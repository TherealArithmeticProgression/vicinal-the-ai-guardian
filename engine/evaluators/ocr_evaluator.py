"""
engine/evaluators/ocr_evaluator.py

OCR Evaluator — extracts text from image/PDF attachments using EasyOCR.

EasyOCR is used rather than pytesseract because:
  - No system-level Tesseract installation required (pure-pip)
  - Better accuracy on degraded/low-res images
  - GPU-acceleratable (falls back to CPU)

Supported attachment types:  IMAGE (.png, .jpg, .jpeg, .bmp, .tiff, .webp)
PDF support is handled via PyMuPDF (fitz) to rasterise pages first,
then each page image is fed through EasyOCR.
"""

from __future__ import annotations

import io
import logging
import time
from pathlib import Path
from typing import ClassVar

from core.errors.exceptions import OCRError, UnsupportedAttachmentError
from core.types.prompt import Attachment, AttachmentType

logger = logging.getLogger(__name__)

_IMAGE_TYPES = {AttachmentType.IMAGE}
_PDF_TYPES   = {AttachmentType.PDF}
_LANGS       = ["en"]


class OCREvaluator:
    """
    Extracts text from attachments in-process (no external service).

    Parameters
    ----------
    languages:
        EasyOCR language codes.  Defaults to English only.
    gpu:
        Whether to attempt GPU acceleration.  Falls back to CPU silently.
    """

    _reader: ClassVar[object | None] = None   # EasyOCR Reader — shared

    def __init__(self, languages: list[str] | None = None, gpu: bool = False) -> None:
        self._languages = languages or _LANGS
        self._gpu       = gpu

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def extract(self, attachment: Attachment) -> str:
        """
        Extract text from a single attachment.

        Raises
        ------
        UnsupportedAttachmentError
            If the attachment type is not IMAGE or PDF.
        OCRError
            If extraction fails for any reason.
        """
        if attachment.type in _IMAGE_TYPES:
            return self._ocr_image_bytes(attachment.data)
        if attachment.type in _PDF_TYPES:
            return self._ocr_pdf_bytes(attachment.data)
        if attachment.type == AttachmentType.TEXT:
            # Plain text attachments just decode and return
            try:
                return attachment.data.decode("utf-8", errors="replace")
            except Exception as exc:
                raise OCRError(f"Failed to decode text attachment: {exc}") from exc

        raise UnsupportedAttachmentError(
            f"OCR is not supported for attachment type '{attachment.type.value}'. "
            "Supported types: image, pdf, text."
        )

    def extract_all(self, attachments: list[Attachment]) -> str:
        """Extract and concatenate text from all attachments."""
        parts: list[str] = []
        for att in attachments:
            try:
                text = self.extract(att)
                if text.strip():
                    parts.append(text.strip())
            except UnsupportedAttachmentError:
                logger.debug("Skipping unsupported attachment: %s", att.type)
            except OCRError as exc:
                logger.warning("OCR failed for attachment '%s': %s", att.filename, exc)
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _ocr_image_bytes(self, data: bytes) -> str:
        t0 = time.perf_counter()
        reader = self._get_reader()

        try:
            import numpy as np
            from PIL import Image

            img = Image.open(io.BytesIO(data)).convert("RGB")
            img_array = np.array(img)
            results = reader.readtext(img_array, detail=0, paragraph=True)
            text = " ".join(results)
        except Exception as exc:
            raise OCRError(f"Image OCR failed: {exc}") from exc

        logger.debug("OCR extracted %d chars in %.1f ms", len(text), (time.perf_counter() - t0) * 1000)
        return text

    def _ocr_pdf_bytes(self, data: bytes) -> str:
        """Rasterise each PDF page and run OCR on each."""
        try:
            import fitz  # PyMuPDF
        except ImportError as exc:
            raise OCRError(
                "PyMuPDF is not installed. Run: pip install pymupdf"
            ) from exc

        all_text: list[str] = []
        try:
            doc = fitz.open(stream=data, filetype="pdf")
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                # Render at 150 DPI — good balance of speed vs. accuracy
                mat = fitz.Matrix(150 / 72, 150 / 72)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                page_text = self._ocr_image_bytes(img_data)
                all_text.append(page_text)
        except OCRError:
            raise
        except Exception as exc:
            raise OCRError(f"PDF OCR failed: {exc}") from exc

        return "\n\n".join(all_text)

    def _get_reader(self):
        if OCREvaluator._reader is None:
            try:
                import easyocr
            except ImportError as exc:
                raise OCRError(
                    "easyocr is not installed. Run: pip install easyocr"
                ) from exc

            logger.info("Initialising EasyOCR reader (languages=%s, gpu=%s) …", self._languages, self._gpu)
            t0 = time.perf_counter()
            OCREvaluator._reader = easyocr.Reader(self._languages, gpu=self._gpu, verbose=False)
            logger.info("EasyOCR ready in %.1f s.", time.perf_counter() - t0)

        return OCREvaluator._reader

"""
data/build_index.py

Build the FAISS threat embedding index from threat_patterns.jsonl.

Run once before using Vicinal:
    python data/build_index.py

Or with a custom patterns file:
    python data/build_index.py --patterns my_threats.jsonl --out custom_index.bin

The script:
  1. Reads each threat pattern from threat_patterns.jsonl
  2. Encodes each text using sentence-transformers (all-MiniLM-L6-v2)
  3. Builds a FAISS IndexFlatL2 (exact search, fast for < 100k vectors)
  4. Writes faiss_index.bin and metadata.json to this directory

For very large datasets (>100k patterns), swap IndexFlatL2 for:
    faiss.IndexIVFFlat — approximate, much faster at scale
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_index")

HERE = Path(__file__).resolve().parent


def build(
    patterns_path: Path,
    index_path: Path,
    meta_path: Path,
    model_name: str = "all-MiniLM-L6-v2",
    device: str = "cpu",
) -> None:
    # ------------------------------------------------------------------ #
    # 1. Load patterns
    # ------------------------------------------------------------------ #
    patterns: list[dict] = []
    with open(patterns_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                patterns.append(json.loads(line))

    if not patterns:
        logger.error("No patterns found in %s", patterns_path)
        sys.exit(1)

    logger.info("Loaded %d threat patterns.", len(patterns))
    texts = [p["text"] for p in patterns]

    # ------------------------------------------------------------------ #
    # 2. Encode
    # ------------------------------------------------------------------ #
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.error("sentence-transformers is not installed. Run: pip install sentence-transformers")
        sys.exit(1)

    logger.info("Loading model '%s' on %s …", model_name, device)
    model = SentenceTransformer(model_name, device=device)

    logger.info("Encoding %d patterns …", len(texts))
    t0 = time.perf_counter()
    embeddings = model.encode(
        texts,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    logger.info("Encoded in %.2f s.  Shape: %s", time.perf_counter() - t0, embeddings.shape)

    # ------------------------------------------------------------------ #
    # 3. Build FAISS index
    # ------------------------------------------------------------------ #
    try:
        import faiss
        import numpy as np
    except ImportError:
        logger.error("faiss-cpu is not installed. Run: pip install faiss-cpu")
        sys.exit(1)

    dim = embeddings.shape[1]
    logger.info("Building IndexFlatL2 (dim=%d) …", dim)
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype("float32"))
    logger.info("Index built: %d vectors.", index.ntotal)

    # ------------------------------------------------------------------ #
    # 4. Save index
    # ------------------------------------------------------------------ #
    faiss.write_index(index, str(index_path))
    logger.info("FAISS index written to %s", index_path)

    # ------------------------------------------------------------------ #
    # 5. Save metadata
    # ------------------------------------------------------------------ #
    meta_records = [
        {
            "id": i,
            "text": p["text"],
            "label": p.get("label", ""),
            "category": p.get("category", "unknown"),
        }
        for i, p in enumerate(patterns)
    ]
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta_records, fh, indent=2, ensure_ascii=False)
    logger.info("Metadata written to %s", meta_path)

    logger.info("Done.  Index is ready to use.")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a FAISS threat embedding index from a JSONL patterns file."
    )
    parser.add_argument(
        "--patterns",
        type=Path,
        default=HERE / "threat_patterns.jsonl",
        help="Path to the threat patterns JSONL file.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=HERE / "faiss_index.bin",
        help="Output path for the FAISS index binary.",
    )
    parser.add_argument(
        "--meta",
        type=Path,
        default=HERE / "metadata.json",
        help="Output path for the metadata JSON file.",
    )
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="sentence-transformers model name.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device for the embedding model.",
    )
    args = parser.parse_args()

    build(
        patterns_path=args.patterns,
        index_path=args.out,
        meta_path=args.meta,
        model_name=args.model,
        device=args.device,
    )


if __name__ == "__main__":
    main()

"""
research/metrics/evaluator.py

Evaluation metrics for baseline comparison experiments.

Given ground-truth labels and baseline predictions, computes:
  - Precision
  - Recall
  - F1 Score
  - False Positive Rate (FPR)
  - Accuracy
  - Average Latency (ms)
  - Area Under ROC Curve (AUC-ROC)

All metrics are returned as a MetricSummary dataclass and can be
exported to a pandas DataFrame for further analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np


@dataclass
class MetricSummary:
    """Full evaluation report for a single baseline run."""
    baseline_name: str
    n_samples: int
    n_threats: int          # ground-truth positive count
    n_benign: int           # ground-truth negative count
    precision: float
    recall: float
    f1: float
    fpr: float              # false positive rate
    accuracy: float
    auc_roc: float
    avg_latency_ms: float
    p95_latency_ms: float
    threshold: float
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"[{self.baseline_name}]  "
            f"F1={self.f1:.3f}  P={self.precision:.3f}  R={self.recall:.3f}  "
            f"FPR={self.fpr:.3f}  AUC={self.auc_roc:.3f}  "
            f"Latency={self.avg_latency_ms:.1f}ms (p95={self.p95_latency_ms:.1f}ms)"
        )


class MetricsEvaluator:
    """Compute evaluation metrics from predictions."""

    @staticmethod
    def compute(
        baseline_name: str,
        y_true: list[int],
        y_scores: list[float],
        latencies_ms: list[float],
        threshold: float = 0.5,
    ) -> MetricSummary:
        """
        Compute a full MetricSummary.

        Parameters
        ----------
        y_true:
            Ground-truth binary labels (1 = threat, 0 = benign).
        y_scores:
            Continuous threat scores from the baseline ∈ [0, 1].
        latencies_ms:
            Per-sample wall-clock time in milliseconds.
        threshold:
            Score ≥ threshold → predicted label = 1.
        """
        y_true_arr   = np.array(y_true, dtype=int)
        y_scores_arr = np.array(y_scores, dtype=float)
        y_pred       = (y_scores_arr >= threshold).astype(int)
        lat_arr      = np.array(latencies_ms, dtype=float)

        tp = int(np.sum((y_pred == 1) & (y_true_arr == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true_arr == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true_arr == 1)))
        tn = int(np.sum((y_pred == 0) & (y_true_arr == 0)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        accuracy  = (tp + tn) / len(y_true_arr) if len(y_true_arr) > 0 else 0.0

        auc = MetricsEvaluator._auc_roc(y_true_arr, y_scores_arr)

        return MetricSummary(
            baseline_name=baseline_name,
            n_samples=len(y_true),
            n_threats=int(y_true_arr.sum()),
            n_benign=int((1 - y_true_arr).sum()),
            precision=precision,
            recall=recall,
            f1=f1,
            fpr=fpr,
            accuracy=accuracy,
            auc_roc=auc,
            avg_latency_ms=float(lat_arr.mean()) if len(lat_arr) else 0.0,
            p95_latency_ms=float(np.percentile(lat_arr, 95)) if len(lat_arr) else 0.0,
            threshold=threshold,
        )

    @staticmethod
    def _auc_roc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """Trapezoidal AUC-ROC (no sklearn dependency if not installed)."""
        try:
            from sklearn.metrics import roc_auc_score
            return float(roc_auc_score(y_true, y_scores))
        except ImportError:
            pass
        except Exception:
            return 0.5   # degenerate case

        # Fallback: manual trapezoidal computation
        thresholds = np.sort(np.unique(y_scores))[::-1]
        tprs, fprs = [0.0], [0.0]
        pos = y_true.sum()
        neg = len(y_true) - pos
        if pos == 0 or neg == 0:
            return 0.5

        for t in thresholds:
            pred = (y_scores >= t).astype(int)
            tp = int(np.sum((pred == 1) & (y_true == 1)))
            fp = int(np.sum((pred == 1) & (y_true == 0)))
            tprs.append(tp / pos)
            fprs.append(fp / neg)

        tprs.append(1.0)
        fprs.append(1.0)
        return float(np.trapz(tprs, fprs))

    @staticmethod
    def compare(summaries: list[MetricSummary]) -> None:
        """Pretty-print a comparison table of baseline results."""
        print("\n" + "=" * 90)
        print(f"{'Baseline':<30} {'F1':>6} {'Prec':>6} {'Rec':>6} {'FPR':>6} {'AUC':>6} {'Lat(ms)':>9}")
        print("-" * 90)
        for s in sorted(summaries, key=lambda x: x.f1, reverse=True):
            print(
                f"{s.baseline_name:<30} "
                f"{s.f1:>6.3f} {s.precision:>6.3f} {s.recall:>6.3f} "
                f"{s.fpr:>6.3f} {s.auc_roc:>6.3f} {s.avg_latency_ms:>9.1f}"
            )
        print("=" * 90 + "\n")

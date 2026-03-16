"""
research/experiments/run_experiment.py

CLI experiment runner — compares multiple baselines on a labelled dataset.

Usage
-----
    # Run with the bundled threat dataset (auto-generates benign samples)
    python research/experiments/run_experiment.py

    # Run with a custom dataset CSV (columns: text, label)
    python research/experiments/run_experiment.py --dataset my_data.csv

    # Run specific baselines only
    python research/experiments/run_experiment.py --baselines keyword vicinal_zero_resource

    # Save results to CSV
    python research/experiments/run_experiment.py --out results/exp_001.csv

Dataset format
--------------
CSV with columns:
    text     : str    — the prompt text
    label    : int    — 1 = threat, 0 = benign

If no dataset is provided, the bundled threat_patterns.jsonl is used as
the positive class, and simple benign prompts are auto-generated.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

# ---- path bootstrap ---------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]
SDK_PATH  = REPO_ROOT / "sdk" / "python"
for p in [str(REPO_ROOT), str(SDK_PATH)]:
    if p not in sys.path:
        sys.path.insert(0, p)
# -----------------------------------------------------------------------


BENIGN_PROMPTS = [
    "What is the capital of France?",
    "Can you summarise this article for me?",
    "Help me write a cover letter for a software engineering position.",
    "What are the best practices for REST API design?",
    "Explain the difference between TCP and UDP.",
    "Translate this paragraph to Spanish.",
    "Write unit tests for this Python function.",
    "What is the time complexity of quicksort?",
    "How do I reverse a linked list in Python?",
    "Give me a recipe for banana bread.",
    "Explain quantum entanglement in simple terms.",
    "What are the main themes in Hamlet?",
    "How does a neural network learn?",
    "What is the difference between supervised and unsupervised learning?",
    "Can you review this code for bugs?",
    "What is the GDP of Germany?",
    "Help me plan a trip to Japan for two weeks.",
    "Write a poem about autumn.",
    "What are the health benefits of meditation?",
    "Explain how DNS resolution works.",
    "Summarise the key points of the Paris Agreement.",
    "What is the difference between a mutex and a semaphore?",
    "How do transformers work in NLP?",
    "Give me ten book recommendations for learning machine learning.",
    "What are the symptoms of vitamin D deficiency?",
    "Explain the concept of compound interest.",
    "How do I use pandas to merge two dataframes?",
    "What is the difference between HTTP and HTTPS?",
    "Help me debug this JavaScript error: Cannot read property 'map' of undefined.",
    "Write a SQL query to find the top 5 customers by revenue.",
]


def load_dataset(path: Path | None) -> tuple[list[str], list[int]]:
    if path and path.exists():
        texts, labels = [], []
        with open(path, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                texts.append(row["text"])
                labels.append(int(row["label"]))
        print(f"Loaded {len(texts)} samples from {path}.")
        return texts, labels

    # Auto-generate from bundled data
    kb_path = REPO_ROOT / "data" / "threat_patterns.jsonl"
    texts, labels = [], []

    with open(kb_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                item = json.loads(line)
                texts.append(item["text"])
                labels.append(1)

    for prompt in BENIGN_PROMPTS:
        texts.append(prompt)
        labels.append(0)

    print(f"Auto-generated dataset: {sum(labels)} threats + {len(labels)-sum(labels)} benign = {len(labels)} total.")
    return texts, labels


def get_baselines(names: list[str] | None):
    from research.baselines.keyword_baseline import KeywordBaseline
    from research.baselines.vicinal_baseline import VicinalBaseline

    available = {
        "keyword":               KeywordBaseline(),
        "vicinal_zero_resource": VicinalBaseline(mode="zero_resource"),
        "vicinal_full":          VicinalBaseline(mode="full"),
    }

    if not names:
        return list(available.values())

    selected = []
    for name in names:
        if name in available:
            selected.append(available[name])
        else:
            print(f"WARNING: Unknown baseline '{name}'. Skipping.")
    return selected


def run(
    dataset_path: Path | None,
    baseline_names: list[str] | None,
    out_path: Path | None,
    verbose: bool = False,
) -> None:
    from research.metrics.evaluator import MetricsEvaluator

    texts, y_true = load_dataset(dataset_path)
    baselines = get_baselines(baseline_names)

    all_summaries = []

    for baseline in baselines:
        print(f"\nRunning baseline: {baseline.name} …")
        y_scores, latencies = [], []

        for i, text in enumerate(texts):
            result = baseline.evaluate(text)
            y_scores.append(result.score)
            latencies.append(result.latency_ms)

            if verbose:
                print(
                    f"  [{i+1:03d}] label={y_true[i]} pred={result.label} "
                    f"score={result.score:.3f} lat={result.latency_ms:.1f}ms"
                )

        summary = MetricsEvaluator.compute(
            baseline_name=baseline.name,
            y_true=y_true,
            y_scores=y_scores,
            latencies_ms=latencies,
            threshold=baseline.threshold,
        )
        all_summaries.append(summary)
        print(summary)

    MetricsEvaluator.compare(all_summaries)

    if out_path:
        import csv as _csv
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [s.to_dict() for s in all_summaries]
        with open(out_path, "w", newline="", encoding="utf-8") as fh:
            writer = _csv.DictWriter(fh, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Results saved to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Vicinal baseline comparison experiment runner.")
    parser.add_argument("--dataset", type=Path, default=None, help="CSV dataset path (text, label).")
    parser.add_argument("--baselines", nargs="*", default=None,
                        choices=["keyword", "vicinal_zero_resource", "vicinal_full"],
                        help="Baselines to run. Default: all.")
    parser.add_argument("--out", type=Path, default=None, help="Output CSV for results.")
    parser.add_argument("--verbose", action="store_true", help="Print per-sample results.")
    args = parser.parse_args()

    run(
        dataset_path=args.dataset,
        baseline_names=args.baselines,
        out_path=args.out,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()

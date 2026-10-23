"""
Measure the auditor against the planted errors in data/raw/ground_truth.csv.

    python evaluate.py

Prints detection precision/recall and, per error type, how many planted
errors were caught with the right primary exception.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from auditor import load_data, run_audit  # noqa: E402

RAW = Path(__file__).resolve().parent / "data" / "raw"


def evaluate(results: pd.DataFrame, truth: pd.DataFrame) -> dict:
    m = results[["line_id", "status", "primary_exception", "exceptions"]].merge(truth, on="line_id")
    is_err = m.truth != "CLEAN"
    flagged = m.status == "EXCEPTION"
    tp = int((is_err & flagged).sum())
    fp = int((~is_err & flagged).sum())
    fn = int((is_err & ~flagged).sum())

    per_type = (
        m[is_err]
        .assign(caught=lambda d: d.apply(lambda r: r.truth in r.exceptions.split("|"), axis=1),
                primary_ok=lambda d: d.truth == d.primary_exception)
        .groupby("truth")[["caught", "primary_ok"]].agg(["sum", "count"])
    )
    return dict(
        tp=tp, fp=fp, fn=fn,
        precision=tp / (tp + fp) if tp + fp else 0.0,
        recall=tp / (tp + fn) if tp + fn else 0.0,
        per_type=per_type,
        false_positives=m[~is_err & flagged],
        misses=m[is_err & ~flagged],
    )


def main():
    results = run_audit(load_data(RAW))
    truth = pd.read_csv(RAW / "ground_truth.csv")
    ev = evaluate(results, truth)
    print(f"Detection  precision {ev['precision']:.1%}  recall {ev['recall']:.1%}  "
          f"(TP {ev['tp']}, FP {ev['fp']}, FN {ev['fn']})")
    print(ev["per_type"].to_string())
    if len(ev["false_positives"]):
        print("\nFalse positives by flag:")
        print(ev["false_positives"].primary_exception.value_counts().to_string())
    if len(ev["misses"]):
        print("\nMissed errors by type:")
        print(ev["misses"].truth.value_counts().to_string())


if __name__ == "__main__":
    main()

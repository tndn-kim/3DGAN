"""Export Structured/Pipeline.py results into structured-results.js for the dashboard.

Usage after running the Structured pipeline and saving its returned dictionary:
    python tools/export_dashboard_results.py --input output/results.pkl --out structured-results.js

The pickle must contain keys: results_before, results_after, and optionally augmented/preprocess.
"""
import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

MODEL_LABELS = {"rf": "Random Forest", "xgb": "XGBoost", "bigru": "BiGRU"}
CLASS_LABELS = [
    "Benign",
    "Analysis & Reconnaissance",
    "System Compromise",
    "DoS & Fuzzers",
]
CLASS_COUNTS_BEFORE = [358332, 17120, 38383, 34080]


def to_builtin(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    return value


def compact_results(results):
    out = {}
    for model, payload in results.items():
        metrics = payload.get("metrics", payload)
        keep = {}
        for key in ["accuracy", "f1_macro", "micro_FNR", "conf_matrix"]:
            if key in metrics:
                keep[key] = to_builtin(metrics[key])
        out[model] = {"metrics": keep}
    return out


def augmented_counts(pipeline_result):
    counts = list(CLASS_COUNTS_BEFORE)
    augmented = pipeline_result.get("augmented") or {}
    for label, arr in augmented.items():
        i = int(label)
        if 0 <= i < len(counts) and hasattr(arr, "shape") and len(arr.shape) == 2:
            counts[i] += int(arr.shape[0])
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="pickle file containing run_pipeline return dict")
    parser.add_argument("--out", default="structured-results.js")
    args = parser.parse_args()

    with open(args.input, "rb") as f:
        result = pickle.load(f)

    dashboard = {
        "actual": True,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "models": [m for m in ["rf", "xgb", "bigru"] if m in result.get("results_before", {})],
        "modelLabels": MODEL_LABELS,
        "classLabels": CLASS_LABELS,
        "classCountsBefore": CLASS_COUNTS_BEFORE,
        "classCountsAfter": augmented_counts(result),
        "before": compact_results(result["results_before"]),
        "after": compact_results(result["results_after"]),
    }

    js = "window.STRUCTURED_RESULTS = " + json.dumps(dashboard, ensure_ascii=False, indent=2) + ";\n"
    Path(args.out).write_text(js, encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

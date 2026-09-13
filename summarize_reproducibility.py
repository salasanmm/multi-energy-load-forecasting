"""Summarize per-run reproducibility manifests for manuscript tables.

The script only reads JSON manifests written by train.py.  It does not infer
missing runs or manufacture aggregate metrics; a model is marked incomplete
when fewer than the requested number of manifests are present.
"""

import argparse
import csv
import glob
import json
import os
from collections import defaultdict


def load_manifests(root):
    grouped = defaultdict(list)
    pattern = os.path.join(root, "**", "reproducibility_run*.json")
    for path in glob.glob(pattern, recursive=True):
        try:
            with open(path, encoding="utf-8") as stream:
                record = json.load(stream)
            if "model" not in record or "run_id" not in record:
                continue
            grouped[os.path.dirname(path)].append((path, record))
        except (OSError, ValueError, TypeError):
            continue
    return grouped


def same_protocol(records):
    if not records:
        return True
    keys = ("selected_hyperparameters", "training_protocol", "data_protocol")
    first = records[0]
    return all(record.get(key) == first.get(key) for record in records[1:] for key in keys)


def summarize(root, expected_runs):
    rows = []
    for directory, entries in sorted(load_manifests(root).items()):
        entries.sort(key=lambda item: int(item[1].get("run_id", 0)))
        records = [record for _, record in entries]
        model = records[0].get("model", os.path.basename(directory))
        metrics = [record.get("final_test_metrics", {}) for record in records]
        rows.append({
            "model_directory": os.path.relpath(directory, root),
            "model": model,
            "runs_completed": len(records),
            "expected_runs": expected_runs,
            "complete": len(records) >= expected_runs,
            "run_ids": ",".join(str(record.get("run_id")) for record in records),
            "seeds": ",".join(str(record.get("seed")) for record in records),
            "protocol_identical": same_protocol(records),
            "parameter_count": records[0].get("total_parameter_count"),
            "trainable_parameter_count": records[0].get("trainable_parameter_count"),
            "MAE_values": ",".join(str(metric.get("MAE")) for metric in metrics),
            "MAPE_values": ",".join(str(metric.get("MAPE")) for metric in metrics),
            "RMSE_values": ",".join(str(metric.get("RMSE")) for metric in metrics),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Summarize train.py reproducibility manifests")
    parser.add_argument("--root", default="./output", help="root directory containing model outputs")
    parser.add_argument("--expected-runs", type=int, default=3,
                        help="number of independent runs required by the manuscript")
    parser.add_argument("--output", default=None, help="CSV output path; defaults to <root>/reproducibility_summary.csv")
    args = parser.parse_args()

    if args.expected_runs < 1:
        parser.error("--expected-runs must be positive")
    rows = summarize(args.root, args.expected_runs)
    output_path = args.output or os.path.join(args.root, "reproducibility_summary.csv")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fieldnames = [
        "model_directory", "model", "runs_completed", "expected_runs", "complete",
        "run_ids", "seeds", "protocol_identical", "parameter_count",
        "trainable_parameter_count", "MAE_values", "MAPE_values", "RMSE_values",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"models": len(rows), "output": output_path, "rows": rows},
                     indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()

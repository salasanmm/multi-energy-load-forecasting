import argparse
import csv
import re
import statistics
from pathlib import Path


def read_overall(root, top_k):
    tag = "dft_top5_811" if top_k == 5 else f"dft_top{top_k}_sens24_811"
    path = root / f"{tag}_rolling_overlap_average_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing summary for top_k={top_k}: {path}")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    return [row for row in rows if int(row["horizon"]) == 24]


def validation_checkpoint_mae(root, top_k):
    if top_k == 5:
        log_dir = root / "run_logs_dft_top5_811"
        names = [f"dft_top5_811_h24_seed{seed}.log" for seed in (2020, 2021, 2022)]
    else:
        log_dir = root / "run_logs_topk_sensitivity_24h"
        names = [f"dft_top{top_k}_sens24_811_seed{seed}.log" for seed in (2020, 2021, 2022)]
    values = []
    for name in names:
        text = (log_dir / name).read_text(encoding="utf-8", errors="replace")
        matches = re.findall(r"kept checkpoint .*?validation mae ([0-9.]+)", text)
        if not matches:
            raise ValueError(f"Missing validation checkpoint MAE in {log_dir / name}")
        values.append(float(matches[-1]))
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="output8_1_1")
    args = parser.parse_args()
    root = Path(args.root)

    rows = []
    for top_k in (1, 3, 5, 10):
        validation_values = validation_checkpoint_mae(root, top_k)
        validation_stats = {
            "validation_checkpoint_MAE_mean": sum(validation_values) / len(validation_values),
            "validation_checkpoint_MAE_std": statistics.stdev(validation_values),
            "validation_checkpoint_MAE_var": statistics.variance(validation_values),
        }
        for summary_row in read_overall(root, top_k):
            rows.append({"top_k": top_k, **summary_row, **validation_stats})

    output = root / "top_k_sensitivity_24h.csv"
    fields = ["top_k", "horizon", "channel", "n_seeds",
              "validation_checkpoint_MAE_mean", "validation_checkpoint_MAE_std",
              "validation_checkpoint_MAE_var"]
    for metric in ("MAE", "MAPE", "RMSE", "DS", "ACCR"):
        fields.extend([f"{metric}_mean", f"{metric}_std", f"{metric}_var"])
    fields.extend(["Params", "FLOPs", "epoch_time_sec_mean", "epoch_time_sec_std", "epoch_time_sec_var"])
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} top-k sensitivity rows to {output}")


if __name__ == "__main__":
    main()

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from util import DataLoaderS


SEASONS = ("spring", "summer", "autumn", "winter")
CHANNELS = ("electricity", "cooling", "heating")
METRICS = ("MAE", "RMSE", "MAPE", "ACCR")


def channel_metrics(y_true, y_pred):
    rows = []
    for index, name in enumerate(CHANNELS):
        actual, predicted = y_true[:, index], y_pred[:, index]
        error = predicted - actual
        nonzero = np.abs(actual) > 1e-8
        mape = float(np.mean(np.abs(error[nonzero] / actual[nonzero])) * 100) if nonzero.any() else float("nan")
        a_std, p_std = float(np.std(actual)), float(np.std(predicted))
        corr = float(np.mean((actual - actual.mean()) * (predicted - predicted.mean())) / (a_std * p_std)) \
            if a_std and p_std else 0.0
        rows.append({
            "channel": name,
            "MAE": float(np.mean(np.abs(error))),
            "RMSE": float(np.sqrt(np.mean(error ** 2))),
            "MAPE": mape,
            "ACCR": corr,
            "MAPE_nonzero_points": int(nonzero.sum()),
        })
    rows.append({
        "channel": "overall",
        **{metric: float(np.mean([row[metric] for row in rows])) for metric in METRICS},
        "MAPE_nonzero_points": int(sum(row["MAPE_nonzero_points"] for row in rows)),
    })
    return rows


def load_model_arrays(root, model, seed, experiment_tag):
    if model == "MFTG-Net":
        result_dir = root / f"MFTG-Net_24_168_{experiment_tag}_seed{seed}" / "result"
    else:
        result_dir = root / "baseline_runs" / f"{model}_24_168_613_seed{seed}" / "result"
    true_path = result_dir / "all_y_true_overlap_average.npy"
    pred_path = result_dir / "all_predict_value_overlap_average.npy"
    if not true_path.exists() or not pred_path.exists():
        raise FileNotFoundError(f"Missing completed result arrays for {model}, seed {seed} in {result_dir}")
    return np.load(true_path), np.load(pred_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="output6_1_3/main_experiment")
    parser.add_argument("--experiment-tag", default="dft_top5_613_seasonal24h")
    parser.add_argument("--data", default="data/dataset_input_jiuzheng.csv")
    args = parser.parse_args()
    root = Path(args.root)
    data = DataLoaderS(args.data, 0.6, 0.1, "cpu", horizon=24, window=168, normalize=2)
    valid_end = int(0.7 * data.n)
    labels = data.season_from_day_of_year(data.row_day_of_year[valid_end:data.n])
    if int((labels == "spring").sum()) == 0 or set(np.unique(labels)) != set(SEASONS):
        raise ValueError("The 6:1:3 test segment no longer covers all four seasons")

    records = []
    for model in ("MFTG-Net", "TimeMixer", "Autoformer", "DLinear", "PatchTST"):
        for seed in (2020, 2021, 2022):
            y_true, y_pred = load_model_arrays(root, model, seed, args.experiment_tag)
            if y_true.shape != y_pred.shape or len(y_true) != len(labels):
                raise ValueError(f"Array/season length mismatch for {model} seed {seed}: "
                                 f"{y_true.shape}, {y_pred.shape}, seasons={len(labels)}")
            for season in SEASONS:
                mask = labels == season
                season_rows = channel_metrics(y_true[mask], y_pred[mask])
                row = next(item for item in season_rows if item["channel"] == "overall")
                records.append({
                    "Model": model, "seed": seed, "season": season,
                    "channel": "overall",
                    **{metric: row[metric] for metric in METRICS},
                    "n_target_timestamps": int(mask.sum()),
                    "MAPE_nonzero_points": row["MAPE_nonzero_points"],
                })

    by_seed = pd.DataFrame(records)
    root.mkdir(parents=True, exist_ok=True)
    summary_path = root / "seasonal_metrics_summary_613_24h_overall_only.csv"
    grouped = by_seed.groupby(["Model", "season", "channel"], sort=False)
    summary = grouped[list(METRICS)].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(str(value) for value in col if value) if isinstance(col, tuple) else col
                       for col in summary.columns]
    sizes = by_seed.groupby(["Model", "season", "channel"], sort=False)[
        ["n_target_timestamps", "MAPE_nonzero_points"]].first().reset_index()
    summary = summary.merge(sizes, on=["Model", "season", "channel"], how="left")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"Wrote {summary_path}")
    print("Test target timestamps by season:", {
        season: int((labels == season).sum()) for season in SEASONS
    })


if __name__ == "__main__":
    main()

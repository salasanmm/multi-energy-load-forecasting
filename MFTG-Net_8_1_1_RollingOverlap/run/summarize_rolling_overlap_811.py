import argparse
import csv
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from get_config import get_config
from metrics import overlap_average


ABLATION_FLAGS = {
    "wo_mssm": "use_mssm", "wo_mstm": "use_mstm",
    "wo_out_residual": "use_out_residual", "wo_regression": "use_regression",
    "wo_pgf": "use_pgf", "wo_crd": "use_crd", "wo_ph": "use_ph",
    "wo_anchor": "use_anchor", "wo_haf": "use_haf",
}


def profile_model(horizon, top_k=5, ablation="full"):
    """Count parameters and PyTorch profiler FLOPs for one origin batch item."""
    model_path = PROJECT_ROOT / "test_model" / "MFGT-Net.py"
    spec = importlib.util.spec_from_file_location("mftg_profile_model", model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = get_config("TimeMixer")
    config.pred_len = horizon
    config.top_k = top_k
    if ablation in ABLATION_FLAGS:
        setattr(config, ABLATION_FLAGS[ablation], False)
    model = module.Model(config).eval()
    params = sum(parameter.numel() for parameter in model.parameters())
    sample = torch.zeros(1, 1, config.enc_in, config.seq_len)
    with torch.no_grad(), torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True
    ) as prof:
        model(sample)
    flops = sum(event.flops or 0 for event in prof.key_averages())
    return int(params), int(flops)


def mean_epoch_time(log_path):
    if log_path is None or not log_path.exists():
        return None
    raw = log_path.read_bytes()
    encoding = "utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8"
    text = raw.decode(encoding, errors="replace")
    seconds = [float(value) for value in re.findall(r"end of epoch\s+\d+\s+\| time:\s+([\d.]+)s", text)]
    return float(np.mean(seconds)) if seconds else None


def metrics(y_true, y_pred):
    error = y_pred - y_true
    delta_axis = 0 if y_true.ndim == 1 else 1
    true_delta = np.diff(y_true, axis=delta_axis)
    pred_delta = np.diff(y_pred, axis=delta_axis)
    if y_true.ndim == 1:
        sigma_true = y_true.std()
        sigma_pred = y_pred.std()
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = ((y_true - y_true.mean()) * (y_pred - y_pred.mean())).mean()
            corr = corr / (sigma_true * sigma_pred) if sigma_true and sigma_pred else 0.0
    else:
        sigma_true = y_true.std(axis=0)
        sigma_pred = y_pred.std(axis=0)
        valid = (sigma_true != 0) & (sigma_pred != 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = ((y_true - y_true.mean(axis=0)) * (y_pred - y_pred.mean(axis=0))).mean(axis=0)
            corr = corr / (sigma_true * sigma_pred)
        corr = float(corr[valid].mean()) if np.any(valid) else 0.0
    return {
        "MAE": float(np.mean(np.abs(error))),
        "MAPE": float(np.mean(np.abs(error / y_true)) * 100),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "DS": float(np.mean((true_delta * pred_delta) >= 0) * 100),
        "ACCR": float(corr),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="output8_1_1")
    parser.add_argument("--experiment-tag", default="dft_top5_811")
    parser.add_argument("--ablation", choices=("full", *ABLATION_FLAGS), default="full")
    args = parser.parse_args()
    root = Path(args.root)
    rows_by_protocol = {"fixed_origin": [], "rolling_overlap_average": []}
    complexity = {}
    for horizon in (24, 48, 72, 96):
        for seed in (2020, 2021, 2022):
            run = root / f"MFTG-Net_{horizon}_168_{args.experiment_tag}_seed{seed}" / "result"
            manifest_path = run / "reproducibility_run1.json"
            true_path, pred_path = run / "all_y_true.npy", run / "all_predict_value.npy"
            if not all(path.exists() for path in (manifest_path, true_path, pred_path)):
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            top_k = int(manifest["selected_hyperparameters"].get("top_k", 5))
            complexity_key = (horizon, top_k)
            if complexity_key not in complexity:
                complexity[complexity_key] = profile_model(horizon, top_k, args.ablation)
            y_true, y_pred = np.load(true_path), np.load(pred_path)
            if y_true.shape != y_pred.shape or y_true.ndim != 3 or y_true.shape[1:] != (horizon, 3):
                raise ValueError(f"Unexpected origin-wise array shape for {run}: {y_true.shape}, {y_pred.shape}")
            overlap_true, overlap_pred = overlap_average(y_true, y_pred)
            protocols = {
                "fixed_origin": (y_true, y_pred),
                "rolling_overlap_average": (overlap_true, overlap_pred),
            }
            epoch_log = run / "data.txt"
            if not epoch_log.exists():
                log_name = f"{args.experiment_tag}_h{horizon}_seed{seed}.log"
                log_candidates = [root / "run_logs" / log_name]
                log_candidates.extend(root.glob(f"run_logs*/{log_name}"))
                log_candidates.append(root / "run_logs" / f"h{horizon}_seed{seed}.log")
                epoch_log = next((path for path in log_candidates if path.exists()), None)
            epoch_seconds = mean_epoch_time(epoch_log)
            for protocol, (protocol_true, protocol_pred) in protocols.items():
                channel_results = {}
                for channel, index in (("electricity", 0), ("cooling", 1), ("heating", 2)):
                    channel_results[channel] = metrics(
                        protocol_true[..., index], protocol_pred[..., index]
                    )
                # Overall is the arithmetic mean of the three channel metrics.
                channel_results["overall"] = {
                    name: float(np.mean([channel_results[key][name]
                                         for key in ("electricity", "cooling", "heating")]))
                    for name in ("MAE", "MAPE", "RMSE", "DS", "ACCR")
                }
                for channel, result in channel_results.items():
                    rows_by_protocol[protocol].append({
                        "horizon": horizon,
                        "seed": seed,
                        "n_origins": len(y_true),
                        "n_scored_values": int(np.prod(protocol_true.shape[:-1])),
                        "channel": channel,
                        **result,
                        "Params": int(manifest["total_parameter_count"]),
                        "FLOPs": complexity[complexity_key][1],
                        "epoch_time_sec": epoch_seconds,
                        "manifest": str(manifest_path),
                    })
    root.mkdir(parents=True, exist_ok=True)
    fields = ["horizon", "seed", "n_origins", "n_scored_values", "channel",
              "MAE", "MAPE", "RMSE", "DS", "ACCR", "Params", "FLOPs",
              "epoch_time_sec", "manifest"]
    summary_fields = ["horizon", "channel", "n_seeds"]
    for metric in ("MAE", "MAPE", "RMSE", "DS", "ACCR"):
        summary_fields.extend([f"{metric}_mean", f"{metric}_std", f"{metric}_var"])
    summary_fields.extend(["Params", "FLOPs", "epoch_time_sec_mean",
                           "epoch_time_sec_std", "epoch_time_sec_var"])
    for protocol, rows in rows_by_protocol.items():
        by_seed = root / f"{args.experiment_tag}_{protocol}_by_seed.csv"
        with by_seed.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        summary_rows = []
        for horizon in (24, 48, 72, 96):
            for channel in ("overall", "electricity", "cooling", "heating"):
                selected = [row for row in rows
                            if row["horizon"] == horizon and row["channel"] == channel]
                if len(selected) != 3:
                    continue
                entry = {"horizon": horizon, "channel": channel, "n_seeds": len(selected),
                         "Params": int(selected[0]["Params"]),
                         "FLOPs": int(selected[0]["FLOPs"])}
                for metric in ("MAE", "MAPE", "RMSE", "DS", "ACCR"):
                    values = np.asarray([row[metric] for row in selected], dtype=float)
                    entry[f"{metric}_mean"] = float(values.mean())
                    entry[f"{metric}_std"] = float(values.std(ddof=1))
                    entry[f"{metric}_var"] = float(values.var(ddof=1))
                epoch_times = np.asarray([row["epoch_time_sec"] for row in selected], dtype=float)
                entry["epoch_time_sec_mean"] = float(epoch_times.mean())
                entry["epoch_time_sec_std"] = float(epoch_times.std(ddof=1))
                entry["epoch_time_sec_var"] = float(epoch_times.var(ddof=1))
                summary_rows.append(entry)
        summary = root / f"{args.experiment_tag}_{protocol}_summary.csv"
        with summary.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=summary_fields)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Wrote {len(rows)} {protocol} per-seed rows to {by_seed}")
        print(f"Wrote {len(summary_rows)} {protocol} mean/std rows to {summary}")


if __name__ == "__main__":
    main()

import argparse
import csv
import importlib.util
import json
import math
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
sys_path = str(ROOT)
import sys
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

from test_model import TimeMixer, DLinear, PatchTST, Autoformer


DATA = ROOT / "data" / "processed" / "kitakyushu_2019_hourly.csv"
OUT = ROOT / "output" / "main_experiment"
DATA_AUX = ROOT / "data" / "processed" / "kitakyushu_2019_hourly_with_aux.csv"
OUT_AUX = ROOT / "output" / "main_experiment_with_aux"
CHANNELS = ["electricity", "cooling", "heating"]
TARGET_COLUMNS = ["electricity_kw", "cooling_kw", "heating_kw"]
INPUT_FEATURES = TARGET_COLUMNS + [
    "outdoor_temp_c", "hot_water_kw", "hour_sin", "hour_cos",
    "weekday_sin", "weekday_cos", "yearday_sin", "yearday_cos", "is_weekend",
]
MODELS = {
    "MFTG-Net": None,
    "TimeMixer": TimeMixer.Model,
    "DLinear": DLinear.Model,
    "PatchTST": PatchTST.Model,
    "Autoformer": Autoformer.Model,
}


def config_for(model_name, horizon):
    cfg = SimpleNamespace(
        task_name="long_term_forecast", is_training=1, model_id="kitakyushu_2019",
        seq_len=168, pred_len=horizon, pred_in_len=horizon, seq_out_len=horizon,
        label_len=48, enc_in=3, dec_in=3, c_out=3, individual=True,
        d_model=64, d_ff=64, n_heads=4, e_layers=2, d_layers=1,
        dropout=0.1, fc_dropout=0.1, head_dropout=0.1, factor=1,
        activation="gelu", output_attention=False, distil=True,
        embed="fixed", freq="h", moving_avg=25, use_norm=1,
        channel_independence=0, down_sampling_window=2, down_sampling_layers=2,
        down_sampling_method="avg", decomp_method="dft_decomp", top_k=5,
        num_kernels=4, patch_len=24, stride=8, padding_patch="end",
        revin=1, affine=0, subtract_last=0, head_type="flatten", pretrain_head=False,
        use_pgf=True, use_crd=True, use_ph=True, use_anchor=True, use_weekly_anchor=False,
        use_haf=True, use_mssm=True, use_mstm=True, use_out_residual=True,
        use_regression=True, pgf_hidden=32, pgf_dropout=0.1, pgf_gate_bias=-3.0,
        crd_hidden=24, crd_dropout=0.1, crd_gate_bias=-3.0,
        ph_hidden=3, ph_kernel=3, ph_dropout=0.1, ph_gate_bias=-3.0,
        anchor_hidden=32, anchor_dropout=0.1, anchor_gate_bias=-3.0,
        anchor_init_scale=0.02, weekly_anchor_hidden=32, weekly_anchor_dropout=0.1,
        weekly_anchor_gate_bias=-3.0, weekly_anchor_init_scale=0.02, haf_init_scale=0.02,
        num_class=3,
    )
    if model_name == "MFTG-Net":
        cfg.e_layers = 3
        cfg.d_model = cfg.d_ff = 64
        cfg.pred_in_len = horizon
        model_file = ROOT / "test_model" / "MFGT-Net.py"
        spec = importlib.util.spec_from_file_location("mftg_net_japan", model_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.Model(cfg), cfg
    if model_name == "TimeMixer":
        cfg.e_layers = 3
        cfg.pred_in_len = horizon
        cfg.moving_avg = 3
    return MODELS[model_name](cfg), cfg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def forward(model_name, model, x, horizon, input_adapter=None):
    if input_adapter is not None:
        x = input_adapter(x)
    if model_name == "MFTG-Net":
        return model(x.unsqueeze(1).transpose(2, 3))
    if model_name == "PatchTST":
        return model(x.unsqueeze(1).transpose(2, 3))
    if model_name == "TimeMixer":
        return model(x)
    if model_name == "DLinear":
        return model(x, None, None, None)
    bsz = x.shape[0]
    x_mark = x.new_zeros((bsz, x.shape[1], 4))
    x_dec = torch.cat((x[:, -48:, :], x.new_zeros((bsz, horizon, 3))), dim=1)
    x_mark_dec = x.new_zeros((bsz, 48 + horizon, 4))
    return model(x, x_mark, x_dec, x_mark_dec)


def origin_windows(series, start, end, seq_len, horizon):
    starts = np.arange(start, end - horizon + 1, dtype=np.int64)
    xs = np.stack([series[t - seq_len:t] for t in starts])
    ys = np.stack([series[t:t + horizon] for t in starts])
    return torch.from_numpy(xs).float(), torch.from_numpy(ys).float(), starts


def overlap_average(predictions, starts, actual, start, end):
    pred_sum = np.zeros((end - start, actual.shape[1]), dtype=np.float64)
    counts = np.zeros(end - start, dtype=np.int64)
    for pred, origin in zip(predictions, starts):
        lo = max(int(origin), start)
        hi = min(int(origin) + pred.shape[0], end)
        if hi > lo:
            pred_sum[lo - start:hi - start] += pred[lo - int(origin):hi - int(origin)]
            counts[lo - start:hi - start] += 1
    keep = counts > 0
    return actual[start:end][keep], pred_sum[keep] / counts[keep, None], keep


def score(y_true, y_pred):
    error = y_pred - y_true
    mae_c = np.mean(np.abs(error), axis=0)
    rmse_c = np.sqrt(np.mean(error ** 2, axis=0))
    mape_c = 100.0 * np.mean(np.abs(error) / (np.abs(y_true) + 1e-2), axis=0)
    diff_true = np.diff(y_true, axis=0)
    diff_pred = np.diff(y_pred, axis=0)
    ds_c = 100.0 * np.mean(np.sign(diff_true) == np.sign(diff_pred), axis=0)
    corr_c = []
    for c in range(y_true.shape[1]):
        if np.std(y_true[:, c]) == 0 or np.std(y_pred[:, c]) == 0:
            corr_c.append(0.0)
        else:
            corr_c.append(float(np.corrcoef(y_true[:, c], y_pred[:, c])[0, 1]))
    channel = {}
    for i, name in enumerate(CHANNELS):
        channel[name] = {"MAE": mae_c[i], "RMSE": rmse_c[i], "MAPE": mape_c[i], "DS": ds_c[i], "ACCR": corr_c[i]}
    channel["overall"] = {
        key: float(np.mean([channel[name][key] for name in CHANNELS]))
        for key in ("MAE", "RMSE", "MAPE", "DS", "ACCR")
    }
    return channel


def make_record(model_name, horizon, seed, channel, metrics, epoch_time, epoch, params, flops, n_points):
    row = {"Model": model_name, "horizon": horizon, "seed": seed, "channel": channel}
    row.update(metrics)
    row.update({"epoch_time_sec": epoch_time, "selected_epoch": epoch, "Params": params,
                "FLOPs_forward": flops, "n_test_timestamps": n_points})
    return row


def estimate_flops(model_name, model, device, horizon, input_adapter=None, input_features=3):
    try:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        x = torch.zeros((1, 168, input_features), device=device)
        with torch.profiler.profile(activities=activities, with_flops=True) as prof:
            with torch.no_grad():
                forward(model_name, model, x, horizon, input_adapter)
        value = sum(event.flops or 0 for event in prof.events())
        return int(value) if value else ""
    except Exception:
        return ""


def train_run(model_name, horizon, seed, device, epochs, batch_size, use_aux=False):
    set_seed(seed)
    data_path = DATA_AUX if use_aux else DATA
    frame = pd.read_csv(data_path)
    targets = frame[TARGET_COLUMNS].to_numpy(dtype=np.float64)
    if use_aux:
        input_raw = frame[INPUT_FEATURES].to_numpy(dtype=np.float64)
    else:
        input_raw = targets
    n = len(targets)
    train_end, valid_end = int(0.8 * n), int(0.9 * n)
    input_scale = np.max(np.abs(input_raw[:train_end]), axis=0)
    input_scale[input_scale == 0] = 1.0
    target_scale = np.max(np.abs(targets[:train_end]), axis=0)
    target_scale[target_scale == 0] = 1.0
    inputs = (input_raw / input_scale).astype(np.float32)
    outputs = (targets / target_scale).astype(np.float32)
    seq_len = 168

    tx, _, _ = origin_windows(inputs, seq_len, train_end, seq_len, horizon)
    _, ty, _ = origin_windows(outputs, seq_len, train_end, seq_len, horizon)
    vx, _, vstarts = origin_windows(inputs, seq_len, valid_end, seq_len, horizon)
    _, vy, _ = origin_windows(outputs, seq_len, valid_end, seq_len, horizon)
    qx, _, qstarts = origin_windows(inputs, seq_len, n, seq_len, horizon)
    _, qy, _ = origin_windows(outputs, seq_len, n, seq_len, horizon)
    if len(tx) == 0 or len(vx) == 0 or len(qx) == 0:
        raise ValueError(f"Insufficient rows for {horizon}-step forecast in chronological 8:1:1 split")

    model, cfg = config_for(model_name, horizon)
    model = model.to(device)
    input_adapter = None
    if use_aux:
        input_adapter = torch.nn.Linear(len(INPUT_FEATURES), len(TARGET_COLUMNS)).to(device)
        with torch.no_grad():
            input_adapter.weight.zero_()
            input_adapter.bias.zero_()
            input_adapter.weight[:, :len(TARGET_COLUMNS)] = torch.eye(len(TARGET_COLUMNS), device=device)
    trainable = list(model.parameters()) + (list(input_adapter.parameters()) if input_adapter is not None else [])
    params = sum(p.numel() for p in trainable if p.requires_grad)
    n_input_features = len(INPUT_FEATURES) if use_aux else len(TARGET_COLUMNS)
    flops = estimate_flops(model_name, model, device, horizon, input_adapter, n_input_features)
    loader = DataLoader(TensorDataset(tx, ty), batch_size=batch_size, shuffle=True, drop_last=False)
    optimizer = torch.optim.Adam(trainable, lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)
    best_val = math.inf
    best_state = None
    best_epoch = 0
    epoch_times = []

    for epoch in range(1, epochs + 1):
        model.train()
        tic = time.perf_counter()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = forward(model_name, model, xb, horizon, input_adapter)
            loss = (torch.abs(pred - yb) / (torch.abs(yb) + 1e-3)).mean() * 100.0 + 0.0005 * torch.abs(pred - yb).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_times.append(time.perf_counter() - tic)

        model.eval()
        predictions = []
        with torch.no_grad():
            for offset in range(0, len(vx), batch_size):
                pred = forward(model_name, model, vx[offset:offset + batch_size].to(device), horizon,
                               input_adapter)
                predictions.append(pred.cpu().numpy())
        val_true, val_pred, _ = overlap_average(np.concatenate(predictions), vstarts, outputs,
                                                 train_end, valid_end)
        val_mae = float(np.mean(np.abs(val_pred - val_true)))
        scheduler.step(val_mae)
        if val_mae < best_val:
            best_val, best_epoch = val_mae, epoch
            best_state = {
                "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "input_adapter": ({k: v.detach().cpu().clone() for k, v in input_adapter.state_dict().items()}
                                  if input_adapter is not None else None),
            }
        print(f"{model_name} H{horizon} seed{seed} epoch {epoch}/{epochs} valid rolling MAE {val_mae:.6f}", flush=True)

    model.load_state_dict(best_state["model"])
    if input_adapter is not None:
        input_adapter.load_state_dict(best_state["input_adapter"])
    model.eval()
    predictions = []
    with torch.no_grad():
        for offset in range(0, len(qx), batch_size):
            pred = forward(model_name, model, qx[offset:offset + batch_size].to(device), horizon,
                           input_adapter)
            predictions.append(pred.cpu().numpy())
    test_true, test_pred, _ = overlap_average(np.concatenate(predictions), qstarts, outputs,
                                               valid_end, n)
    test_true = test_true * target_scale
    test_pred = test_pred * target_scale
    scores = score(test_true, test_pred)
    rows = [make_record(model_name, horizon, seed, ch, values, float(np.mean(epoch_times)), best_epoch,
                        params, flops, len(test_true)) for ch, values in scores.items()]
    return rows


def summarize(rows):
    frame = pd.DataFrame(rows)
    metrics = ["MAE", "RMSE", "MAPE", "DS", "ACCR"]
    grouped = frame.groupby(["Model", "horizon", "channel"], sort=False)
    summary = grouped[metrics + ["epoch_time_sec"]].agg(["mean", "std", "var"]).reset_index()
    summary.columns = ["_".join(str(x) for x in col if x) if isinstance(col, tuple) else col for col in summary.columns]
    complexity = frame.groupby(["Model", "horizon", "channel"], sort=False)[["Params", "FLOPs_forward", "selected_epoch", "n_test_timestamps"]].first().reset_index()
    summary = summary.merge(complexity, on=["Model", "horizon", "channel"], how="left")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--horizons", nargs="+", type=int, default=[24, 48, 72, 96])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2020, 2021, 2022])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--use-aux-features", action="store_true",
                        help="Use historical temperature, hot-water load, and calendar features")
    args = parser.parse_args()
    data_path = DATA_AUX if args.use_aux_features else DATA
    output_dir = OUT_AUX if args.use_aux_features else OUT
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    input_names = INPUT_FEATURES if args.use_aux_features else TARGET_COLUMNS
    print(f"device={device}; data={data_path}; inputs={input_names}; targets={TARGET_COLUMNS}; "
          f"rows={sum(1 for _ in data_path.open(encoding='utf-8-sig')) - 1}", flush=True)
    raw_path = output_dir / "japan_2019_by_seed.csv"
    rows = pd.read_csv(raw_path).to_dict("records") if args.resume and raw_path.exists() else []
    completed = {(str(r["Model"]), int(r["horizon"]), int(r["seed"])) for r in rows}
    for model_name in args.models:
        if model_name not in MODELS:
            raise ValueError(f"Unknown model {model_name!r}; choices: {list(MODELS)}")
        for horizon in args.horizons:
            for seed in args.seeds:
                if (model_name, horizon, seed) in completed:
                    print(f"SKIP completed {model_name} horizon={horizon} seed={seed}", flush=True)
                    continue
                print(f"START {model_name} horizon={horizon} seed={seed}", flush=True)
                rows.extend(train_run(model_name, horizon, seed, device, args.epochs,
                                      args.batch_size, args.use_aux_features))
                completed.add((model_name, horizon, seed))
                pd.DataFrame(rows).to_csv(raw_path, index=False, encoding="utf-8-sig")
                summarize(rows).to_csv(output_dir / "japan_2019_summary.csv", index=False, encoding="utf-8-sig")
                print(f"SAVED {raw_path}", flush=True)
    print(f"Completed {len(args.models) * len(args.horizons) * len(args.seeds)} runs", flush=True)


if __name__ == "__main__":
    main()

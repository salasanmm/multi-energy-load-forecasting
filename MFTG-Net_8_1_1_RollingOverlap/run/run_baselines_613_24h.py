import argparse
import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = ROOT / "baselines"
sys.path.insert(0, str(BASELINE_ROOT))
from test_model import Autoformer, DLinear, PatchTST, TimeMixer


DATA = ROOT / "data" / "dataset_input_jiuzheng.csv"
OUTPUT = ROOT / "output6_1_3" / "main_experiment" / "baseline_runs"
TARGETS = ("electricity", "cooling", "heating")
MODELS = {
    "TimeMixer": TimeMixer.Model,
    "Autoformer": Autoformer.Model,
    "DLinear": DLinear.Model,
    "PatchTST": PatchTST.Model,
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def config_for(model_name, horizon):
    cfg = SimpleNamespace(
        task_name="long_term_forecast", is_training=1, model_id="asu_seasonal_613",
        seq_len=168, pred_len=horizon, pred_in_len=horizon, seq_out_len=horizon,
        label_len=48, enc_in=3, dec_in=3, c_out=3, individual=True,
        d_model=64, d_ff=64, n_heads=4, e_layers=2, d_layers=1,
        dropout=0.1, fc_dropout=0.1, head_dropout=0.1, factor=1,
        activation="gelu", output_attention=False, distil=True,
        embed="fixed", freq="h", moving_avg=25, use_norm=1,
        channel_independence=0, down_sampling_window=2, down_sampling_layers=2,
        down_sampling_method="avg", decomp_method="moving_avg", top_k=5,
        num_kernels=4, patch_len=24, stride=8, padding_patch="end",
        revin=1, affine=0, subtract_last=0, head_type="flatten", pretrain_head=False,
        num_class=3,
    )
    if model_name == "TimeMixer":
        cfg.e_layers = 3
        cfg.moving_avg = 3
    return MODELS[model_name](cfg)


def forward(model_name, model, adapter, x, horizon):
    x = adapter(x)
    if model_name == "PatchTST":
        return model(x.unsqueeze(1).transpose(2, 3))
    if model_name == "TimeMixer":
        return model(x)
    if model_name == "DLinear":
        return model(x, None, None, None)
    batch = x.shape[0]
    x_mark = x.new_zeros((batch, x.shape[1], 4))
    x_dec = torch.cat((x[:, -48:, :], x.new_zeros((batch, horizon, 3))), dim=1)
    x_mark_dec = x.new_zeros((batch, 48 + horizon, 4))
    return model(x, x_mark, x_dec, x_mark_dec)


def windows(inputs, targets, origin_start, origin_end, seq_len, horizon):
    starts = np.arange(origin_start, origin_end - horizon + 1, dtype=np.int64)
    xs = np.stack([inputs[t - seq_len:t] for t in starts])
    ys = np.stack([targets[t:t + horizon] for t in starts])
    return torch.from_numpy(xs).float(), torch.from_numpy(ys).float(), starts


def overlap_average(predictions, starts, actual, start, end):
    pred_sum = np.zeros((end - start, actual.shape[1]), dtype=np.float64)
    count = np.zeros(end - start, dtype=np.int64)
    for pred, origin in zip(predictions, starts):
        lo, hi = max(int(origin), start), min(int(origin) + len(pred), end)
        if lo < hi:
            pred_sum[lo - start:hi - start] += pred[lo - int(origin):hi - int(origin)]
            count[lo - start:hi - start] += 1
    keep = count > 0
    return actual[start:end][keep], pred_sum[keep] / count[keep, None], keep


def estimate_flops(model_name, model, adapter, device, horizon, n_features):
    try:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        sample = torch.zeros((1, 168, n_features), device=device)
        with torch.profiler.profile(activities=activities, with_flops=True) as prof:
            with torch.no_grad():
                forward(model_name, model, adapter, sample, horizon)
        return int(sum(event.flops or 0 for event in prof.events()))
    except Exception:
        return ""


def train_run(model_name, horizon, seed, device, epochs, batch_size):
    set_seed(seed)
    frame = pd.read_csv(DATA)
    raw = frame.to_numpy(dtype=np.float64)
    targets = raw[:, :3]
    n, n_features = raw.shape
    train_end, valid_end = int(0.6 * n), int(0.7 * n)
    input_scale = np.max(np.abs(raw[:train_end]), axis=0)
    input_scale[input_scale == 0] = 1.0
    target_scale = input_scale[:3]
    inputs = (raw / input_scale).astype(np.float32)
    outputs = (targets / target_scale).astype(np.float32)

    seq_len = 168
    tx, ty, _ = windows(inputs, outputs, seq_len, train_end, seq_len, horizon)
    vx, vy, vstarts = windows(inputs, outputs, train_end, valid_end, seq_len, horizon)
    qx, qy, qstarts = windows(inputs, outputs, valid_end, n, seq_len, horizon)
    if min(len(tx), len(vx), len(qx)) == 0:
        raise ValueError(f"Empty split window for {model_name} seed {seed}")

    model = config_for(model_name, horizon).to(device)
    adapter = torch.nn.Linear(n_features, 3).to(device)
    with torch.no_grad():
        adapter.weight.zero_()
        adapter.bias.zero_()
        adapter.weight[:, :3] = torch.eye(3, device=device)
    params = sum(p.numel() for p in list(model.parameters()) + list(adapter.parameters()) if p.requires_grad)
    flops = estimate_flops(model_name, model, adapter, device, horizon, n_features)
    loader = DataLoader(TensorDataset(tx, ty), batch_size=batch_size, shuffle=True, drop_last=False)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(adapter.parameters()),
                                 lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)
    best_val, best_epoch, best_state = math.inf, 0, None
    epoch_times = []

    for epoch in range(1, epochs + 1):
        model.train()
        adapter.train()
        tic = time.perf_counter()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = forward(model_name, model, adapter, xb, horizon)
            error = torch.abs(pred - yb)
            loss = 100.0 * (error / (torch.abs(yb) + 1e-3)).mean() + 0.0005 * error.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(adapter.parameters()), 5.0)
            optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_times.append(time.perf_counter() - tic)

        model.eval()
        adapter.eval()
        val_predictions = []
        with torch.no_grad():
            for offset in range(0, len(vx), batch_size):
                pred = forward(model_name, model, adapter, vx[offset:offset + batch_size].to(device), horizon)
                val_predictions.append(pred.cpu().numpy())
        val_true, val_pred, _ = overlap_average(np.concatenate(val_predictions), vstarts, outputs,
                                                 train_end, valid_end)
        val_true, val_pred = val_true * target_scale, val_pred * target_scale
        val_mae = float(np.abs(val_pred - val_true).mean(axis=0).mean())
        scheduler.step(val_mae)
        if val_mae < best_val:
            best_val, best_epoch = val_mae, epoch
            best_state = (
                {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()},
            )
        print(f"{model_name} H{horizon} seed{seed} epoch {epoch}/{epochs} valid rolling MAE {val_mae:.6f}",
              flush=True)

    model.load_state_dict(best_state[0])
    adapter.load_state_dict(best_state[1])
    model.eval()
    adapter.eval()
    test_predictions = []
    with torch.no_grad():
        for offset in range(0, len(qx), batch_size):
            pred = forward(model_name, model, adapter, qx[offset:offset + batch_size].to(device), horizon)
            test_predictions.append(pred.cpu().numpy())
    test_true, test_pred, keep = overlap_average(np.concatenate(test_predictions), qstarts,
                                                  outputs, valid_end, n)
    test_true, test_pred = test_true * target_scale, test_pred * target_scale

    run_dir = OUTPUT / f"{model_name}_24_168_613_seed{seed}" / "result"
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "all_y_true_overlap_average.npy", test_true)
    np.save(run_dir / "all_predict_value_overlap_average.npy", test_pred)
    metadata = {
        "Model": model_name, "horizon": horizon, "seed": seed,
        "train_ratio": 0.6, "validation_ratio": 0.1, "test_ratio": 0.3,
        "train_end_row_exclusive": train_end, "validation_end_row_exclusive": valid_end,
        "test_target_rows": [valid_end, n], "input_columns": list(frame.columns),
        "target_columns": list(frame.columns[:3]), "input_fusion": f"Linear({n_features}, 3)",
        "selected_epoch": best_epoch, "epoch_time_sec_mean": float(np.mean(epoch_times)),
        "Params": params, "FLOPs_forward": flops,
        "test_target_timestamps": int(keep.sum()),
        "test_rollout_protocol": "hourly rolling origins, averaged by target timestamp",
        "test_used_for_selection": False,
    }
    (run_dir / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[2020, 2021, 2022])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records_path = OUTPUT / "baseline_runs_by_seed.json"
    records = json.loads(records_path.read_text(encoding="utf-8")) if args.resume and records_path.exists() else []
    completed = {(r["Model"], int(r["seed"])) for r in records}
    for model_name in args.models:
        if model_name not in MODELS:
            raise ValueError(f"Unknown baseline {model_name}; choose from {list(MODELS)}")
        for seed in args.seeds:
            if (model_name, seed) in completed:
                print(f"SKIP completed {model_name} seed={seed}", flush=True)
                continue
            print(f"START {model_name} horizon=24 seed={seed}", flush=True)
            records.append(train_run(model_name, 24, seed, device, args.epochs, args.batch_size))
            completed.add((model_name, seed))
            records_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    pd.DataFrame(records).to_csv(OUTPUT / "baseline_runs_by_seed.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()

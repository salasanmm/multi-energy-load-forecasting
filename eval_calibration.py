import numpy as np
import torch
import os
import importlib.util

from get_config import get_config
_model_file = os.path.join(os.path.dirname(__file__), "test_model", "MFGT-Net.py")
_model_spec = importlib.util.spec_from_file_location("mftg_net_impl", _model_file)
if _model_spec is None or _model_spec.loader is None:
    raise ImportError(f"Cannot load proposed model from {_model_file}")
_model_module = importlib.util.module_from_spec(_model_spec)
_model_spec.loader.exec_module(_model_module)
Model = _model_module.Model
from train import args, device, plow, channel_metrics, format_metrics, apply_refiner_mode
from util import DataLoaderS


def affine_per_channel(valid_true, valid_pred, test_pred):
    out = test_pred.copy()
    for c in range(valid_true.shape[-1]):
        x = valid_pred[:, :, c].reshape(-1)
        y = valid_true[:, :, c].reshape(-1)
        a, b = np.polyfit(x, y, 1)
        out[:, :, c] = test_pred[:, :, c] * a + b
    return out


def affine_per_horizon_channel(valid_true, valid_pred, test_pred, clip=(0.85, 1.15)):
    out = test_pred.copy()
    for t in range(valid_true.shape[1]):
        for c in range(valid_true.shape[2]):
            x = valid_pred[:, t, c].reshape(-1)
            y = valid_true[:, t, c].reshape(-1)
            a, b = np.polyfit(x, y, 1)
            if clip is not None:
                a = float(np.clip(a, clip[0], clip[1]))
            out[:, t, c] = test_pred[:, t, c] * a + b
    return out


def bias_per_horizon_channel(valid_true, valid_pred, test_pred, method="median"):
    residual = valid_true - valid_pred
    if method == "mean":
        bias = residual.mean(axis=0, keepdims=True)
    else:
        bias = np.median(residual, axis=0, keepdims=True)
    return test_pred + bias


def blend_candidates(valid_true, valid_pred, test_pred, candidates):
    best = None
    for name, cand_valid, cand_test in candidates:
        for alpha in np.linspace(0.0, 1.0, 21):
            blended_valid = valid_pred * (1.0 - alpha) + cand_valid * alpha
            metrics = channel_metrics(valid_true, blended_valid)
            score = metrics["total"]["mae"]
            if best is None or score < best["score"]:
                blended_test = test_pred * (1.0 - alpha) + cand_test * alpha
                best = {
                    "name": name,
                    "alpha": float(alpha),
                    "score": float(score),
                    "test_pred": blended_test,
                }
    return best


def main():
    config = get_config("TimeMixer")
    data = DataLoaderS(
        args.data,
        getattr(config, "train_ratio", 0.5),
        getattr(config, "valid_ratio", 0.2),
        device,
        args.horizon,
        args.seq_in_len,
        args.normalize,
        exclude_columns=getattr(config, "exclude_columns", None),
    )
    checkpoint = torch.load(args.save, map_location=device)
    model = Model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    mode = checkpoint.get("mode", "pgf_crd_ph")
    apply_refiner_mode(model, mode)

    valid_true, valid_pred = plow(data, data.valid[0], data.valid[1][:, :, :3], model, args.batch_size, mode=mode)
    test_true, test_pred = plow(data, data.test[0], data.test[1][:, :, :3], model, args.batch_size, mode=mode)
    valid_true = valid_true.cpu().numpy()
    valid_pred = valid_pred.cpu().numpy()
    test_true = test_true.cpu().numpy()
    test_pred = test_pred.cpu().numpy()

    print(format_metrics("base valid", channel_metrics(valid_true, valid_pred)))
    print(format_metrics("base test", channel_metrics(test_true, test_pred)))

    aff_c_test = affine_per_channel(valid_true, valid_pred, test_pred)
    aff_c_valid = affine_per_channel(valid_true, valid_pred, valid_pred)
    aff_h_test = affine_per_horizon_channel(valid_true, valid_pred, test_pred)
    aff_h_valid = affine_per_horizon_channel(valid_true, valid_pred, valid_pred)
    bias_med_test = bias_per_horizon_channel(valid_true, valid_pred, test_pred, "median")
    bias_med_valid = bias_per_horizon_channel(valid_true, valid_pred, valid_pred, "median")
    bias_mean_test = bias_per_horizon_channel(valid_true, valid_pred, test_pred, "mean")
    bias_mean_valid = bias_per_horizon_channel(valid_true, valid_pred, valid_pred, "mean")

    candidates = [
        ("affine_channel", aff_c_valid, aff_c_test),
        ("affine_horizon", aff_h_valid, aff_h_test),
        ("bias_median_horizon", bias_med_valid, bias_med_test),
        ("bias_mean_horizon", bias_mean_valid, bias_mean_test),
    ]
    for name, cand_valid, cand_test in candidates:
        print(format_metrics(f"{name} valid", channel_metrics(valid_true, cand_valid)))
        print(format_metrics(f"{name} test", channel_metrics(test_true, cand_test)))

    best = blend_candidates(valid_true, valid_pred, test_pred, candidates)
    print(f"best calibration by valid: {best['name']} alpha={best['alpha']:.2f} valid_mae={best['score']:.4f}")
    print(format_metrics("best calibration test", channel_metrics(test_true, best["test_pred"])))


if __name__ == "__main__":
    main()

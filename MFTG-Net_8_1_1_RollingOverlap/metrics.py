import math

import numpy as np


def MAE(y_true, y_pre):
    y_true = (y_true).reshape((-1, 1))
    y_pre = (y_pre).reshape((-1, 1))
    re = np.abs(y_true - y_pre).mean()
    return re


def RMSE(y_true, y_pre):
    y_true = (y_true).reshape((-1, 1))
    y_pre = (y_pre).reshape((-1, 1))
    re = math.sqrt(((y_true - y_pre) ** 2).mean())
    return re


def MAPE(y_true, y_pre):
    y_true = (y_true).reshape((-1, 1))
    y_pre = (y_pre).reshape((-1, 1))

    # e = (y_true + y_pre) / 2 + 1e-2
    # re = (np.abs(y_true - y_pre) / (np.abs(y_true) + e)).mean()
    re = np.mean(np.abs((y_true - y_pre) / y_true)) * 100

    return re

def MSE(y_true, y_pre):
    y_true = (y_true).reshape((-1, 1))
    y_pre = (y_pre).reshape((-1, 1))
    re = ((y_true - y_pre) ** 2).mean()
    return re


def overlap_average(y_true, y_pred):
    """Average origin forecasts by target timestamp for a rolling forecast."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape or y_true.ndim != 3:
        raise ValueError("Expected matching [origins, horizon, channels] arrays")

    origins, horizon, channels = y_true.shape
    target_count = origins + horizon - 1
    pred_sum = np.zeros((target_count, channels), dtype=np.float64)
    true_sum = np.zeros((target_count, channels), dtype=np.float64)
    counts = np.zeros(target_count, dtype=np.int64)
    for origin in range(origins):
        target_slice = slice(origin, origin + horizon)
        pred_sum[target_slice] += y_pred[origin]
        true_sum[target_slice] += y_true[origin]
        counts[target_slice] += 1

    return true_sum / counts[:, None], pred_sum / counts[:, None]


def overlap_channel_metrics(y_true, y_pred):
    """Score the chronological, target-timestamp series after overlap averaging."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape or y_true.ndim != 2:
        raise ValueError("Expected matching [target timestamps, channels] arrays")

    names = ("electricity", "cooling", "heating")
    result = {}
    for index, name in enumerate(names):
        actual = y_true[:, index]
        predicted = y_pred[:, index]
        error = predicted - actual
        actual_delta = np.diff(actual)
        pred_delta = np.diff(predicted)
        with np.errstate(divide="ignore", invalid="ignore"):
            mape = float(np.mean(np.abs(error / actual)) * 100)
        actual_std = actual.std()
        pred_std = predicted.std()
        corr = float(np.mean((actual - actual.mean()) * (predicted - predicted.mean()))
                     / (actual_std * pred_std)) if actual_std and pred_std else 0.0
        result[name] = {
            "mae": float(np.mean(np.abs(error))),
            "mape": mape,
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "ds": float(np.mean((actual_delta * pred_delta) >= 0) * 100)
            if actual_delta.size else 0.0,
            "corr": corr,
        }
    result["total"] = {
        metric: float(np.mean([result[name][metric] for name in names]))
        for metric in ("mae", "mape", "rmse", "ds", "corr")
    }
    return result

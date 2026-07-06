from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import mutual_info_score


BASE_DIR = Path(r"E:\负荷预测（原件）(2)")
DATA_PATH = BASE_DIR / "data" / "dataset_input_jiuzheng.csv"
OUTPUT_PATH = BASE_DIR / "MIC图.png"

LOAD_COLUMNS = ["KW", "CHWTON", "HTmmBTU"]
FEATURE_COLUMNS = [
    "temperature",
    "dew_point_temperature",
    "station_level_pressure",
    "sea_level_pressure",
    "wet_bulb_temperature",
    "altimeter",
    "DayOfYear_cos",
    "Combined mmBTU",
    "GHG",
]

LOAD_LABELS = ["KW", "CHWTON", "HTmmBTU"]
LOAD_COLORS = ["#68b36b", "#4b8fd8", "#e88735"]

FEATURE_LABELS = {
    "temperature": "temperature",
    "dew_point_temperature": "dew_point\ntemperature",
    "station_level_pressure": "station_level\npressure",
    "sea_level_pressure": "sea_level\npressure",
    "wet_bulb_temperature": "wet_bulb\ntemperature",
    "altimeter": "altimeter",
    "DayOfYear_cos": "DayOfYear\ncos",
    "Combined mmBTU": "Combined\nmmBTU",
    "GHG": "GHG",
}


def bin_continuous(values: np.ndarray, bins: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.array([], dtype=int)
    unique_count = np.unique(values).size
    bins = max(2, min(int(bins), unique_count))
    if bins < 2:
        return np.zeros(values.shape[0], dtype=int)
    try:
        codes = pd.qcut(values, q=bins, labels=False, duplicates="drop")
        codes = pd.Series(codes).fillna(0).astype(int).to_numpy()
        return codes
    except Exception:
        edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
        if edges.size <= 2:
            return np.zeros(values.shape[0], dtype=int)
        return np.digitize(values, edges[1:-1], right=True).astype(int)


def mic_approx(x: np.ndarray, y: np.ndarray, max_bins: int = 9) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[mask], dtype=float)
    y = np.asarray(y[mask], dtype=float)
    if x.size < 10:
        return 0.0

    best = 0.0
    max_bins = max(2, min(int(max_bins), int(np.sqrt(x.size)), 12))

    x_cache: dict[int, np.ndarray] = {}
    y_cache: dict[int, np.ndarray] = {}

    for bx in range(2, max_bins + 1):
        if bx not in x_cache:
            x_cache[bx] = bin_continuous(x, bx)
        xb = x_cache[bx]
        ux = np.unique(xb).size
        if ux < 2:
            continue
        for by in range(2, max_bins + 1):
            if by not in y_cache:
                y_cache[by] = bin_continuous(y, by)
            yb = y_cache[by]
            uy = np.unique(yb).size
            if uy < 2:
                continue
            mi = mutual_info_score(xb, yb)
            norm = np.log(min(ux, uy))
            if norm > 0:
                score = mi / norm
                if score > best:
                    best = score

    return float(min(best, 1.0))


def compute_matrix(df: pd.DataFrame) -> np.ndarray:
    matrix = np.zeros((len(LOAD_COLUMNS), len(FEATURE_COLUMNS)), dtype=float)
    for i, load_col in enumerate(LOAD_COLUMNS):
        load_values = df[load_col].to_numpy(dtype=float)
        for j, feat_col in enumerate(FEATURE_COLUMNS):
            feat_values = df[feat_col].to_numpy(dtype=float)
            matrix[i, j] = mic_approx(load_values, feat_values)
    return matrix


def format_theta_label(theta: float) -> tuple[float, str]:
    angle = np.degrees(theta)
    rotation = -angle
    if rotation < -90:
        rotation += 180
    if rotation > 90:
        rotation -= 180
    return rotation, "center"


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "axes.unicode_minus": False,
        }
    )

    df = pd.read_csv(DATA_PATH)
    data = compute_matrix(df)

    n_features = len(FEATURE_COLUMNS)
    angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False)
    sector_width = 2 * np.pi / n_features
    bar_width = sector_width * 0.23
    offsets = np.array([-bar_width * 1.2, 0.0, bar_width * 1.2])
    colors = LOAD_COLORS

    fig = plt.figure(figsize=(8.6, 8.6), dpi=300)
    ax = fig.add_subplot(111, projection="polar")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    radius_max = 0.60
    ax.set_ylim(0, radius_max)
    yticks = np.array([0.15, 0.30, 0.45, 0.60])
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{v:.2f}".rstrip("0").rstrip(".") for v in yticks], fontsize=11, color="#111111")
    ax.yaxis.grid(True, color="#cfcfcf", linewidth=0.8)
    ax.xaxis.grid(True, color="#d9d9d9", linewidth=0.8)
    ax.spines["polar"].set_color("#bfbfbf")
    ax.spines["polar"].set_linewidth(1.0)

    ax.tick_params(colors="#111111")

    for idx, (load_label, color) in enumerate(zip(LOAD_LABELS, colors)):
        vals = data[idx]
        ax.bar(
            angles + offsets[idx],
            vals,
            width=bar_width,
            bottom=0.0,
            color=color,
            alpha=0.86,
            edgecolor=(1, 1, 1, 0.85),
            linewidth=0.5,
            align="center",
            zorder=3,
        )

    ax.set_xticks(angles)
    ax.set_xticklabels([""] * n_features)

    label_radius = radius_max + 0.035
    for theta, label in zip(angles, FEATURE_COLUMNS):
        rotation, ha = format_theta_label(theta)
        current_radius = label_radius
        if label == "temperature":
            current_radius = radius_max + 0.015
        ax.text(
            theta,
            current_radius,
            FEATURE_LABELS[label],
            ha=ha,
            va="center",
            fontsize=9.5,
            fontweight="bold",
            rotation=rotation,
            rotation_mode="anchor",
            color="#000000",
        )

    ax.set_title("MIC", fontsize=19, fontweight="bold", pad=30, color="#000000")

    legend_handles = [
        Patch(facecolor=colors[i], edgecolor="none", alpha=0.78, label=LOAD_LABELS[i]) for i in range(3)
    ]
    ax.legend(
        handles=legend_handles,
        title="Load",
        loc="upper right",
        bbox_to_anchor=(1.10, 1.00),
        frameon=False,
        fontsize=11,
        title_fontsize=13,
    )

    ax.set_position([0.16, 0.14, 0.56, 0.56])
    fig.subplots_adjust(left=0.10, right=0.84, top=0.90, bottom=0.08)
    fig.savefig(OUTPUT_PATH, dpi=400, bbox_inches="tight", pad_inches=0.32, facecolor="white")


if __name__ == "__main__":
    main()

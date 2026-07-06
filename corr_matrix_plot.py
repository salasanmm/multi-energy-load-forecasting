from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from sklearn.metrics import mutual_info_score


BASE_DIR = Path(r"E:\负荷预测（原件）(2)")
DATA_PATH = BASE_DIR / "data" / "dataset_input_jiuzheng.csv"
OUTPUT_PATH = BASE_DIR / "相关矩阵图.png"

LOADS = ["KW", "CHWTON", "HTmmBTU"]
FEATURES = [
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
LABELS = LOADS + FEATURES

LABEL_MAP = {
    "KW": "KW",
    "CHWTON": "CHWTON",
    "HTmmBTU": "HTmmBTU",
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

PCC_CMAP = LinearSegmentedColormap.from_list("pcc_soft", ["#2c7bb6", "#f7f7f7", "#d7191c"])
MIC_CMAP = LinearSegmentedColormap.from_list("mic_soft", ["#fde0dd", "#fa9fb5", "#dd4a7f", "#980043"])
KDE_FILL = "#b9d9b4"
KDE_LINE = "#4c8a52"
GRID_COLOR = "#d7d7d7"
TEXT_COLOR = "#111111"


def pcc(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    return float(stats.pearsonr(x[mask], y[mask])[0])


def bin_continuous(values: np.ndarray, bins: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.array([], dtype=int)
    bins = max(2, min(int(bins), np.unique(values).size))
    try:
        codes = pd.qcut(values, q=bins, labels=False, duplicates="drop")
        return pd.Series(codes).fillna(0).astype(int).to_numpy()
    except Exception:
        edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
        if edges.size <= 2:
            return np.zeros(values.shape[0], dtype=int)
        return np.digitize(values, edges[1:-1], right=True).astype(int)


def mic_approx(x: np.ndarray, y: np.ndarray, max_bins: int = 8) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[mask], dtype=float)
    y = np.asarray(y[mask], dtype=float)
    if x.size < 10:
        return 0.0

    best = 0.0
    max_bins = max(2, min(max_bins, int(np.sqrt(x.size)), 12))
    x_cache: dict[int, np.ndarray] = {}
    y_cache: dict[int, np.ndarray] = {}

    for bx in range(2, max_bins + 1):
        xb = x_cache.setdefault(bx, bin_continuous(x, bx))
        ux = np.unique(xb).size
        if ux < 2:
            continue
        for by in range(2, max_bins + 1):
            yb = y_cache.setdefault(by, bin_continuous(y, by))
            uy = np.unique(yb).size
            if uy < 2:
                continue
            mi = mutual_info_score(xb, yb)
            denom = np.log(min(ux, uy))
            if denom > 0:
                best = max(best, mi / denom)
    return float(min(best, 1.0))


def kde_density(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 3:
        return np.array([]), np.array([])
    std = np.std(values)
    if std <= 0:
        return np.array([]), np.array([])
    x = np.linspace(values.min() - 0.1 * std, values.max() + 0.1 * std, 256)
    y = stats.gaussian_kde(values)(x)
    y = y / y.max() if y.max() > 0 else y
    return x, y


def panel_style(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
        spine.set_linewidth(0.7)
    ax.set_facecolor("white")


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "axes.unicode_minus": False,
        }
    )

    df = pd.read_csv(DATA_PATH)
    data = df[LABELS]
    n = len(LABELS)

    pcc_mat = np.zeros((n, n), dtype=float)
    mic_mat = np.zeros((n, n), dtype=float)

    for i, a in enumerate(LABELS):
        xa = data[a].to_numpy(dtype=float)
        for j, b in enumerate(LABELS):
            xb = data[b].to_numpy(dtype=float)
            pcc_mat[i, j] = pcc(xa, xb)
            mic_mat[i, j] = mic_approx(xa, xb)

    fig = plt.figure(figsize=(18, 18), dpi=320)
    gs = fig.add_gridspec(n, n, wspace=0.025, hspace=0.025)

    for i in range(n):
        for j in range(n):
            ax = fig.add_subplot(gs[i, j])
            panel_style(ax)

            if i == j:
                x, y = kde_density(data[LABELS[i]].to_numpy(dtype=float))
                if x.size:
                    ax.fill_between(x, 0, y, color=KDE_FILL, alpha=0.55)
                    ax.plot(x, y, color=KDE_LINE, linewidth=1.0)
                ax.set_xlim(left=x.min() if x.size else 0, right=x.max() if x.size else 1)
                ax.set_ylim(0, 1.05)
                ax.text(
                    0.5,
                    0.84,
                    LABEL_MAP[LABELS[i]],
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=12.2,
                    fontweight="bold",
                    color=TEXT_COLOR,
                )
            elif i > j:
                value = mic_mat[i, j]
                cmap = MIC_CMAP
                vmin, vmax = (0, 1)
                ax.imshow([[value]], cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
                ax.text(
                    0.5,
                    0.5,
                    f"{value:.2f}",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=9.4,
                    fontweight="bold",
                    color=TEXT_COLOR,
                )
            else:
                value = pcc_mat[i, j]
                cmap = PCC_CMAP
                vmin, vmax = (-1, 1)
                ax.imshow([[value]], cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
                ax.text(
                    0.5,
                    0.5,
                    f"{value:.2f}",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=11.8,
                    fontweight="bold",
                    color="#000000",
                )

            if i == n - 1:
                ax.set_xlabel(LABEL_MAP[LABELS[j]], fontsize=11.4, fontweight="bold", labelpad=6, color="#000000")
            if j == 0:
                ax.set_ylabel(LABEL_MAP[LABELS[i]], fontsize=11.4, fontweight="bold", labelpad=7, color="#000000")

    cax1 = fig.add_axes([0.925, 0.56, 0.018, 0.29])
    cax2 = fig.add_axes([0.925, 0.15, 0.018, 0.29])
    sm1 = plt.cm.ScalarMappable(cmap=PCC_CMAP, norm=plt.Normalize(-1, 1))
    sm1.set_array([])
    cb1 = fig.colorbar(sm1, cax=cax1)
    cb1.set_ticks([-1, -0.5, 0, 0.5, 1])
    cb1.ax.tick_params(labelsize=12, colors="#000000", width=1.0)
    cb1.set_label("PCC", fontsize=14, fontweight="bold", color="#000000")

    sm2 = plt.cm.ScalarMappable(cmap=MIC_CMAP, norm=plt.Normalize(0, 1))
    sm2.set_array([])
    cb2 = fig.colorbar(sm2, cax=cax2)
    cb2.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cb2.ax.tick_params(labelsize=12, colors="#000000", width=1.0)
    cb2.set_label("MIC", fontsize=14, fontweight="bold", color="#000000")

    fig.suptitle("Correlation Matrix", fontsize=24, fontweight="bold", y=0.995, color="#000000")
    fig.savefig(OUTPUT_PATH, dpi=520, bbox_inches="tight", pad_inches=0.22, facecolor="white")


if __name__ == "__main__":
    main()

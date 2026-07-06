from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap


BASE_DIR = Path(r"E:\负荷预测（原件）(2)")
OUTPUT_DIR = BASE_DIR / "output"
FIG_PATH = BASE_DIR / "预测对比zoom图.png"
FIG_PATH_WPS = BASE_DIR / "预测对比zoom图_WPS清晰排版版.png"
FIG_PATH_PDF = BASE_DIR / "预测对比zoom图_WPS清晰排版版.pdf"
FIG_PATH_SVG = BASE_DIR / "预测对比zoom图_WPS清晰排版版.svg"

PRED_LEN = 96
SEQ_LEN = 168
CHANNELS = [
    ("Electricity Load", 0),
    ("Cooling Load", 1),
    ("Heating Load", 2),
]

MODEL_DIRS = {
    "Proposed": "TimeMixer_96_168",
    "TimesNet": "TimesNet_96_168",
    "PatchTST": "PatchTST_96_168",
    "TSMixer": "TSMixer_96_168",
    "FITS": "FITS_96_168",
    "CFC": "CFC_96_168",
}

STYLE = {
    "Ground Truth": {"color": "#0b0b0b", "lw": 1.8, "ls": "-", "alpha": 1.0},
    "Proposed": {"color": "#e4572e", "lw": 1.55, "ls": "-", "alpha": 0.98},
    "TimesNet": {"color": "#2a9d8f", "lw": 1.05, "ls": "--", "alpha": 0.9},
    "PatchTST": {"color": "#3a86ff", "lw": 1.05, "ls": "-.", "alpha": 0.9},
    "TSMixer": {"color": "#8e6cc8", "lw": 1.0, "ls": ":", "alpha": 0.9},
    "FITS": {"color": "#7bc96f", "lw": 1.0, "ls": "--", "alpha": 0.86},
    "CFC": {"color": "#f4a261", "lw": 1.0, "ls": "-.", "alpha": 0.86},
}

ERROR_CMAP = LinearSegmentedColormap.from_list(
    "error_blue_red",
    ["#08306b", "#4292c6", "#f7f7f7", "#fdae61", "#b2182b"],
)


def load_results() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    truth_path = OUTPUT_DIR / MODEL_DIRS["Proposed"] / "result" / "all_y_true.npy"
    truth = np.load(truth_path)
    preds: dict[str, np.ndarray] = {}
    for name, folder in MODEL_DIRS.items():
        pred_path = OUTPUT_DIR / folder / "result" / "all_predict_value.npy"
        if pred_path.exists():
            preds[name] = np.load(pred_path)
    return truth, preds


def select_sample(truth: np.ndarray, proposed: np.ndarray) -> int:
    ranges = np.ptp(truth[:, :, :3], axis=1)
    denom = np.nanmedian(ranges, axis=0) + 1e-6
    range_score = (ranges / denom).sum(axis=1)
    err = np.mean(np.abs(proposed[:, :, :3] - truth[:, :, :3]), axis=(1, 2))
    err_score = err / (np.nanmedian(err) + 1e-6)
    score = range_score - 0.9 * err_score
    lower, upper = np.percentile(range_score, [55, 92])
    candidates = np.where((range_score >= lower) & (range_score <= upper))[0]
    if candidates.size == 0:
        return int(np.argmax(score))
    return int(candidates[np.argmax(score[candidates])])


def select_zoom_window(y_true: np.ndarray, y_pred: np.ndarray, width: int = 10) -> tuple[int, int]:
    n = len(y_true)
    width = min(width, n)
    best_start = 0
    best_score = -np.inf
    full_range = np.ptp(y_true) + 1e-6
    full_err = np.mean(np.abs(y_pred - y_true)) + 1e-6
    for start in range(0, n - width + 1):
        end = start + width
        seg_true = y_true[start:end]
        seg_pred = y_pred[start:end]
        score = np.ptp(seg_true) / full_range + 0.45 * np.mean(np.abs(seg_pred - seg_true)) / full_err
        if score > best_score:
            best_score = score
            best_start = start
    return best_start, best_start + width - 1


def select_shared_zoom_window(y_true: np.ndarray, y_pred: np.ndarray, width: int = 10) -> tuple[int, int]:
    n = y_true.shape[0]
    width = min(width, n)
    best_start = 0
    best_score = -np.inf
    full_range = np.ptp(y_true, axis=0) + 1e-6
    full_err = np.mean(np.abs(y_pred - y_true), axis=0) + 1e-6
    for start in range(0, n - width + 1):
        end = start + width
        seg_true = y_true[start:end, :]
        seg_pred = y_pred[start:end, :]
        range_score = np.mean(np.ptp(seg_true, axis=0) / full_range)
        error_score = np.mean(np.abs(seg_pred - seg_true) / full_err)
        score = range_score + 0.45 * error_score
        if score > best_score:
            best_score = score
            best_start = start
    return best_start, best_start + width - 1


def add_zoom_box(ax, x0: int, x1: int, y_values: np.ndarray) -> None:
    y_min = float(np.min(y_values[x0 : x1 + 1]))
    y_max = float(np.max(y_values[x0 : x1 + 1]))
    pad = (y_max - y_min) * 0.12 + 1e-6
    rect = Rectangle(
        (x0, y_min - pad),
        x1 - x0,
        (y_max - y_min) + 2 * pad,
        fill=False,
        edgecolor="#d95f5f",
        linewidth=1.0,
        linestyle="--",
        alpha=0.9,
    )
    ax.add_patch(rect)


def plot_series(ax, x: np.ndarray, series: dict[str, np.ndarray], names: list[str]) -> None:
    for name in names:
        style = STYLE[name]
        ax.plot(
            x,
            series[name],
            label=name,
            color=style["color"],
            linewidth=style["lw"],
            linestyle=style["ls"],
            alpha=style["alpha"],
        )


def build_error_heatmap(
    truth: np.ndarray,
    pred: np.ndarray,
    center_idx: int,
    channel: int,
    step_start: int,
    step_end: int,
    n_samples: int = 9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    half = n_samples // 2
    start = int(np.clip(center_idx - half, 0, truth.shape[0] - n_samples))
    sample_indices = np.arange(start, start + n_samples)
    step_indices = np.arange(step_start, step_end + 1)
    errors = np.abs(pred[sample_indices[:, None], step_indices[None, :], channel] - truth[sample_indices[:, None], step_indices[None, :], channel])
    return errors, sample_indices, step_indices


def plot_error_heatmap(ax, errors: np.ndarray, sample_indices: np.ndarray, step_indices: np.ndarray, title: str, vmax: float) -> None:
    im = ax.imshow(errors, aspect="auto", cmap=ERROR_CMAP, vmin=0, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=7.8, fontweight="bold", pad=3)
    ax.set_xticks(np.arange(len(step_indices)))
    ax.set_xticklabels(np.arange(1, len(step_indices) + 1))
    ax.set_yticks([0, len(sample_indices) // 2, len(sample_indices) - 1])
    ax.set_yticklabels(sample_indices[[0, len(sample_indices) // 2, len(sample_indices) - 1]])
    ax.tick_params(axis="both", labelsize=6.8, colors="#0b0b0b", width=0.8, length=2.4)
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#111111")
    return im


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    truth, preds = load_results()
    sample_idx = select_sample(truth, preds["Proposed"])
    shared_z0, shared_z1 = select_shared_zoom_window(
        truth[sample_idx, :, : len(CHANNELS)],
        preds["Proposed"][sample_idx, :, : len(CHANNELS)],
        width=10,
    )
    x = np.arange(PRED_LEN)
    plot_names = ["Ground Truth"] + list(preds.keys())

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(8.45, 6.1),
        dpi=500,
        gridspec_kw={"width_ratios": [4.05, 1.45, 1.25], "wspace": 0.38, "hspace": 0.64},
    )
    fig.patch.set_facecolor("white")

    all_handles = None
    all_labels = None
    heatmap_images = []
    heatmap_vmax_by_channel = [
        float(np.percentile(np.abs(preds["Proposed"][:, :, ch] - truth[:, :, ch]), 97))
        for _, ch in CHANNELS
    ]

    for row, (title, channel) in enumerate(CHANNELS):
        ax = axes[row, 0]
        zoom_ax = axes[row, 1]
        heat_ax = axes[row, 2]

        channel_series = {"Ground Truth": truth[sample_idx, :, channel]}
        for name, arr in preds.items():
            channel_series[name] = arr[sample_idx, :, channel]

        y_true = channel_series["Ground Truth"]
        z0, z1 = shared_z0, shared_z1

        plot_series(ax, x, channel_series, plot_names)
        add_zoom_box(ax, z0, z1, y_true)
        ax.set_title(f"({chr(97 + row)}) {title}", fontsize=8.4, fontweight="bold", pad=3)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_ylabel("Load", fontsize=7.4, fontweight="bold", labelpad=1.5)
        ax.grid(True, color="#d6d6d6", linewidth=0.48, alpha=0.72)
        ax.tick_params(axis="both", labelsize=6.9, colors="#0b0b0b", width=0.8, length=2.6)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("#111111")
        if row == 2:
            ax.set_xlabel("Time Steps (h)", fontsize=7.5, fontweight="bold", labelpad=2)
        else:
            ax.set_xticklabels([])

        plot_series(zoom_ax, x[z0 : z1 + 1], {k: v[z0 : z1 + 1] for k, v in channel_series.items()}, plot_names)
        y_stack = np.vstack([channel_series[name][z0 : z1 + 1] for name in plot_names])
        y_min, y_max = float(np.min(y_stack)), float(np.max(y_stack))
        pad = (y_max - y_min) * 0.12 + 1e-6
        zoom_ax.set_xlim(z0, z1)
        zoom_ax.set_ylim(y_min - pad, y_max + pad)
        zoom_ax.set_title(f"Zoom {z0}-{z1} h", fontsize=7.8, fontweight="bold", pad=3)
        zoom_ax.grid(True, color="#d6d6d6", linewidth=0.48, alpha=0.72)
        zoom_ax.tick_params(axis="both", labelsize=6.4, colors="#0b0b0b", width=0.8, length=2.4)
        for spine in zoom_ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("#111111")
        if row == 2:
            zoom_ax.set_xlabel("Time Steps (h)", fontsize=7.0, fontweight="bold", labelpad=2)
        else:
            zoom_ax.set_xticklabels([])

        errors, sample_indices, step_indices = build_error_heatmap(truth, preds["Proposed"], sample_idx, channel, z0, z1)
        im = plot_error_heatmap(heat_ax, errors, sample_indices, step_indices, "10-step Error Heatmap", heatmap_vmax_by_channel[row])
        heatmap_images.append(im)
        if row == 2:
            heat_ax.set_xlabel("Forecast Steps", fontsize=7.0, fontweight="bold", labelpad=2)
        else:
            heat_ax.set_xticklabels([])

        if all_handles is None:
            all_handles, all_labels = ax.get_legend_handles_labels()

    fig.legend(
        all_handles,
        all_labels,
        loc="lower center",
        ncol=4,
        fontsize=6.9,
        frameon=True,
        framealpha=0.96,
        edgecolor="#d0d0d0",
        handlelength=1.9,
        columnspacing=0.75,
        handletextpad=0.38,
        borderpad=0.35,
        bbox_to_anchor=(0.5, 0.025),
    )
    cbar_ax = fig.add_axes([0.947, 0.20, 0.012, 0.57])
    cbar = fig.colorbar(heatmap_images[0], cax=cbar_ax)
    cbar.set_label("Absolute Error", fontsize=7.2, fontweight="bold", labelpad=3)
    cbar.ax.tick_params(labelsize=6.6, colors="#0b0b0b", width=0.8, length=2.3)

    fig.suptitle(f"Multi-energy Load Forecasting Results (Sample {sample_idx}, 0-100 h)", fontsize=9.6, fontweight="bold", y=0.982)
    fig.subplots_adjust(left=0.070, right=0.930, top=0.910, bottom=0.145)
    for path in (FIG_PATH, FIG_PATH_WPS):
        fig.savefig(path, dpi=700, facecolor="white")
    fig.savefig(FIG_PATH_PDF, facecolor="white")
    fig.savefig(FIG_PATH_SVG, facecolor="white")


if __name__ == "__main__":
    main()

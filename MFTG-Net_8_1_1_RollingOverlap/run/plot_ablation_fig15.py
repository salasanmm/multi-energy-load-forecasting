from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


VARIANTS = [
    "MFTG-Net (full)",
    "w/o MSSM",
    "w/o MSTM",
    "w/o output residual layers",
    "w/o regression layers",
    "w/o PGF gate",
    "w/o CRD refiner",
    "w/o PH refiner",
    "w/o Anchor Fusion",
    "w/o Horizon Affine Refiner",
]
DISPLAY_NAMES = [
    "Full",
    "w/o MSSM",
    "w/o MSTM",
    "w/o OutRes",
    "w/o Reg",
    "w/o PGF",
    "w/o CRD",
    "w/o PH",
    "w/o Anchor",
    "w/o HAF",
]
HORIZONS = [24, 48, 72, 96]
METRICS = [
    ("MAE_mean", "MAE"),
    ("RMSE_mean", "RMSE"),
    ("MAPE_mean", "MAPE (%)"),
]
COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00"]
MARKERS = ["o", "s", "^", "D"]


def plot(csv_path: Path, output_path: Path, preview_path: Path | None) -> None:
    data = pd.read_csv(csv_path)
    required = {"Variant", "horizon", *(metric for metric, _ in METRICS)}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {', '.join(sorted(missing))}")

    data = data[data["channel"].eq("overall")]
    if set(data["Variant"].unique()) != set(VARIANTS):
        raise ValueError("CSV variants do not match the Table 3 ablation variants")
    if set(data["horizon"].unique()) != set(HORIZONS):
        raise ValueError("CSV must contain horizons 24, 48, 72, and 96")

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 4.6), sharex=True)
    x = list(range(len(VARIANTS)))

    for ax, (metric, title) in zip(axes, METRICS):
        subset = data.pivot(index="Variant", columns="horizon", values=metric).reindex(VARIANTS)
        for horizon, color, marker in zip(HORIZONS, COLORS, MARKERS):
            ax.plot(
                x,
                subset[horizon].to_numpy(),
                color=color,
                marker=marker,
                linewidth=1.7,
                markersize=4.8,
                markeredgewidth=0.9,
                label=f"{horizon} h",
            )
        ax.set_title(title, fontweight="semibold", pad=6)
        ax.set_xticks(x, DISPLAY_NAMES, rotation=48, ha="right", rotation_mode="anchor")
        ax.set_xlim(-0.35, len(VARIANTS) - 0.65)
        ax.grid(axis="y", color="#D9DEE5", linewidth=0.65)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#68717D")
        ax.spines["bottom"].set_color("#68717D")
        ax.tick_params(axis="x", length=0, pad=4)
        ax.tick_params(axis="y", length=3, color="#68717D")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=4,
        frameon=False,
        handlelength=1.7,
        columnspacing=1.5,
    )
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.36, top=0.82, wspace=0.38)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.05)
    if preview_path is not None:
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview_path, dpi=220, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview", type=Path)
    args = parser.parse_args()
    plot(args.csv, args.output, args.preview)


if __name__ == "__main__":
    main()

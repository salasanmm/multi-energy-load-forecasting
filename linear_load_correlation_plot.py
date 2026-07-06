from __future__ import annotations

from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(r"E:\负荷预测（原件）(2)")
DATA_PATH = BASE_DIR / "data" / "dataset_input_jiuzheng.csv"
FIG_PATH = BASE_DIR / "电冷热与9因素线性关系图.png"

TARGETS = {
    "KW": "Electric Load",
    "CHWTON": "Cool Load",
    "HTmmBTU": "Heat Load",
}


def wrap_labels(labels: list[str], width: int = 18) -> list[str]:
    return [fill(label.replace("_", " "), width=width) for label in labels]


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "axes.unicode_minus": False,
            "figure.dpi": 500,
            "savefig.dpi": 700,
        }
    )

    df = pd.read_csv(DATA_PATH)
    target_cols = list(TARGETS.keys())
    feature_cols = [col for col in df.columns if col not in target_cols]

    corr = df[feature_cols + target_cols].corr(method="pearson").loc[feature_cols, target_cols]

    x = np.arange(len(feature_cols))
    width = 0.23
    colors = {
        "KW": "#2f78b7",
        "CHWTON": "#f28e2b",
        "HTmmBTU": "#2ca02c",
    }
    offsets = [-width, 0, width]

    fig, ax = plt.subplots(figsize=(9.1, 6.15))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for offset, target in zip(offsets, target_cols):
        values = corr[target].to_numpy(dtype=float)
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=TARGETS[target],
            color=colors[target],
            edgecolor="white",
            linewidth=0.7,
            alpha=0.96,
        )
        for bar, value in zip(bars, values):
            if abs(value) < 0.08:
                continue
            y = value / 2
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8.2,
                fontweight="bold",
                color="white" if abs(value) >= 0.55 else "#111111",
                rotation=90,
            )

    ax.axhline(0, color="#222222", linewidth=1.0)
    ax.set_ylim(-1.08, 1.08)
    ax.set_yticks(np.arange(-1.0, 1.01, 0.25))
    ax.set_ylabel("Pearson Correlation Coefficient (PCC)", fontsize=12.0, fontweight="bold")
    ax.set_xlabel("Input Features", fontsize=12.0, fontweight="bold", labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(wrap_labels(feature_cols, width=14), rotation=32, ha="right", fontsize=10.2, fontweight="bold")
    ax.tick_params(axis="y", labelsize=10.4, width=0.9, colors="#111111")
    ax.tick_params(axis="x", width=0.9, colors="#111111")
    ax.grid(axis="y", linestyle="--", linewidth=0.65, color="#cfcfcf", alpha=0.78)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#222222")

    legend = ax.legend(
        title="Target Loads",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=True,
        framealpha=0.98,
        edgecolor="#d0d0d0",
        fontsize=10.4,
        title_fontsize=10.8,
    )
    legend.get_title().set_fontweight("bold")

    ax.set_title(
        "Linear Correlation between Multi-energy Loads and 9 Input Features",
        fontsize=13.0,
        fontweight="bold",
        pad=28,
    )

    fig.subplots_adjust(left=0.105, right=0.985, top=0.84, bottom=0.31)
    fig.savefig(FIG_PATH, facecolor="white")


if __name__ == "__main__":
    main()

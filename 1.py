from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LinearSegmentedColormap


BASE_DIR = Path(r"E:\负荷预测（原件）(2)")
DATA_PATH = BASE_DIR / "data" / "dataset_input_jiuzheng.csv"
OUTPUT_PATH = BASE_DIR / "图片1.png"

ELECTRIC_CMAP = LinearSegmentedColormap.from_list("electric_soft", ["#d8f2d6", "#8bd18f", "#3ea84f", "#16733a"])
COOLING_CMAP = LinearSegmentedColormap.from_list("cooling_soft", ["#d9e8f6", "#8db9e6", "#4f93d1", "#1d5fa7"])
HEATING_CMAP = LinearSegmentedColormap.from_list("heating_soft", ["#f8ddc6", "#f6b37a", "#e77730", "#b34a10"])


def reshape_daily_hourly(series: pd.Series, hours_per_day: int = 24) -> np.ndarray:
    """Trim to a whole number of days and reshape to (days, hours)."""
    values = series.to_numpy(dtype=float)
    full_days = len(values) // hours_per_day
    trimmed = values[: full_days * hours_per_day]
    return trimmed.reshape(full_days, hours_per_day)


def setup_3d_axis(ax, title: str, zlabel: str, zticks: list[float], zlim: tuple[float, float], cmap: str) -> None:
    ax.set_title(title, fontsize=14, fontweight="bold", pad=6, color=plt.get_cmap(cmap)(0.82))
    ax.set_xlabel("Day", fontsize=11, labelpad=8)
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 24)
    ax.set_zlim(*zlim)
    ax.set_xticks(np.arange(0, 1001, 200))
    ax.set_yticks(np.arange(0, 25, 6))
    ax.set_zticks(zticks)
    ax.zaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    ax.view_init(elev=18, azim=-135)
    ax.grid(True, alpha=0.28)
    ax.tick_params(axis="both", labelsize=8.5, pad=2)
    ax.tick_params(axis="z", labelsize=8.5, pad=2)
    ax.text2D(0.01, 0.085, "Hour", transform=ax.transAxes, fontsize=11, ha="left", va="bottom")
    ax.text2D(
        -0.17,
        0.58,
        zlabel,
        transform=ax.transAxes,
        rotation=90,
        fontsize=9.8,
        ha="center",
        va="center",
        clip_on=False,
        color="black",
        fontweight="bold",
    )
    try:
        ax.set_box_aspect((3.20, 1.16, 0.98))
    except Exception:
        pass
    try:
        ax.xaxis.pane.set_facecolor((1, 1, 1, 0.0))
        ax.yaxis.pane.set_facecolor((1, 1, 1, 0.0))
        ax.zaxis.pane.set_facecolor((1, 1, 1, 0.0))
    except Exception:
        pass


def plot_surface(ax, day_grid, hour_grid, z, cmap: str):
    return ax.plot_surface(
        day_grid,
        hour_grid,
        z,
        cmap=cmap,
        linewidth=0,
        edgecolor="none",
        antialiased=True,
        alpha=0.8,
        shade=True,
    )


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    mapping = [
        ("KW", "Electricity Load", "Electricity Load (×10⁴ kW)", [0, 1, 2, 3], (0, 3.4), ELECTRIC_CMAP),
        ("CHWTON", "Cooling Load", "Cool Load (×10⁴ kW)", [0, 2, 4, 6], (0, 6.4), COOLING_CMAP),
        ("HTmmBTU", "Heating Load", "Heat Load (×10⁴ kW)", [0, 0.2, 0.4], (0, 0.5), HEATING_CMAP),
    ]

    fig = plt.figure(figsize=(13.4, 9.6), dpi=300)
    fig.patch.set_facecolor("white")

    for idx, (column, title, zlabel, zticks, zlim, cmap) in enumerate(mapping, start=1):
        ax = fig.add_subplot(3, 1, idx, projection="3d")
        z = reshape_daily_hourly(df[column]) / 10000.0
        days, hours = z.shape
        day_grid, hour_grid = np.meshgrid(np.arange(1, days + 1), np.arange(0, hours), indexing="ij")

        surf = plot_surface(ax, day_grid, hour_grid, z, cmap)
        setup_3d_axis(ax, title, zlabel, zticks, zlim, cmap)

        cbar = fig.colorbar(surf, ax=ax, shrink=0.78, pad=0.03, aspect=18)
        cbar.ax.tick_params(labelsize=8.5)
        cbar.outline.set_linewidth(0.7)
        cbar.set_ticks(zticks)

    fig.subplots_adjust(left=0.23, right=0.91, top=0.97, bottom=0.04, hspace=0.12)
    fig.savefig(OUTPUT_PATH, dpi=400, bbox_inches="tight", pad_inches=0.05, facecolor="white")


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


BASE_DIR = Path(r"E:\负荷预测（原件）(2)")
OUTPUT_DIR = BASE_DIR / "output"

MY_MODEL_FOLDER = "TimeMixer"
MY_MODEL_NAME = "MY Model"
HORIZONS = [24, 48, 72, 96]

FIG_PATH = BASE_DIR / "项目MY_Model_MAE改进率图.png"
PDF_PATH = BASE_DIR / "项目MY_Model_MAE改进率图.pdf"
CSV_PATH = BASE_DIR / "项目MY_Model_MAE改进率_清晰数据.csv"

# Baseline MAE values transcribed from the reference table provided by the user.
REFERENCE_BASELINE_MAE = {
    "DLinear": [678.4673, 885.5002, 1044.9729, 1161.6257],
    "TimeMixer": [682.4364, 903.1430, 1074.9383, 1167.4664],
    "FiLM": [750.5935, 928.5215, 1089.5145, 1213.8094],
    "TimesNet": [814.4979, 1013.8479, 1123.4011, 1378.5698],
    "TimeXer": [736.1014, 966.3871, 1107.6729, 1231.4666],
    "TSMixer": [762.8230, 964.3477, 1099.4564, 1248.5256],
    "iTransformer": [729.9072, 981.1653, 1158.5640, 1243.5777],
    "FITS": [702.6867, 919.3070, 1084.8245, 1210.7168],
    "CFC": [692.6699, 905.4956, 1033.0853, 1146.2882],
    "SparseTSF": [878.0982, 1088.6006, 1248.0890, 1354.1375],
    "Mamba": [767.2476, 988.4642, 1203.7207, 1350.7268],
}


def load_project_mae(horizon: int) -> float:
    result_dir = OUTPUT_DIR / f"{MY_MODEL_FOLDER}_{horizon}_168" / "result"
    y_true = np.load(result_dir / "all_y_true.npy")
    y_pred = np.load(result_dir / "all_predict_value.npy")
    return float(np.mean(np.abs(y_pred - y_true)))


def collect_my_model_mae() -> list[float]:
    return [load_project_mae(horizon) for horizon in HORIZONS]


def compute_improvement(my_mae: list[float]) -> dict[str, list[float]]:
    my = np.array(my_mae, dtype=float)
    improvement: dict[str, list[float]] = {}
    for model, values in REFERENCE_BASELINE_MAE.items():
        baseline = np.array(values, dtype=float)
        improvement[model] = ((baseline - my) / baseline * 100.0).tolist()
    return improvement


def save_csv(my_mae: list[float], improvement: dict[str, list[float]]) -> None:
    ordered = sorted(REFERENCE_BASELINE_MAE, key=lambda model: np.mean(improvement[model]), reverse=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["MY Model MAE Source", f"Project output: output/{MY_MODEL_FOLDER}_{{24,48,72,96}}_168/result"])
        writer.writerow(["MY Model", *[f"MAE_{h}h" for h in HORIZONS]])
        writer.writerow([MY_MODEL_NAME, *[f"{value:.4f}" for value in my_mae]])
        writer.writerow([])
        writer.writerow(
            [
                "Rank",
                "Reference Baseline Model",
                "Baseline_MAE_24h",
                "Baseline_MAE_48h",
                "Baseline_MAE_72h",
                "Baseline_MAE_96h",
                "MY_Model_MAE_24h",
                "MY_Model_MAE_48h",
                "MY_Model_MAE_72h",
                "MY_Model_MAE_96h",
                "Improvement_24h(%)",
                "Improvement_48h(%)",
                "Improvement_72h(%)",
                "Improvement_96h(%)",
                "Mean Improvement(%)",
            ]
        )
        for rank, model in enumerate(ordered, start=1):
            values = improvement[model]
            writer.writerow(
                [
                    rank,
                    model,
                    *[f"{value:.4f}" for value in REFERENCE_BASELINE_MAE[model]],
                    *[f"{value:.4f}" for value in my_mae],
                    *[f"{value:.2f}" for value in values],
                    f"{np.mean(values):.2f}",
                ]
            )
        writer.writerow([])
        writer.writerow(["Formula", "(Baseline MAE - MY Model MAE) / Baseline MAE * 100%"])


def draw_summary_strip(ax: plt.Axes, improvement_matrix: np.ndarray) -> None:
    horizon_mean = np.mean(improvement_matrix[:, : len(HORIZONS)], axis=0)
    ax.axis("off")
    ax.set_xlim(-0.85, len(HORIZONS))
    ax.set_ylim(0, 1)
    for i, value in enumerate(horizon_mean):
        ax.add_patch(
            plt.Rectangle((i, 0.1), 1.0, 0.75, facecolor="#f1f3f5", edgecolor="#555555", linewidth=0.75)
        )
        ax.text(i + 0.5, 0.48, f"{value:.1f}%", ha="center", va="center", fontsize=8.0, fontweight="bold")
        ax.text(i + 0.5, 0.91, f"{HORIZONS[i]}h", ha="center", va="bottom", fontsize=7.4, fontweight="bold")
    ax.text(-0.08, 0.48, "Mean by horizon", ha="right", va="center", fontsize=8.3, fontweight="bold", clip_on=False)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    my_mae = collect_my_model_mae()
    improvement = compute_improvement(my_mae)
    save_csv(my_mae, improvement)

    ordered_models = sorted(REFERENCE_BASELINE_MAE, key=lambda model: np.mean(improvement[model]), reverse=True)
    matrix = np.array([improvement[model] + [float(np.mean(improvement[model]))] for model in ordered_models])
    values = matrix[:, : len(HORIZONS)]

    bound = float(max(abs(values.min()), abs(values.max())))
    cmap = LinearSegmentedColormap.from_list(
        "project_improvement",
        ["#2b6cb0", "#f5f7fa", "#f6b26b", "#b2182b"],
    )
    norm = TwoSlopeNorm(vmin=-max(6.0, bound), vcenter=0.0, vmax=max(34.0, bound))

    fig = plt.figure(figsize=(7.25, 6.35), dpi=650)
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.0, 0.045],
        height_ratios=[1.0, 0.12],
        wspace=0.08,
        hspace=0.25,
    )
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    ax_strip = fig.add_subplot(gs[1, 0])

    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(HORIZONS) + 1))
    ax.set_xticklabels([f"{h}h" for h in HORIZONS] + ["Mean"], fontsize=8.8, fontweight="bold")
    ax.set_yticks(np.arange(len(ordered_models)))
    ax.set_yticklabels(ordered_models, fontsize=8.4, fontweight="bold")
    ax.set_title("MAE Improvement of MY Model Against Reference Baselines", fontsize=10.3, fontweight="bold", pad=8)
    ax.set_xlabel("Forecasting Horizon", fontsize=9.4, fontweight="bold", labelpad=7)
    ax.set_xticks(np.arange(-0.5, len(HORIZONS) + 1, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ordered_models), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.1)
    ax.axvline(len(HORIZONS) - 0.5, color="#1f1f1f", linewidth=1.15)
    ax.tick_params(axis="both", which="major", length=0)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax.text(
                j,
                i,
                f"{value:.1f}%",
                ha="center",
                va="center",
                fontsize=7.4,
                fontweight="bold",
                color="white" if value >= 25 else "#111111",
            )

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#222222")

    draw_summary_strip(ax_strip, matrix)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Improvement (%)", fontsize=8.7, fontweight="bold", labelpad=5)
    cbar.ax.tick_params(labelsize=7.7, colors="#111111")

    fig.suptitle("Project-based MAE Improvement Analysis of MY Model", fontsize=12.3, fontweight="bold", y=0.985)
    fig.subplots_adjust(left=0.17, right=0.91, top=0.89, bottom=0.12)
    fig.savefig(FIG_PATH, dpi=700, facecolor="white")
    fig.savefig(PDF_PATH, facecolor="white")


if __name__ == "__main__":
    main()

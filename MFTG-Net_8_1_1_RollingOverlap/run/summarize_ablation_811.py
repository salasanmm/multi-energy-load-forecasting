import argparse
import csv
from pathlib import Path


VARIANTS = {
    "wo_mssm": "w/o MSSM",
    "wo_mstm": "w/o MSTM",
    "wo_out_residual": "w/o output residual layers",
    "wo_regression": "w/o regression layers",
    "wo_pgf": "w/o PGF gate",
    "wo_crd": "w/o CRD refiner",
    "wo_ph": "w/o PH refiner",
    "wo_anchor": "w/o Anchor Fusion",
    "wo_haf": "w/o Horizon Affine Refiner",
}


def read_rows(path, variant):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return [{"Variant": variant, **row} for row in csv.DictReader(stream)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="output8_1_1")
    parser.add_argument("--experiment-tag", default="dft_top5_811")
    args = parser.parse_args()
    root = Path(args.root)
    rows = read_rows(
        root / f"{args.experiment_tag}_rolling_overlap_average_summary.csv",
        "MFTG-Net (full)",
    )
    for variant, label in VARIANTS.items():
        tag = f"ablate_modesearch_{variant}_{args.experiment_tag}"
        rows.extend(read_rows(root / f"{tag}_rolling_overlap_average_summary.csv", label))
    if not rows:
        raise SystemExit("No baseline or ablation summary rows were found.")
    fields = ["Variant", "horizon", "channel", "n_seeds"]
    for metric in ("MAE", "MAPE", "RMSE", "DS", "ACCR"):
        fields.extend((f"{metric}_mean", f"{metric}_std", f"{metric}_var"))
    fields.extend(("Params", "FLOPs", "epoch_time_sec_mean", "epoch_time_sec_std", "epoch_time_sec_var"))
    output = root / "ablation_comparison_811_rolling_overlap_average.csv"
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()

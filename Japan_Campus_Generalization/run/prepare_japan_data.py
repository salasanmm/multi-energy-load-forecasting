import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "source" / "Scientific_data_Figure.xlsx"
OUTPUT = ROOT / "data" / "processed" / "kitakyushu_2019_hourly.csv"
MANIFEST = ROOT / "data" / "processed" / "dataset_manifest.json"
AUX_OUTPUT = ROOT / "data" / "processed" / "kitakyushu_2019_hourly_with_aux.csv"
AUX_MANIFEST = ROOT / "data" / "processed" / "dataset_manifest_with_aux.json"


def main():
    source = pd.read_excel(SOURCE, sheet_name="All load1_2004&2019")
    target_columns = {
        "Electricity load_2019 (kW)": "electricity_kw",
        "Cooling load_2019 (kW)": "cooling_kw",
        "Heating load_2019 (kW)": "heating_kw",
    }
    temp_column = next((c for c in source.columns if c.startswith("Temp_2019")), None)
    if temp_column is None:
        raise KeyError("Could not find the 2019 outdoor temperature column")
    auxiliary_columns = {
        temp_column: "outdoor_temp_c",
        "Hot water load_2019 (kW)": "hot_water_kw",
    }
    all_columns = {**target_columns, **auxiliary_columns}
    selected = source[list(all_columns)].rename(columns=all_columns)
    values = selected.to_numpy(dtype=np.float64)
    if values.shape != (8760, 5) or not np.isfinite(values).all():
        raise ValueError(f"Expected 8760 complete hourly rows and 5 load/weather fields, got {values.shape}")
    if (selected[list(target_columns.values()) + ["hot_water_kw"]].to_numpy() < 0).any():
        raise ValueError("Negative values found in the selected 2019 load columns")

    targets = selected[list(target_columns.values())]
    targets.to_csv(OUTPUT, index=False, float_format="%.8f", encoding="utf-8-sig")

    timestamps = pd.date_range("2019-01-01 00:00:00", periods=len(selected), freq="h")
    hour = timestamps.hour.to_numpy()
    weekday = timestamps.dayofweek.to_numpy()
    yearday = timestamps.dayofyear.to_numpy() - 1
    calendar = pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "weekday_sin": np.sin(2 * np.pi * weekday / 7),
        "weekday_cos": np.cos(2 * np.pi * weekday / 7),
        "yearday_sin": np.sin(2 * np.pi * yearday / 365),
        "yearday_cos": np.cos(2 * np.pi * yearday / 365),
        "is_weekend": (weekday >= 5).astype(np.float64),
    })
    with_aux = pd.concat([selected, calendar], axis=1)
    with_aux.to_csv(AUX_OUTPUT, index=False, float_format="%.8f", encoding="utf-8-sig")

    manifest = {
        "dataset": "Kitakyushu campus building energy system",
        "source_workbook": "data/source/Scientific_data_Figure.xlsx",
        "source_sheet": "All load1_2004&2019",
        "source_columns": list(target_columns) + [temp_column, "Hot water load_2019 (kW)"],
        "target_columns": list(targets.columns),
        "year": 2019,
        "timestamp_policy": "8760 chronological hourly records; timestamps reconstructed from 2019-01-01 because the figure workbook uses a leap-year plotting calendar with Feb 29 omitted",
        "rows": int(len(targets)),
        "units": "kW",
        "missing_values": 0,
        "note": "This is the hourly 2019 series included in the authors' public figure workbook, not the complete 20-year Figshare source release.",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    aux_manifest = {
        **manifest,
        "input_columns": list(with_aux.columns),
        "auxiliary_observed_columns": ["outdoor_temp_c", "hot_water_kw"],
        "calendar_columns": list(calendar.columns),
        "auxiliary_data_policy": "Only historical weather and hot-water observations enter each input window; future observed values are never supplied.",
        "output": str(AUX_OUTPUT.relative_to(ROOT)),
    }
    AUX_MANIFEST.write_text(json.dumps(aux_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(targets)} target rows to {OUTPUT}")
    print(f"Wrote {len(with_aux)} rows with auxiliary features to {AUX_OUTPUT}")


if __name__ == "__main__":
    main()

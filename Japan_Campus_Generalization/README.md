# Kitakyushu Japan Campus Validation

This is a separate MFTG-Net external-dataset experiment package. It does not
share data or output files with the ASU Tempe 8:1:1 experiment.

## Data

`data/source/Scientific_data_Figure.xlsx` is the authors' public figure
workbook from the repository associated with the Scientific Data paper. Its
`All load1_2004&2019` sheet contains 8,760 valid hourly records for the 2019
series, including electricity, cooling, heating, outdoor temperature, and hot
water load. The auxiliary-feature experiment also derives hour-of-day,
day-of-week, and day-of-year sine/cosine features plus a weekend flag from a
reconstructed 2019 hourly calendar. The workbook uses a leap-year plotting axis with an empty
February 29; the valid 2019 observations are retained in row order.

This workbook is a figure-data release, not the complete 20-year source data
release. The complete Figshare record is cited in `configs/japan_811.json`.
The experiment therefore must be described as a secondary evaluation on the
public 2019 hourly series, not as a reproduction over all 20 years.

## Experiment

- Chronological split: 80% train, 10% validation, 10% test.
- Prediction targets: electricity, cooling, and heating loads.
- Input: the previous 168 hours of the three loads, outdoor temperature, hot
  water load, and seven deterministic calendar features (12 total: 3 load + 9
  auxiliary).
- Forecast horizons: 24, 48, 72, and 96 hours.
- Models: MFTG-Net, TimeMixer, Autoformer, DLinear, PatchTST.
- Seeds: 2020, 2021, and 2022; 60 epochs; Adam at 0.001; batch size 128.
- Validation and test use hourly rolling origins and average predictions that
  land on the same target timestamp. At each origin, only observations from
  the preceding 168 hours are supplied. Future measured weather, hot-water,
  and load values are never fed to the model.
- Training loss is averaged over each origin's forecast window and training
  batches. Checkpoints are selected by validation rolling-overlap MAE.
- Scaling is fitted from the training partition only. Test data is not used for
  model or checkpoint selection.

The earlier three-column run used only the three target load series as input.
It is retained under `output/main_experiment/` for traceability and should not
be described as using auxiliary variables. The corrected run uses historical
weather and hot-water features and writes separately to
`output/main_experiment_with_aux/`. The 2019-only series supports evaluation
on a second dataset, but its final chronological test segment does not provide
a multi-year seasonal robustness study or transfer from an ASU-trained
checkpoint.

## Run

```powershell
./run/run_japan_811.ps1 -Device cuda:0 -PythonPath "D:\Anaconda3\envs\wzxx\python.exe"
```

Use `-Resume` to skip completed model/horizon/seed combinations. The default
PowerShell entry point runs the auxiliary-feature configuration and writes
under `output/main_experiment_with_aux/`:

- `japan_2019_by_seed.csv`: per-seed and per-channel metrics from the earlier
  detailed run.
- `japan_2019_summary.xlsx` and `japan_2019_summary.csv`: latest supplied
  rerun summary, including channel and overall metrics. The reported means,
  standard deviations, and variances are calculated from seeds 2020, 2021,
  and 2022.
- `japan_2019_overall_paper_metrics.xlsx` and `.csv`: latest supplied overall
  rerun metrics for reporting. Each mean and standard deviation is based on
  seeds 2020, 2021, and 2022.
- `run_logs/kitakyushu_2019_five_models.log`,
  `run_logs/kitakyushu_2019_remaining_models.log`, and
  `run_logs/kitakyushu_2019_autoformer_replacement.log`: run logs for the
  selected five-model comparison. FITS training lines have been removed from
  the two earlier combined logs.

The processed feature data and its manifest are
`data/processed/kitakyushu_2019_hourly_with_aux.csv` and
`data/processed/dataset_manifest_with_aux.json`; the exact configuration is
`configs/japan_811_with_aux.json`. The summary workbook contains MAE, RMSE, direction score, parameter count,
forward FLOPs, selected epoch, and epoch time by channel and overall. ACCR is
reported only in the overall paper-metrics workbook. Overall metrics are the
arithmetic mean of electricity, cooling, and heating channel metrics.

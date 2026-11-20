# MFTG-Net: Primary 8:1:1 and Seasonal 6:1:3 Experiments

This folder contains the MFTG-Net project, the 8:1:1 primary experiment, and
the supplementary 6:1:3 experiment added to evaluate test performance across
all four seasons. Their run interfaces, configurations, logs, and outputs are
kept separate.

The main-run configuration is `configs/mftg_net_rolling_overlap_811.json`.

## Experiment

- Dataset: `data/dataset_input_jiuzheng.csv` (23,807 chronological rows).
- Split: train `[0, 19045)`, validation `[19045, 21426)`, test `[21426, 23807)`.
- Input length: 168 hours; prediction horizons: 24, 48, 72, and 96 hours.
- Seeds: 2020, 2021, and 2022.
- Decomposition: DFT along the temporal dimension, retaining the top 5
  non-zero frequencies independently for each sample and feature at both the
  input decomposition and PDM stages. The trend is the residual after
  reconstructing the selected seasonal frequencies.
- Model-side overlap consistency smoothing: disabled.
- Training loss: weighted MAPE plus `0.0005` times weighted MAE, averaged over
  each forecast window and training batches.
- Validation checkpoint selection: lowest rolling-overlap overall MAE, with
  validation targets restricted to rows `[19045, 21426)`.
- Test protocol: predictions covering the same target timestamp are averaged
  before scoring, with test targets restricted to rows `[21426, 23807)`.
  Later origins use only measurements already observed by that origin;
  forecast values are not fed back as observations.

For each seed and horizon, channel metrics are calculated after target-time
aggregation. The `overall` row is the arithmetic mean of electricity, cooling,
and heating channel metrics. The three-seed summaries report means, sample
standard deviations, and sample variances. Parameters, profiler FLOPs, and
mean epoch time are also included.

## Run

From this directory, activate an environment with PyTorch, NumPy, pandas, and
SciPy, then run:

```powershell
./run/run_rolling_overlap_811.ps1 -Device cuda:0 -PythonPath "<path-to-python.exe>"
```

The script runs all four horizons and three seeds. To regenerate summaries
from existing 8:1:1 output:

```powershell
python ./run/summarize_rolling_overlap_811.py --root ./output8_1_1 --experiment-tag dft_top5_811
```

## Results

- `output8_1_1/dft_top5_811_rolling_overlap_average_summary.csv`: main
  three-seed summary, including per-channel and overall rows.
- `output8_1_1/dft_top5_811_rolling_overlap_average_by_seed.csv`: metrics
  for each seed.
- `output8_1_1/MFTG-Net_<horizon>_168_dft_top5_811_seed<seed>/result/`:
  predictions, targets, metrics, checkpoints, and reproducibility manifest
  for each run.
- `output8_1_1/run_logs_dft_top5_811/`: the 12 main-run logs.
- `output8_1_1/top_k_sensitivity_24h.csv`: combined `k=1, 3, 5, 10` sensitivity
  summary; the `k=5` entries reuse the 24-hour main runs.
- `output8_1_1/dft_top{k}_sens24_811_rolling_overlap_average_{summary,by_seed}.csv`:
  per-k sensitivity summaries and seed-level records for `k=1, 3, 10`.
- `output8_1_1/run_logs_topk_sensitivity_24h/`: the nine additional sensitivity logs.

Each manifest records the split, selected configuration, seed, software
versions, parameter counts, selected epoch, and test metrics.

## Top-k Sensitivity

The sensitivity run compares `k=1, 3, 5, 10` for the 24-hour horizon with
seeds 2020, 2021, and 2022. The `k=5` entries reuse the 24-hour main runs.
Run the other nine configurations with:

```powershell
./run/run_topk_sensitivity_24h.ps1 -Device cuda:0 -PythonPath "D:\Anaconda3\envs\wzxx\python.exe"
```

The combined table is `output8_1_1/top_k_sensitivity_24h.csv`.

## Seasonal 6:1:3 Supplement

This supplementary experiment specifically addresses the reviewer request to
report performance when the chronological test set covers all four seasons.
It does not replace the primary 8:1:1 experiment. The split is 60% train,
10% validation, and 30% test; the test target rows are `[16664, 23807)`.
For the 24-hour horizon, the overlap-averaged test target timestamps include
spring, summer, autumn, and winter (2,053, 1,356, 2,184, and 1,550 timestamps,
respectively). The model selection data remains validation-only.

The comparison uses MFTG-Net, TimeMixer, Autoformer, DLinear, and PatchTST,
with 24-hour forecasts and seeds 2020, 2021, and 2022. All models use the same
chronological split, 168-hour history, and hourly rolling-origin evaluation.
MFTG-Net consumes the 12 input features directly. The four baseline
implementations use a jointly trained linear `12 -> 3` feature-fusion layer
before their three-channel forecasting backbones.
Seasonal metrics are computed after averaging overlapping forecasts at each
target timestamp. MAPE excludes zero-valued targets, for which the percentage
error is undefined. The summary reports MAE, RMSE, MAPE, and ACCR mean and
standard deviation across seeds.

Run the supplementary experiment with:

```powershell
./run/run_seasonal_613_24h.ps1 -Device cuda:0 -PythonPath "D:\Anaconda3\envs\wzxx\python.exe"
```

Its configuration is `configs/mftg_net_rolling_overlap_613_24h.json`; the
interface is `run/run_seasonal_613_24h.ps1`. Results are isolated in
`output6_1_3/main_experiment/`, including `seasonal_metrics_by_seed_613_24h.csv`
and `seasonal_metrics_summary_613_24h.csv`; logs are in
`output6_1_3/run_logs_613_24h/`. Four baseline implementations are loaded
from the local `baselines/` directory, copied from the Japan comparison
implementation.

The overall three-seed results are:

| k | Validation checkpoint MAE | Test MAE mean ± std | Test MAPE | Test RMSE mean ± std | Test DS |
|---:|---:|---:|---:|---:|---:|
| 1 | 1002.01 | 613.11 ± 19.67 | 5.4372% | 850.14 ± 28.54 | 78.35% |
| 3 | 1017.76 | 614.59 ± 13.85 | 5.4574% | 851.69 ± 16.52 | 78.40% |
| 5 | 1005.12 | 613.98 ± 8.26 | 5.4015% | 850.92 ± 13.80 | 78.08% |
| 10 | 998.86 | 597.01 ± 9.37 | 5.3334% | 833.55 ± 6.28 | 78.52% |

The purpose of this analysis is to assess stability across frequency
selections, not to choose `k` by minimizing test error. The overall test MAE
means for `k=1, 3, 5` span only 1.48, and `k=5` has the smallest across-seed
MAE standard deviation (8.26) among the tested settings. `k=10` produces a
lower mean error, which is reported transparently; the manuscript's specified
`k=5` remains the main setting. The CSV also contains per-channel results and
sample variances across seeds.

## Ablation Study

The ablation study uses the same dataset, chronological 8:1:1 split, DFT
top-5 decomposition, 168-hour input, rolling-overlap validation/test scoring,
60 epochs, and seeds 2020/2021/2022 as the main experiment. The full-model
row is taken from the completed `dft_top5_811` runs. The nine ablations are
MSSM, MSTM, output residual layers, regression layers, PGF gate, CRD refiner,
PH refiner, Anchor Fusion, and Horizon Affine Refiner. The current run script
processes the first four variants and skips completed runs. All ablation
variants use the same validation-based refiner-mode selection as the full
model. The removed component is constrained off in every candidate mode, so
mode selection cannot re-enable it.

Run or resume the first-four 48-run batch from this directory:

```powershell
./run/run_ablation_811.ps1 -Device cuda:0 -PythonPath "D:\Anaconda3\envs\wzxx\python.exe"
```

Run logs are stored in `output/run_logs_ablation_811/`. Each variant has its
own per-seed output directory and three-seed summary CSVs under `output/`,
named `ablate_modesearch_wo_<component>_dft_top5_811_rolling_overlap_average_{by_seed,summary}.csv`.
The summaries include channel and arithmetic-mean overall MAE, MAPE, RMSE,
DS, ACCR, seed standard deviations/variances, parameter counts, FLOPs, and
epoch time. Per-run manifests record the ablation flag and resulting model
configuration. After all variants finish, create the combined table including
the completed full-model baseline with:

```powershell
python ./run/summarize_ablation_811.py --root ./output --experiment-tag dft_top5_811
```

This writes `output/ablation_comparison_811_rolling_overlap_average.csv`.
The combined comparison includes these four variants, the previously completed
PGF/CRD/PH/Anchor/HAF variants, and the full-model baseline.

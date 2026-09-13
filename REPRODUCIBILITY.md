# Reproducibility protocol

This file records the protocol used for the revised MFTG-Net experiments. The
current modified model is loaded from `test_model/MFGT-Net.py` by the root
`train.py` entry point; the separate legacy TimeMixer archive is not used by
the revised model.
The model-by-model settings and manuscript parameter counts are maintained in
[`BASELINE_CONFIGS.md`](BASELINE_CONFIGS.md) and
[`configs/baseline_registry.json`](configs/baseline_registry.json).

## Data and task

- Dataset: `data/dataset_input_jiuzheng.csv`.
- Targets: `KW`, `CHWTON`, and `HTmmBTU` (the first three columns).
- Input features: the three targets plus `temperature`,
  `dew_point_temperature`, `station_level_pressure`, `sea_level_pressure`,
  `wet_bulb_temperature`, `altimeter`, `DayOfYear_cos`, `Combined mmBTU`, and
  `GHG`.
- Look-back window: 168 hourly observations.
- Forecast horizons: 24, 48, 72, and 96 hours.
- Chronological split: 50% training, 20% validation, and 30% test.
- Scaling parameters are fitted using training observations only.

## Fixed training protocol

All learned models use the fixed protocol below unless an implementation
cannot support one of the settings; any exception must be recorded with that
model's manifest.

- Optimizer: Adam.
- Learning rate: `1e-3`.
- Weight decay: `1e-5`.
- Batch size: `256`.
- Maximum epochs: `60`.
- Loss: weighted MAPE plus `0.0005` times weighted MAE.
- Gradient clipping norm: `5`.
- Scheduler: `ReduceLROnPlateau`, factor `0.5`, patience `4`, mode `min`.
- Early stopping: disabled.
- Checkpoint selection: lowest validation MAE.
- Seeds for independent runs: `2020`, `2021`, and `2022`.

The experiment does not use an automated hyperparameter sweep. Each selected
configuration is fixed before test evaluation and is represented as a
singleton search space in the run manifest. The test set is never used for
hyperparameter or checkpoint selection.

## Commands

Run one horizon and seed with:

```text
python train.py --horizon 24 --seed 2020 --run_id 1 --device cpu
```

Repeat with `--seed 2021 --run_id 2` and `--seed 2022 --run_id 3`, and replace
`24` with `48`, `72`, or `96` for the other horizons. A GPU can be selected by
replacing `--device cpu` with the available CUDA device.

Each run writes a JSON manifest under
`output/<model>_<horizon>_<window>/result/`. Summarize completed manifests with:

```text
python summarize_reproducibility.py --root ./output --expected-runs 3
```

The summary marks directories with missing runs as incomplete and does not
invent averages for them.

## Repository scope

The repository includes the current modified MFTG-Net source code, data files,
dependency list, fixed protocol, and reproducibility utilities. Validated
MFTG-Net checkpoints were not present in the supplied modified project, so
generated checkpoints are ignored by Git and can be regenerated with the
commands above. The legacy TimeMixer checkpoints archived in
`artifacts/original_time_mixer/` are explicitly labeled and are not revised
MFTG-Net results.

The original project contains source files for DLinear, CFC, FITS, TSMixer,
TimesNet, TimeMixer, iTransformer, FiLM, Mamba, and MFTG-Net. The laboratory
legacy TimeMixer source and original checkpoints are archived under
`artifacts/original_time_mixer/` with their original 8:1:1 protocol. They are
reference artifacts and are not the revised MFTG-Net results. SparseTSF and
TimeXer are listed in the manuscript but were not present in the original
source tree; their external implementation and version must be recorded before
claiming source-level reproduction of those two baselines.

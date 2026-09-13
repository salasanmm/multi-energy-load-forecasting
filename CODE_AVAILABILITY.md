# Code availability and artifact provenance

This note documents the materials released with the revised manuscript so that
the current implementation can be distinguished from the laboratory legacy
baseline.

## Current proposed model

- Model name in the manuscript: `MFTG-Net`.
- Current modified implementation: `test_model/MFGT-Net.py`.
- Training and evaluation entry point: `train.py`.
- Input data: `data/dataset_input_jiuzheng.csv`.
- Fixed protocol and run-manifest format: `REPRODUCIBILITY.md`.
- Baseline architecture and parameter register: `BASELINE_CONFIGS.md` and
  `configs/baseline_registry.json`.

The filename `MFGT-Net.py` is retained from the modified project for backward
compatibility. It is not the name of a second model; all manuscript references
use `MFTG-Net`.

## Reproducing the reported pipeline

From the repository root, install `requirements.txt` and run, for example:

```text
python train.py --horizon 24 --seed 2020 --run_id 1 --device cpu
```

Replace the horizon with `48`, `72`, or `96` and repeat with seeds `2021` and
`2022` for the independent runs. Each completed run writes a JSON manifest and
the selected checkpoint under `output/`. The checkpoint retained by the script
is the one with the lowest validation MAE; the test set is not used for model
selection.

## Archived legacy baseline

`artifacts/original_time_mixer/` contains `test_model/TimeMixer.py` and four
original TimeMixer checkpoints recovered from the laboratory ZIP archive. They
are provided for provenance and baseline verification only. Their original
8:1:1 split, seed, optimizer settings, and checkpoint behavior are recorded in
`artifacts/original_time_mixer/README.md` and
`artifacts/original_time_mixer/legacy_manifest.json`. They must not be
interpreted as revised MFTG-Net checkpoints.

## Checkpoint status

No validated MFTG-Net pretrained checkpoint was available in the supplied
modified project at the time of this release. Therefore, this repository does
not present the archived TimeMixer weights as MFTG-Net weights. The current
MFTG-Net weights are reproducible from `train.py` using the fixed protocol
above; once validated checkpoint files are generated, their SHA-256 hashes and
run manifests should be added beside the corresponding `output/` directory.


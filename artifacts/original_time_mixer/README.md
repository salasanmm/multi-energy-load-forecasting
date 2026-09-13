# Laboratory TimeMixer baseline archive

This directory contains the original laboratory TimeMixer baseline recovered
from `负荷预测（原件）(2).zip`. The source file is now available at
`test_model/TimeMixer.py`; the four checkpoint files correspond to the
original `TimeMixer_<horizon>_168` output directories.

These artifacts are retained for baseline verification only. They were created
with the legacy project protocol and must not be reported as the revised
MFTG-Net results:

- chronological split: 80% train, 10% validation, 10% test;
- look-back window: 168 hours;
- input columns: 12; target columns: the first three load columns;
- horizons: 24, 48, 72, and 96 hours;
- model settings: `d_model=64`, `d_ff=64`, `e_layers=3`,
  `down_sampling_window=2`, `down_sampling_layers=2`, average pooling,
  moving-average decomposition window `3`, and dropout `0.1`;
- optimizer: Adam with learning rate `1.6e-4` and optimizer weight decay
  `1e-5`;
- batch size: `256`; maximum epochs: `60`; gradient clipping norm: `5`;
- legacy scheduler: ReduceLROnPlateau with factor `0.8` and patience `6`;
- legacy run seed: `2020` in the archived `train.py`;
- legacy checkpoint behavior: the file was overwritten during training and
  therefore represents the final saved legacy run, not a validation-selected
  revised checkpoint.

For the revised manuscript experiments, use the root `train.py` entry point,
which loads the modified MFTG-Net implementation from `test_model/MFGT-Net.py`
and uses the documented 5:2:3 protocol in `REPRODUCIBILITY.md`.

## Files

| Horizon | Checkpoint |
| ---: | --- |
| 24 h | `TimeMixer_24_168/model/model_lnn.pt` |
| 48 h | `TimeMixer_48_168/model/model_lnn.pt` |
| 72 h | `TimeMixer_72_168/model/model_lnn.pt` |
| 96 h | `TimeMixer_96_168/model/model_lnn.pt` |

The SHA-256 hashes are recorded in `legacy_manifest.json`.

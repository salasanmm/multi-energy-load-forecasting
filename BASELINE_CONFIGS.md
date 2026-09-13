# Baseline configuration register

This register records the configuration information recoverable from the
original project and the revised experiment protocol. It is intended to make
the comparison in the manuscript auditable without implying that an
unrecorded test-set search was performed.

## Common protocol

All learned models use the same data and training protocol in the revised
experiments:

- 12 input columns: the three load targets plus nine covariates;
- targets: `KW`, `CHWTON`, and `HTmmBTU`;
- look-back: 168 hours;
- horizons: 24, 48, 72, and 96 hours;
- chronological split: 50% train, 20% validation, and 30% test;
- Adam, learning rate `1e-3`, weight decay `1e-5`, batch size `256`;
- maximum 60 epochs, gradient clipping norm `5`;
- ReduceLROnPlateau, factor `0.5`, patience `4`, mode `min`;
- no early stopping; retain the checkpoint with the lowest validation MAE;
- weighted MAPE plus `0.0005` times weighted MAE;
- planned independent seeds: `2020`, `2021`, and `2022` (three runs).

The hyperparameter search space is a singleton for each model: the selected
configuration is fixed before test evaluation. The test set is not used for
hyperparameter or checkpoint selection. A completed run writes its exact
configuration, parameter count, seed, and environment to a JSON manifest under
`output/<model>_<horizon>_<window>/result/`.

## Architecture settings

The following values are taken from the original `get_config.py` and model
source files. `H` denotes the requested forecast horizon. Values marked as
requiring a manifest were not explicitly bound to the repository's main
`train.py` entry point and should be recorded from the implementation used for
the corresponding table result.

| Model | Source in repository | Selected architecture settings | Source status |
| --- | --- | --- | --- |
| DLinear | `test_model/DLinear.py` | `seq_len=168`, `pred_len=H`, `enc_in=12`, moving-average decomposition from the supplied config | included |
| SparseTSF | no source file in the original project | `seq_len=168`, `pred_len=H`, `enc_in=12` | external implementation and commit must be recorded |
| CFC | `test_model/CFC.py` | `in_features=12`, `hidden_size=64`, `backbone_units=64`, `backbone_layers=1`, LeCun activation, gated cell, `seq_in_len=168` | included |
| FITS | `test_model/FITS_copy.py` | `seq_len=168`, `pred_len=H`, `enc_in=12`, `individual=True`, `base_T=24`, `H_order=6` | included |
| TSMixer | `test_model/TSMixer.py` | `seq_len=168`, `pred_len=H`, `enc_in=12`, `d_model=64`, `e_layers=2`, `dropout=0.2` | included |
| TimesNet | `test_model/TimesNet.py` | `seq_len=168`, `pred_len=H`, `enc_in=12`, `d_model=32`, `e_layers=2`, `d_ff=16`, `num_kernels=4`, `top_k=5`, `dropout=0.1` | included |
| TimeMixer | `test_model/timemixerpgf.py` | `seq_len=168`, `pred_len=H`, `enc_in=12`, `d_model=64`, `d_ff=64`, `e_layers=3`, `down_sampling_window=2`, `down_sampling_layers=2`, average pooling, `dropout=0.1` | included; refiner flags must be recorded |
| iTransformer | `test_model/iTransformer.py` | `seq_len=168`, `pred_len=H`, `enc_in=12`; attention and feed-forward dimensions require the run manifest | source included; exact run config required |
| FiLM | `test_model/FiLM.py` | `seq_len=168`, `pred_len=H`, `enc_in=12`, multiscale factors `[1,2,4]`, HiPPO order `256`, spectral modes up to `32`, ratio `0.5` | included |
| TimeXer | no source file in the original project | `seq_len=168`, `pred_len=H`, `enc_in=12` | external implementation and commit must be recorded |
| Mamba | `test_model/Mamba.py` | `seq_len=168`, `pred_len=H`, `enc_in=12`, `expand=2`, `d_conv=4`; `d_model`, `d_state`, and output dimensions require the run manifest | source included; `mamba-ssm` dependency required |
| MFTG-Net (ours) | `test_model/MFGT-Net.py` | `seq_len=168`, `pred_len=H`, `enc_in=12`, `d_model=64`, `d_ff=64`, `e_layers=3`, PDM downsampling `2 x 2`, PGF/CRD/PH/Anchor/HAF enabled | included |

## Parameter counts reported in the manuscript

The following counts are transcribed from the comparison table and are shown
in K parameters. They are reference values for the reported implementation,
not newly inferred values. A run manifest is the authoritative record for a
reproduction because parameter counts can change with a model's horizon.

| Model | H=24 | H=48 | H=72 | H=96 |
| --- | ---: | ---: | ---: | ---: |
| DLinear | 101.95K | 199.30K | 296.64K | 393.98K |
| SparseTSF | 52.42K | 54.43K | 56.45K | 58.46K |
| CFC | 3.69M | 3.73M | 3.76M | 3.79M |
| FITS | 46.63K | 52.12K | 58.96K | 65.75K |
| TSMixer | 1.21M | 1.29M | 1.32M | 1.36M |
| TimesNet | 167.75M | 192.21M | 209.57M | 231.21M |
| TimeMixer | 349.70K | 363.90K | 378.20K | 392.40K |
| iTransformer | 2.71M | 2.79M | 2.82M | 2.86M |
| FiLM | 1.25K | 2.32K | 3.51K | 4.59K |
| TimeXer | 21.76M | 22.01M | 22.56M | 22.98M |
| Mamba | 3.26M | 3.29M | 3.31M | 3.39M |
| MFTG-Net (ours) | 349.67K | 363.93K | 378.18K | 392.44K |

The parameter counts above do not replace independent runs. For every model
used in a revised table, commit the implementation and retain the generated
manifest; for SparseTSF and TimeXer, the external source URL and commit or
release version must also be recorded before claiming full source-level
reproducibility.

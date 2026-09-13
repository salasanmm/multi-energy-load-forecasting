The training script expects `dataset_input_jiuzheng.csv` in this directory.

CSV schema (12 columns):

`KW, CHWTON, HTmmBTU, temperature, dew_point_temperature,
station_level_pressure, sea_level_pressure, wet_bulb_temperature, altimeter,
DayOfYear_cos, Combined mmBTU, GHG`

The first three columns are the forecast targets. The remaining nine columns
are covariates. Keep the column order when preparing a replacement file, or
pass an alternative file with `python train.py --data PATH`.

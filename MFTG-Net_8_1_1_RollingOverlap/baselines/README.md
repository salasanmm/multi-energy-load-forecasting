# Baseline Implementations

The four model files and supporting layers are copied without code changes
from the `Japan_Campus_Generalization` project that produced the Japan campus
comparison. The seasonal 6:1:3 runner adds the same learned 12-to-3 input
fusion layer to each baseline so all four can use the ASU dataset's 12 input
features while forecasting the same three target loads.

Baseline source files are isolated here so the USA experiment does not depend
on the Japan project being present at run time.

param(
    [string]$Device = "cuda:0",
    [string]$PythonPath = "D:\Anaconda3\envs\wzxx\python.exe",
    [int]$Epochs = 60,
    [int]$BatchSize = 128,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
& $PythonPath run\prepare_japan_data.py
$LogDir = Join-Path $Root "output\main_experiment_with_aux\run_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogPath = Join-Path $LogDir "kitakyushu_2019_five_models_with_aux.log"
$Arguments = @(
    "run\run_japan_811.py",
    "--device", $Device,
    "--epochs", $Epochs,
    "--batch-size", $BatchSize,
    "--use-aux-features"
)
if ($Resume) { $Arguments += "--resume" }
& $PythonPath @Arguments 2>&1 | Tee-Object -FilePath $LogPath
if ($LASTEXITCODE -ne 0) { throw "Experiment runner failed with exit code $LASTEXITCODE" }

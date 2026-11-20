param(
    [string]$Device = "cuda:0",
    [string]$PythonPath = "D:\Anaconda3\envs\wzxx\python.exe",
    [int]$Epochs = 60,
    [int]$BatchSize = 256,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "Python executable not found: $PythonPath"
}
$Python = (Resolve-Path -LiteralPath $PythonPath).Path
$OutputRoot = "output6_1_3"
$ExperimentOutput = Join-Path $OutputRoot "main_experiment"
$ExperimentTag = "dft_top5_613_seasonal24h"
$LogDir = Join-Path $OutputRoot "run_logs_613_24h"
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

foreach ($Seed in @(2020, 2021, 2022)) {
    $RunName = "MFTG-Net_24_168_${ExperimentTag}_seed${Seed}"
    $RunDir = Join-Path $ExperimentOutput "$RunName\result"
    $Complete = (Test-Path -LiteralPath (Join-Path $RunDir "all_y_true_overlap_average.npy")) -and
        (Test-Path -LiteralPath (Join-Path $RunDir "final_test_metrics_by_protocol.json"))
    if ($Complete -and $Resume) {
        Write-Host "Skipping completed MFTG-Net seed $Seed"
        continue
    }
    if (Test-Path -LiteralPath $RunDir) {
        throw "Incomplete or existing output for $RunName. Use -Resume for completed runs; inspect the result directory before retrying."
    }

    $LogPath = Join-Path $LogDir "mftg_net_613_24h_seed${Seed}.log"
    $Arguments = @(
        "train.py",
        "--data", "./data/dataset_input_jiuzheng.csv",
        "--horizon", "24",
        "--seq_in_len", "168",
        "--seq_out_len", "24",
        "--train_ratio", "0.6",
        "--valid_ratio", "0.1",
        "--experiment_tag", $ExperimentTag,
        "--output-root", $ExperimentOutput,
        "--seed", "$Seed",
        "--run_id", "1",
        "--num_runs", "3",
        "--device", $Device,
        "--epochs", "$Epochs",
        "--batch_size", "$BatchSize",
        "--evaluation-protocol", "rolling_overlap_average",
        "--top_k", "5"
    )
    Write-Host "Starting MFTG-Net 24h seed $Seed with chronological 6:1:3 split"
    & $Python @Arguments 2>&1 | Tee-Object -FilePath $LogPath
    if ($LASTEXITCODE -ne 0) {
        throw "MFTG-Net failed for seed $Seed. See $LogPath"
    }
}

$BaselineLogName = if ($Resume) { "four_baselines_613_24h_resume.log" } else { "four_baselines_613_24h.log" }
$BaselineLog = Join-Path $LogDir $BaselineLogName
$BaselineArgs = @(
    "run\run_baselines_613_24h.py",
    "--device", $Device,
    "--epochs", "$Epochs",
    "--batch-size", "$BatchSize"
)
if ($Resume) { $BaselineArgs += "--resume" }
& $Python @BaselineArgs 2>&1 | Tee-Object -FilePath $BaselineLog
if ($LASTEXITCODE -ne 0) {
    throw "Baseline runs failed. See $BaselineLog"
}

& $Python "run\summarize_seasonal_613_24h.py" --root $ExperimentOutput --experiment-tag $ExperimentTag
if ($LASTEXITCODE -ne 0) {
    throw "Seasonal summary generation failed."
}

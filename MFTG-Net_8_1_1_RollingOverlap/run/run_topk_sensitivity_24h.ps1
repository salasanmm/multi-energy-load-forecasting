param(
    [string]$Device = "cuda:0",
    [string]$PythonPath = "D:\Anaconda3\envs\wzxx\python.exe"
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "Python executable not found: $PythonPath"
}
$Python = (Resolve-Path -LiteralPath $PythonPath).Path
$Seeds = @(2020, 2021, 2022)
$SensitivityKs = @(1, 3, 10)
$OutputRoot = "output8_1_1"
$LogDir = Join-Path $OutputRoot "run_logs_topk_sensitivity_24h"
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

foreach ($K in $SensitivityKs) {
    $ExperimentTag = "dft_top${K}_sens24_811"
    foreach ($Seed in $Seeds) {
        $RunDir = Join-Path $OutputRoot "MFTG-Net_24_168_${ExperimentTag}_seed${Seed}"
        if (Test-Path $RunDir) {
            throw "Output already exists: $RunDir. Choose a new experiment tag to preserve prior runs."
        }

        $LogPath = Join-Path $LogDir "${ExperimentTag}_seed${Seed}.log"
        $Arguments = @(
            "train.py",
            "--data", "./data/dataset_input_jiuzheng.csv",
            "--horizon", "24",
            "--seq_in_len", "168",
            "--seq_out_len", "24",
            "--train_ratio", "0.8",
            "--valid_ratio", "0.1",
            "--experiment_tag", $ExperimentTag,
            "--output-root", $OutputRoot,
            "--seed", "$Seed",
            "--run_id", "1",
            "--num_runs", "3",
            "--device", $Device,
            "--evaluation-protocol", "rolling_overlap_average",
            "--top_k", "$K"
        )
        & $Python @Arguments 2>&1 | Tee-Object -FilePath $LogPath
        if ($LASTEXITCODE -ne 0) {
            throw "Sensitivity run failed for top_k=$K, seed=$Seed. See $LogPath"
        }
    }

    & $Python "run/summarize_rolling_overlap_811.py" --root $OutputRoot --experiment-tag $ExperimentTag
    if ($LASTEXITCODE -ne 0) {
        throw "Per-k summary generation failed for top_k=$K."
    }
}

& $Python "run/summarize_topk_sensitivity.py" --root $OutputRoot
if ($LASTEXITCODE -ne 0) {
    throw "Top-k sensitivity summary generation failed."
}

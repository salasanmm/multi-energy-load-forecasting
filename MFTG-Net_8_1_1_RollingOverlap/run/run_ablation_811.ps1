param(
    [string]$Device = "cuda:0",
    [string]$PythonPath = "python",
    [ValidateSet("fixed_origin", "rolling_overlap_average")]
    [string]$EvaluationProtocol = "rolling_overlap_average"
)

$ErrorActionPreference = "Stop"
if ($PythonPath -eq "python") {
    $Python = (Get-Command python -ErrorAction Stop).Source
} else {
    if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
        throw "Python executable not found: $PythonPath"
    }
    $Python = (Resolve-Path -LiteralPath $PythonPath).Path
}

$Variants = @(
    "wo_mssm", "wo_mstm", "wo_out_residual", "wo_regression"
)
$Horizons = @(24, 48, 72, 96)
$Seeds = @(2020, 2021, 2022)
$OutputRoot = "output8_1_1"
$LogDir = Join-Path $OutputRoot "run_logs_ablation_811"
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

foreach ($Variant in $Variants) {
    $ExperimentTag = "ablate_modesearch_${Variant}_dft_top5_811"
    foreach ($Horizon in $Horizons) {
        foreach ($Seed in $Seeds) {
            $RunName = "MFTG-Net_${Horizon}_168_${ExperimentTag}_seed${Seed}"
            $RunDir = Join-Path $OutputRoot $RunName
            $MetricsPath = Join-Path $RunDir "result\final_test_metrics_by_protocol.json"
            if (Test-Path -LiteralPath $MetricsPath -PathType Leaf) {
                Write-Host "Skipping completed run $RunName"
                continue
            }
            $LogPath = Join-Path $LogDir "${Variant}_h${Horizon}_seed${Seed}.log"
            $Arguments = @(
                "train.py",
                "--data", "./data/dataset_input_jiuzheng.csv",
                "--horizon", "$Horizon",
                "--seq_in_len", "168",
                "--seq_out_len", "$Horizon",
                "--train_ratio", "0.8",
                "--valid_ratio", "0.1",
                "--experiment_tag", $ExperimentTag,
                "--output-root", $OutputRoot,
                "--seed", "$Seed",
                "--run_id", "1",
                "--num_runs", "3",
                "--device", $Device,
                "--evaluation-protocol", $EvaluationProtocol,
                "--top_k", "5",
                "--ablation", $Variant
            )
            Write-Host "Starting $RunName"
            & $Python @Arguments 2>&1 | Tee-Object -FilePath $LogPath
            if ($LASTEXITCODE -ne 0) {
                throw "Training failed for $RunName. See $LogPath"
            }
            if (-not (Test-Path -LiteralPath $MetricsPath -PathType Leaf)) {
                throw "Training exited without final metrics for $RunName. See $LogPath"
            }
        }
    }
    & $Python "run/summarize_rolling_overlap_811.py" --root $OutputRoot --experiment-tag $ExperimentTag --ablation $Variant
    if ($LASTEXITCODE -ne 0) {
        throw "Summary generation failed for $Variant."
    }
}

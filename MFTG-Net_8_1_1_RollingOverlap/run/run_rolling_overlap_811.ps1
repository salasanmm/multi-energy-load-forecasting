param(
    [string]$Device = "cuda:0",
    [string]$ExperimentTag = "dft_top5_811",
    [string]$OutputRoot = "output8_1_1",
    [string]$PythonPath = "python",
    [ValidateSet("fixed_origin", "rolling_overlap_average")]
    [string]$EvaluationProtocol = "rolling_overlap_average",
    [switch]$UseWeeklyAnchor,
    [switch]$ValidationOnly,
    [double]$ValSelectionFraction = 1.0
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
$Horizons = @(24, 48, 72, 96)
$Seeds = @(2020, 2021, 2022)
$LogDir = Join-Path $OutputRoot "run_logs_dft_top5_811"
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

foreach ($Horizon in $Horizons) {
    foreach ($Seed in $Seeds) {
        $RunName = "MFTG-Net_${Horizon}_168_${ExperimentTag}_seed${Seed}"
        $RunDir = Join-Path $OutputRoot $RunName
        if (Test-Path $RunDir) {
            throw "Output already exists: $RunDir. Choose a new -ExperimentTag to preserve prior runs."
        }

        $LogPath = Join-Path $LogDir "${ExperimentTag}_h${Horizon}_seed${Seed}.log"
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
            "--evaluation-protocol", $EvaluationProtocol
            "--top_k", "5"
        )
        if ($UseWeeklyAnchor) {
            $Arguments += "--use-weekly-anchor"
        }
        if ($ValidationOnly) {
            $Arguments += "--validation-only"
            $Arguments += @("--val-selection-fraction", "$ValSelectionFraction")
        }

        & $Python @Arguments 2>&1 | Tee-Object -FilePath $LogPath
        if ($LASTEXITCODE -ne 0) {
            throw "Training failed for horizon $Horizon, seed $Seed. See $LogPath"
        }
    }
}

if (-not $ValidationOnly) {
    & $Python "run/summarize_rolling_overlap_811.py" --root $OutputRoot --experiment-tag $ExperimentTag
    if ($LASTEXITCODE -ne 0) {
        throw "Summary generation failed."
    }
}

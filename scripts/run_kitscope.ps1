param(
  [string]$RepoRoot = "",
  [string]$DataRoot = "",
  [string]$OutRoot = "",
  [string]$Python = "python",
  [int]$RepresentationEpochs = 25,
  [int]$LimitPerSplitLabel = 0
)

$ErrorActionPreference = "Stop"

function Resolve-PathOrDefault {
  param([string]$Value, [string]$DefaultValue)
  if ([string]::IsNullOrWhiteSpace($Value)) {
    return [System.IO.Path]::GetFullPath($DefaultValue)
  }
  return [System.IO.Path]::GetFullPath($Value)
}

function Invoke-LoggedPython {
  param(
    [string[]]$Arguments,
    [string]$Stdout,
    [string]$Stderr,
    [string]$Step
  )
  & $Python @Arguments 1> $Stdout 2> $Stderr
  if ($LASTEXITCODE -ne 0) {
    throw "$Step failed with exit code $LASTEXITCODE; see $Stdout and $Stderr"
  }
}

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$defaultRepoRoot = Join-Path $scriptRoot ".."
$RepoRoot = Resolve-PathOrDefault -Value $RepoRoot -DefaultValue $defaultRepoRoot
$DataRoot = Resolve-PathOrDefault -Value $DataRoot -DefaultValue (Join-Path $RepoRoot "data\kitscope-release")
$OutRoot = Resolve-PathOrDefault -Value $OutRoot -DefaultValue (Join-Path $RepoRoot "results\kitscope")

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null
$inputRoot = Join-Path $OutRoot "runner_inputs"
$featureShardRoot = Join-Path $OutRoot "feature_shards"
$featureRoot = Join-Path $OutRoot "features_merged"
$resultRoot = Join-Path $OutRoot "results"
New-Item -ItemType Directory -Force -Path $inputRoot, $featureShardRoot, $featureRoot, $resultRoot | Out-Null

$env:PYTHONPATH = (Join-Path $RepoRoot "src")
$statusPath = Join-Path $OutRoot "status.json"

@{
  status = "started"
  started_utc = (Get-Date).ToUniversalTime().ToString("o")
  repo_root = $RepoRoot
  data_root = $DataRoot
  out_root = $OutRoot
} | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 -Path $statusPath

$prepareArgs = @(
  "-B", (Join-Path $RepoRoot "scripts\prepare_public_release_inputs.py"),
  "--data-root", $DataRoot,
  "--out-dir", $inputRoot
)
if ($LimitPerSplitLabel -gt 0) {
  $prepareArgs += @("--limit-per-split-label", "$LimitPerSplitLabel")
}

Invoke-LoggedPython -Step "input preparation" `
  -Stdout (Join-Path $OutRoot "01_prepare_inputs_stdout.log") `
  -Stderr (Join-Path $OutRoot "01_prepare_inputs_stderr.log") `
  -Arguments $prepareArgs

@{
  status = "inputs_prepared"
  updated_utc = (Get-Date).ToUniversalTime().ToString("o")
  input_root = $inputRoot
} | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 -Path $statusPath

$extractionConfig = Join-Path $RepoRoot "configs\universal_interactive_phishing_gateway_oracle_plus_noguard.yaml"
Invoke-LoggedPython -Step "feature extraction" `
  -Stdout (Join-Path $OutRoot "02_feature_shard_stdout.log") `
  -Stderr (Join-Path $OutRoot "02_feature_shard_stderr.log") `
  -Arguments @(
    "-B", (Join-Path $RepoRoot "scripts\preprocess_distributed_dataset.py"),
    "feature-shard",
    "--input-glob", (Join-Path $inputRoot "sample_inputs.txt"),
    "--out-dir", $featureShardRoot,
    "--shard-id", "main_random",
    "--scenario-set", "none",
    "--config", "main_random=$extractionConfig",
    "--expected-post-load-guard-mode", "none",
    "--causal-all-onsets",
    "--candidate-decision-delays", "0.75,1.5,2.5,3.5",
    "--candidate-scan-step", "0.5"
  )

Invoke-LoggedPython -Step "feature merge" `
  -Stdout (Join-Path $OutRoot "03_merge_features_stdout.log") `
  -Stderr (Join-Path $OutRoot "03_merge_features_stderr.log") `
  -Arguments @(
    "-B", (Join-Path $RepoRoot "scripts\preprocess_distributed_dataset.py"),
    "merge-features",
    "--shard-root", $featureShardRoot,
    "--out-root", $featureRoot
  )

$featuresCsv = Join-Path $featureRoot "main_random\features.csv"
$featuresParquet = Join-Path $featureRoot "main_random\features.parquet"
$splitCsv = Join-Path $inputRoot "split_manifest.csv"
$kitLabelsCsv = Join-Path $inputRoot "kit_labels.csv"
if (-not (Test-Path -LiteralPath $featuresParquet)) {
  throw "feature merge did not produce the parquet table required by the formal random-route runner: $featuresParquet"
}

Invoke-LoggedPython -Step "main random fair comparison" `
  -Stdout (Join-Path $OutRoot "04_main_random_stdout.log") `
  -Stderr (Join-Path $OutRoot "04_main_random_stderr.log") `
  -Arguments @(
    "-B", (Join-Path $RepoRoot "scripts\run_fair_multiscale_window_comparison.py"),
    "--features-parquet", $featuresParquet,
    "--column-spec", (Join-Path $RepoRoot "configs\comparison_column_spec.json"),
    "--split-manifest", $splitCsv,
    "--kit-labels-csv", $kitLabelsCsv,
    "--out-dir", $resultRoot,
    "--seed", "42",
    "--alpha", "0.03",
    "--onset-oof-folds", "5",
    "--nms-merge-s", "2.0",
    "--max-decision-delay-s", "3.5",
    "--score-batch-size", "8192",
    "--condition", "five_windows",
    "--variant", "dynamic_rank8",
    "--balanced-kit-dann",
    "--fitdev-refit",
    "--representation-weight-grid", "0,0.1,0.2,0.3,0.4,0.5,0.7,1.0",
    "--representation-epochs", "$RepresentationEpochs",
    "--representation-early-stop-patience", "5",
    "--representation-lam-kit", "0.3",
    "--representation-lam-era", "0.3",
    "--dynamic-rank-budget", "8",
    "--headline-aggregation", "top2",
    "--single-candidate-target-fpr", "0.002",
    "--candidate-decision-delays-s", "0.75,1.5,2.5,3.5"
  )

@{
  status = "complete"
  completed_utc = (Get-Date).ToUniversalTime().ToString("o")
  feature_file = $featuresCsv
  feature_parquet = $featuresParquet
  result_root = $resultRoot
} | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 -Path $statusPath

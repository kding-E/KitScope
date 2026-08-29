$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
python -m kitscope.verify_main --config configs/smoke.yaml --data-root data/smoke --out results/smoke
pytest

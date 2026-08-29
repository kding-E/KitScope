#!/usr/bin/env bash
set -euo pipefail

repo_root=""
data_root=""
out_root=""
python_bin="python"
representation_epochs=25
limit_per_split_label=0

usage() {
  cat <<'EOF'
Usage: scripts/run_kitscope.sh [options]

Options:
  --repo-root PATH                  Repository root. Defaults to this script's parent.
  --data-root PATH                  Dataset root. Defaults to data/kitscope-release.
  --out-root PATH                   Output root. Defaults to results/kitscope.
  --python PATH                     Python executable. Defaults to python.
  --representation-epochs N         Kit-DANN representation epochs. Defaults to 25.
  --limit-per-split-label N         Optional small-run cap. Defaults to 0.
  -h, --help                        Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root|-RepoRoot)
      repo_root="$2"
      shift 2
      ;;
    --data-root|-DataRoot)
      data_root="$2"
      shift 2
      ;;
    --out-root|-OutRoot)
      out_root="$2"
      shift 2
      ;;
    --python|-Python)
      python_bin="$2"
      shift 2
      ;;
    --representation-epochs|-RepresentationEpochs)
      representation_epochs="$2"
      shift 2
      ;;
    --limit-per-split-label|-LimitPerSplitLabel)
      limit_per_split_label="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
start_dir="$(pwd)"

abs_path() {
  local value="$1"
  if [[ "$value" = /* ]]; then
    printf '%s\n' "$value"
  else
    printf '%s\n' "$start_dir/$value"
  fi
}

if [[ -z "$repo_root" ]]; then
  repo_root="$(cd "$script_dir/.." && pwd)"
else
  repo_root="$(abs_path "$repo_root")"
fi

if [[ -z "$data_root" ]]; then
  data_root="$repo_root/data/kitscope-release"
else
  data_root="$(abs_path "$data_root")"
fi

if [[ -z "$out_root" ]]; then
  out_root="$repo_root/results/kitscope"
else
  out_root="$(abs_path "$out_root")"
fi

mkdir -p "$out_root"
input_root="$out_root/runner_inputs"
feature_shard_root="$out_root/feature_shards"
feature_root="$out_root/features_merged"
result_root="$out_root/results"
mkdir -p "$input_root" "$feature_shard_root" "$feature_root" "$result_root"

export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
status_path="$out_root/status.json"

write_status() {
  local status="$1"
  shift
  "$python_bin" - "$status_path" "$status" "$@" <<'PY'
import datetime as _dt
import json
import sys

path = sys.argv[1]
status = sys.argv[2]
items = sys.argv[3:]
payload = {"status": status}
now = _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")
if status == "started":
    payload["started_utc"] = now
elif status == "complete":
    payload["completed_utc"] = now
else:
    payload["updated_utc"] = now
for item in items:
    key, value = item.split("=", 1)
    payload[key] = value
with open(path, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
    f.write("\n")
PY
}

run_python() {
  local step="$1"
  local stdout="$2"
  local stderr="$3"
  shift 3
  "$python_bin" "$@" >"$stdout" 2>"$stderr"
  local exit_code=$?
  if [[ "$exit_code" -ne 0 ]]; then
    echo "$step failed with exit code $exit_code; see $stdout and $stderr" >&2
    exit "$exit_code"
  fi
}

write_status "started" \
  "repo_root=$repo_root" \
  "data_root=$data_root" \
  "out_root=$out_root"

prepare_args=(
  -B "$repo_root/scripts/prepare_public_release_inputs.py"
  --data-root "$data_root"
  --out-dir "$input_root"
)
if [[ "$limit_per_split_label" -gt 0 ]]; then
  prepare_args+=(--limit-per-split-label "$limit_per_split_label")
fi

run_python "input preparation" \
  "$out_root/01_prepare_inputs_stdout.log" \
  "$out_root/01_prepare_inputs_stderr.log" \
  "${prepare_args[@]}"

write_status "inputs_prepared" "input_root=$input_root"

extraction_config="$repo_root/configs/universal_interactive_phishing_gateway_oracle_plus_noguard.yaml"
run_python "feature extraction" \
  "$out_root/02_feature_shard_stdout.log" \
  "$out_root/02_feature_shard_stderr.log" \
  -B "$repo_root/scripts/preprocess_distributed_dataset.py" \
  feature-shard \
  --input-glob "$input_root/sample_inputs.txt" \
  --out-dir "$feature_shard_root" \
  --shard-id main_random \
  --scenario-set none \
  --config "main_random=$extraction_config" \
  --expected-post-load-guard-mode none \
  --causal-all-onsets \
  --candidate-decision-delays 0.75,1.5,2.5,3.5 \
  --candidate-scan-step 0.5

run_python "feature merge" \
  "$out_root/03_merge_features_stdout.log" \
  "$out_root/03_merge_features_stderr.log" \
  -B "$repo_root/scripts/preprocess_distributed_dataset.py" \
  merge-features \
  --shard-root "$feature_shard_root" \
  --out-root "$feature_root"

features_csv="$feature_root/main_random/features.csv"
features_parquet="$feature_root/main_random/features.parquet"
split_csv="$input_root/split_manifest.csv"
kit_labels_csv="$input_root/kit_labels.csv"

if [[ ! -f "$features_parquet" ]]; then
  echo "feature merge did not produce the parquet table required by the formal random-route runner: $features_parquet" >&2
  exit 1
fi

run_python "main random fair comparison" \
  "$out_root/04_main_random_stdout.log" \
  "$out_root/04_main_random_stderr.log" \
  -B "$repo_root/scripts/run_fair_multiscale_window_comparison.py" \
  --features-parquet "$features_parquet" \
  --column-spec "$repo_root/configs/comparison_column_spec.json" \
  --split-manifest "$split_csv" \
  --kit-labels-csv "$kit_labels_csv" \
  --out-dir "$result_root" \
  --seed 42 \
  --alpha 0.03 \
  --onset-oof-folds 5 \
  --nms-merge-s 2.0 \
  --max-decision-delay-s 3.5 \
  --score-batch-size 8192 \
  --condition five_windows \
  --variant dynamic_rank8 \
  --balanced-kit-dann \
  --fitdev-refit \
  --representation-weight-grid 0,0.1,0.2,0.3,0.4,0.5,0.7,1.0 \
  --representation-epochs "$representation_epochs" \
  --representation-early-stop-patience 5 \
  --representation-lam-kit 0.3 \
  --representation-lam-era 0.3 \
  --dynamic-rank-budget 8 \
  --headline-aggregation top2 \
  --single-candidate-target-fpr 0.002 \
  --candidate-decision-delays-s 0.75,1.5,2.5,3.5

write_status "complete" \
  "feature_file=$features_csv" \
  "feature_parquet=$features_parquet" \
  "result_root=$result_root"

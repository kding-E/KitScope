# KitScope

KitScope detects interactive phishing workflows from encrypted traffic metadata.
This repository contains the KitScope implementation, configurations, dataset
interface, label-generation pipeline, evaluation scripts, and tests.

KitScope supports two traffic-visibility profiles:

- `KitScope-CR`: context-rich profile, 425-dimensional route.
- `KitScope-EV`: endpoint-visible profile, 228-dimensional route.

The dataset is hosted outside Git because of file size:

```text
https://ug.link/dxp4800s-xy/filemgr/share-download/?id=59aca5f970d84f35875fa2614a29b39c
```

Extract it under `data/kitscope-release/`, or pass another path to the scripts.
For the traffic-detection pipeline without rebuilding kit-family labels, download
only `pcap.zip` and `json.zip`. Download the remaining archives when rebuilding
the kit-family labels or using the complete dataset bundle.

## Quick Start

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[test,reproduce]"
python -m kitscope.verify_main --config configs/smoke.yaml --data-root data/smoke --out results/smoke
pytest
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[test,reproduce]"
python -m kitscope.verify_main --config configs/smoke.yaml --data-root data/smoke --out results/smoke
pytest
```

The command above uses `data/smoke/` to check installation. Use the full
dataset bundle for KitScope experiments.

## Data Layout

KitScope uses one key, `capture_id`, across all sample files:

```text
data/kitscope-release/
  pcap/<capture_id>.pcap
  har/<capture_id>.har
  json/<capture_id>.json
  screenshots/<capture_id>/*.png
  static_features/<capture_id>/
    html/
    dom_snapshots/
    label_only/html/
    label_only/dom/
    label_only/page_state/
  manifest.jsonl
  splits/url_group_weights.csv
```

`json/<capture_id>.json` stores the per-capture metadata. The static kit-family
pipeline uses `capture_id` to bind that metadata and HAR file to
`static_features/<capture_id>/`.

## Reproducing KitScope

Run KitScope on the full dataset:

Windows PowerShell:

```powershell
.\scripts\run_kitscope.ps1 `
  -DataRoot data\kitscope-release `
  -OutRoot results\kitscope `
  -Python .\.venv\Scripts\python.exe
```

Linux/macOS:

```bash
chmod +x scripts/run_kitscope.sh
./scripts/run_kitscope.sh \
  --data-root data/kitscope-release \
  --out-root results/kitscope \
  --python .venv/bin/python
```

The runner prepares dataset inputs, extracts traffic features, trains KitScope,
and writes predictions and summary metrics under `results/kitscope/`.

## Kit-Family Labels

The Kit-DANN route uses final backend-kit family labels. The code for rebuilding
those labels is included:

```text
scripts/preprocess_distributed_dataset.py static-shard
scripts/preprocess_distributed_dataset.py merge-static
scripts/rebuild_snapshot_kit_labels.py
scripts/build_backend_static_family_assignments.py
scripts/cluster_backend_kit_labels.py
scripts/merge_backend_kit_fragments.py
scripts/merge_backend_kit_static_neighbors.py
scripts/apply_backend_kit_audit_corrections.py
scripts/rebuild_backend_kit_evidence_corrections.py
scripts/build_backend_kit_final_labels.py
scripts/audit_layer2_backend_kit_identity.py
```

The dataset bundle includes `static_features/<capture_id>/` so these scripts
can rebuild the static evidence CSV and final labels from the dataset files.
Use the rebuilt manifest to materialize the runner input `kit_labels.csv`:

Windows PowerShell:

```powershell
python scripts/prepare_public_release_inputs.py `
  --data-root data/kitscope-release `
  --out-dir results/kitscope/runner_inputs

python scripts/preprocess_distributed_dataset.py static-shard `
  --input-list results/kitscope/runner_inputs/static_snapshot_manifest.csv `
  --out-dir results/kitscope/static_shards `
  --shard-id public_static `
  --skip-errors

python scripts/preprocess_distributed_dataset.py merge-static `
  --shard-root results/kitscope/static_shards `
  --out results/kitscope/static_merged/static_features.csv

python scripts/rebuild_snapshot_kit_labels.py `
  --snapshot-manifest results/kitscope/runner_inputs/static_snapshot_manifest.csv `
  --static-features results/kitscope/static_merged/static_features.csv `
  --out-root results/kitscope/kit_labels_final

python scripts/prepare_public_release_inputs.py `
  --data-root data/kitscope-release `
  --out-dir results/kitscope/runner_inputs `
  --kit-label-manifest results/kitscope/kit_labels_final/kit_label_manifest.csv
```

Linux/macOS:

```bash
python scripts/prepare_public_release_inputs.py \
  --data-root data/kitscope-release \
  --out-dir results/kitscope/runner_inputs

python scripts/preprocess_distributed_dataset.py static-shard \
  --input-list results/kitscope/runner_inputs/static_snapshot_manifest.csv \
  --out-dir results/kitscope/static_shards \
  --shard-id public_static \
  --skip-errors

python scripts/preprocess_distributed_dataset.py merge-static \
  --shard-root results/kitscope/static_shards \
  --out results/kitscope/static_merged/static_features.csv

python scripts/rebuild_snapshot_kit_labels.py \
  --snapshot-manifest results/kitscope/runner_inputs/static_snapshot_manifest.csv \
  --static-features results/kitscope/static_merged/static_features.csv \
  --out-root results/kitscope/kit_labels_final

python scripts/prepare_public_release_inputs.py \
  --data-root data/kitscope-release \
  --out-dir results/kitscope/runner_inputs \
  --kit-label-manifest results/kitscope/kit_labels_final/kit_label_manifest.csv
```

## Evaluation

After downloading the dataset bundle:

Windows PowerShell or Linux/macOS:

```bash
python -m kitscope.verify_main --config configs/main_random.yaml --data-root data/kitscope-release --out results/kitscope_eval
```

The evaluation command computes `F1`, `FPR`, `R@3`, `AP`, `U-F1`, and `U-FPR`
from prediction CSVs and URL-equal weights.

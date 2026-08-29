from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def read_manifest(data_root: Path, manifest_name: str = "manifest.jsonl") -> pd.DataFrame:
    manifest = data_root / manifest_name
    if not manifest.is_file() and manifest.suffix.lower() == ".jsonl":
        csv_manifest = manifest.with_suffix(".csv")
        if csv_manifest.is_file():
            manifest = csv_manifest
    if not manifest.is_file():
        raise FileNotFoundError(f"missing release manifest: {manifest}")
    if manifest.suffix.lower() == ".jsonl":
        rows: list[dict] = []
        with manifest.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        frame = pd.DataFrame(rows)
    else:
        frame = pd.read_csv(manifest, dtype=str, keep_default_na=False, low_memory=False)
    if "capture_id" not in frame.columns:
        raise ValueError(f"manifest has no capture_id column: {manifest}")
    if frame["capture_id"].astype(str).duplicated().any():
        raise ValueError("manifest contains duplicate capture_id values")
    return frame.fillna("")


def _existing(path: Path) -> Path | None:
    return path if path.exists() else None


def sample_input_path(data_root: Path, row: pd.Series) -> Path:
    capture_id = str(row.get("capture_id") or "").strip()
    if not capture_id:
        raise ValueError("manifest row has no capture_id")

    json_rel = str(row.get("json_path") or "").strip()
    if json_rel:
        path = _existing(data_root / json_rel)
        if path is not None:
            return path

    by_type = _existing(data_root / "json" / f"{capture_id}.json")
    if by_type is not None:
        return by_type

    sample_dir = str(row.get("sample_dir") or row.get("legacy_sample_dir") or "").strip()
    if sample_dir:
        path = _existing(data_root / sample_dir)
        if path is not None:
            return path

    legacy = _existing(data_root / "samples" / capture_id)
    if legacy is not None:
        return legacy

    raise FileNotFoundError(f"cannot locate public sample input for capture_id={capture_id}")


def attach_sample_inputs(data_root: Path, manifest: pd.DataFrame) -> pd.DataFrame:
    out = manifest.copy()
    out["sample_input"] = [
        str(sample_input_path(data_root, row))
        for _, row in manifest.iterrows()
    ]
    return out


def select_small_batch(
    manifest: pd.DataFrame,
    per_partition_label: int | None = None,
    max_samples: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    frame = manifest.copy()
    if per_partition_label and per_partition_label > 0:
        parts = []
        for (_partition, _label), group in frame.groupby(["partition", "label"], sort=True):
            parts.append(group.sample(n=min(int(per_partition_label), len(group)), random_state=seed))
        frame = pd.concat(parts, ignore_index=True) if parts else frame.iloc[0:0].copy()
    if max_samples and max_samples > 0 and len(frame) > int(max_samples):
        frame = frame.sample(n=int(max_samples), random_state=seed).reset_index(drop=True)
    return frame.reset_index(drop=True)

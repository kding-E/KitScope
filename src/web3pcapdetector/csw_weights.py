"""Validated merge helpers for audited row-level covariate-shift weights."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def _path_key(value: object) -> str:
    return str(value).strip().replace("\\", "/").rstrip("/").casefold()


def _candidate_key(value: object) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(numeric):
        if float(numeric).is_integer():
            return str(int(numeric))
        return format(float(numeric), ".17g")
    return str(value).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def apply_row_weights(
    fit_frame: pd.DataFrame,
    weights_path: str | Path,
    base_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Return normalized fit weights after an exact candidate-window merge.

    Extra rows in the adapter artifact are permitted because the adapter is fit
    before the detector's frozen learned-rank mask.  Every detector fit row,
    however, must have exactly one audited weight.
    """
    path = Path(weights_path)
    weights = pd.read_parquet(path)
    # Static feature tables use candidate_id.  Packet-prefix causal tables use
    # candidate_rank as their stable per-session candidate identity.  Require
    # the same explicit key in both frames and never mix the two schemes.
    if "candidate_id" in fit_frame.columns and "candidate_id" in weights.columns:
        candidate_column = "candidate_id"
    elif "candidate_rank" in fit_frame.columns and "candidate_rank" in weights.columns:
        candidate_column = "candidate_rank"
    else:
        raise ValueError(
            "row-level adaptation weights require a shared candidate_id or candidate_rank key"
        )
    key_columns = ["zip_path", "window_name", candidate_column]
    required = {*key_columns, "applied_weight"}
    missing = sorted(required - set(weights.columns))
    if missing:
        raise ValueError(f"row-level adaptation weights miss columns: {missing}")
    frame_missing = sorted(set(key_columns) - set(fit_frame.columns))
    if frame_missing:
        raise ValueError(f"classifier fit rows miss CSW key columns: {frame_missing}")
    weights = weights.copy()
    fit_keys = fit_frame[key_columns].copy()
    for frame in (weights, fit_keys):
        frame["__path_key"] = frame.zip_path.map(_path_key)
        frame["__window_key"] = frame.window_name.astype(str).str.strip().str.casefold()
        frame["__candidate_key"] = frame[candidate_column].map(_candidate_key)
    merge_keys = ["__path_key", "__window_key", "__candidate_key"]
    if weights.duplicated(merge_keys).any():
        raise ValueError("row-level adaptation weights contain duplicate normalized keys")
    values = pd.to_numeric(weights.applied_weight, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("row-level adaptation weights must be finite and positive")
    weights["__csw_weight"] = values
    joined = fit_keys.merge(
        weights[merge_keys + ["__csw_weight"]], on=merge_keys,
        how="left", validate="one_to_one", sort=False,
    )
    missing_rows = int(joined.__csw_weight.isna().sum())
    if missing_rows:
        raise RuntimeError(f"row-level adaptation weights miss {missing_rows} classifier-fit rows")
    csw = joined.__csw_weight.to_numpy(dtype=float)
    base = np.ones(len(fit_frame), dtype=float) if base_weights is None else np.asarray(base_weights, dtype=float)
    if len(base) != len(fit_frame):
        raise ValueError("base sample-weight length does not match classifier fit rows")
    combined = base * csw
    if not np.isfinite(combined).all() or combined.sum() <= 0:
        raise ValueError("combined row-level adaptation weights are invalid")
    combined *= len(combined) / combined.sum()
    ess = float(combined.sum() ** 2 / np.square(combined).sum())
    audit = {
        "weighting_unit": "row", "weights_path": str(path.resolve()),
        "candidate_key_column": candidate_column,
        "weights_sha256": sha256_file(path), "classifier_fit_rows": int(len(fit_frame)),
        "adapter_weight_rows": int(len(weights)), "missing_classifier_fit_rows": 0,
        "effective_sample_size": ess, "effective_sample_size_fraction": ess / max(1, len(combined)),
    }
    return combined, audit

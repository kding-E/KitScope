#!/usr/bin/env python3
"""Fair adaptive-only versus five-window Layer-1 comparison.

The two conditions consume the same multi-scale, 0.5-second-scan causal onset
pool.  The onset ranker is trained once per representation from adaptive rows
and reused by both conditions.  The only manipulated factor is whether the
phishing classifier sees the adaptive row alone or all five causally available
window rows.  Aggregation is always over distinct candidates, never over two
windows from the same candidate.
"""
from __future__ import annotations

import argparse
import gc
import heapq
import json
import os
import pathlib
import sys
import time
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from run_causal_random_all_onsets_experiment import (  # noqa: E402
    _conformal_pvalues,
    _fit_imputed_model,
    _make_model,
    _metrics,
    _predict,
    _session_normalized_weights,
)
from layer1_kit_dann_representation import (  # noqa: E402
    fit_kit_dann_scorer,
    load_kit_label_map,
)
from run_layer1_openset_suite import _blend_representation_score  # noqa: E402
from web3pcapdetector.csw_weights import apply_row_weights  # noqa: E402


WINDOW_CONDITIONS = {
    "adaptive_only": ["adaptive"],
    "five_windows": ["adaptive", "w2", "w4", "w7", "w10"],
}
VARIANTS = ("all_onsets", "onset_soft_feature", "dynamic_rank8")
CLASSIFIER_MODELS = (
    "lightgbm",
    "xgboost",
    "hist_gradient_boosting",
    "extra_trees",
    "logistic",
)
WINDOW_ORDER = {"adaptive": 0, "w2": 1, "w4": 2, "w7": 3, "w10": 4}


class _ImputedLogisticClassifier:
    """Minimal sklearn-compatible wrapper for the linear sensitivity arm.

    The tree learners retain their native missing-value behavior. Logistic
    regression necessarily adds fit-only median imputation and standardization;
    both transformers are fitted on the same classifier-fit rows and are saved
    with the model, so evaluation never influences preprocessing.
    """

    def __init__(self, seed: int):
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.classifier = LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            max_iter=2000,
            random_state=int(seed),
        )

    def fit(self, X, y, sample_weight=None):
        matrix = self.imputer.fit_transform(X)
        matrix = self.scaler.fit_transform(matrix)
        self.classifier.fit(matrix, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X):
        matrix = self.imputer.transform(X)
        matrix = self.scaler.transform(matrix)
        return self.classifier.predict_proba(matrix)


def _make_classifier_model(kind: str, seed: int):
    """Create only the final phishing classifier for learner sensitivity.

    The causal onset ranker remains the frozen LightGBM implementation imported
    as ``_make_model``. This keeps the experiment scoped to the classifier
    backbone instead of silently changing candidate selection as well.
    """

    kind = str(kind)
    threads = int(os.environ.get("LAYER1_CLASSIFIER_N_JOBS", "-1"))
    if kind == "lightgbm":
        return _make_model(int(seed), ranker=False)
    if kind == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            objective="binary:logistic",
            n_estimators=700,
            max_depth=5,
            learning_rate=0.035,
            subsample=0.90,
            colsample_bytree=0.88,
            reg_lambda=1.5,
            reg_alpha=0.05,
            min_child_weight=2.0,
            tree_method="hist",
            eval_metric="logloss",
            random_state=int(seed),
            n_jobs=threads,
        )
    if kind == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.03,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            random_state=int(seed),
        )
    if kind == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=int(seed),
            n_jobs=threads,
        )
    if kind == "logistic":
        return _ImputedLogisticClassifier(int(seed))
    raise ValueError(f"unsupported classifier model: {kind!r}")


def _write_json(path: pathlib.Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _normalize_paths(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace("\\", "/", regex=False).str.rstrip("/")


def _remote_path_variants(paths: set[str]) -> list[str]:
    values = set(paths)
    values.update(str(path).replace("/", "\\") for path in paths)
    return list(values)


def _candidate_key(session_id: np.ndarray, rank: np.ndarray) -> np.ndarray:
    return (
        np.asarray(session_id, dtype=np.int64) << np.int64(32)
    ) | (np.asarray(rank, dtype=np.int64) & np.int64(0xFFFFFFFF))


def _canonical_path(value: str) -> str:
    return str(value).strip().replace("\\", "/").rstrip("/").casefold()


def _load_manifest(path: pathlib.Path) -> tuple[pd.DataFrame, dict[str, int]]:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "sample_path" not in frame.columns:
        if "zip_path" not in frame.columns:
            raise ValueError("split manifest misses sample_path or zip_path")
        frame["sample_path"] = frame["zip_path"]
    required = {"capture_id", "sample_path", "label", "partition", "fit_role", "source"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"split manifest misses columns: {missing}")
    if frame.capture_id.duplicated().any() or frame.sample_path.duplicated().any():
        raise ValueError("split manifest session identifiers are not unique")
    # Several formal grouped/OOD manifests predate the explicit fine_source
    # column.  In the canonical manifest, fine_source is exactly source for
    # every supported Layer1 source family, so derive that same semantic field
    # instead of rejecting an otherwise identical split.  This is metadata
    # completion only; it does not inspect labels or evaluation outcomes.
    if "fine_source" not in frame.columns:
        frame["fine_source"] = frame["source"]
    if "zip_path" in frame.columns:
        frame = frame.drop(columns=["zip_path"])
    frame = frame.rename(columns={"sample_path": "zip_path"}).copy()
    frame["zip_path"] = _normalize_paths(frame.zip_path)
    frame["y"] = frame.label.str.casefold().eq("phishing").astype(np.int8)
    frame["session_id"] = np.arange(len(frame), dtype=np.int32)
    return frame, dict(zip(frame.zip_path, frame.session_id))


def _window_filter(names: list[str]) -> ds.Expression:
    if len(names) == 1:
        return ds.field("window_name") == names[0]
    return ds.field("window_name").isin(names)


def _scan_to_frame(
    dataset: ds.Dataset,
    columns: list[str],
    windows: list[str],
    session_paths: set[str] | None = None,
) -> pd.DataFrame:
    expression = _window_filter(windows)
    if session_paths is not None:
        expression = expression & ds.field("zip_path").isin(
            _remote_path_variants(session_paths)
        )
    table = dataset.to_table(columns=columns, filter=expression, use_threads=True)
    frame = table.to_pandas(split_blocks=True, self_destruct=True)
    if "zip_path" in frame:
        frame["zip_path"] = _normalize_paths(frame.zip_path)
    return frame


def _attach_ids(
    frame: pd.DataFrame,
    session_to_id: dict[str, int],
) -> pd.DataFrame:
    sid = frame.zip_path.map(session_to_id)
    if sid.isna().any():
        examples = frame.loc[sid.isna(), "zip_path"].drop_duplicates().head().tolist()
        raise ValueError(f"feature rows do not map to split manifest: {examples}")
    frame["session_id"] = sid.astype(np.int32)
    rank = pd.to_numeric(frame.candidate_rank, errors="raise").astype(np.int32)
    frame["candidate_rank"] = rank
    frame["candidate_key"] = _candidate_key(frame.session_id, rank)
    return frame


def _training_weights(
    frame: pd.DataFrame,
    target: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict]:
    """Combine the frozen session-balanced loss with audited causal CSW.

    CSW remains a multiplicative classifier/representation-training weight;
    it never changes the onset ranker, candidate arrival order, dynamic top-k
    admission, calibration, or evaluation selection boundary.
    """
    base = _session_normalized_weights(frame, target)
    path = getattr(args, "precomputed_fit_row_weights", None)
    if not path:
        return base, {"enabled": False}
    combined, audit = apply_row_weights(frame, path, base)
    return combined, {"enabled": True, **audit}


def _validate_causal_csw_contract(
    args: argparse.Namespace,
    manifest: pd.DataFrame,
) -> dict:
    weights_value = getattr(args, "precomputed_fit_row_weights", None)
    summary_value = getattr(args, "adaptation_summary", None)
    if not weights_value and not summary_value:
        return {"enabled": False}
    if not weights_value or not summary_value:
        raise ValueError(
            "packet-causal CSW requires both --precomputed-fit-row-weights "
            "and --adaptation-summary"
        )
    weights_path = pathlib.Path(weights_value)
    summary_path = pathlib.Path(summary_value)
    if not weights_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(
            f"packet-causal CSW artifact missing: {weights_path} / {summary_path}"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    checks = {
        "status": summary.get("status") in {"adapted", "rejected_fallback_frozen"},
        "weighting_unit": summary.get("weighting_unit") == "row",
        "target_partition": summary.get("target_partition") == "evaluation",
        "evaluation_labels_read": int(summary.get("evaluation_labels_read", -1)) == 0,
        "row_support_mode": summary.get("row_support_mode") == "causal_dynamic_pool",
        "candidate_key": summary.get("row_candidate_key_column") == "candidate_rank",
    }
    if not all(checks.values()):
        raise RuntimeError(f"packet-causal CSW contract failed: {checks}")
    weight_sessions = pd.read_parquet(weights_path, columns=["zip_path"])
    weight_keys = set(_normalize_paths(weight_sessions.zip_path))
    fit_keys = set(manifest.loc[manifest.partition.eq("fit"), "zip_path"])
    leaked = weight_keys - fit_keys
    if leaked:
        raise RuntimeError(
            f"packet-causal CSW weights contain {len(leaked)} non-fit sessions"
        )
    return {
        "enabled": True,
        "weights_path": str(weights_path.resolve()),
        "adaptation_summary_path": str(summary_path.resolve()),
        "checks": checks,
        "adaptation_status": summary.get("status"),
        "protocol": summary.get("protocol"),
        "rejected_reasons": summary.get("rejected_reasons", []),
        "fit_weight_sessions": int(len(weight_keys)),
        "evaluation_labels_used": False,
    }


def _mother_table_audit(
    dataset: ds.Dataset,
    manifest: pd.DataFrame,
    session_to_id: dict[str, int],
    max_delay_s: float,
    merge_s: float,
) -> dict:
    columns = [
        "zip_path", "window_name", "window_available_epoch",
        "candidate_start_epoch", "candidate_emit_epoch", "candidate_rank",
        "candidate_source",
    ]
    # The shared causal feature store can be a superset of a dedicated
    # grouped/OOD manifest.  Restrict the audit to the frozen split here;
    # rows belonging to other experiments are not orphans of this split.
    manifest_paths = set(manifest.zip_path)
    frame = _scan_to_frame(
        dataset, columns, WINDOW_CONDITIONS["five_windows"], manifest_paths
    )
    frame = _attach_ids(frame, session_to_id)
    # A deliberately censored observation prefix may end before the ordinary
    # streaming NMS look-ahead closes. In that case the extractor finalizes an
    # otherwise-open candidate exactly at the frozen observation horizon. The
    # split manifest is the authoritative, label-independent source of that
    # horizon. Full-session manifests without ``load_end_epoch`` retain the
    # original strict NMS+delay check unchanged.
    if "load_end_epoch" in manifest.columns:
        end_by_path = dict(zip(
            manifest.zip_path,
            pd.to_numeric(manifest.load_end_epoch, errors="coerce"),
        ))
        frame["observation_end_epoch"] = pd.to_numeric(
            frame.zip_path.map(end_by_path), errors="coerce"
        )
    else:
        frame["observation_end_epoch"] = np.nan
    frame["window_available_epoch"] = pd.to_numeric(
        frame.window_available_epoch, errors="coerce"
    )
    frame["candidate_start_epoch"] = pd.to_numeric(
        frame.candidate_start_epoch, errors="coerce"
    )
    frame["candidate_emit_epoch"] = pd.to_numeric(
        frame.candidate_emit_epoch, errors="coerce"
    )
    finite = frame.dropna(subset=[
        "window_available_epoch", "candidate_start_epoch", "candidate_emit_epoch"
    ])
    regular_emit = (
        finite.candidate_emit_epoch + 1e-7
        >= finite.candidate_start_epoch + float(merge_s) + float(max_delay_s)
    )
    horizon_emit = (
        finite.observation_end_epoch.notna()
        & ((finite.candidate_emit_epoch - finite.observation_end_epoch).abs() <= 1e-7)
        & (finite.candidate_emit_epoch + 1e-7 >= finite.candidate_start_epoch)
    )
    grouped = frame.groupby("candidate_key", sort=False)
    sizes = grouped.size()
    window_unique = grouped.window_name.nunique()
    adaptive_counts = frame.window_name.eq("adaptive").groupby(frame.candidate_key).sum()
    start_span = grouped.candidate_start_epoch.agg(lambda x: float(x.max() - x.min()))
    emit_span = grouped.candidate_emit_epoch.agg(lambda x: float(x.max() - x.min()))
    feature_sessions = set(frame.zip_path.unique())
    manifest_sessions = set(manifest.zip_path)
    sources = frame.drop_duplicates("candidate_key").candidate_source.astype(str)
    audit = {
        "rows": int(len(frame)),
        "candidates": int(len(sizes)),
        "manifest_sessions": int(len(manifest)),
        "feature_sessions": int(len(feature_sessions)),
        "sessions_without_candidate": int(len(manifest_sessions - feature_sessions)),
        "orphan_feature_sessions": int(len(feature_sessions - manifest_sessions)),
        "windows": sorted(frame.window_name.astype(str).unique().tolist()),
        "exactly_five_rows_per_candidate": bool(sizes.eq(5).all()),
        "exactly_five_unique_windows_per_candidate": bool(window_unique.eq(5).all()),
        "exactly_one_adaptive_row_per_candidate": bool(adaptive_counts.eq(1).all()),
        "candidate_start_stable_across_windows": bool(start_span.le(1e-7).all()),
        "candidate_emit_stable_across_windows": bool(emit_span.le(1e-7).all()),
        "window_available_after_emit": bool(
            (finite.window_available_epoch + 1e-7 >= finite.candidate_emit_epoch).all()
        ),
        "emit_after_nms_and_max_delay": bool(regular_emit.all()),
        "emit_after_nms_or_at_observation_horizon": bool((regular_emit | horizon_emit).all()),
        "horizon_finalized_rows": int((~regular_emit & horizon_emit).sum()),
        "horizon_finalized_candidates": int(
            finite.loc[~regular_emit & horizon_emit, "candidate_key"].nunique()
        ),
        "candidate_emit_not_before_start": bool((
            finite.candidate_emit_epoch + 1e-7 >= finite.candidate_start_epoch
        ).all()),
        "traffic_stream_candidates_only": bool(
            sources.str.startswith("traffic_heuristic_stream").all()
        ),
        "candidate_source_counts": sources.value_counts().sort_index().to_dict(),
        "static_complete_session_rank_used": False,
        "single_candidate_override": False,
    }
    audit["passed"] = bool(all([
        audit["orphan_feature_sessions"] == 0,
        audit["exactly_five_rows_per_candidate"],
        audit["exactly_five_unique_windows_per_candidate"],
        audit["exactly_one_adaptive_row_per_candidate"],
        audit["candidate_start_stable_across_windows"],
        audit["candidate_emit_stable_across_windows"],
        audit["window_available_after_emit"],
        audit["emit_after_nms_or_at_observation_horizon"],
        audit["candidate_emit_not_before_start"],
        audit["traffic_stream_candidates_only"],
    ]))
    del frame, grouped, sizes, window_unique, adaptive_counts, start_span, emit_span
    gc.collect()
    if not audit["passed"]:
        raise RuntimeError(f"mother-table fairness audit failed: {audit}")
    return audit


def _session_fold_ids(
    manifest: pd.DataFrame, folds: int, seed: int, *, include_fit_dev: bool = False
) -> dict[int, int]:
    mask = manifest.partition.eq("fit")
    if not include_fit_dev:
        mask &= manifest.fit_role.eq("fit_train")
    fit = manifest[mask][["session_id", "y"]].copy()
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    mapping: dict[int, int] = {}
    for fold, (_, valid) in enumerate(splitter.split(fit.session_id, fit.y)):
        mapping.update({int(value): fold for value in fit.iloc[valid].session_id})
    return mapping


def _fit_onset_ranker(
    adaptive: pd.DataFrame,
    manifest: pd.DataFrame,
    features: list[str],
    folds: int,
    seed: int,
    out_dir: pathlib.Path,
    *,
    include_fit_dev: bool = False,
    dynamic_rank_budget: int = 8,
) -> tuple[pd.DataFrame, dict]:
    # Candidate fusion/inner joins can preserve a sparse source index.  All
    # score arrays below are positional, so normalize once at this boundary
    # before dynamic-rank admission writes by row index.
    adaptive = adaptive.reset_index(drop=True).copy()
    scores = np.full(len(adaptive), np.nan, dtype=np.float32)
    fold_by_id = _session_fold_ids(
        manifest, folds, seed, include_fit_dev=include_fit_dev
    )
    fit_ids = set(fold_by_id)
    positive_fit_ids = set(manifest[
        manifest.session_id.isin(fit_ids) & manifest.y.eq(1)
    ].session_id.astype(int))
    fold_stats = []
    for fold in range(folds):
        valid_ids = {sid for sid, value in fold_by_id.items() if value == fold}
        train_mask = adaptive.session_id.isin(positive_fit_ids - valid_ids).to_numpy()
        valid_mask = adaptive.session_id.isin(valid_ids).to_numpy()
        target = adaptive.loc[train_mask, "candidate_hit_within_3s"].fillna(0).astype(int).to_numpy()
        if len(np.unique(target)) < 2:
            raise ValueError(f"onset OOF fold {fold} lacks both alignment labels")
        weights = _session_normalized_weights(adaptive.loc[train_mask], target)
        imputer, model = _fit_imputed_model(
            adaptive.loc[train_mask], features, target, weights,
            seed + fold, ranker=True,
        )
        scores[valid_mask] = _predict(
            imputer, model, adaptive.loc[valid_mask], features
        ).astype(np.float32)
        fold_stats.append({
            "fold": fold,
            "train_rows": int(train_mask.sum()),
            "scored_rows": int(valid_mask.sum()),
            "alignment_positive_rows": int(target.sum()),
        })
        del imputer, model, target, weights
        gc.collect()

    final_mask = adaptive.session_id.isin(positive_fit_ids).to_numpy()
    final_target = adaptive.loc[final_mask, "candidate_hit_within_3s"].fillna(0).astype(int).to_numpy()
    final_weights = _session_normalized_weights(adaptive.loc[final_mask], final_target)
    final_imputer, final_model = _fit_imputed_model(
        adaptive.loc[final_mask], features, final_target, final_weights,
        seed + 101, ranker=True,
    )
    non_fit_mask = ~adaptive.session_id.isin(fit_ids).to_numpy()
    scores[non_fit_mask] = _predict(
        final_imputer, final_model, adaptive.loc[non_fit_mask], features
    ).astype(np.float32)
    if np.isnan(scores).any():
        raise RuntimeError("onset alignment scores contain missing values")
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"imputer": final_imputer, "model": final_model, "features": features},
        out_dir / "onset_alignment_ranker.joblib",
    )
    onset = adaptive[[
        "zip_path", "session_id", "candidate_key", "candidate_rank",
        "window_available_epoch", "pcap_start_epoch",
    ]].copy()
    onset = onset.rename(columns={"window_available_epoch": "onset_available_epoch"})
    onset["onset_alignment_score"] = scores
    onset["ever_admitted_dynamic_rank8"] = _ever_admitted(
        onset, budget=int(dynamic_rank_budget)
    )
    onset.to_parquet(out_dir / "onset_candidate_map.parquet", index=False)
    protocol = {
        "target": "candidate_hit_within_3s",
        "fit_population": (
            "phishing complete outer-fit sessions; adaptive candidate row only"
            if include_fit_dev else
            "phishing fit_train sessions; adaptive candidate row only"
        ),
        "fit_score_semantics": f"{folds}-fold session-level OOF",
        "non_fit_score_semantics": (
            "ranker frozen on all phishing complete outer-fit sessions"
            if include_fit_dev else
            "ranker frozen on all phishing fit_train sessions"
        ),
        "fitdev_refit": bool(include_fit_dev),
        "folds": fold_stats,
        "final_train_rows": int(final_mask.sum()),
        "final_alignment_positive_rows": int(final_target.sum()),
        "dynamic_rank8_ever_admitted_candidates": int(
            onset.ever_admitted_dynamic_rank8.sum()
        ),
        "dynamic_rank_budget": int(dynamic_rank_budget),
    }
    return onset, protocol


def _ever_admitted(onset: pd.DataFrame, budget: int) -> np.ndarray:
    admitted = np.zeros(len(onset), dtype=bool)
    for _, group in onset.groupby("session_id", sort=False):
        ordered = group.sort_values(
            ["onset_available_epoch", "candidate_rank"], kind="stable"
        )
        heap: list[tuple[float, int, int]] = []
        for arrival, row in enumerate(ordered.itertuples()):
            item = (float(row.onset_alignment_score), -arrival, int(row.Index))
            if len(heap) < budget:
                heapq.heappush(heap, item)
                admitted[int(row.Index)] = True
            elif item[:2] > heap[0][:2]:
                heapq.heapreplace(heap, item)
                admitted[int(row.Index)] = True
    return admitted


def _lookup_candidate_values(
    keys: np.ndarray,
    series: pd.Series,
    name: str,
) -> np.ndarray:
    values = series.reindex(pd.Index(keys)).to_numpy()
    if pd.isna(values).any():
        raise RuntimeError(f"candidate lookup {name} misses rows")
    return values


def _fit_classifier_models(
    dataset: ds.Dataset,
    condition: str,
    manifest: pd.DataFrame,
    session_to_id: dict[str, int],
    onset: pd.DataFrame,
    base_features: list[str],
    seed: int,
    out_dir: pathlib.Path,
    variants: tuple[str, ...] = VARIANTS,
    *,
    include_fit_dev: bool = False,
    args: argparse.Namespace | None = None,
) -> tuple[dict[str, object], dict]:
    fit_mask = manifest.partition.eq("fit")
    if not include_fit_dev:
        fit_mask &= manifest.fit_role.eq("fit_train")
    fit_paths = set(manifest[fit_mask].zip_path)
    columns = ["zip_path", "candidate_rank", "window_name"] + base_features
    fit = _scan_to_frame(
        dataset, columns, WINDOW_CONDITIONS[condition], fit_paths
    )
    fit = _attach_ids(fit, session_to_id)
    onset_index = onset.set_index("candidate_key")
    keys = fit.candidate_key.to_numpy(dtype=np.int64)
    fit["onset_alignment_score"] = _lookup_candidate_values(
        keys, onset_index.onset_alignment_score, "onset_alignment_score"
    ).astype(np.float32)
    admitted = _lookup_candidate_values(
        keys, onset_index.ever_admitted_dynamic_rank8, "dynamic_admission"
    ).astype(bool)
    y_by_session = manifest.set_index("session_id").y
    target_all = y_by_session.reindex(fit.session_id).to_numpy(dtype=np.int8)
    models: dict[str, object] = {}
    stats: dict[str, object] = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    for offset, variant in enumerate(variants):
        mask = admitted if variant == "dynamic_rank8" else np.ones(len(fit), dtype=bool)
        features = list(base_features)
        if variant == "onset_soft_feature":
            features.append("onset_alignment_score")
        target = target_all[mask]
        if len(np.unique(target)) < 2:
            raise ValueError(f"{condition}/{variant} fit rows lack both classes")
        train_view = fit.loc[mask]
        weights, csw_audit = (
            _training_weights(train_view, target, args)
            if args is not None else
            (_session_normalized_weights(train_view, target), {"enabled": False})
        )
        classifier_model = (
            str(getattr(args, "classifier_model", "lightgbm"))
            if args is not None else "lightgbm"
        )
        model = _make_classifier_model(classifier_model, seed + offset)
        # Tree learners retain their native missing-value behavior. The linear
        # arm performs fit-only median imputation and standardization inside its
        # saved wrapper.
        model.fit(train_view[features], target, sample_weight=weights)
        variant_dir = out_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": model,
                "features": features,
                "native_missing": classifier_model != "logistic",
                "classifier_model": classifier_model,
                "preprocessing": (
                    "fit_only_median_imputation_and_standardization"
                    if classifier_model == "logistic"
                    else "native_missing_value_handling"
                ),
            },
            variant_dir / "model.joblib",
        )
        models[variant] = model
        stats[variant] = {
            "fit_rows": int(mask.sum()),
            "fit_sessions": int(train_view.session_id.nunique()),
            "features": len(features),
            "dynamic_ever_admitted_filter": variant == "dynamic_rank8",
            "sample_weighting": "session_normalized_balanced",
            "classifier_model": classifier_model,
            "candidate_onset_ranker_model": "lightgbm_frozen_across_arms",
            "precomputed_row_csw": csw_audit,
            "fitdev_refit": bool(include_fit_dev),
        }
        del train_view, target, weights
        gc.collect()
    del fit, target_all, admitted
    gc.collect()
    return models, stats


def _write_scored_events(
    dataset: ds.Dataset,
    condition: str,
    session_to_id: dict[str, int],
    onset: pd.DataFrame,
    base_features: list[str],
    models: dict[str, object],
    output: pathlib.Path,
    batch_size: int,
    variants: tuple[str, ...] = VARIANTS,
    representation_scorer=None,
    representation_weight: float = 0.0,
    representation_blend_mode: str = "positive_lift",
) -> dict:
    columns = [
        "zip_path", "candidate_rank", "window_name", "window_available_epoch"
    ] + base_features
    split_paths = set(session_to_id)
    split_filter = _window_filter(WINDOW_CONDITIONS[condition]) & ds.field(
        "zip_path"
    ).isin(_remote_path_variants(split_paths))
    scanner = dataset.scanner(
        columns=columns,
        filter=split_filter,
        batch_size=batch_size,
        use_threads=True,
    )
    onset_index = onset.set_index("candidate_key")
    output.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    rows = 0
    try:
        for batch_number, batch in enumerate(scanner.to_batches(), start=1):
            # Dataset filters can yield a zero-row batch at a fragment or row-
            # group boundary.  LightGBM rejects an empty prediction matrix.
            if batch.num_rows == 0:
                continue
            frame = batch.to_pandas(split_blocks=True, self_destruct=True)
            frame["zip_path"] = _normalize_paths(frame.zip_path)
            frame = _attach_ids(frame, session_to_id)
            keys = frame.candidate_key.to_numpy(dtype=np.int64)
            onset_score = _lookup_candidate_values(
                keys, onset_index.onset_alignment_score, "onset_alignment_score"
            ).astype(np.float32)
            onset_available = _lookup_candidate_values(
                keys, onset_index.onset_available_epoch, "onset_available_epoch"
            ).astype(np.float64)
            raw_available = pd.to_numeric(
                frame.window_available_epoch, errors="coerce"
            ).to_numpy(dtype=np.float64)
            frame["onset_alignment_score"] = onset_score
            event_columns = {
                "session_id": frame.session_id.to_numpy(dtype=np.int32),
                "candidate_key": keys,
                "candidate_rank": frame.candidate_rank.to_numpy(dtype=np.int32),
                "window_name": frame.window_name.astype(str).to_numpy(),
                "raw_available_epoch": raw_available,
                "soft_available_epoch": np.maximum(raw_available, onset_available),
                "onset_available_epoch": onset_available,
                "onset_alignment_score": onset_score,
            }
            if "all_onsets" in variants:
                event_columns["all_score"] = models["all_onsets"].predict_proba(
                    frame[base_features]
                )[:, 1].astype(np.float32)
            if "onset_soft_feature" in variants:
                event_columns["soft_score"] = models[
                    "onset_soft_feature"
                ].predict_proba(
                    frame[base_features + ["onset_alignment_score"]]
                )[:, 1].astype(np.float32)
            if "dynamic_rank8" in variants:
                dynamic_base_score = models[
                    "dynamic_rank8"
                ].predict_proba(frame[base_features])[:, 1].astype(np.float32)
                if representation_scorer is not None:
                    dynamic_representation_score = representation_scorer.score(frame)
                    dynamic_score = _blend_representation_score(
                        dynamic_base_score,
                        dynamic_representation_score,
                        float(representation_weight),
                        str(representation_blend_mode),
                    )
                    event_columns["dynamic_base_score"] = dynamic_base_score
                    event_columns["dynamic_dann_score"] = (
                        dynamic_representation_score.astype(np.float32)
                    )
                    event_columns["dynamic_score"] = dynamic_score.astype(np.float32)
                else:
                    event_columns["dynamic_score"] = dynamic_base_score
            events = pd.DataFrame(event_columns)
            table = pa.Table.from_pandas(events, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    output, table.schema, compression="zstd", compression_level=3,
                    use_dictionary=["window_name"], write_statistics=True,
                )
            writer.write_table(table)
            rows += len(events)
            if batch_number % 100 == 0:
                print(json.dumps({
                    "condition": condition, "score_batches": batch_number,
                    "score_rows": rows,
                }), flush=True)
            del frame, events, table
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError(f"no scored events for condition {condition}")
    return {"rows": rows, "event_file": str(output.resolve())}


def _top2_session_scores(
    events: pd.DataFrame,
    score_column: str,
    aggregation: str = "top2",
) -> tuple[dict[int, float], dict[int, int]]:
    candidates = events.groupby(
        ["session_id", "candidate_key"], sort=False
    )[score_column].max().reset_index()
    scores: dict[int, float] = {}
    counts: dict[int, int] = {}
    for sid, group in candidates.groupby("session_id", sort=False):
        values = group[score_column].to_numpy(dtype=float)
        counts[int(sid)] = len(values)
        if aggregation == "max":
            scores[int(sid)] = float(np.max(values)) if len(values) else -1.0
        elif len(values) < 2:
            scores[int(sid)] = -1.0
        else:
            scores[int(sid)] = float(np.mean(np.partition(values, -2)[-2:]))
    return scores, counts


def _simulate_dynamic_session(
    event_group: pd.DataFrame,
    onset_group: pd.DataFrame,
    alpha: float | None = None,
    benign_calibration: np.ndarray | None = None,
    *,
    budget: int = 8,
    aggregation: str = "top2",
    single_candidate_super_threshold: float | None = None,
) -> tuple[float, int, float]:
    combined: list[tuple[float, int, int, int, float]] = []
    ordered_onset = onset_group.sort_values(
        ["onset_available_epoch", "candidate_rank"], kind="stable"
    )
    for arrival, row in enumerate(ordered_onset.itertuples(index=False)):
        combined.append((
            float(row.onset_available_epoch), 0, arrival,
            int(row.candidate_key), float(row.onset_alignment_score),
        ))
    ordered_events = event_group.sort_values(
        ["raw_available_epoch", "candidate_rank", "window_name"], kind="stable"
    )
    for arrival, row in enumerate(ordered_events.itertuples(index=False)):
        combined.append((
            float(row.raw_available_epoch), 1, arrival,
            int(row.candidate_key), float(row.dynamic_score),
        ))
    combined.sort(key=lambda item: (item[0], item[1], item[2]))
    arrival_order: dict[int, int] = {}
    onset_scores: dict[int, float] = {}
    candidate_scores: dict[int, float] = {}
    active: list[int] = []
    best = -1.0
    max_active = 0
    alert_epoch = np.nan
    for epoch, kind, arrival, key, value in combined:
        if kind == 0:
            arrival_order[key] = len(arrival_order)
            onset_scores[key] = value
            active = sorted(
                onset_scores,
                key=lambda item: (-onset_scores[item], arrival_order[item]),
            )[:int(budget)]
            max_active = max(max_active, len(active))
        else:
            candidate_scores[key] = max(candidate_scores.get(key, -1.0), value)
        available = [candidate_scores[key] for key in active if key in candidate_scores]
        if (
            alpha is not None
            and not np.isfinite(alert_epoch)
            and len(available) == 1
            and single_candidate_super_threshold is not None
            and float(available[0]) >= float(single_candidate_super_threshold)
        ):
            alert_epoch = epoch
        if aggregation == "max" and available:
            current = float(np.max(available))
        elif len(available) < 2:
            continue
        else:
            current = float(np.mean(np.partition(np.asarray(available), -2)[-2:]))
        best = max(best, current)
        if (
            alpha is not None and benign_calibration is not None
            and not np.isfinite(alert_epoch)
            and _conformal_pvalues(np.asarray([current]), benign_calibration)[0] <= alpha
        ):
            alert_epoch = epoch
    return best, max_active, alert_epoch


def _dynamic_session_scores(
    events: pd.DataFrame,
    onset: pd.DataFrame,
    *,
    budget: int = 8,
    aggregation: str = "top2",
) -> tuple[dict[int, float], dict[int, int]]:
    onset_groups = {int(sid): group for sid, group in onset.groupby("session_id", sort=False)}
    scores: dict[int, float] = {}
    counts: dict[int, int] = {}
    for sid, group in events.groupby("session_id", sort=False):
        best, count, _ = _simulate_dynamic_session(
            group, onset_groups[int(sid)], budget=int(budget),
            aggregation=str(aggregation),
        )
        scores[int(sid)] = best
        counts[int(sid)] = count
    return scores, counts


def _prepare_representation_rows(
    dataset: ds.Dataset,
    condition: str,
    manifest: pd.DataFrame,
    session_to_id: dict[str, int],
    onset: pd.DataFrame,
    base_features: list[str],
) -> pd.DataFrame:
    fit_paths = set(manifest[manifest.partition.eq("fit")].zip_path)
    columns = [
        "zip_path", "candidate_rank", "window_name", "window_available_epoch"
    ] + base_features
    frame = _scan_to_frame(
        dataset, columns, WINDOW_CONDITIONS[condition], fit_paths
    )
    frame = _attach_ids(frame, session_to_id)
    onset_index = onset.set_index("candidate_key")
    keys = frame.candidate_key.to_numpy(dtype=np.int64)
    frame["onset_available_epoch"] = _lookup_candidate_values(
        keys, onset_index.onset_available_epoch, "onset_available_epoch"
    ).astype(np.float64)
    admitted = _lookup_candidate_values(
        keys, onset_index.ever_admitted_dynamic_rank8, "dynamic_admission"
    ).astype(bool)
    frame = frame.loc[admitted].reset_index(drop=True)
    meta = manifest.set_index("session_id")
    frame["label"] = np.where(
        meta.y.reindex(frame.session_id).to_numpy(dtype=np.int8) == 1,
        "phishing", "benign",
    )
    frame["fit_role"] = meta.fit_role.reindex(frame.session_id).to_numpy()
    return frame


def _fitdev_weight_score(
    valid: pd.DataFrame,
    onset: pd.DataFrame,
    manifest: pd.DataFrame,
    base_score: np.ndarray,
    representation_score: np.ndarray,
    weight: float,
    alpha: float,
    *,
    dynamic_rank_budget: int = 8,
    aggregation: str = "top2",
) -> dict:
    events = valid[[
        "session_id", "candidate_key", "candidate_rank", "window_name",
        "window_available_epoch",
    ]].rename(columns={"window_available_epoch": "raw_available_epoch"}).copy()
    events["dynamic_score"] = _blend_representation_score(
        np.asarray(base_score, dtype=float),
        np.asarray(representation_score, dtype=float),
        float(weight), "positive_lift",
    )
    scores, counts = _dynamic_session_scores(
        events, onset, budget=int(dynamic_rank_budget),
        aggregation=str(aggregation),
    )
    development = manifest[
        manifest.partition.eq("fit") & manifest.fit_role.eq("fit_dev")
    ].copy()
    development["session_score"] = (
        development.session_id.map(scores).fillna(-1.0).astype(float)
    )
    development["candidate_count"] = (
        development.session_id.map(counts).fillna(0).astype(int)
    )
    benign = development.loc[
        development.y.eq(0), "session_score"
    ].to_numpy(dtype=float)
    pvalue = _conformal_pvalues(
        development.session_score.to_numpy(dtype=float), benign
    )
    predicted = pvalue <= float(alpha)
    y = development.y.to_numpy(dtype=int)
    tp = int(np.sum(predicted & (y == 1)))
    fp = int(np.sum(predicted & (y == 0)))
    fn = int(np.sum(~predicted & (y == 1)))
    tn = int(np.sum(~predicted & (y == 0)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "weight": float(weight),
        "plain_f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "fpr": float(fp / max(fp + tn, 1)),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def _capture_date_lookup(manifest: pd.DataFrame):
    if "capture_start_utc" not in manifest:
        raise ValueError("balanced Kit-DANN requires manifest capture_start_utc")
    dates = pd.to_datetime(manifest.capture_start_utc, errors="coerce", utc=True)
    mapping = {
        _canonical_path(path): ("" if pd.isna(date) else date.strftime("%Y%m%d"))
        for path, date in zip(manifest.zip_path, dates)
    }
    return lambda value: mapping.get(_canonical_path(value), "")


def _fit_balanced_dann_selection(
    dataset: ds.Dataset,
    condition: str,
    manifest: pd.DataFrame,
    session_to_id: dict[str, int],
    onset: pd.DataFrame,
    base_features: list[str],
    base_model,
    args: argparse.Namespace,
    out_dir: pathlib.Path,
) -> tuple[object, dict]:
    rows = _prepare_representation_rows(
        dataset, condition, manifest, session_to_id, onset, base_features
    )
    train = rows[rows.fit_role.eq("fit_train")].reset_index(drop=True)
    valid = rows[rows.fit_role.eq("fit_dev")].reset_index(drop=True)
    if train.empty or valid.empty:
        raise ValueError("balanced Kit-DANN selection requires fit_train and fit_dev rows")
    train_y = train.label.eq("phishing").to_numpy(dtype=int)
    train_weight, train_csw_audit = _training_weights(train, train_y, args)
    valid_base = base_model.predict_proba(valid[base_features])[:, 1]
    kit_map = load_kit_label_map(str(args.kit_labels_csv), _canonical_path)
    onset_valid = onset[onset.session_id.isin(set(valid.session_id))].copy()
    weight_grid = [
        float(value) for value in str(args.representation_weight_grid).split(",")
        if str(value).strip()
    ]

    def callback(probability: np.ndarray, epoch: int, _: dict):
        sweep = [
            _fitdev_weight_score(
                valid, onset_valid, manifest, valid_base, probability,
                weight, float(args.alpha),
                dynamic_rank_budget=int(getattr(args, "dynamic_rank_budget", 8)),
                aggregation=str(getattr(args, "headline_aggregation", "top2")),
            )
            for weight in weight_grid
        ]
        selected = max(
            sweep,
            key=lambda item: (
                item["plain_f1"], -item["fpr"], item["recall"], -item["weight"]
            ),
        )
        return float(selected["plain_f1"]), {
            "epoch": int(epoch),
            "selected_weight": float(selected["weight"]),
            "selected": selected,
            "sweep": sweep,
        }

    scorer = fit_kit_dann_scorer(
        train,
        base_features,
        kit_label_map=kit_map,
        capture_date_of=_capture_date_lookup(manifest),
        canonical_path=_canonical_path,
        sample_weight=train_weight,
        valid_df=valid,
        valid_score_callback=callback,
        epochs=int(args.representation_epochs),
        batch_size=int(args.representation_batch_size),
        seed=int(args.seed),
        early_stopping=True,
        early_stop_patience=int(args.representation_early_stop_patience),
        early_stop_metric="plain_f1",
        balance_kit_loss=True,
        balance_era_within_label=True,
        lam_kit=float(getattr(args, "representation_lam_kit", 0.3)),
        lam_dom=float(getattr(args, "representation_lam_era", 0.3)),
    )
    if scorer is None:
        raise RuntimeError("balanced Kit-DANN requires PyTorch")
    metadata = dict(scorer.metadata)
    best_epoch = int(metadata["best_epoch"])
    history_row = next(
        row for row in metadata["history"] if int(row["epoch"]) == best_epoch
    )
    valid_selection = dict(history_row["valid_selection"])
    selection = {
        "enabled": True,
        "model": "balanced_kit_dann",
        "balance_kit_loss": True,
        "balance_era_within_label": True,
        "use_manifest_capture_dates": True,
        "blend_mode": "positive_lift",
        "selection_split": "fit_dev",
        "selection_metric": "plain_f1",
        "selected_epoch": best_epoch,
        "selected_weight": float(valid_selection["selected_weight"]),
        "weight_grid": weight_grid,
        "fit_dev_selection": valid_selection,
        "evaluation_labels_used": False,
        "final_refit": False,
        "precomputed_row_csw": train_csw_audit,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    scorer.save(out_dir / "kit_dann_representation_selection")
    _write_json(out_dir / "representation_selection.json", selection)
    del rows, train, valid
    gc.collect()
    return scorer, selection


def _refit_balanced_dann(
    dataset: ds.Dataset,
    condition: str,
    manifest: pd.DataFrame,
    session_to_id: dict[str, int],
    onset: pd.DataFrame,
    base_features: list[str],
    selection: dict,
    args: argparse.Namespace,
    out_dir: pathlib.Path,
):
    rows = _prepare_representation_rows(
        dataset, condition, manifest, session_to_id, onset, base_features
    )
    train_y = rows.label.eq("phishing").to_numpy(dtype=int)
    weights, refit_csw_audit = _training_weights(rows, train_y, args)
    scorer = fit_kit_dann_scorer(
        rows,
        base_features,
        kit_label_map=load_kit_label_map(
            str(args.kit_labels_csv), _canonical_path
        ),
        capture_date_of=_capture_date_lookup(manifest),
        canonical_path=_canonical_path,
        sample_weight=weights,
        epochs=int(selection["selected_epoch"]),
        batch_size=int(args.representation_batch_size),
        seed=int(args.seed),
        early_stopping=False,
        balance_kit_loss=True,
        balance_era_within_label=True,
        lam_kit=float(getattr(args, "representation_lam_kit", 0.3)),
        lam_dom=float(getattr(args, "representation_lam_era", 0.3)),
    )
    if scorer is None:
        raise RuntimeError("balanced Kit-DANN requires PyTorch")
    scorer.save(out_dir / "kit_dann_representation")
    final_selection = dict(selection)
    final_selection.update({
        "final_refit": True,
        "final_refit_rows": int(len(rows)),
        "final_refit_sessions": int(rows.session_id.nunique()),
        "final_refit_epochs": int(selection["selected_epoch"]),
        "final_refit_precomputed_row_csw": refit_csw_audit,
    })
    _write_json(out_dir / "representation_selection.json", final_selection)
    del rows
    gc.collect()
    return scorer, final_selection


def _session_table(
    manifest: pd.DataFrame,
    scores: dict[int, float],
    counts: dict[int, int],
    alpha: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    table = manifest.copy()
    table["session_score"] = table.session_id.map(scores).fillna(-1.0).astype(float)
    table["candidate_count"] = table.session_id.map(counts).fillna(0).astype(int)
    benign = table[
        table.partition.eq("calibration") & table.y.eq(0)
    ].session_score.to_numpy(dtype=float)
    if len(benign) == 0:
        raise ValueError("calibration partition contains no benign sessions")
    table["conformal_pvalue"] = _conformal_pvalues(
        table.session_score.to_numpy(dtype=float), benign
    )
    table["y_pred"] = table.conformal_pvalue.le(alpha).astype(np.int8)
    return table, benign


def _earliest_top2_alert(
    group: pd.DataFrame,
    score_column: str,
    available_column: str,
    benign: np.ndarray,
    alpha: float,
    *,
    aggregation: str = "top2",
    single_candidate_super_threshold: float | None = None,
) -> float:
    current: dict[int, float] = {}
    ordered = group.assign(
        _window_order=group.window_name.map(WINDOW_ORDER).fillna(99).astype(int)
    ).sort_values([available_column, "candidate_rank", "_window_order"], kind="stable")
    for row in ordered.itertuples(index=False):
        key = int(row.candidate_key)
        current[key] = max(current.get(key, -1.0), float(getattr(row, score_column)))
        if (
            len(current) == 1
            and single_candidate_super_threshold is not None
            and float(next(iter(current.values())))
            >= float(single_candidate_super_threshold)
        ):
            return float(getattr(row, available_column))
        if aggregation == "max" and current:
            score = float(max(current.values()))
        elif len(current) < 2:
            continue
        else:
            values = np.fromiter(current.values(), dtype=float)
            score = float(np.mean(np.partition(values, -2)[-2:]))
        if _conformal_pvalues(np.asarray([score]), benign)[0] <= alpha:
            return float(getattr(row, available_column))
    return np.nan


def _add_latency_metrics(
    table: pd.DataFrame,
    events: pd.DataFrame,
    onset: pd.DataFrame,
    variant: str,
    benign: np.ndarray,
    alpha: float,
    metrics: dict,
    *,
    dynamic_rank_budget: int = 8,
    aggregation: str = "top2",
    single_candidate_super_threshold: float | None = None,
) -> pd.DataFrame:
    latency_scope = table.y_pred.eq(1) & (
        table.partition.eq("evaluation")
        | (table.partition.eq("fit") & table.fit_role.eq("fit_dev"))
    )
    alerted = set(table[latency_scope].session_id.astype(int))
    event_groups = {
        int(sid): group for sid, group in events[events.session_id.isin(alerted)].groupby(
            "session_id", sort=False
        )
    }
    onset_groups = {int(sid): group for sid, group in onset.groupby("session_id", sort=False)}
    starts = onset.groupby("session_id").pcap_start_epoch.min().astype(float).to_dict()
    alerts: dict[int, float] = {}
    for sid in alerted:
        if variant == "all_onsets":
            alerts[sid] = _earliest_top2_alert(
                event_groups[sid], "all_score", "raw_available_epoch", benign, alpha
                , aggregation=aggregation,
                single_candidate_super_threshold=single_candidate_super_threshold,
            )
        elif variant == "onset_soft_feature":
            alerts[sid] = _earliest_top2_alert(
                event_groups[sid], "soft_score", "soft_available_epoch", benign, alpha
                , aggregation=aggregation,
                single_candidate_super_threshold=single_candidate_super_threshold,
            )
        else:
            _, _, alerts[sid] = _simulate_dynamic_session(
                event_groups[sid], onset_groups[sid], alpha, benign,
                budget=int(dynamic_rank_budget), aggregation=aggregation,
                single_candidate_super_threshold=single_candidate_super_threshold,
            )
    table = table.copy()
    table["alert_epoch"] = table.session_id.map(alerts)
    table["latency_from_capture_start_s"] = (
        table.alert_epoch - table.session_id.map(starts)
    )
    latencies = table[
        table.partition.eq("evaluation") & table.y.eq(1) & table.y_pred.eq(1)
    ].latency_from_capture_start_s.dropna()
    metrics["detected_positive_latency_median_s"] = (
        float(latencies.median()) if len(latencies) else None
    )
    metrics["detected_positive_latency_p90_s"] = (
        float(latencies.quantile(0.90)) if len(latencies) else None
    )
    return table


def _record_fit_dev_latency(table: pd.DataFrame, metrics: dict) -> None:
    latencies = table[
        table.partition.eq("fit") & table.fit_role.eq("fit_dev")
        & table.y.eq(1) & table.y_pred.eq(1)
    ].latency_from_capture_start_s.dropna()
    metrics["detected_positive_latency_median_s"] = (
        float(latencies.median()) if len(latencies) else None
    )
    metrics["detected_positive_latency_p90_s"] = (
        float(latencies.quantile(0.90)) if len(latencies) else None
    )


def _evaluate_condition(
    condition: str,
    event_path: pathlib.Path,
    onset: pd.DataFrame,
    manifest: pd.DataFrame,
    alpha: float,
    out_dir: pathlib.Path,
    variants: tuple[str, ...] = VARIANTS,
    *,
    dynamic_rank_budget: int = 8,
    aggregation: str = "top2",
    single_candidate_target_fpr: float = 0.002,
) -> dict:
    events = pq.read_table(event_path).to_pandas(split_blocks=True, self_destruct=True)
    route_data: dict[str, tuple[dict[int, float], dict[int, int]]] = {}
    if "all_onsets" in variants:
        route_data["all_onsets"] = _top2_session_scores(
            events, "all_score", aggregation=aggregation
        )
    if "onset_soft_feature" in variants:
        route_data["onset_soft_feature"] = _top2_session_scores(
            events, "soft_score", aggregation=aggregation
        )
    if "dynamic_rank8" in variants:
        route_data["dynamic_rank8"] = _dynamic_session_scores(
            events, onset, budget=int(dynamic_rank_budget),
            aggregation=aggregation,
        )
    results: dict[str, object] = {"variants": {}}
    for variant, (scores, counts) in route_data.items():
        table, benign = _session_table(manifest, scores, counts, alpha)
        score_column = {
            "all_onsets": "all_score",
            "onset_soft_feature": "soft_score",
            "dynamic_rank8": "dynamic_score",
        }[variant]
        candidate_max = events.groupby(
            ["session_id", "candidate_key"], sort=False
        )[score_column].max().groupby("session_id", sort=False).max()
        table["max_candidate_window_score"] = (
            table.session_id.map(candidate_max).fillna(-1.0).astype(float)
        )
        single_threshold = None
        single_empirical_fpr = None
        if aggregation == "top2" and float(single_candidate_target_fpr) > 0:
            benign_single = table.loc[
                table.partition.eq("calibration") & table.y.eq(0),
                "max_candidate_window_score",
            ].to_numpy(dtype=float)
            finite = benign_single[np.isfinite(benign_single)]
            if len(finite):
                boundary = float(np.quantile(
                    finite, 1.0 - float(single_candidate_target_fpr),
                    method="higher",
                ))
                single_threshold = float(np.nextafter(boundary, np.inf))
                single_empirical_fpr = float(np.mean(finite >= single_threshold))
                single_override = (
                    table.candidate_count.eq(1)
                    & table.max_candidate_window_score.ge(single_threshold)
                )
                table["single_candidate_override"] = single_override
                table["y_pred"] = (table.y_pred.astype(bool) | single_override).astype(np.int8)
            else:
                table["single_candidate_override"] = False
        else:
            table["single_candidate_override"] = False
        evaluation = table[table.partition.eq("evaluation")].copy()
        calibration = table[table.partition.eq("calibration")].copy()
        development = table[
            table.partition.eq("fit") & table.fit_role.eq("fit_dev")
        ].copy()
        # Fit/dev-only component pilots deliberately contain zero evaluation
        # rows so component selection cannot inspect the held-out test set.
        # Preserve that protocol instead of manufacturing a temporary test
        # split merely to satisfy sklearn's non-empty metric input contract.
        eval_metrics = (
            _metrics(evaluation, alpha)
            if len(evaluation)
            else {
                "not_evaluated": True,
                "reason": "split manifest contains no evaluation rows",
                "rows": 0,
            }
        )
        table = _add_latency_metrics(
            table, events, onset, variant, benign, alpha, eval_metrics,
            dynamic_rank_budget=int(dynamic_rank_budget),
            aggregation=aggregation,
            single_candidate_super_threshold=single_threshold,
        )
        variant_dir = out_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        table[table.partition.eq("evaluation")].to_csv(
            variant_dir / "evaluation_predictions.csv", index=False
        )
        table[table.partition.eq("calibration")].to_csv(
            variant_dir / "calibration_predictions.csv", index=False
        )
        table[
            table.partition.eq("fit") & table.fit_role.eq("fit_dev")
        ].to_csv(variant_dir / "fit_dev_predictions.csv", index=False)
        fit_dev_metrics = _metrics(development, alpha)
        _record_fit_dev_latency(table, fit_dev_metrics)
        result = {
            "fit_dev": fit_dev_metrics,
            "calibration": _metrics(calibration, alpha),
            "evaluation": eval_metrics,
            "benign_calibration_sessions": int(len(benign)),
            "aggregation": aggregation,
            "dynamic_rank_budget": int(dynamic_rank_budget),
            "single_candidate_target_fpr": float(single_candidate_target_fpr),
            "single_candidate_super_threshold": single_threshold,
            "single_candidate_empirical_calibration_fpr": single_empirical_fpr,
            "evaluation_single_candidate_overrides": int(
                table.loc[table.partition.eq("evaluation"), "single_candidate_override"].sum()
            ),
        }
        _write_json(variant_dir / "metrics.json", result)
        results["variants"][variant] = result
    _write_json(out_dir / "summary.json", results)
    return results


def _run_representation(
    rep: str,
    dataset: ds.Dataset,
    feature_columns: list[str],
    manifest: pd.DataFrame,
    session_to_id: dict[str, int],
    args: argparse.Namespace,
    out_dir: pathlib.Path,
) -> dict:
    conditions = tuple(
        getattr(args, "conditions", None) or ("adaptive_only", "five_windows")
    )
    variants = tuple(getattr(args, "variants", None) or VARIANTS)
    print(f"[{rep}] load adaptive candidate rows and fit shared onset ranker", flush=True)
    adaptive_columns = [
        "zip_path", "candidate_rank", "candidate_hit_within_3s",
        "window_available_epoch", "pcap_start_epoch",
    ] + feature_columns
    adaptive = _scan_to_frame(
        dataset, adaptive_columns, WINDOW_CONDITIONS["adaptive_only"],
        set(manifest.zip_path),
    )
    adaptive = _attach_ids(adaptive, session_to_id)
    if adaptive.candidate_key.duplicated().any():
        raise RuntimeError("adaptive candidate keys are not unique")
    fitdev_refit = bool(getattr(args, "fitdev_refit", False))
    balanced_dann = bool(getattr(args, "balanced_kit_dann", False))
    dynamic_rank_budget = int(getattr(args, "dynamic_rank_budget", 8))
    headline_aggregation = str(getattr(args, "headline_aggregation", "top2"))
    single_candidate_target_fpr = float(
        getattr(args, "single_candidate_target_fpr", 0.002)
    )
    if fitdev_refit:
        selection_onset, selection_onset_protocol = _fit_onset_ranker(
            adaptive, manifest, feature_columns, int(args.onset_oof_folds),
            int(args.seed), out_dir / "shared_onset_ranker_selection",
            include_fit_dev=False,
            dynamic_rank_budget=dynamic_rank_budget,
        )
        onset, onset_protocol = _fit_onset_ranker(
            adaptive, manifest, feature_columns, int(args.onset_oof_folds),
            int(args.seed), out_dir / "shared_onset_ranker",
            include_fit_dev=True,
            dynamic_rank_budget=dynamic_rank_budget,
        )
    else:
        onset, onset_protocol = _fit_onset_ranker(
            adaptive, manifest, feature_columns, int(args.onset_oof_folds),
            int(args.seed), out_dir / "shared_onset_ranker",
            include_fit_dev=False,
            dynamic_rank_budget=dynamic_rank_budget,
        )
        selection_onset = onset
        selection_onset_protocol = onset_protocol
    del adaptive
    gc.collect()

    condition_results: dict[str, object] = {}
    training_stats: dict[str, object] = {}
    for condition in conditions:
        print(f"[{rep}/{condition}] fit three classifier routes", flush=True)
        condition_dir = out_dir / condition
        representation_scorer = None
        representation_weight = 0.0
        representation_selection = None
        if balanced_dann:
            selection_models, selection_fit_stats = _fit_classifier_models(
                dataset, condition, manifest, session_to_id, selection_onset,
                feature_columns, int(args.seed),
                condition_dir / "models_selection", variants,
                include_fit_dev=False,
                args=args,
            )
            if "dynamic_rank8" not in selection_models:
                raise ValueError("balanced Kit-DANN requires dynamic_rank8 variant")
            selection_representation_scorer, representation_selection = _fit_balanced_dann_selection(
                dataset, condition, manifest, session_to_id, selection_onset,
                feature_columns, selection_models["dynamic_rank8"], args,
                condition_dir / "models_selection" / "dynamic_rank8",
            )
        else:
            selection_models = None
            selection_fit_stats = None

        models, fit_stats = _fit_classifier_models(
            dataset, condition, manifest, session_to_id, onset,
            feature_columns, int(args.seed), condition_dir / "models", variants,
            include_fit_dev=fitdev_refit,
            args=args,
        )
        if balanced_dann:
            if fitdev_refit:
                representation_scorer, representation_selection = _refit_balanced_dann(
                    dataset, condition, manifest, session_to_id, onset,
                    feature_columns, representation_selection, args,
                    condition_dir / "models" / "dynamic_rank8",
                )
            else:
                representation_scorer = selection_representation_scorer
                representation_scorer.save(
                    condition_dir / "models" / "dynamic_rank8"
                    / "kit_dann_representation"
                )
                _write_json(
                    condition_dir / "models" / "dynamic_rank8"
                    / "representation_selection.json",
                    representation_selection,
                )
            representation_weight = float(
                representation_selection["selected_weight"]
            )
        training_stats[condition] = {
            "selection_fit": selection_fit_stats,
            "final_fit": fit_stats,
            "balanced_kit_dann": representation_selection,
        }
        if selection_models is not None:
            del selection_models
            gc.collect()
        event_path = condition_dir / "scored_events.parquet"
        score_stats = _write_scored_events(
            dataset, condition, session_to_id, onset, feature_columns,
            models, event_path, int(args.score_batch_size), variants,
            representation_scorer=representation_scorer,
            representation_weight=representation_weight,
            representation_blend_mode="positive_lift",
        )
        training_stats[condition]["scoring"] = score_stats
        del models
        gc.collect()
        print(f"[{rep}/{condition}] aggregate distinct candidates and calibrate", flush=True)
        condition_results[condition] = _evaluate_condition(
            condition, event_path, onset, manifest, float(args.alpha),
            condition_dir, variants,
            dynamic_rank_budget=dynamic_rank_budget,
            aggregation=headline_aggregation,
            single_candidate_target_fpr=single_candidate_target_fpr,
        )
        gc.collect()

    paired: dict[str, dict[str, float]] = {}
    if {"adaptive_only", "five_windows"}.issubset(condition_results):
        for variant in variants:
            adaptive_metrics = condition_results["adaptive_only"]["variants"][variant]["evaluation"]
            five_metrics = condition_results["five_windows"]["variants"][variant]["evaluation"]
            paired[variant] = {
                metric: float(five_metrics[metric] - adaptive_metrics[metric])
                for metric in ("precision", "recall", "f1", "fpr", "roc_auc", "pr_auc")
            }
    result = {
        "onset_protocol": onset_protocol,
        "selection_onset_protocol": selection_onset_protocol,
        "training_stats": training_stats,
        "conditions": condition_results,
        "five_minus_adaptive_evaluation": paired,
    }
    _write_json(out_dir / "representation_summary.json", result)
    return result


def run(args: argparse.Namespace) -> None:
    started = time.time()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = json.loads(pathlib.Path(args.column_spec).read_text(encoding="utf-8"))
    manifest, session_to_id = _load_manifest(pathlib.Path(args.split_manifest))
    csw_contract = _validate_causal_csw_contract(args, manifest)
    dataset = ds.dataset([str(pathlib.Path(path)) for path in args.features_parquet], format="parquet")
    print("[audit] verify shared candidate pool and five-window completeness", flush=True)
    audit = _mother_table_audit(
        dataset, manifest, session_to_id,
        max_delay_s=float(args.max_decision_delay_s),
        merge_s=float(args.nms_merge_s),
    )
    _write_json(out_dir / "mother_table_audit.json", audit)
    representations = {}
    requested_representations = tuple(
        getattr(args, "representations", None) or ("425", "228")
    )
    for rep in requested_representations:
        representations[rep] = _run_representation(
            rep, dataset, list(spec["representations"][rep]), manifest,
            session_to_id, args, out_dir / rep,
        )
    conditions = tuple(
        getattr(args, "conditions", None) or ("adaptive_only", "five_windows")
    )
    variants = tuple(getattr(args, "variants", None) or VARIANTS)
    raw_delays = getattr(args, "candidate_decision_delays_s", None)
    decision_delays = (
        [float(value) for value in raw_delays.replace(",", " ").split()]
        if raw_delays else [0.75, 1.5, 2.5, 3.5]
    )
    payload = {
        "protocol": {
            "split": "random_zip_legacy_compatible",
            "seed": int(args.seed),
            "scan_step_s": 0.5,
            "candidate_decision_delays_s": decision_delays,
            "nms_merge_s": float(args.nms_merge_s),
            "candidate_pool": "same all causally finalized local onsets",
            "conditions": {
                name: WINDOW_CONDITIONS[name] for name in conditions
            },
            "onset_ranker": "shared between window conditions; adaptive row only",
            "routes": list(variants),
            "aggregation": str(getattr(args, "headline_aggregation", "top2")),
            "soft_feature_availability": "max(window availability, adaptive onset-score availability)",
            "dynamic_rank8": (
                f"prefix-causal top{int(getattr(args, 'dynamic_rank_budget', 8))}-so-far; "
                "persistent maximum over prefixes"
            ),
            "dynamic_rank_budget": int(getattr(args, "dynamic_rank_budget", 8)),
            "single_candidate_override": bool(
                str(getattr(args, "headline_aggregation", "top2")) == "top2"
                and float(getattr(args, "single_candidate_target_fpr", 0.002)) > 0
            ),
            "single_candidate_target_fpr": float(
                getattr(args, "single_candidate_target_fpr", 0.002)
            ),
            "calibration": "same split-conformal benign session maximum",
            "alpha": float(args.alpha),
            "classifier_missing_values": (
                "fit-only median imputation and standardization"
                if str(getattr(args, "classifier_model", "lightgbm")) == "logistic"
                else "learner-native missing-value handling"
            ),
            "classifier_model": str(getattr(args, "classifier_model", "lightgbm")),
            "classifier_preprocessing": (
                "fit_only_median_imputation_and_standardization"
                if str(getattr(args, "classifier_model", "lightgbm")) == "logistic"
                else "native_missing_value_handling"
            ),
            "candidate_onset_ranker_model": "lightgbm_frozen_across_classifier_arms",
            "classifier_sample_weighting": "session_normalized_balanced",
            "precomputed_row_csw": csw_contract,
            "balanced_kit_dann": bool(getattr(args, "balanced_kit_dann", False)),
            "representation_blend_mode": (
                "positive_lift"
                if getattr(args, "balanced_kit_dann", False) else "none"
            ),
            "representation_weight_grid": (
                str(getattr(args, "representation_weight_grid", ""))
                if getattr(args, "balanced_kit_dann", False) else None
            ),
            "fitdev_refit": bool(getattr(args, "fitdev_refit", False)),
            "evaluation_labels_used_for_model_selection": False,
        },
        "mother_table_audit": audit,
        "representations": representations,
        "elapsed_s": time.time() - started,
    }
    _write_json(out_dir / "experiment_summary.json", payload)
    print(json.dumps(payload, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-parquet", action="append", required=True)
    parser.add_argument("--column-spec", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.03)
    parser.add_argument("--onset-oof-folds", type=int, default=5)
    parser.add_argument("--nms-merge-s", type=float, default=2.0)
    parser.add_argument("--max-decision-delay-s", type=float, default=3.5)
    parser.add_argument("--score-batch-size", type=int, default=8192)
    parser.add_argument(
        "--classifier-model",
        choices=CLASSIFIER_MODELS,
        default="lightgbm",
        help=(
            "Final phishing classifier only. Candidate onset ranking remains "
            "the frozen LightGBM route for classifier-backbone sensitivity."
        ),
    )
    parser.add_argument(
        "--representation", dest="representations", action="append",
        choices=("425", "228"),
        help="Representation to run; repeat as needed. Defaults to both.",
    )
    parser.add_argument("--balanced-kit-dann", action="store_true")
    parser.add_argument("--fitdev-refit", action="store_true")
    parser.add_argument(
        "--precomputed-fit-row-weights",
        help=(
            "Audited packet-causal CSW row weights keyed by "
            "zip_path/window_name/candidate_rank."
        ),
    )
    parser.add_argument(
        "--adaptation-summary",
        help="Audit emitted with --precomputed-fit-row-weights.",
    )
    parser.add_argument("--kit-labels-csv")
    parser.add_argument(
        "--representation-weight-grid",
        default="0,0.1,0.2,0.3,0.4,0.5,0.7,1.0",
    )
    parser.add_argument("--representation-epochs", type=int, default=25)
    parser.add_argument("--representation-batch-size", type=int, default=2048)
    parser.add_argument("--representation-early-stop-patience", type=int, default=5)
    parser.add_argument("--representation-lam-kit", type=float, default=0.3)
    parser.add_argument("--representation-lam-era", type=float, default=0.3)
    parser.add_argument("--dynamic-rank-budget", type=int, default=8)
    parser.add_argument(
        "--headline-aggregation", choices=("top2", "max"), default="top2"
    )
    parser.add_argument("--single-candidate-target-fpr", type=float, default=0.002)
    parser.add_argument(
        "--condition", dest="conditions", action="append",
        choices=tuple(WINDOW_CONDITIONS),
        help="Condition to evaluate; repeat as needed. Defaults to both.",
    )
    parser.add_argument(
        "--variant", dest="variants", action="append", choices=VARIANTS,
        help="Classifier/aggregation route to evaluate; repeat as needed.",
    )
    parser.add_argument(
        "--candidate-decision-delays-s", default=None,
        help="Comma-separated delays recorded in the formal protocol.",
    )
    args = parser.parse_args()
    if args.balanced_kit_dann and not args.kit_labels_csv:
        parser.error("--balanced-kit-dann requires --kit-labels-csv")
    if args.dynamic_rank_budget < 1:
        parser.error("--dynamic-rank-budget must be at least 1")
    if not 0 <= args.single_candidate_target_fpr < 1:
        parser.error("--single-candidate-target-fpr must be in [0, 1)")
    # ``fitdev_refit`` also applies to the LightGBM-only ablation: no
    # hyperparameter is selected from evaluation, the fit-only model is merely
    # replaced by an outer-fit (fit_train + fit_dev) model under the already
    # frozen protocol.  The balanced-DANN route additionally freezes its epoch
    # and positive-lift weight on fit_dev before the same outer-fit refit.
    run(args)


if __name__ == "__main__":
    main()

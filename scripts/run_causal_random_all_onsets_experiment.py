#!/usr/bin/env python3
"""Seed-42 random-ZIP Layer-1 experiment under strict causal replay.

Every candidate route consumes causally finalized onsets in availability-time
order.  The dynamic-rank8 comparator maintains an onset-ranker top8-so-far and
persists an alert once any causal prefix crosses its frozen threshold; it never
uses the final complete-session rank retrospectively.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from dataclasses import asdict

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from run_layer1_phishing_detection import feature_columns_for_preset  # noqa: E402


META_COLUMNS = {
    "zip_path", "label", "window_name", "window_available_epoch",
    "candidate_start_epoch", "candidate_emit_epoch", "candidate_rank",
    "candidate_source", "candidate_hit_within_3s", "pcap_start_epoch",
}
REPRESENTATIONS = {
    "425": "kit_l1_core_transport_invariant_stackclean",
    "228": "kit_l1_core_shape_only",
}
VARIANTS = (
    "all_onsets",
    "all_onsets_plus_onset_score",
    "onset_gate_then_phish",
    "dynamic_rank8",
)


def _write_json(path: pathlib.Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _feature_lists(schema_parquet: pathlib.Path) -> dict[str, list[str]]:
    schema = pq.read_schema(schema_parquet)
    frame_data: dict[str, pd.Series] = {}
    for field in schema:
        dtype = "float64" if (
            str(field.type).startswith(("int", "uint", "float", "double", "decimal"))
        ) else "object"
        frame_data[field.name] = pd.Series(dtype=dtype)
    template = pd.DataFrame(frame_data)
    return {
        rep: feature_columns_for_preset(template, preset)
        for rep, preset in REPRESENTATIONS.items()
    }


def _load_features(csv_path: pathlib.Path, schema_parquet: pathlib.Path) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    lists = _feature_lists(schema_parquet)
    header = set(pd.read_csv(csv_path, nrows=0).columns)
    missing_meta = sorted(META_COLUMNS - header)
    if missing_meta:
        raise ValueError(f"causal feature table misses metadata: {missing_meta}")
    usecols = sorted((META_COLUMNS | set(lists["425"]) | set(lists["228"])) & header)
    frame = pd.read_csv(csv_path, usecols=usecols, low_memory=False)
    for column in sorted((set(lists["425"]) | set(lists["228"])) & set(frame.columns)):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in (
        "window_available_epoch", "candidate_start_epoch", "candidate_emit_epoch",
        "candidate_rank", "candidate_hit_within_3s", "pcap_start_epoch",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["zip_path"] = frame.zip_path.astype(str)
    frame["label"] = frame.label.astype(str)
    return frame, lists


def _load_manifest(path: pathlib.Path) -> pd.DataFrame:
    manifest = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"capture_id", "sample_path", "label", "partition", "fit_role", "source", "fine_source"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"split manifest misses columns: {missing}")
    if manifest.capture_id.duplicated().any() or manifest.sample_path.duplicated().any():
        raise ValueError("split manifest must contain unique capture_id and sample_path")
    manifest = manifest.rename(columns={"sample_path": "zip_path"}).copy()
    manifest["zip_path"] = manifest.zip_path.astype(str)
    manifest["y"] = manifest.label.str.casefold().eq("phishing").astype(int)
    return manifest


def _causal_audit(frame: pd.DataFrame, manifest: pd.DataFrame, merge_s: float, max_delay_s: float) -> dict:
    adaptive_only = bool(frame.window_name.astype(str).eq("adaptive").all())
    stream_only = bool(frame.candidate_source.astype(str).str.startswith("traffic_heuristic_stream").all())
    unique = not frame.duplicated(["zip_path", "candidate_rank"]).any()
    finite = frame.dropna(subset=["candidate_start_epoch", "candidate_emit_epoch", "window_available_epoch"])
    nms_floor = finite.candidate_start_epoch + float(merge_s) + float(max_delay_s)
    emit_after_nms = bool((finite.candidate_emit_epoch + 1e-7 >= nms_floor).all())
    window_after_emit = bool((finite.window_available_epoch + 1e-7 >= finite.candidate_emit_epoch).all())
    ranks_chronological = True
    for _, group in frame.groupby("zip_path", sort=False):
        ordered = group.sort_values("candidate_rank")
        if not np.all(np.diff(ordered.candidate_start_epoch.to_numpy(dtype=float)) >= -1e-7):
            ranks_chronological = False
            break
    feature_sessions = set(frame.zip_path)
    manifest_sessions = set(manifest.zip_path)
    audit = {
        "adaptive_only": adaptive_only,
        "traffic_stream_candidates_only": stream_only,
        "one_row_per_candidate": bool(unique),
        "candidate_emit_after_local_nms_finalization": emit_after_nms,
        "window_available_after_candidate_emit": window_after_emit,
        "candidate_rank_is_chronological_only": ranks_chronological,
        "feature_sessions": int(len(feature_sessions)),
        "manifest_sessions": int(len(manifest_sessions)),
        "sessions_without_emitted_onset": int(len(manifest_sessions - feature_sessions)),
        "orphan_feature_sessions": int(len(feature_sessions - manifest_sessions)),
        "rank_used_for_filtering": False,
        "complete_session_future_used_for_selection": False,
    }
    audit["passed"] = bool(all([
        adaptive_only, stream_only, unique, emit_after_nms,
        window_after_emit, ranks_chronological,
        audit["orphan_feature_sessions"] == 0,
    ]))
    if not audit["passed"]:
        raise RuntimeError(f"strict causal audit failed: {audit}")
    return audit


def _session_folds(manifest: pd.DataFrame, sessions: set[str], folds: int, seed: int) -> dict[str, int]:
    table = manifest[manifest.zip_path.isin(sessions)][["zip_path", "y"]].copy()
    if table.empty:
        raise ValueError("no fit sessions available for OOF folds")
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    mapping: dict[str, int] = {}
    for fold, (_, valid) in enumerate(splitter.split(table.zip_path, table.y)):
        mapping.update({str(value): int(fold) for value in table.iloc[valid].zip_path})
    return mapping


def _session_normalized_weights(frame: pd.DataFrame, target: np.ndarray | None = None) -> np.ndarray:
    counts = frame.groupby("zip_path").zip_path.transform("size").to_numpy(dtype=float)
    weights = 1.0 / np.maximum(1.0, counts)
    if target is not None:
        target = np.asarray(target, dtype=int)
        for value in (0, 1):
            mask = target == value
            total = float(weights[mask].sum())
            if total > 0:
                weights[mask] *= 0.5 / total
    weights *= len(weights) / max(float(weights.sum()), 1e-12)
    return weights


def _make_model(seed: int, *, ranker: bool) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        n_estimators=420 if ranker else 650,
        num_leaves=31 if ranker else 63,
        learning_rate=0.035,
        feature_fraction=0.88,
        bagging_fraction=0.90,
        bagging_freq=5,
        min_child_samples=15 if ranker else 12,
        reg_lambda=1.0,
        class_weight=None if ranker else "balanced",
        random_state=int(seed),
        n_jobs=int(os.environ.get("LAYER1_LGBM_N_JOBS", "-1")),
        verbose=-1,
        deterministic=True,
        force_col_wise=True,
    )


def _fit_imputed_model(
    train: pd.DataFrame,
    features: list[str],
    target: np.ndarray,
    weights: np.ndarray,
    seed: int,
    *,
    ranker: bool,
) -> tuple[SimpleImputer, LGBMClassifier]:
    imputer = SimpleImputer(strategy="median")
    matrix = imputer.fit_transform(train[features])
    model = _make_model(seed, ranker=ranker)
    model.fit(matrix, np.asarray(target, dtype=int), sample_weight=weights)
    return imputer, model


def _predict(imputer: SimpleImputer, model: LGBMClassifier, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    if frame.empty:
        return np.empty(0, dtype=float)
    return np.asarray(model.predict_proba(imputer.transform(frame[features]))[:, 1], dtype=float)


def _attach_onset_scores(
    frame: pd.DataFrame,
    manifest: pd.DataFrame,
    features: list[str],
    fit_sessions: set[str],
    folds: int,
    seed: int,
    out_dir: pathlib.Path,
) -> tuple[pd.DataFrame, dict]:
    work = frame.copy()
    work["onset_alignment_score"] = np.nan
    fold_by_session = _session_folds(manifest, fit_sessions, folds, seed)
    positive_fit_sessions = set(
        manifest[
            manifest.zip_path.isin(fit_sessions) & manifest.y.eq(1)
        ].zip_path
    )
    fold_stats = []
    for fold in range(folds):
        valid_sessions = {session for session, value in fold_by_session.items() if value == fold}
        train_rows = work[
            work.zip_path.isin(positive_fit_sessions - valid_sessions)
        ].copy()
        target = train_rows.candidate_hit_within_3s.fillna(0).astype(int).to_numpy()
        if len(np.unique(target)) < 2:
            raise ValueError(f"onset OOF fold {fold} lacks both alignment labels")
        weights = _session_normalized_weights(train_rows, target)
        imputer, model = _fit_imputed_model(
            train_rows, features, target, weights, seed + fold, ranker=True
        )
        valid_mask = work.zip_path.isin(valid_sessions)
        work.loc[valid_mask, "onset_alignment_score"] = _predict(
            imputer, model, work.loc[valid_mask], features
        )
        fold_stats.append({
            "fold": fold,
            "ranker_train_positive_sessions": len(positive_fit_sessions - valid_sessions),
            "scored_fit_sessions": len(valid_sessions),
            "train_rows": len(train_rows),
            "alignment_positive_rows": int(target.sum()),
        })

    final_train = work[work.zip_path.isin(positive_fit_sessions)].copy()
    final_target = final_train.candidate_hit_within_3s.fillna(0).astype(int).to_numpy()
    final_weights = _session_normalized_weights(final_train, final_target)
    final_imputer, final_model = _fit_imputed_model(
        final_train, features, final_target, final_weights, seed + 101, ranker=True
    )
    non_fit_mask = ~work.zip_path.isin(fit_sessions)
    work.loc[non_fit_mask, "onset_alignment_score"] = _predict(
        final_imputer, final_model, work.loc[non_fit_mask], features
    )
    if work[work.zip_path.isin(fit_sessions)].onset_alignment_score.isna().any():
        raise RuntimeError("OOF onset scores are missing for classifier fit rows")
    if work[non_fit_mask].onset_alignment_score.isna().any():
        raise RuntimeError("frozen onset scores are missing for non-fit rows")
    joblib.dump(
        {"imputer": final_imputer, "model": final_model, "features": features},
        out_dir / "onset_alignment_ranker.joblib",
    )
    return work, {
        "ranker_target": "candidate_hit_within_3s",
        "ranker_fit_population": "phishing fit_train sessions only",
        "classifier_fit_score_semantics": f"{folds}-fold session-level OOF",
        "non_fit_score_semantics": "ranker frozen on all phishing fit_train sessions",
        "folds": fold_stats,
        "final_train_rows": int(len(final_train)),
        "final_alignment_positive_rows": int(final_target.sum()),
    }


def _bag_scores(frame: pd.DataFrame, score_column: str, manifest: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby("zip_path", sort=False)[score_column]
    bags = grouped.apply(
        lambda values: float(values.nlargest(2).mean()) if len(values) >= 2 else -1.0
    ).rename("session_score").reset_index()
    counts = grouped.size().rename("candidate_count").reset_index()
    bags = bags.merge(counts, on="zip_path", how="left")
    out = manifest.merge(bags, on="zip_path", how="left", validate="one_to_one")
    out["candidate_count"] = out.candidate_count.fillna(0).astype(int)
    out["session_score"] = out.session_score.fillna(-1.0).astype(float)
    return out


def _dynamic_rank_admitted_mask(
    frame: pd.DataFrame,
    rank_score_column: str = "onset_alignment_score",
    budget: int = 8,
) -> pd.Series:
    """Mark candidates that enter a causal top-k at their arrival time.

    An admitted candidate may be evicted later, but it was genuinely available
    to the downstream scorer before that eviction and therefore remains part of
    the deployment-faithful classifier-fit population.
    """
    admitted = pd.Series(False, index=frame.index, dtype=bool)
    for _, group in frame.groupby("zip_path", sort=False):
        active: list[tuple[float, int, object]] = []
        ordered = group.sort_values(
            ["window_available_epoch", "candidate_rank"], kind="stable"
        )
        for arrival, (index, row) in enumerate(ordered.iterrows()):
            rank_score = float(row[rank_score_column])
            active.append((rank_score, arrival, index))
            active = sorted(active, key=lambda item: (-item[0], item[1]))[:budget]
            if any(item[2] == index for item in active):
                admitted.at[index] = True
    return admitted


def _dynamic_rank_bags(
    frame: pd.DataFrame,
    score_column: str,
    manifest: pd.DataFrame,
    rank_score_column: str = "onset_alignment_score",
    budget: int = 8,
) -> pd.DataFrame:
    """Compute the maximum score reached by any causal top-k-so-far prefix."""
    records: list[dict[str, object]] = []
    for session, group in frame.groupby("zip_path", sort=False):
        active: list[tuple[float, int, float]] = []
        best = -1.0
        max_active = 0
        ordered = group.sort_values(
            ["window_available_epoch", "candidate_rank"], kind="stable"
        )
        for arrival, row in enumerate(ordered.itertuples(index=False)):
            active.append((
                float(getattr(row, rank_score_column)),
                arrival,
                float(getattr(row, score_column)),
            ))
            active = sorted(active, key=lambda item: (-item[0], item[1]))[:budget]
            max_active = max(max_active, len(active))
            if len(active) >= 2:
                current = float(np.mean(sorted(
                    (item[2] for item in active), reverse=True
                )[:2]))
                best = max(best, current)
        records.append({
            "zip_path": session,
            "session_score": best,
            "candidate_count": max_active,
            "observed_candidate_count": int(len(ordered)),
        })
    bags = pd.DataFrame(records)
    out = manifest.merge(bags, on="zip_path", how="left", validate="one_to_one")
    out["candidate_count"] = out.candidate_count.fillna(0).astype(int)
    out["observed_candidate_count"] = out.observed_candidate_count.fillna(0).astype(int)
    out["session_score"] = out.session_score.fillna(-1.0).astype(float)
    return out


def _conformal_pvalues(scores: np.ndarray, benign_calibration: np.ndarray) -> np.ndarray:
    calibration = np.asarray(benign_calibration, dtype=float)
    values = np.asarray(scores, dtype=float)
    return np.asarray([
        (1.0 + float(np.sum(calibration >= value))) / (len(calibration) + 1.0)
        for value in values
    ], dtype=float)


def _metrics(table: pd.DataFrame, alpha: float) -> dict:
    y = table.y.to_numpy(dtype=int)
    pred = table.y_pred.to_numpy(dtype=int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0
    )
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tp = int(((pred == 1) & (y == 1)).sum())
    scores = table.session_score.to_numpy(dtype=float)
    return {
        "n": int(len(table)), "positives": int(y.sum()), "benign": int((y == 0).sum()),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "fpr": float(fp / max(1, fp + tn)),
        "specificity": float(tn / max(1, fp + tn)),
        "roc_auc": float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else None,
        "pr_auc": float(average_precision_score(y, scores)) if len(np.unique(y)) == 2 else None,
        "conformal_alpha": float(alpha),
        "median_candidates": float(table.candidate_count.median()),
        "p90_candidates": float(table.candidate_count.quantile(0.90)),
        "zero_or_one_candidate_sessions": int(table.candidate_count.lt(2).sum()),
    }


def _earliest_alerts(
    rows: pd.DataFrame,
    session_table: pd.DataFrame,
    score_column: str,
    benign_calibration: np.ndarray,
    alpha: float,
) -> pd.DataFrame:
    alerted = set(session_table[session_table.y_pred.eq(1)].zip_path)
    records = []
    for session, group in rows[rows.zip_path.isin(alerted)].groupby("zip_path", sort=False):
        top: list[float] = []
        alert_epoch = np.nan
        for row in group.sort_values("window_available_epoch").itertuples():
            top.append(float(getattr(row, score_column)))
            if len(top) < 2:
                continue
            score = float(np.mean(sorted(top, reverse=True)[:2]))
            pvalue = float(_conformal_pvalues(np.asarray([score]), benign_calibration)[0])
            if pvalue <= alpha:
                alert_epoch = float(row.window_available_epoch)
                break
        start = float(pd.to_numeric(group.pcap_start_epoch, errors="coerce").dropna().iloc[0])
        records.append({
            "zip_path": session,
            "alert_epoch": alert_epoch,
            "latency_from_capture_start_s": alert_epoch - start if np.isfinite(alert_epoch) else np.nan,
        })
    return pd.DataFrame(records)


def _dynamic_rank_earliest_alerts(
    rows: pd.DataFrame,
    session_table: pd.DataFrame,
    score_column: str,
    benign_calibration: np.ndarray,
    alpha: float,
    rank_score_column: str = "onset_alignment_score",
    budget: int = 8,
) -> pd.DataFrame:
    alerted = set(session_table[session_table.y_pred.eq(1)].zip_path)
    records: list[dict[str, object]] = []
    for session, group in rows[rows.zip_path.isin(alerted)].groupby("zip_path", sort=False):
        active: list[tuple[float, int, float]] = []
        alert_epoch = np.nan
        ordered = group.sort_values(
            ["window_available_epoch", "candidate_rank"], kind="stable"
        )
        for arrival, row in enumerate(ordered.itertuples(index=False)):
            active.append((
                float(getattr(row, rank_score_column)),
                arrival,
                float(getattr(row, score_column)),
            ))
            active = sorted(active, key=lambda item: (-item[0], item[1]))[:budget]
            if len(active) < 2:
                continue
            current = float(np.mean(sorted(
                (item[2] for item in active), reverse=True
            )[:2]))
            pvalue = float(_conformal_pvalues(
                np.asarray([current]), benign_calibration
            )[0])
            if pvalue <= alpha:
                alert_epoch = float(row.window_available_epoch)
                break
        start = float(pd.to_numeric(
            group.pcap_start_epoch, errors="coerce"
        ).dropna().iloc[0])
        records.append({
            "zip_path": session,
            "alert_epoch": alert_epoch,
            "latency_from_capture_start_s": (
                alert_epoch - start if np.isfinite(alert_epoch) else np.nan
            ),
        })
    return pd.DataFrame(records)


def _run_representation(
    rep: str,
    frame: pd.DataFrame,
    manifest: pd.DataFrame,
    base_features: list[str],
    out_dir: pathlib.Path,
    alpha: float,
    folds: int,
    seed: int,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    fit_sessions = set(
        manifest[manifest.partition.eq("fit") & manifest.fit_role.eq("fit_train")].zip_path
    )
    scored, onset_protocol = _attach_onset_scores(
        frame, manifest, base_features, fit_sessions, folds, seed, out_dir
    )
    fit_dev_sessions = set(
        manifest[manifest.partition.eq("fit") & manifest.fit_role.eq("fit_dev")].zip_path
    )
    aligned_dev = scored[
        scored.zip_path.isin(fit_dev_sessions)
        & scored.label.str.casefold().eq("phishing")
        & scored.candidate_hit_within_3s.fillna(0).astype(int).eq(1)
    ].onset_alignment_score.to_numpy(dtype=float)
    if len(aligned_dev) == 0:
        raise ValueError("fit_dev has no oracle-aligned positive onset for gate calibration")
    try:
        onset_gate_threshold = float(np.quantile(aligned_dev, 0.05, method="lower"))
    except TypeError:  # NumPy < 1.22
        onset_gate_threshold = float(np.quantile(aligned_dev, 0.05, interpolation="lower"))
    gate_dev = scored[scored.zip_path.isin(fit_dev_sessions)].copy()
    gate_target = gate_dev.candidate_hit_within_3s.fillna(0).astype(int).to_numpy()
    gate_pred = gate_dev.onset_alignment_score.ge(onset_gate_threshold).astype(int).to_numpy()
    gate_tp = int(((gate_target == 1) & (gate_pred == 1)).sum())
    gate_fn = int(((gate_target == 1) & (gate_pred == 0)).sum())
    gate_tn = int(((gate_target == 0) & (gate_pred == 0)).sum())
    gate_fp = int(((gate_target == 0) & (gate_pred == 1)).sum())
    gate_protocol = {
        "threshold_source": "fit_dev aligned-positive 5th percentile",
        "target_alignment_recall": 0.95,
        "threshold": onset_gate_threshold,
        "fit_dev_alignment_recall": gate_tp / max(1, gate_tp + gate_fn),
        "fit_dev_alignment_specificity": gate_tn / max(1, gate_tn + gate_fp),
        "fit_dev_candidate_retention": float(gate_pred.mean()) if len(gate_pred) else 0.0,
        "fit_dev_tp": gate_tp, "fit_dev_fp": gate_fp,
        "fit_dev_tn": gate_tn, "fit_dev_fn": gate_fn,
    }
    fit_rows = scored[scored.zip_path.isin(fit_sessions)].copy()
    results: dict[str, object] = {
        "onset_protocol": onset_protocol,
        "onset_gate_protocol": gate_protocol,
        "variants": {},
    }
    for variant in VARIANTS:
        gated = variant == "onset_gate_then_phish"
        dynamic_rank = variant == "dynamic_rank8"
        # Train the phishing classifier on the complete causal candidate
        # population.  The frozen onset gate is a deployment-time routing
        # decision: candidates rejected by it are never sent to the phishing
        # scorer.  Filtering the phishing-fit rows here can remove every
        # benign row when the gate is clean, leaving the downstream classifier
        # unidentifiable and coupling its training distribution to a hard
        # development threshold.
        if dynamic_rank:
            admitted_mask = _dynamic_rank_admitted_mask(
                fit_rows, rank_score_column="onset_alignment_score", budget=8
            )
            variant_fit = fit_rows[admitted_mask].copy()
        else:
            variant_fit = fit_rows
        target = variant_fit.label.str.casefold().eq("phishing").astype(int).to_numpy()
        if len(np.unique(target)) < 2:
            raise ValueError(f"{variant} fit rows lack both phishing classes")
        weights = _session_normalized_weights(variant_fit)
        features = list(base_features)
        if variant.endswith("plus_onset_score"):
            features.append("onset_alignment_score")
        imputer, model = _fit_imputed_model(
            variant_fit, features, target, weights, seed, ranker=False
        )
        score_column = f"phish_score__{variant}"
        scored[score_column] = _predict(imputer, model, scored, features)
        decision_rows = (
            scored[scored.onset_alignment_score.ge(onset_gate_threshold)].copy()
            if gated else scored
        )
        bags = (
            _dynamic_rank_bags(
                decision_rows,
                score_column,
                manifest,
                rank_score_column="onset_alignment_score",
                budget=8,
            )
            if dynamic_rank else
            _bag_scores(decision_rows, score_column, manifest)
        )
        benign_calibration = bags[
            bags.partition.eq("calibration") & bags.y.eq(0)
        ].session_score.to_numpy(dtype=float)
        if len(benign_calibration) == 0:
            raise ValueError("calibration contains no benign sessions")
        bags["conformal_pvalue"] = _conformal_pvalues(
            bags.session_score.to_numpy(dtype=float), benign_calibration
        )
        bags["y_pred"] = (bags.conformal_pvalue <= float(alpha)).astype(int)
        calibration = bags[bags.partition.eq("calibration")].copy()
        evaluation = bags[bags.partition.eq("evaluation")].copy()
        development = bags[bags.partition.eq("fit") & bags.fit_role.eq("fit_dev")].copy()
        cal_metrics = _metrics(calibration, alpha)
        eval_metrics = _metrics(evaluation, alpha)
        dev_metrics = _metrics(development, alpha)
        alert_rows = (
            _dynamic_rank_earliest_alerts(
                decision_rows,
                evaluation,
                score_column,
                benign_calibration,
                alpha,
                rank_score_column="onset_alignment_score",
                budget=8,
            )
            if dynamic_rank else
            _earliest_alerts(
                decision_rows, evaluation, score_column, benign_calibration, alpha
            )
        )
        evaluation = evaluation.merge(alert_rows, on="zip_path", how="left")
        positive_latencies = evaluation[
            evaluation.y.eq(1) & evaluation.y_pred.eq(1)
        ].latency_from_capture_start_s.dropna()
        eval_metrics["detected_positive_latency_median_s"] = (
            float(positive_latencies.median()) if len(positive_latencies) else None
        )
        eval_metrics["detected_positive_latency_p90_s"] = (
            float(positive_latencies.quantile(0.90)) if len(positive_latencies) else None
        )
        model_dir = out_dir / variant / "phishing_model"
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"imputer": imputer, "model": model, "features": features},
            model_dir / "model.joblib",
        )
        evaluation.to_csv(out_dir / variant / "evaluation_predictions.csv", index=False)
        calibration.to_csv(out_dir / variant / "calibration_predictions.csv", index=False)
        _write_json(out_dir / variant / "metrics.json", {
            "representation": rep,
            "variant": variant,
            "features": len(features),
            "fit_train_rows": len(variant_fit),
            "fit_train_sessions": len(fit_sessions),
            "onset_gate_applied": gated,
            "onset_gate_threshold": onset_gate_threshold if gated else None,
            "dynamic_rank_budget": 8 if dynamic_rank else None,
            "dynamic_rank_score": "onset_alignment_score" if dynamic_rank else None,
            "dynamic_rank_fit_row_retention": (
                float(len(variant_fit) / max(1, len(fit_rows)))
                if dynamic_rank else None
            ),
            "calibration": cal_metrics,
            "fit_dev": dev_metrics,
            "evaluation": eval_metrics,
        })
        results["variants"][variant] = {
            "features": len(features),
            "fit_train_rows": int(len(variant_fit)),
            "dynamic_rank_fit_row_retention": (
                float(len(variant_fit) / max(1, len(fit_rows)))
                if dynamic_rank else None
            ),
            "calibration": cal_metrics,
            "fit_dev": dev_metrics,
            "evaluation": eval_metrics,
        }
    results["paired_delta_evaluation"] = {
        variant: {
            metric: (
                results["variants"][variant]["evaluation"][metric]
                - results["variants"]["all_onsets"]["evaluation"][metric]
            )
            for metric in ("precision", "recall", "f1", "fpr", "roc_auc", "pr_auc")
        }
        for variant in VARIANTS if variant != "all_onsets"
    }
    _write_json(out_dir / "summary.json", results)
    return results


def run(args: argparse.Namespace) -> None:
    started = time.time()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(pathlib.Path(args.split_manifest))
    frame, feature_lists = _load_features(
        pathlib.Path(args.features_csv), pathlib.Path(args.schema_features)
    )
    audit = _causal_audit(
        frame, manifest, merge_s=float(args.nms_merge_s), max_delay_s=float(args.max_decision_delay_s)
    )
    _write_json(out_dir / "causal_audit.json", audit)
    summaries = {}
    for rep in ("425", "228"):
        print(f"[{rep}] fit paired causal candidate routes", flush=True)
        summaries[rep] = _run_representation(
            # _attach_onset_scores makes the one mutable working copy needed by
            # this representation.  Passing another full copy here roughly
            # doubled the peak memory of the 425-column mother table without
            # changing any value.
            rep, frame, manifest, feature_lists[rep], out_dir / rep,
            alpha=float(args.alpha), folds=int(args.onset_oof_folds),
            seed=int(args.seed),
        )
    payload = {
        "protocol": {
            "split": "random_zip_legacy_compatible",
            "seed": int(args.seed),
            "candidate_pool": "all causally finalized local maxima",
            "candidate_rank_gate": "none except dynamic_rank8 comparator",
            "windows": ["adaptive"],
            "aggregation": "strict two-distinct-onset top2 mean",
            "single_candidate_override": False,
            "calibration": "split-conformal benign session score",
            "alpha": float(args.alpha),
            "onset_score_feature": "fit-only OOF stacking; ranker trained only within phishing fit sessions",
            "onset_gate": "fit_dev threshold retaining at least about 95% of oracle-aligned onsets",
            "dynamic_rank8": (
                "availability-time top8-so-far by fit-only onset_alignment_score; "
                "alert persists after the first causal prefix crosses threshold"
            ),
        },
        "causal_audit": audit,
        "representations": summaries,
        "elapsed_s": time.time() - started,
    }
    _write_json(out_dir / "experiment_summary.json", payload)
    print(json.dumps(payload, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-csv", required=True)
    parser.add_argument("--schema-features", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.03)
    parser.add_argument("--onset-oof-folds", type=int, default=5)
    parser.add_argument("--nms-merge-s", type=float, default=2.0)
    parser.add_argument("--max-decision-delay-s", type=float, default=3.5)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

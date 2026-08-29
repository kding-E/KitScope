#!/usr/bin/env python3
"""Open-set Layer 1 suite: held-out stratified calibration + kit-anchored rejection + LOBO.

Addresses the failure mode where a discriminative Layer 1 flags every unfamiliar
benign category as phishing (e.g. the 2026-06-28 final model scored 100% of
unseen AppAct sessions as phishing):

1. Threshold calibration moves from train sessions to a held-out calibration
   split, stratified per benign source group (threshold = max over per-group
   FPR quantiles), so easy negative pools cannot dilute the operating point.
2. A positive-anchored gate models the kit window manifold: kNN distance to a
   reference sample of kit training windows in robust-scaled feature space.
   The decisive form is the session-level gate: a session is phishing only if
   its discriminative score is high AND the distance aggregate of its windows
   is within the kit envelope (tau calibrated on held-out kit sessions), so
   benign traffic that is merely unlike the training benign no longer fires.
3. Open-set validation: benign categories can be held out of training AND
   calibration entirely (no-AppAct, leave-app-groups-out) and are reported as
   unseen-category FPR.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
import re
import shutil
import subprocess
import time
import math
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import RobustScaler

from run_layer1_candidate_phishing_detection import (
    _candidate_eval_mask,
    _sample_labels,
    _source_mask,
    _training_sample_weights,
)
from run_layer1_phishing_detection import (
    feature_columns_for_preset,
    make_model,
    read_second_layer_config,
)
from run_candidate_onset_ranker import (
    _add_diverse_rank,
    _balanced_weights,
    _candidate_rows,
    _feature_columns as _onset_feature_columns,
    _make_ranker,
    _score_model,
)

AGGS = ("max", "topk_mean", "anchor_top2")
DIST_AGGS = ("dist_median_all", "dist_top5_mean")

# Optional session->fine_source relabeling, populated from ``--source-map-csv``.
# When a zip_path is present here it overrides the path-derived source, which is
# how leave-one-dapp-group-out gives blockchain sessions dapp-subgroup labels
# without touching the path-parsing default.  Empty by default => no change.
_SOURCE_OVERRIDE: dict[str, str] = {}
_FILE_HASH_CACHE: dict[str, str] = {}


def _path_key(path: str) -> str:
    """Stable key for joining Windows paths emitted by independent stages."""
    return str(path).strip().replace(chr(92), "/").rstrip("/").casefold()


def _file_sha256(path: str | pathlib.Path) -> str:
    file = pathlib.Path(path)
    key = str(file.resolve())
    if key not in _FILE_HASH_CACHE:
        digest = hashlib.sha256()
        with file.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        _FILE_HASH_CACHE[key] = digest.hexdigest()
    return _FILE_HASH_CACHE[key]


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=pathlib.Path(__file__).resolve().parents[1],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001
        return "unavailable"

# Optional session->capture-date map, populated from ``--split-csv`` when it
# carries a capture_time_utc column.  Used only by --date-balanced-weights;
# sessions absent here fall back to the date embedded in the zip_path name.
_CAPTURE_DATE_OVERRIDE: dict[str, str] = {}
# Anchored to the capture-tool filename tail (_YYYYMMDDTHHMMSSmmm or
# __YYYYMMDD_HHMMSS). A loose (20\d{6}) scan is poisoned by pseudo-dates inside
# millisecond tails ("...T020950426" contains valid-looking "20950426" = year
# 2095) and unix timestamps in public-dataset names, which sends exp(-age/tau)
# recency weights to zero for every real session.
_PATH_DATE_RE = re.compile(r"_(20\d{6})(?:T\d{6,9}|_\d{6})$")


def _capture_date(path: str) -> str:
    override = _CAPTURE_DATE_OVERRIDE.get(str(path))
    if override:
        return override
    stem = str(path).replace(chr(92), "/").rsplit("/", 1)[-1]
    m = _PATH_DATE_RE.search(stem)
    return m.group(1) if m else "na"


def _rep_canon(value: str) -> str:
    """Canonical dataset-relative path for joining external kit labels."""
    text = str(value).replace(chr(92), "/").lower()
    marker = "phish_dataset/"
    return text[text.index(marker):] if marker in text else text


def _date_balanced_weights(train_df: pd.DataFrame, sw) -> np.ndarray:
    """Rescale classifier sample weights so every (label, capture-date) group
    contributes equally: no single capture day's environment dominates."""
    w = np.ones(len(train_df), dtype=float) if sw is None else np.asarray(sw, dtype=float).copy()
    key = pd.DataFrame({
        "lab": train_df["label"].astype(str).to_numpy(),
        "day": [_capture_date(z) for z in train_df["zip_path"].to_numpy()],
    }, index=train_df.index)
    grp_w = key.assign(w=w).groupby(["lab", "day"])["w"].transform("sum")
    w = w / grp_w.to_numpy()
    return w * (len(w) / w.sum())


def _recency_weights(train_df: pd.DataFrame, sw, tau_days: float) -> np.ndarray:
    """Exponential recency decay exp(-age/tau) on classifier sample weights,
    with per-label mass preservation (class prior unchanged). Frozen-model
    concept-drift remedy: the newest capture environments dominate the loss.
    Sessions without a resolvable capture date count as the oldest observed."""
    w = np.ones(len(train_df), dtype=float) if sw is None else np.asarray(sw, dtype=float).copy()
    dates = pd.to_datetime(pd.Series([_capture_date(z) for z in train_df["zip_path"].to_numpy()]),
                           format="%Y%m%d", errors="coerce")
    if int(dates.notna().sum()) == 0:
        return w
    age = (dates.max() - dates).dt.days.astype(float)
    age = age.fillna(float(age.max())).to_numpy()
    out = w * np.exp(-age / float(tau_days))
    lab = train_df["label"].astype(str).to_numpy()
    for value in np.unique(lab):
        mask = lab == value
        got = out[mask].sum()
        if got > 0:
            out[mask] *= w[mask].sum() / got
    return out


def _fine_source(path: str) -> str:
    override = _SOURCE_OVERRIDE.get(_path_key(path))
    if override:
        return str(override)
    p = str(path).replace(chr(92), "/")
    lowered = p.lower()
    if "benign_mobile_appact/" in lowered:
        marker = lowered.index("benign_mobile_appact/") + len("benign_mobile_appact/")
        parts = p[marker:].split("/")
        return "appact/" + (parts[0] if len(parts) > 1 else "ROOT")
    if "benign_appact/" in lowered:
        marker = lowered.index("benign_appact/") + len("benign_appact/")
        parts = p[marker:].split("/")
        return "appact/" + (parts[0] if len(parts) > 1 else "ROOT")
    if "benign_hard_blockchain_act" in lowered or "benign_blockchain_hard" in lowered:
        return "blockchain_hard"
    if "benign_blockchain_act" in lowered or "benign_blockchain/" in lowered:
        return "blockchain"
    if "benign_browser_and_act" in lowered:
        return "browser_same_pipeline"
    if "benign_mobile_browser" in lowered or "benign_browser" in lowered:
        return "browser"
    if "benign" in lowered:
        return "benign_other"
    return "phishing"


def _coarse_group(fine: str) -> str:
    return "appact" if fine.startswith("appact/") else fine


def _robust_source_group(path: str) -> str:
    """Acquisition/threat group used only for fit-time robustness controls."""
    authoritative = _SOURCE_OVERRIDE.get(_path_key(path))
    if authoritative:
        return {
            "phish_interaction": "phishing_interaction",
            "phish_drainer": "phishing_drainer",
        }.get(str(authoritative), _coarse_group(str(authoritative)))
    value = str(path).replace(chr(92), "/")
    lowered = value.lower()
    if "phish_interaction" in lowered:
        return "phishing_interaction"
    if "phish_drainer" in lowered:
        return "phishing_drainer"
    if "benign_hard_blockchain_act" in lowered or "benign_blockchain_hard" in lowered:
        return "blockchain_hard"
    if (
        "benign_browser_and_act" in lowered
        or ("benign_browser" in lowered and "benign_mobile_browser" not in lowered)
    ):
        return "browser_same_pipeline"
    if "benign_mobile_browser" in lowered:
        return "browser_mobile"
    return _coarse_group(_fine_source(path))


def _source_equalized_weights(train_df: pd.DataFrame, base_weights: np.ndarray) -> np.ndarray:
    """Equalize acquisition groups inside each class without using held-out sources."""
    groups = train_df["zip_path"].map(_robust_source_group).to_numpy()
    labels = (train_df["label"].astype(str) == "phishing").to_numpy()
    out = np.asarray(base_weights, dtype=float).copy()
    total_target = float(np.sum(out))
    for is_phishing in (False, True):
        class_mask = labels == is_phishing
        class_groups = sorted(set(groups[class_mask]))
        if not class_groups:
            continue
        per_group_target = (0.5 * total_target) / len(class_groups)
        for group in class_groups:
            mask = class_mask & (groups == group)
            current = float(np.sum(out[mask]))
            if current > 0:
                out[mask] *= per_group_target / current
    return out / max(float(np.mean(out)), 1e-12)


def _eta_squared(x: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Vectorized one-way effect size for every feature column."""
    if len(x) == 0:
        return np.zeros(x.shape[1], dtype=float)
    mean = np.mean(x, axis=0)
    total = np.sum((x - mean) ** 2, axis=0)
    between = np.zeros(x.shape[1], dtype=float)
    for group in np.unique(groups):
        part = x[groups == group]
        if len(part):
            between += len(part) * (np.mean(part, axis=0) - mean) ** 2
    return np.divide(between, total, out=np.zeros_like(between), where=total > 1e-12)


def _source_scrub_features(
    train_df: pd.DataFrame,
    features: list[str],
    keep_fraction: float,
    min_features: int,
    out_dir: pathlib.Path,
) -> tuple[list[str], dict]:
    """Select label-informative features while penalizing fit-source effects.

    All statistics come from fit sessions.  The held-out acquisition source and
    calibration/evaluation sessions are never inspected.  Session medians avoid
    counting the many candidate/window rows from one capture as independent.
    """
    if float(keep_fraction) >= 0.999 or len(features) <= int(min_features):
        return list(features), {"enabled": False, "selected": len(features), "available": len(features)}
    session = train_df[["zip_path", "label", *features]].groupby(
        ["zip_path", "label"], as_index=False
    ).median(numeric_only=True)
    matrix = session[features].replace([np.inf, -np.inf], np.nan)
    # Preserve all feature positions even when one fit-only column is entirely
    # missing.  The audit arrays must stay aligned with ``features``; sklearn's
    # default otherwise drops empty columns during transform.
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    x = imputer.fit_transform(matrix)
    label_groups = session["label"].astype(str).to_numpy()
    label_eta = _eta_squared(x, label_groups)

    benign = session[session["label"].astype(str) != "phishing"].copy()
    benign_groups = benign["zip_path"].map(_robust_source_group).to_numpy()
    if len(np.unique(benign_groups)) >= 2:
        bx = imputer.transform(benign[features].replace([np.inf, -np.inf], np.nan))
        source_eta = _eta_squared(bx, benign_groups)
    else:
        source_eta = np.zeros(len(features), dtype=float)

    # The additive floor prevents tiny, noisy source effects from exploding the
    # ratio.  Label effect breaks ties so constant/uninformative columns sink.
    score = label_eta / (source_eta + 0.02) + 0.05 * label_eta
    n_keep = min(len(features), max(int(min_features), int(math.ceil(len(features) * float(keep_fraction)))))
    order = np.argsort(-score, kind="stable")
    chosen_idx = order[:n_keep]
    chosen = [features[i] for i in chosen_idx]
    audit = pd.DataFrame(
        {
            "feature": features,
            "label_eta2": label_eta,
            "source_eta2_fit_benign": source_eta,
            "source_scrub_score": score,
            "selected": [i in set(chosen_idx.tolist()) for i in range(len(features))],
        }
    ).sort_values(["selected", "source_scrub_score"], ascending=[False, False])
    audit.to_csv(out_dir / "source_scrub_feature_audit.csv", index=False)
    return chosen, {
        "enabled": True,
        "keep_fraction": float(keep_fraction),
        "selected": len(chosen),
        "available": len(features),
        "fit_sessions": int(len(session)),
        "fit_benign_source_groups": sorted(set(benign_groups.tolist())),
    }


def _invariant_feature_screen(
    train_df: pd.DataFrame,
    features: list[str],
    keep_fraction: float,
    min_features: int,
    out_dir: pathlib.Path,
) -> tuple[list[str], dict]:
    """Keep features whose phishing contrast is stable across fit benign sources.

    This is an isolated domain-generalization trial.  For each feature it
    measures a robust standardized phishing-vs-benign effect separately for
    every benign acquisition group available in the fit split.  Features that
    work for only one source, or reverse direction between sources, are ranked
    below features with a consistent worst-source effect.  No calibration,
    evaluation, or held-out-source row is used.
    """
    if float(keep_fraction) >= 0.999 or len(features) <= int(min_features):
        return list(features), {"enabled": False, "selected": len(features), "available": len(features)}
    session = train_df[["zip_path", "label", *features]].groupby(
        ["zip_path", "label"], as_index=False
    ).median(numeric_only=True)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    x = imputer.fit_transform(session[features].replace([np.inf, -np.inf], np.nan))
    is_positive = session["label"].astype(str).to_numpy() == "phishing"
    benign_groups = session.loc[~is_positive, "zip_path"].map(_robust_source_group).to_numpy()
    unique_groups = sorted(set(benign_groups.tolist()))
    if not np.any(is_positive) or len(unique_groups) < 2:
        return list(features), {
            "enabled": False,
            "reason": "requires phishing plus at least two fit benign acquisition groups",
            "selected": len(features),
            "available": len(features),
            "fit_benign_source_groups": unique_groups,
        }

    q75, q25 = np.quantile(x, [0.75, 0.25], axis=0)
    scale = q75 - q25
    fallback = np.std(x, axis=0)
    scale = np.where(scale > 1e-9, scale, np.where(fallback > 1e-9, fallback, 1.0))
    positive_center = np.median(x[is_positive], axis=0)
    effects = []
    benign_x = x[~is_positive]
    for group in unique_groups:
        effects.append((positive_center - np.median(benign_x[benign_groups == group], axis=0)) / scale)
    effect_matrix = np.vstack(effects)
    signs = np.sign(effect_matrix)
    sign_consistency = np.abs(np.mean(signs, axis=0))
    worst_abs_effect = np.min(np.abs(effect_matrix), axis=0)
    median_abs_effect = np.abs(np.median(effect_matrix, axis=0))
    label_eta = _eta_squared(x, session["label"].astype(str).to_numpy())
    source_eta = _eta_squared(benign_x, benign_groups)
    # Worst-source separation is primary.  The consistency factor suppresses
    # source-specific sign reversals; source eta penalizes residual acquisition
    # predictability, while a small floor keeps weak estimates bounded.
    score = (
        worst_abs_effect
        * sign_consistency
        * (0.25 + label_eta)
        / (0.05 + source_eta)
    )
    n_keep = min(len(features), max(int(min_features), int(math.ceil(len(features) * float(keep_fraction)))))
    order = np.argsort(-score, kind="stable")
    chosen_idx = order[:n_keep]
    chosen_set = set(chosen_idx.tolist())
    chosen = [features[i] for i in chosen_idx]
    audit = pd.DataFrame(
        {
            "feature": features,
            "label_eta2": label_eta,
            "benign_source_eta2_fit": source_eta,
            "sign_consistency": sign_consistency,
            "worst_abs_standardized_effect": worst_abs_effect,
            "median_abs_standardized_effect": median_abs_effect,
            "invariant_score": score,
            "selected": [i in chosen_set for i in range(len(features))],
        }
    ).sort_values(["selected", "invariant_score"], ascending=[False, False])
    audit.to_csv(out_dir / "invariant_feature_audit.csv", index=False)
    return chosen, {
        "enabled": True,
        "keep_fraction": float(keep_fraction),
        "selected": len(chosen),
        "available": len(features),
        "fit_sessions": int(len(session)),
        "fit_benign_source_groups": unique_groups,
    }


def _grouped_split_ids(meta: pd.DataFrame, test_size: float, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    """Globally domain-disjoint, approximately stratified session split."""
    g = meta.copy()
    g["stratum"] = g["label"].astype(str) + "|" + g["fine_source"].map(_coarse_group)
    domains = g["domain"].fillna("").astype(str)
    domains = domains.where(domains.str.len() > 0, g["zip_path"].astype(str))
    n_splits = max(2, int(round(1.0 / float(test_size))))
    best: tuple[tuple[float, float, int, int], np.ndarray, np.ndarray] | None = None
    overall = g["stratum"].value_counts(normalize=True)
    for repeat in range(4):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=int(random_state) + repeat,
        )
        for fold, (train_idx, test_idx) in enumerate(
            splitter.split(np.zeros(len(g)), g["stratum"], groups=domains)
        ):
            test_dist = g.iloc[test_idx]["stratum"].value_counts(normalize=True)
            dist_error = float(
                sum(abs(float(test_dist.get(key, 0.0)) - float(value)) for key, value in overall.items())
            )
            size_error = abs(len(test_idx) / max(1, len(g)) - float(test_size))
            missing = int(g.iloc[test_idx]["stratum"].nunique() < g["stratum"].nunique())
            score = (float(missing), size_error, dist_error, repeat * n_splits + fold)
            if best is None or score < best[0]:
                best = (score, train_idx, test_idx)
    if best is None:
        raise RuntimeError("unable to build a globally domain-disjoint split")
    train_idx, test_idx = best[1], best[2]
    train_domains = set(domains.iloc[train_idx])
    test_domains = set(domains.iloc[test_idx])
    overlap = train_domains & test_domains
    if overlap:
        raise RuntimeError(f"domain split leaked {len(overlap)} groups")
    return g.iloc[train_idx]["zip_path"].to_numpy(), g.iloc[test_idx]["zip_path"].to_numpy()


def _parse_split_manifest(
    split: pd.DataFrame,
    known_paths: set[str],
    require_complete: bool = False,
    permitted_missing_feature_keys: set[str] | None = None,
) -> dict:
    """Parse either a legacy train/test CSV or an E0 frozen split manifest.

    E0 manifests use ``sample_path`` plus the three outer partitions and may
    additionally mark fit rows as ``fit_train``/``fit_dev``.  The returned IDs
    always use the exact spelling present in the feature table so downstream
    ``isin`` calls cannot fail because of slash/case differences.
    """
    path_col = "zip_path" if "zip_path" in split.columns else "sample_path" if "sample_path" in split.columns else None
    part_col = "partition" if "partition" in split.columns else "split" if "split" in split.columns else None
    if path_col is None or part_col is None:
        raise ValueError("split CSV requires zip_path/sample_path and partition/split columns")

    work = split.copy()
    work["__path_key"] = work[path_col].astype(str).map(_path_key)
    work["__partition"] = work[part_col].astype(str).str.strip().str.lower().replace(
        {"train": "fit", "valid": "calibration", "validation": "calibration", "test": "evaluation"}
    )
    allowed = {"fit", "calibration", "evaluation", "adaptation_buffer", "excluded"}
    bad = sorted(set(work["__partition"]) - allowed)
    if bad:
        raise ValueError(f"split CSV has unsupported partitions: {bad}")
    conflicts = work.groupby("__path_key")["__partition"].nunique()
    conflicts = conflicts[conflicts > 1]
    if len(conflicts):
        raise ValueError(f"split CSV assigns {len(conflicts)} paths to multiple partitions")
    work = work.drop_duplicates("__path_key", keep="first")

    exact_by_key = {_path_key(path): str(path) for path in known_paths}
    eligible = work[~work["__partition"].isin(["excluded", "adaptation_buffer"])]
    missing_features = sorted(set(eligible["__path_key"]) - set(exact_by_key))
    permitted_missing = {
        _path_key(path) for path in (permitted_missing_feature_keys or set())
    }
    unexpected_missing = sorted(set(missing_features) - permitted_missing)
    if require_complete and unexpected_missing:
        raise ValueError(
            f"split CSV contains {len(unexpected_missing)} eligible sessions absent from features "
            "and absent from the exact validated extraction-error set; "
            f"examples={unexpected_missing[:5]}"
        )

    result: dict[str, object] = {}
    for name in allowed:
        keys = set(work.loc[work["__partition"].eq(name), "__path_key"])
        result[name] = {exact_by_key[key] for key in keys if key in exact_by_key}
    result["fit_dev"] = set()
    result["fit_train"] = set(result["fit"])
    if "fit_role" in work.columns:
        role = work["fit_role"].fillna("").astype(str).str.strip().str.lower()
        dev_keys = set(work.loc[work["__partition"].eq("fit") & role.eq("fit_dev"), "__path_key"])
        train_keys = set(work.loc[work["__partition"].eq("fit") & role.eq("fit_train"), "__path_key"])
        result["fit_dev"] = {exact_by_key[key] for key in dev_keys if key in exact_by_key}
        result["fit_train"] = {exact_by_key[key] for key in train_keys if key in exact_by_key}
        unroled = set(result["fit"]) - set(result["fit_dev"]) - set(result["fit_train"])
        result["fit_train"] = set(result["fit_train"]) | unroled
    result["missing_feature_count"] = len(missing_features)
    result["validated_extraction_error_count"] = len(
        set(missing_features) & permitted_missing
    )
    result["unexpected_missing_feature_count"] = len(unexpected_missing)
    result["unassigned_feature_count"] = len(set(exact_by_key) - set(work["__path_key"]))
    result["is_outer_frozen"] = bool(work["__partition"].eq("calibration").any())
    return result


def _fit_apply_nested_onset_ranker(
    df: pd.DataFrame,
    fit_ids: set[str],
    args: argparse.Namespace,
    out_dir: pathlib.Path,
) -> tuple[pd.DataFrame, dict]:
    """Fit onset ranking on this experiment's fit split only.

    Calibration, outer-test, and whole-source holdout sessions are never used
    to train the upstream supervised ranker.  This avoids the former
    dataset-wide supervised feature-engineering leak and makes the Layer1
    evaluation a genuinely nested protocol.
    """
    cand = _candidate_rows(
        df,
        id_col="zip_path",
        window=str(args.onset_window),
        max_candidate_rank=int(args.max_candidate_rank),
    )
    features = _onset_feature_columns(
        cand,
        feature_mode=str(args.onset_feature_mode),
        include_wallet_vendor=False,
    )
    train = cand[cand["zip_path"].isin(fit_ids)].copy()
    y_train = (
        pd.to_numeric(train["candidate_hit_within_3s"], errors="coerce")
        .fillna(0)
        .astype(int)
        .to_numpy()
    )
    if len(np.unique(y_train)) < 2:
        raise ValueError("nested onset ranker requires positive and negative fit candidates")
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train[features])
    ranker_args = SimpleNamespace(
        model=str(args.onset_model),
        random_state=int(args.random_state),
        max_iter=int(args.onset_max_iter),
        max_leaf_nodes=int(args.onset_max_leaf_nodes),
        learning_rate=float(args.onset_learning_rate),
        l2_regularization=float(args.onset_l2_regularization),
    )
    model = _make_ranker(ranker_args)
    model.fit(x_train, y_train, sample_weight=_balanced_weights(y_train))

    cand = cand.copy()
    cand["learned_onset_score"] = _score_model(model, imputer, features, cand)
    p = np.clip(cand["learned_onset_score"].to_numpy(dtype=float), 1e-9, 1 - 1e-9)
    cand["learned_onset_uncertainty_entropy"] = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    delay = pd.to_numeric(
        cand.get("candidate_decision_delay_s", pd.Series(0.0, index=cand.index)),
        errors="coerce",
    ).fillna(0.0)
    cand["learned_onset_latency_adjusted_score"] = np.clip(
        cand["learned_onset_score"].to_numpy(dtype=float)
        - float(args.onset_latency_penalty_per_s) * delay.to_numpy(dtype=float),
        0.0,
        1.0,
    )
    cand["learned_candidate_rank"] = cand.groupby("zip_path")["learned_onset_score"].rank(
        ascending=False, method="first"
    ).astype(int)
    cand["learned_diverse_rank"] = _add_diverse_rank(
        cand, "learned_onset_score", "zip_path", float(args.onset_diversity_min_gap_s)
    )
    cand["learned_latency_candidate_rank"] = cand.groupby("zip_path")[
        "learned_onset_latency_adjusted_score"
    ].rank(ascending=False, method="first").astype(int)
    cand["learned_latency_diverse_rank"] = _add_diverse_rank(
        cand,
        "learned_onset_latency_adjusted_score",
        "zip_path",
        float(args.onset_diversity_min_gap_s),
    )

    learned_cols = [
        "learned_onset_score",
        "learned_onset_uncertainty_entropy",
        "learned_onset_latency_adjusted_score",
        "learned_candidate_rank",
        "learned_diverse_rank",
        "learned_latency_candidate_rank",
        "learned_latency_diverse_rank",
    ]
    scored_keys = cand[["zip_path", "candidate_rank", *learned_cols]].drop_duplicates(
        ["zip_path", "candidate_rank"]
    )
    # Do not merge all seven scores back through the full 700-column table:
    # pandas materializes the complete frame and briefly exhausts a 32-GB host.
    # Align only the two key columns, then add the small score arrays in place.
    existing_learned = [column for column in learned_cols if column in df.columns]
    if existing_learned:
        df.drop(columns=existing_learned, inplace=True)
    aligned_scores = df[["zip_path", "candidate_rank"]].merge(
        scored_keys,
        on=["zip_path", "candidate_rank"],
        how="left",
        sort=False,
        validate="many_to_one",
    )
    if len(aligned_scores) != len(df):
        raise RuntimeError("nested onset score alignment changed feature row count")
    for column in learned_cols:
        df[column] = aligned_scores[column].to_numpy(copy=False)
    ranked = df
    onset_dir = out_dir / "nested_onset_ranker"
    onset_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"feature": features}).to_csv(onset_dir / "selected_features.csv", index=False)
    pd.DataFrame({"zip_path": sorted(fit_ids)}).to_csv(onset_dir / "fit_sessions.csv", index=False)
    joblib.dump(
        {"model": model, "imputer": imputer, "features": features, "id_column": "zip_path"},
        onset_dir / "candidate_onset_ranker.joblib",
    )
    protocol = {
        "mode": "nested_fit_only",
        "rank_column": "learned_latency_diverse_rank",
        "latency_penalty_per_s": float(args.onset_latency_penalty_per_s),
        "diversity_min_gap_s": float(args.onset_diversity_min_gap_s),
        "fit_sessions": int(train["zip_path"].nunique()),
        "fit_candidate_rows": int(len(train)),
        "scored_sessions": int(cand["zip_path"].nunique()),
        "scored_candidate_rows": int(len(cand)),
        "features": int(len(features)),
        "positive_candidate_rate_fit": float(np.mean(y_train)),
        "calibration_or_test_sessions_used_for_fit": 0,
        "holdout_source_sessions_used_for_fit": 0,
    }
    (onset_dir / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return ranked, protocol


class KitAnchorGate:
    """Positive-anchored open-set gate on the kit window manifold.

    Fit only on phishing (kit) windows: impute -> robust scale -> clip, then
    keep a reference subsample. The gate signal is the kNN distance (k-th
    nearest kit window). Benign families never seen in training sit far from
    the kit manifold even when the discriminative model scores them high.
    """

    def __init__(self, n_ref: int = 20000, knn_k: int = 10, random_state: int = 42):
        self.n_ref = int(n_ref)
        self.knn_k = int(knn_k)
        self.random_state = int(random_state)
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = RobustScaler()
        self.ref_: np.ndarray | None = None
        self.feature_columns: list[str] = []

    def _matrix(self, df: pd.DataFrame) -> np.ndarray:
        x = df[self.feature_columns].to_numpy(dtype=np.float64)
        return np.where(np.isfinite(x), x, np.nan)

    def fit(self, positive_df: pd.DataFrame, feature_columns: list[str]) -> "KitAnchorGate":
        self.feature_columns = list(feature_columns)
        x = self.imputer.fit_transform(self._matrix(positive_df))
        x = np.clip(self.scaler.fit_transform(x), -10.0, 10.0)
        if len(x) > self.n_ref:
            rng = np.random.default_rng(self.random_state)
            x = x[rng.choice(len(x), size=self.n_ref, replace=False)]
        self.ref_ = np.ascontiguousarray(x, dtype=np.float32)
        return self

    def distances(self, df: pd.DataFrame) -> np.ndarray:
        r2 = (self.ref_ ** 2).sum(axis=1)[None, :]
        out = np.empty(len(df), dtype=np.float64)
        k = min(self.knn_k, len(self.ref_) - 1)
        chunk = max(256, int(5e7 // max(len(self.ref_), 1)))
        for lo in range(0, len(df), chunk):
            x = self.imputer.transform(self._matrix(df.iloc[lo:lo + chunk]))
            z = np.clip(self.scaler.transform(x), -10.0, 10.0).astype(np.float32)
            d2 = (z ** 2).sum(axis=1)[:, None] + r2 - 2.0 * (z @ self.ref_.T)
            out[lo:lo + chunk] = np.sqrt(np.maximum(np.partition(d2, k, axis=1)[:, k], 0.0))
        return out

    def save(self, out_dir: pathlib.Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "imputer": self.imputer, "scaler": self.scaler, "ref": self.ref_,
                "knn_k": self.knn_k, "feature_columns": self.feature_columns,
            },
            out_dir / "kit_anchor_gate.joblib",
        )


def _pool_sessions(scored: pd.DataFrame, score_col: str, topk_pool: int) -> pd.DataFrame:
    """Session pooling: score aggregates + kit-distance aggregates per session."""
    columns = ["zip_path", "label", "fine_source", score_col, "dist", "candidate_rank"]
    if "dist_benign" in scored.columns:
        columns.append("dist_benign")
    g = scored[columns].copy()
    g = g.rename(columns={score_col: "s"})
    g = g.sort_values(["zip_path", "s"], ascending=[True, False])
    grp = g.groupby("zip_path", sort=False)
    out = grp.agg(label=("label", "first"), fine_source=("fine_source", "first"),
                  max_score=("s", "max"), n_windows=("s", "size"),
                  dist_median_all=("dist", "median"))
    out["topk_mean_score"] = grp["s"].apply(lambda v: float(v.head(max(1, topk_pool)).mean()))
    out["dist_top5_mean"] = grp["dist"].apply(lambda v: float(v.head(5).mean()))
    if "dist_benign" in g.columns:
        out["dist_benign_median_all"] = grp["dist_benign"].median()
        out["dist_benign_top5_mean"] = grp["dist_benign"].apply(lambda v: float(v.head(5).mean()))
    anchor_max = g.groupby(["zip_path", "candidate_rank"], sort=False)["s"].max().reset_index()
    out["anchor_top2_score"] = anchor_max.groupby("zip_path", sort=False)["s"].apply(
        lambda v: float(v.nlargest(2).mean()))
    return out.reset_index()


def _agg_col(agg: str) -> str:
    return {"max": "max_score", "topk_mean": "topk_mean_score", "anchor_top2": "anchor_top2_score"}[agg]


def _stratified_threshold(sess: pd.DataFrame, agg: str, fpr_target: float) -> tuple[float, dict]:
    """Max over per-benign-source-group FPR quantiles (robust to easy negative pools)."""
    col = _agg_col(agg)
    ben = sess[sess["label"] != "phishing"].copy()
    ben["group"] = ben["fine_source"].map(_coarse_group)
    per_group = {}
    for grp, gdf in ben.groupby("group"):
        theta = float(np.quantile(gdf[col].to_numpy(), 1.0 - fpr_target))
        per_group[grp] = {
            "theta_g": theta,
            "n_calibration_benign": int(len(gdf)),
            "target_fpr": float(fpr_target),
        }
    return (
        max(entry["theta_g"] for entry in per_group.values()) if per_group else 0.5,
        per_group,
    )


def _recall_threshold(sess: pd.DataFrame, agg: str, recall_target: float) -> float:
    col = _agg_col(agg)
    pos = sess[sess["label"] == "phishing"][col].to_numpy()
    if len(pos) == 0:
        return 0.5
    return float(np.quantile(pos, 1.0 - recall_target))


def _split_eval(sess: pd.DataFrame, pred: np.ndarray, unseen_prefixes: list[str]) -> dict:
    is_pos = (sess["label"] == "phishing").to_numpy()
    fine = sess["fine_source"].to_numpy()
    unseen = np.array([any(f.startswith(p) for p in unseen_prefixes) for f in fine]) & ~is_pos
    in_dist = ~unseen

    tp = int((pred & is_pos).sum())
    fn = int((~pred & is_pos).sum())
    fp_in = int((pred & ~is_pos & in_dist).sum())
    recall = tp / max(tp + fn, 1)
    precision_in = tp / max(tp + fp_in, 1)
    out = {
        "recall": recall,
        "precision_in_dist": precision_in,
        "f1_in_dist": 2 * precision_in * recall / max(precision_in + recall, 1e-12),
        "fpr_in_dist_by_group": {},
        "unseen_fpr_by_category": {},
    }
    ben_in_mask = ~is_pos & in_dist
    for grp in sorted({_coarse_group(f) for f in fine[ben_in_mask]}):
        m = ben_in_mask & np.array([_coarse_group(f) == grp for f in fine])
        out["fpr_in_dist_by_group"][grp] = {"fpr": float(pred[m].mean()), "n": int(m.sum())}
    if unseen.any():
        for cat in sorted(set(fine[unseen])):
            m = unseen & (fine == cat)
            out["unseen_fpr_by_category"][cat] = {"fpr": float(pred[m].mean()), "n": int(m.sum())}
        out["unseen_fpr_overall"] = {"fpr": float(pred[unseen].mean()), "n": int(unseen.sum())}
    return out


def _eval_at_threshold(sess: pd.DataFrame, agg: str, thr: float, unseen_prefixes: list[str]) -> dict:
    pred = sess[_agg_col(agg)].to_numpy() >= thr
    out = {"threshold": float(thr)}
    out.update(_split_eval(sess, pred, unseen_prefixes))
    return out


def _parse_weight_grid(text: str) -> list[float]:
    vals: list[float] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"representation blend weight must be in [0,1], got {value}")
        if value not in vals:
            vals.append(value)
    if not vals:
        raise ValueError("--representation-weight-grid produced no candidate weights")
    return vals


def _blend_representation_score(base_score: np.ndarray, rep_score: np.ndarray, weight: float,
                                mode: str) -> np.ndarray:
    w = float(weight)
    if mode == "linear":
        return (1.0 - w) * base_score + w * rep_score
    if mode == "positive_lift":
        return np.clip(base_score + w * np.maximum(rep_score - base_score, 0.0), 0.0, 1.0)
    raise ValueError(f"invalid representation blend mode: {mode!r}")


def _evaluate_representation_weight(
    calib_df: pd.DataFrame,
    *,
    base_score: np.ndarray,
    rep_score: np.ndarray,
    dist: np.ndarray,
    args,
    holdout: list[str],
    weight: float,
    metric_name: str,
) -> dict:
    template = calib_df[["zip_path", "label", "candidate_rank"]].copy()
    template["fine_source"] = template["zip_path"].map(_fine_source)
    template["dist"] = dist
    template["score_plain"] = _blend_representation_score(
        base_score, rep_score, float(weight), str(args.representation_blend_mode)
    )
    if metric_name == "row_auprc":
        y = (calib_df["label"].astype(str) == "phishing").to_numpy().astype(int)
        value = float(average_precision_score(y, template["score_plain"].to_numpy()))
        return {"weight": float(weight), "metric": value, "row_auprc": value}

    sess = _pool_sessions(template, "score_plain", args.topk_pool)
    agg = str(args.representation_tune_agg)
    dist_col = str(args.representation_tune_dist)
    gate_q = float(args.representation_tune_gate_q)
    thr, per_group = _stratified_threshold(sess, agg, float(args.fpr_target))
    cal_pos = sess[sess["label"] == "phishing"]
    tau = float(np.quantile(cal_pos[dist_col].to_numpy(), gate_q)) if len(cal_pos) else float("inf")
    pred = sess[_agg_col(agg)].to_numpy() >= thr
    if metric_name.startswith("session_gate_"):
        pred &= sess[dist_col].to_numpy() <= tau
    ev = _split_eval(sess, pred, holdout)
    value = float(
        ev["recall"]
        if metric_name in {"plain_recall", "session_gate_recall"}
        else ev["f1_in_dist"]
    )
    return {
        "weight": float(weight),
        "metric": value,
        "metric_name": metric_name,
        "threshold": float(thr),
        "threshold_src_fpr_per_group": per_group,
        "tau": tau,
        "recall": float(ev["recall"]),
        "precision_in_dist": float(ev["precision_in_dist"]),
        "f1_in_dist": float(ev["f1_in_dist"]),
    }


def _tune_representation_weight(
    calib_df: pd.DataFrame,
    *,
    base_score: np.ndarray,
    rep_score: np.ndarray,
    dist: np.ndarray,
    args,
    holdout: list[str],
    metric_name: str | None = None,
) -> tuple[float, dict]:
    """Choose one fixed row-score blend weight from held-out calibration rows.

    This is model selection over a predeclared finite grid. Evaluation/test rows
    are never scored here. The default grid includes zero as a fail-safe: the
    balanced Kit-DANN route remains enabled, but a schema may automatically
    reduce its effective fusion weight to zero when fit-only evidence shows no
    gain.
    """
    grid = _parse_weight_grid(args.representation_weight_grid)
    rows: list[dict] = []
    best_row: dict | None = None
    metric_name = str(metric_name or args.representation_tune_metric)
    blend_mode = str(args.representation_blend_mode)
    for w in grid:
        entry = _evaluate_representation_weight(
            calib_df,
            base_score=base_score,
            rep_score=rep_score,
            dist=dist,
            args=args,
            holdout=holdout,
            weight=float(w),
            metric_name=metric_name,
        )
        rows.append(entry)
        if best_row is None or (entry["metric"], -entry["weight"]) > (best_row["metric"], -best_row["weight"]):
            best_row = entry
    assert best_row is not None
    return float(best_row["weight"]), {
        "enabled": True,
        "grid": [float(x) for x in grid],
        "metric": metric_name,
        "blend_mode": blend_mode,
        "selected_weight": float(best_row["weight"]),
        "selected": best_row,
        "sweep": rows,
    }


def _auroc_by_group(sess: pd.DataFrame, col: str, unseen_prefixes: list[str], invert: bool = False) -> dict:
    pos = sess[sess["label"] == "phishing"][col].to_numpy()
    out = {}
    ben = sess[sess["label"] != "phishing"].copy()
    ben["group"] = ben["fine_source"].map(_coarse_group)
    ben["unseen"] = ben["fine_source"].map(lambda f: any(f.startswith(p) for p in unseen_prefixes))
    groups = []
    for group in sorted(set(ben.loc[~ben.unseen, "group"])):
        key = "appact_in_dist" if group == "appact" else str(group)
        groups.append((key, ben[(ben.group == group) & ~ben.unseen]))
    groups.append(("unseen", ben[ben.unseen]))
    for key, gdf in groups:
        if len(gdf) == 0:
            continue
        y = np.concatenate([np.ones(len(pos)), np.zeros(len(gdf))])
        s = np.concatenate([pos, gdf[col].to_numpy()])
        if invert:
            s = -s
        out[key] = {"auroc": float(roc_auc_score(y, s)), "n_benign": int(len(gdf))}
    return out


def _write_run_contract(
    *,
    out_dir: pathlib.Path,
    name: str,
    args,
    metrics: dict,
    session_scores: pd.DataFrame,
    window_scores: pd.DataFrame,
    sess_meta: pd.DataFrame,
    selected_features: list[str],
    eval_frame: pd.DataFrame | None = None,
    score_frame_fn=None,
    tau_window: float | None = None,
) -> None:
    """Emit the paper contract alongside legacy-compatible CSV outputs."""
    headline_variant = str(getattr(args, "headline_variant", "plain"))
    headline_agg = str(getattr(args, "headline_aggregation", "anchor_top2"))
    headline_dist = str(getattr(args, "headline_distance", "dist_median_all"))
    headline_q = float(getattr(args, "headline_gate_q", 0.995))
    if headline_variant == "session_gate":
        headline_key = f"{headline_agg}|{headline_dist}|q{headline_q}|src_fpr"
        headline = metrics["variants"]["session_gate"]["operating_points"][headline_key]
        threshold = float(headline["threshold"])
        tau = float(headline["tau"])
        score_column = f"plain_{_agg_col(headline_agg)}"
        use_support_gate = True
    else:
        aggregate = metrics["variants"][headline_variant]["aggregations"][headline_agg]
        headline_key = f"{headline_variant}|{headline_agg}|src_fpr"
        headline = dict(aggregate["at_src_fpr"])
        threshold = float(aggregate["threshold_src_fpr"])
        headline["threshold"] = threshold
        headline["threshold_src_fpr_per_group"] = aggregate["threshold_src_fpr_per_group"]
        tau = float("inf")
        score_column = f"{headline_variant}_{_agg_col(headline_agg)}"
        use_support_gate = False
    predictions = pd.DataFrame({
        "zip_path": session_scores["zip_path"].astype(str),
        "y_true": session_scores["label"].astype(str).eq("phishing").astype(int),
        "score": session_scores[score_column].astype(float),
        "threshold": threshold,
        "support_distance": session_scores["dist_median_all"].astype(float),
    })
    predictions["y_pred"] = predictions["score"].ge(threshold)
    if use_support_gate:
        predictions["y_pred"] &= predictions["support_distance"].le(tau)
    predictions["y_pred"] = predictions["y_pred"].astype(int)
    meta = sess_meta.copy()
    meta["__key"] = meta["zip_path"].astype(str).map(_path_key)
    meta = meta.drop_duplicates("__key").set_index("__key")
    keys = predictions["zip_path"].map(_path_key)

    def mapped(column: str, default: str = "") -> pd.Series:
        if column not in meta:
            return pd.Series(default, index=predictions.index, dtype="object")
        return keys.map(meta[column]).fillna(default)

    predictions.insert(0, "capture_id", mapped("capture_id"))
    predictions["environment_id"] = mapped("environment_id", "unknown")
    predictions["source_name"] = mapped("source_name").where(
        mapped("source_name").ne(""), session_scores["fine_source"].astype(str)
    )
    positive_group = mapped("positive_supergroup")
    benign_group = mapped("benign_supergroup")
    predictions["supergroup_id"] = np.where(predictions["y_true"].eq(1), positive_group, benign_group)
    predictions["transport_dominance"] = mapped("transport_dominance", "unknown")
    predictions["wallet_family_id"] = mapped("wallet_family_id")
    predictions["phishing_type"] = mapped("phishing_type")
    predictions["heldout_wallet"] = mapped("heldout_wallet")
    predictions["lowo_strength"] = mapped("lowo_strength")
    predictions["alert_time"] = np.nan
    time_column = next(
        (column for column in ("anchor_time_epoch", "feature_anchor_epoch") if column in window_scores),
        None,
    )
    if time_column:
        row_alert_mask = window_scores["score_plain"].astype(float).ge(threshold)
        if use_support_gate:
            row_alert_mask &= window_scores["dist"].astype(float).le(tau)
        row_alert = window_scores[row_alert_mask]
        alert_by_path = row_alert.groupby("zip_path")[time_column].min()
        predictions["alert_time"] = predictions["zip_path"].map(alert_by_path)
    predictions["model_id"] = f"{name}|seed={args.random_state}"
    predictions.to_parquet(out_dir / "predictions.parquet", index=False, compression="zstd")

    permutation_token = str(getattr(args, "permutation_block_token", "") or "").casefold()
    if permutation_token:
        if eval_frame is None or score_frame_fn is None or tau_window is None:
            raise RuntimeError("block permutation requires eval_frame, score_frame_fn, and tau_window")
        block_columns = [
            column for column in selected_features if permutation_token in column.casefold()
        ]
        if not block_columns:
            raise RuntimeError(
                f"permutation block token {permutation_token!r} matches no selected features"
            )
        base_y = predictions["y_true"].to_numpy(dtype=int)
        base_pred = predictions["y_pred"].to_numpy(dtype=int)

        def summary_rates(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
            positive = y_true == 1
            benign = ~positive
            return {
                "recall": float(np.mean(y_pred[positive])) if positive.any() else float("nan"),
                "fpr": float(np.mean(y_pred[benign])) if benign.any() else float("nan"),
            }

        baseline_rates = summary_rates(base_y, base_pred)
        permutation_rows = []
        permutation_rng = np.random.default_rng(int(args.random_state) + 91_337)
        for repeat in range(int(args.permutation_block_repeats)):
            permuted = eval_frame.copy()
            order = permutation_rng.permutation(len(permuted))
            permuted.loc[:, block_columns] = permuted[block_columns].to_numpy()[order]
            permuted_scored = score_frame_fn(permuted)
            permuted_scored["score_window_gate"] = np.where(
                permuted_scored["dist"] <= tau_window,
                permuted_scored["score_plain"],
                0.0,
            )
            variant_column = "score_window_gate" if headline_variant == "window_gate" else "score_plain"
            pooled = _pool_sessions(permuted_scored, variant_column, args.topk_pool)
            score_values = pooled[_agg_col(headline_agg)].to_numpy(dtype=float)
            perm_pred = score_values >= threshold
            if use_support_gate:
                perm_pred &= pooled[headline_dist].to_numpy(dtype=float) <= tau
            permuted_sessions = pooled[["zip_path"]].copy()
            permuted_sessions["y_pred"] = perm_pred.astype(int)
            permuted_sessions = permuted_sessions.merge(
                predictions[["zip_path", "y_true"]], on="zip_path", how="left", validate="one_to_one"
            )
            rates = summary_rates(
                permuted_sessions["y_true"].to_numpy(dtype=int),
                permuted_sessions["y_pred"].to_numpy(dtype=int),
            )
            permutation_rows.append({
                "repeat": repeat, "recall": rates["recall"], "fpr": rates["fpr"],
                "recall_drop": baseline_rates["recall"] - rates["recall"],
                "fpr_change": rates["fpr"] - baseline_rates["fpr"],
            })
        permutation_frame = pd.DataFrame(permutation_rows)
        permutation_frame.to_csv(out_dir / "block_permutation_importance.csv", index=False)
        permutation_summary = {
            "block_token": permutation_token,
            "block_features": block_columns,
            "block_feature_count": len(block_columns),
            "repeats": int(args.permutation_block_repeats),
            "baseline": baseline_rates,
            "mean_recall_drop": float(permutation_frame["recall_drop"].mean()),
            "mean_fpr_change": float(permutation_frame["fpr_change"].mean()),
            "permutation_unit": "candidate_rows_joint_block",
        }
        (out_dir / "block_permutation_importance.json").write_text(
            json.dumps(permutation_summary, indent=2) + "\n", encoding="utf-8"
        )

    group_rows = []
    for group_type, column in (
        ("source", "source_name"), ("environment", "environment_id"), ("overall", None)
    ):
        iterator = [("overall", predictions)] if column is None else predictions.groupby(column, dropna=False)
        for group_id, part in iterator:
            pos = part["y_true"].eq(1)
            ben = ~pos
            pred = part["y_pred"].eq(1)
            group_rows.append({
                "group_type": group_type,
                "group_id": str(group_id),
                "n": int(len(part)),
                "n_positive": int(pos.sum()),
                "n_benign": int(ben.sum()),
                "recall": float(pred[pos].mean()) if pos.any() else np.nan,
                "fpr": float(pred[ben].mean()) if ben.any() else np.nan,
                "precision": float(pos[pred].mean()) if pred.any() else np.nan,
            })
    pd.DataFrame(group_rows).to_csv(out_dir / "group_metrics.csv", index=False)
    (out_dir / "session_metrics.json").write_text(
        json.dumps({"headline_key": headline_key, "headline": headline}, indent=2) + "\n",
        encoding="utf-8",
    )

    effective_config = {**vars(args), "experiment": name, "selected_features": selected_features}
    config_path = out_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(effective_config, sort_keys=True), encoding="utf-8")
    config_sha = _file_sha256(config_path)
    (out_dir / "config_sha256.txt").write_text(config_sha + "\n", encoding="utf-8")
    commit = _git_commit()
    (out_dir / "git_commit.txt").write_text(commit + "\n", encoding="utf-8")
    if args.split_csv:
        shutil.copyfile(args.split_csv, out_dir / "split_manifest.csv")

    input_paths = {
        "features": args.features,
        "model_config": args.config,
        "split_manifest": args.split_csv,
        "source_map": args.source_map_csv,
        "snapshot_manifest": getattr(args, "snapshot_manifest", None),
        "enriched_manifest": getattr(args, "enriched_manifest", None),
        "source_spec": getattr(args, "source_spec", None),
        "feature_validation": getattr(args, "feature_validation_json", None),
        "kit_labels": getattr(args, "representation_kit_labels_csv", None),
        "runner_code": __file__,
    }
    inputs = {}
    for key, value in input_paths.items():
        if value and pathlib.Path(value).is_file():
            path = pathlib.Path(value)
            inputs[key] = {
                "path": str(path.resolve()), "size": int(path.stat().st_size),
                "sha256": _file_sha256(path),
            }
        elif value:
            inputs[key] = {"path": str(pathlib.Path(value).resolve()), "missing": True}
    fingerprint_payload = {
        "experiment": name,
        "config_sha256": config_sha,
        "git_commit": commit,
        "inputs": inputs,
        "feature_schema_sha256": hashlib.sha256(
            "\n".join(selected_features).encode("utf-8")
        ).hexdigest(),
        "predictions_sha256": _file_sha256(out_dir / "predictions.parquet"),
    }
    fingerprint_payload["run_fingerprint"] = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    (out_dir / "run_fingerprint.json").write_text(
        json.dumps(fingerprint_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "runtime.json").write_text(
        json.dumps({"wall_seconds": metrics["setup"]["runtime_s"]}, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "logs.txt").write_text(
        f"experiment={name}\nstatus=complete\nheadline={headline_key}\n", encoding="utf-8"
    )


def run_experiment(name: str, df: pd.DataFrame, sess_meta: pd.DataFrame, base_train_ids: set,
                   base_test_ids: set, args, out_root: pathlib.Path,
                   experiment_holdout: list | None = None,
                   inject_ids: set | None = None,
                   unseen_eval_ids: set | None = None,
                   explicit_calib_ids: set | None = None,
                   fit_dev_ids: set | None = None,
                   enrollment_fit_ids: set | None = None,
                   enrollment_calib_ids: set | None = None) -> dict:
    t0 = time.time()
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    holdout = list(experiment_holdout if experiment_holdout is not None else EXPERIMENTS[name])

    fine_by_id = dict(zip(sess_meta["zip_path"], sess_meta["fine_source"]))
    # ``full_unseen`` is every session of the held-out source(s).  In the default
    # path (no injection) it is exactly the unseen evaluation set, preserving the
    # original behaviour.  Few-shot adaptation moves ``inject_ids`` of them into
    # the training pool and scores FPR on a fixed ``unseen_eval_ids`` reserve.
    full_unseen = {z for z, f in fine_by_id.items() if any(f.startswith(p) for p in holdout)}
    inject_ids = set(inject_ids or set()) & full_unseen
    enrollment_fit_ids = set(enrollment_fit_ids or set()) & full_unseen
    enrollment_calib_ids = set(enrollment_calib_ids or set()) & full_unseen
    if enrollment_fit_ids & enrollment_calib_ids:
        raise RuntimeError("enrollment fit and calibration sessions overlap")
    if inject_ids and (enrollment_fit_ids or enrollment_calib_ids):
        raise RuntimeError("legacy few-shot injection cannot be combined with group enrollment")
    if unseen_eval_ids is None:
        # A frozen E0 manifest makes the outer evaluation partition the only
        # scoreable test population.  Historical auto-splits evaluated every
        # held-out-source row and are retained only for backward compatibility.
        unseen_ids = ((full_unseen & set(base_test_ids)) if explicit_calib_ids is not None else full_unseen) - inject_ids
    else:
        unseen_ids = (set(unseen_eval_ids) & full_unseen) - inject_ids
    trainable_ids = [z for z in base_train_ids if z not in full_unseen]
    if args.split_mode == "domain" and unseen_ids:
        unseen_domains = set(
            sess_meta.loc[sess_meta["zip_path"].isin(unseen_ids), "domain"].fillna("").astype(str)
        )
        trainable_ids = sess_meta.loc[
            sess_meta["zip_path"].isin(trainable_ids)
            & ~sess_meta["domain"].fillna("").astype(str).isin(unseen_domains),
            "zip_path",
        ].tolist()
    if inject_ids or enrollment_fit_ids:
        # Injected held-out benign sessions join the fit/calibration pool after
        # the domain purge so they are never dropped for sharing their own domain.
        trainable_ids = list(dict.fromkeys(
            list(trainable_ids) + sorted(inject_ids | enrollment_fit_ids)
        ))

    # A frozen E0 split supplies the outer fit and calibration IDs verbatim.
    # Legacy manifests with only train/test retain the historical inner split.
    if explicit_calib_ids is not None:
        fit_id_set = set(trainable_ids)
        calib_id_set = (set(explicit_calib_ids) - full_unseen) | enrollment_calib_ids
        if args.split_mode == "domain" and unseen_ids:
            unseen_domains = set(
                sess_meta.loc[sess_meta["zip_path"].isin(unseen_ids), "domain"].fillna("").astype(str)
            )
            calib_id_set -= set(
                sess_meta.loc[
                    sess_meta["zip_path"].isin(calib_id_set)
                    & sess_meta["domain"].fillna("").astype(str).isin(unseen_domains),
                    "zip_path",
                ]
            )
        overlap = fit_id_set & calib_id_set
        if overlap:
            raise RuntimeError(f"frozen split leaked {len(overlap)} sessions between fit and calibration")
    else:
        meta_train = sess_meta[sess_meta["zip_path"].isin(trainable_ids)].copy()
        if args.split_mode == "domain":
            fit_ids, calib_ids = _grouped_split_ids(meta_train, float(args.calib_fraction), int(args.random_state))
        else:
            meta_train["strat"] = meta_train["label"].astype(str) + "|" + meta_train["fine_source"].map(_coarse_group)
            strat = meta_train["strat"] if meta_train["strat"].value_counts().min() >= 2 else meta_train["label"]
            fit_ids, calib_ids = train_test_split(
                meta_train["zip_path"].values, test_size=float(args.calib_fraction),
                random_state=int(args.random_state), stratify=strat.values)
        fit_id_set, calib_id_set = set(fit_ids), set(calib_ids)

    fit_dev_id_set = set(fit_dev_ids or set()) & fit_id_set
    selection_requested = bool(
        getattr(args, "representation_ensemble", False)
        and (
            getattr(args, "representation_early_stopping", False)
            or getattr(args, "representation_tune_weight", False)
        )
    )
    if selection_requested and explicit_calib_ids is not None and not fit_dev_id_set:
        raise RuntimeError(
            "Kit-DANN early stopping/weight selection requires fit_role=fit_dev; "
            "outer calibration is reserved for score calibration and thresholds"
        )
    model_fit_id_set = fit_id_set - fit_dev_id_set if selection_requested else fit_id_set
    if not model_fit_id_set:
        raise RuntimeError("model fit set is empty after reserving fit_dev")

    onset_protocol = {"mode": "precomputed"}
    if args.onset_ranker_mode == "nested":
        df, onset_protocol = _fit_apply_nested_onset_ranker(
            df,
            model_fit_id_set,
            args,
            out_dir,
        )

    # The nested ranker is complete, so discard numeric model features that are
    # outside this representation before producing fit/calibration/eval copies.
    # Protocol metadata (candidate timing/rank/source fields) is retained because
    # ``feature_columns_for_preset(..., "all")`` excludes it by construction.
    selected = feature_columns_for_preset(df, args.feature_preset)
    unused_model_features = sorted(
        set(feature_columns_for_preset(df, "all")) - set(selected)
    )
    if unused_model_features:
        df.drop(columns=unused_model_features, inplace=True)

    train_windows = set(args.train_windows.split(","))
    train_window_mask = df["window_name"].isin(train_windows)
    src_mask = _source_mask(df, {"candidate", "jitter", "oracle"}, args.max_candidate_rank,
                            rank_col=args.candidate_rank_column)
    train_df = df[train_window_mask & src_mask & df["zip_path"].isin(model_fit_id_set)].copy()

    eval_like_mask = _candidate_eval_mask(df, train_windows, args.max_candidate_rank,
                                          rank_col=args.candidate_rank_column)
    calib_df = df[eval_like_mask & df["zip_path"].isin(calib_id_set)].copy()
    fit_dev_df = df[eval_like_mask & df["zip_path"].isin(fit_dev_id_set)].copy()
    eval_ids = {z for z in base_test_ids if z not in full_unseen} | unseen_ids
    eval_df = df[eval_like_mask & df["zip_path"].isin(eval_ids)].copy()

    domain_by_id = dict(zip(sess_meta["zip_path"], sess_meta["domain"].fillna("").astype(str)))
    fit_domains = {domain_by_id[z] for z in fit_id_set}
    calib_domains = {domain_by_id[z] for z in calib_id_set}
    eval_domains = {domain_by_id[z] for z in eval_ids}
    split_audit = {
        "fit_calibration_domain_overlap": int(len(fit_domains & calib_domains)),
        "fit_evaluation_domain_overlap": int(len(fit_domains & eval_domains)),
        "calibration_evaluation_domain_overlap": int(len(calib_domains & eval_domains)),
    }

    print(f"[{name}] train_rows={len(train_df)} calib_rows={len(calib_df)} eval_rows={len(eval_df)} "
          f"unseen_sessions={len(unseen_ids)}", flush=True)

    config = read_second_layer_config(args.config, SimpleNamespace(
        model=args.model, threshold_fpr=args.fpr_target, epochs=None,
        random_state=args.random_state, window="w3"))
    config.window_name = None
    selected, source_scrub_audit = _source_scrub_features(
        train_df,
        selected,
        float(args.source_scrub_keep_fraction),
        int(args.source_scrub_min_features),
        out_dir,
    )
    selected, invariant_screen_audit = _invariant_feature_screen(
        train_df,
        selected,
        float(args.invariant_screen_keep_fraction),
        int(args.source_scrub_min_features),
        out_dir,
    )
    model = make_model(config)
    sw = _training_sample_weights(train_df, args.sample_weighting)
    prequential_weight_audit = {"enabled": False}
    if args.source_weighting == "equalized":
        sw = _source_equalized_weights(train_df, sw)
    # Side-path env-drift options (default off): only the classifier fit input
    # is touched; onset ranker and kit-distance gate stay on the raw rows.
    fit_df, fit_sw = train_df, sw
    if getattr(args, "date_balanced_weights", False):
        fit_sw = _date_balanced_weights(fit_df, fit_sw)
    if getattr(args, "recency_weights", False):
        fit_sw = _recency_weights(fit_df, fit_sw, float(getattr(args, "recency_tau_days", 14.0)))
        ess_r = float(np.square(fit_sw.sum()) / np.square(fit_sw).sum())
        print(f"[{name}] recency weights (tau={getattr(args, 'recency_tau_days', 14.0)}d): "
              f"effective sample size {ess_r:.0f}/{len(fit_sw)}", flush=True)
    if getattr(args, "covariate_shift_weights", False):
        if (
            getattr(args, "precomputed_fit_session_weights", None)
            or getattr(args, "precomputed_fit_row_weights", None)
        ):
            raise RuntimeError("legacy in-process CSW cannot be combined with audited precomputed weights")
        from lightgbm import LGBMClassifier as _DomLGBM

        dom = _DomLGBM(n_estimators=200, learning_rate=0.05, num_leaves=31,
                       subsample=0.8, colsample_bytree=0.8, n_jobs=-1, verbose=-1,
                       random_state=int(args.random_state))
        dom.fit(pd.concat([fit_df[selected], eval_df[selected]], ignore_index=True),
                np.concatenate([np.zeros(len(fit_df)), np.ones(len(eval_df))]))
        p_dom = dom.predict_proba(fit_df[selected])[:, 1]
        clip = float(args.covariate_shift_clip)
        w_cs = np.clip(p_dom / np.clip(1.0 - p_dom, 1e-6, None), 1.0 / clip, clip)
        base_w = np.ones(len(fit_df), dtype=float) if fit_sw is None else np.asarray(fit_sw, dtype=float)
        fit_sw = base_w * w_cs
        fit_sw = fit_sw * (len(fit_sw) / fit_sw.sum())
        ess = float(fit_sw.sum() ** 2 / np.square(fit_sw).sum())
        print(f"[{name}] covariate-shift weights: effective sample size {ess:.0f}/{len(fit_sw)}", flush=True)
    if (
        getattr(args, "precomputed_fit_session_weights", None)
        and getattr(args, "precomputed_fit_row_weights", None)
    ):
        raise RuntimeError("choose exactly one audited CSW weighting unit: session or row")
    if getattr(args, "precomputed_fit_session_weights", None):
        weights_path = pathlib.Path(args.precomputed_fit_session_weights)
        summary_path = pathlib.Path(args.adaptation_summary) if args.adaptation_summary else None
        if summary_path is None or not summary_path.is_file():
            raise RuntimeError("audited adaptation weights require --adaptation-summary")
        adaptation_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if adaptation_summary.get("status") not in {"adapted", "rejected_fallback_frozen"}:
            raise RuntimeError("adaptation summary has an invalid status")
        weights = pd.read_csv(weights_path, low_memory=False)
        required_weight_columns = {"zip_path", "applied_weight"}
        if not required_weight_columns.issubset(weights.columns):
            raise ValueError(f"adaptation weights missing columns: {sorted(required_weight_columns - set(weights.columns))}")
        weights["__key"] = weights["zip_path"].astype(str).map(_path_key)
        if weights["__key"].duplicated().any():
            raise ValueError("adaptation weights contain duplicate sessions")
        forbidden_keys = {
            _path_key(value) for value in (set(calib_id_set) | set(eval_ids))
        }
        leaked = set(weights["__key"]) & forbidden_keys
        if leaked:
            raise RuntimeError(f"adaptation weights contain {len(leaked)} calibration/evaluation sessions")
        weight_map = dict(zip(
            weights["__key"], pd.to_numeric(weights["applied_weight"], errors="raise")
        ))
        row_keys = fit_df["zip_path"].astype(str).map(_path_key)
        missing_weights = sorted(set(row_keys) - set(weight_map))
        if missing_weights:
            raise RuntimeError(f"adaptation weights miss {len(missing_weights)} classifier-fit sessions")
        session_weight = row_keys.map(weight_map).to_numpy(dtype=float)
        base_weight = (
            np.ones(len(fit_df), dtype=float)
            if fit_sw is None else np.asarray(fit_sw, dtype=float)
        )
        fit_sw = base_weight * session_weight
        fit_sw *= len(fit_sw) / fit_sw.sum()
        prequential_weight_audit = {
            "enabled": True,
            "weights_path": str(weights_path.resolve()),
            "weights_sha256": _file_sha256(weights_path),
            "adaptation_summary_path": str(summary_path.resolve()),
            "adaptation_summary_sha256": _file_sha256(summary_path),
            "adaptation_status": adaptation_summary["status"],
            "protocol": adaptation_summary.get("protocol"),
            "target_partition": adaptation_summary.get("target_partition"),
            "rejected_reasons": adaptation_summary.get("rejected_reasons", []),
            "buffer_sessions": adaptation_summary.get("buffer_sessions"),
            "evaluation_feature_rows_read_by_adapter": adaptation_summary.get("evaluation_feature_rows_read"),
            "weighting_unit": "session_median",
        }
    if getattr(args, "precomputed_fit_row_weights", None):
        from web3pcapdetector.csw_weights import apply_row_weights

        weights_path = pathlib.Path(args.precomputed_fit_row_weights)
        summary_path = pathlib.Path(args.adaptation_summary) if args.adaptation_summary else None
        if summary_path is None or not summary_path.is_file():
            raise RuntimeError("audited row adaptation weights require --adaptation-summary")
        adaptation_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if adaptation_summary.get("status") not in {"adapted", "rejected_fallback_frozen"}:
            raise RuntimeError("adaptation summary has an invalid status")
        if adaptation_summary.get("weighting_unit") != "row":
            raise RuntimeError("row-weight detector input does not match adapter weighting_unit")
        weight_sessions = pd.read_parquet(weights_path, columns=["zip_path"])
        forbidden_keys = {
            _path_key(value) for value in (set(calib_id_set) | set(eval_ids))
        }
        leaked = set(weight_sessions.zip_path.astype(str).map(_path_key)) & forbidden_keys
        if leaked:
            raise RuntimeError(
                f"row-level adaptation weights contain {len(leaked)} calibration/evaluation sessions"
            )
        fit_sw, row_weight_audit = apply_row_weights(fit_df, weights_path, fit_sw)
        prequential_weight_audit = {
            "enabled": True,
            **row_weight_audit,
            "adaptation_summary_path": str(summary_path.resolve()),
            "adaptation_summary_sha256": _file_sha256(summary_path),
            "adaptation_status": adaptation_summary["status"],
            "protocol": adaptation_summary.get("protocol"),
            "target_partition": adaptation_summary.get("target_partition"),
            "rejected_reasons": adaptation_summary.get("rejected_reasons", []),
            "buffer_sessions": adaptation_summary.get("buffer_sessions"),
            "evaluation_feature_rows_read_by_adapter": adaptation_summary.get(
                "evaluation_feature_rows_read"
            ),
        }
    jitter_copies = int(getattr(args, "env_jitter_copies", 0) or 0)
    if jitter_copies > 0:
        from screen_env_robust_sidepath import jitter_train_rows

        jrng = np.random.default_rng(int(args.random_state) + 11)
        base_w = np.ones(len(fit_df), dtype=float) if fit_sw is None else np.asarray(fit_sw, dtype=float)
        parts, wparts = [fit_df], [base_w]
        for _ in range(jitter_copies):
            jf = fit_df.copy()
            jf[selected] = jitter_train_rows(fit_df[selected], selected, jrng)
            parts.append(jf)
            wparts.append(base_w)
        fit_df = pd.concat(parts, ignore_index=True)
        fit_sw = np.concatenate(wparts) / (1.0 + jitter_copies)
        print(f"[{name}] env-jitter augmentation: {len(train_df)} -> {len(fit_df)} classifier fit rows", flush=True)
    model.fit(fit_df, feature_columns=selected, sample_weight=fit_sw)

    gate = KitAnchorGate(n_ref=args.gate_ref, knn_k=args.gate_knn_k, random_state=args.random_state)
    gate.fit(train_df[train_df["label"] == "phishing"], selected)
    benign_gate = None
    if args.dual_support_gate:
        benign_gate = KitAnchorGate(n_ref=args.gate_ref, knn_k=args.gate_knn_k, random_state=args.random_state)
        benign_gate.fit(train_df[train_df["label"] != "phishing"], selected)

    rep_scorer = None
    rep_weight = float(args.representation_weight)
    rep_weight_audit = {"enabled": False, "selected_weight": rep_weight}
    rep_train_metadata = None
    selection_base_for_rep = None
    selection_dist_for_rep = None
    rep_selection_df = fit_dev_df if selection_requested else calib_df
    if getattr(args, "representation_ensemble", False):
        from layer1_kit_dann_representation import fit_kit_dann_scorer, load_kit_label_map

        kit_map = load_kit_label_map(args.representation_kit_labels_csv, canonical_path=_rep_canon)
        fit_teacher = None
        if (
            str(args.representation_teacher_loss) != "none"
            and float(args.representation_teacher_loss_weight) > 0
        ):
            fit_teacher = model.predict(fit_df)["second_phish_score"].to_numpy()
        rep_valid_callback = None
        if (
            getattr(args, "representation_early_stopping", False)
            and str(args.representation_early_stop_metric).startswith(("plain_", "session_gate_"))
        ):
            selection_base_for_rep = model.predict(rep_selection_df)["second_phish_score"].to_numpy()
            selection_dist_for_rep = gate.distances(rep_selection_df)

            def rep_valid_callback(rep_prob: np.ndarray, epoch: int, row: dict):
                if getattr(args, "representation_tune_weight", False):
                    _, audit = _tune_representation_weight(
                        rep_selection_df,
                        base_score=selection_base_for_rep,
                        rep_score=rep_prob,
                        dist=selection_dist_for_rep,
                        args=args,
                        holdout=holdout,
                        metric_name=str(args.representation_early_stop_metric),
                    )
                    selected = dict(audit["selected"])
                else:
                    selected = _evaluate_representation_weight(
                        rep_selection_df,
                        base_score=selection_base_for_rep,
                        rep_score=rep_prob,
                        dist=selection_dist_for_rep,
                        args=args,
                        holdout=holdout,
                        weight=float(args.representation_weight),
                        metric_name=str(args.representation_early_stop_metric),
                    )
                selected["epoch"] = int(epoch)
                return float(selected["metric"]), selected

        rep_scorer = fit_kit_dann_scorer(
            fit_df, selected,
            kit_label_map=kit_map, capture_date_of=_capture_date, canonical_path=_rep_canon,
            sample_weight=(np.asarray(fit_sw, dtype=float) if fit_sw is not None else None),
            valid_df=(rep_selection_df if getattr(args, "representation_early_stopping", False) else None),
            teacher_score=fit_teacher,
            valid_score_callback=rep_valid_callback,
            n_eras=int(args.representation_n_eras),
            lam_kit=float(args.representation_lam_kit),
            lam_dom=float(args.representation_lam_era),
            epochs=int(args.representation_epochs), seed=int(args.random_state),
            early_stopping=bool(getattr(args, "representation_early_stopping", False)),
            early_stop_patience=int(args.representation_early_stop_patience),
            early_stop_min_delta=float(args.representation_early_stop_min_delta),
            early_stop_metric=str(args.representation_early_stop_metric),
            early_stop_max_valid_rows=int(args.representation_early_stop_max_valid_rows),
            label_loss=str(args.representation_label_loss),
            focal_gamma=float(args.representation_focal_gamma),
            focal_alpha=float(args.representation_focal_alpha),
            positive_weight=float(args.representation_positive_weight),
            teacher_loss=str(args.representation_teacher_loss),
            teacher_loss_weight=float(args.representation_teacher_loss_weight),
            teacher_margin=float(args.representation_teacher_margin),
            balance_kit_loss=bool(args.representation_balance_kit_loss),
            balance_era_within_label=bool(args.representation_balance_era_within_label),
            auxiliary_weight_clip=float(args.representation_auxiliary_weight_clip),
        )
        cov = int(sum(1 for z in fit_df["zip_path"] if _rep_canon(z) in kit_map))
        rep_train_metadata = getattr(rep_scorer, "metadata", None) if rep_scorer is not None else None
        if rep_scorer is not None:
            rep_scorer.save(out_dir / "kit_dann_representation")
        print(f"[{name}] representation ensemble: kit-DANN MLP fit "
              f"({'ok' if rep_scorer else 'torch-unavailable'}); kit-label coverage "
              f"{cov}/{len(fit_df)} rows; initial_weight={args.representation_weight}", flush=True)

    if rep_scorer is not None and getattr(args, "representation_tune_weight", False):
        selection_base = (selection_base_for_rep if selection_base_for_rep is not None
                          else model.predict(rep_selection_df)["second_phish_score"].to_numpy())
        selection_rep = rep_scorer.score(rep_selection_df)
        selection_dist = (selection_dist_for_rep if selection_dist_for_rep is not None
                          else gate.distances(rep_selection_df))
        rep_weight, rep_weight_audit = _tune_representation_weight(
            rep_selection_df,
            base_score=selection_base,
            rep_score=selection_rep,
            dist=selection_dist,
            args=args,
            holdout=holdout,
        )
        print(f"[{name}] representation ensemble: validation-selected weight={rep_weight:g} "
              f"metric={rep_weight_audit['selected']['metric']:.6f}", flush=True)

    if getattr(args, "representation_selection_only", False):
        if not selection_requested or not fit_dev_id_set:
            raise RuntimeError(
                "--representation-selection-only requires Kit-DANN fit_dev early stopping/weight selection"
            )
        if rep_scorer is None or not rep_train_metadata:
            raise RuntimeError("Kit-DANN selection failed; no representation checkpoint is available")
        selected_epoch = int(rep_train_metadata.get("best_epoch", 0))
        if selected_epoch <= 0:
            raise RuntimeError("Kit-DANN selection produced an invalid best epoch")
        selected_detail = rep_weight_audit.get("selected", {})
        selection_artifact = {
            "schema_version": 2,
            "status": "fit_dev_selected_nonreportable",
            "experiment": name,
            "feature_preset": str(args.feature_preset),
            "features_sha256": _file_sha256(args.features),
            "split_manifest_sha256": _file_sha256(args.split_csv),
            "random_state": int(args.random_state),
            "max_candidate_rank": int(args.max_candidate_rank),
            "candidate_rank_column": str(args.candidate_rank_column),
            "train_windows": str(args.train_windows),
            "onset_ranker_mode": str(args.onset_ranker_mode),
            "sample_weighting": str(args.sample_weighting),
            "fpr_target": float(args.fpr_target),
            "headline_variant": str(args.headline_variant),
            "headline_aggregation": str(args.headline_aggregation),
            "headline_distance": str(args.headline_distance),
            "representation_tune_aggregation": str(args.representation_tune_agg),
            "representation_tune_distance": str(args.representation_tune_dist),
            "representation_tune_gate_q": float(args.representation_tune_gate_q),
            "add_env_shape_ratios": bool(args.add_env_shape_ratios),
            "n_outer_fit_sessions": int(len(fit_id_set)),
            "n_selection_fit_sessions": int(len(model_fit_id_set)),
            "n_fit_dev_sessions": int(len(fit_dev_id_set)),
            "outer_calibration_labels_used_for_selection": False,
            "outer_evaluation_labels_used_for_selection": False,
            "selected_epoch": selected_epoch,
            "selected_weight": float(rep_weight),
            "selection_metric": str(args.representation_tune_metric),
            "selection_metric_value": (
                float(selected_detail["metric"]) if "metric" in selected_detail else None
            ),
            "representation": {
                "blend_mode": str(args.representation_blend_mode),
                "lam_kit": float(args.representation_lam_kit),
                "lam_era": float(args.representation_lam_era),
                "n_eras": int(args.representation_n_eras),
                "label_loss": str(args.representation_label_loss),
                "focal_gamma": float(args.representation_focal_gamma),
                "focal_alpha": float(args.representation_focal_alpha),
                "positive_weight": float(args.representation_positive_weight),
                "teacher_loss": str(args.representation_teacher_loss),
                "teacher_loss_weight": float(args.representation_teacher_loss_weight),
                "teacher_margin": float(args.representation_teacher_margin),
                "balance_kit_loss": bool(args.representation_balance_kit_loss),
                "balance_era_within_label": bool(args.representation_balance_era_within_label),
                "auxiliary_weight_clip": float(args.representation_auxiliary_weight_clip),
                "use_manifest_capture_dates": bool(args.representation_use_manifest_capture_dates),
            },
            "weight_selection": rep_weight_audit,
            "checkpoint_metadata": rep_train_metadata,
            "final_refit_required": True,
        }
        artifact_path = out_dir / "representation_selection.json"
        artifact_path.write_text(
            json.dumps(selection_artifact, indent=2) + "\n", encoding="utf-8"
        )
        selection_metrics = {
            "status": "selection_only_nonreportable",
            "representation_selection": selection_artifact,
        }
        (out_dir / "metrics.json").write_text(
            json.dumps(selection_metrics, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"[{name}] wrote fit-dev selection artifact; outer calibration/evaluation were not scored: "
            f"{artifact_path}",
            flush=True,
        )
        return selection_metrics

    def score_and_gate(part: pd.DataFrame) -> pd.DataFrame:
        pred = model.predict(part)
        keep = ["zip_path", "label", "candidate_rank"] + [
            column for column in ("anchor_time_epoch", "feature_anchor_epoch") if column in part.columns
        ]
        out = part[keep].copy()
        out["fine_source"] = out["zip_path"].map(_fine_source)
        s = pred["second_phish_score"].to_numpy()
        out["score_base"] = s
        if rep_scorer is not None:
            representation_score = rep_scorer.score(part)
            out["score_representation"] = representation_score
            s = _blend_representation_score(
                s,
                representation_score,
                rep_weight,
                str(args.representation_blend_mode),
            )
        out["score_plain"] = s
        out["dist"] = gate.distances(part)
        if benign_gate is not None:
            out["dist_benign"] = benign_gate.distances(part)
        return out

    calib_scored = score_and_gate(calib_df)
    eval_scored = score_and_gate(eval_df)

    # window-level gate ablation: zero out scores of windows outside the kit envelope
    pos_calib_w = calib_scored[calib_scored["label"] == "phishing"]["dist"].to_numpy()
    tau_window = float(np.quantile(pos_calib_w, args.window_gate_q))
    for part in (calib_scored, eval_scored):
        part["score_window_gate"] = np.where(part["dist"] <= tau_window, part["score_plain"], 0.0)

    calib_sess = {v: _pool_sessions(calib_scored, c, args.topk_pool)
                  for v, c in (("plain", "score_plain"), ("window_gate", "score_window_gate"))}
    eval_sess = {v: _pool_sessions(eval_scored, c, args.topk_pool)
                 for v, c in (("plain", "score_plain"), ("window_gate", "score_window_gate"))}

    metrics = {"setup": {
        "experiment": name, "holdout_prefixes": holdout,
        "fewshot_injected_sessions": int(len(inject_ids)),
        "fewshot_reserved_eval_sessions": (int(len(unseen_eval_ids)) if unseen_eval_ids is not None else None),
        "enrollment_fit_sessions": int(len(enrollment_fit_ids)),
        "enrollment_calibration_sessions": int(len(enrollment_calib_ids)),
        "n_outer_fit_sessions": len(fit_id_set),
        "n_fit_sessions": len(model_fit_id_set),
        "n_fit_dev_sessions": len(fit_dev_id_set),
        "n_calib_sessions": len(calib_id_set),
        "n_eval_sessions_in_dist": len(eval_ids) - len(unseen_ids), "n_unseen_sessions": len(unseen_ids),
        "n_train_rows": int(len(train_df)), "n_features": len(selected),
        "model": args.model, "feature_preset": args.feature_preset, "split_mode": args.split_mode,
        "max_candidate_rank": int(args.max_candidate_rank),
        "candidate_rank_column": str(args.candidate_rank_column),
        "frozen_outer_split": bool(explicit_calib_ids is not None),
        "fit_dev_used_for_model_selection": bool(
            (selection_requested and fit_dev_id_set)
            or getattr(args, "_representation_selection_audit", None)
        ),
        "representation_refit_on_all_outer_fit": bool(
            getattr(args, "_representation_selection_audit", None)
        ),
        "outer_calibration_reserved_for_score_and_threshold": bool(explicit_calib_ids is not None),
        "source_weighting": args.source_weighting,
        "source_scrub": source_scrub_audit,
        "invariant_screen": invariant_screen_audit,
        "dual_support_gate": bool(args.dual_support_gate),
        "env_sidepath": {
            "add_env_shape_ratios": bool(getattr(args, "add_env_shape_ratios", False)),
            "env_jitter_copies": int(getattr(args, "env_jitter_copies", 0) or 0),
            "date_balanced_weights": bool(getattr(args, "date_balanced_weights", False)),
            "recency_weights": bool(getattr(args, "recency_weights", False)),
            "recency_tau_days": float(getattr(args, "recency_tau_days", 14.0)),
            "covariate_shift_weights": bool(getattr(args, "covariate_shift_weights", False)),
            "prequential_csw": prequential_weight_audit,
            "representation_ensemble": bool(getattr(args, "representation_ensemble", False)),
            "representation_legacy_linear_fusion": bool(
                getattr(args, "representation_legacy_linear_fusion", False)
            ),
            "representation_weight": float(getattr(args, "representation_weight", 0.5)),
            "representation_blend_mode": str(getattr(args, "representation_blend_mode", "positive_lift")),
            "representation_selected_weight": float(rep_weight),
            "representation_weight_tuning": rep_weight_audit,
            "representation_training": rep_train_metadata,
            "representation_selection_artifact": getattr(
                args, "_representation_selection_audit", None
            ),
            "representation_epochs": int(getattr(args, "representation_epochs", 25)),
            "representation_early_stopping": bool(getattr(args, "representation_early_stopping", True)),
            "representation_lam_kit": float(getattr(args, "representation_lam_kit", 0.3)),
            "representation_lam_era": float(getattr(args, "representation_lam_era", 0.3)),
            "representation_n_eras": int(getattr(args, "representation_n_eras", 6)),
            "representation_label_loss": str(getattr(args, "representation_label_loss", "bce")),
            "representation_positive_weight": float(getattr(args, "representation_positive_weight", 1.0)),
            "representation_teacher_loss": str(getattr(args, "representation_teacher_loss", "none")),
            "representation_teacher_loss_weight": float(getattr(args, "representation_teacher_loss_weight", 0.0)),
            "representation_teacher_margin": float(getattr(args, "representation_teacher_margin", 0.0)),
            "representation_balance_kit_loss": bool(
                getattr(args, "representation_balance_kit_loss", False)
            ),
            "representation_balance_era_within_label": bool(
                getattr(args, "representation_balance_era_within_label", False)
            ),
            "representation_auxiliary_weight_clip": float(
                getattr(args, "representation_auxiliary_weight_clip", 4.0)
            ),
            "representation_use_manifest_capture_dates": bool(
                getattr(args, "representation_use_manifest_capture_dates", False)
            ),
        },
        "split_audit": split_audit,
        "onset_ranker": onset_protocol,
        "fpr_target": args.fpr_target, "recall_target": args.recall_target,
        "headline_variant": str(getattr(args, "headline_variant", "plain")),
        "headline_aggregation": str(getattr(args, "headline_aggregation", "anchor_top2")),
        "headline_distance": str(getattr(args, "headline_distance", "dist_median_all")),
        "headline_gate_q": float(getattr(args, "headline_gate_q", 0.995)),
        "window_gate_q": args.window_gate_q, "tau_window": tau_window,
        "session_gate_qs": [0.99, 0.995], "gate_ref": args.gate_ref, "gate_knn_k": args.gate_knn_k,
    }, "variants": {}}

    # score-threshold variants (plain / window-gated), thresholds from held-out calib
    for vname in ("plain", "window_gate"):
        cs, es = calib_sess[vname], eval_sess[vname]
        vout = {"aggregations": {}}
        for agg in AGGS:
            thr_fpr, per_group = _stratified_threshold(cs, agg, args.fpr_target)
            thr_rec = _recall_threshold(cs, agg, args.recall_target)
            vout["aggregations"][agg] = {
                "threshold_src_fpr": thr_fpr,
                "threshold_src_fpr_per_group": per_group,
                "threshold_recall": thr_rec,
                "at_src_fpr": _eval_at_threshold(es, agg, thr_fpr, holdout),
                "at_recall_target": _eval_at_threshold(es, agg, thr_rec, holdout),
                "auroc": _auroc_by_group(es, _agg_col(agg), holdout),
            }
        metrics["variants"][vname] = vout

    # session-level kit-envelope gate: score >= theta AND session distance <= tau
    cs, es = calib_sess["plain"], eval_sess["plain"]
    cal_pos = cs[cs["label"] == "phishing"]
    sg = {"dist_auroc": {d: _auroc_by_group(es, d, holdout, invert=True) for d in DIST_AGGS},
          "operating_points": {}}
    for agg in AGGS:
        thr_fpr, _ = _stratified_threshold(cs, agg, args.fpr_target)
        thr_rec = _recall_threshold(cs, agg, args.recall_target)
        for dist_col in DIST_AGGS:
            for q in (0.99, 0.995):
                tau = float(np.quantile(cal_pos[dist_col].to_numpy(), q))
                for thr_name, thr in (("src_fpr", thr_fpr), ("recall_target", thr_rec)):
                    pred = (es[_agg_col(agg)].to_numpy() >= thr) & (es[dist_col].to_numpy() <= tau)
                    key = f"{agg}|{dist_col}|q{q}|{thr_name}"
                    entry = {"threshold": float(thr), "tau": tau}
                    entry.update(_split_eval(es, pred, holdout))
                    gate_only = es[dist_col].to_numpy() > tau
                    entry["gate_only_rejection_by_group"] = {
                        grp: float(gate_only[(es["fine_source"].map(_coarse_group) == grp).to_numpy()].mean())
                        for grp in sorted(set(es["fine_source"].map(_coarse_group)))}
                    sg["operating_points"][key] = entry
    sg["diagnostic_low_fpr"] = {}
    diagnostic_agg = str(getattr(args, "headline_aggregation", "anchor_top2"))
    diagnostic_dist = str(getattr(args, "headline_distance", "dist_median_all"))
    diagnostic_q = float(getattr(args, "headline_gate_q", 0.995))
    diagnostic_tau = float(np.quantile(cal_pos[diagnostic_dist].to_numpy(), diagnostic_q))
    for diagnostic_fpr in (0.001, 0.005, 0.01):
        diagnostic_threshold, diagnostic_groups = _stratified_threshold(
            cs, diagnostic_agg, diagnostic_fpr
        )
        diagnostic_pred = (
            es[_agg_col(diagnostic_agg)].to_numpy() >= diagnostic_threshold
        ) & (es[diagnostic_dist].to_numpy() <= diagnostic_tau)
        entry = {
            "target_fpr": diagnostic_fpr,
            "threshold": float(diagnostic_threshold),
            "tau": diagnostic_tau,
            "per_source_calibration": diagnostic_groups,
            "minimum_resolvable_fpr_by_source": {
                group: 1.0 / max(1, int(values["n_calibration_benign"]))
                for group, values in diagnostic_groups.items()
            },
        }
        entry.update(_split_eval(es, diagnostic_pred, holdout))
        sg["diagnostic_low_fpr"][f"fpr_{diagnostic_fpr:g}"] = entry
    metrics["variants"]["session_gate"] = sg

    # Optional density-ratio gate: require positive proximity relative to the
    # fit-benign manifold.  It is an experimental variant and never replaces the
    # positive-only gate unless explicitly requested.
    if benign_gate is not None:
        cs = calib_sess["plain"].copy()
        es = eval_sess["plain"].copy()
        eps = 1e-6
        cs["dual_support_ratio"] = cs["dist_median_all"] / (cs["dist_benign_median_all"] + eps)
        es["dual_support_ratio"] = es["dist_median_all"] / (es["dist_benign_median_all"] + eps)
        cal_pos = cs[cs["label"] == "phishing"]
        dual = {"operating_points": {}, "ratio_auroc": _auroc_by_group(es, "dual_support_ratio", holdout, invert=True)}
        for q in (0.99, 0.995):
            tau_ratio = float(np.quantile(cal_pos["dual_support_ratio"].to_numpy(), q))
            tau_positive = float(np.quantile(cal_pos["dist_median_all"].to_numpy(), q))
            for agg in AGGS:
                thr_fpr, _ = _stratified_threshold(cs, agg, args.fpr_target)
                pred = (
                    (es[_agg_col(agg)].to_numpy() >= thr_fpr)
                    & (es["dist_median_all"].to_numpy() <= tau_positive)
                    & (es["dual_support_ratio"].to_numpy() <= tau_ratio)
                )
                key = f"{agg}|ratio_q{q}|src_fpr"
                entry = {
                    "threshold": float(thr_fpr),
                    "tau_positive": tau_positive,
                    "tau_ratio": tau_ratio,
                }
                entry.update(_split_eval(es, pred, holdout))
                dual["operating_points"][key] = entry
        metrics["variants"]["dual_support_gate"] = dual
        eval_sess["plain"]["dual_support_ratio"] = es["dual_support_ratio"].to_numpy()
        benign_gate.save(out_dir / "benign_anchor_gate")

    merged = eval_sess["plain"].rename(columns={c: f"plain_{c}" for c in
                                                ("max_score", "topk_mean_score", "anchor_top2_score")})
    wg = eval_sess["window_gate"][["zip_path", "max_score", "topk_mean_score", "anchor_top2_score"]]
    merged = merged.merge(wg.rename(columns={c: f"window_gate_{c}" for c in
                                             ("max_score", "topk_mean_score", "anchor_top2_score")}),
                          on="zip_path", how="left")
    merged["unseen"] = merged["fine_source"].map(lambda f: any(f.startswith(p) for p in holdout))
    merged.to_csv(out_dir / "session_predictions.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    eval_scored.to_csv(out_dir / "candidate_window_scores.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    calib_scored.to_csv(out_dir / "calib_window_scores.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    model.save(out_dir / "phishing_detection_model")
    gate.save(out_dir / "kit_anchor_gate")
    metrics["setup"]["runtime_s"] = round(time.time() - t0, 1)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _write_run_contract(
        out_dir=out_dir,
        name=name,
        args=args,
        metrics=metrics,
        session_scores=merged,
        window_scores=eval_scored,
        sess_meta=sess_meta,
        selected_features=selected,
        eval_frame=eval_df,
        score_frame_fn=score_and_gate,
        tau_window=tau_window,
    )
    print(f"[{name}] done in {metrics['setup']['runtime_s']}s", flush=True)
    return metrics


EXPERIMENTS: dict[str, list[str]] = {
    "e1_full_benign": [],
    "e2_no_appact": ["appact"],
    "e3_no_browser": ["browser"],
    "e4_no_blockchain": ["blockchain"],
    "e5_hard_blockchain": ["blockchain_hard"],
    "lobo_fold1": ["appact/Discord", "appact/ClashRoyale", "appact/Line"],
    "lobo_fold2": ["appact/Messenger", "appact/Twitch", "appact/KakaoTalk"],
    "lobo_fold3": ["appact/Zoom", "appact/Slack", "appact/WhatsApp"],
    "lobo_fold4": ["appact/Meet", "appact/Skype", "appact/Webex", "appact/Telegram"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default=r"outputs/experiment_20260702_layer1_appact_benign/ranked_features_merged/features_with_candidate_onset_ranks.parquet")
    parser.add_argument("--out-dir", default="outputs/experiment_20260703_layer1_openset/openset_suite")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--experiments", default=",".join(EXPERIMENTS))
    parser.add_argument("--train-windows", default="adaptive,w2,w4,w7,w10")
    parser.add_argument("--candidate-rank-column", default="learned_latency_diverse_rank")
    parser.add_argument("--onset-ranker-mode", choices=["precomputed", "nested"], default="precomputed",
                        help="nested fits the supervised onset ranker inside each experiment using fit sessions only")
    parser.add_argument("--onset-window", default="adaptive")
    parser.add_argument("--onset-feature-mode", choices=["candidate_only", "candidate_plus_window"], default="candidate_plus_window")
    parser.add_argument("--onset-model", choices=["lightgbm", "xgboost", "catboost", "hist_gradient_boosting"], default="lightgbm")
    parser.add_argument("--onset-max-iter", type=int, default=250)
    parser.add_argument("--onset-learning-rate", type=float, default=0.04)
    parser.add_argument("--onset-max-leaf-nodes", type=int, default=31)
    parser.add_argument("--onset-l2-regularization", type=float, default=0.05)
    parser.add_argument("--onset-diversity-min-gap-s", type=float, default=8.0)
    parser.add_argument("--onset-latency-penalty-per-s", type=float, default=0.02)
    parser.add_argument("--max-candidate-rank", type=int, default=8)
    parser.add_argument("--topk-pool", type=int, default=3)
    parser.add_argument("--model", default="lightgbm")
    parser.add_argument("--feature-preset", default="kit_l1_core")
    parser.add_argument("--sample-weighting", default="rank")
    parser.add_argument("--source-weighting", choices=["none", "equalized"], default="none",
                        help="Experimental fit-only acquisition-group balancing; default leaves the main route unchanged")
    parser.add_argument("--source-scrub-keep-fraction", type=float, default=1.0,
                        help="Experimental fit-only label/source effect feature screen; 1.0 disables it")
    parser.add_argument("--source-scrub-min-features", type=int, default=48)
    parser.add_argument("--invariant-screen-keep-fraction", type=float, default=1.0,
                        help="Experimental fit-only worst-source invariant feature screen; 1.0 disables it")
    parser.add_argument("--representation-ensemble", action="store_true",
                        help="Blend a kit-anchored env-invariant MLP representation (kit-family + "
                             "gradient-reversal era heads) into the score model at row level "
                             "(mean of P(phish)). Frozen-model portability enhancement.")
    parser.add_argument("--representation-weight", type=float, default=0.5,
                        help="Representation weight in [0,1] when validation tuning is disabled.")
    parser.add_argument("--representation-blend-mode", choices=["linear", "positive_lift"], default="positive_lift",
                        help="How to combine LightGBM and representation row scores.")
    parser.add_argument("--representation-legacy-linear-fusion", action="store_true",
                        help="Compatibility switch for the old fixed-0.5 linear fusion. "
                             "Equivalent to --representation-blend-mode linear "
                             "--no-representation-tune-weight --representation-weight 0.5.")
    parser.add_argument("--representation-kit-labels-csv",
                        default="outputs/experiment_20260723_layer1_temporal_kit_scope/"
                                "layer2_backend_kit_static_evidence_route/backend_kit_final_labels/"
                                "backend_kit_final_labels.csv",
                        help="Backend-kit labels CSV for the MLP kit-anchor head.")
    parser.add_argument("--representation-epochs", type=int, default=25,
                        help="Maximum Kit-DANN training epochs. The recommended route uses early stopping.")
    parser.add_argument("--representation-lam-kit", type=float, default=0.3,
                        help="Weight of the training-only kit-family auxiliary loss")
    parser.add_argument("--representation-lam-era", type=float, default=0.3,
                        help="Maximum weight of the capture-era adversarial loss")
    parser.add_argument("--representation-n-eras", type=int, default=6,
                        help="Number of fit-period capture-time partitions")
    parser.add_argument("--representation-early-stopping", dest="representation_early_stopping",
                        action="store_true", default=True,
                        help="Select the Kit-DANN checkpoint using frozen outer-fit fit_dev rows only.")
    parser.add_argument("--no-representation-early-stopping", dest="representation_early_stopping",
                        action="store_false",
                        help="Train Kit-DANN for exactly --representation-epochs epochs.")
    parser.add_argument("--representation-early-stop-patience", type=int, default=5)
    parser.add_argument("--representation-early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--representation-early-stop-metric",
                        choices=["plain_f1", "plain_recall", "session_gate_f1", "session_gate_recall",
                                 "auprc", "auroc", "neg_bce"],
                        default="plain_f1",
                        help="Validation objective for Kit-DANN early stopping. Plain metrics match the "
                             "recommended ungated headline; session-gate metrics are retained for ablations.")
    parser.add_argument("--representation-early-stop-max-valid-rows", type=int, default=100000,
                        help="Subsample validation rows for Kit-DANN early stopping; <=0 uses all rows.")
    parser.add_argument("--representation-label-loss", choices=["bce", "focal"], default="bce",
                        help="Primary Kit-DANN phishing loss.")
    parser.add_argument("--representation-focal-gamma", type=float, default=2.0)
    parser.add_argument("--representation-focal-alpha", type=float, default=0.25)
    parser.add_argument("--representation-positive-weight", type=float, default=1.0,
                        help="Extra positive-class multiplier for the representation label loss.")
    parser.add_argument("--representation-teacher-loss", choices=["none", "mse", "margin"], default="none",
                        help="Optional LightGBM-teacher preservation loss for the representation score.")
    parser.add_argument("--representation-teacher-loss-weight", type=float, default=0.0)
    parser.add_argument("--representation-teacher-margin", type=float, default=0.0,
                        help="Allowed score deviation before the margin teacher loss is applied.")
    parser.add_argument("--representation-balance-kit-loss", action="store_true",
                        help="Use bounded inverse-family weights for the kit auxiliary loss.")
    parser.add_argument("--representation-balance-era-within-label", action="store_true",
                        help="Balance label-by-era cells and exclude eras lacking both labels from the "
                             "adversarial auxiliary loss.")
    parser.add_argument("--representation-auxiliary-weight-clip", type=float, default=4.0,
                        help="Symmetric clipping bound for balanced kit/era auxiliary weights.")
    parser.add_argument("--representation-use-manifest-capture-dates", action="store_true",
                        help="Use source-aware capture_start_utc from the frozen split for Kit-DANN eras. "
                             "Recommended only with label-balanced era loss on this snapshot.")
    parser.add_argument("--representation-tune-weight", dest="representation_tune_weight",
                        action="store_true", default=True,
                        help="Choose the row-score blend weight from a fixed outer-fit fit_dev grid.")
    parser.add_argument("--no-representation-tune-weight", dest="representation_tune_weight",
                        action="store_false",
                        help="Use --representation-weight directly instead of calibration weight tuning.")
    parser.add_argument("--representation-weight-grid", default="0,0.1,0.2,0.3,0.4,0.5,0.7,1.0",
                        help="Comma-separated fit_dev candidates, including 0 as a no-representation guardrail.")
    parser.add_argument("--representation-tune-metric",
                        choices=["plain_f1", "plain_recall", "session_gate_f1", "session_gate_recall",
                                 "row_auprc"],
                        default="plain_f1")
    parser.add_argument("--representation-tune-agg", choices=list(AGGS), default="anchor_top2")
    parser.add_argument("--representation-tune-dist", choices=list(DIST_AGGS), default="dist_median_all")
    parser.add_argument("--representation-tune-gate-q", type=float, default=0.995)
    parser.add_argument(
        "--representation-selection-only",
        action="store_true",
        help=(
            "Stop after fit_train/fit_dev Kit-DANN epoch and blend-weight selection. "
            "Writes representation_selection.json and deliberately produces no reportable evaluation scores."
        ),
    )
    parser.add_argument(
        "--representation-selection-json",
        default=None,
        help=(
            "Frozen fit-dev selection artifact. Its epoch/weight are verified and then applied while all "
            "learned components are refit on the complete outer-fit partition."
        ),
    )
    parser.add_argument("--covariate-shift-weights", action="store_true",
                        help="Side-path (env drift): reweight classifier training rows by clipped odds of an "
                             "adversarial train-vs-eval-period domain classifier fit on UNLABELED features "
                             "(standard covariate-shift correction; needs periodic retraining in deployment).")
    parser.add_argument("--covariate-shift-clip", type=float, default=4.0,
                        help="Clip for covariate-shift odds weights (weights in [1/clip, clip]).")
    parser.add_argument("--precomputed-fit-session-weights", default=None,
                        help="Leakage-safe E7 fit-session weights produced from fit + adaptation_buffer only.")
    parser.add_argument("--precomputed-fit-row-weights", default=None,
                        help="Audited E7 candidate-window weights keyed by path/window/candidate_id.")
    parser.add_argument("--adaptation-summary", default=None,
                        help="E7 adapter audit proving evaluation features/labels were not loaded.")
    parser.add_argument("--add-env-shape-ratios", action="store_true",
                        help="Side-path (env drift): append self-normalized iat/dur shape-ratio columns after load "
                             "(consumed by kit_l1_env_hardened and the explicit *_shape_ratios presets; "
                             "other presets ignore them).")
    parser.add_argument("--env-jitter-copies", type=int, default=0,
                        help="Side-path (env drift): append N environment-jittered copies of the classifier "
                             "training rows (timing scale / packetization perturbation; ranker and gate unaffected).")
    parser.add_argument("--date-balanced-weights", action="store_true",
                        help="Side-path (env drift): rescale classifier sample weights so each (label, capture-date) "
                             "group contributes equally to the loss.")
    parser.add_argument("--recency-weights", action="store_true",
                        help="Side-path (env drift): exponential recency decay on classifier sample weights "
                             "(frozen-model concept-drift remedy; per-label mass preserved).")
    parser.add_argument("--recency-tau-days", type=float, default=14.0,
                        help="Decay constant (days) for --recency-weights.")
    parser.add_argument("--dual-support-gate", action="store_true",
                        help="Add an experimental positive-vs-benign support-ratio gate")
    parser.add_argument("--split-mode", choices=["zip", "domain"], default="zip",
                        help="zip: stratified by session (baseline protocol); domain: grouped so no domain crosses train/calib/test")
    parser.add_argument("--split-csv", default=None,
                        help="Optional legacy train/test CSV or frozen E0 manifest with "
                             "fit/calibration/evaluation and optional fit_role.")
    parser.add_argument("--require-frozen-split", action="store_true",
                        help="Fail unless --split-csv contains an explicit outer calibration partition.")
    parser.add_argument("--require-complete-split", action="store_true",
                        help="Fail when any non-excluded split session is absent from the feature table.")
    parser.add_argument("--source-map-csv", default=None,
                        help="Optional CSV (zip_path,fine_source) relabeling sessions; enables leave-one-dapp-group-out "
                             "without changing the path-derived default source for other sessions.")
    parser.add_argument("--require-source-map", action="store_true",
                        help="Fail unless every loaded feature session has an authoritative source-map row.")
    parser.add_argument("--snapshot-manifest", default=None,
                        help="Frozen E0 sample manifest recorded in each run fingerprint.")
    parser.add_argument("--enriched-manifest", default=None,
                        help="Frozen E0 enriched manifest recorded in each run fingerprint.")
    parser.add_argument("--source-spec", default=None,
                        help="Frozen source whitelist YAML recorded in each run fingerprint.")
    parser.add_argument("--feature-validation-json", default=None,
                        help="Passed streaming validation report supplying the already-computed feature hash.")
    parser.add_argument("--require-run-provenance", action="store_true",
                        help="Require snapshot, enriched manifest, source spec, split, and source map inputs.")
    parser.add_argument("--experiments-json", default=None,
                        help="Optional JSON {experiment_name: [holdout_prefix,...]} merged into the built-in EXPERIMENTS, "
                             "so data-driven holdouts (e.g. dapp groups) are addressable via --experiments.")
    parser.add_argument("--fewshot-experiment", default=None,
                        help="Run a few-shot benign-adaptation curve for this experiment name (e.g. e4_no_blockchain): "
                             "inject k held-out benign sessions into fit/calibration and score FPR on a fixed reserve.")
    parser.add_argument("--fewshot-ks", default="0,10,25,50,100,250,500",
                        help="Comma-separated k values for the few-shot adaptation curve.")
    parser.add_argument("--fewshot-eval-fraction", type=float, default=0.6,
                        help="Fraction of held-out benign sessions reserved as the fixed evaluation pool (never injected).")
    parser.add_argument("--permutation-block-token", default=None,
                        help="Optional selected-feature substring for evaluation-only joint block permutation importance.")
    parser.add_argument("--permutation-block-repeats", type=int, default=20)
    parser.add_argument("--enrollment-manifest", default=None,
                        help="Frozen group-aware E5 manifest with enrollment_fit, enrollment_calibration, "
                             "and evaluation roles. Cannot be combined with legacy --fewshot-experiment.")
    parser.add_argument("--enrollment-mode",
                        choices=["calibration_only", "refit_with_hard_negatives"],
                        default="calibration_only")
    parser.add_argument("--enrollment-experiment", default="e5_hard_blockchain",
                        help="Experiment key whose holdout prefix defines the E5 target benign family.")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--calib-fraction", type=float, default=0.2)
    parser.add_argument("--fpr-target", type=float, default=0.03)
    parser.add_argument("--headline-variant", choices=["plain", "window_gate", "session_gate"],
                        default="plain")
    parser.add_argument("--headline-aggregation", choices=list(AGGS), default="anchor_top2")
    parser.add_argument("--headline-distance", choices=list(DIST_AGGS), default="dist_median_all")
    parser.add_argument("--headline-gate-q", type=float, choices=[0.99, 0.995], default=0.995)
    parser.add_argument("--recall-target", type=float, default=0.975)
    parser.add_argument("--window-gate-q", type=float, default=0.995)
    parser.add_argument("--gate-ref", type=int, default=20000)
    parser.add_argument("--gate-knn-k", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    validated_missing_feature_keys: set[str] = set()
    if args.feature_validation_json:
        validation = json.loads(pathlib.Path(args.feature_validation_json).read_text(encoding="utf-8"))
        feature_path = pathlib.Path(args.features).resolve()
        if not validation.get("passed"):
            raise RuntimeError("feature validation report is not passed")
        if pathlib.Path(validation["features_parquet"]).resolve() != feature_path:
            raise RuntimeError("feature validation report refers to a different feature file")
        if int(validation.get("features_parquet_size_bytes", -1)) != feature_path.stat().st_size:
            raise RuntimeError("validated feature file size changed after validation")
        missing_count = int(validation.get("missing_sessions", 0))
        missing_examples = list(validation.get("missing_examples", []))
        if missing_count:
            if not validation.get("missing_exactly_matches_error_log"):
                raise RuntimeError(
                    "feature validation has missing sessions not exactly matched by its extraction-error log"
                )
            if len(missing_examples) != missing_count:
                raise RuntimeError(
                    "feature validation does not expose the complete validated missing-session set"
                )
            validated_missing_feature_keys = {_path_key(path) for path in missing_examples}
        _FILE_HASH_CACHE[str(feature_path)] = str(validation["features_parquet_sha256"])
    if args.require_run_provenance:
        required_paths = {
            "snapshot_manifest": args.snapshot_manifest,
            "enriched_manifest": args.enriched_manifest,
            "source_spec": args.source_spec,
            "feature_validation_json": args.feature_validation_json,
            "split_csv": args.split_csv,
            "source_map_csv": args.source_map_csv,
        }
        bad = [name for name, value in required_paths.items() if not value or not pathlib.Path(value).is_file()]
        if bad:
            raise RuntimeError(f"missing required run-provenance files: {bad}")
    if args.require_frozen_split and not args.split_csv:
        raise RuntimeError("--require-frozen-split requires --split-csv")
    args._representation_selection_audit = None
    if args.representation_selection_only and args.representation_selection_json:
        raise RuntimeError(
            "--representation-selection-only and --representation-selection-json are mutually exclusive"
        )
    if args.representation_selection_json:
        selection_path = pathlib.Path(args.representation_selection_json)
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection.get("status") != "fit_dev_selected_nonreportable":
            raise RuntimeError("representation selection artifact has an invalid status")
        experiments = [value.strip() for value in str(args.experiments).split(",") if value.strip()]
        if experiments != [str(selection.get("experiment"))]:
            raise RuntimeError(
                "representation selection artifact is bound to exactly one matching experiment"
            )
        expected = {
            "feature_preset": str(args.feature_preset),
            "features_sha256": _file_sha256(args.features),
            "split_manifest_sha256": _file_sha256(args.split_csv),
            "random_state": int(args.random_state),
            "max_candidate_rank": int(args.max_candidate_rank),
            "candidate_rank_column": str(args.candidate_rank_column),
            "train_windows": str(args.train_windows),
            "onset_ranker_mode": str(args.onset_ranker_mode),
            "sample_weighting": str(args.sample_weighting),
            "fpr_target": float(args.fpr_target),
            "headline_variant": str(args.headline_variant),
            "headline_aggregation": str(args.headline_aggregation),
            "headline_distance": str(args.headline_distance),
            "representation_tune_aggregation": str(args.representation_tune_agg),
            "representation_tune_distance": str(args.representation_tune_dist),
            "representation_tune_gate_q": float(args.representation_tune_gate_q),
            "add_env_shape_ratios": bool(args.add_env_shape_ratios),
        }
        mismatches = {
            key: {"artifact": selection.get(key), "requested": value}
            for key, value in expected.items()
            if selection.get(key) != value
        }
        rep_expected = {
            "blend_mode": str(args.representation_blend_mode),
            "lam_kit": float(args.representation_lam_kit),
            "lam_era": float(args.representation_lam_era),
            "n_eras": int(args.representation_n_eras),
            "label_loss": str(args.representation_label_loss),
            "focal_gamma": float(args.representation_focal_gamma),
            "focal_alpha": float(args.representation_focal_alpha),
            "positive_weight": float(args.representation_positive_weight),
            "teacher_loss": str(args.representation_teacher_loss),
            "teacher_loss_weight": float(args.representation_teacher_loss_weight),
            "teacher_margin": float(args.representation_teacher_margin),
            "balance_kit_loss": bool(args.representation_balance_kit_loss),
            "balance_era_within_label": bool(args.representation_balance_era_within_label),
            "auxiliary_weight_clip": float(args.representation_auxiliary_weight_clip),
            "use_manifest_capture_dates": bool(args.representation_use_manifest_capture_dates),
        }
        artifact_rep = selection.get("representation", {})
        mismatches.update({
            f"representation.{key}": {"artifact": artifact_rep.get(key), "requested": value}
            for key, value in rep_expected.items()
            if artifact_rep.get(key) != value
        })
        if mismatches:
            raise RuntimeError(
                "representation selection artifact is incompatible with this refit: "
                + json.dumps(mismatches, sort_keys=True)
            )
        selected_epoch = int(selection.get("selected_epoch", 0))
        selected_weight = float(selection.get("selected_weight", -1.0))
        if selected_epoch <= 0 or not 0.0 <= selected_weight <= 1.0:
            raise RuntimeError("representation selection artifact has invalid epoch/weight")
        args.representation_epochs = selected_epoch
        args.representation_weight = selected_weight
        args.representation_early_stopping = False
        args.representation_tune_weight = False
        args._representation_selection_audit = {
            "path": str(selection_path.resolve()),
            "sha256": _file_sha256(selection_path),
            "selected_epoch": selected_epoch,
            "selected_weight": selected_weight,
            "n_selection_fit_sessions": selection.get("n_selection_fit_sessions"),
            "n_fit_dev_sessions": selection.get("n_fit_dev_sessions"),
            "outer_calibration_labels_used_for_selection": False,
            "outer_evaluation_labels_used_for_selection": False,
        }
        print(
            "loaded fit-dev selection artifact; final training will refit on every outer-fit session: "
            f"epoch={selected_epoch} weight={selected_weight:g}",
            flush=True,
        )
    if args.representation_legacy_linear_fusion:
        args.representation_blend_mode = "linear"
        args.representation_tune_weight = False
        args.representation_weight = 0.5

    out_root = pathlib.Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.require_source_map and not args.source_map_csv:
        raise RuntimeError("--require-source-map requires --source-map-csv")
    _SOURCE_OVERRIDE.clear()
    _CAPTURE_DATE_OVERRIDE.clear()
    smap = None
    if args.source_map_csv:
        smap = pd.read_csv(args.source_map_csv, dtype=str)
        required_source_columns = {"zip_path", "fine_source"}
        missing_source_columns = required_source_columns - set(smap.columns)
        if missing_source_columns:
            raise ValueError(f"source map missing columns: {sorted(missing_source_columns)}")
        smap["__path_key"] = smap["zip_path"].astype(str).map(_path_key)
        conflicts = smap.groupby("__path_key")["fine_source"].nunique()
        conflicts = conflicts[conflicts > 1]
        if len(conflicts):
            raise ValueError(f"source map has {len(conflicts)} conflicting path assignments")
        _SOURCE_OVERRIDE.update(
            dict(zip(smap["__path_key"], smap["fine_source"].astype(str)))
        )
        print(f"applied source map: {len(_SOURCE_OVERRIDE)} sessions relabeled", flush=True)
    if args.experiments_json:
        extra = json.loads(pathlib.Path(args.experiments_json).read_text(encoding="utf-8"))
        EXPERIMENTS.update({str(k): list(v) for k, v in extra.items()})
        print(f"merged {len(extra)} extra experiment holdouts", flush=True)

    print(f"loading {args.features}", flush=True)
    if args.features.endswith(".parquet"):
        df = pd.read_parquet(args.features)
    else:
        df = pd.read_csv(args.features, low_memory=False)
    # The validated parquet contains many repeated string identifiers (session,
    # window, candidate source, and provenance fields).  Expanding every value
    # into a separate Python ``str`` object can add tens of GB before fitting.
    # Categorical scalars preserve the same string comparisons/mapping behavior
    # while keeping one copy per distinct value, which is essential for the
    # repeated LOWO/E2-W cells on 32-GB hosts.
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]) and not isinstance(
            df[c].dtype, pd.CategoricalDtype
        ):
            df[c] = df[c].astype("category")
    print(f"loaded {df.shape}", flush=True)

    if args.add_env_shape_ratios:
        from screen_env_robust_sidepath import add_shape_ratios

        df, added_ratio_cols = add_shape_ratios(df)
        print(f"added {len(added_ratio_cols)} env shape-ratio columns: {added_ratio_cols}", flush=True)

    # base session split: zip_path stratified (identical protocol to the
    # 2026-07-02 baseline run) or domain-grouped (no domain on both sides)
    train_windows = set(args.train_windows.split(","))
    trainable_mask = df["window_name"].isin(train_windows)
    # Keep seeded session splits invariant to feature-extraction / parquet row
    # order.  Guard ablations can contain the same sessions with different
    # window counts and physical row order; without this canonical ordering,
    # identical seeds silently produce different train/calibration/test IDs.
    labels = (
        _sample_labels(df.loc[trainable_mask, ["zip_path", "label", "domain"]], "zip_path")
        .sort_values("zip_path", kind="stable")
        .reset_index(drop=True)
    )
    sess_meta = labels.copy()
    feature_keys = set(sess_meta["zip_path"].astype(str).map(_path_key))
    missing_source_keys = sorted(feature_keys - set(_SOURCE_OVERRIDE))
    if args.require_source_map and missing_source_keys:
        raise RuntimeError(
            f"source map does not cover {len(missing_source_keys)} feature sessions; "
            f"examples={missing_source_keys[:5]}"
        )
    sess_meta["fine_source"] = sess_meta["zip_path"].map(_fine_source)
    sess_meta["__path_key"] = sess_meta["zip_path"].astype(str).map(_path_key)

    def attach_session_metadata(source: pd.DataFrame, path_column: str, columns: list[str]) -> None:
        keyed = source.copy()
        keyed["__path_key"] = keyed[path_column].astype(str).map(_path_key)
        keyed = keyed.drop_duplicates("__path_key").set_index("__path_key")
        for column in columns:
            if column not in keyed:
                continue
            values = sess_meta["__path_key"].map(keyed[column])
            if column in sess_meta:
                existing = sess_meta[column]
                sess_meta[column] = values.where(values.notna() & values.astype(str).ne(""), existing)
            else:
                sess_meta[column] = values

    if smap is not None:
        attach_session_metadata(
            smap, "zip_path", ["capture_id", "source_name", "source_variant", "fine_source"]
        )
    explicit_calib_ids = None
    fit_dev_ids: set[str] = set()
    if args.split_csv:
        sp = pd.read_csv(args.split_csv, dtype=str)
        parsed_split = _parse_split_manifest(
            sp,
            set(labels["zip_path"].astype(str)),
            require_complete=bool(args.require_complete_split),
            permitted_missing_feature_keys=validated_missing_feature_keys,
        )
        time_col = "capture_start_utc" if "capture_start_utc" in sp.columns else "capture_time_utc" if "capture_time_utc" in sp.columns else None
        path_col = "zip_path" if "zip_path" in sp.columns else "sample_path"
        attach_session_metadata(
            sp,
            path_col,
            [
                "capture_id", "environment_id", "source", "source_name", "source_variant",
                "positive_supergroup", "benign_supergroup", "transport_dominance",
                "wallet_family_id", "phishing_type", "benign_service_group_id",
                "hard_benign_target_id", "heldout_wallet", "lowo_strength",
                "constraint_group_id", "split_id_name",
            ],
        )
        if (
            args.date_balanced_weights
            or args.recency_weights
            or args.representation_use_manifest_capture_dates
        ) and time_col:
            ts = pd.to_datetime(sp[time_col], errors="coerce", utc=True, format="mixed")
            _CAPTURE_DATE_OVERRIDE.update({
                str(z): d.strftime("%Y%m%d") for z, d in zip(sp[path_col], ts) if pd.notna(d)})
            print(f"capture-date map from split csv: {len(_CAPTURE_DATE_OVERRIDE)} sessions", flush=True)
        train_ids = sorted(parsed_split["fit"])
        test_ids = sorted(parsed_split["evaluation"])
        fit_dev_ids = set(parsed_split["fit_dev"])
        if parsed_split["is_outer_frozen"]:
            explicit_calib_ids = set(parsed_split["calibration"])
        if args.require_frozen_split and explicit_calib_ids is None:
            raise RuntimeError("--require-frozen-split needs an explicit calibration partition")
        print(f"applied split csv: fit={len(train_ids)} fit_dev={len(fit_dev_ids)} "
              f"calibration={len(explicit_calib_ids or set())} evaluation={len(test_ids)} "
              f"unassigned_features={parsed_split['unassigned_feature_count']} "
              f"missing_features={parsed_split['missing_feature_count']} "
              f"validated_extraction_errors={parsed_split['validated_extraction_error_count']} "
              f"unexpected_missing={parsed_split['unexpected_missing_feature_count']}", flush=True)
        if not train_ids or not test_ids:
            raise RuntimeError("--split-csv produced an empty fit or evaluation set")
        if explicit_calib_ids is not None and not explicit_calib_ids:
            raise RuntimeError("frozen --split-csv produced an empty calibration set")
    elif args.split_mode == "domain":
        train_ids, test_ids = _grouped_split_ids(sess_meta, float(args.test_size), int(args.random_state))
    else:
        train_ids, test_ids = train_test_split(
            labels["zip_path"].values, test_size=float(args.test_size),
            random_state=int(args.random_state), stratify=labels["y"].values)

    all_metrics = {}
    for name in [e.strip() for e in args.experiments.split(",") if e.strip()]:
        all_metrics[name] = run_experiment(
            name, df, sess_meta, set(train_ids), set(test_ids), args, out_root,
            explicit_calib_ids=explicit_calib_ids, fit_dev_ids=fit_dev_ids,
        )

    if args.fewshot_experiment:
        if args.enrollment_manifest:
            raise RuntimeError("legacy few-shot and E5 group enrollment cannot run together")
        fs = _run_fewshot_curve(args.fewshot_experiment, df, sess_meta,
                                set(train_ids), set(test_ids), args, out_root,
                                explicit_calib_ids=explicit_calib_ids, fit_dev_ids=fit_dev_ids)
        all_metrics[f"{args.fewshot_experiment}__fewshot_curve"] = fs

    if args.enrollment_manifest:
        if explicit_calib_ids is None:
            raise RuntimeError("E5 group enrollment requires a frozen outer calibration partition")
        enrollment_metrics = _run_group_enrollment(
            args.enrollment_experiment, df, sess_meta, set(train_ids), set(test_ids),
            args, out_root, explicit_calib_ids=explicit_calib_ids,
            fit_dev_ids=fit_dev_ids,
        )
        all_metrics[f"{args.enrollment_experiment}__{args.enrollment_mode}"] = enrollment_metrics

    (out_root / "suite_metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    print(f"wrote {out_root / 'suite_metrics.json'}", flush=True)


def _fewshot_unseen_fpr(metrics: dict) -> float | None:
    """Pull the headline unseen FPR from a run at the main operating point."""
    op = (
        metrics.get("variants", {}).get("plain", {}).get("aggregations", {})
        .get("anchor_top2", {}).get("at_src_fpr")
    )
    if op and isinstance(op.get("unseen_fpr_overall"), dict):
        return float(op["unseen_fpr_overall"]["fpr"])
    return None


def _run_fewshot_curve(exp_name: str, df: pd.DataFrame, sess_meta: pd.DataFrame,
                       base_train_ids: set, base_test_ids: set, args,
                       out_root: pathlib.Path,
                       explicit_calib_ids: set | None = None,
                       fit_dev_ids: set | None = None) -> dict:
    """Benign few-shot adaptation curve for a held-out-source experiment.

    Reserves a fixed fraction of the held-out benign sessions as an untouched
    evaluation pool, then sweeps k = number of *other* held-out benign sessions
    injected into fit/calibration.  This turns the E4-style "unseen category is
    unrejectable" boundary result into a deployment answer: how much benign
    traffic from a new environment is needed before its FPR collapses.
    """
    holdout = list(EXPERIMENTS[exp_name])
    fine_by_id = dict(zip(sess_meta["zip_path"], sess_meta["fine_source"]))
    full_unseen = sorted(z for z, f in fine_by_id.items() if any(f.startswith(p) for p in holdout))
    rng = np.random.default_rng(int(args.random_state))
    if explicit_calib_ids is not None:
        # Frozen protocol: evaluation rows are never candidates for enrollment.
        eval_candidates = sorted(set(full_unseen) & set(base_test_ids))
        inject_candidates = sorted(set(full_unseen) & set(base_train_ids))
        eval_pool = set(eval_candidates)
        inject_pool = [inject_candidates[i] for i in rng.permutation(len(inject_candidates))]
    else:
        perm = list(rng.permutation(len(full_unseen)))
        n_eval = int(round(float(args.fewshot_eval_fraction) * len(full_unseen)))
        eval_pool = {full_unseen[i] for i in perm[:n_eval]}
        inject_pool = [full_unseen[i] for i in perm[n_eval:]]
    ks = [int(k) for k in str(args.fewshot_ks).split(",") if str(k).strip()]

    curve = {"experiment": exp_name, "holdout_prefixes": holdout,
             "n_held_out_total": len(full_unseen), "n_reserved_eval": len(eval_pool),
             "n_inject_pool": len(inject_pool), "ks": ks, "points": []}
    fs_root = out_root / f"{exp_name}_fewshot"
    for k in ks:
        k_eff = min(k, len(inject_pool))
        inject_ids = set(inject_pool[:k_eff])
        m = run_experiment(f"{exp_name}_fewshot_k{k}", df, sess_meta,
                           base_train_ids, base_test_ids, args, fs_root,
                           experiment_holdout=holdout, inject_ids=inject_ids,
                           unseen_eval_ids=eval_pool,
                           explicit_calib_ids=explicit_calib_ids,
                           fit_dev_ids=fit_dev_ids)
        curve["points"].append({
            "k": k, "k_effective": k_eff,
            "unseen_fpr": _fewshot_unseen_fpr(m),
            "n_unseen_eval": m["setup"]["n_unseen_sessions"],
        })
        print(f"[fewshot {exp_name}] k={k} unseen_fpr={curve['points'][-1]['unseen_fpr']}", flush=True)
    (out_root / f"{exp_name}_fewshot_curve.json").write_text(
        json.dumps(curve, indent=2), encoding="utf-8")
    return curve


def _run_group_enrollment(exp_name: str, df: pd.DataFrame, sess_meta: pd.DataFrame,
                          base_train_ids: set, base_test_ids: set, args,
                          out_root: pathlib.Path, *, explicit_calib_ids: set,
                          fit_dev_ids: set | None = None) -> dict:
    """Run one frozen, service-group-aware E5 enrollment condition."""
    if exp_name not in EXPERIMENTS:
        raise ValueError(f"unknown enrollment experiment: {exp_name}")
    manifest_path = pathlib.Path(args.enrollment_manifest)
    manifest = pd.read_csv(manifest_path, dtype=str)
    path_column = "sample_path" if "sample_path" in manifest else "zip_path" if "zip_path" in manifest else None
    group_column = "benign_service_group_id"
    required = {"role", group_column}
    if path_column is None or not required.issubset(manifest.columns):
        raise ValueError(
            f"enrollment manifest needs sample_path/zip_path, role, and {group_column}"
        )
    allowed_roles = {"enrollment_fit", "enrollment_calibration", "evaluation"}
    unknown_roles = sorted(set(manifest["role"].astype(str)) - allowed_roles)
    if unknown_roles:
        raise ValueError(f"unknown enrollment roles: {unknown_roles}")
    group_role_counts = manifest.groupby(group_column)["role"].nunique()
    if (group_role_counts > 1).any():
        raise RuntimeError(
            f"{int((group_role_counts > 1).sum())} benign service groups cross enrollment roles"
        )
    path_lookup = {
        _path_key(path): str(path)
        for path in sess_meta["zip_path"].astype(str)
    }
    manifest["__zip_path"] = manifest[path_column].astype(str).map(
        lambda value: path_lookup.get(_path_key(value), "")
    )
    if manifest["__zip_path"].eq("").any():
        raise RuntimeError(
            f"enrollment manifest has {int(manifest['__zip_path'].eq('').sum())} sessions missing from features"
        )
    role_ids = {
        role: set(manifest.loc[manifest.role.eq(role), "__zip_path"].astype(str))
        for role in allowed_roles
    }
    if not role_ids["evaluation"]:
        raise RuntimeError("enrollment manifest has no evaluation sessions")
    if not role_ids["evaluation"].issubset(base_test_ids):
        raise RuntimeError("E5 evaluation sessions are not a subset of frozen outer evaluation")
    development_ids = role_ids["enrollment_fit"] | role_ids["enrollment_calibration"]
    if not development_ids.issubset(base_train_ids):
        raise RuntimeError("E5 enrollment development sessions are not a subset of frozen outer fit")
    enrollment_fit = (
        role_ids["enrollment_fit"]
        if args.enrollment_mode == "refit_with_hard_negatives"
        else set()
    )
    name = f"{exp_name}_{args.enrollment_mode}_{manifest_path.stem}"
    metrics = run_experiment(
        name, df, sess_meta, base_train_ids, base_test_ids, args, out_root,
        experiment_holdout=list(EXPERIMENTS[exp_name]),
        unseen_eval_ids=role_ids["evaluation"],
        explicit_calib_ids=explicit_calib_ids,
        fit_dev_ids=fit_dev_ids,
        enrollment_fit_ids=enrollment_fit,
        enrollment_calib_ids=role_ids["enrollment_calibration"],
    )
    metrics["setup"]["enrollment"] = {
        "mode": args.enrollment_mode,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _file_sha256(manifest_path),
        "selected_fit_groups": int(
            manifest.loc[manifest.role.eq("enrollment_fit"), group_column].nunique()
        ),
        "used_fit_groups": int(
            manifest.loc[manifest.role.eq("enrollment_fit"), group_column].nunique()
            if args.enrollment_mode == "refit_with_hard_negatives" else 0
        ),
        "calibration_groups": int(
            manifest.loc[manifest.role.eq("enrollment_calibration"), group_column].nunique()
        ),
        "evaluation_groups": int(
            manifest.loc[manifest.role.eq("evaluation"), group_column].nunique()
        ),
        "group_overlap": 0,
    }
    run_dir = out_root / name
    (run_dir / "session_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    return metrics


if __name__ == "__main__":
    main()

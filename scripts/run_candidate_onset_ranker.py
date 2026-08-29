#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import pathlib

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from web3pcapdetector.models.utils import label_to_binary


LEAK_SUFFIXES = (
    "anchor_error_s",
    "abs_anchor_error_s",
    "offset_from_oracle_s",
    "emit_latency_vs_oracle_s",
    "available_lag_from_oracle_s",
    "_epoch",
    "_rel_s",
)
LEAK_PREFIXES = (
    "candidate_hit_",
    "candidate_hit_oracle_",
)
META_COLS = {
    "sample_id", "zip_path", "label", "domain", "url", "candidate_id", "candidate_source",
    "candidate_reason", "oracle_anchor_epoch", "pcap_start_epoch", "pcap_end_epoch",
    "candidate_start_epoch", "candidate_anchor_epoch", "candidate_emit_epoch", "candidate_available_epoch",
    "candidate_start_rel_s", "candidate_emit_rel_s",
    "anchor_time_epoch", "feature_anchor_epoch", "window_start_epoch", "window_end_epoch",
    "window_available_epoch", "window_latency_vs_oracle_s", "window_available_after_candidate_s",
    "quality_capture_duration_after_anchor_s",
}


def _candidate_rows(df: pd.DataFrame, id_col: str, window: str, max_candidate_rank: int) -> pd.DataFrame:
    # Build the mask on the source frame and copy only the final candidate
    # subset.  Copying the full 1.9M x 700-column feature table before filtering
    # can transiently consume more than 25 GB on the formal 32-GB runner.
    mask = pd.Series(True, index=df.index)
    if window and "window_name" in df.columns:
        mask &= df["window_name"].eq(window)
    src = df.get("candidate_source", pd.Series([""] * len(df), index=df.index))
    rank = pd.to_numeric(
        df.get("candidate_rank", pd.Series([np.nan] * len(df), index=df.index)),
        errors="coerce",
    )
    mask &= src.str.startswith("traffic_heuristic") & rank.ge(1) & rank.le(int(max_candidate_rank))
    work = df.loc[mask].copy()
    if work.empty:
        raise ValueError("no heuristic candidate rows found")
    work["candidate_rank"] = rank.loc[work.index].astype(int)
    return work.drop_duplicates([id_col, "candidate_rank"]).reset_index(drop=True)


def _feature_columns(df: pd.DataFrame, feature_mode: str, include_wallet_vendor: bool = False) -> list[str]:
    cols = []
    for c in df.columns:
        if c in META_COLS:
            continue
        if any(c.startswith(p) for p in LEAK_PREFIXES):
            continue
        if any(c.endswith(s) for s in LEAK_SUFFIXES):
            continue
        if not include_wallet_vendor and any(token in c for token in ("wallet_vendor", "wallet_disturbance", "analytics_ads", "third_party_static", "software_update")):
            continue
        if feature_mode == "candidate_only" and not c.startswith("candidate_"):
            continue
        if feature_mode == "candidate_plus_window" and not (
            c.startswith("candidate_")
            or c.startswith(("base_", "role_", "role_group_", "phase_", "flow_", "proto_", "rpc_"))
        ):
            continue
        if c in {"candidate_rank"}:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return sorted(cols)


def _session_labels(cand: pd.DataFrame, id_col: str) -> pd.DataFrame:
    out = cand[[id_col, "label"]].drop_duplicates(id_col).copy()
    out["y"] = label_to_binary(out["label"].values).astype(int)
    return out


def _balanced_weights(y: np.ndarray) -> np.ndarray:
    y = y.astype(int)
    pos = max(1, int((y == 1).sum()))
    neg = max(1, int((y == 0).sum()))
    w = np.ones(len(y), dtype=float)
    w[y == 1] = len(y) / (2.0 * pos)
    w[y == 0] = len(y) / (2.0 * neg)
    return w


def _rank_metrics(cand: pd.DataFrame, id_col: str, score_col: str) -> dict:
    hit1 = {1: [], 3: [], 5: [], 8: []}
    hit3 = {1: [], 3: [], 5: [], 8: []}
    best_abs = []
    rank1_abs = []
    rank1_err = []
    rows = []
    for session_id, g in cand.groupby(id_col):
        g = g.sort_values(score_col, ascending=False)
        abs_err = pd.to_numeric(g["candidate_abs_anchor_error_s"], errors="coerce")
        err = pd.to_numeric(g["candidate_anchor_error_s"], errors="coerce")
        if abs_err.notna().any():
            best_abs.append(float(abs_err.min()))
            rank1_abs.append(float(abs_err.iloc[0]))
            rank1_err.append(float(err.iloc[0]))
        for k in hit1:
            top = g.head(k)
            top_abs = pd.to_numeric(top["candidate_abs_anchor_error_s"], errors="coerce")
            hit1[k].append(bool((top_abs <= 1.0).any()))
            hit3[k].append(bool((top_abs <= 3.0).any()))
        top1 = g.iloc[0]
        rows.append({
            "session_uid": session_id,
            "sample_id": top1.get("sample_id", ""),
            "zip_path": top1.get("zip_path", ""),
            "label": top1.get("label", ""),
            "domain": top1.get("domain", ""),
            "top1_candidate_rank_original": int(top1.get("candidate_rank", -1)),
            "top1_score": float(top1.get(score_col, np.nan)),
            "top1_anchor_error_s": float(top1.get("candidate_anchor_error_s", np.nan)),
            "top1_abs_anchor_error_s": float(top1.get("candidate_abs_anchor_error_s", np.nan)),
        })
    metrics = {
        "n_sessions": int(cand[id_col].nunique()),
        "rank1_abs_anchor_error_mean": float(np.nanmean(rank1_abs)) if rank1_abs else None,
        "rank1_anchor_error_mean": float(np.nanmean(rank1_err)) if rank1_err else None,
        "closest_abs_anchor_error_mean": float(np.nanmean(best_abs)) if best_abs else None,
    }
    for k in sorted(hit1):
        metrics[f"hit_at_{k}_within_1s"] = float(np.mean(hit1[k])) if hit1[k] else None
        metrics[f"hit_at_{k}_within_3s"] = float(np.mean(hit3[k])) if hit3[k] else None
    return {"metrics": metrics, "top1_rows": rows}


def _add_diverse_rank(cand: pd.DataFrame, score_col: str, id_col: str, min_gap_s: float) -> pd.Series:
    ranks = pd.Series(index=cand.index, dtype=int)
    for _, g in cand.groupby(id_col):
        remaining = list(g.sort_values(score_col, ascending=False).index)
        selected: list[int] = []
        while remaining:
            chosen = None
            for idx in remaining:
                start = float(cand.loc[idx, "candidate_start_rel_s"]) if "candidate_start_rel_s" in cand.columns else float(cand.loc[idx, "candidate_rank"])
                if all(abs(start - float(cand.loc[j, "candidate_start_rel_s"])) >= float(min_gap_s) for j in selected):
                    chosen = idx
                    break
            if chosen is None:
                chosen = remaining[0]
            selected.append(chosen)
            remaining.remove(chosen)
        for r, idx in enumerate(selected, start=1):
            ranks.loc[idx] = r
    return ranks.astype(int)


def _score_model(model, imputer: SimpleImputer, features: list[str], df: pd.DataFrame) -> np.ndarray:
    x = imputer.transform(df[features])
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x)[:, 1], dtype=float)
    raw = np.asarray(model.decision_function(x), dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(raw, -60, 60)))



def _make_ranker(args: argparse.Namespace):
    model_name = str(args.model)
    seed = int(args.random_state)
    if model_name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError("--model lightgbm requires `pip install lightgbm`") from exc
        return LGBMClassifier(
            n_estimators=max(int(args.max_iter), 500),
            num_leaves=int(args.max_leaf_nodes),
            learning_rate=float(args.learning_rate),
            feature_fraction=0.90,
            bagging_fraction=0.90,
            bagging_freq=5,
            min_child_samples=8,
            reg_lambda=max(float(args.l2_regularization), 1e-6),
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
    if model_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError("--model xgboost requires `pip install xgboost`") from exc
        return XGBClassifier(
            n_estimators=max(int(args.max_iter), 500),
            max_depth=4,
            learning_rate=float(args.learning_rate),
            subsample=0.90,
            colsample_bytree=0.90,
            reg_lambda=max(float(args.l2_regularization), 1e-6),
            tree_method="hist",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
            use_label_encoder=False,
        )
    if model_name == "catboost":
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise ImportError("--model catboost requires `pip install catboost`") from exc
        return CatBoostClassifier(
            iterations=max(int(args.max_iter), 500),
            depth=5,
            learning_rate=float(args.learning_rate),
            l2_leaf_reg=max(float(args.l2_regularization), 1e-6),
            loss_function="Logloss",
            auto_class_weights="Balanced",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        )
    return HistGradientBoostingClassifier(
        max_iter=int(args.max_iter),
        learning_rate=float(args.learning_rate),
        max_leaf_nodes=int(args.max_leaf_nodes),
        l2_regularization=float(args.l2_regularization),
        random_state=seed,
    )

def run(args: argparse.Namespace) -> None:
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.features, low_memory=False)
    id_col = str(args.id_column)
    if id_col not in df.columns:
        raise ValueError(f"id column {id_col!r} not found")
    cand = _candidate_rows(df, id_col=id_col, window=args.window, max_candidate_rank=args.max_candidate_rank)
    features = _feature_columns(cand, feature_mode=args.feature_mode, include_wallet_vendor=bool(args.include_wallet_vendor))
    labels = _session_labels(cand, id_col)
    train_ids, test_ids = train_test_split(
        labels[id_col].values,
        test_size=float(args.test_size),
        random_state=int(args.random_state),
        stratify=labels["y"].values,
    )
    train = cand[cand[id_col].isin(set(train_ids))].copy()
    test = cand[cand[id_col].isin(set(test_ids))].copy()
    y_train = pd.to_numeric(train["candidate_hit_within_3s"], errors="coerce").fillna(0).astype(int).values
    y_test = pd.to_numeric(test["candidate_hit_within_3s"], errors="coerce").fillna(0).astype(int).values
    if len(np.unique(y_train)) < 2:
        raise ValueError("candidate ranker training requires positive and negative onset candidates")
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train[features])
    model = _make_ranker(args)
    model.fit(x_train, y_train, sample_weight=_balanced_weights(y_train))
    train_score = _score_model(model, imputer, features, train)
    test_score = _score_model(model, imputer, features, test)
    train["learned_onset_score"] = train_score
    test["learned_onset_score"] = test_score
    latency_penalty = max(0.0, float(args.latency_penalty_per_s))
    for part in (train, test):
        p = np.clip(part["learned_onset_score"].astype(float).values, 1e-9, 1 - 1e-9)
        part["learned_onset_uncertainty_entropy"] = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
        delay = pd.to_numeric(part.get("candidate_decision_delay_s", pd.Series([0.0] * len(part), index=part.index)), errors="coerce").fillna(0.0)
        part["learned_onset_latency_adjusted_score"] = np.clip(
            part["learned_onset_score"].astype(float).values - latency_penalty * delay.astype(float).values,
            0.0,
            1.0,
        )
        part["learned_candidate_rank"] = part.groupby(id_col)["learned_onset_score"].rank(ascending=False, method="first").astype(int)
        part["learned_diverse_rank"] = _add_diverse_rank(part, "learned_onset_score", id_col, float(args.diversity_min_gap_s))
        part["learned_diverse_rank_score"] = -part["learned_diverse_rank"].astype(float)
        part["learned_latency_candidate_rank"] = part.groupby(id_col)["learned_onset_latency_adjusted_score"].rank(ascending=False, method="first").astype(int)
        part["learned_latency_diverse_rank"] = _add_diverse_rank(part, "learned_onset_latency_adjusted_score", id_col, float(args.diversity_min_gap_s))
        part["learned_latency_diverse_rank_score"] = -part["learned_latency_diverse_rank"].astype(float)

    baseline = test.copy()
    baseline["baseline_rank_score"] = -pd.to_numeric(baseline["candidate_rank"], errors="coerce").astype(float)
    base_rank = _rank_metrics(baseline, id_col, "baseline_rank_score")
    learned_rank = _rank_metrics(test, id_col, "learned_onset_score")
    diverse_rank = _rank_metrics(test, id_col, "learned_diverse_rank_score")
    latency_rank = _rank_metrics(test, id_col, "learned_onset_latency_adjusted_score")
    latency_diverse_rank = _rank_metrics(test, id_col, "learned_latency_diverse_rank_score")
    cls_metrics = {}
    if len(np.unique(y_test)) == 2:
        cls_metrics["candidate_auc"] = float(roc_auc_score(y_test, test_score))
        cls_metrics["candidate_auprc"] = float(average_precision_score(y_test, test_score))

    train.to_csv(out_dir / "candidate_ranker_train_scores.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    test.to_csv(out_dir / "candidate_ranker_test_scores.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    pd.DataFrame(base_rank["top1_rows"]).to_csv(out_dir / "baseline_top1_candidates.csv", index=False)
    pd.DataFrame(learned_rank["top1_rows"]).to_csv(out_dir / "learned_top1_candidates.csv", index=False)
    pd.DataFrame({"feature": features}).to_csv(out_dir / "selected_features.csv", index=False)
    joblib.dump({
        "model": model,
        "imputer": imputer,
        "features": features,
        "id_column": id_col,
        "feature_mode": args.feature_mode,
        "window": args.window,
        "include_wallet_vendor": bool(args.include_wallet_vendor),
        "latency_penalty_per_s": latency_penalty,
    }, out_dir / "candidate_onset_ranker.joblib")
    metrics = {
        "setup": {
            "features": args.features,
            "id_column": id_col,
            "window": args.window,
            "max_candidate_rank": int(args.max_candidate_rank),
            "feature_mode": args.feature_mode,
            "include_wallet_vendor": bool(args.include_wallet_vendor),
            "model": str(args.model),
            "diversity_min_gap_s": float(args.diversity_min_gap_s),
            "latency_penalty_per_s": latency_penalty,
            "n_candidate_rows": int(len(cand)),
            "n_train_sessions": int(len(set(train_ids))),
            "n_test_sessions": int(len(set(test_ids))),
            "n_train_rows": int(len(train)),
            "n_test_rows": int(len(test)),
            "n_features": int(len(features)),
            "positive_candidate_rate_train": float(np.mean(y_train)),
            "positive_candidate_rate_test": float(np.mean(y_test)),
        },
        "candidate_classifier": cls_metrics,
        "baseline_original_rank": base_rank["metrics"],
        "learned_rerank": learned_rank["metrics"],
        "learned_diverse_rerank": diverse_rank["metrics"],
        "learned_latency_rerank": latency_rank["metrics"],
        "learned_latency_diverse_rerank": latency_diverse_rank["metrics"],
    }
    (out_dir / "candidate_onset_ranker_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"wrote {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a gateway-only candidate onset learning-to-rank model from oracle labels.")
    parser.add_argument("--features", required=True)
    parser.add_argument("--model", choices=["lightgbm", "xgboost", "catboost", "hist_gradient_boosting"], default="lightgbm")
    parser.add_argument("--out-dir", default="outputs/candidate_onset_ranker")
    parser.add_argument("--id-column", choices=["sample_id", "zip_path"], default="zip_path")
    parser.add_argument("--window", default="adaptive")
    parser.add_argument("--max-candidate-rank", type=int, default=8)
    parser.add_argument("--feature-mode", choices=["candidate_only", "candidate_plus_window"], default="candidate_plus_window")
    parser.add_argument("--include-wallet-vendor", action="store_true")
    parser.add_argument("--diversity-min-gap-s", type=float, default=8.0)
    parser.add_argument("--latency-penalty-per-s", type=float, default=0.02, help="Subtract this value times candidate_decision_delay_s when producing latency-adjusted learned ranks.")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--l2-regularization", type=float, default=0.05)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

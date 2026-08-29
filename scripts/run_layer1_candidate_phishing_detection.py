#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import pathlib
from dataclasses import asdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from run_layer1_phishing_detection import (
    deployment_feature_audit,
    feature_columns_for_preset,
    make_model,
    read_second_layer_config,
)
from web3pcapdetector.models.phishing_detection import compute_binary_metrics, select_threshold_by_fpr
from web3pcapdetector.models.utils import label_to_binary


def _expand_csv(value: str) -> list[str]:
    return [x.strip() for x in str(value or "").replace(",", " ").split() if x.strip()]


def _sample_labels(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    cols = [id_col, "label"]
    if "domain" in df.columns:
        cols.append("domain")
    out = df[cols].drop_duplicates(id_col).copy()
    out["y"] = label_to_binary(out["label"].values).astype(int)
    return out


def _source_mask(df: pd.DataFrame, sources: set[str], max_candidate_rank: int, rank_col: str = "candidate_rank") -> pd.Series:
    src = df.get("candidate_source", pd.Series([""] * len(df), index=df.index))
    rank = pd.to_numeric(df.get("candidate_rank", pd.Series([np.nan] * len(df), index=df.index)), errors="coerce")
    deploy_rank = pd.to_numeric(df.get(rank_col, rank), errors="coerce")
    mask = pd.Series(False, index=df.index)
    if "oracle" in sources:
        mask |= rank.eq(0)
    if "jitter" in sources:
        mask |= src.str.startswith("oracle_jitter_")
    if "candidate" in sources:
        mask |= src.str.startswith("traffic_heuristic") & deploy_rank.ge(1) & deploy_rank.le(int(max_candidate_rank))
    return mask


def _candidate_eval_mask(df: pd.DataFrame, windows: set[str], max_candidate_rank: int, rank_col: str = "candidate_rank") -> pd.Series:
    src = df.get("candidate_source", pd.Series([""] * len(df), index=df.index))
    rank = pd.to_numeric(df.get("candidate_rank", pd.Series([np.nan] * len(df), index=df.index)), errors="coerce")
    deploy_rank = pd.to_numeric(df.get(rank_col, rank), errors="coerce")
    mask = src.str.startswith("traffic_heuristic") & deploy_rank.ge(1) & deploy_rank.le(int(max_candidate_rank))
    if windows and "window_name" in df.columns:
        mask &= df["window_name"].isin(windows)
    return mask


def _session_pool(scored: pd.DataFrame, threshold: float, topk_pool: int, id_col: str) -> pd.DataFrame:
    rows = []
    for session_id, g in scored.groupby(id_col):
        g = g.sort_values("layer1_phish_score", ascending=False)
        best = g.iloc[0]
        topk = g.head(max(1, int(topk_pool)))
        pooled_score = float(topk["layer1_phish_score"].mean())
        anchor_scores = []
        if "candidate_rank" in g.columns:
            for _, anchor_group in g.groupby("candidate_rank"):
                anchor_scores.append(float(anchor_group["layer1_phish_score"].astype(float).max()))
        else:
            anchor_scores = list(g["layer1_phish_score"].astype(float).values)
        anchor_top2 = pd.Series(anchor_scores).nlargest(2)
        anchor_top2_score = float(anchor_top2.mean()) if len(anchor_top2) else float("nan")
        rows.append({
            "session_uid": session_id,
            "id_column": id_col,
            "sample_id": best.get("sample_id", ""),
            "zip_path": best.get("zip_path", ""),
            "label": best.get("label", ""),
            "domain": best.get("domain", ""),
            "n_candidate_windows_scored": int(len(g)),
            "max_score": float(best["layer1_phish_score"]),
            "topk_mean_score": pooled_score,
            "anchor_max_top2_mean_score": anchor_top2_score,
            "max_pred_label": "phishing" if float(best["layer1_phish_score"]) >= threshold else "benign",
            "topk_mean_pred_label": "phishing" if pooled_score >= threshold else "benign",
            "anchor_max_top2_mean_pred_label": "phishing" if anchor_top2_score >= threshold else "benign",
            "best_window_name": best.get("window_name", ""),
            "best_candidate_rank": int(best.get("candidate_rank", -1)) if pd.notna(best.get("candidate_rank", np.nan)) else -1,
            "best_candidate_start_rel_s": float(best.get("candidate_start_rel_s", np.nan)),
            "best_candidate_anchor_error_s": float(best.get("candidate_anchor_error_s", np.nan)),
            "best_window_latency_vs_oracle_s": float(best.get("window_latency_vs_oracle_s", np.nan)),
        })
    return pd.DataFrame(rows)


def _threshold_score_column(mode: str) -> str:
    mapping = {
        "max": "max_score",
        "topk_mean": "topk_mean_score",
        "anchor_top2": "anchor_max_top2_mean_score",
    }
    if mode not in mapping:
        raise ValueError(f"unknown threshold calibration score: {mode}")
    return mapping[mode]


def _attach_scores(df: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    out = pd.concat([df.reset_index(drop=True), pred.reset_index(drop=True)], axis=1)
    out["layer1_phish_score"] = out["second_phish_score"]
    out["layer1_pred_label"] = out["second_pred_label"]
    out["layer1_threshold"] = out["second_threshold"]
    return out


def _training_sample_weights(df: pd.DataFrame, mode: str) -> np.ndarray | None:
    mode = str(mode or "none").lower()
    if mode == "none":
        return None
    weights = np.ones(len(df), dtype=float)
    src = df.get("candidate_source", pd.Series([""] * len(df), index=df.index)).astype(str)
    rank = pd.to_numeric(df.get("candidate_rank", pd.Series([np.nan] * len(df), index=df.index)), errors="coerce")
    abs_err = pd.to_numeric(df.get("candidate_abs_anchor_error_s", pd.Series([np.nan] * len(df), index=df.index)), errors="coerce")
    is_oracle = rank.eq(0)
    is_jitter = src.str.startswith("oracle_jitter_")
    is_candidate = src.str.startswith("traffic_heuristic")
    if mode == "session_normalized_balanced":
        # Weighting frozen by the causal all-onset component experiment: each
        # session first receives unit total mass regardless of its number of
        # causal candidate/window rows; the two phishing classes then receive
        # equal optimizer mass.  There is intentionally no within-session rank
        # preference (that is a different, historical weighting mode).
        session_column = next(
            (name for name in ("zip_path", "sample_id", "session_id") if name in df.columns),
            None,
        )
        if session_column is None:
            raise ValueError(
                "session_normalized_balanced requires zip_path, sample_id, or session_id"
            )
        if "label" not in df.columns:
            raise ValueError("session_normalized_balanced requires label")
        session_keys = df[session_column].astype(str)
        row_counts = session_keys.groupby(session_keys, sort=False).transform("size").to_numpy(dtype=float)
        weights = 1.0 / np.maximum(row_counts, 1.0)
        target = label_to_binary(df["label"].values).astype(int)
        classes = np.unique(target)
        if set(classes.tolist()) != {0, 1}:
            raise ValueError(
                "session_normalized_balanced requires both benign and phishing rows"
            )
        for value in (0, 1):
            mask = target == value
            class_mass = float(weights[mask].sum())
            if class_mass <= 0.0:
                raise ValueError(
                    f"session_normalized_balanced has zero mass for class {value}"
                )
            weights[mask] *= 0.5 / class_mass
        weights *= float(len(weights)) / max(float(weights.sum()), 1e-12)
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
            raise ValueError("session_normalized_balanced produced invalid weights")
        return weights
    if mode in {
        "rank", "rank_position", "position", "rank_session_normalized",
    }:
        weights[is_candidate.values] = (1.0 + 1.0 / np.sqrt(np.maximum(rank[is_candidate].astype(float).values, 1.0)))
        weights[is_oracle.values] = 1.35
        weights[is_jitter.values] = 1.10
        if mode == "rank_session_normalized":
            # Changing max_candidate_rank must not silently give every session
            # more total optimizer mass.  Keep the familiar within-session
            # rank preference, then make every session contribute the same
            # total mass and finally restore mean row weight to one.
            session_column = next(
                (name for name in ("zip_path", "sample_id", "session_id") if name in df.columns),
                None,
            )
            if session_column is None:
                raise ValueError(
                    "rank_session_normalized requires zip_path, sample_id, or session_id"
                )
            session_keys = df[session_column].astype(str)
            group_mass = pd.Series(weights, index=df.index).groupby(
                session_keys, sort=False
            ).transform("sum").to_numpy(dtype=float)
            weights = weights / np.maximum(group_mass, 1e-12)
            weights *= float(len(weights)) / max(float(weights.sum()), 1e-12)
    elif mode in {"oracle_distance", "rank_oracle_distance"}:
        close = np.exp(-np.nan_to_num(abs_err.astype(float).values, nan=8.0) / 6.0)
        weights *= 0.75 + close
        weights[is_candidate.values] *= (1.0 + 0.5 / np.sqrt(np.maximum(rank[is_candidate].astype(float).values, 1.0)))
        weights[is_oracle.values] = 1.50
        weights[is_jitter.values] = np.maximum(weights[is_jitter.values], 1.10)
    else:
        raise ValueError(f"unknown sample weighting mode: {mode}")
    return np.clip(weights, 0.25, 3.0)


def _plot(session_df: pd.DataFrame, max_metrics: dict, pooled_metrics: dict, anchor_metrics: dict, out_path: pathlib.Path) -> None:
    y = label_to_binary(session_df["label"].values).astype(int)
    max_scores = session_df["max_score"].astype(float).values
    pooled_scores = session_df["topk_mean_score"].astype(float).values
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(max_scores[y == 0], bins=24, alpha=0.65, label="benign max", color="#4E79A7")
    axes[0].hist(max_scores[y == 1], bins=24, alpha=0.65, label="phishing max", color="#E15759")
    axes[0].hist(pooled_scores[y == 0], bins=24, alpha=0.35, label="benign top-k mean", color="#76B7B2")
    axes[0].hist(pooled_scores[y == 1], bins=24, alpha=0.35, label="phishing top-k mean", color="#F28E2B")
    axes[0].set_title("Candidate Session Scores")
    axes[0].set_xlabel("score")
    axes[0].set_ylabel("sessions")
    axes[0].legend(frameon=False, fontsize=8)

    keys = ["precision", "recall", "fpr", "f1"]
    x = np.arange(len(keys))
    axes[1].bar(x - 0.25, [max_metrics.get(k, 0.0) for k in keys], width=0.25, label="max", color="#59A14F")
    axes[1].bar(x, [pooled_metrics.get(k, 0.0) for k in keys], width=0.25, label="window top-k", color="#B07AA1")
    axes[1].bar(x + 0.25, [anchor_metrics.get(k, 0.0) for k in keys], width=0.25, label="anchor top-2", color="#F28E2B")
    axes[1].set_xticks(x, keys)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("Session-Level Metrics")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)



def _resolve_candidate_rank_column(df: pd.DataFrame, requested: str) -> str:
    requested = str(requested or "auto")
    if requested != "auto":
        if requested not in df.columns:
            raise ValueError(f"rank column {requested!r} not found")
        return requested
    for col in ["learned_latency_diverse_rank", "learned_latency_candidate_rank", "learned_diverse_rank", "learned_candidate_rank", "candidate_rank"]:
        if col in df.columns:
            return col
    return "candidate_rank"

def run(args: argparse.Namespace) -> None:
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.features, low_memory=False)
    id_col = str(args.id_column)
    if id_col not in df.columns:
        raise ValueError(f"id column {id_col!r} not found in features")
    rank_col = _resolve_candidate_rank_column(df, args.candidate_rank_column)
    train_windows = set(_expand_csv(args.train_windows))
    decision_windows = set(_expand_csv(args.decision_windows))
    if train_windows and "window_name" in df.columns:
        df_trainable = df[df["window_name"].astype(str).isin(train_windows)].copy()
    else:
        df_trainable = df.copy()

    labels = _sample_labels(df_trainable, id_col)
    if len(labels) < 2 or labels["y"].nunique() < 2:
        raise ValueError("candidate Layer 1 training requires both phishing and benign samples")
    train_ids, test_ids = train_test_split(
        labels[id_col].values,
        test_size=float(args.test_size),
        random_state=int(args.random_state),
        stratify=labels["y"].values,
    )
    train_id_set = set(train_ids)
    test_id_set = set(test_ids)

    sources = set(_expand_csv(args.train_sources))
    train_mask = df_trainable[id_col].isin(train_id_set) & _source_mask(df_trainable, sources, args.max_candidate_rank, rank_col=rank_col)
    train_df = df_trainable[train_mask].copy()
    if len(train_df) == 0:
        raise ValueError("no training rows matched train_sources/train_windows")
    if label_to_binary(train_df["label"].values).astype(int).max() == 0 or label_to_binary(train_df["label"].values).astype(int).min() == 1:
        raise ValueError("training rows must contain both classes")

    eval_mask = df[id_col].isin(test_id_set) & _candidate_eval_mask(df, decision_windows, args.max_candidate_rank, rank_col=rank_col)
    eval_df = df[eval_mask].copy()
    if len(eval_df) == 0:
        raise ValueError("no heuristic candidate rows matched decision_windows for holdout evaluation")

    train_df.to_csv(out_dir / "train_candidate_windows.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    eval_df.to_csv(out_dir / "test_candidate_windows.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    config = read_second_layer_config(args.config, args)
    config.window_name = None
    selected_features = feature_columns_for_preset(train_df, args.feature_preset)
    feature_audit = deployment_feature_audit(selected_features)
    pd.DataFrame({"feature": selected_features}).to_csv(out_dir / "selected_features.csv", index=False)
    model = make_model(config)
    sample_weight = _training_sample_weights(train_df, args.sample_weighting)
    if sample_weight is not None:
        train_df.assign(training_sample_weight=sample_weight).to_csv(out_dir / "train_candidate_windows_weighted.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    model.fit(train_df, feature_columns=selected_features, sample_weight=sample_weight)
    if bool(args.recalibrate_threshold_on_train_sessions):
        train_pred = _attach_scores(train_df, model.predict(train_df))
        train_sessions = _session_pool(train_pred, float(train_pred["layer1_threshold"].iloc[0]), args.topk_pool, id_col)
        threshold_score_col = _threshold_score_column(args.threshold_calibration_score)
        model.threshold_ = select_threshold_by_fpr(
            label_to_binary(train_sessions["label"].values),
            train_sessions[threshold_score_col].astype(float).values,
            float(config.threshold_fpr),
        )
    model_dir = out_dir / "phishing_detection_model"
    model.save(model_dir)

    scored = _attach_scores(eval_df, model.predict(eval_df))
    scored.to_csv(out_dir / "candidate_window_scores.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    threshold = float(scored["layer1_threshold"].iloc[0])
    session_df = _session_pool(scored, threshold, args.topk_pool, id_col)
    session_df.to_csv(out_dir / "session_predictions.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    y_session = label_to_binary(session_df["label"].values)
    max_metrics = compute_binary_metrics(y_session, session_df["max_score"].astype(float).values, threshold)
    max_metrics["f1"] = 2 * max_metrics["precision"] * max_metrics["recall"] / max(max_metrics["precision"] + max_metrics["recall"], 1e-12)
    pooled_metrics = compute_binary_metrics(y_session, session_df["topk_mean_score"].astype(float).values, threshold)
    pooled_metrics["f1"] = 2 * pooled_metrics["precision"] * pooled_metrics["recall"] / max(pooled_metrics["precision"] + pooled_metrics["recall"], 1e-12)
    anchor_metrics = compute_binary_metrics(y_session, session_df["anchor_max_top2_mean_score"].astype(float).values, threshold)
    anchor_metrics["f1"] = 2 * anchor_metrics["precision"] * anchor_metrics["recall"] / max(anchor_metrics["precision"] + anchor_metrics["recall"], 1e-12)
    metrics = {
        "setup": {
            "task": "layer1_candidate_window_phishing_detection",
            "features": args.features,
            "train_windows": sorted(train_windows),
            "decision_windows": sorted(decision_windows),
            "train_sources": sorted(sources),
            "id_column": id_col,
            "n_unique_sample_id": int(df["sample_id"].nunique()) if "sample_id" in df.columns else None,
            "n_unique_zip_path": int(df["zip_path"].nunique()) if "zip_path" in df.columns else None,
            "max_candidate_rank": int(args.max_candidate_rank),
            "candidate_rank_column": rank_col,
            "topk_pool": int(args.topk_pool),
            "model": args.model,
            "feature_preset": args.feature_preset,
            "sample_weighting": args.sample_weighting,
            "threshold_calibration_score": args.threshold_calibration_score,
            "threshold_fpr": float(config.threshold_fpr),
            "random_state": int(args.random_state),
            "n_train_samples": int(len(train_id_set)),
            "n_test_samples": int(len(test_id_set)),
            "n_train_rows": int(len(train_df)),
            "n_eval_rows": int(len(eval_df)),
            "n_eval_sessions": int(len(session_df)),
            "n_features": int(len(selected_features)),
            "feature_audit": feature_audit,
            "model_config": asdict(config),
        },
        "session_max": max_metrics,
        "session_topk_mean": pooled_metrics,
        "session_anchor_max_top2_mean": anchor_metrics,
    }
    (out_dir / "layer1_candidate_phishing_detection_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _plot(session_df, max_metrics, pooled_metrics, anchor_metrics, out_dir / "layer1_candidate_phishing_detection_results.png")
    print(json.dumps(metrics, indent=2))
    print(f"wrote {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Layer 1 on oracle/jitter/candidate windows and evaluate candidate top-k session pooling.")
    parser.add_argument("--features", required=True)
    parser.add_argument("--out-dir", default="outputs/experiment_candidate_layer1/layer1_candidate_phishing_detection")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--train-windows", default="adaptive,w3,w6,w10")
    parser.add_argument("--decision-windows", default="adaptive,w3,w6,w10")
    parser.add_argument("--window", default="w3", help=argparse.SUPPRESS)
    parser.add_argument("--train-sources", default="oracle,jitter,candidate")
    parser.add_argument("--id-column", choices=["sample_id", "zip_path"], default="sample_id", help="Column used as the unique session id for split and session pooling.")
    parser.add_argument("--candidate-rank-column", default="auto", help="candidate_rank, learned_candidate_rank, learned_diverse_rank, learned_latency_candidate_rank, learned_latency_diverse_rank, or auto. Use outputs from apply_candidate_onset_ranker.py to activate the learned onset reranker.")
    parser.add_argument("--max-candidate-rank", type=int, default=8)
    parser.add_argument("--topk-pool", type=int, default=3)
    parser.add_argument(
        "--model",
        choices=[
            "mlp",
            "logistic",
            "extra_trees",
            "random_forest",
            "hist_gradient_boosting",
            "lightgbm",
            "xgboost",
            "catboost",
            "gbdt_ensemble",
        ],
        default="gbdt_ensemble",
    )
    parser.add_argument("--feature-preset", choices=["all", "no_quality", "no_wallet_vendor", "no_wallet_vendor_no_quality", "wallet_agnostic", "nuisance_agnostic", "adaptive_core", "kit_l1_core", "core_no_quality", "early_core"], default="kit_l1_core")
    parser.add_argument("--sample-weighting", choices=["none", "rank", "oracle_distance", "rank_oracle_distance"], default="none")
    parser.add_argument("--threshold-calibration-score", choices=["max", "topk_mean", "anchor_top2"], default="max", help="Session score used when --recalibrate-threshold-on-train-sessions is enabled.")
    parser.add_argument("--threshold-fpr", type=float, default=None)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--recalibrate-threshold-on-train-sessions", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

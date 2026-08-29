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

from web3pcapdetector.config import load_config
from web3pcapdetector.models.phishing_detection import (
    GBDTEnsembleBinaryClassifier,
    LogisticBinaryClassifier,
    MLPBinaryClassifier,
    SecondLayerConfig,
    SklearnBinaryClassifier,
    compute_binary_metrics,
)
from web3pcapdetector.models.utils import label_to_binary, numeric_feature_columns, select_window


OUTPUT_COLUMNS = [
    "sample_id",
    "label",
    "domain",
    "url",
    "status",
    "zip_path",
    "window_name",
    "layer1_phish_score",
    "layer1_pred_label",
    "layer1_threshold",
]


DEPLOYMENT_UNAVAILABLE_EXACT_FEATURES = {
    # Window/capture availability or deterministic window design fields, not
    # encrypted traffic observed within the decision window.
    "window_s",
    "coverage_s",
    "base_window_s",
}


DEPLOYMENT_UNAVAILABLE_PREFIXES = (
    # Quality/capture diagnostics include post-window capture duration and other
    # dataset QA signals. They are useful for audits, not deployment decisions.
    "quality_",
    # Visibility sweep controls and diagnostics must never become model inputs.
    "degradation_",
)


def is_deployment_unavailable_feature(column: str) -> bool:
    return column in DEPLOYMENT_UNAVAILABLE_EXACT_FEATURES or column.startswith(DEPLOYMENT_UNAVAILABLE_PREFIXES)


def deployment_feature_audit(columns: list[str]) -> dict[str, list[str]]:
    unavailable = [c for c in columns if is_deployment_unavailable_feature(c)]
    wallet_vendor = [c for c in columns if "wallet_vendor" in c]
    nuisance_roles = [c for c in columns if any(x in c for x in ["wallet_vendor", "analytics_ads", "third_party_static", "software_update"])]
    return {
        "deployment_unavailable": unavailable,
        "wallet_vendor": wallet_vendor,
        "nuisance_role_features": nuisance_roles,
    }


def _without_deployment_unavailable(cols: list[str]) -> list[str]:
    return [c for c in cols if not is_deployment_unavailable_feature(c)]


def _without_wallet_vendor(cols: list[str]) -> list[str]:
    return [c for c in cols if "wallet_vendor" not in c]


def _without_nuisance_roles(cols: list[str]) -> list[str]:
    blocked = ("wallet_vendor", "analytics_ads", "third_party_static", "software_update")
    return [c for c in cols if not any(token in c for token in blocked)]


def read_second_layer_config(config_path: str | None, args: argparse.Namespace) -> SecondLayerConfig:
    cfg = load_config(config_path)
    c = cfg.get("model", {}).get("second_layer", {}) if "model" in cfg else cfg
    return SecondLayerConfig(
        model=str(args.model),
        threshold_fpr=float(args.threshold_fpr if args.threshold_fpr is not None else c.get("threshold_fpr", 0.05)),
        hidden_dims=tuple(c.get("hidden_dims", [128, 64])),
        dropout=float(c.get("dropout", 0.20)),
        epochs=int(args.epochs if args.epochs is not None else c.get("epochs", 80)),
        patience=int(c.get("patience", 12)),
        learning_rate=float(c.get("learning_rate", 1e-3)),
        weight_decay=float(c.get("weight_decay", 1e-4)),
        validation_fraction=float(c.get("validation_fraction", 0.25)),
        random_state=int(args.random_state),
        window_name=str(args.window),
    )


def make_model(config: SecondLayerConfig):
    if config.model == "logistic":
        return LogisticBinaryClassifier(config)
    if config.model == "gbdt_ensemble":
        return GBDTEnsembleBinaryClassifier(config)
    if config.model in {"extra_trees", "random_forest", "hist_gradient_boosting", "lightgbm", "xgboost", "catboost", "gbdt_ensemble"}:
        return SklearnBinaryClassifier(config)
    return MLPBinaryClassifier(config)


def feature_columns_for_preset(df: pd.DataFrame, preset: str) -> list[str]:
    cols = numeric_feature_columns(df, window_name=None, extra_exclude=["is_phishing"])
    if preset.endswith("_no_rpc") and preset != "kit_l1_core_transport_invariant_stackclean_no_rpc":
        base = feature_columns_for_preset(df, preset.removesuffix("_no_rpc"))
        return [column for column in base if "rpc" not in column.casefold()]
    if preset == "kit_l1_core_shape_only_env_hardened":
        # Strict ECH+DoH-safe drift candidate: remove absolute RTT-scale and
        # packetization features from the name-free route, then retain only
        # self-normalized timing/duration ratios as the environment-resistant
        # extension.  The ratios are materialized by --add-env-shape-ratios.
        from screen_env_robust_sidepath import PKT_RE, TIMING_RE

        core = feature_columns_for_preset(df, "kit_l1_core_shape_only")
        kept = [c for c in core if not TIMING_RE.search(c) and not PKT_RE.search(c)]
        extras = [
            c for c in df.columns
            if c.startswith(("iat_shape_", "dur_shape_")) or c == "idle_over_active"
        ]
        return kept + [c for c in extras if c not in kept]
    if preset == "kit_l1_core_transport_invariant_stackclean_shape_ratios":
        core = feature_columns_for_preset(
            df, "kit_l1_core_transport_invariant_stackclean"
        )
        ratios = [
            c for c in df.columns
            if c.startswith(("iat_shape_", "dur_shape_")) or c == "idle_over_active"
        ]
        return core + [c for c in ratios if c not in core]
    if preset == "kit_l1_core_shape_only_shape_ratios":
        # ECH + DoH-safe extension: these ratios use only packet timing and
        # flow-duration fields, never SNI/DNS names, host identity, or roles.
        core = feature_columns_for_preset(df, "kit_l1_core_shape_only")
        ratios = [
            c for c in df.columns
            if c.startswith(("iat_shape_", "dur_shape_")) or c == "idle_over_active"
        ]
        return core + [c for c in ratios if c not in core]
    if preset == "kit_l1_core_transport_invariant_stackclean_no_rpc":
        return [
            column
            for column in feature_columns_for_preset(
                df, "kit_l1_core_transport_invariant_stackclean"
            )
            if "rpc" not in column.casefold()
        ]
    if preset == "all":
        return cols
    if preset == "no_quality":
        return _without_deployment_unavailable(cols)
    if preset == "no_wallet_vendor":
        return _without_wallet_vendor(cols)
    if preset == "no_wallet_vendor_no_quality":
        return _without_deployment_unavailable(_without_wallet_vendor(cols))
    if preset == "wallet_agnostic":
        return [c for c in cols if "wallet_vendor" not in c and "walletconnect" not in c]
    if preset == "nuisance_agnostic":
        return _without_deployment_unavailable(_without_nuisance_roles(cols))
    if preset == "adaptive_core":
        prefixes = ("kitshape_", "view_", "seg_", "base_", "role_", "proto_", "flow_", "rpc_", "phase_", "candidate_")
        return [
            c for c in cols
            if c.startswith(prefixes)
            and not is_deployment_unavailable_feature(c)
            and not any(token in c for token in ("wallet_vendor", "analytics_ads", "third_party_static", "software_update"))
            and not c.endswith("_epoch")
        ]
    if preset == "kit_l1_core":
        # Flow-shape first. Role groups are retained as weak auxiliaries, while
        # raw role/vendor/background features and timing/oracle leakage are
        # excluded so ECH/DoH settings still have a strong core model.
        prefixes = (
            "kitshape_", "view_", "seg_", "base_", "flow_", "proto_", "phase_",
            "candidate_flow_shape_", "candidate_burst_", "candidate_new_flow_count_",
            "candidate_new_host_count_", "candidate_bytes_", "candidate_n_pkts_",
            "role_group_", "role_entropy", "role_dominant", "role_stage_",
        )
        blocked = (
            "wallet_vendor", "wallet_disturbance", "analytics_ads",
            "third_party_static", "software_update", "nuisance", "candidate_score",
            "candidate_rank", "candidate_start_rel", "candidate_emit_rel",
            "candidate_role_trigger", "candidate_role_aux", "candidate_past_wallet",
            "candidate_walletconnect_bytes", "candidate_rpc_bytes", "candidate_identity_bytes",
            "candidate_payment_bytes", "candidate_messaging_bytes", "candidate_form_backend_bytes",
            "candidate_cloud_api_bytes", "candidate_file_storage_bytes", "candidate_first_party_bytes",
            "anchor_error", "hit_within", "oracle",
        )
        return [
            c for c in cols
            if c.startswith(prefixes)
            and not is_deployment_unavailable_feature(c)
            and not any(token in c for token in blocked)
            and not c.endswith("_epoch")
        ]
    if preset == "kit_l1_core_shape_only":
        # Strict ECH + encrypted-DNS mode on top of the paper's transport-
        # invariant stackclean route.  This means generic TCP/UDP443 totals are
        # still excluded: not because ECH/DoH hides them, but because the
        # recommended stackclean route removes transport-regime features that
        # encode the capture/network path.  In addition, remove every field that
        # requires host identity or the role taxonomy derived from SNI/DNS/HAR.
        core = feature_columns_for_preset(df, "kit_l1_core_transport_invariant_stackclean")
        name_free_phase_suffixes = (
            "_burst_sz_max",
            "_bytes_total",
            "_large_pkt_frac",
            "_n_pkts",
        )
        kept: list[str] = []
        for c in core:
            if c.startswith("role_") or "host" in c:
                continue
            if c.startswith("phase_"):
                if c == "phase_activity_decay_30_90_over_0_30" or c.endswith(name_free_phase_suffixes):
                    kept.append(c)
                continue
            kept.append(c)
        return kept
    if preset == "kit_l1_core_transport_invariant":
        # kit_l1_core minus every explicit transport feature (proto_tcp_* /
        # proto_udp443_*). Whether a flow rides QUIC (UDP443) or TCP+TLS is
        # decided by the network path (proxy UDP forwarding), not the kit, so
        # these columns encode capture environment rather than kit behaviour.
        core = feature_columns_for_preset(df, "kit_l1_core")
        return [c for c in core if not c.startswith("proto_")]
    if preset == "kit_l1_core_transport_invariant_stackclean":
        # transport_invariant, additionally minus the three packet-size MIN
        # features. Measured on matched content (TUN-captured vs real-stack
        # browser benign) they separate the capture STACK almost perfectly
        # (AUC 0.98-0.99: the min frame is a header-only ACK synthesized by the
        # local mihomo/Clash-TUN stack) while carrying no portable kit signal;
        # removing them IMPROVES temporal-forward recall (+4.9pp matched,
        # 0.634->0.683) because the model was using them as a capture-environment
        # shortcut. Only these three MINs qualify -- base_iat_min (stack AUC 0.08)
        # and view_*_min carry real kit signal, so they stay
        # (see scripts/screen_capture_stack_free.py + the sz-min ablation).
        capture_stack_mins = {"base_sz_all_min", "base_sz_up_min", "base_sz_down_min"}
        ti = feature_columns_for_preset(df, "kit_l1_core_transport_invariant")
        return [c for c in ti if c not in capture_stack_mins]
    if preset == "paper_ablate_no_packet_shape":
        # Remove local size/direction sequence statistics while retaining burst,
        # temporal, flow, session, and coarse endpoint-role evidence.
        core = feature_columns_for_preset(df, "kit_l1_core_transport_invariant_stackclean")
        def is_local_packet_shape(c: str) -> bool:
            if c.startswith(("view_len_", "view_prefix_", "view_cross_dir_", "view_cross_len_")):
                return True
            if c.startswith(("seg_len_", "seg_dir_")):
                return True
            if c.startswith("base_"):
                return (
                    c.startswith(("base_sz_", "base_bytes_", "base_pkts_"))
                    or c in {
                        "base_large_pkt_frac", "base_small_pkt_frac",
                        "base_ud_byte_ratio", "base_ud_pkt_ratio",
                    }
                )
            return False
        return [c for c in core if not is_local_packet_shape(c)]
    if preset == "paper_ablate_no_temporal":
        # Remove burst, phase-evolution, idle, and relative-timing evidence while
        # retaining packet shape, cross-flow structure, and role context.
        core = feature_columns_for_preset(df, "kit_l1_core_transport_invariant_stackclean")
        temporal_tokens = ("iat", "rhythm", "idle", "active_seconds", "duration", "_dur_", "burst", "rate")
        return [
            c for c in core
            if not c.startswith(("phase_", "seg_"))
            and not any(token in c for token in temporal_tokens)
        ]
    if preset == "paper_ablate_no_crossflow":
        # Remove relations among concurrent flows/endpoints while retaining
        # within-flow packet shape and temporal evolution.
        core = feature_columns_for_preset(df, "kit_l1_core_transport_invariant_stackclean")
        crossflow_exact = {
            "base_n_flows",
            "base_n_server_ips",
            "candidate_new_flow_count_3s",
            "candidate_new_host_count_3s",
        }
        return [
            c for c in core
            if c not in crossflow_exact
            and not c.startswith(("flow_", "kitshape_flow_shape_", "candidate_flow_shape_"))
            and "concurrent" not in c
            and not (c.startswith("role_") and "_flow_frac" in c)
        ]
    if preset == "paper_local_only":
        # Macro-ablation for the paper: retain packet shape plus within-window
        # burst/temporal evolution, but remove cross-flow, endpoint-role and
        # episode-level coordination.  This is stricter than no-crossflow and
        # corresponds to the "Local-only representation" row.
        core = feature_columns_for_preset(df, "kit_l1_core_transport_invariant_stackclean")
        blocked_prefixes = (
            "flow_", "kitshape_flow_shape_", "candidate_flow_shape_",
            "role_", "candidate_new_flow_count_", "candidate_new_host_count_",
        )
        blocked_exact = {"base_n_flows", "base_n_server_ips"}
        return [
            c for c in core
            if c not in blocked_exact
            and not c.startswith(blocked_prefixes)
            and "concurrent" not in c
            and "host" not in c
        ]
    if preset == "kit_l1_env_hardened":
        # Start from stackclean so this candidate never reintroduces the three
        # proxy-stack packet-size minima already audited as shortcuts.  Then
        # remove absolute-timing (RTT-scale) and packetization features and add
        # self-normalized per-session shape ratios.
        from screen_env_robust_sidepath import PKT_RE, TIMING_RE  # side-path only

        core = feature_columns_for_preset(
            df, "kit_l1_core_transport_invariant_stackclean"
        )
        kept = [c for c in core if not TIMING_RE.search(c) and not PKT_RE.search(c)]
        extras = [c for c in df.columns
                  if c.startswith(("iat_shape_", "dur_shape_")) or c == "idle_over_active"]
        return kept + extras
    if preset == "core_no_quality":
        prefixes = ("base_", "role_", "proto_", "flow_", "rpc_", "phase_")
        return [
            c for c in cols
            if c.startswith(prefixes)
            and not is_deployment_unavailable_feature(c)
            and "wallet_vendor" not in c
        ]
    if preset == "early_core":
        return [
            c for c in cols
            if (
                c.startswith(("base_", "role_", "rpc_", "proto_", "flow_"))
                or c.startswith("phase_0_3")
                or c.startswith("phase_3_10")
            )
            and not is_deployment_unavailable_feature(c)
            and "wallet_vendor" not in c
        ]
    raise ValueError(f"unknown feature preset: {preset}")


def attach_layer1_columns(df: pd.DataFrame, pred: pd.DataFrame, selected_features: list[str]) -> pd.DataFrame:
    out = pd.concat([df.reset_index(drop=True), pred.reset_index(drop=True)], axis=1)
    out["layer1_phish_score"] = out["second_phish_score"]
    out["layer1_pred_label"] = out["second_pred_label"]
    out["layer1_threshold"] = out["second_threshold"]
    cols = [c for c in OUTPUT_COLUMNS if c in out.columns]
    feature_cols = [c for c in selected_features if c in out.columns and c not in cols]
    return out[cols + feature_cols]


def plot_results(test_out: pd.DataFrame, metrics: dict, out_path: pathlib.Path) -> None:
    y = label_to_binary(test_out["label"].values).astype(int)
    scores = test_out["layer1_phish_score"].astype(float).values
    benign_scores = scores[y == 0]
    phish_scores = scores[y == 1]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    axes[0].hist(benign_scores, bins=28, alpha=0.65, label="benign", color="#4E79A7")
    axes[0].hist(phish_scores, bins=28, alpha=0.65, label="phishing", color="#E15759")
    axes[0].axvline(float(metrics["threshold"]), color="#111111", linestyle="--", linewidth=1.4, label="threshold")
    axes[0].set_title("Layer 1 Phishing Score Distribution")
    axes[0].set_xlabel("phishing score")
    axes[0].set_ylabel("samples")
    axes[0].legend(frameon=False)

    keys = ["precision", "recall", "fpr", "f1"]
    vals = [float(metrics.get(k, 0.0)) for k in keys]
    colors = ["#59A14F", "#F28E2B", "#B07AA1", "#76B7B2"]
    axes[1].bar(keys, vals, color=colors)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("Layer 1 Holdout Metrics")
    for i, v in enumerate(vals):
        axes[1].text(i, v + 0.025, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.features, low_memory=False)
    dyn = select_window(df, args.window).copy()
    y = label_to_binary(dyn["label"].values).astype(int)
    if len(np.unique(y)) < 2:
        raise ValueError("Layer 1 phishing detection requires both phishing and benign labels")

    idx = np.arange(len(dyn))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=float(args.test_size),
        random_state=int(args.random_state),
        stratify=y,
    )
    train_df = dyn.iloc[train_idx].copy()
    test_df = dyn.iloc[test_idx].copy()
    train_df.to_csv(out_dir / "train_samples.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    test_df.to_csv(out_dir / "test_samples.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    config = read_second_layer_config(args.config, args)
    selected_features = feature_columns_for_preset(train_df, args.feature_preset)
    feature_audit = deployment_feature_audit(selected_features)
    pd.DataFrame({"feature": selected_features}).to_csv(out_dir / "selected_features.csv", index=False)
    model = make_model(config)
    model.fit(train_df, feature_columns=selected_features)
    model_dir = out_dir / "phishing_detection_model"
    model.save(model_dir)

    test_pred = model.predict(test_df)
    test_out = attach_layer1_columns(test_df, test_pred, selected_features)
    test_out.to_csv(out_dir / "holdout_predictions.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    all_pred = model.predict(dyn)
    all_out = attach_layer1_columns(dyn, all_pred, selected_features)
    all_out.to_csv(out_dir / "all_predictions.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    test_metrics = compute_binary_metrics(
        label_to_binary(test_df["label"].values),
        test_out["layer1_phish_score"].astype(float).values,
        float(test_out["layer1_threshold"].iloc[0]),
    )
    test_metrics["f1"] = 2 * test_metrics["precision"] * test_metrics["recall"] / max(test_metrics["precision"] + test_metrics["recall"], 1e-12)
    all_metrics = compute_binary_metrics(
        label_to_binary(dyn["label"].values),
        all_out["layer1_phish_score"].astype(float).values,
        float(all_out["layer1_threshold"].iloc[0]),
    )
    all_metrics["f1"] = 2 * all_metrics["precision"] * all_metrics["recall"] / max(all_metrics["precision"] + all_metrics["recall"], 1e-12)

    metrics = {
        "setup": {
            "task": "layer1_phishing_detection",
            "window": args.window,
            "model": args.model,
            "feature_preset": args.feature_preset,
            "n_features": len(selected_features),
            "feature_audit": feature_audit,
            "test_size": float(args.test_size),
            "threshold_fpr": float(config.threshold_fpr),
            "random_state": int(args.random_state),
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "n_all": int(len(dyn)),
            "n_train_phishing": int(label_to_binary(train_df["label"].values).sum()),
            "n_test_phishing": int(label_to_binary(test_df["label"].values).sum()),
            "model_config": asdict(config),
        },
        "holdout": test_metrics,
        "all_dynamic_samples": all_metrics,
    }
    (out_dir / "layer1_phishing_detection_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_results(test_out, test_metrics, out_dir / "layer1_phishing_detection_results.png")
    print(json.dumps(metrics, indent=2))
    print(f"wrote {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate Layer 1: supervised phishing-vs-benign detection.")
    parser.add_argument("--features", required=True)
    parser.add_argument("--out-dir", default="outputs/experiment_20260507/layer1_phishing_detection")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--window", default="dyn")
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
        ],
        default="lightgbm",
        help=(
            "Layer 1 backbone. Default switched from extra_trees to "
            "lightgbm for the low-latency deployable candidate-window route; "
            "xgboost and catboost are available for robustness checks."
        ),
    )
    parser.add_argument(
        "--feature-preset",
        choices=["all", "no_quality", "no_wallet_vendor", "no_wallet_vendor_no_quality", "wallet_agnostic", "nuisance_agnostic", "adaptive_core", "kit_l1_core", "core_no_quality", "early_core"],
        default="adaptive_core",
        help="Feature pool for Layer 1. no_wallet_vendor_no_quality is the deployment default; no_wallet_vendor is kept for historical comparability.",
    )
    parser.add_argument("--threshold-fpr", type=float, default=None)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

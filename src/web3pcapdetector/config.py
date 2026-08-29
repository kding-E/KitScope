from __future__ import annotations

import copy
import pathlib
from typing import Any, Dict, Optional

import yaml

DEFAULT_CONFIG: Dict[str, Any] = {
    "anchor": {
        # oracle: require an offline interaction-reference event.  Legacy wallet
        # click keys remain supported as one subtype of generic interaction.
        # heuristic: gateway-only near-real-time candidate starts; no UI event needed.
        # auto: use oracle if present, otherwise fallback to heuristic.
        # oracle_and_heuristic: emit oracle plus heuristic candidates for diagnostics only.
        "mode": "oracle",
        "keys": [
            "t_wallet_connect_click",
            "t_walletconnect_connect_click",
            "t_connect_click",
            "t_metamask_connect_click",
        ],
        "primary_session_key": "t_metamask_connect_click",
        "event_fallback_key": "t_metamask_connect_click",
        "allow_network_fallback": True,
        "prepad_s": 0.0,
    },
    "candidate": {
        # Near-real-time onset proposals. A candidate at t becomes available at
        # t + decision_delay_s; fixed w3 features become available at the same time.
        "scan_step_s": 0.5,
        "decision_delay_s": 3.0,
        "immediate_window_s": 3.0,
        "past_quiet_window_s": 5.0,
        "min_after_capture_start_s": 10.0,
        "min_after_first_party_s": 5.0,
        "max_scan_s": 180.0,
        "top_k": 8,
        "merge_within_s": 4.0,
        "min_score": 1.15,
        "min_total_bytes_3s": 2000,
        "min_important_bytes_3s": 3000,
        "min_trigger_bytes_3s": 1500,
        "burst_norm_bytes": 50000.0,
        "quiet_norm_bytes": 20000.0,
        "important_roles": ["unknown", "third_party_backend_or_other", "rpc_provider", "walletconnect"],
        "backend_roles": ["third_party_backend_or_other", "rpc_provider", "walletconnect"],
        "trigger_roles": ["rpc_provider", "walletconnect", "wallet_vendor"],
        "negative_roles": ["third_party_static", "analytics_ads"],
    },
    "dynamic": {
        "min_window_s": 15.0,
        "max_window_s": 90.0,
        "burst_gap_s": 0.35,
        "idle_gap_s": 5.0,
        "tail_pad_s": 0.25,
        "min_burst_packets": 4,
        "min_burst_bytes": 1024,
        "use_capture_stop_if_no_idle": True,
    },
    "features": {
        "burst_gap_s": 0.05,
        "active_bin_s": 1.0,
        "large_packet_threshold": 1000,
        "small_packet_threshold": 80,
        "phases": [[0, 3], [3, 10], [10, 30], [30, 60], [60, 90]],
        "roles": [
            "first_party_site",
            "rpc_provider",
            "wallet_vendor",
            "walletconnect",
            "third_party_static",
            "third_party_backend_or_other",
            "analytics_ads",
            "unknown",
        ],
    },
    "model": {
        "first_layer": {
            "min_cluster_samples": 2,
            "threshold_quantile": 0.95,
            "default_threshold": 4.0,
            "margin_threshold": 0.10,
            "variance_shrinkage": 0.10,
        },
        "second_layer": {
            # Default switched from extra_trees to hist_gradient_boosting on
            # 2026-05-18 after a head-to-head backbone study on the gateway
            # SNI+DNS features: HistGB strictly dominates ExtraTrees on
            # precision, recall, FPR, F1, AUROC and is ~22x faster at inference
            # (median 1.81 ms vs 43.73 ms per session on CPU).  ExtraTrees is
            # retained as a selectable backbone via --model extra_trees for
            # historical comparison.
            "model": "hist_gradient_boosting",
            "threshold_fpr": 0.05,
            "hidden_dims": [128, 64],
            "dropout": 0.20,
            "epochs": 80,
            "patience": 12,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "validation_fraction": 0.25,
            "random_state": 42,
        },
    },
}


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    if path is None:
        return copy.deepcopy(DEFAULT_CONFIG)
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    with p.open("r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    return deep_update(DEFAULT_CONFIG, user_cfg)

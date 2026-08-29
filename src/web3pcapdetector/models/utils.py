from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

META_COLUMNS = {
    "sample_key", "sample_id", "label", "domain", "url", "status", "zip_path", "anchor_mode",
    "anchor_time_epoch", "anchor_source", "anchor_prepad_s", "feature_anchor_epoch",
    "oracle_anchor_epoch", "oracle_anchor_time_epoch", "oracle_anchor_source",
    "candidate_id", "candidate_rank", "candidate_anchor_epoch", "candidate_source",
    "candidate_reason", "candidate_start_epoch", "candidate_emit_epoch", "candidate_available_epoch",
    "candidate_start_rel_s", "candidate_emit_rel_s",
    "candidate_offset_from_oracle_s", "candidate_anchor_error_s", "candidate_abs_anchor_error_s",
    "candidate_emit_latency_vs_oracle_s", "candidate_available_lag_from_oracle_s",
    "candidate_hit_within_1s", "candidate_hit_within_3s", "candidate_hit_within_5s", "candidate_hit_within_10s",
    "candidate_hit_oracle_1s", "candidate_hit_oracle_3s", "candidate_hit_oracle_5s", "candidate_hit_oracle_10s",
    "window_available_epoch", "window_available_after_candidate_s", "window_available_lag_from_oracle_s",
    "window_latency_vs_oracle_s", "heuristic_realtime_risk_score",
    "pcap_start_epoch", "pcap_end_epoch", "session_pcap_start_epoch", "session_pcap_stop_epoch",
    "capture_stop_epoch", "linktype", "local_ips", "role_source", "har_n_hosts", "har_n_server_ips",
    "post_load_guard_epoch", "post_load_guard_source", "post_load_guard_rel_s", "post_load_guard_mode",
    "first_action_epoch", "first_action_source", "first_action_rel_s",
    "traffic_clip_mode", "traffic_clip_epoch",
    "sni_n_hosts", "sni_n_server_ips", "dns_n_hosts", "dns_n_server_ips",
    "filter_suppressed_roles", "filter_suppressed_n_pkts", "filter_suppressed_bytes_total",
    "filter_suppressed_n_flows", "filter_suppressed_n_server_ips",
    "filter_suppressed_host_patterns", "filter_suppressed_host_skip_roles",
    "filter_host_suppressed_n_pkts", "filter_host_suppressed_bytes_total",
    "filter_host_suppressed_n_flows", "filter_host_suppressed_n_server_ips",
    "filter_drop_rpc_fanout_flows", "filter_rpc_fanout_window_s",
    "filter_rpc_fanout_min_flows", "filter_rpc_fanout_min_hosts",
    "filter_rpc_fanout_dropped_n_pkts", "filter_rpc_fanout_dropped_bytes_total",
    "filter_rpc_fanout_dropped_n_flows", "filter_rpc_fanout_dropped_n_server_ips",
    "degradation_hide_sni", "degradation_hide_dns",
    "degradation_sni_visible_rate", "degradation_dns_visible_rate",
    "degradation_sni_dropped_pkts", "degradation_sni_kept_pkts",
    "degradation_dns_dropped_resolvers", "degradation_dns_kept_resolvers",
    "preprocess_config_id", "preprocess_config_sha1", "preprocess_shard_id",
    "preprocess_machine", "source_sample_path", "source_pcap_size_bytes",
    "window_name", "window_start_epoch", "window_end_epoch", "window_end_reason",
    "cluster", "known_cluster", "fold", "split",
    "kit_family", "high_conf_family", "family", "family_label", "family_key", "primary_family_key",
    "known_family", "static_family", "static_family_key", "static_family_key_original",
    "static_family_reason", "static_family_count", "static_family_merge_component_size",
    "merged_static_family_keys", "coverage_static_cluster", "coverage_in_high_conf_family",
    "true_known_cluster", "evidence_tier", "evidence_tier_rank", "evidence_note",
    "primary_evidence_type", "primary_evidence_channel", "supporting_evidence_channels",
    "supporting_repeated_keys", "strong_supporting_key_count", "moderate_supporting_key_count",
    "weak_supporting_key_count", "independent_support_channel_count", "training_family_label_source",

}

CLIENT_STATIC_FEATURE_TOKENS = (
    "family", "static_", "evidence", "supporting", "support_channel",
    "dom", "html", "script", "resource", "asset", "image", "template", "title", "body",
    "drainer", "spender", "contract", "approve", "permit", "walletconnect_project",
    "walletconnect_verify", "infura", "project_id", "api_endpoint", "endpoint_path",
    "url_path", "path_token", "rpc_method", "js_capability", "interaction_pattern",
    "kit_behavior_hash",
)


def is_non_deployable_client_static_column(col: str) -> bool:
    lc = str(col).lower()
    return any(tok in lc for tok in CLIENT_STATIC_FEATURE_TOKENS)


def numeric_feature_columns(df: pd.DataFrame, window_name: Optional[str] = "dyn", extra_exclude: Optional[Iterable[str]] = None) -> List[str]:
    if window_name is not None and "window_name" in df.columns:
        df = df[df["window_name"] == window_name]
    exclude = set(META_COLUMNS)
    if extra_exclude:
        exclude.update(extra_exclude)
    cols = []
    meta_prefixes = ("oracle_", "anchor_", "window_available_", "window_latency_", "learned_", "post_load_guard_", "first_action_", "traffic_clip_")
    for c in df.columns:
        if c in exclude or c.startswith(meta_prefixes):
            continue
        if is_non_deployable_client_static_column(c):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return sorted(cols)


def select_window(df: pd.DataFrame, window_name: str = "dyn") -> pd.DataFrame:
    if window_name is None:
        return df.copy()
    if "window_name" in df.columns:
        return df[df["window_name"] == window_name].copy()
    return df.copy()


def label_to_binary(y: Sequence) -> np.ndarray:
    vals = []
    for v in y:
        s = str(v).lower()
        vals.append(1 if s in {"phishing", "phish", "malicious", "1", "true"} else 0)
    return np.asarray(vals, dtype=np.float32)


def ensure_cluster_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "cluster" not in df.columns:
        df["cluster"] = ""
    return df

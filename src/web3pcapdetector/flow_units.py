from __future__ import annotations

"""Gateway-visible operation-unit extraction for interactive phishing kits.

The default Layer 1 route remains candidate-window based. This module supplies
an auxiliary online segmentation primitive that is also reused by Layer 2.  It
segments a gateway-observed session into short interaction units from bursts of
role-agnostic and role-aware traffic, extracts only unit-internal relative
shape/rhythm features, and suppresses nuisance roles before modelling so the
representation remains independent of subtype-specific client/background traffic.

The extractor intentionally avoids HAR URL paths, DOM, screenshots, click
timestamps, and extension internals unless explicitly requested for debugging.
Default role source is SNI+passive-DNS, matching deployment visibility.
"""

import math
import pathlib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .candidates import resolve_post_load_guard
from .dynamic_window import Burst, build_bursts
from .flow_shape import aggregate_flow_segment_features
from .features import (
    assign_packet_roles_from_har,
    assign_packet_roles_from_sni,
    assign_packet_roles_from_sni_dns,
    assign_packet_roles_har_then_sni,
)
from .har import HarInfo
from .pcap_minimal import Packet, PcapData, read_pcap
from .pipeline import (
    apply_observability_degradation,
    drop_rpc_fanout_flows,
    empty_rpc_fanout_stats,
    suppress_packets_by_host_pattern,
    suppress_packets_by_role,
)
from .roles import normalize_host
from .zipio import extract_zip_sample, load_har_for_sample, load_sample_meta


DEFAULT_IMPORTANT_ROLES = (
    "unknown",
    "first_party_site",
    "third_party_backend_or_other",
    "identity_provider",
    "captcha_challenge",
    "payment_provider",
    "messaging_api",
    "form_backend",
    "cloud_api",
    "object_storage",
    "email_delivery",
    "file_storage",
    "url_shortener",
    "hosting_platform",
    "rpc_provider",
    "walletconnect",
)
DEFAULT_NEGATIVE_ROLES = (
    "third_party_static",
    "analytics_ads",
    "wallet_vendor",
    "software_update",
)
DEFAULT_ROLES = (
    "unknown",
    "first_party_site",
    "hosting_platform",
    "identity_provider",
    "captcha_challenge",
    "payment_provider",
    "messaging_api",
    "form_backend",
    "cloud_api",
    "object_storage",
    "email_delivery",
    "file_storage",
    "url_shortener",
    "rpc_provider",
    "wallet_vendor",
    "walletconnect",
    "third_party_static",
    "third_party_backend_or_other",
    "analytics_ads",
    "software_update",
)


@dataclass(frozen=True)
class OperationUnitConfig:
    role_source: str = "sni_dns"
    feature_mode: str = "clean"
    include_unknown_packets: bool = True
    include_tcp: bool = True
    include_udp443: bool = True
    suppress_roles: Tuple[str, ...] = ("wallet_vendor",)
    suppress_host_patterns: Tuple[str, ...] = ()
    suppress_host_pattern_skip_roles: Tuple[str, ...] = ("first_party_site",)
    drop_rpc_fanout_flows: bool = True
    rpc_fanout_window_s: float = 8.0
    rpc_fanout_min_flows: int = 3
    rpc_fanout_min_hosts: int = 2

    # Online unit segmentation. Candidate start is available at the first
    # significant burst; features only use [unit_start, unit_start+observation_s].
    min_after_capture_start_s: float = 0.5
    min_after_post_load_s: float = 0.0
    max_scan_s: float = 180.0
    burst_gap_s: float = 0.35
    unit_merge_gap_s: float = 1.25
    max_unit_s: float = 12.0
    # Backward-compatible safety cap.  When adaptive_observation=true the
    # extractor closes earlier on evidence saturation or idle timeout.
    observation_s: float = 14.0
    adaptive_observation: bool = True
    min_observation_s: float = 1.25
    idle_gap_s: float = 0.90
    tail_pad_s: float = 0.10
    saturation_bytes: int = 12000
    saturation_important_frac: float = 0.40
    min_unit_packets: int = 3
    min_unit_bytes: int = 800
    min_important_bytes: int = 800
    min_candidate_score: float = 0.35
    top_k_units: int = 16
    merge_candidates_within_s: float = 1.5
    fallback_if_empty: bool = True

    important_roles: Tuple[str, ...] = DEFAULT_IMPORTANT_ROLES
    negative_roles: Tuple[str, ...] = DEFAULT_NEGATIVE_ROLES
    roles: Tuple[str, ...] = DEFAULT_ROLES

    # Feature construction.
    max_prefix_packets: int = 32
    active_bin_s: float = 0.5
    large_packet_threshold: int = 1000
    small_packet_threshold: int = 80


@dataclass(frozen=True)
class OperationUnit:
    start_epoch: float
    end_epoch: float
    feature_end_epoch: float
    score: float
    rank: int
    reason: str
    n_pkts: int
    bytes_total: int
    important_bytes: int
    negative_bytes: int
    start_rel_s: float
    emit_rel_s: float


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def _as_tuple(value, default: Sequence[str] = ()) -> Tuple[str, ...]:
    if value is None:
        return tuple(default)
    if isinstance(value, str):
        items = [x.strip() for x in value.split(",")]
    else:
        items = [str(x).strip() for x in value]
    return tuple(x for x in items if x)


def operation_unit_config_from_dict(cfg: dict | None) -> OperationUnitConfig:
    cfg = cfg or {}
    src = dict(cfg.get("operation_units", {}))
    filtering = dict(cfg.get("filtering", {}))
    for key in [
        "role_source",
        "feature_mode",
        "include_unknown_packets",
        "include_tcp",
        "include_udp443",
        "suppress_roles",
        "suppress_host_patterns",
        "suppress_host_pattern_skip_roles",
        "drop_rpc_fanout_flows",
        "rpc_fanout_window_s",
        "rpc_fanout_min_flows",
        "rpc_fanout_min_hosts",
    ]:
        if key not in src and key in filtering:
            src[key] = filtering[key]
    tuple_keys = {
        "suppress_roles",
        "suppress_host_patterns",
        "suppress_host_pattern_skip_roles",
        "important_roles",
        "negative_roles",
        "roles",
    }
    base = asdict(OperationUnitConfig())
    for key, value in src.items():
        if key not in base:
            continue
        if key in tuple_keys:
            base[key] = _as_tuple(value, base.get(key, ()))
        else:
            base[key] = value
    return OperationUnitConfig(**base)


def _anchor_keys(cfg: dict) -> list[str]:
    anchor_cfg = cfg.get("anchor", {})
    return anchor_cfg.get("keys") or [
        anchor_cfg.get("primary_session_key", "t_metamask_connect_click"),
        anchor_cfg.get("event_fallback_key", "t_metamask_connect_click"),
    ]


def _role_counter(pkts: Sequence[Packet]) -> Counter[str]:
    c: Counter[str] = Counter()
    for p in pkts:
        c[str(p.role or "unknown").lower()] += int(p.length)
    return c


def _sum_roles(counter: Counter[str], roles: Iterable[str]) -> int:
    return int(sum(counter[str(r).lower()] for r in roles))


def _host_parts(host: str) -> list[str]:
    return [h.strip().lower() for h in str(host or "").split("|") if h.strip()]


def _server_ip(packet: Packet) -> str:
    if packet.direction == "up":
        return packet.dst_ip
    if packet.direction == "down":
        return packet.src_ip
    return packet.dst_ip


def prepare_gateway_packets(zip_path: str | pathlib.Path, cfg: dict, allow_har_roles: bool = False) -> tuple[list[Packet], dict, object, PcapData]:
    """Load one sample and return deployment-visible packets plus metadata.

    ``allow_har_roles`` is intentionally false by default. For Layer 1 paper
    experiments this function should run with SNI/DNS roles, not HAR mappings.
    """
    unit_cfg = operation_unit_config_from_dict(cfg)
    role_source = str(unit_cfg.role_source or "sni_dns").lower()
    if role_source in {"har", "har_then_sni"} and not allow_har_roles:
        raise ValueError(
            f"{zip_path}: operation-unit extraction would use role_source={role_source!r}; "
            "pass allow_har_roles=True only for offline debugging."
        )
    if role_source not in {"sni", "sni_dns", "preserved", "har", "har_then_sni"}:
        raise ValueError(f"unknown operation unit role_source={role_source!r}")

    paths = extract_zip_sample(zip_path, require_har=role_source in {"har", "har_then_sni"})
    try:
        meta = load_sample_meta(paths, anchor_key=_anchor_keys(cfg))
        har = load_har_for_sample(paths, meta, feature_mode=unit_cfg.feature_mode) if role_source in {"har", "har_then_sni"} else HarInfo()
        pcap = read_pcap(paths.pcap_path)
        degradation_stats = apply_observability_degradation(pcap, cfg.get("filtering", {}).get("degradation"))
        if role_source == "preserved":
            # Composed E3 event streams already contain the deployment-visible
            # SNI/DNS-derived role attached to each packet.  A continuous trace
            # has multiple visited primary hosts, so recomputing against one
            # synthetic session domain would corrupt those source-capture roles.
            packets = [
                packet for packet in pcap.packets
                if bool(unit_cfg.include_unknown_packets) or str(packet.role) != "unknown"
            ]
        elif role_source == "sni":
            packets = assign_packet_roles_from_sni(
                pcap,
                primary_host=meta.domain,
                feature_mode=unit_cfg.feature_mode,
                include_unknown=bool(unit_cfg.include_unknown_packets),
            )
        elif role_source == "sni_dns":
            packets = assign_packet_roles_from_sni_dns(
                pcap,
                primary_host=meta.domain,
                feature_mode=unit_cfg.feature_mode,
                include_unknown=bool(unit_cfg.include_unknown_packets),
            )
        elif role_source == "har":
            packets = assign_packet_roles_from_har(pcap, har, include_unknown=bool(unit_cfg.include_unknown_packets))
        else:
            packets = assign_packet_roles_har_then_sni(
                pcap,
                har,
                primary_host=meta.domain,
                feature_mode=unit_cfg.feature_mode,
                include_unknown=bool(unit_cfg.include_unknown_packets),
            )
        if not unit_cfg.include_tcp:
            packets = [p for p in packets if p.proto != "TCP"]
        if not unit_cfg.include_udp443:
            packets = [p for p in packets if not p.is_udp443]
        packets, role_suppression = suppress_packets_by_role(packets, unit_cfg.suppress_roles)
        packets, host_suppression = suppress_packets_by_host_pattern(
            packets,
            unit_cfg.suppress_host_patterns,
            skip_roles=unit_cfg.suppress_host_pattern_skip_roles,
        )
        if bool(unit_cfg.drop_rpc_fanout_flows):
            packets, rpc_suppression = drop_rpc_fanout_flows(
                packets,
                window_s=float(unit_cfg.rpc_fanout_window_s),
                min_flows=int(unit_cfg.rpc_fanout_min_flows),
                min_hosts=int(unit_cfg.rpc_fanout_min_hosts),
            )
        else:
            rpc_suppression = empty_rpc_fanout_stats()
        prep_stats = {
            "role_source": role_source,
            "har_n_hosts": len(har.hosts),
            "har_n_server_ips": len(har.ip_to_hosts),
            **degradation_stats,
            **role_suppression,
            **host_suppression,
            **rpc_suppression,
        }
        # Detach tempdir cleanup from caller by materialising all packets and meta first.
        return list(sorted(packets, key=lambda p: p.ts)), prep_stats, meta, pcap
    finally:
        if paths.tempdir is not None:
            paths.tempdir.cleanup()


def _candidate_score(pkts: Sequence[Packet], cfg: OperationUnitConfig) -> tuple[float, str, dict]:
    n = len(pkts)
    total = int(sum(p.length for p in pkts))
    c = _role_counter(pkts)
    important = _sum_roles(c, cfg.important_roles)
    negative = _sum_roles(c, cfg.negative_roles)
    n_flows = len({p.flow_id for p in pkts})
    n_hosts = len({h for p in pkts for h in _host_parts(p.host)})
    if n < int(cfg.min_unit_packets) or total < int(cfg.min_unit_bytes):
        return 0.0, "too_small", {"important": important, "negative": negative, "total": total}
    shape = aggregate_flow_segment_features(pkts, burst_gap_s=float(cfg.burst_gap_s))
    intrinsic = float(shape.get("seg_intrinsic_mean", 0.0))
    intrinsic_max = float(shape.get("seg_intrinsic_max", 0.0))
    high_intrinsic = float(shape.get("seg_high_intrinsic_frac", 0.0))
    byte_score = min(1.0, math.log1p(total) / math.log1p(50_000.0))
    flow_score = min(1.0, n_flows / 6.0)
    # Role is weak auxiliary only; the unit candidate is primarily a stable kit
    # flow-shape/motif unit.
    role_aux = 0.0
    if important >= int(cfg.min_important_bytes):
        role_aux += 0.08
    if c["unknown"] >= 2000:
        role_aux += 0.04
    if c["rpc_provider"] > 0 or c["walletconnect"] > 0:
        role_aux += 0.05
    if c["identity_provider"] > 0 or c["payment_provider"] > 0:
        role_aux += 0.05
    role_aux = min(0.18, role_aux)
    negative_frac = _safe_div(negative, total)
    shape_score = 0.45 * intrinsic + 0.35 * intrinsic_max + 0.20 * high_intrinsic
    score = 0.58 * shape_score + 0.20 * byte_score + 0.12 * flow_score + role_aux - 0.18 * negative_frac
    score = max(0.0, min(1.0, score))
    reasons = []
    if intrinsic_max >= 0.35:
        reasons.append("unit_flow_shape_motif")
    if high_intrinsic >= 0.35:
        reasons.append("multi_flow_unit_shape")
    if byte_score >= 0.50:
        reasons.append("byte_burst")
    if role_aux:
        reasons.append("weak_role_aux")
    if negative_frac >= 0.45:
        reasons.append("negative_role_penalty")
    return score, ";".join(reasons) or "traffic_shape_unit", {"important": important, "negative": negative, "total": total}




def _adaptive_feature_end(
    packets: Sequence[Packet],
    unit_start: float,
    scan_end: float,
    cfg: OperationUnitConfig,
) -> float:
    max_end = min(float(scan_end), float(unit_start) + max(0.25, float(cfg.observation_s)))
    if not bool(cfg.adaptive_observation):
        return max_end
    post = sorted([p for p in packets if float(unit_start) <= p.ts <= max_end], key=lambda p: p.ts)
    if not post:
        return max_end
    min_end = float(unit_start) + max(0.0, float(cfg.min_observation_s))
    important_roles = {str(r).lower() for r in cfg.important_roles}
    bursts = build_bursts(post, burst_gap_s=float(cfg.burst_gap_s))
    for i, b in enumerate(bursts):
        sub = [p for p in post if p.ts <= b.end]
        n = len(sub)
        total = int(sum(p.length for p in sub))
        important = int(sum(p.length for p in sub if str(p.role or "unknown").lower() in important_roles))
        enough = n >= int(cfg.min_unit_packets) and total >= int(cfg.min_unit_bytes) and important >= int(cfg.min_important_bytes)
        if not enough:
            continue
        imp_frac = important / total if total else 0.0
        evidence_end = max(min_end, float(b.end) + max(0.0, float(cfg.tail_pad_s)))
        if total >= int(cfg.saturation_bytes) and imp_frac >= float(cfg.saturation_important_frac):
            return min(max_end, evidence_end)
        nxt = bursts[i + 1] if i + 1 < len(bursts) else None
        if nxt is None:
            return min(max_end, max(evidence_end, float(b.end) + float(cfg.idle_gap_s)))
        gap = float(nxt.start) - float(b.end)
        if gap >= float(cfg.idle_gap_s):
            return min(max_end, max(evidence_end, float(b.end) + float(cfg.idle_gap_s)))
    return min(max_end, max(min_end, float(post[-1].ts) + max(0.0, float(cfg.tail_pad_s))))

def segment_operation_units(
    packets: Sequence[Packet],
    cfg: OperationUnitConfig,
    *,
    pcap_start: Optional[float],
    pcap_end: Optional[float],
    post_load_epoch: Optional[float] = None,
) -> list[OperationUnit]:
    if not packets:
        return []
    packets = sorted(packets, key=lambda p: p.ts)
    start0 = float(pcap_start if pcap_start is not None else packets[0].ts)
    scan_start = start0 + float(cfg.min_after_capture_start_s)
    if post_load_epoch is not None:
        scan_start = max(scan_start, float(post_load_epoch) + max(0.0, float(cfg.min_after_post_load_s)))
    scan_end = min(float(pcap_end if pcap_end is not None else packets[-1].ts), start0 + float(cfg.max_scan_s))
    scan = [p for p in packets if scan_start <= p.ts <= scan_end]
    if not scan:
        scan = [p for p in packets if p.ts <= scan_end]
    # Segment on important packets; features later use all non-suppressed packets
    # in the unit window. This avoids static page resources defining operation
    # boundaries.
    seg_roles = {r.lower() for r in cfg.important_roles}
    seg_pkts = [p for p in scan if str(p.role or "unknown").lower() in seg_roles]
    if not seg_pkts:
        seg_pkts = scan
    bursts = build_bursts(seg_pkts, burst_gap_s=float(cfg.burst_gap_s))
    if not bursts:
        return []

    raw_units: list[tuple[float, float]] = []
    i = 0
    while i < len(bursts):
        b = bursts[i]
        unit_start = float(b.start)
        unit_end = float(b.end)
        j = i + 1
        while j < len(bursts):
            nxt = bursts[j]
            if nxt.start - unit_end > float(cfg.unit_merge_gap_s):
                break
            if nxt.end - unit_start > float(cfg.max_unit_s):
                break
            unit_end = float(nxt.end)
            j += 1
        raw_units.append((unit_start, unit_end))
        i = max(j, i + 1)

    units: list[OperationUnit] = []
    for unit_start, unit_seg_end in raw_units:
        feature_end = _adaptive_feature_end(packets, unit_start, scan_end, cfg)
        if feature_end <= unit_start:
            continue
        sub = [p for p in packets if unit_start <= p.ts <= feature_end]
        score, reason, counts = _candidate_score(sub, cfg)
        if score < float(cfg.min_candidate_score):
            continue
        units.append(OperationUnit(
            start_epoch=float(unit_start),
            end_epoch=float(unit_seg_end),
            feature_end_epoch=float(feature_end),
            score=float(score),
            rank=-1,
            reason=reason,
            n_pkts=int(len(sub)),
            bytes_total=int(counts["total"]),
            important_bytes=int(counts["important"]),
            negative_bytes=int(counts["negative"]),
            start_rel_s=float(unit_start - start0),
            emit_rel_s=float(feature_end - start0),
        ))

    if not units and bool(cfg.fallback_if_empty):
        # Fallback: choose the highest-byte burst after the initial page load. It
        # still uses only an online observation horizon, so this remains deployable.
        candidates = []
        for b in bursts:
            if b.start < scan_start:
                continue
            f_end = _adaptive_feature_end(packets, float(b.start), scan_end, cfg)
            sub = [p for p in packets if b.start <= p.ts <= f_end]
            score, reason, counts = _candidate_score(sub, cfg)
            candidates.append((score, b, sub, reason, counts, f_end))
        if candidates:
            score, b, sub, reason, counts, f_end = max(candidates, key=lambda x: (x[0], x[4]["total"]))
            units.append(OperationUnit(
                start_epoch=float(b.start),
                end_epoch=float(b.end),
                feature_end_epoch=float(f_end),
                score=float(max(score, 0.01)),
                rank=-1,
                reason=f"fallback:{reason}",
                n_pkts=int(len(sub)),
                bytes_total=int(counts["total"]),
                important_bytes=int(counts["important"]),
                negative_bytes=int(counts["negative"]),
                start_rel_s=float(b.start - start0),
                emit_rel_s=float(f_end - start0),
            ))

    # Merge near-duplicate starts and keep top-k by score; final ranking is by
    # time for online processing, while candidate_rank_score records evidence.
    units = sorted(units, key=lambda u: (-u.score, u.start_epoch))
    kept: list[OperationUnit] = []
    for u in units:
        if any(abs(u.start_epoch - v.start_epoch) <= float(cfg.merge_candidates_within_s) for v in kept):
            continue
        kept.append(u)
        if len(kept) >= int(cfg.top_k_units):
            break
    kept = sorted(kept, key=lambda u: u.start_epoch)
    return [OperationUnit(**{**asdict(u), "rank": i}) for i, u in enumerate(kept)]




STAGE_ROLE_GROUPS = {
    "interaction": DEFAULT_IMPORTANT_ROLES,
    "credential_auth": ("identity_provider", "form_backend", "first_party_site"),
    "challenge_gate": ("captcha_challenge",),
    "payment_checkout": ("payment_provider",),
    "exfil_delivery": ("messaging_api", "form_backend", "cloud_api", "object_storage", "email_delivery", "file_storage"),
    "crypto_transaction": ("rpc_provider", "walletconnect"),
    "kit_hosting": ("hosting_platform", "first_party_site"),
    "backend_unknown": ("third_party_backend_or_other", "unknown"),
    "nuisance": DEFAULT_NEGATIVE_ROLES,
}
STAGE_CODES = {name: i + 1 for i, name in enumerate([
    "crypto_transaction", "credential_auth", "payment_checkout", "challenge_gate",
    "exfil_delivery", "kit_hosting", "backend_unknown", "nuisance",
])}


def _dominant_stage(role_bytes: Counter[str], total_bytes: float) -> tuple[str, float, int]:
    best_name, best_val = "none", 0.0
    for name, members in STAGE_ROLE_GROUPS.items():
        if name == "interaction":
            continue
        val = _safe_div(_sum_roles(role_bytes, members), total_bytes)
        if val > best_val:
            best_name, best_val = name, val
    return best_name, float(best_val), int(STAGE_CODES.get(best_name, 0))


def _stats(prefix: str, values: Sequence[float]) -> dict[str, float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_p25": 0.0,
            f"{prefix}_p50": 0.0,
            f"{prefix}_p75": 0.0,
            f"{prefix}_p90": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_min": 0.0,
        }
    arr = np.asarray(vals, dtype=float)
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_p25": float(np.percentile(arr, 25)),
        f"{prefix}_p50": float(np.percentile(arr, 50)),
        f"{prefix}_p75": float(np.percentile(arr, 75)),
        f"{prefix}_p90": float(np.percentile(arr, 90)),
        f"{prefix}_max": float(np.max(arr)),
        f"{prefix}_min": float(np.min(arr)),
    }


def _direction_value(packet: Packet) -> float:
    if packet.direction == "up":
        return 1.0
    if packet.direction == "down":
        return -1.0
    return 0.0


def _sample_prefix(prefix: str, values: Sequence[float], n: int) -> dict[str, float]:
    out: dict[str, float] = {}
    vals = list(values)
    for i in range(int(n)):
        out[f"{prefix}_{i:02d}"] = float(vals[i]) if i < len(vals) and math.isfinite(float(vals[i])) else 0.0
    return out


def _flow_records(pkts: Sequence[Packet], start: float) -> list[dict]:
    by_flow: dict[Tuple, list[Packet]] = defaultdict(list)
    for p in pkts:
        by_flow[p.flow_id].append(p)
    rows = []
    for flow_id, fps in by_flow.items():
        fps = sorted(fps, key=lambda p: p.ts)
        up = [p for p in fps if p.direction == "up"]
        down = [p for p in fps if p.direction == "down"]
        roles = Counter(str(p.role or "unknown").lower() for p in fps)
        rows.append({
            "flow_id": flow_id,
            "first_rel_s": float(fps[0].ts - start),
            "last_rel_s": float(fps[-1].ts - start),
            "duration_s": float(max(0.0, fps[-1].ts - fps[0].ts)),
            "n_pkts": int(len(fps)),
            "bytes_total": int(sum(p.length for p in fps)),
            "bytes_up": int(sum(p.length for p in up)),
            "bytes_down": int(sum(p.length for p in down)),
            "pkts_up": int(len(up)),
            "pkts_down": int(len(down)),
            "server_ip": _server_ip(fps[0]),
            "host_key": "|".join(sorted({h for p in fps for h in _host_parts(p.host)})[:3]),
            "dominant_role": roles.most_common(1)[0][0] if roles else "unknown",
        })
    return sorted(rows, key=lambda r: (r["first_rel_s"], r["flow_id"]))


def extract_unit_features(
    packets: Sequence[Packet],
    unit: OperationUnit,
    cfg: OperationUnitConfig,
    *,
    pcap_start: Optional[float],
    pcap_end: Optional[float],
) -> dict[str, float | str]:
    start = float(unit.start_epoch)
    end = float(unit.feature_end_epoch)
    pkts = sorted([p for p in packets if start <= p.ts <= end], key=lambda p: p.ts)
    n = len(pkts)
    total_bytes = float(sum(p.length for p in pkts))
    duration = max(0.0, end - start)
    out: dict[str, float | str] = {
        "unit_rank": int(unit.rank),
        "unit_start_epoch": start,
        "unit_end_epoch": float(unit.end_epoch),
        "unit_feature_end_epoch": end,
        "unit_observation_s": duration,
        "unit_seg_duration_s": max(0.0, float(unit.end_epoch) - start),
        "unit_emit_after_capture_start_s": float(unit.emit_rel_s),
        "unit_start_after_capture_start_s": float(unit.start_rel_s),
        "unit_candidate_score": float(unit.score),
        "unit_reason": unit.reason,
        "unit_n_pkts": float(n),
        "unit_bytes_total": total_bytes,
        "unit_packet_rate": _safe_div(n, duration),
        "unit_byte_rate": _safe_div(total_bytes, duration),
        "unit_pcap_remaining_s": max(0.0, float(pcap_end or end) - end),
    }
    if not pkts:
        # Still emit a complete feature vector. Stable profile / model imputers
        # can handle zeros, but explicit names improve reproducibility.
        for prefix in ["len_log", "len_delta", "len_offset", "iat_rel", "iat_raw", "flow_bytes", "flow_pkts", "flow_dur", "flow_start_rel"]:
            out.update(_stats(prefix, []))
        out.update(_sample_prefix("prefix_dir", [], cfg.max_prefix_packets))
        out.update(_sample_prefix("prefix_log_len", [], cfg.max_prefix_packets))
        out.update(_sample_prefix("prefix_rel_iat", [], cfg.max_prefix_packets))
        for group in STAGE_ROLE_GROUPS:
            out[f"stage_{group}_byte_frac"] = 0.0
            out[f"stage_{group}_pkt_frac"] = 0.0
        out["stage_dominant_frac"] = 0.0
        out["stage_dominant_code"] = 0.0
        out["stage_dominant_name"] = "none"
        return out

    dirs = [_direction_value(p) for p in pkts]
    lengths = [float(max(0, int(p.payload_len if p.payload_len else p.length))) for p in pkts]
    log_len = np.log1p(np.asarray(lengths, dtype=float))
    len_delta = np.diff(log_len, prepend=log_len[0])
    running_medians = []
    for i in range(len(log_len)):
        running_medians.append(float(np.median(log_len[: i + 1])))
    len_offset = log_len - np.asarray(running_medians, dtype=float)
    ts = np.asarray([float(p.ts) for p in pkts], dtype=float)
    iat_raw = np.diff(ts, prepend=start)
    iat_raw = np.maximum(iat_raw, 0.0)
    positive_iat = iat_raw[iat_raw > 1e-6]
    scale = float(np.median(positive_iat)) if len(positive_iat) else 1.0
    iat_rel = np.clip(iat_raw / max(scale, 1e-6), 0.0, 20.0)
    cum_rel_time = (ts - start) / max(duration, 1e-6)
    cum_rel_time = np.clip(cum_rel_time, 0.0, 1.0)
    sign_switches = sum(1 for a, b in zip(dirs, dirs[1:]) if a != b)
    out["dir_switch_rate"] = _safe_div(sign_switches, max(n - 1, 1))
    out["dir_up_frac"] = _safe_div(sum(1 for d in dirs if d > 0), n)
    out["dir_down_frac"] = _safe_div(sum(1 for d in dirs if d < 0), n)
    out["len_total_payload_or_frame_bytes"] = float(sum(lengths))
    out.update(_stats("len_log", log_len.tolist()))
    out.update(_stats("len_delta", len_delta.tolist()))
    out.update(_stats("len_offset", len_offset.tolist()))
    out.update(_stats("iat_raw", iat_raw.tolist()))
    out.update(_stats("iat_rel", iat_rel.tolist()))
    out.update(_stats("cum_rel_time", cum_rel_time.tolist()))
    out.update(_sample_prefix("prefix_dir", dirs, cfg.max_prefix_packets))
    out.update(_sample_prefix("prefix_log_len", log_len.tolist(), cfg.max_prefix_packets))
    out.update(_sample_prefix("prefix_delta_len", len_delta.tolist(), cfg.max_prefix_packets))
    out.update(_sample_prefix("prefix_rel_iat", iat_rel.tolist(), cfg.max_prefix_packets))
    out.update(_sample_prefix("prefix_cum_rel_time", cum_rel_time.tolist(), cfg.max_prefix_packets))

    roles = [str(r).lower() for r in cfg.roles]
    role_bytes = Counter()
    role_pkts = Counter()
    role_flows: dict[str, set] = defaultdict(set)
    proto_bytes = Counter()
    proto_pkts = Counter()
    for p in pkts:
        role = str(p.role or "unknown").lower()
        proto = str(p.proto or "OTHER").upper()
        role_bytes[role] += int(p.length)
        role_pkts[role] += 1
        role_flows[role].add(p.flow_id)
        proto_bytes[proto] += int(p.length)
        proto_pkts[proto] += 1
    flows = {p.flow_id for p in pkts}
    for role in roles:
        out[f"role_{role}_byte_frac"] = _safe_div(role_bytes[role], total_bytes)
        out[f"role_{role}_pkt_frac"] = _safe_div(role_pkts[role], n)
        out[f"role_{role}_flow_frac"] = _safe_div(len(role_flows[role]), len(flows))
    for group, members in STAGE_ROLE_GROUPS.items():
        out[f"stage_{group}_byte_frac"] = _safe_div(_sum_roles(role_bytes, members), total_bytes)
        out[f"stage_{group}_pkt_frac"] = _safe_div(_sum_roles(role_pkts, members), n)
    stage_name, stage_frac, stage_code = _dominant_stage(role_bytes, total_bytes)
    out["stage_dominant_frac"] = stage_frac
    out["stage_dominant_code"] = float(stage_code)
    out["stage_dominant_name"] = stage_name
    probs = [role_bytes[r] / total_bytes for r in role_bytes if total_bytes > 0 and role_bytes[r] > 0]
    out["role_entropy_bytes"] = float(-sum(p * math.log(p + 1e-12) for p in probs))
    out["role_dominant_byte_frac"] = float(max(probs) if probs else 0.0)
    out["role_important_byte_frac"] = _safe_div(_sum_roles(role_bytes, cfg.important_roles), total_bytes)
    out["role_negative_byte_frac"] = _safe_div(_sum_roles(role_bytes, cfg.negative_roles), total_bytes)
    out["wallet_disturbance_byte_frac"] = _safe_div(role_bytes["wallet_vendor"], total_bytes)
    out["walletconnect_context_byte_frac"] = _safe_div(role_bytes["walletconnect"], total_bytes)
    for proto in ["TCP", "UDP", "OTHER"]:
        out[f"proto_{proto.lower()}_byte_frac"] = _safe_div(proto_bytes[proto], total_bytes)
        out[f"proto_{proto.lower()}_pkt_frac"] = _safe_div(proto_pkts[proto], n)
    out["proto_udp443_pkt_frac"] = _safe_div(sum(1 for p in pkts if p.is_udp443), n)
    out["proto_udp443_byte_frac"] = _safe_div(sum(p.length for p in pkts if p.is_udp443), total_bytes)

    active_bin = max(1e-6, float(cfg.active_bin_s))
    active_bins = {int((p.ts - start) // active_bin) for p in pkts}
    out["unit_active_bins"] = float(len(active_bins))
    out["unit_active_time_frac"] = _safe_div(len(active_bins) * active_bin, duration)
    bursts = build_bursts(pkts, burst_gap_s=float(cfg.burst_gap_s))
    out["burst_count"] = float(len(bursts))
    out.update(_stats("burst_pkts", [b.n_pkts for b in bursts]))
    out.update(_stats("burst_bytes", [b.bytes_total for b in bursts]))
    out.update(_stats("burst_dur", [b.duration for b in bursts]))
    if len(bursts) > 1:
        out.update(_stats("burst_gap", [bursts[i + 1].start - bursts[i].end for i in range(len(bursts) - 1)]))
    else:
        out.update(_stats("burst_gap", []))

    flow_rows = _flow_records(pkts, start)
    out["flow_count"] = float(len(flow_rows))
    out["flow_server_ip_count"] = float(len({r["server_ip"] for r in flow_rows if r["server_ip"]}))
    out["flow_host_key_count"] = float(len({r["host_key"] for r in flow_rows if r["host_key"]}))
    out.update(_stats("flow_bytes", [r["bytes_total"] for r in flow_rows]))
    out.update(_stats("flow_pkts", [r["n_pkts"] for r in flow_rows]))
    out.update(_stats("flow_dur", [r["duration_s"] for r in flow_rows]))
    out.update(_stats("flow_start_rel", [r["first_rel_s"] for r in flow_rows]))
    out["flow_up_byte_frac_mean"] = float(np.mean([_safe_div(r["bytes_up"], r["bytes_total"]) for r in flow_rows])) if flow_rows else 0.0
    out["flow_down_byte_frac_mean"] = float(np.mean([_safe_div(r["bytes_down"], r["bytes_total"]) for r in flow_rows])) if flow_rows else 0.0
    out["large_pkt_frac"] = _safe_div(sum(1 for p in pkts if p.length >= int(cfg.large_packet_threshold)), n)
    out["small_pkt_frac"] = _safe_div(sum(1 for p in pkts if p.length <= int(cfg.small_packet_threshold)), n)
    # Role-free stable-kit unit shape features for Layer 2 prototypes.
    out.update(aggregate_flow_segment_features(pkts, start=start, end=end, burst_gap_s=float(cfg.burst_gap_s)))
    return out


def extract_operation_units_from_zip(
    zip_path: str | pathlib.Path,
    cfg: dict,
    *,
    allow_har_roles: bool = False,
) -> pd.DataFrame:
    unit_cfg = operation_unit_config_from_dict(cfg)
    packets, prep_stats, meta, pcap = prepare_gateway_packets(zip_path, cfg, allow_har_roles=allow_har_roles)
    pcap_start = pcap.pcap_start or meta.pcap_start_epoch or meta.session_start_epoch
    pcap_end = pcap.pcap_end or meta.pcap_stop_epoch or meta.capture_stop_epoch
    post_load_guard_epoch, post_load_guard_source, post_load_guard_mode = resolve_post_load_guard(
        cfg.get("anchor", {}),
        meta_post_load_epoch=meta.post_load_epoch,
        meta_post_load_source=meta.post_load_source,
        packets=packets,
        session_start=meta.session_start_epoch if meta.session_start_epoch is not None else pcap_start,
    )
    units = segment_operation_units(
        packets,
        unit_cfg,
        pcap_start=pcap_start,
        pcap_end=pcap_end,
        post_load_epoch=post_load_guard_epoch,
    )
    rows = []
    for unit in units:
        row = {
            "sample_id": meta.sample_id,
            "label": meta.label,
            "domain": meta.domain,
            "url": meta.url,
            "status": meta.status,
            "zip_path": str(zip_path),
            "pcap_start_epoch": pcap_start,
            "pcap_end_epoch": pcap_end,
            "oracle_anchor_epoch": meta.anchor_epoch,
            "oracle_anchor_source": meta.anchor_source,
            "post_load_guard_mode": post_load_guard_mode,
            "post_load_guard_epoch": post_load_guard_epoch,
            "post_load_guard_source": post_load_guard_source,
            "post_load_guard_rel_s": (float(post_load_guard_epoch) - float(pcap_start)) if (post_load_guard_epoch is not None and pcap_start is not None) else float("nan"),
            "first_action_epoch": meta.first_action_epoch,
            "first_action_source": meta.first_action_source,
            "first_action_rel_s": (float(meta.first_action_epoch) - float(pcap_start)) if (meta.first_action_epoch is not None and pcap_start is not None) else float("nan"),
            "operation_unit_policy": "gateway_visible_adaptive_online_unit",
        }
        row.update(prep_stats)
        row.update(extract_unit_features(packets, unit, unit_cfg, pcap_start=pcap_start, pcap_end=pcap_end))
        rows.append(row)
    return pd.DataFrame(rows)


def extract_operation_units(
    zip_paths: Sequence[str | pathlib.Path],
    cfg: dict,
    *,
    allow_har_roles: bool = False,
    skip_errors: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dfs = []
    errors = []
    for path in zip_paths:
        try:
            df = extract_operation_units_from_zip(path, cfg, allow_har_roles=allow_har_roles)
            if len(df):
                dfs.append(df)
        except Exception as exc:  # noqa: BLE001
            errors.append({"zip_path": str(path), "type": type(exc).__name__, "error": str(exc)})
            if not skip_errors:
                raise
    out = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    return out, pd.DataFrame(errors)

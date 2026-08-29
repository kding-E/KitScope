from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .dynamic_window import Window, build_bursts
from .flow_shape import FlowShapeIndex, aggregate_flow_segment_features, window_flow_shape_features
from .pcap_minimal import Packet, PcapData
from .roles import classify_host, merge_roles, normalize_host


def _server_ip_for_packet(p: Packet) -> str:
    if p.direction == "up":
        return p.dst_ip
    if p.direction == "down":
        return p.src_ip
    return p.dst_ip


def sni_observation_counts(pcap: PcapData) -> tuple[int, int]:
    hosts = set()
    server_ips = set()
    for p in pcap.packets:
        host = normalize_host(p.sni)
        if not host:
            continue
        hosts.add(host)
        server_ips.add(_server_ip_for_packet(p))
    return len(hosts), len(server_ips)


def dns_observation_counts(pcap: PcapData) -> tuple[int, int]:
    hosts = set()
    for mapped_hosts in pcap.dns_ip_to_hosts.values():
        hosts.update(normalize_host(h) for h in mapped_hosts if h)
    return len(hosts), len(pcap.dns_ip_to_hosts)


def _flow_sni_roles(pcap: PcapData, primary_host: str = "", feature_mode: str = "clean") -> Dict[Tuple, Tuple[str, str]]:
    out: Dict[Tuple, Tuple[str, str]] = {}
    for p in pcap.packets:
        host = normalize_host(p.sni)
        if not host:
            continue
        out[p.flow_id] = (host, classify_host(host, primary_host=primary_host, feature_mode=feature_mode))
    return out


def _dns_role_for_packet(
    pcap: PcapData,
    packet: Packet,
    primary_host: str = "",
    feature_mode: str = "clean",
) -> Tuple[str, str] | None:
    hosts = pcap.dns_ip_to_hosts.get(_server_ip_for_packet(packet), set())
    hosts = {normalize_host(h) for h in hosts if normalize_host(h)}
    if not hosts:
        return None
    roles = [classify_host(h, primary_host=primary_host, feature_mode=feature_mode) for h in hosts]
    role = merge_roles(roles)
    host = sorted(hosts)[0] if len(hosts) == 1 else "|".join(sorted(hosts)[:3])
    return host, role


def assign_packet_roles_from_har(pcap: PcapData, har_info, include_unknown: bool = False) -> List[Packet]:
    """Assign host/role to packets using HAR serverIPAddress mapping.

    This intentionally uses host/IP mapping for purification and role abstraction,
    not URL paths or decrypted content.
    """
    out = []
    for p in pcap.packets:
        # The server is usually the non-local side. Direction inference is done in pcap_minimal.
        server_ip = p.dst_ip if p.direction == "up" else p.src_ip if p.direction == "down" else p.dst_ip
        hosts = har_info.ip_to_hosts.get(server_ip, set())
        if hosts:
            roles = [har_info.host_to_role.get(h, "unknown") for h in hosts]
            p.role = merge_roles(roles)
            p.host = sorted(hosts)[0] if len(hosts) == 1 else "|".join(sorted(list(hosts))[:3])
        else:
            p.role = "unknown"
            p.host = ""
        if include_unknown or p.role != "unknown":
            out.append(p)
    return out


def assign_packet_roles_from_sni(
    pcap: PcapData,
    primary_host: str = "",
    feature_mode: str = "clean",
    include_unknown: bool = False,
) -> List[Packet]:
    """Assign host/role to packets using TLS ClientHello SNI observed in PCAP.

    This is a gateway-oriented mode. It never reads HAR data and only labels a
    TCP flow when its ClientHello SNI is visible in the capture.
    """
    out = []
    by_flow = _flow_sni_roles(pcap, primary_host=primary_host, feature_mode=feature_mode)
    for p in pcap.packets:
        host_role = by_flow.get(p.flow_id)
        if host_role:
            p.host, p.role = host_role
        else:
            p.host = ""
            p.role = "unknown"
        if include_unknown or p.role != "unknown":
            out.append(p)
    return out


def assign_packet_roles_from_sni_dns(
    pcap: PcapData,
    primary_host: str = "",
    feature_mode: str = "clean",
    include_unknown: bool = False,
) -> List[Packet]:
    """Assign roles from SNI first, then passive DNS answers observed in PCAP."""
    out = []
    by_flow = _flow_sni_roles(pcap, primary_host=primary_host, feature_mode=feature_mode)
    for p in pcap.packets:
        host_role = by_flow.get(p.flow_id) or _dns_role_for_packet(
            pcap,
            p,
            primary_host=primary_host,
            feature_mode=feature_mode,
        )
        if host_role:
            p.host, p.role = host_role
        else:
            p.host = ""
            p.role = "unknown"
        if include_unknown or p.role != "unknown":
            out.append(p)
    return out


def assign_packet_roles_har_then_sni(
    pcap: PcapData,
    har_info,
    primary_host: str = "",
    feature_mode: str = "clean",
    include_unknown: bool = False,
) -> List[Packet]:
    """Use HAR mapping when available, falling back to SNI for unmatched flows."""
    out = []
    by_flow = _flow_sni_roles(pcap, primary_host=primary_host, feature_mode=feature_mode)
    for p in pcap.packets:
        server_ip = _server_ip_for_packet(p)
        hosts = har_info.ip_to_hosts.get(server_ip, set())
        if hosts:
            roles = [har_info.host_to_role.get(h, "unknown") for h in hosts]
            p.role = merge_roles(roles)
            p.host = sorted(hosts)[0] if len(hosts) == 1 else "|".join(sorted(list(hosts))[:3])
        else:
            host_role = by_flow.get(p.flow_id)
            if host_role:
                p.host, p.role = host_role
            else:
                p.role = "unknown"
                p.host = ""
        if include_unknown or p.role != "unknown":
            out.append(p)
    return out


def assign_packet_roles(pcap: PcapData, har_info, include_unknown: bool = False) -> List[Packet]:
    """Backward-compatible alias for HAR-based role assignment."""
    return assign_packet_roles_from_har(pcap, har_info, include_unknown=include_unknown)


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def _stats(prefix: str, vals: Sequence[float]) -> Dict[str, float]:
    vals = [float(v) for v in vals if v is not None and not math.isnan(float(v))]
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



def _dual_view_prefix_features(pkts: Sequence[Packet], start: float, end: float, cfg: dict) -> Dict[str, float]:
    """Role-free MRCH-style lightweight length/rhythm views."""
    out: Dict[str, float] = {}
    max_n = int(cfg.get("max_prefix_packets", 40) or 40)
    selected = sorted(pkts, key=lambda p: p.ts)[:max(1, max_n)]
    if not selected:
        for prefix in ["view_len_log", "view_len_delta", "view_len_offset", "view_rhythm_rel_iat", "view_rhythm_cum_rel_time"]:
            out.update(_stats(prefix, []))
        out["view_cross_len_iat_corr"] = 0.0
        out["view_cross_dir_switch_rate"] = 0.0
        out["view_cross_up_then_down_rate"] = 0.0
        return out
    lengths = np.asarray([float(max(0, int(p.payload_len if p.payload_len else p.length))) for p in selected], dtype=float)
    log_len = np.log1p(lengths)
    len_delta = np.diff(log_len, prepend=log_len[0])
    med = np.asarray([float(np.median(log_len[: i + 1])) for i in range(len(log_len))], dtype=float)
    len_offset = log_len - med
    ts = np.asarray([float(p.ts) for p in selected], dtype=float)
    iat = np.maximum(np.diff(ts, prepend=float(start)), 0.0)
    pos = iat[iat > 1e-6]
    scale = float(np.median(pos)) if len(pos) else 1.0
    rel_iat = np.clip(iat / max(scale, 1e-6), 0.0, 20.0)
    cum_rel = np.clip((ts - float(start)) / max(float(end - start), 1e-6), 0.0, 1.0)
    dirs = np.asarray([1.0 if p.direction == "up" else -1.0 if p.direction == "down" else 0.0 for p in selected], dtype=float)
    out.update(_stats("view_len_log", log_len.tolist()))
    out.update(_stats("view_len_delta", len_delta.tolist()))
    out.update(_stats("view_len_offset", len_offset.tolist()))
    out.update(_stats("view_rhythm_rel_iat", rel_iat.tolist()))
    out.update(_stats("view_rhythm_cum_rel_time", cum_rel.tolist()))
    out["view_cross_len_iat_corr"] = float(np.corrcoef(log_len, rel_iat)[0, 1]) if len(log_len) >= 3 and np.std(log_len) > 1e-9 and np.std(rel_iat) > 1e-9 else 0.0
    out["view_cross_dir_switch_rate"] = _safe_div(float(np.sum(dirs[1:] != dirs[:-1])) if len(dirs) > 1 else 0.0, max(len(dirs) - 1, 1))
    out["view_cross_up_then_down_rate"] = _safe_div(float(np.sum((dirs[:-1] > 0) & (dirs[1:] < 0))) if len(dirs) > 1 else 0.0, max(len(dirs) - 1, 1))
    sample_n = int(cfg.get("max_view_prefix_positions", 12) or 12)
    for i in range(sample_n):
        out[f"view_prefix_dir_{i:02d}"] = float(dirs[i]) if i < len(dirs) else 0.0
        out[f"view_prefix_log_len_{i:02d}"] = float(log_len[i]) if i < len(log_len) else 0.0
        out[f"view_prefix_rel_iat_{i:02d}"] = float(rel_iat[i]) if i < len(rel_iat) else 0.0
    return out



ROLE_GROUPS = {
    "interaction": (
        "unknown", "first_party_site", "third_party_backend_or_other",
        "identity_provider", "captcha_challenge", "payment_provider",
        "messaging_api", "form_backend", "cloud_api", "object_storage",
        "email_delivery", "file_storage", "url_shortener", "hosting_platform",
        "rpc_provider", "walletconnect",
    ),
    "credential_auth": ("identity_provider", "form_backend", "first_party_site"),
    "challenge_gate": ("captcha_challenge",),
    "payment_checkout": ("payment_provider",),
    "exfil_delivery": ("messaging_api", "form_backend", "cloud_api", "object_storage", "email_delivery", "file_storage"),
    "crypto_transaction": ("rpc_provider", "walletconnect"),
    "kit_hosting": ("hosting_platform", "first_party_site"),
    "backend_unknown": ("third_party_backend_or_other", "unknown"),
    "nuisance": ("wallet_vendor", "analytics_ads", "third_party_static", "software_update"),
}


def _counter_sum(counter: Counter, roles: Iterable[str]) -> int:
    return int(sum(counter[str(r)] for r in roles))


def _add_role_group_features(prefix: str, out: Dict[str, float], role_byte: Counter, role_pkt: Counter, role_flows: Dict[str, set], total_bytes: float, total_pkts: int, total_flows: int) -> None:
    for group, members in ROLE_GROUPS.items():
        b = _counter_sum(role_byte, members)
        p = _counter_sum(role_pkt, members)
        fs = set()
        for m in members:
            fs.update(role_flows.get(m, set()))
        out[f"{prefix}{group}_byte_frac"] = _safe_div(b, total_bytes)
        out[f"{prefix}{group}_pkt_frac"] = _safe_div(p, total_pkts)
        out[f"{prefix}{group}_flow_frac"] = _safe_div(len(fs), total_flows)


def _dominant_stage_from_role_bytes(role_byte: Counter, total_bytes: float) -> tuple[str, float, int]:
    stage_groups = [
        ("crypto_transaction", ROLE_GROUPS["crypto_transaction"]),
        ("credential_auth", ("identity_provider", "form_backend")),
        ("payment_checkout", ROLE_GROUPS["payment_checkout"]),
        ("challenge_gate", ROLE_GROUPS["challenge_gate"]),
        ("exfil_delivery", ("messaging_api", "form_backend", "cloud_api", "object_storage", "email_delivery", "file_storage")),
        ("kit_hosting", ROLE_GROUPS["kit_hosting"]),
        ("backend_unknown", ROLE_GROUPS["backend_unknown"]),
        ("nuisance", ROLE_GROUPS["nuisance"]),
    ]
    codes = {name: i + 1 for i, (name, _) in enumerate(stage_groups)}
    best_name, best_val = "none", 0.0
    for name, members in stage_groups:
        frac = _safe_div(_counter_sum(role_byte, members), total_bytes)
        if frac > best_val:
            best_name, best_val = name, frac
    return best_name, float(best_val), int(codes.get(best_name, 0))


def _basic_counts(pkts: Sequence[Packet], start: float, end: float, cfg: dict, roles: Sequence[str]) -> Dict[str, float]:
    f: Dict[str, float] = {}
    dur = max(0.0, end - start)
    n = len(pkts)
    f["base_n_pkts"] = float(n)
    f["base_window_s"] = dur
    if n == 0:
        # Fill common empty features.
        f.update({
            "base_n_flows": 0.0, "base_n_server_ips": 0.0,
            "base_bytes_total": 0.0, "base_bytes_up": 0.0, "base_bytes_down": 0.0,
            "base_pkts_up": 0.0, "base_pkts_down": 0.0,
            "base_ud_byte_ratio": 0.0, "base_ud_pkt_ratio": 0.0,
            "base_packet_rate": 0.0, "base_byte_rate": 0.0,
            "base_active_seconds": 0.0, "base_max_idle_gap": dur,
        })
        for side in ["all", "up", "down"]:
            f.update(_stats(f"base_sz_{side}", []))
        f.update(_stats("base_iat", []))
        f.update({"base_n_bursts": 0.0, "base_burst_sz_mean": 0.0, "base_burst_sz_max": 0.0})
        for r in roles:
            f[f"role_{r}_byte_frac"] = 0.0
            f[f"role_{r}_pkt_frac"] = 0.0
            f[f"role_{r}_flow_frac"] = 0.0
        for group in ROLE_GROUPS:
            f[f"role_group_{group}_byte_frac"] = 0.0
            f[f"role_group_{group}_pkt_frac"] = 0.0
            f[f"role_group_{group}_flow_frac"] = 0.0
        f["role_stage_dominant_frac"] = 0.0
        f["role_stage_dominant_code"] = 0.0
        return f

    bytes_total = sum(p.length for p in pkts)
    up = [p for p in pkts if p.direction == "up"]
    down = [p for p in pkts if p.direction == "down"]
    bytes_up = sum(p.length for p in up)
    bytes_down = sum(p.length for p in down)
    flows = set(p.flow_id for p in pkts)
    server_ips = set((p.dst_ip if p.direction == "up" else p.src_ip) for p in pkts)
    f.update({
        "base_n_flows": float(len(flows)),
        "base_n_server_ips": float(len(server_ips)),
        "base_bytes_total": float(bytes_total),
        "base_bytes_up": float(bytes_up),
        "base_bytes_down": float(bytes_down),
        "base_pkts_up": float(len(up)),
        "base_pkts_down": float(len(down)),
        "base_ud_byte_ratio": _safe_div(bytes_up, bytes_down),
        "base_ud_pkt_ratio": _safe_div(len(up), len(down)),
        "base_packet_rate": _safe_div(n, dur),
        "base_byte_rate": _safe_div(bytes_total, dur),
    })
    active_bin = float(cfg.get("active_bin_s", 1.0))
    bins = set(int((p.ts - start) // active_bin) for p in pkts if p.ts >= start)
    f["base_active_seconds"] = float(len(bins) * active_bin)
    ts = [p.ts for p in pkts]
    iats = np.diff(ts).tolist() if len(ts) > 1 else []
    f["base_max_idle_gap"] = float(max(iats) if iats else dur)
    for side, arr in [("all", pkts), ("up", up), ("down", down)]:
        f.update(_stats(f"base_sz_{side}", [p.length for p in arr]))
    f.update(_stats("base_iat", iats))
    large_thr = float(cfg.get("large_packet_threshold", 1000))
    small_thr = float(cfg.get("small_packet_threshold", 80))
    f["base_large_pkt_frac"] = _safe_div(sum(1 for p in pkts if p.length >= large_thr), n)
    f["base_small_pkt_frac"] = _safe_div(sum(1 for p in pkts if p.length <= small_thr), n)
    bursts = build_bursts(pkts, float(cfg.get("burst_gap_s", 0.05)))
    burst_sizes = [b.n_pkts for b in bursts]
    burst_bytes = [b.bytes_total for b in bursts]
    f["base_n_bursts"] = float(len(bursts))
    f["base_burst_sz_mean"] = float(np.mean(burst_sizes)) if burst_sizes else 0.0
    f["base_burst_sz_max"] = float(max(burst_sizes)) if burst_sizes else 0.0
    f["base_burst_bytes_max"] = float(max(burst_bytes)) if burst_bytes else 0.0
    f["base_burst_bytes_mean"] = float(np.mean(burst_bytes)) if burst_bytes else 0.0

    role_byte = Counter()
    role_pkt = Counter()
    role_flows = defaultdict(set)
    for p in pkts:
        r = p.role or "unknown"
        role_byte[r] += p.length
        role_pkt[r] += 1
        role_flows[r].add(p.flow_id)
    for r in roles:
        f[f"role_{r}_byte_frac"] = _safe_div(role_byte[r], bytes_total)
        f[f"role_{r}_pkt_frac"] = _safe_div(role_pkt[r], n)
        f[f"role_{r}_flow_frac"] = _safe_div(len(role_flows[r]), len(flows))
    _add_role_group_features("role_group_", f, role_byte, role_pkt, role_flows, bytes_total, n, len(flows))
    stage_name, stage_frac, stage_code = _dominant_stage_from_role_bytes(role_byte, bytes_total)
    f["role_stage_dominant_frac"] = stage_frac
    f["role_stage_dominant_code"] = float(stage_code)
    probs = [role_byte[r] / bytes_total for r in role_byte if bytes_total > 0 and role_byte[r] > 0]
    f["role_entropy_bytes"] = float(-sum(p * math.log(p + 1e-12) for p in probs))
    f["role_dominant_byte_frac"] = float(max(probs) if probs else 0.0)
    f["role_rpc_to_first_party_byte_ratio"] = _safe_div(role_byte["rpc_provider"], role_byte["first_party_site"])
    f["role_backend_to_first_party_byte_ratio"] = _safe_div(role_byte["third_party_backend_or_other"], role_byte["first_party_site"])
    f["role_static_to_first_party_byte_ratio"] = _safe_div(role_byte["third_party_static"], role_byte["first_party_site"])
    f["role_exfil_to_nuisance_byte_ratio"] = _safe_div(
        _counter_sum(role_byte, ROLE_GROUPS["exfil_delivery"]),
        _counter_sum(role_byte, ROLE_GROUPS["nuisance"]),
    )
    f["role_interaction_to_nuisance_byte_ratio"] = _safe_div(
        _counter_sum(role_byte, ROLE_GROUPS["interaction"]),
        _counter_sum(role_byte, ROLE_GROUPS["nuisance"]),
    )

    # Protocol features.
    tcp = [p for p in pkts if p.proto == "TCP"]
    udp443 = [p for p in pkts if p.is_udp443]
    f["proto_tcp_byte_frac"] = _safe_div(sum(p.length for p in tcp), bytes_total)
    f["proto_tcp_pkt_frac"] = _safe_div(len(tcp), n)
    f["proto_udp443_byte_frac"] = _safe_div(sum(p.length for p in udp443), bytes_total)
    f["proto_udp443_pkt_frac"] = _safe_div(len(udp443), n)
    for r in ["first_party_site", "third_party_backend_or_other", "rpc_provider", "third_party_static"]:
        rb = sum(p.length for p in udp443 if p.role == r)
        f[f"proto_udp443_{r}_byte_frac"] = _safe_div(rb, bytes_total)

    # Flow stats.
    by_flow: Dict[Tuple, List[Packet]] = defaultdict(list)
    for p in pkts:
        by_flow[p.flow_id].append(p)
    flow_pkts, flow_bytes, flow_durs = [], [], []
    for fl, fps in by_flow.items():
        flow_pkts.append(len(fps))
        flow_bytes.append(sum(p.length for p in fps))
        flow_durs.append(max(p.ts for p in fps) - min(p.ts for p in fps) if len(fps) > 1 else 0.0)
    f.update(_stats("flow_pkts", flow_pkts))
    f.update(_stats("flow_bytes", flow_bytes))
    f.update(_stats("flow_dur", flow_durs))
    # Concurrent flows by 1s bucket.
    bucket_flows = defaultdict(set)
    for fl, fps in by_flow.items():
        a = int((min(p.ts for p in fps) - start) // 1.0)
        b = int((max(p.ts for p in fps) - start) // 1.0)
        for bucket in range(max(0, a), max(0, b) + 1):
            bucket_flows[bucket].add(fl)
    conc = [len(v) for v in bucket_flows.values()]
    f["flow_max_concurrent_1s"] = float(max(conc) if conc else 0.0)
    f["flow_mean_concurrent_1s"] = float(np.mean(conc) if conc else 0.0)

    # RPC-specific.
    rpc = [p for p in pkts if p.role == "rpc_provider"]
    rpc_bytes = sum(p.length for p in rpc)
    rpc_ts = [p.ts for p in rpc]
    rpc_bins = set(int((p.ts - start) // active_bin) for p in rpc)
    f["rpc_byte_frac"] = _safe_div(rpc_bytes, bytes_total)
    f["rpc_pkt_frac"] = _safe_div(len(rpc), n)
    f["rpc_flow_frac"] = _safe_div(len(set(p.flow_id for p in rpc)), len(flows))
    f["rpc_active_seconds"] = float(len(rpc_bins) * active_bin)
    f["rpc_persistence_score"] = f["rpc_byte_frac"] * _safe_div(f["rpc_active_seconds"], dur) * math.log1p(len(rpc)) if dur > 0 else 0.0
    f["rpc_first_delay_s"] = float(min(rpc_ts) - start) if rpc_ts else -1.0
    f.update(_stats("rpc_iat", np.diff(rpc_ts).tolist() if len(rpc_ts) > 1 else []))
    return f


def _phase_features(pkts: Sequence[Packet], start: float, cfg: dict, roles: Sequence[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    phases = cfg.get("phases", [[0, 3], [3, 10], [10, 30], [30, 60], [60, 90]])
    for a, b in phases:
        sub = [p for p in pkts if start + float(a) <= p.ts < start + float(b)]
        prefix = f"phase_{int(a)}_{int(b)}"
        n = len(sub)
        total = sum(p.length for p in sub)
        out[f"{prefix}_n_pkts"] = float(n)
        out[f"{prefix}_bytes_total"] = float(total)
        out[f"{prefix}_large_pkt_frac"] = _safe_div(sum(1 for p in sub if p.length >= cfg.get("large_packet_threshold", 1000)), n)
        bursts = build_bursts(sub, float(cfg.get("burst_gap_s", 0.05))) if sub else []
        out[f"{prefix}_burst_sz_max"] = float(max([bb.n_pkts for bb in bursts]) if bursts else 0.0)
        for r in roles:
            rb = sum(p.length for p in sub if p.role == r)
            out[f"{prefix}_{r}_byte_frac"] = _safe_div(rb, total)
            out[f"{prefix}_{r}_bytes"] = float(rb)
    # Convenience aliases used in analysis.  Generic interaction aliases are
    # the primary Layer-1 names; subtype-specific fields remain for compatibility.
    out["phase_immediate_rpc_burst_bytes_0_3s"] = out.get("phase_0_3_rpc_provider_bytes", 0.0)
    out["phase_immediate_backend_burst_bytes_0_3s"] = out.get("phase_0_3_third_party_backend_or_other_bytes", 0.0)
    out["phase_immediate_identity_bytes_0_3s"] = out.get("phase_0_3_identity_provider_bytes", 0.0)
    out["phase_immediate_captcha_bytes_0_3s"] = out.get("phase_0_3_captcha_challenge_bytes", 0.0)
    out["phase_immediate_payment_bytes_0_3s"] = out.get("phase_0_3_payment_provider_bytes", 0.0)
    out["phase_immediate_messaging_bytes_0_3s"] = out.get("phase_0_3_messaging_api_bytes", 0.0)
    out["phase_interaction_backendlike_bytes_0_3s"] = (
        out.get("phase_0_3_unknown_bytes", 0.0)
        + out.get("phase_0_3_third_party_backend_or_other_bytes", 0.0)
        + out.get("phase_0_3_identity_provider_bytes", 0.0)
        + out.get("phase_0_3_payment_provider_bytes", 0.0)
        + out.get("phase_0_3_messaging_api_bytes", 0.0)
        + out.get("phase_0_3_rpc_provider_bytes", 0.0)
        + out.get("phase_0_3_walletconnect_bytes", 0.0)
    )
    out["phase_interaction_exfil_bytes_0_3s"] = (
        out.get("phase_0_3_messaging_api_bytes", 0.0)
        + out.get("phase_0_3_form_backend_bytes", 0.0)
        + out.get("phase_0_3_cloud_api_bytes", 0.0)
        + out.get("phase_0_3_object_storage_bytes", 0.0)
        + out.get("phase_0_3_email_delivery_bytes", 0.0)
        + out.get("phase_0_3_file_storage_bytes", 0.0)
    )
    out["phase_interaction_auth_challenge_payment_bytes_0_3s"] = (
        out.get("phase_0_3_identity_provider_bytes", 0.0)
        + out.get("phase_0_3_captcha_challenge_bytes", 0.0)
        + out.get("phase_0_3_payment_provider_bytes", 0.0)
    )
    out["phase_interaction_crypto_bytes_0_3s"] = (
        out.get("phase_0_3_rpc_provider_bytes", 0.0)
        + out.get("phase_0_3_walletconnect_bytes", 0.0)
    )
    out["phase_interaction_nuisance_bytes_0_3s"] = (
        out.get("phase_0_3_wallet_vendor_bytes", 0.0)
        + out.get("phase_0_3_third_party_static_bytes", 0.0)
        + out.get("phase_0_3_analytics_ads_bytes", 0.0)
        + out.get("phase_0_3_software_update_bytes", 0.0)
    )
    out["phase_delayed_static_burst_bytes_10_30s"] = out.get("phase_10_30_third_party_static_bytes", 0.0)
    out["phase_late_rpc_bytes_30_90s"] = out.get("phase_30_60_rpc_provider_bytes", 0.0) + out.get("phase_60_90_rpc_provider_bytes", 0.0)
    early = out.get("phase_0_3_bytes_total", 0.0) + out.get("phase_3_10_bytes_total", 0.0) + out.get("phase_10_30_bytes_total", 0.0)
    late = out.get("phase_30_60_bytes_total", 0.0) + out.get("phase_60_90_bytes_total", 0.0)
    out["phase_activity_decay_30_90_over_0_30"] = _safe_div(late, early)
    return out


def extract_window_features(
    packets: Sequence[Packet],
    window: Window,
    cfg: dict,
    roles: Optional[Sequence[str]] = None,
    capture_end: Optional[float] = None,
    shape_index: FlowShapeIndex | None = None,
) -> Dict[str, float | str]:
    roles = list(roles or cfg.get("roles") or [])
    start, end = window.start, window.end
    pkts = [p for p in packets if start <= p.ts <= end]
    out: Dict[str, float | str] = {
        "window_name": window.name,
        "window_start_epoch": start,
        "window_end_epoch": end,
        "window_end_reason": window.reason,
        "window_s": max(0.0, end - start),
        "coverage_s": max(0.0, end - start),
        "quality_capture_duration_after_anchor_s": max(0.0, (capture_end or end) - start),
        "quality_is_shorter_than_60s": 1.0 if max(0.0, end - start) < 60 else 0.0,
        "quality_n_packets_in_window": float(len(pkts)),
    }
    out.update(_basic_counts(pkts, start, end, cfg, roles))
    out.update(_dual_view_prefix_features(pkts, start, end, cfg))
    out.update(aggregate_flow_segment_features(pkts, start=start, end=end, burst_gap_s=float(cfg.get("burst_gap_s", 0.05) or 0.05)))
    out.update(window_flow_shape_features(shape_index if shape_index is not None else packets, start=start, end=end, pre_s=float(cfg.get("shape_pre_window_s", 6.0) or 6.0), burst_gap_s=float(cfg.get("shape_burst_gap_s", cfg.get("burst_gap_s", 0.35)) or cfg.get("burst_gap_s", 0.35))))
    out.update(_phase_features(pkts, start, cfg, roles))
    return out

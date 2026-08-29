from __future__ import annotations

import glob
import hashlib
import json
import math
import pathlib
import re
from urllib.parse import urlparse
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

from .candidates import CandidateAnchor, generate_candidate_anchors, make_oracle_candidate, resolve_post_load_guard
from .config import load_config
from .dynamic_window import Window, adaptive_episode_window, dynamic_episode_window, fixed_windows
from .flow_shape import build_flow_shape_index
from .features import (
    assign_packet_roles_from_har,
    assign_packet_roles_from_sni,
    assign_packet_roles_from_sni_dns,
    assign_packet_roles_har_then_sni,
    dns_observation_counts,
    extract_window_features,
    sni_observation_counts,
)
from .har import HarInfo
from .pcap_minimal import PcapData, read_pcap
from .zipio import extract_zip_sample, load_har_for_sample, load_sample_meta


def _stable_unit(seed: str, salt: str) -> float:
    """Deterministic [0, 1) hash for reproducible per-flow/per-host sampling."""
    h = hashlib.sha1(f"{salt}|{seed}".encode("utf-8", errors="ignore")).digest()
    return int.from_bytes(h[:8], "big") / float(1 << 64)


def unify_transport_stream(pcap: PcapData, mode: str, quic_control_max_bytes: int = 48) -> Dict:
    """Make TCP443 and QUIC/UDP443 traffic comparable BEFORE feature computation.

    Whether a flow rides QUIC or TCP+TLS is decided by the capture path (proxy
    UDP443 forwarding), not by the kit, so flow-shape features inherit a capture
    -environment fingerprint. This transform (side-path, opt-in via
    ``filtering.transport_unify``) removes the transport-specific artifacts at
    the packet-stream level so every downstream feature is computed on a
    transport-neutral stream:

    mode "norm"
      - drop TCP443 transport-control packets (payload_len == 0: pure
        ACK/SYN/FIN/RST) and UDP443 datagrams < quic_control_max_bytes
        (QUIC ACK/control-only) -> data-packet timing/counts on both stacks
      - rewrite packet ``length`` to the L4 payload for 443 traffic ->
        removes the TCP/UDP header and re-segmentation size offsets
    mode "norm_merge"
      - additionally collapse every 443 flow between the same endpoint pair
        into one logical flow id (unifies h2-vs-h3 connection racing)

    Mutates ``pcap.packets`` in place, like apply_observability_degradation.
    Must run before role assignment. Returns a stats dict for auditing.
    """
    mode = str(mode or "none").lower()
    stats = {
        "transport_unify_mode": mode,
        "transport_unify_dropped_tcp_ctrl": 0,
        "transport_unify_dropped_quic_ctrl": 0,
        "transport_unify_rewritten": 0,
        "transport_unify_merged_flows": 0,
    }
    if mode == "none":
        return stats
    if mode not in {"norm", "norm_merge"}:
        raise ValueError(f"unknown filtering.transport_unify={mode!r}; expected none, norm, or norm_merge")
    kept = []
    for p in pcap.packets:
        on443 = p.src_port == 443 or p.dst_port == 443
        if p.proto == "TCP" and on443:
            if int(p.payload_len or 0) <= 0:
                stats["transport_unify_dropped_tcp_ctrl"] += 1
                continue
            p.length = int(p.payload_len)
            stats["transport_unify_rewritten"] += 1
        elif p.is_udp443:
            if int(p.payload_len or 0) < int(quic_control_max_bytes):
                stats["transport_unify_dropped_quic_ctrl"] += 1
                continue
            p.length = int(p.payload_len)
            stats["transport_unify_rewritten"] += 1
        if mode == "norm_merge" and on443 and p.proto in ("TCP", "UDP"):
            lo, hi = sorted((p.src_ip, p.dst_ip))
            merged = ("ENC443", lo, hi)
            if p.flow_id != merged:
                stats["transport_unify_merged_flows"] += 1
            p.flow_id = merged
        kept.append(p)
    pcap.packets = kept
    return stats


def apply_observability_degradation(pcap: PcapData, degradation_cfg: Optional[Dict]) -> Dict:
    """Simulate future gateway observability loss (ECH, Encrypted DNS).

    The function mutates ``pcap.packets`` and ``pcap.dns_ip_to_hosts`` in place
    to model the encrypted-traffic-analysis residual once SNI and/or passive DNS
    are no longer visible to the gateway.  It must run before any role-assignment
    function, otherwise role labels will already encode the now-hidden hints.

    Supported keys
    --------------
    hide_sni : bool
        Drop the TLS ClientHello SNI from every packet (ECH worst case).
    sni_visible_rate : float in [0, 1]
        Fraction of flows that still leak SNI (e.g. ECH GREASE / non-compliant
        endpoints).  Selection is deterministic per flow id.  Ignored if
        hide_sni is false.
    hide_dns : bool
        Hide passive DNS A/AAAA answers (Encrypted DNS / DoH / DoT worst case).
    dns_visible_rate : float in [0, 1]
        Fraction of resolver answers that still leak (split-horizon DoH
        fallback or plaintext A/AAAA leakage).
    seed : str
        Salt for the deterministic sampling, so repeated runs are reproducible.

    Returns a small stats dict that is merged into every output row, so any
    downstream comparison can verify the requested degradation actually took
    effect on a sample.
    """
    cfg = dict(degradation_cfg or {})
    hide_sni = bool(cfg.get("hide_sni", False))
    hide_dns = bool(cfg.get("hide_dns", False))
    sni_visible_rate = float(cfg.get("sni_visible_rate", 0.0) or 0.0)
    dns_visible_rate = float(cfg.get("dns_visible_rate", 0.0) or 0.0)
    seed = str(cfg.get("seed", "ech-edns-default"))

    stats = {
        "degradation_hide_sni": int(bool(hide_sni)),
        "degradation_hide_dns": int(bool(hide_dns)),
        "degradation_sni_visible_rate": float(max(0.0, min(1.0, sni_visible_rate))),
        "degradation_dns_visible_rate": float(max(0.0, min(1.0, dns_visible_rate))),
        "degradation_sni_dropped_pkts": 0,
        "degradation_sni_kept_pkts": 0,
        "degradation_dns_dropped_resolvers": 0,
        "degradation_dns_kept_resolvers": 0,
    }

    if hide_sni:
        keep_rate = max(0.0, min(1.0, sni_visible_rate))
        flow_keep: Dict = {}
        dropped = 0
        kept = 0
        for p in pcap.packets:
            if not p.sni:
                continue
            flow_key = p.flow_id or (p.src_ip, p.src_port, p.dst_ip, p.dst_port)
            decision = flow_keep.get(flow_key)
            if decision is None:
                decision = _stable_unit(repr(flow_key), seed + "|sni") < keep_rate
                flow_keep[flow_key] = decision
            if decision:
                kept += 1
            else:
                p.sni = ""
                dropped += 1
        stats["degradation_sni_dropped_pkts"] = int(dropped)
        stats["degradation_sni_kept_pkts"] = int(kept)

    if hide_dns and pcap.dns_ip_to_hosts:
        keep_rate = max(0.0, min(1.0, dns_visible_rate))
        kept_map: Dict[str, set] = {}
        dropped = 0
        kept = 0
        for server_ip, hosts in pcap.dns_ip_to_hosts.items():
            if _stable_unit(server_ip, seed + "|dns") < keep_rate:
                kept_map[server_ip] = hosts
                kept += 1
            else:
                dropped += 1
        pcap.dns_ip_to_hosts = kept_map
        stats["degradation_dns_dropped_resolvers"] = int(dropped)
        stats["degradation_dns_kept_resolvers"] = int(kept)

    return stats


def _normalise_role_list(value) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        items = value.split(",")
    else:
        items = list(value)
    return {str(item).strip().lower() for item in items if str(item).strip()}


def _packet_server_ip(packet) -> str:
    if packet.direction == "up":
        return packet.dst_ip
    if packet.direction == "down":
        return packet.src_ip
    return packet.dst_ip


def suppress_packets_by_role(packets: Sequence, roles: Iterable[str]) -> tuple[list, dict]:
    """Remove role-labelled traffic using only gateway-observable host roles."""
    suppressed_roles = _normalise_role_list(roles)
    if not suppressed_roles:
        return list(packets), {
            "filter_suppressed_roles": "",
            "filter_suppressed_n_pkts": 0,
            "filter_suppressed_bytes_total": 0,
            "filter_suppressed_n_flows": 0,
            "filter_suppressed_n_server_ips": 0,
        }
    kept = []
    dropped = []
    for packet in packets:
        role = str(packet.role or "unknown").lower()
        if role in suppressed_roles:
            dropped.append(packet)
        else:
            kept.append(packet)
    return kept, {
        "filter_suppressed_roles": ",".join(sorted(suppressed_roles)),
        "filter_suppressed_n_pkts": int(len(dropped)),
        "filter_suppressed_bytes_total": int(sum(p.length for p in dropped)),
        "filter_suppressed_n_flows": int(len({p.flow_id for p in dropped})),
        "filter_suppressed_n_server_ips": int(len({_packet_server_ip(p) for p in dropped})),
    }


def _host_parts(host: str) -> list[str]:
    return [part.strip().lower() for part in str(host or "").split("|") if part.strip()]


def load_har_request_hosts(har_path: str | None) -> set[str]:
    """Return normalized request hosts from a browser HAR file."""
    if not har_path:
        return set()
    try:
        with open(har_path, "r", encoding="utf-8-sig") as f:
            obj = json.load(f)
    except Exception:
        return set()
    entries = ((obj.get("log") or {}).get("entries") or []) if isinstance(obj, dict) else []
    hosts: set[str] = set()
    for entry in entries:
        req = (entry or {}).get("request") or {}
        url = str(req.get("url") or "")
        try:
            host = (urlparse(url).hostname or "").strip().lower().strip(".")
        except Exception:
            host = ""
        if host:
            hosts.add(host)
    return hosts


def _host_in_reference(host: str, reference_hosts: set[str]) -> bool:
    host = str(host or "").strip().lower().strip(".")
    if not host or not reference_hosts:
        return False
    if host in reference_hosts:
        return True
    return any(host.endswith("." + ref) for ref in reference_hosts if ref)


def suppress_packets_by_host_pattern(
    packets: Sequence,
    patterns: Iterable[str],
    skip_roles: Iterable[str] = ("first_party_site",),
    keep_hosts: Iterable[str] | None = None,
) -> tuple[list, dict]:
    raw_patterns = [str(pattern).strip() for pattern in (patterns or []) if str(pattern).strip()]
    skip_role_set = _normalise_role_list(skip_roles)
    keep_host_set = {str(host).strip().lower().strip(".") for host in (keep_hosts or []) if str(host).strip()}
    if not raw_patterns:
        return list(packets), {
            "filter_suppressed_host_patterns": "",
            "filter_suppressed_host_skip_roles": ",".join(sorted(skip_role_set)),
            "filter_suppressed_host_keep_hosts": int(bool(keep_host_set)),
            "filter_host_suppressed_n_pkts": 0,
            "filter_host_suppressed_bytes_total": 0,
            "filter_host_suppressed_n_flows": 0,
            "filter_host_suppressed_n_server_ips": 0,
        }
    compiled = [re.compile(pattern, re.I) for pattern in raw_patterns]
    kept = []
    dropped = []
    for packet in packets:
        role = str(packet.role or "unknown").lower()
        if role in skip_role_set:
            kept.append(packet)
            continue
        hosts = _host_parts(packet.host)
        host_matches = any(regex.search(host) for host in hosts for regex in compiled)
        if host_matches:
            if keep_host_set and any(_host_in_reference(host, keep_host_set) for host in hosts):
                kept.append(packet)
            else:
                dropped.append(packet)
        else:
            kept.append(packet)
    return kept, {
        "filter_suppressed_host_patterns": " | ".join(raw_patterns),
        "filter_suppressed_host_skip_roles": ",".join(sorted(skip_role_set)),
        "filter_suppressed_host_keep_hosts": int(bool(keep_host_set)),
        "filter_host_suppressed_n_pkts": int(len(dropped)),
        "filter_host_suppressed_bytes_total": int(sum(p.length for p in dropped)),
        "filter_host_suppressed_n_flows": int(len({p.flow_id for p in dropped})),
        "filter_host_suppressed_n_server_ips": int(len({_packet_server_ip(p) for p in dropped})),
    }


def suppress_packets_by_role_unless_har_host(
    packets: Sequence,
    roles: Iterable[str],
    har_hosts: Iterable[str],
    skip_roles: Iterable[str] = ("first_party_site",),
) -> tuple[list, dict]:
    """Suppress role-labelled traffic only when its host is absent from HAR."""
    suppressed_roles = _normalise_role_list(roles)
    skip_role_set = _normalise_role_list(skip_roles)
    har_host_set = {str(host).strip().lower().strip(".") for host in (har_hosts or []) if str(host).strip()}
    if not suppressed_roles:
        return list(packets), {
            "filter_har_guarded_roles": "",
            "filter_har_guarded_n_pkts": 0,
            "filter_har_guarded_bytes_total": 0,
            "filter_har_guarded_n_flows": 0,
            "filter_har_guarded_n_server_ips": 0,
            "filter_har_guarded_har_hosts": int(len(har_host_set)),
        }
    kept = []
    dropped = []
    for packet in packets:
        role = str(packet.role or "unknown").lower()
        if role in skip_role_set or role not in suppressed_roles:
            kept.append(packet)
            continue
        hosts = _host_parts(packet.host)
        if hosts and har_host_set and any(_host_in_reference(host, har_host_set) for host in hosts):
            kept.append(packet)
        else:
            dropped.append(packet)
    return kept, {
        "filter_har_guarded_roles": ",".join(sorted(suppressed_roles)),
        "filter_har_guarded_n_pkts": int(len(dropped)),
        "filter_har_guarded_bytes_total": int(sum(p.length for p in dropped)),
        "filter_har_guarded_n_flows": int(len({p.flow_id for p in dropped})),
        "filter_har_guarded_n_server_ips": int(len({_packet_server_ip(p) for p in dropped})),
        "filter_har_guarded_har_hosts": int(len(har_host_set)),
    }


def drop_rpc_fanout_flows(
    packets: Sequence,
    window_s: float = 10.0,
    min_flows: int = 3,
    min_hosts: int = 2,
) -> tuple[list, dict]:
    """Drop short bursts of many public-RPC flows, a common wallet-refresh shape."""
    flow_first: dict = {}
    flow_hosts: dict = {}
    flow_roles: dict = {}
    for packet in packets:
        flow_first[packet.flow_id] = min(flow_first.get(packet.flow_id, packet.ts), packet.ts)
        flow_hosts.setdefault(packet.flow_id, set()).update(_host_parts(packet.host))
        flow_roles.setdefault(packet.flow_id, set()).add(str(packet.role or "unknown").lower())

    rpc_flows = [
        (flow_id, float(flow_first[flow_id]), flow_hosts.get(flow_id, set()))
        for flow_id, roles in flow_roles.items()
        if "rpc_provider" in roles
    ]
    rpc_flows.sort(key=lambda item: item[1])
    marked = set()
    span = max(0.0, float(window_s or 0.0))
    min_flows = max(1, int(min_flows))
    min_hosts = max(1, int(min_hosts))
    for i, (_, start, _) in enumerate(rpc_flows):
        members = []
        hosts = set()
        for flow_id, other_start, flow_host_set in rpc_flows[i:]:
            if other_start > start + span:
                break
            members.append(flow_id)
            hosts.update(flow_host_set)
        if len(members) >= min_flows and len(hosts) >= min_hosts:
            marked.update(members)

    if not marked:
        return list(packets), {
            "filter_drop_rpc_fanout_flows": 1,
            "filter_rpc_fanout_window_s": float(window_s or 0.0),
            "filter_rpc_fanout_min_flows": int(min_flows),
            "filter_rpc_fanout_min_hosts": int(min_hosts),
            "filter_rpc_fanout_dropped_n_pkts": 0,
            "filter_rpc_fanout_dropped_bytes_total": 0,
            "filter_rpc_fanout_dropped_n_flows": 0,
            "filter_rpc_fanout_dropped_n_server_ips": 0,
        }
    kept = []
    dropped = []
    for packet in packets:
        if packet.flow_id in marked:
            dropped.append(packet)
        else:
            kept.append(packet)
    return kept, {
        "filter_drop_rpc_fanout_flows": 1,
        "filter_rpc_fanout_window_s": float(window_s or 0.0),
        "filter_rpc_fanout_min_flows": int(min_flows),
        "filter_rpc_fanout_min_hosts": int(min_hosts),
        "filter_rpc_fanout_dropped_n_pkts": int(len(dropped)),
        "filter_rpc_fanout_dropped_bytes_total": int(sum(p.length for p in dropped)),
        "filter_rpc_fanout_dropped_n_flows": int(len({p.flow_id for p in dropped})),
        "filter_rpc_fanout_dropped_n_server_ips": int(len({_packet_server_ip(p) for p in dropped})),
    }


def empty_rpc_fanout_stats() -> dict:
    return {
        "filter_drop_rpc_fanout_flows": 0,
        "filter_rpc_fanout_window_s": 0.0,
        "filter_rpc_fanout_min_flows": 0,
        "filter_rpc_fanout_min_hosts": 0,
        "filter_rpc_fanout_dropped_n_pkts": 0,
        "filter_rpc_fanout_dropped_bytes_total": 0,
        "filter_rpc_fanout_dropped_n_flows": 0,
        "filter_rpc_fanout_dropped_n_server_ips": 0,
    }


def _float_list(value) -> list[float]:
    if value is None:
        return []
    if isinstance(value, str):
        items = value.replace(",", " ").split()
    else:
        items = list(value)
    out = []
    for item in items:
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            continue
    return out





def _heuristic_window_risk_score(row: dict) -> float:
    """Transparent fallback score for candidate/window ranking when no trained model is supplied.

    This is not the supervised Layer-1 detector. It is intentionally simple and
    only helps scripts rank near-real-time candidate windows in a single-ZIP demo.
    """
    early_bytes = float(row.get("phase_0_3_bytes_total", 0.0) or 0.0)
    early_unknown_backend = (
        float(row.get("phase_0_3_unknown_bytes", 0.0) or 0.0)
        + float(row.get("phase_0_3_third_party_backend_or_other_bytes", 0.0) or 0.0)
        + float(row.get("phase_0_3_identity_provider_bytes", 0.0) or 0.0)
        + float(row.get("phase_0_3_captcha_challenge_bytes", 0.0) or 0.0)
        + float(row.get("phase_0_3_payment_provider_bytes", 0.0) or 0.0)
        + float(row.get("phase_0_3_messaging_api_bytes", 0.0) or 0.0)
        + float(row.get("phase_0_3_file_storage_bytes", 0.0) or 0.0)
        + float(row.get("phase_0_3_rpc_provider_bytes", 0.0) or 0.0)
        + float(row.get("phase_0_3_walletconnect_bytes", 0.0) or 0.0)
    )
    early_frac = early_unknown_backend / early_bytes if early_bytes else 0.0
    byte_score = min(1.0, math.log1p(max(0.0, early_bytes)) / math.log1p(50000.0))
    flow_score = min(1.0, float(row.get("base_n_flows", 0.0) or 0.0) / 20.0)
    cand_score = min(1.0, float(row.get("candidate_score", 0.0) or 0.0) / 3.5)
    wallet_context = 1.0 if (
        float(row.get("candidate_wallet_vendor_bytes_3s", 0.0) or 0.0) >= 8000
        or float(row.get("candidate_past_wallet_vendor_bytes", 0.0) or 0.0) >= 12000
    ) else 0.0
    score = 0.55 * cand_score + 0.20 * byte_score + 0.15 * early_frac + 0.05 * flow_score + 0.05 * wallet_context
    return max(0.0, min(1.0, score))


def _candidate_emit_at_observation_end(
    candidate_emit_epoch: float,
    packet_end_epoch: Optional[float],
) -> float:
    """Finalize an otherwise-open candidate at an explicit observation end.

    Continuous candidate NMS normally delays emission until its look-ahead
    neighborhood closes.  A HAR load-only sample is instead a deliberately
    censored stream: once its frozen observation horizon is reached, there are
    no later packets in the experiment.  Finalizing the candidate at that
    horizon keeps the reported availability causal with respect to the observed
    prefix and, critically, never attributes post-boundary evidence to it.
    Full-session extraction (``packet_end_epoch is None``) is unchanged.
    """
    emit_epoch = float(candidate_emit_epoch)
    if packet_end_epoch is None:
        return emit_epoch
    return min(emit_epoch, float(packet_end_epoch))

def extract_features_from_zip(
    zip_path: str,
    cfg: Dict,
    *,
    packet_start_epoch: Optional[float] = None,
    packet_end_epoch: Optional[float] = None,
) -> pd.DataFrame:
    filtering_cfg = cfg.get("filtering", {})
    role_source = str(filtering_cfg.get("role_source", "har")).lower()
    if role_source not in {"har", "sni", "sni_dns", "preserved", "har_then_sni"}:
        raise ValueError(
            f"unknown filtering.role_source={role_source!r}; expected har, sni, sni_dns, "
            "preserved, or har_then_sni"
        )
    paths = extract_zip_sample(zip_path, require_har=role_source in {"har", "har_then_sni"})
    try:
        anchor_cfg = cfg.get("anchor", {})
        anchor_keys = anchor_cfg.get("keys") or [
            anchor_cfg.get("primary_session_key", "t_metamask_connect_click"),
            anchor_cfg.get("event_fallback_key", "t_metamask_connect_click"),
        ]
        meta = load_sample_meta(paths, anchor_key=anchor_keys)
        feature_mode = filtering_cfg.get("feature_mode", "clean")
        har = load_har_for_sample(paths, meta, feature_mode=feature_mode) if role_source in {"har", "har_then_sni"} else HarInfo()
        pcap = read_pcap(
            paths.pcap_path,
            start_epoch=packet_start_epoch,
            end_epoch=packet_end_epoch,
        )
        if (packet_start_epoch is not None or packet_end_epoch is not None) and not pcap.packets:
            raise ValueError(
                f"{zip_path}: HAR target-page load interval retained no parseable packets"
            )
        transport_unify_stats = unify_transport_stream(
            pcap,
            str(filtering_cfg.get("transport_unify", "none") or "none"),
            int(filtering_cfg.get("quic_control_max_bytes", 48) or 48),
        )
        degradation_stats = apply_observability_degradation(pcap, filtering_cfg.get("degradation"))
        include_unknown = bool(filtering_cfg.get("include_unknown_packets", False))
        if role_source == "preserved":
            packets = [
                packet for packet in pcap.packets
                if include_unknown or str(packet.role) != "unknown"
            ]
        elif role_source == "har":
            packets = assign_packet_roles_from_har(pcap, har, include_unknown=include_unknown)
        elif role_source == "sni":
            packets = assign_packet_roles_from_sni(
                pcap,
                primary_host=meta.domain,
                feature_mode=feature_mode,
                include_unknown=include_unknown,
            )
        elif role_source == "sni_dns":
            packets = assign_packet_roles_from_sni_dns(
                pcap,
                primary_host=meta.domain,
                feature_mode=feature_mode,
                include_unknown=include_unknown,
            )
        else:
            packets = assign_packet_roles_har_then_sni(
                pcap,
                har,
                primary_host=meta.domain,
                feature_mode=feature_mode,
                include_unknown=include_unknown,
            )
        if not filtering_cfg.get("include_tcp", True):
            packets = [p for p in packets if p.proto != "TCP"]
        if not filtering_cfg.get("include_udp443", True):
            packets = [p for p in packets if not p.is_udp443]

        capture_end = meta.pcap_stop_epoch or meta.capture_stop_epoch or pcap.pcap_end
        if packet_end_epoch is not None:
            # The load-only observation is evaluated at the externally observed
            # HAR boundary.  A fixed window that extends beyond that horizon is
            # therefore a censored prefix available at the boundary, not at the
            # timestamp of the last retained packet.
            capture_end = float(packet_end_epoch)
        # Outside an explicit observation interval, fall back to the observed
        # PCAP end when session metadata is absent or inconsistent.  With an
        # explicit HAR end, keep that end as the censored observation horizon
        # even when the final packet arrived earlier.
        if packet_end_epoch is None and pcap.pcap_end and (
            capture_end is None
            or pcap.pcap_end < capture_end
            or pcap.pcap_end > capture_end + 2
        ):
            # Prefer observed end for feature windows, because PCAP timestamps are ground truth for packets.
            capture_end = pcap.pcap_end

        # Candidate onset generation must see wallet/RPC/unknown traffic before
        # role suppression. Layer-1 feature extraction below can still suppress
        # wallet-vendor traffic to avoid learning wallet-specific shortcuts.
        candidate_packets = list(packets)

        packets, suppression_stats = suppress_packets_by_role(
            packets,
            filtering_cfg.get("suppress_roles", []),
        )
        packets, host_suppression_stats = suppress_packets_by_host_pattern(
            packets,
            filtering_cfg.get("suppress_host_patterns", []),
            skip_roles=filtering_cfg.get("suppress_host_pattern_skip_roles", ["first_party_site"]),
        )
        if bool(filtering_cfg.get("drop_rpc_fanout_flows", False)):
            packets, rpc_fanout_stats = drop_rpc_fanout_flows(
                packets,
                window_s=float(filtering_cfg.get("rpc_fanout_window_s", 10.0) or 10.0),
                min_flows=int(filtering_cfg.get("rpc_fanout_min_flows", 3) or 3),
                min_hosts=int(filtering_cfg.get("rpc_fanout_min_hosts", 2) or 2),
            )
        else:
            rpc_fanout_stats = empty_rpc_fanout_stats()

        # Experimental counterfactual used to measure how much of Layer-1 is
        # carried by pre-interaction page-load traffic.  Clipping the packet
        # sequences here (before candidate generation and flow-shape indexing)
        # is stricter than merely discarding candidates whose start precedes the
        # action: candidate and kit-shape features otherwise look back 4--6 s.
        feature_cfg = cfg.get("features", {})
        clip_before_first_action = bool(feature_cfg.get("clip_before_first_action", False))
        traffic_clip_epoch = None
        if clip_before_first_action:
            if meta.first_action_epoch is None:
                raise ValueError(
                    f"{zip_path}: features.clip_before_first_action requires first_action_epoch"
                )
            traffic_clip_epoch = float(meta.first_action_epoch)
            candidate_packets = [p for p in candidate_packets if p.ts >= traffic_clip_epoch]
            packets = [p for p in packets if p.ts >= traffic_clip_epoch]
        feature_flow_shape_index = build_flow_shape_index(packets)

        anchor_prepad_s = max(0.0, float(anchor_cfg.get("prepad_s", 0.0) or 0.0))
        jitter_offsets_s = [x for x in _float_list(anchor_cfg.get("jitter_offsets_s")) if abs(x) > 1e-9]
        post_load_guard_epoch, post_load_guard_source, post_load_guard_mode = resolve_post_load_guard(
            anchor_cfg,
            meta_post_load_epoch=meta.post_load_epoch,
            meta_post_load_source=meta.post_load_source,
            packets=candidate_packets,
            session_start=meta.session_start_epoch if meta.session_start_epoch is not None else pcap.pcap_start,
        )
        anchor_mode = str(anchor_cfg.get("mode", "oracle") or "oracle").strip().lower()
        if anchor_mode in {"event", "event_oracle", "events"}:
            anchor_mode = "oracle"
        if anchor_mode in {"near_realtime", "near-real-time", "gateway", "network", "traffic"}:
            anchor_mode = "heuristic"

        anchor_items: list[CandidateAnchor] = []

        def _add_oracle_anchor(rank: int = 0) -> None:
            if meta.anchor_epoch is None:
                return
            anchor_items.append(make_oracle_candidate(
                float(meta.anchor_epoch),
                source=meta.anchor_source or "oracle",
                pcap_start=pcap.pcap_start,
                rank=rank,
                decision_delay_s=0.0,
            ))

        def _add_oracle_jitter(start_rank: int = 900) -> None:
            if meta.anchor_epoch is None or not jitter_offsets_s:
                return
            for i, offset_s in enumerate(jitter_offsets_s, start=1):
                start_epoch = float(meta.anchor_epoch) + float(offset_s)
                if pcap.pcap_start is not None:
                    start_epoch = max(float(pcap.pcap_start), start_epoch)
                if capture_end is not None:
                    start_epoch = min(float(capture_end), start_epoch)
                anchor_items.append(make_oracle_candidate(
                    start_epoch,
                    source=f"oracle_jitter_{offset_s:+.3f}s",
                    pcap_start=pcap.pcap_start,
                    rank=start_rank + i,
                    decision_delay_s=0.0,
                ))

        if anchor_mode == "oracle":
            _add_oracle_anchor(rank=0)
            _add_oracle_jitter(start_rank=900)
            if not anchor_items:
                raise ValueError(f"{zip_path}: cannot locate an offline interaction-reference anchor from keys={anchor_keys}")
        elif anchor_mode in {"heuristic", "candidate", "realtime"}:
            anchor_items = generate_candidate_anchors(
                candidate_packets,
                cfg.get("candidate", {}),
                pcap_start=pcap.pcap_start,
                pcap_end=capture_end or pcap.pcap_end,
                session_start=meta.session_start_epoch,
                post_load_epoch=post_load_guard_epoch,
            )
            if not anchor_items:
                raise ValueError(f"{zip_path}: traffic heuristic produced no candidate anchors")
        elif anchor_mode in {"auto"}:
            if meta.anchor_epoch is not None:
                _add_oracle_anchor(rank=0)
                _add_oracle_jitter(start_rank=900)
            else:
                anchor_items = generate_candidate_anchors(
                    candidate_packets,
                    cfg.get("candidate", {}),
                    pcap_start=pcap.pcap_start,
                    pcap_end=capture_end or pcap.pcap_end,
                    session_start=meta.session_start_epoch,
                    post_load_epoch=post_load_guard_epoch,
                )
            if not anchor_items:
                raise ValueError(f"{zip_path}: no candidate anchors generated for anchor.mode=auto")
        elif anchor_mode in {"sample_start", "interval_start", "observation_start"}:
            # Strict load-prefix experiment: the externally frozen interval
            # start is the sole onset.  No later application action is read or
            # used for alignment on this route.
            start_epoch = (
                float(packet_start_epoch)
                if packet_start_epoch is not None
                else float(pcap.pcap_start)
            )
            anchor_items.append(make_oracle_candidate(
                start_epoch,
                source="observation_interval_start",
                pcap_start=start_epoch,
                rank=0,
                decision_delay_s=0.0,
            ))
        elif anchor_mode in {"both", "oracle_plus_heuristic", "oracle_and_heuristic", "hybrid"}:
            _add_oracle_anchor(rank=0)
            _add_oracle_jitter(start_rank=900)
            heur = generate_candidate_anchors(
                candidate_packets,
                cfg.get("candidate", {}),
                pcap_start=pcap.pcap_start,
                pcap_end=capture_end or pcap.pcap_end,
                session_start=meta.session_start_epoch,
                post_load_epoch=post_load_guard_epoch,
            )
            for cand in heur:
                if meta.anchor_epoch is not None and abs(cand.start_epoch - float(meta.anchor_epoch)) < 1e-6:
                    continue
                anchor_items.append(cand)
            if not anchor_items:
                raise ValueError(f"{zip_path}: no oracle or heuristic anchors available")
        else:
            raise ValueError(f"unknown anchor.mode={anchor_mode!r}; expected oracle, heuristic, auto, or oracle_plus_heuristic")

        rows = []
        roles = feature_cfg.get("roles", [])
        sni_n_hosts, sni_n_server_ips = sni_observation_counts(pcap)
        dns_n_hosts, dns_n_server_ips = dns_observation_counts(pcap)
        for cand in anchor_items:
            feature_anchor = float(cand.start_epoch)
            if anchor_prepad_s:
                feature_anchor = feature_anchor - anchor_prepad_s
                if pcap.pcap_start is not None:
                    feature_anchor = max(float(pcap.pcap_start), feature_anchor)
                if post_load_guard_epoch is not None:
                    feature_anchor = max(float(post_load_guard_epoch), feature_anchor)
            windows = []
            window_cfg = cfg.get("windows", {})
            if window_cfg.get("emit_observation_horizon", False):
                if capture_end is None:
                    raise ValueError(
                        f"{zip_path}: observation-horizon window requires an explicit interval end"
                    )
                windows.append(Window(
                    "adaptive",
                    feature_anchor,
                    float(capture_end),
                    "full_observation_interval",
                ))
            elif window_cfg.get("emit_adaptive", False):
                adaptive_cfg = dict(cfg.get("adaptive_window", {}))
                adaptive_cfg.pop("prepad_s", None)
                windows.append(adaptive_episode_window(packets, feature_anchor, capture_end, **adaptive_cfg))
            if window_cfg.get("emit_dynamic", True):
                dyn_cfg = dict(cfg.get("dynamic", {}))
                dyn_cfg.pop("prepad_s", None)  # Backward-compatible with older configs that placed it here.
                windows.append(dynamic_episode_window(packets, feature_anchor, capture_end, **dyn_cfg))
            fixed_seconds = window_cfg.get("fixed_seconds", [3, 10, 30, 60, 90])
            fixed = fixed_windows(feature_anchor, capture_end, fixed_seconds)
            if window_cfg.get("require_complete_fixed_windows", False):
                fixed = [
                    window
                    for window, seconds in zip(fixed, fixed_seconds)
                    if capture_end is None
                    or feature_anchor + float(seconds) <= float(capture_end) + 1e-7
                ]
            windows.extend(fixed)

            interval_start_mode = anchor_mode in {
                "sample_start", "interval_start", "observation_start"
            }
            candidate_row = cand.to_row(
                None if interval_start_mode else meta.anchor_epoch
            )
            effective_candidate_emit = _candidate_emit_at_observation_end(
                float(cand.emit_epoch),
                packet_end_epoch,
            )
            if effective_candidate_emit != float(cand.emit_epoch):
                candidate_row["candidate_emit_epoch"] = effective_candidate_emit
                candidate_row["candidate_available_epoch"] = effective_candidate_emit
                if pcap.pcap_start is not None:
                    candidate_row["candidate_emit_rel_s"] = (
                        effective_candidate_emit - float(pcap.pcap_start)
                    )
            for w in windows:
                feats = extract_window_features(
                    packets,
                    w,
                    feature_cfg,
                    roles=roles,
                    capture_end=capture_end,
                    shape_index=feature_flow_shape_index,
                )
                available_epoch = max(effective_candidate_emit, float(w.end))
                row = {
                    "sample_id": meta.sample_id,
                    "label": meta.label,
                    "domain": meta.domain,
                    "url": meta.url,
                    "status": meta.status,
                    "zip_path": zip_path,
                    "anchor_mode": anchor_mode,
                    "anchor_time_epoch": float(cand.start_epoch),
                    "anchor_source": cand.source,
                    "anchor_prepad_s": anchor_prepad_s,
                    "feature_anchor_epoch": feature_anchor,
                    "oracle_anchor_epoch": None if interval_start_mode else meta.anchor_epoch,
                    "oracle_anchor_time_epoch": None if interval_start_mode else meta.anchor_epoch,
                    "oracle_anchor_source": "" if interval_start_mode else meta.anchor_source,
                    "window_available_epoch": available_epoch,
                    "window_available_after_candidate_s": max(0.0, available_epoch - float(cand.start_epoch)),
                    "window_latency_vs_oracle_s": (available_epoch - float(meta.anchor_epoch)) if meta.anchor_epoch is not None else float("nan"),
                    "window_available_lag_from_oracle_s": (available_epoch - float(meta.anchor_epoch)) if meta.anchor_epoch is not None else float("nan"),
                    "pcap_start_epoch": (
                        float(packet_start_epoch)
                        if interval_start_mode and packet_start_epoch is not None
                        else pcap.pcap_start
                    ),
                    "pcap_end_epoch": pcap.pcap_end,
                    "packet_interval_start_epoch": packet_start_epoch,
                    "packet_interval_end_epoch": packet_end_epoch,
                    "session_pcap_start_epoch": meta.pcap_start_epoch,
                    "session_pcap_stop_epoch": meta.pcap_stop_epoch,
                    "capture_stop_epoch": meta.capture_stop_epoch,
                    "post_load_guard_mode": post_load_guard_mode,
                    "post_load_guard_epoch": post_load_guard_epoch,
                    "post_load_guard_source": post_load_guard_source,
                    "post_load_guard_rel_s": (float(post_load_guard_epoch) - float(pcap.pcap_start)) if (post_load_guard_epoch is not None and pcap.pcap_start is not None) else float("nan"),
                    "first_action_epoch": meta.first_action_epoch,
                    "first_action_source": meta.first_action_source,
                    "first_action_rel_s": (float(meta.first_action_epoch) - float(pcap.pcap_start)) if (meta.first_action_epoch is not None and pcap.pcap_start is not None) else float("nan"),
                    "traffic_clip_mode": "first_action" if clip_before_first_action else "none",
                    "traffic_clip_epoch": traffic_clip_epoch,
                    "linktype": pcap.linktype,
                    "local_ips": ",".join(pcap.local_ips),
                    "role_source": role_source,
                    "har_n_hosts": len(har.hosts),
                    "har_n_server_ips": len(har.ip_to_hosts),
                    "sni_n_hosts": sni_n_hosts,
                    "sni_n_server_ips": sni_n_server_ips,
                    "dns_n_hosts": dns_n_hosts,
                    "dns_n_server_ips": dns_n_server_ips,
                }
                row.update(candidate_row)
                row.update(suppression_stats)
                row.update(host_suppression_stats)
                row.update(rpc_fanout_stats)
                row.update(transport_unify_stats)
                row.update(degradation_stats)
                row.update(feats)
                row["heuristic_realtime_risk_score"] = _heuristic_window_risk_score(row)
                rows.append(row)
        return pd.DataFrame(rows)
    finally:
        if paths.tempdir is not None:
            paths.tempdir.cleanup()

def _expand_input_globs(input_glob: str | Sequence[str]) -> List[str]:
    patterns = [input_glob] if isinstance(input_glob, str) else list(input_glob)
    paths: List[str] = []
    for pattern in patterns:
        paths.extend(glob.glob(pattern))
    return sorted(dict.fromkeys(paths))


def extract_features(
    input_glob: str | Sequence[str],
    config_path: Optional[str],
    out: str,
    skip_errors: bool = False,
    error_log: Optional[str] = None,
) -> pd.DataFrame:
    cfg = load_config(config_path)
    paths = _expand_input_globs(input_glob)
    if not paths:
        raise FileNotFoundError(f"No files matched {input_glob}")
    dfs = []
    errors = []
    for p in paths:
        print(f"[extract] {p}")
        try:
            dfs.append(extract_features_from_zip(p, cfg))
        except Exception as exc:
            if not skip_errors:
                raise
            print(f"[extract] skip {p}: {exc}")
            errors.append({"path": p, "error": str(exc), "type": type(exc).__name__})
    if not dfs:
        raise RuntimeError("No samples were successfully extracted")
    df = pd.concat(dfs, ignore_index=True)
    out_path = pathlib.Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    if error_log and errors:
        err_path = pathlib.Path(error_log)
        err_path.parent.mkdir(parents=True, exist_ok=True)
        err_path.write_text(json.dumps(errors, indent=2), encoding="utf-8")
        print(f"[extract] wrote error log {err_path} skipped={len(errors)}")
    print(f"[extract] wrote {out_path} rows={len(df)} cols={len(df.columns)}")
    return df

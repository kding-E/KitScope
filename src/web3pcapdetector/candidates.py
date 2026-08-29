from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from .pcap_minimal import Packet
from .flow_shape import FlowShapeIndex, build_flow_shape_index, local_flow_change_features
from .roles import naive_registrable_domain


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def _host_parts(host: str) -> List[str]:
    return [p.strip().lower() for p in str(host or "").split("|") if p.strip()]


def _role_byte_counter(pkts: Sequence[Packet]) -> Counter:
    c: Counter = Counter()
    for p in pkts:
        c[str(p.role or "unknown").lower()] += int(p.length)
    return c


def _sum_roles(counter: Counter, roles: Iterable[str]) -> int:
    return int(sum(counter[str(r).lower()] for r in roles))


def _split_roles(value, default: Sequence[str]) -> List[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        items = value.split(",")
    else:
        items = list(value)
    out = [str(x).strip().lower() for x in items if str(x).strip()]
    return out or list(default)


@dataclass
class CandidateAnchor:
    """A traffic-only proposed interaction-relevant traffic-episode onset.

    `start_epoch` is the proposed feature anchor.  `emit_epoch` is the earliest
    time a near-real-time gateway could emit this candidate, because the scoring
    rule looks ahead by `decision_delay_s` seconds.  The onset is not assumed to
    occur after a wallet connection or even after the first user action.
    """

    start_epoch: float
    emit_epoch: float
    rank: int
    score: float
    source: str
    reason: str
    decision_delay_s: float
    scan_step_s: float
    immediate_window_s: float
    past_window_s: float
    start_rel_s: float = 0.0
    emit_rel_s: float = 0.0
    n_pkts_3s: int = 0
    bytes_3s: int = 0
    important_bytes_3s: int = 0
    unknown_bytes_3s: int = 0
    backend_bytes_3s: int = 0
    rpc_bytes_3s: int = 0
    walletconnect_bytes_3s: int = 0
    wallet_vendor_bytes_3s: int = 0
    static_analytics_bytes_3s: int = 0
    identity_bytes_3s: int = 0
    captcha_bytes_3s: int = 0
    payment_bytes_3s: int = 0
    messaging_bytes_3s: int = 0
    form_backend_bytes_3s: int = 0
    cloud_api_bytes_3s: int = 0
    object_storage_bytes_3s: int = 0
    email_delivery_bytes_3s: int = 0
    hosting_bytes_3s: int = 0
    file_storage_bytes_3s: int = 0
    url_shortener_bytes_3s: int = 0
    first_party_bytes_3s: int = 0
    interaction_bytes_3s: int = 0
    nuisance_bytes_3s: int = 0
    past_important_bytes: int = 0
    past_wallet_vendor_bytes: int = 0
    new_flow_count_3s: int = 0
    new_host_count_3s: int = 0
    new_endpoint_count_3s: int = 0
    new_site_count_3s: int = 0
    new_web_flow_count_3s: int = 0
    new_http_flow_count_3s: int = 0
    visible_site_open_epoch: float = float("nan")
    endpoint_site_open_epoch: float = float("nan")
    flow_shape_salience: float = 0.0
    flow_shape_salience_top3_mean: float = 0.0
    flow_shape_change_score: float = 0.0
    flow_shape_intrinsic_score: float = 0.0
    flow_shape_changed_byte_frac: float = 0.0
    flow_shape_stable_nochange_byte_frac: float = 0.0
    flow_shape_active_flow_count: float = 0.0
    flow_shape_high_salience_flow_count: float = 0.0
    flow_shape_multiflow_coordination: float = 0.0
    quiet_score: float = 0.0
    burst_score: float = 0.0
    role_trigger_score: float = 0.0
    role_aux_score: float = 0.0
    stable_nochange_penalty: float = 0.0
    static_penalty: float = 0.0
    wallet_vendor_only_penalty: float = 0.0

    def to_row(self, oracle_anchor_epoch: Optional[float] = None) -> Dict[str, float | int | str]:
        candidate_id = "oracle" if int(self.rank) == 0 or str(self.source).startswith("session.json") else f"heur_{max(1, int(self.rank)):02d}"
        row: Dict[str, float | int | str] = {
            "candidate_id": candidate_id,
            "candidate_rank": int(self.rank),
            "candidate_source": self.source,
            "candidate_reason": self.reason,
            "candidate_start_epoch": float(self.start_epoch),
            "candidate_anchor_epoch": float(self.start_epoch),
            "candidate_emit_epoch": float(self.emit_epoch),
            "candidate_available_epoch": float(self.emit_epoch),
            "candidate_start_rel_s": float(self.start_rel_s),
            "candidate_emit_rel_s": float(self.emit_rel_s),
            "candidate_score": float(self.score),
            "candidate_decision_delay_s": float(self.decision_delay_s),
            "candidate_scan_step_s": float(self.scan_step_s),
            "candidate_immediate_window_s": float(self.immediate_window_s),
            "candidate_past_window_s": float(self.past_window_s),
            "candidate_n_pkts_3s": int(self.n_pkts_3s),
            "candidate_bytes_3s": int(self.bytes_3s),
            "candidate_important_bytes_3s": int(self.important_bytes_3s),
            "candidate_unknown_bytes_3s": int(self.unknown_bytes_3s),
            "candidate_backend_bytes_3s": int(self.backend_bytes_3s),
            "candidate_rpc_bytes_3s": int(self.rpc_bytes_3s),
            "candidate_walletconnect_bytes_3s": int(self.walletconnect_bytes_3s),
            "candidate_wallet_vendor_bytes_3s": int(self.wallet_vendor_bytes_3s),
            "candidate_static_analytics_bytes_3s": int(self.static_analytics_bytes_3s),
            "candidate_identity_bytes_3s": int(self.identity_bytes_3s),
            "candidate_captcha_bytes_3s": int(self.captcha_bytes_3s),
            "candidate_payment_bytes_3s": int(self.payment_bytes_3s),
            "candidate_messaging_bytes_3s": int(self.messaging_bytes_3s),
            "candidate_form_backend_bytes_3s": int(self.form_backend_bytes_3s),
            "candidate_cloud_api_bytes_3s": int(self.cloud_api_bytes_3s),
            "candidate_object_storage_bytes_3s": int(self.object_storage_bytes_3s),
            "candidate_email_delivery_bytes_3s": int(self.email_delivery_bytes_3s),
            "candidate_hosting_bytes_3s": int(self.hosting_bytes_3s),
            "candidate_file_storage_bytes_3s": int(self.file_storage_bytes_3s),
            "candidate_url_shortener_bytes_3s": int(self.url_shortener_bytes_3s),
            "candidate_first_party_bytes_3s": int(self.first_party_bytes_3s),
            "candidate_interaction_bytes_3s": int(self.interaction_bytes_3s),
            "candidate_nuisance_bytes_3s": int(self.nuisance_bytes_3s),
            "candidate_past_important_bytes": int(self.past_important_bytes),
            "candidate_past_wallet_vendor_bytes": int(self.past_wallet_vendor_bytes),
            "candidate_new_flow_count_3s": int(self.new_flow_count_3s),
            "candidate_new_host_count_3s": int(self.new_host_count_3s),
            "candidate_new_endpoint_count_3s": int(self.new_endpoint_count_3s),
            "candidate_new_site_count_3s": int(self.new_site_count_3s),
            "candidate_new_web_flow_count_3s": int(self.new_web_flow_count_3s),
            "candidate_new_http_flow_count_3s": int(self.new_http_flow_count_3s),
            "candidate_visible_site_open_epoch": float(self.visible_site_open_epoch),
            "candidate_endpoint_site_open_epoch": float(self.endpoint_site_open_epoch),
            "candidate_flow_shape_salience": float(self.flow_shape_salience),
            "candidate_flow_shape_salience_top3_mean": float(self.flow_shape_salience_top3_mean),
            "candidate_flow_shape_change_score": float(self.flow_shape_change_score),
            "candidate_flow_shape_intrinsic_score": float(self.flow_shape_intrinsic_score),
            "candidate_flow_shape_changed_byte_frac": float(self.flow_shape_changed_byte_frac),
            "candidate_flow_shape_stable_nochange_byte_frac": float(self.flow_shape_stable_nochange_byte_frac),
            "candidate_flow_shape_active_flow_count": float(self.flow_shape_active_flow_count),
            "candidate_flow_shape_high_salience_flow_count": float(self.flow_shape_high_salience_flow_count),
            "candidate_flow_shape_multiflow_coordination": float(self.flow_shape_multiflow_coordination),
            "candidate_quiet_score": float(self.quiet_score),
            "candidate_burst_score": float(self.burst_score),
            "candidate_role_trigger_score": float(self.role_trigger_score),
            "candidate_role_aux_score": float(self.role_aux_score),
            "candidate_stable_nochange_penalty": float(self.stable_nochange_penalty),
            "candidate_static_penalty": float(self.static_penalty),
            "candidate_wallet_vendor_only_penalty": float(self.wallet_vendor_only_penalty),
        }
        if oracle_anchor_epoch is not None:
            err = float(self.start_epoch - oracle_anchor_epoch)
            row.update({
                "candidate_anchor_error_s": err,
                "candidate_abs_anchor_error_s": abs(err),
                "candidate_offset_from_oracle_s": err,
                "candidate_hit_within_1s": 1 if abs(err) <= 1.0 else 0,
                "candidate_hit_within_3s": 1 if abs(err) <= 3.0 else 0,
                "candidate_hit_within_5s": 1 if abs(err) <= 5.0 else 0,
                "candidate_hit_within_10s": 1 if abs(err) <= 10.0 else 0,
                "candidate_hit_oracle_1s": 1 if abs(err) <= 1.0 else 0,
                "candidate_hit_oracle_3s": 1 if abs(err) <= 3.0 else 0,
                "candidate_hit_oracle_5s": 1 if abs(err) <= 5.0 else 0,
                "candidate_hit_oracle_10s": 1 if abs(err) <= 10.0 else 0,
                "candidate_emit_latency_vs_oracle_s": float(self.emit_epoch - oracle_anchor_epoch),
                "candidate_available_lag_from_oracle_s": float(self.emit_epoch - oracle_anchor_epoch),
            })
        else:
            row.update({
                "candidate_anchor_error_s": float("nan"),
                "candidate_abs_anchor_error_s": float("nan"),
                "candidate_offset_from_oracle_s": float("nan"),
                "candidate_hit_within_1s": 0,
                "candidate_hit_within_3s": 0,
                "candidate_hit_within_5s": 0,
                "candidate_hit_within_10s": 0,
                "candidate_hit_oracle_1s": 0,
                "candidate_hit_oracle_3s": 0,
                "candidate_hit_oracle_5s": 0,
                "candidate_hit_oracle_10s": 0,
                "candidate_emit_latency_vs_oracle_s": float("nan"),
                "candidate_available_lag_from_oracle_s": float("nan"),
            })
        return row


def make_oracle_candidate(
    anchor_epoch: float,
    *,
    source: str,
    pcap_start: Optional[float],
    rank: int = 0,
    decision_delay_s: float = 0.0,
) -> CandidateAnchor:
    pcap_start = float(pcap_start if pcap_start is not None else anchor_epoch)
    return CandidateAnchor(
        start_epoch=float(anchor_epoch),
        emit_epoch=float(anchor_epoch + max(0.0, decision_delay_s)),
        rank=int(rank),
        score=999.0,
        source=source or "oracle",
        reason="oracle_event_anchor",
        decision_delay_s=max(0.0, float(decision_delay_s)),
        scan_step_s=0.0,
        immediate_window_s=0.0,
        past_window_s=0.0,
        start_rel_s=float(anchor_epoch - pcap_start),
        emit_rel_s=float(anchor_epoch + max(0.0, decision_delay_s) - pcap_start),
    )


def _first_packet_time_for_role(packets: Sequence[Packet], roles: Iterable[str]) -> Optional[float]:
    role_set = {str(r).lower() for r in roles}
    vals = [p.ts for p in packets if str(p.role or "unknown").lower() in role_set]
    return min(vals) if vals else None


def _flow_first_times(packets: Sequence[Packet]) -> Dict[Tuple, float]:
    first: Dict[Tuple, float] = {}
    for p in packets:
        if p.flow_id not in first or p.ts < first[p.flow_id]:
            first[p.flow_id] = p.ts
    return first


def _host_first_times(packets: Sequence[Packet]) -> Dict[str, float]:
    first: Dict[str, float] = {}
    for p in packets:
        for host in _host_parts(p.host):
            if host not in first or p.ts < first[host]:
                first[host] = p.ts
    return first


_WEB_SERVICE_PORTS = frozenset({80, 443, 8080, 8443})


def _remote_endpoint(packet: Packet, *, web_only: bool = False) -> Tuple[str, int] | None:
    """Return the gateway-observable remote IP/service-port identity.

    Normal visibility can use SNI/host novelty, whereas ECH/DoH must fall
    back to an address-level identity.  The client ephemeral port is excluded
    deliberately: a reconnect to the same remote service is not a new site.
    """
    if packet.direction == "up":
        endpoint = (str(packet.dst_ip), int(packet.dst_port))
        return endpoint if not web_only or endpoint[1] in _WEB_SERVICE_PORTS else None
    if packet.direction == "down":
        endpoint = (str(packet.src_ip), int(packet.src_port))
        return endpoint if not web_only or endpoint[1] in _WEB_SERVICE_PORTS else None
    server_ports = {53, 80, 443, 8080, 8443}
    if int(packet.dst_port) in server_ports and int(packet.src_port) not in server_ports:
        endpoint = (str(packet.dst_ip), int(packet.dst_port))
        return endpoint if not web_only or endpoint[1] in _WEB_SERVICE_PORTS else None
    if int(packet.src_port) in server_ports and int(packet.dst_port) not in server_ports:
        endpoint = (str(packet.src_ip), int(packet.src_port))
        return endpoint if not web_only or endpoint[1] in _WEB_SERVICE_PORTS else None
    return None


def _endpoint_first_times(packets: Sequence[Packet]) -> Dict[Tuple[str, int], float]:
    first: Dict[Tuple[str, int], float] = {}
    for packet in packets:
        endpoint = _remote_endpoint(packet, web_only=True)
        if endpoint is None:
            continue
        if endpoint not in first or packet.ts < first[endpoint]:
            first[endpoint] = packet.ts
    return first


def _site_first_times(packets: Sequence[Packet]) -> Dict[str, float]:
    first: Dict[str, float] = {}
    for packet in packets:
        for host in _host_parts(packet.host or packet.sni):
            site = naive_registrable_domain(host)
            if not site:
                continue
            if site not in first or packet.ts < first[site]:
                first[site] = packet.ts
    return first


def _attach_site_open_evidence(
    candidate: CandidateAnchor,
    packets: Sequence[Packet] | FlowShapeIndex,
    endpoint_first: Dict[Tuple[str, int], float],
    site_first: Dict[str, float],
    flow_first: Dict[Tuple, float],
) -> CandidateAnchor:
    immediate = _packets_between(
        packets,
        float(candidate.start_epoch),
        float(candidate.start_epoch) + float(candidate.decision_delay_s),
    )
    start = float(candidate.start_epoch)
    end = start + float(candidate.decision_delay_s)
    web_packets = [
        packet for packet in immediate
        if _remote_endpoint(packet, web_only=True) is not None
    ]
    web_flows = {
        packet.flow_id for packet in web_packets
        if start <= float(flow_first.get(packet.flow_id, packet.ts)) <= end
    }
    endpoints = {
        endpoint for packet in web_packets
        if (endpoint := _remote_endpoint(packet, web_only=True)) is not None
        and start <= float(endpoint_first.get(endpoint, packet.ts)) <= end
    }
    sites = {
        site
        for packet in web_packets
        for host in _host_parts(packet.host or packet.sni)
        if (site := naive_registrable_domain(host))
        and start <= float(site_first.get(site, packet.ts)) <= end
    }
    http_flows = {
        packet.flow_id for packet in web_packets
        if (_remote_endpoint(packet, web_only=True) or ("", 0))[1] in {80, 8080}
        and start <= float(flow_first.get(packet.flow_id, packet.ts)) <= end
    }
    candidate.new_endpoint_count_3s = int(len(endpoints))
    candidate.new_site_count_3s = int(len(sites))
    candidate.new_web_flow_count_3s = int(len(web_flows))
    candidate.new_http_flow_count_3s = int(len(http_flows))
    if sites:
        candidate.visible_site_open_epoch = float(
            min(site_first[site] for site in sites)
        )
    if endpoints:
        candidate.endpoint_site_open_epoch = float(
            min(endpoint_first[endpoint] for endpoint in endpoints)
        )
    return candidate


def attach_continuous_site_open_evidence(
    candidates: Sequence[CandidateAnchor],
    packets: Sequence[Packet],
) -> List[CandidateAnchor]:
    """Attach the independent site-open pass to cached onset proposals.

    Onset proposal generation and site-open detection are deliberately
    separable.  This helper permits a frozen full-stream proposal cache to be
    reused: the long packet stream is indexed once, site/endpoint burst
    evidence is attached without regenerating proposal scores or Layer-1
    windows, and the caller can then join proposals to alert-session
    boundaries.  No label, source-unit boundary, idle gap, or rolling horizon
    is consulted.
    """
    ordered_packets = sorted(
        [packet for packet in packets if packet.ts is not None],
        key=lambda packet: packet.ts,
    )
    if not ordered_packets:
        return list(candidates)
    packet_index = build_flow_shape_index(ordered_packets)
    flow_first = _flow_first_times(ordered_packets)
    endpoint_first = _endpoint_first_times(ordered_packets)
    site_first = _site_first_times(ordered_packets)
    for candidate in candidates:
        _attach_site_open_evidence(
            candidate, packet_index, endpoint_first, site_first, flow_first
        )
    return list(candidates)


def detect_continuous_site_open_boundaries(
    packets: Sequence[Packet],
    cfg: Optional[dict] = None,
    *,
    visibility: str,
) -> List[dict]:
    """Detect site-open reset boundaries in an independent full-stream pass.

    The pass deliberately does not consume onset candidates.  It first finds
    novel visible site identities (SNI/Host) or novel remote Web IP:service-port
    identities, groups their first observations into short traffic bursts, and
    retains bursts with enough newly opened Web flows/bytes.  Cached onset
    proposals can therefore be generated before this pass and joined to the
    resulting intervals without changing either detector.
    """
    cfg = dict(cfg or {})
    if visibility not in {"sni_or_host", "endpoint"}:
        raise ValueError(f"unsupported alert-session visibility: {visibility!r}")
    ordered = sorted(
        [packet for packet in packets if packet.ts is not None],
        key=lambda packet: float(packet.ts),
    )
    if not ordered:
        return []

    merge_s = max(0.0, float(cfg.get("boundary_merge_s", 3.5) or 3.5))
    lookahead_s = max(
        0.25, float(cfg.get("site_open_lookahead_s", 3.5) or 3.5)
    )
    past_s = max(0.0, float(cfg.get("site_open_past_s", 4.0) or 4.0))
    min_visible_novelty = max(1, int(cfg.get("min_visible_novelty", 1) or 1))
    min_visible_navigation_sites = max(
        1, int(cfg.get("min_visible_navigation_sites", 1) or 1)
    )
    min_endpoint_novelty = max(1, int(cfg.get("min_endpoint_novelty", 2) or 2))
    min_web_flows = max(1, int(cfg.get("min_web_flows", 2) or 2))
    min_burst_bytes = max(1, int(cfg.get("min_site_open_bytes", 900) or 900))
    quiet_score_min = float(cfg.get("quiet_score_min", 0.35) or 0.35)
    quiet_norm_bytes = max(
        1.0, float(cfg.get("site_open_quiet_norm_bytes", 20000) or 20000)
    )
    high_fanout_web_flows = max(
        min_web_flows, int(cfg.get("high_fanout_web_flows", 12) or 12)
    )

    # By default an identity opens a session only on its first observation in
    # the long stream.  The optional normal-visibility reactivation horizon
    # adds a second, still causal trigger when the *same* visible site returns
    # after being absent for the frozen interval.  This is identity-local state,
    # not an idle-gap/rolling pre-segmentation of the client stream.
    reactivation_quiet_s = (
        max(0.0, float(cfg.get("visible_identity_reactivation_quiet_s", 0.0) or 0.0))
        if visibility == "sni_or_host"
        else 0.0
    )
    identity_last: Dict[object, float] = {}
    identity_events: List[Tuple[float, object, bool]] = []
    navigation_like_identities: set[object] = set()
    flow_first: Dict[Tuple, Tuple[float, int]] = {}
    web_packets: List[Packet] = []
    for packet in ordered:
        endpoint = _remote_endpoint(packet, web_only=True)
        if endpoint is None:
            continue
        web_packets.append(packet)
        flow_id = packet.flow_id
        prior_flow = flow_first.get(flow_id)
        if prior_flow is None or float(packet.ts) < prior_flow[0]:
            flow_first[flow_id] = (float(packet.ts), int(endpoint[1]))
        identities: Iterable[object]
        if visibility == "endpoint":
            identities = (endpoint,)
        else:
            visible_identities = []
            for host in _host_parts(packet.host or packet.sni):
                site = naive_registrable_domain(host)
                if not site:
                    continue
                visible_identities.append(site)
                normalized_host = host.lower().rstrip(".")
                if normalized_host in {site, f"www.{site}"}:
                    navigation_like_identities.add(site)
            identities = tuple(visible_identities)
        for identity in identities:
            epoch = float(packet.ts)
            prior = identity_last.get(identity)
            if prior is None:
                identity_events.append((epoch, identity, False))
            elif reactivation_quiet_s > 0.0 and epoch - float(prior) >= reactivation_quiet_s:
                identity_events.append((epoch, identity, True))
            identity_last[identity] = epoch
    if not identity_events:
        return []

    identity_events.sort(
        key=lambda row: (float(row[0]), str(row[1]), bool(row[2])),
    )
    groups: List[List[Tuple[float, object, bool]]] = []
    for event in identity_events:
        if groups and float(event[0]) - float(groups[-1][-1][0]) <= merge_s:
            groups[-1].append(event)
        else:
            groups.append([event])

    flow_events = sorted(
        (epoch, port, flow_id) for flow_id, (epoch, port) in flow_first.items()
    )
    flow_epochs = [float(row[0]) for row in flow_events]
    web_times = [float(packet.ts) for packet in web_packets]
    web_prefix = [0]
    for packet in web_packets:
        web_prefix.append(web_prefix[-1] + int(packet.length))

    def byte_sum(start: float, end: float) -> int:
        left = bisect_left(web_times, float(start))
        right = bisect_left(web_times, float(end))
        return int(web_prefix[right] - web_prefix[left])

    raw_boundaries: List[dict] = []
    novelty_min = (
        min_visible_novelty if visibility == "sni_or_host" else min_endpoint_novelty
    )
    for group in groups:
        epoch = float(group[0][0])
        stop = epoch + lookahead_s
        left = bisect_left(flow_epochs, epoch)
        right = bisect_right(flow_epochs, stop)
        local_flows = flow_events[left:right]
        web_flow_count = int(len(local_flows))
        http_flow_count = int(sum(int(row[1]) in {80, 8080} for row in local_flows))
        burst_bytes = byte_sum(epoch, stop)
        past_bytes = byte_sum(epoch - past_s, epoch) if past_s > 0 else 0
        quiet_score = _clamp(1.0 - past_bytes / quiet_norm_bytes, 0.0, 1.0)
        novelty = int(len({identity for _, identity, _ in group}))
        reactivated_novelty = int(len({
            identity for _, identity, reactivated in group if reactivated
        }))
        navigation_sites = int(sum(
            identity in navigation_like_identities
            for identity in {identity for _, identity, _ in group}
        ))
        opening_context = True
        if visibility == "endpoint":
            opening_context = bool(
                quiet_score >= quiet_score_min
                or web_flow_count >= high_fanout_web_flows
                or http_flow_count > 0
            )
        if not (
            novelty >= novelty_min
            and (
                visibility == "endpoint"
                or navigation_sites >= min_visible_navigation_sites
            )
            and web_flow_count >= min_web_flows
            and burst_bytes >= min_burst_bytes
            and opening_context
        ):
            continue
        raw_boundaries.append({
            "boundary_epoch": epoch,
            "novelty": novelty,
            "reactivated_novelty": reactivated_novelty,
            "navigation_sites": navigation_sites,
            "web_flows": web_flow_count,
            "http_flows": http_flow_count,
            "burst_bytes": int(burst_bytes),
            "quiet_score": float(quiet_score),
            "score": float(math.log1p(burst_bytes)),
            "source": (
                "independent_full_stream_site_open_reactivation_pass"
                if reactivated_novelty > 0
                else "independent_full_stream_site_open_pass"
            ),
            "merged_triggers": int(len(group)),
        })
    return raw_boundaries


def _packets_between(packets: Sequence[Packet], start: float, end: float) -> List[Packet]:
    if isinstance(packets, FlowShapeIndex):
        return packets.packets_between(start, end)
    return [p for p in packets if start <= p.ts < end]


def _score_candidate_at(
    t: float,
    packets: Sequence[Packet],
    flow_first: Dict[Tuple, float],
    host_first: Dict[str, float],
    cfg: dict,
    pcap_start: float,
    packet_index: FlowShapeIndex | None = None,
) -> Optional[CandidateAnchor]:
    immediate_s = max(0.25, float(cfg.get("decision_delay_s", cfg.get("immediate_window_s", 3.0)) or 3.0))
    past_s = max(0.0, float(cfg.get("past_quiet_window_s", 5.0) or 5.0))
    scan_step_s = max(0.05, float(cfg.get("scan_step_s", 0.5) or 0.5))
    important_roles = _split_roles(
        cfg.get("important_roles"),
        [
            "unknown", "first_party_site", "third_party_backend_or_other",
            "identity_provider", "captcha_challenge", "payment_provider",
            "messaging_api", "form_backend", "cloud_api", "object_storage",
            "email_delivery", "file_storage", "url_shortener", "hosting_platform",
            "rpc_provider", "walletconnect",
        ],
    )
    backend_roles = _split_roles(
        cfg.get("backend_roles"),
        [
            "first_party_site", "third_party_backend_or_other", "identity_provider",
            "payment_provider", "messaging_api", "form_backend", "cloud_api",
            "object_storage", "email_delivery", "file_storage", "hosting_platform",
            "rpc_provider", "walletconnect",
        ],
    )
    trigger_roles = _split_roles(
        cfg.get("trigger_roles"),
        [
            "first_party_site", "third_party_backend_or_other", "identity_provider",
            "captcha_challenge", "payment_provider", "messaging_api",
            "form_backend", "cloud_api", "object_storage", "email_delivery",
            "file_storage", "hosting_platform", "rpc_provider", "walletconnect", "wallet_vendor",
        ],
    )
    negative_roles = _split_roles(cfg.get("negative_roles"), ["third_party_static", "analytics_ads", "software_update"])

    query_packets = packet_index if packet_index is not None else packets
    immediate = _packets_between(query_packets, t, t + immediate_s)
    if not immediate:
        return None
    past = _packets_between(query_packets, t - past_s, t) if past_s > 0 else []
    c_now = _role_byte_counter(immediate)
    c_past = _role_byte_counter(past)

    total_bytes = int(sum(p.length for p in immediate))
    important_bytes = _sum_roles(c_now, important_roles)
    backend_bytes = _sum_roles(c_now, backend_roles)
    unknown_bytes = int(c_now["unknown"])
    rpc_bytes = int(c_now["rpc_provider"])
    walletconnect_bytes = int(c_now["walletconnect"])
    wallet_vendor_bytes = int(c_now["wallet_vendor"])
    identity_bytes = int(c_now["identity_provider"])
    captcha_bytes = int(c_now["captcha_challenge"])
    payment_bytes = int(c_now["payment_provider"])
    messaging_bytes = int(c_now["messaging_api"])
    form_backend_bytes = int(c_now["form_backend"])
    cloud_api_bytes = int(c_now["cloud_api"])
    object_storage_bytes = int(c_now["object_storage"])
    email_delivery_bytes = int(c_now["email_delivery"])
    hosting_bytes = int(c_now["hosting_platform"])
    file_storage_bytes = int(c_now["file_storage"])
    url_shortener_bytes = int(c_now["url_shortener"])
    first_party_bytes = int(c_now["first_party_site"])
    negative_bytes = _sum_roles(c_now, negative_roles)
    nuisance_bytes = negative_bytes + wallet_vendor_bytes
    interaction_bytes = _sum_roles(c_now, important_roles)
    past_important = _sum_roles(c_past, important_roles)
    past_wallet_vendor = int(c_past["wallet_vendor"])
    trigger_bytes = _sum_roles(c_now, trigger_roles)

    min_important = int(cfg.get("min_important_bytes_3s", 3000) or 3000)
    min_total = int(cfg.get("min_total_bytes_3s", 2000) or 2000)
    if total_bytes < min_total:
        return None
    # Low-role/ECH mode: allow candidates with strong local flow-shape evidence
    # even when SNI/DNS roles are unavailable or misleading.
    # shape is computed below; keep this coarse filter permissive.
    if important_bytes < max(1, min_important // 3) and trigger_bytes < max(1, int(cfg.get("min_trigger_bytes_3s", 1500) or 1500) // 3) and total_bytes < max(1, int(cfg.get("min_total_bytes_3s", 2000) or 2000) * 2):
        return None

    immediate_flow_ids = {p.flow_id for p in immediate}
    new_flow_count = sum(1 for fl in immediate_flow_ids if flow_first.get(fl, t) >= t)
    immediate_hosts = {h for p in immediate for h in _host_parts(p.host)}
    new_host_count = sum(1 for h in immediate_hosts if host_first.get(h, t) >= t)

    shape = local_flow_change_features(
        query_packets,
        t,
        post_s=immediate_s,
        pre_s=float(cfg.get("shape_pre_window_s", max(4.0, past_s)) or max(4.0, past_s)),
        burst_gap_s=float(cfg.get("shape_burst_gap_s", cfg.get("burst_gap_s", 0.35)) or cfg.get("burst_gap_s", 0.35)),
        min_post_packets=int(cfg.get("shape_min_post_packets", 1) or 1),
    )

    burst_norm = max(1.0, float(cfg.get("burst_norm_bytes", 50000) or 50000))
    burst_score = _clamp(math.log1p(important_bytes) / math.log1p(burst_norm), 0.0, 1.5)
    backend_score = _clamp(math.log1p(max(backend_bytes, 0)) / math.log1p(burst_norm), 0.0, 1.5)
    quiet_norm = max(1.0, float(cfg.get("quiet_norm_bytes", 20000) or 20000))
    quiet_score = _clamp(1.0 - (past_important / quiet_norm), 0.0, 1.0)
    new_flow_score = _clamp(new_flow_count / max(1.0, float(cfg.get("new_flow_norm", 8) or 8)), 0.0, 1.0)
    new_host_score = _clamp(new_host_count / max(1.0, float(cfg.get("new_host_norm", 4) or 4)), 0.0, 1.0)

    # Role is auxiliary.  ECH/DoH can hide or distort roles, so this score is
    # capped and cannot dominate the candidate.  The main evidence is traffic
    # shape/rhythm salience below.
    role_trigger_score = 0.0
    if walletconnect_bytes > 0:
        role_trigger_score += 0.10
    if rpc_bytes > 0:
        role_trigger_score += 0.10
    if backend_bytes >= int(cfg.get("backend_trigger_bytes_3s", 4000) or 4000):
        role_trigger_score += 0.08
    if identity_bytes >= int(cfg.get("identity_trigger_bytes_3s", 1200) or 1200):
        role_trigger_score += 0.08
    if captcha_bytes >= int(cfg.get("captcha_trigger_bytes_3s", 800) or 800):
        role_trigger_score += 0.05
    if payment_bytes >= int(cfg.get("payment_trigger_bytes_3s", 1200) or 1200):
        role_trigger_score += 0.08
    if messaging_bytes >= int(cfg.get("messaging_trigger_bytes_3s", 800) or 800):
        role_trigger_score += 0.08
    if form_backend_bytes >= int(cfg.get("form_backend_trigger_bytes_3s", 800) or 800):
        role_trigger_score += 0.08
    if cloud_api_bytes >= int(cfg.get("cloud_api_trigger_bytes_3s", 1000) or 1000):
        role_trigger_score += 0.08
    if object_storage_bytes >= int(cfg.get("object_storage_trigger_bytes_3s", 1000) or 1000):
        role_trigger_score += 0.06
    if email_delivery_bytes >= int(cfg.get("email_delivery_trigger_bytes_3s", 600) or 600):
        role_trigger_score += 0.06
    if file_storage_bytes >= int(cfg.get("file_storage_trigger_bytes_3s", 1000) or 1000):
        role_trigger_score += 0.06
    if first_party_bytes >= int(cfg.get("first_party_trigger_bytes_3s", 3500) or 3500):
        role_trigger_score += 0.05
    if unknown_bytes >= int(cfg.get("unknown_trigger_bytes_3s", 12000) or 12000):
        role_trigger_score += 0.05
    role_aux_score = min(float(cfg.get("role_aux_cap", 0.30) or 0.30), role_trigger_score)

    static_frac = _safe_div(negative_bytes, total_bytes)
    static_penalty = 0.0
    if static_frac > float(cfg.get("static_frac_penalty_threshold", 0.35) or 0.35):
        static_penalty = min(1.0, static_frac) * float(cfg.get("static_penalty_weight", 0.85) or 0.85)

    vendor_only_penalty = 0.0
    vendor_frac = _safe_div(wallet_vendor_bytes, total_bytes)
    if vendor_frac > float(cfg.get("wallet_vendor_only_frac_threshold", 0.70) or 0.70) and backend_bytes < int(cfg.get("backend_trigger_bytes_3s", 4000) or 4000):
        vendor_only_penalty = float(cfg.get("wallet_vendor_only_penalty_weight", 0.75) or 0.75)

    shape_salience = float(shape["flow_shape_salience"])
    shape_top3 = float(shape["flow_shape_salience_top3_mean"])
    shape_change = float(shape["flow_shape_change_mean"])
    shape_intrinsic = float(shape["flow_shape_intrinsic_mean"])
    shape_multiflow = float(shape["flow_shape_multiflow_coordination"])
    stable_nochange_penalty = float(cfg.get("w_stable_nochange", 0.45) or 0.45) * float(shape["flow_shape_stable_nochange_byte_frac"])
    # Kit-centric score: local shape/rhythm change and intrinsic interaction
    # motifs are primary.  New-flow/new-host and role are weak auxiliaries only.
    score = (
        float(cfg.get("w_shape_salience", 1.35) or 1.35) * shape_salience
        + float(cfg.get("w_shape_top3", 0.70) or 0.70) * shape_top3
        + float(cfg.get("w_shape_change", 0.65) or 0.65) * shape_change
        + float(cfg.get("w_shape_intrinsic", 0.55) or 0.55) * shape_intrinsic
        + float(cfg.get("w_shape_multiflow", 0.45) or 0.45) * shape_multiflow
        + float(cfg.get("w_burst", 0.30) or 0.30) * burst_score
        + float(cfg.get("w_backend", 0.18) or 0.18) * backend_score
        + float(cfg.get("w_quiet", 0.08) or 0.08) * quiet_score
        + float(cfg.get("w_new_flow", 0.06) or 0.06) * new_flow_score
        + float(cfg.get("w_new_host", 0.04) or 0.04) * new_host_score
        + role_aux_score
        - static_penalty
        - vendor_only_penalty
        - stable_nochange_penalty
    )


    min_score = float(cfg.get("min_score", 1.15) or 1.15)
    if score < min_score:
        return None

    reasons: List[str] = []
    if shape_salience >= float(cfg.get("reason_shape_salience", 0.25) or 0.25):
        reasons.append("flow_shape_salience")
    if shape_change >= float(cfg.get("reason_shape_change", 0.18) or 0.18):
        reasons.append("same_flow_shape_change")
    if shape_intrinsic >= float(cfg.get("reason_shape_intrinsic", 0.30) or 0.30):
        reasons.append("interaction_motif")
    if shape_multiflow >= float(cfg.get("reason_multiflow", 0.25) or 0.25):
        reasons.append("multi_flow_shape_coordination")
    if important_bytes >= min_important:
        reasons.append("important_burst")
    if backend_bytes >= int(cfg.get("backend_trigger_bytes_3s", 4000) or 4000):
        reasons.append("backend_or_rpc")
    if identity_bytes >= int(cfg.get("identity_trigger_bytes_3s", 1200) or 1200):
        reasons.append("identity")
    if captcha_bytes >= int(cfg.get("captcha_trigger_bytes_3s", 800) or 800):
        reasons.append("captcha")
    if payment_bytes >= int(cfg.get("payment_trigger_bytes_3s", 1200) or 1200):
        reasons.append("payment")
    if messaging_bytes >= int(cfg.get("messaging_trigger_bytes_3s", 800) or 800):
        reasons.append("messaging")
    if form_backend_bytes >= int(cfg.get("form_backend_trigger_bytes_3s", 800) or 800):
        reasons.append("form_backend")
    if cloud_api_bytes >= int(cfg.get("cloud_api_trigger_bytes_3s", 1000) or 1000):
        reasons.append("cloud_api")
    if object_storage_bytes >= int(cfg.get("object_storage_trigger_bytes_3s", 1000) or 1000):
        reasons.append("object_storage")
    if email_delivery_bytes >= int(cfg.get("email_delivery_trigger_bytes_3s", 600) or 600):
        reasons.append("email_delivery")
    if file_storage_bytes >= int(cfg.get("file_storage_trigger_bytes_3s", 1000) or 1000):
        reasons.append("file_storage")
    if first_party_bytes >= int(cfg.get("first_party_trigger_bytes_3s", 3500) or 3500):
        reasons.append("first_party_followup")
    if unknown_bytes >= int(cfg.get("unknown_trigger_bytes_3s", 12000) or 12000):
        reasons.append("unknown_burst")
    if walletconnect_bytes > 0:
        reasons.append("walletconnect")
    if rpc_bytes > 0:
        reasons.append("rpc")
    # wallet/vendor context is not a positive reason; it is reported only via bytes fields.
    if quiet_score >= 0.75:
        reasons.append("post_quiet")
    if new_flow_count > 0:
        reasons.append("new_flow")
    if new_host_count > 0:
        reasons.append("new_host")
    if static_penalty:
        reasons.append("static_penalty")
    if stable_nochange_penalty:
        reasons.append("stable_no_local_shape_change_penalty")
    if vendor_only_penalty:
        reasons.append("wallet_vendor_only_penalty")

    emit = t + immediate_s
    return CandidateAnchor(
        start_epoch=float(t),
        emit_epoch=float(emit),
        rank=-1,
        score=float(score),
        source="traffic_heuristic",
        reason=";".join(reasons) or "traffic_shape",
        decision_delay_s=float(immediate_s),
        scan_step_s=float(scan_step_s),
        immediate_window_s=float(immediate_s),
        past_window_s=float(past_s),
        start_rel_s=float(t - pcap_start),
        emit_rel_s=float(emit - pcap_start),
        n_pkts_3s=len(immediate),
        bytes_3s=total_bytes,
        important_bytes_3s=important_bytes,
        unknown_bytes_3s=unknown_bytes,
        backend_bytes_3s=int(c_now["third_party_backend_or_other"]),
        rpc_bytes_3s=rpc_bytes,
        walletconnect_bytes_3s=walletconnect_bytes,
        wallet_vendor_bytes_3s=wallet_vendor_bytes,
        static_analytics_bytes_3s=negative_bytes,
        identity_bytes_3s=identity_bytes,
        captcha_bytes_3s=captcha_bytes,
        payment_bytes_3s=payment_bytes,
        messaging_bytes_3s=messaging_bytes,
        form_backend_bytes_3s=form_backend_bytes,
        cloud_api_bytes_3s=cloud_api_bytes,
        object_storage_bytes_3s=object_storage_bytes,
        email_delivery_bytes_3s=email_delivery_bytes,
        hosting_bytes_3s=hosting_bytes,
        file_storage_bytes_3s=file_storage_bytes,
        url_shortener_bytes_3s=url_shortener_bytes,
        first_party_bytes_3s=first_party_bytes,
        interaction_bytes_3s=interaction_bytes,
        nuisance_bytes_3s=nuisance_bytes,
        past_important_bytes=past_important,
        past_wallet_vendor_bytes=past_wallet_vendor,
        new_flow_count_3s=int(new_flow_count),
        new_host_count_3s=int(new_host_count),
        flow_shape_salience=float(shape_salience),
        flow_shape_salience_top3_mean=float(shape_top3),
        flow_shape_change_score=float(shape_change),
        flow_shape_intrinsic_score=float(shape_intrinsic),
        flow_shape_changed_byte_frac=float(shape["flow_shape_changed_byte_frac"]),
        flow_shape_stable_nochange_byte_frac=float(shape["flow_shape_stable_nochange_byte_frac"]),
        flow_shape_active_flow_count=float(shape["flow_shape_active_flow_count"]),
        flow_shape_high_salience_flow_count=float(shape["flow_shape_high_salience_flow_count"]),
        flow_shape_multiflow_coordination=float(shape_multiflow),
        quiet_score=float(quiet_score),
        burst_score=float(burst_score),
        role_trigger_score=float(role_trigger_score),
        role_aux_score=float(role_aux_score),
        stable_nochange_penalty=float(stable_nochange_penalty),
        static_penalty=float(static_penalty),
        wallet_vendor_only_penalty=float(vendor_only_penalty),
    )


def _merge_candidates(candidates: Sequence[CandidateAnchor], merge_within_s: float, top_k: int) -> List[CandidateAnchor]:
    if not candidates:
        return []
    merge_within_s = max(0.0, float(merge_within_s))
    # Keep the strongest local maxima; this avoids emitting many near-duplicates
    # from the same 3-second burst.
    sorted_by_score = sorted(candidates, key=lambda c: (-c.score, c.start_epoch))
    kept: List[CandidateAnchor] = []
    for cand in sorted_by_score:
        if any(abs(cand.start_epoch - k.start_epoch) <= merge_within_s for k in kept):
            continue
        kept.append(cand)
        if len(kept) >= top_k:
            break
    kept.sort(key=lambda c: c.start_epoch)
    # Rank by score while preserving the start time separately.
    for rank, cand in enumerate(sorted(kept, key=lambda c: (-c.score, c.start_epoch)), start=1):
        cand.rank = rank
    return sorted(kept, key=lambda c: c.rank)


def _normalise_candidate_config(cfg: Optional[dict]) -> dict:
    out = dict(cfg or {})
    # Backward-compatible aliases used by earlier experimental configs.
    if "decision_delay_s" not in out:
        for k in ("decision_lookahead_s", "lookahead_s", "immediate_window_s"):
            if k in out:
                out["decision_delay_s"] = out[k]
                break
    if "min_after_capture_start_s" not in out:
        for k in ("min_after_pcap_start_s", "min_after_start_s"):
            if k in out:
                out["min_after_capture_start_s"] = out[k]
                break
    if "max_scan_s" not in out:
        for k in ("max_after_pcap_start_s", "max_after_start_s"):
            if k in out:
                out["max_scan_s"] = out[k]
                break
    if "merge_within_s" not in out and "nms_gap_s" in out:
        out["merge_within_s"] = out["nms_gap_s"]
    if "min_trigger_bytes_3s" not in out and "min_trigger_bytes" in out:
        out["min_trigger_bytes_3s"] = out["min_trigger_bytes"]
    if "min_total_bytes_3s" not in out and "min_future_bytes" in out:
        out["min_total_bytes_3s"] = out["min_future_bytes"]
    return out


def generate_candidate_anchors(
    packets: Sequence[Packet],
    cfg: Optional[dict] = None,
    *,
    pcap_start: Optional[float] = None,
    pcap_end: Optional[float] = None,
    session_start: Optional[float] = None,
    post_load_epoch: Optional[float] = None,
) -> List[CandidateAnchor]:
    """Generate traffic-episode onset candidates from gateway-observable traffic.

    The default configuration is deliberately high-recall: it returns several
    candidate anchors per site session.  Downstream layer-1 scoring should run on
    each candidate and aggregate at session level.  User-action timestamps, when
    available, are evaluation references rather than candidate prerequisites.
    """
    cfg = _normalise_candidate_config(cfg)
    if bool(cfg.get("causal_all_local_maxima", False)):
        # Strict deployment replay: do not form a capture-global top-k.  The
        # continuous iterator freezes each local NMS decision only after its
        # full causal look-ahead has elapsed and yields every surviving onset.
        # Session/oracle instrumentation is intentionally not consulted here.
        return list(
            iter_continuous_candidate_anchors(
                packets,
                cfg,
                pcap_start=pcap_start,
                pcap_end=pcap_end,
            )
        )
    pkts = sorted([p for p in packets if p.ts is not None], key=lambda p: p.ts)
    if not pkts:
        return []
    start0 = float(pcap_start if pcap_start is not None else pkts[0].ts)
    end0 = float(pcap_end if pcap_end is not None else pkts[-1].ts)
    if "decision_delays_s" in cfg and cfg.get("decision_delays_s") is not None:
        raw_delays = cfg.get("decision_delays_s")
        if isinstance(raw_delays, str):
            delay_items = raw_delays.replace(",", " ").split()
        else:
            delay_items = list(raw_delays)
        decision_delays_s = sorted({max(0.25, float(x)) for x in delay_items}) or [max(0.25, float(cfg.get("decision_delay_s", cfg.get("immediate_window_s", 3.0)) or 3.0))]
    else:
        decision_delays_s = [max(0.25, float(cfg.get("decision_delay_s", cfg.get("immediate_window_s", 3.0)) or 3.0))]
    decision_delay_s = max(decision_delays_s)
    scan_step_s = max(0.05, float(cfg.get("scan_step_s", 0.5) or 0.5))
    top_k = max(1, int(cfg.get("top_k", 8) or 8))
    merge_within_s = max(0.0, float(cfg.get("merge_within_s", 4.0) or 4.0))

    first_party_first = _first_packet_time_for_role(pkts, ["first_party_site"])
    base_session_start = float(session_start if session_start is not None else (first_party_first if first_party_first is not None else start0))
    min_after_capture = max(0.0, float(cfg["min_after_capture_start_s"] if "min_after_capture_start_s" in cfg else 12.0))
    min_after_first_party = max(0.0, float(cfg["min_after_first_party_s"] if "min_after_first_party_s" in cfg else 5.0))
    scan_start = start0 + min_after_capture
    if first_party_first is not None:
        scan_start = max(scan_start, first_party_first + min_after_first_party)
    if post_load_epoch is not None:
        scan_start = max(
            scan_start,
            float(post_load_epoch) + max(0.0, float(cfg.get("min_after_post_load_s", 0.0) or 0.0)),
        )
    # If the capture starts long before browser navigation, session_start can be
    # supplied by a gateway sessionizer; do not scan before its post-load guard.
    scan_start = max(scan_start, base_session_start + max(0.0, float(cfg.get("min_after_session_start_s", 0.0) or 0.0)))
    max_scan_s = cfg.get("max_scan_s", None)
    scan_end = end0 - decision_delay_s
    if max_scan_s is not None:
        scan_end = min(scan_end, base_session_start + max(1.0, float(max_scan_s)))
    if scan_end <= scan_start:
        return []

    flow_first = _flow_first_times(pkts)
    host_first = _host_first_times(pkts)
    endpoint_first = _endpoint_first_times(pkts)
    site_first = _site_first_times(pkts)
    packet_index = build_flow_shape_index(pkts)
    candidates: List[CandidateAnchor] = []
    t = scan_start
    # Avoid accumulating floating point error too much by using an integer step loop.
    n_steps = int(max(0, math.floor((scan_end - scan_start) / scan_step_s))) + 1
    for i in range(n_steps):
        t = scan_start + i * scan_step_s
        for delay_s in decision_delays_s:
            local_cfg = dict(cfg)
            local_cfg["decision_delay_s"] = float(delay_s)
            cand = _score_candidate_at(t, pkts, flow_first, host_first, local_cfg, pcap_start=start0, packet_index=packet_index)
            if cand is not None:
                _attach_site_open_evidence(
                    cand, packet_index, endpoint_first, site_first, flow_first
                )
                if len(decision_delays_s) > 1:
                    cand.source = f"traffic_heuristic_ms{float(delay_s):g}s"
                    cand.reason = f"multi_scale_{float(delay_s):g}s;" + cand.reason
                candidates.append(cand)
    merged = _merge_candidates(candidates, merge_within_s=merge_within_s, top_k=top_k)
    if not merged and bool(cfg.get("fallback_if_empty", cfg.get("force_fallback", False))):
        # Conservative fallback: use the strongest non-initial burst after scan_start.
        # It is marked with a low score so downstream aggregation can treat it as weak.
        fallback_cfg = dict(cfg)
        fallback_cfg["min_score"] = -999.0
        fallback_cfg["min_total_bytes_3s"] = 1
        fallback_cfg["min_important_bytes_3s"] = 1
        fallback_cfg["min_trigger_bytes_3s"] = 1
        loose: List[CandidateAnchor] = []
        for i in range(n_steps):
            t = scan_start + i * scan_step_s
            for delay_s in decision_delays_s:
                fallback_cfg["decision_delay_s"] = float(delay_s)
                cand = _score_candidate_at(t, pkts, flow_first, host_first, fallback_cfg, pcap_start=start0, packet_index=packet_index)
                if cand is not None:
                    _attach_site_open_evidence(
                        cand, packet_index, endpoint_first, site_first, flow_first
                    )
                    cand.source = "traffic_heuristic_fallback"
                    cand.reason = "fallback;" + cand.reason
                    loose.append(cand)
        merged = _merge_candidates(loose, merge_within_s=merge_within_s, top_k=top_k)
    # Ensure rank order is by score, not by chronological order.
    ranked = sorted(merged, key=lambda c: (c.rank, c.start_epoch))
    return ranked


def iter_continuous_candidate_anchors(
    packets: Sequence[Packet],
    cfg: Optional[dict] = None,
    *,
    pcap_start: Optional[float] = None,
    pcap_end: Optional[float] = None,
    scan_block_s: float = 60.0,
) -> Iterator[CandidateAnchor]:
    """Yield traffic-only onset candidates across one unsegmented stream.

    This is the continuous counterpart of :func:`generate_candidate_anchors`.
    It deliberately does *not* reset candidate state at inactivity gaps, source
    capture boundaries, or rolling-horizon boundaries.  ``scan_block_s`` only
    bounds temporary computation; the packet/time indexes and the NMS carry are
    shared across blocks, so an onset and its pre/post context can cross a block
    boundary unchanged.

    The historical per-capture ``max_scan_s`` and ``top_k`` limits do not apply
    to a long-lived stream.  Applying them globally would inspect only the first
    few minutes or keep only a handful of candidates for an entire client-day.
    Instead, every local maximum passing the frozen traffic heuristic is yielded
    in chronological order.  Downstream Layer-1 scoring owns bounded causal
    candidate state and alert cooldown.
    """
    cfg = _normalise_candidate_config(cfg)
    pkts = sorted([p for p in packets if p.ts is not None], key=lambda p: p.ts)
    if not pkts:
        return

    start0 = float(pcap_start if pcap_start is not None else pkts[0].ts)
    end0 = float(pcap_end if pcap_end is not None else pkts[-1].ts)
    if "decision_delays_s" in cfg and cfg.get("decision_delays_s") is not None:
        raw_delays = cfg.get("decision_delays_s")
        if isinstance(raw_delays, str):
            delay_items = raw_delays.replace(",", " ").split()
        else:
            delay_items = list(raw_delays)
        decision_delays_s = sorted({max(0.25, float(x)) for x in delay_items})
    else:
        decision_delays_s = [
            max(
                0.25,
                float(
                    cfg.get(
                        "decision_delay_s", cfg.get("immediate_window_s", 3.0)
                    )
                    or 3.0
                ),
            )
        ]
    if not decision_delays_s:
        decision_delays_s = [3.0]

    scan_step_s = max(0.05, float(cfg.get("scan_step_s", 0.5) or 0.5))
    merge_within_s = max(0.0, float(cfg.get("merge_within_s", 4.0) or 4.0))
    block_s = max(scan_step_s, float(scan_block_s))
    scan_start = start0 + max(
        0.0, float(cfg.get("min_after_capture_start_s", 0.0) or 0.0)
    )
    if not bool(cfg.get("skip_first_party_scan_guard", False)):
        first_party_first = _first_packet_time_for_role(pkts, ["first_party_site"])
        if first_party_first is not None:
            scan_start = max(
                scan_start,
                float(first_party_first)
                + max(0.0, float(cfg.get("min_after_first_party_s", 0.0) or 0.0)),
            )
    scan_end = end0 - min(decision_delays_s)
    if scan_end < scan_start:
        return

    packet_index = build_flow_shape_index(pkts)
    flow_first = _flow_first_times(pkts)
    host_first = _host_first_times(pkts)
    endpoint_first = _endpoint_first_times(pkts)
    site_first = _site_first_times(pkts)
    carry: List[CandidateAnchor] = []
    ordinal = 0
    block_start = scan_start
    epsilon = scan_step_s * 1e-6
    while block_start <= scan_end + epsilon:
        block_end = min(scan_end, block_start + block_s)
        offset_steps = max(0, int(math.ceil((block_start - start0) / scan_step_s - 1e-9)))
        first_t = start0 + offset_steps * scan_step_s
        n_steps = int(max(0, math.floor((block_end - first_t) / scan_step_s + 1e-9))) + 1
        raw: List[CandidateAnchor] = list(carry)
        for i in range(n_steps):
            t = first_t + i * scan_step_s
            if t < block_start - epsilon or t > block_end + epsilon:
                continue
            for delay_s in decision_delays_s:
                if t + delay_s > end0 + epsilon:
                    continue
                local_cfg = dict(cfg)
                local_cfg["decision_delay_s"] = float(delay_s)
                cand = _score_candidate_at(
                    t,
                    pkts,
                    flow_first,
                    host_first,
                    local_cfg,
                    pcap_start=start0,
                    packet_index=packet_index,
                )
                if cand is None:
                    continue
                _attach_site_open_evidence(
                    cand, packet_index, endpoint_first, site_first, flow_first
                )
                if len(decision_delays_s) > 1:
                    cand.source = f"traffic_heuristic_stream_ms{float(delay_s):g}s"
                    cand.reason = f"continuous_multi_scale_{float(delay_s):g}s;" + cand.reason
                else:
                    cand.source = "traffic_heuristic_stream"
                raw.append(cand)

        merged = _merge_candidates(
            raw, merge_within_s=merge_within_s, top_k=max(1, len(raw))
        )
        is_last = block_end >= scan_end - epsilon
        finalize_before = float("inf") if is_last else block_end - merge_within_s
        finalized = sorted(
            (cand for cand in merged if cand.start_epoch < finalize_before),
            key=lambda cand: (cand.start_epoch, cand.emit_epoch, -cand.score),
        )
        carry = [] if is_last else [
            cand for cand in merged if cand.start_epoch >= finalize_before
        ]
        for cand in finalized:
            # Selecting the strongest local maximum needs the complete NMS
            # neighbourhood.  Although the candidate's own traffic window may
            # close earlier, it is not deployably observable as an NMS survivor
            # until every later proposal within merge_within_s has itself had
            # enough traffic to finish the longest decision window.
            causal_nms_available = (
                float(cand.start_epoch)
                + float(merge_within_s)
                + float(max(decision_delays_s))
            )
            cand.emit_epoch = max(float(cand.emit_epoch), causal_nms_available)
            cand.emit_rel_s = float(cand.emit_epoch - start0)
            ordinal += 1
            cand.rank = ordinal
            yield cand
        if is_last:
            break
        block_start = block_end + scan_step_s


def build_continuous_alert_session_assignments(
    candidates: Sequence[CandidateAnchor],
    cfg: Optional[dict] = None,
    *,
    visibility: str,
    site_open_boundaries: Optional[Sequence[dict]] = None,
) -> List[dict]:
    """Map continuous onset candidates into traffic-triggered alert sessions.

    The formal path supplies ``site_open_boundaries`` from
    :func:`detect_continuous_site_open_boundaries`, an independent full-stream
    traffic pass.  Onset proposals are then assigned by interval only.  The
    candidate-local boundary fallback is retained for old callers and focused
    unit tests, but is not the corrected E3 formal semantics.

    A candidate whose decision interval crosses the reset is ineligible on
    both sides, preventing future-page traffic from leaking into the prior
    alert.

    Normal visibility uses novelty of registrable SNI/host identities.  The
    ECH+DoH view uses novelty of remote Web-service IP:port identities only;
    DNS resolvers, push channels, and client ephemeral ports are excluded.
    Source-capture IDs, labelled visits, inactivity episodes, and rolling
    horizons are never inputs.
    """
    cfg = dict(cfg or {})
    if visibility not in {"sni_or_host", "endpoint"}:
        raise ValueError(f"unsupported alert-session visibility: {visibility!r}")
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            float(candidate.start_epoch), float(candidate.emit_epoch), -float(candidate.score)
        ),
    )
    if not ordered:
        return []
    quiet_score_min = float(cfg.get("quiet_score_min", 0.35) or 0.35)
    min_burst_score = max(0.0, float(cfg.get("min_burst_score", 0.10) or 0.10))
    min_visible_novelty = max(1, int(cfg.get("min_visible_novelty", 1) or 1))
    min_endpoint_novelty = max(1, int(cfg.get("min_endpoint_novelty", 2) or 2))
    min_web_flows = max(1, int(cfg.get("min_web_flows", 2) or 2))
    high_fanout_web_flows = max(
        min_web_flows, int(cfg.get("high_fanout_web_flows", 12) or 12)
    )
    min_http_flows = max(1, int(cfg.get("min_http_flows", 1) or 1))
    endpoint_quiet_min_web_flows = max(
        min_web_flows,
        int(cfg.get("endpoint_quiet_min_web_flows", 6) or 6),
    )
    boundary_merge_s = max(0.0, float(cfg.get("boundary_merge_s", 3.5) or 3.5))

    trigger_candidates: List[dict] = []
    if site_open_boundaries is not None:
        for raw in site_open_boundaries:
            boundary_epoch = float(raw["boundary_epoch"])
            if not math.isfinite(boundary_epoch):
                continue
            trigger_candidates.append({
                **dict(raw),
                "candidate": None,
                "boundary_epoch": boundary_epoch,
                "novelty": int(raw.get("novelty", 0)),
                "web_flows": int(raw.get("web_flows", 0)),
                "score": float(raw.get("score", 0.0)),
            })
    else:
        for candidate in ordered:
            novelty = (
                int(candidate.new_site_count_3s)
                if visibility == "sni_or_host"
                else int(candidate.new_endpoint_count_3s)
            )
            novelty_min = (
                min_visible_novelty
                if visibility == "sni_or_host"
                else min_endpoint_novelty
            )
            web_flows = int(candidate.new_web_flow_count_3s)
            quiet_transition = float(candidate.quiet_score) >= quiet_score_min
            if visibility == "endpoint":
                opening_context = bool(
                    web_flows >= high_fanout_web_flows
                    or (
                        quiet_transition
                        and web_flows >= endpoint_quiet_min_web_flows
                        and int(candidate.new_http_flow_count_3s) >= min_http_flows
                    )
                )
            else:
                opening_context = bool(
                    quiet_transition
                    or int(candidate.new_http_flow_count_3s) >= min_http_flows
                    or web_flows >= high_fanout_web_flows
                )
            site_open_like = bool(
                novelty >= novelty_min
                and web_flows >= min_web_flows
                and float(candidate.burst_score) >= min_burst_score
                and opening_context
            )
            boundary_epoch = (
                float(candidate.visible_site_open_epoch)
                if visibility == "sni_or_host"
                else float(candidate.endpoint_site_open_epoch)
            )
            if site_open_like and math.isfinite(boundary_epoch):
                trigger_candidates.append({
                    "candidate": candidate,
                    "boundary_epoch": boundary_epoch,
                    "novelty": novelty,
                    "web_flows": web_flows,
                    "score": float(candidate.score),
                })

    boundaries: List[dict] = []
    for trigger in sorted(
        trigger_candidates,
        key=lambda row: (
            float(row["boundary_epoch"]),
            -int(row["web_flows"]),
            -float(row.get("score", 0.0)),
        ),
    ):
        if boundaries and (
            float(trigger["boundary_epoch"])
            - float(boundaries[-1]["boundary_epoch"])
            <= boundary_merge_s
        ):
            boundaries[-1]["merged_triggers"] += int(
                trigger.get("merged_triggers", 1)
            )
            if (
                int(trigger["web_flows"]), float(trigger.get("score", 0.0))
            ) > (
                int(boundaries[-1]["web_flows"]), float(boundaries[-1].get("score", 0.0))
            ):
                preserved_epoch = float(boundaries[-1]["boundary_epoch"])
                merged = int(boundaries[-1]["merged_triggers"])
                boundaries[-1] = {**trigger, "boundary_epoch": preserved_epoch,
                                  "merged_triggers": merged}
            continue
        boundaries.append({
            **trigger,
            "merged_triggers": int(trigger.get("merged_triggers", 1)),
        })
    for index, boundary in enumerate(boundaries, start=1):
        boundary["alert_session_index"] = index
        boundary["session_end_epoch"] = (
            float(boundaries[index]["boundary_epoch"])
            if index < len(boundaries)
            else float("inf")
        )

    assignments: List[dict] = []
    local_ranks: Dict[int, int] = {}
    for candidate in ordered:
        candidate_start = float(candidate.start_epoch)
        candidate_emit = float(candidate.emit_epoch)
        active = None
        for boundary in boundaries:
            if float(boundary["boundary_epoch"]) <= candidate_start + 1e-9:
                active = boundary
            else:
                break
        eligible = bool(
            active is not None
            and candidate_emit < float(active["session_end_epoch"]) - 1e-9
        )
        session_index = int(active["alert_session_index"]) if active else 0
        if eligible:
            local_ranks[session_index] = local_ranks.get(session_index, 0) + 1
        local_rank = int(local_ranks.get(session_index, 0)) if eligible else 0
        assignments.append({
            "candidate": candidate,
            "alert_session_index": int(session_index),
            "alert_session_start_epoch": (
                float(active["boundary_epoch"]) if active else float("nan")
            ),
            "alert_session_end_epoch": (
                float(active["session_end_epoch"]) if active else float("nan")
            ),
            "alert_session_candidate_rank": int(local_rank),
            "alert_session_candidate_eligible": bool(eligible),
            "alert_session_candidate_crosses_reset": bool(
                active is not None
                and candidate_emit >= float(active["session_end_epoch"]) - 1e-9
            ),
            "alert_session_trigger": bool(eligible and local_rank == 1),
            "alert_session_trigger_fallback": False,
            "alert_session_trigger_visibility": visibility,
            "alert_session_trigger_novelty": (
                int(active["novelty"]) if active else 0
            ),
            "alert_session_trigger_web_flows": (
                int(active["web_flows"]) if active else 0
            ),
            "alert_session_trigger_site_open_like": bool(active is not None),
            "alert_session_trigger_merged_bursts": (
                int(active["merged_triggers"]) if active else 0
            ),
            "alert_session_count": int(len(boundaries)),
        })
    return assignments


def estimate_traffic_post_load_epoch(
    packets: Sequence[Packet],
    *,
    session_start: Optional[float] = None,
    cfg: Optional[dict] = None,
) -> Tuple[Optional[float], str]:
    """Estimate the end of the initial page-load burst from packet metadata only.

    Gateway-deployable replacement for the instrumentation-derived post-load
    guard (browser load events / HAR spans).  The load phase is modelled as the
    initial dense packet activity after session start; it ends at the first
    burst boundary where both hold:

      1. cumulative volume looks like a page load
         (``min_load_bytes`` / ``min_load_packets``), and
      2. the activity is followed by a low-rate quiet period of at least
         ``quiet_gap_s`` (per-bin byte rate <= ``quiet_max_bytes_per_s``;
         strict packet silence rarely happens on busy pages because of
         keepalives and analytics trickle).

    Causality: the returned epoch T is the end of the last loud bin; a gateway
    confirms "the load went quiet at T" by T + quiet_gap_s.  Candidates are
    anchored at t >= T and emit at t + decision_delay, so with the default
    delays the earliest emission trails the online confirmation time by at
    most a fraction of a second; conservative deployments can add the residual
    (quiet_gap_s - settle_pad_s - min decision delay, <=0.5s with defaults) to
    the emission accounting.

    Failure modes: if the volume threshold is never reached within
    ``max_load_s`` there is no page-load-like burst and no clipping is applied
    (returns session start).  If volume is reached but no quiet gap appears,
    the estimate caps at ``max_load_s`` after session start.  A session whose
    user interaction overlaps the load tail without a quiet gap in between gets
    a guard after the interaction onset; the no-guard ablation and
    ``scripts/audit_traffic_post_load_guard.py`` bound this risk.

    ``risk_cap_s`` optionally clips the estimated boundary to a latency learned
    from training-only interaction traces.  It is a suppression-safety control:
    inference remains traffic-only, while a low training quantile (for example
    alpha=0.01) limits how often a hard guard can land after an interaction.
    Because the cap can move the boundary into the load tail, downstream code
    should prefer a soft phase feature over irreversible candidate deletion.
    """
    cfg = dict(cfg or {})
    # Defaults follow the 2026-07-10 full-dataset audit (8,163 instrumented
    # sessions): quiet 10KB/s + gap 1.0s + cap 20s gave 0.0% guard-past-onset+3s
    # on blockchain/drainer and 7.7% on fast-clicking interaction captures,
    # with the guard median within 0.1s of the instrumentation guard.
    quiet_gap_s = max(0.1, float(cfg.get("quiet_gap_s", 1.0) or 1.0))
    quiet_max_bytes_per_s = max(0.0, float(cfg.get("quiet_max_bytes_per_s", 10000.0) or 0.0))
    quiet_bin_s = max(0.05, float(cfg.get("quiet_bin_s", 0.25) or 0.25))
    min_load_bytes = max(1, int(cfg.get("min_load_bytes", 30000) or 30000))
    min_load_packets = max(1, int(cfg.get("min_load_packets", 25) or 25))
    max_load_s = max(1.0, float(cfg.get("max_load_s", 20.0) or 20.0))
    settle_pad_s = max(0.0, float(cfg.get("settle_pad_s", 0.25) or 0.25))
    risk_cap_raw = cfg.get("risk_cap_s", None)
    risk_cap_s = None if risk_cap_raw is None else max(0.0, float(risk_cap_raw))

    pkts = sorted((p for p in packets if p.ts is not None), key=lambda p: p.ts)
    if not pkts:
        return None, "traffic_post_load.no_packets"
    t0 = float(session_start if session_start is not None else pkts[0].ts)
    horizon = t0 + max_load_s

    def _risk_bounded(epoch: float, source: str) -> Tuple[float, str]:
        if risk_cap_s is None:
            return epoch, source
        cap = t0 + risk_cap_s
        if epoch <= cap:
            return epoch, source
        return cap, f"{source}+risk_cap_{risk_cap_s:g}s"

    scoped = [p for p in pkts if t0 <= p.ts <= horizon]
    if not scoped:
        return _risk_bounded(t0, "traffic_post_load.no_packets_after_session_start")

    # Rate-based quiet: strict packet silence rarely happens on busy pages
    # (keepalives, analytics trickle), so a bin only blocks the quiet run when
    # its byte rate exceeds quiet_max_bytes_per_s.  The load ends at the last
    # loud bin that is followed by quiet_gap_s of low-rate bins, provided the
    # loud phase reached page-load volume by then.
    n_bins = int(math.ceil(max_load_s / quiet_bin_s))
    bin_bytes = [0] * n_bins
    bin_pkts = [0] * n_bins
    for p in scoped:
        i = min(n_bins - 1, int((float(p.ts) - t0) / quiet_bin_s))
        bin_bytes[i] += int(p.length)
        bin_pkts[i] += 1
    quiet_bin_max = quiet_max_bytes_per_s * quiet_bin_s
    quiet_bins_needed = max(1, int(math.ceil(quiet_gap_s / quiet_bin_s)))

    cum_bytes = 0
    cum_pkts = 0
    loud_cum_bytes = 0
    loud_cum_pkts = 0
    last_loud_end = t0
    quiet_run = 0
    for i in range(n_bins):
        cum_bytes += bin_bytes[i]
        cum_pkts += bin_pkts[i]
        if bin_bytes[i] > quiet_bin_max:
            quiet_run = 0
            last_loud_end = t0 + (i + 1) * quiet_bin_s
            loud_cum_bytes = cum_bytes
            loud_cum_pkts = cum_pkts
            continue
        quiet_run += 1
        if quiet_run >= quiet_bins_needed and loud_cum_bytes >= min_load_bytes and loud_cum_pkts >= min_load_packets:
            end = min(last_loud_end + settle_pad_s, horizon)
            return _risk_bounded(
                end,
                f"traffic_post_load.quiet_after_{loud_cum_bytes}b_{loud_cum_pkts}p",
            )
    if cum_bytes >= min_load_bytes and cum_pkts >= min_load_packets:
        # Page-load-like volume but never quiet inside the horizon: cap.
        return _risk_bounded(
            horizon,
            f"traffic_post_load.max_load_cap_{cum_bytes}b_{cum_pkts}p",
        )
    # Below the volume threshold there is nothing page-load-like to guard on.
    return _risk_bounded(t0, "traffic_post_load.no_load_burst")


def resolve_post_load_guard(
    anchor_cfg: Optional[dict],
    *,
    meta_post_load_epoch: Optional[float],
    meta_post_load_source: str,
    packets: Sequence[Packet],
    session_start: Optional[float],
) -> Tuple[Optional[float], str, str]:
    """Resolve the effective post-load guard for candidate generation.

    ``anchor.post_load_guard_mode`` selects the source:
      - ``instrumentation`` (legacy fallback when the key is omitted): browser
        load events / HAR spans from the sample metadata.  Offline-only; a real
        gateway cannot observe these.
      - ``none``: no guard (ablation).
      - ``traffic``: gateway-deployable traffic-only estimate, tunable via the
        ``anchor.traffic_post_load`` sub-dict.

    Returns ``(guard_epoch, guard_source, guard_mode)``.
    """
    cfg = dict(anchor_cfg or {})
    # Keep the implicit instrumentation fallback only so historical configs and
    # archived experiments remain reproducible.  Every current extraction config
    # must set this key explicitly; the parallel extractor can enforce the
    # expected mode at the experiment boundary.
    mode = str(cfg.get("post_load_guard_mode", "instrumentation") or "instrumentation").strip().lower()
    if mode in {"instrumentation", "meta", "events"}:
        return meta_post_load_epoch, str(meta_post_load_source or ""), "instrumentation"
    if mode in {"none", "off", "disabled"}:
        return None, "post_load_guard_disabled", "none"
    if mode in {"traffic", "traffic_only", "gateway"}:
        epoch, source = estimate_traffic_post_load_epoch(
            packets,
            session_start=session_start,
            cfg=cfg.get("traffic_post_load", {}) or {},
        )
        return epoch, source, "traffic"
    raise ValueError(
        f"unknown anchor.post_load_guard_mode={mode!r}; expected instrumentation, none, or traffic"
    )

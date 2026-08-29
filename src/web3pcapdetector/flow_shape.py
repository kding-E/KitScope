from __future__ import annotations

"""Traffic-shape primitives for kit-centric Layer 1/Layer 2 features.

These helpers intentionally do not depend on SNI/DNS role labels.  They describe
what a flow segment does locally: length morphology, rhythm, direction motifs,
burstiness, and how the segment differs from the same flow immediately before a
candidate onset.  This replaces the older idea of demoting long-lived flows; a
long-lived kit connection remains strong evidence when its local shape changes
or matches a phishing interaction motif.
"""

import math
from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .dynamic_window import build_bursts
from .pcap_minimal import Packet


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def _stats(prefix: str, values: Sequence[float]) -> dict[str, float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return {f"{prefix}_{k}": 0.0 for k in ["mean", "std", "p50", "p90", "max"]}
    arr = np.asarray(vals, dtype=float)
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_p50": float(np.percentile(arr, 50)),
        f"{prefix}_p90": float(np.percentile(arr, 90)),
        f"{prefix}_max": float(np.max(arr)),
    }


def direction_value(p: Packet) -> int:
    if p.direction == "up":
        return 1
    if p.direction == "down":
        return -1
    return 0


def flow_groups(packets: Sequence[Packet]) -> dict[Tuple, list[Packet]]:
    out: dict[Tuple, list[Packet]] = defaultdict(list)
    for p in packets:
        out[p.flow_id].append(p)
    return {k: sorted(v, key=lambda x: x.ts) for k, v in out.items()}


@dataclass
class FlowShapeIndex:
    """Per-sample time index for repeated local flow-shape queries."""

    packets: list[Packet]
    packet_ts: list[float]
    by_flow: dict[Tuple, list[Packet]]
    flow_ts: dict[Tuple, list[float]]

    @classmethod
    def from_packets(cls, packets: Sequence[Packet]) -> "FlowShapeIndex":
        ordered = sorted([p for p in packets if p.ts is not None], key=lambda p: p.ts)
        by_flow: dict[Tuple, list[Packet]] = defaultdict(list)
        for p in ordered:
            by_flow[p.flow_id].append(p)
        by_flow = {k: v for k, v in by_flow.items()}
        return cls(
            packets=ordered,
            packet_ts=[float(p.ts) for p in ordered],
            by_flow=by_flow,
            flow_ts={k: [float(p.ts) for p in v] for k, v in by_flow.items()},
        )

    def packets_between(self, start: float, end: float) -> list[Packet]:
        lo = bisect_left(self.packet_ts, float(start))
        hi = bisect_left(self.packet_ts, float(end))
        return self.packets[lo:hi]

    def active_flow_ids_between(self, start: float, end: float) -> set[Tuple]:
        return {p.flow_id for p in self.packets_between(start, end)}

    def flow_segment(self, flow_id: Tuple, start: float, end: float) -> list[Packet]:
        fps = self.by_flow.get(flow_id, [])
        if not fps:
            return []
        ts = self.flow_ts.get(flow_id, [])
        lo = bisect_left(ts, float(start))
        hi = bisect_left(ts, float(end))
        return fps[lo:hi]


def build_flow_shape_index(packets: Sequence[Packet]) -> FlowShapeIndex:
    return FlowShapeIndex.from_packets(packets)


def _vector_values(stats: dict[str, float]) -> np.ndarray:
    # A compact, bounded vector for pre/post distance.  Constants are broad
    # deployment-scale normalisers, not fitted from role/domain data.
    names_and_scales = [
        ("n_pkts_log", 4.0),
        ("bytes_log", 12.0),
        ("duration_log", 3.5),
        ("up_pkt_frac", 1.0),
        ("up_byte_frac", 1.0),
        ("dir_switch_rate", 1.0),
        ("up_then_down_rate", 1.0),
        ("large_pkt_frac", 1.0),
        ("small_pkt_frac", 1.0),
        ("len_log_mean", 8.0),
        ("len_log_std", 4.0),
        ("len_log_p90", 8.0),
        ("iat_rel_p50", 8.0),
        ("iat_rel_p90", 12.0),
        ("burst_count_log", 3.0),
        ("burst_bytes_max_log", 12.0),
    ]
    return np.asarray([float(stats.get(n, 0.0)) / max(s, 1e-6) for n, s in names_and_scales], dtype=float)


def segment_stats(packets: Sequence[Packet], *, start: float | None = None, end: float | None = None, burst_gap_s: float = 0.35) -> dict[str, float]:
    pkts = sorted([p for p in packets if (start is None or p.ts >= start) and (end is None or p.ts < end)], key=lambda p: p.ts)
    n = len(pkts)
    if not pkts:
        return {
            "n_pkts": 0.0,
            "n_pkts_log": 0.0,
            "bytes_total": 0.0,
            "bytes_log": 0.0,
            "duration_s": 0.0,
            "duration_log": 0.0,
            "up_pkt_frac": 0.0,
            "up_byte_frac": 0.0,
            "dir_switch_rate": 0.0,
            "up_then_down_rate": 0.0,
            "large_pkt_frac": 0.0,
            "small_pkt_frac": 0.0,
            "len_log_mean": 0.0,
            "len_log_std": 0.0,
            "len_log_p90": 0.0,
            "iat_rel_p50": 0.0,
            "iat_rel_p90": 0.0,
            "burst_count": 0.0,
            "burst_count_log": 0.0,
            "burst_bytes_max_log": 0.0,
            "intrinsic_salience": 0.0,
        }
    lengths = np.asarray([float(max(0, int(p.payload_len if p.payload_len else p.length))) for p in pkts], dtype=float)
    total = float(np.sum(lengths))
    ts = np.asarray([float(p.ts) for p in pkts], dtype=float)
    duration = float(max(0.0, ts[-1] - ts[0])) if n > 1 else 0.0
    dirs = np.asarray([direction_value(p) for p in pkts], dtype=float)
    up_mask = dirs > 0
    down_mask = dirs < 0
    iats = np.diff(ts) if len(ts) > 1 else np.asarray([], dtype=float)
    pos_iats = iats[iats > 1e-6]
    scale = float(np.median(pos_iats)) if len(pos_iats) else 1.0
    rel_iats = np.clip(iats / max(scale, 1e-6), 0.0, 30.0) if len(iats) else np.asarray([], dtype=float)
    log_len = np.log1p(lengths)
    switches = float(np.sum(dirs[1:] != dirs[:-1])) if len(dirs) > 1 else 0.0
    up_then_down = float(np.sum((dirs[:-1] > 0) & (dirs[1:] < 0))) if len(dirs) > 1 else 0.0
    bursts = build_bursts(pkts, burst_gap_s=float(burst_gap_s))
    max_burst_bytes = max([b.bytes_total for b in bursts], default=0)
    up_bytes = float(np.sum(lengths[up_mask])) if len(lengths) else 0.0
    stat = {
        "n_pkts": float(n),
        "n_pkts_log": float(math.log1p(n)),
        "bytes_total": total,
        "bytes_log": float(math.log1p(total)),
        "duration_s": duration,
        "duration_log": float(math.log1p(duration)),
        "up_pkt_frac": safe_div(float(np.sum(up_mask)), n),
        "down_pkt_frac": safe_div(float(np.sum(down_mask)), n),
        "up_byte_frac": safe_div(up_bytes, total),
        "dir_switch_rate": safe_div(switches, max(n - 1, 1)),
        "up_then_down_rate": safe_div(up_then_down, max(n - 1, 1)),
        "large_pkt_frac": safe_div(float(np.sum(lengths >= 1000.0)), n),
        "small_pkt_frac": safe_div(float(np.sum(lengths <= 80.0)), n),
        "len_log_mean": float(np.mean(log_len)),
        "len_log_std": float(np.std(log_len)),
        "len_log_p90": float(np.percentile(log_len, 90)),
        "iat_rel_p50": float(np.percentile(rel_iats, 50)) if len(rel_iats) else 0.0,
        "iat_rel_p90": float(np.percentile(rel_iats, 90)) if len(rel_iats) else 0.0,
        "burst_count": float(len(bursts)),
        "burst_count_log": float(math.log1p(len(bursts))),
        "burst_bytes_max_log": float(math.log1p(max_burst_bytes)),
    }
    # Role-free intrinsic phishing-interaction salience: request/response shape,
    # packet volume, burstiness, and direction alternation.  This deliberately
    # does not care whether the flow is new or old.
    vol = min(1.0, stat["bytes_log"] / math.log1p(20000.0))
    pkt = min(1.0, stat["n_pkts_log"] / math.log1p(18.0))
    burst = min(1.0, stat["burst_bytes_max_log"] / math.log1p(16000.0))
    bidir = 1.0 - abs(stat["up_byte_frac"] - 0.5) * 2.0
    motif = min(1.0, 0.5 * stat["dir_switch_rate"] + 0.5 * stat["up_then_down_rate"])
    size_floor = min(1.0, total / 900.0)
    stat["intrinsic_salience"] = float(np.clip((0.25 * vol + 0.20 * pkt + 0.25 * burst + 0.15 * bidir + 0.15 * motif) * size_floor, 0.0, 1.0))
    return stat


def prepost_distance(pre_stats: dict[str, float], post_stats: dict[str, float]) -> float:
    pre = _vector_values(pre_stats)
    post = _vector_values(post_stats)
    return float(np.clip(np.sqrt(np.mean((post - pre) ** 2)), 0.0, 3.0) / 3.0)


def local_flow_change_features(
    packets: Sequence[Packet] | FlowShapeIndex,
    t: float,
    *,
    post_s: float,
    pre_s: float = 6.0,
    burst_gap_s: float = 0.35,
    min_post_packets: int = 1,
) -> dict[str, float]:
    """Compute candidate-local flow-shape salience.

    Long-lived flows are not penalised.  They become strong evidence when their
    post-candidate segment changes shape/rhythm relative to the same flow's
    previous segment or when their post-candidate segment has strong intrinsic
    request/response/burst motifs.  Only segments that are old *and* locally
    unchanged with low intrinsic salience contribute to stable-no-change mass.
    """
    index = packets if isinstance(packets, FlowShapeIndex) else FlowShapeIndex.from_packets(packets)
    active_flow_ids = index.active_flow_ids_between(float(t), float(t) + float(post_s))
    per_flow = []
    total_post_bytes = 0.0
    stable_nochange_bytes = 0.0
    changed_bytes = 0.0
    for flow_id in active_flow_ids:
        post = index.flow_segment(flow_id, float(t), float(t) + float(post_s))
        if len(post) < int(min_post_packets):
            continue
        pre = index.flow_segment(flow_id, float(t) - float(pre_s), float(t))
        post_stats = segment_stats(post, burst_gap_s=burst_gap_s)
        pre_stats = segment_stats(pre, burst_gap_s=burst_gap_s)
        change = prepost_distance(pre_stats, post_stats) if pre else post_stats["intrinsic_salience"]
        intrinsic = float(post_stats["intrinsic_salience"])
        # Stable kit feature score: local change and intrinsic motif both count.
        salience = float(np.clip(0.55 * change + 0.45 * intrinsic, 0.0, 1.0))
        bytes_total = float(post_stats["bytes_total"])
        total_post_bytes += bytes_total
        if pre and change < 0.08 and intrinsic < 0.22:
            stable_nochange_bytes += bytes_total
        if change >= 0.18 or intrinsic >= 0.35:
            changed_bytes += bytes_total
        per_flow.append({
            "flow_id": flow_id,
            "salience": salience,
            "shape_change": change,
            "intrinsic": intrinsic,
            "bytes": bytes_total,
            "n_pkts": post_stats["n_pkts"],
            "had_pre": 1.0 if pre else 0.0,
        })
    if not per_flow:
        return {
            "flow_shape_salience": 0.0,
            "flow_shape_salience_top3_mean": 0.0,
            "flow_shape_change_max": 0.0,
            "flow_shape_change_mean": 0.0,
            "flow_shape_intrinsic_max": 0.0,
            "flow_shape_intrinsic_mean": 0.0,
            "flow_shape_changed_byte_frac": 0.0,
            "flow_shape_stable_nochange_byte_frac": 0.0,
            "flow_shape_active_flow_count": 0.0,
            "flow_shape_high_salience_flow_count": 0.0,
            "flow_shape_pre_context_frac": 0.0,
            "flow_shape_multiflow_coordination": 0.0,
        }
    sal = np.asarray([r["salience"] for r in per_flow], dtype=float)
    chg = np.asarray([r["shape_change"] for r in per_flow], dtype=float)
    intr = np.asarray([r["intrinsic"] for r in per_flow], dtype=float)
    weights = np.asarray([max(1.0, r["bytes"]) for r in per_flow], dtype=float)
    high = sal >= 0.35
    top3 = np.sort(sal)[-min(3, len(sal)):]
    return {
        "flow_shape_salience": float(np.average(sal, weights=weights)),
        "flow_shape_salience_top3_mean": float(np.mean(top3)) if len(top3) else 0.0,
        "flow_shape_change_max": float(np.max(chg)),
        "flow_shape_change_mean": float(np.average(chg, weights=weights)),
        "flow_shape_intrinsic_max": float(np.max(intr)),
        "flow_shape_intrinsic_mean": float(np.average(intr, weights=weights)),
        "flow_shape_changed_byte_frac": safe_div(changed_bytes, total_post_bytes),
        "flow_shape_stable_nochange_byte_frac": safe_div(stable_nochange_bytes, total_post_bytes),
        "flow_shape_active_flow_count": float(len(per_flow)),
        "flow_shape_high_salience_flow_count": float(np.sum(high)),
        "flow_shape_pre_context_frac": safe_div(float(sum(r["had_pre"] for r in per_flow)), len(per_flow)),
        "flow_shape_multiflow_coordination": float(np.clip(np.sum(high) / 3.0, 0.0, 1.0) * np.clip(np.mean(top3) if len(top3) else 0.0, 0.0, 1.0)),
    }


def window_flow_shape_features(packets: Sequence[Packet], *, start: float, end: float, pre_s: float = 6.0, burst_gap_s: float = 0.35) -> dict[str, float]:
    post_s = max(0.0, float(end) - float(start))
    feats = local_flow_change_features(packets, float(start), post_s=post_s, pre_s=float(pre_s), burst_gap_s=float(burst_gap_s))
    return {f"kitshape_{k}": float(v) for k, v in feats.items()}


def aggregate_flow_segment_features(packets: Sequence[Packet], *, start: float | None = None, end: float | None = None, burst_gap_s: float = 0.35) -> dict[str, float]:
    """Aggregate role-free per-flow segment features for Layer2 unit vectors."""
    groups = flow_groups([p for p in packets if (start is None or p.ts >= start) and (end is None or p.ts < end)])
    stats = [segment_stats(v, burst_gap_s=burst_gap_s) for v in groups.values() if v]
    if not stats:
        out = {}
        for prefix in ["seg_intrinsic", "seg_len_log_mean", "seg_iat_rel_p50", "seg_dir_switch", "seg_burst_bytes"]:
            out.update(_stats(prefix, []))
        out["seg_count"] = 0.0
        out["seg_high_intrinsic_frac"] = 0.0
        return out
    out = {"seg_count": float(len(stats))}
    out.update(_stats("seg_intrinsic", [s["intrinsic_salience"] for s in stats]))
    out.update(_stats("seg_len_log_mean", [s["len_log_mean"] for s in stats]))
    out.update(_stats("seg_iat_rel_p50", [s["iat_rel_p50"] for s in stats]))
    out.update(_stats("seg_dir_switch", [s["dir_switch_rate"] for s in stats]))
    out.update(_stats("seg_burst_bytes", [s["burst_bytes_max_log"] for s in stats]))
    out["seg_high_intrinsic_frac"] = safe_div(sum(1 for s in stats if s["intrinsic_salience"] >= 0.35), len(stats))
    return out

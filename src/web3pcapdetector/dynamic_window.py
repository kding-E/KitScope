from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from .pcap_minimal import Packet


@dataclass
class Window:
    name: str
    start: float
    end: float
    reason: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class Burst:
    start: float
    end: float
    n_pkts: int
    bytes_total: int

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def build_bursts(packets: Sequence[Packet], burst_gap_s: float) -> List[Burst]:
    if not packets:
        return []
    pkts = sorted(packets, key=lambda p: p.ts)
    bursts: List[Burst] = []
    b_start = pkts[0].ts
    b_end = pkts[0].ts
    b_n = 1
    b_bytes = pkts[0].length
    last = pkts[0].ts
    for p in pkts[1:]:
        if p.ts - last > burst_gap_s:
            bursts.append(Burst(b_start, b_end, b_n, b_bytes))
            b_start = p.ts
            b_n = 0
            b_bytes = 0
        b_end = p.ts
        b_n += 1
        b_bytes += p.length
        last = p.ts
    bursts.append(Burst(b_start, b_end, b_n, b_bytes))
    return bursts


def dynamic_episode_window(
    packets: Sequence[Packet],
    anchor: float,
    capture_end: Optional[float],
    *,
    min_window_s: float = 15.0,
    max_window_s: float = 90.0,
    burst_gap_s: float = 0.35,
    idle_gap_s: float = 5.0,
    tail_pad_s: float = 0.25,
    min_burst_packets: int = 4,
    min_burst_bytes: int = 1024,
    use_capture_stop_if_no_idle: bool = True,
) -> Window:
    """Return a dynamic interaction-relevant traffic-episode window.

    The episode starts at a traffic candidate or an offline interaction-reference
    anchor. It ends at the first meaningful burst followed by a sufficiently long
    idle gap, or at capture stop if the capture ended before an idle gap. This is
    intentionally not a fixed 60s rule, so short completed episodes remain valid.
    """
    hard_end = anchor + max_window_s
    if capture_end is not None:
        hard_end = min(hard_end, capture_end)
    post = [p for p in packets if anchor <= p.ts <= hard_end]
    if not post:
        return Window("dyn", anchor, hard_end, "no_post_packets")
    bursts = build_bursts(post, burst_gap_s=burst_gap_s)
    sig = [b for b in bursts if b.n_pkts >= min_burst_packets or b.bytes_total >= min_burst_bytes]
    if not sig:
        end = min(hard_end, post[-1].ts + tail_pad_s)
        return Window("dyn", anchor, end, "no_significant_burst")

    # Search for an idle gap after a significant burst and after the minimum duration.
    for i, b in enumerate(sig):
        if b.end - anchor < min_window_s:
            continue
        nxt = sig[i + 1] if i + 1 < len(sig) else None
        if nxt is not None:
            gap = nxt.start - b.end
            if gap >= idle_gap_s:
                end = min(hard_end, b.end + tail_pad_s)
                return Window("dyn", anchor, end, f"idle_gap_{gap:.3f}s_after_burst")
        else:
            # Last significant burst. If no more significant bursts, the episode ends there.
            end = min(hard_end, b.end + tail_pad_s)
            if capture_end is not None and capture_end <= end + idle_gap_s and use_capture_stop_if_no_idle:
                # Capture ended around the prompt or after activity; keep the actual capture tail.
                return Window("dyn", anchor, hard_end, "capture_stop_after_last_burst")
            return Window("dyn", anchor, end, "last_significant_burst")

    # No significant burst reached the minimum episode length. For a short capture
    # stop, keep the full capture; otherwise cap at min_window_s rather than
    # drifting to a hard 90s window.
    if capture_end is not None and capture_end <= anchor + min_window_s:
        return Window("dyn", anchor, hard_end, "capture_stop_before_min_window")
    end = min(hard_end, anchor + min_window_s)
    return Window("dyn", anchor, end, "min_window_no_late_burst")


def _split_role_list(value, default: Sequence[str]) -> set[str]:
    if value is None:
        return {str(x).lower() for x in default}
    if isinstance(value, str):
        items = value.replace(",", " ").split()
    else:
        items = list(value)
    out = {str(x).strip().lower() for x in items if str(x).strip()}
    return out or {str(x).lower() for x in default}


def adaptive_episode_window(
    packets: Sequence[Packet],
    anchor: float,
    capture_end: Optional[float],
    *,
    min_window_s: float = 1.25,
    max_window_s: float = 14.0,
    burst_gap_s: float = 0.35,
    idle_gap_s: float = 0.90,
    tail_pad_s: float = 0.10,
    min_packets: int = 4,
    min_bytes: int = 1200,
    min_important_bytes: int = 900,
    important_roles: Sequence[str] = (
        "unknown", "first_party_site", "third_party_backend_or_other",
        "identity_provider", "captcha_challenge", "payment_provider",
        "messaging_api", "form_backend", "cloud_api", "object_storage",
        "email_delivery", "file_storage", "url_shortener", "hosting_platform",
        "rpc_provider", "walletconnect",
    ),
    nuisance_roles: Sequence[str] = ("wallet_vendor", "analytics_ads", "third_party_static", "software_update"),
    close_on_saturation: bool = True,
    saturation_bytes: int = 12000,
    saturation_important_frac: float = 0.40,
) -> Window:
    """Return a low-latency adaptive candidate window.

    Unlike the old fixed observation horizon, this window closes when enough
    interaction evidence has been observed and either the local burst becomes
    idle or the interaction evidence saturates.  The maximum duration is only a
    safety cap; the common case is governed by observed packet/burst structure.
    This is deployable because the emitted end is never earlier than the time at
    which the gateway has observed the required evidence or the idle timeout.
    """
    hard_end = float(anchor) + max(0.25, float(max_window_s))
    if capture_end is not None:
        hard_end = min(hard_end, float(capture_end))
    post = sorted([p for p in packets if float(anchor) <= p.ts <= hard_end], key=lambda p: p.ts)
    if not post:
        return Window("adaptive", float(anchor), hard_end, "adaptive_no_post_packets")

    important = _split_role_list(important_roles, ())
    nuisance = _split_role_list(nuisance_roles, ())
    min_t = float(anchor) + max(0.0, float(min_window_s))
    min_packets = max(1, int(min_packets))
    min_bytes = max(1, int(min_bytes))
    min_important_bytes = max(0, int(min_important_bytes))

    def evidence_until(t: float) -> tuple[int, int, int, int]:
        sub = [p for p in post if p.ts <= t]
        n = len(sub)
        total = int(sum(p.length for p in sub))
        imp = int(sum(p.length for p in sub if str(p.role or "unknown").lower() in important))
        nui = int(sum(p.length for p in sub if str(p.role or "unknown").lower() in nuisance))
        return n, total, imp, nui

    bursts = build_bursts(post, burst_gap_s=float(burst_gap_s))
    last_seen = post[-1].ts
    for i, b in enumerate(bursts):
        candidate_end = max(float(b.end) + max(0.0, float(tail_pad_s)), min_t)
        if candidate_end > hard_end:
            break
        n, total, imp, nui = evidence_until(float(b.end))
        enough = n >= min_packets and total >= min_bytes and (imp >= min_important_bytes or min_important_bytes == 0)
        if not enough:
            continue
        imp_frac = imp / total if total else 0.0
        nui_frac = nui / total if total else 0.0
        if bool(close_on_saturation) and total >= int(saturation_bytes) and imp_frac >= float(saturation_important_frac) and nui_frac < 0.80:
            return Window("adaptive", float(anchor), min(candidate_end, hard_end), "adaptive_saturated_evidence")
        next_b = bursts[i + 1] if i + 1 < len(bursts) else None
        if next_b is None:
            # In offline extraction we know this is the last observed burst.  A
            # gateway would emit after the idle timeout; include that idle wait
            # in the window end so latency accounting remains conservative.
            end = min(max(candidate_end, float(b.end) + float(idle_gap_s)), hard_end)
            return Window("adaptive", float(anchor), end, "adaptive_last_burst_idle_timeout")
        gap = float(next_b.start) - float(b.end)
        if gap >= float(idle_gap_s):
            end = min(max(candidate_end, float(b.end) + float(idle_gap_s)), hard_end)
            return Window("adaptive", float(anchor), end, f"adaptive_idle_gap_{gap:.3f}s")

    # If the interaction never reaches the evidence gate, keep a short prefix up
    # to the last observed packet or the safety cap.  This produces a weak but
    # usable candidate row instead of silently dropping the session.
    end = min(max(min_t, float(last_seen) + max(0.0, float(tail_pad_s))), hard_end)
    return Window("adaptive", float(anchor), end, "adaptive_weak_evidence_cap")


# Backward-compatible API aliases.  Historical scripts and serialized experiment
# recipes still import these names; new code should use the neutral episode names.
dynamic_postconnect_window = dynamic_episode_window
adaptive_postconnect_window = adaptive_episode_window


def fixed_windows(anchor: float, capture_end: Optional[float], seconds: Sequence[float]) -> List[Window]:
    wins: List[Window] = []
    for s in seconds:
        end = anchor + float(s)
        reason = f"fixed_{s:g}s"
        if capture_end is not None and end > capture_end:
            end = capture_end
            reason = f"fixed_{s:g}s_clipped_by_capture"
        wins.append(Window(f"w{int(s) if float(s).is_integer() else s}", anchor, end, reason))
    return wins

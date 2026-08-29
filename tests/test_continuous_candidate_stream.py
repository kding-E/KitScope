from __future__ import annotations

import web3pcapdetector.candidates as candidates
from web3pcapdetector.candidates import CandidateAnchor, iter_continuous_candidate_anchors
from web3pcapdetector.pcap_minimal import Packet


def _packet(ts: float, role: str = "unknown") -> Packet:
    return Packet(
        ts=ts,
        length=1200,
        ip_version=4,
        proto="TCP",
        src_ip="10.0.0.1",
        dst_ip="198.18.0.1",
        src_port=12345,
        dst_port=443,
        direction="up",
        payload_len=1100,
        flow_id=("10.0.0.1", 12345, "198.18.0.1", 443, "TCP"),
        role=role,
    )


def test_continuous_scanner_waits_for_first_party_role_by_default(monkeypatch):
    def fake_score(t, packets, flow_first, host_first, cfg, pcap_start, packet_index=None):
        rounded = round(float(t), 6)
        if rounded not in {10.0, 105.0}:
            return None
        return CandidateAnchor(
            start_epoch=rounded,
            emit_epoch=rounded + 1.0,
            rank=-1,
            score=2.0,
            source="traffic_heuristic",
            reason="test",
            decision_delay_s=1.0,
            scan_step_s=1.0,
            immediate_window_s=1.0,
            past_window_s=4.0,
        )

    monkeypatch.setattr(candidates, "_score_candidate_at", fake_score)
    packets = [_packet(0.0), _packet(100.0, role="first_party_site"), _packet(120.0)]
    observed = list(
        iter_continuous_candidate_anchors(
            packets,
            {
                "scan_step_s": 1.0,
                "decision_delays_s": [1.0],
                "min_after_capture_start_s": 0.0,
                "min_after_first_party_s": 5.0,
                "merge_within_s": 2.0,
            },
            pcap_start=0.0,
            pcap_end=120.0,
            scan_block_s=30.0,
        )
    )
    assert [item.start_epoch for item in observed] == [105.0]


def test_continuous_scanner_can_skip_first_party_guard_for_load_only_routes(monkeypatch):
    def fake_score(t, packets, flow_first, host_first, cfg, pcap_start, packet_index=None):
        rounded = round(float(t), 6)
        if rounded != 10.0:
            return None
        return CandidateAnchor(
            start_epoch=rounded,
            emit_epoch=rounded + 1.0,
            rank=-1,
            score=2.0,
            source="traffic_heuristic",
            reason="test",
            decision_delay_s=1.0,
            scan_step_s=1.0,
            immediate_window_s=1.0,
            past_window_s=4.0,
        )

    monkeypatch.setattr(candidates, "_score_candidate_at", fake_score)
    packets = [_packet(0.0), _packet(100.0, role="first_party_site"), _packet(120.0)]
    observed = list(
        iter_continuous_candidate_anchors(
            packets,
            {
                "scan_step_s": 1.0,
                "decision_delays_s": [1.0],
                "min_after_capture_start_s": 0.0,
                "min_after_first_party_s": 5.0,
                "skip_first_party_scan_guard": True,
                "merge_within_s": 2.0,
            },
            pcap_start=0.0,
            pcap_end=120.0,
            scan_block_s=30.0,
        )
    )
    assert [item.start_epoch for item in observed] == [10.0]

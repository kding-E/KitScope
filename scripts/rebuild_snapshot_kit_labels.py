#!/usr/bin/env python3
"""Rebuild the static kit-label chain for one immutable snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_dataset_key(value: str) -> str:
    text = str(value).strip().replace("\\", "/").casefold().rstrip("/")
    marker = "phish_dataset/"
    return text[text.index(marker):] if marker in text else text


def _add_canonical_dataset_key(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "canonical_dataset_key" not in out.columns:
        path_col = "sample_path" if "sample_path" in out.columns else "zip_path" if "zip_path" in out.columns else None
        if path_col is not None:
            out["canonical_dataset_key"] = out[path_col].map(canonical_dataset_key)
    return out


def _usable_unique_key(frame: pd.DataFrame, column: str) -> bool:
    if column not in frame.columns:
        return False
    values = frame[column].fillna("").astype(str)
    if values.eq("").any():
        return False
    return not values.duplicated().any()


def _select_join_key(left: pd.DataFrame, right: pd.DataFrame, left_name: str, right_name: str) -> str:
    for column in ("capture_id", "sample_key", "canonical_dataset_key"):
        if _usable_unique_key(left, column) and _usable_unique_key(right, column):
            return column
    raise ValueError(f"cannot find a unique key shared by {left_name} and {right_name}")


def _run(python: str, arguments: list[str], log_dir: Path, step: str) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    command = [python, "-u", *arguments]
    stdout_path = log_dir / f"{step}.stdout.log"
    stderr_path = log_dir / f"{step}.stderr.log"
    print(f"[kit-label-rebuild] {step}: {' '.join(command)}", flush=True)
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        proc = subprocess.run(command, cwd=str(ROOT), env=env, stdout=stdout, stderr=stderr)
    if proc.returncode:
        raise RuntimeError(f"{step} failed with exit {proc.returncode}; see {stderr_path}")


def _build_kit_manifest(snapshot_path: Path, static_features: Path, assignments_path: Path,
                        labels_path: Path, out_path: Path) -> dict:
    snapshot = _add_canonical_dataset_key(pd.read_csv(snapshot_path, low_memory=False, dtype=str))
    snapshot = snapshot[snapshot["label"].eq("phishing")].copy()
    assignments = _add_canonical_dataset_key(pd.read_csv(assignments_path, low_memory=False, dtype=str))
    labels = _add_canonical_dataset_key(pd.read_csv(labels_path, low_memory=False, dtype=str))

    assignment_join_key = _select_join_key(snapshot, assignments, "snapshot", "assignments")
    label_join_key = _select_join_key(snapshot, labels, "snapshot", "final labels")
    outside = set(labels[label_join_key].astype(str)) - set(snapshot[label_join_key].astype(str))
    if outside:
        raise ValueError(f"final labels contain {len(outside)} records outside the frozen snapshot")

    assignment_columns = [
        "static_family_key", "evidence_tier", "evidence_note",
        "primary_evidence_type", "primary_evidence_channel",
        "independent_support_channel_count", "strong_supporting_key_count",
    ]
    assignment_columns = [column for column in assignment_columns if column in assignments and column != assignment_join_key]
    label_columns = [
        "static_family_key", "kit_cluster",
        "backend_kit_training_label", "label_correction_final", "ndist",
    ]
    label_columns = [column for column in label_columns if column in labels and column != label_join_key]
    assignment_view = assignments[[assignment_join_key, *assignment_columns]].copy()
    assignment_view = assignment_view.rename(
        columns={assignment_join_key: "__assignment_join_key", "static_family_key": "assignment_family_key"}
    )
    label_view = labels[[label_join_key, *label_columns]].copy()
    label_view = label_view.rename(
        columns={label_join_key: "__label_join_key", "static_family_key": "kit_family_id", "kit_cluster": "backend_cluster_id"}
    )
    snapshot = snapshot.assign(
        __assignment_join_key=snapshot[assignment_join_key].astype(str),
        __label_join_key=snapshot[label_join_key].astype(str),
    )
    merged = snapshot.merge(
        assignment_view,
        on="__assignment_join_key", how="left", validate="one_to_one",
    ).merge(
        label_view,
        on="__label_join_key", how="left", validate="one_to_one",
    )
    for column, default in (
        ("kit_family_id", ""),
        ("backend_cluster_id", ""),
        ("evidence_tier", "unknown"),
        ("assignment_family_key", ""),
    ):
        if column not in merged.columns:
            merged[column] = default
    merged["kit_family_id"] = merged["kit_family_id"].fillna("").astype(str)
    merged["backend_cluster_id"] = merged["backend_cluster_id"].fillna("").astype(str)
    merged["kit_evidence_tier"] = (
        merged["evidence_tier"].fillna("").astype(str).replace({"": "unknown"})
    )
    training = merged.get("backend_kit_training_label", pd.Series("0", index=merged.index))
    merged["fit_eligible"] = training.astype(str).str.lower().isin({"1", "true", "yes"}) & merged["kit_family_id"].ne("")
    merged["label_version"] = "final_static_evidence_snapshot"
    merged["label_source"] = "multi_party_static_evidence"
    static_sha = sha256_file(static_features)
    merged["label_hash"] = merged.apply(
        lambda row: hashlib.sha256(
            "\x1f".join([
                str(row.capture_id), str(row.kit_family_id), str(row.backend_cluster_id),
                str(row.kit_evidence_tier), str(row.get("primary_evidence_channel", "")),
                static_sha,
            ]).encode("utf-8")
        ).hexdigest(),
        axis=1,
    )
    keep = [
        "capture_id", "sample_path", "source", "source_variant", "kit_family_id",
        "backend_cluster_id", "kit_evidence_tier", "evidence_note",
        "primary_evidence_type", "primary_evidence_channel",
        "independent_support_channel_count", "strong_supporting_key_count",
        "label_version", "label_source", "label_hash", "fit_eligible",
    ]
    keep = [column for column in keep if column in merged]
    merged[keep].sort_values("capture_id", kind="mergesort").to_csv(out_path, index=False)
    summary = {
        "snapshot_manifest": str(snapshot_path.resolve()),
        "snapshot_manifest_sha256": sha256_file(snapshot_path),
        "static_features": str(static_features.resolve()),
        "static_features_sha256": static_sha,
        "assignments_sha256": sha256_file(assignments_path),
        "kit_labels_sha256": sha256_file(labels_path),
        "kit_label_manifest": str(out_path.resolve()),
        "kit_label_manifest_sha256": sha256_file(out_path),
        "assignment_join_key": assignment_join_key,
        "label_join_key": label_join_key,
        "phishing_sessions": int(len(merged)),
        "assignment_coverage": int(merged.get("assignment_family_key", pd.Series("", index=merged.index)).fillna("").ne("").sum()),
        "kit_labeled_sessions": int(merged["kit_family_id"].ne("").sum()),
        "fit_eligible_sessions": int(merged["fit_eligible"].sum()),
        "kit_families": int(merged.loc[merged.kit_family_id.ne(""), "kit_family_id"].nunique()),
        "evidence_tier_counts": merged["kit_evidence_tier"].value_counts(dropna=False).to_dict(),
    }
    return summary


def run(args: argparse.Namespace) -> int:
    python = str(Path(args.python).resolve()) if args.python else sys.executable
    snapshot_path = Path(args.snapshot_manifest)
    static_features = Path(args.static_features)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    logs = out_root / "logs"
    assignments_dir = out_root / "static_assignments"
    cluster_dir = out_root / "backend_kit_label_clusters"
    fragment_dir = out_root / "backend_kit_fragment_labels"
    static_neighbor_dir = out_root / "backend_kit_static_neighbor_labels"
    audit_dir = out_root / "backend_kit_audit_labels"
    evidence_dir = out_root / "backend_kit_evidence_labels"
    final_dir = out_root / "backend_kit_final_labels"
    assignments = assignments_dir / "static_family_label_assignments.csv"

    steps = [
        ("assignments", ["scripts/build_backend_static_family_assignments.py", "--static-features", str(static_features), "--out-dir", str(assignments_dir), "--min-count", str(args.min_count)]),
        ("cluster", [
            "scripts/cluster_backend_kit_labels.py",
            "--assignments-csv", str(assignments),
            "--out-dir", str(cluster_dir),
            "--min-family-samples", str(args.cluster_min_family_samples),
        ]),
        ("fragment", ["scripts/merge_backend_kit_fragments.py", "--assignments-csv", str(assignments), "--cluster-labels-csv", str(cluster_dir / "backend_kit_label_clusters.csv"), "--out-dir", str(fragment_dir)]),
        ("static_neighbor", ["scripts/merge_backend_kit_static_neighbors.py", "--assignments-csv", str(assignments), "--fragment-labels-csv", str(fragment_dir / "backend_kit_fragment_labels.csv"), "--out-dir", str(static_neighbor_dir)]),
        ("audit", ["scripts/apply_backend_kit_audit_corrections.py", "--assignments-csv", str(assignments), "--static-neighbor-labels-csv", str(static_neighbor_dir / "backend_kit_static_neighbor_labels.csv"), "--out-dir", str(audit_dir)]),
        ("evidence", ["scripts/rebuild_backend_kit_evidence_corrections.py", "--assignments-csv", str(assignments), "--static-neighbor-labels-csv", str(static_neighbor_dir / "backend_kit_static_neighbor_labels.csv"), "--audit-labels-csv", str(audit_dir / "backend_kit_audit_labels.csv"), "--out-dir", str(evidence_dir)]),
        ("final", ["scripts/build_backend_kit_final_labels.py", "--assignments-csv", str(assignments), "--evidence-labels-csv", str(evidence_dir / "backend_kit_evidence_labels.csv"), "--out-dir", str(final_dir)]),
    ]
    for step, arguments in steps:
        _run(python, arguments, logs, step)

    labels_path = final_dir / "backend_kit_final_labels.csv"
    kit_manifest = out_root / "kit_label_manifest.csv"
    summary = _build_kit_manifest(snapshot_path, static_features, assignments, labels_path, kit_manifest)
    if args.old_labels:
        stability_dir = out_root / "label_stability_vs_previous"
        _run(python, [
            "scripts/audit_layer2_label_stability.py",
            "--old-labels", str(args.old_labels),
            "--new-labels", str(final_dir / "trace_sample_split_kit_labels.csv"),
            "--out-dir", str(stability_dir),
        ], logs, "stability")
        summary["previous_labels"] = str(Path(args.old_labels).resolve())
        summary["previous_labels_sha256"] = sha256_file(Path(args.old_labels))
    summary["code_sha256"] = {
        path.name: sha256_file(path)
        for path in [Path(__file__), *[ROOT / arguments[0] for _, arguments in steps]]
    }
    (out_root / "kit_label_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-manifest", required=True)
    parser.add_argument("--static-features", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--old-labels", default=None)
    parser.add_argument("--min-count", type=int, default=4)
    parser.add_argument("--cluster-min-family-samples", type=int, default=5)
    parser.add_argument("--python", default=None)
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))

#!/usr/bin/env python3
from __future__ import annotations

"""Build audited backend-kit labels as a conservative correction on top of static-neighbor labels.

The static-neighbor labels are traffic-observable, but the manual strict-HAR audit found
two places where that is not aligned with the research target: distinguishing
actual phishing kit families.

The audited correction keeps static-neighbor provenance and applies only audited, evidence-backed
corrections:

* Split the weak static-neighbor super-family 67e63193 back to its fragment labels. Its static-neighbor
  merge was driven by a high-frequency inline hash and common wallet/provider
  APIs, while the core backend hosts, scripts, WalletConnect project ids, and
  interaction patterns are not consistently shared.
* Merge same-kit fragments where static DOM/HAR evidence is stronger than the
  previous boundary: exact domain overlap plus matching HTML structure/backend
  workflow for the 5e70/c213/93cb cluster, and distinctive shared scripts for
  the 62a/67a vote-site cluster.

The script does not use Layer2 traffic predictions as evidence.
"""

import argparse
import csv
import json
import pathlib
import sys
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from merge_backend_kit_static_neighbors import _as_set, _edge_metrics, _family_profile, _jaccard, _short  # noqa: E402


SPLIT_TO_FRAGMENT = {
    "backend_kit:67e63193ef99a9883b": "split_weak_static_neighbor_superfamily_to_fragment",
}

MERGE_TO_TARGET = {
    "backend_kit:93cbcd481fc75afdad": "backend_kit:5e70c2da68c309abd5",
    "backend_kit:c21318f9170b265fa7": "backend_kit:5e70c2da68c309abd5",
    "backend_kit:67a8fa8f91d0187aba": "backend_kit:62a11ff014262b0ad6",
}

EVIDENCE_COLS = [
    "backend_endpoints",
    "backend_request_schemas",
    "backend_route_set_hashes",
    "backend_flow_hashes",
    "backend_kit_hashes",
    "script_content_hashes",
    "inline_script_hashes",
    "body_template_hash",
    "html_structure_hashes",
    "resource_graph_hashes",
    "kit_behavior_hashes",
    "image_asset_hashes",
    "walletconnect_project_ids",
    "js_capabilities",
    "interaction_pattern",
]
PASSTHROUGH_COLS = [
    "capture_id", "sample_key", "label", "source", "source_variant",
    "source_folder", "fine_source", "partition", "fit_role",
]


def _token_set(rows: pd.DataFrame, col: str) -> set[str]:
    out: set[str] = set()
    if col not in rows.columns:
        return out
    for value in rows[col].fillna(""):
        parsed = _as_set(value)
        if parsed:
            out |= {x for x in parsed if x not in {"", "[]", "nan", "none", "null"}}
        elif str(value).strip() and str(value).strip().lower() not in {"nan", "none", "null"}:
            out.add(str(value).strip().lower())
    return out


def _top_values(rows: pd.DataFrame, col: str, n: int = 8) -> list[tuple[str, int]]:
    counter: Counter[str] = Counter()
    if col not in rows.columns:
        return []
    for value in rows[col].fillna(""):
        parsed = _as_set(value)
        if parsed:
            counter.update(x for x in parsed if x not in {"", "[]", "nan", "none", "null"})
        elif str(value).strip() and str(value).strip().lower() not in {"nan", "none", "null"}:
            counter.update([str(value).strip().lower()])
    return counter.most_common(n)


def _compare_groups(name_a: str, rows_a: pd.DataFrame, name_b: str, rows_b: pd.DataFrame) -> dict[str, Any]:
    profile_a = _family_profile(name_a, rows_a)
    profile_b = _family_profile(name_b, rows_b)
    edge = _edge_metrics(profile_a, profile_b)
    extra: dict[str, Any] = {}
    for col in EVIDENCE_COLS:
        set_a = _token_set(rows_a, col)
        set_b = _token_set(rows_b, col)
        if set_a or set_b:
            extra[col] = {
                "a_n": int(len(set_a)),
                "b_n": int(len(set_b)),
                "shared": int(len(set_a & set_b)),
                "jaccard": round(float(_jaccard(set_a, set_b)), 4),
                "shared_examples": sorted(set_a & set_b)[:8],
            }
    domains_a = set(rows_a["domain"].dropna().astype(str))
    domains_b = set(rows_b["domain"].dropna().astype(str))
    return {
        "a": name_a,
        "b": name_b,
        "a_short": _short(name_a),
        "b_short": _short(name_b),
        "a_samples": int(len(rows_a)),
        "b_samples": int(len(rows_b)),
        "a_domains": int(len(domains_a)),
        "b_domains": int(len(domains_b)),
        "domain_overlap": int(len(domains_a & domains_b)),
        "domain_overlap_examples": sorted(domains_a & domains_b)[:20],
        "edge": {
            k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else int(v) if isinstance(v, (int, np.integer)) else v)
            for k, v in edge.items()
        },
        "evidence_overlap": extra,
    }


def _build_evidence_report(frame: pd.DataFrame) -> dict[str, Any]:
    report: dict[str, Any] = {
        "route": "backend_kit_audit_labels_conservative_static_correction",
        "policy": {
            "uses_traffic_predictions": False,
            "split_to_fragment": SPLIT_TO_FRAGMENT,
            "merge_to_target": MERGE_TO_TARGET,
        },
        "corrections": [],
    }

    for static_neighbor_family, reason in SPLIT_TO_FRAGMENT.items():
        rows = frame[frame["static_family_key_static_neighbor"].astype(str).eq(static_neighbor_family)]
        members = sorted(rows["static_family_key_fragment"].dropna().astype(str).unique())
        pair_evidence = []
        for i, fam_a in enumerate(members):
            for fam_b in members[i + 1:]:
                pair_evidence.append(
                    _compare_groups(
                        fam_a,
                        rows[rows["static_family_key_fragment"].astype(str).eq(fam_a)],
                        fam_b,
                        rows[rows["static_family_key_fragment"].astype(str).eq(fam_b)],
                    )
                )
        report["corrections"].append(
            {
                "type": "split",
                "reason": reason,
                "source_static_neighbor_family": static_neighbor_family,
                "source_static_neighbor_short": _short(static_neighbor_family),
                "n_samples": int(len(rows)),
                "n_domains": int(rows["domain"].dropna().astype(str).nunique()),
                "fragment_members": members,
                "fragment_member_shorts": [_short(x) for x in members],
                "pair_evidence": pair_evidence,
                "aibot2_after_split": frame[frame["domain"].astype(str).str.lower().eq("aibot2.vip")][
                    ["domain", "sample_id", "static_family_key_fragment", "static_family_key_audit"]
                ].drop_duplicates().to_dict("records"),
            }
        )

    for source, target in MERGE_TO_TARGET.items():
        source_rows = frame[frame["static_family_key_static_neighbor"].astype(str).eq(source)]
        target_rows = frame[frame["static_family_key_static_neighbor"].astype(str).eq(target)]
        report["corrections"].append(
            {
                "type": "merge",
                "reason": "audited_static_dom_har_neighbor",
                "source_static_neighbor_family": source,
                "source_static_neighbor_short": _short(source),
                "target_audit_family": target,
                "target_audit_short": _short(target),
                "n_source_samples": int(len(source_rows)),
                "n_target_samples_before": int(len(target_rows)),
                "evidence": _compare_groups(source, source_rows, target, target_rows),
                "source_top_patterns": _top_values(source_rows, "interaction_pattern"),
                "target_top_patterns": _top_values(target_rows, "interaction_pattern"),
            }
        )
    return report


def build_audit_labels(args: argparse.Namespace) -> dict[str, Any]:
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(args.static_neighbor_labels_csv, low_memory=False)
    assignments = pd.read_csv(args.assignments_csv, low_memory=False, usecols=lambda c: c in {"zip_path", *EVIDENCE_COLS})
    labels["zip_path"] = labels["zip_path"].astype(str)
    assignments["zip_path"] = assignments["zip_path"].astype(str)
    frame = labels.merge(assignments, on="zip_path", how="left")
    frame = frame[frame["static_family_key"].astype(str).str.startswith("backend_kit:")].copy()
    frame["static_family_key_static_neighbor"] = frame["static_family_key"].astype(str)

    if "static_family_key_fragment" not in frame.columns:
        raise ValueError("Static-neighbor labels must include static_family_key_fragment for audit correction provenance.")

    frame["static_family_key_audit"] = frame["static_family_key_static_neighbor"]
    frame["label_correction_audit"] = "unchanged_from_static_neighbor"

    for static_neighbor_family, reason in SPLIT_TO_FRAGMENT.items():
        mask = frame["static_family_key_static_neighbor"].astype(str).eq(static_neighbor_family)
        frame.loc[mask, "static_family_key_audit"] = frame.loc[mask, "static_family_key_fragment"].astype(str)
        frame.loc[mask, "label_correction_audit"] = reason

    for source, target in MERGE_TO_TARGET.items():
        mask = frame["static_family_key_static_neighbor"].astype(str).eq(source)
        frame.loc[mask, "static_family_key_audit"] = target
        frame.loc[mask, "label_correction_audit"] = "merge_static_dom_har_neighbor_to_" + _short(target)

    evidence_report = _build_evidence_report(frame)

    keep = [
        "zip_path",
        "sample_id",
        "domain",
        *[c for c in PASSTHROUGH_COLS if c in frame.columns],
        "static_family_key_audit",
        "static_family_key_static_neighbor",
        "static_family_key_fragment",
        "static_family_key_cluster",
        "kit_family_key",
        "kit_cluster",
        "ndist",
        "label_correction_audit",
    ]
    keep = list(dict.fromkeys(keep))
    keep = [c for c in keep if c in frame.columns]
    labels_audit = frame[keep].copy()
    labels_audit = labels_audit.rename(columns={"static_family_key_audit": "static_family_key"})
    labels_audit["backend_kit_training_label"] = 1
    labels_audit.to_csv(out / "backend_kit_audit_labels.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    summary = (
        frame.groupby("static_family_key_audit")
        .agg(
            n_samples=("zip_path", "size"),
            n_domains=("domain", "nunique"),
            n_static_neighbor_families=("static_family_key_static_neighbor", "nunique"),
            n_fragment_families=("static_family_key_fragment", "nunique"),
            static_neighbor_members=("static_family_key_static_neighbor", lambda s: " || ".join(sorted(set(map(str, s))))),
            fragment_members=("static_family_key_fragment", lambda s: " || ".join(sorted(set(map(str, s))))),
            corrections=("label_correction_audit", lambda s: " || ".join(sorted(set(map(str, s))))),
        )
        .reset_index()
        .rename(columns={"static_family_key_audit": "static_family_key"})
        .sort_values(["n_samples", "n_static_neighbor_families", "static_family_key"], ascending=[False, False, True])
    )
    summary.to_csv(out / "backend_kit_audit_labels_family_summary.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    changed = frame[frame["static_family_key_audit"].astype(str).ne(frame["static_family_key_static_neighbor"].astype(str))].copy()
    changed[
        [
            "zip_path",
            "sample_id",
            "domain",
            "static_family_key_fragment",
            "static_family_key_static_neighbor",
            "static_family_key_audit",
            "label_correction_audit",
        ]
    ].to_csv(out / "backend_kit_audit_labels_corrections.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    validation = {
        **evidence_report,
        "inputs": {
            "static_neighbor_labels_csv": str(args.static_neighbor_labels_csv),
            "assignments_csv": str(args.assignments_csv),
        },
        "n_samples": int(len(frame)),
        "n_families_static_neighbor": int(frame["static_family_key_static_neighbor"].nunique()),
        "n_families_audit": int(frame["static_family_key_audit"].nunique()),
        "n_changed_samples": int(len(changed)),
        "changed_by_reason": dict(Counter(changed["label_correction_audit"].astype(str))),
        "outputs": {
            "labels": str(out / "backend_kit_audit_labels.csv"),
            "family_summary": str(out / "backend_kit_audit_labels_family_summary.csv"),
            "corrections": str(out / "backend_kit_audit_labels_corrections.csv"),
        },
    }
    (out / "backend_kit_audit_labels_validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8",
    )
    return validation


def write_trace_splits(args: argparse.Namespace) -> dict[str, Any]:
    out = pathlib.Path(args.out_dir)
    labels = pd.read_csv(out / "backend_kit_audit_labels.csv", low_memory=False)
    labels["true_family"] = labels["static_family_key"].astype(str)
    labels["training_family_label"] = labels["true_family"]
    if args.base_split_csv:
        base_split = pd.read_csv(args.base_split_csv, low_memory=False)
        label_map = dict(zip(base_split["zip_path"].astype(str), base_split.get("label", "phishing")))
        labels["label"] = labels["zip_path"].astype(str).map(label_map).fillna("phishing")
    else:
        labels["label"] = "phishing"

    split_cols = [
        "zip_path", "sample_id", "domain",
        *[c for c in ("capture_id", "sample_key", "source", "source_variant", "source_folder", "fine_source") if c in labels.columns],
        "label", "true_family", "training_family_label",
    ]
    split_input = labels[list(dict.fromkeys(split_cols))].copy()
    n_rows = len(split_input)
    test_size = float(args.test_size)
    if n_rows < 2:
        train_idx = np.arange(n_rows)
        test_idx = np.array([], dtype=int)
    else:
        n_test = int(np.ceil(test_size * n_rows)) if 0.0 < test_size < 1.0 else int(test_size)
        n_test = max(1, min(n_rows - 1, n_test))
        family_counts = split_input["true_family"].astype(str).value_counts()
        n_families = len(family_counts)
        stratify = (
            split_input["true_family"].astype(str)
            if n_families > 1 and family_counts.min() >= 2 and n_test >= n_families and (n_rows - n_test) >= n_families
            else None
        )
        train_idx, test_idx = train_test_split(
            np.arange(n_rows),
            test_size=test_size,
            random_state=int(args.random_state),
            stratify=stratify,
        )
    split_input["split"] = "train"
    split_input.loc[test_idx, "split"] = "test"
    split_input = split_input.sort_values(["split", "true_family", "zip_path"]).reset_index(drop=True)
    split_input.to_csv(out / "trace_sample_split_audit_labels.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    split_input[split_input["split"].eq("train")].to_csv(out / "split_train_pred_audit_labels.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    split_input[split_input["split"].eq("test")].to_csv(out / "split_test_pred_audit_labels.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    summary = {
        "rows": int(len(split_input)),
        "families": int(split_input["true_family"].nunique()),
        "train": int((split_input["split"] == "train").sum()),
        "test": int((split_input["split"] == "test").sum()),
        "families_train_ge15": int((split_input[split_input["split"].eq("train")].groupby("true_family").size() >= 15).sum()),
        "trace_sample_split_audit_labels": str(out / "trace_sample_split_audit_labels.csv"),
    }
    (out / "trace_sample_split_audit_labels_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build audited backend-kit labels from static-neighbor labels plus DOM/HAR evidence.")
    parser.add_argument("--static-neighbor-labels-csv", required=True)
    parser.add_argument("--assignments-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--base-split-csv", default="")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    validation = build_audit_labels(args)
    split_summary = write_trace_splits(args)
    print(
        json.dumps(
            {
                "n_samples": validation["n_samples"],
                "n_families_static_neighbor": validation["n_families_static_neighbor"],
                "n_families_audit": validation["n_families_audit"],
                "n_changed_samples": validation["n_changed_samples"],
                "changed_by_reason": validation["changed_by_reason"],
                "split": split_summary,
                "out_dir": str(args.out_dir),
            },
            indent=2,
            ensure_ascii=False,
            default=float,
        )
    )


if __name__ == "__main__":
    main()

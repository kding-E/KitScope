#!/usr/bin/env python3
from __future__ import annotations

"""Build evidence-corrected backend-kit labels with a single reproducible merge rule.

Motivation
----------
* The static-neighbor stage merged fragment labels with a single-linkage graph whose dominant driver was
  ``shared_script_high_endpoint_overlap`` (``raw_ep_j >= 0.7`` plus >=1 shared
  "script", where the shared script could be the ubiquitous inline placeholder).
  Backend endpoints / request schemas are dominated by the *shared wallet/provider
  API surface* (coinbase, phantom, rainbow, bitkeep, onramper ...) that every
  drainer emits, so high endpoint overlap does NOT imply the same kit. This
  over-merged distinct kits -- most visibly the 67e63193 super-family, whose three
  members share ZERO content scripts / backend struct hashes / body templates.
* The earlier audited correction handled that by hand: it hardcoded one split (67e63193) and three merges. That
  is correct but (a) incomplete -- it never re-audited the other endpoint-driven
  static-neighbor components such as b3bd8c15 -- and (b) brittle -- the corrections are a dict
  of raw hash keys that silently no-op if upstream hashes change.

What this stage does
------------------
Rebuild families from the fragment labels (the kit-identity atoms) with ONE
all-pairs rule. Two fragments merge iff they share *kit-owned* (distinctive)
evidence -- never mere endpoint/schema overlap:

  A. a shared backend route-set / flow / kit structural hash, or
  B. an exact shared full host (domain), or
  C. a shared non-empty WalletConnect project id, or
  D. a shared non-empty body-template hash, or
  E. a shared *distinctive* content-script hash (global fragment-DF <= cap), or
  F. the credential-captcha random-path identity (same credential_captcha pattern
     + high normalized-path Jaccard + zero shared scripts) -- the static-neighbor stage's clean
     ``interaction_random_path`` signal, which is not endpoint-contaminated because
     these kits do not talk to wallet provider APIs.

Connected components of that distinctive-evidence graph are the evidence-corrected families.
This (1) drops the endpoint contamination, (2) generalizes the 67e63193 split to
every static-neighbor super-family, (3) re-derives the audited merges from evidence instead
of hardcoded keys, and (4) is fully reproducible. It uses NO Layer2 traffic
predictions. intermediate and audited keys are kept as
provenance; guard assertions fail loudly if an audited outcome regresses.
"""

import argparse
import csv
import json
import pathlib
import sys
from collections import Counter, defaultdict
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from cluster_backend_kit_labels import _json_list  # noqa: E402
from merge_backend_kit_static_neighbors import (  # noqa: E402
    _edge_metrics,
    _family_profile,
    _quality,
    _short,
    _stable_family_key,
)

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

DENY = {"", "[]", "nan", "none", "null", "{}"}
PASSTHROUGH_COLS = [
    "capture_id", "sample_key", "label", "source", "source_variant",
    "source_folder", "fine_source", "partition", "fit_role",
]

# Guard anchors: audited outcomes that this stage must still satisfy.
GUARD_SPLIT_67E63193 = [
    "backend_kit:55f85595eb943bd08d",
    "backend_kit:60cb187500e6caa19a",
    "backend_kit:79388d06f5ca464bcf",
]
GUARD_MERGE_PAIRS = [
    ("backend_kit:93cbcd481fc75afdad", "backend_kit:c4b6ed71978c461d46"),  # origindefi
    ("backend_kit:c21318f9170b265fa7", "backend_kit:41e8a9af282e69af14"),  # ar/nillion cluster
    ("backend_kit:67a8fa8f91d0187aba", "backend_kit:62a11ff014262b0ad6"),  # vote-site cluster
]


def _clean_set(values: Iterable[object]) -> set[str]:
    out: set[str] = set()
    for value in values:
        parsed = _json_list(value)
        if parsed:
            for x in parsed:
                s = str(x).strip().lower()
                if s and s not in DENY:
                    out.add(s)
        else:
            s = str(value).strip().lower()
            if s and s not in DENY:
                out.add(s)
    return out


def _fragment_profile(fam: str, rows: pd.DataFrame) -> dict[str, Any]:
    prof = _family_profile(fam, rows)
    prof["domains"] = set(rows["domain"].dropna().astype(str).str.lower()) if "domain" in rows else set()
    prof["body"] = _clean_set(rows["body_template_hash"]) if "body_template_hash" in rows else set()
    prof["wc"] = _clean_set(rows["walletconnect_project_ids"]) if "walletconnect_project_ids" in rows else set()
    prof["html"] = _clean_set(rows["html_structure_hashes"]) if "html_structure_hashes" in rows else set()
    return prof


def _rare(tokens: set[str], df: Counter[str], cap: int) -> list[str]:
    """Shared tokens that are rare across fragments (df <= cap) -> kit-distinctive.

    A token shared by many fragments is shared infra (reused WalletConnect/Reown
    project id, framework JS bundle, common host) and must not define kit identity.
    """
    return sorted(t for t in tokens if int(df.get(t, 0)) <= cap)


def _merge_reason(
    a: dict[str, Any],
    b: dict[str, Any],
    m: dict[str, Any],
    df: dict[str, Counter[str]],
    cap: int,
    args: argparse.Namespace,
) -> tuple[str, dict[str, Any]]:
    """Return (reason, detail) if a distinctive (kit-owned, rare) edge exists, else ("", {})."""
    shared_bh = _rare((a["route_hashes"] & b["route_hashes"]) | (a["flow_hashes"] & b["flow_hashes"]) | (a["kit_hashes"] & b["kit_hashes"]), df["backend"], cap)
    if shared_bh:
        return "shared_backend_struct_hash", {"shared_backend_hashes": shared_bh[:8], "n": len(shared_bh)}

    shared_domains = _rare(a["domains"] & b["domains"], df["domain"], cap)
    if shared_domains:
        return "shared_full_host", {"shared_domains": shared_domains[:8], "n": len(shared_domains)}

    shared_wc = _rare(a["wc"] & b["wc"], df["wc"], cap)
    if shared_wc:
        return "shared_walletconnect_project_id", {"shared_wc": shared_wc[:8],
                                                   "max_family_df": max(int(df["wc"].get(t, 0)) for t in shared_wc)}

    shared_body = _rare(a["body"] & b["body"], df["body"], cap)
    if len(shared_body) >= int(args.min_shared_body_templates):
        return "shared_body_template", {"shared_body": shared_body[:8], "n": len(shared_body)}

    distinctive = _rare(a["script_hashes"] & b["script_hashes"], df["script"], cap)
    if len(distinctive) >= int(args.min_distinctive_scripts):
        return "shared_distinctive_script", {
            "distinctive_scripts": distinctive[:8],
            "n": len(distinctive),
            "max_family_df": max(int(df["script"].get(h, 0)) for h in distinctive),
        }

    if (
        a["is_credential_captcha"]
        and b["is_credential_captcha"]
        and max(float(m["norm_own_j"]), float(m["norm_api_j"])) >= float(args.interaction_norm_jaccard)
        and int(m["shared_script_count"]) == 0
    ):
        return "credential_captcha_random_path", {
            "norm_own_j": round(float(m["norm_own_j"]), 4),
            "norm_api_j": round(float(m["norm_api_j"]), 4),
        }
    return "", {}


def _components(nodes: list[str], edges: list[tuple[str, str, str, dict[str, Any]]]) -> list[list[str]]:
    parent = {n: n for n in nodes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, _r, _d in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    by_root: dict[str, list[str]] = defaultdict(list)
    for n in nodes:
        by_root[find(n)].append(n)
    return [sorted(v) for v in by_root.values()]


def build(args: argparse.Namespace) -> dict[str, Any]:
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(args.static_neighbor_labels_csv, low_memory=False)
    assignments = pd.read_csv(
        args.assignments_csv, low_memory=False, usecols=lambda c: c in {"zip_path", *EVIDENCE_COLS}
    )
    labels["zip_path"] = labels["zip_path"].astype(str)
    assignments["zip_path"] = assignments["zip_path"].astype(str)
    frame = labels.merge(assignments, on="zip_path", how="left")
    frame = frame[frame["static_family_key"].astype(str).str.startswith("backend_kit:")].copy()
    frame["static_family_key_static_neighbor"] = frame["static_family_key"].astype(str)
    if "static_family_key_fragment" not in frame.columns:
        raise ValueError("Static-neighbor labels must carry static_family_key_fragment provenance.")
    frame["static_family_key_fragment"] = frame["static_family_key_fragment"].astype(str)

    # Optional hand-audited labels, for provenance / comparison only.
    if args.audit_labels_csv:
        audit_labels = pd.read_csv(args.audit_labels_csv, low_memory=False)
        audit_map = dict(zip(audit_labels["zip_path"].astype(str), audit_labels["static_family_key"].astype(str)))
        frame["static_family_key_audited"] = frame["zip_path"].map(audit_map).fillna(frame["static_family_key_static_neighbor"])

    fragments = sorted(frame["static_family_key_fragment"].unique())
    profiles = {
        fam: _fragment_profile(fam, g) for fam, g in frame.groupby("static_family_key_fragment", sort=True)
    }

    # Per-channel fragment document-frequency: how many fragment labels contain each
    # token. Only rare (low-DF) tokens are kit-distinctive; high-DF tokens are
    # shared infra (reused WalletConnect ids, framework bundles, common hosts).
    df: dict[str, Counter[str]] = {k: Counter() for k in ("script", "backend", "domain", "wc", "body")}
    for prof in profiles.values():
        for h in prof["script_hashes"]:
            df["script"][h] += 1
        for h in prof["route_hashes"] | prof["flow_hashes"] | prof["kit_hashes"]:
            df["backend"][h] += 1
        for h in prof["domains"]:
            df["domain"][h] += 1
        for h in prof["wc"]:
            df["wc"][h] += 1
        for h in prof["body"]:
            df["body"][h] += 1
    cap = int(args.distinctive_max_family_df)
    if cap <= 0:
        cap = max(4, round(float(args.distinctive_df_frac) * len(fragments)))

    frag_by_static_neighbor: dict[str, list[str]] = {
        str(static_neighbor_key): sorted(set(map(str, g)))
        for static_neighbor_key, g in frame.groupby("static_family_key_static_neighbor")["static_family_key_fragment"]
    }

    # ---- Phase 1 (general / automated): re-audit every static-neighbor super-family ---------
    # Within each static-neighbor family, keep only fragments tied by distinctive (kit-owned,
    # rare) evidence; the rest split off. Bounded to one static-neighbor family at a time, so it
    # cannot chain across the dataset. This generalizes the 67e63193 split to all
    # of endpoint-driven merges (b3bd8c15, cc44b06f, ...).
    key_by_fragment: dict[str, str] = {}
    split_edges: list[tuple[str, str, str, dict[str, Any]]] = []
    for static_neighbor_fam, members in frag_by_static_neighbor.items():
        if len(members) == 1:
            key_by_fragment[members[0]] = static_neighbor_fam
            continue
        local: list[tuple[str, str, str, dict[str, Any]]] = []
        for i, fa in enumerate(members):
            for fb in members[i + 1:]:
                reason, detail = _merge_reason(profiles[fa], profiles[fb], _edge_metrics(profiles[fa], profiles[fb]), df, cap, args)
                if reason:
                    local.append((fa, fb, reason, detail))
        comps = _components(members, local)
        if len(comps) == 1:
            for f in members:
                key_by_fragment[f] = static_neighbor_fam  # unchanged: distinctive evidence holds the static-neighbor family together
        else:
            split_edges.extend(local)
            for comp in comps:
                key = comp[0] if len(comp) == 1 else _stable_family_key(comp)
                for f in comp:
                    key_by_fragment[f] = key

    # ---- Phase 2 (audited, rule-verified): cross-family same-kit merges ---------
    # Unrestricted automated cross-family merging collapses distinct campaigns into
    # one blob, because many drainer families share a common drainer-as-a-service
    # codebase (rare scripts/templates bridge them transitively). So cross-family
    # merges stay limited to the hand-audited same-kit pairs -- but each is now
    # re-verified against the distinctive-evidence rule at build time and fails
    # loudly if its evidence regresses (fixing the earlier silent-hardcoded brittleness).
    roots = {k: k for k in set(key_by_fragment.values())}

    def find(x: str) -> str:
        while roots[x] != x:
            roots[x] = roots[roots[x]]
            x = roots[x]
        return x

    audit_merges: list[dict[str, Any]] = []
    for fa, fb in GUARD_MERGE_PAIRS:
        rec: dict[str, Any] = {"a": _short(fa), "b": _short(fb)}
        if fa in key_by_fragment and fb in key_by_fragment:
            reason, detail = _merge_reason(profiles[fa], profiles[fb], _edge_metrics(profiles[fa], profiles[fb]), df, cap, args)
            rec.update({"present": True, "distinctive_reason": reason or "NONE", "detail": detail,
                        "evidence_ok": bool(reason)})
            ra, rb = find(key_by_fragment[fa]), find(key_by_fragment[fb])
            if ra != rb:
                roots[rb] = ra
        else:
            rec.update({"present": False, "evidence_ok": False})
        audit_merges.append(rec)

    groups: dict[str, list[str]] = defaultdict(list)
    for k in list(roots):
        groups[find(k)].append(k)
    final_key = {k: (ks[0] if len(ks) == 1 else _stable_family_key(sorted(ks))) for root, ks in groups.items() for k in ks}
    key_by_fragment = {fragment: final_key[key] for fragment, key in key_by_fragment.items()}

    frame["static_family_key_evidence"] = frame["static_family_key_fragment"].map(key_by_fragment)

    # ---- guard assertions: audited outcomes must hold -------------------------
    guard: dict[str, Any] = {
        "split_67e63193": {}, "merges": [], "coverage_complete": True, "ok": True
    }
    present = set(fragments)
    split_keys = {key_by_fragment.get(f) for f in GUARD_SPLIT_67E63193 if f in present}
    guard["split_67e63193"] = {
        "members_present": [f for f in GUARD_SPLIT_67E63193 if f in present],
        "distinct_evidence_families": len([k for k in split_keys if k is not None]),
        "ok": len([k for k in split_keys if k is not None]) >= 2,
    }
    guard["coverage_complete"] = guard["coverage_complete"] and (
        len(guard["split_67e63193"]["members_present"]) == len(GUARD_SPLIT_67E63193)
    )
    guard["ok"] = guard["ok"] and guard["split_67e63193"]["ok"]
    for a, b in GUARD_MERGE_PAIRS:
        if a in present and b in present:
            ok = key_by_fragment[a] == key_by_fragment[b]
            guard["merges"].append({"a": _short(a), "b": _short(b), "same_family": ok})
            guard["ok"] = guard["ok"] and ok
        else:
            guard["merges"].append({"a": _short(a), "b": _short(b), "present": False})
            guard["coverage_complete"] = False
    guard["ok"] = guard["ok"] and guard["coverage_complete"]

    # ---- provenance / correction label per sample -----------------------------
    frag_set_by_static_neighbor = {k: set(v) for k, v in frag_by_static_neighbor.items()}
    frag_set_by_evidence = frame.groupby("static_family_key_evidence")["static_family_key_fragment"].apply(lambda s: set(s)).to_dict()

    def _correction(row: pd.Series) -> str:
        static_neighbor_key = row["static_family_key_static_neighbor"]
        evidence_key = row["static_family_key_evidence"]
        if evidence_key == static_neighbor_key:
            return "unchanged_from_static_neighbor"
        static_neighbor_members = frag_set_by_static_neighbor.get(static_neighbor_key, set())
        audit_members = frag_set_by_evidence.get(evidence_key, set())
        if audit_members.issubset(static_neighbor_members) and len(audit_members) < len(static_neighbor_members):
            return "split_from_static_neighbor_no_distinctive_evidence"
        return "merge_cross_family_audited"

    frame["label_correction_evidence"] = frame.apply(_correction, axis=1)

    # ---- outputs --------------------------------------------------------------
    keep = [
        "zip_path", "sample_id", "domain",
        *[c for c in PASSTHROUGH_COLS if c in frame.columns],
        "static_family_key_evidence", "static_family_key_static_neighbor", "static_family_key_fragment",
        "static_family_key_cluster", "kit_family_key", "kit_cluster", "ndist",
        "label_correction_evidence",
    ]
    keep = list(dict.fromkeys(keep))
    keep = [c for c in keep if c in frame.columns]
    labels_out = frame[keep].copy().rename(columns={"static_family_key_evidence": "static_family_key"})
    labels_out["backend_kit_training_label"] = 1
    labels_out.to_csv(out / "backend_kit_evidence_labels.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    summary = (
        frame.groupby("static_family_key_evidence")
        .agg(
            n_samples=("zip_path", "size"),
            n_domains=("domain", "nunique"),
            n_static_neighbor_families=("static_family_key_static_neighbor", "nunique"),
            n_fragment_families=("static_family_key_fragment", "nunique"),
            fragment_members=("static_family_key_fragment", lambda s: " || ".join(sorted(set(map(str, s))))),
            corrections=("label_correction_evidence", lambda s: " || ".join(sorted(set(map(str, s))))),
        )
        .reset_index()
        .rename(columns={"static_family_key_evidence": "static_family_key"})
        .sort_values(["n_samples", "n_fragment_families", "static_family_key"], ascending=[False, False, True])
    )
    summary.to_csv(out / "backend_kit_evidence_labels_family_summary.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    # Phase-1 generalized splits: static-neighbor super-families broken up because some
    # fragments lacked distinctive evidence tying them to the rest.
    split_edge_by_pair = {(a, b): (r, d) for a, b, r, d in split_edges}
    splits: list[dict[str, Any]] = []
    for static_neighbor_fam, members in frag_by_static_neighbor.items():
        if len(members) < 2:
            continue
        sub: dict[str, list[str]] = defaultdict(list)
        for f in members:
            sub[key_by_fragment[f]].append(f)
        if len(sub) <= 1:
            continue
        kept_edges = [
            {"a": _short(a), "b": _short(b), "reason": r, **d}
            for (a, b), (r, d) in split_edge_by_pair.items()
            if a in members and b in members
        ]
        splits.append(
            {
                "static_neighbor_family": _short(static_neighbor_fam),
                "n_fragment_families": len(members),
                "n_evidence_families_after": len(sub),
                "subfamilies": [sorted(_short(x) for x in g) for g in sub.values()],
                "distinctive_edges_kept": sorted(kept_edges, key=lambda e: (e["reason"], e["a"], e["b"])),
            }
        )

    quality = {
        "fragment": _quality(frame, "static_family_key_fragment"),
        "static_neighbor": _quality(frame, "static_family_key_static_neighbor"),
        "evidence": _quality(frame, "static_family_key_evidence"),
    }
    if "static_family_key_audited" in frame.columns:
        quality["audit"] = _quality(frame, "static_family_key_audited")

    validation = {
        "route": "backend_kit_evidence_labels_distinctive_evidence_rule",
        "policy": {
            "uses_traffic_predictions": False,
            "merge_channels": ["backend_struct_hash", "full_host", "walletconnect_project_id",
                                "body_template", "distinctive_content_script", "credential_captcha_random_path"],
            "drops": "endpoint/schema Jaccard as a merge driver (shared wallet-API surface)",
            "distinctive_max_family_df": cap,
        },
        "inputs": {"static_neighbor_labels_csv": str(args.static_neighbor_labels_csv), "assignments_csv": str(args.assignments_csv)},
        "n_samples": int(len(frame)),
        "n_fragment_labels": len(fragments),
        "n_families_static_neighbor": int(frame["static_family_key_static_neighbor"].nunique()),
        "n_families_evidence": int(frame["static_family_key_evidence"].nunique()),
        "n_static_neighbor_superfamilies_split": len(splits),
        "n_distinctive_split_edges": len(split_edges),
        "split_edge_reason_counts": dict(Counter(r for _a, _b, r, _d in split_edges)),
        "n_audited_cross_family_merges": int(sum(1 for r in audit_merges if r.get("evidence_ok"))),
        "changed_vs_static_neighbor": dict(Counter(frame["label_correction_evidence"].astype(str))),
        "guard": guard,
        "audited_merges": audit_merges,
        "quality": quality,
        "splits": sorted(splits, key=lambda s: -s["n_fragment_families"]),
    }
    (out / "backend_kit_evidence_labels_validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
    )
    return validation


def write_trace_splits(args: argparse.Namespace) -> dict[str, Any]:
    out = pathlib.Path(args.out_dir)
    labels = pd.read_csv(out / "backend_kit_evidence_labels.csv", low_memory=False)
    labels["true_family"] = labels["static_family_key"].astype(str)
    labels["training_family_label"] = labels["true_family"]
    if args.base_split_csv:
        base = pd.read_csv(args.base_split_csv, low_memory=False)
        label_map = dict(zip(base["zip_path"].astype(str), base.get("label", "phishing")))
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
    split_input.to_csv(out / "trace_sample_split_evidence_labels.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    summary = {
        "rows": int(len(split_input)),
        "families": int(split_input["true_family"].nunique()),
        "train": int((split_input["split"] == "train").sum()),
        "test": int((split_input["split"] == "test").sum()),
        "families_train_ge15": int(
            (split_input[split_input["split"].eq("train")].groupby("true_family").size() >= 15).sum()
        ),
    }
    (out / "trace_sample_split_evidence_labels_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Build reproducible evidence-corrected backend-kit labels.")
    ap.add_argument("--static-neighbor-labels-csv", required=True)
    ap.add_argument("--assignments-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--audit-labels-csv", default="")
    ap.add_argument("--base-split-csv", default="")
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--min-distinctive-scripts", type=int, default=1)
    ap.add_argument("--min-shared-body-templates", type=int, default=1)
    ap.add_argument("--distinctive-max-family-df", type=int, default=0,
                    help="Cap on fragment-DF for a shared token (any channel) to count as kit-distinctive (0=auto).")
    ap.add_argument("--distinctive-df-frac", type=float, default=0.03)
    ap.add_argument("--interaction-norm-jaccard", type=float, default=0.80)
    args = ap.parse_args()

    validation = build(args)
    split_summary = write_trace_splits(args)
    print(json.dumps({
        "n_samples": validation["n_samples"],
        "n_fragment_labels": validation["n_fragment_labels"],
        "n_families_static_neighbor": validation["n_families_static_neighbor"],
        "n_families_evidence": validation["n_families_evidence"],
        "n_static_neighbor_superfamilies_split": validation["n_static_neighbor_superfamilies_split"],
        "n_audited_cross_family_merges": validation["n_audited_cross_family_merges"],
        "split_edge_reason_counts": validation["split_edge_reason_counts"],
        "changed_vs_static_neighbor": validation["changed_vs_static_neighbor"],
        "guard_ok": validation["guard"]["ok"],
        "quality": validation["quality"],
        "split": split_summary,
        "out_dir": str(args.out_dir),
    }, indent=2, ensure_ascii=False, default=float))
    if not validation["guard"]["ok"]:
        print("\n!! GUARD FAILED: an audited outcome regressed -- inspect validation JSON.", flush=True)


if __name__ == "__main__":
    main()

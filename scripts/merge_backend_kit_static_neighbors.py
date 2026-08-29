#!/usr/bin/env python3
from __future__ import annotations

"""Build static-neighbor backend-kit labels from fragment labels plus static-neighbor rules.

Fragment labels represent kit identity: it merges clustered fragments only when distinctive JS
bundle evidence says two backend-kit families are the same kit. The 2026-06-28
domain-grouped Layer2 audit showed two residual problems:

1. Some non-Web3 interaction captures use random backend path salts such as
   ``/fwtsplckie/api`` and ``/lxkaetdhfu/api``. These salts define separate fragment
   families although the observable workflow is the same credential-captcha kit.
2. Some drainer families are statically close enough that encrypted traffic
   cannot reliably separate them. This stage exposes a traffic-observable family label by
   merging only strongly supported static neighbors.

This stage keeps fragment provenance in ``static_family_key_fragment`` and writes the
new label to ``static_family_key`` so existing Layer2 scripts can consume it.
It uses no domain/TLD/brand signal and does not use traffic predictions.
"""

import argparse
import csv
import hashlib
import json
import math
import pathlib
import re
import sys
from collections import Counter, defaultdict
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from cluster_backend_kit_labels import _json_list, own_evidence_tokens  # noqa: E402

PASSTHROUGH_COLS = [
    "capture_id", "sample_key", "label", "source", "source_variant",
    "source_folder", "fine_source", "partition", "fit_role",
]


RANDOM_API_SEGMENT_RE = re.compile(r"/[a-z0-9]{7,}(?=/api(?:/|\?| |$))", re.I)
GLOBAL_OBJECT_SENTINEL_RE = re.compile(r"/\+\(__(?:globalobject|uint8array)[^)]+\)\+", re.I)
INDEX_PAGE_RE = re.compile(r"/index_[0-9]+(?=\.html|$)", re.I)


def _short(key: str) -> str:
    text = str(key or "")
    return text.split(":", 1)[-1][:8] if ":" in text else text[:8]


def _stable_family_key(members: Iterable[str]) -> str:
    seed = "\n".join(sorted(str(x) for x in members))
    return "backend_kit:" + hashlib.sha256(seed.encode("utf-8", "ignore")).hexdigest()[:18]


def _as_set(value: object) -> set[str]:
    return {str(x).lower() for x in _json_list(value) if str(x).strip()}


def _normalise_random_backend_token(token: str) -> str:
    text = str(token or "").lower()
    text = RANDOM_API_SEGMENT_RE.sub("/{rand}", text)
    text = GLOBAL_OBJECT_SENTINEL_RE.sub("/{js_sentinel}", text)
    text = INDEX_PAGE_RE.sub("/index_{n}", text)
    return text


def _interaction_pattern(row: pd.Series) -> str:
    vals = _json_list(row.get("interaction_pattern", "[]"))
    if vals:
        return str(vals[0])
    return str(row.get("interaction_pattern", "") or "")


def _family_majority(values: list[str]) -> tuple[str, float]:
    vals = [str(v) for v in values if str(v)]
    if not vals:
        return "", 0.0
    c = Counter(vals)
    val, n = c.most_common(1)[0]
    return val, float(n / len(vals))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return float(len(a & b) / max(1, len(a | b)))


def _set_coverage(rows: pd.DataFrame, col: str) -> dict[str, float]:
    c: Counter[str] = Counter()
    for value in rows[col] if col in rows.columns else []:
        c.update(_as_set(value))
    n = max(1, len(rows))
    return {k: float(v / n) for k, v in c.items()}


def _family_profile(fam: str, rows: pd.DataFrame) -> dict[str, Any]:
    raw_own: set[str] = set()
    norm_own: set[str] = set()
    raw_ep: set[str] = set()
    raw_sc: set[str] = set()
    norm_ep: set[str] = set()
    norm_sc: set[str] = set()
    route_hashes: set[str] = set()
    flow_hashes: set[str] = set()
    kit_hashes: set[str] = set()
    script_hashes: set[str] = set()
    inline_hashes: set[str] = set()
    capabilities: set[str] = set()
    patterns: list[str] = []

    for _, row in rows.iterrows():
        own = {str(t).lower() for t in own_evidence_tokens(row)}
        raw_own |= own
        normed = {_normalise_random_backend_token(t) for t in own}
        norm_own |= normed
        raw_ep |= {t for t in own if t.startswith("ep:")}
        raw_sc |= {t for t in own if t.startswith("sc:")}
        norm_ep |= {t for t in normed if t.startswith("ep:")}
        norm_sc |= {t for t in normed if t.startswith("sc:")}
        route_hashes |= _as_set(row.get("backend_route_set_hashes", "[]"))
        flow_hashes |= _as_set(row.get("backend_flow_hashes", "[]"))
        kit_hashes |= _as_set(row.get("backend_kit_hashes", "[]"))
        script_hashes |= _as_set(row.get("script_content_hashes", "[]"))
        inline_hashes |= _as_set(row.get("inline_script_hashes", "[]"))
        capabilities |= _as_set(row.get("js_capabilities", "[]"))
        pat = _interaction_pattern(row)
        if pat:
            patterns.append(pat)

    majority_pattern, pattern_frac = _family_majority(patterns)
    return {
        "family": fam,
        "n_samples": int(len(rows)),
        "n_domains": int(rows["domain"].astype(str).nunique()) if "domain" in rows else 0,
        "raw_own": raw_own,
        "norm_own": norm_own,
        "raw_ep": raw_ep,
        "raw_sc": raw_sc,
        "norm_ep": norm_ep,
        "norm_sc": norm_sc,
        "route_hashes": route_hashes,
        "flow_hashes": flow_hashes,
        "kit_hashes": kit_hashes,
        "script_hashes": script_hashes,
        "inline_hashes": inline_hashes,
        "script_cov": _set_coverage(rows, "script_content_hashes"),
        "inline_cov": _set_coverage(rows, "inline_script_hashes"),
        "capabilities": capabilities,
        "majority_pattern": majority_pattern,
        "majority_pattern_frac": pattern_frac,
        "is_credential_captcha": majority_pattern == "credential_captcha_cloaked" and pattern_frac >= 0.75,
    }


def _edge_metrics(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    shared_script = (a["script_hashes"] | a["inline_hashes"]) & (b["script_hashes"] | b["inline_hashes"])
    shared_backend_hash = (a["route_hashes"] & b["route_hashes"]) | (a["flow_hashes"] & b["flow_hashes"]) | (a["kit_hashes"] & b["kit_hashes"])
    a_norm_api = {t for t in a["norm_own"] if "/{rand}/api" in t}
    b_norm_api = {t for t in b["norm_own"] if "/{rand}/api" in t}
    min_script_cov = 0.0
    for tok in shared_script:
        ca = max(a["script_cov"].get(tok, 0.0), a["inline_cov"].get(tok, 0.0))
        cb = max(b["script_cov"].get(tok, 0.0), b["inline_cov"].get(tok, 0.0))
        min_script_cov = max(min_script_cov, min(ca, cb))
    return {
        "raw_own_j": _jaccard(a["raw_own"], b["raw_own"]),
        "norm_own_j": _jaccard(a["norm_own"], b["norm_own"]),
        "raw_ep_j": _jaccard(a["raw_ep"], b["raw_ep"]),
        "raw_sc_j": _jaccard(a["raw_sc"], b["raw_sc"]),
        "norm_ep_j": _jaccard(a["norm_ep"], b["norm_ep"]),
        "norm_sc_j": _jaccard(a["norm_sc"], b["norm_sc"]),
        "norm_api_j": _jaccard(a_norm_api, b_norm_api),
        "shared_script_count": len(shared_script),
        "shared_backend_hash_count": len(shared_backend_hash),
        "shared_script_min_coverage": min_script_cov,
        "same_majority_pattern": bool(a["majority_pattern"] and a["majority_pattern"] == b["majority_pattern"]),
        "majority_pattern": a["majority_pattern"] if a["majority_pattern"] == b["majority_pattern"] else "",
    }


def _merge_reason(a: dict[str, Any], b: dict[str, Any], m: dict[str, Any], args: argparse.Namespace) -> str:
    if (
        a["is_credential_captcha"]
        and b["is_credential_captcha"]
        and max(float(m["norm_own_j"]), float(m["norm_api_j"])) >= float(args.interaction_norm_jaccard)
        and m["shared_script_count"] == 0
    ):
        return "interaction_random_path"
    static_j = max(float(m["raw_ep_j"]), float(m["raw_sc_j"]), float(m["norm_ep_j"]), float(m["norm_sc_j"]))
    if m["shared_backend_hash_count"] > 0 and static_j >= float(args.backend_hash_min_jaccard):
        return "shared_backend_hash_static_neighbor"
    if m["shared_script_count"] > 0:
        if float(m["raw_ep_j"]) >= float(args.static_neighbor_endpoint_jaccard):
            return "shared_script_high_endpoint_overlap"
        if float(m["raw_sc_j"]) >= float(args.static_neighbor_schema_jaccard):
            return "shared_script_high_schema_overlap"
        if (
            m["shared_script_count"] >= int(args.static_neighbor_min_shared_scripts)
            and static_j >= float(args.static_neighbor_script_jaccard)
            and float(m["shared_script_min_coverage"]) >= float(args.static_neighbor_min_script_coverage)
        ):
            return "shared_script_static_neighbor"
    return ""


def _components(nodes: list[str], edges: list[tuple[str, str, str, dict[str, Any]]]) -> list[list[str]]:
    parent = {n: n for n in nodes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b, _reason, _metrics in edges:
        union(a, b)
    by_root: dict[str, list[str]] = defaultdict(list)
    for n in nodes:
        by_root[find(n)].append(n)
    return [sorted(v) for v in by_root.values()]


def _quality(frame: pd.DataFrame, fam_col: str) -> dict[str, float]:
    profiles = {
        fam: _family_profile(str(fam), g)
        for fam, g in frame.groupby(fam_col, sort=True)
        if str(fam)
    }
    keys = [k for k, p in profiles.items() if p["n_samples"] >= 4]
    script_shares = []
    for fam in keys:
        sub = frame[frame[fam_col].astype(str).eq(fam)]
        c: Counter[str] = Counter()
        for col in ("script_content_hashes", "inline_script_hashes"):
            if col in sub.columns:
                for value in sub[col]:
                    c.update(_as_set(value))
        script_shares.append(max(c.values()) / len(sub) if c else 0.0)
    hi_raw = hi_norm = tot = 0
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            pa, pb = profiles[a], profiles[b]
            if not pa["raw_own"] or not pb["raw_own"]:
                continue
            tot += 1
            if _jaccard(pa["raw_own"], pb["raw_own"]) >= 0.5:
                hi_raw += 1
            if _jaccard(pa["norm_own"], pb["norm_own"]) >= 0.5:
                hi_norm += 1
    return {
        "families_ge4": float(len(keys)),
        "within_family_scripthash_share": float(np.mean(script_shares)) if script_shares else 0.0,
        "cross_family_raw_own_overlap_ge0.5_frac": float(hi_raw / tot) if tot else 0.0,
        "cross_family_norm_own_overlap_ge0.5_frac": float(hi_norm / tot) if tot else 0.0,
    }


def build_static_neighbor_labels(args: argparse.Namespace) -> dict[str, Any]:
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    fragment_labels = pd.read_csv(args.fragment_labels_csv, low_memory=False)
    ev = pd.read_csv(args.assignments_csv, low_memory=False)
    fragment_labels["zip_path"] = fragment_labels["zip_path"].astype(str)
    ev["zip_path"] = ev["zip_path"].astype(str)
    ev_cols = [
        "zip_path", "label", "backend_endpoints", "backend_request_schemas",
        "backend_route_set_hashes", "backend_flow_hashes", "backend_kit_hashes",
        "script_content_hashes", "inline_script_hashes", "addresses",
        "drainer_spenders", "js_capabilities", "interaction_pattern",
    ]
    merged = fragment_labels.merge(ev[[c for c in ev_cols if c in ev.columns]], on="zip_path", how="left", suffixes=("", "_ev"))
    merged = merged[merged["static_family_key"].astype(str).str.startswith("backend_kit:")].copy()
    merged["static_family_key_fragment"] = merged["static_family_key"].astype(str)

    profiles = {
        fam: _family_profile(str(fam), g)
        for fam, g in merged.groupby("static_family_key_fragment", sort=True)
    }
    families = sorted(profiles)
    edges: list[tuple[str, str, str, dict[str, Any]]] = []
    for i, fa in enumerate(families):
        for fb in families[i + 1:]:
            m = _edge_metrics(profiles[fa], profiles[fb])
            reason = _merge_reason(profiles[fa], profiles[fb], m, args)
            if reason:
                edges.append((fa, fb, reason, m))

    comps = _components(families, edges)
    static_neighbor_key_by_fragment: dict[str, str] = {}
    comp_edges: dict[tuple[str, ...], list[tuple[str, str, str, dict[str, Any]]]] = defaultdict(list)
    comp_lookup = {fam: tuple(comp) for comp in comps for fam in comp}
    for edge in edges:
        comp_edges[comp_lookup[edge[0]]].append(edge)
    merge_records: list[dict[str, Any]] = []
    for comp in comps:
        if len(comp) == 1:
            static_neighbor_key = comp[0]
        else:
            static_neighbor_key = _stable_family_key(comp)
        for fam in comp:
            static_neighbor_key_by_fragment[fam] = static_neighbor_key
        if len(comp) > 1:
            component_rows = merged[merged["static_family_key_fragment"].isin(comp)]
            reasons = Counter(edge[2] for edge in comp_edges[tuple(comp)])
            merge_records.append(
                {
                    "static_neighbor_family": _short(static_neighbor_key),
                    "static_neighbor_family_full": static_neighbor_key,
                    "members": [_short(x) for x in comp],
                    "members_full": comp,
                    "n_samples": int(len(component_rows)),
                    "n_domains": int(component_rows["domain"].astype(str).nunique()),
                    "n_fragment_families": int(len(comp)),
                    "reasons": dict(reasons),
                    "edges": [
                        {
                            "a": _short(a),
                            "b": _short(b),
                            "a_full": a,
                            "b_full": b,
                            "reason": reason,
                            **{
                                k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else int(v) if isinstance(v, (int, np.integer)) else v)
                                for k, v in metrics.items()
                            },
                        }
                        for a, b, reason, metrics in sorted(comp_edges[tuple(comp)], key=lambda e: (e[2], e[0], e[1]))
                    ],
                }
            )

    merged["static_family_key"] = merged["static_family_key_fragment"].map(static_neighbor_key_by_fragment)
    if "static_family_key_cluster" not in merged.columns:
        merged["static_family_key_cluster"] = ""

    keep = [
        "zip_path", "sample_id", "domain",
        *[c for c in PASSTHROUGH_COLS if c in merged.columns],
        "static_family_key", "static_family_key_fragment",
        "static_family_key_cluster", "kit_family_key", "kit_cluster", "ndist",
    ]
    keep = list(dict.fromkeys(keep))
    keep = [c for c in keep if c in merged.columns]
    labels = merged[keep].copy()
    labels["backend_kit_training_label"] = 1
    labels.to_csv(out / "backend_kit_static_neighbor_labels.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    fam_summary = (
        merged.groupby("static_family_key")
        .agg(
            n_samples=("zip_path", "size"),
            n_domains=("domain", "nunique"),
            n_fragment_families=("static_family_key_fragment", "nunique"),
            fragment_members=("static_family_key_fragment", lambda s: " || ".join(sorted(set(map(str, s))))),
        )
        .reset_index()
        .sort_values(["n_samples", "n_fragment_families", "static_family_key"], ascending=[False, False, True])
    )
    fam_summary.to_csv(out / "backend_kit_static_neighbor_labels_family_summary.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    quality_fragment = _quality(merged, "static_family_key_fragment")
    quality_static_neighbor = _quality(merged, "static_family_key")
    validation = {
        "route": "backend_kit_static_neighbor_labels_static_neighbor_superfamily",
        "inputs": {
            "fragment_labels_csv": str(args.fragment_labels_csv),
            "assignments_csv": str(args.assignments_csv),
        },
        "params": vars(args),
        "n_samples": int(len(labels)),
        "n_families_fragment": int(labels["static_family_key_fragment"].nunique()),
        "n_families_static_neighbor": int(labels["static_family_key"].nunique()),
        "n_merge_components": int(sum(1 for comp in comps if len(comp) > 1)),
        "n_edges": int(len(edges)),
        "quality": {
            "fragment": quality_fragment,
            "static_neighbor": quality_static_neighbor,
            "raw_cross_family_overlap_delta": round(quality_static_neighbor["cross_family_raw_own_overlap_ge0.5_frac"] - quality_fragment["cross_family_raw_own_overlap_ge0.5_frac"], 6),
            "norm_cross_family_overlap_delta": round(quality_static_neighbor["cross_family_norm_own_overlap_ge0.5_frac"] - quality_fragment["cross_family_norm_own_overlap_ge0.5_frac"], 6),
        },
        "merges": sorted(merge_records, key=lambda r: (-r["n_samples"], r["static_neighbor_family_full"])),
    }
    (out / "backend_kit_static_neighbor_labels_merge_validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
    )
    return validation


def write_trace_splits(args: argparse.Namespace) -> dict[str, Any]:
    out = pathlib.Path(args.out_dir)
    labels = pd.read_csv(out / "backend_kit_static_neighbor_labels.csv", low_memory=False)
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
        n_test = int(math.ceil(test_size * n_rows)) if 0.0 < test_size < 1.0 else int(test_size)
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
    split_input.to_csv(out / "trace_sample_split_static_neighbors.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    split_input[split_input["split"].eq("train")].to_csv(out / "split_train_pred_static_neighbors.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    split_input[split_input["split"].eq("test")].to_csv(out / "split_test_pred_static_neighbors.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    summary = {
        "rows": int(len(split_input)),
        "families": int(split_input["true_family"].nunique()),
        "train": int((split_input["split"] == "train").sum()),
        "test": int((split_input["split"] == "test").sum()),
        "families_train_ge15": int((split_input[split_input["split"].eq("train")].groupby("true_family").size() >= 15).sum()),
        "trace_sample_split_static_neighbors": str(out / "trace_sample_split_static_neighbors.csv"),
    }
    (out / "trace_sample_split_static_neighbors_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Build traffic-observable backend-kit labels from fragment labels.")
    ap.add_argument("--assignments-csv", required=True)
    ap.add_argument("--fragment-labels-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--base-split-csv", default="")
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--interaction-norm-jaccard", type=float, default=0.80)
    ap.add_argument("--static-neighbor-endpoint-jaccard", type=float, default=0.70)
    ap.add_argument("--static-neighbor-schema-jaccard", type=float, default=0.70)
    ap.add_argument("--static-neighbor-script-jaccard", type=float, default=0.35)
    ap.add_argument("--static-neighbor-min-shared-scripts", type=int, default=2)
    ap.add_argument("--static-neighbor-min-script-coverage", type=float, default=0.05)
    ap.add_argument("--backend-hash-min-jaccard", type=float, default=0.35)
    args = ap.parse_args()

    validation = build_static_neighbor_labels(args)
    split_summary = write_trace_splits(args)
    print(json.dumps({
        "n_samples": validation["n_samples"],
        "n_families_fragment": validation["n_families_fragment"],
        "n_families_static_neighbor": validation["n_families_static_neighbor"],
        "n_merge_components": validation["n_merge_components"],
        "n_edges": validation["n_edges"],
        "quality": validation["quality"],
        "split": split_summary,
        "out_dir": str(args.out_dir),
    }, indent=2, ensure_ascii=False, default=float))
    print("\nmerges:")
    for merge in validation["merges"]:
        print(f"  {merge['static_neighbor_family']} <= {merge['members']} ({merge['n_samples']} samples) {merge['reasons']}")


if __name__ == "__main__":
    main()

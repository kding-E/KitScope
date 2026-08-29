#!/usr/bin/env python3
from __future__ import annotations

"""Build final backend-kit labels with a non-transitive same-kit merge.

The evidence-correction stage deliberately does not auto-merge across static-neighbor families, because a single-link
"strong shared token" rule chains many drainer families into one blob. The error
analysis of the evidence-correction downstream run showed the residual misclassifications fall
into two groups:

* genuine model / traffic confusions between families that share little static
  evidence (not fixable by relabeling), and
* a handful of genuinely same-kit families kept split (e.g. the bep20* operator,
  the 135461223.site cluster), which the model "confuses" because they are one kit.

The final label builder fixes only the second group, with a non-transitive method that cannot chain:
average-linkage agglomerative clustering over the evidence-corrected families, with distance
= 1 - max-over-channel Jaccard of *distinctive* (low family-DF) tokens. Average
linkage merges a family into a cluster only if it is similar to the cluster on
average, so the sparse host-bridge chains that produced the blob do not form.

Per-channel Jaccard (not absolute shared counts) is used on purpose: two large
families can share a dozen exact hosts yet still be different operators; Jaccard
normalises by family size. Inspecting the data, genuine same-kit pairs sit at
Jaccard >= ~0.25 while the blob's chaining edges are < 0.1.

Validated downstream (domain-grouped, train_n>=15, same bg_strict features, only
labels swapped): The final labels raise supported-family macro-F1 to 0.7943 at 37 families,
above the evidence-correction baseline (0.7635 / 38). Force-merging the whole blob into one family instead
*lowers* macro-F1 to 0.7794 (25 families), confirming the blob is not one kit.

No Layer2 traffic predictions are used. Intermediate family keys are kept as provenance.
"""

import argparse
import csv
import hashlib
import json
import pathlib
import sys
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from merge_backend_kit_static_neighbors import _quality, _short  # noqa: E402
from rebuild_backend_kit_evidence_corrections import EVIDENCE_COLS, _clean_set  # noqa: E402

CHANNELS = {
    "script": "script_content_hashes",
    "host": "domain",
    "body": "body_template_hash",
    "wc": "walletconnect_project_ids",
    "route": "backend_route_set_hashes",
    "flow": "backend_flow_hashes",
    "kit": "backend_kit_hashes",
}
PASSTHROUGH_COLS = [
    "capture_id", "sample_key", "label", "source", "source_variant",
    "source_folder", "fine_source", "partition", "fit_role",
]

# audited same-kit outcomes that must survive (verified, not hardcoded merges)
GUARD_SAME_FAMILY = [
    ("backend_kit:93cbcd481fc75afdad", "backend_kit:c4b6ed71978c461d46"),
    ("backend_kit:c21318f9170b265fa7", "backend_kit:41e8a9af282e69af14"),
    ("backend_kit:67a8fa8f91d0187aba", "backend_kit:62a11ff014262b0ad6"),
]


def _sample_tokens(row: dict[str, Any]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for ch, col in CHANNELS.items():
        if ch == "host":
            d = row.get("domain")
            out[ch] = {str(d).lower()} if (d is not None and str(d).strip() and str(d).lower() != "nan") else set()
        else:
            out[ch] = _clean_set([row.get(col, "")])
    return out


def build(args: argparse.Namespace) -> dict[str, Any]:
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    lab = pd.read_csv(args.evidence_labels_csv, low_memory=False)
    asg = pd.read_csv(args.assignments_csv, low_memory=False, usecols=lambda c: c in {"zip_path", *EVIDENCE_COLS})
    lab["zip_path"] = lab["zip_path"].astype(str)
    asg["zip_path"] = asg["zip_path"].astype(str)
    f = lab.merge(asg, on="zip_path", how="left")
    f = f[f["static_family_key"].astype(str).str.startswith("backend_kit:")].copy()
    f["fam"] = f["static_family_key"].astype(str)  # evidence-corrected key

    # per-family distinctive token sets (token rare across families)
    fam_counts = {ch: defaultdict(Counter) for ch in CHANNELS}
    for row in f.to_dict("records"):
        toks = _sample_tokens(row)
        for ch, ts in toks.items():
            for t in ts:
                fam_counts[ch][row["fam"]][t] += 1
    fam_df = {ch: Counter() for ch in CHANNELS}
    for ch in CHANNELS:
        for fam, c in fam_counts[ch].items():
            for t in c:
                fam_df[ch][t] += 1
    cap = int(args.distinctive_max_family_df)
    fams = sorted(f["fam"].unique())
    if cap <= 0:
        cap = max(4, round(float(args.distinctive_df_frac) * len(fams)))
    dset = {ch: {fam: {t for t in fam_counts[ch][fam] if fam_df[ch][t] <= cap} for fam in fams} for ch in CHANNELS}

    def maxjac(a: str, b: str) -> tuple[float, dict[str, float]]:
        best = 0.0
        per: dict[str, float] = {}
        for ch in CHANNELS:
            A, Bb = dset[ch][a], dset[ch][b]
            if A and Bb:
                u = len(A | Bb)
                j = len(A & Bb) / u if u else 0.0
                if j > 0:
                    per[ch] = round(j, 4)
                best = max(best, j)
        return best, per

    n = len(fams)
    dist = np.ones((n, n))
    np.fill_diagonal(dist, 0.0)
    per_pair: dict[tuple[int, int], dict[str, float]] = {}
    for i in range(n):
        for j in range(i + 1, n):
            mj, per = maxjac(fams[i], fams[j])
            dist[i, j] = dist[j, i] = 1.0 - mj
            if mj >= float(args.sim_threshold):
                per_pair[(i, j)] = {"max_jaccard": round(mj, 4), **{f"jac_{k}": v for k, v in per.items()}}

    cl = AgglomerativeClustering(
        n_clusters=None, distance_threshold=1.0 - float(args.sim_threshold),
        metric="precomputed", linkage="average",
    )
    clab = cl.fit_predict(dist)
    groups: dict[int, list[str]] = defaultdict(list)
    for fam, c in zip(fams, clab):
        groups[int(c)].append(fam)

    def key_for(members: list[str]) -> str:
        if len(members) == 1:
            return members[0]
        return "backend_kit:" + hashlib.sha256("\n".join(sorted(members)).encode()).hexdigest()[:18]

    key_by_evidence = {m: key_for(members) for members in groups.values() for m in members}
    f["static_family_key_final"] = f["fam"].map(key_by_evidence)

    # ---- guard: audited same-kit pairs intact + no blob -----------------------
    final_by_fragment = dict(zip(f["static_family_key_fragment"].astype(str), f["static_family_key_final"].astype(str)))
    guard = {
        "same_family_pairs": [], "max_family_evidence_fragments": 0,
        "coverage_complete": True, "ok": True,
    }
    for a, b in GUARD_SAME_FAMILY:
        if a in final_by_fragment and b in final_by_fragment:
            ok = final_by_fragment[a] == final_by_fragment[b]
            guard["same_family_pairs"].append({"a": _short(a), "b": _short(b), "same": ok})
            guard["ok"] = guard["ok"] and ok
        else:
            guard["same_family_pairs"].append({"a": _short(a), "b": _short(b), "present": False})
            guard["coverage_complete"] = False
    biggest = max((len(set(m)) for m in [
        f[f["static_family_key_final"].eq(k)]["fam"].unique() for k in f["static_family_key_final"].unique()
    ]), default=0)
    guard["max_family_evidence_fragments"] = int(biggest)
    guard["ok"] = guard["ok"] and biggest <= int(args.max_merge_families)
    guard["ok"] = guard["ok"] and guard["coverage_complete"]

    # ---- merges report --------------------------------------------------------
    merges = []
    fidx = {fam: i for i, fam in enumerate(fams)}
    for members in groups.values():
        if len(members) < 2:
            continue
        rows = f[f["fam"].isin(members)]
        edges = []
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                ia, ib = sorted((fidx[a], fidx[b]))
                if (ia, ib) in per_pair:
                    edges.append({"a": _short(a), "b": _short(b), **per_pair[(ia, ib)]})
        merges.append({
            "final_family": _short(key_by_evidence[members[0]]),
            "n_samples": int(len(rows)),
            "n_domains": int(rows["domain"].nunique()),
            "members": [_short(x) for x in members],
            "edges": sorted(edges, key=lambda e: -e["max_jaccard"]),
        })

    # ---- outputs --------------------------------------------------------------
    def correction(row: pd.Series) -> str:
        return "unchanged_from_evidence_labels" if row["static_family_key_final"] == row["fam"] else "merge_same_kit_community"

    f["label_correction_final"] = f.apply(correction, axis=1)
    keep = [
        "zip_path", "sample_id", "domain",
        *[c for c in PASSTHROUGH_COLS if c in f.columns],
        "static_family_key_final", "static_family_key",
        "static_family_key_fragment", "static_family_key_cluster", "kit_family_key", "kit_cluster",
        "ndist", "label_correction_final",
    ]
    keep = list(dict.fromkeys(keep))
    keep = [c for c in keep if c in f.columns]
    labels_out = f[keep].copy().rename(columns={"static_family_key_final": "static_family_key",
                                                 "static_family_key": "static_family_key_evidence"})
    labels_out["backend_kit_training_label"] = 1
    labels_out.to_csv(out / "backend_kit_final_labels.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    (f.groupby("static_family_key_final").agg(
        n_samples=("zip_path", "size"), n_domains=("domain", "nunique"),
        n_evidence_families=("fam", "nunique"),
        members=("fam", lambda s: " || ".join(sorted(set(map(str, s))))),
    ).reset_index().rename(columns={"static_family_key_final": "static_family_key"})
     .sort_values(["n_samples", "static_family_key"], ascending=[False, True])
     ).to_csv(out / "backend_kit_final_labels_family_summary.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    validation = {
        "route": "backend_kit_final_labels_average_linkage_same_kit_merge",
        "policy": {"uses_traffic_predictions": False, "similarity": "max-over-channel Jaccard of distinctive tokens",
                   "linkage": "average (non-transitive)", "sim_threshold": float(args.sim_threshold),
                   "distinctive_max_family_df": cap},
        "inputs": {"evidence_labels_csv": str(args.evidence_labels_csv), "assignments_csv": str(args.assignments_csv)},
        "n_samples": int(len(f)),
        "n_families_evidence": int(f["fam"].nunique()),
        "n_families_final": int(f["static_family_key_final"].nunique()),
        "n_merges": len(merges),
        "n_changed_samples": int((f["label_correction_final"] != "unchanged_from_evidence_labels").sum()),
        "guard": guard,
        "quality": {"evidence": _quality(f, "fam"), "final": _quality(f, "static_family_key_final")},
        "merges": sorted(merges, key=lambda r: -r["n_samples"]),
    }
    (out / "backend_kit_final_labels_validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    # trace split (runner re-does the grouped split; stratified file for record)
    split_cols = [
        "zip_path", "sample_id", "domain",
        *[c for c in ("capture_id", "sample_key", "source", "source_variant", "source_folder", "fine_source") if c in f.columns],
    ]
    d = f[list(dict.fromkeys(split_cols))].copy()
    d["true_family"] = f["static_family_key_final"].astype(str)
    d["training_family_label"] = d["true_family"]
    if args.base_split_csv:
        base = pd.read_csv(args.base_split_csv, low_memory=False)
        lmap = dict(zip(base["zip_path"].astype(str), base.get("label", "phishing")))
        d["label"] = d["zip_path"].astype(str).map(lmap).fillna("phishing")
    else:
        d["label"] = "phishing"
    n_rows = len(d)
    test_size = float(args.test_size)
    if n_rows < 2:
        te = np.array([], dtype=int)
    else:
        n_test = int(np.ceil(test_size * n_rows)) if 0.0 < test_size < 1.0 else int(test_size)
        n_test = max(1, min(n_rows - 1, n_test))
        family_counts = d["true_family"].astype(str).value_counts()
        n_families = len(family_counts)
        stratify = (
            d["true_family"].astype(str)
            if n_families > 1 and family_counts.min() >= 2 and n_test >= n_families and (n_rows - n_test) >= n_families
            else None
        )
        _tr, te = train_test_split(
            np.arange(n_rows),
            test_size=test_size,
            random_state=int(args.random_state),
            stratify=stratify,
        )
    d["split"] = "train"
    d.iloc[te, d.columns.get_loc("split")] = "test"
    final_cols = [
        "zip_path", "sample_id", "domain",
        *[c for c in ("capture_id", "sample_key", "source", "source_variant", "source_folder", "fine_source") if c in d.columns],
        "label", "true_family", "training_family_label", "split",
    ]
    d = d[list(dict.fromkeys(final_cols))]
    d.sort_values(["split", "true_family", "zip_path"]).to_csv(out / "trace_sample_split_kit_labels.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    return validation


def main() -> None:
    ap = argparse.ArgumentParser(description="Build final same-kit-merged backend-kit labels (non-transitive).")
    ap.add_argument("--evidence-labels-csv", required=True)
    ap.add_argument("--assignments-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--base-split-csv", default="")
    ap.add_argument("--sim-threshold", type=float, default=0.25)
    ap.add_argument("--distinctive-max-family-df", type=int, default=6)
    ap.add_argument("--distinctive-df-frac", type=float, default=0.03)
    ap.add_argument("--max-merge-families", type=int, default=6,
                    help="Guard: a final family must not absorb more than this many evidence-corrected families (blob tripwire).")
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()
    v = build(args)
    print(json.dumps({
        "n_families_evidence": v["n_families_evidence"], "n_families_final": v["n_families_final"],
        "n_merges": v["n_merges"], "n_changed_samples": v["n_changed_samples"],
        "guard_ok": v["guard"]["ok"], "max_family_evidence_fragments": v["guard"]["max_family_evidence_fragments"],
        "quality": v["quality"], "out_dir": str(args.out_dir),
    }, indent=2, ensure_ascii=False, default=float))
    for mg in v["merges"]:
        print(f"  {mg['final_family']} <= {mg['members']} ({mg['n_samples']} samples)")
    if not v["guard"]["ok"]:
        print("\n!! GUARD FAILED: blob tripwire or audited pair regressed.", flush=True)


if __name__ == "__main__":
    main()

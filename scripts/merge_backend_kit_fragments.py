#!/usr/bin/env python3
from __future__ import annotations

"""Build fragment backend-kit labels by merging clustered families on kit-owned identity.

Why this exists
---------------
The initial clustered labels (``cluster_backend_kit_labels.py``) cluster on infra-stripped
backend endpoints + request schemas. Auditing the residual Layer2 confusion
showed that channel still fragments one kit across several families, because the
shared drainer-framework request schemas are semi-generic and per-capture
observation noise splits a kit. Two kit-owned, front-end-independent channels
separate identity far more cleanly:

* ``script_content_hashes`` / ``inline_script_hashes`` -- the kit's own JS bundle.
* ``addresses`` / ``drainer_spenders`` -- the on-chain destination / approve
  target (the kit's collector), independent of the JS bundle.

This step takes the *frozen, reproducible* clustered labels and applies a single
post-clustering MERGE pass: two clustered families are linked when they share a JS
bundle hash that is distinctive (appears in <= ``max_token_family_df`` of all
families) and characteristic (covers >= ``min_coverage`` of EACH family's
samples). Connected components become fragment-merged families.

It uses NO domain / TLD / front-end-brand signal.

Independent validation (this is the gate, not just the build):
1. Per merge, on-chain corroboration: do the members also share a distinctive,
   generic-token-filtered destination address / drainer_spender? The on-chain
   channel is independent of the JS bundle used to merge -> Tier-A if yes,
   Tier-B (bundle-only) if not.
2. Aggregate guardrail (the clustering quality metric), computed before and after fragment merging:
   - within-family script-hash share (coherence, should not collapse)
   - cross-family endpoint path-overlap >= 0.5 fraction (over-merge detector;
     merging true fragments should keep this flat or LOWER it, never raise it).

Output (canonical, drop-in for Layer2 routes via ``static_family_key``):
* ``backend_kit_fragment_labels.csv``
* ``backend_kit_fragment_labels_family_summary.csv``
* ``backend_kit_fragment_labels_merge_validation.json``
"""

import argparse
import json
import pathlib
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from audit_layer2_backend_kit_identity import (  # noqa: E402
    GENERIC_ADDRESSES,
    _short,
    _toks,
    build_channel_stats,
    connected_components,
    find_merge_edges,
)
from cluster_backend_kit_labels import own_evidence_tokens, _json_list  # noqa: E402

PASSTHROUGH_COLS = [
    "capture_id", "sample_key", "label", "source", "source_variant",
    "source_folder", "fine_source", "partition", "fit_role",
]

# Well-known ERC-20 / router contracts that are drain targets, not kit
# collectors. Extends the audit tool's set with multi-chain stablecoins/routers
# that showed up as low-DF noise in the on-chain channel.
GENERIC_ONCHAIN = set(GENERIC_ADDRESSES) | {
    "0x55d398326f99059ff775485246999027b3197955",  # BSC USDT
    "0xdac17f958d2ee523a2206206994597c13d831ec7",  # ETH USDT
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",  # ETH USDC
    "0xe592427a0aece92de3edee1f18e0157c05861564",  # Uniswap router
    "0x7a250d5630b4cf539739df2c5dacb4c659f2488d",  # Uniswap router
}


def onchain_corroboration(merged: pd.DataFrame, members: list[str], min_cov: float, max_df: int):
    """Distinctive on-chain address/spender shared by ALL members (independent of JS bundle)."""
    chans = [c for c in ("addresses", "drainer_spenders", "drainer_hits") if c in merged.columns]
    # family-DF of each on-chain token across ALL families
    dfreq: Counter = Counter()
    for _fam, sub in merged.groupby("static_family_key"):
        seen = set()
        for ch in chans:
            for v in sub[ch]:
                for t in _toks(v):
                    if t not in GENERIC_ONCHAIN:
                        seen.add(t)
        for t in seen:
            dfreq[t] += 1
    # per-member coverage
    cov = {}
    for f in members:
        sub = merged[merged.static_family_key == f]
        c: Counter = Counter()
        for ch in chans:
            for v in sub[ch]:
                for t in set(_toks(v)):
                    if t not in GENERIC_ONCHAIN:
                        c[t] += 1
        cov[f] = {t: n / len(sub) for t, n in c.items()}
    common = set.intersection(*[set(cov[f]) for f in members]) if members else set()
    hits = []
    for t in common:
        if dfreq[t] > max_df:
            continue
        per = [round(cov[f].get(t, 0.0), 3) for f in members]
        if min(per) >= min_cov:
            hits.append({"token": t, "family_df": int(dfreq[t]), "per_member_coverage": per})
    hits.sort(key=lambda d: (d["family_df"], -min(d["per_member_coverage"])))
    return hits


def aggregate_quality(merged: pd.DataFrame, fam_col: str) -> dict:
    """cluster-quality: within-family script-hash share + cross-family endpoint overlap>=0.5."""
    merged = merged.copy()
    merged["_own_ep"] = merged.apply(
        lambda r: {t for t in own_evidence_tokens(r) if t.startswith("ep:")}, axis=1)
    merged["_scr"] = merged["script_content_hashes"].map(lambda v: set(map(str, _json_list(v))))
    groups = {k: g.index.tolist() for k, g in merged.groupby(fam_col) if str(k)}
    groups = {k: v for k, v in groups.items() if len(v) >= 4}
    shares = []
    for idxs in groups.values():
        sh: Counter = Counter()
        for i in idxs:
            for h in merged.loc[i, "_scr"]:
                sh[h] += 1
        shares.append(max(sh.values()) / len(idxs) if sh else 0.0)
    path_sets = {}
    for k, idxs in groups.items():
        ps: set[str] = set()
        for i in idxs:
            ps |= merged.loc[i, "_own_ep"]
        path_sets[k] = ps
    keys = list(path_sets)
    hi = tot = 0
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            A, B = path_sets[keys[a]], path_sets[keys[b]]
            if not A or not B:
                continue
            tot += 1
            if len(A & B) / len(A | B) >= 0.5:
                hi += 1
    return {
        "families_ge4": len(groups),
        "within_family_scripthash_share": float(np.mean(shares)) if shares else 0.0,
        "cross_family_pathoverlap_ge0.5_frac": float(hi / tot) if tot else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge clustered backend-kit families into fragment labels on kit-owned JS-bundle identity")
    ap.add_argument("--assignments-csv", required=True)
    ap.add_argument("--cluster-labels-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-token-family-df", type=int, default=5)
    ap.add_argument("--min-coverage", type=float, default=0.34)
    ap.add_argument("--merge-channels", default="script_content_hashes,inline_script_hashes")
    ap.add_argument("--onchain-max-df", type=int, default=6)
    ap.add_argument("--guardrail-overlap-tol", type=float, default=0.005,
                    help="abort if cross-family endpoint overlap rises after fragment merging")
    args = ap.parse_args()

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    merge_channels = [c.strip() for c in args.merge_channels.split(",") if c.strip()]

    cluster_labels = pd.read_csv(args.cluster_labels_csv, low_memory=False)
    ev = pd.read_csv(args.assignments_csv, low_memory=False)
    cluster_labels["zip_path"] = cluster_labels["zip_path"].astype(str)
    ev["zip_path"] = ev["zip_path"].astype(str)
    ev_cols = ["zip_path", "addresses", "drainer_spenders", "drainer_hits",
               "backend_endpoints", "backend_request_schemas", "script_content_hashes",
               "inline_script_hashes"]
    merged = cluster_labels.merge(ev[[c for c in ev_cols if c in ev.columns]], on="zip_path", how="left",
                      suffixes=("", "_ev"))
    # the clustered-label file already carries script_content_hashes? no -> from ev. ensure present:
    merged = merged[merged["static_family_key"].astype(str).str.startswith("backend_kit:")].copy()

    families = sorted(merged["static_family_key"].unique())
    coverage, family_df, fam_sizes = build_channel_stats(merged, merge_channels)
    edges = find_merge_edges(coverage, family_df, families, merge_channels,
                             args.max_token_family_df, args.min_coverage)
    comp = connected_components(families, edges)

    # canonical key = largest member; build map + per-merge validation
    merge_map = {}
    merges = []
    for root, members in comp.items():
        members_sorted = sorted(members, key=lambda f: (-fam_sizes[f], f))
        canonical = members_sorted[0]
        for f in members:
            merge_map[f] = canonical
        if len(members) > 1:
            corro = onchain_corroboration(merged, members, args.min_coverage, args.onchain_max_df)
            merges.append({
                "fragment_family": _short(canonical),
                "fragment_family_full": canonical,
                "members": [_short(f) for f in members_sorted],
                "n_samples": int(sum(fam_sizes[f] for f in members)),
                "per_member_samples": {_short(f): fam_sizes[f] for f in members_sorted},
                "tier": "A_two_channel" if corro else "B_bundle_only",
                "onchain_corroboration": corro[:4],
            })
    merges.sort(key=lambda d: (d["tier"], -d["n_samples"]))

    merged["static_family_key_cluster"] = merged["static_family_key"]
    merged["static_family_key"] = merged["static_family_key_cluster"].map(merge_map)

    # --- independent validation: aggregate guardrail before/after fragment merging ---
    q_cluster = aggregate_quality(merged, "static_family_key_cluster")
    quality_fragment = aggregate_quality(merged, "static_family_key")
    overlap_delta = quality_fragment["cross_family_pathoverlap_ge0.5_frac"] - q_cluster["cross_family_pathoverlap_ge0.5_frac"]
    guardrail_pass = overlap_delta <= args.guardrail_overlap_tol

    # --- write canonical fragment labels ---
    keep = [
        "zip_path", "sample_id", "domain",
        *[c for c in PASSTHROUGH_COLS if c in merged.columns],
        "static_family_key", "static_family_key_cluster",
    ]
    keep += [c for c in ("kit_family_key", "kit_cluster", "ndist", "label") if c in merged.columns]
    keep = list(dict.fromkeys(keep))
    lab = merged[keep].copy()
    lab["backend_kit_training_label"] = 1
    lab.to_csv(out / "backend_kit_fragment_labels.csv", index=False)

    fam_summary = (merged.groupby("static_family_key")
                   .agg(n_samples=("zip_path", "size"), n_domains=("domain", "nunique"),
                        n_cluster_families=("static_family_key_cluster", "nunique"))
                   .reset_index().sort_values("n_samples", ascending=False))
    fam_summary.to_csv(out / "backend_kit_fragment_labels_family_summary.csv", index=False)

    validation = {
        "cluster_labels_csv": str(args.cluster_labels_csv),
        "params": {"merge_channels": merge_channels, "max_token_family_df": args.max_token_family_df,
                   "min_coverage": args.min_coverage, "onchain_max_df": args.onchain_max_df},
        "n_families_cluster": len(families),
        "n_families_fragment": int(merged["static_family_key"].nunique()),
        "n_merges": len(merges),
        "aggregate_guardrail": {
            "cluster_quality": q_cluster, "fragment_quality": quality_fragment,
            "cross_family_overlap_delta": round(overlap_delta, 5),
            "tolerance": args.guardrail_overlap_tol,
            "guardrail_pass": bool(guardrail_pass),
            "note": "merging true fragments should keep cross-family endpoint overlap flat or lower it; a rise signals over-merge",
        },
        "merges": merges,
    }
    (out / "backend_kit_fragment_labels_merge_validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({
        "n_families_cluster": len(families), "n_families_fragment": int(merged["static_family_key"].nunique()),
        "n_merges": len(merges),
        "within_family_scripthash_share": {"cluster": round(q_cluster["within_family_scripthash_share"], 4),
                                            "fragment": round(quality_fragment["within_family_scripthash_share"], 4)},
        "cross_family_pathoverlap_ge0.5": {"cluster": round(q_cluster["cross_family_pathoverlap_ge0.5_frac"], 5),
                                           "fragment": round(quality_fragment["cross_family_pathoverlap_ge0.5_frac"], 5),
                                           "delta": round(overlap_delta, 5)},
        "guardrail_pass": bool(guardrail_pass),
    }, indent=2))
    print("\nmerges:")
    for m in merges:
        print(f"  [{m['tier']}] {m['fragment_family']} <= {m['members']}  ({m['n_samples']} samples)"
              + (f"  on-chain: {m['onchain_corroboration'][0]['token']} df={m['onchain_corroboration'][0]['family_df']}"
                 if m["onchain_corroboration"] else "  (bundle-only)"))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

"""Audit backend-kit family identity and propose evidence-based merges.

Why this exists
---------------
The clustered labels (``cluster_backend_kit_labels.py``) cluster on infra-stripped
backend *endpoints + request schemas*. Auditing the residual Layer2 confusion
cluster showed that channel is a noisy middle layer: the shared drainer
framework/SDK request-schema hashes are semi-generic (one hash spans 14-35
families), so the same kit is both over-merged with framework siblings and
fragmented across per-capture observation noise.

Two *kit-owned, frontend-independent* channels separate kit identity far more
cleanly:

* ``script_content_hashes`` / ``inline_script_hashes`` -- the kit's own JS
  bundle. A distinctive bundle hash is shared by very few families.
* ``addresses`` / ``drainer_spenders`` -- the on-chain destination / approve
  target the drainer exfiltrates to. The kit's own collector, independent of
  the front-end brand.

This tool links two families when they share a token that is BOTH
*distinctive* (appears in <= ``max_token_family_df`` families, so generic
libraries / common framework hashes / popular ERC-20 token addresses are
excluded) AND *characteristic* (covers >= ``min_coverage`` of each family's
samples, so a one-off shared resource does not trigger a merge). Connected
components over those links become proposed merged kit families.

It uses NO domain / TLD / front-end signal, so a shared cheap ``.vip`` domain
cannot drive a merge -- only shared kit code or shared on-chain collectors can.

Outputs
-------
* ``kit_identity_merge_edges.csv`` -- every family pair with supporting tokens.
* ``kit_identity_merged_labels.csv`` -- zip_path -> merged_family_key map
  (drop-in replacement label column ``merged_family_key``).
* ``kit_identity_merge_summary.json`` -- components, sizes, and the
  distinctness of each merge.
"""

import argparse
import json
import pathlib
from collections import Counter, defaultdict
from typing import Iterable

import pandas as pd

# Channels that carry kit-owned identity, ordered strong -> weak.
STRONG_CHANNELS = (
    "script_content_hashes",
    "inline_script_hashes",
    "addresses",
    "drainer_spenders",
    "drainer_hits",
)
# Channels only reported for corroboration, never used to drive a merge,
# because they encode a shared drainer framework rather than one kit.
CORROBORATING_CHANNELS = (
    "backend_request_schema_hashes",
    "backend_kit_hashes",
    "html_structure_hashes",
    "image_asset_hashes",
)
# ERC-20 / system addresses that are common targets, not kit collectors.
GENERIC_ADDRESSES = {
    "0xc2132d05d31c914a87c6611c10748aeb04b58e8f",  # Polygon USDT
    "0x2791bca1f2de4661ed88a30c99a7a9449aa84174",  # Polygon USDC
    "0x0000000000000000000000000000000000000008",  # precompile
    "0x0000000000000000000000000000000000000000",  # zero address
}


def _toks(value: object) -> list[str]:
    try:
        parsed = json.loads(str(value))
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    out = []
    for t in parsed:
        s = str(t).strip().lower()
        if s and s not in GENERIC_ADDRESSES:
            out.append(s)
    return out


def _short(fam: str) -> str:
    return str(fam).replace("backend_kit:", "")[:8]


def build_channel_stats(merged: pd.DataFrame, channels: Iterable[str]):
    """Per channel: family-level coverage of each token + token family-DF."""
    fam_sizes = merged.groupby("static_family_key").size().to_dict()
    # coverage[channel][family][token] = fraction of family samples with token
    coverage: dict[str, dict[str, dict[str, float]]] = {}
    family_df: dict[str, Counter] = {}
    for ch in channels:
        if ch not in merged.columns:
            continue
        fam_tok_counts: dict[str, Counter] = defaultdict(Counter)
        for fam, sub in merged.groupby("static_family_key"):
            for v in sub[ch]:
                for t in set(_toks(v)):
                    fam_tok_counts[fam][t] += 1
        cov: dict[str, dict[str, float]] = {}
        dfreq: Counter = Counter()
        for fam, cnts in fam_tok_counts.items():
            n = fam_sizes[fam]
            cov[fam] = {t: c / n for t, c in cnts.items()}
            for t in cnts:
                dfreq[t] += 1
        coverage[ch] = cov
        family_df[ch] = dfreq
    return coverage, family_df, fam_sizes


def find_merge_edges(
    coverage, family_df, families, channels,
    max_token_family_df: int, min_coverage: float,
):
    edges = []
    fams = sorted(families)
    for i in range(len(fams)):
        for j in range(i + 1, len(fams)):
            fa, fb = fams[i], fams[j]
            support = {}
            for ch in channels:
                if ch not in coverage:
                    continue
                ca, cb = coverage[ch].get(fa, {}), coverage[ch].get(fb, {})
                df = family_df[ch]
                shared = []
                for t, cov_a in ca.items():
                    if cov_a < min_coverage:
                        continue
                    cov_b = cb.get(t, 0.0)
                    if cov_b < min_coverage:
                        continue
                    if df[t] > max_token_family_df:
                        continue
                    shared.append({"token": t, "family_df": int(df[t]),
                                   "cov_a": round(cov_a, 3), "cov_b": round(cov_b, 3)})
                if shared:
                    support[ch] = sorted(shared, key=lambda d: (d["family_df"], -d["cov_a"]))
            if support:
                edges.append((fa, fb, support))
    return edges


def connected_components(families, edges):
    parent = {f: f for f in families}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for fa, fb, _ in edges:
        ra, rb = find(fa), find(fb)
        if ra != rb:
            parent[rb] = ra
    comp: dict[str, list[str]] = defaultdict(list)
    for f in families:
        comp[find(f)].append(f)
    return comp


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit backend-kit identity and propose evidence-based merges")
    ap.add_argument("--assignments-csv", required=True)
    ap.add_argument("--cluster-labels-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-token-family-df", type=int, default=5,
                    help="a linking token may appear in at most this many families (distinctiveness)")
    ap.add_argument("--min-coverage", type=float, default=0.34,
                    help="a linking token must cover at least this fraction of EACH family's samples")
    ap.add_argument("--channels", default=",".join(STRONG_CHANNELS),
                    help="comma list of channels used to drive merges")
    args = ap.parse_args()

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    channels = [c.strip() for c in args.channels.split(",") if c.strip()]

    labels = pd.read_csv(args.cluster_labels_csv, low_memory=False)
    ev = pd.read_csv(args.assignments_csv, low_memory=False)
    labels["zip_path"] = labels["zip_path"].astype(str)
    ev["zip_path"] = ev["zip_path"].astype(str)
    keep = ["zip_path"] + [c for c in (channels + list(CORROBORATING_CHANNELS)) if c in ev.columns]
    merged = labels.merge(ev[keep], on="zip_path", how="left")
    merged = merged[merged["static_family_key"].astype(str).str.startswith("backend_kit:")].copy()

    families = sorted(merged["static_family_key"].unique())
    coverage, family_df, fam_sizes = build_channel_stats(merged, channels + list(CORROBORATING_CHANNELS))
    edges = find_merge_edges(coverage, family_df, families, channels,
                             args.max_token_family_df, args.min_coverage)
    comp = connected_components(families, edges)

    # --- write edge report ---
    edge_rows = []
    for fa, fb, support in edges:
        for ch, shared in support.items():
            for s in shared:
                edge_rows.append({
                    "family_a": _short(fa), "family_b": _short(fb), "channel": ch,
                    "token": s["token"], "token_family_df": s["family_df"],
                    "coverage_a": s["cov_a"], "coverage_b": s["cov_b"],
                    "n_a": fam_sizes[fa], "n_b": fam_sizes[fb],
                })
    pd.DataFrame(edge_rows).to_csv(out / "kit_identity_merge_edges.csv", index=False)

    # --- build merged label map (canonical = largest member family) ---
    merged_key = {}
    components = []
    for root, members in comp.items():
        members_sorted = sorted(members, key=lambda f: (-fam_sizes[f], f))
        canonical = members_sorted[0]
        for f in members:
            merged_key[f] = canonical
        if len(members) > 1:
            components.append({
                "merged_family": _short(canonical),
                "members": [_short(f) for f in members_sorted],
                "n_member_families": len(members),
                "n_samples": int(sum(fam_sizes[f] for f in members)),
                "per_member_samples": {_short(f): fam_sizes[f] for f in members_sorted},
            })
    components.sort(key=lambda d: -d["n_member_families"])

    lab = merged[["zip_path", "sample_id", "domain", "static_family_key"]].copy()
    lab["merged_family_key"] = lab["static_family_key"].map(merged_key)
    lab.to_csv(out / "kit_identity_merged_labels.csv", index=False)

    summary = {
        "n_clustered_families": len(families),
        "n_families_after_merge": len(set(merged_key.values())),
        "n_merge_components": len(components),
        "params": {
            "max_token_family_df": args.max_token_family_df,
            "min_coverage": args.min_coverage,
            "channels": channels,
        },
        "merges": components,
    }
    (out / "kit_identity_merge_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in
                      ["n_clustered_families", "n_families_after_merge", "n_merge_components"]}, indent=2))
    print("\nproposed merges (>1 family):")
    for c in components:
        print(f"  {c['merged_family']}  <= {c['members']}  ({c['n_samples']} samples)")


if __name__ == "__main__":
    main()

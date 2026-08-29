#!/usr/bin/env python3
from __future__ import annotations

"""Rebuild backend-kit ground-truth labels from per-sample static/HAR evidence.

Why this exists
---------------
The strict exact-hash labels key each kit family on a
hash of the sample's whole backend material (endpoints + schemas + roles + flow).
Two problems make those labels wrong as kit identity:

1. **Third-party wallet infrastructure leaks into the key.** The most common
   "backend endpoints" are not the phishing kit's own server: they are
   WalletConnect / Web3Modal / Reown, Coinbase / Phantom / Rainbow / Rabby / OKX /
   Bitget wallet APIs, on-ramp and analytics endpoints, and Cloudflare RUM. Every
   phishing site that connects a given wallet emits these, so they both (a) pull
   unrelated kits together and (b), because each capture observes a slightly
   different subset, split one kit across many hashes.

2. **Exact set hashing is brittle.** A single missing/extra observed endpoint
   changes the hash, fragmenting one kit into several families.

Audit on the exact-hash labels: 12% of family pairs share >=0.5 Jaccard
on their own (infra-stripped) endpoint paths -- i.e. they are very likely the
same kit split apart.

What this does
--------------
- Strips third-party wallet/analytics/CDN infrastructure from the backend
  evidence, keeping only kit-owned endpoints + request schemas.
- Builds an IDF-weighted evidence vector per sample (ubiquitous leftovers get ~0
  weight) and forms families by average-linkage agglomerative clustering at a
  cosine-distance threshold. This is robust to observation noise: a kit survives a
  missing endpoint.
- Validates families on an *independent* evidence channel never used to build
  them -- shared front-end script-content hashes -- and reports cross-family
  endpoint overlap. The clustering step cuts strong cross-family overlap from 12% to ~1.4% while
  keeping within-family script-hash sharing ~0.75.

Output schema is compatible with the Layer2 routes: `static_family_key` uses the
`backend_kit:` prefix, joined to the operation-unit / packet tables on `zip_path`.

Limitation: there is no external kit-name ground truth, so this is a principled
re-derivation of kit identity from the collected evidence, weak-validated on an
independent channel -- not a labelled gold set.
"""

import argparse
import hashlib
import json
import math
import pathlib
import re
from collections import Counter, defaultdict
from typing import Sequence

import numpy as np
import pandas as pd

# Third-party wallet / analytics / CDN infrastructure. Endpoints matching these
# are emitted by the wallet stack or shared SaaS, not by the phishing kit's own
# backend, so they must not define kit identity.
INFRA_PATTERN = re.compile(
    r"(walletconnect\.org|web3modal\.org|reown|rainbow\.me|coinbase\.com|"
    r"phantom\.app|rabby\.io|onramper\.com|debank\.com|okx\.|okex\.org|okx\.cab|okx\.ac|"
    r"bitkeep|bitget|bitapi|chainnear|lymryy|ljhsnzpc|trustwallet|metamask\.io|"
    r"infura\.io|alchemy|quicknode|ankr\.com|cloudflare|1\.1\.1\.1|gstatic|googleapis|"
    r"google-analytics|googletagmanager|sentry|amplitude|mixpanel|segment\.io|"
    r"cdn-cgi|/appkit/|getwallets|getwalletimage|getassetimage|getanalyticsconfig|"
    r"/metrics$|/amp$|/rum$)",
    re.I,
)

PASSTHROUGH_COLS = [
    "capture_id", "sample_key", "label", "source", "source_variant",
    "source_folder", "fine_source", "partition", "fit_role",
]


def _json_list(value: object) -> list:
    try:
        parsed = json.loads(str(value))
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def is_infra_endpoint(endpoint: str) -> bool:
    e = str(endpoint)
    # The WalletConnect/Reown AppKit calls all carry a projectId query parameter.
    if "projectid" in e.lower():
        return True
    return bool(INFRA_PATTERN.search(e))


def strip_host(endpoint: str) -> str:
    e = str(endpoint)
    if e and not e.startswith("/") and "/" in e:
        return "/" + e.split("/", 1)[1]
    return e


def own_evidence_tokens(row: pd.Series, schema_clip: int = 120) -> set[str]:
    """Kit-owned backend evidence: infra-stripped endpoint paths + request schemas."""
    tokens: set[str] = set()
    for ep in _json_list(row.get("backend_endpoints", "[]")):
        if is_infra_endpoint(ep):
            continue
        tokens.add("ep:" + strip_host(ep))
    for sc in _json_list(row.get("backend_request_schemas", "[]")):
        sc = str(sc)
        if is_infra_endpoint(sc):
            continue
        tokens.add("sc:" + sc[:schema_clip])
    return tokens


def _family_key(members_tokens: Sequence[set[str]], dfreq: Counter, n_docs: int) -> str:
    """Stable kit key: hash the high-IDF tokens common to most of the cluster."""
    counts: Counter = Counter()
    for toks in members_tokens:
        for t in toks:
            counts[t] += 1
    n = len(members_tokens)
    core = sorted(
        t for t, c in counts.items()
        if c >= max(2, math.ceil(0.5 * n)) and dfreq[t] <= 0.30 * n_docs
    )
    if not core:
        core = sorted(counts)[:20]
    digest = hashlib.sha256("\n".join(core).encode("utf-8", "ignore")).hexdigest()[:18]
    return f"backend_kit:{digest}"


def build_labels(
    assignments_csv: pathlib.Path,
    out_dir: pathlib.Path,
    distance_threshold: float = 0.55,
    min_distinctive: int = 2,
    distinctive_df_frac: float = 0.06,
    drop_df_frac: float = 0.30,
    min_family_samples: int = 5,
    label_join_csv: pathlib.Path | None = None,
) -> dict:
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    out_dir.mkdir(parents=True, exist_ok=True)
    usecols = [
        "zip_path", "sample_id", "domain", "static_family_key",
        "backend_endpoints", "backend_request_schemas", "backend_role_sequence",
        "script_content_hashes",
        *PASSTHROUGH_COLS,
    ]
    df = pd.read_csv(assignments_csv, low_memory=False)
    df = df[[c for c in usecols if c in df.columns]].copy()
    df["zip_path"] = df["zip_path"].astype(str)
    n_docs = len(df)

    df["own"] = df.apply(own_evidence_tokens, axis=1)
    df["scr"] = df["script_content_hashes"].map(lambda v: set(map(str, _json_list(v))))

    dfreq: Counter = Counter()
    for toks in df["own"]:
        for t in set(toks):
            dfreq[t] += 1
    distinctive_cut = distinctive_df_frac * n_docs
    df["ndist"] = df["own"].map(lambda t: sum(1 for x in t if dfreq[x] <= distinctive_cut))

    label_map: dict[str, str] = {}
    if label_join_csv and label_join_csv.exists():
        lj = pd.read_csv(label_join_csv, low_memory=False)
        idc = "zip_path" if "zip_path" in lj.columns else ("sample_id" if "sample_id" in lj.columns else None)
        if idc and "label" in lj.columns:
            label_map = dict(zip(lj[idc].astype(str), lj["label"].astype(str)))

    elig = df[df["ndist"] >= int(min_distinctive)].reset_index(drop=True)
    if len(elig) < 2:
        raise RuntimeError("too few samples with distinctive kit-owned backend evidence")

    vocab = sorted(set().union(*elig["own"]))
    vi = {t: i for i, t in enumerate(vocab)}
    drop_cut = drop_df_frac * n_docs
    M = np.zeros((len(elig), len(vocab)), dtype=np.float32)
    for i, toks in enumerate(elig["own"]):
        for t in toks:
            if dfreq[t] >= drop_cut:
                continue
            M[i, vi[t]] = math.log((n_docs + 1) / (dfreq[t] + 1)) + 1.0
    norm = np.linalg.norm(M, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    M = M / norm
    D = np.clip(1.0 - M @ M.T, 0.0, 2.0)
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method="average")
    cluster = fcluster(Z, t=float(distance_threshold), criterion="distance")

    members: dict[int, list[int]] = defaultdict(list)
    for i, c in enumerate(cluster):
        members[int(c)].append(i)
    keyed: dict[int, str] = {}
    for c, idxs in members.items():
        keyed[c] = _family_key([elig.iloc[i]["own"] for i in idxs], dfreq, n_docs)
    elig["kit_cluster"] = cluster
    elig["kit_family_key"] = [keyed[int(c)] for c in cluster]

    fam_counts = elig["kit_family_key"].value_counts()
    eligible_families = set(fam_counts[fam_counts >= int(min_family_samples)].index)
    elig["static_family_key"] = elig["kit_family_key"].where(elig["kit_family_key"].isin(eligible_families), "")

    # --- weak-supervision validation on the independent script-hash channel ---
    def quality(group_col: str, frame: pd.DataFrame) -> dict:
        groups = {k: g.index.tolist() for k, g in frame.groupby(group_col) if str(k)}
        groups = {k: v for k, v in groups.items() if len(v) >= 4}
        shares = []
        for _, idxs in groups.items():
            sh: Counter = Counter()
            for i in idxs:
                for h in frame.loc[i, "scr"]:
                    sh[h] += 1
            shares.append(max(sh.values()) / len(idxs) if sh else 0.0)
        path_sets = {}
        for k, idxs in groups.items():
            ps: set[str] = set()
            for i in idxs:
                ps |= {t for t in frame.loc[i, "own"] if t.startswith("ep:")}
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

    strict = elig[elig["static_family_key"].ne("")].reset_index(drop=True)
    cluster_quality = quality("static_family_key", strict)

    out_cols = [
        "zip_path", "sample_id", "domain",
        *[c for c in PASSTHROUGH_COLS if c in strict.columns],
        "static_family_key", "kit_family_key", "kit_cluster", "ndist",
    ]
    strict_out = strict[out_cols].copy()
    if label_map:
        strict_out["label"] = strict_out["zip_path"].map(lambda z: label_map.get(str(z), ""))
    strict_out["backend_kit_training_label"] = 1
    labels_path = out_dir / "backend_kit_label_clusters.csv"
    strict_out.to_csv(labels_path, index=False)

    # Full assignment (including dropped/ineligible) for transparency.
    elig[out_cols].to_csv(out_dir / "backend_kit_label_clusters_all_eligible.csv", index=False)

    fam_summary = (
        strict.groupby("static_family_key")
        .agg(n_samples=("zip_path", "size"), n_domains=("domain", "nunique"))
        .reset_index()
        .sort_values("n_samples", ascending=False)
    )
    fam_summary.to_csv(out_dir / "backend_kit_label_clusters_family_summary.csv", index=False)

    summary = {
        "assignments_csv": str(assignments_csv),
        "n_docs": int(n_docs),
        "n_eligible": int(len(elig)),
        "n_labelled_strict": int(len(strict)),
        "n_families_strict": int(strict["static_family_key"].nunique()),
        "distance_threshold": float(distance_threshold),
        "min_distinctive": int(min_distinctive),
        "min_family_samples": int(min_family_samples),
        "cluster_quality": cluster_quality,
        "label_distribution": (
            Counter(strict_out["label"]).most_common() if "label" in strict_out.columns else "unknown"
        ),
    }
    (out_dir / "backend_kit_label_clusters_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Rebuild backend-kit ground-truth labels with infra-stripped IDF evidence clustering")
    ap.add_argument("--assignments-csv", required=True, help="backend_static_family_label_assignments.csv with per-sample evidence columns")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--distance-threshold", type=float, default=0.55)
    ap.add_argument("--min-distinctive", type=int, default=2)
    ap.add_argument("--distinctive-df-frac", type=float, default=0.06)
    ap.add_argument("--drop-df-frac", type=float, default=0.30)
    ap.add_argument("--min-family-samples", type=int, default=5)
    ap.add_argument("--label-join-csv", default=None, help="optional CSV with zip_path/sample_id + label for phishing/benign reporting")
    args = ap.parse_args()
    summary = build_labels(
        assignments_csv=pathlib.Path(args.assignments_csv),
        out_dir=pathlib.Path(args.out_dir),
        distance_threshold=float(args.distance_threshold),
        min_distinctive=int(args.min_distinctive),
        distinctive_df_frac=float(args.distinctive_df_frac),
        drop_df_frac=float(args.drop_df_frac),
        min_family_samples=int(args.min_family_samples),
        label_join_csv=pathlib.Path(args.label_join_csv) if args.label_join_csv else None,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()

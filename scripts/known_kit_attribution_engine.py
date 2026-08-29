#!/usr/bin/env python3
"""Internal engine for Layer 2 known kit/drainer attribution.

Use ``scripts/run_layer2_known_kit_attribution.py`` for the canonical public
entry point. This file keeps the original implementation name so older outputs
and imports remain reproducible.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
import re
import sys
from collections import Counter
from itertools import combinations
from dataclasses import asdict, dataclass
from typing import Sequence

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    adjusted_rand_score,
    average_precision_score,
    normalized_mutual_info_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import RobustScaler

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from audit_cluster_family_evidence import family_features, feature_type
from run_optimized_known_phish_cluster_experiment import candidate_pool, transform_frame
from web3pcapdetector.models.utils import numeric_feature_columns, select_window


PHISH_LABELS = {"phishing", "phish", "malicious", "1", "true"}
EVIDENCE_TIER_RANK = {"weak": 1, "moderate": 2, "strong": 3}
SCRIPT_DISTINCTIVE_RE = re.compile(
    r"(drainer|wallet[-_]?connect|lucifer|onboard|fpbundle|modals|secureproxy|mining|permit|approve)",
    re.I,
)
GENERIC_DRAINER_SYMBOLS = {"drainer"}
SUPPORT_MIN_COVERAGE = 0.50

# Backend-kit identity is stricter than backend interaction schema.  A request
# schema such as "wallet_address + auth_address" is a phishing workflow class,
# not necessarily the same server-side kit.  It is therefore support evidence by
# default; Layer2 kit-family pseudo-labels should be generated only from route,
# flow, and backend-kit material that includes the server interface topology.
BACKEND_KIT_PRIMARY_TYPES = {
    "backend_kit",
    "backend_flow_hash",
    "backend_route_set_hash",
}
BACKEND_PRIMARY_TYPES = BACKEND_KIT_PRIMARY_TYPES | {
    "backend_request_schema_hash",
    "backend_callsite_hash",
    "backend_endpoint",
}
STRONG_PRIMARY_TYPES = {
    "backend_kit",
}
MODERATE_PRIMARY_TYPES = {
    "backend_flow_hash",
    "backend_route_set_hash",
}
SUPPORT_ONLY_TYPES = {
    "backend_request_schema_hash",
    "backend_callsite_hash",
    "backend_endpoint",
    "inline_script_hash",
    "script_content_hash",
    "drainer_spender",
    "drainer_contract",
    "walletconnect_verify",
    "walletconnect_project",
    "infura_project",
    "kit_behavior_hash",
    "resource_graph_hash",
    "api_endpoint",
    "backend_host_pattern",
    "backend_role",
    "drainer_path",
    "drainer_symbol",
    "script_local",
    "script_host_path",
    "html_structure_hash",
    "image_asset_hash",
    "js_capability_profile",
    "interaction_pattern",
    "danger_method_combo",
    "template",
    "title",
    "body_template",
    "claim_words",
    "address",
}
WEAK_SUPPORT_TYPES = SUPPORT_ONLY_TYPES
EVIDENCE_CHANNEL_BY_TYPE = {
    "backend_kit": "backend_kit",
    "backend_request_schema_hash": "backend_schema",
    "backend_flow_hash": "backend_flow",
    "backend_route_set_hash": "backend_route_set",
    "backend_callsite_hash": "backend_callsite",
    "backend_endpoint": "backend_endpoint",
    "backend_host_pattern": "backend_host",
    "backend_role": "backend_role",
    "inline_script_hash": "source_code_support",
    "script_content_hash": "source_code_support",
    "drainer_spender": "operator_wallet_support",
    "drainer_contract": "operator_wallet_support",
    "walletconnect_verify": "walletconnect_verify_support",
    "walletconnect_project": "walletconnect_project_support",
    "infura_project": "rpc_infra_support",
    "kit_behavior_hash": "web3_behavior_support",
    "resource_graph_hash": "resource_graph_support",
    "api_endpoint": "api_support",
    "drainer_path": "source_path_support",
    "drainer_symbol": "source_text_support",
    "script_local": "resource_path_support",
    "script_host_path": "resource_path_support",
    "html_structure_hash": "html_structure_support",
    "image_asset_hash": "asset_support",
    "js_capability_profile": "capability_profile_support",
    "interaction_pattern": "interaction_pattern_support",
    "danger_method_combo": "web3_behavior_support",
    "template": "template_support",
    "title": "template_support",
    "body_template": "template_support",
    "claim_words": "template_support",
    "address": "address_support",
}
CHOOSE_FAMILY_PRIORITY = [
    ("backend_kit", 130),
    ("backend_flow", 115),
    ("backend_route_set", 105),
]
CANONICAL_KEY_TYPE_PRIORITY = {
    "backend_kit": 100,
    "backend_flow_hash": 85,
    "backend_route_set_hash": 80,
    # Support-only evidence can appear in audit/merge diagnostics but should not
    # become canonical backend-kit identity unless explicitly requested later.
    "backend_request_schema_hash": 40,
    "backend_callsite_hash": 35,
    "backend_endpoint": 20,
}
MERGEABLE_BACKEND_TYPES = {
    "backend_kit",
    "backend_flow_hash",
    "backend_route_set_hash",
}


@dataclass(frozen=True)
class Candidate:
    pool: str
    topn: int
    pca: int | None
    algorithm: str
    knn_k: int
    threshold_quantile: float
    variance_shrinkage: float
    margin_threshold: float


def is_phishing(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(PHISH_LABELS)


def first_value(values: object) -> str:
    if isinstance(values, list) and values:
        return str(values[0])
    return ""


def family_key_type(key: str) -> str:
    return str(key).split(":", 1)[0]


def evidence_channel(key: str) -> str:
    return EVIDENCE_CHANNEL_BY_TYPE.get(family_key_type(key), family_key_type(key))


def evidence_strength(key: str) -> str:
    kind = family_key_type(key)
    if kind in STRONG_PRIMARY_TYPES:
        return "strong"
    if kind in MODERATE_PRIMARY_TYPES:
        return "moderate"
    return "weak"


def _compound_family_key(strong_parts: Sequence[str]) -> str:
    material = "\n".join(sorted(str(part) for part in strong_parts if part))
    digest = hashlib.sha256(material.encode("utf-8", errors="ignore")).hexdigest()[:18]
    return f"compound_family:{digest}"


def sample_candidate_keys(meta: dict, features: set[str]) -> dict[str, list[str]]:
    """Return backend-kit candidate keys plus non-primary support evidence.

    ``static_family_key`` is now intended to approximate the same backend kit:
    shared server endpoints, request/response schema, route set, and backend
    state machine. Frontend bundle hashes, spender addresses, WalletConnect IDs,
    behaviour hashes, resource graphs and templates are retained only as support
    evidence. They do not define a family by themselves.
    """
    out: dict[str, list[str]] = {
        "backend_kit": [],
        "backend_schema": [],
        "backend_flow": [],
        "backend_route_set": [],
        "backend_callsite": [],
        "backend_endpoint": [],
        "support": [],
    }
    title = str(meta.get("title") or "")
    body_hash = str(meta.get("body_template_hash") or "")
    if title and body_hash:
        out["support"].append(f"template:{title}|body_template:{body_hash}")

    for feature in sorted(features):
        kind = feature_type(feature)
        if kind == "backend_kit":
            out["backend_kit"].append(feature)
        elif kind == "backend_flow_hash":
            out["backend_flow"].append(feature)
        elif kind == "backend_route_set_hash":
            out["backend_route_set"].append(feature)
        elif kind in SUPPORT_ONLY_TYPES or kind.startswith("js_capability"):
            # Request schemas, single endpoints and frontend callsites are
            # valuable corroborating evidence, but they are too generic to be a
            # phishing-kit family identity under the backend-kit definition.
            out["support"].append(feature)

    return {key: sorted(set(values)) for key, values in out.items()}


def choose_family_key(candidates: dict[str, list[str]], counts: Counter[str], min_count: int) -> tuple[str, str, int]:
    best_key = ""
    best_reason = ""
    best_count = 0
    best_score = -1
    for reason, base_score in CHOOSE_FAMILY_PRIORITY:
        for key in candidates.get(reason, []):
            count = int(counts.get(key, 0))
            if count < min_count:
                continue
            # Prefer high-quality evidence, then support count. Templates are only
            # a last-resort weak label and will be filtered by evidence tier later.
            score = base_score * 1_000_000 + EVIDENCE_TIER_RANK.get(evidence_strength(key), 1) * 10_000 + count
            if not best_key or score > best_score:
                best_score = score
                best_key = key
                best_reason = reason
                best_count = count
    return best_key, best_reason, best_count


def evidence_tier(
    key: str,
    reason: str,
    count: int,
    support_count: int,
    top_support_coverage: float,
    independent_support_channel_count: int,
    strong_supporting_key_count: int,
) -> tuple[str, str]:
    primary_type = family_key_type(key)
    if primary_type not in BACKEND_PRIMARY_TYPES:
        return "weak", "non-backend evidence is support-only and cannot define a backend kit family"

    has_independent_support = independent_support_channel_count >= 1 and top_support_coverage >= SUPPORT_MIN_COVERAGE

    if primary_type == "backend_kit":
        if has_independent_support or support_count >= 1 or count >= 4:
            return "strong", "backend-kit key repeats and includes backend route/schema/flow topology"
        return "moderate", "backend-kit key repeats, but independent support is sparse"

    if primary_type == "backend_flow_hash":
        if has_independent_support or strong_supporting_key_count >= 1:
            return "strong", "backend flow repeats with independent backend/support evidence"
        if count >= 4:
            return "moderate", "backend flow repeats across several samples"
        return "weak", "backend flow is sparse and lacks corroborating backend evidence"

    if primary_type == "backend_route_set_hash":
        if independent_support_channel_count >= 2 or strong_supporting_key_count >= 1:
            return "strong", "backend route-set repeats with schema/flow support"
        if has_independent_support or count >= 4:
            return "moderate", "backend route-set repeats with limited support"
        return "weak", "backend route-set repeats but has little schema or flow support"

    if primary_type == "backend_request_schema_hash":
        return "weak", "request schema is a workflow/superfamily clue, not a backend-kit identity"

    if primary_type == "backend_callsite_hash":
        return "weak", "frontend callsite can be loader/glue reuse and is support-only by default"

    if primary_type == "backend_endpoint":
        return "weak", "single endpoint/path alone is not sufficient to define a backend kit"

    return "weak", "unrecognised backend evidence"


def build_family_evidence_validation(labels: pd.DataFrame, min_count: int, out_dir: pathlib.Path) -> pd.DataFrame:
    candidate_sets: dict[str, set[str]] = {}
    for _, row in labels.iterrows():
        try:
            candidate_sets[str(row["zip_path"])] = set(json.loads(str(row["candidate_family_keys"])))
        except json.JSONDecodeError:
            candidate_sets[str(row["zip_path"])] = set()

    rows = []
    assigned = labels[labels["static_family_key"].astype(str).ne("")].copy()
    for key, group in assigned.groupby("static_family_key", sort=True):
        n = len(group)
        reason = str(group["static_family_reason"].iloc[0])
        primary_channel = evidence_channel(str(key))
        primary_type = family_key_type(str(key))
        counter: Counter[str] = Counter()
        for path in group["zip_path"].astype(str):
            counter.update(candidate_sets.get(path, set()))

        repeated_support = []
        independent_channels: set[str] = set()
        strong_supporting_key_count = 0
        for cand, support in counter.most_common():
            if cand == key:
                continue
            coverage = support / max(n, 1)
            if support >= min_count or (support >= 2 and coverage >= SUPPORT_MIN_COVERAGE):
                channel = evidence_channel(cand)
                strength = evidence_strength(cand)
                repeated_support.append((cand, support, coverage, channel, strength))
                if coverage >= SUPPORT_MIN_COVERAGE and channel != primary_channel:
                    independent_channels.add(channel)
                if coverage >= SUPPORT_MIN_COVERAGE and strength in {"strong", "moderate"}:
                    strong_supporting_key_count += 1

        support_count = sum(1 for _, _, cov, _, _ in repeated_support if cov >= SUPPORT_MIN_COVERAGE)
        top_support_coverage = max((cov for _, _, cov, _, _ in repeated_support), default=0.0)
        independent_support_channel_count = len(independent_channels)
        tier, note = evidence_tier(
            str(key),
            reason,
            n,
            support_count,
            top_support_coverage,
            independent_support_channel_count,
            strong_supporting_key_count,
        )
        rows.append({
            "static_family_key": key,
            "static_family_reason": reason,
            "primary_evidence_type": primary_type,
            "primary_evidence_channel": primary_channel,
            "n_samples": int(n),
            "evidence_tier": tier,
            "evidence_tier_rank": EVIDENCE_TIER_RANK[tier],
            "supporting_repeated_key_count": int(support_count),
            "independent_support_channel_count": int(independent_support_channel_count),
            "supporting_evidence_channels": ",".join(sorted(independent_channels)),
            "strong_supporting_key_count": int(strong_supporting_key_count),
            "top_supporting_coverage": float(top_support_coverage),
            "n_domains": int(group["domain"].astype(str).nunique()),
            "n_titles": int(group["title"].astype(str).nunique()),
            "n_body_templates": int(group["body_template_hash"].astype(str).nunique()),
            "supporting_repeated_keys": "\n".join(
                f"{cand} | type={family_key_type(cand)} | channel={channel} | strength={strength} | support={support}/{n} | coverage={coverage:.2f}"
                for cand, support, coverage, channel, strength in repeated_support[:30]
            ),
            "representative_samples": "\n".join(
                f"{r.sample_id}:{r.domain}" for r in group[["sample_id", "domain"]].head(10).itertuples(index=False)
            ),
            "evidence_note": note,
        })

    validation_columns = [
        "static_family_key",
        "static_family_reason",
        "primary_evidence_type",
        "primary_evidence_channel",
        "n_samples",
        "evidence_tier",
        "evidence_tier_rank",
        "supporting_repeated_key_count",
        "independent_support_channel_count",
        "supporting_evidence_channels",
        "strong_supporting_key_count",
        "top_supporting_coverage",
        "n_domains",
        "n_titles",
        "n_body_templates",
        "supporting_repeated_keys",
        "representative_samples",
        "evidence_note",
    ]
    if rows:
        validation = pd.DataFrame(rows, columns=validation_columns).sort_values(
            ["evidence_tier_rank", "n_samples", "static_family_key"],
            ascending=[False, False, True],
        )
    else:
        validation = pd.DataFrame(columns=validation_columns)
    validation.to_csv(out_dir / "static_family_evidence_validation.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    labels = labels.copy()
    for column in [
        "evidence_tier",
        "evidence_note",
        "supporting_repeated_keys",
        "primary_evidence_type",
        "primary_evidence_channel",
        "independent_support_channel_count",
        "supporting_evidence_channels",
        "strong_supporting_key_count",
    ]:
        if column in {"independent_support_channel_count", "strong_supporting_key_count"}:
            default = 0
        else:
            default = ""
        mapping = dict(zip(validation["static_family_key"], validation[column])) if not validation.empty else {}
        labels[column] = labels["static_family_key"].map(mapping).fillna(default)

    md = ["# Static Family Evidence Validation", ""]
    md.append("Evidence tiers are pseudo-label quality gates for backend-kit Layer2 training labels.")
    md.append("Request-schema-only, endpoint-only, frontend/source, spender and template evidence are support-only; high-confidence family labels require backend kit/flow/route topology.")
    md.append("")
    for _, row in validation.iterrows():
        md.append(f"## {row['evidence_tier'].upper()} - {row['static_family_key']}")
        md.append(f"- samples: {row['n_samples']}, reason: {row['static_family_reason']}")
        md.append(f"- primary evidence: {row['primary_evidence_type']} / {row['primary_evidence_channel']}")
        md.append(f"- domains/titles/body templates: {row['n_domains']}/{row['n_titles']}/{row['n_body_templates']}")
        md.append(
            f"- supporting repeated keys: {row['supporting_repeated_key_count']}, "
            f"independent channels: {row['independent_support_channel_count']} ({row['supporting_evidence_channels']}), "
            f"strong/moderate support keys: {row['strong_supporting_key_count']}, "
            f"top coverage: {row['top_supporting_coverage']:.2f}"
        )
        md.append(f"- note: {row['evidence_note']}")
        top = str(row["supporting_repeated_keys"]).splitlines()[:10]
        if top:
            md.append("- supporting evidence:")
            for line in top:
                md.append(f"  - {line}")
        reps = str(row["representative_samples"]).splitlines()[:8]
        if reps:
            md.append("- representative samples:")
            for line in reps:
                md.append(f"  - {line}")
        md.append("")
    (out_dir / "static_family_evidence_validation.md").write_text("\n".join(md), encoding="utf-8")
    return labels

def _json_list_value(value: object) -> list:
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    try:
        parsed = json.loads(str(value))
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return []


def _static_meta_from_precomputed(row: pd.Series) -> tuple[dict, set[str]]:
    features = set(_json_list_value(row.get("family_features_json", "[]")))
    meta = {
        "sample_id": str(row.get("sample_id", "")),
        "domain": str(row.get("domain", "")),
        "url": str(row.get("url", "")),
        "status": str(row.get("status", "")),
        "connect_confirmed": bool(row.get("connect_confirmed", False)),
        "signature_prompt_seen": bool(row.get("signature_prompt_seen", False)),
        "high_risk_wallet_prompt_seen": bool(row.get("high_risk_wallet_prompt_seen", False)),
        "title": str(row.get("title", "")),
        "body_template_hash": str(row.get("body_template_hash", "")),
        "body_tokens": _json_list_value(row.get("body_tokens", "[]")),
        "methods": _json_list_value(row.get("methods", "[]")),
        "addresses": _json_list_value(row.get("addresses", "[]")),
        "walletconnect_project_ids": _json_list_value(row.get("walletconnect_project_ids", "[]")),
        "walletconnect_verify_ids": _json_list_value(row.get("walletconnect_verify_ids", "[]")),
        "drainer_hits": _json_list_value(row.get("drainer_hits", "[]")),
        "drainer_spenders": _json_list_value(row.get("drainer_spenders", "[]")),
        "script_tokens": _json_list_value(row.get("script_tokens", "[]")),
        "api_endpoints": _json_list_value(row.get("api_endpoints", "[]")),
        "backend_endpoints": _json_list_value(row.get("backend_endpoints", "[]")),
        "backend_observations": _json_list_value(row.get("backend_observations", "[]")),
        "backend_request_schemas": _json_list_value(row.get("backend_request_schemas", "[]")),
        "backend_roles": _json_list_value(row.get("backend_roles", "[]")),
        "backend_role_sequence": _json_list_value(row.get("backend_role_sequence", "[]")),
        "backend_route_set_hashes": _json_list_value(row.get("backend_route_set_hashes", "[]")),
        "backend_request_schema_hashes": _json_list_value(row.get("backend_request_schema_hashes", "[]")),
        "backend_flow_hashes": _json_list_value(row.get("backend_flow_hashes", "[]")),
        "backend_callsite_hashes": _json_list_value(row.get("backend_callsite_hashes", "[]")),
        "backend_kit_hashes": _json_list_value(row.get("backend_kit_hashes", "[]")),
        "claim_words": _json_list_value(row.get("claim_words", "[]")),
        "infura_project_ids": _json_list_value(row.get("infura_project_ids", "[]")),
        "js_capabilities": _json_list_value(row.get("js_capabilities", "[]")),
        "interaction_pattern": str(row.get("interaction_pattern", "")),
        "inline_script_hashes": _json_list_value(row.get("inline_script_hashes", "[]")),
        "script_content_hashes": _json_list_value(row.get("script_content_hashes", "[]")),
        "resource_graph_hashes": _json_list_value(row.get("resource_graph_hashes", "[]")),
        "html_structure_hashes": _json_list_value(row.get("html_structure_hashes", "[]")),
        "image_asset_hashes": _json_list_value(row.get("image_asset_hashes", "[]")),
        "kit_behavior_hashes": _json_list_value(row.get("kit_behavior_hashes", "[]")),
        "rpc_methods": _json_list_value(row.get("rpc_methods", "[]")),
    }
    return meta, features


class UnionFind:
    def __init__(self, items: Sequence[str]):
        self.parent = {str(item): str(item) for item in items}

    def find(self, item: str) -> str:
        item = str(item)
        parent = self.parent.setdefault(item, item)
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, a: str, b: str) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def merge_static_family_anchors(
    labels: pd.DataFrame,
    min_count: int,
    out_dir: pathlib.Path,
    enabled: bool = True,
    min_overlap: float = 0.80,
    min_edge_support: int | None = None,
) -> pd.DataFrame:
    """Merge static family anchors that are near-subsets of the same evidence graph.

    This uses HAR/static evidence only while constructing confirmed family anchors.
    The downstream attribution model still receives only encrypted-traffic features.
    """
    labels = labels.copy()
    assigned_keys = sorted(k for k in labels["static_family_key"].astype(str).unique() if k)
    if not enabled or not assigned_keys:
        return labels

    assigned_set = set(assigned_keys)
    support: Counter[str] = Counter()
    cooccur: Counter[tuple[str, str]] = Counter()
    for raw in labels["candidate_family_keys"].astype(str):
        try:
            keys = sorted(set(json.loads(raw)) & assigned_set)
        except json.JSONDecodeError:
            keys = []
        for key in keys:
            support[key] += 1
        for a, b in combinations(keys, 2):
            cooccur[(a, b)] += 1

    uf = UnionFind(assigned_keys)
    edge_rows = []
    min_overlap = float(min_overlap)
    min_edge_support = max(3, int(min_edge_support if min_edge_support is not None else min_count))
    for (a, b), n in sorted(cooccur.items()):
        if n < min_edge_support:
            continue
        overlap = n / max(min(int(support[a]), int(support[b])), 1)
        jaccard = n / max(int(support[a]) + int(support[b]) - n, 1)
        type_a = family_key_type(a)
        type_b = family_key_type(b)
        # Do not merge through generic connector keys such as a single endpoint.
        # Otherwise one common path like /index/index/save_erc_data.html can
        # collapse multiple backend kits into one component.
        if type_a not in MERGEABLE_BACKEND_TYPES or type_b not in MERGEABLE_BACKEND_TYPES:
            continue
        if overlap >= min_overlap:
            uf.union(a, b)
            edge_rows.append({
                "family_key_a": a,
                "family_key_b": b,
                "cooccurrence": int(n),
                "support_a": int(support[a]),
                "support_b": int(support[b]),
                "overlap_min": float(overlap),
                "jaccard": float(jaccard),
            })

    if not edge_rows:
        labels["merged_static_family_keys"] = labels["static_family_key"].astype(str)
        labels["static_family_merge_component_size"] = labels["static_family_key"].astype(str).map(lambda x: 1 if x else 0)
        return labels

    components: dict[str, list[str]] = {}
    for key in assigned_keys:
        components.setdefault(uf.find(key), []).append(key)

    key_to_canonical: dict[str, str] = {}
    key_to_members: dict[str, list[str]] = {}
    for members in components.values():
        members = sorted(members, key=lambda k: (-CANONICAL_KEY_TYPE_PRIORITY.get(family_key_type(k), 0), -int(support[k]), k))
        canonical = members[0]
        for key in members:
            key_to_canonical[key] = canonical
            key_to_members[key] = members

    original_key = labels["static_family_key"].astype(str)
    labels["static_family_key_original"] = original_key
    labels["static_family_key"] = original_key.map(lambda key: key_to_canonical.get(key, key))
    labels["merged_static_family_keys"] = original_key.map(lambda key: " || ".join(key_to_members.get(key, [key])) if key else "")
    labels["static_family_merge_component_size"] = original_key.map(lambda key: len(key_to_members.get(key, [])) if key else 0)
    merged_counts = labels["static_family_key"].value_counts().to_dict()
    labels["static_family_count"] = labels["static_family_key"].map(merged_counts).fillna(labels["static_family_count"]).astype(int)

    pd.DataFrame(edge_rows).sort_values(["overlap_min", "cooccurrence"], ascending=[False, False]).to_csv(
        out_dir / "static_family_anchor_merge_edges.csv",
        index=False,
        quoting=csv.QUOTE_MINIMAL,
    )
    component_rows = []
    for members in components.values():
        if len(members) <= 1:
            continue
        members = sorted(members, key=lambda k: (-int(support[k]), k))
        component_rows.append({
            "canonical_static_family_key": members[0],
            "component_size": len(members),
            "component_keys": "\n".join(members),
            "component_supports": "\n".join(f"{key} | support={support[key]}" for key in members),
        })
    pd.DataFrame(component_rows).to_csv(out_dir / "static_family_anchor_merge_components.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    return labels


def build_static_family_labels(
    phishing: pd.DataFrame,
    min_count: int,
    out_dir: pathlib.Path,
    include_har: bool = True,
    anchor_merge: bool = True,
    anchor_merge_min_overlap: float = 0.80,
    anchor_merge_min_cooccur: int | None = None,
    precomputed_static_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    counts: Counter[str] = Counter()
    per_sample_candidates: list[dict[str, list[str]]] = []
    per_sample_features: list[set[str]] = []

    if precomputed_static_features is not None:
        source = precomputed_static_features.copy()
        if "include_har" in source.columns:
            source = source[pd.to_numeric(source["include_har"], errors="coerce").fillna(1).astype(int).eq(1 if include_har else 0)].copy()
        if "sample_key" in phishing.columns and "sample_key" in source.columns:
            source = source[source["sample_key"].astype(str).isin(set(phishing["sample_key"].astype(str)))].copy()
        else:
            source = source[source["zip_path"].astype(str).isin(set(phishing["zip_path"].astype(str)))].copy()
        if source.empty:
            raise ValueError("precomputed static family features did not match any phishing samples")
        static_iter = []
        for _, row in source.iterrows():
            meta, features = _static_meta_from_precomputed(row)
            static_iter.append((row, meta, features))
    else:
        static_iter = []
        for _, row in phishing.iterrows():
            sample_dir = pathlib.Path(str(row["zip_path"]))
            meta, features = family_features(sample_dir, include_har=include_har)
            static_iter.append((row, meta, features))

    for row, meta, features in static_iter:
        candidates = sample_candidate_keys(meta, features)
        flat = sorted({key for values in candidates.values() for key in values})
        counts.update(flat)
        per_sample_candidates.append(candidates)
        per_sample_features.append(features)
        out_row = {
            "zip_path": str(row.get("zip_path", "")),
            "sample_id": row.get("sample_id", meta.get("sample_id")),
            "domain": row.get("domain", meta.get("domain")),
            "url": row.get("url", meta.get("url")),
            "title": meta.get("title", ""),
            "body_template_hash": meta.get("body_template_hash", ""),
            "methods": json.dumps(meta.get("methods", []), ensure_ascii=False),
            "addresses": json.dumps(meta.get("addresses", []), ensure_ascii=False),
            "walletconnect_project_ids": json.dumps(meta.get("walletconnect_project_ids", []), ensure_ascii=False),
            "walletconnect_verify_ids": json.dumps(meta.get("walletconnect_verify_ids", []), ensure_ascii=False),
            "drainer_hits": json.dumps(meta.get("drainer_hits", []), ensure_ascii=False),
            "drainer_spenders": json.dumps(meta.get("drainer_spenders", []), ensure_ascii=False),
            "script_tokens": json.dumps(meta.get("script_tokens", []), ensure_ascii=False),
            "api_endpoints": json.dumps(meta.get("api_endpoints", []), ensure_ascii=False),
            "backend_endpoints": json.dumps(meta.get("backend_endpoints", []), ensure_ascii=False),
            "backend_observations": json.dumps(meta.get("backend_observations", []), ensure_ascii=False),
            "backend_request_schemas": json.dumps(meta.get("backend_request_schemas", []), ensure_ascii=False),
            "backend_roles": json.dumps(meta.get("backend_roles", []), ensure_ascii=False),
            "backend_role_sequence": json.dumps(meta.get("backend_role_sequence", []), ensure_ascii=False),
            "backend_route_set_hashes": json.dumps(meta.get("backend_route_set_hashes", []), ensure_ascii=False),
            "backend_request_schema_hashes": json.dumps(meta.get("backend_request_schema_hashes", []), ensure_ascii=False),
            "backend_flow_hashes": json.dumps(meta.get("backend_flow_hashes", []), ensure_ascii=False),
            "backend_callsite_hashes": json.dumps(meta.get("backend_callsite_hashes", []), ensure_ascii=False),
            "backend_kit_hashes": json.dumps(meta.get("backend_kit_hashes", []), ensure_ascii=False),
            "infura_project_ids": json.dumps(meta.get("infura_project_ids", []), ensure_ascii=False),
            "js_capabilities": json.dumps(meta.get("js_capabilities", []), ensure_ascii=False),
            "interaction_pattern": meta.get("interaction_pattern", ""),
            "inline_script_hashes": json.dumps(meta.get("inline_script_hashes", []), ensure_ascii=False),
            "script_content_hashes": json.dumps(meta.get("script_content_hashes", []), ensure_ascii=False),
            "resource_graph_hashes": json.dumps(meta.get("resource_graph_hashes", []), ensure_ascii=False),
            "html_structure_hashes": json.dumps(meta.get("html_structure_hashes", []), ensure_ascii=False),
            "image_asset_hashes": json.dumps(meta.get("image_asset_hashes", []), ensure_ascii=False),
            "kit_behavior_hashes": json.dumps(meta.get("kit_behavior_hashes", []), ensure_ascii=False),
            "rpc_methods": json.dumps(meta.get("rpc_methods", []), ensure_ascii=False),
            "candidate_family_keys": json.dumps(flat, ensure_ascii=False),
            "n_family_features": len(features),
        }
        if "sample_key" in row:
            out_row["sample_key"] = row.get("sample_key", "")
        for column in (
            "capture_id",
            "label",
            "source",
            "source_variant",
            "source_folder",
            "fine_source",
            "partition",
            "fit_role",
        ):
            if column in row:
                out_row[column] = row.get(column, "")
        rows.append(out_row)

    assigned = []
    for row, candidates, features in zip(rows, per_sample_candidates, per_sample_features):
        key, reason, count = choose_family_key(candidates, counts, min_count)
        row = dict(row)
        row["static_family_key"] = key
        row["static_family_reason"] = reason
        row["static_family_count"] = count
        row["high_conf_family"] = bool(key)
        row["family_features_json"] = json.dumps(sorted(features), ensure_ascii=False)
        assigned.append(row)

    labels = merge_static_family_anchors(
        pd.DataFrame(assigned),
        min_count,
        out_dir,
        enabled=anchor_merge,
        min_overlap=anchor_merge_min_overlap,
        min_edge_support=anchor_merge_min_cooccur,
    )
    labels = build_family_evidence_validation(labels, min_count, out_dir)
    tier_rank = labels["evidence_tier"].astype(str).str.lower().map(EVIDENCE_TIER_RANK).fillna(0)
    primary_type = labels["static_family_key"].astype(str).map(family_key_type)
    labels["high_conf_family"] = (
        labels["static_family_key"].astype(str).ne("")
        & primary_type.isin(BACKEND_KIT_PRIMARY_TYPES)
        & tier_rank.ge(EVIDENCE_TIER_RANK["moderate"])
    )
    labels.to_csv(out_dir / "static_family_label_assignments.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    counts_df = (
        labels[labels["high_conf_family"]]
        .groupby(["static_family_key", "static_family_reason"], as_index=False)
        .size()
        .rename(columns={"size": "n_samples"})
        .sort_values(["n_samples", "static_family_reason", "static_family_key"], ascending=[False, True, True])
    )
    counts_df.to_csv(out_dir / "static_family_counts.csv", index=False)
    pd.DataFrame({"candidate_family_key": list(counts.keys()), "global_count": list(counts.values())}).sort_values(
        "global_count", ascending=False
    ).to_csv(out_dir / "static_family_candidate_counts.csv", index=False)
    return labels


def split_by_family(
    labelled: pd.DataFrame,
    min_family_size: int,
    min_evidence_tier: str,
    train_fraction: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    rng = np.random.default_rng(random_state)
    family_to_cluster: dict[str, str] = {}
    train_parts = []
    test_parts = []
    min_rank = EVIDENCE_TIER_RANK.get(str(min_evidence_tier).lower(), 2)
    tier_rank = labelled["evidence_tier"].astype(str).str.lower().map(EVIDENCE_TIER_RANK).fillna(0)
    eligible = labelled[labelled["static_family_key"].astype(str).ne("") & tier_rank.ge(min_rank)].copy()
    counts = eligible["static_family_key"].value_counts()
    known_families = [fam for fam, n in counts.items() if int(n) >= int(min_family_size)]
    known_families = sorted(known_families, key=lambda f: (-int(counts[f]), f))
    family_to_cluster = {fam: f"known_kit_c{i:02d}" for i, fam in enumerate(known_families)}

    for fam in known_families:
        part = eligible[eligible["static_family_key"].eq(fam)].copy()
        idx = np.asarray(part.index)
        rng.shuffle(idx)
        n_train = int(np.floor(len(idx) * float(train_fraction)))
        n_train = min(max(n_train, 2), len(idx) - 1)
        train_parts.append(part.loc[idx[:n_train]].copy())
        test_parts.append(part.loc[idx[n_train:]].copy())

    if not train_parts:
        raise RuntimeError("No static family has enough samples for a train/test split")
    train = pd.concat(train_parts, ignore_index=False)
    test = pd.concat(test_parts, ignore_index=False)
    train["true_known_cluster"] = train["static_family_key"].map(family_to_cluster)
    test["true_known_cluster"] = test["static_family_key"].map(family_to_cluster)
    return train, test, family_to_cluster


def build_external_coverage_audit_set(
    labelled: pd.DataFrame,
    excluded_paths: set[str],
    high_conf_family_to_cluster: dict[str, str],
    min_family_size: int,
    min_evidence_tier: str = "moderate",
) -> tuple[pd.DataFrame, dict[str, str]]:
    min_rank = EVIDENCE_TIER_RANK.get(str(min_evidence_tier).lower(), 2)
    tier_rank = labelled["evidence_tier"].astype(str).str.lower().map(EVIDENCE_TIER_RANK).fillna(0)
    eligible = labelled[labelled["static_family_key"].astype(str).ne("") & tier_rank.ge(min_rank)].copy()
    counts = eligible["static_family_key"].value_counts()
    coverage_families = [fam for fam, n in counts.items() if int(n) >= int(min_family_size)]
    coverage_families = sorted(coverage_families, key=lambda f: (-int(counts[f]), f))
    coverage_family_to_cluster = {fam: f"coverage_kit_c{i:02d}" for i, fam in enumerate(coverage_families)}
    out = eligible[
        eligible["static_family_key"].isin(coverage_families)
        & ~eligible["zip_path"].astype(str).isin(excluded_paths)
    ].copy()
    if out.empty:
        out["coverage_static_cluster"] = ""
        out["coverage_in_high_conf_family"] = False
        out["true_known_cluster"] = ""
        return out, coverage_family_to_cluster
    out["coverage_static_cluster"] = out["static_family_key"].map(coverage_family_to_cluster).fillna("")
    out["coverage_in_high_conf_family"] = out["static_family_key"].isin(high_conf_family_to_cluster)
    out["true_known_cluster"] = out["static_family_key"].map(high_conf_family_to_cluster).fillna("")
    return out, coverage_family_to_cluster


def external_coverage_audit_metrics(external: pd.DataFrame, pred: pd.DataFrame) -> dict:
    if external.empty or pred.empty:
        return {
            "external_coverage_audit_n": int(len(external)),
            "external_coverage_audit_overlap_known_family_n": 0,
            "external_coverage_audit_new_family_n": 0,
        }
    true = external["true_known_cluster"].astype(str).values
    overlap = true != ""
    nearest = pred["nearest_known_cluster"].astype(str).values
    alerts = pred["known_family_alert"].astype(int).values
    correct = nearest == true
    correct_alert = (alerts == 1) & correct & overlap
    new_family = ~overlap
    metrics = {
        "external_coverage_audit_n": int(len(external)),
        "external_coverage_audit_overlap_known_family_n": int(overlap.sum()),
        "external_coverage_audit_new_family_n": int(new_family.sum()),
        "external_coverage_audit_alert_rate": float(alerts.mean()) if len(alerts) else 0.0,
        "external_coverage_audit_nearest_accuracy_on_overlap": float(correct[overlap].mean()) if overlap.any() else 0.0,
        "external_coverage_audit_correct_accept_rate_on_overlap": float(correct_alert[overlap].mean()) if overlap.any() else 0.0,
        "external_coverage_audit_accuracy_among_alerts_on_overlap": float(correct_alert.sum() / max(((alerts == 1) & overlap).sum(), 1)),
        "external_coverage_audit_alert_rate_on_new_family": float(alerts[new_family].mean()) if new_family.any() else 0.0,
    }
    return metrics


def write_external_coverage_audit_summary(external_out: pd.DataFrame, out_path: pathlib.Path) -> None:
    if external_out.empty:
        pd.DataFrame().to_csv(out_path, index=False)
        return
    rows = []
    for key, group in external_out.groupby("static_family_key", sort=True):
        nearest_counts = group["nearest_known_cluster"].astype(str).value_counts()
        alert_rate = float(group["known_family_alert"].astype(int).mean()) if len(group) else 0.0
        overlap = group["true_known_cluster"].astype(str).ne("")
        if overlap.any():
            correct_nearest = (
                group.loc[overlap, "nearest_known_cluster"].astype(str).values
                == group.loc[overlap, "true_known_cluster"].astype(str).values
            )
            overlap_acc = float(correct_nearest.mean())
        else:
            overlap_acc = 0.0
        rows.append({
            "static_family_key": key,
            "coverage_static_cluster": group["coverage_static_cluster"].astype(str).iloc[0],
            "evidence_tier": group["evidence_tier"].astype(str).iloc[0],
            "n_samples": int(len(group)),
            "in_high_conf_family": bool(group["coverage_in_high_conf_family"].astype(bool).any()),
            "expected_high_conf_cluster": group["true_known_cluster"].astype(str).replace("", np.nan).dropna().iloc[0]
            if group["true_known_cluster"].astype(str).ne("").any() else "",
            "top_predicted_cluster": nearest_counts.index[0] if len(nearest_counts) else "",
            "top_predicted_cluster_count": int(nearest_counts.iloc[0]) if len(nearest_counts) else 0,
            "alert_rate": alert_rate,
            "nearest_accuracy_on_overlap": overlap_acc,
            "representative_samples": "\n".join(
                f"{r.sample_id}:{r.domain}" for r in group[["sample_id", "domain"]].head(8).itertuples(index=False)
            ),
        })
    pd.DataFrame(rows).sort_values(["in_high_conf_family", "n_samples"], ascending=[False, False]).to_csv(
        out_path,
        index=False,
        quoting=csv.QUOTE_MINIMAL,
    )


def normalized(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    vmax = float(np.nanmax(values)) if values.size else 0.0
    if not np.isfinite(vmax) or vmax <= 0:
        return np.zeros_like(values)
    return values / vmax


WALLET_AGNOSTIC_EXCLUDE_TOKENS = ("wallet_vendor", "walletconnect", "rpc_provider", "rpc_")
WALLET_AGNOSTIC_EXCLUDE_PREFIXES = ("rpc_",)
NO_WALLET_VENDOR_EXCLUDE_TOKENS = ("wallet_vendor",)
TRAFFIC_FAMILY_EXCLUDE_TOKENS = (
    "wallet_vendor",
    "first_party",
    "first_party_site",
    "static_to_first_party",
    "third_party_static",
    "delayed_static",
    "analytics_ads",
    "har_",
)
SHAPE_INCLUDE_TOKENS = (
    "_frac",
    "_ratio",
    "entropy",
    "iat_",
    "dur_",
    "delay",
    "active_seconds",
    "max_idle_gap",
    "concurrent",
    "window_s",
    "coverage_s",
)
SHAPE_EXCLUDE_TOKENS = (
    "_bytes",
    "_pkts",
    "_n_pkts",
    "bytes_",
    "pkts_",
    "byte_rate",
    "packet_rate",
    "burst_sz",
    "burst_bytes",
    "_n_flows",
    "_n_server_ips",
)
FAMILY_POOL_EXACT_EXCLUDE = {"base_window_s", "coverage_s", "window_s"}
FAMILY_POOL_EXCLUDE_PREFIXES = (
    "anchor_",
    "capture_",
    "filter_",
    "har_",
    "pcap_",
    "quality_",
    "session_",
    "window_",
)


def feature_allowed_by_preset(col: str, preset: str) -> bool:
    lc = col.lower()
    normalized_preset = str(preset or "traffic_family").lower()
    if normalized_preset == "all":
        return True
    if normalized_preset == "no_wallet_vendor":
        return not any(tok in lc for tok in NO_WALLET_VENDOR_EXCLUDE_TOKENS)
    if normalized_preset == "traffic_family":
        return not any(tok in lc for tok in TRAFFIC_FAMILY_EXCLUDE_TOKENS)
    if normalized_preset in {"wallet_agnostic", "strict_wallet_agnostic"}:
        if any(tok in lc for tok in WALLET_AGNOSTIC_EXCLUDE_TOKENS):
            return False
        if any(lc.startswith(prefix) for prefix in WALLET_AGNOSTIC_EXCLUDE_PREFIXES):
            return False
    if normalized_preset == "strict_wallet_agnostic" and "unknown" in lc:
        return False
    if normalized_preset == "wallet_agnostic_shape":
        if any(tok in lc for tok in WALLET_AGNOSTIC_EXCLUDE_TOKENS):
            return False
        if any(lc.startswith(prefix) for prefix in WALLET_AGNOSTIC_EXCLUDE_PREFIXES):
            return False
        if any(tok in lc for tok in SHAPE_EXCLUDE_TOKENS):
            return False
        return any(tok in lc for tok in SHAPE_INCLUDE_TOKENS)
    return True


def apply_feature_preset(cols: Sequence[str], preset: str) -> list[str]:
    return [c for c in cols if feature_allowed_by_preset(c, preset)]


def clip_scaled(values: np.ndarray, scaled_clip: float | None) -> np.ndarray:
    if scaled_clip is None or float(scaled_clip) <= 0:
        return values
    return np.clip(values, -float(scaled_clip), float(scaled_clip))


def family_candidate_pool(cols: Sequence[str], pool: str) -> list[str]:
    if pool not in {"wallet_rpc_backend", "postconnect_no_site_static", "timing_core", "cross_family"}:
        return candidate_pool(cols, pool)
    out: list[str] = []
    for c in cols:
        lc = c.lower()
        if c in FAMILY_POOL_EXACT_EXCLUDE or lc.startswith(FAMILY_POOL_EXCLUDE_PREFIXES):
            continue
        if pool == "wallet_rpc_backend":
            if any(tok in lc for tok in ["walletconnect", "rpc_", "rpc_provider", "backend", "udp443"]):
                out.append(c)
        elif pool == "postconnect_no_site_static":
            if any(tok in lc for tok in ["first_party_site", "static_to_first_party", "har_", "quality_", "pcap_"]):
                continue
            if c.startswith(("base_", "role_", "phase_", "rpc_", "proto_", "flow_")):
                out.append(c)
        elif pool == "timing_core":
            if any(tok in lc for tok in ["delay", "dur", "iat", "burst", "flow_", "rate", "active", "n_pkts", "bytes"]):
                if c.startswith(("base_", "role_", "phase_", "rpc_", "proto_", "flow_")):
                    out.append(c)
        elif pool == "cross_family":
            if any(tok in lc for tok in TRAFFIC_FAMILY_EXCLUDE_TOKENS):
                continue
            if not c.startswith(("base_", "role_", "phase_", "rpc_", "proto_", "flow_")):
                continue
            if c.startswith("base_"):
                if any(tok in lc for tok in ["iat", "active", "idle", "ud_", "large_pkt_frac", "small_pkt_frac", "burst_sz", "n_bursts"]):
                    out.append(c)
            elif c.startswith("phase_"):
                if any(tok in lc for tok in ["walletconnect", "rpc_provider", "backend_or_other", "unknown"]):
                    if not lc.endswith("_bytes"):
                        out.append(c)
                elif any(tok in lc for tok in ["large_pkt_frac", "burst_sz", "activity_decay"]):
                    out.append(c)
            elif c.startswith("role_"):
                if any(tok in lc for tok in ["walletconnect", "rpc_provider", "backend_or_other", "unknown", "entropy", "dominant"]):
                    out.append(c)
            elif c.startswith(("rpc_", "flow_")):
                out.append(c)
            elif c.startswith("proto_") and not any(tok in lc for tok in ["first_party_site", "third_party_static"]):
                out.append(c)
    return out


def family_fisher_scores(z: np.ndarray, labels: Sequence[str]) -> np.ndarray:
    labels = np.asarray(labels, dtype=str)
    families = np.unique(labels)
    if len(families) < 2 or len(labels) <= len(families):
        return np.zeros(z.shape[1], dtype=float)
    overall = np.nanmean(z, axis=0)
    between = np.zeros(z.shape[1], dtype=float)
    within = np.zeros(z.shape[1], dtype=float)
    for family in families:
        part = z[labels == family]
        if len(part) == 0:
            continue
        mu = np.nanmean(part, axis=0)
        between += len(part) * (mu - overall) ** 2
        within += np.nansum((part - mu) ** 2, axis=0)
    between /= max(len(families) - 1, 1)
    within /= max(len(labels) - len(families), 1)
    return between / (within + 1e-6)


def selected_feature_weights(selected: Sequence[str], ranking: pd.DataFrame) -> np.ndarray:
    if ranking.empty or "metric_weight" not in ranking.columns:
        return np.ones(len(selected), dtype=float)
    score_by_feature = ranking.set_index("feature")["metric_weight"].to_dict()
    weights = np.asarray([float(score_by_feature.get(feature, 1.0)) for feature in selected], dtype=float)
    weights = np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0)
    weights = np.sqrt(np.maximum(weights, 0.0) + 0.05)
    mean = float(weights.mean()) if weights.size else 1.0
    if mean <= 0 or not np.isfinite(mean):
        return np.ones(len(selected), dtype=float)
    return np.clip(weights / mean, 0.25, 4.0)


def select_family_aware_features(
    known_train: pd.DataFrame,
    benign_calibration: pd.DataFrame,
    cols: Sequence[str],
    topn: int,
    corr_threshold: float,
    family_weight: float,
    binary_weight: float,
    random_state: int,
    scaled_clip: float | None,
) -> tuple[list[str], pd.DataFrame]:
    labelled = pd.concat([known_train, benign_calibration], ignore_index=True)
    x_all = transform_frame(labelled, cols)
    miss = x_all.isna().mean()
    var = x_all.var(numeric_only=True)
    usable = [c for c in cols if miss.get(c, 1.0) < 0.8 and var.get(c, 0.0) > 1e-12]
    if not usable:
        return [], pd.DataFrame()

    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    z_all = clip_scaled(scaler.fit_transform(imputer.fit_transform(transform_frame(labelled, usable))), scaled_clip)

    z_family = clip_scaled(scaler.transform(imputer.transform(transform_frame(known_train, usable))), scaled_clip)
    y_family = known_train["true_known_cluster"].astype(str).values
    fam_mi = mutual_info_classif(z_family, y_family, random_state=random_state, discrete_features=False)
    fam_fisher = family_fisher_scores(z_family, y_family)

    y_binary = np.r_[np.ones(len(known_train)), np.zeros(len(benign_calibration))]
    bin_mi = mutual_info_classif(z_all, y_binary, random_state=random_state, discrete_features=False)
    family_signal = 0.55 * normalized(fam_mi) + 0.45 * normalized(fam_fisher)
    combined = float(family_weight) * family_signal + float(binary_weight) * normalized(bin_mi)
    ranking = pd.DataFrame({
        "feature": usable,
        "family_mi": fam_mi,
        "family_fisher": fam_fisher,
        "binary_mi": bin_mi,
        "family_signal": family_signal,
        "combined_score": combined,
        "metric_weight": family_signal,
    }).sort_values("combined_score", ascending=False)

    ordered = ranking["feature"].tolist()
    zdf = pd.DataFrame(z_all, columns=usable)[ordered]
    picked: list[str] = []
    for c in ordered:
        if len(picked) >= int(topn):
            break
        if not picked:
            picked.append(c)
            continue
        max_corr = zdf[picked].corrwith(zdf[c]).abs().max()
        if pd.isna(max_corr) or max_corr < float(corr_threshold):
            picked.append(c)
    return picked, ranking


def fit_preprocess(
    train: pd.DataFrame,
    parts: Sequence[pd.DataFrame],
    cols: Sequence[str],
    pca_components: int | None,
    random_state: int,
    scaled_clip: float | None,
):
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    x_train = clip_scaled(scaler.fit_transform(imputer.fit_transform(transform_frame(train, cols))), scaled_clip)
    pca = None
    if pca_components is not None:
        n = min(int(pca_components), x_train.shape[1], x_train.shape[0] - 1)
        if n >= 2:
            pca = PCA(n_components=n, random_state=random_state, whiten=True)
            x_train = pca.fit_transform(x_train)
    transformed = []
    for part in parts:
        z = clip_scaled(scaler.transform(imputer.transform(transform_frame(part, cols))), scaled_clip)
        if pca is not None:
            z = pca.transform(z)
        transformed.append(z)
    return imputer, scaler, pca, transformed


class FamilyPrototypeModel:
    def __init__(
        self,
        threshold_quantile: float,
        variance_shrinkage: float,
        margin_threshold: float = 1.0,
        feature_weights: np.ndarray | None = None,
    ):
        self.threshold_quantile = float(threshold_quantile)
        self.variance_shrinkage = float(variance_shrinkage)
        self.margin_threshold = float(margin_threshold)
        self.feature_weights_ = None if feature_weights is None else np.asarray(feature_weights, dtype=float)
        self.family_names_: list[str] = []
        self.family_sizes_: dict[str, int] = {}
        self.mu_: dict[str, np.ndarray] = {}
        self.var_: dict[str, np.ndarray] = {}
        self.thresholds_: dict[str, float] = {}

    def fit(self, x: np.ndarray, labels: Sequence[str]) -> "FamilyPrototypeModel":
        labels = np.asarray(labels, dtype=str)
        global_var = np.var(x, axis=0) + 1e-6
        for family in sorted(np.unique(labels)):
            xc = x[labels == family]
            mu = xc.mean(axis=0)
            var_emp = np.var(xc, axis=0) if len(xc) > 1 else global_var
            var = (1.0 - self.variance_shrinkage) * var_emp + self.variance_shrinkage * global_var + 1e-6
            dist = self._distance_to(xc, mu, var)
            threshold = float(np.quantile(dist, self.threshold_quantile)) if len(dist) else 1.0
            self.family_names_.append(family)
            self.family_sizes_[family] = int(len(xc))
            self.mu_[family] = mu
            self.var_[family] = var
            self.thresholds_[family] = max(threshold, 1e-6)
        return self

    @staticmethod
    def _weighted_mean(values: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
        if weights is None:
            return np.mean(values, axis=1)
        weights = np.asarray(weights, dtype=float)
        denom = max(float(np.sum(weights)), 1e-12)
        return np.sum(values * weights, axis=1) / denom

    def _distance_to(self, x: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
        return np.sqrt(self._weighted_mean(((x - mu) ** 2) / var, self.feature_weights_))

    def predict(self, x: np.ndarray) -> pd.DataFrame:
        distances = [self._distance_to(x, self.mu_[family], self.var_[family]) for family in self.family_names_]
        dmat = np.vstack(distances).T
        best_idx = dmat.argmin(axis=1)
        nearest = [self.family_names_[i] for i in best_idx]
        best = dmat[np.arange(len(x)), best_idx]
        if dmat.shape[1] > 1:
            partitioned = np.partition(dmat, 1, axis=1)
            second = partitioned[:, 1]
        else:
            second = np.full(len(x), np.inf)
        margin = second / np.maximum(best, 1e-9)
        thresholds = np.asarray([self.thresholds_[family] for family in nearest], dtype=float)
        ratios = best / thresholds
        return pd.DataFrame({
            "nearest_known_cluster": nearest,
            "nearest_family_distance": best,
            "second_nearest_family_distance": second,
            "family_distance_margin": margin,
            "family_threshold": thresholds,
            "family_distance_ratio": ratios,
            "known_family_alert": ((ratios <= 1.0) & (margin >= self.margin_threshold)).astype(int),
        })


class FamilyKnnModel:
    def __init__(
        self,
        threshold_quantile: float,
        margin_threshold: float = 1.0,
        knn_k: int = 3,
        feature_weights: np.ndarray | None = None,
    ):
        self.threshold_quantile = float(threshold_quantile)
        self.margin_threshold = float(margin_threshold)
        self.knn_k = max(1, int(knn_k))
        self.feature_weights_ = None if feature_weights is None else np.asarray(feature_weights, dtype=float)
        self.family_names_: list[str] = []
        self.family_sizes_: dict[str, int] = {}
        self.thresholds_: dict[str, float] = {}
        self.x_train_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None

    def fit(self, x: np.ndarray, labels: Sequence[str]) -> "FamilyKnnModel":
        self.x_train_ = np.asarray(x, dtype=float)
        self.labels_ = np.asarray(labels, dtype=str)
        self.family_names_ = sorted(np.unique(self.labels_).tolist())
        self.family_sizes_ = {family: int((self.labels_ == family).sum()) for family in self.family_names_}
        dmat = self._distance_matrix(self.x_train_, self.x_train_)
        np.fill_diagonal(dmat, np.inf)
        fallback = np.nanmedian(dmat[np.isfinite(dmat)]) if np.isfinite(dmat).any() else 1.0
        for family in self.family_names_:
            idx = np.where(self.labels_ == family)[0]
            scores = []
            for i in idx:
                same = idx[idx != i]
                if len(same) == 0:
                    scores.append(float(fallback))
                    continue
                d = np.sort(dmat[i, same])
                k = min(self.knn_k, len(d))
                scores.append(float(np.mean(d[:k])))
            threshold = float(np.quantile(scores, self.threshold_quantile)) if scores else float(fallback)
            self.thresholds_[family] = max(threshold, 1e-6)
        return self

    def _distance_matrix(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        diff = x[:, None, :] - y[None, :, :]
        sq = diff ** 2
        if self.feature_weights_ is not None:
            weights = np.asarray(self.feature_weights_, dtype=float)
            sq = sq * weights
            return np.sqrt(np.sum(sq, axis=2) / max(float(weights.sum()), 1e-12))
        return np.sqrt(np.mean(sq, axis=2))

    def predict(self, x: np.ndarray) -> pd.DataFrame:
        if self.x_train_ is None or self.labels_ is None:
            raise RuntimeError("FamilyKnnModel is not fitted")
        dmat = self._distance_matrix(np.asarray(x, dtype=float), self.x_train_)
        family_scores = []
        for family in self.family_names_:
            idx = np.where(self.labels_ == family)[0]
            k = min(self.knn_k, len(idx))
            family_dist = np.partition(dmat[:, idx], k - 1, axis=1)[:, :k]
            family_scores.append(family_dist.mean(axis=1))
        score_mat = np.vstack(family_scores).T
        best_idx = score_mat.argmin(axis=1)
        nearest = [self.family_names_[i] for i in best_idx]
        best = score_mat[np.arange(len(x)), best_idx]
        if score_mat.shape[1] > 1:
            partitioned = np.partition(score_mat, 1, axis=1)
            second = partitioned[:, 1]
        else:
            second = np.full(len(x), np.inf)
        margin = second / np.maximum(best, 1e-9)
        thresholds = np.asarray([self.thresholds_[family] for family in nearest], dtype=float)
        ratios = best / thresholds
        return pd.DataFrame({
            "nearest_known_cluster": nearest,
            "nearest_family_distance": best,
            "second_nearest_family_distance": second,
            "family_distance_margin": margin,
            "family_threshold": thresholds,
            "family_distance_ratio": ratios,
            "known_family_alert": ((ratios <= 1.0) & (margin >= self.margin_threshold)).astype(int),
        })


class FamilyExtraTreesModel:
    def __init__(
        self,
        threshold_quantile: float,
        margin_threshold: float,
        target_fpr: float,
        random_state: int,
    ):
        self.threshold_quantile = float(threshold_quantile)
        self.margin_threshold = float(margin_threshold)
        self.target_fpr = float(target_fpr)
        self.random_state = int(random_state)
        self.family_names_: list[str] = []
        self.family_sizes_: dict[str, int] = {}
        self.prob_threshold_: float = 0.5
        self.distance_threshold_: float = 0.5
        self.model_: ExtraTreesClassifier | None = None

    def _make_model(self, seed_offset: int = 0) -> ExtraTreesClassifier:
        return ExtraTreesClassifier(
            n_estimators=300,
            max_features="sqrt",
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=self.random_state + int(seed_offset),
            n_jobs=-1,
        )

    def fit(self, x: np.ndarray, labels: Sequence[str], x_benign_calib: np.ndarray | None = None) -> "FamilyExtraTreesModel":
        labels = np.asarray(labels, dtype=str)
        self.model_ = self._make_model()
        self.model_.fit(x, labels)
        self.family_names_ = [str(c) for c in self.model_.classes_]
        self.family_sizes_ = {family: int((labels == family).sum()) for family in self.family_names_}

        counts = pd.Series(labels).value_counts()
        n_splits = int(min(5, counts.min())) if len(counts) else 0
        if n_splits >= 2:
            oof_conf = np.zeros(len(labels), dtype=float)
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
            for fold, (train_idx, val_idx) in enumerate(cv.split(x, labels)):
                fold_model = self._make_model(seed_offset=fold + 1)
                fold_model.fit(x[train_idx], labels[train_idx])
                oof_conf[val_idx] = fold_model.predict_proba(x[val_idx]).max(axis=1)
            train_conf = oof_conf
        else:
            train_conf = self.model_.predict_proba(x).max(axis=1)
        # Tree probabilities are not calibrated enough for a hard benign-calibration
        # quantile: benign samples often receive high max-probability for some family.
        # Use the anchor-confidence quantile as the candidate gate and let the outer
        # objective penalize measured benign FPR on the held-out benign split.
        train_threshold = float(np.quantile(train_conf, max(0.0, 1.0 - self.threshold_quantile)))
        self.prob_threshold_ = float(np.clip(train_threshold, 0.0, 0.999999))
        self.distance_threshold_ = max(1.0 - self.prob_threshold_, 1e-6)
        return self

    def predict(self, x: np.ndarray) -> pd.DataFrame:
        if self.model_ is None:
            raise RuntimeError("FamilyExtraTreesModel is not fitted")
        proba = self.model_.predict_proba(x)
        best_idx = proba.argmax(axis=1)
        nearest = [str(self.model_.classes_[i]) for i in best_idx]
        best_prob = proba[np.arange(len(x)), best_idx]
        if proba.shape[1] > 1:
            partitioned = np.partition(proba, -2, axis=1)
            second_prob = partitioned[:, -2]
        else:
            second_prob = np.zeros(len(x), dtype=float)
        margin = best_prob / np.maximum(second_prob, 1e-9)
        distance = 1.0 - best_prob
        second_distance = 1.0 - second_prob
        ratios = distance / self.distance_threshold_
        thresholds = np.full(len(x), self.distance_threshold_, dtype=float)
        return pd.DataFrame({
            "nearest_known_cluster": nearest,
            "nearest_family_distance": distance,
            "second_nearest_family_distance": second_distance,
            "family_distance_margin": margin,
            "family_threshold": thresholds,
            "family_distance_ratio": ratios,
            "known_family_alert": ((best_prob >= self.prob_threshold_) & (margin >= self.margin_threshold)).astype(int),
        })


def calibrate_metric_thresholds(
    model,
    x_known_calib: np.ndarray,
    known_calib: pd.DataFrame,
    x_benign_calib: np.ndarray,
    target_fpr: float,
) -> dict:
    """Calibrate metric-family thresholds with high-confidence holdout and benign traffic.

    The nearest-family classifier is unchanged. Only the open-set accept gate is
    adjusted using distances, margins, and benign calibration traffic.
    """
    if not hasattr(model, "thresholds_") or not getattr(model, "thresholds_", None):
        return {
            "threshold_calibration_enabled": False,
            "threshold_calibration_reason": "model has no metric thresholds",
        }
    old_thresholds = {str(k): float(v) for k, v in model.thresholds_.items()}
    known_pred = model.predict(x_known_calib)
    true = known_calib["true_known_cluster"].astype(str).values
    nearest = known_pred["nearest_known_cluster"].astype(str).values
    distances = known_pred["nearest_family_distance"].astype(float).values
    correct_nearest = nearest == true

    calibrated = dict(old_thresholds)
    family_rows = []
    q = float(getattr(model, "threshold_quantile", 0.8))
    for family in getattr(model, "family_names_", []):
        mask = (true == family) & correct_nearest
        n_correct = int(mask.sum())
        if n_correct:
            threshold = max(float(np.quantile(distances[mask], q)), 1e-6)
            calibrated[str(family)] = threshold
            source = "high_conf_holdout_correct_nearest"
        else:
            threshold = old_thresholds.get(str(family), 1.0)
            source = "train_leave_one_fallback"
        family_rows.append({
            "known_cluster": str(family),
            "old_threshold": old_thresholds.get(str(family), np.nan),
            "holdout_calibrated_threshold": float(threshold),
            "n_holdout_correct_nearest": n_correct,
            "source": source,
        })

    model.thresholds_ = {family: max(float(thr), 1e-6) for family, thr in calibrated.items()}
    benign_before = model.predict(x_benign_calib) if len(x_benign_calib) else pd.DataFrame()
    fpr_before = float(benign_before["known_family_alert"].mean()) if len(benign_before) else 0.0
    scale = 1.0
    if len(benign_before) and fpr_before > float(target_fpr):
        margin = benign_before["family_distance_margin"].astype(float).values
        ratio = benign_before["family_distance_ratio"].astype(float).values
        margin_pass = margin >= float(getattr(model, "margin_threshold", 1.0))
        if margin_pass.any():
            candidate_scale = float(np.quantile(ratio[margin_pass], np.clip(float(target_fpr), 0.0, 1.0)))
            if np.isfinite(candidate_scale):
                scale = min(1.0, max(candidate_scale, 1e-6))
                model.thresholds_ = {
                    family: max(float(thr) * scale, 1e-6)
                    for family, thr in model.thresholds_.items()
                }

    benign_after = model.predict(x_benign_calib) if len(x_benign_calib) else pd.DataFrame()
    known_after = model.predict(x_known_calib) if len(x_known_calib) else pd.DataFrame()
    if len(known_after):
        known_alert = known_after["known_family_alert"].astype(int).values
        known_nearest = known_after["nearest_known_cluster"].astype(str).values
        known_correct_alert = (known_alert == 1) & (known_nearest == true)
    else:
        known_alert = np.asarray([], dtype=int)
        known_correct_alert = np.asarray([], dtype=bool)
    return {
        "threshold_calibration_enabled": True,
        "threshold_calibration_reason": "high_conf_holdout_distance_quantile_capped_by_benign_calibration",
        "threshold_calibration_target_fpr": float(target_fpr),
        "threshold_calibration_global_scale": float(scale),
        "threshold_calibration_benign_fpr_before_scale": fpr_before,
        "threshold_calibration_benign_fpr_after_scale": float(benign_after["known_family_alert"].mean()) if len(benign_after) else 0.0,
        "threshold_calibration_known_holdout_nearest_accuracy": float(correct_nearest.mean()) if len(correct_nearest) else 0.0,
        "threshold_calibration_known_holdout_accept_recall": float(known_alert.mean()) if len(known_alert) else 0.0,
        "threshold_calibration_known_holdout_correct_accept_rate": float(known_correct_alert.mean()) if len(known_correct_alert) else 0.0,
        "threshold_calibration_family_thresholds": family_rows,
    }


def safe_auc(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(y, score)), float(average_precision_score(y, score))


def evaluate_candidate(
    model: FamilyPrototypeModel,
    x_known_test: np.ndarray,
    known_test: pd.DataFrame,
    x_benign_test: np.ndarray,
    x_unknown_phish: np.ndarray,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    known_pred = model.predict(x_known_test)
    benign_pred = model.predict(x_benign_test)
    unknown_pred = model.predict(x_unknown_phish) if len(x_unknown_phish) else pd.DataFrame()

    true = known_test["true_known_cluster"].astype(str).values
    nearest = known_pred["nearest_known_cluster"].astype(str).values
    alerts = known_pred["known_family_alert"].astype(int).values
    correct_nearest = nearest == true
    correct_alert = (alerts == 1) & correct_nearest

    known_ratio = known_pred["family_distance_ratio"].astype(float).values
    benign_ratio = benign_pred["family_distance_ratio"].astype(float).values
    y_auc = np.r_[np.ones(len(known_ratio)), np.zeros(len(benign_ratio))]
    score_auc = -np.r_[known_ratio, benign_ratio]
    auroc, auprc = safe_auc(y_auc, score_auc)
    metrics = {
        "known_family_test_nearest_accuracy": float(correct_nearest.mean()) if len(correct_nearest) else 0.0,
        "known_family_test_accept_recall": float(alerts.mean()) if len(alerts) else 0.0,
        "known_family_test_correct_accept_rate": float(correct_alert.mean()) if len(correct_alert) else 0.0,
        "known_family_test_accuracy_among_alerts": float(correct_alert.sum() / max(alerts.sum(), 1)),
        "benign_test_fpr": float(benign_pred["known_family_alert"].mean()) if len(benign_pred) else 0.0,
        "unknown_phish_affinity": float(unknown_pred["known_family_alert"].mean()) if len(unknown_pred) else 0.0,
        "known_vs_benign_distance_auroc": auroc,
        "known_vs_benign_distance_auprc": auprc,
        "known_family_nmi_nearest": float(normalized_mutual_info_score(true, nearest)) if len(true) else 0.0,
        "known_family_ari_nearest": float(adjusted_rand_score(true, nearest)) if len(true) else 0.0,
    }
    return metrics, known_pred, benign_pred, unknown_pred


def objective(metrics: dict, target_fpr: float) -> float:
    fpr = float(metrics["benign_test_fpr"])
    penalty = 2.5 * max(0.0, fpr - float(target_fpr))
    return (
        1.6 * float(metrics["known_family_test_nearest_accuracy"])
        + 1.0 * float(metrics["known_family_nmi_nearest"])
        + 0.8 * float(metrics["known_family_ari_nearest"])
        + 0.8 * float(metrics["known_family_test_correct_accept_rate"])
        + 0.3 * float(metrics["known_family_test_accept_recall"])
        + 0.4 * float(metrics["known_vs_benign_distance_auroc"])
        - 2.0 * fpr
        - 0.2 * float(metrics["unknown_phish_affinity"])
        - penalty
    )


def attach(df: pd.DataFrame, pred: pd.DataFrame, split: str) -> pd.DataFrame:
    out = pd.concat([df.reset_index(drop=True), pred.reset_index(drop=True)], axis=1)
    out["split"] = split
    return out


def plot_results(
    known_train_pred: pd.DataFrame,
    known_test_pred: pd.DataFrame,
    benign_pred: pd.DataFrame,
    unknown_pred: pd.DataFrame,
    family_sizes: dict[str, int],
    metrics: dict,
    pca_points: dict,
    out_path: pathlib.Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))
    fig.suptitle("Family-Aware First Layer", fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    keys = list(family_sizes.keys())
    vals = [family_sizes[k] for k in keys]
    ax.bar(keys, vals, color="#4E79A7")
    ax.set_title("Known Kit Family Prototype Sizes")
    ax.set_ylabel("Training phishing samples")
    ax.tick_params(axis="x", rotation=60, labelsize=8)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[0, 1]
    train_ratio = known_train_pred["family_distance_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
    test_ratio = known_test_pred["family_distance_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
    benign_ratio = benign_pred["family_distance_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
    unknown_ratio = unknown_pred["family_distance_ratio"].replace([np.inf, -np.inf], np.nan).dropna() if len(unknown_pred) else pd.Series(dtype=float)
    combined = pd.concat([train_ratio, test_ratio, benign_ratio, unknown_ratio], ignore_index=True)
    max_x = float(np.nanpercentile(combined, 98)) if len(combined) else 3.0
    bins = np.linspace(0, max(1.8, max_x), 36)
    ax.hist(benign_ratio, bins=bins, alpha=0.62, color="#7F7F7F", label="benign test")
    ax.hist(unknown_ratio, bins=bins, alpha=0.52, color="#F28E2B", label="other phishing")
    ax.hist(test_ratio, bins=bins, alpha=0.70, color="#59A14F", label="known-family test")
    ax.hist(train_ratio, bins=bins, alpha=0.30, color="#4E79A7", label="known-family train")
    ax.axvline(1.0, color="#222222", linestyle="--", linewidth=1.5)
    ax.set_title("Distance to Nearest Known Kit Prototype")
    ax.set_xlabel("nearest distance / family threshold")
    ax.set_ylabel("Samples")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 0]
    labels = [
        "nearest acc",
        "correct accept",
        "benign FPR",
        "AUROC",
        "NMI",
    ]
    vals = [
        metrics["known_family_test_nearest_accuracy"],
        metrics["known_family_test_correct_accept_rate"],
        metrics["benign_test_fpr"],
        metrics["known_vs_benign_distance_auroc"],
        metrics["known_family_nmi_nearest"],
    ]
    colors = ["#59A14F", "#59A14F", "#E15759", "#4E79A7", "#4E79A7"]
    ax.bar(labels, vals, color=colors)
    ax.set_ylim(0, 1.05)
    ax.set_title("Holdout Metrics")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 1]
    if pca_points:
        ax.scatter(pca_points["benign"][:, 0], pca_points["benign"][:, 1], s=12, alpha=0.22, color="#7F7F7F", label="benign")
        if len(pca_points["unknown"]):
            ax.scatter(pca_points["unknown"][:, 0], pca_points["unknown"][:, 1], s=16, alpha=0.45, color="#F28E2B", marker="x", label="other phishing")
        fams = np.asarray(pca_points["train_family"])
        for fam in sorted(np.unique(fams))[:14]:
            mask = fams == fam
            ax.scatter(pca_points["train"][mask, 0], pca_points["train"][mask, 1], s=20, alpha=0.82, label=fam)
        ax.set_title("2D View of Selected Traffic Space")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(frameon=False, fontsize=6, ncol=2)
        ax.grid(alpha=0.20)
    else:
        ax.set_axis_off()

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.features, low_memory=False)
    dyn = select_window(df, args.window).copy()
    dyn["is_phishing"] = is_phishing(dyn["label"]).astype(int)
    phishing = dyn[dyn["is_phishing"].eq(1)].copy()
    benign = dyn[dyn["is_phishing"].eq(0)].copy()
    if phishing.empty or benign.empty:
        raise ValueError("Need both phishing and benign dynamic samples")

    include_har_static_evidence = not bool(getattr(args, "exclude_har_static_evidence", False))
    static_family_features_path = getattr(args, "static_family_features", None)
    precomputed_static_features = (
        pd.read_csv(static_family_features_path, low_memory=False)
        if static_family_features_path
        else None
    )
    static_labels = build_static_family_labels(
        phishing,
        int(args.static_family_min_count),
        out_dir,
        include_har=include_har_static_evidence,
        anchor_merge=not bool(getattr(args, "disable_anchor_merge", False)),
        anchor_merge_min_overlap=float(getattr(args, "anchor_merge_min_overlap", 0.80)),
        anchor_merge_min_cooccur=getattr(args, "anchor_merge_min_cooccur", None),
        precomputed_static_features=precomputed_static_features,
    )
    family_join_col = "sample_key" if "sample_key" in phishing.columns and "sample_key" in static_labels.columns else "zip_path"
    label_cols = [
        family_join_col, "static_family_key", "static_family_reason", "static_family_count",
        "high_conf_family", "evidence_tier", "evidence_note", "supporting_repeated_keys",
    ]
    phishing_labeled = phishing.merge(
        static_labels[label_cols].drop_duplicates(family_join_col),
        on=family_join_col,
        how="left",
    )
    phishing_labeled["static_family_key"] = phishing_labeled["static_family_key"].fillna("")
    phishing_labeled["evidence_tier"] = phishing_labeled["evidence_tier"].fillna("")

    known_train, known_test, family_to_cluster = split_by_family(
        phishing_labeled,
        int(args.train_family_min_size),
        str(args.min_evidence_tier),
        float(args.family_train_fraction),
        int(args.random_state),
    )
    known_paths = set(known_train["zip_path"].astype(str)) | set(known_test["zip_path"].astype(str))
    unknown_phish = phishing_labeled[~phishing_labeled["zip_path"].astype(str).isin(known_paths)].copy()
    external_coverage = pd.DataFrame()
    coverage_family_to_cluster: dict[str, str] = {}
    external_coverage_enabled = (
        bool(getattr(args, "external_coverage_audit", False))
        and str(args.min_evidence_tier).lower() == "strong"
    )
    if external_coverage_enabled:
        external_coverage, coverage_family_to_cluster = build_external_coverage_audit_set(
            phishing_labeled,
            excluded_paths=known_paths,
            high_conf_family_to_cluster=family_to_cluster,
            min_family_size=int(getattr(args, "external_coverage_min_size", args.train_family_min_size)),
            min_evidence_tier=str(getattr(args, "external_coverage_min_evidence_tier", "moderate")),
        )

    benign_calib_paths, benign_test_paths = train_test_split(
        benign["zip_path"].astype(str).values,
        train_size=float(args.benign_calibration_fraction),
        random_state=int(args.random_state),
        shuffle=True,
    )
    benign_calib = benign[benign["zip_path"].astype(str).isin(set(benign_calib_paths))].copy()
    benign_test = benign[benign["zip_path"].astype(str).isin(set(benign_test_paths))].copy()

    evidence_by_key = static_labels.drop_duplicates("static_family_key").set_index("static_family_key")
    cluster_map = pd.DataFrame(
        [{
            "known_cluster": cluster,
            "static_family_key": key,
            "static_family_reason": evidence_by_key.at[key, "static_family_reason"] if key in evidence_by_key.index else "",
            "evidence_tier": evidence_by_key.at[key, "evidence_tier"] if key in evidence_by_key.index else "",
            "evidence_note": evidence_by_key.at[key, "evidence_note"] if key in evidence_by_key.index else "",
        } for key, cluster in family_to_cluster.items()]
    ).sort_values("known_cluster")
    cluster_map.to_csv(out_dir / "known_family_cluster_map.csv", index=False)

    feature_preset = str(getattr(args, "feature_preset", "traffic_family") or "traffic_family")
    all_numeric = numeric_feature_columns(dyn, window_name=None, extra_exclude=["cluster", "known_cluster", "is_phishing"])
    all_numeric = apply_feature_preset(all_numeric, feature_preset)
    algorithms = [str(a).lower() for a in getattr(args, "algorithms", ["knn"])]
    knn_values = [int(k) for k in getattr(args, "knn_k", [1, 3, 5])]
    search_rows = []
    cache: dict[int, dict] = {}
    for pool in args.pools:
        pool_cols = family_candidate_pool(all_numeric, pool)
        if not pool_cols:
            continue
        for topn in args.topn:
            selected, ranking = select_family_aware_features(
                known_train,
                benign_calib,
                pool_cols,
                int(topn),
                float(args.corr_threshold),
                float(args.family_weight),
                float(args.binary_weight),
                int(args.random_state),
                float(getattr(args, "scaled_clip", 3.0)),
            )
            if len(selected) < 3:
                continue
            raw_feature_weights = selected_feature_weights(selected, ranking)
            for pca_value in args.pca_components:
                pca_n = None if str(pca_value).lower() == "none" else int(pca_value)
                try:
                    imputer, scaler, pca, transformed = fit_preprocess(
                        known_train,
                        [known_train, known_test, benign_calib, benign_test, unknown_phish, external_coverage],
                        selected,
                        pca_n,
                        int(args.random_state),
                        float(getattr(args, "scaled_clip", 3.0)),
                    )
                except Exception:
                    continue
                x_known_train, x_known_test, x_benign_calib, x_benign_test, x_unknown, x_external_coverage = transformed
                feature_weights = None if pca is not None else raw_feature_weights
                for q in args.threshold_quantiles:
                    for margin in args.margin_thresholds:
                        candidate_specs = []
                        if "knn" in algorithms:
                            candidate_specs.extend(("knn", int(k), 0.0) for k in knn_values)
                        if "prototype" in algorithms:
                            candidate_specs.extend(("prototype", 0, float(s)) for s in args.variance_shrinkage)
                        if "extra_trees" in algorithms and pca is None:
                            candidate_specs.append(("extra_trees", 0, 0.0))
                        for algorithm, knn_k, shrinkage in candidate_specs:
                            candidate = Candidate(
                                pool=pool,
                                topn=int(topn),
                                pca=pca_n,
                                algorithm=algorithm,
                                knn_k=int(knn_k),
                                threshold_quantile=float(q),
                                variance_shrinkage=float(shrinkage),
                                margin_threshold=float(margin),
                            )
                            train_labels = known_train["true_known_cluster"].astype(str).values
                            if algorithm == "knn":
                                model = FamilyKnnModel(
                                    float(q),
                                    margin_threshold=float(margin),
                                    knn_k=int(knn_k),
                                    feature_weights=feature_weights,
                                ).fit(x_known_train, train_labels)
                            elif algorithm == "extra_trees":
                                model = FamilyExtraTreesModel(
                                    float(q),
                                    margin_threshold=float(margin),
                                    target_fpr=float(args.target_fpr),
                                    random_state=int(args.random_state),
                                ).fit(x_known_train, train_labels, x_benign_calib=x_benign_calib)
                            else:
                                model = FamilyPrototypeModel(
                                    float(q),
                                    float(shrinkage),
                                    float(margin),
                                    feature_weights=feature_weights,
                                ).fit(x_known_train, train_labels)
                            calibration = {"threshold_calibration_enabled": False}
                            if bool(getattr(args, "calibrate_thresholds", False)):
                                calibration = calibrate_metric_thresholds(
                                    model,
                                    x_known_test,
                                    known_test,
                                    x_benign_calib,
                                    float(args.target_fpr),
                                )
                            metrics, known_test_pred, benign_pred, unknown_pred = evaluate_candidate(
                                model,
                                x_known_test,
                                known_test,
                                x_benign_test,
                                x_unknown,
                            )
                            row = {
                                **asdict(candidate),
                                "n_features": len(selected),
                                "n_known_families": len(family_to_cluster),
                                "threshold_calibration_enabled": bool(calibration.get("threshold_calibration_enabled", False)),
                                "threshold_calibration_global_scale": calibration.get("threshold_calibration_global_scale"),
                                "threshold_calibration_benign_fpr_after_scale": calibration.get("threshold_calibration_benign_fpr_after_scale"),
                                **metrics,
                            }
                            row["objective"] = objective(metrics, float(args.target_fpr))
                            cache_key = len(search_rows)
                            row["cache_key"] = cache_key
                            search_rows.append(row)
                            cache[cache_key] = {
                                "candidate": candidate,
                                "selected": selected,
                                "ranking": ranking,
                                "imputer": imputer,
                                "scaler": scaler,
                                "pca": pca,
                                "x_known_train": x_known_train,
                                "x_known_test": x_known_test,
                                "x_benign_calib": x_benign_calib,
                                "x_benign_test": x_benign_test,
                                "x_unknown": x_unknown,
                                "x_external_coverage": x_external_coverage,
                                "model": model,
                                "calibration": calibration,
                                "metrics": metrics,
                                "known_test_pred": known_test_pred,
                                "benign_pred": benign_pred,
                                "unknown_pred": unknown_pred,
                            }

    if not search_rows:
        raise RuntimeError("No valid known kit/drainer attribution candidates were produced")

    search = pd.DataFrame(search_rows).sort_values("objective", ascending=False).reset_index(drop=True)
    search.to_csv(out_dir / "family_aware_search.csv", index=False)
    best_key = int(search.iloc[0]["cache_key"])
    best = cache[best_key]
    model = best["model"]

    known_train_pred = model.predict(best["x_known_train"])
    known_test_pred = best["known_test_pred"]
    benign_pred = best["benign_pred"]
    unknown_pred = best["unknown_pred"]
    external_coverage_pred = model.predict(best["x_external_coverage"]) if len(external_coverage) else pd.DataFrame()

    train_out = attach(known_train, known_train_pred, "known_family_train")
    test_out = attach(known_test, known_test_pred, "known_family_test")
    benign_out = attach(benign_test, benign_pred, "benign_test")
    unknown_out = attach(unknown_phish, unknown_pred, "other_phishing")
    external_coverage_out = attach(external_coverage, external_coverage_pred, "coverage_priority_external_audit") if len(external_coverage) else pd.DataFrame()
    for frame in [train_out, test_out, benign_out, unknown_out]:
        if "true_known_cluster" not in frame.columns:
            frame["true_known_cluster"] = ""
    train_out.to_csv(out_dir / "known_family_train_predictions.csv", index=False)
    test_out.to_csv(out_dir / "known_family_test_predictions.csv", index=False)
    benign_out.to_csv(out_dir / "benign_test_predictions.csv", index=False)
    unknown_out.to_csv(out_dir / "other_phishing_predictions.csv", index=False)
    if len(external_coverage_out):
        external_coverage_out.to_csv(out_dir / "coverage_priority_external_predictions.csv", index=False)
        write_external_coverage_audit_summary(
            external_coverage_out,
            out_dir / "coverage_priority_external_audit_summary.csv",
        )
    pd.concat([train_out, test_out, benign_out, unknown_out], ignore_index=True).to_csv(
        out_dir / "family_aware_first_layer_predictions.csv", index=False
    )

    pd.DataFrame({"feature": best["selected"]}).to_csv(out_dir / "selected_features.csv", index=False)
    best["ranking"].to_csv(out_dir / "feature_ranking.csv", index=False)

    setup = {
        "window": args.window,
        "static_family_min_count": int(args.static_family_min_count),
        "include_har_static_evidence": bool(include_har_static_evidence),
        "static_family_features": str(static_family_features_path or ""),
        "static_family_join_col": family_join_col,
        "anchor_merge_enabled": not bool(getattr(args, "disable_anchor_merge", False)),
        "anchor_merge_min_overlap": float(getattr(args, "anchor_merge_min_overlap", 0.80)),
        "anchor_merge_min_cooccur": getattr(args, "anchor_merge_min_cooccur", None),
        "feature_preset": feature_preset,
        "traffic_feature_policy": "HAR/static evidence is used only for family-anchor pseudo-labels; attribution features are numeric encrypted-traffic features after preset/pool filtering.",
        "training_policy": (
            "Default production policy: train the family classifier only on high-confidence strong anchors; "
            "use benign calibration plus high-confidence holdout to calibrate the distance/margin open-set gate; "
            "keep coverage-priority samples out of training and emit them only as external/generalization/audit predictions."
        ),
        "threshold_calibration_policy": "Metric thresholds are fitted from high-confidence holdout distances and capped by benign calibration traffic.",
        "threshold_calibration": best.get("calibration", {}),
        "external_coverage_audit_enabled": bool(external_coverage_enabled),
        "external_coverage_min_evidence_tier": str(getattr(args, "external_coverage_min_evidence_tier", "moderate")),
        "external_coverage_min_size": int(getattr(args, "external_coverage_min_size", args.train_family_min_size)),
        "algorithms": algorithms,
        "knn_k": knn_values,
        "scaled_clip": float(getattr(args, "scaled_clip", 3.0)),
        "train_family_min_size": int(args.train_family_min_size),
        "min_evidence_tier": str(args.min_evidence_tier),
        "family_train_fraction": float(args.family_train_fraction),
        "benign_calibration_fraction": float(args.benign_calibration_fraction),
        "random_state": int(args.random_state),
        "n_phishing_dyn": int(len(phishing)),
        "n_benign_dyn": int(len(benign)),
        "n_known_family_train": int(len(known_train)),
        "n_known_family_test": int(len(known_test)),
        "n_other_phishing": int(len(unknown_phish)),
        "n_external_coverage_audit": int(len(external_coverage)),
        "n_benign_calibration": int(len(benign_calib)),
        "n_benign_test": int(len(benign_test)),
        "n_known_families": int(len(family_to_cluster)),
        "n_external_coverage_families": int(len(coverage_family_to_cluster)),
        "static_family_evidence_tier_counts": static_labels["evidence_tier"].value_counts(dropna=False).to_dict(),
        "best_candidate": asdict(best["candidate"]),
        "selected_feature_count": len(best["selected"]),
        "selected_features": best["selected"],
        "family_cluster_sizes": model.family_sizes_,
    }
    external_metrics = external_coverage_audit_metrics(external_coverage, external_coverage_pred)
    full_metrics = {"setup": setup, **best["metrics"], **external_metrics}
    (out_dir / "family_aware_metrics.json").write_text(json.dumps(full_metrics, indent=2), encoding="utf-8")
    joblib.dump(
        {
            "candidate": best["candidate"],
            "feature_preset": feature_preset,
            "scaled_clip": float(getattr(args, "scaled_clip", 3.0)),
            "selected_features": best["selected"],
            "imputer": best["imputer"],
            "scaler": best["scaler"],
            "pca": best["pca"],
            "family_to_cluster": family_to_cluster,
            "prototype_model": model,
        },
        out_dir / "family_aware_first_layer_model.joblib",
    )

    pca_points = {}
    all_x = np.vstack([best["x_known_train"], best["x_known_test"], best["x_benign_test"], best["x_unknown"]])
    if all_x.shape[0] >= 4 and all_x.shape[1] >= 2:
        plot_pca = PCA(n_components=2, random_state=int(args.random_state))
        plot_pca.fit(all_x)
        pca_points = {
            "train": plot_pca.transform(best["x_known_train"]),
            "known_test": plot_pca.transform(best["x_known_test"]),
            "benign": plot_pca.transform(best["x_benign_test"]),
            "unknown": plot_pca.transform(best["x_unknown"]) if len(best["x_unknown"]) else np.empty((0, 2)),
            "train_family": known_train["true_known_cluster"].astype(str).values,
        }
    plot_results(
        train_out,
        test_out,
        benign_out,
        unknown_out,
        model.family_sizes_,
        full_metrics,
        pca_points,
        out_dir / "family_aware_first_layer_results.png",
    )

    print(json.dumps(full_metrics, indent=2))
    print(f"wrote {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build known kit/drainer attribution from static kit evidence and traffic features.")
    parser.add_argument("--features", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--window", default="dyn")
    parser.add_argument("--static-family-features", default=None)
    parser.add_argument("--static-family-min-count", type=int, default=4)
    parser.add_argument(
        "--exclude-har-static-evidence",
        action="store_true",
        help="Do not scan browser.har while building offline static family pseudo-labels.",
    )
    parser.add_argument(
        "--disable-anchor-merge",
        action="store_true",
        help="Do not merge highly co-occurring static evidence keys before creating family anchors.",
    )
    parser.add_argument("--anchor-merge-min-overlap", type=float, default=0.80)
    parser.add_argument("--anchor-merge-min-cooccur", type=int, default=None)
    parser.add_argument("--train-family-min-size", type=int, default=4)
    parser.add_argument("--min-evidence-tier", choices=["weak", "moderate", "strong"], default="moderate")
    parser.add_argument("--family-train-fraction", type=float, default=0.70)
    parser.add_argument("--benign-calibration-fraction", type=float, default=0.50)
    parser.add_argument("--target-fpr", type=float, default=0.10)
    parser.add_argument(
        "--disable-threshold-calibration",
        dest="calibrate_thresholds",
        action="store_false",
        default=True,
        help="Disable distance-threshold calibration from high-confidence holdout plus benign calibration.",
    )
    parser.add_argument(
        "--external-coverage-audit",
        dest="external_coverage_audit",
        action="store_true",
        default=True,
        help="For strong/high-confidence runs, predict moderate+ coverage samples as external audit only.",
    )
    parser.add_argument("--skip-external-coverage-audit", dest="external_coverage_audit", action="store_false")
    parser.add_argument("--external-coverage-min-evidence-tier", choices=["weak", "moderate", "strong"], default="moderate")
    parser.add_argument("--external-coverage-min-size", type=int, default=5)
    parser.add_argument("--corr-threshold", type=float, default=0.92)
    parser.add_argument("--family-weight", type=float, default=0.70)
    parser.add_argument("--binary-weight", type=float, default=0.30)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--feature-preset",
        choices=["traffic_family", "no_wallet_vendor", "wallet_agnostic", "wallet_agnostic_shape", "strict_wallet_agnostic", "all"],
        default="traffic_family",
        help="Limit Layer 2 traffic features. traffic_family removes wallet-vendor and front-end/site features for cross-URL family attribution.",
    )
    parser.add_argument(
        "--scaled-clip",
        type=float,
        default=3.0,
        help="Clip robust-scaled Layer 2 features to +/- this value; use 0 to disable.",
    )
    parser.add_argument("--pools", nargs="+", default=[
        "cross_family", "wallet_rpc_backend", "timing_core", "postconnect_no_site_static",
    ])
    parser.add_argument("--algorithms", nargs="+", choices=["knn", "prototype", "extra_trees"], default=["knn", "prototype"])
    parser.add_argument("--knn-k", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--topn", nargs="+", type=int, default=[20, 40, 80, 120])
    parser.add_argument("--pca-components", nargs="+", default=["none", "10", "20"])
    parser.add_argument("--threshold-quantiles", nargs="+", type=float, default=[0.5, 0.7, 0.8, 0.9])
    parser.add_argument("--variance-shrinkage", nargs="+", type=float, default=[0.05, 0.15, 0.30])
    parser.add_argument("--margin-thresholds", nargs="+", type=float, default=[1.0, 1.05, 1.10, 1.20, 1.35])
    run(parser.parse_args())


if __name__ == "__main__":
    main()

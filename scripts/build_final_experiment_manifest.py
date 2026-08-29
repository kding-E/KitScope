#!/usr/bin/env python3
"""Build the metadata-enriched manifest and E0 coverage audits.

The raw snapshot remains immutable.  This script joins only collection-time
metadata and optional Layer-2 static-evidence labels.  Missing research fields
are kept explicit and are never inferred from model predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from urllib.parse import urlparse

import pandas as pd

from prepare_final_layer1_features import canonical_dataset_key


MANAGED_HOST_SUFFIXES = (
    ".vercel.app",
    ".pages.dev",
    ".netlify.app",
    ".workers.dev",
    ".github.io",
    ".web.app",
    ".firebaseapp.com",
    ".glitch.me",
    ".replit.app",
    ".render.com",
    ".fly.dev",
)
COMMON_SECOND_LEVEL_SUFFIXES = {
    "ac.uk", "co.id", "co.in", "co.jp", "co.kr", "co.th", "co.uk", "co.za",
    "com.ar", "com.au", "com.bd", "com.br", "com.cn", "com.co", "com.eg",
    "com.hk", "com.kh", "com.mm", "com.mx", "com.my", "com.ng", "com.pe",
    "com.ph", "com.pk", "com.pl", "com.ru", "com.sa", "com.sg", "com.tr",
    "com.tw", "com.ua", "com.ve", "com.vn", "edu.cn", "firm.in", "go.th",
    "gov.uk", "ne.jp", "net.au", "net.cn", "net.in", "or.jp", "or.th",
    "org.au", "org.cn", "org.in", "org.uk",
}
KNOWN_WALLETS = (
    "metamask", "rabby", "rainbow", "coinbase", "phantom", "okx", "bitget",
    "trust", "walletconnect", "brave", "argent", "zerion",
)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_id(prefix: str, *values: object, length: int = 24) -> str:
    material = "\x1f".join(str(v or "").strip().lower() for v in values)
    return f"{prefix}:{_sha256_text(material)[:length]}"


def _clean(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def _read_json(path: Path) -> dict:
    try:
        obj = json.loads(path.read_text(encoding="utf-8-sig", errors="replace"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _sample_metadata(sample_path: Path, source: str = "") -> tuple[dict, str]:
    # The two public-mobile sources are individual PCAP/JSON payload files and
    # never carry our collector metadata.  Avoid a remote-drive stat per row.
    if source in {"appact", "browser_mobile"}:
        return {}, "none"
    for name in ("session.json", "sample.json", "sample_private.json"):
        candidate = sample_path / name
        if candidate.exists():
            return _read_json(candidate), name
    return {}, "missing"


def _host(value: object) -> str:
    text = _clean(value).lower().strip(". ")
    if "://" in text:
        try:
            text = (urlparse(text).hostname or "").lower().strip(".")
        except Exception:
            text = ""
    if ":" in text and not text.startswith("["):
        text = text.split(":", 1)[0]
    return text


def registrable_domain(host: object) -> str:
    """Dataset grouping key with tenant isolation for managed hosting."""
    normalized = _host(host)
    if not normalized:
        return ""
    if normalized.endswith(MANAGED_HOST_SUFFIXES):
        return normalized
    parts = [part for part in normalized.split(".") if part]
    if len(parts) <= 2:
        return normalized
    suffix2 = ".".join(parts[-2:])
    if suffix2 in COMMON_SECOND_LEVEL_SUFFIXES and len(parts) >= 3:
        return ".".join(parts[-3:])
    return suffix2


def _parse_note_fields(notes: object) -> dict[str, str]:
    fields: dict[str, str] = {}
    for item in re.split(r"[;\n]", _clean(notes)):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        fields[key.strip().lower()] = value.strip()
    return fields


def _timestamp_from_name(value: str) -> str:
    patterns = (
        (r"(20\d{6})T(\d{6})(?:\d{3})?", "%Y%m%d%H%M%S"),
        (r"_(20\d{6})_(\d{6})(?:\D|$)", "%Y%m%d%H%M%S"),
    )
    for pattern, fmt in patterns:
        match = re.search(pattern, value)
        if match:
            try:
                parsed = datetime.strptime("".join(match.groups()), fmt).replace(tzinfo=timezone.utc)
                return parsed.isoformat().replace("+00:00", "Z")
            except ValueError:
                pass
    epoch = re.match(r"^(1[0-9]{9})(?:_|$)", Path(value).stem)
    if epoch:
        try:
            return datetime.fromtimestamp(int(epoch.group(1)), tz=timezone.utc).isoformat().replace(
                "+00:00", "Z"
            )
        except (ValueError, OSError, OverflowError):
            pass
    return ""


def _normalize_timestamp(value: object) -> str:
    text = _clean(value)
    if not text:
        return ""
    parsed = pd.to_datetime(text, utc=True, errors="coerce")
    if pd.isna(parsed):
        return ""
    return parsed.isoformat().replace("+00:00", "Z")


def _metadata_domain(meta: dict, path: Path, source: str) -> str:
    sample = meta.get("sample", {}) or {}
    source_row = meta.get("source_row", {}) or {}
    candidates = (
        meta.get("domain"),
        meta.get("url"),
        meta.get("target_url"),
        sample.get("url"),
        source_row.get("url"),
        source_row.get("openphish_url"),
        source_row.get("phishunt_domain"),
        source_row.get("certpl_domain"),
    )
    for value in candidates:
        host = _host(value)
        if host:
            return host
    name = path.name
    if source == "browser_mobile":
        match = re.search(r"^\d+_(.+?)_20\d{6}T?\d{6}", name)
        if match:
            return _host(match.group(1).replace("_", "."))
    match = re.search(r"_(?:https?__)?([A-Za-z0-9.-]+?)(?:__|_20\d{6})", name)
    return _host(match.group(1)) if match else ""


def _wallet(meta: dict, path: Path, source: str) -> tuple[str, str, str]:
    wallet = meta.get("wallet", {}) or {}
    browser = meta.get("browser", {}) or {}
    config = meta.get("config", {}) or {}
    candidates = (
        wallet.get("type"), wallet.get("display_name"), browser.get("wallet_type"),
        browser.get("wallet_display_name"), config.get("active_wallet"),
        config.get("wallet_display_name"),
    )
    selected = ""
    for value in candidates:
        text = _clean(value).lower().replace(" ", "_")
        if text:
            selected = text
            break
    if not selected:
        lower_name = path.name.lower()
        for known in KNOWN_WALLETS:
            if re.search(rf"(?:^|_){re.escape(known)}(?:_|$)", lower_name):
                selected = known
                break
    flags = meta.get("flags", {}) or {}
    if not selected and source in {"phish_drainer", "blockchain"} and (
        config.get("metamask_extension_id") or any("metamask" in str(k) and bool(v) for k, v in flags.items())
    ):
        selected = "metamask"
    if not selected:
        return "", "NA", ""
    version = _clean(wallet.get("version") or config.get("wallet_version"))
    return selected, "browser_extension", version


def _interaction_outcome(meta: dict) -> str:
    flags = meta.get("flags", {}) or {}
    loop = meta.get("loop_result", {}) or {}
    if bool(flags.get("signature_prompt_seen") or flags.get("high_risk_wallet_prompt_seen")):
        return "signature_prompt"
    if bool(flags.get("connect_confirmed")):
        return "connect_confirmed"
    if bool(flags.get("walletconnect_qr_seen")):
        return "walletconnect_qr"
    if int(loop.get("dummy_submits") or 0) > 0:
        return "form_submit"
    if bool(flags.get("connect_trigger_clicked")):
        return "connect_clicked"
    if _clean(meta.get("outcome_category")):
        return _clean(meta.get("outcome_category"))
    if _clean(meta.get("status")):
        return _clean(meta.get("status"))
    return "page_only"


def _site_scale(notes: str, source: str) -> tuple[str, str, str]:
    if source != "blockchain_hard":
        return "unknown", "not_available", "not_audited"
    fields = _parse_note_fields(notes)
    evidence = " ".join((fields.get("rationale", ""), notes)).lower()
    normalized_evidence = re.sub(r"[_-]+", " ", evidence)
    if re.search(r"\b(small|micro)\b", normalized_evidence):
        return "small", f"session_metadata:{fields.get('rationale', notes)}", "metadata_explicit"
    if re.search(r"\b(niche|regional|emerging)\b", normalized_evidence):
        return "niche", f"session_metadata:{fields.get('rationale', notes)}", "metadata_explicit"
    if re.search(r"\b(mainstream|major|large|blue chip|top tier)\b", normalized_evidence):
        return "mainstream", f"session_metadata:{fields.get('rationale', notes)}", "metadata_explicit"
    hard_reason = fields.get("hard_reason", "").lower()
    source_name = fields.get("source", "").lower()
    tvl_text = fields.get("tvl_usd", "")
    try:
        tvl = float(tvl_text.replace(",", ""))
    except (TypeError, ValueError):
        tvl = None
    # These thresholds are frozen metadata-only rules.  They use the TVL
    # snapshot already stored by the target collector, never detector scores.
    # The collector's explicit long-tail designation remains niche even when
    # its point-in-time TVL exceeds the small-site cutoff.
    if "defillama_long_tail" in hard_reason:
        tier = "small" if tvl is not None and tvl <= 100_000 else "niche"
        return tier, f"collection_target:{source_name};hard_reason={hard_reason};tvl_usd={tvl_text}", "collection_metadata_rule"
    if tvl is not None and source_name.startswith("local:"):
        tier = "small" if tvl <= 100_000 else "niche" if tvl <= 5_000_000 else "mainstream"
        return tier, f"collection_target:{source_name};hard_reason={hard_reason};tvl_usd={tvl_text}", "collection_metadata_rule"
    return "unknown", f"session_metadata:{fields.get('rationale', notes)}", "manual_review_required"


def _source_specific_fields(row: pd.Series, meta: dict, metadata_file: str) -> dict:
    source = _clean(row["source"])
    path = Path(_clean(row["sample_path"]))
    sample = meta.get("sample", {}) or {}
    source_row = meta.get("source_row", {}) or {}
    notes = _clean(meta.get("notes") or sample.get("notes") or source_row.get("notes"))
    note_fields = _parse_note_fields(notes)
    domain = _metadata_domain(meta, path, source)
    reg_domain = registrable_domain(domain)

    if source == "appact":
        app_category = PureWindowsPath(_clean(row["relative_sample_path"])).parts[1]
        benign_behavior = "mobile_app"
        service_material = f"appact:{app_category.lower()}"
    elif source == "browser_mobile":
        parts = PureWindowsPath(_clean(row["relative_sample_path"])).parts
        site_split = parts[2] if len(parts) > 2 else "unknown"
        benign_behavior = "browser"
        service_material = f"browser_mobile:{reg_domain or site_split.lower()}"
    elif source == "browser_same_pipeline":
        interaction_type = _clean(source_row.get("interaction_type"))
        vertical = _clean(meta.get("vertical") or source_row.get("vertical"))
        benign_behavior = interaction_type or vertical or "interactive_browser"
        service_material = f"browser_same_pipeline:{reg_domain or meta.get('sample_id') or path.name}"
    elif source in {"blockchain", "blockchain_hard"}:
        benign_behavior = "dapp_wallet" if source == "blockchain_hard" else "dapp"
        service_material = f"dapp:{reg_domain or sample.get('sample_id') or path.name}"
    else:
        benign_behavior = "NA"
        service_material = ""

    wallet_family, wallet_client_type, wallet_version = _wallet(meta, path, source)
    sample_native_id = _clean(
        sample.get("sample_id") or meta.get("sample_id") or meta.get("session_id") or source_row.get("sample_id")
    )
    hard_target = ""
    if source == "blockchain_hard":
        # Newer rough-wallet captures retain their original curated target in
        # notes; older captures use the native benign_collection_* sample ID.
        target_material = _clean(note_fields.get("source_sample_id")) or sample_native_id
        if not target_material:
            target_match = re.search(
                r"(benign_(?:collection|rough_wallet|hard_whitelist)_\d+)", path.name, re.I
            )
            target_material = target_match.group(1) if target_match else ""
        if target_material:
            hard_target = _stable_id("hard_target", target_material)
    if hard_target:
        service_material = hard_target

    start = _normalize_timestamp(
        meta.get("start_utc_ts") or meta.get("created_ts_utc") or (meta.get("pcap", {}) or {}).get("start_ts_utc")
    ) or _timestamp_from_name(path.name)
    end = _normalize_timestamp(
        meta.get("end_utc_ts") or meta.get("finished_ts_utc") or (meta.get("pcap", {}) or {}).get("stop_ts_utc")
    )
    capture_month = start[:7] if start else "unknown"

    browser = meta.get("browser", {}) or {}
    context = browser.get("context_kwargs_redacted", {}) or {}
    ua = _clean(context.get("user_agent"))
    version_match = re.search(r"(?:Chrome|Chromium)/(\d+(?:\.\d+)*)", ua)
    browser_version = version_match.group(1) if version_match else ""
    template = ((meta.get("browser_profile", {}) or {}).get("template_fingerprint", {}) or {})
    template_hash = _clean(template.get("sha256_name_size"))
    interface = _clean((meta.get("pcap", {}) or {}).get("interface") or (meta.get("config", {}) or {}).get("network_interface"))
    collector = (
        "windows_v05" if metadata_file in {"sample.json", "sample_private.json"}
        else "multiwallet_web3" if meta.get("wallet")
        else "legacy_web3" if metadata_file == "session.json"
        else "public_mobile"
    )
    sensor_version = f"{collector}:dumpcap" if meta else f"{collector}:source_capture"
    access_network_id = _stable_id("network", interface or source)
    client_stack_id = _stable_id("client", collector, ua, wallet_family or "no_wallet")
    environment_id = _stable_id(
        "environment", sensor_version, access_network_id, client_stack_id, capture_month
    )
    browser_profile_id = _stable_id("profile", template_hash or collector, source)

    automation_policy_material = json.dumps(
        {
            "collector": collector,
            "mode": meta.get("mode"),
            "capture_filter": (meta.get("pcap", {}) or {}).get("capture_filter"),
            "status_policy": meta.get("network_artifact_retention"),
        },
        sort_keys=True,
        ensure_ascii=True,
        default=str,
    )
    automation_policy_id = _stable_id("automation", automation_policy_material)

    if source == "phish_interaction":
        phishing_type = "credential"
    elif source == "phish_drainer":
        phishing_type = "drainer"
    else:
        phishing_type = "NA"

    if sample_native_id:
        collection_material = f"{source}:{sample_native_id}"
    else:
        collection_material = f"{source}:{row['relative_sample_path']}"
    collection_unit = _stable_id("collection", collection_material)
    source_url = _clean(meta.get("url") or sample.get("url") or source_row.get("url"))
    source_url_hash = _clean(
        (meta.get("url_redacted", {}) or {}).get("url_sha256") or source_row.get("url_sha256")
    ) or (_sha256_text(source_url) if source_url else "")

    scale_tier, scale_evidence, scale_status = _site_scale(notes, source)
    hosting_type = _clean(note_fields.get("hosting_type"))
    if not hosting_type and domain.endswith(MANAGED_HOST_SUFFIXES):
        hosting_type = "shared_host"
    hosting_type = hosting_type or "unknown"
    if source == "phish_drainer" and scale_tier == "unknown" and hosting_type == "shared_host":
        # Pre-score scale proxy allowed by the protocol: a campaign-specific
        # tenant on a shared hosting suffix is treated as a small hosted site.
        # Custom-domain drainers remain unknown unless a ranking/target record
        # provides stronger collection-time evidence.
        managed_suffix = next((suffix for suffix in MANAGED_HOST_SUFFIXES if domain.endswith(suffix)), "shared_host")
        scale_tier = "small"
        scale_evidence = f"collection-time managed hosting tenant: {managed_suffix}"
        scale_status = "hosting_tenant_proxy"
    category = _clean(note_fields.get("category"))
    if not category and source == "browser_same_pipeline":
        category = _clean(source_row.get("vertical") or meta.get("vertical"))

    return {
        "phishing_type": phishing_type,
        "environment_id": environment_id,
        "sensor_version": sensor_version,
        "capture_path_id": collector,
        "access_network_id": access_network_id,
        "client_stack_id": client_stack_id,
        "browser_version": browser_version or "unknown",
        "browser_profile_id": browser_profile_id,
        "capture_start_utc": start,
        "capture_end_utc": end,
        "capture_era": capture_month,
        "domain": domain,
        "registrable_domain": reg_domain,
        "source_url_hash": source_url_hash,
        "campaign_id": "",
        "backend_cluster_id": "",
        "drainer_deployment_id": (
            _stable_id("deployment", sample_native_id or reg_domain) if source == "phish_drainer" else ""
        ),
        "benign_behavior_family": benign_behavior,
        "benign_service_group_id": _stable_id("benign_service", service_material) if service_material else "",
        "hard_benign_target_id": hard_target,
        "dapp_category": category or "NA",
        "site_scale_tier": scale_tier,
        "site_scale_evidence": scale_evidence,
        "site_scale_evidence_status": scale_status,
        "hosting_type": hosting_type,
        "interaction_outcome": _interaction_outcome(meta),
        "collection_unit_id": collection_unit,
        "automation_policy_id": automation_policy_id,
        "transport_dominance": "unknown_pending_feature_join",
        "wallet_family_id": wallet_family,
        "wallet_client_type": wallet_client_type,
        "wallet_version": wallet_version or "unknown",
        "metadata_file": metadata_file,
        "metadata_available": metadata_file not in {"none", "missing"},
        "metadata_notes": notes,
    }


class DisjointSet:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def _load_kit_labels(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(columns=[
            "canonical_dataset_key", "kit_family_id", "backend_cluster_id",
            "kit_evidence_tier", "kit_label_hash", "kit_fit_eligible",
        ])
    frame = pd.read_csv(path, low_memory=False, usecols=lambda c: c in {
        "zip_path", "sample_path", "static_family_key", "true_family", "kit_family_id",
        "kit_cluster", "backend_cluster_id", "evidence_tier", "kit_evidence_tier",
        "label_correction_final", "backend_kit_training_label", "fit_eligible",
        "label_hash", "kit_label_hash",
    })
    path_column = "zip_path" if "zip_path" in frame else "sample_path" if "sample_path" in frame else None
    if path_column is None:
        raise ValueError("kit label file requires zip_path or sample_path")
    frame["canonical_dataset_key"] = frame[path_column].map(canonical_dataset_key)
    if "kit_family_id" not in frame:
        source = "static_family_key" if "static_family_key" in frame else "true_family"
        frame["kit_family_id"] = frame[source].map(_clean)
    else:
        frame["kit_family_id"] = frame["kit_family_id"].map(_clean)
    if "backend_cluster_id" not in frame:
        frame["backend_cluster_id"] = frame["kit_cluster"].map(_clean) if "kit_cluster" in frame else ""
    frame["kit_evidence_tier"] = (
        frame["kit_evidence_tier"] if "kit_evidence_tier" in frame
        else frame["evidence_tier"] if "evidence_tier" in frame else "unknown"
    )
    frame["kit_label_hash"] = (
        frame["kit_label_hash"] if "kit_label_hash" in frame
        else frame["label_hash"] if "label_hash" in frame else ""
    )
    fit_source = (
        frame["fit_eligible"] if "fit_eligible" in frame
        else frame["backend_kit_training_label"] if "backend_kit_training_label" in frame
        else frame["kit_family_id"].ne("")
    )
    frame["kit_fit_eligible"] = fit_source.astype(str).str.lower().isin({"1", "true", "yes"})
    if frame["canonical_dataset_key"].duplicated().any():
        duplicate = frame[frame["canonical_dataset_key"].duplicated(False)]
        inconsistent = duplicate.groupby("canonical_dataset_key")["kit_family_id"].nunique()
        if (inconsistent > 1).any():
            raise ValueError("kit label file maps a sample path to multiple kit families")
        frame = frame.drop_duplicates("canonical_dataset_key")
    return frame[[
        "canonical_dataset_key", "kit_family_id", "backend_cluster_id",
        "kit_evidence_tier", "kit_label_hash", "kit_fit_eligible",
    ]]


def _load_transport_dominance(path: Path | None) -> pd.DataFrame:
    """Derive label-blind session transport metadata from frozen Layer1 rows."""
    columns = [
        "zip_path", "proto_tcp_byte_frac", "proto_tcp_pkt_frac",
        "proto_udp443_byte_frac", "proto_udp443_pkt_frac",
    ]
    if path is None:
        return pd.DataFrame(columns=[
            "canonical_dataset_key", "transport_dominance_feature",
            "transport_tcp_fraction", "transport_udp443_fraction",
        ])
    if path.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(path, columns=columns)
    else:
        frame = pd.read_csv(path, usecols=columns, low_memory=False)
    frame["canonical_dataset_key"] = frame["zip_path"].map(canonical_dataset_key)
    for column in columns[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    aggregate = frame.groupby("canonical_dataset_key", as_index=False).agg({
        "proto_tcp_byte_frac": "median", "proto_tcp_pkt_frac": "median",
        "proto_udp443_byte_frac": "median", "proto_udp443_pkt_frac": "median",
    })
    aggregate["transport_tcp_fraction"] = aggregate[[
        "proto_tcp_byte_frac", "proto_tcp_pkt_frac"
    ]].max(axis=1)
    aggregate["transport_udp443_fraction"] = aggregate[[
        "proto_udp443_byte_frac", "proto_udp443_pkt_frac"
    ]].max(axis=1)

    def classify(row: pd.Series) -> str:
        tcp = float(row.transport_tcp_fraction) if pd.notna(row.transport_tcp_fraction) else 0.0
        udp = float(row.transport_udp443_fraction) if pd.notna(row.transport_udp443_fraction) else 0.0
        if tcp >= 0.80 and tcp >= udp + 0.10:
            return "tls_tcp"
        if udp >= 0.80 and udp >= tcp + 0.10:
            return "quic"
        if tcp > 0 or udp > 0:
            return "mixed"
        return "unknown"

    aggregate["transport_dominance_feature"] = aggregate.apply(classify, axis=1)
    return aggregate[[
        "canonical_dataset_key", "transport_dominance_feature",
        "transport_tcp_fraction", "transport_udp443_fraction",
    ]]


def _positive_supergroups(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    dsu = DisjointSet()
    phishing = frame[frame["label"].eq("phishing")]
    for row in phishing.itertuples(index=False):
        values = []
        for column, prefix in (
            ("kit_family_id", "kit"), ("campaign_id", "campaign"),
            ("backend_cluster_id", "backend"),
        ):
            value = _clean(getattr(row, column))
            if value:
                values.append(f"{prefix}:{value}")
        for value in values[1:]:
            dsu.union(values[0], value)

    groups: list[str] = []
    fallback: list[bool] = []
    for row in frame.itertuples(index=False):
        if row.label != "phishing":
            groups.append("")
            fallback.append(False)
            continue
        values = [
            f"kit:{_clean(row.kit_family_id)}" if _clean(row.kit_family_id) else "",
            f"campaign:{_clean(row.campaign_id)}" if _clean(row.campaign_id) else "",
            f"backend:{_clean(row.backend_cluster_id)}" if _clean(row.backend_cluster_id) else "",
        ]
        values = [value for value in values if value]
        if values:
            root = min(dsu.find(value) for value in values)
            groups.append(_stable_id("positive_component", root))
            fallback.append(False)
        else:
            groups.append(_stable_id("positive_fallback", row.collection_unit_id, row.registrable_domain))
            fallback.append(True)
    return pd.Series(groups, index=frame.index), pd.Series(fallback, index=frame.index)


def _write_audits(frame: pd.DataFrame, out_dir: Path) -> None:
    audit_dir = out_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    overview = (
        frame.groupby(["label", "source", "source_name", "source_variant"], dropna=False)
        .agg(
            sessions=("capture_id", "size"),
            collection_units=("collection_unit_id", "nunique"),
            domains=("registrable_domain", lambda s: s[s.ne("")].nunique()),
            capture_start=("capture_start_utc", "min"),
            capture_end=("capture_end_utc", "max"),
        )
        .reset_index()
    )
    overview.to_csv(audit_dir / "dataset_overview.csv", index=False)

    environment = (
        frame.groupby(["environment_id", "sensor_version", "capture_path_id", "client_stack_id"], dropna=False)
        .agg(
            sessions=("capture_id", "size"),
            labels=("label", lambda s: "|".join(sorted(set(s)))),
            sources=("source", lambda s: "|".join(sorted(set(s)))),
            start=("capture_start_utc", "min"),
            end=("capture_end_utc", "max"),
        )
        .reset_index()
    )
    environment.to_csv(audit_dir / "environment_coverage.csv", index=False)

    wallet_rows = []
    wallet_frame = frame[frame["wallet_family_id"].ne("")]
    for wallet, part in wallet_frame.groupby("wallet_family_id", dropna=False):
        positive = part[part["phishing_type"].eq("drainer")]
        benign = part[part["label"].eq("benign")]
        wallet_rows.append({
            "wallet_family_id": wallet,
            "wallet_client_type": "|".join(sorted(set(part["wallet_client_type"]))),
            "num_drainer_sessions": int(len(positive)),
            "num_drainer_deployments": int(positive["drainer_deployment_id"].replace("", pd.NA).nunique()),
            "num_positive_supergroups": int(positive["positive_supergroup"].replace("", pd.NA).nunique()),
            "num_benign_wallet_sessions": int(len(benign)),
            "num_benign_service_groups": int(benign["benign_service_group_id"].replace("", pd.NA).nunique()),
            "collection_start": part["capture_start_utc"].min(),
            "collection_end": part["capture_end_utc"].max(),
            "environment_ids": "|".join(sorted(set(part["environment_id"]))),
            "client_stack_ids": "|".join(sorted(set(part["client_stack_id"]))),
        })
    pd.DataFrame(wallet_rows).to_csv(audit_dir / "wallet_coverage.csv", index=False)

    benign = frame[frame["label"].eq("benign")]
    (
        benign.groupby(["benign_behavior_family", "source", "source_variant"], dropna=False)
        .agg(sessions=("capture_id", "size"), service_groups=("benign_service_group_id", "nunique"))
        .reset_index()
        .to_csv(audit_dir / "benign_behavior_coverage.csv", index=False)
    )

    hard = frame[frame["source"].eq("blockchain_hard")]
    if len(hard):
        (
            hard.groupby("hard_benign_target_id", dropna=False)
            .agg(
                domain=("registrable_domain", lambda s: "|".join(sorted(set(s)))),
                dapp_category=("dapp_category", lambda s: "|".join(sorted(set(s)))),
                site_scale_tier=("site_scale_tier", lambda s: "|".join(sorted(set(s)))),
                site_scale_evidence=("site_scale_evidence", "first"),
                site_scale_evidence_status=("site_scale_evidence_status", lambda s: "|".join(sorted(set(s)))),
                hosting_type=("hosting_type", lambda s: "|".join(sorted(set(s)))),
                wallets=("wallet_family_id", lambda s: "|".join(sorted(set(x for x in s if x)))),
                interaction_outcomes=("interaction_outcome", lambda s: "|".join(sorted(set(s)))),
                collection_start=("capture_start_utc", "min"),
                collection_end=("capture_end_utc", "max"),
                sessions=("capture_id", "size"),
                collection_units=("collection_unit_id", "nunique"),
            )
            .reset_index()
            .to_csv(audit_dir / "hard_benign_site_coverage.csv", index=False)
        )
    else:
        pd.DataFrame().to_csv(audit_dir / "hard_benign_site_coverage.csv", index=False)

    required = [
        "environment_id", "capture_start_utc", "registrable_domain", "collection_unit_id",
        "automation_policy_id", "transport_dominance",
    ]
    missingness = {
        column: {"missing": int(frame[column].fillna("").eq("").sum()), "total": int(len(frame))}
        for column in required
    }
    missingness["benign_service_group_id_benign"] = {
        "missing": int(benign["benign_service_group_id"].fillna("").eq("").sum()),
        "total": int(len(benign)),
    }
    missingness["positive_supergroup_phishing"] = {
        "missing": int(frame.loc[frame.label.eq("phishing"), "positive_supergroup"].fillna("").eq("").sum()),
        "total": int(frame.label.eq("phishing").sum()),
    }
    (audit_dir / "group_missingness.json").write_text(
        json.dumps(missingness, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def build(args: argparse.Namespace) -> int:
    snapshot_path = Path(args.snapshot_manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot = pd.read_csv(snapshot_path, dtype=str, low_memory=False)
    if snapshot["capture_id"].duplicated().any():
        raise ValueError("snapshot capture_id must be unique")

    records = []
    for index, row in snapshot.iterrows():
        path = Path(_clean(row["sample_path"]))
        meta, metadata_file = _sample_metadata(path, _clean(row["source"]))
        record = row.to_dict()
        record.update(_source_specific_fields(row, meta, metadata_file))
        record["canonical_dataset_key"] = canonical_dataset_key(path)
        records.append(record)
        if (index + 1) % 1000 == 0:
            print(f"[enriched-manifest] metadata {index + 1}/{len(snapshot)}", flush=True)
    frame = pd.DataFrame(records)

    kit_path = Path(args.kit_labels) if args.kit_labels else None
    kit = _load_kit_labels(kit_path)
    frame = frame.merge(kit, on="canonical_dataset_key", how="left", suffixes=("", "_kit"), validate="one_to_one")
    for column in ("kit_family_id", "backend_cluster_id", "kit_evidence_tier", "kit_label_hash"):
        kit_column = f"{column}_kit"
        if kit_column in frame:
            frame[column] = frame[kit_column].fillna(frame.get(column, "")).map(_clean)
            frame.drop(columns=[kit_column], inplace=True)
        elif column not in frame:
            frame[column] = ""
        else:
            frame[column] = frame[column].fillna("").map(_clean)
    kit_fit_column = "kit_fit_eligible_kit" if "kit_fit_eligible_kit" in frame else "kit_fit_eligible"
    if kit_fit_column in frame:
        frame["kit_fit_eligible"] = frame[kit_fit_column].astype(str).str.lower().isin({"1", "true", "yes"})
        if kit_fit_column != "kit_fit_eligible":
            frame.drop(columns=[kit_fit_column], inplace=True)
    else:
        frame["kit_fit_eligible"] = False

    frame["positive_supergroup"], frame["positive_supergroup_is_fallback"] = _positive_supergroups(frame)
    frame["benign_supergroup"] = frame["benign_service_group_id"].where(frame.label.eq("benign"), "")
    frame["wallet_positive_group"] = frame.apply(
        lambda r: _stable_id("wallet_positive", r.wallet_family_id, r.positive_supergroup)
        if r.label == "phishing" and _clean(r.wallet_family_id) else "", axis=1
    )
    frame["wallet_benign_group"] = frame.apply(
        lambda r: _stable_id("wallet_benign", r.wallet_family_id, r.benign_service_group_id)
        if r.label == "benign" and _clean(r.wallet_family_id) else "", axis=1
    )
    feature_path = Path(args.features) if getattr(args, "features", "") else None
    transport = _load_transport_dominance(feature_path)
    frame = frame.merge(transport, on="canonical_dataset_key", how="left", validate="one_to_one")
    if feature_path is not None:
        frame["transport_dominance"] = frame["transport_dominance_feature"].fillna("unknown")
        frame["transport_dominance_source"] = frame["transport_dominance_feature"].map(
            lambda value: "frozen_layer1_protocol_fraction_median" if pd.notna(value)
            else "missing_structured_extraction"
        )
    else:
        frame["transport_dominance_source"] = "pending_feature_join"
    frame.drop(columns=["transport_dominance_feature"], errors="ignore", inplace=True)
    frame.drop(columns=["canonical_dataset_key"], inplace=True)

    if frame.loc[frame.label.eq("benign"), "benign_service_group_id"].fillna("").eq("").any():
        raise ValueError("benign_service_group_id is missing for one or more benign samples")
    if frame.loc[frame.label.eq("phishing"), "positive_supergroup"].fillna("").eq("").any():
        raise ValueError("positive_supergroup is missing for one or more phishing samples")
    hard = frame[frame.source.eq("blockchain_hard")]
    if hard["hard_benign_target_id"].fillna("").eq("").any():
        raise ValueError("hard_benign_target_id is missing for hard-benign samples")

    parquet_path = out_dir / "enriched_manifest.parquet"
    csv_path = out_dir / "enriched_manifest.csv"
    frame.to_parquet(parquet_path, index=False, compression="zstd")
    frame.to_csv(csv_path, index=False)
    frame.to_csv(out_dir / "eligible_main_manifest.csv", index=False)
    _write_audits(frame, out_dir)

    kit_manifest_columns = [
        "capture_id", "sample_path", "kit_family_id", "backend_cluster_id",
        "kit_evidence_tier", "kit_label_hash", "kit_fit_eligible",
        "positive_supergroup", "positive_supergroup_is_fallback",
    ]
    kit_manifest = frame.loc[frame.label.eq("phishing"), kit_manifest_columns].copy()
    kit_manifest["label_source"] = str(kit_path.resolve()) if kit_path else "missing"
    kit_manifest["label_hash"] = kit_manifest.apply(
        lambda r: _clean(r.kit_label_hash) or _sha256_text(
            f"{r.capture_id}\x1f{r.kit_family_id}\x1f{r.backend_cluster_id}\x1f{r.kit_evidence_tier}"
        ), axis=1
    )
    kit_manifest["fit_eligible"] = kit_manifest["kit_fit_eligible"] & kit_manifest["kit_family_id"].ne("")
    kit_manifest.to_csv(out_dir / "kit_label_manifest.csv", index=False)

    summary = {
        "snapshot_manifest": str(snapshot_path.resolve()),
        "snapshot_manifest_sha256": sha256_file(snapshot_path),
        "enriched_manifest": str(parquet_path.resolve()),
        "enriched_manifest_sha256": sha256_file(parquet_path),
        "sessions": int(len(frame)),
        "labels": frame["label"].value_counts().to_dict(),
        "sources": frame["source"].value_counts().to_dict(),
        "metadata_available": int(frame["metadata_available"].sum()),
        "kit_label_source": str(kit_path.resolve()) if kit_path else None,
        "kit_labeled_phishing": int(frame.loc[frame.label.eq("phishing"), "kit_family_id"].ne("").sum()),
        "phishing_sessions": int(frame.label.eq("phishing").sum()),
        "positive_fallback_sessions": int(frame["positive_supergroup_is_fallback"].sum()),
        "hard_benign_targets": int(hard["hard_benign_target_id"].nunique()),
        "hard_benign_scale_tiers": hard["site_scale_tier"].value_counts().to_dict(),
        "transport_dominance": frame["transport_dominance"].value_counts(dropna=False).to_dict(),
        "transport_feature_source": str(feature_path.resolve()) if feature_path else None,
        "transport_feature_source_sha256": sha256_file(feature_path) if feature_path else None,
    }
    (out_dir / "enriched_manifest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-manifest", required=True)
    parser.add_argument("--kit-labels", default="")
    parser.add_argument("--features", default="",
                        help="Frozen Layer1 feature table used only for label-blind transport dominance.")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    return build(args)


if __name__ == "__main__":
    raise SystemExit(main())

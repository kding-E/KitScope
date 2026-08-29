#!/usr/bin/env python3
"""Distributed dataset preprocessing and shard merge utilities.

This script is intended for the case where raw Web3 capture samples are too
large to keep on the training machine. Run shard commands on machines that have
local raw samples, copy the small outputs back, then run merge commands once on
the training machine.
"""
from __future__ import annotations

import argparse
import copy
import csv
import glob
import hashlib
import json
import pathlib
import shutil
import socket
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence
from urllib.parse import urlparse

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
SCRIPTS_DIR = ROOT / "scripts"
EXPORTER_DIR = ROOT / "evolution" / "layer2_family_classifier"
for _path in (SRC_DIR, SCRIPTS_DIR, EXPORTER_DIR):
    text = str(_path)
    if text not in sys.path:
        sys.path.insert(0, text)

from web3pcapdetector.config import load_config  # noqa: E402
from web3pcapdetector.pipeline import extract_features_from_zip  # noqa: E402
from web3pcapdetector.roles import host_from_url, normalize_host  # noqa: E402


SCENARIO_CONFIGS: Dict[str, str] = {
    "default": "configs/default.yaml",
    "sni_only": "configs/sni_only.yaml",
    "gateway_sni_dns_unknown": "configs/gateway_sni_dns_unknown.yaml",
    "full_sni": "configs/gateway_ech_full_sni.yaml",
    "no_sni": "configs/gateway_ech_no_sni.yaml",
    "no_dns": "configs/gateway_ech_no_dns.yaml",
    "no_sni_no_dns": "configs/gateway_ech_no_sni_no_dns.yaml",
    "candidate_jitter": "configs/realtime_gateway_oracle_plus_heuristic_jitter.yaml",
    "candidate_oracle_plus_heuristic": "configs/realtime_gateway_oracle_plus_heuristic.yaml",
    "near_realtime": "configs/near_realtime_gateway.yaml",
}

SCENARIO_SETS: Dict[str, List[str]] = {
    "base": ["default", "sni_only", "gateway_sni_dns_unknown"],
    "gateway": ["gateway_sni_dns_unknown"],
    "ech_edns_ablation": ["full_sni", "no_sni", "no_dns", "no_sni_no_dns"],
    "candidate_training": ["candidate_jitter"],
    "near_realtime": ["near_realtime"],
    "all_core": [
        "gateway_sni_dns_unknown",
        "full_sni",
        "no_sni",
        "no_dns",
        "no_sni_no_dns",
        "candidate_jitter",
        "near_realtime",
    ],
}

TEXT_EXTS = {".html", ".json", ".jsonl", ".log", ".txt", ".har", ".js"}
STATIC_SKIP_DIRS = {"screenshots"}
MAX_STATIC_FILE_BYTES = 8_000_000

FEATURE_ROW_KEY_COLUMNS = [
    "preprocess_config_id",
    "sample_key",
    "sample_id",
    "label",
    "domain",
    "anchor_mode",
    "anchor_source",
    "candidate_rank",
    "candidate_source",
    "window_name",
    "anchor_time_epoch",
    "feature_anchor_epoch",
]


@dataclass
class ScenarioSpec:
    scenario_id: str
    config_path: Optional[pathlib.Path]
    config: dict
    generated_from: str = ""


def _safe_id(value: str) -> str:
    out = []
    for ch in str(value).strip():
        if ch.isalnum() or ch in {"_", "-", "."}:
            out.append(ch)
        else:
            out.append("_")
    text = "".join(out).strip("._-")
    if not text:
        raise ValueError(f"invalid empty id from {value!r}")
    return text


def _resolve_project_path(path: str | pathlib.Path) -> pathlib.Path:
    p = pathlib.Path(path)
    if p.is_absolute():
        return p
    return ROOT / p


def _json_sha1(obj: object) -> str:
    text = json.dumps(obj, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _read_json(path: pathlib.Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def _find_zip_member(names: Sequence[str], suffix: str) -> Optional[str]:
    matches = [name for name in names if name.replace("\\", "/").endswith(suffix)]
    if not matches:
        return None
    return sorted(matches, key=len)[0]


def _read_session_from_sample(path: pathlib.Path) -> dict:
    if path.is_file() and path.suffix.lower() == ".json":
        obj = _read_json(path)
        schema = str(obj.get("schema_version") or "")
        if "pcap_path" in obj or schema.startswith("kitscope-public-sample"):
            return obj
    if path.is_dir():
        for name in ("session.json", "sample.json", "sample_private.json"):
            p = path / name
            if p.exists():
                obj = _read_json(p)
                if name == "session.json":
                    return obj
                return _sample_json_to_session(obj, path)
        raise FileNotFoundError(f"{path}: missing session.json/sample.json")
    with zipfile.ZipFile(path) as zf:
        member = (
            _find_zip_member(zf.namelist(), "session.json")
            or _find_zip_member(zf.namelist(), "sample.json")
            or _find_zip_member(zf.namelist(), "sample_private.json")
        )
        if not member:
            raise FileNotFoundError(f"{path}: missing session.json/sample.json")
        with zf.open(member) as f:
            obj = json.loads(f.read().decode("utf-8-sig", errors="ignore"))
            if pathlib.PurePosixPath(member.replace("\\", "/")).name == "session.json":
                return obj
            return _sample_json_to_session(obj, path)


def _sample_json_to_session(sample: dict, path: pathlib.Path) -> dict:
    source = sample.get("source_row", {}) or {}
    host = (
        source.get("openphish_host")
        or source.get("host")
        or source.get("hostname")
        or sample.get("domain")
        or sample.get("host")
        or ""
    )
    url = (
        sample.get("url")
        or sample.get("target_url")
        or source.get("url")
        or source.get("openphish_url")
        or source.get("original_url")
        or (f"https://{host}" if host else "")
    )
    return {
        "session_id": sample.get("session_id") or sample.get("sample_id") or path.stem,
        "domain": normalize_host(host or host_from_url(url) or path.stem),
        "sample": {
            "sample_id": sample.get("sample_id") or sample.get("id") or path.stem,
            "label": sample.get("label") or ("phishing" if "phish" in str(path).lower() else "unknown"),
            "url": url,
            "notes": source.get("notes") or "",
        },
        "status": sample.get("status") or sample.get("result") or "",
        "flags": sample.get("flags", {}) or {},
    }


def _pcap_size_from_sample(path: pathlib.Path) -> int:
    if path.is_file() and path.suffix.lower() == ".json":
        try:
            obj = _read_json(path)
            schema = str(obj.get("schema_version") or "")
            if "pcap_path" in obj or schema.startswith("kitscope-public-sample"):
                capture_id = str(obj.get("capture_id") or path.stem)
                rel = str(obj.get("pcap_path") or f"pcap/{capture_id}.pcap")
                pcap_path = pathlib.Path(rel)
                if not pcap_path.is_absolute():
                    root = path.parent.parent if path.parent.name.lower() == "json" else path.parent
                    pcap_path = root / pcap_path
                return int(pcap_path.stat().st_size) if pcap_path.exists() else 0
        except Exception:
            return 0
    if path.is_dir():
        for name in ("capture.pcap", "capture.pcapng", "traffic.pcapng", "traffic.pcap"):
            p = path / name
            if p.exists():
                return int(p.stat().st_size)
        return 0
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            member = (
                _find_zip_member(names, "capture.pcap")
                or _find_zip_member(names, "capture.pcapng")
                or _find_zip_member(names, "traffic.pcapng")
                or _find_zip_member(names, "traffic.pcap")
            )
            return int(zf.getinfo(member).file_size) if member else 0
    except Exception:
        return 0


def _infer_label(path: pathlib.Path, session: dict) -> str:
    sample = session.get("sample", {}) or {}
    label = sample.get("label")
    if label:
        return str(label)
    name = path.name.lower()
    if name.startswith("phishing") or "phish" in name:
        return "phishing"
    if name.startswith("benign"):
        return "benign"
    return "unknown"


def sample_identity(path: str | pathlib.Path) -> dict:
    p = pathlib.Path(path)
    try:
        session = _read_session_from_sample(p)
    except Exception:
        session = {}
    sample = session.get("sample", {}) or {}
    url = str(sample.get("url") or "")
    source_folder = (
        session.get("source_folder")
        or session.get("original_folder")
        or session.get("original_folder_name")
        or session.get("capture_folder")
        or (session.get("source_row", {}) or {}).get("source_folder")
        or ""
    )
    source_stem = pathlib.Path(str(source_folder)).name if source_folder else ""
    sample_id = str(sample.get("sample_id") or session.get("session_id") or source_stem or p.stem)
    label = _infer_label(p, session)
    domain = normalize_host(session.get("domain") or host_from_url(url) or source_stem or p.stem)
    event_times = session.get("event_times", {}) or {}
    material = {
        "sample_id": sample_id,
        "session_id": str(session.get("session_id") or ""),
        "label": label,
        "domain": domain,
        "url": url,
        "pcap_size": _pcap_size_from_sample(p),
        "pcap_start": str(event_times.get("t_pcap_start") or ""),
        "pcap_stop": str(event_times.get("t_pcap_stop") or ""),
    }
    sample_key = hashlib.sha1(
        json.dumps(material, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()[:24]
    return {
        "sample_key": sample_key,
        "sample_id": sample_id,
        "label": label,
        "domain": domain,
        "url": url,
        "source_pcap_size_bytes": int(material["pcap_size"]),
    }


def _expand_paths(patterns: str | Sequence[str]) -> List[str]:
    raw = [patterns] if isinstance(patterns, str) else list(patterns)
    paths: List[str] = []
    for pattern in raw:
        manifest = pathlib.Path(str(pattern))
        if manifest.exists() and manifest.suffix.lower() in {".csv", ".txt"}:
            if manifest.suffix.lower() == ".csv":
                df = pd.read_csv(manifest, low_memory=False)
                path_col = next(
                    (col for col in ("zip_path", "sample_path", "source_sample_path", "path") if col in df.columns),
                    None,
                )
                if path_col is None:
                    raise ValueError(
                        f"{manifest}: CSV manifest must contain zip_path, sample_path, "
                        "source_sample_path, or path"
                    )
                paths.extend(df[path_col].dropna().astype(str).tolist())
            else:
                paths.extend(
                    line.strip()
                    for line in manifest.read_text(encoding="utf-8").splitlines()
                    if line.strip() and not line.lstrip().startswith("#")
                )
            continue
        matches = glob.glob(str(pattern))
        if not matches and pathlib.Path(str(pattern)).exists():
            matches = [str(pattern)]
        paths.extend(matches)
    return sorted(dict.fromkeys(paths))


def _expand_static_records(patterns: str | Sequence[str]) -> List[Dict[str, str]]:
    raw = [patterns] if isinstance(patterns, str) else list(patterns)
    records: List[Dict[str, str]] = []
    for pattern in raw:
        manifest = pathlib.Path(str(pattern))
        if manifest.exists() and manifest.suffix.lower() in {".csv", ".txt"}:
            if manifest.suffix.lower() == ".csv":
                df = pd.read_csv(manifest, low_memory=False, dtype=str, keep_default_na=False).fillna("")
                path_col = next(
                    (
                        col
                        for col in (
                            "static_sample_path",
                            "sample_path",
                            "source_sample_path",
                            "zip_path",
                            "path",
                        )
                        if col in df.columns
                    ),
                    None,
                )
                if path_col is None:
                    raise ValueError(
                        f"{manifest}: CSV manifest must contain static_sample_path, sample_path, "
                        "source_sample_path, zip_path, or path"
                    )
                for _, row in df.iterrows():
                    sample_path = str(row.get(path_col, "")).strip()
                    if not sample_path:
                        continue
                    record = {str(col): str(row.get(col, "")) for col in df.columns}
                    record["sample_path"] = sample_path
                    record.setdefault("source_sample_path", sample_path)
                    records.append(record)
            else:
                records.extend(
                    {"sample_path": line.strip(), "source_sample_path": line.strip()}
                    for line in manifest.read_text(encoding="utf-8").splitlines()
                    if line.strip() and not line.lstrip().startswith("#")
                )
            continue
        matches = glob.glob(str(pattern))
        if not matches and pathlib.Path(str(pattern)).exists():
            matches = [str(pattern)]
        records.extend({"sample_path": str(path), "source_sample_path": str(path)} for path in matches)

    deduped: List[Dict[str, str]] = []
    seen: set[str] = set()
    for record in records:
        key = str(record.get("sample_path", "")).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def _infer_release_root_from_static_sample(sample_path: pathlib.Path) -> Optional[pathlib.Path]:
    if sample_path.parent.name.lower() == "static_features":
        return sample_path.parent.parent
    if sample_path.parent.name.lower() in {"json", "har", "pcap"}:
        return sample_path.parent.parent
    return None


def _record_capture_id(record: Dict[str, str], sample_path: pathlib.Path) -> str:
    capture_id = str(record.get("capture_id") or "").strip()
    if capture_id:
        return capture_id
    if sample_path.parent.name.lower() == "static_features":
        return sample_path.name
    return sample_path.stem


def _resolve_release_record_path(
    record: Dict[str, str],
    key: str,
    sample_path: pathlib.Path,
    folder: str,
    suffix: str,
) -> Optional[pathlib.Path]:
    capture_id = _record_capture_id(record, sample_path)
    release_root = _infer_release_root_from_static_sample(sample_path)
    raw = str(record.get(key) or "").strip()
    if raw:
        path = pathlib.Path(raw)
        if path.is_absolute():
            return path
        if release_root is not None:
            return release_root / path
        return sample_path / path
    if release_root is not None and capture_id:
        path = release_root / folder / f"{capture_id}.{suffix}"
        if path.exists():
            return path
    return None


def _static_record_context(record: Dict[str, str]) -> Dict[str, object]:
    sample_path = pathlib.Path(str(record.get("sample_path", "")))
    capture_id = _record_capture_id(record, sample_path)
    json_path = _resolve_release_record_path(record, "json_path", sample_path, "json", "json")
    har_path = _resolve_release_record_path(record, "har_path", sample_path, "har", "har")
    return {
        "capture_id": capture_id,
        "session_path": json_path if json_path is not None and json_path.exists() else None,
        "har_path": har_path if har_path is not None and har_path.exists() else None,
    }


def _identity_from_static_record(record: Dict[str, str], context: Dict[str, object]) -> dict:
    sample_path = pathlib.Path(str(record.get("sample_path", "")))
    session_path = context.get("session_path")
    ident = sample_identity(session_path if isinstance(session_path, pathlib.Path) else sample_path)
    capture_id = str(record.get("capture_id") or context.get("capture_id") or "").strip()
    if capture_id:
        ident["capture_id"] = capture_id
    for column in (
        "sample_id",
        "label",
        "domain",
        "url",
        "source",
        "source_variant",
        "source_folder",
        "fine_source",
        "partition",
        "fit_role",
    ):
        value = str(record.get(column) or "").strip()
        if value:
            ident[column] = value
    return ident


def _manifest_sha256(path: str | None) -> str | None:
    if not path:
        return None
    manifest = pathlib.Path(path)
    if not manifest.is_file():
        return None
    digest = hashlib.sha256()
    with manifest.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_name_path(value: str) -> tuple[str, pathlib.Path]:
    if "=" in value:
        name, path = value.split("=", 1)
        return _safe_id(name), _resolve_project_path(path)
    path = _resolve_project_path(value)
    return _safe_id(path.stem), path


def _visibility_cfg(base_cfg: dict, sni_visible_rate: float, dns_visible_rate: float) -> dict:
    cfg = copy.deepcopy(base_cfg)
    filtering = cfg.setdefault("filtering", {})
    degradation = dict(filtering.get("degradation") or {})
    degradation["hide_sni"] = bool(sni_visible_rate < 1.0)
    degradation["hide_dns"] = bool(dns_visible_rate < 1.0)
    degradation["sni_visible_rate"] = float(sni_visible_rate)
    degradation["dns_visible_rate"] = float(dns_visible_rate)
    degradation["seed"] = f"distributed_sweep_sni_{sni_visible_rate:.2f}_dns_{dns_visible_rate:.2f}"
    filtering["degradation"] = degradation
    return cfg


def _scenario_specs(args: argparse.Namespace) -> List[ScenarioSpec]:
    specs: List[ScenarioSpec] = []
    seen: set[str] = set()

    def add_config(scenario_id: str, config_path: pathlib.Path, generated_from: str = "") -> None:
        sid = _safe_id(scenario_id)
        if sid in seen:
            return
        seen.add(sid)
        specs.append(ScenarioSpec(sid, config_path, load_config(str(config_path)), generated_from))

    for set_name in args.scenario_set or []:
        if set_name == "none":
            continue
        if set_name == "ech_edns_sweep":
            base_path = _resolve_project_path(args.base_config)
            base_cfg = load_config(str(base_path))
            rates = sorted({float(max(0.0, min(1.0, r))) for r in args.sweep_rates})
            if 1.0 not in rates:
                rates.append(1.0)
                rates.sort()
            cells: list[tuple[str, float, float]] = []
            for rate in rates:
                cells.append(("sni", rate, 1.0))
            for rate in rates:
                if rate != 1.0:
                    cells.append(("dns", 1.0, rate))
            for axis, sni_rate, dns_rate in cells:
                sid = f"{axis}_{sni_rate:.2f}_{dns_rate:.2f}"
                if sid in seen:
                    continue
                seen.add(sid)
                specs.append(
                    ScenarioSpec(
                        sid,
                        base_path,
                        _visibility_cfg(base_cfg, sni_rate, dns_rate),
                        generated_from=f"{base_path};sni={sni_rate:.2f};dns={dns_rate:.2f}",
                    )
                )
            continue

        for scenario_id in SCENARIO_SETS[set_name]:
            add_config(scenario_id, _resolve_project_path(SCENARIO_CONFIGS[scenario_id]))

    for item in args.config or []:
        scenario_id, config_path = _parse_name_path(item)
        add_config(scenario_id, config_path)

    if not specs:
        raise ValueError("no scenario selected; use --scenario-set or --config")
    return specs


def _parse_float_list(value: str) -> list[float]:
    return [
        float(item)
        for item in str(value or "").replace(",", " ").split()
        if item.strip()
    ]


def _apply_feature_extraction_overrides(
    specs: Sequence[ScenarioSpec],
    args: argparse.Namespace,
) -> List[ScenarioSpec]:
    out: List[ScenarioSpec] = []
    for scenario in specs:
        cfg = copy.deepcopy(scenario.config)
        generated_from = scenario.generated_from

        if args.causal_all_onsets:
            anchor = cfg.setdefault("anchor", {})
            anchor["mode"] = "heuristic"
            anchor["post_load_guard_mode"] = "none"
            candidate = cfg.setdefault("candidate", {})
            candidate["causal_all_local_maxima"] = True
            candidate["fallback_if_empty"] = False
            candidate["min_after_capture_start_s"] = 0.0
            candidate["min_after_first_party_s"] = 0.0
            candidate["min_after_post_load_s"] = 0.0
            candidate["max_scan_s"] = None
            candidate.pop("skip_first_party_scan_guard", None)
            if args.candidate_decision_delays:
                candidate["decision_delays_s"] = _parse_float_list(args.candidate_decision_delays)
            if args.candidate_scan_step is not None:
                candidate["scan_step_s"] = float(args.candidate_scan_step)
            suffix = (
                "causal_all_onsets=true;"
                f"candidate_decision_delays={args.candidate_decision_delays or ''};"
                f"candidate_scan_step={args.candidate_scan_step if args.candidate_scan_step is not None else ''}"
            )
            generated_from = f"{generated_from};{suffix}" if generated_from else suffix

        expected_guard = str(args.expected_post_load_guard_mode or "").strip().lower()
        if expected_guard:
            anchor_cfg = cfg.get("anchor", {}) if isinstance(cfg.get("anchor", {}), dict) else {}
            actual_guard = str(anchor_cfg.get("post_load_guard_mode", "instrumentation") or "instrumentation").strip().lower()
            if actual_guard != expected_guard:
                raise ValueError(
                    f"{scenario.scenario_id}: expected post_load_guard_mode={expected_guard!r}, "
                    f"got {actual_guard!r}"
                )

        out.append(
            ScenarioSpec(
                scenario.scenario_id,
                scenario.config_path,
                cfg,
                generated_from=generated_from,
            )
        )
    return out


def _ensure_sample_key(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "sample_key" not in out.columns:
        parts = []
        for col in ("sample_id", "label", "domain", "url"):
            if col in out.columns:
                parts.append(out[col].astype(str))
        if parts:
            material = parts[0]
            for part in parts[1:]:
                material = material + "|" + part
            out["sample_key"] = material.map(lambda x: hashlib.sha1(x.encode("utf-8", errors="ignore")).hexdigest()[:24])
        else:
            out["sample_key"] = ""
    return out


def _dedupe_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = _ensure_sample_key(df)
    subset = [c for c in FEATURE_ROW_KEY_COLUMNS if c in out.columns]
    if not subset:
        subset = list(out.columns)
    out = out.drop_duplicates(subset=subset, keep="first").copy()
    sort_cols = [c for c in ["preprocess_config_id", "label", "sample_id", "domain", "sample_key", "candidate_rank", "window_name"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, kind="mergesort")
    return out.reset_index(drop=True)


def _read_existing_features(path: pathlib.Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _feature_sample_keys(path: pathlib.Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        existing = pd.read_csv(path, usecols=lambda c: c == "sample_key")
    except Exception:
        return set()
    if "sample_key" not in existing.columns:
        return set()
    return set(existing["sample_key"].astype(str))


def _attach_preprocess_columns(
    df: pd.DataFrame,
    ident: dict,
    sample_path: str,
    scenario: ScenarioSpec,
    config_sha1: str,
    shard_id: str,
    machine: str,
) -> pd.DataFrame:
    out = df.copy()
    out["sample_key"] = ident.get("sample_key", "")
    out["preprocess_config_id"] = scenario.scenario_id
    out["preprocess_config_sha1"] = config_sha1
    out["preprocess_shard_id"] = shard_id
    out["preprocess_machine"] = machine
    out["source_sample_path"] = str(sample_path)
    out["source_pcap_size_bytes"] = int(ident.get("source_pcap_size_bytes") or 0)
    return out


def _write_json(path: pathlib.Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def cmd_feature_shard(args: argparse.Namespace) -> None:
    specs = _apply_feature_extraction_overrides(_scenario_specs(args), args)
    paths = _expand_paths(args.input_glob)
    if args.limit:
        paths = paths[: int(args.limit)]
    if not paths:
        raise FileNotFoundError(f"no samples matched {args.input_glob}")

    shard_id = _safe_id(args.shard_id or socket.gethostname())
    machine = socket.gethostname()
    out_root = pathlib.Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for scenario in specs:
        scenario_dir = out_root / shard_id / scenario.scenario_id
        scenario_dir.mkdir(parents=True, exist_ok=True)
        features_csv = scenario_dir / "features.csv"
        errors: List[dict] = []
        manifest_rows: List[dict] = []
        existing = _read_existing_features(features_csv) if args.resume else pd.DataFrame()
        done_keys = _feature_sample_keys(features_csv) if args.resume else set()
        dfs: List[pd.DataFrame] = [existing] if not existing.empty else []
        config_sha1 = _json_sha1(scenario.config)

        print(f"[feature-shard] scenario={scenario.scenario_id} paths={len(paths)} out={features_csv}")
        for index, sample_path in enumerate(paths, start=1):
            ident = sample_identity(sample_path)
            if ident["sample_key"] in done_keys:
                continue
            try:
                print(f"[feature-shard] {scenario.scenario_id} {index}/{len(paths)} {sample_path}")
                part = extract_features_from_zip(sample_path, scenario.config)
                part = _attach_preprocess_columns(
                    part,
                    ident=ident,
                    sample_path=sample_path,
                    scenario=scenario,
                    config_sha1=config_sha1,
                    shard_id=shard_id,
                    machine=machine,
                )
                dfs.append(part)
                done_keys.add(str(ident["sample_key"]))
                manifest_rows.append({
                    **ident,
                    "source_sample_path": str(sample_path),
                    "preprocess_config_id": scenario.scenario_id,
                    "preprocess_config_sha1": config_sha1,
                    "preprocess_shard_id": shard_id,
                    "status": "ok",
                    "rows": int(len(part)),
                })
            except Exception as exc:  # noqa: BLE001
                row = {
                    **ident,
                    "source_sample_path": str(sample_path),
                    "preprocess_config_id": scenario.scenario_id,
                    "preprocess_config_sha1": config_sha1,
                    "preprocess_shard_id": shard_id,
                    "status": "error",
                    "error": str(exc),
                    "type": type(exc).__name__,
                }
                errors.append(row)
                manifest_rows.append(row)
                if not args.skip_errors:
                    raise
                print(f"[feature-shard] skip {sample_path}: {exc}")

        if not dfs:
            raise RuntimeError(f"no features extracted for scenario {scenario.scenario_id}")
        merged = _dedupe_features(pd.concat(dfs, ignore_index=True))
        merged.to_csv(features_csv, index=False, quoting=csv.QUOTE_MINIMAL)
        pd.DataFrame(manifest_rows).to_csv(scenario_dir / "sample_manifest.csv", index=False, quoting=csv.QUOTE_MINIMAL)
        if errors:
            pd.DataFrame(errors).to_csv(scenario_dir / "extract_errors.csv", index=False, quoting=csv.QUOTE_MINIMAL)
            _write_json(scenario_dir / "extract_errors.json", errors)
        _write_json(
            scenario_dir / "shard_metadata.json",
            {
                "kind": "feature_shard",
                "scenario_id": scenario.scenario_id,
                "config_path": str(scenario.config_path) if scenario.config_path else "",
                "generated_from": scenario.generated_from,
                "config_sha1": config_sha1,
                "shard_id": shard_id,
                "machine": machine,
                "n_input_paths": len(paths),
                "n_rows": int(len(merged)),
                "n_unique_samples": int(merged["sample_key"].nunique()) if "sample_key" in merged.columns else None,
                "causal_all_onsets": bool(args.causal_all_onsets),
                "candidate_decision_delays": str(args.candidate_decision_delays or ""),
                "candidate_scan_step": args.candidate_scan_step,
                "expected_post_load_guard_mode": str(args.expected_post_load_guard_mode or ""),
                "effective_anchor_mode": scenario.config.get("anchor", {}).get("mode", ""),
                "effective_post_load_guard_mode": scenario.config.get("anchor", {}).get("post_load_guard_mode", ""),
                "effective_causal_all_local_maxima": bool(scenario.config.get("candidate", {}).get("causal_all_local_maxima", False)),
                "effective_fallback_if_empty": bool(scenario.config.get("candidate", {}).get("fallback_if_empty", False)),
                "effective_skip_first_party_scan_guard": bool(scenario.config.get("candidate", {}).get("skip_first_party_scan_guard", False)),
            },
        )
        print(f"[feature-shard] wrote {features_csv} rows={len(merged)}")


def _discover_files(roots: Sequence[str] | None, globs: Sequence[str] | None, filename: str) -> List[pathlib.Path]:
    files: List[pathlib.Path] = []
    for root in roots or []:
        p = pathlib.Path(root)
        if p.is_file() and p.name == filename:
            files.append(p)
        elif p.exists():
            files.extend(p.rglob(filename))
    for pattern in globs or []:
        files.extend(pathlib.Path(p) for p in glob.glob(pattern, recursive=True))
    return sorted(dict.fromkeys(p.resolve() for p in files if p.exists()))


def _infer_config_id_from_feature_path(path: pathlib.Path) -> str:
    if path.parent.name:
        return _safe_id(path.parent.name)
    return "features"


def cmd_merge_features(args: argparse.Namespace) -> None:
    out_root = pathlib.Path(args.out_root)
    files = _discover_files(args.shard_root, args.shard_glob, "features.csv")
    if args.include_existing and out_root.exists():
        files.extend(out_root.rglob("features.csv"))
        files = sorted(dict.fromkeys(p.resolve() for p in files if p.exists()))
    if not files:
        raise FileNotFoundError("no feature shard CSVs found")

    by_config: Dict[str, List[pd.DataFrame]] = {}
    source_rows = []
    for file in files:
        df = pd.read_csv(file, low_memory=False)
        if df.empty:
            continue
        if "preprocess_config_id" not in df.columns:
            df["preprocess_config_id"] = _infer_config_id_from_feature_path(file)
        for config_id, part in df.groupby(df["preprocess_config_id"].astype(str), sort=True):
            by_config.setdefault(_safe_id(config_id), []).append(part.copy())
        source_rows.append({
            "path": str(file),
            "rows": int(len(df)),
            "configs": ",".join(sorted(df["preprocess_config_id"].astype(str).unique())),
        })

    summary_rows = []
    for config_id, parts in sorted(by_config.items()):
        merged = _dedupe_features(pd.concat(parts, ignore_index=True))
        out_dir = out_root / config_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / "features.csv"
        out_parquet = out_dir / "features.parquet"
        merged.to_csv(out_csv, index=False, quoting=csv.QUOTE_MINIMAL)
        try:
            merged.to_parquet(out_parquet, index=False)
        except ImportError as exc:
            raise RuntimeError(
                "merge-features requires pyarrow or fastparquet to write the "
                "parquet table consumed by the formal random-route runner; "
                "install the reproduce extras with pip install -e .[reproduce]"
            ) from exc
        summary_rows.append({
            "preprocess_config_id": config_id,
            "features_csv": str(out_csv),
            "features_parquet": str(out_parquet),
            "rows": int(len(merged)),
            "unique_samples": int(merged["sample_key"].nunique()) if "sample_key" in merged.columns else None,
            "unique_sample_ids": int(merged["sample_id"].nunique()) if "sample_id" in merged.columns else None,
        })
        print(f"[merge-features] wrote {out_csv} and {out_parquet} rows={len(merged)}")

    pd.DataFrame(summary_rows).to_csv(out_root / "distributed_features_summary.csv", index=False)
    pd.DataFrame(source_rows).to_csv(out_root / "distributed_feature_sources.csv", index=False)


def _zip_member_allowed_for_static(name: str, size: int, include_har: bool) -> bool:
    clean = name.replace("\\", "/")
    parts = [p for p in clean.split("/") if p]
    if not parts or any(part in {"..", "."} for part in parts):
        return False
    if any(part.lower() in STATIC_SKIP_DIRS for part in parts):
        return False
    suffix = pathlib.PurePosixPath(clean).suffix.lower()
    if suffix == ".har" and not include_har:
        return False
    if suffix not in TEXT_EXTS:
        return False
    return int(size) <= MAX_STATIC_FILE_BYTES


def _extract_static_zip_to_temp(path: pathlib.Path, include_har: bool) -> tuple[tempfile.TemporaryDirectory, pathlib.Path]:
    tmp = tempfile.TemporaryDirectory(prefix="web3_static_")
    tmp_root = pathlib.Path(tmp.name).resolve()
    session_candidates: List[pathlib.Path] = []
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            if not _zip_member_allowed_for_static(info.filename, info.file_size, include_har):
                continue
            rel = pathlib.PurePosixPath(info.filename)
            dest = (tmp_root / pathlib.Path(*rel.parts)).resolve()
            if not str(dest).startswith(str(tmp_root)):
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, dest.open("wb") as out:
                shutil.copyfileobj(src, out)
            if dest.name in {"session.json", "sample.json", "sample_private.json"}:
                session_candidates.append(dest)
    if not session_candidates:
        tmp.cleanup()
        raise FileNotFoundError(f"{path}: missing session.json/sample.json in static zip extraction")
    sample_root = sorted(session_candidates, key=lambda p: len(str(p)))[0].parent
    return tmp, sample_root


def _static_sample_root(path: pathlib.Path, include_har: bool) -> tuple[Optional[tempfile.TemporaryDirectory], pathlib.Path]:
    if path.is_dir():
        return None, path
    return _extract_static_zip_to_temp(path, include_har)


def _json_list(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value if value is not None else [], ensure_ascii=False)


def cmd_static_shard(args: argparse.Namespace) -> None:
    from audit_cluster_family_evidence import family_features  # noqa: E402
    from known_kit_attribution_engine import sample_candidate_keys  # noqa: E402

    if args.input_list and args.input_glob:
        raise ValueError("use exactly one of --input-list or --input-glob")
    input_spec = [args.input_list] if args.input_list else args.input_glob
    records = _expand_static_records(input_spec)
    if args.limit:
        records = records[: int(args.limit)]
    if not records:
        raise FileNotFoundError(f"no samples matched {input_spec}")

    include_har = not bool(args.exclude_har)
    wanted_labels = {str(x).lower() for x in (args.labels or [])}
    shard_id = _safe_id(args.shard_id or socket.gethostname())
    out_dir = pathlib.Path(args.out_dir) / shard_id / ("static_with_har" if include_har else "static_no_har")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "static_family_features.csv"
    try:
        existing = pd.read_csv(out_csv, low_memory=False) if args.resume and out_csv.exists() else pd.DataFrame()
    except pd.errors.EmptyDataError:
        existing = pd.DataFrame()
    done_keys = set()
    if not existing.empty and "sample_key" in existing.columns:
        done_keys = set(existing["sample_key"].astype(str))

    rows = existing.to_dict("records") if not existing.empty else []
    errors = []
    print(f"[static-shard] include_har={include_har} paths={len(records)} out={out_csv}")

    def _write_static_checkpoint() -> None:
        df = pd.DataFrame(rows)
        if not df.empty:
            subset = ["sample_key", "include_har"] if "sample_key" in df.columns else ["sample_id", "domain", "include_har"]
            df = df.drop_duplicates(subset=subset, keep="first").sort_values(subset, kind="mergesort")
        df.to_csv(out_csv, index=False, quoting=csv.QUOTE_MINIMAL)
        if errors:
            pd.DataFrame(errors).to_csv(out_dir / "static_errors.csv", index=False)
            _write_json(out_dir / "static_errors.json", errors)

    for index, record in enumerate(records, start=1):
        sample_path = str(record.get("sample_path", ""))
        context = _static_record_context(record)
        ident = _identity_from_static_record(record, context)
        if wanted_labels and str(ident.get("label", "")).lower() not in wanted_labels:
            continue
        if ident["sample_key"] in done_keys:
            continue
        tmp = None
        try:
            print(f"[static-shard] {index}/{len(records)} {sample_path}")
            tmp, sample_root = _static_sample_root(pathlib.Path(sample_path), include_har=include_har)
            session_path = context.get("session_path")
            har_path = context.get("har_path")
            meta, features = family_features(
                sample_root,
                include_har=include_har,
                session_path=session_path if isinstance(session_path, pathlib.Path) else None,
                har_path=har_path if isinstance(har_path, pathlib.Path) else None,
            )
            candidates = sample_candidate_keys(meta, features)
            flat = sorted({key for values in candidates.values() for key in values})
            row = {
                **ident,
                "zip_path": str(sample_path),
                "source_sample_path": str(sample_path),
                "metadata_json_path": str(session_path) if isinstance(session_path, pathlib.Path) else "",
                "har_path": str(har_path) if isinstance(har_path, pathlib.Path) else "",
                "preprocess_shard_id": shard_id,
                "preprocess_machine": socket.gethostname(),
                "include_har": int(include_har),
                "n_family_features": int(len(features)),
                "family_features_json": json.dumps(sorted(features), ensure_ascii=False),
                "candidate_family_keys": json.dumps(flat, ensure_ascii=False),
                "title": meta.get("title", ""),
                "status": meta.get("status", ""),
                "body_template_hash": meta.get("body_template_hash", ""),
                "connect_confirmed": bool(meta.get("connect_confirmed", False)),
                "signature_prompt_seen": bool(meta.get("signature_prompt_seen", False)),
                "high_risk_wallet_prompt_seen": bool(meta.get("high_risk_wallet_prompt_seen", False)),
                "methods": _json_list(meta.get("methods", [])),
                "addresses": _json_list(meta.get("addresses", [])),
                "walletconnect_project_ids": _json_list(meta.get("walletconnect_project_ids", [])),
                "walletconnect_verify_ids": _json_list(meta.get("walletconnect_verify_ids", [])),
                "drainer_hits": _json_list(meta.get("drainer_hits", [])),
                "drainer_spenders": _json_list(meta.get("drainer_spenders", [])),
                "script_tokens": _json_list(meta.get("script_tokens", [])),
                "api_endpoints": _json_list(meta.get("api_endpoints", [])),
                "backend_endpoints": _json_list(meta.get("backend_endpoints", [])),
                "backend_observations": _json_list(meta.get("backend_observations", [])),
                "backend_request_schemas": _json_list(meta.get("backend_request_schemas", [])),
                "backend_roles": _json_list(meta.get("backend_roles", [])),
                "backend_role_sequence": _json_list(meta.get("backend_role_sequence", [])),
                "backend_route_set_hashes": _json_list(meta.get("backend_route_set_hashes", [])),
                "backend_request_schema_hashes": _json_list(meta.get("backend_request_schema_hashes", [])),
                "backend_flow_hashes": _json_list(meta.get("backend_flow_hashes", [])),
                "backend_callsite_hashes": _json_list(meta.get("backend_callsite_hashes", [])),
                "backend_kit_hashes": _json_list(meta.get("backend_kit_hashes", [])),
                "claim_words": _json_list(meta.get("claim_words", [])),
                "body_tokens": _json_list(meta.get("body_tokens", [])),
                "infura_project_ids": _json_list(meta.get("infura_project_ids", [])),
                "js_capabilities": _json_list(meta.get("js_capabilities", [])),
                "interaction_pattern": meta.get("interaction_pattern", ""),
                "inline_script_hashes": _json_list(meta.get("inline_script_hashes", [])),
                "script_content_hashes": _json_list(meta.get("script_content_hashes", [])),
                "resource_graph_hashes": _json_list(meta.get("resource_graph_hashes", [])),
                "html_structure_hashes": _json_list(meta.get("html_structure_hashes", [])),
                "image_asset_hashes": _json_list(meta.get("image_asset_hashes", [])),
                "kit_behavior_hashes": _json_list(meta.get("kit_behavior_hashes", [])),
                "rpc_methods": _json_list(meta.get("rpc_methods", [])),
            }
            rows.append(row)
            done_keys.add(str(ident["sample_key"]))
            if len(rows) % 25 == 0:
                _write_static_checkpoint()
        except Exception as exc:  # noqa: BLE001
            error = {
                **ident,
                "source_sample_path": str(sample_path),
                "include_har": int(include_har),
                "error": str(exc),
                "type": type(exc).__name__,
            }
            errors.append(error)
            if not args.skip_errors:
                raise
            print(f"[static-shard] skip {sample_path}: {exc}")
            if len(errors) % 25 == 0:
                _write_static_checkpoint()
        finally:
            if tmp is not None:
                tmp.cleanup()

    _write_static_checkpoint()
    try:
        df = pd.read_csv(out_csv, low_memory=False) if out_csv.exists() else pd.DataFrame()
    except pd.errors.EmptyDataError:
        df = pd.DataFrame()
    _write_json(
        out_dir / "static_shard_metadata.json",
        {
            "kind": "static_family_features",
            "include_har": include_har,
            "shard_id": shard_id,
            "rows": int(len(df)),
            "errors": int(len(errors)),
            "input_mode": "manifest" if args.input_list else "glob",
            "input_manifest": str(pathlib.Path(args.input_list).resolve()) if args.input_list else None,
            "input_manifest_sha256": _manifest_sha256(args.input_list),
            "input_paths": int(len(records)),
        },
    )
    print(f"[static-shard] wrote {out_csv} rows={len(df)}")


def cmd_merge_static(args: argparse.Namespace) -> None:
    files = _discover_files(args.shard_root, args.shard_glob, "static_family_features.csv")
    if not files:
        raise FileNotFoundError("no static_family_features.csv shard files found")
    parts = []
    for file in files:
        df = pd.read_csv(file, low_memory=False)
        if not df.empty:
            parts.append(df)
    if not parts:
        raise RuntimeError("static shards were empty")
    merged = pd.concat(parts, ignore_index=True)
    if "include_har" not in merged.columns:
        merged["include_har"] = 1
    subset = ["sample_key", "include_har"] if "sample_key" in merged.columns else ["sample_id", "domain", "include_har"]
    merged = merged.drop_duplicates(subset=subset, keep="first").sort_values(subset, kind="mergesort")
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False, quoting=csv.QUOTE_MINIMAL)
    summary = pd.DataFrame([{
        "static_family_features": str(out),
        "rows": int(len(merged)),
        "unique_samples": int(merged["sample_key"].nunique()) if "sample_key" in merged.columns else None,
        "include_har_values": ",".join(str(x) for x in sorted(merged["include_har"].astype(int).unique())),
    }])
    summary.to_csv(out.parent / "static_family_features_summary.csv", index=False)
    print(f"[merge-static] wrote {out} rows={len(merged)}")


def cmd_sequence_shard(args: argparse.Namespace) -> None:
    from export_packet_sequences import run as export_packet_sequences_run  # noqa: E402

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    export_packet_sequences_run(args)
    _write_json(
        out_dir / "sequence_shard_metadata.json",
        {
            "kind": "packet_sequence_shard",
            "features": str(args.features),
            "config": str(args.config),
            "shard_id": args.shard_id or socket.gethostname(),
            "machine": socket.gethostname(),
        },
    )


def _discover_sequence_npzs(args: argparse.Namespace) -> List[pathlib.Path]:
    files = _discover_files(args.sequence_root, args.sequence_glob, "packet_sequences.npz")
    return sorted(dict.fromkeys(files))


def _sequence_key(row: pd.Series, fallback_index: int) -> str:
    if "sample_key" in row and pd.notna(row["sample_key"]) and str(row["sample_key"]):
        return str(row["sample_key"])
    parts = []
    for col in ("sample_id", "label", "domain"):
        if col in row:
            parts.append(str(row[col]))
    if parts:
        return "|".join(parts)
    return f"row:{fallback_index}"


def cmd_merge_sequences(args: argparse.Namespace) -> None:
    files = _discover_sequence_npzs(args)
    if not files:
        raise FileNotFoundError("no packet_sequences.npz files found")

    cats = []
    conts = []
    masks = []
    manifest_rows = []
    seen: set[str] = set()
    expected_shape = None
    for npz_path in files:
        manifest_path = npz_path.parent / "packet_sequence_manifest.csv"
        if manifest_path.exists():
            manifest = pd.read_csv(manifest_path, low_memory=False)
        else:
            manifest = pd.DataFrame()
        npz = np.load(npz_path, allow_pickle=True)
        cat = np.asarray(npz["cat"], dtype=np.int64)
        cont = np.asarray(npz["cont"], dtype=np.float32)
        mask = np.asarray(npz["mask"], dtype=bool)
        if expected_shape is None:
            expected_shape = (cat.shape[1:], cont.shape[1:], mask.shape[1:])
        elif expected_shape != (cat.shape[1:], cont.shape[1:], mask.shape[1:]):
            raise ValueError(f"{npz_path}: sequence shape differs from earlier shards")
        n = int(cat.shape[0])
        if manifest.empty:
            manifest = pd.DataFrame({
                "sample_id": npz["sample_id"].astype(str) if "sample_id" in npz.files else ["" for _ in range(n)],
                "label": npz["label"].astype(str) if "label" in npz.files else ["" for _ in range(n)],
                "domain": npz["domain"].astype(str) if "domain" in npz.files else ["" for _ in range(n)],
                "zip_path": npz["zip_path"].astype(str) if "zip_path" in npz.files else ["" for _ in range(n)],
            })
        for i in range(n):
            row = manifest.iloc[i].copy() if i < len(manifest) else pd.Series(dtype=object)
            key = _sequence_key(row, fallback_index=len(manifest_rows))
            if key in seen:
                continue
            seen.add(key)
            cats.append(cat[i])
            conts.append(cont[i])
            masks.append(mask[i])
            record = row.to_dict()
            record["seq_index"] = len(manifest_rows)
            record["source_sequence_npz"] = str(npz_path)
            record["sequence_dedup_key"] = key
            manifest_rows.append(record)

    if not manifest_rows:
        raise RuntimeError("no sequences remained after deduplication")
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(out_dir / "packet_sequence_manifest.csv", index=False)
    arrays = {
        "cat": np.stack(cats).astype(np.int64),
        "cont": np.stack(conts).astype(np.float32),
        "mask": np.stack(masks).astype(bool),
        "sample_id": manifest.get("sample_id", pd.Series([""] * len(manifest))).astype(str).values,
        "zip_path": manifest.get("zip_path", pd.Series([""] * len(manifest))).astype(str).values,
        "label": manifest.get("label", pd.Series([""] * len(manifest))).astype(str).values,
        "domain": manifest.get("domain", pd.Series([""] * len(manifest))).astype(str).values,
    }
    if "sample_key" in manifest.columns:
        arrays["sample_key"] = manifest["sample_key"].astype(str).values
    np.savez_compressed(out_dir / "packet_sequences.npz", **arrays)
    _write_json(
        out_dir / "packet_sequence_merge_metadata.json",
        {
            "sources": [str(p) for p in files],
            "n_sequences": int(len(manifest)),
            "dedupe_key": "sample_key or sample_id|label|domain",
        },
    )
    print(f"[merge-sequences] wrote {out_dir / 'packet_sequences.npz'} sequences={len(manifest)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Distributed preprocessing for Web3 PCAP detector datasets.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    scenario_choices = sorted([*SCENARIO_SETS.keys(), "ech_edns_sweep", "none"])

    p = sub.add_parser("feature-shard", help="Extract feature CSV shards on a machine with raw samples.")
    p.add_argument("--input-glob", nargs="+", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--shard-id", default=None)
    p.add_argument("--scenario-set", nargs="+", choices=scenario_choices, default=["gateway"])
    p.add_argument("--config", nargs="*", default=[], help="Extra config specs, either name=path or path.")
    p.add_argument("--base-config", default="configs/gateway_ech_full_sni.yaml")
    p.add_argument("--sweep-rates", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--expected-post-load-guard-mode", default="")
    p.add_argument("--causal-all-onsets", action="store_true",
                   help="Use every causally finalized local onset, matching the KitScope main random route.")
    p.add_argument("--candidate-decision-delays", default="",
                   help="Optional comma-separated causal onset decision delays override.")
    p.add_argument("--candidate-scan-step", type=float, default=None,
                   help="Optional causal onset scan cadence override in seconds.")
    p.add_argument("--skip-errors", action="store_true")
    p.add_argument("--resume", dest="resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(func=cmd_feature_shard)

    p = sub.add_parser("merge-features", help="Merge and deduplicate feature shards into experiment-ready features.csv files.")
    p.add_argument("--shard-root", nargs="*", default=[])
    p.add_argument("--shard-glob", nargs="*", default=[])
    p.add_argument("--out-root", required=True)
    p.add_argument("--include-existing", action="store_true", help="Include existing out-root/*/features.csv for incremental merges.")
    p.set_defaults(func=cmd_merge_features)

    p = sub.add_parser("static-shard", help="Extract Layer 2 static family evidence shards on a machine with raw samples.")
    p.add_argument("--input-glob", nargs="*", default=[])
    p.add_argument("--input-list", default=None,
                   help="Exact .txt/.csv snapshot manifest; mutually exclusive with --input-glob.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--shard-id", default=None)
    p.add_argument("--labels", nargs="*", default=["phishing"])
    p.add_argument("--exclude-har", action="store_true")
    p.add_argument("--skip-errors", action="store_true")
    p.add_argument("--resume", dest="resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(func=cmd_static_shard)

    p = sub.add_parser("merge-static", help="Merge and deduplicate static family evidence shards.")
    p.add_argument("--shard-root", nargs="*", default=[])
    p.add_argument("--shard-glob", nargs="*", default=[])
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_merge_static)

    p = sub.add_parser("sequence-shard", help="Export packet sequence shards on a machine with raw samples.")
    p.add_argument("--features", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--shard-id", default=None)
    p.add_argument("--window", default="dyn")
    p.add_argument("--max-packets", type=int, default=128)
    p.add_argument("--sequence-window-s", type=float, default=90.0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--limit-per-label", type=int, default=None)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--skip-errors", action="store_true")
    p.add_argument("--allow-har-roles", action="store_true")
    p.set_defaults(func=cmd_sequence_shard)

    p = sub.add_parser("merge-sequences", help="Merge and deduplicate packet sequence NPZ shards.")
    p.add_argument("--sequence-root", nargs="*", default=[])
    p.add_argument("--sequence-glob", nargs="*", default=[])
    p.add_argument("--out-dir", required=True)
    p.set_defaults(func=cmd_merge_sequences)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import pathlib
import re
import tempfile
import zipfile
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .har import HarInfo, parse_har_obj
from .roles import host_from_url, normalize_host
from .timeutil import parse_time_to_epoch


@dataclass
class SamplePaths:
    zip_path: str
    root: str
    pcap_path: str
    har_path: Optional[str]
    session_path: str
    events_path: Optional[str] = None
    tempdir: Optional[tempfile.TemporaryDirectory] = None


@dataclass
class SampleMeta:
    sample_id: str
    label: str
    domain: str
    url: str
    status: str
    session: dict
    anchor_epoch: Optional[float]
    anchor_source: str
    pcap_start_epoch: Optional[float]
    pcap_stop_epoch: Optional[float]
    capture_stop_epoch: Optional[float]
    session_start_epoch: Optional[float]
    session_end_epoch: Optional[float]
    post_load_epoch: Optional[float]
    post_load_source: str
    first_action_epoch: Optional[float]
    first_action_source: str


def _find_member(names: List[str], suffix: str) -> Optional[str]:
    suffix = suffix.replace("\\", "/")
    matches = [n for n in names if n.replace("\\", "/").endswith(suffix)]
    if not matches:
        return None
    # Prefer shortest member path to avoid nested duplicates.
    return sorted(matches, key=len)[0]


def _find_first_member(names: List[str], suffixes: Sequence[str]) -> Optional[str]:
    for suffix in suffixes:
        match = _find_member(names, suffix)
        if match:
            return match
    return None


def _first_existing(path: pathlib.Path, names: Sequence[str]) -> Optional[pathlib.Path]:
    for name in names:
        candidate = path / name
        if candidate.exists():
            return candidate
    return None


def _first_existing_json(path: pathlib.Path, names: Sequence[str]) -> Optional[pathlib.Path]:
    first: Optional[pathlib.Path] = None
    for name in names:
        candidate = path / name
        if not candidate.exists():
            continue
        if first is None:
            first = candidate
        try:
            with candidate.open("r", encoding="utf-8-sig") as f:
                json.load(f)
            return candidate
        except Exception:
            continue
    return first


def _generated_session_for_raw_capture(path: pathlib.Path) -> dict:
    label = _infer_label_from_path(path)
    domain = _infer_domain_from_path(path)
    return {
        "sample_id": path.stem,
        "label": label,
        "domain": domain,
        "status": "raw_capture",
        "sample": {
            "sample_id": path.stem,
            "label": label,
            "url": f"https://{domain}" if domain else "",
        },
        "source_row": {
            "sample_id": path.stem,
            "host": domain,
            "source_path": str(path),
        },
        "event_times": {},
    }


def _raw_capture_file(path: pathlib.Path) -> bool:
    name = path.name.lower()
    suffix = path.suffix.lower()
    if suffix in {".pcap", ".pcapng"}:
        return True
    return suffix == ".json" and "labeled_biflows_all_packets_encryption_metadata" in name


def _public_release_root(json_path: pathlib.Path) -> pathlib.Path:
    if json_path.parent.name.lower() == "json":
        return json_path.parent.parent
    return json_path.parent


def _resolve_public_path(release_root: pathlib.Path, value: str, fallback: str) -> pathlib.Path:
    raw = str(value or fallback).strip()
    path = pathlib.Path(raw)
    return path if path.is_absolute() else release_root / path


def _extract_public_sample_json(src: pathlib.Path, require_har: bool) -> Optional[SamplePaths]:
    if src.suffix.lower() != ".json":
        return None
    try:
        with src.open("r", encoding="utf-8-sig") as handle:
            obj = json.load(handle)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    schema = str(obj.get("schema_version") or "")
    if "pcap_path" not in obj and not schema.startswith("kitscope-public-sample"):
        return None

    release_root = _public_release_root(src)
    capture_id = str(obj.get("capture_id") or src.stem)
    pcap_path = _resolve_public_path(
        release_root,
        str(obj.get("pcap_path") or ""),
        f"pcap/{capture_id}.pcap",
    )
    har_path = _resolve_public_path(
        release_root,
        str(obj.get("har_path") or ""),
        f"har/{capture_id}.har",
    )
    missing = []
    if not pcap_path.exists():
        missing.append(f"pcap_path={pcap_path}")
    if require_har and not har_path.exists():
        missing.append(f"har_path={har_path}")
    if missing:
        raise FileNotFoundError(f"{src}: missing {missing}")

    tmp = tempfile.TemporaryDirectory(prefix="web3pcap_public_")
    events_path: pathlib.Path | None = None
    events = obj.get("events")
    if isinstance(events, list):
        events_path = pathlib.Path(tmp.name) / "events.jsonl"
        with events_path.open("w", encoding="utf-8", newline="\n") as handle:
            for event in events:
                if isinstance(event, dict):
                    handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    return SamplePaths(
        zip_path=str(src),
        root=str(release_root),
        pcap_path=str(pcap_path),
        har_path=str(har_path) if har_path.exists() else None,
        session_path=str(src),
        events_path=str(events_path) if events_path else None,
        tempdir=tmp,
    )


def extract_zip_sample(zip_path: str | pathlib.Path, require_har: bool = True) -> SamplePaths:
    zip_path = str(zip_path)
    src = pathlib.Path(zip_path)
    public_sample = _extract_public_sample_json(src, require_har=require_har) if src.is_file() else None
    if public_sample is not None:
        return public_sample
    if src.is_file() and _raw_capture_file(src):
        if require_har:
            raise FileNotFoundError(f"{zip_path}: raw capture files do not include browser.har; use a gateway config with role_source=sni or sni_dns")
        tmp = tempfile.TemporaryDirectory(prefix="web3pcap_raw_")
        root = pathlib.Path(tmp.name)
        session_path = root / "session.json"
        session_path.write_text(json.dumps(_generated_session_for_raw_capture(src), indent=2), encoding="utf-8")
        return SamplePaths(
            zip_path=zip_path,
            root=str(root),
            pcap_path=str(src),
            har_path=None,
            session_path=str(session_path),
            events_path=None,
            tempdir=tmp,
        )
    if src.is_dir():
        pcap_path = _first_existing(src, ["capture.pcap", "capture.pcapng", "traffic.pcap", "traffic.pcapng"])
        har_path = _first_existing(src, ["browser.har", "label_only/browser.har"])
        session_path = _first_existing_json(src, ["session.json", "sample.json", "sample_private.json"])
        events_path = src / "events.jsonl"
        missing = [
            k for k, v in {
                "capture.pcap/capture.pcapng/traffic.pcapng": pcap_path,
                "session.json/sample.json": session_path,
            }.items()
            if v is None or not v.exists()
        ]
        if require_har and (har_path is None or not har_path.exists()):
            missing.append("browser.har")
        if missing:
            raise FileNotFoundError(f"{zip_path}: missing {missing}")
        return SamplePaths(
            zip_path=zip_path,
            root=str(src),
            pcap_path=str(pcap_path),
            har_path=str(har_path) if har_path and har_path.exists() else None,
            session_path=str(session_path),
            events_path=str(events_path) if events_path.exists() else None,
            tempdir=None,
        )
    tmp = tempfile.TemporaryDirectory(prefix="web3pcap_")
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        pcap_member = _find_first_member(names, ["capture.pcap", "capture.pcapng", "traffic.pcap", "traffic.pcapng"])
        har_member = _find_first_member(names, ["browser.har", "label_only/browser.har"])
        session_member = _find_first_member(names, ["session.json", "sample.json", "sample_private.json"])
        events_member = _find_member(names, "events.jsonl")
        missing = [
            k for k, v in {
                "capture.pcap/capture.pcapng/traffic.pcapng": pcap_member,
                "session.json/sample.json": session_member,
            }.items()
            if not v
        ]
        if require_har and not har_member:
            missing.append("browser.har")
        if missing:
            raise FileNotFoundError(f"{zip_path}: missing {missing}")
        for m in [pcap_member, har_member, session_member, events_member]:
            if m:
                zf.extract(m, tmp.name)
        root = pathlib.Path(tmp.name)
        return SamplePaths(
            zip_path=zip_path,
            root=str(root),
            pcap_path=str(root / pcap_member),
            har_path=str(root / har_member) if har_member else None,
            session_path=str(root / session_member),
            events_path=str(root / events_member) if events_member else None,
            tempdir=tmp,
        )


def read_json(path: str | pathlib.Path) -> dict:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def read_events_jsonl(path: Optional[str | pathlib.Path]) -> List[dict]:
    if not path:
        return []
    events = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def _event_time_from_session(session: dict, key: str) -> Optional[float]:
    et = session.get("event_times", {}) or {}
    if key in session:
        return parse_time_to_epoch(session.get(key))
    if key in et:
        return parse_time_to_epoch(et.get(key))
    return None


def _event_time_from_events(events: List[dict], key: str) -> Optional[float]:
    vals = []
    for e in events:
        # Flexible JSONL event formats.
        if key in e:
            vals.append(parse_time_to_epoch(e.get(key)))
        name = e.get("event_name") or e.get("event_type") or e.get("event") or e.get("type") or e.get("name")
        if name == key:
            vals.append(parse_time_to_epoch(
                e.get("utc_ts")
                or e.get("ts_utc")
                or e.get("start_ts_utc")
                or e.get("stop_ts_utc")
                or e.get("time")
                or e.get("ts")
                or e.get("timestamp")
                or e.get("epoch")
            ))
        if e.get("key") == key:
            vals.append(parse_time_to_epoch(e.get("value") or e.get("time") or e.get("ts")))
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None


def _as_anchor_keys(anchor_key: str | Sequence[str]) -> List[str]:
    if isinstance(anchor_key, str):
        keys = [anchor_key]
    else:
        keys = [str(k) for k in anchor_key]
    return list(dict.fromkeys([k for k in keys if k]))


def _infer_label_from_path(path: str | pathlib.Path) -> str:
    parts = [part.lower() for part in pathlib.Path(path).parts]
    name = pathlib.Path(path).name.lower()
    if name.startswith("benign") or any("benign" in part for part in parts):
        return "benign"
    if name.startswith("phishing") or any("phish" in part for part in parts):
        return "phishing"
    return "unknown"


def _infer_domain_from_path(path: str | pathlib.Path) -> str:
    stem = pathlib.Path(path).stem
    match = re.match(r"^[0-9a-f]{8,}_(.+?)__\d{8}", stem, flags=re.I)
    if match:
        return normalize_host(match.group(1))
    return normalize_host(stem)


def _event_name(e: dict) -> str:
    return str(e.get("event_name") or e.get("event_type") or e.get("event") or e.get("type") or e.get("name") or "")


def _event_epoch(e: dict) -> Optional[float]:
    return parse_time_to_epoch(
        e.get("utc_ts")
        or e.get("ts_utc")
        or e.get("start_ts_utc")
        or e.get("stop_ts_utc")
        or e.get("time")
        or e.get("ts")
        or e.get("timestamp")
        or e.get("epoch")
    )


def _first_named_event(events: List[dict], names: Sequence[str]) -> tuple[Optional[float], str]:
    wanted = {str(n) for n in names}
    vals: list[tuple[float, str]] = []
    for e in events:
        name = _event_name(e)
        if name not in wanted:
            continue
        ts = _event_epoch(e)
        if ts is not None:
            vals.append((float(ts), name))
    if not vals:
        return None, ""
    ts, name = min(vals, key=lambda item: item[0])
    return ts, f"events.jsonl.first.{name}"


def _event_times(events: List[dict], names: Sequence[str]) -> list[float]:
    wanted = {str(n) for n in names}
    vals = []
    for e in events:
        if _event_name(e) not in wanted:
            continue
        ts = _event_epoch(e)
        if ts is not None:
            vals.append(float(ts))
    return sorted(vals)


def _first_event_time(events: List[dict], names: Sequence[str]) -> Optional[float]:
    vals = _event_times(events, names)
    return vals[0] if vals else None


def _last_event_time(events: List[dict], names: Sequence[str]) -> Optional[float]:
    vals = _event_times(events, names)
    return vals[-1] if vals else None


_INTERACTION_ANCHOR_EVENTS = (
    "dummy_submit_form_button",
    "wild_dummy_submit",
    "form_submit",
    "submit",
    "action_click_text",
    "action_click_locator",
    "action_click_coordinates",
    "action_click",
)

_POST_LOAD_EVENTS = (
    "networkidle",
    "page_load",
    "load",
    "domcontentloaded",
    "page_domcontentloaded",
    "t_initial_page_loaded",
)


def _derive_post_load_epoch(session: dict, events: List[dict], first_action_epoch: Optional[float]) -> tuple[Optional[float], str]:
    et = session.get("event_times", {}) or {}
    vals: list[tuple[float, str]] = []
    initial = parse_time_to_epoch(et.get("t_initial_page_loaded"))
    if initial is not None:
        vals.append((float(initial), "session.json.event_times.t_initial_page_loaded"))
    for name in _POST_LOAD_EVENTS:
        for ts in _event_times(events, [name]):
            vals.append((float(ts), f"events.jsonl.{name}"))
    if not vals:
        return None, ""
    if first_action_epoch is not None:
        prior = [(ts, src) for ts, src in vals if ts <= float(first_action_epoch) + 1e-6]
        if prior:
            return max(prior, key=lambda item: item[0])
        earliest = min(vals, key=lambda item: item[0])
        if earliest[0] <= float(first_action_epoch) + 2.0:
            return earliest[0], earliest[1]
        return float(first_action_epoch), "events.jsonl.first_action_before_load_event"
    return max(vals, key=lambda item: item[0])


def _derive_har_post_load_epoch(har_path: Optional[str], first_action_epoch: Optional[float]) -> tuple[Optional[float], str]:
    if not har_path:
        return None, ""
    try:
        with open(har_path, "r", encoding="utf-8-sig") as f:
            obj = json.load(f)
    except Exception:
        return None, ""
    entries = (obj.get("log", {}) if isinstance(obj, dict) else {}).get("entries", []) or []
    spans: list[tuple[float, float]] = []
    for entry in entries:
        start = parse_time_to_epoch(entry.get("startedDateTime"))
        if start is None:
            continue
        try:
            dur = max(0.0, float(entry.get("time") or 0.0) / 1000.0)
        except (TypeError, ValueError):
            dur = 0.0
        spans.append((float(start), float(start) + dur))
    if not spans:
        return None, ""
    if first_action_epoch is not None:
        before_end = [end for start, end in spans if end <= float(first_action_epoch) + 1e-6]
        if before_end:
            return max(before_end), "har.request_end_before_first_action"
        before_start = [start for start, _end in spans if start <= float(first_action_epoch) + 1e-6]
        if before_start:
            return max(before_start), "har.request_start_before_first_action"
        return None, ""
    return max(end for _start, end in spans), "har.max_request_end"


def load_sample_meta(paths: SamplePaths, anchor_key: str | Sequence[str] = "t_metamask_connect_click") -> SampleMeta:
    session = read_json(paths.session_path)
    sample = session.get("sample", {}) or {}
    source_row = session.get("source_row", {}) or {}
    sample_id = str(sample.get("sample_id") or session.get("capture_id") or session.get("sample_id") or source_row.get("sample_id") or session.get("session_id") or pathlib.Path(paths.zip_path).stem)
    label = str(sample.get("label") or session.get("label") or _infer_label_from_path(paths.zip_path))
    domain = normalize_host(
        session.get("domain")
        or source_row.get("openphish_host")
        or source_row.get("host")
        or host_from_url(sample.get("url", ""))
        or _infer_domain_from_path(paths.zip_path)
    )
    url = sample.get("url") or f"https://{domain}"
    status = str(session.get("status") or "")
    et = session.get("event_times", {}) or {}
    anchor_keys = _as_anchor_keys(anchor_key)
    anchor_epoch = None
    anchor_source = ""
    for key in anchor_keys:
        anchor_epoch = _event_time_from_session(session, key)
        if anchor_epoch is not None:
            anchor_source = f"session.json.event_times.{key}"
            break
    if anchor_epoch is None:
        events = read_events_jsonl(paths.events_path)
        for key in anchor_keys:
            anchor_epoch = _event_time_from_events(events, key)
            if anchor_epoch is not None:
                anchor_source = f"events.jsonl.last.{key}"
                break
        if anchor_epoch is None:
            anchor_epoch, anchor_source = _first_named_event(events, _INTERACTION_ANCHOR_EVENTS)
        if anchor_epoch is None:
            anchor_source = "missing"
    else:
        events = read_events_jsonl(paths.events_path)
    first_action_epoch, first_action_source = _first_named_event(events, _INTERACTION_ANCHOR_EVENTS)
    if first_action_epoch is None and anchor_epoch is not None:
        first_action_epoch = anchor_epoch
        first_action_source = anchor_source
    pcap_start = parse_time_to_epoch(et.get("t_pcap_start"))
    pcap_stop = parse_time_to_epoch(et.get("t_pcap_stop"))
    capture_stop = parse_time_to_epoch(et.get("t_capture_stop"))
    session_start = parse_time_to_epoch(et.get("t_session_start") or session.get("start_utc_ts"))
    session_end = parse_time_to_epoch(et.get("t_session_end") or session.get("end_utc_ts"))
    pcap_start = pcap_start or _first_event_time(events, ["t_pcap_start"])
    pcap_stop = pcap_stop or _last_event_time(events, ["t_pcap_stop"])
    capture_stop = capture_stop or _last_event_time(events, ["t_capture_stop"])
    session_start = session_start or _first_event_time(events, ["t_session_start"])
    session_end = session_end or _last_event_time(events, ["t_session_end"])
    pcap_obj = session.get("pcap", {}) or {}
    pcap_start = pcap_start or parse_time_to_epoch(pcap_obj.get("start_ts_utc") or pcap_obj.get("started_utc"))
    pcap_stop = pcap_stop or parse_time_to_epoch(pcap_obj.get("stop_ts_utc") or pcap_obj.get("ended_utc"))
    capture_stop = capture_stop or pcap_stop
    session_start = session_start or parse_time_to_epoch(session.get("created_ts_utc"))
    session_end = session_end or parse_time_to_epoch(session.get("finished_ts_utc"))
    post_load_epoch, post_load_source = _derive_post_load_epoch(session, events, first_action_epoch)
    if post_load_epoch is None:
        post_load_epoch, post_load_source = _derive_har_post_load_epoch(paths.har_path, first_action_epoch)
    return SampleMeta(
        sample_id=sample_id,
        label=label,
        domain=domain,
        url=url,
        status=status,
        session=session,
        anchor_epoch=anchor_epoch,
        anchor_source=anchor_source,
        pcap_start_epoch=pcap_start,
        pcap_stop_epoch=pcap_stop,
        capture_stop_epoch=capture_stop,
        session_start_epoch=session_start,
        session_end_epoch=session_end,
        post_load_epoch=post_load_epoch,
        post_load_source=post_load_source,
        first_action_epoch=first_action_epoch,
        first_action_source=first_action_source,
    )


def load_har_for_sample(paths: SamplePaths, meta: SampleMeta, feature_mode: str = "clean") -> HarInfo:
    if not paths.har_path:
        return HarInfo()
    with open(paths.har_path, "r", encoding="utf-8-sig") as f:
        obj = json.load(f)
    return parse_har_obj(obj, primary_host=meta.domain, feature_mode=feature_mode)

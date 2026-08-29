from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .roles import classify_host, host_from_url, merge_roles, normalize_host
from .timeutil import parse_time_to_epoch


@dataclass
class HarInfo:
    hosts: Set[str] = field(default_factory=set)
    ip_to_hosts: Dict[str, Set[str]] = field(default_factory=dict)
    host_to_ips: Dict[str, Set[str]] = field(default_factory=dict)
    host_to_role: Dict[str, str] = field(default_factory=dict)
    ip_to_role: Dict[str, str] = field(default_factory=dict)
    entries: List[dict] = field(default_factory=list)


def _clean_server_ip(ip: Optional[str]) -> str:
    if not ip:
        return ""
    ip = str(ip).strip()
    if ip.startswith("[") and "]" in ip:
        ip = ip[1:ip.index("]")]
    # HAR sometimes stores "1.2.3.4:443".
    if ip.count(":") == 1 and "." in ip:
        ip = ip.split(":", 1)[0]
    return ip


def parse_har_obj(obj: dict, primary_host: str = "", feature_mode: str = "clean") -> HarInfo:
    info = HarInfo()
    log = obj.get("log", {}) if isinstance(obj, dict) else {}
    entries = log.get("entries", []) or []
    for e in entries:
        req = e.get("request", {}) or {}
        url = req.get("url", "")
        host = host_from_url(url)
        if not host:
            continue
        info.hosts.add(host)
        role = classify_host(host, primary_host=primary_host, feature_mode=feature_mode)
        info.host_to_role[host] = role
        server_ip = _clean_server_ip(e.get("serverIPAddress"))
        if server_ip:
            info.ip_to_hosts.setdefault(server_ip, set()).add(host)
            info.host_to_ips.setdefault(host, set()).add(server_ip)
        info.entries.append(
            {
                "host": host,
                "role": role,
                "server_ip": server_ip,
                "started_epoch": parse_time_to_epoch(e.get("startedDateTime")),
                "time_ms": e.get("time"),
                # Body sizes are kept only for debugging reports. They are not model features.
                "transfer_size": (e.get("response", {}) or {}).get("_transferSize"),
                "body_size": (e.get("response", {}) or {}).get("bodySize"),
            }
        )
    for ip, hosts in info.ip_to_hosts.items():
        roles = [info.host_to_role.get(h, "unknown") for h in hosts]
        info.ip_to_role[ip] = merge_roles(roles)
    return info


def load_har(path: str | pathlib.Path, primary_host: str = "", feature_mode: str = "clean") -> HarInfo:
    with open(path, "r", encoding="utf-8-sig") as f:
        obj = json.load(f)
    return parse_har_obj(obj, primary_host=primary_host, feature_mode=feature_mode)

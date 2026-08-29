#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import json
import pathlib
import re
from collections import Counter, defaultdict
from itertools import combinations
from statistics import mean
from urllib.parse import parse_qsl, urlparse
from typing import Sequence

import pandas as pd


TEXT_EXTS = {".html", ".json", ".jsonl", ".log", ".txt", ".har", ".js"}
SKIP_DIRS = {"screenshots"}
MAX_FILE_BYTES = 8_000_000

SCRIPT_RE = re.compile(r"<script\b[^>]*\bsrc=[\"']([^\"']+)[\"']", re.I)
TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.S)
PROJECT_ID_RE = re.compile(r"projectId[\"'=:%20\s]+([a-f0-9]{16,64})", re.I)
WC_VERIFY_RE = re.compile(r"verify\.walletconnect\.com/([a-f0-9]{12,64})", re.I)
ADDRESS_RE = re.compile(r"\b0x[a-fA-F0-9]{40}\b")
METHOD_RE = re.compile(
    r"\b(eth_sendTransaction|wallet_switchEthereumChain|wallet_addEthereumChain|eth_signTypedData(?:_v[34])?|personal_sign|eth_sign|setApprovalForAll|isApprovedForAll|increaseAllowance|permit2|PermitSingle|PermitBatch)\b|(?<![a-zA-Z0-9_$])approve\s*\(",
    re.I,
)
API_RE = re.compile(r"\b(?:fetch|axios\.(?:get|post)|XMLHttpRequest\.open)\s*\(\s*[\"']([^\"']+)", re.I)
JQUERY_AJAX_URL_RE = re.compile(r"\$\.ajax\s*\(\s*\{.*?\burl\s*:\s*[\"']([^\"']+)", re.I | re.S)
GENERIC_URL_LITERAL_RE = re.compile(r"\burl\s*:\s*[\"']([^\"']+?\.(?:php|asp|aspx|jsp|json|html|do|action))(?:[?#][^\"']*)?[\"']", re.I)
DRainer_RE = re.compile(r"\b(DrainerPopup|new\s+Drainer|Draining\s+(?:started|finished)|assets/js/drainer(?:/[^\"'<>\s]+)?|drainer\.js|wallet\s*drainer|drainer)\b", re.I)
OBF_BUNDLE_RE = re.compile(r"\b[a-f0-9]{3,}[_-][a-f0-9]{3,}\.js\b|/[a-f0-9]{8,}\.[0-9]+\.js\b", re.I)
CLAIM_WORD_RE = re.compile(r"\b(claim|airdrop|migrate|reward|presale|eligib(?:le|ility)|verify|rectify|validate|mint|bridge|refund|revoke|swap)\b", re.I)
INLINE_SCRIPT_BLOCK_RE = re.compile(r"<script\b(?![^>]*\bsrc\s*=)[^>]*>(.*?)</script>", re.I | re.S)
HTML_TAG_RE = re.compile(r"<\s*/?\s*([a-zA-Z][a-zA-Z0-9:-]*)\b([^>]*)>", re.S)
ID_CLASS_RE = re.compile(r"\b(id|class|name|data-testid|aria-label)\s*=\s*[\"']([^\"']{1,120})[\"']", re.I)
JS_VAR_ADDRESS_RE = re.compile(
    r"\b(?:const|let|var)?\s*([A-Za-z_$][\w$]{0,80})\s*=\s*[\"'](0x[a-fA-F0-9]{40})[\"']",
    re.I,
)
DICT_ADDRESS_RE = re.compile(r"[\"']([A-Za-z0-9_$:-]{1,80})[\"']\s*:\s*[\"'](0x[a-fA-F0-9]{40})[\"']", re.I)
SPENDER_VAR_HINT_RE = re.compile(
    r"(?:auth|authori[sz]ed|spender|operator|affiliate|receiver|recipient|receive|merchant|owner|drain|target|to[_-]?address|wallet)",
    re.I,
)
TOKEN_VAR_HINT_RE = re.compile(r"(?:token|contract|approveaddr|coin|asset)", re.I)
APPROVAL_CALL_ARG_RE = re.compile(
    r"\.(?:approve|increaseAllowance|permit|setApprovalForAll)\s*\(\s*(?:[\"'](0x[a-fA-F0-9]{40})[\"']|([A-Za-z_$][\w$]{0,80}))",
    re.I,
)
SEND_TX_TO_RE = re.compile(r"\bto\s*:\s*(?:[\"'](0x[a-fA-F0-9]{40})[\"']|([A-Za-z_$][\w$]{0,80}))", re.I)
RPC_METHOD_LITERAL_RE = re.compile(r"[\"']method[\"']\s*:\s*[\"']([^\"']+)[\"']", re.I)
INFURA_KEY_RE = re.compile(r"\b(?:infura(?:[_-]?key)?|projectId|project_id)\b[^\n\r]{0,80}?[\"']([a-f0-9]{24,64})[\"']", re.I)

COMMON_TOKEN_ADDRESSES = {
    # Frequently embedded token/DeFi contract addresses. These are not family/operator clues.
    "0x0000000000000000000000000000000000000000",
    "0x00000000000c2e074ec69a0dfb2997ba6c7d2e1e",
    "0x0000000000095413afc295d19edeb1ad7b71c952",
    "0x0bc529c00c6401aef6d220be8c6ea1667f6ad93e",
    "0x1494ca1f11d487c2bbe4543e90080aeba4ba3c2b",
    "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984",
    "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599",
    "0x3432b6a60d23ca0dfca7761b7ab56459d9c964d0",
    "0x514910771af9ca656af840dff83e8264ecf986ca",
    "0x6b175474e89094c44da98b954eedeac495271d0f",
    "0x6b3595068778dd592e39a122f4f5a5cf09c90fe2",
    "0x7fc66500c84a76ad7e9c93437bfc5ac33e2ddae9",
    "0x853d955acef822db058eb8505911ed77f175b99e",
    "0x956f47f50a910163d8bf957cf5846d573e7f87ca",
    "0x9f8f72aa9304c8b593d555f12ef6589cc3a579a2",
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
    "0xa1faa113cbe53436df28ff0aee54275c13b40975",
    "0xa47c8bf37f92abed4a126bda807a7b7498661acd",
    "0xba11d00c5f74255f56a5e366f4f77f5a186d7f55",
    "0xc00e94cb662c3520282e6f5717214004a7f26888",
    "0xc011a73ee8576fb46f5e1c5751ca3b9fe0af2a6f",
    "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",
    "0xc7283b66eb1eb5fb86327f08e1b5816b0720212b",
    "0xc834fa996fa3bec7aad3693af486ae53d8aa8b50",
    "0xd2877702675e6ceb975b4a1dff9fb7baf4c91ea9",
    "0xd533a949740bb3306d119cc777fa900ba034cd52",
    "0xdac17f958d2ee523a2206206994597c13d831ec7",
    "0xe28b3b32b6c345a34ff64674606124dd5aceca30",
    "0xeef9f339514298c6a857efcfc1a762af84438dee",
}
COMMON_RPC_PROJECT_IDS = {
    # MetaMask extension default Infura project ID from captures; not a kit-family clue.
    "b6bf7d3508c941499b10025c0776eaf8",
}

COMMON_HOSTS = {
    "www.w3.org", "fonts.googleapis.com", "fonts.gstatic.com", "cdn.jsdelivr.net", "cdnjs.cloudflare.com",
    "unpkg.com", "ajax.googleapis.com", "www.google-analytics.com", "www.googletagmanager.com",
    "www.gstatic.com", "connect.facebook.net", "browser-intake-datadoghq.com",
    "static.cloudflareinsights.com", "cdn.tailwindcss.com",
}

# Hosts and paths that are common utilities/noise. They may be useful for page
# behaviour auditing, but they must not define a backend kit family.
PUBLIC_UTILITY_HOSTS = {
    "api.ipify.org", "api64.ipify.org", "ipinfo.io", "ipapi.co", "ip-api.com", "api.myip.com",
    "ifconfig.me", "icanhazip.com", "checkip.amazonaws.com", "geolocation-db.com", "api.db-ip.com",
    "www.google-analytics.com", "www.googletagmanager.com", "stats.g.doubleclick.net",
    "browser-intake-datadoghq.com", "static.cloudflareinsights.com",
    "rpc.walletconnect.com", "relay.walletconnect.com", "verify.walletconnect.com",
}
PUBLIC_UTILITY_HOST_SUFFIXES = (
    ".infura.io", ".alchemy.com", ".alchemyapi.io", ".ankr.com", ".walletconnect.com",
    ".google-analytics.com", ".googletagmanager.com", ".gstatic.com", ".googleapis.com",
    ".cloudflare.com", ".cloudflareinsights.com", ".sentry.io", ".datadoghq.com",
)
BACKEND_STATIC_EXTS = {
    ".js", ".mjs", ".css", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".ico",
    ".woff", ".woff2", ".ttf", ".otf", ".map", ".mp4", ".webm", ".txt", ".xml",
}
BACKEND_ROUTE_HINT_RE = re.compile(
    r"(?:/|\b)(?:api|ajax|wallet|user|index|admin|app|auth|connect|claim|verify|address|asset|erc|nft|token|approve|approval|allowance|balance|money|save|getaddress|postinfo|submit|login|send|notify|collect|drain|telegram|discord|webhook)(?:/|\b|_|-)",
    re.I,
)
BACKEND_FIELD_HINT_RE = re.compile(
    r"^(?:address|account|wallet|wallet_address|auth_address|authorized_address|spender|operator|owner|from|to|chain|chainid|chain_id|bizhong|symbol|token|contract|amount|balance|money|assets?|invite|invite_code|uid|user|email|password|phone|code|otp|seed|mnemonic|privatekey|signature|tx|hash|ip|ua)$",
    re.I,
)
BACKEND_OBJECT_IGNORE_KEYS = {
    "url", "type", "method", "headers", "body", "data", "params", "success", "error", "complete",
    "async", "cache", "contenttype", "content_type", "datatype", "data_type", "timeout", "credentials",
    "mode", "redirect", "referrer", "signal", "then", "catch", "finally", "json", "text", "blob",
    # Capture/session metadata that may appear next to HTML snippets in local files.
    "sample", "sample_id", "session_id", "label", "domain", "status", "flags", "url",
    "connect_confirmed", "signature_prompt_seen", "high_risk_wallet_prompt_seen",
    "color", "border", "border_collapse", "content", "display", "font", "font_family", "font_size",
    "font_style", "list_style", "padding", "position", "text_decoration", "viewport", "description",
    "keywords", "format_detection", "apple_mobile_web_app_capable", "apple_mobile_web_app_status_bar_style",
    "ready_state", "provider", "core", "conv", "style", "class", "id",
}
BACKEND_ROLE_PATTERNS = [
    ("approval_report", re.compile(r"save_erc|auth_address|approve|approval|allowance|permit|setapproval|spender", re.I)),
    ("asset_or_balance_report", re.compile(r"money|balance|asset|token|erc|nft|collect", re.I)),
    ("address_config", re.compile(r"getaddress|address_base|config|settings|init|bootstrap", re.I)),
    ("wallet_session", re.compile(r"wallet|connect|account|chain", re.I)),
    ("credential_capture", re.compile(r"login|password|email|phone|otp|code|seed|mnemonic|private", re.I)),
    ("messaging_exfil", re.compile(r"telegram|discord|webhook|notify|send", re.I)),
    ("generic_submit", re.compile(r"submit|save|post|collect|report", re.I)),
]
COMMON_CAPTURE_ADDRESSES = {
    # The instrumented browser wallet used during capture. It appears on many pages and is not a drainer-family clue.
    "0xcac6a6837a42af4aaab26d8849408d4d9e7ba975",
}
COMMON_SCRIPT_TOKENS = {
    "jquery", "bootstrap", "react", "react-dom", "web3.min.js", "ethers", "sweetalert", "sentry",
    "chunk", "polyfill", "runtime", "main.js", "app.js", "index.js",
    "pre-load.js", "./common-", "./ui-", "./scripts/lockdown", "./scripts/policy-load", "./scripts/snow.js", "./scripts/use-snow.js",
}
COMMON_JS_LIBRARY_TOKENS = {
    "jquery", "bootstrap", "web3.min", "ethers", "web3modal", "walletconnect", "wallet-connect",
    "web3provider", "clipboard", "layer.js", "flexible.js", "bignumber", "lodash", "vue", "react",
    "polyfill", "runtime", "vendor", "chunk-vendors",
}
CAPABILITY_PATTERNS = {
    "wallet_api": re.compile(r"ethereum\.(?:request|enable|send|sendAsync)|eth_requestAccounts|wallet_requestPermissions|web3\.eth\.getAccounts", re.I),
    "walletconnect": re.compile(r"WalletConnect|walletconnect|Web3Modal|display_uri|wc:", re.I),
    "token_approval": re.compile(r"\.(?:approve|increaseAllowance|setApprovalForAll)\s*\(|allowance\s*\(|PermitSingle|PermitBatch|permit2", re.I),
    "balance_sweep": re.compile(r"balanceOf\s*\(|getBalance\s*\(|getMostValuableAssets|decimals\s*\(|tokenOfOwnerByIndex", re.I),
    "tx_signing": re.compile(r"eth_sendTransaction|eth_signTypedData(?:_v[34])?|personal_sign|eth_sign|sendTransaction\s*\(", re.I),
    "rpc_provider": re.compile(r"new\s+Web3\s*\(|HttpProvider|cloudflare-eth|infura|alchemy|ankr|rpc\.", re.I),
    "backend_exfil": re.compile(r"\$\.ajax|fetch\s*\(|XMLHttpRequest|axios\.|postInfo\s*\(|auth_address|save_erc_data|auth_address_money", re.I),
    "realtime_socket": re.compile(r"new\s+WebSocket|WebSocket\s*\(|socket\.io|\bio\s*\(\s*[\"'](?:wss?:)?//", re.I),
    "fingerprinting": re.compile(r"navigator\.(?:userAgent|webdriver|plugins|languages|deviceMemory|hardwareConcurrency)|screen\.|canvas|webgl|localStorage|sessionStorage", re.I),
    "captcha_cloak": re.compile(r"captcha|turnstile|cf-challenge|cloudflare|HeadlessChrome|webdriver|bot\b", re.I),
    "clipboard": re.compile(r"ClipboardJS|navigator\.clipboard|document\.execCommand\s*\(\s*[\"']copy", re.I),
}
FEATURE_WEIGHTS = {
    # Backend-kit evidence. These are the only feature families that should
    # define Layer-2 backend family pseudo-labels.
    "backend_kit": 10.0,
    "backend_request_schema_hash": 8.5,
    "backend_flow_hash": 8.0,
    "backend_route_set_hash": 7.5,
    "backend_callsite_hash": 7.0,
    "backend_endpoint": 5.0,
    "backend_host_pattern": 4.0,
    "backend_role": 2.0,
    # Support evidence. Strong for auditing, not sufficient to define backend kit identity.
    "inline_script_hash": 5.0,
    "script_content_hash": 4.5,
    "drainer_spender": 4.0,
    "drainer_contract": 3.5,
    "drainer_symbol": 3.0,
    "drainer_path": 3.0,
    "kit_behavior_hash": 3.0,
    "resource_graph_hash": 2.5,
    "danger_method_combo": 2.5,
    "walletconnect_project": 2.0,
    "walletconnect_verify": 2.5,
    "infura_project": 1.5,
    "address": 1.0,
    "api_endpoint": 1.0,
    "js_capability_profile": 1.5,
    "script_local": 1.2,
    "script_host_path": 1.0,
    "html_structure_hash": 1.0,
    "image_asset_hash": 1.0,
    "obf_bundle": 1.0,
    "title": 0.6,
    "body_template": 0.6,
    "claim_words": 0.4,
}


def read_text(path: pathlib.Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig", errors="ignore")
    except Exception:
        return ""


def read_json(path: pathlib.Path) -> dict:
    try:
        return json.loads(read_text(path))
    except Exception:
        return {}


def _normalise_host_for_static(value: str) -> str:
    text = str(value or "").strip().lower()
    if "://" in text:
        text = urlparse(text).netloc.lower()
    text = text.split("@")[-1].split("/", 1)[0].split(":", 1)[0].strip(".")
    if text.startswith("www."):
        return text[4:]
    return text


def sample_json_to_session(sample: dict, sample_dir: pathlib.Path) -> dict:
    """Return a session-like metadata dict for non-Web3 interaction captures."""
    source = sample.get("source_row", {}) or {}
    artifacts = sample.get("artifacts", {}) or {}
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
        "session_id": sample.get("session_id") or sample.get("sample_id") or sample_dir.name,
        "domain": _normalise_host_for_static(str(host or urlparse(str(url)).netloc or sample_dir.name)),
        "sample": {
            "sample_id": sample.get("sample_id") or sample.get("id") or sample_dir.name,
            "label": sample.get("label") or ("phishing" if "phish" in str(sample_dir).lower() else ""),
            "url": url,
            "notes": source.get("notes") or "",
        },
        "status": sample.get("status") or sample.get("result") or "",
        "flags": sample.get("flags", {}) or {},
        "artifacts": artifacts,
        "_source_schema": "sample.json",
    }


def read_capture_session(sample_dir: pathlib.Path, session_path: pathlib.Path | None = None) -> dict:
    session = read_json(session_path) if session_path else read_json(sample_dir / "session.json")
    if session:
        session.setdefault("_source_schema", (session_path or sample_dir / "session.json").name)
        return session
    sample = read_json(sample_dir / "sample.json") or read_json(sample_dir / "sample_private.json")
    if sample:
        return sample_json_to_session(sample, sample_dir)
    return sample_json_to_session({}, sample_dir)


def _metadata_base_dir(sample_dir: pathlib.Path, session_path: pathlib.Path | None) -> pathlib.Path:
    if session_path:
        parent = session_path.parent
        if parent.name.lower() == "json":
            return parent.parent
        return parent
    if sample_dir.parent.name.lower() == "static_features":
        return sample_dir.parent.parent
    return sample_dir


def _append_har_candidates(candidates: list[str], obj: dict) -> None:
    if not isinstance(obj, dict):
        return
    artifacts = obj.get("artifacts", {}) or {}
    if isinstance(artifacts, dict) and artifacts.get("har"):
        candidates.append(str(artifacts.get("har")))
    paths = obj.get("paths", {}) or {}
    if isinstance(paths, dict):
        for key in ("har", "har_path", "browser_har"):
            if paths.get(key):
                candidates.append(str(paths.get(key)))
    browser = obj.get("browser", {}) or {}
    if isinstance(browser, dict) and browser.get("har_path"):
        candidates.append(str(browser.get("har_path")))
    if obj.get("har_path"):
        candidates.append(str(obj.get("har_path")))


def find_har_path(
    sample_dir: pathlib.Path,
    session_path: pathlib.Path | None = None,
    har_path: pathlib.Path | None = None,
) -> pathlib.Path | None:
    if har_path is not None and har_path.exists():
        return har_path
    session = read_json(session_path) if session_path else read_json(sample_dir / "session.json")
    sample = read_json(sample_dir / "sample.json") or read_json(sample_dir / "sample_private.json")
    candidates: list[str] = []
    for obj in (session, sample):
        _append_har_candidates(candidates, obj)
    capture_id = str(session.get("capture_id") or sample_dir.name if isinstance(session, dict) else sample_dir.name)
    if capture_id:
        candidates.append(f"har/{capture_id}.har")
    candidates.extend(["browser.har", "label_only/browser.har"])
    base_dir = _metadata_base_dir(sample_dir, session_path)
    for raw in candidates:
        if not raw:
            continue
        path = pathlib.Path(raw)
        if not path.is_absolute():
            for base in (base_dir, sample_dir):
                candidate = base / path
                if candidate.exists():
                    return candidate
            continue
        if path.exists():
            return path
    return None


def iter_text_files(root: pathlib.Path, exclude_exts: set[str] | None = None):
    exclude_exts = {e.lower() for e in (exclude_exts or set())}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        name = path.name.lower()
        rel = str(path.relative_to(root)).replace("\\", "/").lower()
        if name.startswith("metamask_") or "/metamask_" in rel or "chrome-extension" in rel:
            continue
        suffix = path.suffix.lower()
        if suffix in exclude_exts:
            continue
        if suffix not in TEXT_EXTS:
            continue
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        yield path


def norm_url_token(url: str, primary_domain: str) -> str:
    url = html.unescape(url.strip())
    if not url or url.startswith("data:") or url.startswith("blob:"):
        return ""
    parsed = urlparse(url)
    if parsed.netloc:
        host = parsed.netloc.lower()
        path = parsed.path.lower()
        if host in COMMON_HOSTS:
            return ""
        if host == primary_domain.lower():
            return path
        return f"{host}{path}"
    return parsed.path.lower() or url.lower()


def useful_script(src: str) -> bool:
    s = src.lower()
    if not s or s in {"/", "#"}:
        return False
    return not any(tok in s for tok in COMMON_SCRIPT_TOKENS)


def normalize_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"0x[a-fA-F0-9]{40}", " ADDR ", text)
    text = re.sub(r"[a-f0-9]{16,64}", " HEX ", text, flags=re.I)
    text = re.sub(r"https?://[^\s\"'<>]+", " URL ", text)
    text = re.sub(r"\b\d+(?:[.,]\d+)*\b", " NUM ", text)
    text = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff$]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def short_hash(text: str, n: int = 12) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:n]


def content_short_hash(raw: bytes | str, n: int = 16) -> str:
    if isinstance(raw, str):
        raw = raw.encode("utf-8", errors="ignore")
    return hashlib.sha256(raw).hexdigest()[:n]


def decode_har_content(content: dict) -> str:
    raw = decode_har_content_bytes(content)
    if not raw:
        return ""
    return raw.decode("utf-8", errors="ignore")


def decode_har_content_bytes(content: dict) -> bytes:
    text = content.get("text") or ""
    if not text:
        return b""
    if str(content.get("encoding", "")).lower() == "base64":
        try:
            return base64.b64decode(text, validate=False)
        except Exception:
            return b""
    return str(text).encode("utf-8", errors="ignore")


def strip_js_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    text = re.sub(r"(^|[^:])//.*", r"\1 ", text)
    return text


def normalize_javascript_for_hash(text: str) -> str:
    text = html.unescape(text or "")
    text = strip_js_comments(text)
    text = re.sub(r"0x[a-fA-F0-9]{40}", " ADDR ", text)
    text = re.sub(r"[a-f0-9]{24,64}", " HEX ", text, flags=re.I)
    text = re.sub(r"https?://[^\s\"'<>]+", " URL ", text)
    text = re.sub(r"['\"][^'\"]{0,240}['\"]", " STR ", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\b", " NUM ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:300_000]


def normalize_html_structure(text: str) -> str:
    parts: list[str] = []
    for match in HTML_TAG_RE.finditer(text or ""):
        tag = match.group(1).lower()
        attrs = []
        for key, value in ID_CLASS_RE.findall(match.group(2) or ""):
            norm_value = normalize_text(value)[:48]
            if norm_value:
                attrs.append(f"{key.lower()}={norm_value}")
        if attrs:
            parts.append(f"{tag}[{','.join(sorted(attrs)[:5])}]")
        else:
            parts.append(tag)
        if len(parts) >= 1_000:
            break
    return " ".join(parts)


def html_to_visible_text(text: str) -> str:
    text = re.sub(r"<script\b.*?</script>", " ", text or "", flags=re.I | re.S)
    text = re.sub(r"<style\b.*?</style>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    return normalize_text(text)


def is_extension_or_wallet_url(url: str) -> bool:
    lower = str(url or "").lower()
    return lower.startswith("chrome-extension://") or "nkbihfbeogaeaoehlefnkodbefgpgknn" in lower or "metamask" in lower


def host_of(url: str) -> str:
    return urlparse(str(url or "")).netloc.lower()


def is_first_party_url(url: str, primary_domain: str) -> bool:
    host = host_of(url)
    if not host:
        return True
    primary_domain = primary_domain.lower()
    return host == primary_domain or host.endswith("." + primary_domain)


def is_js_resource(url: str, mime: str) -> bool:
    path = urlparse(str(url or "")).path.lower()
    return "javascript" in str(mime or "").lower() or path.endswith(".js")


def is_html_resource(url: str, mime: str) -> bool:
    path = urlparse(str(url or "")).path.lower()
    m = str(mime or "").lower()
    return "text/html" in m or path.endswith((".html", "/"))


def is_image_resource(url: str, mime: str) -> bool:
    path = urlparse(str(url or "")).path.lower()
    m = str(mime or "").lower()
    return m.startswith("image/") or path.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".ico"))


def likely_common_js_library(url_token: str, text: str) -> bool:
    lower = (url_token or "").lower()
    if any(tok in lower for tok in COMMON_JS_LIBRARY_TOKENS):
        return True
    prefix = (text or "")[:2_000].lower()
    common_markers = (
        "jquery javascript library", "web3modal", "walletconnectprovider", "walletconnect provider",
        "web3.js", "clipboard.js", "bignumber.js", "bootstrap", "react.production",
    )
    return any(marker in prefix for marker in common_markers)


def detect_capabilities(text: str) -> set[str]:
    return {name for name, pattern in CAPABILITY_PATTERNS.items() if pattern.search(text or "")}


def classify_interaction_pattern(text: str, capabilities: set[str], api_endpoints: Sequence[str]) -> str:
    has_wallet = bool({"wallet_api", "walletconnect"} & capabilities)
    has_approval = "token_approval" in capabilities
    has_backend = bool(api_endpoints) or "backend_exfil" in capabilities
    has_realtime = "realtime_socket" in capabilities
    has_captcha = "captcha_cloak" in capabilities
    has_forms = bool(re.search(r"<(?:form|input|select|textarea)\b", text or "", re.I))
    if has_wallet and has_approval and has_backend and has_realtime:
        return "wallet_approval_multi_stage_realtime"
    if has_wallet and has_approval and has_backend:
        return "wallet_approval_multi_stage_backend"
    if has_wallet and has_approval:
        return "wallet_approval_local"
    if has_wallet and has_backend:
        return "wallet_connect_backend"
    if has_realtime and has_forms:
        return "credential_multi_stage_realtime"
    if has_captcha and has_forms:
        return "credential_captcha_cloaked"
    if has_forms and has_backend:
        return "credential_multi_stage_backend"
    if has_forms:
        return "credential_single_stage"
    if has_wallet:
        return "wallet_connect_only"
    return "unknown"


def extract_har_site_resources(
    sample_dir: pathlib.Path,
    primary_domain: str,
    include_har: bool,
    session_path: pathlib.Path | None = None,
    har_path: pathlib.Path | None = None,
) -> dict:
    resources = {
        "text_parts": [],
        "html_docs": [],
        "inline_scripts": [],
        "first_party_js": [],
        "script_tokens": set(),
        "local_script_paths": set(),
        "resource_sequence": [],
        "image_hashes": set(),
        "request_payloads": [],
    }
    if not include_har:
        return resources
    resolved_har = find_har_path(sample_dir, session_path=session_path, har_path=har_path)
    if resolved_har is None:
        return resources
    try:
        har = json.loads(read_text(resolved_har))
    except Exception:
        return resources
    for entry in har.get("log", {}).get("entries", []) or []:
        request = entry.get("request", {}) or {}
        response = entry.get("response", {}) or {}
        url = str(request.get("url") or "")
        if not url or is_extension_or_wallet_url(url):
            continue
        method = str(request.get("method") or "GET").upper()
        parsed = urlparse(url)
        path = parsed.path or "/"
        token = norm_url_token(url, primary_domain)
        mime = str((response.get("content", {}) or {}).get("mimeType") or "")
        text = decode_har_content(response.get("content", {}) or {})
        post_text = str(((request.get("postData", {}) or {}).get("text") or ""))
        if post_text and not is_extension_or_wallet_url(str(request.get("headers", ""))):
            resources["request_payloads"].append(post_text[:100_000])
            resources["text_parts"].append(post_text[:100_000])
            for rpc_method in RPC_METHOD_LITERAL_RE.findall(post_text):
                resources.setdefault("rpc_methods", set()).add(str(rpc_method).lower())
        if token and is_first_party_url(url, primary_domain):
            ext = pathlib.PurePosixPath(path).suffix.lower()
            if ext in {".js", ".css", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".ico"} or method == "GET":
                resources["resource_sequence"].append(token[:160])
        if text and is_first_party_url(url, primary_domain):
            resources["text_parts"].append(text[:500_000])
        elif text and is_js_resource(url, mime) and not is_first_party_url(url, primary_domain):
            # External JS can still expose capabilities, but avoid using it as a code family hash.
            resources["text_parts"].append(text[:120_000])
        if is_html_resource(url, mime) and text:
            resources["html_docs"].append(text[:500_000])
            for src in SCRIPT_RE.findall(text):
                stoken = norm_url_token(src, primary_domain)
                if stoken and useful_script(stoken):
                    resources["script_tokens"].add(stoken)
                    if not re.match(r"^[a-z0-9.-]+\.", stoken):
                        resources["local_script_paths"].add(stoken)
            for block in INLINE_SCRIPT_BLOCK_RE.findall(text):
                if block and len(block.strip()) >= 80:
                    resources["inline_scripts"].append(block)
        if is_js_resource(url, mime) and text:
            stoken = token or path.lower()
            if stoken and useful_script(stoken):
                resources["script_tokens"].add(stoken)
                if is_first_party_url(url, primary_domain):
                    resources["local_script_paths"].add(stoken)
            if is_first_party_url(url, primary_domain) and not likely_common_js_library(stoken, text):
                resources["first_party_js"].append((stoken, text[:500_000]))
        if is_image_resource(url, mime) and text and is_first_party_url(url, primary_domain):
            if token and not any(skip in token.lower() for skip in {"favicon", "logo"}):
                raw_image = decode_har_content_bytes(response.get("content", {}) or {})
                resources["image_hashes"].add(content_short_hash(raw_image[:200_000], n=14))
    return resources


def extract_var_addresses(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for name, address in JS_VAR_ADDRESS_RE.findall(text or ""):
        out[str(name).lower()] = str(address).lower()
    for name, address in DICT_ADDRESS_RE.findall(text or ""):
        # Keep dictionary addresses for token allowlist filtering and evidence reports.
        out[str(name).lower()] = str(address).lower()
    return out


def extract_drainer_spenders(text: str) -> set[str]:
    var_addresses = extract_var_addresses(text)
    candidates: set[str] = set()
    for name, address in var_addresses.items():
        if address in COMMON_TOKEN_ADDRESSES or address in COMMON_CAPTURE_ADDRESSES:
            continue
        if SPENDER_VAR_HINT_RE.search(name) and not TOKEN_VAR_HINT_RE.search(name):
            candidates.add(address)
    for direct, varname in APPROVAL_CALL_ARG_RE.findall(text or ""):
        address = (direct or var_addresses.get(str(varname).lower(), "")).lower()
        if address and address not in COMMON_TOKEN_ADDRESSES and address not in COMMON_CAPTURE_ADDRESSES:
            candidates.add(address)
    for direct, varname in SEND_TX_TO_RE.findall(text or ""):
        address = (direct or var_addresses.get(str(varname).lower(), "")).lower()
        if address and address not in COMMON_TOKEN_ADDRESSES and address not in COMMON_CAPTURE_ADDRESSES:
            candidates.add(address)
    return candidates


def extract_api_endpoints(text: str, primary_domain: str) -> list[str]:
    raw = []
    raw.extend(API_RE.findall(text or ""))
    raw.extend(JQUERY_AJAX_URL_RE.findall(text or ""))
    raw.extend(GENERIC_URL_LITERAL_RE.findall(text or ""))
    endpoints = sorted({norm_url_token(endpoint, primary_domain) for endpoint in raw})
    return [e for e in endpoints if e and not e.startswith("chrome-extension")]


def _host_is_public_utility(host: str) -> bool:
    host = (host or "").lower().split(":", 1)[0]
    if not host:
        return False
    if host in COMMON_HOSTS or host in PUBLIC_UTILITY_HOSTS:
        return True
    return any(host.endswith(suffix) for suffix in PUBLIC_UTILITY_HOST_SUFFIXES)


def _normalise_query_names(query: str) -> list[str]:
    names: set[str] = set()
    for key, _ in parse_qsl(query or "", keep_blank_values=True):
        key = normalize_text(str(key)).replace(" ", "_")[:48]
        if key:
            names.add(key)
    return sorted(names)


def _normalise_backend_path(path: str, query: str = "") -> str:
    path = html.unescape(path or "/").strip()
    if not path.startswith("/"):
        path = "/" + path
    path = path.lower()
    path = re.sub(r"0x[a-f0-9]{40}", "{addr}", path, flags=re.I)
    path = re.sub(r"[a-f0-9]{32,64}", "{hex}", path, flags=re.I)
    path = re.sub(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", "{uuid}", path, flags=re.I)
    path = re.sub(r"/\d{3,}(?=/|$)", "/{num}", path)
    path = re.sub(r"/\d+(?=/|$)", "/{n}", path)
    path = re.sub(r"/{2,}", "/", path).rstrip("/") or "/"
    qnames = _normalise_query_names(query)
    if qnames:
        path += "?" + "&".join(qnames[:12])
    return path


def canonical_backend_endpoint(raw_url: str, primary_domain: str) -> str:
    raw_url = html.unescape(str(raw_url or "").strip())
    if not raw_url or raw_url.startswith(("data:", "blob:", "javascript:", "mailto:", "tel:")):
        return ""
    parsed = urlparse(raw_url)
    if not parsed.netloc and raw_url.startswith("//"):
        parsed = urlparse("https:" + raw_url)
    host = parsed.netloc.lower().split("@")[-1].split(":", 1)[0]
    path = parsed.path or "/"
    suffix = pathlib.PurePosixPath(path).suffix.lower()
    if suffix in BACKEND_STATIC_EXTS:
        return ""
    if host and _host_is_public_utility(host):
        return ""
    endpoint_path = _normalise_backend_path(path, parsed.query)
    if endpoint_path in {"/", "/index.html", "/index.php", "/home", "/main"}:
        # Landing pages are frontend/presentation, not backend-kit routes.
        return ""
    primary = (primary_domain or "").lower().split(":", 1)[0]
    if host and not (host == primary or host.endswith("." + primary)):
        return f"{host}{endpoint_path}"
    return endpoint_path


def _clean_backend_field_name(name: str) -> str:
    name = normalize_text(str(name or "")).replace(" ", "_")
    name = re.sub(r"_+", "_", name).strip("_")[:64]
    if not name or name in BACKEND_OBJECT_IGNORE_KEYS:
        return ""
    if name.startswith(("html_", "head_", "body_", "script_", "style_", "meta_", "title_")):
        return ""
    return name


def backend_field_is_schema_signal(field: str) -> bool:
    field = _clean_backend_field_name(field)
    if not field:
        return False
    if len(field) > 42 or "$" in field:
        return False
    if field.count("_") > 5 and not field.startswith(("auth_", "wallet_", "chain_", "invite_")):
        return False
    if BACKEND_FIELD_HINT_RE.match(field):
        return True
    if re.search(r"(?:^|_)(?:num|amount|money|balance|assets?|token|symbol|chain|invite|auth|address|wallet|account|signature|password|email|otp|seed|mnemonic|code|msg)(?:$|_)", field, re.I):
        return True
    return False


def _collect_json_keys(obj: object, out: set[str], depth: int = 0) -> None:
    if depth > 3:
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            clean = _clean_backend_field_name(str(key))
            if clean and clean not in BACKEND_OBJECT_IGNORE_KEYS:
                out.add(clean)
            _collect_json_keys(value, out, depth + 1)
    elif isinstance(obj, list):
        for value in obj[:20]:
            _collect_json_keys(value, out, depth + 1)


def extract_backend_field_names(text: str) -> list[str]:
    text = html.unescape(str(text or ""))
    if not text:
        return []
    fields: set[str] = set()
    stripped = text.strip()
    if stripped[:1] in {"{", "["}:
        try:
            obj = json.loads(stripped)
            _collect_json_keys(obj, fields)
        except Exception:
            pass
    if "=" in stripped and not fields:
        for key, _ in parse_qsl(stripped, keep_blank_values=True):
            clean = _clean_backend_field_name(key)
            if clean and clean not in BACKEND_OBJECT_IGNORE_KEYS:
                fields.add(clean)
    # JavaScript object literal / data:{...} fallback.
    for key in re.findall(r"[\{,]\s*[\"']?([A-Za-z_$][\w$-]{1,64})[\"']?\s*:", text[:20_000]):
        clean = _clean_backend_field_name(key)
        if clean and clean not in BACKEND_OBJECT_IGNORE_KEYS:
            fields.add(clean)
    # Form fallback.
    for key in re.findall(r"\bname\s*=\s*[\"']([^\"']{1,80})[\"']", text[:20_000], flags=re.I):
        clean = _clean_backend_field_name(key)
        if clean:
            fields.add(clean)
    return sorted(fields)[:40]


def extract_response_schema_keys(text: str, mime: str = "") -> list[str]:
    text = str(text or "").strip()
    if not text or len(text) > 120_000:
        return []
    keys: set[str] = set()
    if "json" in str(mime or "").lower() or text[:1] in {"{", "["}:
        try:
            obj = json.loads(text)
            _collect_json_keys(obj, keys)
        except Exception:
            pass
    if not keys:
        # Lightweight fallback for JSON-like API responses.
        for key in re.findall(r"[\{,]\s*[\"']([A-Za-z_$][\w$-]{1,64})[\"']\s*:", text[:20_000]):
            clean = _clean_backend_field_name(key)
            if clean and clean not in BACKEND_OBJECT_IGNORE_KEYS:
                keys.add(clean)
    return sorted(keys)[:40]


def classify_backend_role(endpoint: str, fields: Sequence[str] = (), response_keys: Sequence[str] = ()) -> str:
    material = " ".join([endpoint or "", *fields, *response_keys]).lower()
    for role, pattern in BACKEND_ROLE_PATTERNS:
        if pattern.search(material):
            return role
    return "generic_backend"


def is_public_endpoint_token(endpoint: str) -> bool:
    endpoint = str(endpoint or "").lower()
    if not endpoint or endpoint.startswith("/"):
        return False
    host = endpoint.split("/", 1)[0].split(":", 1)[0]
    return _host_is_public_utility(host)


def is_json_rpc_backend_noise(endpoint: str, fields: Sequence[str] = (), context: str = "") -> bool:
    endpoint_l = str(endpoint or "").lower()
    field_set = {_clean_backend_field_name(f) for f in fields}
    context_l = str(context or "").lower()[:20_000]
    if re.search(r"(?:^|/)(?:rpc|jsonrpc)(?:$|[/?#])", endpoint_l):
        if {"jsonrpc", "id", "method"} & field_set or re.search(r"\b(?:eth_|wallet_|net_|web3_)", context_l):
            return True
    if "jsonrpc" in context_l and re.search(r"[\"']method[\"']\s*:\s*[\"'](?:eth_|wallet_|net_|web3_)", context_l):
        return True
    return False


def backend_endpoint_has_signal(endpoint: str, method: str = "", fields: Sequence[str] = (), context: str = "") -> bool:
    endpoint = str(endpoint or "")
    if not endpoint:
        return False
    path = endpoint.split("/", 1)[1] if re.match(r"^[a-z0-9.-]+/", endpoint) else endpoint
    path_only = path.split("?", 1)[0]
    suffix = pathlib.PurePosixPath(path_only).suffix.lower()
    if suffix in BACKEND_STATIC_EXTS:
        return False
    if endpoint.startswith(("api.ipify.org", "ipinfo.io", "ip-api.com", "api.myip.com")):
        return False
    method_u = str(method or "").upper()
    field_set = {_clean_backend_field_name(f) for f in fields}
    if field_set & {"auth_address", "authorized_address", "address", "wallet", "account", "bizhong", "token", "chainid", "chain_id", "amount", "balance", "money", "signature", "password", "email", "otp", "seed", "mnemonic"}:
        return True
    if any(BACKEND_FIELD_HINT_RE.match(f or "") for f in field_set):
        return True
    if method_u in {"POST", "PUT", "PATCH", "DELETE"} and not endpoint.endswith((".js", ".css")):
        return True
    if BACKEND_ROUTE_HINT_RE.search(endpoint):
        return True
    if CAPABILITY_PATTERNS["backend_exfil"].search(context or ""):
        return True
    return False


def _backend_observation_key(obs: dict) -> tuple:
    return (
        str(obs.get("source", "")),
        str(obs.get("method", "")).upper(),
        str(obs.get("endpoint", "")),
        tuple(obs.get("request_fields", []) or []),
        tuple(obs.get("response_keys", []) or []),
    )


def dedupe_backend_observations(observations: Sequence[dict]) -> list[dict]:
    seen: set[tuple] = set()
    out: list[dict] = []
    for obs in observations:
        if not obs.get("endpoint"):
            continue
        key = _backend_observation_key(obs)
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(obs))
    return out


def extract_backend_observations_from_text(text: str, primary_domain: str) -> list[dict]:
    text = str(text or "")
    observations: list[dict] = []
    patterns = [API_RE, JQUERY_AJAX_URL_RE, GENERIC_URL_LITERAL_RE]
    for pattern in patterns:
        for match in pattern.finditer(text):
            raw = match.group(1)
            endpoint = canonical_backend_endpoint(raw, primary_domain)
            if not endpoint:
                continue
            ctx = text[max(0, match.start() - 1200): min(len(text), match.end() + 2200)]
            method = ""
            method_match = re.search(r"\b(?:type|method)\s*:\s*[\"']?(GET|POST|PUT|PATCH|DELETE)[\"']?", ctx, re.I)
            if method_match:
                method = method_match.group(1).upper()
            elif pattern is JQUERY_AJAX_URL_RE:
                method = "POST" if re.search(r"\bdata\s*:", ctx, re.I) else "GET"
            fields = [f for f in extract_backend_field_names(ctx) if backend_field_is_schema_signal(f)]
            if is_json_rpc_backend_noise(endpoint, fields, ctx):
                continue
            if not backend_endpoint_has_signal(endpoint, method, fields, ctx):
                continue
            role = classify_backend_role(endpoint, fields)
            callsite_material = normalize_javascript_for_hash(ctx)[:3000]
            observations.append({
                "source": "js",
                "order": int(match.start()),
                "method": method or "UNKNOWN",
                "endpoint": endpoint,
                "request_fields": fields,
                "response_keys": [],
                "role": role,
                "callsite_hash": short_hash(f"{endpoint}|{role}|{','.join(fields)}|{callsite_material}", n=14),
            })
    return dedupe_backend_observations(observations)


def extract_backend_observations_from_har(
    sample_dir: pathlib.Path,
    primary_domain: str,
    include_har: bool,
    session_path: pathlib.Path | None = None,
    har_path: pathlib.Path | None = None,
) -> list[dict]:
    if not include_har:
        return []
    resolved_har = find_har_path(sample_dir, session_path=session_path, har_path=har_path)
    if resolved_har is None:
        return []
    try:
        har = json.loads(read_text(resolved_har))
    except Exception:
        return []
    observations: list[dict] = []
    for idx, entry in enumerate(har.get("log", {}).get("entries", []) or []):
        request = entry.get("request", {}) or {}
        response = entry.get("response", {}) or {}
        raw_url = str(request.get("url") or "")
        if not raw_url or is_extension_or_wallet_url(raw_url):
            continue
        endpoint = canonical_backend_endpoint(raw_url, primary_domain)
        if not endpoint:
            continue
        method = str(request.get("method") or "GET").upper()
        post_text = str(((request.get("postData", {}) or {}).get("text") or ""))
        parsed = urlparse(raw_url)
        query_fields = _normalise_query_names(parsed.query)
        raw_request_fields = sorted(set(query_fields + extract_backend_field_names(post_text)))
        request_fields = [f for f in raw_request_fields if backend_field_is_schema_signal(f) or f in {"lang", "locale"}][:40]
        content = response.get("content", {}) or {}
        mime = str(content.get("mimeType") or "")
        response_text = decode_har_content(content)
        response_keys = extract_response_schema_keys(response_text, mime)
        ctx = post_text + "\n" + response_text[:4000]
        if is_json_rpc_backend_noise(endpoint, request_fields, ctx):
            continue
        if not backend_endpoint_has_signal(endpoint, method, request_fields, ctx):
            continue
        role = classify_backend_role(endpoint, request_fields, response_keys)
        observations.append({
            "source": "har",
            "order": int(idx),
            "method": method,
            "endpoint": endpoint,
            "request_fields": request_fields,
            "response_keys": response_keys,
            "role": role,
            "status": int(response.get("status") or 0),
            "mime": mime.split(";", 1)[0].lower(),
            "callsite_hash": "",
        })
    return dedupe_backend_observations(observations)


def canonical_backend_field_classes(fields: Sequence[str]) -> list[str]:
    classes: set[str] = set()
    for raw in fields or []:
        field = _clean_backend_field_name(str(raw))
        if not field:
            continue
        if field in {"lang", "locale"}:
            classes.add("locale")
        elif "invite" in field:
            classes.add("invite_code")
        elif re.search(r"auth|authori[sz]ed|spender|operator", field, re.I):
            classes.add("auth_or_spender_address")
        elif re.search(r"address|wallet|account|^from$|^to$", field, re.I):
            classes.add("wallet_address")
        elif re.search(r"bizhong|symbol|token|contract|asset", field, re.I):
            classes.add("asset_identifier")
        elif re.search(r"num|amount|money|balance|price|total|value", field, re.I):
            classes.add("asset_value")
        elif re.search(r"signature|tx|hash", field, re.I):
            classes.add("tx_or_signature")
        elif re.search(r"email|password|phone|otp|code|seed|mnemonic|private", field, re.I):
            classes.add("credential_or_status")
        elif field in {"code", "msg", "message", "status"}:
            classes.add("response_status")
        elif backend_field_is_schema_signal(field):
            classes.add(field)
    return sorted(classes)[:20]


def backend_observation_schema(obs: dict) -> str:
    fields = ",".join(canonical_backend_field_classes(obs.get("request_fields") or []))
    response_keys = ",".join(canonical_backend_field_classes(obs.get("response_keys") or []))
    return f"{str(obs.get('method', 'UNKNOWN')).upper()} {obs.get('endpoint', '')} | role={obs.get('role', '')} | fields={fields} | response={response_keys}"


def build_backend_material(observations: Sequence[dict]) -> dict:
    endpoints = sorted({str(obs.get("endpoint")) for obs in observations if obs.get("endpoint")})
    schemas = sorted({backend_observation_schema(obs) for obs in observations if obs.get("endpoint")})
    ordered = sorted(observations, key=lambda obs: (str(obs.get("source")), int(obs.get("order", 0))))
    role_sequence = [f"{obs.get('role', 'generic_backend')}:{obs.get('endpoint', '')}" for obs in ordered if obs.get("endpoint")]
    callsites = sorted({str(obs.get("callsite_hash")) for obs in observations if obs.get("callsite_hash")})
    roles = sorted({str(obs.get("role")) for obs in observations if obs.get("role")})
    route_set_hashes: set[str] = set()
    schema_hashes: set[str] = set()
    flow_hashes: set[str] = set()
    backend_kit_hashes: set[str] = set()
    if endpoints:
        route_set_hashes.add(short_hash(json.dumps(endpoints[:40], sort_keys=True), n=16))
    for schema in schemas:
        schema_hashes.add(short_hash(schema, n=16))
    if len(role_sequence) >= 2:
        flow_hashes.add(short_hash(json.dumps(role_sequence[:50], ensure_ascii=False), n=16))
    # A backend-kit key must be backend-defined: route set + request/response
    # schema and, where available, role/flow order. Source hashes and spender
    # addresses are deliberately excluded so frontend/operator reuse does not
    # collapse different server kits.
    if endpoints and schemas and (len(endpoints) >= 2 or len(schemas) >= 2 or flow_hashes):
        kit_material = {
            "endpoints": endpoints[:40],
            "schemas": schemas[:40],
            "roles": roles,
            "flow": role_sequence[:50],
        }
        backend_kit_hashes.add(short_hash(json.dumps(kit_material, sort_keys=True, ensure_ascii=False), n=18))
    return {
        "endpoints": endpoints,
        "schemas": schemas,
        "roles": roles,
        "role_sequence": role_sequence,
        "route_set_hashes": sorted(route_set_hashes),
        "schema_hashes": sorted(schema_hashes),
        "flow_hashes": sorted(flow_hashes),
        "callsite_hashes": sorted(callsites),
        "backend_kit_hashes": sorted(backend_kit_hashes),
    }


def dom_body_text(root: pathlib.Path) -> str:
    candidates = [
        root / "dom_snapshots" / "connect_confirmed.json",
        root / "dom_snapshots" / "wallet_prompt_post_connect_watch.json",
        root / "dom_snapshots" / "metamask_option_clicked.json",
        root / "dom_snapshots" / "wallet_choice_modal.json",
        root / "dom_snapshots" / "initial_page_loaded.json",
        root / "label_only" / "dom" / "connect_confirmed.json",
        root / "label_only" / "dom" / "wallet_prompt_post_connect_watch.json",
        root / "label_only" / "dom" / "metamask_option_clicked.json",
        root / "label_only" / "dom" / "wallet_choice_modal.json",
        root / "label_only" / "dom" / "initial_page_loaded.json",
        root / "label_only" / "page_state" / "connect_confirmed.json",
        root / "label_only" / "page_state" / "initial_page_loaded.json",
    ]
    for path in candidates:
        obj = read_json(path)
        text = (
            obj.get("body_text_excerpt")
            or obj.get("body_text")
            or obj.get("visible_text")
            or obj.get("html_excerpt")
            or obj.get("html")
            or ""
        )
        if text:
            return str(text)
    return ""


def html_title(root: pathlib.Path) -> str:
    html_dirs = [root / "html", root / "label_only" / "html"]
    for html_dir in html_dirs:
        for path in sorted(html_dir.glob("*.html")) if html_dir.exists() else []:
            text = read_text(path)
            m = TITLE_RE.search(text)
            if m:
                return normalize_text(m.group(1))[:120]
    json_dirs = [root / "dom_snapshots", root / "label_only" / "dom", root / "label_only" / "page_state"]
    for json_dir in json_dirs:
        for path in sorted(json_dir.glob("*.json")) if json_dir.exists() else []:
            obj = read_json(path)
            title = obj.get("title")
            if title:
                return normalize_text(str(title))[:120]
    return ""


def token_set_from_body(text: str) -> set[str]:
    norm = normalize_text(text)
    toks = [t for t in norm.split() if len(t) >= 3]
    return set(toks[:200])


def family_features(
    sample_dir: pathlib.Path,
    include_har: bool = True,
    session_path: pathlib.Path | None = None,
    har_path: pathlib.Path | None = None,
) -> tuple[dict, set[str]]:
    """Extract static kit-family evidence from a capture directory.

    The extractor deliberately separates high-evidence kit/operator anchors
    (source hashes, spender/operator addresses, WalletConnect verification IDs,
    backend API routes) from weak presentation cues (title/body/template). The
    downstream Layer-2 label builder can then require independent channels before
    treating a repeated static key as a strong pseudo-label.
    """
    session = read_capture_session(sample_dir, session_path=session_path)
    sample = session.get("sample", {}) or {}
    primary_domain = str(session.get("domain") or urlparse(str(sample.get("url", ""))).netloc or sample_dir.name).lower()
    sample_id = str(sample.get("sample_id") or session.get("session_id") or sample_dir.name)
    status = str(session.get("status") or "")
    flags = session.get("flags", {}) or {}

    text_parts: list[str] = []
    html_docs: list[str] = []
    inline_scripts: list[str] = []
    first_party_js: list[tuple[str, str]] = []
    script_tokens: set[str] = set()
    local_script_paths: set[str] = set()
    resource_sequence: list[str] = []
    image_hashes: set[str] = set()
    rpc_methods: set[str] = set()

    # Parse HAR with response bodies explicitly. Adding browser.har wholesale to
    # text_parts weakens evidence with extension/internal noise and prevents
    # response-type-specific hashes.
    har_resources = extract_har_site_resources(
        sample_dir,
        primary_domain,
        include_har=include_har,
        session_path=session_path,
        har_path=har_path,
    )
    text_parts.extend(har_resources.get("text_parts", []))
    html_docs.extend(har_resources.get("html_docs", []))
    inline_scripts.extend(har_resources.get("inline_scripts", []))
    first_party_js.extend(har_resources.get("first_party_js", []))
    script_tokens.update(har_resources.get("script_tokens", set()))
    local_script_paths.update(har_resources.get("local_script_paths", set()))
    resource_sequence.extend(har_resources.get("resource_sequence", []))
    image_hashes.update(har_resources.get("image_hashes", set()))
    rpc_methods.update(har_resources.get("rpc_methods", set()))

    for path in iter_text_files(sample_dir, exclude_exts={".har"}):
        text = read_text(path)
        if not text:
            continue
        low_rel = str(path.relative_to(sample_dir)).replace("\\", "/").lower()
        if "/metamask_" in low_rel or low_rel.startswith("html/metamask_"):
            continue
        text_parts.append(text[:500_000])
        if path.suffix.lower() in {".html", ".htm"}:
            html_docs.append(text[:500_000])
            for src in SCRIPT_RE.findall(text):
                token = norm_url_token(src, primary_domain)
                if token and useful_script(token):
                    script_tokens.add(token)
                    if not re.match(r"^[a-z0-9.-]+\.", token):
                        local_script_paths.add(token)
            for block in INLINE_SCRIPT_BLOCK_RE.findall(text):
                if block and len(block.strip()) >= 80:
                    inline_scripts.append(block)
        elif path.suffix.lower() == ".js":
            token = "/" + low_rel if not low_rel.startswith("/") else low_rel
            if useful_script(token) and not likely_common_js_library(token, text):
                first_party_js.append((token, text[:500_000]))

    blob = "\n".join(text_parts)

    body = dom_body_text(sample_dir)
    if not body and html_docs:
        body = html_to_visible_text("\n".join(html_docs[:3]))[:5000]
    body_norm = normalize_text(body)
    title = html_title(sample_dir)
    if not title:
        for doc in html_docs[:5]:
            m = TITLE_RE.search(doc)
            if m:
                title = normalize_text(m.group(1))[:120]
                break

    methods: set[str] = set()
    for match in METHOD_RE.finditer(blob):
        raw = match.group(0).lower()
        token = "approve_call" if raw.startswith("approve") else raw
        methods.add(token)
    methods.update(str(method).lower() for method in rpc_methods if method)
    methods_list = sorted(methods)

    all_addresses = sorted({a.lower() for a in ADDRESS_RE.findall(blob)} - COMMON_CAPTURE_ADDRESSES)
    evidence_addresses = [a for a in all_addresses if a not in COMMON_TOKEN_ADDRESSES]
    project_ids = sorted({p.lower() for p in PROJECT_ID_RE.findall(blob)} - COMMON_RPC_PROJECT_IDS)
    verify_ids = sorted({p.lower() for p in WC_VERIFY_RE.findall(blob)})
    infura_keys = sorted({p.lower() for p in INFURA_KEY_RE.findall(blob)} - COMMON_RPC_PROJECT_IDS)
    drainer_hits = sorted({normalize_text(m.group(1))[:80] for m in DRainer_RE.finditer(blob)})
    obf_bundles = sorted({m.group(0).lower().lstrip("/") for m in OBF_BUNDLE_RE.finditer(blob)})
    api_endpoints = [endpoint for endpoint in extract_api_endpoints(blob, primary_domain) if not is_public_endpoint_token(endpoint)]
    backend_observations = dedupe_backend_observations(
        extract_backend_observations_from_har(
            sample_dir,
            primary_domain,
            include_har=include_har,
            session_path=session_path,
            har_path=har_path,
        )
        + extract_backend_observations_from_text(blob, primary_domain)
    )
    backend_material = build_backend_material(backend_observations)
    backend_endpoints = backend_material["endpoints"]
    drainer_spenders = sorted(extract_drainer_spenders(blob))
    js_capabilities = sorted(detect_capabilities(blob))
    interaction_pattern = classify_interaction_pattern(blob, set(js_capabilities), backend_endpoints)
    claim_words = sorted({m.group(1).lower() for m in CLAIM_WORD_RE.finditer(body + "\n" + title)})

    inline_script_hashes: set[str] = set()
    for block in inline_scripts:
        norm = normalize_javascript_for_hash(block)
        if len(norm) < 120:
            continue
        block_caps = detect_capabilities(block)
        high_signal = bool(block_caps & {"wallet_api", "walletconnect", "token_approval", "tx_signing", "backend_exfil", "balance_sweep"})
        if high_signal or re.search(r"\b(?:approve|auth_address|save_erc_data|eth_sendTransaction|WalletConnect)\b", block, re.I):
            inline_script_hashes.add(short_hash(norm, n=16))

    script_content_hashes: set[str] = set()
    for token, js_text in first_party_js:
        norm = normalize_javascript_for_hash(js_text)
        if len(norm) < 120:
            continue
        block_caps = detect_capabilities(js_text)
        if block_caps or re.search(r"\b(?:approve|permit|eth_sendTransaction|WalletConnect|auth_address)\b", js_text, re.I):
            script_content_hashes.add(short_hash(f"{token}\n{norm}", n=16))

    html_structure_hashes: set[str] = set()
    for doc in html_docs[:10]:
        norm_html = normalize_html_structure(doc)
        if len(norm_html) >= 80:
            html_structure_hashes.add(short_hash(norm_html, n=14))

    resource_graph_hashes: set[str] = set()
    graph_tokens = [token for token in resource_sequence if token]
    if len(graph_tokens) >= 3:
        resource_graph_hashes.add(short_hash("\n".join(graph_tokens[:80]), n=14))

    kit_behavior_hashes: set[str] = set()
    high_risk_methods = [
        m for m in methods_list
        if m in {
            "eth_sendtransaction", "eth_signtypeddata", "eth_signtypeddata_v4", "eth_signtypeddata_v3", "personal_sign",
            "eth_sign", "approve_call", "setapprovalforall", "increaseallowance", "permit2",
            "permitsingle", "permitbatch",
        }
    ]
    behavior_material = {
        "interaction": interaction_pattern,
        "capabilities": js_capabilities,
        "high_risk_methods": high_risk_methods,
        "backend_api": backend_endpoints[:8],
        "has_spender": bool(drainer_spenders),
        "has_walletconnect_verify": bool(verify_ids),
    }
    if js_capabilities or high_risk_methods or backend_endpoints or drainer_spenders or verify_ids:
        kit_behavior_hashes.add(short_hash(json.dumps(behavior_material, sort_keys=True), n=14))

    features: set[str] = set()
    for h in backend_material["backend_kit_hashes"]:
        features.add(f"backend_kit:{h}")
    for h in backend_material["schema_hashes"]:
        features.add(f"backend_request_schema_hash:{h}")
    for h in backend_material["flow_hashes"]:
        features.add(f"backend_flow_hash:{h}")
    for h in backend_material["route_set_hashes"]:
        features.add(f"backend_route_set_hash:{h}")
    for h in backend_material["callsite_hashes"][:20]:
        features.add(f"backend_callsite_hash:{h}")
    for endpoint in backend_endpoints[:80]:
        features.add(f"backend_endpoint:{endpoint}")
        if endpoint and not endpoint.startswith("/"):
            features.add(f"backend_host_pattern:{endpoint.split('/', 1)[0]}")
    for role in backend_material["roles"]:
        features.add(f"backend_role:{role}")
    for p in project_ids:
        features.add(f"walletconnect_project:{p}")
    for p in verify_ids:
        features.add(f"walletconnect_verify:{p}")
    for p in infura_keys:
        features.add(f"infura_project:{p}")
    for a in evidence_addresses[:80]:
        features.add(f"address:{a}")
    for a in drainer_spenders[:40]:
        features.add(f"drainer_spender:{a}")
    for m in methods_list:
        features.add(f"method:{m}")
    if high_risk_methods:
        features.add("danger_method_combo:" + "+".join(sorted(high_risk_methods)))
    for d in drainer_hits:
        if "assets js drainer" in d or "drainer js" in d:
            features.add(f"drainer_path:{d}")
        else:
            features.add(f"drainer_symbol:{d}")
    for s in sorted(script_tokens):
        if "drainer" in s:
            features.add(f"drainer_path:{s}")
        elif s in local_script_paths:
            features.add(f"script_local:{s}")
        else:
            features.add(f"script_host_path:{s}")
    for h in sorted(inline_script_hashes):
        features.add(f"inline_script_hash:{h}")
    for h in sorted(script_content_hashes):
        features.add(f"script_content_hash:{h}")
    for h in sorted(resource_graph_hashes):
        features.add(f"resource_graph_hash:{h}")
    for h in sorted(html_structure_hashes):
        features.add(f"html_structure_hash:{h}")
    for h in sorted(image_hashes)[:30]:
        features.add(f"image_asset_hash:{h}")
    if js_capabilities:
        profile_hash = short_hash("\n".join(js_capabilities), n=14)
        features.add(f"js_capability_profile:{profile_hash}")
        for cap in js_capabilities:
            features.add(f"js_capability:{cap}")
    if interaction_pattern and interaction_pattern != "unknown":
        features.add(f"interaction_pattern:{interaction_pattern}")
    for h in sorted(kit_behavior_hashes):
        features.add(f"kit_behavior_hash:{h}")
    for b in obf_bundles[:20]:
        features.add(f"obf_bundle:{b}")
    for a in api_endpoints[:50]:
        if not is_public_endpoint_token(a):
            features.add(f"api_endpoint:{a}")
    if title:
        features.add(f"title:{title}")
    if body_norm:
        features.add(f"body_template:{short_hash(body_norm[:2500])}")
    if claim_words:
        features.add("claim_words:" + "+".join(claim_words[:12]))

    meta = {
        "sample_id": sample_id,
        "domain": primary_domain,
        "url": sample.get("url") or "",
        "status": status,
        "connect_confirmed": bool(flags.get("connect_confirmed")),
        "signature_prompt_seen": bool(flags.get("signature_prompt_seen")),
        "high_risk_wallet_prompt_seen": bool(flags.get("high_risk_wallet_prompt_seen")),
        "title": title,
        "body_template_hash": short_hash(body_norm[:2500]) if body_norm else "",
        "body_tokens": sorted(token_set_from_body(body)),
        "methods": methods_list,
        "addresses": evidence_addresses[:80],
        "all_addresses": all_addresses[:120],
        "walletconnect_project_ids": project_ids,
        "walletconnect_verify_ids": verify_ids,
        "infura_project_ids": infura_keys,
        "drainer_hits": drainer_hits,
        "drainer_spenders": drainer_spenders,
        "script_tokens": sorted(script_tokens)[:160],
        "api_endpoints": api_endpoints[:80],
        "backend_endpoints": backend_endpoints[:80],
        "backend_observations": backend_observations[:120],
        "backend_request_schemas": backend_material["schemas"][:80],
        "backend_roles": backend_material["roles"],
        "backend_role_sequence": backend_material["role_sequence"][:80],
        "backend_route_set_hashes": backend_material["route_set_hashes"],
        "backend_request_schema_hashes": backend_material["schema_hashes"],
        "backend_flow_hashes": backend_material["flow_hashes"],
        "backend_callsite_hashes": backend_material["callsite_hashes"],
        "backend_kit_hashes": backend_material["backend_kit_hashes"],
        "claim_words": claim_words,
        "js_capabilities": js_capabilities,
        "interaction_pattern": interaction_pattern,
        "inline_script_hashes": sorted(inline_script_hashes),
        "script_content_hashes": sorted(script_content_hashes),
        "resource_graph_hashes": sorted(resource_graph_hashes),
        "html_structure_hashes": sorted(html_structure_hashes),
        "image_asset_hashes": sorted(image_hashes)[:30],
        "kit_behavior_hashes": sorted(kit_behavior_hashes),
        "rpc_methods": sorted(rpc_methods),
    }
    return meta, features

def feature_type(feature: str) -> str:
    return feature.split(":", 1)[0]


def weighted_shared_score(shared_features: list[str], coverage: float, uniqueish_count: int) -> float:
    weight = sum(FEATURE_WEIGHTS.get(feature_type(f), 1.0) for f in shared_features)
    return weight * coverage + 1.5 * uniqueish_count


def pairwise_jaccard(sets: list[set[str]]) -> float:
    if len(sets) < 2:
        return 1.0
    vals = []
    for a, b in combinations(sets, 2):
        union = a | b
        vals.append(len(a & b) / len(union) if union else 0.0)
    return float(mean(vals)) if vals else 0.0


def verdict_for(row: dict) -> str:
    if row["n_samples"] < 2:
        return "insufficient_cluster_size"
    if row["strong_unique_shared_count"] >= 2 and row["shared_coverage_max"] >= 0.5:
        return "strong_same_drainer_or_kit"
    if row["strong_unique_shared_count"] >= 1 and row["shared_coverage_max"] >= 0.67:
        return "strong_same_drainer_or_kit"
    if row["shared_feature_count"] >= 3 and row["shared_coverage_max"] >= 0.5:
        return "moderate_same_template_or_infra"
    if row["body_token_jaccard_mean"] >= 0.45 and row["n_samples"] <= 5:
        return "moderate_same_template_small_cluster"
    return "weak_or_mixed_cluster"


def audit(args: argparse.Namespace) -> None:
    pred = pd.read_csv(args.predictions, low_memory=False)
    cluster_col = args.cluster_col
    pred = pred[pred[cluster_col].astype(str).str.startswith("known_phish_")].copy()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_rows = []
    sample_features: dict[str, set[str]] = {}
    sample_body_sets: dict[str, set[str]] = {}
    cluster_samples: dict[str, list[str]] = defaultdict(list)
    feature_to_clusters: dict[str, set[str]] = defaultdict(set)

    for _, row in pred.iterrows():
        cluster = str(row[cluster_col])
        sample_dir = pathlib.Path(str(row["zip_path"]))
        if not sample_dir.exists():
            continue
        meta, features = family_features(sample_dir)
        sample_key = f"{cluster}:{meta['sample_id']}:{meta['domain']}"
        sample_features[sample_key] = features
        sample_body_sets[sample_key] = set(meta.pop("body_tokens", []))
        cluster_samples[cluster].append(sample_key)
        for f in features:
            feature_to_clusters[f].add(cluster)
        sample_rows.append({
            "cluster": cluster,
            "sample_key": sample_key,
            "zip_path": str(sample_dir),
            "n_family_features": len(features),
            "family_features_json": json.dumps(sorted(features), ensure_ascii=False),
            **{k: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v for k, v in meta.items()},
        })

    # Second pass after global feature-to-cluster map is known.
    cluster_rows = []
    shared_rows = []
    for cluster, keys in sorted(cluster_samples.items()):
        n = len(keys)
        counters = Counter()
        for key in keys:
            counters.update(sample_features[key])
        min_count = max(2, int(np_ceil(n * float(args.min_coverage)))) if n > 2 else n
        shared = [f for f, c in counters.items() if c >= min_count]
        shared.sort(key=lambda f: (-(counters[f] / n), len(feature_to_clusters[f]), feature_type(f), f))
        uniqueish = [f for f in shared if len(feature_to_clusters[f]) <= int(args.max_cross_clusters)]
        strong_unique = [
            f for f in uniqueish
            if feature_type(f) in {
                "compound_family", "inline_script_hash", "script_content_hash", "drainer_spender", "drainer_contract",
                "drainer_symbol", "drainer_path", "kit_behavior_hash", "resource_graph_hash", "danger_method_combo",
                "walletconnect_project", "walletconnect_verify", "infura_project", "address", "api_endpoint",
            }
        ]
        max_cov = max((counters[f] / n for f in shared), default=0.0)
        body_j = pairwise_jaccard([sample_body_sets[k] for k in keys])
        score = weighted_shared_score(uniqueish, max_cov, len(strong_unique))
        row = {
            "cluster": cluster,
            "n_samples": n,
            "shared_feature_count": len(shared),
            "uniqueish_shared_count": len(uniqueish),
            "strong_unique_shared_count": len(strong_unique),
            "shared_coverage_max": max_cov,
            "body_token_jaccard_mean": body_j,
            "family_score": score,
            "top_shared_features": "\n".join(
                f"{f} | coverage={counters[f]}/{n} | clusters={len(feature_to_clusters[f])}"
                for f in uniqueish[:20]
            ),
            "top_all_shared_features": "\n".join(
                f"{f} | coverage={counters[f]}/{n} | clusters={len(feature_to_clusters[f])}"
                for f in shared[:30]
            ),
            "representative_samples": "\n".join(keys[:8]),
        }
        row["verdict"] = verdict_for(row)
        cluster_rows.append(row)
        for f in shared:
            shared_rows.append({
                "cluster": cluster,
                "feature": f,
                "feature_type": feature_type(f),
                "sample_coverage": counters[f],
                "n_samples": n,
                "coverage_frac": counters[f] / n,
                "n_clusters_with_feature": len(feature_to_clusters[f]),
                "clusters_with_feature": ",".join(sorted(feature_to_clusters[f])),
                "uniqueish": len(feature_to_clusters[f]) <= int(args.max_cross_clusters),
            })

    sample_df = pd.DataFrame(sample_rows).sort_values(["cluster", "sample_id"])
    cluster_df = pd.DataFrame(cluster_rows).sort_values(["verdict", "family_score"], ascending=[True, False])
    shared_df = pd.DataFrame(shared_rows).sort_values(["cluster", "uniqueish", "coverage_frac"], ascending=[True, False, False])

    sample_df.to_csv(out_dir / "cluster_family_sample_fingerprints.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    cluster_df.to_csv(out_dir / "cluster_family_evidence.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    shared_df.to_csv(out_dir / "cluster_shared_family_features.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    md = ["# Cluster Family Evidence", ""]
    md.append("This report checks whether each traffic cluster maps to the same drainer kit/template/infrastructure family.")
    md.append("")
    for _, row in cluster_df.sort_values("cluster").iterrows():
        md.append(f"## {row['cluster']} - {row['verdict']}")
        md.append(f"- samples: {row['n_samples']}")
        md.append(f"- shared features: {row['shared_feature_count']}, unique-ish shared: {row['uniqueish_shared_count']}, strong unique-ish shared: {row['strong_unique_shared_count']}")
        md.append(f"- max shared coverage: {row['shared_coverage_max']:.2f}")
        md.append(f"- body-token mean Jaccard: {row['body_token_jaccard_mean']:.3f}")
        md.append("- top unique-ish shared fingerprints:")
        top = str(row["top_shared_features"]).splitlines()[:10]
        if top:
            for line in top:
                md.append(f"  - {line}")
        else:
            md.append("  - none")
        md.append("- representative samples:")
        for line in str(row["representative_samples"]).splitlines()[:6]:
            md.append(f"  - {line}")
        md.append("")
    (out_dir / "cluster_family_evidence_report.md").write_text("\n".join(md), encoding="utf-8")

    print(cluster_df[[
        "cluster", "n_samples", "verdict", "uniqueish_shared_count", "strong_unique_shared_count",
        "shared_coverage_max", "body_token_jaccard_mean", "family_score",
    ]].sort_values("cluster").to_string(index=False))
    print(f"wrote {out_dir}")


def np_ceil(x: float) -> int:
    return int(-(-x // 1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether clustered phishing samples share the same drainer family fingerprints.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cluster-col", default="known_cluster")
    parser.add_argument("--min-coverage", type=float, default=0.50)
    parser.add_argument("--max-cross-clusters", type=int, default=2)
    audit(parser.parse_args())


if __name__ == "__main__":
    main()

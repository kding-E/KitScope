from __future__ import annotations

import json
import ipaddress
import math
import pathlib
import socket
import struct
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

DLT_NULL = 0
DLT_EN10MB = 1
DLT_RAW = 101
DLT_LINUX_SLL = 113
DLT_LINUX_SLL2 = 276

PCAPNG_SECTION_HEADER = b"\x0a\x0d\x0d\x0a"
PCAPNG_BYTE_ORDER_MAGIC_LE = b"\x4d\x3c\x2b\x1a"
PCAPNG_BYTE_ORDER_MAGIC_BE = b"\x1a\x2b\x3c\x4d"
PCAPNG_BLOCK_IDB = 0x00000001
PCAPNG_BLOCK_PB = 0x00000002
PCAPNG_BLOCK_EPB = 0x00000006
PCAPNG_OPT_ENDOFOPT = 0
PCAPNG_OPT_IF_TSRESOL = 9
PCAPNG_OPT_IF_TSOFFSET = 14

TCP_FLAG_FIN = 0x01
TCP_FLAG_SYN = 0x02
TCP_FLAG_RST = 0x04
TCP_FLAG_PUSH = 0x08
TCP_FLAG_ACK = 0x10


@dataclass
class Packet:
    ts: float
    length: int
    ip_version: int
    proto: str
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    direction: str = "unknown"  # up, down, unknown
    tcp_flags: int = 0
    payload_len: int = 0
    raw_payload: bytes = b""
    ip_header_bytes: int = 0
    l4_header_bytes: int = 0
    flow_id: Tuple = field(default_factory=tuple)
    role: str = "unknown"
    host: str = ""
    sni: str = ""
    dns_answers: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)
    source_unit_id: str = ""
    source_capture_id: str = ""

    @property
    def is_udp443(self) -> bool:
        return self.proto == "UDP" and (self.src_port == 443 or self.dst_port == 443)


@dataclass
class PcapData:
    packets: List[Packet]
    linktype: int
    pcap_start: Optional[float]
    pcap_end: Optional[float]
    local_ips: List[str]
    dns_ip_to_hosts: Dict[str, set[str]] = field(default_factory=dict)
    source_flow_count: int = 0


@dataclass
class _PcapRecord:
    ts: float
    length: int
    data: bytes
    linktype: int


@dataclass
class _PcapNgInterface:
    linktype: int
    ts_scale: float = 1e-6
    ts_offset: float = 0.0


def _inet_ntop(version: int, data: bytes) -> str:
    family = socket.AF_INET if version == 4 else socket.AF_INET6
    return socket.inet_ntop(family, data)


def _is_private_or_local(ip: str) -> bool:
    try:
        obj = ipaddress.ip_address(ip)
        return obj.is_private or obj.is_loopback or obj.is_link_local
    except Exception:
        return False


def _canonical_flow(proto: str, src: str, sport: int, dst: str, dport: int) -> Tuple:
    a = (src, sport)
    b = (dst, dport)
    if a <= b:
        return (proto, src, sport, dst, dport)
    return (proto, dst, dport, src, sport)


def _read_dns_name(data: bytes, offset: int, depth: int = 0) -> Tuple[str, int]:
    if depth > 8:
        return "", offset
    labels: List[str] = []
    pos = offset
    jumped = False
    end_pos = offset
    while pos < len(data):
        length = data[pos]
        if length & 0xC0 == 0xC0:
            if pos + 1 >= len(data):
                return "", offset
            ptr = ((length & 0x3F) << 8) | data[pos + 1]
            name, _ = _read_dns_name(data, ptr, depth + 1)
            if name:
                labels.append(name)
            pos += 2
            jumped = True
            end_pos = pos
            break
        if length == 0:
            pos += 1
            end_pos = pos
            break
        pos += 1
        if pos + length > len(data):
            return "", offset
        try:
            labels.append(data[pos:pos + length].decode("idna").lower())
        except Exception:
            return "", offset
        pos += length
    if not jumped:
        end_pos = pos
    return ".".join([p for p in labels if p]).strip("."), end_pos


def _parse_dns_answers(payload: bytes) -> Tuple[Tuple[str, str], ...]:
    if len(payload) < 12:
        return tuple()
    try:
        _tid, flags, qdcount, ancount, _nscount, _arcount = struct.unpack("!HHHHHH", payload[:12])
        # Only use responses. The QR bit is the high bit in DNS flags.
        if not (flags & 0x8000):
            return tuple()
        pos = 12
        query_names: List[str] = []
        for _ in range(min(qdcount, 20)):
            qname, pos = _read_dns_name(payload, pos)
            if qname:
                query_names.append(qname)
            if pos + 4 > len(payload):
                return tuple()
            pos += 4
        answers: List[Tuple[str, str]] = []
        for _ in range(min(ancount, 80)):
            name, pos = _read_dns_name(payload, pos)
            if pos + 10 > len(payload):
                break
            rtype, _rclass, _ttl, rdlen = struct.unpack("!HHIH", payload[pos:pos + 10])
            pos += 10
            if pos + rdlen > len(payload):
                break
            rdata = payload[pos:pos + rdlen]
            pos += rdlen
            host = name or (query_names[0] if query_names else "")
            if not host:
                continue
            if rtype == 1 and rdlen == 4:
                answers.append((_inet_ntop(4, rdata), host))
            elif rtype == 28 and rdlen == 16:
                answers.append((_inet_ntop(6, rdata), host))
        return tuple(answers)
    except Exception:
        return tuple()


def _parse_ipv4(pkt: bytes) -> Optional[Tuple[int, str, str, bytes, int]]:
    if len(pkt) < 20:
        return None
    first = pkt[0]
    version = first >> 4
    if version != 4:
        return None
    ihl = (first & 0x0F) * 4
    if ihl < 20 or len(pkt) < ihl:
        return None
    total_len = struct.unpack("!H", pkt[2:4])[0]
    if total_len <= 0 or total_len > len(pkt):
        total_len = len(pkt)
    proto = pkt[9]
    src = _inet_ntop(4, pkt[12:16])
    dst = _inet_ntop(4, pkt[16:20])
    frag = struct.unpack("!H", pkt[6:8])[0]
    frag_offset = frag & 0x1FFF
    more_frags = bool(frag & 0x2000)
    # Only parse first fragments. Non-first fragments do not contain TCP/UDP headers.
    if frag_offset != 0:
        return None
    payload = pkt[ihl:total_len]
    return proto, src, dst, payload, 4


def _skip_ipv6_ext_headers(next_header: int, payload: bytes) -> Tuple[int, bytes]:
    # Minimal extension-header skipping for hop-by-hop, routing, fragment, destination opts.
    ext_headers = {0, 43, 44, 60}
    nh = next_header
    data = payload
    for _ in range(8):
        if nh not in ext_headers or len(data) < 8:
            break
        if nh == 44:  # Fragment header fixed 8 bytes.
            nh = data[0]
            # Non-first fragment has offset != 0.
            frag_off = struct.unpack("!H", data[2:4])[0] >> 3
            if frag_off != 0:
                return nh, b""
            data = data[8:]
        else:
            nh2 = data[0]
            hdr_len = (data[1] + 1) * 8
            if len(data) < hdr_len:
                return nh, b""
            nh = nh2
            data = data[hdr_len:]
    return nh, data


def _parse_ipv6(pkt: bytes) -> Optional[Tuple[int, str, str, bytes, int]]:
    if len(pkt) < 40:
        return None
    if pkt[0] >> 4 != 6:
        return None
    payload_len = struct.unpack("!H", pkt[4:6])[0]
    next_header = pkt[6]
    src = _inet_ntop(6, pkt[8:24])
    dst = _inet_ntop(6, pkt[24:40])
    payload = pkt[40:40 + payload_len] if payload_len else pkt[40:]
    nh, payload = _skip_ipv6_ext_headers(next_header, payload)
    return nh, src, dst, payload, 6


def _parse_transport(ts: float, incl_len: int, ip_tuple: Tuple[int, str, str, bytes, int]) -> Optional[Packet]:
    proto_num, src, dst, payload, ipver = ip_tuple
    if proto_num == 6:  # TCP
        if len(payload) < 20:
            return None
        sport, dport = struct.unpack("!HH", payload[0:4])
        data_offset = (payload[12] >> 4) * 4
        if data_offset < 20 or len(payload) < data_offset:
            return None
        flags = payload[13]
        payload_len = max(0, len(payload) - data_offset)
        flow = _canonical_flow("TCP", src, sport, dst, dport)
        data = payload[data_offset:]
        sni = _parse_tls_sni(data) if payload_len and (sport == 443 or dport == 443) else ""
        return Packet(ts, incl_len, ipver, "TCP", src, dst, sport, dport, tcp_flags=flags, payload_len=payload_len, flow_id=flow, sni=sni)
    if proto_num == 17:  # UDP
        if len(payload) < 8:
            return None
        sport, dport, ulen = struct.unpack("!HHH", payload[:6])
        payload_len = max(0, min(len(payload), ulen) - 8) if ulen else max(0, len(payload) - 8)
        data = payload[8:8 + payload_len]
        flow = _canonical_flow("UDP", src, sport, dst, dport)
        dns_answers = _parse_dns_answers(data) if sport == 53 or dport == 53 else tuple()
        return Packet(ts, incl_len, ipver, "UDP", src, dst, sport, dport, payload_len=payload_len, flow_id=flow, dns_answers=dns_answers)
    return None


def _packet_payload_by_linktype(data: bytes, linktype: int) -> Optional[bytes]:
    if linktype == DLT_RAW:
        return data
    if linktype == DLT_EN10MB:
        if len(data) < 14:
            return None
        eth_type = struct.unpack("!H", data[12:14])[0]
        off = 14
        # VLAN tags.
        while eth_type in (0x8100, 0x88A8) and len(data) >= off + 4:
            eth_type = struct.unpack("!H", data[off + 2:off + 4])[0]
            off += 4
        if eth_type not in (0x0800, 0x86DD):
            return None
        return data[off:]
    if linktype == DLT_LINUX_SLL:
        if len(data) < 16:
            return None
        eth_type = struct.unpack("!H", data[14:16])[0]
        if eth_type not in (0x0800, 0x86DD):
            return None
        return data[16:]
    if linktype == DLT_LINUX_SLL2:
        if len(data) < 20:
            return None
        eth_type = struct.unpack("!H", data[0:2])[0]
        if eth_type not in (0x0800, 0x86DD):
            return None
        return data[20:]
    if linktype == DLT_NULL:
        if len(data) < 4:
            return None
        # BSD loopback family is native endian; simply inspect the next nibble too.
        return data[4:]
    return None


def _parse_ip(payload: bytes) -> Optional[Tuple[int, str, str, bytes, int]]:
    if not payload:
        return None
    version = payload[0] >> 4
    if version == 4:
        return _parse_ipv4(payload)
    if version == 6:
        return _parse_ipv6(payload)
    return None


def _parse_client_hello_sni(body: bytes) -> str:
    """Best-effort parser for the SNI extension in a single TLS ClientHello.

    This intentionally does not attempt TCP reassembly. It only succeeds when
    the complete ClientHello is present in one captured TCP payload.
    """
    try:
        pos = 0
        if len(body) < 34:
            return ""
        pos += 2  # legacy_version
        pos += 32  # random
        if pos + 1 > len(body):
            return ""
        session_len = body[pos]
        pos += 1 + session_len
        if pos + 2 > len(body):
            return ""
        cipher_len = struct.unpack("!H", body[pos:pos + 2])[0]
        pos += 2 + cipher_len
        if pos + 1 > len(body):
            return ""
        compression_len = body[pos]
        pos += 1 + compression_len
        if pos + 2 > len(body):
            return ""
        extensions_len = struct.unpack("!H", body[pos:pos + 2])[0]
        pos += 2
        end = min(len(body), pos + extensions_len)
        while pos + 4 <= end:
            ext_type, ext_len = struct.unpack("!HH", body[pos:pos + 4])
            pos += 4
            ext_end = pos + ext_len
            if ext_end > end:
                return ""
            if ext_type == 0:  # server_name
                if pos + 2 > ext_end:
                    return ""
                list_len = struct.unpack("!H", body[pos:pos + 2])[0]
                name_pos = pos + 2
                list_end = min(ext_end, name_pos + list_len)
                while name_pos + 3 <= list_end:
                    name_type = body[name_pos]
                    name_len = struct.unpack("!H", body[name_pos + 1:name_pos + 3])[0]
                    name_pos += 3
                    name_end = name_pos + name_len
                    if name_end > list_end:
                        return ""
                    if name_type == 0:
                        try:
                            return body[name_pos:name_end].decode("idna").strip().lower().strip(".")
                        except Exception:
                            return ""
                    name_pos = name_end
            pos = ext_end
    except Exception:
        return ""
    return ""


def _parse_tls_sni(payload: bytes) -> str:
    """Parse SNI from TLS records at the start of a TCP payload."""
    pos = 0
    while pos + 5 <= len(payload):
        content_type = payload[pos]
        record_len = struct.unpack("!H", payload[pos + 3:pos + 5])[0]
        record_start = pos + 5
        record_end = record_start + record_len
        if record_len <= 0 or record_end > len(payload):
            return ""
        if content_type == 22:  # handshake
            hpos = record_start
            while hpos + 4 <= record_end:
                handshake_type = payload[hpos]
                handshake_len = int.from_bytes(payload[hpos + 1:hpos + 4], "big")
                body_start = hpos + 4
                body_end = body_start + handshake_len
                if handshake_len <= 0 or body_end > record_end:
                    return ""
                if handshake_type == 1:  # ClientHello
                    return _parse_client_hello_sni(payload[body_start:body_end])
                hpos = body_end
        pos = record_end
    return ""


def _read_pcap_records(path: str) -> Tuple[int, List[_PcapRecord]]:
    records: List[_PcapRecord] = []
    with open(path, "rb") as f:
        gh = f.read(24)
        if len(gh) < 24:
            raise ValueError(f"{path}: invalid PCAP")
        magic = gh[:4]
        if magic == PCAPNG_SECTION_HEADER:
            return _read_pcapng_records(path, gh + f.read())
        if magic == b"\xd4\xc3\xb2\xa1":
            endian = "<"; scale = 1e-6
        elif magic == b"\xa1\xb2\xc3\xd4":
            endian = ">"; scale = 1e-6
        elif magic == b"\x4d\x3c\xb2\xa1":
            endian = "<"; scale = 1e-9
        elif magic == b"\xa1\xb2\x3c\x4d":
            endian = ">"; scale = 1e-9
        else:
            raise ValueError(f"{path}: unsupported file, not classic PCAP or PCAPNG")
        vmaj, vmin, thiszone, sigfigs, snaplen, network = struct.unpack(endian + "HHiiii", gh[4:24])
        linktype = network & 0xFFFF
        rec_hdr = struct.Struct(endian + "IIII")
        while True:
            rh = f.read(16)
            if not rh:
                break
            if len(rh) < 16:
                break
            ts_sec, ts_frac, incl_len, orig_len = rec_hdr.unpack(rh)
            data = f.read(incl_len)
            if len(data) < incl_len:
                break
            ts = ts_sec + ts_frac * scale
            records.append(_PcapRecord(ts, int(orig_len or incl_len), data, linktype))
    return linktype, records


def _align32(n: int) -> int:
    return (n + 3) & ~3


def _parse_pcapng_options(body: bytes, start: int, endian: str) -> Dict[int, List[bytes]]:
    options: Dict[int, List[bytes]] = {}
    pos = start
    while pos + 4 <= len(body):
        code, length = struct.unpack_from(endian + "HH", body, pos)
        pos += 4
        if code == PCAPNG_OPT_ENDOFOPT:
            break
        if pos + length > len(body):
            break
        value = body[pos:pos + length]
        options.setdefault(code, []).append(value)
        pos += _align32(length)
    return options


def _pcapng_ts_scale(options: Dict[int, List[bytes]]) -> float:
    vals = options.get(PCAPNG_OPT_IF_TSRESOL) or []
    if not vals:
        return 1e-6
    b = vals[0][0] if vals[0] else 6
    if b & 0x80:
        return 2.0 ** -(b & 0x7F)
    return 10.0 ** -b


def _pcapng_ts_offset(options: Dict[int, List[bytes]], endian: str) -> float:
    vals = options.get(PCAPNG_OPT_IF_TSOFFSET) or []
    if vals and len(vals[0]) >= 8:
        return float(struct.unpack(endian + "q", vals[0][:8])[0])
    return 0.0


def _read_pcapng_records(path: str, data: bytes) -> Tuple[int, List[_PcapRecord]]:
    records: List[_PcapRecord] = []
    interfaces: List[_PcapNgInterface] = []
    endian: Optional[str] = None
    primary_linktype: Optional[int] = None
    offset = 0
    size = len(data)
    while offset + 12 <= size:
        if data[offset:offset + 4] == PCAPNG_SECTION_HEADER:
            bom = data[offset + 8:offset + 12]
            if bom == PCAPNG_BYTE_ORDER_MAGIC_LE:
                endian = "<"
            elif bom == PCAPNG_BYTE_ORDER_MAGIC_BE:
                endian = ">"
            else:
                raise ValueError(f"{path}: invalid PCAPNG byte-order magic")
            total_len = struct.unpack_from(endian + "I", data, offset + 4)[0]
            if total_len < 28 or offset + total_len > size:
                raise ValueError(f"{path}: invalid PCAPNG section block length")
            trailer_len = struct.unpack_from(endian + "I", data, offset + total_len - 4)[0]
            if trailer_len != total_len:
                raise ValueError(f"{path}: invalid PCAPNG section block trailer")
            interfaces = []
            offset += total_len
            continue

        if endian is None:
            raise ValueError(f"{path}: invalid PCAPNG, missing section header")

        block_type, total_len = struct.unpack_from(endian + "II", data, offset)
        if total_len < 12 or offset + total_len > size:
            raise ValueError(f"{path}: invalid PCAPNG block length")
        trailer_len = struct.unpack_from(endian + "I", data, offset + total_len - 4)[0]
        if trailer_len != total_len:
            raise ValueError(f"{path}: invalid PCAPNG block trailer")
        body = data[offset + 8:offset + total_len - 4]

        if block_type == PCAPNG_BLOCK_IDB and len(body) >= 8:
            linktype, _reserved, _snaplen = struct.unpack_from(endian + "HHI", body, 0)
            options = _parse_pcapng_options(body, 8, endian)
            iface = _PcapNgInterface(
                linktype=int(linktype),
                ts_scale=_pcapng_ts_scale(options),
                ts_offset=_pcapng_ts_offset(options, endian),
            )
            interfaces.append(iface)
            if primary_linktype is None:
                primary_linktype = iface.linktype

        elif block_type == PCAPNG_BLOCK_EPB and len(body) >= 20:
            interface_id, ts_high, ts_low, cap_len, orig_len = struct.unpack_from(endian + "IIIII", body, 0)
            if interface_id < len(interfaces) and len(body) >= 20 + _align32(cap_len):
                iface = interfaces[interface_id]
                ts_raw = (int(ts_high) << 32) | int(ts_low)
                packet = body[20:20 + cap_len]
                records.append(_PcapRecord(
                    ts=float(ts_raw * iface.ts_scale + iface.ts_offset),
                    length=int(orig_len or cap_len),
                    data=packet,
                    linktype=iface.linktype,
                ))

        elif block_type == PCAPNG_BLOCK_PB and len(body) >= 20:
            interface_id, _drops, ts_high, ts_low, cap_len, orig_len = struct.unpack_from(endian + "HHIIII", body, 0)
            if interface_id < len(interfaces) and len(body) >= 20 + _align32(cap_len):
                iface = interfaces[interface_id]
                ts_raw = (int(ts_high) << 32) | int(ts_low)
                packet = body[20:20 + cap_len]
                records.append(_PcapRecord(
                    ts=float(ts_raw * iface.ts_scale + iface.ts_offset),
                    length=int(orig_len or cap_len),
                    data=packet,
                    linktype=iface.linktype,
                ))

        offset += total_len

    if primary_linktype is None:
        primary_linktype = records[0].linktype if records else -1
    return primary_linktype, records


def _infer_local_ips_prepass(parsed_packets: List[Packet]) -> List[str]:
    counts: Dict[str, int] = {}
    syn_clients: Dict[str, int] = {}
    for p in parsed_packets:
        counts[p.src_ip] = counts.get(p.src_ip, 0) + 1
        counts[p.dst_ip] = counts.get(p.dst_ip, 0) + 1
        if p.proto == "TCP" and (p.tcp_flags & TCP_FLAG_SYN) and not (p.tcp_flags & TCP_FLAG_ACK):
            syn_clients[p.src_ip] = syn_clients.get(p.src_ip, 0) + 1
    private_ips = [ip for ip in counts if _is_private_or_local(ip)]
    if private_ips:
        return sorted(private_ips, key=lambda ip: (-counts[ip], ip))
    if syn_clients:
        return sorted(syn_clients, key=lambda ip: (-syn_clients[ip], ip))
    if counts:
        return [max(counts, key=counts.get)]
    return []


def _assign_directions(packets: List[Packet], local_ips: List[str]) -> None:
    local_set = set(local_ips)
    tcp_clients: Dict[Tuple, str] = {}
    for p in packets:
        if p.proto == "TCP" and (p.tcp_flags & TCP_FLAG_SYN) and not (p.tcp_flags & TCP_FLAG_ACK):
            tcp_clients[p.flow_id] = p.src_ip
    for p in packets:
        if p.src_ip in local_set and p.dst_ip not in local_set:
            p.direction = "up"
        elif p.dst_ip in local_set and p.src_ip not in local_set:
            p.direction = "down"
        elif p.flow_id in tcp_clients:
            p.direction = "up" if p.src_ip == tcp_clients[p.flow_id] else "down"
        else:
            # Heuristic for HTTP(S) style flows.
            if p.dst_port in (80, 443, 8080, 8443) and p.src_port > 1024:
                p.direction = "up"
            elif p.src_port in (80, 443, 8080, 8443) and p.dst_port > 1024:
                p.direction = "down"
            else:
                p.direction = "unknown"


def _tcp_flags_from_mirage(value) -> int:
    if value is None:
        return 0
    if isinstance(value, (int, float)) and not math.isnan(float(value)):
        return int(value)
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return 0
    try:
        return int(text, 0)
    except ValueError:
        pass
    flags = 0
    for ch in text.upper():
        if ch == "F":
            flags |= TCP_FLAG_FIN
        elif ch == "S":
            flags |= TCP_FLAG_SYN
        elif ch == "R":
            flags |= TCP_FLAG_RST
        elif ch == "P":
            flags |= TCP_FLAG_PUSH
        elif ch == "A":
            flags |= TCP_FLAG_ACK
    return flags


def _safe_float(value, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(out) else out


def _safe_int(value, default: int = 0) -> int:
    try:
        if value is None:
            return default
        out = float(value)
        if math.isnan(out):
            return default
        return int(out)
    except (TypeError, ValueError):
        return default


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _mirage_payload_bytes(value) -> bytes:
    """Decode only MIRAGE's network payload field into its recorded bytes."""
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, list):
        try:
            return bytes(int(item) & 0xFF for item in value)
        except (TypeError, ValueError):
            return b""
    if isinstance(value, str):
        text = value.strip().replace(" ", "").replace(":", "")
        try:
            return bytes.fromhex(text)
        except ValueError:
            return b""
    return b""


def _read_mirage_json(path: str) -> PcapData:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"{path}: unsupported Mirage/AppAct JSON, expected object keyed by flow")

    packets: List[Packet] = []
    for flow_key, flow_obj in obj.items():
        parts = [p.strip() for p in str(flow_key).split(",")]
        if len(parts) < 5:
            continue
        src_ip, src_port_s, dst_ip, dst_port_s, proto_s = parts[:5]
        src_port = _safe_int(src_port_s, 0)
        dst_port = _safe_int(dst_port_s, 0)
        proto_num = _safe_int(proto_s, 0)
        proto = "TCP" if proto_num == 6 else "UDP" if proto_num == 17 else str(proto_num or "IP")
        if proto not in {"TCP", "UDP"}:
            continue

        packet_data = (flow_obj or {}).get("packet_data", {}) if isinstance(flow_obj, dict) else {}
        ts_values = _as_list(packet_data.get("timestamp"))
        dirs = _as_list(packet_data.get("packet_dir"))
        lengths = _as_list(packet_data.get("IP_packet_bytes"))
        payload_lens = _as_list(packet_data.get("L4_payload_bytes"))
        raw_payloads = _as_list(packet_data.get("L4_raw_payload"))
        ip_header_lens = _as_list(packet_data.get("IP_header_bytes"))
        l4_header_lens = _as_list(packet_data.get("L4_header_bytes"))
        flags_values = _as_list(packet_data.get("TCP_flags"))
        roles = _as_list(packet_data.get("role"))
        hosts = _as_list(packet_data.get("host"))
        snis = _as_list(packet_data.get("sni"))
        dns_values = _as_list(packet_data.get("dns_answers"))
        source_unit_ids = _as_list(packet_data.get("source_unit_id"))
        source_capture_ids = _as_list(packet_data.get("source_capture_id"))
        n = len(ts_values)
        if n == 0:
            continue
        flow_id = _canonical_flow(proto, src_ip, src_port, dst_ip, dst_port)
        for i in range(n):
            ts = _safe_float(ts_values[i], default=float("nan"))
            if math.isnan(ts):
                continue
            direction_code = _safe_int(dirs[i], 0) if i < len(dirs) else 0
            if direction_code == 1:
                pkt_src, pkt_sport, pkt_dst, pkt_dport = dst_ip, dst_port, src_ip, src_port
                direction = "down"
            else:
                pkt_src, pkt_sport, pkt_dst, pkt_dport = src_ip, src_port, dst_ip, dst_port
                direction = "up"
            length = _safe_int(lengths[i], 0) if i < len(lengths) else 0
            payload_len = _safe_int(payload_lens[i], 0) if i < len(payload_lens) else 0
            raw_payload = _mirage_payload_bytes(raw_payloads[i]) if i < len(raw_payloads) else b""
            ip_header_len = _safe_int(ip_header_lens[i], 0) if i < len(ip_header_lens) else 0
            l4_header_len = _safe_int(l4_header_lens[i], 0) if i < len(l4_header_lens) else 0
            tcp_flags = _tcp_flags_from_mirage(flags_values[i]) if proto == "TCP" and i < len(flags_values) else 0
            dns_answers = ()
            if i < len(dns_values) and isinstance(dns_values[i], (list, tuple)):
                dns_answers = tuple(
                    (str(item[0]), str(item[1]))
                    for item in dns_values[i]
                    if isinstance(item, (list, tuple)) and len(item) >= 2
                )
            packets.append(Packet(
                ts=ts,
                length=max(0, length),
                ip_version=6 if ":" in pkt_src or ":" in pkt_dst else 4,
                proto=proto,
                src_ip=pkt_src,
                dst_ip=pkt_dst,
                src_port=pkt_sport,
                dst_port=pkt_dport,
                direction=direction,
                tcp_flags=tcp_flags,
                payload_len=max(0, payload_len),
                raw_payload=raw_payload,
                ip_header_bytes=max(0, ip_header_len),
                l4_header_bytes=max(0, l4_header_len),
                flow_id=flow_id,
                role=str(roles[i]) if i < len(roles) else "unknown",
                host=str(hosts[i]) if i < len(hosts) else "",
                sni=str(snis[i]) if i < len(snis) else "",
                dns_answers=dns_answers,
                source_unit_id=str(source_unit_ids[i]) if i < len(source_unit_ids) else "",
                source_capture_id=str(source_capture_ids[i]) if i < len(source_capture_ids) else "",
            ))

    packets.sort(key=lambda p: p.ts)
    local_ips = _infer_local_ips_prepass(packets)
    if local_ips:
        _assign_directions(packets, local_ips)
    pcap_start = packets[0].ts if packets else None
    pcap_end = packets[-1].ts if packets else None
    dns_ip_to_hosts: Dict[str, set[str]] = {}
    for packet in packets:
        for ip, host in packet.dns_answers:
            if ip and host:
                dns_ip_to_hosts.setdefault(ip, set()).add(host)
    return PcapData(
        packets=packets,
        linktype=-1,
        pcap_start=pcap_start,
        pcap_end=pcap_end,
        local_ips=local_ips,
        dns_ip_to_hosts=dns_ip_to_hosts,
        source_flow_count=len(obj),
    )


def read_pcap(
    path: str,
    *,
    start_epoch: Optional[float] = None,
    end_epoch: Optional[float] = None,
) -> PcapData:
    if pathlib.Path(path).suffix.lower() == ".json":
        result = _read_mirage_json(path)
        if start_epoch is None and end_epoch is None:
            return result
        packets = [
            packet
            for packet in result.packets
            if (start_epoch is None or packet.ts >= float(start_epoch))
            and (end_epoch is None or packet.ts <= float(end_epoch))
        ]
        local_ips = _infer_local_ips_prepass(packets)
        _assign_directions(packets, local_ips)
        dns_ip_to_hosts: Dict[str, set[str]] = {}
        for packet in packets:
            for ip, host in packet.dns_answers:
                if ip and host:
                    dns_ip_to_hosts.setdefault(ip, set()).add(host)
        return PcapData(
            packets=packets,
            linktype=result.linktype,
            pcap_start=packets[0].ts if packets else None,
            pcap_end=packets[-1].ts if packets else None,
            local_ips=local_ips,
            dns_ip_to_hosts=dns_ip_to_hosts,
            source_flow_count=result.source_flow_count,
        )
    linktype, records = _read_pcap_records(path)
    packets: List[Packet] = []
    for rec in records:
        if start_epoch is not None and rec.ts < float(start_epoch):
            continue
        if end_epoch is not None and rec.ts > float(end_epoch):
            continue
        payload = _packet_payload_by_linktype(rec.data, rec.linktype)
        if payload is None:
            continue
        ipt = _parse_ip(payload)
        if ipt is None:
            continue
        p = _parse_transport(rec.ts, rec.length, ipt)
        if p is not None:
            # For DLT_RAW, orig_len is IP-packet length. For Ethernet, orig_len includes link header.
            p.length = int(rec.length)
            packets.append(p)
    packets.sort(key=lambda p: p.ts)
    local_ips = _infer_local_ips_prepass(packets)
    _assign_directions(packets, local_ips)
    dns_ip_to_hosts: Dict[str, set[str]] = {}
    for p in packets:
        for ip, host in p.dns_answers:
            if ip and host:
                dns_ip_to_hosts.setdefault(ip, set()).add(host)
    pcap_start = packets[0].ts if packets else None
    pcap_end = packets[-1].ts if packets else None
    return PcapData(
        packets=packets,
        linktype=linktype,
        pcap_start=pcap_start,
        pcap_end=pcap_end,
        local_ips=local_ips,
        dns_ip_to_hosts=dns_ip_to_hosts,
    )

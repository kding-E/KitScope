from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Any


def parse_time_to_epoch(value: Any) -> Optional[float]:
    """Parse ISO8601 strings or numeric epoch timestamps into seconds."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            pass
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(s).timestamp()
        except ValueError:
            # Last resort for timestamps without timezone.
            try:
                dt = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ")
                return dt.replace(tzinfo=timezone.utc).timestamp()
            except Exception:
                return None
    return None

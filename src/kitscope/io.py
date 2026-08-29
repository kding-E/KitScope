from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"missing data file: {path}")
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def first_column(frame: pd.DataFrame, names: Iterable[str], kind: str) -> str:
    for name in names:
        if name in frame.columns:
            return name
    tried = ", ".join(names)
    raise ValueError(f"could not find {kind} column; tried: {tried}")


def resolve_data_path(data_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return data_root / path

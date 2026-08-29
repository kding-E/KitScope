#!/usr/bin/env python3
"""Prepare and merge an incremental Layer-1 extraction for a frozen snapshot.

The July 2026 feature table uses an older drive letter and older names for
three dataset directories.  This utility compares paths by their canonical
location below ``phish_dataset`` and never by basename alone, writes the exact
missing-extraction input list, and can later merge the reusable rows with the
new extraction while rewriting ``zip_path`` to the frozen manifest path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PureWindowsPath
from typing import Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.dataset as ds
import pyarrow.parquet as pq


DATASET_DIR_ALIASES = {
    "benign_appact": "benign_mobile_appact",
    "benign_blockchain": "benign_blockchain_act",
    "benign_browser": "benign_mobile_browser",
}


def canonical_dataset_key(value: object) -> str:
    """Return a case-insensitive path key relative to ``phish_dataset``."""
    parts = list(PureWindowsPath(str(value)).parts)
    lower = [part.lower() for part in parts]
    if "phish_dataset" in lower:
        parts = parts[lower.index("phish_dataset") + 1 :]
    normalized = [part.lower() for part in parts]
    if normalized:
        normalized[0] = DATASET_DIR_ALIASES.get(normalized[0], normalized[0])
    return "/".join(normalized)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_path_column(path: Path) -> pd.Series:
    if path.suffix.lower() in {".parquet", ".pq"}:
        values = pq.read_table(path, columns=["zip_path"])["zip_path"].to_pylist()
        return pd.Series(values, dtype="string")
    return pd.read_csv(path, usecols=["zip_path"], dtype="string")["zip_path"]


def _assert_unique_mapping(frame: pd.DataFrame, key: str, value: str, name: str) -> None:
    counts = frame[[key, value]].drop_duplicates().groupby(key, dropna=False)[value].nunique()
    collisions = counts[counts > 1]
    if not collisions.empty:
        raise ValueError(f"{name} has {len(collisions)} canonical-path collisions")


def prepare(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    features_path = Path(args.reused_features)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_path, dtype=str, low_memory=False)
    required = {"capture_id", "sample_path", "label", "source", "source_name", "source_variant"}
    missing_columns = sorted(required - set(manifest.columns))
    if missing_columns:
        raise ValueError(f"manifest is missing required columns: {missing_columns}")
    if manifest["capture_id"].duplicated().any():
        raise ValueError("manifest capture_id values are not unique")

    manifest["canonical_dataset_key"] = manifest["sample_path"].map(canonical_dataset_key)
    _assert_unique_mapping(manifest, "canonical_dataset_key", "sample_path", "manifest")

    old_paths = read_path_column(features_path).dropna().drop_duplicates().to_frame("old_zip_path")
    old_paths["canonical_dataset_key"] = old_paths["old_zip_path"].map(canonical_dataset_key)
    _assert_unique_mapping(old_paths, "canonical_dataset_key", "old_zip_path", "reused feature table")

    joined = old_paths.merge(
        manifest[
            [
                "canonical_dataset_key",
                "capture_id",
                "sample_path",
                "label",
                "source",
                "source_name",
                "source_variant",
            ]
        ],
        on="canonical_dataset_key",
        how="left",
        validate="one_to_one",
    )
    reusable = joined[joined["capture_id"].notna()].copy()
    outside = joined[joined["capture_id"].isna()].copy()
    covered_keys = set(reusable["canonical_dataset_key"])
    covered = manifest[manifest["canonical_dataset_key"].isin(covered_keys)].copy()
    missing = manifest[~manifest["canonical_dataset_key"].isin(covered_keys)].copy()

    reusable.drop(columns=["canonical_dataset_key"]).to_csv(out_dir / "reused_path_map.csv", index=False)
    outside.drop(columns=["canonical_dataset_key"]).to_csv(
        out_dir / "reused_sessions_outside_snapshot.csv", index=False
    )
    covered.drop(columns=["canonical_dataset_key"]).to_csv(out_dir / "covered_manifest.csv", index=False)
    missing.drop(columns=["canonical_dataset_key"]).to_csv(out_dir / "missing_manifest.csv", index=False)
    (out_dir / "missing_input_paths.txt").write_text(
        "\n".join(missing["sample_path"].astype(str)) + ("\n" if len(missing) else ""),
        encoding="utf-8",
    )

    by_source = (
        manifest.assign(covered=manifest["canonical_dataset_key"].isin(covered_keys))
        .groupby(["source", "source_name", "source_variant", "label"], dropna=False)["covered"]
        .agg(total="size", covered="sum")
        .reset_index()
    )
    by_source["missing"] = by_source["total"] - by_source["covered"]
    by_source.to_csv(out_dir / "coverage_by_source.csv", index=False)

    summary = {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "reused_features": str(features_path.resolve()),
        "reused_features_sha256": sha256_file(features_path),
        "manifest_sessions": int(len(manifest)),
        "reused_feature_sessions": int(len(old_paths)),
        "covered_sessions": int(len(covered)),
        "missing_sessions": int(len(missing)),
        "reused_sessions_outside_snapshot": int(len(outside)),
        "coverage": float(len(covered) / len(manifest)) if len(manifest) else 0.0,
        "path_aliases": DATASET_DIR_ALIASES,
        "missing_input_paths": str((out_dir / "missing_input_paths.txt").resolve()),
        "reused_path_map": str((out_dir / "reused_path_map.csv").resolve()),
    }
    (out_dir / "coverage_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _rewrite_reused_batches(
    features: Path,
    path_map: dict[str, str],
    writer: pq.ParquetWriter,
    batch_size: int,
) -> tuple[int, set[str]]:
    rows = 0
    sessions: set[str] = set()
    scanner = ds.dataset(features, format="parquet").scanner(batch_size=batch_size)
    for batch in scanner.to_batches():
        old_paths = batch.column(batch.schema.get_field_index("zip_path")).to_pylist()
        keep_indices = [i for i, value in enumerate(old_paths) if str(value) in path_map]
        if not keep_indices:
            continue
        kept = batch.take(pa.array(keep_indices, type=pa.int64()))
        rewritten = [path_map[str(old_paths[i])] for i in keep_indices]
        column_index = kept.schema.get_field_index("zip_path")
        kept = kept.set_column(column_index, "zip_path", pa.array(rewritten, type=pa.string()))
        writer.write_batch(kept)
        rows += kept.num_rows
        sessions.update(rewritten)
    return rows, sessions


def _cast_table_to_schema(table: pa.Table, schema: pa.Schema) -> pa.Table:
    arrays: list[pa.Array | pa.ChunkedArray] = []
    for field in schema:
        if field.name not in table.column_names:
            arrays.append(pa.nulls(table.num_rows, type=field.type))
            continue
        column = table[field.name]
        if column.type != field.type:
            column = pc.cast(column, field.type, safe=False)
        arrays.append(column)
    return pa.Table.from_arrays(arrays, schema=schema)


def _read_incremental_batches(path: Path, schema: pa.Schema, block_size: int) -> Iterable[pa.RecordBatch]:
    if path.suffix.lower() in {".parquet", ".pq"}:
        for batch in ds.dataset(path, format="parquet").scanner(batch_size=50_000).to_batches():
            yield _cast_table_to_schema(pa.Table.from_batches([batch]), schema).to_batches()[0]
        return
    reader = pacsv.open_csv(
        path,
        read_options=pacsv.ReadOptions(block_size=block_size, use_threads=True),
        convert_options=pacsv.ConvertOptions(strings_can_be_null=True),
    )
    for batch in reader:
        yield _cast_table_to_schema(pa.Table.from_batches([batch]), schema).to_batches()[0]


def merge(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    reused_path = Path(args.reused_features)
    incremental_path = Path(args.incremental_features)
    prep_dir = Path(args.prepared_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_path, dtype=str, low_memory=False)
    allowed = set(manifest["sample_path"].astype(str))
    path_map_frame = pd.read_csv(prep_dir / "reused_path_map.csv", dtype=str)
    path_map = dict(zip(path_map_frame["old_zip_path"], path_map_frame["sample_path"]))
    base_schema = pq.ParquetFile(reused_path).schema_arrow
    tmp = output.with_name(output.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    writer = pq.ParquetWriter(tmp, base_schema, compression="zstd")
    try:
        reused_rows, sessions = _rewrite_reused_batches(
            reused_path, path_map, writer, int(args.batch_size)
        )
        incremental_rows = 0
        for batch in _read_incremental_batches(
            incremental_path, base_schema, int(args.csv_block_size)
        ):
            paths = [str(value) for value in batch.column(batch.schema.get_field_index("zip_path")).to_pylist()]
            invalid = sorted(set(paths) - allowed)
            if invalid:
                raise ValueError(f"incremental features contain {len(invalid)} paths outside manifest")
            writer.write_batch(batch)
            incremental_rows += batch.num_rows
            sessions.update(paths)
    finally:
        writer.close()

    missing = sorted(allowed - sessions)
    if missing and not args.allow_missing:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"merged feature table is missing {len(missing)} manifest sessions")
    tmp.replace(output)
    report = {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "reused_features": str(reused_path.resolve()),
        "reused_features_sha256": sha256_file(reused_path),
        "incremental_features": str(incremental_path.resolve()),
        "incremental_features_sha256": sha256_file(incremental_path),
        "output": str(output.resolve()),
        "output_sha256": sha256_file(output),
        "reused_rows": int(reused_rows),
        "incremental_rows": int(incremental_rows),
        "merged_rows": int(reused_rows + incremental_rows),
        "manifest_sessions": int(len(allowed)),
        "merged_sessions": int(len(sessions)),
        "missing_sessions": int(len(missing)),
        "missing_paths": missing,
    }
    output.with_suffix(".merge.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prep = subparsers.add_parser("prepare")
    prep.add_argument("--manifest", required=True)
    prep.add_argument("--reused-features", required=True)
    prep.add_argument("--out-dir", required=True)
    prep.set_defaults(func=prepare)

    combine = subparsers.add_parser("merge")
    combine.add_argument("--manifest", required=True)
    combine.add_argument("--reused-features", required=True)
    combine.add_argument("--incremental-features", required=True)
    combine.add_argument("--prepared-dir", required=True)
    combine.add_argument("--output", required=True)
    combine.add_argument("--batch-size", type=int, default=50_000)
    combine.add_argument("--csv-block-size", type=int, default=64 * 1024 * 1024)
    combine.add_argument("--allow-missing", action="store_true")
    combine.set_defaults(func=merge)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

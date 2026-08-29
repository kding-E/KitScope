#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _read_manifest(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        frame = pd.DataFrame(rows)
    else:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
    if "capture_id" not in frame.columns:
        raise ValueError(f"{path}: missing capture_id")
    if frame["capture_id"].astype(str).duplicated().any():
        raise ValueError(f"{path}: duplicate capture_id values")
    return frame.fillna("")


def _rel_or_default(row: pd.Series, key: str, folder: str, suffix: str) -> str:
    value = str(row.get(key) or "").strip()
    if value:
        return value.replace("\\", "/")
    return f"{folder}/{row['capture_id']}.{suffix}"


def _select(frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = frame.copy()
    if args.limit_per_split_label and int(args.limit_per_split_label) > 0:
        parts: list[pd.DataFrame] = []
        fit = out[out["partition"].astype(str).eq("fit")].copy()
        other = out[~out["partition"].astype(str).eq("fit")].copy()
        if len(fit):
            fit_role = fit.get("fit_role", pd.Series("fit_train", index=fit.index)).fillna("")
            fit = fit.assign(__fit_role=fit_role.where(fit_role.astype(str).ne(""), "fit_train"))
            for (_role, _label), group in fit.groupby(["__fit_role", "label"], sort=True):
                parts.append(group.sample(n=min(int(args.limit_per_split_label), len(group)), random_state=int(args.random_state)))
        for (_partition, _label), group in other.groupby(["partition", "label"], sort=True):
            parts.append(group.sample(n=min(int(args.limit_per_split_label), len(group)), random_state=int(args.random_state)))
        out = pd.concat(parts, ignore_index=True).drop(columns=["__fit_role"], errors="ignore") if parts else out.iloc[0:0]
    if args.max_samples and int(args.max_samples) > 0 and len(out) > int(args.max_samples):
        out = out.sample(n=int(args.max_samples), random_state=int(args.random_state)).reset_index(drop=True)
    return out.reset_index(drop=True)


def _require(path: Path, kind: str, capture_id: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{capture_id}: missing {kind}: {path}")


def _load_kit_label_manifest(path: str | None) -> pd.DataFrame | None:
    if not path:
        return None
    frame = pd.read_csv(Path(path), dtype=str, keep_default_na=False, low_memory=False).fillna("")
    if "capture_id" not in frame.columns:
        raise ValueError(f"{path}: missing capture_id")
    key_col = next(
        (column for column in ("kit_family_id", "static_family_key", "kit_family_key") if column in frame.columns),
        None,
    )
    if key_col is None:
        raise ValueError(f"{path}: missing kit_family_id/static_family_key/kit_family_key")
    keep = ["capture_id", key_col, *(["fit_eligible"] if "fit_eligible" in frame.columns else [])]
    out = frame[keep].copy()
    out = out.rename(columns={key_col: "kit_family_id", "fit_eligible": "label_fit_eligible"})
    out = out[out["capture_id"].astype(str).ne("")]
    if out["capture_id"].astype(str).duplicated().any():
        raise ValueError(f"{path}: duplicate capture_id values")
    return out


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = data_root / args.manifest
    manifest = _select(_read_manifest(manifest_path), args)
    rebuilt_kit_labels = _load_kit_label_manifest(args.kit_label_manifest)

    records: list[dict[str, Any]] = []
    static_records: list[dict[str, Any]] = []
    for _, row in manifest.iterrows():
        capture_id = str(row["capture_id"])
        pcap_rel = _rel_or_default(row, "pcap_path", "pcap", "pcap")
        har_rel = _rel_or_default(row, "har_path", "har", "har")
        json_rel = _rel_or_default(row, "json_path", "json", "json")
        pcap_path = data_root / pcap_rel
        har_path = data_root / har_rel
        json_path = data_root / json_rel
        _require(pcap_path, "pcap", capture_id)
        _require(har_path, "har", capture_id)
        _require(json_path, "json", capture_id)
        static_dir = data_root / "static_features" / capture_id
        screenshot_dir = data_root / "screenshots" / capture_id
        record = row.to_dict()
        record.update(
            {
                "capture_id": capture_id,
                "pcap_path": pcap_rel,
                "har_path": har_rel,
                "json_path": json_rel,
                "zip_path": str(json_path),
                "sample_path": str(json_path),
                "traffic_sample_path": str(json_path),
                "static_sample_path": str(static_dir) if static_dir.exists() else "",
                "screenshots_dir": str(screenshot_dir) if screenshot_dir.exists() else "",
            }
        )
        records.append(record)
        if static_dir.exists():
            static_records.append(
                {
                    "capture_id": capture_id,
                    "sample_path": str(static_dir),
                    "json_path": str(json_path),
                    "har_path": str(har_path),
                    "label": record.get("label", ""),
                    "sample_id": (record.get("sample") or {}).get("sample_id", "") if isinstance(record.get("sample"), dict) else record.get("sample_id", ""),
                    "domain": record.get("domain", ""),
                    "url": record.get("url", ""),
                    "source": record.get("source", ""),
                    "source_variant": record.get("source_variant", ""),
                    "source_folder": record.get("source_folder", ""),
                    "fine_source": record.get("fine_source", ""),
                    "partition": record.get("partition", ""),
                    "fit_role": record.get("fit_role", ""),
                }
            )

    selected = pd.DataFrame(records).fillna("")
    selected.to_csv(out_dir / "manifest_selected.csv", index=False)
    (out_dir / "sample_inputs.txt").write_text(
        "\n".join(selected["traffic_sample_path"].astype(str).tolist()) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(static_records).to_csv(out_dir / "static_snapshot_manifest.csv", index=False)
    (out_dir / "static_sample_inputs.txt").write_text(
        "\n".join(row["sample_path"] for row in static_records) + ("\n" if static_records else ""),
        encoding="utf-8",
    )
    selected[
        ["capture_id", "traffic_sample_path", "static_sample_path", "screenshots_dir"]
    ].to_csv(out_dir / "public_path_map.csv", index=False)

    split_cols = [
        "zip_path", "sample_path", "capture_id", "label", "partition", "fit_role", "capture_start_utc",
        "environment_id", "source", "source_name", "source_variant", "fine_source",
        "positive_supergroup", "benign_supergroup", "wallet_family_id", "kit_family_id",
        "backend_cluster_id", "domain", "url",
    ]
    selected[[column for column in split_cols if column in selected.columns]].to_csv(out_dir / "split_manifest.csv", index=False)
    selected[[column for column in ["zip_path", "capture_id", "source_name", "source_variant", "fine_source"] if column in selected.columns]].to_csv(
        out_dir / "source_map.csv", index=False
    )

    kit_columns = ["zip_path", "capture_id", "partition"]
    kit_columns += [column for column in ("kit_family_id", "backend_cluster_id") if column in selected.columns]
    kit = selected[kit_columns].copy()
    kit_label_source = "release_manifest"
    if rebuilt_kit_labels is not None:
        kit = kit.drop(columns=["kit_family_id", "backend_cluster_id"], errors="ignore").merge(
            rebuilt_kit_labels,
            on="capture_id",
            how="left",
            validate="one_to_one",
        )
        kit["kit_family_id"] = kit["kit_family_id"].fillna("").astype(str)
        quality_eligible = kit.get("label_fit_eligible", pd.Series("true", index=kit.index)).astype(str).str.lower().isin(
            {"1", "true", "yes"}
        )
        kit["fit_eligible"] = kit["partition"].astype(str).eq("fit") & quality_eligible & kit["kit_family_id"].ne("")
        kit_label_source = "rebuilt_kit_label_manifest"
    else:
        if "kit_family_id" not in kit.columns:
            kit["kit_family_id"] = ""
        if "backend_cluster_id" not in kit.columns:
            kit["backend_cluster_id"] = ""
        family = kit["kit_family_id"].fillna("").astype(str)
        backend = kit["backend_cluster_id"].fillna("").astype(str)
        kit["kit_family_id"] = family.where(family.ne(""), backend)
        kit["fit_eligible"] = kit["partition"].astype(str).eq("fit") & kit["kit_family_id"].astype(str).ne("")
    kit[["zip_path", "capture_id", "kit_family_id", "fit_eligible"]].to_csv(out_dir / "kit_labels.csv", index=False)

    report = {
        "schema": "kitscope_public_release_inputs",
        "data_root": str(data_root),
        "manifest": str(manifest_path),
        "selected_samples": int(len(selected)),
        "static_feature_dirs": int(len(static_records)),
        "kit_family_nonempty": int(kit["kit_family_id"].astype(str).ne("").sum()),
        "kit_label_source": kit_label_source,
        "files": {
            "sample_inputs": str(out_dir / "sample_inputs.txt"),
            "split_manifest": str(out_dir / "split_manifest.csv"),
            "source_map": str(out_dir / "source_map.csv"),
            "kit_labels": str(out_dir / "kit_labels.csv"),
            "static_snapshot_manifest": str(out_dir / "static_snapshot_manifest.csv"),
            "public_path_map": str(out_dir / "public_path_map.csv"),
        },
    }
    (out_dir / "prepare_public_release_inputs_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare public-layout KitScope release inputs for the original runners.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--manifest", default="manifest.jsonl")
    parser.add_argument("--kit-label-manifest", default=None,
                        help="Optional rebuilt kit_label_manifest.csv to materialize runner-compatible kit_labels.csv.")
    parser.add_argument("--limit-per-split-label", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    return parser


if __name__ == "__main__":
    prepare(build_parser().parse_args())

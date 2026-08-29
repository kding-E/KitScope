#!/usr/bin/env python3
"""Build Layer 2 static-family assignments from merged precomputed evidence.

This is the extraction-only bridge needed by the static kit-label pipeline. The
older experiment driver also trains an unrelated attribution model after
building the same assignments, which is unnecessary for the recommended
traffic-only Layer 2 route and makes a full rerun needlessly expensive.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
for path in (ROOT / "scripts", ROOT / "evolution" / "layer2_family_classifier"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from known_kit_attribution_engine import build_static_family_labels  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-features", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--min-count", type=int, default=4)
    parser.add_argument("--exclude-har", action="store_true")
    parser.add_argument("--disable-anchor-merge", action="store_true")
    parser.add_argument("--anchor-merge-min-overlap", type=float, default=0.80)
    parser.add_argument("--anchor-merge-min-cooccur", type=int, default=None)
    args = parser.parse_args()

    static = pd.read_csv(args.static_features, low_memory=False)
    if "label" in static.columns:
        phishing = static[static["label"].astype(str).str.lower().str.contains("phish")].copy()
    else:
        phishing = static.copy()
    if phishing.empty:
        raise ValueError("no phishing rows found in merged static features")

    out_dir = pathlib.Path(args.out_dir)
    labels = build_static_family_labels(
        phishing=phishing,
        min_count=int(args.min_count),
        out_dir=out_dir,
        include_har=not bool(args.exclude_har),
        anchor_merge=not bool(args.disable_anchor_merge),
        anchor_merge_min_overlap=float(args.anchor_merge_min_overlap),
        anchor_merge_min_cooccur=args.anchor_merge_min_cooccur,
        precomputed_static_features=static,
    )
    print(
        f"[static-assignments] rows={len(labels)} "
        f"high_conf={int(labels['high_conf_family'].fillna(False).astype(bool).sum())} "
        f"out={out_dir / 'static_family_label_assignments.csv'}"
    )


if __name__ == "__main__":
    main()

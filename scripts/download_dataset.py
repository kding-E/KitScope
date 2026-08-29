#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

import yaml


PLACEHOLDER_PREFIX = "TODO_REPLACE"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and unpack the KitScope dataset.")
    parser.add_argument("--config", type=Path, default=Path("configs/main_random.yaml"))
    parser.add_argument("--out", type=Path, default=Path("data/kitscope-release"))
    parser.add_argument("--url", default=None, help="override dataset_url from the config")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    url = args.url or str(config.get("dataset_url", ""))
    if not url or url.startswith(PLACEHOLDER_PREFIX):
        raise SystemExit(
            "dataset_url is still a placeholder. Replace it in configs/main_random.yaml "
            "or pass --url before running this downloader."
        )

    args.out.mkdir(parents=True, exist_ok=True)
    archive = args.out.with_suffix(".zip")
    print(f"downloading {url}")
    with urllib.request.urlopen(url) as response, archive.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    print(f"wrote {archive}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(args.out)
    print(f"extracted to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

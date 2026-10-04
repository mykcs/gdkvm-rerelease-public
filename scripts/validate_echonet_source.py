#!/usr/bin/env python3
"""Validate a locally authorized EchoNet-Dynamic source tree.

This script never downloads EchoNet-Dynamic and never stores a download URL.
Each user must obtain access through Stanford's current official access path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gdkvm_data.contract import ECHONET_EXPECTED_VIDEOS, load_echonet_filelist


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    filelist = args.root / "FileList.csv"
    tracings = args.root / "VolumeTracings.csv"
    videos = args.root / "Videos"
    for path in (filelist, tracings):
        if not path.is_file():
            raise SystemExit(f"missing required EchoNet file: {path}")

    splits = load_echonet_filelist(filelist)
    result = {
        "dataset": "EchoNet-Dynamic",
        "split_counts": {key: len(value) for key, value in splits.items()},
        "expected_split_counts": ECHONET_EXPECTED_VIDEOS,
        "filelist": str(filelist),
        "volume_tracings": str(tracings),
        "videos_directory_exists": videos.is_dir(),
        "authorized_source_required": True,
    }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

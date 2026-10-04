#!/usr/bin/env python3
"""Build a CAMUS contiguous NumPy cache from an official source layout."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gdkvm_data.camus import build_npy_cache_auto


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help=(
            "Either download_camus.py output (database_split.zip + patients/) "
            "or canonical CAMUS_public (database_split/ + database_nifti/)"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--patient-limit", type=int)
    parser.add_argument("--available-only", action="store_true")
    args = parser.parse_args()

    output = build_npy_cache_auto(
        args.source,
        args.output,
        split=args.split,
        size=args.size,
        frame_count=args.frames,
        patient_limit=args.patient_limit,
        available_only=args.available_only,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build an EchoNet-Dynamic 10-frame cache from an authorized local source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gdkvm_data.echonet import build_echonet_npy_cache


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", type=str.upper, choices=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--limit-samples", type=int)
    args = parser.parse_args()

    try:
        output = build_echonet_npy_cache(
            args.source,
            args.output,
            split=args.split,
            size=args.size,
            frame_count=args.frames,
            limit_samples=args.limit_samples,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

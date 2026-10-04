#!/usr/bin/env python3
"""Verify a GDKVM checkpoint byte identity before use."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--bytes", type=int)
    args = parser.parse_args()

    if not args.path.is_file():
        raise SystemExit(f"checkpoint not found: {args.path}")

    actual_bytes = args.path.stat().st_size
    actual_sha = sha256_file(args.path)
    ok = actual_sha.lower() == args.sha256.lower()
    if args.bytes is not None:
        ok = ok and actual_bytes == args.bytes

    result = {
        "path_name": args.path.name,
        "actual_bytes": actual_bytes,
        "actual_sha256": actual_sha,
        "expected_bytes": args.bytes,
        "expected_sha256": args.sha256.lower(),
        "verified": ok,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

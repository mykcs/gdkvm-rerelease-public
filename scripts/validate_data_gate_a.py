#!/usr/bin/env python3
"""Run the deterministic Gate A validation for the dataset workstream."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--online-camus",
        action="store_true",
        help="also verify the current official CAMUS metadata/split endpoint",
    )
    args = parser.parse_args()

    run(sys.executable, "scripts/validate_repo.py")
    run(
        sys.executable,
        "-m",
        "unittest",
        "-v",
        "tests/test_data_contract.py",
        "tests/test_camus_converter.py",
        "tests/test_echonet_converter.py",
        "tests/test_benchmark_receipt.py",
    )

    if args.online_camus:
        with tempfile.TemporaryDirectory(prefix="gdkvm-camus-probe-") as tmp:
            receipt = Path(tmp) / "camus-official.json"
            run(
                sys.executable,
                "scripts/probe_camus_official.py",
                "--output",
                str(receipt),
            )

    print("DATA GATE A: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

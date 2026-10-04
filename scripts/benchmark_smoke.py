#!/usr/bin/env python3
"""Dataset-free smoke test for the shared benchmark receipt harness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gdkvm_bench import PhaseRecorder, environment_snapshot, write_receipt


def deterministic_work() -> int:
    return sum((index * index) % 997 for index in range(20_000))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-sha", required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=8)
    args = parser.parse_args()

    for _ in range(args.warmup):
        deterministic_work()

    recorder = PhaseRecorder()
    checksum = 0
    for _ in range(args.iterations):
        with recorder.phase("synthetic_cpu_step"):
            checksum ^= deterministic_work()

    receipt = {
        "schema_version": 1,
        "kind": "harness-smoke",
        "identity": {
            "code_sha": args.code_sha,
            "eval_protocol_id": "gdkvm-rerelease-v1",
            "runtime_profile": "dataset-free-harness-smoke",
        },
        "workload": {
            "data": "synthetic",
            "purpose": "validate receipt/timing plumbing only",
        },
        "environment": environment_snapshot(),
        "measurement": {
            "warmup_iterations": args.warmup,
            "measured_iterations": args.iterations,
            "phases": recorder.summary(),
            "checksum": checksum,
        },
    }
    write_receipt(args.output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

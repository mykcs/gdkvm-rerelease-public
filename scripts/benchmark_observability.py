#!/usr/bin/env python3
"""Dataset-free microbenchmark for the GDKVM observability layer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gdkvm_bench import environment_snapshot, write_receipt
from gdkvm_observability import ExperimentLogger, WandbSink


class InMemoryRun:
    def __init__(self) -> None:
        self.logs = []
        self.summary = {}

    def log(self, payload, *, step):
        self.logs.append((step, dict(payload)))


def measure(iterations: int, *, projected: bool) -> dict[str, float | int]:
    with tempfile.TemporaryDirectory(prefix="gdkvm-observe-bench-") as tmp:
        run = InMemoryRun()
        projection = WandbSink(run) if projected else None
        logger = ExperimentLogger(Path(tmp) / "events.jsonl", projection=projection)

        start = time.perf_counter()
        for step in range(iterations):
            logger.log_train_step(
                step=step,
                total_loss=1.0 / (step + 1),
                lr=2e-4,
                loss_components={"dice": 0.2, "ce": 0.8},
                grad_norm=1.5,
                performance={"step_s": 0.05},
            )
        elapsed = time.perf_counter() - start
        bytes_written = (Path(tmp) / "events.jsonl").stat().st_size
        return {
            "iterations": iterations,
            "elapsed_s": elapsed,
            "events_per_s": iterations / max(elapsed, 1e-12),
            "mean_us_per_event": elapsed * 1e6 / iterations,
            "journal_bytes": bytes_written,
            "remote_log_calls": len(run.logs),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-sha", required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()
    if args.iterations < 1:
        raise SystemExit("--iterations must be >= 1")

    local_only = measure(args.iterations, projected=False)
    projected = measure(args.iterations, projected=True)

    receipt = {
        "schema_version": 1,
        "kind": "observability-benchmark",
        "identity": {
            "code_sha": args.code_sha,
            "eval_protocol_id": "gdkvm-rerelease-v1",
            "runtime_profile": "dataset-free-observability",
        },
        "workload": {
            "events": "train-step scalar bundle",
            "projection": "in-memory W&B-compatible sink; no network",
        },
        "environment": environment_snapshot(),
        "measurement": {
            "warmup_iterations": 0,
            "measured_iterations": args.iterations,
            "local_only": local_only,
            "local_plus_projection": projected,
        },
    }
    write_receipt(args.output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

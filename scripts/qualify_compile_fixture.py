#!/usr/bin/env python3
"""Tiny eager-vs-compile qualification fixture.

This is a runtime/API correctness gate, not a GDKVM end-to-end performance claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


def synchronize(torch, device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def stats(values):
    ordered = sorted(values)
    mid = len(ordered) // 2
    median = (
        ordered[mid]
        if len(ordered) % 2
        else 0.5 * (ordered[mid - 1] + ordered[mid])
    )
    return {
        "n": len(values),
        "mean_s": sum(values) / len(values),
        "median_s": median,
        "min_s": min(values),
        "max_s": max(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-sha", required=True)
    parser.add_argument("--profile", default="fixture")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--modes",
        default="eager,default,reduce-overhead,max-autotune-no-cudagraphs",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    import torch.nn as nn

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    torch.manual_seed(17)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(17)

    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.GELU(),
        nn.Conv2d(16, 16, 3, padding=1),
        nn.GELU(),
        nn.Conv2d(16, 4, 1),
    ).eval().to(device)
    x = torch.randn(4, 3, 64, 64, device=device)

    with torch.inference_mode():
        reference = model(x).detach().clone()

    requested = [item.strip() for item in args.modes.split(",") if item.strip()]
    results = {}

    for mode in requested:
        if mode == "eager":
            candidate = model
        else:
            if not callable(getattr(torch, "compile", None)):
                results[mode] = {"status": "unsupported", "reason": "torch.compile missing"}
                continue
            try:
                candidate = torch.compile(model, mode=mode)
            except Exception as exc:
                results[mode] = {
                    "status": "compile-construction-failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
                continue

        try:
            synchronize(torch, device)
            cold_start = time.perf_counter()
            with torch.inference_mode():
                first = candidate(x)
            synchronize(torch, device)
            cold_s = time.perf_counter() - cold_start

            max_abs = float((first - reference).abs().max().item())
            denom = reference.abs().clamp_min(1e-8)
            max_rel = float(((first - reference).abs() / denom).max().item())

            for _ in range(args.warmup):
                with torch.inference_mode():
                    candidate(x)
            synchronize(torch, device)

            timings = []
            for _ in range(args.iterations):
                synchronize(torch, device)
                start = time.perf_counter()
                with torch.inference_mode():
                    candidate(x)
                synchronize(torch, device)
                timings.append(time.perf_counter() - start)

            results[mode] = {
                "status": "ok",
                "cold_first_call_s": cold_s,
                "steady_state": stats(timings),
                "max_abs_error_vs_eager": max_abs,
                "max_rel_error_vs_eager": max_rel,
            }
        except Exception as exc:
            results[mode] = {
                "status": "execution-failed",
                "reason": f"{type(exc).__name__}: {exc}",
            }

    receipt = {
        "schema_version": 1,
        "kind": "runtime-compile-fixture",
        "identity": {
            "code_sha": args.code_sha,
            "eval_protocol_id": "gdkvm-rerelease-v1",
            "runtime_profile": args.profile,
        },
        "workload": {
            "fixture": "3x-conv2d-gelu",
            "shape": [4, 3, 64, 64],
            "device": device,
            "dtype": "float32",
        },
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_runtime": getattr(torch.version, "cuda", None),
            "cuda_available": bool(torch.cuda.is_available()),
            "device_name": (
                torch.cuda.get_device_name(0)
                if device.startswith("cuda")
                else device
            ),
        },
        "measurement": {
            "warmup_iterations": args.warmup,
            "measured_iterations": args.iterations,
            "modes": results,
        },
        "warning": "Tiny fixture only; never report these timings as GDKVM end-to-end speedups.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

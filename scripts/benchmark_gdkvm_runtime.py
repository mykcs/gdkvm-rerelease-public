#!/usr/bin/env python3
"""Qualify eager/torch.compile model-forward behavior on a CUDA runtime.

This fixture intentionally uses random, deterministic weights and synthetic data.
It tests runtime/compiler compatibility and relative execution behavior, not model accuracy.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", default=["eager", "default"])
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()

def jsonable_counters(torch: Any) -> dict[str, dict[str, int]]:
    try:
        counters = torch._dynamo.utils.counters
    except Exception:
        return {}
    result: dict[str, dict[str, int]] = {}
    for group, values in counters.items():
        try:
            result[str(group)] = {str(k): int(v) for k, v in values.items()}
        except Exception:
            continue
    return result

def tensor_signature(torch: Any, output: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            detached = value.detach().float()
            result[key] = {
                "shape": list(value.shape),
                "sum": float(detached.sum().cpu()),
                "mean": float(detached.mean().cpu()),
            }
    return result

def compare_outputs(torch: Any, reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    details: dict[str, Any] = {}
    maxima: list[float] = []
    for key, ref in reference.items():
        cand = candidate.get(key)
        if not isinstance(ref, torch.Tensor) or not isinstance(cand, torch.Tensor):
            continue
        if ref.shape != cand.shape:
            details[key] = {"shape_match": False}
            continue
        diff = (ref.detach().float() - cand.detach().float()).abs()
        max_abs = float(diff.max().cpu()) if diff.numel() else 0.0
        mean_abs = float(diff.mean().cpu()) if diff.numel() else 0.0
        details[key] = {
            "shape_match": True,
            "max_abs": max_abs,
            "mean_abs": mean_abs,
        }
        maxima.append(max_abs)
    return {
        "max_abs_over_tensors": max(maxima) if maxima else None,
        "per_tensor": details,
    }

def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()
    sys.path.insert(0, str(source_root))

    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")

    # Avoid network downloads while preserving architecture.
    from model.utils import resnet
    original18 = resnet.resnet18
    original50 = resnet.resnet50
    resnet.resnet18 = lambda pretrained=True, **kwargs: original18(pretrained=False, **kwargs)
    resnet.resnet50 = lambda pretrained=True, **kwargs: original50(pretrained=False, **kwargs)
    from model.gdkvm01 import GDKVM

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    reference_model = GDKVM(
        model_type="small",
        image_encoder_type="resnet18",
        mask_encoder_type="resnet18",
    ).eval()
    state = copy.deepcopy(reference_model.state_dict())
    del reference_model

    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    rgb = torch.rand(
        (1, args.frames, 1, args.size, args.size),
        generator=generator,
        dtype=torch.float32,
    )
    ff_gt = torch.zeros((1, 1, 1, args.size, args.size), dtype=torch.long)
    input_cpu = {
        "rgb": rgb,
        "ff_gt": ff_gt,
        # Trainer normalizes this CPU metadata before the compiled model
        # boundary. Keep the runtime fixture on the same model-side contract.
        "info": {"num_objects": [1]},
    }

    def to_device_input() -> dict[str, Any]:
        return {
            "rgb": input_cpu["rgb"].to(device),
            "ff_gt": input_cpu["ff_gt"].to(device),
            # Keep metadata as Python integers, matching Trainer's normalized boundary.
            "info": {"num_objects": list(input_cpu["info"]["num_objects"])},
        }

    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "gdkvm-runtime-qualification",
        "identity": {
            "source_root": str(source_root),
            "source_gdkvm_sha256": None,
            "seed": args.seed,
        },
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
        },
        "workload": {
            "model": "small/resnet18/resnet18",
            "batch_size": 1,
            "frames": args.frames,
            "size": args.size,
            "dtype": "float32",
        },
        "modes": {},
    }

    import hashlib
    source_file = source_root / "model" / "gdkvm01.py"
    report["identity"]["source_gdkvm_sha256"] = hashlib.sha256(source_file.read_bytes()).hexdigest()

    if device.type == "cuda":
        report["environment"].update({
            "gpu_name": torch.cuda.get_device_name(device),
            "gpu_capability": list(torch.cuda.get_device_capability(device)),
            "device_count_visible": torch.cuda.device_count(),
        })

    eager_reference: dict[str, Any] | None = None

    for mode in args.modes:
        torch.manual_seed(args.seed)
        model = GDKVM(
            model_type="small",
            image_encoder_type="resnet18",
            mask_encoder_type="resnet18",
        )
        model.load_state_dict(state)
        model = model.to(device).eval()
        if device.type == "cuda":
            model = model.to(memory_format=torch.channels_last)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        try:
            if mode == "eager":
                executable = model
            else:
                try:
                    torch._dynamo.reset()
                    torch._dynamo.utils.counters.clear()
                except Exception:
                    pass
                executable = torch.compile(model, mode=mode)

            data = to_device_input()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            cold_start = time.perf_counter()
            with torch.inference_mode():
                first = executable(data)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            cold_s = time.perf_counter() - cold_start

            for _ in range(args.warmup):
                with torch.inference_mode():
                    executable(to_device_input())
            if device.type == "cuda":
                torch.cuda.synchronize(device)

            samples: list[float] = []
            last = first
            for _ in range(args.iterations):
                data = to_device_input()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
                with torch.inference_mode():
                    last = executable(data)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                samples.append(time.perf_counter() - start)

            if mode == "eager":
                eager_reference = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in last.items()}
                comparison = {"reference": True}
            elif eager_reference is not None:
                comparison = compare_outputs(
                    torch,
                    eager_reference,
                    {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in last.items()},
                )
            else:
                comparison = {"reference_missing": True}

            entry = {
                "status": "PASS",
                "cold_start_s": cold_s,
                "warmup_iterations": args.warmup,
                "measured_iterations": args.iterations,
                "mean_s": sum(samples) / len(samples),
                "min_s": min(samples),
                "max_s": max(samples),
                "samples_s": samples,
                "comparison_to_eager": comparison,
                "output_signature": tensor_signature(torch, last),
                "dynamo_counters": jsonable_counters(torch) if mode != "eager" else {},
            }
            if device.type == "cuda":
                entry["peak_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
                entry["peak_memory_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
            report["modes"][mode] = entry
        except Exception as exc:
            report["modes"][mode] = {
                "status": "FAIL",
                "error": f"{type(exc).__name__}: {exc}",
                "dynamo_counters": jsonable_counters(torch) if mode != "eager" else {},
            }
        finally:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

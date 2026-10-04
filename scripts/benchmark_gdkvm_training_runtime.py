#!/usr/bin/env python3
"""Synthetic GDKVM training-step runtime qualification.

This measures forward + backward + AdamW on deterministic synthetic data.
It is a runtime engineering fixture, not a scientific-training result.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("eager", "default"), default="eager")
    parser.add_argument("--amp", choices=("none", "fp16", "bf16"), default="none")
    parser.add_argument("--channels-last", choices=("on", "off"), default="on")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261001)
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


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()
    sys.path.insert(0, str(source_root))

    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    from model.utils import resnet

    original18 = resnet.resnet18
    original50 = resnet.resnet50
    resnet.resnet18 = lambda pretrained=True, **kwargs: original18(pretrained=False, **kwargs)
    resnet.resnet50 = lambda pretrained=True, **kwargs: original50(pretrained=False, **kwargs)

    from model.gdkvm01 import GDKVM

    device = torch.device(args.device)
    channels_last = args.channels_last == "on"
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    reference = GDKVM(
        model_type="small",
        image_encoder_type="resnet18",
        mask_encoder_type="resnet18",
        use_channels_last=channels_last,
    )
    initial_state = copy.deepcopy(reference.state_dict())
    del reference

    model = GDKVM(
        model_type="small",
        image_encoder_type="resnet18",
        mask_encoder_type="resnet18",
        use_channels_last=channels_last,
    )
    model.load_state_dict(initial_state)
    model = model.to(device).train()
    if channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.01,
        foreach=True,
    )

    if args.mode == "default":
        try:
            torch._dynamo.reset()
            torch._dynamo.utils.counters.clear()
        except Exception:
            pass
        executable = torch.compile(model, mode="default")
    else:
        executable = model

    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    rgb = torch.rand(
        (1, args.frames, 1, args.size, args.size),
        generator=generator,
        dtype=torch.float32,
    )
    ff_gt = torch.zeros((1, 1, 1, args.size, args.size), dtype=torch.long)

    def make_input() -> dict[str, Any]:
        return {
            "rgb": rgb.to(device),
            "ff_gt": ff_gt.to(device),
            "info": {"num_objects": [1]},
        }

    amp_enabled = args.amp != "none"
    amp_dtype = {
        "none": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.amp]
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda" and args.amp == "fp16"),
    )

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def step() -> float:
        optimizer.zero_grad(set_to_none=True)
        data = make_input()
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype if amp_enabled else None,
            enabled=amp_enabled,
        ):
            out = executable(data)
            logits = [
                value
                for key, value in out.items()
                if key.startswith("logits_") and isinstance(value, torch.Tensor)
            ]
            if not logits:
                raise RuntimeError("runtime fixture produced no logits")
            loss = sum(value.float().square().mean() for value in logits)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        return float(loss.detach().cpu())

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    synchronize()
    cold_start = time.perf_counter()
    first_loss = step()
    synchronize()
    cold_s = time.perf_counter() - cold_start

    warmup_losses = []
    for _ in range(args.warmup):
        warmup_losses.append(step())
    synchronize()

    timings = []
    losses = []
    for _ in range(args.iterations):
        synchronize()
        start = time.perf_counter()
        losses.append(step())
        synchronize()
        timings.append(time.perf_counter() - start)

    finite_losses = all(
        value == value and abs(value) != float("inf")
        for value in [first_loss, *warmup_losses, *losses]
    )

    report = {
        "schema_version": 1,
        "kind": "gdkvm-training-runtime-qualification",
        "identity": {
            "seed": args.seed,
            "runtime_mode": args.mode,
            "amp": args.amp,
            "channels_last": channels_last,
        },
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_runtime": getattr(torch.version, "cuda", None),
            "device": str(device),
        },
        "workload": {
            "model": "small/resnet18/resnet18",
            "batch_size": 1,
            "frames": args.frames,
            "size": args.size,
            "optimizer": "AdamW foreach",
            "loss": "mean-square over logits outputs",
            "scientific_training_result": False,
        },
        "measurement": {
            "cold_start_s": cold_s,
            "first_loss": first_loss,
            "warmup_iterations": args.warmup,
            "measured_iterations": args.iterations,
            "samples_s": timings,
            "losses": losses,
            "mean_s": sum(timings) / len(timings),
            "min_s": min(timings),
            "max_s": max(timings),
            "all_losses_finite": finite_losses,
            "dynamo_counters": jsonable_counters(torch) if args.mode != "eager" else {},
        },
    }
    if device.type == "cuda":
        report["environment"].update(
            {
                "gpu_name": torch.cuda.get_device_name(device),
                "gpu_capability": list(torch.cuda.get_device_capability(device)),
            }
        )
        report["measurement"]["peak_memory_allocated_bytes"] = int(
            torch.cuda.max_memory_allocated(device)
        )
        report["measurement"]["peak_memory_reserved_bytes"] = int(
            torch.cuda.max_memory_reserved(device)
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""One-step eager-vs-compile training equivalence qualification for GDKVM.

This checks numerical agreement at the point where compiler reordering enters:
forward loss, gradients, and one optimizer update from identical initial state.
It is not a scientific training result.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def tensor_diff(torch: Any, left: Any, right: Any) -> dict[str, float]:
    diff = (left.detach().float() - right.detach().float()).abs()
    denom = left.detach().float().abs().clamp_min(1e-8)
    return {
        "max_abs": float(diff.max().cpu()) if diff.numel() else 0.0,
        "mean_abs": float(diff.mean().cpu()) if diff.numel() else 0.0,
        "max_rel": float((diff / denom).max().cpu()) if diff.numel() else 0.0,
    }


def max_named_diff(torch: Any, left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    max_abs = 0.0
    max_mean_abs = 0.0
    for name in sorted(set(left) & set(right)):
        if left[name] is None or right[name] is None:
            continue
        row = tensor_diff(torch, left[name], right[name])
        rows[name] = row
        max_abs = max(max_abs, row["max_abs"])
        max_mean_abs = max(max_mean_abs, row["mean_abs"])
    return {
        "max_abs_over_tensors": max_abs,
        "max_mean_abs_over_tensors": max_mean_abs,
        "per_tensor": rows,
    }


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
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    base = GDKVM(
        model_type="small",
        image_encoder_type="resnet18",
        mask_encoder_type="resnet18",
        use_channels_last=True,
    )
    initial_state = copy.deepcopy(base.state_dict())
    del base

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

    def run(mode: str) -> dict[str, Any]:
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)

        model = GDKVM(
            model_type="small",
            image_encoder_type="resnet18",
            mask_encoder_type="resnet18",
            use_channels_last=True,
        )
        model.load_state_dict(initial_state)
        model = model.to(device).train()
        if device.type == "cuda":
            model = model.to(memory_format=torch.channels_last)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.01,
            foreach=True,
        )
        executable = model if mode == "eager" else torch.compile(model, mode="default")

        optimizer.zero_grad(set_to_none=True)
        out = executable(make_input())
        logits = [
            value
            for key, value in out.items()
            if key.startswith("logits_") and isinstance(value, torch.Tensor)
        ]
        if not logits:
            raise RuntimeError("no logits from GDKVM fixture")
        loss = sum(value.float().square().mean() for value in logits)
        loss.backward()

        gradients = {
            name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
            for name, parameter in model.named_parameters()
        }
        optimizer.step()
        parameters = {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters()
        }

        return {
            "loss": float(loss.detach().cpu()),
            "gradients": gradients,
            "parameters": parameters,
        }

    eager = run("eager")
    compiled = run("default")

    report = {
        "schema_version": 1,
        "kind": "gdkvm-training-one-step-equivalence",
        "identity": {
            "seed": args.seed,
            "compile_mode": "default",
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
            "optimizer": "AdamW foreach lr=1e-4 wd=0.01",
            "scientific_training_result": False,
        },
        "comparison": {
            "eager_loss": eager["loss"],
            "compiled_loss": compiled["loss"],
            "loss_abs_diff": abs(eager["loss"] - compiled["loss"]),
            "gradients": max_named_diff(
                torch, eager["gradients"], compiled["gradients"]
            ),
            "parameters_after_one_step": max_named_diff(
                torch, eager["parameters"], compiled["parameters"]
            ),
        },
        "warning": "One-step synthetic numerical qualification only; multi-step floating-point trajectories may diverge.",
    }

    if device.type == "cuda":
        report["environment"].update(
            {
                "gpu_name": torch.cuda.get_device_name(device),
                "gpu_capability": list(torch.cuda.get_device_capability(device)),
            }
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

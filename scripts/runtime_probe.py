#!/usr/bin/env python3
"""Emit a machine-readable runtime capability probe."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="unknown")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    result: dict[str, object] = {
        "schema_version": 1,
        "profile_id": args.profile,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }

    try:
        import torch
    except ImportError as exc:
        result["torch_available"] = False
        result["torch_error"] = f"{type(exc).__name__}: {exc}"
        text = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")
        print(text, end="")
        return 2 if args.require_cuda else 0

    result["torch_available"] = True
    result["torch_version"] = torch.__version__
    result["cuda_runtime"] = getattr(torch.version, "cuda", None)
    result["compile_available"] = callable(getattr(torch, "compile", None))
    result["cuda_available"] = bool(torch.cuda.is_available())
    result["cudnn_version"] = (
        torch.backends.cudnn.version()
        if hasattr(torch.backends, "cudnn")
        else None
    )

    if result["compile_available"]:
        try:
            result["compile_modes"] = sorted(torch._inductor.list_mode_options())
        except Exception as exc:
            result["compile_modes_error"] = f"{type(exc).__name__}: {exc}"

    if torch.cuda.is_available():
        devices = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_bytes": int(props.total_memory),
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
            )
        result["cuda_devices"] = devices
        try:
            result["tf32_matmul_allowed"] = bool(
                torch.backends.cuda.matmul.allow_tf32
            )
        except Exception:
            pass

    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")

    if args.require_cuda and not torch.cuda.is_available():
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

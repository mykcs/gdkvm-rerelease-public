"""Shared benchmark/reproduction receipt helpers."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from hashlib import sha256
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Iterator, Mapping, Sequence


SCHEMA_VERSION = 1


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command_probe(args: Sequence[str]) -> tuple[str | None, str | None]:
    """Return stdout on success or a compact diagnostic on failure."""
    try:
        result = subprocess.run(
            list(args),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    value = result.stdout.strip()
    if result.returncode == 0 and value:
        return value, None
    error = result.stderr.strip() or value or f"exit code {result.returncode}"
    return None, error


def environment_snapshot() -> dict[str, object]:
    snapshot: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "tempdir": tempfile.gettempdir(),
    }
    nvidia, nvidia_error = _command_probe(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ]
    )
    snapshot["nvidia_smi_available"] = bool(nvidia)
    if nvidia:
        snapshot["nvidia_gpus"] = [line.strip() for line in nvidia.splitlines()]
    elif nvidia_error:
        snapshot["nvidia_smi_error"] = nvidia_error
    try:
        import torch  # type: ignore
    except ImportError:
        snapshot["torch"] = None
    else:
        snapshot["torch"] = getattr(torch, "__version__", "unknown")
        snapshot["cuda_runtime"] = getattr(torch.version, "cuda", None)
        cudnn = None
        try:
            cudnn = torch.backends.cudnn.version()
        except Exception:
            pass
        snapshot["cudnn"] = cudnn
    return snapshot


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("values must be non-empty")
    if not 0 <= q <= 1:
        raise ValueError("q must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


@dataclass
class PhaseRecorder:
    synchronize: Callable[[], None] | None = None
    samples: dict[str, list[float]] = field(default_factory=dict)

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        if self.synchronize:
            self.synchronize()
        start = time.perf_counter()
        try:
            yield
        finally:
            if self.synchronize:
                self.synchronize()
            elapsed = time.perf_counter() - start
            self.samples.setdefault(name, []).append(elapsed)

    def summary(self) -> dict[str, dict[str, float | int]]:
        output: dict[str, dict[str, float | int]] = {}
        for name, values in self.samples.items():
            output[name] = {
                "n": len(values),
                "mean_s": statistics.fmean(values),
                "median_s": statistics.median(values),
                "p95_s": percentile(values, 0.95),
                "min_s": min(values),
                "max_s": max(values),
            }
        return output


def validate_receipt(receipt: Mapping[str, object]) -> None:
    required_top = {
        "schema_version",
        "kind",
        "identity",
        "workload",
        "environment",
        "measurement",
    }
    missing = sorted(required_top - set(receipt))
    if missing:
        raise ValueError(f"receipt missing top-level fields: {missing}")
    if receipt["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {receipt['schema_version']}")

    identity = receipt["identity"]
    if not isinstance(identity, Mapping):
        raise ValueError("identity must be an object")
    required_identity = {
        "code_sha",
        "eval_protocol_id",
        "runtime_profile",
    }
    missing_identity = sorted(required_identity - set(identity))
    if missing_identity:
        raise ValueError(f"receipt identity missing fields: {missing_identity}")

    measurement = receipt["measurement"]
    if not isinstance(measurement, Mapping):
        raise ValueError("measurement must be an object")
    for key in ("warmup_iterations", "measured_iterations"):
        if key not in measurement:
            raise ValueError(f"measurement missing {key}")

    if receipt["kind"] == "performance":
        required_performance_identity = {
            "checkpoint_sha256",
            "dataset_id",
            "split_id",
            "compile_mode",
        }
        missing_performance_identity = sorted(
            required_performance_identity - set(identity)
        )
        if missing_performance_identity:
            raise ValueError(
                "performance receipt identity missing fields: "
                f"{missing_performance_identity}"
            )

        workload = receipt["workload"]
        if not isinstance(workload, Mapping):
            raise ValueError("workload must be an object")
        required_workload = {
            "input_resolution",
            "sequence_length",
            "batch_size",
            "dtype",
            "amp",
            "world_size",
            "logging_profile",
            "data_cache_format",
        }
        missing_workload = sorted(required_workload - set(workload))
        if missing_workload:
            raise ValueError(
                f"performance receipt workload missing fields: {missing_workload}"
            )
        if "phases" not in measurement:
            raise ValueError("performance receipt measurement missing phases")


def write_receipt(path: str | Path, receipt: Mapping[str, object]) -> None:
    validate_receipt(receipt)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

"""Pure helpers for reproducible training identities."""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def json_safe(value: Any) -> Any:
    """Convert NumPy-backed receipt values into strict JSON-compatible values."""
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_manifest_identity(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"cache manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "manifest_sha256": sha256_file(manifest_path),
        "dataset": manifest.get("dataset"),
        "split": manifest.get("split") or manifest.get("selection", {}).get("split"),
        "format": manifest.get("format"),
        "shape": manifest.get("shape"),
        "sample_count": manifest.get("sample_count")
        or manifest.get("converted_samples")
        or (manifest.get("shape") or [None])[0],
        "data_protocol_id": manifest.get("data_protocol_id"),
        "partial_source": bool(manifest.get("partial_source", False)),
        "patient_count": manifest.get("patient_count"),
        "selection": manifest.get("selection"),
        "source_is_full_video_tree": manifest.get("source_is_full_video_tree"),
    }


def checkpoint_identity(path: str | Path, *, role: str) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "filename": path.name,
        "byte_size": path.stat().st_size,
        "sha256": sha256_file(path),
        "role": str(role),
    }


def memory_fraction_for_limit(limit_gib: float | None, total_memory_bytes: int) -> float | None:
    if limit_gib is None:
        return None
    limit_gib = float(limit_gib)
    if limit_gib <= 0:
        raise ValueError("memory_limit_gib must be > 0")
    if total_memory_bytes <= 0:
        raise ValueError("total_memory_bytes must be > 0")
    requested = limit_gib * (1024**3)
    return min(1.0, requested / float(total_memory_bytes))


def validate_formal_cache_identity(
    identity: dict[str, Any],
    *,
    role: str,
    expected_dataset: str,
    expected_samples: int,
) -> None:
    """Fail closed when a cache is not the declared full formal split."""
    role = str(role).lower()
    observed_dataset = str(identity.get("dataset") or "")
    if observed_dataset != str(expected_dataset):
        raise ValueError(
            f"{role} cache dataset {observed_dataset!r} != {expected_dataset!r}"
        )

    observed_split = str(identity.get("split") or "").lower()
    if observed_split != role:
        raise ValueError(
            f"{role} cache split {observed_split!r} does not match role {role!r}"
        )

    observed_count = identity.get("sample_count")
    if int(observed_count or -1) != int(expected_samples):
        raise ValueError(
            f"{role} cache sample_count {observed_count} != expected {expected_samples}"
        )

    if identity.get("partial_source"):
        raise ValueError(f"{role} cache is marked partial_source=true")

    selection = identity.get("selection")
    if isinstance(selection, dict) and selection.get("limit_samples") is not None:
        raise ValueError(f"{role} cache was built with a sample limit")

    if identity.get("source_is_full_video_tree") is False:
        raise ValueError(f"{role} cache was not built from the full source video tree")

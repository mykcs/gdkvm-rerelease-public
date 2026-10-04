#!/usr/bin/env python3
"""Download the public CAMUS NIfTI release from official CREATIS Girder."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import urllib.parse
import urllib.request
import zipfile


BASE = "https://humanheart-project.creatis.insa-lyon.fr/database/api/v1"
PATIENT_PARENT_ID = "63fde55f73e9f004868fb7ac"
SPLIT_FOLDER_ID = "66e27d12961576b1bad4e4e1"


def request_json(url: str, *, ipv4: bool = False):
    if ipv4:
        curl = shutil.which("curl")
        if curl is None:
            raise RuntimeError("curl is required for --ipv4")
        result = subprocess.run(
            [
                curl,
                "--ipv4",
                "--fail",
                "--location",
                "--silent",
                "--show-error",
                "--connect-timeout",
                "15",
                "--max-time",
                "60",
                "--retry",
                "5",
                "--retry-all-errors",
                url,
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=75,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"curl JSON request failed ({result.returncode}): "
                f"{result.stderr.decode(errors='replace').strip()}"
            )
        return json.loads(result.stdout)
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_zip(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            return archive.testzip() is None
    except zipfile.BadZipFile:
        return False


def list_patients(*, ipv4: bool = False) -> list[dict]:
    query = urllib.parse.urlencode(
        {
            "parentType": "folder",
            "parentId": PATIENT_PARENT_ID,
            "limit": 1000,
            "offset": 0,
            "sort": "name",
            "sortdir": 1,
        }
    )
    rows = request_json(f"{BASE}/folder?{query}", ipv4=ipv4)
    rows = sorted(rows, key=lambda row: row["name"])
    if len(rows) != 500:
        raise RuntimeError(f"official CAMUS folder count {len(rows)} != 500")
    expected = [f"patient{i:04d}" for i in range(1, 501)]
    if [row["name"] for row in rows] != expected:
        raise RuntimeError("official CAMUS patient folders are not patient0001..patient0500")
    return rows


def download(url: str, destination: Path, *, ipv4: bool = False) -> None:
    """Download with resumable retries because the public Girder can close streams."""
    if valid_zip(destination):
        return
    curl = shutil.which("curl")
    if curl is None:
        raise RuntimeError("curl is required for resumable CAMUS downloads")
    part = destination.with_suffix(destination.suffix + ".part")
    command = [
        curl,
        *(["--ipv4"] if ipv4 else []),
        "--fail",
        "--location",
        "--retry",
        "20",
        "--retry-delay",
        "2",
        "--retry-all-errors",
        "--continue-at",
        "-",
        "--output",
        str(part),
        url,
    ]
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"curl failed with exit code {result.returncode}: {url}")
    if not valid_zip(part):
        raise RuntimeError(f"download completed but zip validation failed: {url}")
    part.replace(destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--split", choices=("train", "val", "test"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ipv4", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 64:
        raise SystemExit("--workers must be between 1 and 64")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be >= 1")

    output = args.output
    patient_dir = output / "patients"
    output.mkdir(parents=True, exist_ok=True)
    patient_dir.mkdir(parents=True, exist_ok=True)

    # Download/validate the split before choosing a subset so --limit is
    # deterministic inside the requested scientific split rather than by
    # numeric patient ID.
    split_zip = output / "database_split.zip"
    download(f"{BASE}/folder/{SPLIT_FOLDER_ID}/download", split_zip, ipv4=args.ipv4)
    with zipfile.ZipFile(split_zip) as archive:
        split_mapping = {}
        names = {
            "subgroup_training.txt": "train",
            "subgroup_validation.txt": "val",
            "subgroup_testing.txt": "test",
        }
        for member in archive.namelist():
            name = Path(member).name
            if name in names:
                split_mapping[names[name]] = [
                    line.strip()
                    for line in archive.read(member).decode("utf-8").splitlines()
                    if line.strip()
                ]
    if set(split_mapping) != {"train", "val", "test"}:
        raise RuntimeError("official CAMUS split archive is incomplete")
    canonical_split = json.dumps(
        split_mapping, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")

    rows = list_patients(ipv4=args.ipv4)
    if args.split is not None:
        allowed = set(split_mapping[args.split])
        rows = [row for row in rows if row["name"] in allowed]
        if len(rows) != len(allowed):
            raise RuntimeError(
                f"official folder inventory does not match {args.split} split"
            )
    if args.limit is not None:
        rows = rows[: args.limit]

    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                download,
                f"{BASE}/folder/{row['_id']}/download",
                patient_dir / f"{row['name']}.zip",
                ipv4=args.ipv4,
            ): row
            for row in rows
        }
        for future in as_completed(futures):
            row = futures[future]
            try:
                future.result()
                print(f"downloaded {row['name']}", flush=True)
            except Exception as exc:
                failures.append(
                    {"name": row["name"], "error": f"{type(exc).__name__}: {exc}"}
                )
                print(f"FAILED {row['name']}: {exc}", flush=True)

    expected_complete_count = (
        len(split_mapping[args.split]) if args.split is not None else 500
    )
    manifest = {
        "source": "official-creatis-girder",
        "api_base": BASE,
        "patient_parent_id": PATIENT_PARENT_ID,
        "split_folder_id": SPLIT_FOLDER_ID,
        "selection": {
            "split": args.split,
            "limit": args.limit,
        },
        "requested_patients": len(rows),
        "complete_for_selection": not failures
        and args.limit is None
        and len(rows) == expected_complete_count,
        "failures": failures,
        "split_archive": {
            "bytes": split_zip.stat().st_size,
            "transport_sha256": sha256_file(split_zip),
            "content_sha256": sha256(canonical_split).hexdigest(),
            "counts": {key: len(value) for key, value in split_mapping.items()},
        },
        "patients": [],
    }
    for row in rows:
        archive = patient_dir / f"{row['name']}.zip"
        if valid_zip(archive):
            manifest["patients"].append(
                {
                    "name": row["name"],
                    "girder_folder_id": row["_id"],
                    "girder_uncompressed_bytes": int(row.get("size", 0)),
                    "archive_bytes": archive.stat().st_size,
                    "archive_sha256": sha256_file(archive),
                }
            )

    (output / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if failures:
        raise SystemExit(f"{len(failures)} CAMUS downloads failed")
    print(f"CAMUS download manifest: {output / 'download_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

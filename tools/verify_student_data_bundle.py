#!/usr/bin/env python3
"""Verify installed student data against its SHA-256 manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(manifest_path: Path) -> tuple[int, int]:
    manifest_path = manifest_path.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    files = manifest.get("files", [])
    for item in files:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            failures.append(f"unsafe manifest path: {relative}")
            continue
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
            continue
        if path.stat().st_size != int(item["size_bytes"]):
            failures.append(f"size mismatch: {relative}")
            continue
        if _sha256(path) != item["sha256"]:
            failures.append(f"checksum mismatch: {relative}")
    if failures:
        raise ValueError("Student data verification failed:\n" + "\n".join(failures))
    return len(files), sum(int(item["size_bytes"]) for item in files)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest",
        nargs="?",
        type=Path,
        default=ROOT / "student_data_bundle" / "manifest.json",
    )
    args = parser.parse_args(argv)
    file_count, total_bytes = verify(args.manifest)
    print(f"Verified {file_count} files ({total_bytes} bytes).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

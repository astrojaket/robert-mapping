#!/usr/bin/env python3
"""Verify or restore the archived UNITY profile arrays.

The script uses paths relative to its own file. Run it from any working
directory with the Python standard library.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "archive" / "profiles_manifest.json"
MAX_PART_BYTES = 40 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest() -> dict[str, Any]:
    try:
        with MANIFEST.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Manifest not found: {MANIFEST}") from exc
    if manifest.get("schema_version") != 1:
        raise RuntimeError("Unsupported profile archive manifest schema")
    if not isinstance(manifest.get("studies"), dict):
        raise RuntimeError("Manifest has no studies object")
    return manifest


def checked_paths(study: str, record: dict[str, Any]) -> list[tuple[Path, dict[str, Any]]]:
    paths: list[tuple[Path, dict[str, Any]]] = []
    parts = record.get("parts")
    if not isinstance(parts, list) or not parts:
        raise RuntimeError(f"{study}: manifest has no parts")
    for part in parts:
        relative = Path(part["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"{study}: unsafe part path {relative}")
        path = MANIFEST.parent / relative
        paths.append((path, part))
    return paths


def verify_study(study: str, record: dict[str, Any]) -> None:
    digest = hashlib.sha256()
    byte_count = 0
    for path, part in checked_paths(study, record):
        if not path.is_file():
            raise RuntimeError(f"{study}: missing part {path.relative_to(MANIFEST.parent)}")
        size = path.stat().st_size
        if size != part.get("bytes"):
            raise RuntimeError(f"{study}: wrong byte count for {path.name}: {size}")
        if size > MAX_PART_BYTES:
            raise RuntimeError(f"{study}: part exceeds 40 MiB: {path.name}")
        part_hash = sha256_file(path)
        if part_hash != part.get("sha256"):
            raise RuntimeError(f"{study}: SHA-256 mismatch for {path.name}")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
                byte_count += len(block)
    if byte_count != record.get("bytes"):
        raise RuntimeError(f"{study}: combined byte count does not match manifest")
    if digest.hexdigest() != record.get("sha256"):
        raise RuntimeError(f"{study}: combined SHA-256 does not match manifest")


def verify_all(manifest: dict[str, Any], only_study: str | None = None) -> list[str]:
    studies = manifest["studies"]
    if only_study is not None and only_study not in studies:
        raise RuntimeError(f"Unknown study: {only_study}")
    names = [only_study] if only_study is not None else sorted(studies)
    for name in names:
        verify_study(name, studies[name])
    return names


def restore_study(study: str, record: dict[str, Any], force: bool) -> str:
    verify_study(study, record)
    relative_target = Path(record["restore_path"])
    if relative_target.is_absolute() or ".." in relative_target.parts:
        raise RuntimeError(f"{study}: unsafe restore path {relative_target}")
    target = ROOT / relative_target
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        if not target.is_file():
            raise RuntimeError(f"{study}: restore target exists and is not a file: {target}")
        current_hash = sha256_file(target)
        if current_hash == record["sha256"] and target.stat().st_size == record["bytes"]:
            return f"{study}: already restored and verified ({target.relative_to(ROOT)})"
        if not force:
            raise RuntimeError(
                f"{study}: {target.relative_to(ROOT)} exists with different contents; "
                "use --force to replace it"
            )

    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as output:
            for part_path, _part in checked_paths(study, record):
                with part_path.open("rb") as part_stream:
                    for block in iter(lambda: part_stream.read(1024 * 1024), b""):
                        output.write(block)
            output.flush()
            os.fsync(output.fileno())
        if temporary.stat().st_size != record["bytes"] or sha256_file(temporary) != record["sha256"]:
            raise RuntimeError(f"{study}: restored temporary file failed verification")
        os.replace(temporary, target)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return f"{study}: restored and verified ({target.relative_to(ROOT)})"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--restore", action="store_true", help="verify and restore profile files")
    action.add_argument(
        "--verify-restored",
        action="store_true",
        help="verify complete profile files already in reference/",
    )
    parser.add_argument("--study", help="limit --restore or verification to one study")
    parser.add_argument("--force", action="store_true", help="replace a different restored file")
    args = parser.parse_args(argv)
    if args.force and not args.restore:
        parser.error("--force requires --restore")

    try:
        manifest = read_manifest()
        names = verify_all(manifest, args.study)
        if args.verify_restored:
            for name in names:
                record = manifest["studies"][name]
                target = ROOT / record["restore_path"]
                if not target.is_file():
                    raise RuntimeError(f"{name}: restored file not found: {target.relative_to(ROOT)}")
                if target.stat().st_size != record["bytes"] or sha256_file(target) != record["sha256"]:
                    raise RuntimeError(f"{name}: restored file failed verification")
                print(f"{name}: restored file verified ({target.relative_to(ROOT)})")
        elif args.restore:
            for name in names:
                print(restore_study(name, manifest["studies"][name], args.force))
        else:
            for name in names:
                print(f"{name}: archive parts verified")
        return 0
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

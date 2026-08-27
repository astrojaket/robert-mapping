#!/usr/bin/env python3
"""Download published light curves without running scientific analysis."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path
import shutil
import sys
from typing import Any
from urllib.request import Request, urlopen

import yaml


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch author-released eclipse and phase-curve data only."
    )
    parser.add_argument("catalog", type=Path)
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--workers", type=int, default=3, choices=(1, 2, 3))
    parser.add_argument("--list", action="store_true")
    return parser


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict) or not isinstance(value.get("datasets"), list):
        raise ValueError("catalog must contain a datasets list")
    return value


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - archive checksum, not security
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fetch(root: Path, dataset: dict[str, Any], item: dict[str, Any]) -> str:
    target = root / str(dataset["directory"]) / "source" / str(item["name"])
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = item.get("md5")
    if target.exists() and (expected is None or _md5(target) == expected):
        return f"ready   {dataset['id']}: {target.name}"
    partial = target.with_name(target.name + ".partial")
    request = Request(str(item["url"]), headers={"User-Agent": "robert-mapping/0.1"})
    with urlopen(request, timeout=120) as response, partial.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024)
    if expected is not None and _md5(partial) != expected:
        raise ValueError(f"checksum failed for {dataset['id']}: {target.name}")
    partial.replace(target)
    return f"fetched {dataset['id']}: {target.name}"


def main() -> int:
    args = _parser().parse_args()
    catalog_path = args.catalog.resolve()
    catalog = _load(catalog_path)
    datasets = catalog["datasets"]
    selected_ids = set(args.dataset)
    if selected_ids:
        known = {str(item["id"]) for item in datasets}
        unknown = sorted(selected_ids - known)
        if unknown:
            raise ValueError(f"unknown dataset id(s): {', '.join(unknown)}")
        datasets = [item for item in datasets if item["id"] in selected_ids]
    if args.list:
        for item in datasets:
            print(f"{item['id']:<36} {item['state']}")
        return 0

    work: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for dataset in datasets:
        for item in dataset.get("files", []):
            work.append((dataset, item))
    root = catalog_path.parent
    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_fetch, root, dataset, item): (dataset, item)
            for dataset, item in work
        }
        for future in as_completed(futures):
            dataset, item = futures[future]
            try:
                print(future.result())
            except Exception as exc:  # keep other independent downloads running
                failures += 1
                print(
                    f"failed  {dataset['id']}: {item['name']}: {exc}",
                    file=sys.stderr,
                )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

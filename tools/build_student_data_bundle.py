#!/usr/bin/env python3
"""Build a checksum-verified student data bundle.

The bundle contains the local data needed for the WASP-18b benchmark and the
current WASP-121b study. It does not contain fit results or simulated data.
The archive paths are relative to the repository root, so a student can
extract the archive directly into a clean clone.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import tarfile
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "dist" / f"robert-mapping-student-data-{date.today().isoformat()}.tar.gz"

WASP18_ROOT = (
    ROOT
    / "literature_data"
    / "WASP-18b"
    / "JWST-NIRISS-SOSS"
    / "source"
    / "WASP-18b 3D Mapping Archive"
)

WASP18_FILES = (
    WASP18_ROOT / "eigenspectra" / "spec_lambin_25.npz",
    WASP18_ROOT / "theresa" / "inputs" / "spec_lambin_25.npz",
    WASP18_ROOT / "theresa" / "inputs" / "niriss-firstorder.txt",
    WASP18_ROOT / "theresa" / "inputs" / "phoenix-spectrum.txt",
)

EXCLUDED_NAMES = {".DS_Store"}
EXCLUDED_SUFFIXES = {".partial"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _valid_file(path: Path) -> bool:
    return (
        path.is_file()
        and path.name not in EXCLUDED_NAMES
        and path.suffix not in EXCLUDED_SUFFIXES
    )


def selected_files() -> tuple[Path, ...]:
    """Return the allow-listed source and prepared data files."""

    selected: set[Path] = set()

    wasp121 = ROOT / "literature_data" / "WASP-121b"
    if wasp121.is_dir():
        selected.update(path for path in wasp121.rglob("*") if _valid_file(path))

    # These are the original local SOSS products used to make the canonical
    # audit-only prepared files. They are not production-ready observations.
    soss_originals = ROOT / "WASP-121b" / "SOSS"
    if soss_originals.is_dir():
        selected.update(path for path in soss_originals.rglob("*") if _valid_file(path))

    selected.update(path for path in WASP18_FILES if _valid_file(path))

    prepared_wasp18 = (
        ROOT
        / "literature_data"
        / "WASP-18b"
        / "JWST-NIRISS-SOSS"
        / "prepared"
        / "25bin"
    )
    if prepared_wasp18.is_dir():
        selected.update(path for path in prepared_wasp18.rglob("*") if _valid_file(path))

    # Published WASP-18b temperature maps are used by the one-to-one
    # wavelength comparison. The large unrelated sphere arrays are excluded.
    reference = WASP18_ROOT / "eigenspectra" / "Figure1"
    selected.update(path for path in reference.glob("temp_wave_*.npz") if _valid_file(path))

    return tuple(sorted(selected, key=_relative))


def _state_for(path: Path) -> str:
    relative = _relative(path)
    if "WASP-18b" in relative:
        return "benchmark_ready"
    if relative.endswith(("white_nrs1.npz", "white_nrs2.npz", "white_light_curve.npz")):
        if "JWST-NIRSpec-G395H" in relative or "JWST-MIRI-LRS" in relative:
            return "production_white_light_ready"
    if "JWST-NIRSpec-G395H/prepared/spectroscopic" in relative:
        return "prepared_spectral_requires_map_validation"
    if "JWST-MIRI-LRS/prepared/spectroscopic" in relative:
        return "prepared_spectral_requires_map_validation"
    if "/source/" in relative:
        return "source_or_provenance"
    return "audit_only_not_production_ready"


def _manifest(files: Iterable[Path]) -> dict[str, object]:
    entries = []
    for path in files:
        entries.append(
            {
                "path": _relative(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "state": _state_for(path),
            }
        )
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repository": "https://github.com/astrojaket/robert-mapping",
        "contains_results": False,
        "contains_simulations": False,
        "file_count": len(entries),
        "total_uncompressed_bytes": sum(item["size_bytes"] for item in entries),
        "sources": {
            "WASP-18b JWST/NIRISS SOSS": "https://doi.org/10.5281/zenodo.14751570",
            "WASP-121b JWST/NIRSpec G395H": "https://zenodo.org/records/20651891",
            "WASP-121b JWST/MIRI LRS": "https://zenodo.org/records/20767846",
            "WASP-121b HST/WFC3 G141": "https://doi.org/10.1038/s41550-021-01592-w",
        },
        "important_limits": [
            "Only the WASP-121b NIRSpec NRS1, NRS2, and MIRI broadband white curves are current production inputs.",
            "WASP-121b NIRISS, HST, TESS, and SMARTS products remain audit-only for the reasons in docs/wasp121b_observation_suite.md.",
            "The prepared spectral products require white-light validation and injection recovery before scientific map fits.",
            "Large posterior result folders are deliberately excluded.",
        ],
        "files": entries,
    }


def _readme_text() -> str:
    return """robert-mapping student data bundle

Extract this archive in the root of a clean robert-mapping clone:

    tar -xzf robert-mapping-student-data-YYYY-MM-DD.tar.gz -C /path/to/robert-mapping

Then activate the eclipse-mapping environment and run:

    python tools/verify_student_data_bundle.py student_data_bundle/manifest.json
    robert-mapping doctor

Start with docs/student_learning_path.md.

The bundle contains source and prepared data. It contains no posterior results
and no simulated light curves. A file being present does not mean that it is a
production-ready map input. Read each state in student_data_bundle/manifest.json
and the audit in docs/wasp121b_observation_suite.md.
"""


def _add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    info.mtime = 0
    info.mode = 0o644
    archive.addfile(info, io.BytesIO(payload))


def verify_archive(path: Path) -> tuple[int, int]:
    """Verify every archived data file against the embedded manifest."""

    with tarfile.open(path, mode="r:gz") as archive:
        manifest_member = archive.extractfile("student_data_bundle/manifest.json")
        if manifest_member is None:
            raise ValueError("The bundle has no embedded manifest.")
        manifest = json.load(manifest_member)
        members = {member.name: member for member in archive.getmembers()}
        for item in manifest["files"]:
            member = members.get(item["path"])
            if member is None or not member.isfile():
                raise ValueError(f"The bundle is missing {item['path']}.")
            if member.size != int(item["size_bytes"]):
                raise ValueError(f"The archived size is wrong for {item['path']}.")
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError(f"Could not read {item['path']} from the bundle.")
            digest = hashlib.sha256()
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
            if digest.hexdigest() != item["sha256"]:
                raise ValueError(f"The archived checksum is wrong for {item['path']}.")
    return int(manifest["file_count"]), int(manifest["total_uncompressed_bytes"])


def build_bundle(output: Path) -> tuple[Path, dict[str, object]]:
    files = selected_files()
    if not files:
        raise FileNotFoundError("No student data files were found.")
    missing = [path for path in WASP18_FILES if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required WASP-18b files: " + ", ".join(map(str, missing)))

    manifest = _manifest(files)
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    try:
        with tarfile.open(partial, mode="w:gz", compresslevel=6) as archive:
            _add_bytes(archive, "student_data_bundle/README.txt", _readme_text().encode("utf-8"))
            _add_bytes(
                archive,
                "student_data_bundle/manifest.json",
                (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
            )
            for path in files:
                archive.add(path, arcname=_relative(path), recursive=False)
        partial.replace(output)
    finally:
        if partial.exists():
            partial.unlink()

    verify_archive(output)

    sidecar = output.with_suffix(output.suffix + ".sha256")
    sidecar.write_text(f"{_sha256(output)}  {output.name}\n", encoding="utf-8")
    manifest_sidecar = output.with_suffix(output.suffix + ".manifest.json")
    manifest_sidecar.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output, manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    output, manifest = build_bundle(args.output)
    print(f"Bundle: {output}")
    print(f"Files: {manifest['file_count']}")
    print(f"Uncompressed bytes: {manifest['total_uncompressed_bytes']}")
    print(f"Archive bytes: {output.stat().st_size}")
    print(f"SHA256: {_sha256(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Prepare WASP-121b NIRSpec G395H spectroscopic light curves.

This is a data-only command.  It validates the nine public Zenodo arrays,
splits NRS1 and NRS2 by the wavelength gap, and writes compact detector NPZ
files.  It never runs a fit or a simulation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robert_mapping.data.nirspec_spectroscopic import (
    DEFAULT_OUTPUT_DIRECTORY,
    DEFAULT_SOURCE_DIRECTORY,
    prepare_wasp121_nirspec_spectroscopic,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and prepare the WASP-121b NIRSpec spectroscopic release "
            "without inference or simulation."
        )
    )
    parser.add_argument(
        "--source-directory",
        type=Path,
        default=DEFAULT_SOURCE_DIRECTORY,
        help="Directory containing the nine downloaded source text files.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory for the detector NPZ products and audit manifest.",
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=None,
        help="Optional checksum manifest; default is download_manifest.json in the source directory.",
    )
    parser.add_argument(
        "--skip-checksums",
        action="store_true",
        help="Skip MD5 comparison with the source manifest (shape checks still run).",
    )
    args = parser.parse_args()
    audit = prepare_wasp121_nirspec_spectroscopic(
        args.source_directory,
        args.output_directory,
        source_manifest=args.source_manifest,
        verify_checksums=not args.skip_checksums,
    )
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

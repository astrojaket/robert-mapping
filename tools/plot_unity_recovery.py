#!/usr/bin/env python3
"""Compatibility entry point; use UNITY_Theory/code for new UNITY work."""
from pathlib import Path
import runpy

CANONICAL = Path(__file__).resolve().parents[1] / "UNITY_Theory" / "code" / "plot_unity_recovery.py"
if __name__ == "__main__":
    runpy.run_path(str(CANONICAL), run_name="__main__")
else:
    # Keep direct imports used by earlier notebooks working.
    globals().update({key: value for key, value in runpy.run_path(str(CANONICAL)).items()
                      if not key.startswith("__")})

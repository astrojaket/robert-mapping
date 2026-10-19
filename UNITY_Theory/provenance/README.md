# UNITY archive and provenance notes

The copied validation files and artifact manifests are historical snapshots from `results/unity_erc_20261009`. Their embedded hashes describe the original run files. Path cleanup changes some copied JSON and text files, so historical hashes do not validate those edited copies. The package manifest records each original and packaged hash.

The four raw `profiles.npz` files are stored as lossless parts under `archive/`. Use `archive_profiles.py --restore` to rebuild them under `reference/`.

Absolute local paths are removed from copied JSON and text metadata. CSV files and online PandExo pickle files are copied byte for byte.

Seed pairing for standardized 50 and 100 ppm sensitivity runs: both noise levels use the same deterministic trial seed for each target. Seed derivation fixes noise index at 0 and uses the master seed, target index, and trial index. The expected-noise input in `data/inputs/unity_expected_noise.json` has one PandExo-derived noise level per target and a separate 100-trial seed note. Some historical notes describe an earlier one-draw run or an older seed description. Treat those notes as source snapshot history. Use the packaged input and current run metadata for the current run scope.

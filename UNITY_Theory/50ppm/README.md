# 50ppm

This folder shows the saved run from input settings through injected and retrieved maps. This case uses a noise input of 50 ppm for each target.

The organizer does not run fits. It selects the saved rows for this noise case. The saved trial number stays in every table. The trial 0 light curves and maps use the saved row `trial = 0`; they do not use a row chosen by its fit result.

The saved `seed_target_index` stays with each target. The run uses paired standard noise draws across noise levels and scenarios. The trial number and seed index keep that pairing when you rerun one target.

## Folders

- `01_initialisation/` has the case input and saved run settings.
- `02_injection/` has the injected degree-2 map coefficients for each target and scenario.
- `03_light_curves/` has one CSV file per target. These files contain trial-0 rows for both fit scopes.
- `04_retrieval/recovery_trials.csv` has every saved draw and fit scope for this noise case.
- `04_retrieval/retrieved_map_coefficients.csv` has the nine best-fit harmonic coefficients for every recovery row.
- `04_retrieval/profiles_source.json` points to the shared fit-profile archive.
- `05_summary/` has the case summary, map figures, and figure caption.

The main result has 6 targets, 100 draws per target, and these scenarios: temperature_scaled, zero_offset, constant_30_deg, uniform_null. It has one row for each draw and fit scope. The case summary gives detection counts out of all draws. It also gives central 1σ and 2σ ranges for longitude bias and recovered longitude, plus the bias standard deviation. The map coefficients use the canonical degree-2 spherical-harmonic basis. Injected maps have unit planetary full-phase flux. Retrieved maps use the saved uniform and hotspot amplitudes. The `depth_ppm` value converts map intensity to stellar-ppm scale. The right map is the conditional best fit from trial 0, even when its saved detection result is false.

Run the organizer again with `python UNITY_Theory/run.py organize --source reference`. After a new complete run, use `python UNITY_Theory/run.py organize --source generated`.

To run this case from its saved input alone, use:

```sh
python UNITY_Theory/code/unity_injection_recovery.py --inputs UNITY_Theory/50ppm/01_initialisation/input.json --output UNITY_Theory/generated/50ppm --trials 100 --exposure-subsamples 8 --disk-radial 32 --disk-azimuth 128 --chunk-size 128
```

After running this case alone, rebuild its maps from that output with:

```sh
python UNITY_Theory/code/build_case_folders.py --source generated --case 50ppm --bundle-directory UNITY_Theory/generated/50ppm --output-root UNITY_Theory/generated/cases --expected-trials 100
```

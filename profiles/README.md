# Compute profiles

`robert-mapping` uses one Conda environment name on all supported machines:
`eclipse-mapping`.

The profiles in this directory keep the numerical backend on the CPU and use
one numerical thread until the YAML settings are loaded. Use the profile that matches the machine before
starting Python. Do not mix the Apple Silicon environment with a Linux
environment.

## Laptop (Apple Silicon)

```bash
conda env create --file environment-osx-arm64.yml
conda activate eclipse-mapping
source profiles/laptop.env
robert-mapping doctor examples/production_wasp43b_nuts.yml
robert-mapping validate examples/production_wasp43b_nuts.yml
```

If the environment already exists, use `conda env update` with the same file.

## Glamdring (Linux workstation)

```bash
conda env create --file environment-linux-64-cpu.yml
conda activate eclipse-mapping
source profiles/glamdring.env
robert-mapping doctor examples/production_wasp43b_nuts.yml
robert-mapping validate examples/production_wasp43b_nuts.yml
```

## SLURM CPU job

Create the environment once on a login node. Load the site Conda module first
if your cluster requires modules. Then submit a job with:

```bash
sbatch profiles/slurm_fit.sbatch examples/production_wasp178b_whitened.yml
```

The batch template requests one task and three CPUs by default. Change the
allocation with `sbatch --cpus-per-task=8 profiles/slurm_fit.sbatch CONFIG`.
The template passes the allocated CPU count to `fit --cores`. It does
not contain a site-specific partition, account, or module name. Add those
settings in a local copy if your cluster requires them.

The optional second argument changes the output directory for that job:

```bash
sbatch profiles/slurm_fit.sbatch examples/production_wasp178b_whitened.yml \
  results/slurm_wasp178b
```

## Configuration rule

Every fit configuration must contain this compute block, or use these values
as its defaults:

```yaml
compute:
  profile: local       # use slurm in a SLURM-specific copy
  jax_platform: cpu
  max_cpus: 3          # total CPU budget
  workers: 1          # independent target/fold processes where supported
  threads: 1          # numerical threads per process
```

The three-CPU value is a default, not a fixed limit. Keep workers times threads
within the CPU budget. Sampling chains are separate from the worker count.
All requested chains run; they run in sequence if the available JAX devices
or CPU budget cannot support parallel chains. See
[parallel execution](../docs/parallel_execution.md) for the command-specific controls.

## Reproducible environments

The environment YAML files specify dependency ranges. They are not solved
lock files. For a repeatable installation, generate a lock file from the
shared specification on a machine with `conda-lock`:

```bash
conda-lock lock --file environment.yml --platform osx-arm64 --platform linux-64
```

Install the lock file that matches the target platform. Never use an
`osx-arm64` lock file on Glamdring or a SLURM worker.

Test the solved environment on each target platform before the student
handoff. Commit the generated lock file after those checks pass.

# Compute profiles

`robert-mapping` uses one Conda environment name on all supported machines:
`eclipse-mapping`.

The profiles in this directory keep the numerical backend on the CPU and cap
the run at three CPUs. Use the profile that matches the machine before
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

The batch template requests one task and three CPUs. It sets the same
three-thread CPU environment as the laptop and Glamdring profiles. It does
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
  max_cpus: 3
  threads: 3
```

Use no more than three parallel chains in a CPU fit. The current NumPyro
backend launches parallel chains on separate host devices, so the chain count
and CPU cap must agree.

## Reproducible environments

For a repeatable installation, generate platform lock files from the shared
specification on a machine with `conda-lock`:

```bash
conda-lock lock --file environment.yml --platform osx-arm64 --platform linux-64
```

Install the lock file that matches the target platform. Never use an
`osx-arm64` lock file on Glamdring or a SLURM worker.

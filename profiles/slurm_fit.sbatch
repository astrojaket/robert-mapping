#!/usr/bin/env bash
#SBATCH --job-name=robert-mapping
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --output=robert-mapping-%j.out
#SBATCH --error=robert-mapping-%j.err
#SBATCH --time=01:00:00

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: sbatch profiles/slurm_fit.sbatch CONFIG [OUTPUT_DIR]" >&2
    exit 2
fi

cd "${SLURM_SUBMIT_DIR:-$PWD}"

if ! command -v conda >/dev/null 2>&1; then
    echo "Conda is not available. Load your site's Conda module before sbatch." >&2
    exit 1
fi

# A non-interactive SLURM shell does not load the Conda shell function by
# default. This makes activation explicit and keeps the environment portable.
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate eclipse-mapping
source profiles/slurm.env

CONFIG="$1"
if [[ $# -eq 2 ]]; then
    OUTPUT_ARGS=(--output-dir "$2")
else
    OUTPUT_ARGS=()
fi

srun --ntasks=1 --cpus-per-task=3 robert-mapping doctor "$CONFIG"
srun --ntasks=1 --cpus-per-task=3 robert-mapping fit "$CONFIG" "${OUTPUT_ARGS[@]}"

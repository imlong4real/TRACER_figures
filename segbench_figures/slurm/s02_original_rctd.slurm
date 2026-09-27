#!/bin/bash
#SBATCH --account=adeshpa6_tracer
#SBATCH --partition=shared
#SBATCH --job-name=sb_orig
#SBATCH --output=/scratch4/adeshpa6/tracer_campaign/final/logs/segbench_v2/%x_%j.log
# usage: sbatch [--cpus/--mem/--time] s02_original_rctd.slurm <mode> <setting> [method]
set -euo pipefail
export PYTHONPATH=/scratch4/adeshpa6/TRACER/src
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export RETICULATE_PYTHON=/scratch4/adeshpa6/TRACER/.venv_rcc/bin/python
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../scripts" && pwd)"
cd "$SCRIPT_DIR"
echo "host=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?} args=$*  $(date)"
/usr/bin/time -v /scratch4/adeshpa6/TRACER/.venv_rcc/bin/python -u s02_original_rctd.py "$@"
echo "done $(date)"

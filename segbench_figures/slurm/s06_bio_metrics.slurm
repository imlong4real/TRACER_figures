#!/bin/bash
#SBATCH --account=adeshpa6_tracer
#SBATCH --partition=shared
#SBATCH --job-name=sb_bio
#SBATCH --output=/scratch4/adeshpa6/tracer_campaign/final/logs/segbench_v2/%x_%j.log
# usage: sbatch s06_bio_metrics.slurm <setting> [<setting> ...]
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../scripts" && pwd)"
cd "$SCRIPT_DIR"
for s in "$@"; do
  echo "=== $s $(date)"
  /scratch4/adeshpa6/TRACER/.venv_rcc/bin/python -u s06_bio_metrics.py "$s"
done

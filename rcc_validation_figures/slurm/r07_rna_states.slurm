#!/bin/bash
#SBATCH --job-name=r07_rna
#SBATCH --output=/scratch4/adeshpa6/tracer_campaign/final/logs/r07_rna_states_%j.log
#SBATCH --partition=shared
#SBATCH --account=adeshpa6_tracer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4:00:00
set -euo pipefail
export MPLCONFIGDIR=/scratch4/adeshpa6/xenium_rcc/tracer/.mplconfig
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMBA_NUM_THREADS=8
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../scripts" && pwd)"
cd "$SCRIPT_DIR"
/usr/bin/time -v /scratch4/adeshpa6/TRACER/.venv_rcc/bin/python -u r07_rna_states.py

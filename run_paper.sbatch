#!/bin/bash
#SBATCH --job-name=bbrl_acc2027
#SBATCH --partition=short-40core
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --output=bbrl-%j.out
#SBATCH --error=bbrl-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
# Activate the desired environment before sbatch, or pass PYTHON_BIN explicitly.
PYTHON_BIN="${PYTHON_BIN:-python}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
"$PYTHON_BIN" -m pytest -q
"$PYTHON_BIN" benchmark_bbrl.py --strict-environment \
  --out-dir "outputs/paper_${SLURM_JOB_ID}" \
  --n-values "${N_VALUES:-8,16,32,64,128}" \
  --points "${POINTS:-100}" --seeds "${SEEDS:-0,1,2,3,4}"

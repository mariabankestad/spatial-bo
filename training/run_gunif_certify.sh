#!/bin/bash
# Certification of the g_uniform surrogate at 16 held-out (N, ell) nodes with
# fresh reference designs (gunif_certify.py): one Slurm array task per
# dimension, D = 2 + I. Writes cert_gu_D<D>.json.
#
#   sbatch --array=0-4 run_gunif_certify.sh
#SBATCH --job-name=gunif_cert
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/gunif_cert_%A_%a.out
#SBATCH --error=logs/gunif_cert_%A_%a.err
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
I="${SLURM_ARRAY_TASK_ID:-0}"; D=$(( 2 + I ))

# module load ...   # your Python environment with torch, numpy, scipy
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
export BO_FIELDGU_DIR="$REPO/results/field_gu"
mkdir -p "$REPO/logs"
cd "$REPO/training"
python -u gunif_certify.py --D "$D" --R 3000 --k 4 --workers 8

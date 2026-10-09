#!/bin/bash
# Continuous g_uniform surrogate over (N, ell) from the calibrated node
# fields of run_gunif_nodes.sh (gunif_fit.py): one Slurm array task per
# dimension, D = 2 + I. Writes results/field_gu/field_gu_D<D>.pt, which
# pack_field.py turns into the release file.
#
#   sbatch --array=0-4 run_gunif_fit.sh
#SBATCH --job-name=gunif_fit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/gunif_fit_%A_%a.out
#SBATCH --error=logs/gunif_fit_%A_%a.err
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
I="${SLURM_ARRAY_TASK_ID:-0}"; D=$(( 2 + I ))

# module load ...   # your Python environment with torch, numpy, scipy
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
export GUNIF_OUT="$REPO/results/gunif_nodes" BO_FIELDGU_DIR="$REPO/results/field_gu"
mkdir -p "$REPO/logs" "$BO_FIELDGU_DIR"
cd "$REPO/training"
python -u gunif_fit.py --D "$D" --min_nodes 20

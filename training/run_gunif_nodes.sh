#!/bin/bash
# Calibration of g_uniform at the nodes of a 5 x 5 log-spaced grid of
# (N, ell) over the trained range, N in [2D, 150], ell in [0.05, 3] ([0.02, 3]
# in D = 2), for D = 2..6: one Slurm array task per node, 125 tasks. Task I:
# D = (2, 3, 4, 5, 6)[I div 25], node I mod 25 with N index (I mod 25) div 5
# and ell index I mod 5. The calibrated node fields are the targets of
# run_gunif_fit.sh.
#
#   sbatch --array=0-124 run_gunif_nodes.sh
#SBATCH --job-name=gunif_nodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/gunif_nodes_%A_%a.out
#SBATCH --error=logs/gunif_nodes_%A_%a.err
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
I="${SLURM_ARRAY_TASK_ID:-0}"
DS=(2 3 4 5 6); D="${DS[$(( I / 25 ))]}"; NODE=$(( I % 25 )); IN=$(( NODE / 5 )); IE=$(( NODE % 5 ))
read -r N ELL < <(python3 -c "
D=$D; i_n=$IN; i_e=$IE
n_lo, n_hi = 2*D, 150; e_lo, e_hi = (0.02 if D == 2 else 0.05), 3.0
print(int(round(n_lo*(n_hi/n_lo)**(i_n/4))), e_lo*(e_hi/e_lo)**(i_e/4))")

# module load ...   # your Python environment with torch, numpy, scipy
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
export GUNIF_OUT="$REPO/results/gunif_nodes"
mkdir -p "$REPO/logs" "$GUNIF_OUT"
cd "$REPO/training"
python -u gunif_feasibility.py --D "$D" --N "$N" --ell "$ELL" --init info \
    --R 2000 --R_ho 1000 --R_cert 3000 --iters 40 --eta 0.1 --workers 8

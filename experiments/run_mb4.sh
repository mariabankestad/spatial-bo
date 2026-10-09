#!/bin/bash
# Controlled synthetic experiment (Section 4.1): one Slurm array task per
# (exploration scale, dimension, placement of the global minimum, seed).
# The unweighted rule and the three weighted rules at lambda = 1, 2, 4, 8,
# initial design 2D+2, budget 25/40/55 evaluations for D = 2, 3, 4,
# noise-free observations, g normalized by its domain mean. Standard GP-LCB
# is run at the reference scale only.
#
#   sbatch --array=0-824 run_mb4.sh
#
# Task I: scale block I div 275 (0.5, 1 and 1.5 times the reference
# lengthscale), then (dimension, placement) pair (I mod 275) div 25 and seed
# I mod 25. Results go to results/bo_mb4_norm/mb4_<D>d_<placement>_s<seed>_mb4_l<lambda><suffix>.npz;
# mb4_early.py and mb4_overall.py score them. Adapt the SBATCH lines and the
# environment line to your cluster.
#SBATCH --job-name=bo_mb4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/bo_mb4_%A_%a.out
#SBATCH --error=logs/bo_mb4_%A_%a.err
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
I="${SLURM_ARRAY_TASK_ID:-0}"
SCALES=("mult:0.5" "paper" "mult:1.5"); SUFF=("_s05" "" "_s15")
SC=$(( I / 275 )); I=$(( I % 275 ))
RULE="${SCALES[$SC]}"; SF="${SUFF[$SC]}"
PAIRS=(2:corner 2:face 2:interior
       3:corner 3:edge 3:face 3:interior
       4:corner 4:edge 4:face 4:interior)
PAIR="${PAIRS[$(( I / 25 ))]}"; SEED=$(( I % 25 ))
D="${PAIR%%:*}"; V="${PAIR##*:}"
FUNC="mb4_${D}d_${V}"
if [[ "$SC" == 1 ]]; then ARMS="fit_fit,fit_fix,fit_gfix,fit_gefffix,fit_gufix"; else ARMS="fit_fix,fit_gfix,fit_gefffix,fit_gufix"; fi

# module load ...   # your Python environment with torch, numpy, scipy
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
export BO_RESULTS_DIR="$REPO/results/bo_mb4_norm"
mkdir -p "$BO_RESULTS_DIR" "$REPO/logs"
cd "$REPO/experiments"
for LAM in 1 2 4 8; do
  python -u run_bo.py --func "$FUNC" --seed "$SEED" --noise_y 0 --noise_model 1e-4 \
      --g_ell ref --g_norm mean --ell_rule "$RULE" --lam "$LAM" --methods "$ARMS" --tag "mb4_l${LAM}${SF}"
done

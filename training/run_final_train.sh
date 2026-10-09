#!/bin/bash
# Training of the released g_geo and g_eff surrogates: one Slurm array task
# per (field, dimension, seed), 20 tasks. Task I: field I div 10 (0 = g_geo,
# 1 = g_eff), D = 2 + (I mod 10) div 2, seed I mod 2. The two seeds of a
# field and dimension are packed into one release file by pack_field.py.
#
#   sbatch --array=0-19 run_final_train.sh
#
# g_geo: 600k steps, K = 8 designs per row; g_eff: 300k steps, K = 32 (G
# head) and 64 (V head) designs per row. Both use the final schedule and
# sampler and the trained range N in [2D, 150], ell in [0.05, 3] ([0.02, 3]
# for g_geo in D = 2). The trainers checkpoint every 2% of the steps and
# resume with --resume, so a task interrupted by the wall-time limit can be
# resubmitted. The gold validation nodes of gen_field_v2.py (g_geo) and
# gen_field_geff.py (g_eff) must exist in the output directory.
#SBATCH --job-name=field_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=23:00:00
#SBATCH --output=logs/field_train_%A_%a.out
#SBATCH --error=logs/field_train_%A_%a.err
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
I="${SLURM_ARRAY_TASK_ID:-0}"
FIELDS=(v2 geff); FIELD="${FIELDS[$(( I / 10 ))]}"
D=$(( 2 + (I % 10) / 2 )); SEED=$(( I % 2 ))

# module load ...   # your Python environment with torch, numpy, scipy
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
export FIELD_N_LO_MULT=2 FIELD_N_HI=150
mkdir -p "$REPO/logs"
cd "$REPO/training"
if [[ "$FIELD" == v2 ]]; then
  [[ "$D" == 2 ]] && export F2_ELL_LO=0.02
  python -u train_field_online.py --D "$D" --seed "$SEED" --steps 600000 --K 8 \
      --schedule final --loss rowrel --sampler final --resume
else
  python -u train_field_geff_online.py --D "$D" --seed "$SEED" --steps 300000 \
      --K 32 --KV 64 --schedule final --sampler final --resume
fi

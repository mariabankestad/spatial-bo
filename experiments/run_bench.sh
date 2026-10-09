#!/bin/bash -l
# Benchmark experiment (Section 4.2): one Slurm array task per (function, seed).
# All rules of run_bo.py on one function and one seed, lambda = 2, initial
# design 2D+2, budget 25/40/55/70/85 evaluations for D = 2..6, noise-free
# observations, exploration lengthscale 0.268 sqrt(D/2).
#
#   sbatch --array=0-1399 run_bench.sh                                   # 14 functions x 100 seeds, pool-mean normalization
#   G_NORM=design ARMS=fit_gfix,fit_gefffix,fit_gufix sbatch --array=0-1399 run_bench.sh   # design-mean normalization (main results)
#   LAM=4 sbatch --array=0-1399 run_bench.sh                             # lambda = 4 (appendix)
#
# Task J runs function J mod 14 with seed J div 14. Results go to
# results/bo_bench/<function>_s<seed>_<tag>.npz; run_bench_score.sh scores
# them. Adapt the SBATCH lines and the environment line to your cluster.
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH -J bo_bench
#SBATCH -o logs/bench_%A_%a.out
#SBATCH -e logs/bench_%A_%a.err
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
J="${SLURM_ARRAY_TASK_ID:-0}"
LAM="${LAM:-2}"
G_NORM="${G_NORM:-mean}"
ARMS="${ARMS:-fit_fit,fit_fix,fit_gfix,fit_gefffix,fit_gufix,random}"
TASKS=(beale levy2d levy_3d hartmann_3d styblinski_tang_3d hartmann_4d levy_4d michalewicz_5d
       hartmann_6d branin camel6 rastrigin_3d rosenbrock_4d zakharov_5d)
NF=${#TASKS[@]}
FUNC="${TASKS[$(( J % NF ))]}"; SEED=$(( J / NF ))
TAG="bench_l${LAM}"; [[ "$G_NORM" == "mean" ]] || TAG="${TAG}_${G_NORM}"

# module load ...   # your Python environment with torch, numpy, scipy
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
export BO_RESULTS_DIR="$REPO/results/bo_bench"
mkdir -p "$BO_RESULTS_DIR" "$REPO/logs"
cd "$REPO/experiments"
python -u run_bo.py --func "$FUNC" --seed "$SEED" --noise_y 0 --noise_model 1e-4 \
    --g_ell ref --g_norm "$G_NORM" --ell_rule paper --lam "$LAM" --methods "$ARMS" --tag "$TAG"

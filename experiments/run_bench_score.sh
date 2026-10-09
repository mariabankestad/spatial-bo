#!/bin/bash -l
# Score the benchmark runs of run_bench.sh (eval_bench.py): paired regret
# differences against the g = 1 rule with bootstrap intervals, written to
# results/bench_score_<tag>.{json,md}. Chain it to the array with
# --dependency=afterany:<job id>, or run it directly once the runs exist.
#
#   sbatch run_bench_score.sh                   # tag bench_l2
#   G_NORM=design sbatch run_bench_score.sh     # tag bench_l2_design, baseline from bench_l2 (same seeds)
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:20:00
#SBATCH -J bo_bench_score
#SBATCH -o logs/bench_score_%j.out
#SBATCH -e logs/bench_score_%j.err
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAM="${LAM:-2}"; G_NORM="${G_NORM:-mean}"; TAG="bench_l${LAM}"; [[ "$G_NORM" == "mean" ]] || TAG="${TAG}_${G_NORM}"

# module load ...   # your Python environment with torch, numpy, scipy
export OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
export BO_RESULTS_DIR="$REPO/results/bo_bench"
cp -n "$REPO/results/hstar.json" "$BO_RESULTS_DIR/hstar.json" 2>/dev/null || true
cd "$REPO/experiments"
python -u eval_bench.py --tag "$TAG" --g_norm "$G_NORM" --baseline_tag "bench_l${LAM}" \
    --tasks "$REPO/results/benchmark_tasks.json" --out "$REPO/results/bench_score_${TAG}"

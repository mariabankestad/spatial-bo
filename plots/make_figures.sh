#!/bin/bash
# Recreate the result figures and tables of the paper in plots/out/.
# The scripts read the scored summaries in ../results and the fields in
# ../fields; the per-seed runs (results/bo_mb4_norm, results/bo_mb4_ref,
# results/bo_bench), distributed separately, are needed for the synthetic
# section and the trajectory figures.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p out

# Controlled synthetic experiment (Section 4.1, Appendix C)
python plot_mb4_combined.py                       # fig_mb4_combined.pdf
python make_firsthit_table.py                     # tab_mb4_firsthit.tex, tab_mb4_hitfrac.tex

# Benchmark functions (Section 4.2, Appendix D)
python plot_bench.py --mode init                  # fig_bench_init.pdf
python plot_bench.py --mode init --score ../results/bench_score_bench_l2.json --suffix _poolmean   # fig_bench_init_poolmean.pdf
python plot_bench_traj.py                         # fig_bench_traj.pdf
python plot_bench_traj.py --all                   # fig_bench_traj_all.pdf
python make_bench_table.py                        # tab_bench_final.tex

echo "figures and tables written to plots/out/"

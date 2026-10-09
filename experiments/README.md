# Experiments

`run_bo.py` runs the exploration rules on one function and one seed. Rule
names in the code and in the result files:

| name | rule |
|---|---|
| `fit_fit` | standard GP-LCB |
| `fit_fix` | g = 1 |
| `fit_gfix` | g_geo |
| `fit_gefffix` | g_eff |
| `fit_gufix` | g_uniform |
| `random` | uniform random candidates |

**Controlled synthetic experiment** (Section 4.1). `run_mb4.sh` runs
`run_bo.py` over the placements of the global minimum, seeds and exploration
weights; `mb4_early.py` and `mb4_overall.py` score the runs.

**Benchmark functions** (Section 4.2). `run_bench.sh` and
`run_bench_score.sh`; the settings are listed in `benchmark_protocol.md`.
The two robot-pushing tasks need the simulator of Wang et al. (2017), which
is not included.

**Weight-field training** (Appendix A). `../training/run_final_train.sh`
trains `geo` and `eff`, `run_gate_field.sh` certifies them;
`run_gunif_nodes.sh`, `run_gunif_fit.sh` and `run_gunif_certify.sh`
calibrate and certify `uniform`. `pack_field.py` packs the trained networks
into the files in `../fields/`.

**Figures and tables.** `../plots/make_figures.sh` writes them to
`plots/out/` from the scored summaries in `../results/`. The raw per-seed
trajectories (about 170 MB) are distributed separately as a release asset.

The launch scripts are Slurm array jobs; adapt the `#SBATCH` lines and the
environment line to your cluster.

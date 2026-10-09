# Benchmark experiment: protocol

The settings of the benchmark experiment (Section 4.2 and Appendix D of the
paper), as run by `run_bench.sh` and scored by `run_bench_score.sh`.

## Tasks

- Eight functions fixed before any run: Beale (2), Levy (2, 3, 4), Hartmann
  (3, 4), Styblinski–Tang (3), Michalewicz (5); Hartmann (6) as a
  higher-dimensional test; and five functions used during development:
  Branin, six-hump camel, Rastrigin (3), Rosenbrock (4), Zakharov (5).
- Every function is evaluated on the unit cube, mapped to its classical
  domain internally (`spatialbo/functions.py`), and standardized over a fixed
  set of 8192 Sobol points.
- `results/benchmark_tasks.{json,md}` (`bench_tasks.py`) records, per task,
  the location of the optimum, its distance to the nearest wall, that
  distance relative to the spacing of the budget, and the resulting class
  (interior, face, edge, corner).

## Rules

- Standard GP-LCB (sigma at the fitted lengthscale); the unweighted rule
  g = 1; g_geo, g_uniform and g_eff, all with sigma at the reference
  lengthscale 0.268 sqrt(D/2); uniform random as a reference.
- g is divided by its average at the current state. The main results use
  the average over the evaluated points (`--g_norm design`, tag
  `bench_l2_design`); the average over a fixed 1024-point Sobol set of the
  domain (`--g_norm mean`, tag `bench_l2`) is reported in the appendix. The
  g = 1, standard GP-LCB and random runs are shared between the two, as they
  do not depend on the normalization.

## Common settings

- Initial design of 2D + 2 scrambled-Sobol points per seed, shared by all
  rules; budget 25, 40, 55, 70, 85 evaluations for D = 2..6.
- lambda = 2 for every task and rule, fixed in advance; lambda = 4 as a
  robustness check.
- Noise-free observations; GP noise variance 1e-4.
- GP: Matérn-5/2, unit output scale, zero mean, one lengthscale fitted by
  MAP under the prior of Hvarfner et al. (2024), bounds [0.02, 3], six
  restarts.
- Acquisition optimizer: 512 shared Sobol candidates, the 32 best plus 10
  Boltzmann-sampled starts, one stacked L-BFGS-B with gradients through g.
- 100 paired seeds per task (a 25-seed pilot, then 75 more with the same
  settings).

## Readouts

- Simple regret r_t = h* − best observed value through evaluation t, with
  h* the numerically polished optimum of the standardized objective
  (`results/hstar.json`).
- Paired differences against g = 1 after 2(D + 2) evaluations and at the end
  of the budget, with 90% paired bootstrap intervals (20 000 resamples);
  D + 2 and the midpoint of the budget as additional checkpoints.
- Tasks are ordered by the distance of the optimum to the boundary.

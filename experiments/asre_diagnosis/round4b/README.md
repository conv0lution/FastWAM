# ASRE Stage 2 Round 4B

This directory implements the frozen low-rank, scene-specific feature-subspace
gate. It uses uncentered per-layer/per-KV SVD on a task-stratified 80-episode
fit partition, strict 20-episode holdout diagnostics, nested matched-rank random
orthogonal controls, paired online evaluation, and held-out-only offline replay.

Run from the repository root after committing the reviewed implementation:

```bash
ROUND4B_GPU_IDS="4 5 6 7" \
ROUND4B_OUTPUT_ROOT="$PWD/asre_results/round4b_subspace" \
ROUND4B_LAUNCH_STAGGER_SECONDS=70 \
./experiments/asre_diagnosis/round4b/run_full_experiment.sh
```

The driver is resumable and never launches a later experiment.

## Cumulative-energy follow-up

The analysis-only follow-up reconstructs complete rank-0..3072 calibration and
held-out cumulative ΔZ energy curves. It performs one authorized cache-only
pass over the frozen 100 held-out states to persist missing Grams, recovers the
missing eigensystem only from the original calibration Grams, and does not run
online episodes, environment rollouts, or held-out SVD fitting.

Run it only after committing the reviewed follow-up implementation:

```bash
ROUND4B_ENERGY_GPU_IDS="4 5 6 7" \
ROUND4B_ENERGY_LAUNCH_STAGGER_SECONDS=70 \
./experiments/asre_diagnosis/round4b/run_energy_curve_analysis.sh
```

All new artifacts are isolated under
`asre_results/round4b_subspace/energy_curve_analysis/`; existing Round 4B
aggregate and calibration artifacts are read-only inputs.

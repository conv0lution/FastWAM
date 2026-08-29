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

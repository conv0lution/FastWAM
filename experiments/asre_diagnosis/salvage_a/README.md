# ASRE Salvage A — Action-Sensitive Feature Subspace

This is the registered final compact-ASRE salvage test. It freezes a new
task-stratified 50/50 split, creates split-local donor derangements, fits
matched-calibration SVD and full-action-path ActionAware bases at ranks 36 and
97, reuses the preregistered nested random controls, and evaluates exactly eight
conditions in two four-GPU waves.

The sole primary decision scope is the frozen held-out 50 episodes. Calibration
50 and all-100 results are descriptive. The driver stops after aggregation and
plotting regardless of whether the result is STRONG, MODERATE, or WEAK.

Run only from a committed clean worktree:

```bash
SALVAGE_A_GPU_IDS="4 5 6 7" \
SALVAGE_A_OUTPUT_ROOT="$PWD/asre_results/salvage_a_action_sensitive" \
SALVAGE_A_LAUNCH_STAGGER_SECONDS=70 \
./experiments/asre_diagnosis/salvage_a/run_full_experiment.sh
```

The pipeline is fail-closed and resume-safe. A partial run may be restarted
with the identical command and source commit. If the source commit changes, use
a new child directory under `asre_results/salvage_a_action_sensitive/`.

Important artifacts:

- `split/calibration_split_manifest.json`
- `donors/donor_mapping.json`
- `calibration/state_selection_manifest.json`
- `calibration/differentiable_path_report.json`
- `calibration/basis_manifest.json`
- `calibration/subspace_diagnostics.json`
- `machinery_report.json`
- `aggregate/salvage_a_summary.md`
- `aggregate/result_summary_for_gpt.md`

No additional rank, cross-suite, RoboTwin, alternative gradient objective,
world-prediction experiment, semantic probe, or later stage is launched.

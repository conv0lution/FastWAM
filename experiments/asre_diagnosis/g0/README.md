# ASRE G0 — Cross-Suite Generalization Gate

G0 tests whether the frozen LIBERO-Spatial Stage-1 retrieval dependencies
generalize to the repository's canonical `libero_object`, `libero_goal`, and
`libero_10` suites under the same `libero_uncond_2cam224.pt` checkpoint.
It is not Stage 2 Round 4A and contains no token, head, or subspace sparsity.

## Frozen protocol

The suites run sequentially in this exact order. Within a suite, four
independent non-DDP workers run concurrently:

| Condition slot | Condition | Current video retrieval | Replacement |
|---:|---|---|---|
| 0 | `full_current` | layers 0–29 | none |
| 1 | `late_current_15_29` | layers 15–29 | none |
| 2 | `early_current_00_19` | layers 0–19 | none |
| 3 | `late_wrong_scene_15_29` | layers 15–29 | same-task next-trial wrong scene on layers 15–29 |

Each suite uses tasks 0–9, trials 0–9, seed 42, action horizon 32, 10
inference steps, a 10-step replan interval, compiled action inference, and the
existing gripper binarization. Rollout video saving is disabled. Suite task
language is encoded normally by the checkpoint's text encoder; the frozen
Spatial prompt cache is deliberately not reused across suites.

The wrong-scene donor is the first model-ready policy-query observation after
30 evaluator dummy/wait steps. The mapping is frozen before outcome inspection:

```text
donor_trial = (recipient_trial + 1) mod 10
```

## Formal run

First commit the implementation. Formal preflight refuses dirty source because
the result metadata must identify the exact code revision. The driver never
commits, pushes, or launches Stage 2.

```bash
/home/zhaizicheng/miniconda3/envs/fastwam/bin/python3.10 \
  -m experiments.asre_diagnosis.g0.run_g0 \
  --checkpoint /local_home/zhaizicheng/fastwam_assets/checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  --dataset-stats /local_home/zhaizicheng/fastwam_assets/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  --valid-manifest asre_results/round2/state_bank_valid_manifest.json \
  --state-bank-dir asre_results/state_bank \
  --output-root asre_results/g0_cross_suite \
  --python /home/zhaizicheng/miniconda3/envs/fastwam/bin/python3.10 \
  --gpu-ids 4 5 6 7
```

On the current host, physical GPUs 0–3 are occupied by another four-way job,
while 4–7 are the four dedicated cards. `--gpu-ids` is therefore mandatory.
The four values map to the condition slots in table order and the physical GPU
inventory is frozen in launcher provenance. Each selected card must have at
least 22000 MiB free at launch.

The same formal command is also available as:

```bash
./experiments/asre_diagnosis/g0/run_full_experiment.sh
```

To preserve an earlier failed attempt while starting from a newly committed
revision, choose a child output directory:

```bash
G0_OUTPUT_ROOT="$PWD/asre_results/g0_cross_suite/retry_YYYYMMDD" \
  ./experiments/asre_diagnosis/g0/run_full_experiment.sh
```

For development-only suite/config inspection on an uncommitted checkout:

```bash
/home/zhaizicheng/miniconda3/envs/fastwam/bin/python3.10 \
  -m experiments.asre_diagnosis.g0.run_g0 \
  --checkpoint /local_home/zhaizicheng/fastwam_assets/checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  --dataset-stats /local_home/zhaizicheng/fastwam_assets/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  --valid-manifest asre_results/round2/state_bank_valid_manifest.json \
  --state-bank-dir asre_results/state_bank \
  --output-root asre_results/g0_cross_suite \
  --gpu-ids 4 5 6 7 \
  --allow-dirty --preflight-only
```

`--allow-dirty` cannot proceed beyond preflight. A formal run performs the
full ASRE unit suite, fixed-input machinery controls A–D, donor capture, a
two-trial four-condition smoke test, and the full four-condition evaluation
for each suite before advancing to the next suite. A nonzero child exit or
invalid/missing result fails the suite and terminates its peers.

## Resume and artifacts

Verified-complete condition directories are skipped. Partial conditions are
resumed only when launcher identity, Git/artifact hashes, suite/trial protocol,
GPU mapping, condition schedule, and donor bundle all match. A recorded live
child blocks a second launcher.

Artifacts are rooted at `asre_results/g0_cross_suite/`:

```text
preflight_report.json
machinery_report.json
libero_object/{donors,smoke,full}/
libero_goal/{donors,smoke,full}/
libero_10/{donors,smoke,full}/
aggregate/
  condition_summary.csv
  primary_contrasts.csv
  paired_transitions.csv
  task_success.csv
  cross_suite_descriptive.csv
  g0_summary.json
  g0_summary.md
  result_summary_for_gpt.md
  plots/
logs/
```

The three new suite analyses are primary. LIBERO-Spatial is imported from the
frozen Round-2/Round-3B online results and labeled `frozen prior result`.
The optional pooled hierarchy (`suite -> task -> paired trial`) is descriptive
and does not replace suite-level conclusions.

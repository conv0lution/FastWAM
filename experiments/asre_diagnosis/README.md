# ASRE (Action-Sufficient Representation): video K/V layer diagnosis

This directory implements the training-free diagnosis described in the experiment plan. It does not use DDP and does not change training, weights, video prefill, text cross-attention, proprioception, schedulers, or inference hyperparameters.

All commands below assume the repository root as the working directory and the Fast-WAM environment (including the external LIBERO package) is active.

To execute the complete guarded pipeline with the checkpoint paths configured
for this host, run:

```bash
./experiments/asre_diagnosis/run_full_experiment.sh
```

The script resumes the existing state bank, writes stage logs under
`asre_results/logs/`, places state-bank Wan inference on GPU 0 and T5 plus
EGL rendering on GPU 1, and only starts the full online matrix after both smoke
stages pass. Paths and the two state-bank GPUs can be overridden with
`CHECKPOINT`, `DATASET_STATS`, `ASRE_OUTPUT_ROOT`, `STATE_MODEL_GPU`, and
`STATE_AUX_GPU` environment variables.

## 1. Local unit tests

```bash
PYTHONPATH=src:. python -m unittest discover \
  -s experiments/asre_diagnosis/tests -v
```

## 2. Collect the fixed-state bank

This runs baseline rollouts for all 10 `libero_spatial` tasks, 10 episodes per task, and stores the first five policy-query states from every episode.

```bash
CUDA_VISIBLE_DEVICES=0,1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
MUJOCO_EGL_DEVICE_ID=1 python experiments/asre_diagnosis/collect_state_bank.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=/path/to/checkpoint \
  EVALUATION.device=cuda:0 \
  EVALUATION.text_encoder_device=cuda:1 \
  EVALUATION.task_suite_name=libero_spatial \
  EVALUATION.task_ids='[0,1,2,3,4,5,6,7,8,9]' \
  EVALUATION.num_trials=10 \
  EVALUATION.dataset_stats_path=/path/to/dataset_stats.json \
  ASRE_DIAGNOSIS.state_bank_dir=/path/to/asre/state_bank
```

## 3. Mandatory checkpoint-level correctness smoke test

Test A compares the original public `infer_action` call against an enabled diagnosis with an empty disabled-layer tuple. Test B drops one middle layer. Test C drops all layers.

```bash
python experiments/asre_diagnosis/smoke_test.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=/path/to/checkpoint \
  model.load_text_encoder=false \
  EVALUATION.dataset_stats_path=/path/to/dataset_stats.json \
  ASRE_DIAGNOSIS.state_bank_dir=/path/to/asre/state_bank
```

The machine-readable report is written to `state_bank/smoke_test_report.json` by default.

## 4. Fixed-state offline replay

To launch all eight conditions, safely resume completed condition outputs,
then aggregate and plot the results, run:

```bash
./experiments/asre_diagnosis/run_offline_analysis.sh
```

The launcher explicitly clears inherited MuJoCo/EGL variables because fixed-
state replay does not render. It skips a condition only when its `summary.csv`
contains all state-bank samples and refuses to overwrite partial JSONL output.

Run one process per condition, setting `condition_index` from 0 through 7. The index is resolved against `model.mot.num_layers`; for a 30-layer model the names are baseline, the six five-layer groups, and drop-all.

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/asre_diagnosis/replay_state_bank.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=/path/to/checkpoint \
  EVALUATION.num_trials=10 \
  EVALUATION.dataset_stats_path=/path/to/dataset_stats.json \
  ASRE_DIAGNOSIS.enabled=true \
  ASRE_DIAGNOSIS.condition_index=0 \
  ASRE_DIAGNOSIS.state_bank_dir=/path/to/asre/state_bank \
  ASRE_DIAGNOSIS.offline_output_dir=/path/to/asre/offline
```

Repeat concurrently on GPUs 1–7 with the matching condition index. Each process writes `per_sample.jsonl`, `summary.csv`, and `run_metadata.json` under its own condition directory.

## 5. Online two-condition smoke

This runs task 0 for two trials under baseline and drop-all. Each condition gets
one Wan GPU and one T5/EGL GPU.

```bash
python experiments/asre_diagnosis/launch_eight_gpu.py \
  --mode smoke \
  --checkpoint /path/to/checkpoint \
  --dataset-stats-path /path/to/dataset_stats.json \
  --output-root /path/to/asre/online_smoke
```

## 6. Full eight-GPU online matrix

The launcher refuses a full run unless both checkpoint-level Test A/B/C and the online smoke launch succeeded.

```bash
python experiments/asre_diagnosis/launch_eight_gpu.py \
  --mode full \
  --checkpoint /path/to/checkpoint \
  --dataset-stats-path /path/to/dataset_stats.json \
  --output-root /path/to/asre/online_full \
  --smoke-summary /path/to/asre/online_smoke/launcher_summary.json \
  --correctness-report /path/to/asre/state_bank/smoke_test_report.json
```

The full launcher uses four independent two-GPU slots `(0,1)`, `(2,3)`,
`(4,5)`, and `(6,7)`. Wan runs on the first GPU and T5 plus EGL on the second.
Conditions 0–3 run in the first wave and 4–7 in the second, so every process has
the same memory-safe placement. It retains the existing `compile_action_infer`,
scheduler, action horizon, inference-step, replan, preprocessing, and gripper
settings, and disables rollout video encoding by default. Re-running the same
command skips result files for completed tasks and resumes missing tasks.

The launcher derives the layer count from the selected task/model config. For a custom checkpoint whose architecture intentionally differs from that config, pass `--num-layers`; the loaded model remains the source of truth for intervention validation and condition resolution.

## 7. Aggregate and plot

```bash
python experiments/asre_diagnosis/aggregate_results.py \
  --online-root /path/to/asre/online_full \
  --offline-root /path/to/asre/offline \
  --output-dir /path/to/asre/aggregate

python experiments/asre_diagnosis/plot_results.py \
  --aggregate-dir /path/to/asre/aggregate
```

Aggregation writes the requested summary table, paired transition counts, per-task deltas, and paired-bootstrap confidence intervals. Plotting writes the four requested PNG figures. `matplotlib` is only needed in the analysis environment; it is not added as a model training/evaluation dependency.

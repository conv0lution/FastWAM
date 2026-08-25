# ASRE Round 2：充分性与检索结构实验

本目录实现固定的八条件 Round-2 协议。它复用 Round 1 已验证的 video K/V
干预，仅把人类可读的 `enabled_video_retrieval_layers` 编译为其补集
`disabled_video_layers`。没有修改 `mot.py`、模型权重、video prefill、扩散步数、
文本/本体感觉条件或调度器。

所有命令都从 FastWAM 仓库根目录执行。正式运行前应先提交本轮代码，否则
`run_metadata.json` 中记录的是上一次提交的 Git hash，不能完整标识当前实现。

## 最短完整命令

deep57 上的 checkpoint 和 dataset-stats 默认路径已经写入脚本：

```bash
cd /home/zhaizicheng/Reasoning/FastWAM
./experiments/asre_diagnosis/round2/run_full_experiment.sh
```

执行顺序为：固定状态库 QC、八条件离线回放、C0/C4/C7 在线 smoke、八条件
在线 full、聚合、绘图。脚本不会写入或覆盖 Round-1 目录；Round-2 的唯一默认
根目录是 `asre_results/round2/`。

如需覆盖路径：

```bash
CHECKPOINT=/path/to/checkpoint.pt \
DATASET_STATS=/path/to/dataset_stats.json \
ASRE_ROUND1_OUTPUT_ROOT=/path/to/existing/asre_results \
ASRE_ROUND2_OUTPUT_ROOT=/path/to/asre_results/round2 \
ASRE_ROUND1_SUMMARY=/path/to/asre_results/aggregate/summary.csv \
./experiments/asre_diagnosis/round2/run_full_experiment.sh
```

## 本地 smoke / 单元测试

不要使用这台机器指向 Python 2 的系统 `python`。直接使用实验环境解释器：

```bash
PY=/home/zhaizicheng/miniconda3/envs/fastwam/bin/python
PYTHONPATH="$PWD/src:$PWD:$PWD/../LIBERO${PYTHONPATH:+:$PYTHONPATH}" \
  "$PY" -m unittest discover -s experiments/asre_diagnosis/tests -v
```

## 固定 QC 与离线回放

下列命令先用固定规则（raw max-abs `<=1e-6`、形状/finite 检查、gripper
精确一致）验证原始 500 个状态，然后把同一个有序有效 ID 集合用于全部八个条件：

```bash
./experiments/asre_diagnosis/round2/run_offline.sh
```

它生成不可变的 `state_bank_valid_manifest.json` 和 `prompt_context_cache.pt`。
后者复用状态库中每项任务的原始文本 context，使在线进程不再加载 T5，因此可以
做到一个条件占一张 A5000。

## 在线三条件 smoke

先完成上面的 QC，然后执行：

```bash
PY=/home/zhaizicheng/miniconda3/envs/fastwam/bin/python
CKPT=/local_home/zhaizicheng/fastwam_assets/checkpoints/fastwam_release/libero_uncond_2cam224.pt
STATS=/local_home/zhaizicheng/fastwam_assets/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json

"$PY" experiments/asre_diagnosis/round2/launch_eight_gpu.py \
  --mode smoke \
  --python "$PY" \
  --checkpoint "$CKPT" \
  --dataset-stats-path "$STATS" \
  --valid-manifest "$PWD/asre_results/round2/state_bank_valid_manifest.json" \
  --output-root "$PWD/asre_results/round2/online_smoke"
```

Smoke 固定运行 C0、C4、C7，各用物理 GPU 0、4、7，在 task 0 上做 2 个
paired trials。任何一个条件失败时都不会开放 full gate。

## 完整八卡在线矩阵

```bash
"$PY" experiments/asre_diagnosis/round2/launch_eight_gpu.py \
  --mode full \
  --python "$PY" \
  --checkpoint "$CKPT" \
  --dataset-stats-path "$STATS" \
  --valid-manifest "$PWD/asre_results/round2/state_bank_valid_manifest.json" \
  --output-root "$PWD/asre_results/round2/online_full" \
  --smoke-summary "$PWD/asre_results/round2/online_smoke/launcher_summary.json"
```

映射固定为 C0–C7 对应物理 GPU 0–7，不使用 DDP。每个进程内部只看见自己的
GPU，因此模型设备始终是 `cuda:0`；MuJoCo EGL 仍使用物理 GPU 序号。每个条件有
独立目录、attempt log 和状态文件。完整任务会零写跳过，部分任务只能在 provenance
和配置完全相同的情况下恢复。同一 output root 由非阻塞文件锁保护；评估子进程会
继承该锁，因此即使父 launcher 被 `SIGKILL`，也不会在孤儿进程尚存时重复占卡启动。

## 聚合与绘图

```bash
"$PY" experiments/asre_diagnosis/round2/aggregate_results.py \
  --online-root "$PWD/asre_results/round2/online_full" \
  --offline-root "$PWD/asre_results/round2/offline" \
  --round1-summary "$PWD/asre_results/aggregate/summary.csv" \
  --output-dir "$PWD/asre_results/round2/aggregate"

MPLCONFIGDIR="$PWD/asre_results/round2/.matplotlib" \
"$PY" experiments/asre_diagnosis/round2/plot_results.py \
  --aggregate-dir "$PWD/asre_results/round2/aggregate"
```

聚合器从在线 metadata 指向的不可变 QC manifest 自动读取实际有效样本数和顺序；
不会假设它一定等于 499。它严格要求 8×100 paired episodes，并输出 paired 与
task-hierarchical bootstrap 95% CI、exact McNemar、七项 Holm 校正、预设比较、
task heatmap 数据、动作维度分解、replan-stage 统计及机器诊断。
同时会生成 `round1_round2_joint_summary.csv`，并报告 Round-2 baseline repeat
相对 Round-1 baseline 的成功率差；这只是 aggregate-rate 复现检查，不冒充
episode-level 一致性证明。

`executed_prefix_norm_rms` 是动作 0–9 的六维连续物理动作差除以 dataset global
std 后的 RMS。`full_chunk_norm_rms_0_31` 使用相同物理/std 定义但覆盖动作 0–31；
`round1_raw_output_full_chunk_rms` 则单独保留 Round-1 原始 min/max 归一化输出空间
指标，二者不能混为一项。

## 预期输出树

```text
asre_results/
├── state_bank/                         # 原 Round-1 输入，只读
└── round2/
    ├── state_bank_valid_manifest.json
    ├── prompt_context_cache.pt
    ├── logs/
    ├── offline/
    │   └── <C0...C7>/
    │       ├── per_sample.jsonl
    │       ├── summary.csv
    │       └── run_metadata.json
    ├── online_smoke/
    ├── online_full/
    │   └── <C0...C7>/
    │       ├── run_metadata.json
    │       ├── launcher_status.json
    │       └── libero_spatial/gpu0_task*_results.json
    └── aggregate/
        ├── summary.csv
        ├── summary.json
        ├── paired_transitions.csv
        ├── task_success_rates.csv
        ├── task_success_delta.csv
        ├── offline_action_dimensions.csv
        ├── replan_stage_metrics.csv
        ├── planned_comparisons.csv
        ├── round1_round2_joint_summary.csv
        ├── diagnostic_summary.{json,md}
        ├── aggregate_metadata.json
        └── plots/                       # 五张主图 + 一张 exploratory replan 图
```

候选充分性标签只用于筛选：点估计损失不超过 5 个百分点，且没有灾难性 task
pattern。这里在查看 Round-2 结果前固定将“灾难性”操作化为：baseline 的该 task
成功率至少 80%，同时干预条件下降至少 50 个百分点；诊断中的“强退化”阈值固定为
整体下降至少 20 个百分点。这些标签不是正式 non-inferiority 结论。八个条件完成后
应停止增加新 layer schedule，先联合 Round 1 解释结果。

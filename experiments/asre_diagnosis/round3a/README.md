# ASRE Round 3A：晚半层 2^3 因子实验补全

Round 3A 只补齐晚半层因子设计缺失的三个 cell，不继续搜索任意 layer
schedule，也不以效率优化为目标。三个二元区域固定为：

- A：action layers 15–19；
- B：action layers 20–24；
- C：action layers 25–29。

所有 Round-3A cell 都关闭 layers 0–14 的直接 video K/V retrieval。新条件为：

| Cell | 条件 | enabled layers | physical GPU |
|---|---|---|---:|
| 000 | `keep_none_late` | none | 0 |
| 010 | `keep_20_24` | 20–24 | 1 |
| 110 | `keep_15_24` | 15–24 | 2 |

其余五个 cell 直接读取已经冻结的 Round-2 原始结果，不会重跑：
`keep_25_29` (001)、`keep_20_29` (011)、`keep_15_19` (100)、
`keep_15_19_25_29` (101) 和 `keep_15_29` (111)。

本目录复用 Round 1/2 已验证的 `disabled_video_layers` 干预、Round-2
action metrics 和同一份 QC-valid 499-state manifest。它不改模型层、video
prefill、权重、hidden states、文本/本体感觉条件、scheduler、action horizon 或
inference-step count。

## 运行前必须先提交代码

launcher 和 offline replay 都会要求 Git 代码工作树干净（已生成的
`asre_results/round3a/aggregate/` 结果除外）。请先提交 Round-3A 实现，让
`run_metadata.json` 中的 commit 真正标识运行代码：

```bash
cd /home/zhaizicheng/Reasoning/FastWAM
git status
# 检查并自行 git add / git commit；脚本不会自动提交、清理或 push。
```

Round-2 tag 和结果只读；不要移动、删除或重写
`asre_results/round2/`。原始 Round-2 结果不在 Git 中，必须保留在本机或另行备份。

## 一键完整执行

默认路径适用于 deep57：

```bash
./experiments/asre_diagnosis/round3a/run_full_experiment.sh
```

固定顺序为：

1. 只读预检 frozen Round-2 五个 cell 的 online/offline 完整性与配对；
2. 校验 frozen Round-2 manifest 与所有哈希，三条件 499-state offline replay；
3. 三条件 task 0 × 2 paired-trial smoke，并验证 action trace 数值有限；
4. smoke gate 通过后三条件运行全部 10 tasks × 10 trials；
5. 把 Round-2 五个 cell 与 Round-3A 三个 cell 严格配对聚合；
6. 生成五张因子分析主图并停止。

如需覆盖位置：

```bash
CHECKPOINT=/path/to/libero_uncond_2cam224.pt \
DATASET_STATS=/path/to/libero_uncond_2cam224_dataset_stats.json \
ASRE_ROUND1_OUTPUT_ROOT=/path/to/asre_results \
ASRE_ROUND2_OUTPUT_ROOT=/path/to/asre_results/round2 \
ASRE_ROUND3A_OUTPUT_ROOT=/path/to/asre_results/round3a \
./experiments/asre_diagnosis/round3a/run_full_experiment.sh
```

路径覆盖仍必须与 Round-2 QC-valid manifest 中记录的 checkpoint、dataset stats、
source manifest 和 prompt cache 的绝对路径及 SHA256 完全一致。

## 分阶段命令

离线 replay：

```bash
./experiments/asre_diagnosis/round3a/run_offline.sh
```

该脚本不会调用 Round-2 QC 生成器，也不会改写 valid manifest。它先校验 manifest
声明的 checkpoint、dataset stats、Round-1 state manifest 和 prompt cache 哈希，
然后将同一有序 499-state population 分配到 GPU 0/1/2。已有完整条件会在严格验证
metadata、样本 ID 顺序和指标 schema 后零写跳过；partial/unknown 目录会 fail closed。

在线 smoke：

```bash
PY=/home/zhaizicheng/miniconda3/envs/fastwam/bin/python
CKPT=/local_home/zhaizicheng/fastwam_assets/checkpoints/fastwam_release/libero_uncond_2cam224.pt
STATS=/local_home/zhaizicheng/fastwam_assets/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json

"$PY" experiments/asre_diagnosis/round3a/launch_three_gpu.py \
  --mode smoke \
  --python "$PY" \
  --checkpoint "$CKPT" \
  --dataset-stats-path "$STATS" \
  --valid-manifest "$PWD/asre_results/round2/state_bank_valid_manifest.json" \
  --round2-reference-metadata \
    "$PWD/asre_results/round2/online_full/keep_15_29/run_metadata.json" \
  --output-root "$PWD/asre_results/round3a/online_smoke"
```

完整在线运行：

```bash
"$PY" experiments/asre_diagnosis/round3a/launch_three_gpu.py \
  --mode full \
  --python "$PY" \
  --checkpoint "$CKPT" \
  --dataset-stats-path "$STATS" \
  --valid-manifest "$PWD/asre_results/round2/state_bank_valid_manifest.json" \
  --round2-reference-metadata \
    "$PWD/asre_results/round2/online_full/keep_15_29/run_metadata.json" \
  --output-root "$PWD/asre_results/round3a/online_full" \
  --smoke-summary "$PWD/asre_results/round3a/online_smoke/launcher_summary.json"
```

不使用 DDP。三个 condition 各占物理 GPU 0/1/2，进程内模型设备均为
`cuda:0`，MuJoCo EGL 使用对应物理 GPU ID。launcher 对 output root 持有非阻塞
文件锁；完整 condition 零写跳过，partial condition 只在 commit、配置、provenance
和已完成 task 文件全部匹配时恢复。

聚合与绘图：

```bash
"$PY" experiments/asre_diagnosis/round3a/aggregate_factorial.py \
  --round2-online-root "$PWD/asre_results/round2/online_full" \
  --round2-offline-root "$PWD/asre_results/round2/offline" \
  --round3a-online-root "$PWD/asre_results/round3a/online_full" \
  --round3a-offline-root "$PWD/asre_results/round3a/offline" \
  --valid-manifest "$PWD/asre_results/round2/state_bank_valid_manifest.json" \
  --output-dir "$PWD/asre_results/round3a/aggregate"

MPLCONFIGDIR="$PWD/asre_results/round3a/.matplotlib" \
"$PY" experiments/asre_diagnosis/round3a/plot_factorial.py \
  --aggregate-dir "$PWD/asre_results/round3a/aggregate"
```

聚合器必须联合 bootstrap 八维 paired outcome vector，而不是独立 bootstrap 每个
cell；输出 episode-paired 与 task-hierarchical 95% CI。七个 factorial contrasts
如报告 hypothesis-test p-value，则作为同一个 family 做 Holm correction。概率尺度
contrast 是实验交互摘要，不应被描述为神经计算的完整因果模型。

## 预期输出

```text
asre_results/
├── round2/                              # 全部只读
│   ├── state_bank_valid_manifest.json
│   ├── offline/
│   └── online_full/
└── round3a/
    ├── logs/
    ├── offline/<new condition>/
    │   ├── per_sample.jsonl
    │   ├── summary.csv
    │   └── run_metadata.json
    ├── online_smoke/
    ├── online_full/<new condition>/
    └── aggregate/
        ├── factorial_cells.{csv,json}
        ├── factorial_simple_effects.csv
        ├── factorial_interactions.csv
        ├── factorial_task_success.csv
        ├── factorial_offline_metrics.csv
        ├── factorial_paired_transitions.csv
        ├── factorial_replan_stage_metrics.csv
        ├── factorial_summary.{json,md}
        ├── aggregate_metadata.json
        └── plots/
            ├── figure_A_complete_factorial_cell_plot.png
            ├── figure_B_contextual_contribution_of_B.png
            ├── figure_C_pairwise_interaction_summary.png
            ├── figure_D_retrieval_schedule_schematic.png
            └── figure_E_task_level_factorial_heatmap.png
```

完整八 cell 聚合与五张图生成后达到 stop rule。不得由这些脚本继续启动单层 sweep、
head/token/channel pruning、K/V replacement、stale K/V、residual patching、新 suite
或更多 seed。

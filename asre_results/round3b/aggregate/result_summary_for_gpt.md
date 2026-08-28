# Fast-WAM ASRE Round 3B: Matched-Shape K/V Replacement Result Summary

## Executive result

The pre-specified decision is **GO** (`strong_content_support`). The observed correct-to-wrong loss is at least 20 percentage points, and the task-hierarchical interval excludes losses of 5 points or less.

The primary wrong-minus-correct contrast is -95.0 pp, with paired 95% CI [-99.0 pp, -90.0 pp] and task-hierarchical 95% CI [-100.0 pp, -88.0 pp].

## Online success

| Condition | Success | Paired 95% CI | Task-hierarchical 95% CI |
|---|---:|---:|---:|
| Correct Current K/V | 95/100 (95.0%) | [90.0%, 99.0%] | [88.0%, 100.0%] |
| Wrong Same-Task Scene K/V | 0/100 (0.0%) | [0.0%, 0.0%] | [0.0%, 0.0%] |
| No Video K/V | 0/100 (0.0%) | [0.0%, 0.0%] | [0.0%, 0.0%] |

## Paired contrasts

| Contrast (target - reference) | Delta | Paired 95% CI | Task 95% CI | Induced | Rescued | McNemar p |
|---|---:|---:|---:|---:|---:|---:|
| `wrong_minus_correct` | -95.0 pp | [-99.0 pp, -90.0 pp] | [-100.0 pp, -88.0 pp] | 95 | 0 | 5.049e-29 |
| `no_video_minus_correct` | -95.0 pp | [-99.0 pp, -90.0 pp] | [-100.0 pp, -88.0 pp] | 95 | 0 | 5.049e-29 |
| `wrong_minus_no_video` | +0.0 pp | [+0.0 pp, +0.0 pp] | [+0.0 pp, +0.0 pp] | 0 | 0 | 1 |

The descriptive `content_gap_fraction` is 1.000. It is not a mediation fraction or variance-explained estimate.

## Per-task outcomes

| Task | Correct | Wrong scene | No video | Wrong - correct |
|---:|---:|---:|---:|---:|
| 0 | 100.0% | 0.0% | 0.0% | -100.0 pp |
| 1 | 80.0% | 0.0% | 0.0% | -80.0 pp |
| 2 | 100.0% | 0.0% | 0.0% | -100.0 pp |
| 3 | 100.0% | 0.0% | 0.0% | -100.0 pp |
| 4 | 100.0% | 0.0% | 0.0% | -100.0 pp |
| 5 | 90.0% | 0.0% | 0.0% | -90.0 pp |
| 6 | 100.0% | 0.0% | 0.0% | -100.0 pp |
| 7 | 100.0% | 0.0% | 0.0% | -100.0 pp |
| 8 | 90.0% | 0.0% | 0.0% | -90.0 pp |
| 9 | 90.0% | 0.0% | 0.0% | -90.0 pp |

## Offline fixed-state action sensitivity

| Condition | Prefix RMS | Full RMS | Translation RMS | Rotation RMS | Cosine | Grip flip |
|---|---:|---:|---:|---:|---:|---:|
| Correct Current K/V | 0.1367 | 0.1370 | 0.1463 | 0.1189 | 0.9933 | 1.06% |
| Wrong Same-Task Scene K/V | 0.3700 | 0.4248 | 0.3154 | 0.3996 | 0.9487 | 3.93% |
| No Video K/V | 0.9641 | 0.9689 | 1.1457 | 0.6429 | 0.3120 | 61.52% |

The direct wrong-vs-correct executed-prefix RMS is 0.3418 (episode-clustered 95% CI [0.3243, 0.3601]).

## Machinery and donor controls

- Self replacement passed: `True`; max raw-action absolute difference `0.0`.
- Online donors use `(recipient_trial + 1) mod 10` within each task and a fixed first-policy-query visual observation after the evaluator's 30 dummy steps.
- Offline donors are cyclic derangements within the same task and saved replan index, always from a different episode.
- Donor K/V is recomputed at every recipient replan using the fixed donor image plus current recipient text context and proprioception; this isolates scene content, not temporal freshness.
- Layer-wise current/donor K/V scale summaries for layers 15-29 are in `kv_scale_sanity.csv`; shapes, dtypes and finiteness were strictly checked.

## Provenance and scope

- Round-3A frozen parent: `ASRE-round3a-factorial` / `d36383c16d974ba1e5a750c088327a7a88baa8fb`.
- Validated Round-3A run code: `92e842c79f5209b31c2653944cc0c1719e95eb9e`.
- Round-3B source commit: `fbefcee942c0a40e9eb1f06e60a9a958dea3f2e1`.
- Checkpoint SHA256: `1000437cfcf55c000094f79a2600634c502bcb5b492476b94bf8509883a49579`.
- Dataset statistics SHA256: `30f81ad7d5076e97323e3328bce003e01a04cb21327b5bacd21bb72846768638`.
- Valid state manifest SHA256: `a05ebb6b50d64eb022103269a0eef595e749fe8c16e49e996ba08669f44378cd`.
- Prompt cache SHA256: `4eab1fbcede2efd6bba48a5bb3f9eaca0e9e94b03d05ceb497255d4dda82ba9a`.
- Online donor mapping SHA256: `b072adef3aebda9e5449f6e9bc90baade8e84e4703c9fa500642157b59643a6c`.
- Donor observation manifest SHA256: `3a568b0a47e7cacf26914c881c859362d84743d4b5194017c5d126a0d85ba5bd`.
- One checkpoint, one seed and LIBERO-Spatial only; 10 trials per task.
- Fixed wrong-scene content does not isolate temporal freshness.
- The Round-3B stop rule was applied; no stale/cross-task/key-only/value-only, patching, sparsity, pruning, additional-suite or additional-checkpoint experiment is part of this run.

## Questions for GPT discussion

1. Is the GO/VERIFY/STOP interpretation appropriately calibrated to the paired and task-hierarchical uncertainty?
2. How strongly can the result revise the causal interpretation of the Rounds 1-3A deletion experiments?
3. If classification is VERIFY, which single matched-shape control should be pre-registered next without broadening into sparsity experiments?

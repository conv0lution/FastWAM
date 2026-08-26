# Fast-WAM ASRE Round 3A: Complete Late-Half Factorial Result Summary

## Executive summary

Round 3A completed the pre-registered late-half `2^3` factorial for direct
video K/V retrieval in the Fast-WAM action stream. All factorial cells keep
direct video retrieval disabled at action layers 0–14 and define:

- A = retrieval at layers 15–19;
- B = retrieval at layers 20–24;
- C = retrieval at layers 25–29.

Five cells came from the frozen Round-2 run and three missing cells were run
under the same paired protocol. The complete success pattern is:

> `000=0%`, `001=0%`, `010=0%`, `011=49%`, `100=8%`, `101=86%`,
> `110=61%`, and `111=95%`.

The central result is not a ranking of individual layer groups. It is a
strongly context-dependent cooperative structure:

1. B alone does not improve observed success over no late retrieval: both
   `010` and `000` are 0/100.
2. B is highly valuable when exactly one outer region is present. Adding B to
   A raises success from 8% to 61% (`+53 pp`), and adding B to C raises it from
   0% to 49% (`+49 pp`).
3. When A and C are already both present, adding B raises success from 86% to
   95% (`+9 pp`). The episode-paired interval is `[+2,+17] pp`, but the
   task-hierarchical interval is `[-1,+20] pp`; this smaller increment is not
   robust under the report's across-task support criterion.
4. A and C retain a positive averaged interaction of `+31.5 pp`, with paired
   95% CI `[+23.0,+40.0] pp`, task-hierarchical CI `[+13.5,+50.5] pp`, and
   Holm-adjusted sign-flip `p=7.00e-5`. Thus, the A+C complementarity proposed
   after Round 2 remains visible in the completed factorial.
5. The three-way A×B×C contrast is very large and negative: `-93 pp`, with
   paired 95% CI `[-111,-75] pp`, task-hierarchical CI `[-132,-53] pp`, and
   Holm-adjusted `p=7.00e-5`. On the additive probability scale, the
   benefit of forming any pair is much larger when the third region is absent
   than when it is present. This is evidence of factorial context dependence,
   not evidence that one region mechanistically inhibits another.
6. The averaged AB and BC interactions are only `+6.5 pp` and `+2.5 pp` and
   are not statistically supported. This must not be read as absence of
   contextual interaction: large positive and negative conditional
   interactions cancel in those averages. The significant three-way contrast
   exposes that cancellation.
7. Fixed-state action RMS remains informative but is not a surrogate for
   closed-loop success. Cells `011` and `101` have almost identical
   executed-prefix RMS (`0.3245` and `0.3279`) but success rates of 49% and
   86%. Cell `110` has a larger RMS of `0.5212` yet reaches 61%.

The strongest defensible conclusion is that late direct visual retrieval is
distributed, cooperative, and context dependent. None of A, B, or C is
individually sufficient. Any pair recovers substantial behavior, A+C is the
strongest pair, and all three together recover near-baseline performance.
Semantic labels such as “bridge,” “grounding,” or “refinement” remain
hypotheses rather than measured functions.

## Intervention and factorial definition

The model has 30 action layers. A cell's `keep` schedule identifies only the
action layers allowed to consume their corresponding cached video K/V. At all
other action layers, the validated diagnosis intervention omits video K/V
from action attention.

All action and video Transformer blocks still execute normally. The
intervention does not:

- delete or skip model layers;
- zero video features or K/V tensors;
- alter video prefill or hidden states;
- change model weights, text conditioning, or proprioception;
- change the scheduler, action horizon, inference-step count, or replanning
  interval.

The factorial therefore diagnoses access to direct cached video K/V. It does
not identify a minimal executable circuit or establish a compute-saving
architecture. Visual information may still propagate through the action
residual stream when direct retrieval is disabled.

The mapping is:

| Cell | A | B | C | Condition | Enabled direct video-retrieval layers | Provenance |
|---|---:|---:|---:|---|---|---|
| 000 | 0 | 0 | 0 | `keep_none_late` | none | new Round 3A |
| 001 | 0 | 0 | 1 | `keep_25_29` | 25–29 | frozen Round 2 |
| 010 | 0 | 1 | 0 | `keep_20_24` | 20–24 | new Round 3A |
| 011 | 0 | 1 | 1 | `keep_20_29` | 20–29 | frozen Round 2 |
| 100 | 1 | 0 | 0 | `keep_15_19` | 15–19 | frozen Round 2 |
| 101 | 1 | 0 | 1 | `keep_15_19_25_29` | 15–19 and 25–29 | frozen Round 2 |
| 110 | 1 | 1 | 0 | `keep_15_24` | 15–24 | new Round 3A |
| 111 | 1 | 1 | 1 | `keep_15_29` | 15–29 | frozen Round 2 |

Layers 0–14 are disabled in every factorial cell. Accordingly, all main and
interaction effects below are conditional on the absence of direct retrieval
at layers 0–14. The separate full-retrieval Round-2 baseline was 97%; the
best factorial cell, `111`, was 95%.

## Experimental protocol and completion audit

- Checkpoint: `libero_uncond_2cam224.pt`
- Suite: LIBERO-Spatial, all 10 tasks
- Online evaluation: 10 paired trials per task and cell, 100 episodes per
  cell
- New Round-3A online rollouts: 300 full episodes plus 6 smoke episodes
- Complete factorial dataset: 800 paired cell-episodes, including the five
  frozen Round-2 cells
- Seed: 42, with the same task/trial identities and ordering
- Action horizon: 32
- Inference steps: 10
- Replan interval: 10 environment steps
- Gripper binarization: enabled, as in Round 2
- Compiled action inference: enabled, as in Round 2
- Prompt context: frozen Round-2 cache
- New-condition GPU mapping: `000 -> GPU0`, `010 -> GPU1`, `110 -> GPU2`
- GPU model: NVIDIA RTX A5000; no DDP
- PyTorch: 2.7.1+cu128; CUDA: 12.8
- Rollout videos: not saved

The smoke gate ran task 0 with two paired trials for each new cell. All three
conditions completed with the expected retrieval configuration, finite action
traces, matching task/trial IDs, and compatible checkpoint, prompt cache, and
state-manifest provenance.

Each new offline condition replayed the same ordered 499 QC-valid states from
100 episode clusters. The three `per_sample.jsonl` files contain exactly 499
records each. The full online output contains exactly 10 task result files per
new condition, and every launcher condition state is `complete`. The driver
finished with exit code 0 after offline replay, smoke, full evaluation,
aggregation, and all five plots.

A post-run scan found no traceback, exception, OOM, EGL error, NaN, Inf,
interrupted run, or failed launcher state in the completed Round-3A artifacts.
The Round-3A aggregation/statistics/plot tests were rerun after completion:
`22 passed, 6 subtests passed`.

Recorded provenance is:

- Round-3A code and run commit:
  `92e842c79f5209b31c2653944cc0c1719e95eb9e`
- Frozen Round-2 factorial-cell run commit:
  `dc0e65621c011250431fc8c27e9aadc4943cc826`
- Round-2 frozen tag: `ASRE-round2-sufficiency` at `7d7d033`
- Checkpoint SHA256:
  `1000437cfcf55c000094f79a2600634c502bcb5b492476b94bf8509883a49579`
- Dataset-statistics SHA256:
  `30f81ad7d5076e97323e3328bce003e01a04cb21327b5bacd21bb72846768638`
- Round-1 source-manifest SHA256:
  `ed582c1420a705b08105acc5481f93ef899d6677eefa48a78a3f6ed0b1f50ab8`
- Round-2 valid-manifest SHA256:
  `a05ebb6b50d64eb022103269a0eef595e749fe8c16e49e996ba08669f44378cd`
- Prompt-context-cache SHA256:
  `4eab1fbcede2efd6bba48a5bb3f9eaca0e9e94b03d05ceb497255d4dda82ba9a`

The current Git worktree has no modified tracked Round-1 or Round-2 artifact.
Only `asre_results/round3a/` is untracked. The validated model-level
intervention under `src/fastwam/models/` was unchanged. Round-3A source changes
only registered the new protocol/conditions, added launch, replay, aggregation,
statistics and plotting code, and extended result-resume metadata validation.

## Statistical definitions

All success effects use the additive probability scale. Percentage-point
values are probability effects multiplied by 100.

The paired bootstrap resamples the complete eight-condition outcome vector
jointly by matched task/trial key. The task-hierarchical paired bootstrap
resamples tasks and then matched trials within task. Both use 10,000
replicates and seed 0. Intervals are pointwise, not simultaneous.

The seven main, pairwise, and three-way factorial contrasts additionally use
a two-sided joint Monte-Carlo paired sign-flip test with 100,000 samples and
seed 0. Holm correction is applied across that family of seven. The test
depends on sign-exchangeability of episode-level contrast scores and is not
claimed to be a design-exact randomized-treatment test.

Simple-effect p-values are exact paired McNemar values. They are descriptive
and unadjusted; they are not members of the seven-contrast Holm family.
Effect sizes and paired/task-hierarchical intervals are primary.

For all-zero cells, the empirical bootstrap intervals are necessarily
`[0,0]` because every observed outcome is a failure. These degenerate
intervals summarize this finite paired sample; they do not prove that the
population success probability is mathematically zero.

Let `yabc` be the success probability for cell `abc`. The seven balanced
factorial estimands are:

- `main_A = 0.25 * [(y100-y000)+(y101-y001)+(y110-y010)+(y111-y011)]`
- `main_B = 0.25 * [(y010-y000)+(y011-y001)+(y110-y100)+(y111-y101)]`
- `main_C = 0.25 * [(y001-y000)+(y011-y010)+(y101-y100)+(y111-y110)]`
- `AB = 0.5 * [(y110-y010-y100+y000)+(y111-y011-y101+y001)]`
- `AC = 0.5 * [(y101-y001-y100+y000)+(y111-y011-y110+y010)]`
- `BC = 0.5 * [(y011-y001-y010+y000)+(y111-y101-y110+y100)]`
- `ABC = (y111-y101-y011+y001)-(y110-y100-y010+y000)`

These probability-scale contrasts are experimental summaries, not a complete
causal model of neural computation.

## Complete factorial cell results

`Executed-prefix RMS` is normalized RMS over six continuous action dimensions
and action indices 0–9, the portion executed before replanning. Its interval
in the source table is clustered by episode. Cosine and gripper flip are also
computed over indices 0–9.

| Cell | Condition | Success [paired 95% CI] | Task-hierarchical 95% CI | Executed-prefix RMS | Cosine | Grip flip |
|---|---|---:|---:|---:|---:|---:|
| 000 | `keep_none_late` | 0% [0, 0] | [0, 0] | 0.9641 | 0.3120 | 61.52% |
| 001 | `keep_25_29` | 0% [0, 0] | [0, 0] | 0.6763 | 0.7497 | 19.38% |
| 010 | `keep_20_24` | 0% [0, 0] | [0, 0] | 0.7488 | 0.7274 | 36.37% |
| 011 | `keep_20_29` | 49% [39, 59] | [30, 69] | 0.3245 | 0.9495 | 6.15% |
| 100 | `keep_15_19` | 8% [3, 14] | [0, 20] | 0.7239 | 0.7390 | 25.69% |
| 101 | `keep_15_19_25_29` | 86% [79, 92] | [75, 95] | 0.3279 | 0.9685 | 1.32% |
| 110 | `keep_15_24` | 61% [51, 70] | [35, 85] | 0.5212 | 0.9319 | 19.20% |
| 111 | `keep_15_29` | 95% [90, 99] | [88, 100] | 0.1367 | 0.9933 | 1.06% |

There is a clear descriptive progression: zero or very low success for all
single-region cells, substantial recovery for every two-region cell, and the
best result with all three regions. It is not a strict count-of-regions rule:
the three pairs differ materially (`BC=49%`, `AB=61%`, `AC=86%`).

## All contextual simple effects

An effect is target minus reference. “Paired CI” and “task CI” are joint
episode-paired and task-hierarchical paired bootstrap intervals.

| Factor added | Fixed context | Transition | Effect | Paired 95% CI | Task 95% CI | Exact McNemar p |
|---|---|---|---:|---:|---:|---:|
| A | B=0, C=0 | 000 -> 100 | +8 pp | [+3, +14] | [0, +20] | 0.00781 |
| A | B=1, C=0 | 010 -> 110 | +61 pp | [+51, +70] | [+35, +85] | 8.67e-19 |
| A | B=0, C=1 | 001 -> 101 | +86 pp | [+79, +92] | [+75, +95] | 2.58e-26 |
| A | B=1, C=1 | 011 -> 111 | +46 pp | [+36, +56] | [+28, +64] | 3.48e-13 |
| B | A=0, C=0 | 000 -> 010 | 0 pp | [0, 0] | [0, 0] | 1.0 |
| B | A=1, C=0 | 100 -> 110 | +53 pp | [+43, +63] | [+29, +77] | 3.11e-15 |
| B | A=0, C=1 | 001 -> 011 | +49 pp | [+39, +59] | [+30, +69] | 3.55e-15 |
| B | A=1, C=1 | 101 -> 111 | +9 pp | [+2, +17] | [-1, +20] | 0.0352 |
| C | A=0, B=0 | 000 -> 001 | 0 pp | [0, 0] | [0, 0] | 1.0 |
| C | A=1, B=0 | 100 -> 101 | +78 pp | [+68, +87] | [+62, +91] | 1.41e-21 |
| C | A=0, B=1 | 010 -> 011 | +49 pp | [+39, +59] | [+30, +69] | 3.55e-15 |
| C | A=1, B=1 | 110 -> 111 | +34 pp | [+23, +45] | [+7, +62] | 5.65e-8 |

The paired transitions make the B result especially concrete:

- `000 -> 010`: no successes in either condition;
- `100 -> 110`: 54 failures were rescued and 1 success was lost, net `+53`;
- `001 -> 011`: 49 failures were rescued and none was lost, net `+49`;
- `101 -> 111`: 12 failures were rescued and 3 successes were lost, net `+9`.

A is the only region with a nonzero observed effect in every context, although
its alone-context task-hierarchical interval touches zero. C has no observed
standalone benefit but is strongly positive whenever A or B is present. B has
no observed standalone benefit, is strongly positive with either A or C
alone, and has a smaller, task-uncertain increment when A+C are already
present.

## Main and interaction contrasts

| Contrast | Effect | Paired 95% CI | Task-hierarchical 95% CI | Raw sign-flip p | Holm p |
|---|---:|---:|---:|---:|---:|
| Main A | +50.25 pp | [+46.25, +54.25] | [+41.75, +59.00] | 1.00e-5 | 7.00e-5 |
| Main B | +27.75 pp | [+23.75, +31.75] | [+19.24, +36.50] | 1.00e-5 | 7.00e-5 |
| Main C | +40.25 pp | [+35.50, +45.00] | [+29.50, +49.25] | 1.00e-5 | 7.00e-5 |
| A×B averaged over C | +6.5 pp | [-1.0, +14.0] | [-5.5, +19.5] | 0.1257 | 0.2514 |
| A×C averaged over B | +31.5 pp | [+23.0, +40.0] | [+13.5, +50.5] | 1.00e-5 | 7.00e-5 |
| B×C averaged over A | +2.5 pp | [-5.0, +10.0] | [-13.0, +17.5] | 0.5970 | 0.5970 |
| A×B×C | -93.0 pp | [-111.0, -75.0] | [-132.0, -53.0] | 1.00e-5 | 7.00e-5 |

The main effects are averages across four very different contexts. They show
that each region contributes positively on average, with A largest, then C,
then B. They do not imply an ordered semantic pipeline.

The interaction decomposition explains the apparently small average AB and
BC terms:

- AB interaction when C=0: `+53 pp`; when C=1: `-40 pp`; average `+6.5 pp`.
- AC interaction when B=0: `+78 pp`; when B=1: `-15 pp`; average `+31.5 pp`.
- BC interaction when A=0: `+49 pp`; when A=1: `-44 pp`; average `+2.5 pp`.

In each case, the pairwise interaction becomes 93 pp more negative when the
third factor is enabled. This common change is exactly the `ABC=-93 pp`
contrast. Large positive pair synergy in sparse contexts changes to
diminishing returns in the dense context. Because success is bounded and
`111` is already at 95%, ceiling effects may contribute to this negative
probability-scale three-way contrast. It should not be labeled neural
antagonism or inhibition.

## Answers to the pre-specified questions

### Q1. Does B alone perform above no retrieval?

No observed evidence. Both `y010` and `y000` are 0/100, giving `0 pp` and
identical all-failure paired outcomes. This does not prove equality of their
population success probabilities.

### Q2. Does adding B to A help when C is absent?

Yes. `y110-y100=+53 pp`, with paired CI `[+43,+63]`, task-hierarchical CI
`[+29,+77]`, and 54 rescues versus 1 induced failure.

### Q3. Does adding B to C help when A is absent?

Yes. `y011-y001=+49 pp`, with paired CI `[+39,+59]`, task-hierarchical CI
`[+30,+69]`, and 49 rescues with no induced failures.

### Q4. Does adding B to A+C help when both outer regions are present?

The observed increment is `+9 pp`. Its paired CI excludes zero, but the
task-hierarchical CI is `[-1,+20] pp`, and the unadjusted McNemar p-value is
0.0352. This is compatible with a modest benefit but does not meet the stated
criterion for task-generalized support.

### Q5. Is B strongly context dependent?

Yes on the observed probability scale. Its four simple effects are `0`,
`+53`, `+49`, and `+9 pp`. The completed factorial shows that B contributes
little alone, strongly complements either A or C when the other is absent,
and adds much less when both are present. The strong three-way contrast is the
formal summary of how these conditional relationships change.

### Q6. Is strong A+C complementarity still visible?

Yes, when averaged over B: `A×C=+31.5 pp`, with both paired intervals excluding
zero and Holm-adjusted `p=7.00e-5`. It is not uniform: AC interaction is
`+78 pp` when B is absent and `-15 pp` when B is present. Thus A+C
complementarity is supported but itself context dependent.

### Q7. Is there evidence for a genuine A×B×C interaction?

There is strong evidence for a nonzero three-way experimental contrast on the
additive success-probability scale: `-93 pp`, with both intervals excluding
zero and the sign-flip p-value surviving correction across the seven-contrast
Holm family. “Genuine” should be restricted to this statistical estimand. The
contrast does not identify a semantic three-stage mechanism, and its magnitude
may partly reflect the bounded outcome scale and near-ceiling `111` cell.

## Task-level heterogeneity

Each entry is success among 10 paired trials.

| Task | y000 | y001 | y010 | y011 | y100 | y101 | y110 | y111 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0% | 0% | 0% | 100% | 10% | 90% | 100% | 100% |
| 1 | 0% | 0% | 0% | 20% | 0% | 90% | 100% | 80% |
| 2 | 0% | 0% | 0% | 70% | 0% | 100% | 70% | 100% |
| 3 | 0% | 0% | 0% | 70% | 20% | 100% | 60% | 100% |
| 4 | 0% | 0% | 0% | 50% | 0% | 60% | 0% | 100% |
| 5 | 0% | 0% | 0% | 10% | 50% | 80% | 100% | 90% |
| 6 | 0% | 0% | 0% | 80% | 0% | 100% | 70% | 100% |
| 7 | 0% | 0% | 0% | 10% | 0% | 90% | 10% | 100% |
| 8 | 0% | 0% | 0% | 50% | 0% | 70% | 100% | 90% |
| 9 | 0% | 0% | 0% | 30% | 0% | 80% | 0% | 90% |

The new AB cell `110` is particularly heterogeneous: it reaches 100% on
tasks 0, 1, 5, and 8, but 0% on tasks 4 and 9 and 10% on task 7. Adding C to
form `111` produces the largest descriptive rescues on tasks 4 (`0 -> 100%`),
7 (`10 -> 100%`), and 9 (`0 -> 90%`). Conversely, tasks 1 and 8 are 20 and
10 pp lower in `111` than `110`; with only 10 trials per task, these should
not be interpreted as harm or improvement without more replication.

The AC cell `101` is much more uniformly successful than BC cell `011`, even
though their aggregate offline RMS values are almost identical. This
reinforces that action-error structure and closed-loop dynamics are important.

## Offline action sensitivity

| Cell | Success | Executed-prefix RMS | Full-chunk RMS | Translation RMS | Rotation RMS | Cosine | Grip flip |
|---|---:|---:|---:|---:|---:|---:|---:|
| 000 | 0% | 0.9641 | 0.9689 | 1.1457 | 0.6429 | 0.3120 | 61.52% |
| 001 | 0% | 0.6763 | 0.7672 | 0.7937 | 0.4775 | 0.7497 | 19.38% |
| 010 | 0% | 0.7488 | 0.7909 | 0.8840 | 0.5101 | 0.7274 | 36.37% |
| 011 | 49% | 0.3245 | 0.3722 | 0.3667 | 0.2478 | 0.9495 | 6.15% |
| 100 | 8% | 0.7239 | 0.6999 | 0.8356 | 0.5202 | 0.7390 | 25.69% |
| 101 | 86% | 0.3279 | 0.2981 | 0.3406 | 0.2910 | 0.9685 | 1.32% |
| 110 | 61% | 0.5212 | 0.5133 | 0.5820 | 0.4011 | 0.9319 | 19.20% |
| 111 | 95% | 0.1367 | 0.1370 | 0.1463 | 0.1189 | 0.9933 | 1.06% |

More retained regions generally reduce fixed-state deviation, but a single
offline scalar does not determine behavior. The most important counterexample
is `011` versus `101`: RMS is effectively tied, while success differs by
37 pp. Their cosine and gripper errors differ substantially, suggesting that
direction and action-channel structure matter. Even those richer offline
metrics do not capture intervened-policy visitation or accumulated closed-loop
errors.

Mean executed-prefix RMS over the five saved baseline replan positions is:

| Cell | Replan 0 | Replan 1 | Replan 2 | Replan 3 | Replan 4 |
|---|---:|---:|---:|---:|---:|
| 000 | 1.156 | 1.101 | 0.969 | 0.769 | 0.827 |
| 001 | 0.675 | 0.656 | 0.693 | 0.647 | 0.711 |
| 010 | 0.916 | 0.857 | 0.742 | 0.591 | 0.640 |
| 011 | 0.333 | 0.322 | 0.328 | 0.307 | 0.333 |
| 100 | 0.966 | 0.815 | 0.648 | 0.565 | 0.627 |
| 101 | 0.432 | 0.319 | 0.285 | 0.285 | 0.319 |
| 110 | 0.711 | 0.538 | 0.463 | 0.434 | 0.461 |
| 111 | 0.166 | 0.144 | 0.129 | 0.120 | 0.124 |

Replan 0 contains 99 samples because the single QC-excluded state was the
first query of the first episode; every later replan stage contains 100. The
offline states come from baseline visitation, so the decreasing curves in
several cells must not be described as adaptation or self-repair.

## Joint interpretation across Rounds 1–3A

Round 1 measured marginal necessity by deleting one five-layer retrieval
window while leaving the others enabled. Round 2 measured conditional
sufficiency under sparse keep schedules. Round 3A completed the factorial
among the three late windows. These are related but distinct estimands.

The combined evidence supports:

1. **Late retrieval is collectively important.** Removing all late direct
   retrieval gives 0%, while retaining all of A+B+C gives 95%.
2. **No late region is individually sufficient.** A, B, and C alone give 8%,
   0%, and 0%.
3. **Every pair has substantial but unequal capacity.** BC gives 49%, AB 61%,
   and AC 86%. Thus, useful behavior is distributed rather than confined to a
   single five-layer window.
4. **A and C are the strongest pair.** Their positive averaged factorial
   interaction is supported after multiplicity correction, consistent with
   the Round-2 complementarity result.
5. **B is not irrelevant.** Its Round-1 marginal deletion cost was only 8 pp,
   and B alone is 0%, but B contributes 49–53 pp when paired with exactly one
   outer region. The full factorial directly demonstrates why a weak marginal
   deletion effect cannot be equated with lack of conditional value.
6. **The late-half computation is not additive.** Large sparse-context pair
   gains, smaller dense-context increments, and the `-93 pp` three-way
   contrast reject a simple context-free contribution account on the success
   scale.
7. **Early-layer conclusions remain conditional.** The factorial fixes 0–14
   off. The 95% `111` result remains consistent with collective omission of
   early direct retrieval in this screening run, but it is not a formal
   non-inferiority result and does not show that early blocks or early visual
   representations are useless.

## Observed, supported, and hypothetical claims

### Observed

- The eight success rates and all paired task/trial outcomes in the tables
  above were directly measured.
- B's four simple effects are 0, +53, +49, and +9 pp.
- A+C reaches 86%, all three regions reach 95%, and full Round-2 retrieval
  reaches 97%.
- Offline action deviations differ across cells and do not map one-to-one to
  online success.

### Supported by the pre-specified uncertainty rule

- A, B, and C each have a positive average main effect.
- The averaged A×C interaction is positive.
- Effects of B with A alone and C alone are positive.
- The A×B×C probability-scale contrast is strongly nonzero and negative.
- The contribution of each region depends on which other regions retain
  direct video retrieval; a context-free additive account is inadequate.

### Hypotheses requiring mechanism controls

- B may coordinate, bridge, or correct information established or consumed by
  the outer late regions.
- A may establish action–vision coupling and C may perform final re-grounding.
- Residual-stream transport may explain why a region can matter even when
  fresh retrieval is absent elsewhere.

None of those semantic roles is identified by removal of video K/V alone.
Removing keys changes both visual content availability and attention key-set
size/softmax normalization. Matched-shape wrong-observation or stale-K/V
controls and residual patching would be needed to separate these mechanisms.
They are outside the Round-3A stop rule and were not launched.

## Defensible paper-level wording

A conservative statement is:

> Completing a paired `2^3` factorial over Fast-WAM's three late action-layer
> video-retrieval regions revealed a cooperative and strongly
> context-dependent pathway. None of the three five-layer regions was
> individually sufficient on LIBERO-Spatial, whereas each two-region schedule
> recovered substantial but unequal success and the complete late half
> reached 95%. The middle region contributed no observed success alone but
> added 49–53 percentage points when paired with either outer region, while
> its increment was smaller once both outer regions were present. These
> results support distributed late visual conditioning and reject a simple
> context-free additive account, without assigning semantic functions to
> individual regions.

Avoid stronger statements such as:

- “B is a proven bridge/refinement module.”
- “A, B, and C form a minimal circuit.”
- “The negative three-way interaction means the regions inhibit one another.”
- “Layers 0–14 are useless.”
- “The 15–29 schedule is proven non-inferior or action equivalent.”
- “The experiment demonstrates deployable pruning or a compute reduction.”

## Scope limits

- One checkpoint, one seed, and one LIBERO suite were evaluated.
- There are only 10 trials per task, limiting task-specific stability and
  formal non-inferiority claims.
- Bootstrap intervals are pointwise; simple-effect p-values are unadjusted.
- The sign-flip p-values rely on an exchangeability assumption and have a
  Monte-Carlo floor near `1e-5`.
- Probability-scale interactions are sensitive to the bounded success scale
  and possible ceiling effects.
- The offline bank contains correlated replan states from baseline visitation,
  not states reached by intervened policies.
- Five-layer regions cannot localize importance to individual layers, heads,
  tokens, or channels.
- Retrieval removal confounds missing visual content with a changed attention
  key set and softmax normalization.
- Near-equal task success does not imply action equivalence.

## Commands and implementation audit

The successful end-to-end command was:

```bash
cd /home/zhaizicheng/Reasoning/FastWAM
set -o pipefail
./experiments/asre_diagnosis/round3a/run_full_experiment.sh \
  2>&1 | tee asre_results/round3a/logs/driver.log
```

That script ran frozen-Round-2 preflight, offline replay, online smoke, online
full evaluation, aggregation, and plotting in order. The explicit aggregation
and plot commands were:

```bash
PY=/home/zhaizicheng/miniconda3/envs/fastwam/bin/python

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

Commit `92e842c` changed these source groups:

- `.gitignore`;
- `experiments/asre_diagnosis/common.py`;
- the new `experiments/asre_diagnosis/round3a/` package, README, launch/replay,
  aggregation/statistics, plotting, and shell entry points;
- `experiments/asre_diagnosis/tests/test_common.py` and five Round-3A test
  modules;
- `experiments/libero/eval_libero_single.py` for Round-3A protocol-aware
  resume metadata validation.

No file under the validated core model implementation
`src/fastwam/models/` changed. No Round-1 or Round-2 result was overwritten or
modified. No Round 3B or additional schedule was launched.

## Result artifacts

The primary aggregate directory is `asre_results/round3a/aggregate/` and
contains:

- `factorial_cells.csv` and `factorial_cells.json`: all cells, online/offline
  metrics, and provenance;
- `factorial_simple_effects.csv`: all 12 contextual effects;
- `factorial_interactions.csv`: exact formulas, main effects, pairwise and
  three-way contrasts, intervals, and multiplicity-adjusted p-values;
- `factorial_paired_transitions.csv`: matched outcome transitions and McNemar
  statistics;
- `factorial_task_success.csv`: task-level cell rates;
- `factorial_offline_metrics.csv`: 499-state offline metrics;
- `factorial_replan_stage_metrics.csv`: exploratory stage summaries;
- `factorial_summary.json` and `factorial_summary.md`: machine-readable and
  compact machine-generated summaries;
- `aggregate_metadata.json`: analysis version, seeds, roots, and manifest
  identity;
- `plots/figure_A_complete_factorial_cell_plot.png`;
- `plots/figure_B_contextual_contribution_of_B.png`;
- `plots/figure_C_pairwise_interaction_summary.png`;
- `plots/figure_D_retrieval_schedule_schematic.png`;
- `plots/figure_E_task_level_factorial_heatmap.png`.

Raw new-condition outputs are under:

- `asre_results/round3a/offline/`;
- `asre_results/round3a/online_smoke/`;
- `asre_results/round3a/online_full/`;
- `asre_results/round3a/logs/driver.log`.

## Questions for GPT discussion

1. What is the strongest publishable statement about the full factorial that
   does not turn statistical context dependence into an unsupported semantic
   layer-role claim?
2. How should the large negative three-way probability-scale interaction be
   presented given the near-ceiling `111` cell and cancellation in the
   averaged AB and BC interactions?
3. Should the main paper figure lead with the eight-cell pattern, the four B
   simple effects, or the sparse-versus-dense interaction reversal?
4. Does the `011` versus `101` offline-RMS tie but 37 pp success gap justify a
   stronger emphasis on action direction/gripper structure and closed-loop
   evaluation?
5. Which matched-shape K/V replacement control would best distinguish missing
   current visual content from key-set/softmax artifacts?
6. What confirmatory sample size, seeds, checkpoints, and suites would be
   appropriate before calling `keep_15_29` non-inferior within a 5 pp margin?
7. After respecting the Round-3A stop rule, should the next mechanistic study
   prioritize matched-shape K/V replacement, residual-stream patching, or a
   deliberately staged combination?

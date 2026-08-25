# Fast-WAM ASRE Round 2: Sufficiency and Retrieval-Structure Result Summary

## Executive summary

Round 2 tested which action layers must retain **direct access to cached video
K/V** after Round 1 identified a non-uniform necessity profile across the 30
action layers.

The main result is:

> Direct video retrieval in the complete late half, layers 15–29, is the only
> Round-2 schedule that passes the pre-specified 100-episode screening rule:
> 95% success versus 97% for the same-run baseline. This is a candidate
> action-sufficient retrieval schedule, not a formal demonstration of
> equivalence or non-inferiority.

The other principal findings are:

1. Early-only retrieval is grossly insufficient: `keep_00_14` obtains 0%
   success and `keep_00_19` obtains 12%.
2. The two Round-1 critical windows are not individually sufficient:
   `keep_15_19` obtains 8%, and `keep_25_29` obtains 0%.
3. The two windows are strongly complementary: enabling both raises success
   from 8%/0% individually to 86%. However, the combined schedule is still 11
   percentage points below baseline, with a task-hierarchical 95% CI of
   [-22, -2] pp and Holm-adjusted McNemar p=0.0148. Thus, complementarity is
   supported, but joint sufficiency is not.
4. Layer-group effects are strongly contextual and inconsistent with a simple
   additive interpretation. In particular, layers 20–24 cannot be called
   unimportant merely because their Round-1 group deletion had a relatively
   small aggregate effect.
5. Fixed-state action deviation is informative but is not a surrogate for
   closed-loop success. `keep_20_29` and `keep_15_19_25_29` have almost the
   same executed-prefix normalized RMS (0.3245 and 0.3279), but their online
   success rates are 49% and 86%.

The strongest joint Round-1/Round-2 interpretation is therefore a
**collaborative, multi-stage, context-dependent late visual-conditioning
pathway**, rather than a single independently sufficient layer window.

## Intervention and terminology

The model has 30 action layers. A Round-2 `keep` schedule specifies the action
layers that may directly consume the cached video K/V tokens. At every other
action layer, only those video K/V tokens are omitted from that layer's action
attention computation.

All 30 Transformer blocks are still executed. The intervention does not:

- delete or skip model layers;
- zero video features or K/V tensors;
- alter video prefill computation;
- modify hidden states, model weights, text conditioning, or proprioception;
- change the diffusion scheduler, inference-step count, or action horizon.

Consequently, `keep_15_29` means “enable direct video retrieval only at action
layers 15–29,” not “run only the last 15 model layers.” Visual information may
still propagate through the action residual stream at layers where direct
video retrieval is disabled.

The eight conditions were:

| Condition | Direct video-retrieval layers | Disabled retrieval layers | Scientific role |
|---|---|---|---|
| `baseline_round2` | 0–29 | none | Same-run reference |
| `keep_15_29` | 15–29 | 0–14 | Late-half sufficiency |
| `keep_20_29` | 20–29 | 0–19 | Conditional contribution of 15–19 |
| `keep_25_29` | 25–29 | 0–24 | Final-window sufficiency |
| `keep_00_14` | 0–14 | 15–29 | Early retrieval plus residual propagation |
| `keep_00_19` | 0–19 | 20–29 | Early retrieval including 15–19 |
| `keep_15_19` | 15–19 | 0–14 and 20–29 | Middle-window sufficiency |
| `keep_15_19_25_29` | 15–19 and 25–29 | 0–14 and 20–24 | Critical-window complementarity |

## Experimental protocol and integrity checks

- Checkpoint: `libero_uncond_2cam224.pt`
- Suite: LIBERO-Spatial, all 10 tasks
- Online evaluation: 10 paired trials per task and condition, or 100 episodes
  per condition and 800 Round-2 rollouts total
- Seed: 42, with strict task/trial identity across Round-2 conditions
- Action horizon: 32
- Inference steps: 10
- Replan interval: 10 environment steps
- Gripper binarization: enabled
- Compiled action inference: enabled
- Text conditioning: exact stored Round-1 prompt contexts; the T5 encoder was
  not independently reloaded by each process
- GPU execution: one independent condition per A5000; no DDP
- Rollout videos: not saved

The Round-1 state bank contained 500 baseline-visitation states from 100
episodes and five replanning positions per episode. Before viewing Round-2
condition results, a fixed QC rule required a baseline raw-action maximum
absolute replay error no greater than 1e-6, finite tensors, matching [32, 7]
shapes, and exact post-processed gripper values. Exactly one sample,
`task00_episode00_replan000`, exceeded the tolerance with a maximum raw error
of 0.0078125. It was excluded for every condition. Each offline condition
therefore replayed the same ordered 499 samples from the same 100 episode
clusters.

The baseline, early-only, and combined-window smoke conditions completed
before the full launch. All eight offline replays and all eight full online
conditions completed. Each online result contained all 10 tasks, 100 unique
task/trial keys, and the expected retrieval configuration. No condition was
silently resumed from an incompatible result directory.

The run logs were also checked for OOM, EGL, traceback, NaN, and Inf failures;
none were found in the completed matrix. The first aggregation attempt then
stopped before producing results because the metadata validator interpreted a
present and legitimate `sigma_shift: null` as a missing field. The validator
was corrected to distinguish a missing key from an allowed null value, a
regression test was added, all 41 diagnosis tests passed, and aggregation and
plotting were rerun successfully. This was a post-processing validation bug;
the GPU-generated offline and online results were not rerun or altered.

Round-1 and Round-2 baselines both obtained 97/100. In addition to the
machine-generated aggregate-rate check, a post-hoc direct audit of the raw
per-task result files found exact success/failure identity for every task and
trial: 97 success-success pairs, 3 failure-failure pairs, and no cross-round
transitions. This audit concerns binary episode outcomes, not byte-identical
action trajectories.

Recorded provenance includes:

- checkpoint SHA256:
  `1000437cfcf55c000094f79a2600634c502bcb5b492476b94bf8509883a49579`
- dataset-statistics SHA256:
  `30f81ad7d5076e97323e3328bce003e01a04cb21327b5bacd21bb72846768638`
- Round-1 source state-manifest SHA256:
  `ed582c1420a705b08105acc5481f93ef899d6677eefa48a78a3f6ed0b1f50ab8`
- Round-2 valid-manifest SHA256:
  `a05ebb6b50d64eb022103269a0eef595e749fe8c16e49e996ba08669f44378cd`
- prompt-context-cache SHA256:
  `4eab1fbcede2efd6bba48a5bb3f9eaca0e9e94b03d05ceb497255d4dda82ba9a`
- recorded Git commit: `dc0e65621c011250431fc8c27e9aadc4943cc826`
- GPU: NVIDIA RTX A5000
- PyTorch: 2.7.1+cu128; CUDA: 12.8

## Statistical definitions and decision rule

For all baseline comparisons, the success-rate difference is intervention
minus Round-2 baseline. The paired 95% CI resamples paired episodes. The
task-hierarchical 95% CI first resamples tasks and then paired episodes within
each sampled task. Both use 10,000 bootstrap replicates with bootstrap seed 0.

Exact McNemar tests use paired binary outcomes. Holm adjustment covers the
pre-defined family of seven intervention-versus-baseline comparisons. Exact
p-values for non-baseline planned comparisons are reported as raw p-values
and should not be described as Holm-adjusted across all planned contrasts.

The pre-specified **screening** label requires:

- observed aggregate success loss no greater than 5 percentage points; and
- no catastrophic task, defined as a task with baseline success at least 80%
  and an intervention-minus-baseline loss of at least 50 percentage points.

This rule selects candidates for confirmation. It is not a formal
non-inferiority test. In particular, a non-significant McNemar result does not
establish equivalence.

## Complete Round-2 results

`Executed-prefix RMS` is the RMS normalized deviation over the six continuous
action dimensions and action indices 0–9, the portion executed before the
next policy query. Its CI is clustered by episode. Cosine and gripper flip are
also computed over action indices 0–9.

| Condition | # retrieval layers | Executed-prefix RMS [95% cluster CI] | Cosine | Grip flip | Success | Delta | Paired 95% CI | Task-hierarchical 95% CI | Screening result |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `baseline_round2` | 30 | 0.0000 [0.0000, 0.0000] | 1.0000 | 0.00% | 97% | 0 pp | [0, 0] pp | [0, 0] pp | reference |
| `keep_15_29` | 15 | 0.1367 [0.1330, 0.1404] | 0.9933 | 1.06% | **95%** | **-2 pp** | **[-7, +3] pp** | **[-9, +4] pp** | **candidate; no catastrophic task** |
| `keep_20_29` | 10 | 0.3245 [0.3167, 0.3327] | 0.9495 | 6.15% | 49% | -48 pp | [-58, -38] pp | [-67, -29] pp | not candidate; tasks 1, 4, 5, 7, 9 catastrophic |
| `keep_25_29` | 5 | 0.6763 [0.6578, 0.6944] | 0.7497 | 19.38% | 0% | -97 pp | [-100, -93] pp | [-100, -91] pp | not candidate; all tasks catastrophic |
| `keep_00_14` | 15 | 0.8489 [0.8156, 0.8807] | 0.5149 | 43.09% | 0% | -97 pp | [-100, -93] pp | [-100, -91] pp | not candidate; all tasks catastrophic |
| `keep_00_19` | 20 | 0.6032 [0.5773, 0.6284] | 0.8448 | 19.26% | 12% | -85 pp | [-92, -77] pp | [-99, -65] pp | not candidate; tasks 0–4 and 6–9 catastrophic |
| `keep_15_19` | 5 | 0.7239 [0.6947, 0.7532] | 0.7390 | 25.69% | 8% | -89 pp | [-95, -83] pp | [-99, -73] pp | not candidate; tasks 0–4 and 6–9 catastrophic |
| `keep_15_19_25_29` | 10 | 0.3279 [0.3180, 0.3377] | 0.9685 | 1.32% | **86%** | **-11 pp** | **[-19, -4] pp** | **[-22, -2] pp** | not candidate; no catastrophic task |

`keep_15_29` is the only screening candidate. Its task-hierarchical CI still
permits a true loss as large as 9 pp, so the result is appropriately phrased
as “consistent with collective dispensability of early-layer direct video
retrieval when the full late half remains available,” not “proven
action-sufficient” or “proven non-inferior.”

## Paired outcome transitions relative to baseline

The four transition columns are baseline success/intervention success (SS),
baseline success/intervention failure (SF), baseline failure/intervention
success (FS), and baseline failure/intervention failure (FF).

| Condition | SS | SF / induced failures | FS / rescued successes | FF | Exact McNemar p | Holm-adjusted p |
|---|---:|---:|---:|---:|---:|---:|
| `keep_15_29` | 93 | 4 | 2 | 1 | 0.6875 | 0.6875 |
| `keep_20_29` | 48 | 49 | 1 | 2 | 9.06e-14 | 2.72e-13 |
| `keep_25_29` | 0 | 97 | 0 | 3 | 1.26e-29 | 8.84e-29 |
| `keep_00_14` | 0 | 97 | 0 | 3 | 1.26e-29 | 8.84e-29 |
| `keep_00_19` | 11 | 86 | 1 | 2 | 1.14e-24 | 4.55e-24 |
| `keep_15_19` | 8 | 89 | 0 | 3 | 3.23e-27 | 1.62e-26 |
| `keep_15_19_25_29` | 84 | 13 | 2 | 1 | 0.007385 | 0.01477 |

The two apparent “rescues” for `keep_15_29` and for the combined critical
schedule are part of the paired stochastic outcome pattern and should not be
interpreted as an intervention benefit.

## Pre-specified mechanistic comparisons

The eight conditions were designed around directional comparisons rather
than an unordered condition ranking.

| ID | Reference -> target | Question | Paired delta [95% CI] | Task-hierarchical 95% CI | Induced / rescued | Exact p | Interpretation |
|---|---|---|---:|---:|---:|---:|---|
| A | baseline -> `keep_15_29` | Are layers 0–14 collectively dispensable when 15–29 remain? | -2 pp [-7, +3] | [-9, +4] pp | 4 / 2 | 0.6875 | Passes screening, but no equivalence proof |
| B1 | baseline -> `keep_00_14` | Is early-half-only retrieval sufficient? | -97 pp [-100, -93] | [-100, -91] pp | 97 / 0 | 1.26e-29 | No |
| B2 | baseline -> `keep_00_19` | Is retrieval through layer 19 sufficient? | -85 pp [-92, -77] | [-99, -65] pp | 86 / 1 | 1.14e-24 | No |
| B3 | `keep_00_14` -> `keep_00_19` | What does 15–19 contribute in an early-only context? | +12 pp [+6, +19] | [+1, +28] pp | 0 / 12 | 0.000488 | Helpful, but far from sufficient |
| C | `keep_15_29` -> `keep_20_29` | What is the contribution of 15–19 when 20–29 remain? | -46 pp [-56, -36] | [-64, -28] pp | 47 / 1 | 3.48e-13 | 15–19 is critical in this context |
| D | baseline -> `keep_25_29` | Is final-window-only retrieval sufficient? | -97 pp [-100, -93] | [-100, -91] pp | 97 / 0 | 1.26e-29 | No |
| E1 | baseline -> `keep_15_19` | Is middle-window-only retrieval sufficient? | -89 pp [-95, -83] | [-99, -73] pp | 89 / 0 | 3.23e-27 | No |
| E2 | baseline -> `keep_25_29` | Is final-window-only retrieval sufficient? | -97 pp [-100, -93] | [-100, -91] pp | 97 / 0 | 1.26e-29 | No |
| E3 | baseline -> combined critical windows | Are the two windows jointly sufficient? | -11 pp [-19, -4] | [-22, -2] pp | 13 / 2 | 0.007385 | No under the screening rule |
| E4 | `keep_15_19` -> combined | What does adding 25–29 rescue? | +78 pp [+69, +87] | [+62, +91] pp | 2 / 80 | 1.41e-21 | Strong complementarity |
| E5 | `keep_25_29` -> combined | What does adding 15–19 rescue? | +86 pp [+79, +92] | [+75, +95] pp | 0 / 86 | 2.58e-26 | Strong complementarity |

For A, B1, B2, D, E1, E2, and E3, the baseline-family Holm-adjusted values
are given in the complete-results table or transition table. B3, C, E4, and
E5 are direct non-baseline planned contrasts; their p-values above are raw
exact paired values.

The machine diagnostic reports Pattern C as false because Pattern C required
both complementarity **and** candidate-level joint sufficiency. The data do
support complementarity; they do not support joint sufficiency. “Pattern C is
false” must therefore not be paraphrased as “the windows are not
complementary.”

## Task-level heterogeneity

| Task ID | Baseline | Keep 15–29 | Keep 20–29 | Keep 25–29 | Keep 0–14 | Keep 0–19 | Keep 15–19 | Keep 15–19 + 25–29 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 100% | 100% | 100% | 0% | 0% | 10% | 10% | 90% |
| 1 | 100% | 80% | 20% | 0% | 0% | 0% | 0% | 90% |
| 2 | 100% | 100% | 70% | 0% | 0% | 0% | 0% | 100% |
| 3 | 100% | 100% | 70% | 0% | 0% | 20% | 20% | 100% |
| 4 | 100% | 100% | 50% | 0% | 0% | 0% | 0% | 60% |
| 5 | 80% | 90% | 10% | 0% | 0% | 70% | 50% | 80% |
| 6 | 100% | 100% | 80% | 0% | 0% | 0% | 0% | 100% |
| 7 | 100% | 100% | 10% | 0% | 0% | 0% | 0% | 90% |
| 8 | 90% | 90% | 50% | 0% | 0% | 20% | 0% | 70% |
| 9 | 100% | 90% | 30% | 0% | 0% | 0% | 0% | 80% |

Notable patterns are:

- `keep_15_29` is relatively stable across tasks. The visible losses are task
  1 (-20 pp) and task 9 (-10 pp). Task 5 changes from 80% to 90%, but with
  only 10 episodes per task this must not be called an improvement.
- The combined critical schedule has no task meeting the pre-defined
  catastrophic threshold, but it is not uniformly stable: task 4 drops 40 pp,
  and tasks 8 and 9 drop 20 pp.
- `keep_20_29` is highly heterogeneous. Task 0 remains at 100%, whereas tasks
  1, 4, 5, 7, and 9 meet the catastrophic criterion.
- `keep_00_19` and `keep_15_19` retain some success mainly on task 5 and a few
  isolated tasks. `keep_25_29` and `keep_00_14` fail every task.

There are only 10 trials per task, so these patterns are exploratory rather
than stable estimates of task-specific effects.

## Offline action sensitivity

The continuous action decomposition is:

| Condition | Translation RMS | Rotation RMS | Executed-prefix grip flip | Online success |
|---|---:|---:|---:|---:|
| `keep_15_29` | 0.1463 | 0.1189 | 1.06% | 95% |
| `keep_20_29` | 0.3667 | 0.2478 | 6.15% | 49% |
| `keep_25_29` | 0.7937 | 0.4775 | 19.38% | 0% |
| `keep_00_14` | 1.0127 | 0.5567 | 43.09% | 0% |
| `keep_00_19` | 0.6813 | 0.4498 | 19.26% | 12% |
| `keep_15_19` | 0.8356 | 0.5202 | 25.69% | 8% |
| `keep_15_19_25_29` | 0.3406 | 0.2910 | 1.32% | 86% |

Translation RMS exceeds rotation RMS under every intervention. The
`delta_position_x` deviation is generally the largest single continuous
dimension, although x and z are almost tied for `keep_15_29`. These are
descriptive patterns, not causal mediation results. Gripper flip is a rate and
should not be numerically compared as though it were another continuous RMS
dimension.

The strongest caution against using one offline scalar as a behavioral proxy
is the contrast between `keep_20_29` and the combined critical schedule. Their
continuous executed-prefix RMS values are nearly identical, yet the combined
schedule has a higher cosine similarity (0.9685 versus 0.9495), a much lower
gripper flip rate (1.32% versus 6.15%), and 37 pp higher online success. The
direction and structure of action error, closed-loop dynamics, and task
tolerance matter in addition to mean deviation magnitude.

`keep_15_29` also produces a clearly nonzero fixed-state action deviation
while retaining 95% success. Near-baseline task success therefore does not
imply action equivalence.

## Exploratory replan-stage analysis

Mean executed-prefix normalized RMS over the five saved baseline policy-query
positions was:

| Condition | Replan 0 | Replan 1 | Replan 2 | Replan 3 | Replan 4 |
|---|---:|---:|---:|---:|---:|
| `keep_15_29` | 0.166 | 0.144 | 0.129 | 0.120 | 0.124 |
| `keep_20_29` | 0.333 | 0.322 | 0.328 | 0.307 | 0.333 |
| `keep_25_29` | 0.675 | 0.656 | 0.693 | 0.647 | 0.711 |
| `keep_00_14` | 1.043 | 0.980 | 0.849 | 0.662 | 0.713 |
| `keep_00_19` | 0.861 | 0.667 | 0.517 | 0.459 | 0.514 |
| `keep_15_19` | 0.966 | 0.815 | 0.648 | 0.565 | 0.627 |
| `keep_15_19_25_29` | 0.432 | 0.319 | 0.285 | 0.285 | 0.319 |

Replan 0 has 99 samples because the excluded state was the first query of the
first episode; each other stage has 100. Most high-deviation conditions fall
toward replan 3 and rebound slightly at replan 4, but no formal stage-change
test was pre-specified. The states all come from baseline visitation and the
stage distributions differ, so this must not be interpreted as policy
adaptation or self-repair.

## Joint Round-1 and Round-2 interpretation

Round 1 measured the marginal **necessity** of each five-layer window when all
other direct video-retrieval layers remained enabled:

| Round-1 condition | Success | Delta from Round-1 baseline |
|---|---:|---:|
| baseline | 97% | 0 pp |
| drop 00–04 | 97% | 0 pp |
| drop 05–09 | 95% | -2 pp |
| drop 10–14 | 97% | 0 pp |
| drop 15–19 | 70% | -27 pp |
| drop 20–24 | 89% | -8 pp |
| drop 25–29 | 67% | -30 pp |
| drop all | 0% | -97 pp |

Round 2 instead measured **conditional sufficiency** under restricted keep
schedules. These are different estimands, so their numerical effects must not
be added or treated as interchangeable.

Together, the two rounds support the following claims:

1. **Early direct retrieval is collectively dispensable only in a particular
   context.** The three Round-1 deletions within layers 0–14 cause 0, -2, and 0
   pp aggregate changes, and Round-2 `keep_15_29` obtains 95%. This supports
   early-layer direct retrieval being jointly removable in this screening run
   when all of 15–29 remains available. It does not show that early layers or
   early visual representations are useless.
2. **Layers 15–19 form a strongly influential, context-dependent region.**
   Their Round-1 deletion costs 27 pp. Removing them from the late-half
   schedule changes 95% (`keep_15_29`) to 49% (`keep_20_29`), a pre-specified
   paired contrast of -46 pp. Yet 15–19 alone achieves only 8%.
3. **Layers 25–29 are necessary in many contexts but not sufficient.** Their
   Round-1 deletion costs 30 pp, whereas keeping only 25–29 yields 0%.
4. **The two critical windows are complementary but not jointly sufficient.**
   They yield 8% and 0% alone and 86% together, but the combined schedule
   remains detectably below the 97% baseline and fails the screening rule.
5. **Layers 20–24 are likely redundant or conditionally coordinating, not
   unimportant.** Round-1 deletion of 20–24 costs only 8 pp and is not robust
   after the full correction/uncertainty analysis. However, two descriptive,
   post-hoc schedule contrasts point to a larger sparse-context role:
   adding 20–24 to final-only retrieval changes 0% (`keep_25_29`) to 49%
   (`keep_20_29`), and adding 20–24 to the combined critical schedule changes
   86% to 95% (`keep_15_29`). These post-hoc contrasts motivate, but do not
   establish, a bridging or compensatory mechanism.
6. **A simple additive account is not supported.** A window's marginal cost
   when the rest of the network retains video access does not predict its
   isolated or sparse sufficiency. A complete factorial would be needed to
   estimate the interactions formally.

A plausible mechanistic picture is that layers 15–19 establish an effective
action–vision coupling stage, layers 20–24 provide context-dependent bridging
or correction, and layers 25–29 perform final re-grounding before action
readout. These role labels are hypotheses; the current deletion/keep
interventions do not directly measure semantic function.

## Hypothesis status after Round 2

| Hypothesis | Status | Evidence |
|---|---|---|
| Direct retrieval at 0–14 can be jointly omitted when 15–29 remain | Candidate support | 95% versus 97%, but task-hierarchical CI extends to -9 pp |
| Early retrieval followed only by residual propagation is sufficient | Strongly rejected | `keep_00_14=0%`; `keep_00_19=12%` |
| 15–19 is an influential region | Supported | Round-1 deletion -27 pp; Round-2 conditional removal -46 pp |
| 15–19 alone is sufficient | Rejected | `keep_15_19=8%` |
| 25–29 alone is sufficient | Strongly rejected | `keep_25_29=0%` |
| 15–19 and 25–29 are complementary | Supported | 8%/0% alone; 86% combined |
| 15–19 and 25–29 are jointly sufficient | Rejected under current screening rule | Combined is -11 pp; hierarchical CI [-22, -2] pp |
| 20–24 is irrelevant | Not supported | Weak marginal effect but large descriptive conditional effects |
| A simple additive account explains the layer windows | Not supported | Necessity and sufficiency depend strongly on retained context; the factorial is incomplete |
| Offline mean RMS fully explains online success | Rejected | Similar RMS can accompany 49% and 86% success |

The machine-generated pattern flags are A=true, B=true, C=false, D=false,
and E=false. Here C=false means “complementary **and jointly sufficient**” is
false, while E=false occurs because `keep_15_29` does not strongly degrade.

## Defensible paper-level wording

A conservative paper-level statement would be:

> Across a fixed Fast-WAM checkpoint on LIBERO-Spatial, complementary
> necessity and keep-only interventions reveal a heterogeneous,
> context-dependent late visual-conditioning pathway in the action stream.
> Direct video retrieval at layers 15–29 is the only schedule that preserves
> near-baseline performance under the pre-specified screening rule, while
> early-only retrieval and either critical late window alone are insufficient.
> Layers 15–19 and 25–29 show strong behavioral complementarity, but their
> sparse union remains significantly below baseline, implying that effective
> retrieval depends on interactions across the broader late-half computation.

Avoid the following stronger statements:

- “Layers 0–14 are useless.”
- “Layers 15–29 are proven action-sufficient or non-inferior.”
- “Layers 15–19 and 25–29 are a sufficient minimal circuit.”
- “Only the final five layers contain the task-relevant visual information.”
- “Layers 20–24 are unimportant.”
- “The intervention demonstrates a deployable 2x compute reduction.”

The experiment diagnoses direct video K/V access. It does not delete action
blocks, establish a minimal circuit, or yet demonstrate a runtime/memory gain.

## Scope limits and unresolved mechanisms

- Results come from one checkpoint, one seed, and LIBERO-Spatial only.
- Ten episodes per task are insufficient for formal non-inferiority or stable
  task-specific estimates.
- The offline bank contains correlated replanning states and samples only the
  baseline visitation distribution; it does not capture states reached by
  intervened policies or closed-loop error accumulation.
- Round-1 offline summaries used the original 500-state bank, whereas Round 2
  uses the fixed 499-state QC-valid subset. Cross-round offline values are
  useful descriptively but are not identical-population estimates.
- Five-layer group schedules cannot localize a result to a single layer or
  attention head.
- Removing video K/V changes both available visual content and the attention
  key set/softmax normalization. The current result cannot distinguish loss
  of correct visual semantics from structural attention effects.
- Near-baseline success can coexist with materially different actions because
  the benchmark has behavioral tolerance; success equality is not action
  equivalence.
- No compression, pruning, retraining, or out-of-distribution claim has been
  tested.

## Most informative next decisions for discussion

The Round-2 stop rule has been followed: no additional schedules or Round-3
experiments have been launched. Two distinct goals imply different next
steps.

If the goal is **mechanistic identification**, a compact completion of the
late-half factorial is more informative than a broad single-layer sweep.
Define A=15–19, B=20–24, and C=25–29 while always disabling 0–14. Existing
Round-2 conditions cover C, BC, A, AC, and ABC. The missing cells are:

- no direct video retrieval in the late half (000), rerun under the same
  Round-2 protocol rather than substituted from a different round;
- B only: `keep_20_24`;
- AB: `keep_15_24`.

Together these would complete the 2^3 factorial and permit explicit
interaction estimands for A, B, and C. This proposal is post-hoc and should be
pre-registered before execution.

The next mechanism control should preserve attention shape while replacing
correct K/V, for example with wrong-observation or temporally stale same-layer
video K/V. This would help separate dependence on current visual content from
the structural effect of removing keys and changing softmax normalization.
Action-residual patching could then distinguish late fresh retrieval from
visual information already carried in the residual stream.

If the goal is an **action-sufficient schedule or efficiency claim**, first
confirm `keep_15_29` in a pre-registered non-inferiority design with an
explicit margin, substantially more episodes/seeds, additional checkpoints,
and additional LIBERO suites. Only after confirmation would it be reasonable
to implement and benchmark an actual sparse-attention or compute-saving path.

## Questions for GPT discussion

1. What is the strongest publishable claim that preserves the distinction
   between “complementary” and “jointly sufficient”?
2. Is the proposed late-half 2^3 factorial the best next mechanistic design?
   Which interaction estimands and multiplicity correction should be
   pre-specified?
3. Which matched-shape K/V replacement control best distinguishes current
   visual-content dependence from key-set/softmax artifacts?
4. How should action-residual patching be designed to separate late
   re-grounding from residual transport of early visual information?
5. For a 5 pp non-inferiority margin, how many task-stratified episodes, seeds,
   suites, and checkpoints would be needed to confirm `keep_15_29`?
6. Which Round-2 result should be the main paper figure: the complete schedule
   comparison, the critical-window complementarity plot, or a combined
   necessity/sufficiency schematic?
7. Should the 20–24 follow-up prioritize the complete factorial, K/V
   replacement, or both in a staged design?

## Available result artifacts

The compact aggregate directory contains:

- `summary.csv` and `summary.json`: complete condition-level metrics;
- `planned_comparisons.csv`: pre-specified paired contrasts;
- `paired_transitions.csv`: outcome transition counts and McNemar statistics;
- `task_success_rates.csv` and `task_success_delta.csv`: task-level outcomes;
- `offline_action_dimensions.csv`: dimension/group decomposition;
- `replan_stage_metrics.csv`: exploratory temporal analysis;
- `round1_round2_joint_summary.csv`: shared Round-1/Round-2 overview;
- `diagnostic_summary.md` and `diagnostic_summary.json`: machine pattern flags;
- five main plots and one exploratory replan-stage plot.

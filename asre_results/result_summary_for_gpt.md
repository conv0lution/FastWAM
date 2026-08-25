# Fast-WAM ASRE Layer-wise Video K/V Diagnosis: Complete Result Summary

## Objective and intervention

We performed a training-free causal diagnosis on the official Fast-WAM
checkpoint to test whether the layer-wise video representations consumed by
the action stream have heterogeneous importance for action generation and
LIBERO task success.

The model has 30 action layers. At a disabled layer, we removed only the
cached video K/V tokens from that layer's action-attention computation. We did
not zero video features, skip video blocks, alter model weights, change text or
proprioception conditioning, or change inference hyperparameters.

The eight conditions were:

- baseline: no disabled layers
- drop_00_04
- drop_05_09
- drop_10_14
- drop_15_19
- drop_20_24
- drop_25_29
- drop_all: all 30 layers disabled

## Experimental setup and validation

- Checkpoint: `libero_uncond_2cam224.pt`
- Suite: LIBERO-Spatial, all 10 tasks
- Online evaluation: 10 paired episodes per task and condition, 100 episodes
  per condition, 800 rollouts total
- Seed: 42, with identical initial-state ordering across conditions
- Action horizon: 32
- Inference steps: 10
- Replan interval: 10 environment steps
- Offline state bank: 500 shared baseline states, comprising 10 tasks x 10
  episodes x the first 5 policy queries (environment steps 30, 40, 50, 60,
  and 70)
- Every offline condition replayed exactly the same 500 model-ready inputs and
  action-noise seed.

Correctness checks passed:

- Empty intervention versus the original inference path: max and mean action
  difference were both exactly 0.
- Disabling middle layer 15 changed the action, with max difference 0.0546875.
- Disabling all layers returned a finite action tensor of shape [32, 7].
- All online and offline processes exited successfully. No OOM, EGL error,
  traceback, NaN, or Inf was found.

## Complete results

`Offline normalized action L2` below is actually RMS deviation over the six
continuous normalized action dimensions and the 32-step action horizon.
Gripper flip is measured after action post-processing and binarization.

| Condition | Offline normalized action RMS | Action cosine | Gripper flip | Online success | Delta vs baseline | Paired bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.000003 | 1.000000 | 0.000% | 97% | 0 pp | [0, 0] pp |
| drop_00_04 | 0.018932 | 0.999084 | 0.119% | 97% | 0 pp | [-4, +4] pp |
| drop_05_09 | 0.008223 | 0.999704 | 0.100% | 95% | -2 pp | [-6, +2] pp |
| drop_10_14 | 0.030407 | 0.997210 | 0.806% | 97% | 0 pp | [-4, +4] pp |
| drop_15_19 | 0.066418 | 0.986178 | 1.813% | 70% | **-27 pp** | **[-36, -18] pp** |
| drop_20_24 | 0.058808 | 0.988067 | 0.506% | 89% | -8 pp | [-15, -2] pp |
| drop_25_29 | 0.121510 | 0.969660 | 1.994% | 67% | **-30 pp** | **[-40, -20] pp** |
| drop_all | 0.339453 | 0.262351 | 59.644% | 0% | **-97 pp** | **[-100, -93] pp** |

Paired baseline-to-ablation outcome transitions were:

- drop_00_04: 2 successes became failures and 2 failures became successes
- drop_05_09: 3 successes became failures and 1 failure became a success
- drop_10_14: 2 successes became failures and 2 failures became successes
- drop_15_19: 29 successes became failures and 2 failures became successes
- drop_20_24: 10 successes became failures and 2 failures became successes
- drop_25_29: 32 successes became failures and 2 failures became successes
- drop_all: all 97 baseline successes became failures

Exact paired McNemar tests gave p=4.63e-7 for drop_15_19, p=6.94e-8 for
drop_25_29, and p=1.26e-29 for drop_all. Drop_20_24 had an uncorrected
p=0.0386 but was not significant after correcting for the multiple layer-group
comparisons. A task-hierarchical bootstrap likewise gave approximate 95% CIs
of [-42, -13] pp for drop_15_19, [-20, +3] pp for drop_20_24, and [-57, -5]
pp for drop_25_29.

## Offline-to-online relationship

Across only the six equal-width five-layer ablations, excluding baseline and
the high-leverage drop_all positive control:

- Offline normalized action RMS versus online success degradation:
  - Pearson r=0.887, p=0.0183
  - Spearman rho=0.812, p=0.0499
- One minus action cosine versus online success degradation:
  - Pearson r=0.895, p=0.0161
  - Spearman rho=0.812, p=0.0499

The p-values above are the usual parametric correlation-test values. Because
there are only six layer-group points, exact two-sided permutation checks are
also informative: for normalized action RMS they give approximately p=0.0083
for Pearson and p=0.072 for Spearman. The association should therefore be
described as strong but exploratory, rather than as a precisely estimated or
fully robust population-level law.

Including drop_all increases the L2 Pearson correlation to 0.986, but this is
strongly driven by the extreme drop_all point and should not be the primary
statistic.

Thus, fixed-state action sensitivity is strongly associated with aggregate
closed-loop behavioral degradation at the layer-group level, but the
relationship is not one-to-one. In particular, drop_15_19 and drop_20_24 have
similar average offline deviations (0.0664 versus 0.0588) but very different
online losses (-27 versus -8 percentage points).

The same offline magnitude is not a reliable task-level failure predictor.
For example, under drop_25_29, task 0 has a relatively large offline deviation
but remains 10/10 successful, whereas task 7 has a smaller offline deviation
but falls from 10/10 to 0/10. Task geometry, tolerance, and closed-loop error
accumulation therefore matter in addition to immediate action deviation.

## Task-level heterogeneity

- drop_15_19 causes broad damage:
  - task 9: -60 pp
  - task 7: -50 pp
  - task 4: -40 pp
  - tasks 1, 2, and 5: -30 pp each
- drop_20_24 is more selective:
  - task 1: -40 pp
  - task 8: -30 pp
  - task 4: -20 pp
- drop_25_29 is strongly task selective:
  - task 7 (bowl on stove): 10/10 to 0/10
  - task 9 (bowl on wooden cabinet): 10/10 to 0/10
  - task 4: -50 pp
  - task 6: -40 pp
  - task 3: -30 pp
  - tasks 0, 1, 2, and 5 remain 10/10

Each task has only 10 episodes, so these task-level patterns are exploratory.
Small apparent improvements should not be interpreted as benefits.

## Temporal observations

Mean normalized action RMS by baseline replan index was:

- drop_15_19: 0.0869, 0.0662, 0.0607, 0.0590, 0.0593
- drop_20_24: 0.0732, 0.0555, 0.0456, 0.0492, 0.0705
- drop_25_29: 0.1483, 0.1180, 0.1022, 0.1081, 0.1309

The key groups therefore perturb the policy from the first saved query rather
than only at late rollout stages. Within each predicted 32-action chunk,
drop_15_19 remains broadly stable across the horizon, while drop_20_24 and
drop_25_29 have their largest executed-action deviations at the first one or
two horizon positions.

## Main interpretation

The evidence supports heterogeneous and non-monotonic causal importance of
the layer-wise video K/V pathways consumed by the action stream. Layers 15-19
and 25-29 are the strongest critical regions, while 20-24 is a weaker and more
task-specific region. Removing K/V access in layers 0-14 produces measurable
action changes but no detectable aggregate success loss, consistent with
redundancy, compensation by later layers, or task tolerance. Removing all
video K/V reduces success to zero, showing that the overall video-to-action
pathway is indispensable.

The result should not be stated as "layers 0-14 are useless." The intervention
only removes direct video K/V access at selected action layers. Visual
information may already be carried in the residual action representation or
retrieved again by later layers. Removing K/V also changes the available key
set and attention softmax normalization.

This is a diagnosis result, not yet evidence that layers can be deleted or
that the model can be compressed without retraining.

## Reproducibility caveat and scope limits

- Baseline replay is exact for 499/500 samples. The first sample, retained from
  an interrupted pre-resume collection run, has a tiny normalized RMS of
  0.001625; the aggregate baseline mean is only 3.25e-6. Excluding it makes the
  baseline exactly zero and does not change any ranking or conclusion.
- The state bank covers the first five policy queries on the baseline
  visitation distribution, not states visited by the ablated policies.
- The five states within an episode are correlated; offline uncertainty should
  be clustered by episode, and generalization across tasks should use a task-
  hierarchical analysis.
- Results are from one checkpoint, one seed, and LIBERO-Spatial only.
- Five-layer group ablations cannot localize importance to an individual
  layer.
- Checkpoint and manifest SHA256 hashes were not recorded, although paths,
  configuration, Git commit, and metadata match across runs.

## Questions for discussion

1. What is the strongest defensible paper-level claim from these results?
2. How should we distinguish fresh visual retrieval in late action layers from
   redundant visual information already carried in the action residual stream?
3. Which follow-up is most informative: single-layer ablations around layers
   15-19 and 25-29, cumulative prefix/suffix ablations, or controlled K/V
   replacement interventions?
4. Which additional analyses should be produced from the existing per-sample
   data (episode-clustered confidence intervals, replan curves, horizon curves,
   task-conditioned metrics, or failure-transition analysis)?
5. How should the main figures be redesigned so that drop_all does not dominate
   the scale and paired uncertainty is visible?

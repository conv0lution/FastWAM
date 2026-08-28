# Fast-WAM ASRE Round 3B

Round 3B is the matched-shape video K/V replacement control. It asks whether
the late action stream needs the correct scene-specific visual representation,
or whether the structural presence of realistic video K/V tokens is enough.

The frozen three conditions are:

| Condition | Layers 0-14 | Layers 15-29 |
|---|---|---|
| `late_current_correct` | disabled | current-scene K/V |
| `late_wrong_scene` | disabled | fixed same-task/different-trial donor K/V |
| `late_no_video` | disabled | disabled |

Online donor trial `i` is `(i + 1) mod 10` within the same task. The donor
image is captured at the first policy query after 30 dummy steps. Because
video prefill also consumes the appended proprioceptive context, the fixed
donor image is reused but donor K/V is recomputed at every recipient replan
with current recipient text/proprioception. This is a wrong fixed-scene
control, not a temporal-staleness experiment.

The end-to-end command is:

```bash
cd /home/zhaizicheng/Reasoning/FastWAM
mkdir -p asre_results/round3b/logs
set -o pipefail
./experiments/asre_diagnosis/round3b/run_full_experiment.sh \
  2>&1 | tee asre_results/round3b/logs/driver.log
```

The default physical GPU assignment is `0,1,2`. When another isolated
three-GPU set is available, record it explicitly without changing condition
order, for example:

```bash
ROUND3B_GPU_IDS=4,5,6 \
ROUND3B_LAUNCH_STAGGER_SECONDS=70 \
./experiments/asre_diagnosis/round3b/run_full_experiment.sh \
  2>&1 | tee asre_results/round3b/logs/driver.log
```

The optional stagger only separates host-side model construction to reduce
peak RAM pressure. The three conditions still use isolated GPUs and overlap
during the substantive rollout.

The driver validates the frozen Round-3A parent, captures and freezes donor
observations, runs the same-input numerical identity gate, replays the exact
499-state offline bank, runs a two-trial task-0 smoke on three isolated GPUs,
then runs 100 paired episodes per condition. It stops after aggregation and
the five pre-specified figures.

Raw donor tensors, action traces, offline actions/cache audits, rollout
outputs, and logs remain ignored. Only the compact aggregate tables,
summaries, provenance, and figures under `asre_results/round3b/aggregate/`
are intended for a result commit.

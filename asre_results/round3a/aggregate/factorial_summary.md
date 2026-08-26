# ASRE Round 3A: complete late-half 2^3 factorial

This summary combines five immutable Round-2 cells with the three new
Round-3A cells. Effects are on the success-probability scale; percentage-
point values are probability effects multiplied by 100.

## Observed

| Cell | A | B | C | Condition | Success | Executed-prefix RMS |
|---|---:|---:|---:|---|---:|---:|
| 000 | 0 | 0 | 0 | `keep_none_late` | 0.0% | 0.9641 |
| 001 | 0 | 0 | 1 | `keep_25_29` | 0.0% | 0.6763 |
| 010 | 0 | 1 | 0 | `keep_20_24` | 0.0% | 0.7488 |
| 011 | 0 | 1 | 1 | `keep_20_29` | 49.0% | 0.3245 |
| 100 | 1 | 0 | 0 | `keep_15_19` | 8.0% | 0.7239 |
| 101 | 1 | 0 | 1 | `keep_15_19_25_29` | 86.0% | 0.3279 |
| 110 | 1 | 1 | 0 | `keep_15_24` | 61.0% | 0.5212 |
| 111 | 1 | 1 | 1 | `keep_15_29` | 95.0% | 0.1367 |

### Contextual effects

- `simple_A_B0_C0`: +8.0 pp; paired 95% CI [+3.0, +14.0] pp; task-hierarchical [+0.0, +20.0] pp.
- `simple_A_B1_C0`: +61.0 pp; paired 95% CI [+51.0, +70.0] pp; task-hierarchical [+35.0, +85.0] pp.
- `simple_A_B0_C1`: +86.0 pp; paired 95% CI [+79.0, +92.0] pp; task-hierarchical [+75.0, +95.0] pp.
- `simple_A_B1_C1`: +46.0 pp; paired 95% CI [+36.0, +56.0] pp; task-hierarchical [+28.0, +64.0] pp.
- `simple_B_A0_C0`: +0.0 pp; paired 95% CI [+0.0, +0.0] pp; task-hierarchical [+0.0, +0.0] pp.
- `simple_B_A1_C0`: +53.0 pp; paired 95% CI [+43.0, +63.0] pp; task-hierarchical [+29.0, +77.0] pp.
- `simple_B_A0_C1`: +49.0 pp; paired 95% CI [+39.0, +59.0] pp; task-hierarchical [+30.0, +69.0] pp.
- `simple_B_A1_C1`: +9.0 pp; paired 95% CI [+2.0, +17.0] pp; task-hierarchical [-1.0, +20.0] pp.
- `simple_C_A0_B0`: +0.0 pp; paired 95% CI [+0.0, +0.0] pp; task-hierarchical [+0.0, +0.0] pp.
- `simple_C_A1_B0`: +78.0 pp; paired 95% CI [+68.0, +87.0] pp; task-hierarchical [+62.0, +91.0] pp.
- `simple_C_A0_B1`: +49.0 pp; paired 95% CI [+39.0, +59.0] pp; task-hierarchical [+30.0, +69.0] pp.
- `simple_C_A1_B1`: +34.0 pp; paired 95% CI [+23.0, +45.0] pp; task-hierarchical [+7.0, +62.0] pp.

### Seven factorial contrasts

- `main_A`: +50.2 pp; paired 95% CI [+46.2, +54.2] pp; task-hierarchical [+41.8, +59.0] pp; raw sign-flip p=9.9999e-06, Holm p=6.99993e-05.
- `main_B`: +27.7 pp; paired 95% CI [+23.8, +31.8] pp; task-hierarchical [+19.2, +36.5] pp; raw sign-flip p=9.9999e-06, Holm p=6.99993e-05.
- `main_C`: +40.2 pp; paired 95% CI [+35.5, +45.0] pp; task-hierarchical [+29.5, +49.3] pp; raw sign-flip p=9.9999e-06, Holm p=6.99993e-05.
- `interaction_AB`: +6.5 pp; paired 95% CI [-1.0, +14.0] pp; task-hierarchical [-5.5, +19.5] pp; raw sign-flip p=0.125699, Holm p=0.251397.
- `interaction_AC`: +31.5 pp; paired 95% CI [+23.0, +40.0] pp; task-hierarchical [+13.5, +50.5] pp; raw sign-flip p=9.9999e-06, Holm p=6.99993e-05.
- `interaction_BC`: +2.5 pp; paired 95% CI [-5.0, +10.0] pp; task-hierarchical [-13.0, +17.5] pp; raw sign-flip p=0.597004, Holm p=0.597004.
- `interaction_ABC`: -93.0 pp; paired 95% CI [-111.0, -75.0] pp; task-hierarchical [-132.0, -53.0] pp; raw sign-flip p=9.9999e-06, Holm p=6.99993e-05.

### Pre-specified scientific questions

- **Q1 Does B alone perform above no retrieval?** Observed effect +0.0 pp; uncertain; at least one pointwise 95% CI includes zero.
- **Q2 Does adding B to A help when C is absent?** Observed effect +53.0 pp; positive; both pointwise 95% CIs exclude zero.
- **Q3 Does adding B to C help when A is absent?** Observed effect +49.0 pp; positive; both pointwise 95% CIs exclude zero.
- **Q4 Does adding B to A+C help?** Observed effect +9.0 pp; uncertain; at least one pointwise 95% CI includes zero.
- **Q5 Is B strongly context-dependent?** B simple effects range from +0.0 to +53.0 pp. At least one B-related interaction has both pointwise CIs excluding zero.
- **Q6 Is averaged A+C complementarity visible?** A×C=+31.5 pp; positive; both pointwise 95% CIs exclude zero. This is an averaged factorial interaction, not an established semantic mechanism.
- **Q7 Is there evidence for an A×B×C interaction?** A×B×C=-93.0 pp; negative; both pointwise 95% CIs exclude zero; Holm p=6.99993e-05.

## Supported

Only directions for which both the paired and task-hierarchical pointwise
95% intervals exclude zero are described above as supported. Holm correction
applies only to the seven main/pairwise/three-way factorial contrasts; the
12 McNemar simple-effect p-values are descriptive and unadjusted.

The interaction contrasts summarize experimental dependence on retained
retrieval context. They are not a complete causal model of neural computation.

## Hypothesis

Terms such as bridging, grounding, refinement, or staged retrieval remain
functional hypotheses. Removing K/V also changes the attention key set and
softmax normalization. Matched-shape K/V replacement or residual patching would
be required to separate visual-content dependence from structural attention
effects, and those experiments are outside the Round-3A stop rule.

## Statistical scope

- Paired episodes: 100
- Tasks: 10
- Joint bootstrap replicates: 10000
- Bootstrap intervals are pointwise, not simultaneous.
- Factorial p-values use a Monte-Carlo paired sign-flip test whose
  interpretation depends on sign-exchangeability of episode-level contrast
  scores; this is not claimed to be a design-exact randomized-treatment test.
- Results remain one checkpoint, one seed, and LIBERO-Spatial only.

# Fast-WAM ASRE Salvage B

Salvage B is the final, pre-registered world-vs-action functional
dissociation gate. It does not search for another compact subspace and it
never launches a later ASRE stage.

The implementation uses Preferred Path A. The shared causal interface is the
same first-frame video K/V cache used by the frozen Round-4C action
intervention, with layers 0–14 disabled and layers 15–29 supplied from one of
exactly four conditions:

- `current_all`
- `wrong_all`
- `svd_r97`
- `svd_r170` (primary)

The world consumer is a future-only factorization of the stock
`first_frame_causal` video computation. It accepts no raw first-frame token or
residual. Each draw starts the two future latent frames from pure Gaussian
noise, follows the frozen native 10-step video scheduler, and scores the final
future latent against the causal-VAE target. The real future target never
enters inference. A real-checkpoint machinery gate must show that the
factorized all-current, no-disabled denoising step agrees with the stock joint
future output before any world metric is evaluated.

## Execution order

`run_full_experiment.sh` performs the following fail-closed sequence:

1. static architecture and native-objective audit;
2. frozen provenance/action-reuse preflight;
3. CPU tests;
4. freeze and hash 100 task-stratified official LIBERO-Spatial clips;
5. freeze exactly four future-latent noise draws per clip, one native inference
   schedule, and hashes of the decoded scoring targets;
6. run the real-checkpoint shared-interface and native-metric machinery gate;
7. evaluate only `current_all` and `wrong_all` on four independent single-GPU
   workers;
8. stop as `WORLD-ENDPOINT-UNINFORMATIVE` unless mean
   `loss(wrong)-loss(current) > 0` and its paired-bootstrap 95% CI has a
   strictly positive lower endpoint;
9. only after that gate, evaluate `svd_r97` and `svd_r170`;
10. aggregate action/world recovery, functional dissociation, uncertainty,
    figures, and the final classification; then stop unconditionally.

No DDP is used. Frozen Round-4C action episodes are verified and reused; they
are not rerun. The GPU work is limited to the shared-interface machinery check
and native world-loss evaluation.

The stock joint call and the cache-factorized call use mathematically
equivalent causal masks but different fully masked SDPA sequence extents. Their
real-checkpoint equivalence gate therefore uses
`max(0.2%, 2 * execution-dtype epsilon)` relative RMSE, while still requiring
exact future inputs, identical shapes, finite values, the same cache objects,
intervention reach to both consumers, and no raw-prefix bypass. Pointwise
`allclose` is recorded descriptively because it is not stable across BF16 SDPA
partition shapes.

## Run

The formal preflight requires all source changes outside the selected output
root to be committed. From the repository root:

```bash
SALVAGE_B_GPU_IDS="4 5 6 7" \
SALVAGE_B_OUTPUT_ROOT="$PWD/asre_results/salvage_b_world_action_dissociation" \
SALVAGE_B_LAUNCH_STAGGER_SECONDS=70 \
./experiments/asre_diagnosis/salvage_b/run_full_experiment.sh
```

The driver safely reuses compatible completed artifacts in the same output
root. It refuses incompatible resumes and refuses to overwrite frozen
manifests. Use a fresh child directory under
`asre_results/salvage_b_world_action_dissociation/` after changing source or
inputs.

The final human/GPT handoff is generated automatically at:

```text
<output-root>/aggregate/result_summary_for_gpt.md
```

No manual model-assisted analysis is needed to create that report.

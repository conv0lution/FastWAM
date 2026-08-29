# ASRE Stage 2 — Round 4C

Round 4C evaluates energy-controlled action sufficiency on LIBERO-Spatial using
the exact frozen Round-4B SVD basis prefixes at ranks 36, 97, and 170. It reruns
same-round current/wrong endpoints, uses four GPUs in two waves, and stops after
aggregation and plotting.

```bash
ROUND4C_GPU_IDS="4 5 6 7" \
ROUND4C_OUTPUT_ROOT="$PWD/asre_results/round4c_energy_sufficiency" \
ROUND4C_LAUNCH_STAGGER_SECONDS=70 \
./experiments/asre_diagnosis/round4c/run_full_experiment.sh
```

The launcher is fail-closed and resume-safe. Raw results remain under
`asre_results/` and are not committed.

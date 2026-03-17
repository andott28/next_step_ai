# SCA Root-Cause Diagnosis Report (2026-03-13)

## Scope
This run focused on isolating whether the strict sparse quality collapse is mainly due to:
- sparse execution backend (`dense` vs `cuda_spmm`),
- specific layer bands being sparsified,
- or cumulative dense-vs-sparse hidden/logit drift during autoregressive decode.

Raw artifact:
- `results/sca_root_cause_diagnosis_v1/sca_diagnosis_results.json`

## Experiments Run
Conditions (2 prompts, greedy decode, short continuation):
- `dense_baseline_np_off` (neuroplasticity disabled)
- `sca_dense_all_layers` (SCA on, `spmm_impl=dense`)
- `sca_cuda_all_layers` (SCA on, `spmm_impl=cuda_spmm`)
- `sca_cuda_mid_8_23` (SCA on only layers 8-23, `cuda_spmm`)
- `sca_cuda_late_24_29` (SCA on only layers 24-29, `cuda_spmm`)

Drift probe:
- Same prefix expanded for 6 autoregressive steps.
- At each step, compare dense pass vs sparse pass on the same token prefix:
  - mean layer cosine similarity,
  - final layer cosine similarity,
  - KL between dense and sparse next-token distributions.
- Also tracked route overlap on layers 14-17 across steps.

## Key Findings

### 1) Failure exists even when sparse execution backend is `dense`
`sca_dense_all_layers` already degrades output:
- `max_run_mean=4.0`, `rep_frac_mean=0.2308`

So repetition is not primarily a `cuda_spmm`-only implementation bug.

### 2) `cuda_spmm` worsens degradation slightly, but does not create it from zero
`sca_cuda_all_layers` vs `sca_dense_all_layers`:
- `max_run_mean`: `4.5` vs `4.0`
- `rep_frac_mean`: `0.2692` vs `0.2308`

Backend differences matter, but they are secondary to the core SCA behavior.

### 3) Mid-band sparsification is the most damaging in this probe
`sca_cuda_mid_8_23` is worst:
- `max_run_mean=5.0`
- `rep_frac_mean=0.3419`

Late-band only (`24-29`) is less destructive:
- `max_run_mean=2.5`
- `rep_frac_mean=0.1667`

This indicates the central/mid computational band is high-leverage for collapse risk.

### 4) No dense fallback masking was detected
All tested SCA conditions had:
- `max_fallback_triggered=0.0`

So these failures are real sparse-path behavior, not hidden fallback artifacts.

### 5) Dense-vs-sparse geometry divergence is severe from the first generated steps
Drift probe:
- mean layer cosine stays very low (`~0.045` to `~0.068`)
- final layer cosine near zero / slightly negative at some steps
- logits KL remains high (`~5.3` to `~8.6`)

Interpretation: sparse decode states are in a meaningfully different geometry than dense states almost immediately.

### 6) Routing is relatively stable while collapse still occurs
Route overlap in layers 14-17 was often high (`jaccard_prev` frequently `1.0`, occasional `0.5`).
Generated next token repeated strongly (`next_token_id` remained constant across steps in this probe).

Interpretation: collapse can happen even with mostly stable routing, so unstable routing alone is not the primary cause.

## Root-Cause Ranking (from this run)
1. **Primary:** SCA all-layer manifold distortion causes immediate dense-vs-sparse decode drift.
2. **Secondary:** `cuda_spmm` adds some additional degradation, but does not explain the full issue.
3. **Layer sensitivity:** Mid-band sparsification contributes most strongly to collapse in this setup.
4. **Not primary:** Hidden dense fallback or gross route instability.

## Practical Conclusion
The main fix should target **SCA behavior and layer strategy**, not only kernel/backend tweaks.

Most promising next direction:
- constrain or redesign SCA in the mid band first,
- add decode-stability objectives that penalize multi-step dense-vs-sparse drift,
- keep strict sparse execution, but reduce manifold damage before the final decode stack.

## Caveats
- This was a short-prompt, short-horizon diagnosis, not a full benchmark.
- Metrics are still strong enough to support directional conclusions because the failure signal is large.

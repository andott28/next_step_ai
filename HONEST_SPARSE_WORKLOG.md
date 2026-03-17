# Honest Sparse SCA Worklog (Current Session)

## Scope of this log
This document summarizes what was implemented, tested, and diagnosed in the recent SCA/hybrid strict-sparse debugging cycle.

The focus was:
- remove hidden dense fallback behavior,
- align training with the runtime that matters,
- isolate strict runtime failures,
- add runtime honesty checks and diagnostics.

---

## 1) Starting diagnosis (before latest code changes)

### Main observed symptom
- Short strict prompts often collapsed into repetitive subword loops (examples like `...olateolate...`).

### Key evidence gathered
- Short strict runs showed `mean_selected_pages_per_step = 0.0`, so long-range sparse-attention retrieval was not active for those probes.
- Dense hybrid baseline (without sparse MLP path) remained coherent on the same prompts.
- Sparse path behavior depended heavily on runtime mode (strict/non-strict/cache settings).

### Early root-cause direction
- The failure was in decode/runtime interaction with sparse path, not primarily in long-range retrieval.

---

## 2) Runtime fixes completed before this user request

### A) Task-bias mismatch fix
- Inference now respects the `disable_task_bias_injection` value stored in SCA recalibration artifact metadata.
- This removed a calibration/runtime mismatch where recalibration was run with task bias disabled but inference could re-enable it.

### B) Strict repetition penalty default fix
- Strict decode repetition penalty was made opt-in rather than always-on in strict mode.
- This removed a strong strict-only logit perturbation that could destabilize sparse decode.

### C) Extension/tooling path simplification
- Active runtime path no longer depends on the custom C++ extension path in the hybrid loader.

### D) Verification
- Targeted tests passed (`34 passed`) after these changes.

---

## 3) User-requested ordered tasks completed in this cycle

User asked to do these in order:
1. Add CLI override to run strict settings with fallback disabled but `spmm_impl=dense`.
2. Recalibrate on actual active strict layer band (not stale/later-only band).
3. Surface fallback explicitly in inference diagnostics.

All three were implemented and executed.

---

## 4) Code changes implemented in this cycle

## 4.1 CLI override for strict + dense backend diagnostics

### File
- `llama3_neuroplastic/run_hybrid_gqa_mamba_inference.py`

### Added flags
- `--sca-spmm-impl {dense,cuda_spmm,torch_block_sparse}`
- `--sca-fallback-threshold`
- `--allow-strict-noncuda-spmm`

### Supporting runtime change
- `llama3_neuroplastic/bounded_context.py`
  - `validate_sparse_attention_runtime(...)` now accepts a diagnostic override argument to allow non-`cuda_spmm` backend under strict checks for isolation experiments.

### Supporting model config wiring
- `llama3_neuroplastic/neuroplastic_llama_gqa_mamba.py`
  - added `strict_runtime_allow_noncuda_spmm_diagnostic` config field and threaded it into strict runtime validation call.

---

## 4.2 Explicit fallback diagnostics surfaced

### Files
- `llama3_neuroplastic/sca_sparse_mlp.py`
- `llama3_neuroplastic/neuroplastic_llama_gqa_mamba.py`
- `llama3_neuroplastic/run_hybrid_gqa_mamba_inference.py`

### New diagnostics data
Per-layer:
- `dense_fallback_rate`
- `triggered_steps`
- `total_steps`
- `last_fallback_triggered`

Aggregate:
- `mean_dense_fallback_rate`

### Important correction
- Decode-guard/buffer-disabled layers are no longer counted as fallback events. They are treated as intentional dense path for disabled layers, not “fallback.”

---

## 4.3 Honest sparse default behavior + adaptive routing capacity

### Fallback default set to zero
- `stability_dense_fallback_threshold` default changed to `0.0`.
- Files:
  - `llama3_neuroplastic/sca_sparse_config.py`
  - `llama3_neuroplastic/neuroplastic_llama_gqa_mamba.py`
  - inference CLI default logic in `run_hybrid_gqa_mamba_inference.py`

### Runtime honesty gate
- Inference now has `--require-honest-sparse` (default true).
- If sparse MLP enabled and fallback rate > 0, run fails explicitly.

### Adaptive top-k routing (allow sparse path to use more active blocks as needed)
- Added to config:
  - `adaptive_top_k`
  - `adaptive_top_k_min`
  - `adaptive_top_k_max`
  - `adaptive_top_k_min_score_ratio`
  - computed `route_top_k` property
- Implemented in:
  - `llama3_neuroplastic/sca_sparse_config.py`
  - `llama3_neuroplastic/sca_sparse_adapter.py`
  - routing now computes up to `route_top_k` and prunes per-row by relative score threshold.
  - `llama3_neuroplastic/sca_sparse_mlp.py` updated to use `route_top_k` where needed.

### Inference CLI support
- Added:
  - `--sca-adaptive-top-k`
  - `--sca-adaptive-top-k-min`
  - `--sca-adaptive-top-k-max`
  - `--sca-adaptive-top-k-min-score-ratio`

---

## 4.4 Recalibration runner alignment to runtime + failure mode anchoring

### File
- `llama3_neuroplastic/run_sca_recalibration_from_hybrid_baseline.py`

### Changes
- Added adaptive top-k CLI/config controls (same fields as runtime).
- Added CE anchoring term (`--ce-weight`) to better anchor language decode behavior.
- Strengthened rollout defaults to start earlier and run more often.
- Added fallback mean metric in saved recalibration metrics (`mean_dense_fallback_rate`).
- Layer selection default now uses active sparse layer band (based on model sparse-layer enable function), not hybrid artifact layer selection.
- Speech-anchor default now uses the tail of selected training layers, not fixed upper model tail.
- Rollout KL handling made robust against non-finite values (skip non-finite rollout term instead of hard crash).

---

## 5) Tests and commands run

### Repeated unit test check
- `pytest tests/test_hybrid_gqa_mamba.py tests/test_sca_sparse_mlp.py -q`
- Result: `34 passed` (multiple times after changes).

### Key isolation runs
- strict + `cuda_spmm` + fallback disabled.
- strict + `dense` backend override + fallback disabled.
- non-strict + fallback disabled.
- non-strict default (before fallback removal) with new fallback diagnostics.

### Key finding from fallback diagnostics
- In non-strict default mode, active sparse layers were often `dense_fallback_rate = 1.0`, proving “good” non-strict text was frequently fallback-assisted rather than true sparse execution.

---

## 6) Recalibration runs performed in this cycle

## 6.1 Active strict-band recalibration smoke
- Output:
  - `results/sca_recalibration_strict_active_band_smoke/sca_recalibrated_state.pt`
  - `results/sca_recalibration_strict_active_band_smoke/sca_recalibration_metrics.json`

### Notes
- Fuller run initially hit OOM with larger setup.
- Reduced smoke run completed.

## 6.2 Honest sparse methodology smoke
- Output:
  - `results/sca_recalibration_honest_sparse_smoke/sca_recalibrated_state.pt`
  - `results/sca_recalibration_honest_sparse_smoke/sca_recalibration_metrics.json`

## 6.3 Honest sparse methodology smoke (auto layer selection validated)
- Output:
  - `results/sca_recalibration_honest_sparse_smoke2/sca_recalibrated_state.pt`
  - `results/sca_recalibration_honest_sparse_smoke2/sca_recalibration_metrics.json`

### Validation result
- Layer selection correctly resolved to active strict band `0..17`.
- Speech anchor layers resolved to `12..17`.
- Fallback rate stayed `0.0` (honest sparse path).

---

## 7) Current diagnosis (latest)

1. **Main failure is true sparse-path decode instability**, not long-range retrieval:
   - strict short runs still show no long-range page selection in these probes.

2. **`cuda_spmm` is not the sole culprit**:
   - strict runs with dense backend override also collapse similarly when fallback is disabled.

3. **Fallback previously masked quality**:
   - non-strict “good” behavior often depended on dense fallback.

4. **Training methodology is now closer to correct target**:
   - runtime parity enabled,
   - active strict layer band selection default fixed,
   - fallback default removed,
   - explicit honesty checks and metrics added.

5. **Quality is still not solved yet**:
   - honest sparse smoke runs still produce unstable text patterns under strict generation,
   - indicating more training/optimization is needed under the corrected methodology.

---

## 8) Artifacts most relevant right now

- Strict runtime diagnostics (active-band checkpoint):
  - `results/strict_cuda_spmm_diag_activeband.json`
  - `results/strict_dense_spmm_diag_activeband.json`

- Non-strict fallback exposure:
  - `results/nonstrict_default_fallback_diag.json`

- Honest sparse smoke recalibration:
  - `results/sca_recalibration_honest_sparse_smoke2/sca_recalibration_metrics.json`
  - `results/sca_recalibration_honest_sparse_smoke2/sca_recalibrated_state.pt`

---

## 9) Practical status

- The stack is now instrumented to prevent self-deception from fallback-assisted quality.
- Runtime and training defaults are aligned with “true sparse” expectations.
- Remaining work is model-quality stabilization under honest sparse decode, not hidden fallback tuning.

---

## 10) New combined fix pass (implemented together)

Date: 2026-03-16

This pass implemented the full combined methodology requested for a robust, scalable fix:

1. Stagewise layer-band training (lower active band first, then full band)
2. Per-layer residual norm control in local sparse geometry loss
3. Logit-centric objective under strict runtime parity (with rollout refinement)
4. Bounded adaptive sparsity retained (no dense fallback rescue)

### 10.1 Code changes

- Modified:
  - `llama3_neuroplastic/run_sca_recalibration_from_hybrid_baseline.py`
  - `llama3_neuroplastic/neuroplastic_llama_gqa_mamba.py`

### 10.2 Recalibration runner updates

Added a new default mode:
- `recalibration_mode = strict_sparse_stability`

Added stagewise training support:
- `--staged-training-enabled`
- `--stage1-steps-ratio`
- `--stage-split-ratio`
- `--stage2-lr-scale`

Added explicit stability controls:
- `--delta-norm-cap`
- `--delta-norm-cap-weight`
- `--lower-layer-loss-weight`
- `--upper-layer-loss-weight`
- `--upper-layer-scale-cap`
- `--output-scale-cap-tail-fraction`

Behavior changes in training loop:
- Builds a stage plan from selected layers.
- Reconfigures trainable layer indices per stage.
- Recreates optimizer with stage-scaled LR.
- Uses weighted local loss in strict stability mode:
  - local geometry (weighted)
  - logits KL
  - CE
  - rollout KL (scheduled)
- Masks `sca_layer_output_scale` gradients outside currently active stage.
- Clamps layer output scales after each optimizer step (tighter cap on upper tail layers).
- Logs stage name and local loss each progress line.
- Stores `stage_plan` in metrics artifacts.

### 10.3 Local loss helper updates (model)

`compute_sca_local_recalibration_loss(...)` now accepts:
- `layer_weight_map`
- `delta_norm_cap`
- `delta_norm_cap_weight`

New computations:
- per-layer `delta_norm_ratio = ||sparse-dense||^2 / ||dense||^2`
- cap penalty `relu(delta_norm_ratio - delta_norm_cap)^2`
- optional per-layer weighting before aggregation

New metrics returned:
- `loss_delta_norm_ratio`
- `loss_delta_norm_cap`
- `per_layer_delta_norm_ratio`

### 10.4 Defaults tuned toward strict stability

Updated runner defaults for the new mode:
- stronger early logits alignment:
  - `logits_kl_weight=0.08`
  - `logits_warmup_steps=2`
- CE emphasis increased:
  - `ce_weight=0.35`
- local geometry reduced in strict mode:
  - `local_mlp_weight=0.1`
- rollout made less aggressive and later:
  - `rollout_start_step=8`
  - `rollout_every=4`
  - `rollout_max_new_tokens=2`
  - `rollout_kl_weight=0.05`

### 10.5 Validation run after patch

Executed:
- `pytest tests/test_hybrid_gqa_mamba.py tests/test_sca_sparse_mlp.py -q`

Result:
- `34 passed`

Also verified runner CLI parsing with:
- `python -m llama3_neuroplastic.run_sca_recalibration_from_hybrid_baseline --help`

### 10.6 Why this pass matters

This pass removes the previous mismatch where training could look successful but fail under strict sparse decode. The methodology now directly targets:
- honest sparse runtime behavior,
- upper-layer instability control,
- autoregressive next-token stability under self-generated prefixes,
- scalable sparse adaptation via staged layer activation instead of fixed hardcoded exemptions.

---

## 11) Top-k ablation and sparse-MLP formulation diagnosis

Date: 2026-03-16

This pass revisited the old `top_k` assumption directly under honest strict sparse runtime, using fast targeted probes instead of long full training loops.

### 11.1 What was tested

- Strict sparse MLP inference with:
  - fallback `= 0`
  - sparse attention disabled for short-prompt diagnosis
  - exact fixed `top_k` values
  - no adaptive routing
- Candidate values tested:
  - `3`
  - `4`
  - `6`
  - `8`
  - `12`
  - `16`

Artifacts:
- `results/topk_probe_3.json`
- `results/topk_probe_4.json`
- `results/topk_probe_6.json`
- `results/topk_probe_8.json`
- `results/topk_probe_12.json`
- `results/topk_probe_16.json`

### 11.2 Outcome of the top-k sweep

There was **no acceptable `top_k` in the tested range**.

Observed strict outputs:
- `k=3`: loops like `321321...` and `olateolate...`
- `k=4`: same failure shape
- `k=6`: same failure shape
- `k=8`: same failure shape
- `k=12`: still broken, different junk pattern
- `k=16`: still broken, different junk pattern

Measured sparse cost increased as expected:
- `k=3`: touched fraction about `0.0132`
- `k=4`: about `0.0176`
- `k=6`: about `0.0264`
- `k=8`: about `0.0352`
- `k=12`: about `0.0527`
- `k=16`: about `0.0703`

But semantic quality did **not** recover.

### 11.3 Direct MLP norm-ratio check across k

One-prompt sparse-vs-dense MLP norm ratio:
- `k=3`: `0.00233`
- `k=4`: `0.00345`
- `k=8`: `0.00869`
- `k=16`: `0.02472`

Interpretation:
- Larger `k` improves sparse MLP output norm somewhat.
- Even at `k=16`, sparse MLP output norm is only about `2.5%` of dense MLP output norm.
- This is nowhere near enough to preserve decode geometry.

### 11.4 Stronger root-cause diagnosis inside the sparse MLP

The main issue is now considered **structural**, not just a bad `k` choice.

Direct wrapper analysis showed:
- mean active feature fraction around `3/128 = 0.0234375` in the original setting
- about `97.7%` of dense MLP output energy lies outside the selected block mask
- removing output masking helps only modestly
- bank manifest loading does not materially improve one-step alignment

So the best current diagnosis is:

- not mainly routing quality
- not mainly the bank path
- not mainly output masking
- but the fact that the formulation sparsifies the **input basis** too aggressively before the MLP

### 11.5 Practical consequence

The previous ablation that selected `3/128` was almost certainly optimizing the wrong proxy.  
However, the new honest ablation also shows that simply picking a larger `k` is **not enough** with the current sparse MLP formula.

The next phase should therefore focus on changing the sparse MLP formulation itself, then re-running the ablation on the new formulation.

## 12) Fast sparse-placement probe inside the MLP

I ran a cheap layer-local probe on real captured MLP inputs from the 8B hybrid model using one short prompt:

- prompt: `Describe a sparse MLP routing system.`
- no full generation loop
- no retraining
- same routing (`top_k=3`, fallback off)
- compare where sparsity is inserted in the MLP chain

Tested variants:

1. `current_input_and_output_mask`
   - current formulation: mask hidden input before `gate_proj`/`up_proj`, then mask output
2. `input_mask_no_output_mask`
   - still mask hidden input before `gate_proj`/`up_proj`, but do not mask final output
3. `output_side_only_sparse_down`
   - keep full hidden input for `gate_proj`/`up_proj`
   - keep full intermediate activation
   - apply sparsity only when selecting output blocks in `down_proj`
4. `intermediate_group_mask_then_dense_down`
   - keep full hidden input for `gate_proj`/`up_proj`
   - apply a block-group mask in intermediate activation space
   - then use dense `down_proj`
5. `dense_output_mask_only`
   - full dense MLP, then mask only the final output blocks

Aggregate result across 18 active layers:

- `current_input_and_output_mask`
  - mean norm ratio: `0.002512`
  - mean cosine: `0.018180`
- `input_mask_no_output_mask`
  - mean norm ratio: `0.013673`
  - mean cosine: `0.129884`
- `output_side_only_sparse_down`
  - mean norm ratio: `0.146233`
  - mean cosine: `0.146119`
- `intermediate_group_mask_then_dense_down`
  - mean norm ratio: `0.120412`
  - mean cosine: `0.162992`
- `dense_output_mask_only`
  - mean norm ratio: `0.146230`
  - mean cosine: `0.146120`

Tail layers 12-17 show the same pattern:

- current input-mask formulation stays around `0.0024 - 0.0027` norm ratio
- input-mask without output mask rises only to about `0.012 - 0.015`
- output-side-only sparse down reaches about `0.144 - 0.165`
- intermediate-group masking reaches about `0.098 - 0.140`

Interpretation:

- The current formulation is by far the worst.
- Removing output masking helps, but only slightly.
- The big gain comes from preserving the full hidden input through `gate_proj` and `up_proj`.
- Once `gate/up` see the full residual stream, both output-side sparsity and intermediate-space sparsity preserve much more of the dense MLP signal.
- `output_side_only_sparse_down` and `dense_output_mask_only` are nearly identical in this probe, which means the main damage is not at the final output mask; it happens earlier when the input basis is sparsified.

Conclusion:

- For this Llama-style gated MLP, the best fast-tested place to put sparsity is after the gated interaction, not before it.
- Input-basis sparsity is the structurally wrong placement.
- The next redesign should preserve full-input `gate_proj`/`up_proj` and move sparsity to intermediate activation space and/or the `down_proj` output contribution.

# Strict Short-Prompt Stability Analysis

## Question
Why are outputs still not fully semantically stable on strict short prompts?

This report answers that with direct evidence from the current 8B hybrid + SCA runtime, not with theory-only speculation.

## Bottom Line

The current strict short-prompt instability is **not** primarily caused by long-range sparse attention paging, and it is **not** visible on prompt-time teacher-forced forwards.

The two verified causes are:

1. **Inference is running with task-bias injection enabled even though recalibration disabled it.**
2. **`strict_fully_sparse` adds a decode-time repetition-penalty path that by itself is enough to corrupt generation, even when sparse attention paging is disabled.**

Together these explain why short strict prompts still degrade even after speech-anchor recalibration.

## What Was Tested

### 1. Is sparse-attention paging the cause on short prompts?

Checked diagnostic artifacts from strict short-prompt runs:

- [results/_tmp_probe_v3_after_guard12.json](/c:/Users/andre/Desktop/Overføre/next_step_ai/results/_tmp_probe_v3_after_guard12.json)

Observed:

- `mean_selected_pages_per_step = 0.0`
- `mean_bytes_cpu_to_gpu_per_step = 0.0`

Conclusion:

- For these short prompts, the model is **not fetching long-range archive pages at all**.
- So the short-prompt instability is **not** caused by the two-stage sparse-attention paging path.

### 2. Is the hybrid baseline itself unstable?

Ran dense hybrid baseline on the same prompt:

- prompt: `Describe a sparse MLP routing system.`

Observed dense output:

- `Describe a sparse MLP routing system. The system should be able to route a`

Conclusion:

- The hybrid baseline is semantically stable on this prompt.
- The failure is introduced by the sparse path / strict runtime, not by the hybrid baseline.

### 3. Does sparse MLP alone degrade output?

Ran:

- hybrid baseline only
- hybrid + sparse MLP
- hybrid + sparse MLP + sparse attention
- hybrid + sparse MLP + strict sparse path

Observed:

- dense hybrid: coherent
- sparse MLP, no cache: degraded but still mostly grammatical
- sparse MLP, cache on: repetitive
- sparse MLP + sparse attention, non-strict: prompt-copy repetition
- sparse MLP + strict sparse: token corruption / junk continuation

Conclusion:

- Sparse MLP path is the first place quality regresses.
- Strict mode makes that worse.

### 4. Is the problem already present on prompt-time forward passes?

Generated direct dense-vs-sparse forward comparisons:

- [results/strict_short_prompt_drift_analysis_v1.json](/c:/Users/andre/Desktop/Overføre/next_step_ai/results/strict_short_prompt_drift_analysis_v1.json)
- [results/strict_decode_vs_prefill_analysis_v1.json](/c:/Users/andre/Desktop/Overføre/next_step_ai/results/strict_decode_vs_prefill_analysis_v1.json)
- [results/strict_cached_rollout_drift_analysis_v2.json](/c:/Users/andre/Desktop/Overføre/next_step_ai/results/strict_cached_rollout_drift_analysis_v2.json)

Observed on prompt-time / teacher-forced forward:

- dense and sparse logits were effectively identical
- final hidden cosine was ~1.0
- hidden MSE was ~0

Conclusion:

- The model does **not** meaningfully diverge on the initial prompt encoding.
- The failure shows up during **autoregressive decode behavior**, not during prompt-time forward alone.

## Verified Cause 1: Task-Bias Injection Mismatch

### Code fact

`run_sca_recalibration_from_hybrid_baseline.py` explicitly sets:

- `model.disable_task_bias_injection = not bool(cfg.include_task_embedding)`

With default recalibration settings, that means:

- **task bias injection is disabled during recalibration**

Relevant code:

- [run_sca_recalibration_from_hybrid_baseline.py](/c:/Users/andre/Desktop/Overføre/next_step_ai/llama3_neuroplastic/run_sca_recalibration_from_hybrid_baseline.py)
- [neuroplastic_llama_gqa_mamba.py](/c:/Users/andre/Desktop/Overføre/next_step_ai/llama3_neuroplastic/neuroplastic_llama_gqa_mamba.py)

The model forward applies task embedding bias only when neuroplasticity is enabled and task-bias injection is not disabled.

### Runtime fact

The inference runner does **not** set `disable_task_bias_injection=True` after loading the recalibrated SCA artifact.

Relevant file:

- [run_hybrid_gqa_mamba_inference.py](/c:/Users/andre/Desktop/Overføre/next_step_ai/llama3_neuroplastic/run_hybrid_gqa_mamba_inference.py)

### Direct toggle test

Same checkpoint, same prompt, same cached greedy decode, only one change:

- `disable_task_bias_injection=False`
- `disable_task_bias_injection=True`

Observed:

- `False`:
  - `Describe a sparse MLP routing system.`
  - immediate failure / no coherent continuation
- `True`:
  - `Describe a sparse MLP routing system. The system should be able to`

Conclusion:

- This is a confirmed calibration/runtime mismatch.
- The deployed runtime is adding a task-bias term that the recalibration pass explicitly excluded.
- That mismatch alone is enough to break short-prompt semantics.

## Verified Cause 2: Strict Decode Path Is Corrupting Generation

### Code fact

When `strict_fully_sparse=True`, generation applies an extra score modification:

- `_apply_repetition_penalty_to_scores(...)`

Relevant code:

- [neuroplastic_llama_gqa_mamba.py](/c:/Users/andre/Desktop/Overføre/next_step_ai/llama3_neuroplastic/neuroplastic_llama_gqa_mamba.py)

This penalty is applied over the generated-token tail and penalizes every token that has already appeared in the window.

### Isolation test

To isolate strict behavior from sparse-attention paging, ran:

- sparse MLP enabled
- task bias disabled
- sparse attention disabled
- `strict_fully_sparse=True`

Observed output:

- `Describe a sparse MLP routing system.olate meteoruple.Distance Sever metabol-ing ing`

This is the critical result:

- strict corruption occurs **even when sparse attention paging is disabled**

Conclusion:

- The strict short-prompt failure is not a paging problem.
- The strict decode branch itself is sufficient to produce semantic corruption.

### Why this branch is harmful here

The repetition penalty is being applied to a sparse path that is already more fragile than the dense baseline.

In practice that means:

- the sparse model’s next-token distribution is already flatter / less stable than the dense baseline on rollout
- the strict penalty then suppresses common continuation tokens seen in the recent window
- once high-probability valid tokens are penalized, argmax shifts into low-probability junk subwords
- those junk tokens then become part of the prefix, making subsequent steps even less stable

This matches the observed behavior:

- non-strict cached decode with task bias disabled can still be coherent
- strict decode immediately falls into malformed subword continuations

## What Did *Not* Turn Out To Be The Main Cause

### Not long-range sparse attention paging

Short prompts never touched long-range archive pages in the tested runs.

### Not the hybrid baseline

Dense hybrid output remains coherent on the same prompts.

### Not prompt-time dense-vs-sparse hidden/logit drift

Teacher-forced prompt passes remained nearly identical.

### Not `eval()` / train mode

A direct same-instance check with and without `model.eval()` on the coherent non-strict cached run produced the same output on this prompt.

So lack of `eval()` is not the primary reason for this specific failure mode.

## Final Causal Chain

For strict short prompts, the current failure mode is:

1. Recalibration trains with task-bias injection disabled.
2. Inference does not honor that setting by default.
3. That already pushes sparse decode off the calibrated manifold.
4. Strict mode then applies an aggressive repetition penalty on top of the sparse decode distribution.
5. Because short prompts are not using long-range paging, there is no attention-retrieval explanation to hide behind.
6. The result is semantic instability, repetition, or malformed subword loops.

## Practical Meaning

If the goal is to explain why strict short prompts are still unstable today, the answer is:

- **primary deployed mismatch:** task-bias injection is on at inference even though recalibration turned it off
- **primary strict-mode breaker:** the strict repetition-penalty branch is damaging sparse decode directly

This is why the outputs are still not fully semantically stable.

# Hybrid-Baseline SCA Recalibration Execution Report

## Date / Environment
- Date: 2026-03-12
- Machine: single GPU (`NVIDIA GeForce RTX 2080`)
- Python env used for runs: `verification_env\Scripts\python`

## What Was Implemented
- Added new recalibration runner:
  - `llama3_neuroplastic/run_sca_recalibration_from_hybrid_baseline.py`
- Added sparse MLP alignment capture support:
  - `llama3_neuroplastic/sca_sparse_mlp.py`
  - New fields/methods for alignment snapshots + fallback tracking.
- Added hybrid model recalibration APIs and task-bias control:
  - `llama3_neuroplastic/neuroplastic_llama_gqa_mamba.py`
  - Added:
    - `prepare_sca_local_recalibration(...)`
    - `set_mlp_alignment_capture(...)`
    - `compute_sca_local_recalibration_loss(...)`
    - `export_sca_recalibration_state()`
    - `load_sca_recalibration_state(...)`
  - Added runtime flag:
    - `disable_task_bias_injection`
- Added inference compatibility for recalibrated SCA artifact:
  - `llama3_neuroplastic/run_hybrid_gqa_mamba_inference.py`
  - New CLI option: `--sca-recalibrated-checkpoint`
- Added/updated tests:
  - `tests/test_sca_sparse_mlp.py`
  - `tests/test_hybrid_gqa_mamba.py`

## What Was Tested

### Static / Unit-Level Validation
- `python -m py_compile` on touched modules: passed.
- `verification_env\Scripts\python -m pytest -q tests/test_sca_sparse_mlp.py tests/test_hybrid_gqa_mamba.py`
  - Result: `25 passed`

### Runtime Recalibration Jobs

1. Export-only run
- Output dir: `results/sca_recalibration_export_only`
- Artifacts written:
  - `sca_recalibrated_state.pt`
  - `sca_recalibration_metrics.json`
- Status: success

2. 2-layer smoke run (8 steps, initial settings)
- Output dir: `results/sca_recalibration_2layers_8steps`
- Layers: `14-15`
- Status: success
- Key metrics:
  - Initial/Final loss: `0.0 -> 0.0`
  - Fallback rate by layer: `1.0` for both layers
- Interpretation: dense fallback dominated; optimization signal was effectively bypassed.

3. 2-layer smoke run (8 steps, no fast fallback)
- Output dir: `results/sca_recalibration_2layers_8steps_nofallback`
- Layers: `14-15`
- Status: success
- Key metrics:
  - Initial/Final loss: `0.7558002472 -> 0.0`
  - Fallback rate by layer: `0.0` for both layers
- Acceptance check:
  - Loss decreased by >=50% over 8 steps: yes

4. 4-layer smoke run (8 steps, no fast fallback)
- Output dir: `results/sca_recalibration_4layers_8steps_nofallback`
- Layers: `14-17`
- Status: success
- Key metrics:
  - Initial/Final loss: `0.7680563331 -> 0.0`
  - Fallback rate by layer: `0.0` for all four layers
- Acceptance check:
  - Loss decreased by >=50% over 8 steps: yes

### End-to-End Load + Inference Check
- Command path tested:
  - `run_hybrid_gqa_mamba_inference.py`
  - loaded both hybrid checkpoint and recalibrated SCA checkpoint
- Load output:
  - hybrid state load: success
  - recalibration state load: success (`loaded_items=2, missing_items=0`)
- Generation executed successfully.

## Issues Found and Fixes Applied During Execution

1. Hybrid layer mismatch on checkpoint load
- Symptom: strict load failed when model instantiated hybrid layers different from checkpoint payload.
- Fix:
  - Runner now reads checkpoint metadata (`layer_selection`, rank/threshold/state_dim) and aligns model hybrid init.
  - Runner uses `strict=False` for hybrid state load to support subset calibration from a broader checkpoint artifact.

2. No gradient path in local loss (runtime backward failure)
- Symptom: `RuntimeError: tensor does not require grad`.
- Root cause: alignment capture detached sparse output tensor before loss computation.
- Fix:
  - Kept `sparse_mlp_out` attached to graph in alignment capture.
  - Kept dense reference detached (`no_grad`) as intended.

3. Fallback domination in default smoke run
- Symptom: fallback rate 100%, loss fixed at 0.0.
- Resolution:
  - Re-ran smoke jobs with `--disable-fast-fallback` for meaningful calibration signal.
  - Metrics now clearly expose fallback rates, matching plan intent.

## Artifacts Produced
- `results/sca_recalibration_export_only/sca_recalibrated_state.pt`
- `results/sca_recalibration_export_only/sca_recalibration_metrics.json`
- `results/sca_recalibration_2layers_8steps/sca_recalibrated_state.pt`
- `results/sca_recalibration_2layers_8steps/sca_recalibration_metrics.json`
- `results/sca_recalibration_2layers_8steps_nofallback/sca_recalibrated_state.pt`
- `results/sca_recalibration_2layers_8steps_nofallback/sca_recalibration_metrics.json`
- `results/sca_recalibration_4layers_8steps_nofallback/sca_recalibrated_state.pt`
- `results/sca_recalibration_4layers_8steps_nofallback/sca_recalibration_metrics.json`

## Final Findings
- The recalibration pipeline is implemented and runnable end-to-end on the available 8B hybrid artifact.
- Unit/integration tests for new APIs and capture/loss/export/load behavior are passing.
- Smoke recalibration objectives are met when fast fallback is disabled in this environment.
- Fallback telemetry is essential; default settings can mask calibration by routing through dense fallback.

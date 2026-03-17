# Current Model File Usage Report

- Generated (UTC): 2026-03-17T21:38:46.283402+00:00
- Repository root: `C:\Users\andre\Desktop\Overføre\next_step_ai`
- Files audited in table: `335`
- `verification_env` included in table: `no`
- `verification_env` file count (excluded from table): `26977`

## Current Model Scope

- Runtime entrypoint roots:
  - `llama3_neuroplastic/run_hybrid_gqa_mamba_inference.py`
  - `llama3_neuroplastic/neuroplastic_llama_gqa_mamba.py`
  - `llama3_neuroplastic/sca_sparse_mlp.py`
  - `llama3_neuroplastic/sca_sparse_config.py`
- Calibration/diagnostic entrypoint roots:
  - `llama3_neuroplastic/run_sca_recalibration_from_hybrid_baseline.py`
  - `llama3_neuroplastic/init_learned_basis_from_dense_mlp.py`
  - `llama3_neuroplastic/run_sca_root_issue_locator.py`
  - `llama3_neuroplastic/run_sca_diagnostic_wipe.py`
  - `llama3_neuroplastic/run_decoder_mirror_sca_calibration.py`

## Category Counts

| category | count |
|---|---:|
| `CODE_NOT_IN_CURRENT_MODEL_PATH` | 34 |
| `CONFIG_OR_DATA` | 9 |
| `DOC_OR_ASSET` | 22 |
| `OTHER` | 2 |
| `RESULT_ARTIFACT` | 241 |
| `TEST_ONLY` | 11 |
| `USED_CURRENT_MODEL_CALIBRATION` | 5 |
| `USED_CURRENT_MODEL_RUNTIME` | 11 |

## Per-File Classification

| path | category | used_in_current_model | reason |
|---|---|---|---|
| `405B_SWITCH_READINESS_REPORT.md` | `DOC_OR_ASSET` | `no` | documentation or static asset |
| `_tmp_old_vs_current_lists.txt` | `DOC_OR_ASSET` | `no` | documentation or static asset |
| `_tmp_old_vs_current_lists_adjusted.txt` | `DOC_OR_ASSET` | `no` | documentation or static asset |
| `_tmp_old_vs_current_table.md` | `DOC_OR_ASSET` | `no` | documentation or static asset |
| `_tmp_used_vs_unused_table.md` | `DOC_OR_ASSET` | `no` | documentation or static asset |
| `accelerate_config.yaml` | `CONFIG_OR_DATA` | `optional` | used only when explicitly passed to scripts |
| `adaptive_components/custom_activation.py` | `CODE_NOT_IN_CURRENT_MODEL_PATH` | `no` | python module/script outside current runtime+calibration dependency graph |
| `benchmark_comparison_analysis.py` | `CODE_NOT_IN_CURRENT_MODEL_PATH` | `no` | python module/script outside current runtime+calibration dependency graph |
| `benchmark_config.yaml` | `CONFIG_OR_DATA` | `optional` | used only when explicitly passed to scripts |
| `CHAT_CHANGES_SUMMARY.md` | `DOC_OR_ASSET` | `no` | documentation or static asset |
| `DEFINITIVE_STRICT_OUTPUT_ROOT_CAUSE_REPORT.md` | `DOC_OR_ASSET` | `no` | documentation or static asset |
| `example_configs/llama_sca_objective_adaptive.yaml` | `CONFIG_OR_DATA` | `optional` | used only when explicitly passed to scripts |
| `example_configs/llama_sca_objective_bench_cuda.yaml` | `CONFIG_OR_DATA` | `optional` | used only when explicitly passed to scripts |
| `example_configs/llama_sca_objective_bench_dense.yaml` | `CONFIG_OR_DATA` | `optional` | used only when explicitly passed to scripts |
| `example_configs/llama_sca_objective_dense_poc_tune.yaml` | `CONFIG_OR_DATA` | `optional` | used only when explicitly passed to scripts |
| `example_configs/llama_sca_objective_grouped_gemm.yaml` | `CONFIG_OR_DATA` | `optional` | used only when explicitly passed to scripts |
| `example_configs/llama_sca_posthoc_ablation.yaml` | `CONFIG_OR_DATA` | `optional` | used only when explicitly passed to scripts |
| `example_configs/performance_test_config.yaml` | `CONFIG_OR_DATA` | `optional` | used only when explicitly passed to scripts |
| `extract_bio_metrics.py` | `CODE_NOT_IN_CURRENT_MODEL_PATH` | `no` | python module/script outside current runtime+calibration dependency graph |
| `generate_current_model_file_usage_report.py` | `CODE_NOT_IN_CURRENT_MODEL_PATH` | `no` | python module/script outside current runtime+calibration dependency graph |
| `GROUPED_GEMM_CUSTOM_KERNEL_PLAN.md` | `DOC_OR_ASSET` | `no` | documentation or static asset |
| `HONEST_SPARSE_WORKLOG.md` | `DOC_OR_ASSET` | `no` | documentation or static asset |
| `llama3_neuroplastic/bench_grouped_micro.py` | `CODE_NOT_IN_CURRENT_MODEL_PATH` | `no` | python module/script outside current runtime+calibration dependency graph |
| `llama3_neuroplastic/bench_grouped_row_gemm.py` | `CODE_NOT_IN_CURRENT_MODEL_PATH` | `no` | python module/script outside current runtime+calibration dependency graph |
| `llama3_neuroplastic/bounded_context.py` | `USED_CURRENT_MODEL_RUNTIME` | `yes` | reachable from runtime inference entrypoint(s) |
| `llama3_neuroplastic/check_grouped_correctness.py` | `CODE_NOT_IN_CURRENT_MODEL_PATH` | `no` | python module/script outside current runtime+calibration dependency graph |
| `llama3_neuroplastic/cuda/__init__.py` | `CODE_NOT_IN_CURRENT_MODEL_PATH` | `no` | python module/script outside current runtime+calibration dependency graph |
| `llama3_neuroplastic/cuda/sca_sparse_bindings.cpp` | `OTHER` | `no` | not part of current model runtime path |
| `llama3_neuroplastic/cuda/sca_sparse_kernels.cu` | `OTHER` | `no` | not part of current model runtime path |
| `llama3_neuroplastic/cuda/sca_sparse_loader.py` | `CODE_NOT_IN_CURRENT_MODEL_PATH` | `no` | python module/script outside current runtime+calibration dependency graph |
| `llama3_neuroplastic/generate_qualitative.py` | `CODE_NOT_IN_CURRENT_MODEL_PATH` | `no` | python module/script outside current runtime+calibration dependency graph |
| `llama3_neuroplastic/generated_sample.txt` | `DOC_OR_ASSET` | `no` | documentation or static asset |
| `llama3_neuroplastic/gqa_mamba_rank_collapse.py` | `USED_CURRENT_MODEL_RUNTIME` | `yes` | reachable from runtime inference entrypoint(s) |
| `llama3_neuroplastic/init_learned_basis_from_dense_mlp.py` | `USED_CURRENT_MODEL_CALIBRATION` | `yes` | reachable from calibration/diagnostic entrypoint(s) |
| `llama3_neuroplastic/llama_evaluation_pipeline.py` | `CODE_NOT_IN_CURRENT_MODEL_PATH` | `no` | python module/script outside current runtime+calibration dependency graph |
| `llama3_neuroplastic/neuroplastic_llama.py` | `CODE_NOT_IN_CURRENT_MODEL_PATH` | `no` | python module/script outside current runtime+calibration dependency graph |
| `llama3_neuroplastic/neuroplastic_llama_backup_pre_mamba.py` | `CODE_NOT_IN_CURRENT_MODEL_PATH` | `no` | python module/script outside current runtime+calibration dependency graph |
| `llama3_neuroplastic/neuroplastic_llama_gqa_mamba.py` | `USED_CURRENT_MODEL_RUNTIME` | `yes` | reachable from runtime inference entrypoint(s) |
| `llama3_neuroplastic/neuroplastic_llama_interpolated_sca_v2.py` | `CODE_NOT_IN_CURRENT_MODEL_PATH` | `no` | python module/script outside current runtime+calibration dependency graph |
| `llama3_neuroplastic/objective_losses.py` | `CODE_NOT_IN_CURRENT_MODEL_PATH` | `no` | python module/script outside current runtime+calibration dependency graph |
| `llama3_neuroplastic/objective_scheduler.py` | `CODE_NOT_IN_CURRENT_MODEL_PATH` | `no` | python module/script outside current runtime+calibration dependency graph |
| `llama3_neuroplastic/paged_sparse_attention.py` | `USED_CURRENT_MODEL_RUNTIME` | `yes` | reachable from runtime inference entrypoint(s) |
| `llama3_neuroplastic/profile_triton_sparse.py` | `CODE_NOT_IN_CURRENT_MODEL_PATH` | `no` | python module/script outside current runtime+calibration dependency graph |
| `llama3_neuroplastic/run_both_minimal.py` | `CODE_NOT_IN_CURRENT_MODEL_PATH` | `no` | python module/script outside current runtime+calibration dependency graph |
| `llama3_neuroplastic/run_decoder_mirror_sca_calibration.py` | `USED_CURRENT_MODEL_CALIBRATION` | `yes` | reachable from calibration/diagnostic entrypoint(s) |
| `llama3_neuroplastic/run_gqa_mamba_rank_collapse_pipeline.py` | `CODE_NOT_IN_CURRENT_MODEL_PATH` | `no` | python module/script outside current runtime+calibration dependency graph |
| `llama3_neuroplastic/run_gqa_mamba_sync_calibration.py` | `CODE_NOT_IN_CURRENT_MODEL_PATH` | `no` | python module/script outside current runtime+calibration dependency graph |
| `llama3_neuroplastic/run_hybrid_gqa_mamba_inference.py` | `USED_CURRENT_MODEL_RUNTIME` | `yes` | reachable from runtime inference entrypoint(s) |
| `llama3_neuroplastic/run_llama_benchmark_1000.py` | `CODE_NOT_IN_CURRENT_MODEL_PATH` | `no` | python module/script outside current runtime+calibration dependency graph |
| `llama3_neuroplastic/run_llama_benchmark_1000_added_buffer_interpolateinject.py` | `CODE_NOT_IN_CURRENT_MODEL_PATH` | `no` | python module/script outside current runtime+calibration dependency graph |
| `llama3_neuroplastic/run_sca_diagnostic_wipe.py` | `USED_CURRENT_MODEL_CALIBRATION` | `yes` | reachable from calibration/diagnostic entrypoint(s) |
| `llama3_neuroplastic/run_sca_flops_grid_ablation.py` | `CODE_NOT_IN_CURRENT_MODEL_PATH` | `no` | python module/script outside current runtime+calibration dependency graph |
| `llama3_neuroplastic/run_sca_posthoc_ablation.py` | `CODE_NOT_IN_CURRENT_MODEL_PATH` | `no` | python module/script outside current runtime+calibration dependency graph |
| `llama3_neuroplastic/run_sca_recalibration_from_hybrid_baseline.py` | `USED_CURRENT_MODEL_CALIBRATION` | `yes` | reachable from calibration/diagnostic entrypoint(s) |
| `llama3_neuroplastic/run_sca_root_issue_locator.py` | `USED_CURRENT_MODEL_CALIBRATION` | `yes` | reachable from calibration/diagnostic entrypoint(s) |
| `llama3_neuroplastic/run_sca_topk_ablation.py` | `CODE_NOT_IN_CURRENT_MODEL_PATH` | `no` | python module/script outside current runtime+calibration dependency graph |
| `llama3_neuroplastic/run_sparse_mlp_bank_export.py` | `CODE_NOT_IN_CURRENT_MODEL_PATH` | `no` | python module/script outside current runtime+calibration dependency graph |
| `llama3_neuroplastic/sca_decoder_mirror.py` | `USED_CURRENT_MODEL_RUNTIME` | `yes` | reachable from runtime inference entrypoint(s) |
| `llama3_neuroplastic/sca_sparse_adapter.py` | `USED_CURRENT_MODEL_RUNTIME` | `yes` | reachable from runtime inference entrypoint(s) |
| `llama3_neuroplastic/sca_sparse_config.py` | `USED_CURRENT_MODEL_RUNTIME` | `yes` | reachable from runtime inference entrypoint(s) |
| `llama3_neuroplastic/sca_sparse_mlp.py` | `USED_CURRENT_MODEL_RUNTIME` | `yes` | reachable from runtime inference entrypoint(s) |
| `llama3_neuroplastic/test_cola_quick.py` | `CODE_NOT_IN_CURRENT_MODEL_PATH` | `no` | python module/script outside current runtime+calibration dependency graph |
| `llama3_neuroplastic/train_llama_sca_objective.py` | `CODE_NOT_IN_CURRENT_MODEL_PATH` | `no` | python module/script outside current runtime+calibration dependency graph |
| `llama3_neuroplastic/triton_sca_gate.py` | `USED_CURRENT_MODEL_RUNTIME` | `yes` | reachable from runtime inference entrypoint(s) |
| `llama3_neuroplastic/triton_sparse_mlp.py` | `USED_CURRENT_MODEL_RUNTIME` | `yes` | reachable from runtime inference entrypoint(s) |
| `llama3_neuroplastic/tune_generation.py` | `CODE_NOT_IN_CURRENT_MODEL_PATH` | `no` | python module/script outside current runtime+calibration dependency graph |
| `llama3_neuroplastic/tuning_results.txt` | `DOC_OR_ASSET` | `no` | documentation or static asset |
| `llama3_neuroplastic/verify_gains.py` | `CODE_NOT_IN_CURRENT_MODEL_PATH` | `no` | python module/script outside current runtime+calibration dependency graph |
| `main.py` | `CODE_NOT_IN_CURRENT_MODEL_PATH` | `no` | python module/script outside current runtime+calibration dependency graph |
| `newtests.md` | `DOC_OR_ASSET` | `no` | documentation or static asset |
| `requirements.txt` | `DOC_OR_ASSET` | `no` | documentation or static asset |
| `research_paper2.tex` | `DOC_OR_ASSET` | `no` | documentation or static asset |
| `results/405b_switch_readiness_gates.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/405b_switch_readiness_gates_speech_anchor_v1.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/405b_switch_readiness_gates_speech_anchor_v2_reprobe.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/405b_switch_readiness_gates_speech_anchor_v3_reprobe_allow_cache.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/405b_switch_readiness_gates_speech_anchor_v4.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/405b_switch_readiness_gates_v2.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/_tmp_diag_long_allow_cache.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/_tmp_diag_strict_run.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/_tmp_learned_basis_init_smoke.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/_tmp_learned_basis_rollout_smoke/sca_recalibrated_state.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/_tmp_learned_basis_rollout_smoke/sca_recalibration_metrics.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/_tmp_learned_basis_v3_signal_check/sca_recalibrated_state.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/_tmp_learned_basis_v3_signal_check/sca_recalibration_metrics.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/_tmp_probe_after_backend_switch.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/_tmp_probe_v3.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/_tmp_probe_v3_after_cap.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/_tmp_probe_v3_after_guard.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/_tmp_probe_v3_after_guard12.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/_tmp_probe_v3_after_rep_penalty.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/_tmp_probe_v3_long.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/_tmp_recal_with_learned_init_smoke/sca_recalibrated_state.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/_tmp_recal_with_learned_init_smoke/sca_recalibration_metrics.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/_tmp_spatial_only_check/sca_recalibrated_state.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/_tmp_spatial_only_check/sca_recalibration_metrics.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/_tmp_wrap_diag.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/_tmp_wrap_long_diag.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/decoder_mirror_calibration_full_sparse_gate_v1/decoder_mirror_calibration_metrics.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/decoder_mirror_calibration_full_sparse_gate_v1/decoder_mirror_sca_state.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/decoder_mirror_calibration_full_sparse_gate_v2/decoder_mirror_calibration_metrics.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/decoder_mirror_calibration_full_sparse_gate_v2/decoder_mirror_sca_state.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/dense_hybrid_eval.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/dense_topk_mask_generation_probe.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/dense_topk_mask_generation_probe_hi.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/dense_topk_mask_layerband_probe.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/glue_evaluation_results_20251227_161118.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/glue_evaluation_results_20251227_170238.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/glue_evaluation_results_20251227_172906.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/glue_evaluation_results_20260109_155856.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/glue_evaluation_results_20260109_160358.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/gqa_mamba_8b_no_teacher/gqa_mamba_collapsed_attention_only.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/gqa_mamba_8b_no_teacher/gqa_mamba_no_teacher_metrics.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/grouped_row_bench_large.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/grouped_row_bench_small.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/grouped_row_correctness.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/grouped_row_microbench.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/hybrid_8b_export/hybrid_attention_metrics.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/hybrid_8b_export/hybrid_attention_state.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/hybrid_8b_local_geometry/hybrid_attention_metrics.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/hybrid_8b_local_geometry/hybrid_attention_state.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/hybrid_8b_local_geometry_v2/hybrid_attention_metrics.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/hybrid_8b_local_geometry_v2/hybrid_attention_state.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/hybrid_8b_small_local/hybrid_attention_metrics.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/hybrid_8b_small_local/hybrid_attention_state.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/learned_basis_init_v1.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/long_context_strict_sparse_attention_diag_speech_anchor_v1.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/long_context_strict_sparse_attention_diag_with_mirror_v2.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/long_context_strict_sparse_attention_eval.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/long_context_strict_sparse_attention_eval_speech_anchor_v1.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/long_context_strict_sparse_attention_eval_with_mirror_v2.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/mlp_sparse_placement_probe.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/nonstrict_default_fallback_diag.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/nonstrict_dense_f0_diag.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/oracle_intermediate_group_topk_sweep.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/oracle_output_block_routing_probe.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/oracle_output_block_topk_sweep.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/oracle_output_block_topk_sweep_hi.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/readiness_sparse_attention_strict_quick.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/readiness_sparse_attention_strict_quick_speech_anchor_v1.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/readiness_sparse_attention_strict_quick_with_mirror_v2.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/readiness_sparse_attention_strict_quick_with_mirror_v3.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/readiness_sparse_baseline_quick.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/readiness_sparse_baseline_quick_speech_anchor_v1.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/readiness_sparse_baseline_quick_v2.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_diagnostic_wipe_v2.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_mlp_spmm_repro5/model_step_0000005/config.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_mlp_spmm_repro5/model_step_0000005/neuroplastic_llama_sca_v2.bin` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_mlp_spmm_repro5/objective_step_0000005.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_mlp_spmm_run1_repro/model_step_0000002/config.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_mlp_spmm_run1_repro/model_step_0000002/neuroplastic_llama_sca_v2.bin` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_mlp_spmm_run1_repro/objective_step_0000002.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_mlp_spmm_smoke/model_step_0000001/config.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_mlp_spmm_smoke/model_step_0000001/neuroplastic_llama_sca_v2.bin` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_mlp_spmm_smoke/objective_step_0000001.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_mlp_spmm_smoke2/model_step_0000001/config.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_mlp_spmm_smoke2/model_step_0000001/neuroplastic_llama_sca_v2.bin` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_mlp_spmm_smoke2/objective_step_0000001.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_posthoc_ablation/perf_after_mech_opt.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_posthoc_ablation/perf_after_opt.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_posthoc_ablation/perf_after_opt2.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_posthoc_ablation/perf_after_opt3.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_posthoc_ablation/perf_after_v2_strict.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_posthoc_ablation/perf_before_opt.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_posthoc_ablation/perf_tune_base.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_posthoc_ablation/perf_tune_bo128.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_posthoc_ablation/perf_tune_bo128_bk64.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_posthoc_ablation/perf_tune_bo128_bk64_ow8.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_posthoc_ablation/perf_tune_bo128_w8.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_posthoc_ablation/perf_tune_bo64_w8.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_posthoc_ablation/perf_user_opt.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_posthoc_ablation/perf_user_opt_2nd.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_posthoc_ablation/results.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_posthoc_ablation/smoke.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_posthoc_ablation/smoke9_int4.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_posthoc_ablation/smoke_int4.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_recalibration_2layers_8steps/sca_recalibrated_state.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_recalibration_2layers_8steps/sca_recalibration_metrics.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_recalibration_2layers_8steps_nofallback/sca_recalibrated_state.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_recalibration_2layers_8steps_nofallback/sca_recalibration_metrics.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_recalibration_4layers_8steps_nofallback/sca_recalibrated_state.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_recalibration_4layers_8steps_nofallback/sca_recalibration_metrics.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_recalibration_4layers_logits_refine/sca_recalibrated_state.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_recalibration_4layers_logits_refine/sca_recalibration_metrics.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_recalibration_4layers_logits_refine_34/sca_recalibrated_state.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_recalibration_4layers_logits_refine_34/sca_recalibration_metrics.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_recalibration_export_only/sca_recalibrated_state.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_recalibration_export_only/sca_recalibration_metrics.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_recalibration_full_sparse_gate/sca_recalibrated_state.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_recalibration_full_sparse_gate/sca_recalibration_metrics.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_recalibration_full_sparse_gate_v2/sca_recalibrated_state.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_recalibration_full_sparse_gate_v2/sca_recalibration_metrics.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_recalibration_full_sparse_gate_vcvars/sca_recalibrated_state.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_recalibration_full_sparse_gate_vcvars/sca_recalibration_metrics.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_recalibration_full_sparse_gate_vs2022/sca_recalibrated_state.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_recalibration_full_sparse_gate_vs2022/sca_recalibration_metrics.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_recalibration_honest_sparse_smoke/sca_recalibrated_state.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_recalibration_honest_sparse_smoke/sca_recalibration_metrics.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_recalibration_honest_sparse_smoke2/sca_recalibrated_state.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_recalibration_honest_sparse_smoke2/sca_recalibration_metrics.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_recalibration_honest_sparse_v1_stable/sca_recalibrated_state.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_recalibration_honest_sparse_v1_stable/sca_recalibration_metrics.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_recalibration_honest_sparse_v4_no_speech/sca_recalibrated_state.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_recalibration_honest_sparse_v4_no_speech/sca_recalibration_metrics.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_recalibration_honest_sparse_v6/sca_recalibrated_state.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_recalibration_honest_sparse_v6/sca_recalibration_metrics.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_recalibration_honest_sparse_v7/sca_recalibrated_state.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_recalibration_honest_sparse_v7/sca_recalibration_metrics.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_recalibration_honest_sparse_v8_logits_rollout/sca_recalibrated_state.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_recalibration_honest_sparse_v8_logits_rollout/sca_recalibration_metrics.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_recalibration_learned_basis_init_v1/sca_recalibrated_state.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_recalibration_learned_basis_init_v1/sca_recalibration_metrics.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_recalibration_learned_basis_init_v2/sca_recalibrated_state.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_recalibration_learned_basis_init_v2/sca_recalibration_metrics.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_recalibration_learned_basis_v1_pilot/sca_recalibrated_state.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_recalibration_learned_basis_v1_pilot/sca_recalibration_metrics.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_recalibration_learned_basis_v2_strict/sca_recalibrated_state.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_recalibration_learned_basis_v2_strict/sca_recalibration_metrics.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_recalibration_spatial_logits_lr1e4/sca_recalibrated_state.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_recalibration_spatial_logits_lr1e4/sca_recalibration_metrics.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_recalibration_spatial_logits_w0/sca_recalibrated_state.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_recalibration_spatial_logits_w0/sca_recalibration_metrics.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_recalibration_speech_anchor_v1/sca_recalibrated_state.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_recalibration_speech_anchor_v1/sca_recalibration_metrics.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_recalibration_speech_anchor_v2_smoke/sca_recalibrated_state.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_recalibration_speech_anchor_v2_smoke/sca_recalibration_metrics.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_recalibration_speech_anchor_v3_smoke/sca_recalibrated_state.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_recalibration_speech_anchor_v3_smoke/sca_recalibration_metrics.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_recalibration_strict_active_band_smoke/sca_recalibrated_state.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_recalibration_strict_active_band_smoke/sca_recalibration_metrics.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_recalibration_task_safe/sca_recalibrated_state.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_recalibration_task_safe/sca_recalibration_metrics.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_recalibration_triton_gate_smoke/sca_recalibrated_state.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_recalibration_triton_gate_smoke/sca_recalibration_metrics.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_root_cause_diagnosis_v1/sca_diagnosis_results.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sca_root_issue_locator_v2.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sparse_attention_nonstrict_eval.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sparse_attention_strict_eval.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sparse_baseline_eval.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sparse_mlp_bank_8b_full_fp16/layer_000_mlp_blocks.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sparse_mlp_bank_8b_full_fp16/layer_001_mlp_blocks.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sparse_mlp_bank_8b_full_fp16/layer_002_mlp_blocks.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sparse_mlp_bank_8b_full_fp16/layer_003_mlp_blocks.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sparse_mlp_bank_8b_full_fp16/layer_004_mlp_blocks.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sparse_mlp_bank_8b_full_fp16/layer_005_mlp_blocks.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sparse_mlp_bank_8b_full_fp16/layer_006_mlp_blocks.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sparse_mlp_bank_8b_full_fp16/layer_007_mlp_blocks.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sparse_mlp_bank_8b_full_fp16/layer_008_mlp_blocks.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sparse_mlp_bank_8b_full_fp16/layer_009_mlp_blocks.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sparse_mlp_bank_8b_full_fp16/layer_010_mlp_blocks.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sparse_mlp_bank_8b_full_fp16/layer_011_mlp_blocks.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sparse_mlp_bank_8b_full_fp16/layer_012_mlp_blocks.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sparse_mlp_bank_8b_full_fp16/layer_013_mlp_blocks.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sparse_mlp_bank_8b_full_fp16/layer_014_mlp_blocks.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sparse_mlp_bank_8b_full_fp16/layer_015_mlp_blocks.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sparse_mlp_bank_8b_full_fp16/layer_016_mlp_blocks.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sparse_mlp_bank_8b_full_fp16/layer_017_mlp_blocks.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sparse_mlp_bank_8b_full_fp16/layer_018_mlp_blocks.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sparse_mlp_bank_8b_full_fp16/layer_019_mlp_blocks.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sparse_mlp_bank_8b_full_fp16/layer_020_mlp_blocks.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sparse_mlp_bank_8b_full_fp16/layer_021_mlp_blocks.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sparse_mlp_bank_8b_full_fp16/layer_022_mlp_blocks.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sparse_mlp_bank_8b_full_fp16/layer_023_mlp_blocks.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sparse_mlp_bank_8b_full_fp16/layer_024_mlp_blocks.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sparse_mlp_bank_8b_full_fp16/layer_025_mlp_blocks.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sparse_mlp_bank_8b_full_fp16/layer_026_mlp_blocks.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sparse_mlp_bank_8b_full_fp16/layer_027_mlp_blocks.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sparse_mlp_bank_8b_full_fp16/layer_028_mlp_blocks.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sparse_mlp_bank_8b_full_fp16/layer_029_mlp_blocks.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sparse_mlp_bank_8b_full_fp16/layer_030_mlp_blocks.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sparse_mlp_bank_8b_full_fp16/layer_031_mlp_blocks.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sparse_mlp_bank_8b_full_fp16/mlp_bank_manifest.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sparse_mlp_bank_8b_mid_fp16/layer_008_mlp_blocks.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sparse_mlp_bank_8b_mid_fp16/layer_009_mlp_blocks.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sparse_mlp_bank_8b_mid_fp16/layer_010_mlp_blocks.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sparse_mlp_bank_8b_mid_fp16/layer_011_mlp_blocks.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sparse_mlp_bank_8b_mid_fp16/layer_012_mlp_blocks.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sparse_mlp_bank_8b_mid_fp16/layer_013_mlp_blocks.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sparse_mlp_bank_8b_mid_fp16/layer_014_mlp_blocks.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sparse_mlp_bank_8b_mid_fp16/layer_015_mlp_blocks.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sparse_mlp_bank_8b_mid_fp16/layer_016_mlp_blocks.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sparse_mlp_bank_8b_mid_fp16/layer_017_mlp_blocks.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sparse_mlp_bank_8b_mid_fp16/layer_018_mlp_blocks.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sparse_mlp_bank_8b_mid_fp16/layer_019_mlp_blocks.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sparse_mlp_bank_8b_mid_fp16/layer_020_mlp_blocks.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sparse_mlp_bank_8b_mid_fp16/layer_021_mlp_blocks.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sparse_mlp_bank_8b_mid_fp16/layer_022_mlp_blocks.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sparse_mlp_bank_8b_mid_fp16/layer_023_mlp_blocks.pt` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/sparse_mlp_bank_8b_mid_fp16/mlp_bank_manifest.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/strict_cached_rollout_drift_analysis_v2.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/strict_cuda_spmm_diag.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/strict_cuda_spmm_diag_activeband.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/strict_decode_vs_prefill_analysis_v1.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/strict_dense_spmm_diag.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/strict_dense_spmm_diag_activeband.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/strict_penalty_flip_analysis_v1.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/strict_probe_after_runtime_fixes.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/strict_probe_intermediate_group_v1.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/strict_probe_learned_basis_init_v2.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/strict_probe_learned_basis_v2_strict.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/strict_probe_output_sparse_v1.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/strict_probe_v6.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/strict_probe_v7.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/strict_probe_v8.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/strict_probe_v8_guard24.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/strict_rollout_drift_analysis_v1.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/strict_short_prompt_drift_analysis_v1.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/topk_probe_12.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/topk_probe_16.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/topk_probe_3.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/topk_probe_4.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/topk_probe_6.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `results/topk_probe_8.json` | `RESULT_ARTIFACT` | `no` | generated checkpoint/metric artifact; load only when explicitly provided |
| `SCA_BACKEND_SWITCH_NOTE.md` | `DOC_OR_ASSET` | `no` | documentation or static asset |
| `sca_grid_size_notes.md` | `DOC_OR_ASSET` | `no` | documentation or static asset |
| `SCA_RECALIBRATION_EXECUTION_REPORT.md` | `DOC_OR_ASSET` | `no` | documentation or static asset |
| `SCA_ROOT_CAUSE_DIAGNOSIS_REPORT.md` | `DOC_OR_ASSET` | `no` | documentation or static asset |
| `semantic_attractor_comparison.png` | `DOC_OR_ASSET` | `no` | documentation or static asset |
| `setup.py` | `CODE_NOT_IN_CURRENT_MODEL_PATH` | `no` | python module/script outside current runtime+calibration dependency graph |
| `simple_evaluation_pipeline.py` | `CODE_NOT_IN_CURRENT_MODEL_PATH` | `no` | python module/script outside current runtime+calibration dependency graph |
| `STRICT_SHORT_PROMPT_STABILITY_ANALYSIS.md` | `DOC_OR_ASSET` | `no` | documentation or static asset |
| `tests/test_bounded_context.py` | `TEST_ONLY` | `no` | used by test runs, not runtime model path |
| `tests/test_gqa_mamba_rank_collapse.py` | `TEST_ONLY` | `no` | used by test runs, not runtime model path |
| `tests/test_hybrid_gqa_mamba.py` | `TEST_ONLY` | `no` | used by test runs, not runtime model path |
| `tests/test_objective_losses.py` | `TEST_ONLY` | `no` | used by test runs, not runtime model path |
| `tests/test_objective_scheduler.py` | `TEST_ONLY` | `no` | used by test runs, not runtime model path |
| `tests/test_paged_sparse_attention.py` | `TEST_ONLY` | `no` | used by test runs, not runtime model path |
| `tests/test_sca_decoder_mirror.py` | `TEST_ONLY` | `no` | used by test runs, not runtime model path |
| `tests/test_sca_sparse_adapter.py` | `TEST_ONLY` | `no` | used by test runs, not runtime model path |
| `tests/test_sca_sparse_mlp.py` | `TEST_ONLY` | `no` | used by test runs, not runtime model path |
| `tests/test_sparse_mlp_bank_export.py` | `TEST_ONLY` | `no` | used by test runs, not runtime model path |
| `tests/test_train_llama_objective_smoke.py` | `TEST_ONLY` | `no` | used by test runs, not runtime model path |
| `THREE_TIER_SPARSE_ATTENTION_IMPLEMENTATION_REPORT.md` | `DOC_OR_ASSET` | `no` | documentation or static asset |
| `TRITON_SCA_GATE_MIGRATION_REPORT.md` | `DOC_OR_ASSET` | `no` | documentation or static asset |
| `visualize_semantic_attractor.py` | `CODE_NOT_IN_CURRENT_MODEL_PATH` | `no` | python module/script outside current runtime+calibration dependency graph |

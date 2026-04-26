from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ThroughputTargets:
    decode_tok_s: float = 3.30
    mean_decode_ms_per_token: float = 303.0
    decode_avg_mb_per_layer: float = 23.0


THROUGHPUT_TARGETS = ThroughputTargets()

THROUGHPUT_PROBE_DEFAULTS: dict[str, Any] = {
    "sparse_top_k": 208,
    "sparse_basis_top_k": 64,
    "sparse_mlp_execution": "exact_blockwise_sparse",
    "sparse_mlp_prefill_mode": "hot_cache",
    "vram_hot_cache_gb": 5.25,
    "pre_warm": True,
    "calibrate_hot_cache": True,
    "hot_cache_calibration_tokens": 64,
    "attn_active_heads": 5,
    "attn_min_active_heads": 5,
    "attn_max_active_heads": 5,
    "sparse_attn_prefill_mode": "sparse",
    "sparse_kv_prefill_mode": "sparse",
}

CONTRACT_SCHEMA_VERSION = 1


def normalize_throughput_contract(raw_value: str | None) -> str:
    value = str(raw_value or "off").strip().lower() or "off"
    if value in {"off", "none", "false", "0"}:
        return "off"
    if value in {"probe", "fast_path"}:
        return "probe"
    if value in {"strict", "enforce"}:
        return "strict"
    raise RuntimeError("throughput-contract must be one of: off, probe, strict")


def apply_throughput_probe_defaults(args: Any) -> None:
    for key, value in THROUGHPUT_PROBE_DEFAULTS.items():
        setattr(args, key, value)


def build_contract_check(*, name: str, passed: bool, actual: Any, expected: Any) -> dict[str, Any]:
    return {
        "name": str(name),
        "passed": bool(passed),
        "actual": actual,
        "expected": expected,
    }


def _traffic_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    traffic = payload.get("traffic")
    if isinstance(traffic, dict):
        return traffic
    traffic_report = payload.get("traffic_report")
    if isinstance(traffic_report, dict):
        return traffic_report
    return {}


def _decode_tok_s(payload: dict[str, Any]) -> float:
    if "decode_tok_s" in payload:
        return float(payload.get("decode_tok_s", 0.0))
    return float(payload.get("decode_tokens_per_second", 0.0))


def build_throughput_contract_report(
    payload: dict[str, Any],
    *,
    include_sparse_kv_checks: bool = False,
) -> dict[str, Any]:
    runtime_status = dict(payload.get("runtime_status", {}) or {})
    traffic = _traffic_from_payload(payload)
    decode_traffic = dict(traffic.get("decode", {}) or {})
    new_tokens = int(payload.get("new_tokens", 0))
    expected_layer_visits = int(runtime_status.get("num_layers", 0)) * new_tokens
    actual_layer_visits = int(decode_traffic.get("layer_visits", 0))
    decode_tok_s = _decode_tok_s(payload)

    checks = [
        build_contract_check(
            name="decode_tok_s",
            passed=decode_tok_s >= THROUGHPUT_TARGETS.decode_tok_s,
            actual=decode_tok_s,
            expected=f">={THROUGHPUT_TARGETS.decode_tok_s}",
        ),
        build_contract_check(
            name="mean_decode_ms_per_token",
            passed=float(payload.get("mean_decode_ms_per_token", float("inf")))
            <= THROUGHPUT_TARGETS.mean_decode_ms_per_token,
            actual=float(payload.get("mean_decode_ms_per_token", 0.0)),
            expected=f"<={THROUGHPUT_TARGETS.mean_decode_ms_per_token}",
        ),
        build_contract_check(
            name="decode_avg_mb_per_layer",
            passed=float(payload.get("decode_avg_mb_per_layer", float("inf")))
            <= THROUGHPUT_TARGETS.decode_avg_mb_per_layer,
            actual=float(payload.get("decode_avg_mb_per_layer", 0.0)),
            expected=f"<={THROUGHPUT_TARGETS.decode_avg_mb_per_layer}",
        ),
        build_contract_check(
            name="decode_layer_visits",
            passed=expected_layer_visits > 0 and actual_layer_visits == expected_layer_visits,
            actual=actual_layer_visits,
            expected=expected_layer_visits,
        ),
        build_contract_check(
            name="lm_head_on_gpu",
            passed=bool(runtime_status.get("lm_head_on_gpu", False)),
            actual=bool(runtime_status.get("lm_head_on_gpu", False)),
            expected=True,
        ),
        build_contract_check(
            name="lm_head_mode",
            passed=str(runtime_status.get("lm_head_mode", "")) == "gpu_nf4",
            actual=str(runtime_status.get("lm_head_mode", "")),
            expected="gpu_nf4",
        ),
        build_contract_check(
            name="decode_backend",
            passed=str(runtime_status.get("decode_backend", "")) == "single_kernel_sparse_decode_sm75",
            actual=str(runtime_status.get("decode_backend", "")),
            expected="single_kernel_sparse_decode_sm75",
        ),
        build_contract_check(
            name="attn_backend_decode",
            passed=str(runtime_status.get("attn_backend_decode", "")) == "compact_sparse_v1",
            actual=str(runtime_status.get("attn_backend_decode", "")),
            expected="compact_sparse_v1",
        ),
        build_contract_check(
            name="compact_sparse_attention_steps",
            passed=int(runtime_status.get("compact_sparse_attention_steps", 0)) > 0,
            actual=int(runtime_status.get("compact_sparse_attention_steps", 0)),
            expected=">0",
        ),
        build_contract_check(
            name="vram_hot_cache_live_calibrated",
            passed=bool(runtime_status.get("vram_hot_cache_live_calibrated", False)),
            actual=bool(runtime_status.get("vram_hot_cache_live_calibrated", False)),
            expected=True,
        ),
        build_contract_check(
            name="decode_mlp_cold_blocks_streamed",
            passed=int(runtime_status.get("decode_mlp_cold_blocks_streamed", 0)) == 0,
            actual=int(runtime_status.get("decode_mlp_cold_blocks_streamed", 0)),
            expected=0,
        ),
        build_contract_check(
            name="decode_down_cold_blocks_streamed",
            passed=int(runtime_status.get("decode_down_cold_blocks_streamed", 0)) == 0,
            actual=int(runtime_status.get("decode_down_cold_blocks_streamed", 0)),
            expected=0,
        ),
    ]
    if include_sparse_kv_checks:
        checks.extend(
            [
                build_contract_check(
                    name="sparse_attention_layers",
                    passed=int(runtime_status.get("sparse_attention_layers", 0))
                    == int(runtime_status.get("num_layers", 0)),
                    actual=int(runtime_status.get("sparse_attention_layers", 0)),
                    expected=int(runtime_status.get("num_layers", 0)),
                ),
                build_contract_check(
                    name="sparse_kv_enabled_for_decode",
                    passed=bool(runtime_status.get("sparse_kv_enabled_for_decode", False)),
                    actual=bool(runtime_status.get("sparse_kv_enabled_for_decode", False)),
                    expected=True,
                ),
            ]
        )

    failed = [check["name"] for check in checks if not bool(check["passed"])]
    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "contract": "strict",
        "passed": len(failed) == 0,
        "failed_checks": failed,
        "checks": checks,
    }

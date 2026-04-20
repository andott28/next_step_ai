from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

try:
    from transformers import AutoConfig
except ImportError:
    AutoConfig = None


def _as_int(value: Any, default: int = 0) -> int:
    if torch.is_tensor(value):
        if int(value.numel()) == 1:
            return int(value.detach().cpu().view(-1)[0].item())
        return default
    try:
        return int(value)
    except Exception:
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    if torch.is_tensor(value):
        if int(value.numel()) == 1:
            return float(value.detach().cpu().view(-1)[0].item())
        return default
    try:
        return float(value)
    except Exception:
        return default


def _layer_stat(layer_state: dict[str, Any], stats: dict[str, Any], layer_key: str, name: str, default: Any) -> Any:
    if name in layer_state:
        return layer_state[name]
    layer_stats = stats.get(layer_key, {}) if isinstance(stats, dict) else {}
    if isinstance(layer_stats, dict) and name in layer_stats:
        return layer_stats[name]
    return default


def _load_model_shape(model_name: str, local_files_only: bool) -> dict[str, int]:
    if not model_name:
        return {}
    if AutoConfig is None:
        raise RuntimeError("transformers is required when --model-name is provided")
    config = AutoConfig.from_pretrained(str(model_name), local_files_only=bool(local_files_only))
    return {
        "hidden_size": int(getattr(config, "hidden_size", 0)),
        "intermediate_size": int(getattr(config, "intermediate_size", 0)),
        "num_hidden_layers": int(getattr(config, "num_hidden_layers", 0)),
    }


def _expected_blocks_for_domain(model_shape: dict[str, int], block_domain: str, block_size: int) -> int | None:
    if not model_shape:
        return None
    width = int(model_shape["hidden_size"] if block_domain == "output" else model_shape["intermediate_size"])
    if width <= 0 or block_size <= 0:
        return None
    if width % block_size != 0:
        return -1
    return int(width // block_size)


def _validate_layer(
    *,
    layer_key: str,
    layer_state: dict[str, Any],
    stats: dict[str, Any],
    config: dict[str, Any],
    model_shape: dict[str, int],
    artifact_target: str,
    block_domain: str,
    block_size: int,
    config_num_blocks: int,
    min_explained_variance: float,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    layer_summary: dict[str, Any] = {"layer": int(layer_key)}

    enc_w = layer_state.get("encoder_weight")
    enc_b = layer_state.get("encoder_bias")
    if not torch.is_tensor(enc_w) or int(enc_w.ndim) != 2:
        errors.append(f"layer {layer_key}: encoder_weight must be a rank-2 tensor")
        return layer_summary, errors
    if not torch.is_tensor(enc_b) or int(enc_b.ndim) != 1:
        errors.append(f"layer {layer_key}: encoder_bias must be a rank-1 tensor")
        return layer_summary, errors

    basis_rank = int(enc_w.shape[0])
    hidden_size = int(enc_w.shape[1])
    layer_summary["basis_rank"] = basis_rank
    layer_summary["encoder_hidden_size"] = hidden_size
    if int(enc_b.numel()) != basis_rank:
        errors.append(
            f"layer {layer_key}: encoder_bias has {int(enc_b.numel())} values, expected basis_rank={basis_rank}"
        )
    if model_shape and hidden_size != int(model_shape["hidden_size"]):
        errors.append(
            f"layer {layer_key}: encoder input dim {hidden_size} != model hidden_size {model_shape['hidden_size']}"
        )

    rank_effective = _as_int(_layer_stat(layer_state, stats, layer_key, "rank_effective", basis_rank), basis_rank)
    evr = _as_float(_layer_stat(layer_state, stats, layer_key, "explained_variance_ratio", 1.0), 1.0)
    samples = _as_float(_layer_stat(layer_state, stats, layer_key, "samples", -1.0), -1.0)
    layer_summary["rank_effective"] = rank_effective
    layer_summary["explained_variance_ratio"] = evr
    layer_summary["samples"] = samples
    if rank_effective < 1 or rank_effective > basis_rank:
        errors.append(f"layer {layer_key}: rank_effective={rank_effective} is outside [1, {basis_rank}]")
    if min_explained_variance > 0.0 and evr < min_explained_variance:
        errors.append(
            f"layer {layer_key}: explained_variance_ratio={evr:.6f} < required {min_explained_variance:.6f}"
        )

    expected_blocks = _expected_blocks_for_domain(model_shape, block_domain, block_size)
    if expected_blocks == -1:
        errors.append(f"model {block_domain} width is not divisible by block_size={block_size}")

    if block_domain == "intermediate" or artifact_target == "intermediate_block_scores":
        score_weight = layer_state.get("score_weight")
        score_bias = layer_state.get("score_bias")
        if not torch.is_tensor(score_weight) or int(score_weight.ndim) != 2:
            errors.append(f"layer {layer_key}: intermediate artifact requires rank-2 score_weight")
            return layer_summary, errors
        if int(score_weight.shape[1]) != basis_rank:
            errors.append(
                f"layer {layer_key}: score_weight basis dim {int(score_weight.shape[1])} != basis_rank={basis_rank}"
            )
        num_blocks = int(score_weight.shape[0])
        if not torch.is_tensor(score_bias) or int(score_bias.ndim) != 1:
            errors.append(f"layer {layer_key}: intermediate artifact requires rank-1 score_bias")
        elif int(score_bias.numel()) != num_blocks:
            errors.append(
                f"layer {layer_key}: score_bias has {int(score_bias.numel())} values, expected {num_blocks}"
            )
        block_importance = layer_state.get("block_importance")
        if torch.is_tensor(block_importance) and int(block_importance.numel()) != num_blocks:
            errors.append(
                f"layer {layer_key}: block_importance has {int(block_importance.numel())} values, expected {num_blocks}"
            )
        layer_summary["num_blocks"] = num_blocks
        layer_summary["payload"] = "score_weight"
    else:
        decoder_blocks = layer_state.get("decoder_blocks")
        decoder_bias = layer_state.get("decoder_bias")
        if not torch.is_tensor(decoder_blocks) or int(decoder_blocks.ndim) != 3:
            errors.append(f"layer {layer_key}: output artifact requires rank-3 decoder_blocks")
            return layer_summary, errors
        if int(decoder_blocks.shape[1]) != basis_rank:
            errors.append(
                f"layer {layer_key}: decoder basis dim {int(decoder_blocks.shape[1])} != basis_rank={basis_rank}"
            )
        if int(decoder_blocks.shape[2]) != block_size:
            errors.append(
                f"layer {layer_key}: decoder block size {int(decoder_blocks.shape[2])} != config block_size={block_size}"
            )
        num_blocks = int(decoder_blocks.shape[0])
        if not torch.is_tensor(decoder_bias) or int(decoder_bias.ndim) != 2:
            errors.append(f"layer {layer_key}: output artifact requires rank-2 decoder_bias")
        elif tuple(decoder_bias.shape) != (num_blocks, block_size):
            errors.append(
                f"layer {layer_key}: decoder_bias shape {tuple(decoder_bias.shape)} != {(num_blocks, block_size)}"
            )
        layer_summary["num_blocks"] = num_blocks
        layer_summary["payload"] = "decoder_blocks"

    if config_num_blocks > 0 and int(layer_summary["num_blocks"]) != config_num_blocks:
        errors.append(
            f"layer {layer_key}: num_blocks={int(layer_summary['num_blocks'])} != config num_blocks={config_num_blocks}"
        )
    if expected_blocks is not None and expected_blocks >= 0 and int(layer_summary["num_blocks"]) != expected_blocks:
        errors.append(
            f"layer {layer_key}: num_blocks={int(layer_summary['num_blocks'])} != expected {expected_blocks} "
            f"for model {block_domain} width and block_size={block_size}"
        )
    if int(config.get("basis_rank", basis_rank)) != basis_rank:
        errors.append(
            f"layer {layer_key}: basis_rank={basis_rank} != config basis_rank={int(config.get('basis_rank'))}"
        )

    return layer_summary, errors


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a sparse MLP basis/router checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the sparse MLP checkpoint .pt file.")
    parser.add_argument("--model-name", type=str, default="", help="Optional HF model id or local snapshot for shape checks.")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--expect-target",
        type=str,
        default="intermediate_block_scores",
        choices=["any", "output_reconstruction", "intermediate_block_scores"],
    )
    parser.add_argument(
        "--expect-block-domain",
        type=str,
        default="intermediate",
        choices=["any", "output", "intermediate"],
    )
    parser.add_argument("--expect-layers", type=int, default=0, help="Require exactly this many layer states when > 0.")
    parser.add_argument("--expect-num-blocks", type=int, default=0, help="Require every layer to expose this block count.")
    parser.add_argument("--min-layers", type=int, default=1, help="Require at least this many layer states.")
    parser.add_argument("--min-explained-variance", type=float, default=0.0)
    parser.add_argument("--json-out", type=str, default="", help="Optional path to write the JSON validation report.")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    checkpoint_path = Path(args.checkpoint)
    payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError("Checkpoint payload must be a dict")

    config = payload.get("config", {})
    if not isinstance(config, dict):
        raise RuntimeError("Checkpoint config must be a dict")
    layer_states = payload.get("layer_states", {})
    if not isinstance(layer_states, dict):
        raise RuntimeError("Checkpoint layer_states must be a dict")
    stats = payload.get("stats", {})
    if not isinstance(stats, dict):
        stats = {}

    artifact_target = str(config.get("artifact_target", "output_reconstruction")).strip().lower()
    block_domain = str(
        config.get("block_domain", "intermediate" if artifact_target == "intermediate_block_scores" else "output")
    ).strip().lower()
    block_size = int(config.get("block_size", 0))
    config_num_blocks = int(config.get("num_blocks", 0))
    model_shape = _load_model_shape(str(args.model_name), bool(args.local_files_only)) if args.model_name else {}

    errors: list[str] = []
    if args.expect_target != "any" and artifact_target != str(args.expect_target):
        errors.append(f"artifact_target={artifact_target!r}, expected {args.expect_target!r}")
    if args.expect_block_domain != "any" and block_domain != str(args.expect_block_domain):
        errors.append(f"block_domain={block_domain!r}, expected {args.expect_block_domain!r}")
    if block_size <= 0:
        errors.append(f"block_size must be > 0, got {block_size}")
    encoder_activation = str(config.get("encoder_activation", "linear")).strip().lower()
    if encoder_activation != "linear":
        errors.append(f"encoder_activation={encoder_activation!r}, expected 'linear'")
    recommended_execution = str(config.get("recommended_execution", "")).strip().lower()
    if artifact_target == "intermediate_block_scores" and recommended_execution not in {
        "",
        "exact_intermediate_sparse",
    }:
        errors.append(
            f"recommended_execution={recommended_execution!r}, expected 'exact_intermediate_sparse' "
            "for intermediate_block_scores"
        )
    if artifact_target == "output_reconstruction" and recommended_execution not in {
        "",
        "output_basis_surrogate",
    }:
        errors.append(
            f"recommended_execution={recommended_execution!r}, expected 'output_basis_surrogate' "
            "for output_reconstruction"
        )
    if int(args.expect_layers) > 0 and len(layer_states) != int(args.expect_layers):
        errors.append(f"layer count {len(layer_states)} != expected {int(args.expect_layers)}")
    if len(layer_states) < int(args.min_layers):
        errors.append(f"layer count {len(layer_states)} < required minimum {int(args.min_layers)}")

    layer_summaries: list[dict[str, Any]] = []
    for layer_key in sorted(layer_states.keys(), key=lambda value: int(value)):
        state = layer_states[layer_key]
        if not isinstance(state, dict):
            errors.append(f"layer {layer_key}: state must be a dict")
            continue
        layer_summary, layer_errors = _validate_layer(
            layer_key=str(layer_key),
            layer_state=state,
            stats=stats,
            config=config,
            model_shape=model_shape,
            artifact_target=artifact_target,
            block_domain=block_domain,
            block_size=block_size,
            config_num_blocks=config_num_blocks,
            min_explained_variance=float(args.min_explained_variance),
        )
        layer_summaries.append(layer_summary)
        errors.extend(layer_errors)

    evrs = [float(item["explained_variance_ratio"]) for item in layer_summaries if "explained_variance_ratio" in item]
    num_blocks = sorted({int(item["num_blocks"]) for item in layer_summaries if "num_blocks" in item})
    if int(args.expect_num_blocks) > 0 and num_blocks != [int(args.expect_num_blocks)]:
        errors.append(f"unique_num_blocks={num_blocks} != expected [{int(args.expect_num_blocks)}]")
    report: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "artifact_target": artifact_target,
        "block_domain": block_domain,
        "block_size": block_size,
        "config_num_blocks": config_num_blocks,
        "basis_rank": int(config.get("basis_rank", 0)),
        "basis_top_k": int(config.get("basis_top_k", 0)),
        "encoder_activation": encoder_activation,
        "recommended_execution": recommended_execution,
        "layers": len(layer_states),
        "unique_num_blocks": num_blocks,
        "model_shape": model_shape,
        "explained_variance": {
            "min": min(evrs) if evrs else None,
            "mean": (sum(evrs) / len(evrs)) if evrs else None,
            "max": max(evrs) if evrs else None,
        },
        "sample_layers": layer_summaries[: min(5, len(layer_summaries))],
        "errors": errors,
    }

    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered + "\n", encoding="utf-8")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

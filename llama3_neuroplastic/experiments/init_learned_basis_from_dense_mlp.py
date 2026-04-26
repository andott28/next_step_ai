from __future__ import annotations

import argparse
import json
import sys
from itertools import chain
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer

from llama3_neuroplastic.layer_selection import parse_layer_selection

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

try:
    from ..basis_fitting import fit_block_score_basis, fit_layer_basis
    from ..performance_utils import configure_runtime_environment, resolve_dataloader_kwargs
    from .init_kv_basis import _fit_kv_basis as _fit_kv_basis_fn
    from .streaming_llama_runtime import StreamingLlamaRuntime
except ImportError:
    from init_kv_basis import _fit_kv_basis as _fit_kv_basis_fn
    from streaming_llama_runtime import StreamingLlamaRuntime

    from llama3_neuroplastic.basis_fitting import fit_block_score_basis, fit_layer_basis
    from llama3_neuroplastic.performance_utils import configure_runtime_environment, resolve_dataloader_kwargs

NeuroplasticLlama = None


def _parse_layers(spec: str | None) -> list[int] | None:
    return parse_layer_selection(spec, all_as_none=True)


def _slice_layer_chunk(
    layers: list[int],
    *,
    chunk_size: int,
    chunk_index: int,
) -> list[int]:
    if int(chunk_size) <= 0:
        return list(layers)
    if int(chunk_index) < 0:
        raise ValueError("--layer-chunk-index must be >= 0")
    start = int(chunk_size) * int(chunk_index)
    end = start + int(chunk_size)
    return list(layers[start:end])


def _layers_for_output_save(
    *,
    selected_layers: list[int],
    layer_states: dict[str, dict[str, torch.Tensor]],
) -> list[int]:
    output_layers = {int(idx) for idx in selected_layers}
    output_layers.update(int(idx) for idx in layer_states)
    return sorted(output_layers)


def group_tokenized_texts(examples: dict[str, list], *, seq_len: int) -> dict[str, Any]:
    ids = list(chain.from_iterable(examples["input_ids"]))
    total = (len(ids) // int(seq_len)) * int(seq_len)
    if total == 0:
        return {"input_ids": [], "attention_mask": []}
    chunks = [ids[i : i + int(seq_len)] for i in range(0, total, int(seq_len))]
    return {
        "input_ids": chunks,
        "attention_mask": [[1] * int(seq_len) for _ in chunks],
    }


def _build_dataloader(
    tokenizer: Any,
    dataset_name: str,
    dataset_config: str,
    dataset_split: str,
    text_column: str,
    max_seq_length: int,
    batch_size: int,
    max_samples: int,
    dataloader_num_workers: int,
    dataloader_prefetch_factor: int,
    dataloader_persistent_workers: bool,
    min_text_chars: int = 200,
) -> DataLoader:
    if load_dataset is None:
        raise RuntimeError("datasets package is required")
    dataset = load_dataset(dataset_name, dataset_config, split=dataset_split)
    _min_chars = max(int(min_text_chars), 1)
    filtered = dataset.filter(
        lambda x: isinstance(x[text_column], str) and len(x[text_column].strip()) >= _min_chars
    )
    if len(filtered) == 0:

        print(
            f"[dataloader] WARNING: no examples with >= {_min_chars} chars in "
            f"'{dataset_name}/{dataset_config}'. Falling back to min_text_chars=1.",
            flush=True,
        )
        filtered = dataset.filter(
            lambda x: isinstance(x[text_column], str) and len(x[text_column].strip()) >= 1
        )
    if len(filtered) == 0:
        raise RuntimeError(
            f"Dataset '{dataset_name}/{dataset_config}' has no non-empty text in column '{text_column}'."
        )





    _raw_limit = max(int(max_samples) * 200 + 5000, 10000)
    if len(filtered) > _raw_limit:
        filtered = filtered.select(range(_raw_limit))


    def tokenize_fn(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(
            batch[text_column],
            truncation=False,
            padding=False,
        )

    tokenized = filtered.map(tokenize_fn, batched=True, remove_columns=filtered.column_names)





    _seq_len = int(max_seq_length)

    def group_texts(examples: dict[str, list]) -> dict[str, Any]:
        return group_tokenized_texts(examples, seq_len=_seq_len)

    grouped = tokenized.map(
        group_texts,
        batched=True,
        batch_size=1000,
        remove_columns=list(tokenized.column_names),
    )

    if max_samples > 0:
        grouped = grouped.select(range(min(int(max_samples), len(grouped))))

    if len(grouped) == 0:
        raise RuntimeError(
            f"Dataset produced 0 packed sequences of length {_seq_len}. "
            f"Try lowering --max-seq-length or increasing --max-samples."
        )

    grouped.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return DataLoader(
        grouped,
        **resolve_dataloader_kwargs(
            batch_size=int(batch_size),
            num_workers=int(dataloader_num_workers),
            prefetch_factor=int(dataloader_prefetch_factor),
            persistent_workers=bool(dataloader_persistent_workers),
        ),
    )


def _choose_layers(model: NeuroplasticLlama, explicit_layers: list[int] | None) -> list[int]:
    if explicit_layers is not None:
        return sorted(explicit_layers)
    total_layers = len(model.model.model.layers)
    active = [idx for idx in range(total_layers) if bool(model._sparse_layer_enabled(int(idx)))]
    return sorted(active if active else list(range(total_layers)))


def _choose_layers_from_layout(
    *,
    total_layers: int,
    explicit_layers: list[int] | None,
    bottom_buffer_layers: int,
    decode_guard_layers: int,
    dense_anchor_stride: int,
    top_buffer_layers: int = 2,
) -> list[int]:
    if explicit_layers is not None:
        return sorted(explicit_layers)
    lower = max(int(bottom_buffer_layers), 0)
    upper = int(total_layers) - max(int(top_buffer_layers), 0) - max(int(decode_guard_layers), 0)
    out: list[int] = []
    for layer_idx in range(int(total_layers)):
        if not (lower <= int(layer_idx) < max(upper, 0)):
            continue
        if int(dense_anchor_stride) > 0:
            active_offset = int(layer_idx) - lower + 1
            if active_offset > 0 and active_offset % int(dense_anchor_stride) == 0:
                continue
        out.append(int(layer_idx))
    return out


def _fit_layer_basis(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    basis_rank: int,
    block_size: int,
    pca_method: str,
    pca_batch_rows: int,
) -> dict[str, Any]:
    return fit_layer_basis(
        x=x,
        y=y,
        basis_rank=int(basis_rank),
        block_size=int(block_size),
        pca_method=str(pca_method),
        pca_batch_rows=int(pca_batch_rows),
    )


def _artifact_target_to_block_domain(artifact_target: str) -> str:
    target = str(artifact_target).strip().lower()
    if target == "output_reconstruction":
        return "output"
    if target == "intermediate_block_scores":
        return "intermediate"
    raise ValueError(f"Unsupported artifact target: {artifact_target}")


def _artifact_target_to_recommended_execution(artifact_target: str) -> str:
    target = str(artifact_target).strip().lower()
    if target == "output_reconstruction":
        return "output_basis_surrogate"
    if target == "intermediate_block_scores":
        return "exact_blockwise_sparse"
    raise ValueError(f"Unsupported artifact target: {artifact_target}")


def _fit_sparse_artifact(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    artifact_target: str,
    basis_rank: int,
    block_size: int,
    pca_method: str,
    pca_batch_rows: int,
) -> dict[str, Any]:
    target = str(artifact_target).strip().lower()
    if target == "output_reconstruction":
        return _fit_layer_basis(
            x=x,
            y=y,
            basis_rank=int(basis_rank),
            block_size=int(block_size),
            pca_method=str(pca_method),
            pca_batch_rows=int(pca_batch_rows),
        )
    if target == "intermediate_block_scores":
        return fit_block_score_basis(
            x=x,
            block_scores=y,
            basis_rank=int(basis_rank),
            pca_method=str(pca_method),
            pca_batch_rows=int(pca_batch_rows),
        )
    raise ValueError(f"Unsupported artifact target: {artifact_target}")


def _layer_state_from_fitted(*, artifact_target: str, fitted: dict[str, Any]) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {
        "encoder_weight": fitted["encoder_weight"],
        "encoder_bias": fitted["encoder_bias"],
        "scale": fitted["scale"],
    }
    target = str(artifact_target).strip().lower()
    if target == "output_reconstruction":
        state.update(
            {
                "decoder_blocks": fitted["decoder_blocks"],
                "decoder_bias": fitted["decoder_bias"],
            }
        )
    elif target == "intermediate_block_scores":
        state.update(
            {
                "score_weight": fitted["score_weight"],
                "score_bias": fitted["score_bias"],
                "block_importance": fitted["block_importance"],
            }
        )
    else:
        raise ValueError(f"Unsupported artifact target: {artifact_target}")
    state["samples"] = torch.tensor(float(fitted["samples"]), dtype=torch.float32)
    state["rank_effective"] = torch.tensor(float(fitted["rank_effective"]), dtype=torch.float32)
    state["explained_variance_ratio"] = torch.tensor(float(fitted["explained_variance_ratio"]), dtype=torch.float32)
    return state


def _load_profile_overrides(path: str) -> dict[str, dict[int, int]]:
    if not str(path).strip():
        return {
            "basis_rank_by_layer": {},
            "basis_top_k_by_layer": {},
            "top_k_by_layer": {},
        }
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    overrides = payload.get("recommended_overrides", {}) if isinstance(payload, dict) else {}
    out: dict[str, dict[int, int]] = {
        "basis_rank_by_layer": {},
        "basis_top_k_by_layer": {},
        "top_k_by_layer": {},
    }
    for key in out:
        raw = overrides.get(key, {}) if isinstance(overrides, dict) else {}
        if isinstance(raw, dict):
            out[key] = {int(layer_idx): int(value) for layer_idx, value in raw.items()}
    return out


def _collect_alignment_rows(
    model: NeuroplasticLlama,
    selected_layers: list[int],
    *,
    attention_mask: torch.Tensor,
    layer_x: dict[int, list[torch.Tensor]],
    layer_y: dict[int, list[torch.Tensor]],
    layer_rows: dict[int, int],
    max_rows: int,
) -> None:
    valid = attention_mask.reshape(-1).to(dtype=torch.bool)
    for layer_idx in selected_layers:
        if layer_rows[layer_idx] >= max_rows:
            continue
        align = model.sca_sparse_mlps[layer_idx].get_last_alignment()
        if align is None:
            continue
        mlp_input = align.get("mlp_input")
        dense_out = align.get("dense_mlp_out")
        if mlp_input is None or dense_out is None:
            continue
        x = mlp_input.reshape(-1, mlp_input.shape[-1])[valid]
        y = dense_out.reshape(-1, dense_out.shape[-1])[valid]
        if x.numel() == 0 or y.numel() == 0:
            continue
        remain = max_rows - layer_rows[layer_idx]
        take = int(min(remain, x.shape[0]))
        layer_x[layer_idx].append(x[:take].detach().cpu().float())
        layer_y[layer_idx].append(y[:take].detach().cpu().float())
        layer_rows[layer_idx] += take


def _build_basis_payload(
    *,
    args: argparse.Namespace,
    selected_layers: list[int],
    layer_states: dict[str, dict[str, torch.Tensor]],
    stats: dict[str, dict[str, float]],
    block_size: int,
    num_blocks: int,
    overrides: dict[str, dict[int, int]],
    existing_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifact_target = str(args.artifact_target).strip().lower()
    config_payload: dict[str, Any] = dict(existing_config or {})
    config_payload.update(
        {
            "sparse_placement": "learned_basis",
            "artifact_target": artifact_target,
            "block_domain": _artifact_target_to_block_domain(artifact_target),
            "encoder_activation": "linear",
            "recommended_execution": _artifact_target_to_recommended_execution(artifact_target),
            "routing_mode": str(config_payload.get("routing_mode", args.sca_routing_mode)),
            "bottom_buffer_layers": int(config_payload.get("bottom_buffer_layers", args.sca_bottom_buffer_layers)),
            "decode_guard_layers": int(config_payload.get("decode_guard_layers", args.sca_decode_guard_layers)),
            "dense_anchor_stride": int(config_payload.get("dense_anchor_stride", args.sca_dense_anchor_stride)),
            "basis_rank": int(config_payload.get("basis_rank", args.basis_rank)),


            "basis_top_k": int(config_payload.get("basis_top_k", args.basis_top_k)),
            "pca_method": str(config_payload.get("pca_method", args.pca_method)),
            "pca_batch_rows": int(config_payload.get("pca_batch_rows", args.pca_batch_rows)),
            "basis_rank_by_layer": config_payload.get("basis_rank_by_layer", overrides["basis_rank_by_layer"]),
            "basis_top_k_by_layer": config_payload.get("basis_top_k_by_layer", overrides["basis_top_k_by_layer"]),
            "top_k_by_layer": config_payload.get("top_k_by_layer", overrides["top_k_by_layer"]),
            "block_size": int(config_payload.get("block_size", block_size)),
            "num_blocks": int(config_payload.get("num_blocks", num_blocks)),
            "intermediate_block_score_metric": "mean_abs",
            "streaming_harness_used": bool(config_payload.get("streaming_harness_used", args.use_streaming_harness)),
        }
    )
    return {
        "model_name": str(args.model_name),
        "hybrid_checkpoint_path": str(args.hybrid_checkpoint),
        "config": config_payload,
        "layer_selection": list(selected_layers),
        "layer_states": layer_states,
        "stats": stats,
    }


def _save_basis_resume(
    *,
    resume_path: Path,
    layer_states: dict[str, dict[str, torch.Tensor]],
    stats: dict[str, dict[str, float]],
    layer_x: dict[int, list[torch.Tensor]],
    layer_y: dict[int, list[torch.Tensor]],
    selected_layers: list[int],
    include_buffers: bool,
    layer_kv_x: dict[int, list[torch.Tensor]] | None = None,
    layer_kv_rows: dict[int, int] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "layer_states": layer_states,
        "stats": stats,
    }
    if bool(include_buffers):
        payload["layer_x"] = {idx: layer_x[idx] for idx in selected_layers if layer_x[idx]}
        payload["layer_y"] = {idx: layer_y[idx] for idx in selected_layers if layer_y[idx]}
    if bool(include_buffers) and layer_kv_x is not None:
        payload["layer_kv_x"] = {idx: layer_kv_x[idx] for idx in selected_layers if layer_kv_x.get(idx)}
    if layer_kv_rows is not None:
        payload["layer_kv_rows"] = dict(layer_kv_rows)
    torch.save(payload, resume_path)


def _save_basis_output(
    *,
    output_path: Path,
    args: argparse.Namespace,
    selected_layers: list[int],
    layer_states: dict[str, dict[str, torch.Tensor]],
    stats: dict[str, dict[str, float]],
    block_size: int,
    num_blocks: int,
    overrides: dict[str, dict[int, int]],
    existing_config: dict[str, Any] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        _build_basis_payload(
            args=args,
            selected_layers=selected_layers,
            layer_states=layer_states,
            stats=stats,
            block_size=block_size,
            num_blocks=num_blocks,
            overrides=overrides,
            existing_config=existing_config,
        ),
        output_path,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Initialize learned-basis sparse MLP from dense MLP activations.")
    p.add_argument("--model-name", type=str, required=True)
    p.add_argument("--hybrid-checkpoint", type=str, default="")
    p.add_argument("--output-path", type=str, required=True)
    p.add_argument("--layers", type=str, default="")
    p.add_argument("--max-samples", type=int, default=256)
    p.add_argument("--max-seq-length", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-rows-per-layer", type=int, default=1024)
    p.add_argument("--basis-rank", type=int, default=64)
    p.add_argument("--basis-top-k", type=int, default=64)
    p.add_argument("--pca-method", type=str, default="auto", choices=["auto", "lowrank", "incremental"])
    p.add_argument("--pca-batch-rows", type=int, default=1024)
    p.add_argument(
        "--artifact-target",
        type=str,
        default="intermediate_block_scores",
        choices=["output_reconstruction", "intermediate_block_scores"],
        help="Artifact to fit. output_reconstruction keeps the legacy output-space surrogate. "
             "intermediate_block_scores fits a router over true FFN intermediate block scores "
             "for exact sparse gate/up/down execution.",
    )
    p.add_argument("--sca-block-size", type=int, default=32)
    p.add_argument("--sca-bottom-buffer-layers", type=int, default=2)
    p.add_argument("--sca-decode-guard-layers", type=int, default=12)
    p.add_argument("--sca-dense-anchor-stride", type=int, default=0)
    p.add_argument("--dataloader-num-workers", type=int, default=0)
    p.add_argument("--dataloader-prefetch-factor", type=int, default=2)
    p.add_argument("--dataloader-persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dense-rollout-tokens", type=int, default=8)
    p.add_argument("--resume-save-every-batches", type=int, default=1)
    p.add_argument(
        "--resume-save-buffers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Persist raw collected activation buffers in .resume.pt files. Disabled by default.",
    )
    p.add_argument("--write-partial-output-every-batches", type=int, default=1)
    p.add_argument("--max-batches", type=int, default=0)
    p.add_argument("--only-missing-from-output", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--no-resume", action="store_true", default=False,
                   help="Ignore any existing checkpoint and start data collection from scratch.")
    p.add_argument("--layer-chunk-size", type=int, default=0)
    p.add_argument("--layer-chunk-index", type=int, default=0)
    p.add_argument("--plan-only", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--profile-path", type=str, default="")
    p.add_argument("--use-streaming-harness", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--taylor-layers", type=str, default="")
    p.add_argument(
        "--taylor-feature-map",
        type=str,
        default="hybrid_performer",
        choices=["hybrid_performer", "elu", "taylor"],
    )
    p.add_argument("--taylor-local-window", type=int, default=64)
    p.add_argument("--taylor-feature-dim", type=int, default=64)
    p.add_argument("--taylor-state-decay", type=float, default=1.0)
    p.add_argument(
        "--sca-routing-mode",
        type=str,
        choices=["spatial_grid", "semantic_latent"],
        default="semantic_latent",
    )
    p.add_argument("--dataset-name", type=str, default="wikitext")
    p.add_argument("--dataset-config", type=str, default="wikitext-2-raw-v1")
    p.add_argument("--dataset-split", type=str, default="train")
    p.add_argument("--text-column", type=str, default="text")
    p.add_argument("--min-text-chars", type=int, default=200,
                   help="Minimum character count (after strip) to keep a calibration sample. "
                        "Filters out blank lines and short section headers in wikitext-style datasets.")
    p.add_argument(
        "--kv-basis-output-path", type=str, default="",
        help="If set, K/V column-block routing basis is fitted in the same forward pass "
             "and saved to this path. Only supported with --use-streaming-harness.",
    )
    p.add_argument("--kv-basis-rank", type=int, default=32,
                   help="Encoder rank for K/V router (default 32).")
    p.add_argument("--kv-basis-top-k", type=int, default=51,
                   help="Active column-blocks at inference (default 51 = 10%% of 512).")
    p.add_argument("--kv-max-rows", type=int, default=256,
                   help="Max rows to collect per layer for KV basis fitting (default 256).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if int(args.basis_top_k) <= 0 or int(args.basis_top_k) > int(args.basis_rank):
        raise RuntimeError("--basis-top-k must be in [1, --basis-rank]")
    configure_runtime_environment(cudnn_benchmark=True)
    if str(args.artifact_target).strip().lower() == "intermediate_block_scores" and not bool(args.use_streaming_harness):
        raise RuntimeError(
            "--artifact-target intermediate_block_scores currently requires --use-streaming-harness "
            "so exact intermediate block scores can be collected from the streamed dense MLP."
        )
    if bool(args.use_streaming_harness) and int(args.batch_size) != 1:
        raise RuntimeError("Streaming harness mode currently requires --batch-size 1")
    if (not bool(args.use_streaming_harness)) and not str(args.hybrid_checkpoint).strip():
        raise RuntimeError("--hybrid-checkpoint is required unless --use-streaming-harness is enabled")
    _output_path = Path(args.output_path)
    _resume_path = _output_path.with_suffix(".resume.pt")
    _existing_output: dict[str, Any] = {}
    if _output_path.exists():
        try:
            _loaded_output = torch.load(_output_path, map_location="cpu")
            if isinstance(_loaded_output, dict):
                _existing_output = _loaded_output
        except Exception:
            print(f"[warn] Could not load existing output at {_output_path} (corrupt?), starting fresh.")
            _output_path.unlink(missing_ok=True)
    _existing_config = (
        dict(_existing_output.get("config", {})) if isinstance(_existing_output.get("config", {}), dict) else {}
    )
    _existing_layer_selection = [int(idx) for idx in _existing_output.get("layer_selection", [])]
    _effective_bottom_buffer_layers = int(
        _existing_config.get("bottom_buffer_layers", args.sca_bottom_buffer_layers)
    )
    _effective_decode_guard_layers = int(
        _existing_config.get("decode_guard_layers", args.sca_decode_guard_layers)
    )
    _effective_dense_anchor_stride = int(
        _existing_config.get("dense_anchor_stride", args.sca_dense_anchor_stride)
    )
    _effective_routing_mode = str(_existing_config.get("routing_mode", args.sca_routing_mode))
    _effective_block_size = int(_existing_config.get("block_size", args.sca_block_size))
    _effective_basis_rank = int(_existing_config.get("basis_rank", args.basis_rank))
    _effective_basis_top_k = int(_existing_config.get("basis_top_k", args.basis_top_k))
    _effective_pca_method = str(_existing_config.get("pca_method", args.pca_method))
    _effective_pca_batch_rows = int(_existing_config.get("pca_batch_rows", args.pca_batch_rows))
    _artifact_target = str(_existing_config.get("artifact_target", args.artifact_target)).strip().lower()
    if bool(args.plan_only) and bool(args.use_streaming_harness):
        config = AutoConfig.from_pretrained(
            args.model_name,
            local_files_only=bool(args.local_files_only),
        )
        _explicit_layers = _parse_layers(args.layers)
        if _explicit_layers is not None:
            requested_layers = list(_explicit_layers)
        elif _existing_layer_selection:
            requested_layers = list(_existing_layer_selection)
        else:
            requested_layers = _choose_layers_from_layout(
                total_layers=int(config.num_hidden_layers),
                explicit_layers=None,
                bottom_buffer_layers=_effective_bottom_buffer_layers,
                decode_guard_layers=_effective_decode_guard_layers,
                dense_anchor_stride=_effective_dense_anchor_stride,
            )
        layer_states: dict[str, dict[str, torch.Tensor]] = {}
        if not bool(args.no_resume) and _resume_path.exists():
            _resume_data = torch.load(_resume_path, map_location="cpu")
            layer_states = _resume_data.get("layer_states", {})
        elif not bool(args.no_resume) and _existing_output:
            layer_states = _existing_output.get("layer_states", {})
        completed_layers = sorted(int(_k) for _k in layer_states)
        selected_layers = list(requested_layers)
        if bool(args.only_missing_from_output):
            selected_layers = [int(idx) for idx in selected_layers if int(idx) not in set(completed_layers)]
        selected_layers = _slice_layer_chunk(
            selected_layers,
            chunk_size=int(args.layer_chunk_size),
            chunk_index=int(args.layer_chunk_index),
        )
        print(
            json.dumps(
                {
                    "output_path": str(_output_path),
                    "requested_layers": requested_layers,
                    "completed_layers": completed_layers,
                    "selected_layers": selected_layers,
                    "remaining_layers": [
                        int(idx) for idx in requested_layers if int(idx) not in set(completed_layers)
                    ],
                    "chunk_size": int(args.layer_chunk_size),
                    "chunk_index": int(args.layer_chunk_index),
                    "max_batches": int(args.max_batches),
                },
                indent=2,
            )
        )
        return

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        local_files_only=bool(args.local_files_only),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataloader = _build_dataloader(
        tokenizer=tokenizer,
        dataset_name=str(args.dataset_name),
        dataset_config=str(args.dataset_config),
        dataset_split=str(args.dataset_split),
        text_column=str(args.text_column),
        max_seq_length=int(args.max_seq_length),
        batch_size=int(args.batch_size),
        max_samples=int(args.max_samples),
        dataloader_num_workers=int(args.dataloader_num_workers),
        dataloader_prefetch_factor=int(args.dataloader_prefetch_factor),
        dataloader_persistent_workers=bool(args.dataloader_persistent_workers),
        min_text_chars=int(args.min_text_chars),
    )

    artifact: dict[str, Any] = {}
    model: NeuroplasticLlama | None = None
    runtime: StreamingLlamaRuntime | None = None
    _batch_count: int = 0
    if bool(args.use_streaming_harness):
        parsed_taylor_layers = _parse_layers(args.taylor_layers)
        runtime = StreamingLlamaRuntime(
            model_name_or_path=str(args.model_name),
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            dtype=torch.float16,
            taylor_layers=[] if parsed_taylor_layers is None else parsed_taylor_layers,
            taylor_feature_map=str(args.taylor_feature_map),
            taylor_local_window=int(args.taylor_local_window),
            taylor_feature_dim=int(args.taylor_feature_dim),
            taylor_state_decay=float(args.taylor_state_decay),
            local_files_only=bool(args.local_files_only),
            ram_cache=False,
            materialize_lm_head=False,
            sparse_basis_path=str(_output_path) if _existing_output and _existing_output.get("layer_states") else None,
            vram_hot_cache_gb=0.0,
        )
        _explicit_layers = _parse_layers(args.layers)
        if _explicit_layers is not None:
            selected_layers = list(_explicit_layers)
        elif _existing_layer_selection:
            selected_layers = list(_existing_layer_selection)
        else:
            selected_layers = _choose_layers_from_layout(
                total_layers=int(runtime.num_layers),
                explicit_layers=None,
                bottom_buffer_layers=_effective_bottom_buffer_layers,
                decode_guard_layers=_effective_decode_guard_layers,
                dense_anchor_stride=_effective_dense_anchor_stride,
            )
        block_size = int(_effective_block_size)
        if _artifact_target == "output_reconstruction":
            hidden_size = int(runtime.config.hidden_size)
            num_blocks = int(hidden_size // max(block_size, 1))
            if num_blocks * block_size != hidden_size:
                raise RuntimeError("hidden_size must be divisible by --sca-block-size for learned-basis init")
        elif _artifact_target == "intermediate_block_scores":
            intermediate_size = int(getattr(runtime.config, "intermediate_size", 0))
            num_blocks = int(intermediate_size // max(block_size, 1))
            if num_blocks * block_size != intermediate_size:
                raise RuntimeError(
                    "intermediate_size must be divisible by --sca-block-size for intermediate block-score routing init"
                )
        else:
            raise RuntimeError(f"Unsupported artifact target: {_artifact_target}")
    else:
        if NeuroplasticLlama is None:
            raise RuntimeError(
                "The legacy NeuroplasticLlama fitting path is unavailable. "
                "Use --use-streaming-harness for the maintained streaming basis fitting path."
            )
        artifact = torch.load(args.hybrid_checkpoint, map_location="cpu")
        if not isinstance(artifact, dict):
            raise RuntimeError("Hybrid checkpoint must be a dict artifact")

        model = NeuroplasticLlama(
            model_name=args.model_name,
            neuroplasticity_enabled=True,
            sca_use_cuda=True,
            sca_spmm_impl="cuda_spmm",
            sca_block_size=int(_effective_block_size),
            sca_basis_rank=int(_effective_basis_rank),
            sca_basis_top_k=int(_effective_basis_top_k),
            sca_routing_mode=str(_effective_routing_mode),
            sca_bottom_buffer_layers=int(_effective_bottom_buffer_layers),
            sca_dense_anchor_stride=int(_effective_dense_anchor_stride),
            sca_decode_guard_layers=int(_effective_decode_guard_layers),
            sca_sparse_placement="learned_basis",
            sca_stability_dense_fallback_threshold=0.0,
            attention_hybrid_enabled=True,
            attention_hybrid_layers=artifact.get("layer_selection"),
            attention_hybrid_target_rank=artifact.get("target_rank", 16),
            attention_hybrid_variance_threshold=float(artifact.get("variance_threshold", 0.90)),
            attention_hybrid_state_dim=artifact.get("state_dim"),
            attention_hybrid_force_no_cache=True,
        )
        model.load_hybrid_attention_state(str(args.hybrid_checkpoint), strict=False)
        model.disable_task_bias_injection = True
        selected_layers = _choose_layers(model, _parse_layers(args.layers))
        model.set_mlp_alignment_capture(True, selected_layers)
        model.neuroplasticity_enabled = False
        model.eval()
        if _artifact_target != "output_reconstruction":
            raise RuntimeError(
                "--artifact-target intermediate_block_scores is currently supported only with --use-streaming-harness"
            )
        block_size = int(model.sca_config.block_size)
        num_blocks = int(model.sca_config.num_blocks)

    requested_layers = list(selected_layers)
    layer_states: dict[str, dict[str, torch.Tensor]] = {}
    stats: dict[str, dict[str, float]] = {}
    _loaded_layer_x: dict[int, list[torch.Tensor]] = {}
    _loaded_layer_y: dict[int, list[torch.Tensor]] = {}
    _loaded_layer_kv_x: dict[int, list[torch.Tensor]] = {}
    _loaded_layer_kv_rows: dict[int, int] = {}
    if not bool(args.no_resume) and _resume_path.exists():
        _resume_data = torch.load(_resume_path, map_location="cpu")
        layer_states = _resume_data.get("layer_states", {})
        stats = _resume_data.get("stats", {})
        _loaded_layer_x = {int(_k): _xs for _k, _xs in _resume_data.get("layer_x", {}).items()}
        _loaded_layer_y = {int(_k): _ys for _k, _ys in _resume_data.get("layer_y", {}).items()}
        _loaded_layer_kv_x = {int(_k): _xs for _k, _xs in _resume_data.get("layer_kv_x", {}).items()}
        _loaded_layer_kv_rows = {int(_k): int(_v) for _k, _v in _resume_data.get("layer_kv_rows", {}).items()}
        _kv_resumed = sum(v for v in _loaded_layer_kv_rows.values())
        print(f"[resume] loaded {len(layer_states)}/{len(selected_layers)} fitted layers, "
              f"{sum(1 for idx in _loaded_layer_x if int(idx) not in {int(s) for s in layer_states})} "
              f"partially collected, {_kv_resumed} kv rows, from {_resume_path}")
    elif not bool(args.no_resume) and _output_path.exists():
        _resume_data = torch.load(_output_path, map_location="cpu")
        layer_states = _resume_data.get("layer_states", {})
        stats = _resume_data.get("stats", {})
        print(
            f"[resume] loaded {len(layer_states)}/{len(selected_layers)} fitted layers from {_output_path}",
            flush=True,
        )

    _completed_layer_ids = {int(_k) for _k in layer_states}
    if bool(args.only_missing_from_output):
        selected_layers = [int(idx) for idx in selected_layers if int(idx) not in _completed_layer_ids]
    selected_layers = _slice_layer_chunk(
        selected_layers,
        chunk_size=int(args.layer_chunk_size),
        chunk_index=int(args.layer_chunk_index),
    )
    if bool(args.plan_only):
        print(
            json.dumps(
                {
                    "output_path": str(_output_path),
                    "requested_layers": requested_layers,
                    "completed_layers": sorted(_completed_layer_ids),
                    "selected_layers": selected_layers,
                    "remaining_layers": [
                        int(idx) for idx in requested_layers if int(idx) not in _completed_layer_ids
                    ],
                    "chunk_size": int(args.layer_chunk_size),
                    "chunk_index": int(args.layer_chunk_index),
                    "max_batches": int(args.max_batches),
                },
                indent=2,
            )
        )
        return
    if not selected_layers:
        print(
            json.dumps(
                {
                    "output_path": str(_output_path),
                    "layers_initialized": len(layer_states),
                    "selected_layers": [],
                    "status": "no pending layers selected",
                },
                indent=2,
            )
        )
        return

    layer_x: dict[int, list[torch.Tensor]] = {int(idx): [] for idx in selected_layers}
    layer_y: dict[int, list[torch.Tensor]] = {int(idx): [] for idx in selected_layers}
    layer_rows: dict[int, int] = {int(idx): 0 for idx in selected_layers}


    _kv_output_path: Path | None = (
        Path(str(args.kv_basis_output_path))
        if bool(args.use_streaming_harness) and str(args.kv_basis_output_path).strip()
        else None
    )
    _do_kv = _kv_output_path is not None
    layer_kv_x: dict[int, list[torch.Tensor]] = {int(idx): [] for idx in selected_layers} if _do_kv else {}
    layer_kv_rows: dict[int, int] = {int(idx): 0 for idx in selected_layers} if _do_kv else {}
    kv_layer_states: dict[str, Any] = {}
    max_rows = int(max(args.max_rows_per_layer, 32))
    _kv_max_rows = int(max(args.kv_max_rows, 32)) if _do_kv else max_rows
    rollout_tokens = int(max(args.dense_rollout_tokens, 1))
    for _k in layer_states:
        if int(_k) in layer_rows:
            layer_rows[int(_k)] = max_rows
    for _k, _xs in _loaded_layer_x.items():
        if int(_k) in layer_x and int(_k) not in _completed_layer_ids:
            layer_x[int(_k)] = _xs
            layer_rows[int(_k)] = sum(t.shape[0] for t in _xs)
    for _k, _ys in _loaded_layer_y.items():
        if int(_k) in layer_y and int(_k) not in _completed_layer_ids:
            layer_y[int(_k)] = _ys
    if _do_kv:
        for _k, _xs in _loaded_layer_kv_x.items():
            if int(_k) in layer_kv_x:
                layer_kv_x[int(_k)] = _xs
        for _k, _v in _loaded_layer_kv_rows.items():
            if int(_k) in layer_kv_rows:
                layer_kv_rows[int(_k)] = int(_v)

    _resume_save_every_batches = max(int(args.resume_save_every_batches), 0)
    _partial_output_every_batches = max(int(args.write_partial_output_every_batches), 0)
    overrides = _load_profile_overrides(str(args.profile_path))















    _fit_threshold = max(int(_effective_basis_rank) - 1, 1)





    _EVR_LOW: float = 0.65
    _EVR_HIGH: float = 0.85
    _layer_next_fit_at: dict[int, int] = {int(idx): _fit_threshold for idx in selected_layers}


    for batch in dataloader:
        if runtime is not None:
            runtime.collect_dense_mlp_rows(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                selected_layers=selected_layers,
                layer_x=layer_x,
                layer_y=layer_y,
                layer_rows=layer_rows,
                max_rows=max_rows,
                artifact_target=_artifact_target,
                layer_kv_x=layer_kv_x if _do_kv else None,
                layer_kv_rows=layer_kv_rows if _do_kv else None,
                kv_max_rows=_kv_max_rows,
            )
        else:
            assert model is not None
            input_ids = batch["input_ids"].to(model.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(model.device, non_blocking=True)
            with torch.no_grad():
                _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                    task_id=0,
                )
            _collect_alignment_rows(
                model,
                selected_layers,
                attention_mask=attention_mask,
                layer_x=layer_x,
                layer_y=layer_y,
                layer_rows=layer_rows,
                max_rows=max_rows,
            )
            if not all(v >= max_rows for v in layer_rows.values()):
                valid_lengths = attention_mask.sum(dim=-1).tolist()
                for row_idx, valid_len in enumerate(valid_lengths):
                    prefix_len = int(max(1, min(64, int(valid_len) - rollout_tokens)))
                    prefix_ids = input_ids[row_idx : row_idx + 1, :prefix_len]
                    prefix_mask = attention_mask[row_idx : row_idx + 1, :prefix_len]
                    with torch.no_grad():
                        generated = model.generate(
                            input_ids=prefix_ids,
                            attention_mask=prefix_mask,
                            max_new_tokens=rollout_tokens,
                            do_sample=False,
                            use_cache=True,
                            task_id=0,
                        )
                        generated_mask = torch.ones_like(generated, device=model.device)
                        _ = model(
                            input_ids=generated,
                            attention_mask=generated_mask,
                            use_cache=False,
                            return_dict=True,
                            task_id=0,
                        )
                    _collect_alignment_rows(
                        model,
                        selected_layers,
                        attention_mask=generated_mask,
                        layer_x=layer_x,
                        layer_y=layer_y,
                        layer_rows=layer_rows,
                        max_rows=max_rows,
                    )
                    if all(v >= max_rows for v in layer_rows.values()):
                        break
        _batch_count += 1
        _rows_done = sum(1 for idx in selected_layers if str(idx) in layer_states)
        _rows_partial = sum(1 for idx in selected_layers if layer_rows[idx] > 0 and str(idx) not in layer_states)
        _saved_resume = False
        _saved_output = False
        if _resume_save_every_batches > 0 and _batch_count % _resume_save_every_batches == 0:
            _save_basis_resume(
                resume_path=_resume_path,
                layer_states=layer_states,
                stats=stats,
                layer_x=layer_x,
                layer_y=layer_y,
                selected_layers=selected_layers,
                include_buffers=bool(args.resume_save_buffers),
                layer_kv_x=layer_kv_x if _do_kv else None,
                layer_kv_rows=layer_kv_rows if _do_kv else None,
            )
            _saved_resume = True
        if _partial_output_every_batches > 0 and _batch_count % _partial_output_every_batches == 0:
            _save_basis_output(
                output_path=_output_path,
                args=args,
                selected_layers=_layers_for_output_save(selected_layers=selected_layers, layer_states=layer_states),
                layer_states=layer_states,
                stats=stats,
                block_size=block_size,
                num_blocks=num_blocks,
                overrides=overrides,
                existing_config=_existing_config,
            )
            _saved_output = True
        _min_rows = min((layer_rows[idx] for idx in selected_layers if str(idx) not in layer_states), default=0)
        _max_rows_seen = max((layer_rows[idx] for idx in selected_layers if str(idx) not in layer_states), default=0)
        _unfitted_idxs = [idx for idx in selected_layers if str(idx) not in layer_states]
        _next_fit_min = min((_layer_next_fit_at[idx] for idx in _unfitted_idxs), default=0)
        _row_info = f" rows=[{_min_rows}..{_max_rows_seen}] next_fit>={_next_fit_min}"
        if _do_kv:
            _kv_min = min((layer_kv_rows.get(int(idx), 0) for idx in selected_layers), default=0)
            _kv_max_seen = max((layer_kv_rows.get(int(idx), 0) for idx in selected_layers), default=0)
            _kv_pending = sum(1 for idx in selected_layers if layer_kv_rows.get(int(idx), 0) < _kv_max_rows)
            _row_info += f" kv=[{_kv_min}..{_kv_max_seen}] kv_pending={_kv_pending}"
        if _saved_resume and _saved_output:
            print(f"[pass done] {_rows_done} fitted, {_rows_partial} collecting{_row_info} [res+out]", flush=True)
        elif _saved_resume:
            print(f"[pass done] {_rows_done} fitted, {_rows_partial} collecting{_row_info} [res]", flush=True)
        elif _saved_output:
            print(f"[pass done] {_rows_done} fitted, {_rows_partial} collecting{_row_info} [out]", flush=True)
        else:
            print(f"[pass done] {_rows_done} fitted, {_rows_partial} collecting{_row_info}", flush=True)








        for layer_idx in selected_layers:
            if str(layer_idx) in layer_states or layer_rows[layer_idx] < _layer_next_fit_at[layer_idx]:
                continue
            _xs = layer_x[layer_idx]
            _ys = layer_y[layer_idx]
            if not _xs or not _ys:
                continue
            _x = torch.cat(_xs, dim=0)
            _y = torch.cat(_ys, dim=0)
            _fitted = _fit_sparse_artifact(
                _x,
                _y,
                artifact_target=_artifact_target,
                basis_rank=int(_effective_basis_rank),
                block_size=int(block_size),
                pca_method=str(_effective_pca_method),
                pca_batch_rows=int(_effective_pca_batch_rows),
            )
            _evr = float(_fitted.get("explained_variance_ratio", 1.0))
            _at_capacity = layer_rows[layer_idx] >= max_rows
            if _evr >= _EVR_LOW or _at_capacity:

                _evr_tag = (
                    "excellent" if _evr >= _EVR_HIGH
                    else ("acceptable" if _evr >= _EVR_LOW else "best-effort")
                )
                layer_states[str(layer_idx)] = _layer_state_from_fitted(
                    artifact_target=_artifact_target,
                    fitted=_fitted,
                )
                stats[str(layer_idx)] = {
                    "samples": float(_fitted["samples"]),
                    "rank_effective": float(_fitted["rank_effective"]),
                    "explained_variance_ratio": _evr,
                    "pca_method": str(_fitted.get("pca_method", _effective_pca_method)),
                }

                layer_rows[layer_idx] = max_rows
                layer_x[layer_idx].clear()
                layer_y[layer_idx].clear()
                _save_basis_resume(
                    resume_path=_resume_path,
                    layer_states=layer_states,
                    stats=stats,
                    layer_x=layer_x,
                    layer_y=layer_y,
                    selected_layers=selected_layers,
                    include_buffers=bool(args.resume_save_buffers),
                    layer_kv_x=layer_kv_x if _do_kv else None,
                    layer_kv_rows=layer_kv_rows if _do_kv else None,
                )
                _save_basis_output(
                    output_path=_output_path,
                    args=args,
                    selected_layers=_layers_for_output_save(selected_layers=selected_layers, layer_states=layer_states),
                    layer_states=layer_states,
                    stats=stats,
                    block_size=block_size,
                    num_blocks=num_blocks,
                    overrides=overrides,
                    existing_config=_existing_config,
                )
                pct_done = int(len(layer_states) * 100 // max(len(selected_layers), 1))
                print(
                    f"[checkpoint] layer {layer_idx}: EVR={_evr:.3f} ({_evr_tag}),"
                    f" {_fitted['samples']} samples —"
                    f" {len(layer_states)}/{len(selected_layers)} fitted ({pct_done}%)"
                    f" — resume + partial output saved",
                    flush=True,
                )
            else:


                _next_milestone = min(layer_rows[layer_idx] * 2, max_rows)
                _layer_next_fit_at[layer_idx] = _next_milestone
                print(
                    f"[refit] layer {layer_idx}: EVR={_evr:.3f} < {_EVR_LOW} with"
                    f" {layer_rows[layer_idx]} rows — keeping data, re-fit at {_next_milestone} rows",
                    flush=True,
                )
        if int(args.max_batches) > 0 and _batch_count >= int(args.max_batches):
            print(f"[bounded] reached max_batches={int(args.max_batches)}; stopping early", flush=True)
            break
        _mlp_done = all(v >= max_rows for v in layer_rows.values())
        _kv_done = (
            not _do_kv
            or all(layer_kv_rows.get(int(idx), 0) >= _kv_max_rows for idx in selected_layers)
        )
        if _mlp_done and _kv_done:
            break


    for layer_idx in selected_layers:
        if str(layer_idx) in layer_states:
            continue
        xs = layer_x[layer_idx]
        ys = layer_y[layer_idx]
        if not xs or not ys:
            print(
                f"WARNING: layer {layer_idx} collected no rows and will be absent from the checkpoint.",
                file=sys.stderr,
            )
            continue
        x = torch.cat(xs, dim=0)
        y = torch.cat(ys, dim=0)
        fitted = _fit_sparse_artifact(
            x=x,
            y=y,
            artifact_target=_artifact_target,
            basis_rank=int(_effective_basis_rank),
            block_size=int(block_size),
            pca_method=str(_effective_pca_method),
            pca_batch_rows=int(_effective_pca_batch_rows),
        )
        layer_states[str(layer_idx)] = _layer_state_from_fitted(
            artifact_target=_artifact_target,
            fitted=fitted,
        )
        stats[str(layer_idx)] = {
            "samples": float(fitted["samples"]),
            "rank_effective": float(fitted["rank_effective"]),
            "explained_variance_ratio": float(fitted["explained_variance_ratio"]),
            "pca_method": str(fitted.get("pca_method", _effective_pca_method)),
        }
    missing_layers = [idx for idx in selected_layers if str(idx) not in layer_states]
    if missing_layers:
        print(
            f"WARNING: {len(missing_layers)}/{len(selected_layers)} selected layers have no data"
            f" and are absent from the checkpoint: {missing_layers}",
            file=sys.stderr,
        )

    _save_basis_output(
        output_path=_output_path,
        args=args,
        selected_layers=_layers_for_output_save(selected_layers=selected_layers, layer_states=layer_states),
        layer_states=layer_states,
        stats=stats,
        block_size=block_size,
        num_blocks=num_blocks,
        overrides=overrides,
        existing_config=_existing_config,
    )
    print(json.dumps({"output_path": str(_output_path), "layers_initialized": len(layer_states), "stats": stats}, indent=2))


    if _do_kv and _kv_output_path is not None and runtime is not None:
        print("[kv_basis] fitting K/V routing basis from co-collected activations...", flush=True)
        _kv_hidden_size = int(getattr(runtime.config, "hidden_size", 16384))
        _kv_kv_hidden = int(
            getattr(runtime.config, "num_key_value_heads", 8)
            * getattr(runtime.config, "head_dim", 128)
        )
        _kv_block_size = 32
        _kv_num_col_blocks = _kv_hidden_size // _kv_block_size
        for layer_idx in selected_layers:
            xs = layer_kv_x.get(int(layer_idx), [])
            if not xs:
                print(f"[kv_basis] layer {layer_idx}: no data — skipping", flush=True)
                continue
            x = torch.cat(xs, dim=0)
            print(f"[kv_basis] layer {layer_idx}: fitting on {int(x.shape[0])} rows...", flush=True)
            fitted_kv = _fit_kv_basis_fn(
                x=x,
                kv_out=torch.zeros(int(x.shape[0]), _kv_kv_hidden),
                basis_rank=int(args.kv_basis_rank),
                block_size=_kv_block_size,
                hidden_size=_kv_hidden_size,
                pca_method=str(_effective_pca_method),
                pca_batch_rows=int(_effective_pca_batch_rows),
            )
            kv_layer_states[str(layer_idx)] = {
                "encoder_weight":  fitted_kv["encoder_weight"],
                "encoder_bias":    fitted_kv["encoder_bias"],
                "decoder_blocks":  fitted_kv["decoder_blocks"],
                "decoder_bias":    fitted_kv["decoder_bias"],
                "block_importance": fitted_kv["block_importance"],
            }
        _kv_output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": {
                    "hidden_size":      _kv_hidden_size,
                    "kv_hidden":        _kv_kv_hidden,
                    "block_size":       _kv_block_size,
                    "num_col_blocks":   _kv_num_col_blocks,
                    "basis_rank":       int(args.kv_basis_rank),
                    "top_k":            int(args.kv_basis_top_k),
                    "separate_k_v_routing": False,
                },
                "layer_states": kv_layer_states,
            },
            _kv_output_path,
        )
        print(f"[kv_basis] saved {len(kv_layer_states)} layers → {_kv_output_path}", flush=True)


if __name__ == "__main__":
    main()

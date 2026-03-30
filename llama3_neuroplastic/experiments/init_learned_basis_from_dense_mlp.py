from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover
    load_dataset = None

try:
    from ..basis_fitting import fit_layer_basis
    from ..performance_utils import configure_runtime_environment, resolve_dataloader_kwargs
    from .neuroplastic_llama_gqa_mamba import NeuroplasticLlama
    from .streaming_llama_runtime import StreamingLlamaRuntime
except ImportError:  # pragma: no cover
    from llama3_neuroplastic.basis_fitting import fit_layer_basis
    from llama3_neuroplastic.performance_utils import configure_runtime_environment, resolve_dataloader_kwargs
    from neuroplastic_llama_gqa_mamba import NeuroplasticLlama
    from streaming_llama_runtime import StreamingLlamaRuntime


def _parse_layers(spec: str | None) -> Optional[List[int]]:
    if spec is None or spec.strip() == "":
        return None
    out: set[int] = set()
    for part in spec.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            s, e = token.split("-", 1)
            start = int(s)
            end = int(e)
            if end < start:
                raise ValueError(f"Invalid layer range: {token}")
            out.update(range(start, end + 1))
        else:
            out.add(int(token))
    return sorted(out)


def _slice_layer_chunk(
    layers: List[int],
    *,
    chunk_size: int,
    chunk_index: int,
) -> List[int]:
    if int(chunk_size) <= 0:
        return list(layers)
    if int(chunk_index) < 0:
        raise ValueError("--layer-chunk-index must be >= 0")
    start = int(chunk_size) * int(chunk_index)
    end = start + int(chunk_size)
    return list(layers[start:end])


def _layers_for_output_save(
    *,
    selected_layers: List[int],
    layer_states: Dict[str, Dict[str, torch.Tensor]],
) -> List[int]:
    output_layers = {int(idx) for idx in selected_layers}
    output_layers.update(int(idx) for idx in layer_states.keys())
    return sorted(output_layers)


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
) -> DataLoader:
    if load_dataset is None:
        raise RuntimeError("datasets package is required")
    dataset = load_dataset(dataset_name, dataset_config, split=dataset_split)
    if max_samples > 0:
        dataset = dataset.select(range(min(int(max_samples), len(dataset))))
    dataset = dataset.filter(lambda x: isinstance(x[text_column], str) and len(x[text_column].strip()) > 0)

    def tokenize_fn(batch: Dict[str, List[str]]) -> Dict[str, Any]:
        return tokenizer(
            batch[text_column],
            truncation=True,
            padding="max_length",
            max_length=max_seq_length,
        )

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return DataLoader(
        tokenized,
        **resolve_dataloader_kwargs(
            batch_size=int(batch_size),
            num_workers=int(dataloader_num_workers),
            prefetch_factor=int(dataloader_prefetch_factor),
            persistent_workers=bool(dataloader_persistent_workers),
        ),
    )


def _choose_layers(model: NeuroplasticLlama, explicit_layers: Optional[List[int]]) -> List[int]:
    if explicit_layers is not None:
        return sorted(explicit_layers)
    total_layers = len(model.model.model.layers)
    active = [idx for idx in range(total_layers) if bool(model._sparse_layer_enabled(int(idx)))]
    return sorted(active if active else list(range(total_layers)))


def _choose_layers_from_layout(
    *,
    total_layers: int,
    explicit_layers: Optional[List[int]],
    bottom_buffer_layers: int,
    decode_guard_layers: int,
    dense_anchor_stride: int,
    top_buffer_layers: int = 2,
) -> List[int]:
    if explicit_layers is not None:
        return sorted(explicit_layers)
    lower = max(int(bottom_buffer_layers), 0)
    upper = int(total_layers) - max(int(top_buffer_layers), 0) - max(int(decode_guard_layers), 0)
    out: List[int] = []
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
) -> Dict[str, Any]:
    return fit_layer_basis(
        x=x,
        y=y,
        basis_rank=int(basis_rank),
        block_size=int(block_size),
        pca_method=str(pca_method),
        pca_batch_rows=int(pca_batch_rows),
    )


def _load_profile_overrides(path: str) -> Dict[str, Dict[int, int]]:
    if not str(path).strip():
        return {
            "basis_rank_by_layer": {},
            "basis_top_k_by_layer": {},
            "top_k_by_layer": {},
        }
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    overrides = payload.get("recommended_overrides", {}) if isinstance(payload, dict) else {}
    out: Dict[str, Dict[int, int]] = {
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
    selected_layers: List[int],
    *,
    attention_mask: torch.Tensor,
    layer_x: Dict[int, List[torch.Tensor]],
    layer_y: Dict[int, List[torch.Tensor]],
    layer_rows: Dict[int, int],
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
    selected_layers: List[int],
    layer_states: Dict[str, Dict[str, torch.Tensor]],
    stats: Dict[str, Dict[str, float]],
    block_size: int,
    num_blocks: int,
    overrides: Dict[str, Dict[int, int]],
    existing_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    config_payload: Dict[str, Any] = dict(existing_config or {})
    config_payload.update(
        {
            "sparse_placement": "learned_basis",
            "routing_mode": str(config_payload.get("routing_mode", args.sca_routing_mode)),
            "bottom_buffer_layers": int(config_payload.get("bottom_buffer_layers", args.sca_bottom_buffer_layers)),
            "decode_guard_layers": int(config_payload.get("decode_guard_layers", args.sca_decode_guard_layers)),
            "dense_anchor_stride": int(config_payload.get("dense_anchor_stride", args.sca_dense_anchor_stride)),
            "basis_rank": int(config_payload.get("basis_rank", args.basis_rank)),
            "pca_method": str(config_payload.get("pca_method", args.pca_method)),
            "pca_batch_rows": int(config_payload.get("pca_batch_rows", args.pca_batch_rows)),
            "basis_rank_by_layer": config_payload.get("basis_rank_by_layer", overrides["basis_rank_by_layer"]),
            "basis_top_k_by_layer": config_payload.get("basis_top_k_by_layer", overrides["basis_top_k_by_layer"]),
            "top_k_by_layer": config_payload.get("top_k_by_layer", overrides["top_k_by_layer"]),
            "block_size": int(config_payload.get("block_size", block_size)),
            "num_blocks": int(config_payload.get("num_blocks", num_blocks)),
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
    layer_states: Dict[str, Dict[str, torch.Tensor]],
    stats: Dict[str, Dict[str, float]],
    layer_x: Dict[int, List[torch.Tensor]],
    layer_y: Dict[int, List[torch.Tensor]],
    selected_layers: List[int],
    include_buffers: bool,
) -> None:
    payload: Dict[str, Any] = {
        "layer_states": layer_states,
        "stats": stats,
    }
    if bool(include_buffers):
        payload["layer_x"] = {idx: layer_x[idx] for idx in selected_layers if layer_x[idx]}
        payload["layer_y"] = {idx: layer_y[idx] for idx in selected_layers if layer_y[idx]}
    torch.save(payload, resume_path)


def _save_basis_output(
    *,
    output_path: Path,
    args: argparse.Namespace,
    selected_layers: List[int],
    layer_states: Dict[str, Dict[str, torch.Tensor]],
    stats: Dict[str, Dict[str, float]],
    block_size: int,
    num_blocks: int,
    overrides: Dict[str, Dict[int, int]],
    existing_config: Optional[Dict[str, Any]] = None,
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
    p.add_argument("--pca-method", type=str, default="auto", choices=["auto", "lowrank", "incremental"])
    p.add_argument("--pca-batch-rows", type=int, default=1024)
    p.add_argument("--sca-block-size", type=int, default=32)
    p.add_argument("--sca-bottom-buffer-layers", type=int, default=2)
    p.add_argument("--sca-decode-guard-layers", type=int, default=12)
    p.add_argument("--sca-dense-anchor-stride", type=int, default=0)
    p.add_argument("--dataloader-num-workers", type=int, default=0)
    p.add_argument("--dataloader-prefetch-factor", type=int, default=2)
    p.add_argument("--dataloader-persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dense-rollout-tokens", type=int, default=8)
    p.add_argument("--resume-save-every-batches", type=int, default=1)
    p.add_argument("--write-partial-output-every-batches", type=int, default=1)
    p.add_argument("--max-batches", type=int, default=0)
    p.add_argument("--only-missing-from-output", action=argparse.BooleanOptionalAction, default=False)
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
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    configure_runtime_environment(cudnn_benchmark=True)
    if bool(args.use_streaming_harness) and int(args.batch_size) != 1:
        raise RuntimeError("Streaming harness mode currently requires --batch-size 1")
    if (not bool(args.use_streaming_harness)) and not str(args.hybrid_checkpoint).strip():
        raise RuntimeError("--hybrid-checkpoint is required unless --use-streaming-harness is enabled")
    _output_path = Path(args.output_path)
    _resume_path = _output_path.with_suffix(".resume.pt")
    _existing_output: Dict[str, Any] = {}
    if _output_path.exists():
        _loaded_output = torch.load(_output_path, map_location="cpu")
        if isinstance(_loaded_output, dict):
            _existing_output = _loaded_output
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
    _effective_pca_method = str(_existing_config.get("pca_method", args.pca_method))
    _effective_pca_batch_rows = int(_existing_config.get("pca_batch_rows", args.pca_batch_rows))
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
        layer_states: Dict[str, Dict[str, torch.Tensor]] = {}
        if _resume_path.exists():
            _resume_data = torch.load(_resume_path, map_location="cpu")
            layer_states = _resume_data.get("layer_states", {})
        elif _existing_output:
            layer_states = _existing_output.get("layer_states", {})
        completed_layers = sorted(int(_k) for _k in layer_states.keys())
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
    )

    artifact: Dict[str, Any] = {}
    model: Optional[NeuroplasticLlama] = None
    runtime: Optional[StreamingLlamaRuntime] = None
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
        hidden_size = int(runtime.config.hidden_size)
        num_blocks = int(hidden_size // max(block_size, 1))
        if num_blocks * block_size != hidden_size:
            raise RuntimeError("hidden_size must be divisible by --sca-block-size for learned-basis init")
    else:
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
            sca_basis_top_k=max(1, int(_effective_basis_rank) // 4),
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
        block_size = int(model.sca_config.block_size)
        num_blocks = int(model.sca_config.num_blocks)

    requested_layers = list(selected_layers)
    layer_states: Dict[str, Dict[str, torch.Tensor]] = {}
    stats: Dict[str, Dict[str, float]] = {}
    _loaded_layer_x: Dict[int, List[torch.Tensor]] = {}
    _loaded_layer_y: Dict[int, List[torch.Tensor]] = {}
    if _resume_path.exists():
        _resume_data = torch.load(_resume_path, map_location="cpu")
        layer_states = _resume_data.get("layer_states", {})
        stats = _resume_data.get("stats", {})
        _loaded_layer_x = {int(_k): _xs for _k, _xs in _resume_data.get("layer_x", {}).items()}
        _loaded_layer_y = {int(_k): _ys for _k, _ys in _resume_data.get("layer_y", {}).items()}
        print(f"[resume] loaded {len(layer_states)}/{len(selected_layers)} fitted layers, "
              f"{sum(1 for idx in _loaded_layer_x if int(idx) not in {int(s) for s in layer_states})} "
              f"partially collected, from {_resume_path}")
    elif _output_path.exists():
        _resume_data = torch.load(_output_path, map_location="cpu")
        layer_states = _resume_data.get("layer_states", {})
        stats = _resume_data.get("stats", {})
        print(
            f"[resume] loaded {len(layer_states)}/{len(selected_layers)} fitted layers from {_output_path}",
            flush=True,
        )

    _completed_layer_ids = {int(_k) for _k in layer_states.keys()}
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

    layer_x: Dict[int, List[torch.Tensor]] = {int(idx): [] for idx in selected_layers}
    layer_y: Dict[int, List[torch.Tensor]] = {int(idx): [] for idx in selected_layers}
    layer_rows: Dict[int, int] = {int(idx): 0 for idx in selected_layers}
    max_rows = int(max(args.max_rows_per_layer, 32))
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

    _resume_save_every_batches = max(int(args.resume_save_every_batches), 0)
    _partial_output_every_batches = max(int(args.write_partial_output_every_batches), 0)
    overrides = _load_profile_overrides(str(args.profile_path))

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
                include_buffers=True,
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
        if _saved_resume and _saved_output:
            print(f"[pass done] {_rows_done} fitted, {_rows_partial} collecting — resume + partial output saved", flush=True)
        elif _saved_resume:
            print(f"[pass done] {_rows_done} fitted, {_rows_partial} collecting — resume saved", flush=True)
        elif _saved_output:
            print(f"[pass done] {_rows_done} fitted, {_rows_partial} collecting — partial output saved", flush=True)
        else:
            print(f"[pass done] {_rows_done} fitted, {_rows_partial} collecting", flush=True)
        # Fit and free any layer that has collected enough rows. Using 1/8 of max_rows
        # as the threshold so layers are saved early and the resume file is written
        # after just a few batches rather than waiting for all layers to be fully saturated.
        _fit_threshold = max(64, max_rows // 8)
        for layer_idx in selected_layers:
            if str(layer_idx) in layer_states or layer_rows[layer_idx] < _fit_threshold:
                continue
            _xs = layer_x[layer_idx]
            _ys = layer_y[layer_idx]
            if not _xs or not _ys:
                continue
            _x = torch.cat(_xs, dim=0)
            _y = torch.cat(_ys, dim=0)
            _fitted = _fit_layer_basis(
                _x,
                _y,
                basis_rank=int(_effective_basis_rank),
                block_size=int(block_size),
                pca_method=str(_effective_pca_method),
                pca_batch_rows=int(_effective_pca_batch_rows),
            )
            layer_states[str(layer_idx)] = {
                "encoder_weight": _fitted["encoder_weight"],
                "encoder_bias": _fitted["encoder_bias"],
                "decoder_blocks": _fitted["decoder_blocks"],
                "decoder_bias": _fitted["decoder_bias"],
                "scale": _fitted["scale"],
            }
            stats[str(layer_idx)] = {
                "samples": float(_fitted["samples"]),
                "rank_effective": float(_fitted["rank_effective"]),
                "explained_variance_ratio": float(_fitted["explained_variance_ratio"]),
                "pca_method": str(_fitted.get("pca_method", _effective_pca_method)),
            }
            # Once a layer is fitted, stop collecting more token rows for it in
            # the current run. Resume loading already treats fitted layers this way.
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
                include_buffers=True,
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
                f"[checkpoint] {len(layer_states)}/{len(selected_layers)} layers fitted ({pct_done}%)"
                f" — resume + partial output saved"
            )
        if int(args.max_batches) > 0 and _batch_count >= int(args.max_batches):
            print(f"[bounded] reached max_batches={int(args.max_batches)}; stopping early", flush=True)
            break
        if all(v >= max_rows for v in layer_rows.values()):
            break

    # Fit any layers that stopped short of max_rows (e.g. dataset exhausted early)
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
        fitted = _fit_layer_basis(
            x=x,
            y=y,
            basis_rank=int(_effective_basis_rank),
            block_size=int(block_size),
            pca_method=str(_effective_pca_method),
            pca_batch_rows=int(_effective_pca_batch_rows),
        )
        layer_states[str(layer_idx)] = {
            "encoder_weight": fitted["encoder_weight"],
            "encoder_bias": fitted["encoder_bias"],
            "decoder_blocks": fitted["decoder_blocks"],
            "decoder_bias": fitted["decoder_bias"],
            "scale": fitted["scale"],
        }
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


if __name__ == "__main__":
    main()

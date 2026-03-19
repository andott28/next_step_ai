from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover
    load_dataset = None

try:
    from .neuroplastic_llama_gqa_mamba import NeuroplasticLlama
except ImportError:  # pragma: no cover
    from neuroplastic_llama_gqa_mamba import NeuroplasticLlama


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


def _build_dataloader(
    tokenizer: Any,
    dataset_name: str,
    dataset_config: str,
    dataset_split: str,
    text_column: str,
    max_seq_length: int,
    batch_size: int,
    max_samples: int,
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
    return DataLoader(tokenized, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())


def _choose_layers(model: NeuroplasticLlama, explicit_layers: Optional[List[int]]) -> List[int]:
    if explicit_layers is not None:
        return sorted(explicit_layers)
    total_layers = len(model.model.model.layers)
    active = [idx for idx in range(total_layers) if bool(model._sparse_layer_enabled(int(idx)))]
    return sorted(active if active else list(range(total_layers)))


def _fit_layer_basis(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    basis_rank: int,
    block_size: int,
) -> Dict[str, Any]:
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise RuntimeError("x/y must be 2D with matching rows")
    hidden_size = int(y.shape[1])
    num_blocks = hidden_size // int(block_size)
    if num_blocks * int(block_size) != hidden_size:
        raise RuntimeError("hidden_size must be divisible by block_size")

    y_mean = y.mean(dim=0)
    y_centered = y - y_mean
    rows = int(y_centered.shape[0])
    rank_eff = int(max(min(int(basis_rank), rows, hidden_size), 1))

    _u, s, v = torch.pca_lowrank(y_centered, q=rank_eff, center=False, niter=2)
    v = v[:, :rank_eff].contiguous()
    coeff = y_centered @ v

    ones = torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device)
    x_aug = torch.cat([x, ones], dim=-1)
    lsq = torch.linalg.lstsq(x_aug, coeff)
    proj = lsq.solution
    enc_w_eff = proj[:-1, :].transpose(0, 1).contiguous()
    enc_b_eff = proj[-1, :].contiguous()

    basis = v.transpose(0, 1).contiguous()
    if rank_eff < int(basis_rank):
        pad_rows = int(basis_rank) - rank_eff
        basis = torch.cat([basis, torch.zeros((pad_rows, hidden_size), dtype=basis.dtype, device=basis.device)], dim=0)
        enc_w_eff = torch.cat(
            [enc_w_eff, torch.zeros((pad_rows, enc_w_eff.shape[1]), dtype=enc_w_eff.dtype, device=enc_w_eff.device)],
            dim=0,
        )
        enc_b_eff = torch.cat([enc_b_eff, torch.zeros((pad_rows,), dtype=enc_b_eff.dtype, device=enc_b_eff.device)], dim=0)

    decoder_blocks = basis.view(int(basis_rank), num_blocks, int(block_size)).permute(1, 0, 2).contiguous()
    decoder_bias = y_mean.view(num_blocks, int(block_size)).contiguous()

    total_var = y_centered.pow(2).sum().clamp_min(1e-8)
    captured_var = coeff.pow(2).sum()
    explained = float((captured_var / total_var).detach().cpu().item())
    return {
        "encoder_weight": enc_w_eff.detach().cpu().float(),
        "encoder_bias": enc_b_eff.detach().cpu().float(),
        "decoder_blocks": decoder_blocks.detach().cpu().float(),
        "decoder_bias": decoder_bias.detach().cpu().float(),
        "scale": torch.tensor(1.0, dtype=torch.float32),
        "samples": int(x.shape[0]),
        "explained_variance_ratio": explained,
        "rank_effective": int(rank_eff),
    }


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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Initialize learned-basis sparse MLP from dense MLP activations.")
    p.add_argument("--model-name", type=str, required=True)
    p.add_argument("--hybrid-checkpoint", type=str, required=True)
    p.add_argument("--output-path", type=str, required=True)
    p.add_argument("--layers", type=str, default="")
    p.add_argument("--max-samples", type=int, default=256)
    p.add_argument("--max-seq-length", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-rows-per-layer", type=int, default=4096)
    p.add_argument("--basis-rank", type=int, default=64)
    p.add_argument("--sca-bottom-buffer-layers", type=int, default=2)
    p.add_argument("--sca-decode-guard-layers", type=int, default=12)
    p.add_argument("--dense-rollout-tokens", type=int, default=8)
    p.add_argument("--profile-path", type=str, default="")
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
    artifact = torch.load(args.hybrid_checkpoint, map_location="cpu")
    if not isinstance(artifact, dict):
        raise RuntimeError("Hybrid checkpoint must be a dict artifact")

    model = NeuroplasticLlama(
        model_name=args.model_name,
        neuroplasticity_enabled=True,
        sca_use_cuda=True,
        sca_spmm_impl="cuda_spmm",
        sca_basis_rank=int(args.basis_rank),
        sca_basis_top_k=max(1, int(args.basis_rank) // 4),
        sca_routing_mode=str(args.sca_routing_mode),
        sca_bottom_buffer_layers=int(args.sca_bottom_buffer_layers),
        sca_decode_guard_layers=int(args.sca_decode_guard_layers),
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

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
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
    )

    selected_layers = _choose_layers(model, _parse_layers(args.layers))
    model.set_mlp_alignment_capture(True, selected_layers)
    model.neuroplasticity_enabled = False
    model.eval()

    layer_x: Dict[int, List[torch.Tensor]] = {int(idx): [] for idx in selected_layers}
    layer_y: Dict[int, List[torch.Tensor]] = {int(idx): [] for idx in selected_layers}
    layer_rows: Dict[int, int] = {int(idx): 0 for idx in selected_layers}
    max_rows = int(max(args.max_rows_per_layer, 32))
    rollout_tokens = int(max(args.dense_rollout_tokens, 1))

    for batch in dataloader:
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
        if all(v >= max_rows for v in layer_rows.values()):
            break

    overrides = _load_profile_overrides(str(args.profile_path))
    layer_states: Dict[str, Dict[str, torch.Tensor]] = {}
    stats: Dict[str, Dict[str, float]] = {}
    for layer_idx in selected_layers:
        xs = layer_x[layer_idx]
        ys = layer_y[layer_idx]
        if not xs or not ys:
            continue
        x = torch.cat(xs, dim=0)
        y = torch.cat(ys, dim=0)
        fitted = _fit_layer_basis(
            x=x,
            y=y,
            basis_rank=int(args.basis_rank),
            block_size=int(model.sca_config.block_size),
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
        }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "model_name": str(args.model_name),
        "hybrid_checkpoint_path": str(args.hybrid_checkpoint),
        "config": {
            "sparse_placement": "learned_basis",
            "routing_mode": str(args.sca_routing_mode),
            "bottom_buffer_layers": int(args.sca_bottom_buffer_layers),
            "decode_guard_layers": int(args.sca_decode_guard_layers),
            "basis_rank": int(args.basis_rank),
            "basis_rank_by_layer": overrides["basis_rank_by_layer"],
            "basis_top_k_by_layer": overrides["basis_top_k_by_layer"],
            "top_k_by_layer": overrides["top_k_by_layer"],
            "block_size": int(model.sca_config.block_size),
            "num_blocks": int(model.sca_config.num_blocks),
        },
        "layer_selection": list(selected_layers),
        "layer_states": layer_states,
        "stats": stats,
    }
    torch.save(payload, output_path)
    print(json.dumps({"output_path": str(output_path), "layers_initialized": len(layer_states), "stats": stats}, indent=2))


if __name__ == "__main__":
    main()

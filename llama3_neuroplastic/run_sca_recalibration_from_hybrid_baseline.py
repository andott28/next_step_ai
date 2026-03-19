from __future__ import annotations

import argparse
import json
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover
    load_dataset = None

try:
    from .neuroplastic_llama_gqa_mamba import NeuroplasticLlama
    from .strict_decode_metrics import evaluate_decode_prefixes, final_hidden_cosine
except ImportError:  # pragma: no cover
    from neuroplastic_llama_gqa_mamba import NeuroplasticLlama
    from strict_decode_metrics import evaluate_decode_prefixes, final_hidden_cosine


@dataclass
class SCALocalRecalibrationConfig:
    model_name: str
    hybrid_checkpoint: str
    output_dir: str
    learned_basis_init_checkpoint: str = ""
    layers: str = ""
    max_samples: int = 256
    max_seq_length: int = 128
    batch_size: int = 1
    steps: int = 128
    lr: float = 1e-5
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    task_id: int = 0
    include_task_embedding: bool = False
    include_spatial_proj: bool = True
    include_layer_output_scale: bool = True
    loss_mode: str = "mse_plus_norm"
    top_k: int = 3
    adaptive_top_k: bool = True
    adaptive_top_k_min: int = 3
    adaptive_top_k_max: int = 12
    adaptive_top_k_min_score_ratio: float = 0.15
    block_size: int = 32
    basis_rank: int = 32
    basis_top_k: int = 8
    soft_mask: bool = True
    sparse_placement: str = "learned_basis"
    sca_routing_mode: str = "semantic_latent"
    sca_bottom_buffer_layers: int = 2
    sca_decode_guard_layers: int = 12
    semantic_block_score_normalized: bool = False
    grouped_row_gemm: bool = False
    disable_fast_fallback: bool = True
    save_every: int = 0
    verbose: bool = False
    recalibration_mode: str = "strict_sparse_stability"
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    dataset_split: str = "train"
    text_column: str = "text"
    logits_kl_weight: float = 0.2
    logits_warmup_steps: int = 2
    speech_anchor_layers: str = ""
    speech_anchor_weight: float = 1.0
    speech_anchor_cosine_weight: float = 0.1
    speech_anchor_final_hidden_weight: float = 0.25
    speech_anchor_delta_norm_weight: float = 0.05
    local_mlp_weight: float = 0.02
    local_mlp_loss_cap: float = 200.0
    local_mlp_log_compress: bool = True
    ce_weight: float = 0.5
    layer_output_scale_penalty_weight: float = 0.05
    decode_scale_stage1_freeze: bool = True
    decode_scale_stage1_value: float = 0.75
    decode_scale_min: float = 0.35
    entropy_floor: float = 8.0
    entropy_floor_weight: float = 0.05
    rollout_entropy_floor_weight: float = 0.05
    latent_support_concentration_weight: float = 0.05
    latent_support_similarity_weight: float = 0.05
    delta_norm_cap: float = 0.25
    delta_norm_cap_weight: float = 0.1
    staged_training_enabled: bool = True
    stage1_steps_ratio: float = 0.6
    stage_split_ratio: float = 0.67
    stage2_lr_scale: float = 0.5
    progressive_depth_enabled: bool = True
    progressive_depth_group_size: int = 2
    lower_layer_loss_weight: float = 1.0
    upper_layer_loss_weight: float = 0.35
    upper_layer_scale_cap: float = 0.75
    output_scale_cap_tail_fraction: float = 0.34
    nonfinite_recovery_enabled: bool = True
    nonfinite_max_retries_per_step: int = 2
    nonfinite_lr_backoff: float = 0.5
    nonfinite_param_abs_cap: float = 10.0
    skip_nonfinite_dense_kl: bool = True
    rollout_refinement_enabled: bool = True
    rollout_start_step: int = 8
    rollout_every: int = 4
    rollout_max_new_tokens: int = 2
    rollout_kl_weight: float = 0.05
    validation_prefix_count: int = 128
    validation_prefix_length: int = 64
    strict_decode_upper_layer_cap_enabled: bool = False
    strict_runtime_parity: bool = True
    strict_runtime_local_window_tokens: int = 128
    strict_runtime_sink_tokens: int = 8
    strict_runtime_page_size_tokens: int = 32
    strict_runtime_top_k_pages: int = 4
    strict_runtime_retrieval_head_groups: str = "0"
    strict_runtime_retrieval_start_layer: int = -1
    seed: int = 42


def _parse_args() -> SCALocalRecalibrationConfig:
    p = argparse.ArgumentParser(description="Recalibrate SCA sparse MLP path against hybrid baseline locally.")
    p.add_argument("--model-name", type=str, required=True)
    p.add_argument("--hybrid-checkpoint", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--learned-basis-init-checkpoint", type=str, default="")
    p.add_argument("--layers", type=str, default="")
    p.add_argument("--max-samples", type=int, default=256)
    p.add_argument("--max-seq-length", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--include-task-embedding", action="store_true")
    p.add_argument("--include-spatial-proj", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-layer-output-scale", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--loss-mode", type=str, default="mse_plus_norm", choices=["mse", "cosine", "mse_plus_norm"])
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--adaptive-top-k", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--adaptive-top-k-min", type=int, default=3)
    p.add_argument("--adaptive-top-k-max", type=int, default=12)
    p.add_argument("--adaptive-top-k-min-score-ratio", type=float, default=0.15)
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--basis-rank", type=int, default=32)
    p.add_argument("--basis-top-k", type=int, default=8)
    p.add_argument("--soft-mask", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--sca-sparse-placement",
        dest="sparse_placement",
        type=str,
        default="learned_basis",
        choices=["input_mask", "output_sparse", "intermediate_group", "learned_basis"],
    )
    p.add_argument(
        "--sca-routing-mode",
        type=str,
        default="semantic_latent",
        choices=["spatial_grid", "semantic_latent"],
    )
    p.add_argument("--sca-bottom-buffer-layers", type=int, default=2)
    p.add_argument("--sca-decode-guard-layers", type=int, default=12)
    p.add_argument("--semantic-block-score-normalized", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--grouped-row-gemm", action="store_true")
    p.add_argument("--disable-fast-fallback", action="store_true", default=True)
    p.add_argument("--logits-kl-weight", type=float, default=0.2)
    p.add_argument("--logits-warmup-steps", type=int, default=2)
    p.add_argument("--speech-anchor-layers", type=str, default="")
    p.add_argument("--speech-anchor-weight", type=float, default=1.0)
    p.add_argument("--speech-anchor-cosine-weight", type=float, default=0.1)
    p.add_argument("--speech-anchor-final-hidden-weight", type=float, default=0.25)
    p.add_argument("--speech-anchor-delta-norm-weight", type=float, default=0.05)
    p.add_argument("--local-mlp-weight", type=float, default=0.02)
    p.add_argument("--local-mlp-loss-cap", type=float, default=200.0)
    p.add_argument("--local-mlp-log-compress", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ce-weight", type=float, default=0.5)
    p.add_argument("--layer-output-scale-penalty-weight", type=float, default=0.05)
    p.add_argument("--decode-scale-stage1-freeze", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--decode-scale-stage1-value", type=float, default=0.75)
    p.add_argument("--decode-scale-min", type=float, default=0.35)
    p.add_argument("--entropy-floor", type=float, default=8.0)
    p.add_argument("--entropy-floor-weight", type=float, default=0.05)
    p.add_argument("--rollout-entropy-floor-weight", type=float, default=0.05)
    p.add_argument("--latent-support-concentration-weight", type=float, default=0.05)
    p.add_argument("--latent-support-similarity-weight", type=float, default=0.05)
    p.add_argument("--delta-norm-cap", type=float, default=0.25)
    p.add_argument("--delta-norm-cap-weight", type=float, default=0.1)
    p.add_argument("--staged-training-enabled", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--stage1-steps-ratio", type=float, default=0.6)
    p.add_argument("--stage-split-ratio", type=float, default=0.67)
    p.add_argument("--stage2-lr-scale", type=float, default=0.5)
    p.add_argument("--progressive-depth-enabled", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--progressive-depth-group-size", type=int, default=2)
    p.add_argument("--lower-layer-loss-weight", type=float, default=1.0)
    p.add_argument("--upper-layer-loss-weight", type=float, default=0.35)
    p.add_argument("--upper-layer-scale-cap", type=float, default=0.75)
    p.add_argument("--output-scale-cap-tail-fraction", type=float, default=0.34)
    p.add_argument("--nonfinite-recovery-enabled", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--nonfinite-max-retries-per-step", type=int, default=2)
    p.add_argument("--nonfinite-lr-backoff", type=float, default=0.5)
    p.add_argument("--nonfinite-param-abs-cap", type=float, default=10.0)
    p.add_argument("--skip-nonfinite-dense-kl", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--rollout-refinement-enabled", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--rollout-start-step", type=int, default=8)
    p.add_argument("--rollout-every", type=int, default=4)
    p.add_argument("--rollout-max-new-tokens", type=int, default=2)
    p.add_argument("--rollout-kl-weight", type=float, default=0.05)
    p.add_argument("--validation-prefix-count", type=int, default=128)
    p.add_argument("--validation-prefix-length", type=int, default=64)
    p.add_argument(
        "--strict-decode-upper-layer-cap-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument("--strict-runtime-parity", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--strict-runtime-local-window-tokens", type=int, default=128)
    p.add_argument("--strict-runtime-sink-tokens", type=int, default=8)
    p.add_argument("--strict-runtime-page-size-tokens", type=int, default=32)
    p.add_argument("--strict-runtime-top-k-pages", type=int, default=4)
    p.add_argument("--strict-runtime-retrieval-head-groups", type=str, default="0")
    p.add_argument("--strict-runtime-retrieval-start-layer", type=int, default=-1)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--recalibration-mode",
        type=str,
        default="local_mlp_geometry",
        choices=[
            "local_mlp_geometry",
            "local_mlp_geometry_plus_logits",
            "speech_layer_anchor",
            "speech_layer_anchor_plus_logits",
            "strict_sparse_stability",
            "decode_manifold_alignment",
            "export_only",
        ],
    )
    args = p.parse_args()
    return SCALocalRecalibrationConfig(**vars(args))


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_layer_selection(spec: str | None) -> List[int] | None:
    if spec is None or spec.strip() == "":
        return None
    out: set[int] = set()
    for part in spec.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_str, end_str = token.split("-", 1)
            start = int(start_str)
            end = int(end_str)
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
        raise RuntimeError("datasets package is required for SCA recalibration")

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
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def _collect_batches(dataloader: Iterable[Dict[str, torch.Tensor]], max_batches: int) -> List[Dict[str, torch.Tensor]]:
    out: List[Dict[str, torch.Tensor]] = []
    for idx, batch in enumerate(dataloader):
        if idx >= max_batches:
            break
        out.append(
            {
                "input_ids": batch["input_ids"].cpu(),
                "attention_mask": batch["attention_mask"].cpu(),
            }
        )
    if not out:
        raise RuntimeError("No batches collected from dataset")
    return out


def _build_validation_prefixes(
    tokenizer: Any,
    *,
    dataset_name: str,
    dataset_config: str,
    text_column: str,
    max_prefix_length: int,
    count: int,
) -> List[Dict[str, torch.Tensor]]:
    if load_dataset is None:
        return []
    dataset = load_dataset(dataset_name, dataset_config, split="validation")
    dataset = dataset.filter(lambda x: isinstance(x[text_column], str) and len(x[text_column].strip()) > 0)
    prefixes: List[Dict[str, torch.Tensor]] = []
    for row in dataset:
        encoded = tokenizer(
            row[text_column],
            truncation=True,
            max_length=int(max_prefix_length) + 8,
            return_tensors="pt",
        )
        if encoded["input_ids"].shape[-1] < 9:
            continue
        prefix_len = int(min(max_prefix_length, encoded["input_ids"].shape[-1] - 8))
        prefixes.append(
            {
                "input_ids": encoded["input_ids"][:, :prefix_len].cpu(),
                "attention_mask": encoded["attention_mask"][:, :prefix_len].cpu(),
            }
        )
        if len(prefixes) >= int(count):
            break
    return prefixes


def _load_hybrid_artifact(path: str) -> Dict[str, Any]:
    blob = torch.load(path, map_location="cpu")
    if not isinstance(blob, dict):
        raise RuntimeError("Hybrid checkpoint must be a dict artifact")
    return blob


def _choose_layer_selection(model: NeuroplasticLlama, explicit_layers: Optional[List[int]]) -> List[int]:
    if explicit_layers is not None:
        return sorted(explicit_layers)
    total_layers = len(model.model.model.layers)
    active_sparse = [idx for idx in range(total_layers) if bool(model._sparse_layer_enabled(int(idx)))]
    if active_sparse:
        return sorted(active_sparse)
    cutoff = max(total_layers - int(getattr(model, "buffer_layers", 0)), 0)
    return list(range(cutoff))


def _choose_speech_anchor_layers(
    model: NeuroplasticLlama,
    explicit_layers: Optional[List[int]],
    selected_layers: Optional[List[int]] = None,
) -> List[int]:
    if explicit_layers is not None:
        return sorted(explicit_layers)
    if selected_layers:
        tail = sorted(int(v) for v in selected_layers)
        return tail[-min(6, len(tail)) :]
    total_layers = len(model.model.model.layers)
    active_sparse = [idx for idx in range(total_layers) if bool(model._sparse_layer_enabled(int(idx)))]
    if active_sparse:
        tail = sorted(active_sparse)
        return tail[-min(6, len(tail)) :]
    cutoff = max(total_layers - int(getattr(model, "buffer_layers", 0)), 0)
    end = max(cutoff, 0)
    start = max(end - 6, 0)
    return list(range(start, end))


def _compute_speech_anchor_loss(
    dense_hidden_states: Tuple[torch.Tensor, ...],
    sparse_hidden_states: Tuple[torch.Tensor, ...],
    layer_indices: List[int],
    cosine_weight: float = 0.1,
    final_hidden_weight: float = 0.25,
    delta_norm_weight: float = 0.05,
) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, float]]:
    if not dense_hidden_states or not sparse_hidden_states:
        raise RuntimeError("Hidden states are required for speech-layer anchor loss")
    per_layer: Dict[str, float] = {}
    losses: List[torch.Tensor] = []
    for layer_idx in layer_indices:
        hs_idx = int(layer_idx) + 1  # hidden_states[0] is embedding output
        if hs_idx < 0 or hs_idx >= len(dense_hidden_states) or hs_idx >= len(sparse_hidden_states):
            continue
        dense_h = dense_hidden_states[hs_idx].float()
        sparse_h = sparse_hidden_states[hs_idx].float()
        mse = F.mse_loss(sparse_h, dense_h)
        cos = 1.0 - F.cosine_similarity(
            sparse_h.reshape(1, -1),
            dense_h.reshape(1, -1),
            dim=-1,
            eps=1e-6,
        ).mean()
        layer_loss = mse + (float(cosine_weight) * cos)
        per_layer[str(layer_idx)] = float(layer_loss.detach().cpu().item())
        losses.append(layer_loss)
    if not losses:
        raise RuntimeError("No valid speech anchor layers were available in hidden states")
    stacked = torch.stack(losses)
    anchor_loss = stacked.mean()
    dense_final = dense_hidden_states[-1].float()
    sparse_final = sparse_hidden_states[-1].float()
    final_hidden_mse = F.mse_loss(sparse_final, dense_final)
    delta_norm_ratio = (sparse_final - dense_final).pow(2).mean() / dense_final.pow(2).mean().clamp_min(1e-6)
    total = anchor_loss + (float(final_hidden_weight) * final_hidden_mse) + (float(delta_norm_weight) * delta_norm_ratio)
    extras = {
        "speech_anchor_base": float(anchor_loss.detach().cpu().item()),
        "speech_anchor_final_hidden_mse": float(final_hidden_mse.detach().cpu().item()),
        "speech_anchor_delta_norm_ratio": float(delta_norm_ratio.detach().cpu().item()),
    }
    return total, per_layer, extras


def _save_artifacts(
    model: NeuroplasticLlama,
    out_dir: Path,
    metrics: Dict[str, Any],
) -> tuple[Path, Path]:
    state_path = out_dir / "sca_recalibrated_state.pt"
    metrics_path = out_dir / "sca_recalibration_metrics.json"
    state = model.export_sca_recalibration_state()
    torch.save(state, state_path)
    payload = dict(metrics)
    payload["state_path"] = str(state_path)
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return state_path, metrics_path


def _text_quality_metrics(text: str) -> Dict[str, float]:
    tokens = re.findall(r"\w+|[^\w\s]", text)
    if not tokens:
        return {
            "distinct1": 0.0,
            "max_run": 0.0,
            "rep_frac": 0.0,
            "punct_frac": 0.0,
        }
    distinct1 = float(len(set(tokens)) / len(tokens))
    max_run = 1
    run = 1
    rep = 0
    punct = 0
    for tok in tokens:
        if re.fullmatch(r"[^\w\s]+", tok):
            punct += 1
    for idx in range(1, len(tokens)):
        if tokens[idx] == tokens[idx - 1]:
            run += 1
            rep += 1
            max_run = max(max_run, run)
        else:
            run = 1
    return {
        "distinct1": distinct1,
        "max_run": float(max_run),
        "rep_frac": float(rep / max(len(tokens) - 1, 1)),
        "punct_frac": float(punct / len(tokens)),
        "char_repeat4_frac": float(_char_repeat4_fraction(text)),
        "degenerate": float(1.0 if (max_run >= 4 or (rep / max(len(tokens) - 1, 1)) > 0.2 or _char_repeat4_fraction(text) > 0.2) else 0.0),
    }


def _char_repeat4_fraction(text: str) -> float:
    s = text.lower()
    if len(s) < 8:
        return 0.0
    total = max(len(s) - 3, 1)
    rep = 0
    for idx in range(len(s) - 7):
        if s[idx : idx + 4] == s[idx + 4 : idx + 8]:
            rep += 1
    return float(rep / total)


def _evaluate_smoke_quality(
    model: NeuroplasticLlama,
    tokenizer: Any,
    prompts: List[str],
    task_id: int,
    max_new_tokens: int = 12,
    use_cache: bool = False,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for prompt in prompts:
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded["attention_mask"].to(model.device)
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                top_k=20,
                top_p=0.9,
                use_cache=bool(use_cache),
                task_id=task_id,
            )
        text = tokenizer.decode(generated[0], skip_special_tokens=True)
        rows.append({"prompt": prompt, "text": text, **_text_quality_metrics(text)})
    agg = {
        "distinct1_mean": float(sum(r["distinct1"] for r in rows) / max(len(rows), 1)),
        "max_run_mean": float(sum(r["max_run"] for r in rows) / max(len(rows), 1)),
        "rep_frac_mean": float(sum(r["rep_frac"] for r in rows) / max(len(rows), 1)),
        "punct_frac_mean": float(sum(r["punct_frac"] for r in rows) / max(len(rows), 1)),
        "char_repeat4_frac_mean": float(sum(r["char_repeat4_frac"] for r in rows) / max(len(rows), 1)),
        "degenerate_frac": float(sum(r["degenerate"] for r in rows) / max(len(rows), 1)),
    }
    return {"rows": rows, "agg": agg}


def _quality_gate_not_worse(candidate: Dict[str, Any], baseline: Dict[str, Any]) -> Tuple[bool, Dict[str, float]]:
    c = candidate["agg"]
    b = baseline["agg"]
    checks = {
        "distinct1_delta": float(c["distinct1_mean"] - b["distinct1_mean"]),
        "max_run_delta": float(c["max_run_mean"] - b["max_run_mean"]),
        "rep_frac_delta": float(c["rep_frac_mean"] - b["rep_frac_mean"]),
        "punct_frac_delta": float(c["punct_frac_mean"] - b["punct_frac_mean"]),
        "char_repeat4_frac_delta": float(c["char_repeat4_frac_mean"] - b["char_repeat4_frac_mean"]),
        "degenerate_frac_delta": float(c["degenerate_frac"] - b["degenerate_frac"]),
    }
    passed = (
        checks["distinct1_delta"] >= -0.05
        and checks["max_run_delta"] <= 1.0
        and checks["rep_frac_delta"] <= 0.05
        and checks["punct_frac_delta"] <= 0.05
        and checks["char_repeat4_frac_delta"] <= 0.05
        and checks["degenerate_frac_delta"] <= 0.05
    )
    return bool(passed), checks


def _configure_strict_runtime_parity(model: NeuroplasticLlama, cfg: SCALocalRecalibrationConfig) -> None:
    if not bool(cfg.strict_runtime_parity):
        return
    retrieval_groups = [int(x.strip()) for x in str(cfg.strict_runtime_retrieval_head_groups).split(",") if x.strip()]
    retrieval_start = None if int(cfg.strict_runtime_retrieval_start_layer) < 0 else int(cfg.strict_runtime_retrieval_start_layer)
    model.set_sparse_attention_mode(
        enabled=True,
        local_window_tokens=int(cfg.strict_runtime_local_window_tokens),
        sink_tokens=int(cfg.strict_runtime_sink_tokens),
        page_size_tokens=int(cfg.strict_runtime_page_size_tokens),
        retrieval_top_k_pages=int(cfg.strict_runtime_top_k_pages),
        retrieval_head_group_ids=retrieval_groups,
        retrieval_start_layer=retrieval_start,
        archive_cpu_dtype="int4",
        strict_fully_sparse=True,
    )


def _build_stage_plan(selected_layers: List[int], total_steps: int, cfg: SCALocalRecalibrationConfig) -> List[Dict[str, Any]]:
    if total_steps <= 0:
        return []
    if str(cfg.recalibration_mode) == "decode_manifold_alignment":
        if bool(cfg.progressive_depth_enabled):
            group_size = max(int(cfg.progressive_depth_group_size), 1)
            groups = [
                list(selected_layers[idx : idx + group_size])
                for idx in range(0, len(selected_layers), group_size)
                if selected_layers[idx : idx + group_size]
            ]
            if not groups:
                return []
            base_steps = max(total_steps // len(groups), 0)
            remainder = max(total_steps - (base_steps * len(groups)), 0)
            cumulative: List[int] = []
            plan: List[Dict[str, Any]] = []
            for stage_idx, group in enumerate(groups):
                stage_steps = int(base_steps + (1 if stage_idx < remainder else 0))
                if stage_steps <= 0:
                    continue
                cumulative.extend(group)
                plan.append(
                    {
                        "name": f"progressive_depth_{stage_idx + 1}",
                        "layers": list(group),
                        "active_sparse_layers": list(cumulative),
                        "steps": stage_steps,
                        "lr_scale": 1.0 if stage_idx == 0 else float(cfg.stage2_lr_scale),
                    }
                )
            return plan
        split_at = max(1, min(len(selected_layers), int(round(len(selected_layers) * float(cfg.stage1_steps_ratio)))))
        high_impact = list(sorted(selected_layers)[:split_at])
        stage1_steps = max(1, min(total_steps - 1, int(round(total_steps * 0.6)))) if total_steps > 1 else int(total_steps)
        stage2_steps = max(total_steps - stage1_steps, 0)
        plan: List[Dict[str, Any]] = [
            {
                "name": "decode_manifold_stage1",
                "layers": list(high_impact),
                "active_sparse_layers": list(high_impact),
                "steps": int(stage1_steps),
                "lr_scale": 1.0,
            },
        ]
        if stage2_steps > 0:
            plan.append(
                {
                    "name": "decode_manifold_stage2",
                    "layers": list(selected_layers),
                    "active_sparse_layers": list(selected_layers),
                    "steps": int(stage2_steps),
                    "lr_scale": 0.5,
                }
            )
        return plan
    if (not bool(cfg.staged_training_enabled)) or len(selected_layers) <= 2:
        return [{"name": "full", "layers": list(selected_layers), "steps": int(total_steps), "lr_scale": 1.0}]

    split_at = max(1, min(len(selected_layers) - 1, int(round(len(selected_layers) * float(cfg.stage_split_ratio)))))
    lower_layers = list(selected_layers[:split_at])
    full_layers = list(selected_layers)
    stage1_steps = max(1, min(total_steps - 1, int(round(total_steps * float(cfg.stage1_steps_ratio)))))
    stage2_steps = max(total_steps - stage1_steps, 0)
    plan: List[Dict[str, Any]] = [
        {
            "name": "lower_band",
            "layers": lower_layers,
            "active_sparse_layers": lower_layers,
            "steps": int(stage1_steps),
            "lr_scale": 1.0,
        },
    ]
    if stage2_steps > 0:
        plan.append(
            {
                "name": "full_band",
                "layers": full_layers,
                "active_sparse_layers": full_layers,
                "steps": int(stage2_steps),
                "lr_scale": float(cfg.stage2_lr_scale),
            }
        )
    return plan


def _final_hidden_cosine_loss(
    dense_hidden_states: Tuple[torch.Tensor, ...],
    sparse_hidden_states: Tuple[torch.Tensor, ...],
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    cosine = final_hidden_cosine(
        dense_hidden=dense_hidden_states[-1],
        sparse_hidden=sparse_hidden_states[-1],
        attention_mask=attention_mask,
    )
    return 1.0 - cosine.mean()


def _layerwise_hidden_cosine_loss(
    dense_hidden_states: Tuple[torch.Tensor, ...],
    sparse_hidden_states: Tuple[torch.Tensor, ...],
    attention_mask: torch.Tensor,
    layer_indices: List[int],
) -> tuple[torch.Tensor, Dict[str, float]]:
    if not dense_hidden_states or not sparse_hidden_states:
        zero = torch.zeros((), device=attention_mask.device, dtype=torch.float32)
        return zero, {
            "hidden_alignment_layers_used": 0.0,
            "hidden_alignment_final_cosine_mean": 0.0,
            "hidden_alignment_layer_cosine_mean": 0.0,
        }
    losses: List[torch.Tensor] = []
    cosine_values: List[float] = []
    used_layers = 0
    for layer_idx in layer_indices:
        hs_idx = int(layer_idx) + 1
        if hs_idx < 0 or hs_idx >= len(dense_hidden_states) or hs_idx >= len(sparse_hidden_states):
            continue
        cosine = final_hidden_cosine(
            dense_hidden=dense_hidden_states[hs_idx],
            sparse_hidden=sparse_hidden_states[hs_idx],
            attention_mask=attention_mask,
        )
        loss = 1.0 - cosine.mean()
        losses.append(loss)
        cosine_values.append(float(cosine.mean().detach().cpu().item()))
        used_layers += 1
    final_cos_loss = _final_hidden_cosine_loss(
        dense_hidden_states=dense_hidden_states,
        sparse_hidden_states=sparse_hidden_states,
        attention_mask=attention_mask,
    )
    final_cos_mean = 1.0 - float(final_cos_loss.detach().cpu().item())
    if losses:
        return torch.stack(losses).mean(), {
            "hidden_alignment_layers_used": float(used_layers),
            "hidden_alignment_final_cosine_mean": float(final_cos_mean),
            "hidden_alignment_layer_cosine_mean": float(sum(cosine_values) / max(len(cosine_values), 1)),
        }
    return final_cos_loss, {
        "hidden_alignment_layers_used": 0.0,
        "hidden_alignment_final_cosine_mean": float(final_cos_mean),
        "hidden_alignment_layer_cosine_mean": float(final_cos_mean),
    }


def _build_rollout_prefix_from_batch(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    reserve_tokens: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    prefixes_ids: List[torch.Tensor] = []
    prefixes_mask: List[torch.Tensor] = []
    for row_idx in range(int(input_ids.shape[0])):
        valid_len = int(attention_mask[row_idx].sum().item())
        prefix_len = int(max(1, min(64, valid_len - int(reserve_tokens))))
        prefixes_ids.append(input_ids[row_idx : row_idx + 1, :prefix_len])
        prefixes_mask.append(attention_mask[row_idx : row_idx + 1, :prefix_len])
    max_len = max(int(t.shape[-1]) for t in prefixes_ids)
    padded_ids: List[torch.Tensor] = []
    padded_masks: List[torch.Tensor] = []
    for prefix_ids, prefix_mask in zip(prefixes_ids, prefixes_mask):
        pad = max_len - int(prefix_ids.shape[-1])
        if pad <= 0:
            padded_ids.append(prefix_ids)
            padded_masks.append(prefix_mask)
            continue
        padded_ids.append(F.pad(prefix_ids, (0, pad), value=0))
        padded_masks.append(F.pad(prefix_mask, (0, pad), value=0))
    return torch.cat(padded_ids, dim=0), torch.cat(padded_masks, dim=0)


def _compute_decode_manifold_rollout_alignment(
    model: NeuroplasticLlama,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    task_id: int,
    max_new_tokens: int,
    layer_indices: Optional[List[int]] = None,
) -> tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    prefix_ids, prefix_mask = _build_rollout_prefix_from_batch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        reserve_tokens=int(max_new_tokens),
    )
    prev_enabled = bool(model.neuroplasticity_enabled)
    with torch.no_grad():
        model.neuroplasticity_enabled = True
        generated = model.generate(
            input_ids=prefix_ids,
            attention_mask=prefix_mask,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            use_cache=True,
            task_id=int(task_id),
        ).detach()
    generated_mask = torch.ones_like(generated, device=model.device)
    model.neuroplasticity_enabled = False
    with torch.no_grad():
        dense_out = model(
            input_ids=generated,
            attention_mask=generated_mask,
            use_cache=False,
            return_dict=True,
            output_hidden_states=True,
            task_id=int(task_id),
        )
    model.neuroplasticity_enabled = True
    sparse_out = model(
        input_ids=generated,
        attention_mask=generated_mask,
        use_cache=False,
        return_dict=True,
        output_hidden_states=True,
        task_id=int(task_id),
    )
    sparse_shift = sparse_out.logits[:, :-1, :].float()
    dense_shift = dense_out.logits[:, :-1, :].float()
    valid = generated_mask[:, 1:].to(dtype=torch.bool)
    kl_tensor = F.kl_div(
        F.log_softmax(sparse_shift, dim=-1),
        F.softmax(dense_shift, dim=-1),
        reduction="none",
    ).sum(dim=-1)
    rollout_kl = kl_tensor[valid].mean() if torch.any(valid) else torch.zeros((), device=model.device, dtype=torch.float32)
    rollout_cos, rollout_cos_metrics = _layerwise_hidden_cosine_loss(
        dense_hidden_states=tuple(dense_out.hidden_states),
        sparse_hidden_states=tuple(sparse_out.hidden_states),
        attention_mask=generated_mask,
        layer_indices=list(layer_indices or []),
    )
    model.neuroplasticity_enabled = prev_enabled
    aux = {
        "rollout_sequence_length": float(generated.shape[-1]),
        "rollout_batch_size": float(generated.shape[0]),
    }
    aux.update(rollout_cos_metrics)
    return rollout_kl, rollout_cos, aux


def _build_layer_weight_map(
    selected_layers: List[int],
    active_layers: List[int],
    lower_weight: float,
    upper_weight: float,
) -> Dict[int, float]:
    if not selected_layers:
        return {}
    max_active = int(max(active_layers)) if active_layers else int(max(selected_layers))
    out: Dict[int, float] = {}
    for layer_idx in selected_layers:
        out[int(layer_idx)] = float(lower_weight if int(layer_idx) <= max_active else upper_weight)
    return out


def _mask_layer_output_scale_grad(model: NeuroplasticLlama, active_layers: List[int]) -> None:
    if not hasattr(model, "sca_layer_output_scale"):
        return
    param = model.sca_layer_output_scale
    if param.grad is None:
        return
    grad = param.grad
    keep = torch.zeros_like(grad, dtype=torch.bool)
    for idx in active_layers:
        if 0 <= int(idx) < keep.numel():
            keep[int(idx)] = True
    grad[~keep] = 0.0


def _apply_output_scale_caps(model: NeuroplasticLlama, cfg: SCALocalRecalibrationConfig, selected_layers: List[int]) -> None:
    if not hasattr(model, "sca_layer_output_scale"):
        return
    scales = model.sca_layer_output_scale.data
    if scales.numel() == 0:
        return
    tail_count = max(1, int(round(len(selected_layers) * float(cfg.output_scale_cap_tail_fraction))))
    tail_start_layer = selected_layers[-tail_count] if selected_layers else int(scales.numel() - tail_count)
    for layer_idx in range(scales.numel()):
        cap = 1.0
        if int(layer_idx) >= int(tail_start_layer):
            cap = float(cfg.upper_layer_scale_cap)
        floor = 0.0
        if selected_layers and int(layer_idx) in selected_layers:
            floor = float(cfg.decode_scale_min)
        scales[layer_idx].clamp_(min=floor, max=cap)


def _enforce_stage1_decode_scale_freeze(
    model: NeuroplasticLlama,
    cfg: SCALocalRecalibrationConfig,
    selected_layers: List[int],
    active_layers: List[int],
) -> None:
    if not bool(cfg.decode_scale_stage1_freeze):
        return
    if not hasattr(model, "sca_layer_output_scale"):
        return
    scales = model.sca_layer_output_scale.data
    if scales.numel() == 0 or not selected_layers:
        return
    active_set = {int(v) for v in active_layers}
    decode_layers = [int(v) for v in selected_layers if int(v) not in active_set]
    if not decode_layers:
        return
    value = float(cfg.decode_scale_stage1_value)
    for layer_idx in decode_layers:
        if 0 <= layer_idx < scales.numel():
            scales[layer_idx].fill_(value)


def _apply_entropy_floor_penalty(logits: torch.Tensor, entropy_floor: float) -> torch.Tensor:
    probs = torch.softmax(logits.float(), dim=-1)
    entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)
    return torch.relu(torch.tensor(float(entropy_floor), device=entropy.device, dtype=entropy.dtype) - entropy).mean()


def _compute_latent_support_regularizer(
    model: NeuroplasticLlama,
    layer_indices: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    latent_terms: List[torch.Tensor] = []
    block_terms: List[torch.Tensor] = []
    norms: List[float] = []
    active_fracs: List[float] = []
    coords_used: List[float] = []
    usage_entropies: List[float] = []
    support_overlaps: List[float] = []
    support_uniques: List[float] = []
    max_latent_importance: List[float] = []
    max_latent_load: List[float] = []
    max_block_importance: List[float] = []
    max_block_load: List[float] = []
    selected = set(int(v) for v in layer_indices) if layer_indices is not None else None
    for layer_idx, wrapper in enumerate(model.sca_sparse_mlps):
        if selected is not None and int(layer_idx) not in selected:
            continue
        stats = wrapper.get_last_learned_basis_stats()
        if stats is None:
            continue
        norms.append(float(stats.get("mean_latent_norm", 0.0)))
        active_fracs.append(float(stats.get("active_latent_fraction", 0.0)))
        coords_used.append(float(stats.get("active_latent_coords_used", 0.0)))
        usage_entropies.append(float(stats.get("latent_usage_entropy", 0.0)))
        support_overlaps.append(float(stats.get("support_overlap_mean", 0.0)))
        support_uniques.append(float(stats.get("support_unique_fraction", 0.0)))
        max_latent_importance.append(float(stats.get("max_latent_importance_fraction", 0.0)))
        max_latent_load.append(float(stats.get("max_latent_load_fraction", 0.0)))
        max_block_importance.append(float(stats.get("max_block_importance_fraction", 0.0)))
        max_block_load.append(float(stats.get("max_block_load_fraction", 0.0)))

        reg_tensors = wrapper.get_last_learned_basis_regularizer_tensors()
        if not isinstance(reg_tensors, dict):
            continue
        for key in ("_latent_importance_probs", "_latent_load_probs"):
            tensor = reg_tensors.get(key)
            if torch.is_tensor(tensor) and tensor.numel() > 0:
                uniform = torch.full_like(tensor, 1.0 / max(int(tensor.numel()), 1))
                latent_terms.append((tensor - uniform).pow(2).mean())
        for key in ("_block_importance_probs", "_block_load_probs"):
            tensor = reg_tensors.get(key)
            if torch.is_tensor(tensor) and tensor.numel() > 0:
                uniform = torch.full_like(tensor, 1.0 / max(int(tensor.numel()), 1))
                block_terms.append((tensor - uniform).pow(2).mean())
    zero = torch.zeros((), device=model.device, dtype=torch.float32)
    latent_balance = torch.stack(latent_terms).mean() if latent_terms else zero
    block_balance = torch.stack(block_terms).mean() if block_terms else zero
    metrics = {
        "mean_latent_norm": float(sum(norms) / max(len(norms), 1)),
        "mean_active_latent_fraction": float(sum(active_fracs) / max(len(active_fracs), 1)),
        "mean_active_latent_coords_used": float(sum(coords_used) / max(len(coords_used), 1)),
        "mean_latent_usage_entropy": float(sum(usage_entropies) / max(len(usage_entropies), 1)),
        "mean_latent_support_overlap": float(sum(support_overlaps) / max(len(support_overlaps), 1)),
        "mean_latent_support_unique_fraction": float(sum(support_uniques) / max(len(support_uniques), 1)),
        "max_latent_importance_fraction": float(sum(max_latent_importance) / max(len(max_latent_importance), 1)),
        "max_latent_load_fraction": float(sum(max_latent_load) / max(len(max_latent_load), 1)),
        "max_block_importance_fraction": float(sum(max_block_importance) / max(len(max_block_importance), 1)),
        "max_block_load_fraction": float(sum(max_block_load) / max(len(max_block_load), 1)),
        "latent_balance_penalty": float(latent_balance.detach().cpu().item()) if latent_terms else 0.0,
        "block_balance_penalty": float(block_balance.detach().cpu().item()) if block_terms else 0.0,
    }
    return latent_balance, block_balance, metrics


def _sanitize_trainable_parameters(
    trainable: List[torch.nn.Parameter],
    *,
    abs_cap: float,
) -> None:
    cap = float(max(abs_cap, 1.0))
    with torch.no_grad():
        for param in trainable:
            if param is None or param.data is None:
                continue
            data = param.data
            if not torch.isfinite(data).all():
                data.nan_to_num_(nan=0.0, posinf=cap, neginf=-cap)
            data.clamp_(min=-cap, max=cap)


def _backoff_optimizer_lr(optimizer: AdamW, factor: float) -> None:
    scale = float(factor)
    if scale <= 0.0 or scale >= 1.0:
        return
    for group in optimizer.param_groups:
        current_lr = float(group.get("lr", 0.0))
        group["lr"] = max(current_lr * scale, 1e-8)


def _compute_rollout_refinement_kl(
    model: NeuroplasticLlama,
    tokenizer: Any,
    prompts: List[str],
    task_id: int,
    max_new_tokens: int,
    entropy_floor: float,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    losses: List[torch.Tensor] = []
    entropy_penalties: List[torch.Tensor] = []
    quality_rows: List[Dict[str, float]] = []
    for prompt in prompts:
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded["attention_mask"].to(model.device)
        with torch.no_grad():
            model.neuroplasticity_enabled = True
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
                top_k=20,
                top_p=0.9,
                use_cache=True,
                task_id=int(task_id),
            ).detach()
        text = tokenizer.decode(generated[0], skip_special_tokens=True)
        quality_rows.append(_text_quality_metrics(text))
        full_mask = torch.ones_like(generated, device=model.device)
        model.neuroplasticity_enabled = False
        with torch.no_grad():
            dense_out = model(
                input_ids=generated,
                attention_mask=full_mask,
                use_cache=False,
                return_dict=True,
                task_id=int(task_id),
            )
        model.neuroplasticity_enabled = True
        sparse_out = model(
            input_ids=generated,
            attention_mask=full_mask,
            use_cache=False,
            return_dict=True,
            task_id=int(task_id),
        )
        sparse_logp = F.log_softmax(sparse_out.logits.float(), dim=-1)
        dense_p = F.softmax(dense_out.logits.detach().float(), dim=-1)
        kl = F.kl_div(sparse_logp, dense_p, reduction="batchmean")
        if not torch.isfinite(kl):
            continue
        losses.append(kl)
        entropy_penalties.append(_apply_entropy_floor_penalty(sparse_out.logits, entropy_floor=float(entropy_floor)))
    if not losses:
        zero = torch.zeros((), device=model.device, dtype=torch.float32)
        return zero, zero, {"rollout_rep_frac_mean": 0.0, "rollout_max_run_mean": 0.0, "rollout_degenerate_frac": 0.0}
    quality = {
        "rollout_rep_frac_mean": float(sum(r["rep_frac"] for r in quality_rows) / max(len(quality_rows), 1)),
        "rollout_max_run_mean": float(sum(r["max_run"] for r in quality_rows) / max(len(quality_rows), 1)),
        "rollout_degenerate_frac": float(sum(r["degenerate"] for r in quality_rows) / max(len(quality_rows), 1)),
    }
    return torch.stack(losses).mean(), torch.stack(entropy_penalties).mean(), quality


def main() -> None:
    cfg = _parse_args()
    _set_seed(cfg.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    artifact = _load_hybrid_artifact(cfg.hybrid_checkpoint)
    selected_layers = _parse_layer_selection(cfg.layers)
    if not bool(cfg.disable_fast_fallback):
        raise RuntimeError("Fast fallback must be disabled for full-sparse recalibration criterion")
    if bool(cfg.grouped_row_gemm):
        raise RuntimeError("grouped_row_gemm must be disabled for full-sparse recalibration criterion")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required because sca_use_cuda must be enabled")
    fallback_threshold = 0.0
    target_rank = artifact.get("target_rank", 16)
    variance_threshold = float(artifact.get("variance_threshold", 0.90))
    state_dim = artifact.get("state_dim")

    model = NeuroplasticLlama(
        model_name=cfg.model_name,
        neuroplasticity_enabled=True,
        sca_use_cuda=True,
        sca_spmm_impl="cuda_spmm",
        sca_basis_rank=int(cfg.basis_rank),
        sca_basis_top_k=int(cfg.basis_top_k),
        sca_top_k=int(cfg.top_k),
        sca_routing_mode=str(cfg.sca_routing_mode),
        sca_bottom_buffer_layers=int(cfg.sca_bottom_buffer_layers),
        sca_decode_guard_layers=int(cfg.sca_decode_guard_layers),
        sca_semantic_block_score_normalized=bool(cfg.semantic_block_score_normalized),
        sca_adaptive_top_k=bool(cfg.adaptive_top_k),
        sca_adaptive_top_k_min=int(cfg.adaptive_top_k_min),
        sca_adaptive_top_k_max=int(cfg.adaptive_top_k_max),
        sca_adaptive_top_k_min_score_ratio=float(cfg.adaptive_top_k_min_score_ratio),
        sca_block_size=int(cfg.block_size),
        sca_sparse_placement=str(cfg.sparse_placement),
        sca_soft_mask=bool(cfg.soft_mask),
        sca_grouped_row_gemm=bool(cfg.grouped_row_gemm),
        sca_stability_dense_fallback_threshold=float(fallback_threshold),
        attention_hybrid_enabled=True,
        attention_hybrid_layers=selected_layers,
        attention_hybrid_target_rank=target_rank,
        attention_hybrid_variance_threshold=variance_threshold,
        attention_hybrid_state_dim=state_dim,
        attention_hybrid_force_no_cache=not bool(cfg.strict_runtime_parity),
        attention_sparse_mode=bool(cfg.strict_runtime_parity),
        attention_local_window_tokens=int(cfg.strict_runtime_local_window_tokens),
        attention_sink_tokens=int(cfg.strict_runtime_sink_tokens),
        attention_page_size_tokens=int(cfg.strict_runtime_page_size_tokens),
        attention_retrieval_top_k_pages=int(cfg.strict_runtime_top_k_pages),
        attention_archive_cpu_dtype="int4",
        attention_disable_ssd_fetch_in_decode=True,
        attention_force_single_model_runtime=True,
        strict_decode_enable_repetition_penalty=False,
        strict_decode_upper_layer_cap_enabled=bool(cfg.strict_decode_upper_layer_cap_enabled),
    )
    model.load_hybrid_attention_state(cfg.hybrid_checkpoint, strict=False)
    if str(cfg.learned_basis_init_checkpoint).strip():
        model.load_learned_basis_init(str(cfg.learned_basis_init_checkpoint).strip(), strict=True)
    model.disable_task_bias_injection = not bool(cfg.include_task_embedding)
    model.strict_decode_enable_repetition_penalty = False
    model.strict_decode_upper_layer_cap_enabled = bool(cfg.strict_decode_upper_layer_cap_enabled)
    _configure_strict_runtime_parity(model, cfg)

    selected_layers = _choose_layer_selection(model, selected_layers)
    speech_anchor_layers = _choose_speech_anchor_layers(
        model,
        _parse_layer_selection(cfg.speech_anchor_layers),
        selected_layers=selected_layers,
    )
    stage_plan = _build_stage_plan(selected_layers=selected_layers, total_steps=int(cfg.steps), cfg=cfg)
    if cfg.verbose:
        print(json.dumps({"stage_plan": stage_plan, "selected_layers": selected_layers}, indent=2))

    trainable = model.prepare_sca_local_recalibration(
        include_task_embedding=bool(cfg.include_task_embedding and cfg.recalibration_mode != "decode_manifold_alignment"),
        include_spatial_proj=bool(cfg.include_spatial_proj and cfg.recalibration_mode != "decode_manifold_alignment"),
        include_adapter_scale=False,
        include_layer_output_scale=bool(cfg.include_layer_output_scale),
        layer_indices=list(stage_plan[0]["layers"]) if stage_plan else selected_layers,
        active_sparse_layer_indices=(
            list(stage_plan[0].get("active_sparse_layers", stage_plan[0]["layers"])) if stage_plan else selected_layers
        ),
        recalibration_mode=cfg.recalibration_mode,
        hybrid_checkpoint_path=cfg.hybrid_checkpoint,
    )

    if cfg.recalibration_mode == "export_only" or int(cfg.steps) <= 0:
        metrics = {
            "mode": "export_only",
            "model_name": cfg.model_name,
            "layer_selection": selected_layers,
            "speech_anchor_layers": speech_anchor_layers,
            "steps": 0,
            "initial_loss": None,
            "final_loss": None,
            "min_loss": None,
            "mean_touched_weight_fraction": 0.0,
            "estimated_bytes_fetched_per_token": 0.0,
            "fallback_rate_by_layer": {},
            "elapsed_seconds": float(time.perf_counter() - t0),
            "config": asdict(cfg),
        }
        state_path, metrics_path = _save_artifacts(model, out_dir, metrics)
        print(json.dumps(metrics, indent=2))
        print(f"Saved recalibrated SCA state to: {state_path}")
        print(f"Saved recalibration metrics to: {metrics_path}")
        return

    if not trainable:
        raise RuntimeError("No trainable parameters found for SCA recalibration")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    validation_prefixes = _build_validation_prefixes(
        tokenizer,
        dataset_name=cfg.dataset_name,
        dataset_config=cfg.dataset_config,
        text_column=cfg.text_column,
        max_prefix_length=int(cfg.validation_prefix_length),
        count=int(cfg.validation_prefix_count),
    )

    dataloader = _build_dataloader(
        tokenizer=tokenizer,
        dataset_name=cfg.dataset_name,
        dataset_config=cfg.dataset_config,
        dataset_split=cfg.dataset_split,
        text_column=cfg.text_column,
        max_seq_length=cfg.max_seq_length,
        batch_size=cfg.batch_size,
        max_samples=cfg.max_samples,
    )
    cached_batches = _collect_batches(dataloader, max_batches=max(cfg.steps, 1))
    smoke_prompts = [
        "Write one sentence about Norway.",
        "Explain why recurrent models can reduce memory traffic.",
        "Summarize grouped-query attention in two sentences.",
        "Describe a sparse MLP routing system.",
        "Give one short fact about Mamba-2.",
    ]
    baseline_quality = _evaluate_smoke_quality(
        model=model,
        tokenizer=tokenizer,
        prompts=smoke_prompts,
        task_id=int(cfg.task_id),
        use_cache=bool(cfg.strict_runtime_parity),
    )
    baseline_validation = None
    if cfg.recalibration_mode == "decode_manifold_alignment" and validation_prefixes:
        model.set_sparse_attention_mode(strict_fully_sparse=False)
        baseline_validation = evaluate_decode_prefixes(
            model,
            tokenizer,
            validation_prefixes,
            task_id=int(cfg.task_id),
            rollout_horizon=16,
            use_cache=bool(cfg.strict_runtime_parity),
        )
        model.set_sparse_attention_mode(strict_fully_sparse=True)

    losses: List[float] = []
    fallback_accum: Dict[str, List[float]] = {}
    layer_loss_last: Dict[str, float] = {}
    layer_norm_last: Dict[str, float] = {}
    touched_values: List[float] = []
    bytes_values: List[float] = []
    fallback_values_mean: List[float] = []
    rollout_kl_values: List[float] = []
    rollout_entropy_values: List[float] = []
    rollout_rep_frac_values: List[float] = []
    rollout_max_run_values: List[float] = []
    rollout_degenerate_values: List[float] = []
    latent_norm_values: List[float] = []
    latent_active_fraction_values: List[float] = []
    latent_coords_used_values: List[float] = []
    latent_usage_entropy_values: List[float] = []
    latent_support_overlap_values: List[float] = []
    latent_support_unique_fraction_values: List[float] = []
    nonfinite_events = 0
    skipped_dense_kl_steps = 0

    total_steps = int(cfg.steps)
    global_step = 0
    stage_idx = 0
    stage_remaining = int(stage_plan[0]["steps"]) if stage_plan else 0
    train_stage_layers = list(stage_plan[0]["layers"]) if stage_plan else list(selected_layers)
    active_sparse_stage_layers = (
        list(stage_plan[0].get("active_sparse_layers", stage_plan[0]["layers"])) if stage_plan else list(selected_layers)
    )
    active_stage_lr_scale = float(stage_plan[0]["lr_scale"]) if stage_plan else 1.0
    optimizer = AdamW(trainable, lr=float(cfg.lr) * float(active_stage_lr_scale), weight_decay=cfg.weight_decay)
    _enforce_stage1_decode_scale_freeze(model, cfg=cfg, selected_layers=selected_layers, active_layers=train_stage_layers)

    for step in range(total_steps):
        if stage_remaining <= 0 and stage_idx + 1 < len(stage_plan):
            stage_idx += 1
            train_stage_layers = list(stage_plan[stage_idx]["layers"])
            active_sparse_stage_layers = list(stage_plan[stage_idx].get("active_sparse_layers", train_stage_layers))
            active_stage_lr_scale = float(stage_plan[stage_idx]["lr_scale"])
            stage_remaining = int(stage_plan[stage_idx]["steps"])
            trainable = model.prepare_sca_local_recalibration(
                include_task_embedding=bool(cfg.include_task_embedding and cfg.recalibration_mode != "decode_manifold_alignment"),
                include_spatial_proj=bool(cfg.include_spatial_proj and cfg.recalibration_mode != "decode_manifold_alignment"),
                include_adapter_scale=False,
                include_layer_output_scale=bool(cfg.include_layer_output_scale),
                layer_indices=train_stage_layers,
                active_sparse_layer_indices=active_sparse_stage_layers,
                recalibration_mode=cfg.recalibration_mode,
                hybrid_checkpoint_path=cfg.hybrid_checkpoint,
            )
            optimizer = AdamW(trainable, lr=float(cfg.lr) * float(active_stage_lr_scale), weight_decay=cfg.weight_decay)
            _enforce_stage1_decode_scale_freeze(model, cfg=cfg, selected_layers=selected_layers, active_layers=train_stage_layers)
            if cfg.verbose:
                print(
                    f"[sca-recal] switching-stage name={stage_plan[stage_idx]['name']} "
                    f"train_layers={train_stage_layers} active_sparse_layers={active_sparse_stage_layers} "
                    f"lr_scale={active_stage_lr_scale:.3f}"
                )

        batch = cached_batches[step % len(cached_batches)]
        input_ids = batch["input_ids"].to(model.device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(model.device, non_blocking=True)
        labels = input_ids.masked_fill(attention_mask == 0, -100)
        decode_manifold_mode = cfg.recalibration_mode == "decode_manifold_alignment"
        aux: Dict[str, Any] = {}
        total_loss = torch.zeros((), device=model.device, dtype=torch.float32)
        local_loss = torch.zeros((), device=model.device, dtype=torch.float32)
        speech_anchor_loss = torch.zeros((), device=model.device, dtype=torch.float32)
        ce_loss = torch.zeros((), device=model.device, dtype=torch.float32)

        if decode_manifold_mode:
            model.neuroplasticity_enabled = False
            with torch.no_grad():
                dense_out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                    output_hidden_states=True,
                    task_id=int(cfg.task_id),
                )
            model.neuroplasticity_enabled = True
            sparse_out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
                output_hidden_states=True,
                task_id=int(cfg.task_id),
            )
            ce_loss = model.model.loss_function(
                logits=sparse_out.logits,
                labels=labels,
                vocab_size=int(model.model.config.vocab_size),
            )
            teacher_kl_tensor = F.kl_div(
                F.log_softmax(sparse_out.logits.float(), dim=-1),
                F.softmax(dense_out.logits.detach().float(), dim=-1),
                reduction="none",
            ).sum(dim=-1)
            teacher_valid = attention_mask.to(dtype=torch.bool)
            teacher_kl = teacher_kl_tensor[teacher_valid].mean() if torch.any(teacher_valid) else torch.zeros((), device=model.device, dtype=torch.float32)
            teacher_cos, teacher_cos_metrics = _layerwise_hidden_cosine_loss(
                dense_hidden_states=tuple(dense_out.hidden_states),
                sparse_hidden_states=tuple(sparse_out.hidden_states),
                attention_mask=attention_mask,
                layer_indices=train_stage_layers,
            )
            layer_weight_map = {int(layer_idx): 1.0 for layer_idx in train_stage_layers}
            local_loss, local_aux = model.compute_sca_local_recalibration_loss(
                loss_mode=cfg.loss_mode,
                layer_weight_map=layer_weight_map,
                delta_norm_cap=float(cfg.delta_norm_cap),
                delta_norm_cap_weight=float(cfg.delta_norm_cap_weight),
            )
            aux.update(local_aux)
            latent_balance, block_balance, balance_metrics = _compute_latent_support_regularizer(
                model,
                layer_indices=train_stage_layers,
            )
            total_loss = (
                ce_loss
                + (0.5 * teacher_kl)
                + (0.25 * teacher_cos)
                + (0.05 * local_loss)
                + (0.1 * latent_balance)
                + (0.1 * block_balance)
            )
            aux["ce_loss"] = float(ce_loss.detach().cpu().item())
            aux["teacher_kl"] = float(teacher_kl.detach().cpu().item())
            aux["teacher_final_hidden_cosine_loss"] = float(teacher_cos.detach().cpu().item())
            aux.update({f"teacher_{k}": v for k, v in teacher_cos_metrics.items()})
            aux["local_loss_raw"] = float(local_loss.detach().cpu().item())
            aux["local_loss_used"] = float(local_loss.detach().cpu().item())
            aux["latent_balance_penalty"] = float(latent_balance.detach().cpu().item())
            aux["block_balance_penalty"] = float(block_balance.detach().cpu().item())
            aux.update(balance_metrics)

            if stage_idx >= 1:
                rollout_kl, rollout_cos, rollout_aux = _compute_decode_manifold_rollout_alignment(
                    model=model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    task_id=int(cfg.task_id),
                    max_new_tokens=8,
                    layer_indices=active_sparse_stage_layers,
                )
                total_loss = total_loss + rollout_kl + (0.5 * rollout_cos)
                aux["rollout_kl"] = float(rollout_kl.detach().cpu().item())
                aux["rollout_hidden_cosine_loss"] = float(rollout_cos.detach().cpu().item())
                aux.update(rollout_aux)
                rollout_kl_values.append(float(rollout_kl.detach().cpu().item()))
            sparse_out = sparse_out
        else:
            use_speech_anchor = cfg.recalibration_mode in {"speech_layer_anchor", "speech_layer_anchor_plus_logits"}
            use_logits_kl = cfg.recalibration_mode in {
                "local_mlp_geometry_plus_logits",
                "speech_layer_anchor_plus_logits",
                "strict_sparse_stability",
            }
            use_local_geometry = cfg.recalibration_mode in {
                "local_mlp_geometry",
                "local_mlp_geometry_plus_logits",
                "strict_sparse_stability",
            }
            dense_logits = None
            dense_hidden_states = None
            if (use_logits_kl and step >= int(cfg.logits_warmup_steps)) or use_speech_anchor:
                prev_enabled = model.neuroplasticity_enabled
                model.neuroplasticity_enabled = False
                with torch.no_grad():
                    dense_out = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        return_dict=True,
                        output_hidden_states=bool(use_speech_anchor),
                        task_id=int(cfg.task_id),
                    )
                    if use_logits_kl and step >= int(cfg.logits_warmup_steps):
                        dense_logits = dense_out.logits.detach().float()
                        if not torch.isfinite(dense_logits).all():
                            if bool(cfg.skip_nonfinite_dense_kl):
                                dense_logits = None
                                skipped_dense_kl_steps += 1
                                nonfinite_events += 1
                                if cfg.verbose:
                                    print(f"[sca-recal] skipped-dense-kl step={step+1} reason=nonfinite_dense_logits")
                            else:
                                raise RuntimeError(f"Non-finite dense reference logits at step {step+1}")
                    if use_speech_anchor:
                        dense_hidden_states = tuple(dense_out.hidden_states) if dense_out.hidden_states is not None else None
                model.neuroplasticity_enabled = prev_enabled

            model.neuroplasticity_enabled = True
            sparse_out = None
            retries = int(max(0, cfg.nonfinite_max_retries_per_step))
            for attempt in range(retries + 1):
                candidate = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                    output_hidden_states=bool(use_speech_anchor),
                    task_id=int(cfg.task_id),
                )
                if torch.isfinite(candidate.logits).all():
                    sparse_out = candidate
                    break
                nonfinite_events += 1
                if not bool(cfg.nonfinite_recovery_enabled) or attempt >= retries:
                    raise RuntimeError(f"Non-finite sparse logits at step {step+1}")
                _sanitize_trainable_parameters(trainable, abs_cap=float(cfg.nonfinite_param_abs_cap))
                _apply_output_scale_caps(model, cfg=cfg, selected_layers=selected_layers)
                _backoff_optimizer_lr(optimizer, factor=float(cfg.nonfinite_lr_backoff))
                optimizer.zero_grad(set_to_none=True)
                if cfg.verbose:
                    print(
                        f"[sca-recal] recovered-nonfinite step={step+1} attempt={attempt+1} "
                        f"new_lr={optimizer.param_groups[0]['lr']:.2e}"
                    )
            if sparse_out is None:
                raise RuntimeError(f"Failed to recover sparse forward at step {step+1}")

            if use_local_geometry:
                layer_weight_map = _build_layer_weight_map(
                    selected_layers=selected_layers,
                    active_layers=train_stage_layers,
                    lower_weight=float(cfg.lower_layer_loss_weight),
                    upper_weight=float(cfg.upper_layer_loss_weight),
                )
                local_loss, aux = model.compute_sca_local_recalibration_loss(
                    loss_mode=cfg.loss_mode,
                    layer_weight_map=layer_weight_map,
                    delta_norm_cap=float(cfg.delta_norm_cap),
                    delta_norm_cap_weight=float(cfg.delta_norm_cap_weight),
                )
                if not torch.isfinite(local_loss):
                    raise RuntimeError(f"Non-finite local loss at step {step+1}")
                local_weight = float(cfg.local_mlp_weight) if cfg.recalibration_mode == "strict_sparse_stability" else 1.0
                local_term = local_loss
                if float(cfg.local_mlp_loss_cap) > 0.0:
                    local_term = torch.clamp(local_term, max=float(cfg.local_mlp_loss_cap))
                if bool(cfg.local_mlp_log_compress):
                    local_term = torch.log1p(local_term.clamp_min(0.0))
                total_loss = total_loss + (local_weight * local_term)
                aux["local_loss_raw"] = float(local_loss.detach().cpu().item())
                aux["local_loss_used"] = float(local_term.detach().cpu().item())
            elif use_speech_anchor:
                sparse_hidden_states = tuple(sparse_out.hidden_states) if sparse_out.hidden_states is not None else None
                if dense_hidden_states is None or sparse_hidden_states is None:
                    raise RuntimeError("speech_layer_anchor mode requires dense and sparse hidden states")
                speech_anchor_loss, speech_anchor_per_layer, speech_anchor_extras = _compute_speech_anchor_loss(
                    dense_hidden_states=dense_hidden_states,
                    sparse_hidden_states=sparse_hidden_states,
                    layer_indices=speech_anchor_layers,
                    cosine_weight=float(cfg.speech_anchor_cosine_weight),
                    final_hidden_weight=float(cfg.speech_anchor_final_hidden_weight),
                    delta_norm_weight=float(cfg.speech_anchor_delta_norm_weight),
                )
                aux["speech_anchor_per_layer"] = speech_anchor_per_layer
                aux["speech_anchor_layers"] = list(speech_anchor_layers)
                aux["speech_anchor_loss"] = float(speech_anchor_loss.detach().cpu().item())
                aux.update(speech_anchor_extras)
                total_loss = total_loss + (float(cfg.speech_anchor_weight) * speech_anchor_loss)
                if float(cfg.local_mlp_weight) > 0.0:
                    local_loss, local_aux = model.compute_sca_local_recalibration_loss(loss_mode=cfg.loss_mode)
                    aux["local_aux"] = local_aux
                    total_loss = total_loss + (float(cfg.local_mlp_weight) * local_loss)

            if dense_logits is not None:
                sparse_logp = F.log_softmax(sparse_out.logits.float(), dim=-1)
                dense_p = F.softmax(dense_logits, dim=-1)
                kl = F.kl_div(sparse_logp, dense_p, reduction="batchmean")
                if not torch.isfinite(kl):
                    raise RuntimeError(f"Non-finite KL loss at step {step+1}")
                total_loss = total_loss + (float(cfg.logits_kl_weight) * kl)
                aux["logits_kl"] = float(kl.detach().cpu().item())
            if float(cfg.ce_weight) > 0.0:
                ce_loss = model.model.loss_function(
                    logits=sparse_out.logits,
                    labels=labels,
                    vocab_size=int(model.model.config.vocab_size),
                )
                if not torch.isfinite(ce_loss):
                    raise RuntimeError(f"Non-finite CE loss at step {step+1}")
                total_loss = total_loss + (float(cfg.ce_weight) * ce_loss)
                aux["ce_loss"] = float(ce_loss.detach().cpu().item())
            if float(cfg.entropy_floor_weight) > 0.0:
                entropy_penalty = _apply_entropy_floor_penalty(sparse_out.logits, entropy_floor=float(cfg.entropy_floor))
                total_loss = total_loss + (float(cfg.entropy_floor_weight) * entropy_penalty)
                aux["entropy_floor_penalty"] = float(entropy_penalty.detach().cpu().item())
            latent_concentration, latent_similarity, latent_metrics = _compute_latent_support_regularizer(
                model,
                layer_indices=train_stage_layers,
            )
            if float(cfg.latent_support_concentration_weight) > 0.0 or float(cfg.latent_support_similarity_weight) > 0.0:
                latent_penalty = sparse_out.logits.new_zeros(())
                if float(cfg.latent_support_concentration_weight) > 0.0:
                    latent_penalty = latent_penalty + (
                        float(cfg.latent_support_concentration_weight)
                        * latent_concentration.to(device=sparse_out.logits.device, dtype=sparse_out.logits.dtype)
                    )
                if float(cfg.latent_support_similarity_weight) > 0.0:
                    latent_penalty = latent_penalty + (
                        float(cfg.latent_support_similarity_weight)
                        * latent_similarity.to(device=sparse_out.logits.device, dtype=sparse_out.logits.dtype)
                    )
                total_loss = total_loss + latent_penalty
                aux["latent_support_penalty"] = float(latent_penalty.detach().cpu().item())
            aux.update(latent_metrics)
            if float(cfg.layer_output_scale_penalty_weight) > 0.0 and hasattr(model, "sca_layer_output_scale"):
                decode_start = max(int(len(model.sca_sparse_mlps) * 0.75), 0)
                decode_scales = model.sca_layer_output_scale[decode_start:].float()
                if decode_scales.numel() > 0:
                    upper_penalty = torch.relu(decode_scales - 1.0).pow(2).mean()
                    target_penalty = (decode_scales.mean() - 0.85).pow(2)
                    scale_penalty = upper_penalty + (0.5 * target_penalty)
                    total_loss = total_loss + (float(cfg.layer_output_scale_penalty_weight) * scale_penalty)
                    aux["layer_output_scale_penalty"] = float(scale_penalty.detach().cpu().item())
        use_rollout = bool((not decode_manifold_mode) and cfg.rollout_refinement_enabled) and step >= int(cfg.rollout_start_step)
        do_rollout_step = bool(use_rollout and int(cfg.rollout_every) > 0 and ((step - int(cfg.rollout_start_step)) % int(cfg.rollout_every) == 0))
        if not torch.isfinite(total_loss):
            raise RuntimeError(f"Non-finite total loss at step {step+1}")

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        _mask_layer_output_scale_grad(model, train_stage_layers)
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=float(cfg.grad_clip))
        optimizer.step()
        _apply_output_scale_caps(model, cfg=cfg, selected_layers=selected_layers)
        _enforce_stage1_decode_scale_freeze(model, cfg=cfg, selected_layers=selected_layers, active_layers=train_stage_layers)

        if do_rollout_step:
            rollout_prompts = smoke_prompts[:1]
            optimizer.zero_grad(set_to_none=True)
            rollout_kl, rollout_entropy_penalty, rollout_quality = _compute_rollout_refinement_kl(
                model=model,
                tokenizer=tokenizer,
                prompts=rollout_prompts,
                task_id=int(cfg.task_id),
                max_new_tokens=int(cfg.rollout_max_new_tokens),
                entropy_floor=float(cfg.entropy_floor),
            )
            if bool(torch.isfinite(rollout_kl)) and bool(getattr(rollout_kl, "requires_grad", False)):
                rollout_loss = (
                    float(cfg.rollout_kl_weight) * rollout_kl
                    + float(cfg.rollout_entropy_floor_weight) * rollout_entropy_penalty
                )
                rollout_loss.backward()
                _mask_layer_output_scale_grad(model, train_stage_layers)
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(trainable, max_norm=float(cfg.grad_clip))
                optimizer.step()
                _apply_output_scale_caps(model, cfg=cfg, selected_layers=selected_layers)
                _enforce_stage1_decode_scale_freeze(model, cfg=cfg, selected_layers=selected_layers, active_layers=train_stage_layers)
                aux["rollout_kl"] = float(rollout_kl.detach().cpu().item())
                aux["rollout_entropy_penalty"] = float(rollout_entropy_penalty.detach().cpu().item())
                aux.update(rollout_quality)
                rollout_kl_values.append(float(rollout_kl.detach().cpu().item()))
                rollout_entropy_values.append(float(rollout_entropy_penalty.detach().cpu().item()))
                rollout_rep_frac_values.append(float(rollout_quality.get("rollout_rep_frac_mean", 0.0)))
                rollout_max_run_values.append(float(rollout_quality.get("rollout_max_run_mean", 0.0)))
                rollout_degenerate_values.append(float(rollout_quality.get("rollout_degenerate_frac", 0.0)))

        step_loss = float(total_loss.detach().cpu().item())
        losses.append(step_loss)
        layer_loss_last = dict(aux.get("per_layer_loss", {}))
        layer_norm_last = dict(aux.get("per_layer_norm", {}))
        if "speech_anchor_per_layer" in aux:
            layer_loss_last = dict(aux.get("speech_anchor_per_layer", {}))
        for key, value in aux.get("fallback_rate_by_layer", {}).items():
            fallback_accum.setdefault(str(key), []).append(float(value))

        sparse_diag = model.get_sparse_mlp_diagnostics()
        touched_values.append(float(sparse_diag.get("mean_touched_weight_fraction", 0.0)))
        bytes_values.append(float(sparse_diag.get("estimated_bytes_fetched_per_token", 0.0)))
        fallback_values_mean.append(float(sparse_diag.get("mean_dense_fallback_rate", 0.0)))
        latent_norm_values.append(float(sparse_diag.get("mean_latent_norm", 0.0)))
        latent_active_fraction_values.append(float(sparse_diag.get("mean_active_latent_fraction", 0.0)))
        latent_coords_used_values.append(float(sparse_diag.get("mean_active_latent_coords_used", 0.0)))
        latent_usage_entropy_values.append(float(sparse_diag.get("mean_latent_usage_entropy", 0.0)))
        latent_support_overlap_values.append(float(sparse_diag.get("mean_latent_support_overlap", 0.0)))
        latent_support_unique_fraction_values.append(float(sparse_diag.get("mean_latent_support_unique_fraction", 0.0)))

        if cfg.verbose or step == 0 or (step + 1) % 10 == 0:
            print(
                f"[sca-recal] step={step+1}/{cfg.steps} "
                f"stage={stage_plan[stage_idx]['name'] if stage_plan else 'full'} "
                f"loss={step_loss:.6f} layers={aux.get('layers_captured', 0)} "
                f"local={aux.get('local_loss_used', float(local_loss.detach().cpu().item()) if torch.is_tensor(local_loss) else 0.0):.6f} "
                f"speech={aux.get('speech_anchor_loss', 0.0):.6f} "
                f"ce={aux.get('ce_loss', 0.0):.6f} "
                f"rollout_kl={aux.get('rollout_kl', 0.0):.6f} "
                f"fallback={aux.get('fallback_rate', 0.0):.4f}"
            )
        stage_remaining -= 1
        global_step += 1

        if cfg.save_every > 0 and (step + 1) % int(cfg.save_every) == 0:
            interim_metrics = {
                "mode": cfg.recalibration_mode,
                "steps": int(step + 1),
                "initial_loss": float(losses[0]) if losses else None,
                "final_loss": float(losses[-1]) if losses else None,
                "min_loss": float(min(losses)) if losses else None,
                "per_layer_loss": layer_loss_last,
                "per_layer_norm": layer_norm_last,
                "mean_touched_weight_fraction": float(sum(touched_values) / max(len(touched_values), 1)),
                "estimated_bytes_fetched_per_token": float(sum(bytes_values) / max(len(bytes_values), 1)),
                "mean_dense_fallback_rate": float(sum(fallback_values_mean) / max(len(fallback_values_mean), 1)),
                "mean_latent_norm": float(sum(latent_norm_values) / max(len(latent_norm_values), 1)),
                "mean_active_latent_fraction": float(sum(latent_active_fraction_values) / max(len(latent_active_fraction_values), 1)),
                "mean_active_latent_coords_used": float(sum(latent_coords_used_values) / max(len(latent_coords_used_values), 1)),
                "mean_latent_usage_entropy": float(sum(latent_usage_entropy_values) / max(len(latent_usage_entropy_values), 1)),
                "mean_latent_support_overlap": float(sum(latent_support_overlap_values) / max(len(latent_support_overlap_values), 1)),
                "mean_latent_support_unique_fraction": float(
                    sum(latent_support_unique_fraction_values) / max(len(latent_support_unique_fraction_values), 1)
                ),
                "nonfinite_events": int(nonfinite_events),
                "skipped_dense_kl_steps": int(skipped_dense_kl_steps),
                "fallback_rate_by_layer": {
                    k: float(sum(v) / max(len(v), 1)) for k, v in fallback_accum.items()
                },
                "elapsed_seconds": float(time.perf_counter() - t0),
                "layer_selection": selected_layers,
                "stage_plan": stage_plan,
                "config": asdict(cfg),
            }
            _save_artifacts(model, out_dir, interim_metrics)

    fallback_rate_by_layer = {k: float(sum(v) / max(len(v), 1)) for k, v in fallback_accum.items()}
    warned_layers = [k for k, v in fallback_rate_by_layer.items() if float(v) > 0.5]
    metrics = {
        "mode": cfg.recalibration_mode,
        "steps": int(cfg.steps),
        "model_name": cfg.model_name,
        "hybrid_checkpoint": cfg.hybrid_checkpoint,
        "layer_selection": selected_layers,
        "speech_anchor_layers": speech_anchor_layers,
        "trainable_params": int(sum(int(p.numel()) for p in trainable)),
        "initial_loss": float(losses[0]) if losses else None,
        "final_loss": float(losses[-1]) if losses else None,
        "min_loss": float(min(losses)) if losses else None,
        "per_layer_loss": layer_loss_last,
        "per_layer_norm": layer_norm_last,
        "mean_touched_weight_fraction": float(sum(touched_values) / max(len(touched_values), 1)),
        "estimated_bytes_fetched_per_token": float(sum(bytes_values) / max(len(bytes_values), 1)),
        "mean_dense_fallback_rate": float(sum(fallback_values_mean) / max(len(fallback_values_mean), 1)),
        "mean_latent_norm": float(sum(latent_norm_values) / max(len(latent_norm_values), 1)),
        "mean_active_latent_fraction": float(sum(latent_active_fraction_values) / max(len(latent_active_fraction_values), 1)),
        "mean_active_latent_coords_used": float(sum(latent_coords_used_values) / max(len(latent_coords_used_values), 1)),
        "mean_latent_usage_entropy": float(sum(latent_usage_entropy_values) / max(len(latent_usage_entropy_values), 1)),
        "mean_latent_support_overlap": float(sum(latent_support_overlap_values) / max(len(latent_support_overlap_values), 1)),
        "mean_latent_support_unique_fraction": float(
            sum(latent_support_unique_fraction_values) / max(len(latent_support_unique_fraction_values), 1)
        ),
        "runtime_sca_use_cuda": bool(model.sca_config.use_cuda),
        "runtime_sca_spmm_impl": str(model.sca_config.spmm_impl),
        "nonfinite_events": int(nonfinite_events),
        "skipped_dense_kl_steps": int(skipped_dense_kl_steps),
        "fallback_rate_by_layer": fallback_rate_by_layer,
        "fallback_warning_layers_over_50pct": warned_layers,
        "elapsed_seconds": float(time.perf_counter() - t0),
        "mean_rollout_kl": float(sum(rollout_kl_values) / max(len(rollout_kl_values), 1)) if rollout_kl_values else 0.0,
        "mean_rollout_entropy_penalty": float(sum(rollout_entropy_values) / max(len(rollout_entropy_values), 1))
        if rollout_entropy_values
        else 0.0,
        "mean_rollout_rep_frac": float(sum(rollout_rep_frac_values) / max(len(rollout_rep_frac_values), 1))
        if rollout_rep_frac_values
        else 0.0,
        "mean_rollout_max_run": float(sum(rollout_max_run_values) / max(len(rollout_max_run_values), 1))
        if rollout_max_run_values
        else 0.0,
        "mean_rollout_degenerate_frac": float(sum(rollout_degenerate_values) / max(len(rollout_degenerate_values), 1))
        if rollout_degenerate_values
        else 0.0,
        "decode_layer_output_scale_mean": float(
            model.sca_layer_output_scale[max(int(len(model.sca_sparse_mlps) * 0.75), 0) :].detach().float().mean().cpu().item()
        )
        if hasattr(model, "sca_layer_output_scale") and len(model.sca_sparse_mlps) > 0
        else 1.0,
        "config": asdict(cfg),
        "stage_plan": stage_plan,
    }
    candidate_quality = _evaluate_smoke_quality(
        model=model,
        tokenizer=tokenizer,
        prompts=smoke_prompts,
        task_id=int(cfg.task_id),
        use_cache=bool(cfg.strict_runtime_parity),
    )
    quality_ok, quality_delta = _quality_gate_not_worse(candidate_quality, baseline_quality)
    metrics["quality_baseline"] = baseline_quality["agg"]
    metrics["quality_candidate"] = candidate_quality["agg"]
    metrics["quality_delta"] = quality_delta
    if cfg.recalibration_mode == "decode_manifold_alignment" and validation_prefixes:
        model.set_sparse_attention_mode(strict_fully_sparse=True)
        strict_validation = evaluate_decode_prefixes(
            model,
            tokenizer,
            validation_prefixes,
            task_id=int(cfg.task_id),
            rollout_horizon=16,
            use_cache=bool(cfg.strict_runtime_parity),
        )
        metrics["strict_validation"] = strict_validation
        metrics["non_strict_validation_baseline"] = baseline_validation
        baseline_deg = 0.0 if baseline_validation is None else float(
            baseline_validation.get("quality", {}).get("degenerate_frac", 0.0)
        )
        balance = strict_validation.get("balance", {})
        quality_ok = bool(
            float(metrics["mean_dense_fallback_rate"]) == 0.0
            and float(strict_validation.get("dense_top1_rank_median", 0.0)) == 1.0
            and float(strict_validation.get("dense_top1_rank_p95", 0.0)) <= 5.0
            and float(strict_validation.get("rollout_kl_mean", 0.0)) <= 1.0
            and float(strict_validation.get("final_hidden_cosine_mean", 0.0)) >= 0.95
            and float(strict_validation.get("quality", {}).get("degenerate_frac", 0.0)) <= (baseline_deg + 0.05)
            and float(balance.get("max_latent_importance_fraction", 0.0)) <= 0.35
            and float(balance.get("max_block_importance_fraction", 0.0)) <= 0.20
        )
        metrics["quality_delta"] = {
            **quality_delta,
            "strict_dense_top1_rank_median": float(strict_validation.get("dense_top1_rank_median", 0.0)),
            "strict_dense_top1_rank_p95": float(strict_validation.get("dense_top1_rank_p95", 0.0)),
            "strict_rollout_kl_mean": float(strict_validation.get("rollout_kl_mean", 0.0)),
            "strict_final_hidden_cosine_mean": float(strict_validation.get("final_hidden_cosine_mean", 0.0)),
            "strict_degenerate_frac": float(strict_validation.get("quality", {}).get("degenerate_frac", 0.0)),
            "non_strict_baseline_degenerate_frac": float(baseline_deg),
            "max_latent_importance_fraction": float(balance.get("max_latent_importance_fraction", 0.0)),
            "max_block_importance_fraction": float(balance.get("max_block_importance_fraction", 0.0)),
        }
    metrics["quality_gate_passed"] = bool(quality_ok)
    state_path, metrics_path = _save_artifacts(model, out_dir, metrics)
    if not quality_ok:
        metrics["quality_gate_failure_reason"] = "candidate quality regressed beyond allowed delta"
        metrics["state_path"] = str(state_path)
        metrics["metrics_path"] = str(metrics_path)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(metrics, indent=2))
        print(
            f"Quality gate failed (see {metrics_path}). Candidate checkpoint saved to {state_path} for inspection."
        )
        return
    print(json.dumps(metrics, indent=2))
    print(f"Saved recalibrated SCA state to: {state_path}")
    print(f"Saved recalibration metrics to: {metrics_path}")


if __name__ == "__main__":
    main()

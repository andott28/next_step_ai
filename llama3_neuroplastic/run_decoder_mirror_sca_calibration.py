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
from torch.optim import AdamW
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


SMOKE_PROMPTS = [
    "Write one sentence about Norway.",
    "Explain why recurrent models can reduce memory traffic.",
    "Summarize grouped-query attention in two sentences.",
    "Describe a sparse MLP routing system.",
    "Give one short fact about Mamba-2.",
]


@dataclass
class DecoderMirrorCalibrationConfig:
    model_name: str
    hybrid_checkpoint: str
    sca_recalibrated_checkpoint: str
    output_dir: str
    source_layers: str = ""
    mirror_top_k: int = 0
    mirror_rank: int = 4
    steps: int = 128
    lr: float = 1e-5
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    max_samples: int = 256
    max_seq_length: int = 128
    batch_size: int = 1
    task_id: int = 0
    logits_kl_weight: float = 1.0
    ce_weight: float = 0.25
    delta_penalty_weight: float = 0.01
    route_prior_scale_init: float = 0.25
    residual_scale_init: float = 0.0
    disable_rollout_refinement: bool = False
    rollout_start_step: int = 64
    rollout_steps: int = 16
    rollout_max_new_tokens: int = 8
    save_every: int = 0
    verbose: bool = False
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    dataset_split: str = "train"
    text_column: str = "text"
    seed: int = 42


def _parse_args() -> DecoderMirrorCalibrationConfig:
    p = argparse.ArgumentParser(description="Calibrate sparse decoder mirror against dense baseline logits.")
    p.add_argument("--model-name", type=str, required=True)
    p.add_argument("--hybrid-checkpoint", type=str, required=True)
    p.add_argument("--sca-recalibrated-checkpoint", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--source-layers", type=str, default="")
    p.add_argument("--mirror-top-k", type=int, default=0)
    p.add_argument("--mirror-rank", type=int, default=4)
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--max-samples", type=int, default=256)
    p.add_argument("--max-seq-length", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--logits-kl-weight", type=float, default=1.0)
    p.add_argument("--ce-weight", type=float, default=0.25)
    p.add_argument("--delta-penalty-weight", type=float, default=0.01)
    p.add_argument("--route-prior-scale-init", type=float, default=0.25)
    p.add_argument("--residual-scale-init", type=float, default=0.0)
    p.add_argument("--disable-rollout-refinement", action="store_true")
    p.add_argument("--rollout-start-step", type=int, default=64)
    p.add_argument("--rollout-steps", type=int, default=16)
    p.add_argument("--rollout-max-new-tokens", type=int, default=8)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--verbose", action="store_true")
    return DecoderMirrorCalibrationConfig(**vars(p.parse_args()))


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
        raise RuntimeError("datasets package is required for decoder mirror calibration")

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


def _load_hybrid_artifact(path: str) -> Dict[str, Any]:
    blob = torch.load(path, map_location="cpu")
    if not isinstance(blob, dict):
        raise RuntimeError("Hybrid checkpoint must be a dict artifact")
    return blob


def _text_quality_metrics(text: str) -> Dict[str, float]:
    tokens = re.findall(r"\w+|[^\w\s]", text)
    if not tokens:
        return {"distinct1": 0.0, "max_run": 0.0, "rep_frac": 0.0, "punct_frac": 0.0}
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
    }


def _evaluate_smoke_quality(
    model: NeuroplasticLlama,
    tokenizer: Any,
    prompts: List[str],
    task_id: int,
    max_new_tokens: int = 12,
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
                task_id=task_id,
            )
        text = tokenizer.decode(generated[0], skip_special_tokens=True)
        rows.append({"prompt": prompt, "text": text, **_text_quality_metrics(text)})
    agg = {
        "distinct1_mean": float(sum(r["distinct1"] for r in rows) / max(len(rows), 1)),
        "max_run_mean": float(sum(r["max_run"] for r in rows) / max(len(rows), 1)),
        "rep_frac_mean": float(sum(r["rep_frac"] for r in rows) / max(len(rows), 1)),
        "punct_frac_mean": float(sum(r["punct_frac"] for r in rows) / max(len(rows), 1)),
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
    }
    passed = (
        checks["distinct1_delta"] >= -0.05
        and checks["max_run_delta"] <= 1.0
        and checks["rep_frac_delta"] <= 0.05
        and checks["punct_frac_delta"] <= 0.05
    )
    return bool(passed), checks


def _save_artifacts(model: NeuroplasticLlama, out_dir: Path, metrics: Dict[str, Any]) -> Tuple[Path, Path]:
    state_path = out_dir / "decoder_mirror_sca_state.pt"
    metrics_path = out_dir / "decoder_mirror_calibration_metrics.json"
    torch.save(model.export_decoder_mirror_state(), state_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return state_path, metrics_path


def _rollout_refinement_loss(
    model: NeuroplasticLlama,
    tokenizer: Any,
    prompts: List[str],
    cfg: DecoderMirrorCalibrationConfig,
) -> torch.Tensor:
    losses: List[torch.Tensor] = []
    prompt_count = max(1, min(int(cfg.rollout_steps), len(prompts)))
    for prompt in prompts[:prompt_count]:
        encoded = tokenizer(prompt, return_tensors="pt")
        prompt_ids = encoded["input_ids"].to(model.device)
        prompt_mask = encoded["attention_mask"].to(model.device)
        with torch.no_grad():
            model.neuroplasticity_enabled = True
            model.set_decoder_mirror_enabled(True)
            generated = model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                max_new_tokens=int(cfg.rollout_max_new_tokens),
                do_sample=False,
                top_k=20,
                top_p=0.9,
                use_cache=False,
                task_id=int(cfg.task_id),
            ).detach()
        full_mask = torch.ones_like(generated, device=model.device)
        model.neuroplasticity_enabled = False
        model.set_decoder_mirror_enabled(False)
        with torch.no_grad():
            dense_out = model(
                input_ids=generated,
                attention_mask=full_mask,
                use_cache=False,
                return_dict=True,
                task_id=int(cfg.task_id),
            )
        model.neuroplasticity_enabled = True
        model.set_decoder_mirror_enabled(True)
        mirror_out = model(
            input_ids=generated,
            attention_mask=full_mask,
            use_cache=False,
            return_dict=True,
            task_id=int(cfg.task_id),
        )
        rollout_loss, _ = model.compute_decoder_mirror_calibration_loss(
            mirror_logits=mirror_out.logits,
            dense_logits=dense_out.logits.detach(),
            labels=None,
            logits_kl_weight=1.0,
            ce_weight=0.0,
            delta_penalty_weight=0.0,
        )
        losses.append(rollout_loss)
    if not losses:
        return torch.zeros((), device=model.device, dtype=torch.float32)
    return torch.stack(losses).mean()


def main() -> None:
    cfg = _parse_args()
    _set_seed(cfg.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    artifact = _load_hybrid_artifact(cfg.hybrid_checkpoint)
    artifact_layers = artifact.get("layer_selection")
    selected_hybrid_layers = None
    if isinstance(artifact_layers, list) and artifact_layers:
        selected_hybrid_layers = [int(v) for v in artifact_layers]
    source_layers = _parse_layer_selection(cfg.source_layers)
    target_rank = artifact.get("target_rank", 16)
    variance_threshold = float(artifact.get("variance_threshold", 0.90))
    state_dim = artifact.get("state_dim")

    model = NeuroplasticLlama(
        model_name=cfg.model_name,
        neuroplasticity_enabled=True,
        sca_use_cuda=True,
        sca_spmm_impl="cuda_spmm",
        sca_stability_dense_fallback_threshold=0.0,
        attention_hybrid_enabled=True,
        attention_hybrid_layers=selected_hybrid_layers,
        attention_hybrid_target_rank=target_rank,
        attention_hybrid_variance_threshold=variance_threshold,
        attention_hybrid_state_dim=state_dim,
        attention_hybrid_force_no_cache=True,
        decoder_mirror_enabled=True,
        decoder_mirror_top_k=(None if int(cfg.mirror_top_k) <= 0 else int(cfg.mirror_top_k)),
        decoder_mirror_rank=int(cfg.mirror_rank),
        decoder_mirror_route_conditioned=True,
        decoder_mirror_source_layers=source_layers,
        decoder_mirror_route_prior_scale_init=float(cfg.route_prior_scale_init),
        decoder_mirror_residual_scale_init=float(cfg.residual_scale_init),
    )
    model.load_hybrid_attention_state(cfg.hybrid_checkpoint, strict=True)
    model.load_sca_recalibration_state(cfg.sca_recalibrated_checkpoint, strict=True)
    model.disable_task_bias_injection = True

    trainable = model.prepare_decoder_mirror_calibration(
        source_layer_indices=source_layers,
        top_k=(None if int(cfg.mirror_top_k) <= 0 else int(cfg.mirror_top_k)),
        rank=int(cfg.mirror_rank),
        route_conditioned=True,
        calibration_mode="decoder_co_warp",
        hybrid_checkpoint_path=cfg.hybrid_checkpoint,
        sca_recalibrated_checkpoint_path=cfg.sca_recalibrated_checkpoint,
    )
    if not trainable:
        raise RuntimeError("No trainable decoder mirror parameters were found")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataloader = _build_dataloader(
        tokenizer=tokenizer,
        dataset_name=cfg.dataset_name,
        dataset_config=cfg.dataset_config,
        dataset_split=cfg.dataset_split,
        text_column=cfg.text_column,
        max_seq_length=int(cfg.max_seq_length),
        batch_size=int(cfg.batch_size),
        max_samples=int(cfg.max_samples),
    )
    cached_batches = _collect_batches(dataloader, max_batches=max(int(cfg.steps), 1))
    optimizer = AdamW(trainable, lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

    model.neuroplasticity_enabled = True
    model.set_decoder_mirror_enabled(False)
    baseline_quality = _evaluate_smoke_quality(model, tokenizer, SMOKE_PROMPTS, int(cfg.task_id))
    model.set_decoder_mirror_enabled(True)

    losses: List[float] = []
    logits_kl_values: List[float] = []
    ce_values: List[float] = []
    delta_values: List[float] = []
    active_block_values: List[float] = []
    touched_values: List[float] = []
    fallback_accum: Dict[str, List[float]] = {}

    for step in range(int(cfg.steps)):
        batch = cached_batches[step % len(cached_batches)]
        input_ids = batch["input_ids"].to(model.device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(model.device, non_blocking=True)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        model.neuroplasticity_enabled = False
        model.set_decoder_mirror_enabled(False)
        with torch.no_grad():
            dense_out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=None,
                use_cache=False,
                return_dict=True,
                task_id=int(cfg.task_id),
            )

        model.neuroplasticity_enabled = True
        model.set_decoder_mirror_enabled(True)
        mirror_out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
            use_cache=False,
            return_dict=True,
            task_id=int(cfg.task_id),
        )
        total_loss, aux = model.compute_decoder_mirror_calibration_loss(
            mirror_logits=mirror_out.logits,
            dense_logits=dense_out.logits.detach(),
            labels=labels,
            logits_kl_weight=float(cfg.logits_kl_weight),
            ce_weight=float(cfg.ce_weight),
            delta_penalty_weight=float(cfg.delta_penalty_weight),
        )

        if not bool(cfg.disable_rollout_refinement) and step >= int(cfg.rollout_start_step):
            total_loss = total_loss + _rollout_refinement_loss(model, tokenizer, SMOKE_PROMPTS, cfg)

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        if float(cfg.grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=float(cfg.grad_clip))
        optimizer.step()

        losses.append(float(total_loss.detach().cpu().item()))
        logits_kl_values.append(float(aux["loss_logits_kl"]))
        ce_values.append(float(aux["loss_ce"]))
        delta_values.append(float(aux["loss_delta_norm"]))
        active_block_values.append(float(aux.get("mean_active_blocks", 0.0)))
        touched_values.append(float(aux.get("mean_touched_weight_fraction", 0.0)))
        for key, value in aux.get("fallback_rate_by_layer", {}).items():
            fallback_accum.setdefault(str(key), []).append(float(value))

        if cfg.verbose or step == 0 or (step + 1) % 10 == 0:
            print(
                f"[decoder-mirror] step={step+1}/{cfg.steps} loss={losses[-1]:.6f} "
                f"kl={logits_kl_values[-1]:.6f} delta={delta_values[-1]:.6f}"
            )

        if int(cfg.save_every) > 0 and (step + 1) % int(cfg.save_every) == 0:
            interim_metrics = {
                "initial_loss": float(losses[0]) if losses else None,
                "final_loss": float(losses[-1]) if losses else None,
                "min_loss": float(min(losses)) if losses else None,
                "initial_logits_kl": float(logits_kl_values[0]) if logits_kl_values else None,
                "final_logits_kl": float(logits_kl_values[-1]) if logits_kl_values else None,
                "initial_ce": float(ce_values[0]) if ce_values else None,
                "final_ce": float(ce_values[-1]) if ce_values else None,
                "mean_delta_norm_ratio": float(sum(delta_values) / max(len(delta_values), 1)),
                "mean_active_blocks": float(sum(active_block_values) / max(len(active_block_values), 1)),
                "mean_touched_weight_fraction": float(sum(touched_values) / max(len(touched_values), 1)),
                "fallback_rate_by_layer": {k: float(sum(v) / max(len(v), 1)) for k, v in fallback_accum.items()},
                "elapsed_seconds": float(time.perf_counter() - t0),
                "steps": int(step + 1),
                "rollout_refinement_enabled": not bool(cfg.disable_rollout_refinement),
            }
            _save_artifacts(model, out_dir, interim_metrics)

    model.neuroplasticity_enabled = True
    model.set_decoder_mirror_enabled(True)
    candidate_quality = _evaluate_smoke_quality(model, tokenizer, SMOKE_PROMPTS, int(cfg.task_id))
    quality_ok, quality_delta = _quality_gate_not_worse(candidate_quality, baseline_quality)
    fallback_rate_by_layer = {k: float(sum(v) / max(len(v), 1)) for k, v in fallback_accum.items()}
    metrics = {
        "initial_loss": float(losses[0]) if losses else None,
        "final_loss": float(losses[-1]) if losses else None,
        "min_loss": float(min(losses)) if losses else None,
        "initial_logits_kl": float(logits_kl_values[0]) if logits_kl_values else None,
        "final_logits_kl": float(logits_kl_values[-1]) if logits_kl_values else None,
        "initial_ce": float(ce_values[0]) if ce_values else None,
        "final_ce": float(ce_values[-1]) if ce_values else None,
        "mean_delta_norm_ratio": float(sum(delta_values) / max(len(delta_values), 1)),
        "mean_active_blocks": float(sum(active_block_values) / max(len(active_block_values), 1)),
        "mean_touched_weight_fraction": float(sum(touched_values) / max(len(touched_values), 1)),
        "fallback_rate_by_layer": fallback_rate_by_layer,
        "quality_baseline": baseline_quality["agg"],
        "quality_candidate": candidate_quality["agg"],
        "quality_delta": quality_delta,
        "quality_gate_passed": bool(quality_ok),
        "elapsed_seconds": float(time.perf_counter() - t0),
        "steps": int(cfg.steps),
        "rollout_refinement_enabled": not bool(cfg.disable_rollout_refinement),
        "config": asdict(cfg),
    }
    state_path, metrics_path = _save_artifacts(model, out_dir, metrics)
    print(json.dumps(metrics, indent=2))
    print(f"Saved decoder mirror state to: {state_path}")
    print(f"Saved decoder mirror metrics to: {metrics_path}")


if __name__ == "__main__":
    main()

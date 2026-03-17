#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover
    load_dataset = None

try:
    from .gqa_mamba_rank_collapse import sync_alignment_loss
    from .neuroplastic_llama_backup_pre_mamba import NeuroplasticLlama as TeacherNeuroplasticLlama
    from .neuroplastic_llama_gqa_mamba import NeuroplasticLlama as StudentNeuroplasticLlama
except ImportError:  # Script-mode fallback
    from gqa_mamba_rank_collapse import sync_alignment_loss
    from neuroplastic_llama_backup_pre_mamba import NeuroplasticLlama as TeacherNeuroplasticLlama
    from neuroplastic_llama_gqa_mamba import NeuroplasticLlama as StudentNeuroplasticLlama


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloader(
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
        raise RuntimeError("datasets package is required for sync calibration")

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


def _select_logits(logits: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "last_token":
        return logits[:, -1, :]
    if mode == "full_sequence":
        return logits
    raise ValueError(f"Unsupported logit_target_mode: {mode}")


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


def _freeze_for_local_geometry(student: StudentNeuroplasticLlama, include_output_bias: bool = False) -> List[torch.nn.Parameter]:
    if hasattr(student, "prepare_local_geometry_calibration"):
        return list(student.prepare_local_geometry_calibration(include_output_bias=include_output_bias))
    raise RuntimeError("Student model does not support local geometry calibration")


def _extract_hybrid_attention_state(student: StudentNeuroplasticLlama) -> Dict[str, Any]:
    if hasattr(student, "export_hybrid_attention_state"):
        return dict(student.export_hybrid_attention_state())
    raise RuntimeError("Student model does not support hybrid attention export")


def _save_hybrid_artifact(
    student: StudentNeuroplasticLlama,
    out_dir: Path,
    args: argparse.Namespace,
    calibration_mode: str,
    extra_metrics: Dict[str, Any],
) -> tuple[Path, Path]:
    state_path = out_dir / "hybrid_attention_state.pt"
    metrics_path = out_dir / "hybrid_attention_metrics.json"
    export = _extract_hybrid_attention_state(student)
    payload = {
        "model_name": args.model_name,
        "layer_selection": export.get("layer_selection", []),
        "target_rank": args.target_rank,
        "variance_threshold": args.variance_threshold,
        "state_dim": args.state_dim,
        "calibration_mode": calibration_mode,
        "hybrid_attention_state_dict": export.get("hybrid_attention_state_dict", {}),
        "mix_values_by_layer": export.get("mix_values_by_layer", {}),
    }
    torch.save(payload, state_path)
    mix_map = export.get("mix_values_by_layer", {})
    finite_mix = [float(v) for v in mix_map.values() if isinstance(v, (float, int)) and torch.isfinite(torch.tensor(float(v)))]
    metrics = {
        "model_name": args.model_name,
        "calibration_mode": calibration_mode,
        "layer_selection": export.get("layer_selection", []),
        "mean_mix": (float(sum(finite_mix) / max(len(finite_mix), 1)) if finite_mix else 0.0),
        "nonfinite_mix_layers": int(len(mix_map) - len(finite_mix)),
        **extra_metrics,
        "state_path": str(state_path),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return state_path, metrics_path


def main() -> None:
    p = argparse.ArgumentParser(description="Short sync/calibration pass for GQA-Mamba replaced attention.")
    p.add_argument("--model-name", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="results/gqa_mamba_sync")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--calibration-mode",
        type=str,
        default="export_only",
        choices=["local_geometry", "offline_teacher", "export_only"],
    )
    p.add_argument("--layers", type=str, default="", help="Layer selection, e.g. '8-23' or '8,9,10,11'")
    p.add_argument("--dataset-name", type=str, default="wikitext")
    p.add_argument("--dataset-config", type=str, default="wikitext-2-raw-v1")
    p.add_argument("--dataset-split", type=str, default="train")
    p.add_argument("--text-column", type=str, default="text")
    p.add_argument("--max-samples", type=int, default=256)
    p.add_argument("--max-seq-length", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--sync-steps", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--loss-mode", type=str, default="mse", choices=["mse", "cosine"])
    p.add_argument("--logit-target-mode", type=str, default="last_token", choices=["last_token", "full_sequence"])
    p.add_argument("--target-rank", type=int, default=None)
    p.add_argument("--variance-threshold", type=float, default=0.90)
    p.add_argument("--state-dim", type=int, default=None)
    p.add_argument("--mix-init", type=float, default=0.05)
    p.add_argument("--mix-regularization", type=float, default=0.01)
    p.add_argument("--norm-regularization", type=float, default=0.01)
    p.add_argument("--sanitize-params", action="store_true", help="Clamp/repair non-finite hybrid params after each step")
    p.add_argument("--include-output-bias", action="store_true")
    p.add_argument("--verbose-init", action="store_true", help="Print per-layer replacement progress")
    args = p.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_layers = _parse_layer_selection(args.layers)

    t0 = time.perf_counter()
    print("[sync] loading student model + replacing attention", flush=True)
    student = StudentNeuroplasticLlama(
        model_name=args.model_name,
        neuroplasticity_enabled=False,
        sca_use_cuda=False,
        attention_hybrid_enabled=True,
        attention_hybrid_layers=selected_layers,
        attention_hybrid_mix_init=args.mix_init,
        attention_hybrid_target_rank=args.target_rank,
        attention_hybrid_variance_threshold=args.variance_threshold,
        attention_hybrid_state_dim=args.state_dim,
        attention_hybrid_force_no_cache=True,
        attention_gqa_verbose=bool(args.verbose_init),
    )
    student.train()

    if args.calibration_mode == "export_only" or int(args.sync_steps) <= 0:
        elapsed = time.perf_counter() - t0
        state_path, metadata_path = _save_hybrid_artifact(
            student=student,
            out_dir=out_dir,
            args=args,
            calibration_mode="export_only",
            extra_metrics={
                "mode": "export_only",
                "duration_seconds": float(elapsed),
            },
        )
        metrics = {
            "mode": "export_only",
            "duration_seconds": float(elapsed),
            "state_path": str(state_path),
        }
        print(json.dumps(metrics, indent=2))
        print(f"Saved hybrid attention state to: {state_path}")
        print(f"Saved metrics to: {metadata_path}")
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataloader = build_dataloader(
        tokenizer=tokenizer,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        text_column=args.text_column,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
    )
    cached_batches = _collect_batches(dataloader, max_batches=max(args.sync_steps, 1))
    print(f"[sync] collected {len(cached_batches)} tokenized batches", flush=True)

    teacher_targets: List[torch.Tensor] = []
    if args.calibration_mode == "offline_teacher":
        print("[sync] loading teacher model", flush=True)
        teacher = TeacherNeuroplasticLlama(
            model_name=args.model_name,
            neuroplasticity_enabled=False,
            sca_use_cuda=False,
        )
        teacher.eval()
        print("[sync] caching teacher logits", flush=True)

        with torch.no_grad():
            for i, batch in enumerate(cached_batches):
                input_ids = batch["input_ids"].to(teacher.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(teacher.device, non_blocking=True)
                out = teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                    task_id=0,
                )
                target = _select_logits(out.logits.float(), args.logit_target_mode).cpu()
                teacher_targets.append(target)
                if (i + 1) % 8 == 0 or i == 0:
                    print(f"[sync] cached teacher logits {i+1}/{len(cached_batches)}", flush=True)

        del teacher
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    trainable = _freeze_for_local_geometry(student, include_output_bias=args.include_output_bias)
    if not trainable:
        raise RuntimeError("No trainable hybrid attention parameters found")

    print(f"[sync] trainable attention params: {sum(int(p.numel()) for p in trainable):,}", flush=True)
    optimizer = AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    losses: List[float] = []
    print("[sync] starting calibration steps", flush=True)
    for step in range(int(args.sync_steps)):
        idx = step % len(cached_batches)
        batch = cached_batches[idx]
        input_ids = batch["input_ids"].to(student.device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(student.device, non_blocking=True)

        step_t0 = time.perf_counter()
        print(f"[sync] step {step+1}/{args.sync_steps}: forward start", flush=True)
        out = student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
            task_id=0,
        )
        print(
            f"[sync] step {step+1}/{args.sync_steps}: forward done ({time.perf_counter() - step_t0:.2f}s)",
            flush=True,
        )
        if args.calibration_mode == "local_geometry":
            loss, aux_metrics = student.compute_local_geometry_loss(
                mix_regularization=args.mix_regularization,
                norm_regularization=args.norm_regularization,
            )
        else:
            teacher_logits = teacher_targets[idx].to(student.device, non_blocking=True)
            student_logits = _select_logits(out.logits.float(), args.logit_target_mode)
            loss = sync_alignment_loss(student_logits=student_logits, teacher_logits=teacher_logits, mode=args.loss_mode)
            aux_metrics = {"hybrid_layers": float(len(student.get_effective_hybrid_layers()))}

        optimizer.zero_grad(set_to_none=True)
        bwd_t0 = time.perf_counter()
        print(f"[sync] step {step+1}/{args.sync_steps}: backward start", flush=True)
        loss.backward()
        print(
            f"[sync] step {step+1}/{args.sync_steps}: backward done ({time.perf_counter() - bwd_t0:.2f}s)",
            flush=True,
        )
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=float(args.grad_clip))
        optimizer.step()
        repair_info = {}
        if args.sanitize_params and hasattr(student, "sanitize_hybrid_parameters"):
            repair_info = student.sanitize_hybrid_parameters()
        losses.append(float(loss.item()))

        if (step + 1) % 10 == 0 or step == 0:
            print(f"[sync] step={step+1}/{args.sync_steps} loss={losses[-1]:.6f} aux={aux_metrics} repair={repair_info}")

    elapsed = time.perf_counter() - t0
    state_path, metadata_path = _save_hybrid_artifact(
        student=student,
        out_dir=out_dir,
        args=args,
        calibration_mode=args.calibration_mode,
        extra_metrics={
            "steps": int(args.sync_steps),
            "loss_mode": args.loss_mode,
            "logit_target_mode": args.logit_target_mode,
            "initial_loss": float(losses[0]) if losses else None,
            "final_loss": float(losses[-1]) if losses else None,
            "min_loss": float(min(losses)) if losses else None,
            "duration_seconds": float(elapsed),
        },
    )
    metrics = json.loads(metadata_path.read_text(encoding="utf-8"))

    print(json.dumps(metrics, indent=2))
    print(f"Saved hybrid attention state to: {state_path}")
    print(f"Saved sync metrics to: {metadata_path}")


if __name__ == "__main__":
    main()

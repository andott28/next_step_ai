import argparse
import json
import logging
import os
import sys
import time
import warnings
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from transformers import AutoTokenizer

warnings.filterwarnings("ignore")

sys.path.append(os.path.join(os.getcwd(), "llama3_neuroplastic"))
from neuroplastic_llama_interpolated_sca_v2 import NeuroplasticLlamaInterpolatedSCAV2
from run_llama_benchmark_1000_added_buffer_interpolateinject import (
    BASE_MODEL_NAME,
    _build_prompts,
    _score_candidates_batch,
    _task_candidates,
)
from sca_sparse_config import build_block_centers, build_inhibition_matrix


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
BENCHMARK_TASK = "mrpc"
BENCHMARK_METRIC = "Accuracy"

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def _parse_int_list(raw: str) -> List[int]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    if not values:
        raise ValueError("Expected at least one integer value")
    return values


def _parse_str_list(raw: str) -> List[str]:
    values = [x.strip().lower() for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("Expected at least one mode")
    return values


def _validate_modes(modes: List[str]) -> List[str]:
    allowed = {"sparse", "dense"}
    invalid = [m for m in modes if m not in allowed]
    if invalid:
        raise ValueError(f"Unsupported modes: {invalid}. Supported modes: {sorted(allowed)}")
    return modes


def _apply_sca_setting(
    model: NeuroplasticLlamaInterpolatedSCAV2,
    mode: str,
    grid_size: int,
    sparse_top_k: int | None,
    disable_grid: bool,
) -> Dict[str, float]:
    base = model.base_model
    if not disable_grid:
        base.sca_config.grid_size = int(grid_size)

    num_blocks = int(base.sca_config.num_blocks)
    if mode == "dense":
        top_k = num_blocks
        if hasattr(base.sca_config, "soft_mask"):
            base.sca_config.soft_mask = False
    elif mode == "sparse":
        if sparse_top_k is None:
            raise ValueError("sparse_top_k must be provided for sparse mode")
        top_k = int(sparse_top_k)
        if hasattr(base.sca_config, "soft_mask"):
            base.sca_config.soft_mask = True
    else:
        raise ValueError(f"Unknown mode: {mode}")

    top_k = max(1, min(top_k, num_blocks))
    base.sca_config.top_k = top_k

    if not disable_grid:
        centers = build_block_centers(base.sca_config).to(device=base.block_centers.device, dtype=base.block_centers.dtype)
        inhibition = build_inhibition_matrix(centers.float(), radius=base.sca_config.inhibition_radius).to(
            device=base.inhibition_matrix.device,
            dtype=base.inhibition_matrix.dtype,
        )
        with torch.no_grad():
            base.block_centers.copy_(centers)
            base.inhibition_matrix.copy_(inhibition)

    return {
        "top_k": float(top_k),
        "num_blocks": float(num_blocks),
        "active_ratio": float(top_k) / float(num_blocks),
    }


def _forward_logits(
    model: NeuroplasticLlamaInterpolatedSCAV2,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: str,
    task_id: int,
) -> torch.Tensor:
    with torch.inference_mode():
        if device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                    return_dict=True,
                    use_cache=False,
                    task_id=task_id,
                )
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                return_dict=True,
                use_cache=False,
                task_id=task_id,
            )
    return outputs.logits.float()


def _is_oom_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda out of memory" in text


def _prepare_mrpc_cache(
    tokenizer: AutoTokenizer,
    dataset,
    max_length: int,
) -> Dict[str, object]:
    batch = dataset[: len(dataset)]
    prompts, labels = _build_prompts(BENCHMARK_TASK, batch, tokenizer)
    enc = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {
        "prompts": prompts,
        "labels": np.array(labels, dtype=np.int64),
        "input_ids": enc.input_ids.cpu(),
        "attention_mask": enc.attention_mask.cpu(),
    }


def _evaluate_mrpc(
    model: NeuroplasticLlamaInterpolatedSCAV2,
    tokenizer: AutoTokenizer,
    mrpc_cache: Dict[str, object],
    task_id: int,
    batch_size: int,
    device: str,
) -> Tuple[float, str, int, float]:
    candidates, label_map, mode = _task_candidates(BENCHMARK_TASK)
    if mode != "classification":
        raise RuntimeError(f"Expected classification mode for {BENCHMARK_TASK}, got {mode}")
    candidate_token_ids = [tokenizer(c, add_special_tokens=False)["input_ids"] for c in candidates]
    fast_path = all(len(ids) == 1 for ids in candidate_token_ids)

    all_preds = []
    labels_all = mrpc_cache["labels"]
    prompts_all = mrpc_cache["prompts"]
    input_ids_all = mrpc_cache["input_ids"]
    attention_mask_all = mrpc_cache["attention_mask"]
    total_n = int(len(labels_all))
    start = time.perf_counter()
    effective_bs = max(1, int(batch_size))
    idx = 0
    pbar = tqdm(total=total_n, desc=BENCHMARK_TASK, leave=False)
    while idx < total_n:
        end = min(idx + effective_bs, total_n)
        batch_labels = labels_all[idx:end]
        try:
            if fast_path:
                input_ids = input_ids_all[idx:end].to(device)
                attention_mask = attention_mask_all[idx:end].to(device)
                logits = _forward_logits(
                    model=model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    device=device,
                    task_id=task_id,
                )
                log_probs = F.log_softmax(logits, dim=-1)
                seq_lens = attention_mask.sum(dim=1)
                last_pos = seq_lens - 1
                b_idx = torch.arange(input_ids.size(0), device=device)
                score_cols = []
                for ids in candidate_token_ids:
                    token_id = int(ids[0])
                    score_cols.append(log_probs[b_idx, last_pos, token_id])
                scores = torch.stack(score_cols, dim=-1).cpu()
            else:
                scores = _score_candidates_batch(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=prompts_all[idx:end],
                    candidates=candidates,
                    device=device,
                    max_length=input_ids_all.shape[1],
                    task_id=task_id,
                )
        except RuntimeError as exc:
            if device == "cuda" and _is_oom_error(exc) and effective_bs > 1:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                new_bs = max(1, effective_bs // 2)
                logger.warning(f"OOM at batch_size={effective_bs}, retrying with batch_size={new_bs}")
                effective_bs = new_bs
                continue
            raise

        pred_idx = torch.argmax(scores, dim=-1).numpy()
        preds = label_map[pred_idx]
        all_preds.extend(preds.tolist())
        idx = end
        pbar.update(len(batch_labels))
    pbar.close()

    elapsed = time.perf_counter() - start
    y_true = np.array(labels_all, dtype=np.int64)
    y_pred = np.array(all_preds, dtype=np.int64)
    score = float(accuracy_score(y_true, y_pred))
    return score, BENCHMARK_METRIC, total_n, elapsed


def run() -> None:
    parser = argparse.ArgumentParser(description="MRPC-only ablation runner for SCA FLOP-saving routing and grid sizes.")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Optional trained SCA checkpoint directory.")
    parser.add_argument("--model-name", type=str, default=BASE_MODEL_NAME)
    parser.add_argument("--modes", type=str, default="sparse,dense", help="Comma list: sparse,dense")
    parser.add_argument("--grid-sizes", type=str, default="2,4,6", help="Comma list of grid sizes.")
    parser.add_argument("--task-id", type=int, default=0, help="Fixed task_id for deterministic comparison.")
    parser.add_argument(
        "--sparse-top-k-values",
        type=str,
        default="1,2,3,4,8",
        help="Comma list of top-k values to sweep in sparse mode.",
    )
    parser.add_argument(
        "--sparse-top-k",
        type=int,
        default=None,
        help="Optional single top-k for sparse mode (overrides --sparse-top-k-values).",
    )
    parser.add_argument("--max-samples", type=int, default=96, help="Max MRPC validation samples (0 = full).")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument(
        "--disable-grid",
        action="store_true",
        help="Disable grid remapping/rebuild; keeps checkpoint/current block centers and only changes top_k.",
    )
    parser.add_argument("--output-json", type=str, default=None, help="Optional path to save full JSON results.")
    args = parser.parse_args()

    modes = _validate_modes(_parse_str_list(args.modes))
    grid_sizes = _parse_int_list(args.grid_sizes)
    sparse_top_k_values = _parse_int_list(args.sparse_top_k_values)
    if args.sparse_top_k is not None:
        sparse_top_k_values = [int(args.sparse_top_k)]
    sparse_top_k_values = [int(v) for v in sparse_top_k_values if int(v) > 0]
    if not sparse_top_k_values:
        raise ValueError("Expected at least one sparse top-k value > 0")

    n_grids = 1 if args.disable_grid else len(grid_sizes)
    n_sparse = n_grids * (len(sparse_top_k_values) if "sparse" in modes else 0)
    n_dense = n_grids if "dense" in modes else 0
    logger.info(
        f"Ablation plan: {n_sparse + n_dense} settings "
        f"(sparse={n_sparse}, dense={n_dense}) over grids={grid_sizes if not args.disable_grid else ['off']} top_k={sparse_top_k_values}"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"

    if args.checkpoint_dir:
        logger.info(f"Loading checkpoint: {args.checkpoint_dir}")
        model = NeuroplasticLlamaInterpolatedSCAV2.from_pretrained(
            args.checkpoint_dir,
            neuroplasticity_enabled=True,
        )
    else:
        logger.info("No checkpoint-dir provided. Using fresh model init.")
        model = NeuroplasticLlamaInterpolatedSCAV2(
            model_name=args.model_name,
            neuroplasticity_enabled=True,
            sca_use_cuda=False,
        )
    model.to(device)
    model.eval()
    model.base_model.collect_bio_gate_telemetry = False

    dataset = load_dataset("glue", BENCHMARK_TASK)["validation"]
    if args.max_samples > 0:
        dataset = dataset.select(range(min(len(dataset), args.max_samples)))
    logger.info(f"Loaded {BENCHMARK_TASK.upper()} validation samples: {len(dataset)}")
    mrpc_cache = _prepare_mrpc_cache(
        tokenizer=tokenizer,
        dataset=dataset,
        max_length=args.max_length,
    )

    results = []
    effective_grids = [grid_sizes[0]] if args.disable_grid else grid_sizes
    for mode in modes:
        for grid_size in effective_grids:
            mode_top_ks = sparse_top_k_values if mode == "sparse" else [None]
            for top_k_choice in mode_top_ks:
                setting_info = _apply_sca_setting(
                    model=model,
                    mode=mode,
                    grid_size=grid_size,
                    sparse_top_k=top_k_choice,
                    disable_grid=args.disable_grid,
                )
                logger.info(
                    f"Running setting mode={mode} grid={'off' if args.disable_grid else grid_size} top_k={int(setting_info['top_k'])}/{int(setting_info['num_blocks'])}"
                )

                benchmark_result: Dict[str, object]
                total_samples = 0
                total_time = 0.0
                score = float("nan")
                metric = BENCHMARK_METRIC
                try:
                    score, metric, n_samples, elapsed = _evaluate_mrpc(
                        model=model,
                        tokenizer=tokenizer,
                        mrpc_cache=mrpc_cache,
                        task_id=args.task_id,
                        batch_size=args.batch_size,
                        device=device,
                    )
                    benchmark_result = {
                        "score": float(score),
                        "metric": metric,
                        "samples": int(n_samples),
                        "seconds": float(elapsed),
                        "samples_per_second": float(n_samples / max(elapsed, 1e-9)),
                    }
                    total_samples = n_samples
                    total_time = elapsed
                    logger.info(
                        f"  {BENCHMARK_TASK.upper():<8} {metric:<8}={score:.4f}  {n_samples / max(elapsed, 1e-9):.2f} samples/s"
                    )
                except Exception as exc:
                    benchmark_result = {"error": str(exc)}
                    logger.error(f"  {BENCHMARK_TASK.upper()} failed: {exc}")

                row = {
                    "mode": mode,
                    "grid_size": int(grid_size),
                    "grid_disabled": bool(args.disable_grid),
                    "grid_label": "off" if args.disable_grid else str(int(grid_size)),
                    "benchmark": BENCHMARK_TASK,
                    "metric": metric,
                    "score": float(score),
                    "top_k": int(setting_info["top_k"]),
                    "num_blocks": int(setting_info["num_blocks"]),
                    "active_ratio": float(setting_info["active_ratio"]),
                    "sparsity_percent": float((1.0 - setting_info["active_ratio"]) * 100.0),
                    "total_samples": int(total_samples),
                    "total_seconds": float(total_time),
                    "throughput_samples_per_second": float(total_samples / max(total_time, 1e-9)),
                    "benchmark_result": benchmark_result,
                }
                results.append(row)

    print("\n" + "=" * 90)
    print(f"SCA FLOP-SAVING + GRID ABLATION SUMMARY ({BENCHMARK_TASK.upper()})")
    print("=" * 90)
    print(
        f"{'Mode':<10} {'Grid':<6} {'TopK':<10} {'Active%':<8} {'Sparse%':<8} {'Metric':<10} "
        f"{'Score':<8} {'Sec':<9} {'Samples/s':<10} {'dScore':<8} {'dS/s':<8}"
    )
    dense_by_grid = {str(r["grid_label"]): r for r in results if r["mode"] == "dense"}
    for row in results:
        d_score = "n/a"
        d_sps = "n/a"
        dense_ref = dense_by_grid.get(str(row["grid_label"]))
        if dense_ref is not None and row["mode"] != "dense":
            d_score = f"{(row['score'] - dense_ref['score']):+.4f}"
            d_sps = f"{(row['throughput_samples_per_second'] - dense_ref['throughput_samples_per_second']):+.2f}"
        print(
            f"{row['mode']:<10} {row['grid_label']:<6} {row['top_k']}/{row['num_blocks']:<7} "
            f"{100.0 * row['active_ratio']:<7.2f} {row['sparsity_percent']:<7.2f} "
            f"{row['metric']:<10} {row['score']:<8.4f} "
            f"{row['total_seconds']:<9.2f} {row['throughput_samples_per_second']:<10.2f} {d_score:<8} {d_sps:<8}"
        )
    print("=" * 90)

    if args.output_json:
        payload = {
            "model_name": args.model_name,
            "checkpoint_dir": args.checkpoint_dir,
            "task_id": args.task_id,
            "benchmark": BENCHMARK_TASK,
            "max_samples": args.max_samples,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "disable_grid": bool(args.disable_grid),
            "results": results,
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"Wrote JSON results to {args.output_json}")


if __name__ == "__main__":
    run()

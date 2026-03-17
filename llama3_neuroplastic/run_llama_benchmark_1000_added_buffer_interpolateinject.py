import argparse
import logging
import os
import sys
import warnings
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.metrics import accuracy_score, matthews_corrcoef
from tqdm import tqdm
from transformers import AutoTokenizer

warnings.filterwarnings("ignore")

sys.path.append(os.path.join(os.getcwd(), "llama3_neuroplastic"))
from neuroplastic_llama_interpolated_sca_v2 import NeuroplasticLlamaInterpolatedSCAV2


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
TASKS = ["cola", "sst2", "mrpc", "qqp", "stsb", "mnli", "qnli", "rte"]
BASE_MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def _task_id(task: str) -> int:
    return TASKS.index(task)


def _task_candidates(task: str):
    if task == "cola":
        return ["Yes", "No"], np.array([1, 0], dtype=np.int64), "classification"
    if task == "sst2":
        return ["Positive", "Negative"], np.array([1, 0], dtype=np.int64), "classification"
    if task in ["mrpc", "qqp"]:
        return ["Yes", "No"], np.array([1, 0], dtype=np.int64), "classification"
    if task == "mnli":
        return ["entailment", "neutral", "contradiction"], np.array([0, 1, 2], dtype=np.int64), "classification"
    if task in ["qnli", "rte"]:
        return ["Yes", "No"], np.array([0, 1], dtype=np.int64), "classification"
    if task == "stsb":
        return ["0", "1", "2", "3", "4", "5"], np.array([0, 1, 2, 3, 4, 5], dtype=np.float32), "regression"
    raise ValueError(f"Unsupported task: {task}")


def _to_chat_prompt(tokenizer: AutoTokenizer, user_content: str) -> str:
    messages = [{"role": "user", "content": user_content}]
    if hasattr(tokenizer, "apply_chat_template"):
        template = getattr(tokenizer, "chat_template", None)
        if template:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Fallback for tokenizer variants that do not ship chat_template metadata.
    # Keeps deterministic model-native framing consistent with Llama-style chat formatting.
    return (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
        f"{user_content}\n"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )


def _build_prompts(task: str, batch, tokenizer: AutoTokenizer) -> Tuple[List[str], np.ndarray]:
    labels = np.array(batch["label"])
    prompts: List[str] = []
    for i in range(len(labels)):
        if task == "cola":
            user_content = (
                "Determine if the following sentence is grammatically acceptable.\n"
                f"Sentence: {batch['sentence'][i]}\n"
                "Reply with exactly one word: Yes or No."
            )
        elif task == "sst2":
            user_content = (
                "Determine the sentiment of the following sentence.\n"
                f"Sentence: {batch['sentence'][i]}\n"
                "Reply with exactly one word: Positive or Negative."
            )
        elif task == "mrpc":
            user_content = (
                "Determine if the following two sentences are paraphrases of each other.\n"
                f"Sentence 1: {batch['sentence1'][i]}\n"
                f"Sentence 2: {batch['sentence2'][i]}\n"
                "Reply with exactly one word: Yes or No."
            )
        elif task == "qqp":
            user_content = (
                "Determine if the following two questions have the same meaning.\n"
                f"Question 1: {batch['question1'][i]}\n"
                f"Question 2: {batch['question2'][i]}\n"
                "Reply with exactly one word: Yes or No."
            )
        elif task == "stsb":
            user_content = (
                "Rate the semantic similarity between the following two sentences.\n"
                f"Sentence 1: {batch['sentence1'][i]}\n"
                f"Sentence 2: {batch['sentence2'][i]}\n"
                "Reply with exactly one integer from 0 to 5."
            )
        elif task == "mnli":
            user_content = (
                "Read the premise and hypothesis and determine the relationship.\n"
                f"Premise: {batch['premise'][i]}\n"
                f"Hypothesis: {batch['hypothesis'][i]}\n"
                "Reply with exactly one word: entailment, neutral, or contradiction."
            )
        elif task == "qnli":
            user_content = (
                "Determine whether the sentence answers the question.\n"
                f"Question: {batch['question'][i]}\n"
                f"Sentence: {batch['sentence'][i]}\n"
                "Reply with exactly one word: Yes or No."
            )
        elif task == "rte":
            user_content = (
                "Determine whether sentence 2 is entailed by sentence 1.\n"
                f"Sentence 1: {batch['sentence1'][i]}\n"
                f"Sentence 2: {batch['sentence2'][i]}\n"
                "Reply with exactly one word: Yes or No."
            )
        else:
            raise ValueError(f"Unsupported task: {task}")
        prompts.append(_to_chat_prompt(tokenizer, user_content))
    return prompts, labels


def _disable_random_fresh_neuroplastic(model: NeuroplasticLlamaInterpolatedSCAV2) -> None:
    """
    Fresh mode policy:
    - Keep the base pretrained LM fresh from hub (no local checkpoint).
    - Disable random untrained neuroplastic perturbations.
    - Preserve the Layer-2 fluency route.
    """
    base = model.base_model
    with torch.no_grad():
        if hasattr(base, "task_embedding"):
            base.task_embedding.weight.zero_()
        if hasattr(base, "spatial_proj"):
            base.spatial_proj.weight.zero_()
            base.spatial_proj.bias.zero_()
        if hasattr(base, "sca_config"):
            base.sca_config.top_k = int(base.sca_config.num_blocks)
            base.sca_config.soft_mask = False
            base.sca_config.spmm_impl = "dense"
        if hasattr(base, "refractory_until"):
            base.refractory_until.zero_()
    logger.info("Fresh mode: neutralized sparse routing + task params; evaluating base model + Layer-2 fluency route")


def _forward_logits(
    model: NeuroplasticLlamaInterpolatedSCAV2,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: str,
    task_id: int,
):
    with torch.no_grad():
        if device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                    return_dict=True,
                    task_id=task_id,
                )
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                return_dict=True,
                task_id=task_id,
            )
    return outputs.logits.float()


def _score_candidates_batch(
    model: NeuroplasticLlamaInterpolatedSCAV2,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    candidates: List[str],
    device: str,
    max_length: int,
    task_id: int,
):
    scores = torch.zeros((len(prompts), len(candidates)), dtype=torch.float32)
    for c_idx, candidate in enumerate(candidates):
        cand_ids = tokenizer(candidate, add_special_tokens=False)["input_ids"]
        cand_len = len(cand_ids)
        if cand_len == 0:
            raise RuntimeError(f"Candidate tokenization is empty: {candidate!r}")

        full_texts = [p + candidate for p in prompts]
        enc = tokenizer(full_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        input_ids = enc.input_ids.to(device)
        attention_mask = enc.attention_mask.to(device)

        logits = _forward_logits(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            device=device,
            task_id=task_id,
        )
        log_probs = F.log_softmax(logits, dim=-1)

        seq_lens = attention_mask.sum(dim=1)
        start_pos = seq_lens - cand_len
        b_idx = torch.arange(input_ids.size(0), device=device)

        sample_scores = torch.full((input_ids.size(0),), -1e9, dtype=torch.float32, device=device)
        valid = start_pos >= 1
        if valid.any():
            vb = b_idx[valid]
            vs = start_pos[valid]
            vscore = torch.zeros(vb.shape[0], dtype=torch.float32, device=device)
            for step in range(cand_len):
                token_pos = vs + step
                target_ids = input_ids[vb, token_pos]
                vscore += log_probs[vb, token_pos - 1, target_ids]
            sample_scores[valid] = vscore

        scores[:, c_idx] = sample_scores.cpu()
    return scores


def _pearson_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return 0.0
    if np.std(y_true) == 0.0 or np.std(y_pred) == 0.0:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def run_benchmark():
    parser = argparse.ArgumentParser(description="Minimal live GLUE benchmark")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=192)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="Optional override for task_id (use the same id for all tasks).",
    )
    parser.add_argument("--neuroplastic-off", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"

    if args.checkpoint_dir and not args.fresh:
        model = NeuroplasticLlamaInterpolatedSCAV2.from_pretrained(
            args.checkpoint_dir,
            neuroplasticity_enabled=not args.neuroplastic_off,
        )
    else:
        model = NeuroplasticLlamaInterpolatedSCAV2(
            model_name=BASE_MODEL_NAME,
            neuroplasticity_enabled=not args.neuroplastic_off,
            sca_use_cuda=False,
        )
        if args.fresh and (not args.neuroplastic_off):
            _disable_random_fresh_neuroplastic(model)
    model.to(device)
    model.eval()
    model.base_model.collect_bio_gate_telemetry = False
    fixed_task_id = args.task_id
    if fixed_task_id is not None:
        logger.info(f"Using fixed task_id={fixed_task_id} for all tasks")

    results = {}
    metrics = {}

    for task in TASKS:
        logger.info(f"Running {task.upper()}...")
        try:
            dataset = load_dataset("glue", task)
            val_split = "validation_matched" if task == "mnli" else "validation"
            dataset = dataset[val_split]
            if args.max_samples:
                dataset = dataset.select(range(min(len(dataset), args.max_samples)))

            candidates, label_map, mode = _task_candidates(task)
            task_id = fixed_task_id if fixed_task_id is not None else _task_id(task)
            all_preds = []
            all_labels = []

            for i in tqdm(range(0, len(dataset), args.batch_size), desc=task):
                batch = dataset[i : i + args.batch_size]
                prompts, labels = _build_prompts(task, batch, tokenizer)
                scores = _score_candidates_batch(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    candidates=candidates,
                    device=device,
                    max_length=args.max_length,
                    task_id=task_id,
                )
                if mode == "regression":
                    probs = torch.softmax(scores, dim=-1).numpy()
                    preds = probs @ label_map
                else:
                    pred_idx = torch.argmax(scores, dim=-1).numpy()
                    preds = label_map[pred_idx]
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.tolist())

            y_true = np.array(all_labels)
            y_pred = np.array(all_preds)

            if mode == "regression":
                score = _pearson_corr(y_true.astype(np.float32), y_pred.astype(np.float32))
                metric = "Pearson"
            elif task == "cola":
                score = float(matthews_corrcoef(y_true.astype(np.int64), y_pred.astype(np.int64)))
                metric = "MCC"
            else:
                score = float(accuracy_score(y_true.astype(np.int64), y_pred.astype(np.int64)))
                metric = "Accuracy"

            logger.info(f"{task.upper()} {metric}: {score:.4f}")
            results[task] = score
            metrics[task] = metric
        except Exception as exc:
            logger.error(f"{task.upper()} failed: {exc}")

    print("\n" + "=" * 40)
    print("MINIMAL LIVE BENCHMARK SUMMARY")
    print("=" * 40)
    for task in TASKS:
        if task in results:
            print(f"{task.upper():<10} | {results[task]:.4f} ({metrics[task]})")
    print("=" * 40)


if __name__ == "__main__":
    run_benchmark()

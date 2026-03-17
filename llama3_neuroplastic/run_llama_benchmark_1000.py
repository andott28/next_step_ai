import os
import sys
import logging
import argparse
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, matthews_corrcoef
from tqdm import tqdm
import warnings

# Filter warnings
warnings.filterwarnings("ignore")

# Add the llama3_neuroplastic directory to the path if needed
sys.path.append(os.path.join(os.getcwd(), "llama3_neuroplastic"))
from neuroplastic_llama import NeuroplasticLlama

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Increase recursion limit
sys.setrecursionlimit(5000)

# PIN SEED
SEED = 42
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def get_embeddings(model, tokenizer, dataset, device, batch_size):
    embeddings = []
    labels = []

    logger.info(f"Processing {len(dataset)} samples (Batch Size={batch_size})...")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i : i + batch_size]

        if "sentence" in batch:
            texts = batch["sentence"]
        elif "sentence1" in batch:
            texts = [f"{s1} {s2}" for s1, s2 in zip(batch["sentence1"], batch["sentence2"])]
        elif "question1" in batch:
            texts = [f"{q1} {q2}" for q1, q2 in zip(batch["question1"], batch["question2"])]
        elif "premise" in batch:
            texts = [f"{p} {h}" for p, h in zip(batch["premise"], batch["hypothesis"])]
        else:
            continue

        if "label" not in batch:
            continue
        batch_labels = batch["label"]

        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )

            last_hidden = outputs.hidden_states[-1]
            pooled = last_hidden[:, -1, :]

        embeddings.append(pooled.cpu().numpy())
        labels.extend(batch_labels)

    if not embeddings:
        return np.array([]), np.array([])

    return np.concatenate(embeddings, axis=0), np.array(labels)


def run_benchmark():
    parser = argparse.ArgumentParser(description="Llama-3 GLUE Evaluation (Trained Probes)")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples per task")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Optional SCA-v2 model directory (contains config.json and neuroplastic_llama_sca_v2.bin).",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model_name = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
    tasks = ["cola", "sst2", "mrpc", "qqp", "stsb", "mnli", "qnli", "rte"]

    logger.info("STARTING LLAMA-3 BENCHMARK (Trained Probe Mode)")
    logger.info(f"Configuration: Uncapped={args.max_samples is None} | Batch={args.batch_size} | Device={device}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if args.checkpoint_dir:
        logger.info(f"Loading SCA-v2 checkpoint from {args.checkpoint_dir}")
        model = NeuroplasticLlama.from_pretrained(
            args.checkpoint_dir,
            neuroplasticity_enabled=True,
        )
    else:
        model = NeuroplasticLlama(
            model_name=base_model_name,
            neuroplasticity_enabled=True,
            sca_block_size=32,
            sca_block_rank=4,
            sca_top_k=3,
            sca_sigma=1.0,
            sca_refractory_steps=100,
            sca_inhibition_lambda=0.0,
            sca_use_cuda=True,
        )
    model.to(device)
    model.eval()

    try:
        model.task_coordinator.task_integrator.task_biological_params["task_0"]["noise_intensity"] = 0.0005
        logger.info("Configured SCA-v2 params: Top-K=3 | Block=32 | Rank=4 | Noise=0.0005")
    except Exception as exc:
        logger.warning(f"Failed to set biological params: {exc}")

    results = {}

    for task in tasks:
        logger.info(f"\nExample Task: {task.upper()}")
        try:
            dataset = load_dataset("glue", task)
            val_split = "validation_matched" if task == "mnli" else "validation"

            train_limit = 5000
            train_data = dataset["train"]
            if len(train_data) > train_limit:
                train_data = train_data.select(range(train_limit))

            logger.info("  -> Extracting Train Embeddings...")
            x_train, y_train = get_embeddings(model, tokenizer, train_data, device, args.batch_size)

            val_data = dataset[val_split]
            if args.max_samples:
                val_data = val_data.select(range(min(len(val_data), args.max_samples)))

            logger.info(f"  -> Extracting Validation Embeddings ({len(val_data)} samples)...")
            x_val, y_val = get_embeddings(model, tokenizer, val_data, device, args.batch_size)

            logger.info("  -> Training Probe...")
            if task == "stsb":
                clf = Ridge(alpha=1.0)
            else:
                clf = LogisticRegression(max_iter=1000, n_jobs=-1)

            clf.fit(x_train, y_train)
            preds = clf.predict(x_val)

            if task == "cola":
                score = matthews_corrcoef(y_val, preds)
                metric = "MCC"
            elif task in ["mrpc", "qqp"]:
                score = accuracy_score(y_val, preds)
                metric = "Accuracy"
            elif task == "stsb":
                from scipy.stats import pearsonr

                score, _ = pearsonr(y_val, preds)
                metric = "Pearson"
            else:
                score = accuracy_score(y_val, preds)
                metric = "Accuracy"

            logger.info(f"-> {task.upper()} RESULT: {score:.4f} ({metric})")
            results[task] = score

        except Exception as exc:
            logger.error(f"Failed Task {task}: {exc}")

    print("\n" + "=" * 50)
    print("FINAL BENCHMARK RESULTS")
    print("=" * 50)
    for task, score in results.items():
        print(f"{task.upper()}: {score:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    run_benchmark()

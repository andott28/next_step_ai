#!/usr/bin/env python3
"""
Modular GLUE evaluation pipeline main orchestrator.
Replaces simple_evaluation_pipeline.py class logic with modular files.
Preserves all logic: seeds per run, MNLI/AX splits, error handling.
Improvements: batch inference, direct STSB pearson, config-driven.
"""

import argparse
import json
import logging
import os
import time
import torch
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Tuple
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from scipy.stats import pearsonr
from datasets import load_dataset

# Local imports
from evaluation import config, data_loader, model_utils, metrics, stats, reporter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model_configs() -> List[Dict[str, str]]:
    """Load model configs from config.py, add neuroplastic if exists."""
    model_configs = config.DEFAULT_MODEL_CONFIGS.copy()
    
    for path in config.NEUROPLASTIC_PATHS:
        if os.path.exists(path):
            model_configs.append({
                'name': 'neuroplastic-transformer',
                'path': path
            })
            logger.info(f"Added neuroplastic model from {path}")
            break
    
    logger.info(f"Loaded {len(model_configs)} model configs: {[c['name'] for c in model_configs]}")
    return model_configs

def get_embeddings(model, tokenizer, dataset, device, max_samples=None, batch_size=32):
    """Extract embeddings for linear probing."""
    embeddings = []
    labels = []
    
    if max_samples:
        dataset = dataset.select(range(min(len(dataset), max_samples)))
    
    # Ensure correct pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        
        # Prepare text (simplified logic)
        texts = []
        if 'sentence' in batch:
            texts = batch['sentence']
        elif 'sentence1' in batch:
            texts = [f"{s1} [SEP] {s2}" for s1, s2 in zip(batch['sentence1'], batch['sentence2'])]
        elif 'premise' in batch:
            texts = [f"{p} [SEP] {h}" for p, h in zip(batch['premise'], batch['hypothesis'])]
        elif 'question' in batch:
            texts = [f"{q} [SEP] {s}" for q, s in zip(batch['question'], batch['sentence'])]
        else:
             texts = [str(x) for x in batch.values()[0]] # Fallback

        if 'label' not in batch:
            continue
            
        inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # Use last hidden state of last token
            last_hidden = outputs.hidden_states[-1]
            pooled = last_hidden[:, -1, :]
            
        embeddings.append(pooled.cpu().numpy())
        labels.extend(batch['label'])
        
    if not embeddings:
        return np.array([]), np.array([])
        
    return np.concatenate(embeddings, axis=0), np.array(labels)

def evaluate_task(
    model: Any,
    tokenizer: Any,
    task_name: str,
    run_seed: int,
    max_samples: int = config.DEFAULT_PARAMS['max_samples'],
    batch_size: int = 32,
    device: torch.device = None
) -> Dict[str, Any]:
    """
    Evaluate single task using TRAINED LINEAR PROBE.
    """
    try:
        logger.info(f"--- Processing {task_name} (Trained Probe Mode) ---")
        
        # Load Data DIRECTLY (Bypass data_loader to get 'train' split)
        dataset = load_dataset('glue', task_name)
        
        # 1. EXTRACT TRAIN FEATURES (Subset for speed)
        train_max = 5000 
        logger.info(f"  -> Extracting Train Features (Max {train_max})...")
        X_train, y_train = get_embeddings(model, tokenizer, dataset['train'], device, train_max, batch_size)
        
        # 2. EXTRACT VAL FEATURES (Full or Capped)
        logger.info(f"  -> Extracting Val Features...")
        if task_name == 'mnli':
            val_data = dataset['validation_matched']
        else:
            val_data = dataset['validation']
            
        X_val, y_val = get_embeddings(model, tokenizer, val_data, device, max_samples, batch_size)
        
        # 3. TRAIN PROBE
        logger.info(f"  -> Training Probe...")
        if task_name == 'stsb':
            clf = Ridge(alpha=1.0)
        else:
            clf = LogisticRegression(max_iter=1000, n_jobs=-1)
            
        clf.fit(X_train, y_train)
        
        # 4. PREDICT
        preds = clf.predict(X_val)
        
        # 5. SCORE
        metric_info = config.GLUE_SUITE[task_name]
        
        if task_name == 'stsb':
            primary_score, _ = pearsonr(y_val, preds)
            metric_results = {'pearson': primary_score}
        elif task_name == 'cola':
            primary_score = matthews_corrcoef(y_val, preds)
            metric_results = {'mcc': primary_score}
        elif task_name in ['mrpc', 'qqp']:
            primary_score = accuracy_score(y_val, preds)
            f1 = f1_score(y_val, preds)
            metric_results = {'accuracy': primary_score, 'f1': f1}
        else:
            primary_score = accuracy_score(y_val, preds)
            metric_results = {'accuracy': primary_score}
            
        logger.info(f"👉 {task_name.upper()} RESULT: {primary_score:.4f}")
        
        # Format for reporter compatibility
        # We dummy references/predictions purely for the reporter logic if needed, 
        # but reporter mainly uses primary_score and results dict.
        
        return {
            'task': task_name,
            'metric': metric_info['metric'],
            'description': metric_info['description'],
            'results': metric_results,
            'primary_score': float(primary_score),
            'num_samples': len(preds),
            'error_rate': 0.0,
            'run_seed': run_seed
        }
    except Exception as e:
        logger.error(f"Task {task_name} failed: {e}")
        return {'task': task_name, 'error': str(e), 'primary_score': 0.0}

def run_evaluation(
    model_configs: List[Dict[str, str]],
    num_runs: int,
    random_seed: int,
    max_samples: int = None,
    batch_size: int = 32
) -> Dict[str, Any]:
    """
    Main evaluation loop: load models, run N times per model/task, compute stats.
    Supports RESUME functionality via 'checkpoint.json'.
    """
    if max_samples is None:
        max_samples = config.DEFAULT_PARAMS['max_samples']
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'timestamp': timestamp,
        'num_runs': num_runs,
        'random_seed': random_seed,
        'models': {}
    }

    # CHECKPOINTING: Load existing checkpoint if it exists
    checkpoint_path = "checkpoint.json"
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            logger.info(f"🔄 Found checkpoint file! Resuming from state...")
            # We merge checkpoint data into 'results', but keep new timestamp/config if needed.
            # actually, simplest is to assume configs didn't change.
            results['models'] = checkpoint_data.get('models', {})
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")

    # Load all models once
    loaded_models = {}
    for config_item in model_configs:
        model, tokenizer, device = model_utils.load_model_and_tokenizer(config_item)
        if model is not None:
            loaded_models[config_item['name']] = (model, tokenizer, device)
        else:
            logger.error(f"Failed to load {config_item['name']}")
    
    # Evaluate each model
    for model_name, (model, tokenizer, device) in loaded_models.items():
        logger.info(f"Evaluating {model_name}")
        
        # Ensure model entry exists
        if model_name not in results['models']:
            results['models'][model_name] = {
                'model_name': model_name,
                'glue_results': {},
                'run_scores': [],
                'successful_evaluations': 0,
                'total_evaluations': 0,
                'overall_score': 0.0
            }
        
        model_results = results['models'][model_name]
        
        # Multiple runs for stats
        for run_id in range(num_runs):
            run_seed = random_seed + run_id
            
            # Set seeds per run
            np.random.seed(run_seed)
            torch.manual_seed(run_seed)
            if torch.cuda.is_available():
                try:
                    torch.cuda.manual_seed_all(run_seed)
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
                except Exception as e:
                    logger.warning(f"Could not set CUDA seed: {e}")
            
            # Skip logging if run seems fully complete? No, just check per task.
            logger.info(f"  Run {run_id+1}/{num_runs} (seed={run_seed})")
            
            run_total = 0
            
            for task_name in config.GLUE_SUITE.keys():
                run_total += 1
                result_key = f"{task_name}_run_{run_id}"

                # RESUME CHECK: If result exists and is valid, skip
                if result_key in model_results['glue_results']:
                    logger.info(f"    ⏩ Skipping {task_name} (Run {run_id}) - Found in checkpoint.")
                    continue

                logger.info(f"    ▶️ Running {task_name} (Run {run_id})...")
                result = evaluate_task(model, tokenizer, task_name, run_seed, max_samples, batch_size, device)
                
                model_results['glue_results'][result_key] = result
                
                # SAVE CHECKPOINT AFTER EACH TASK
                with open(checkpoint_path, 'w') as f:
                    json.dump(results, f, indent=2)
            
            # Recompute run stats dynamically from stored results
            current_run_scores = []
            successful_runs = 0
            for task_key in config.GLUE_SUITE.keys():
                rk = f"{task_key}_run_{run_id}"
                if rk in model_results['glue_results']:
                    res = model_results['glue_results'][rk]
                    if 'error' not in res:
                        current_run_scores.append(res['primary_score'])
                        successful_runs += 1
            
            # This logic is a bit flawed for resume because 'run_scores' list in model_results 
            # might not be populated if we just loaded 'glue_results'. 
            # Better to reconstruct stats at the END of all runs.
        
        # Reconstruct high-level stats from granular glue_results
        # This handles cases where we resumed and 'run_scores' list was empty or partial
        final_run_scores = []
        total_succ = 0
        total_evals = 0
        overall_sum = 0.0

        for r_id in range(num_runs):
            run_s = []
            for t_name in config.GLUE_SUITE.keys():
                total_evals += 1
                rk = f"{t_name}_run_{r_id}"
                if rk in model_results['glue_results']:
                    res = model_results['glue_results'][rk]
                    if 'error' not in res:
                        run_s.append(res['primary_score'])
                        total_succ += 1
            
            if run_s:
                avg = np.mean(run_s)
                final_run_scores.append(avg)
                overall_sum += avg
        
        model_results['run_scores'] = final_run_scores
        model_results['successful_evaluations'] = total_succ
        model_results['total_evaluations'] = total_evals
        model_results['overall_score'] = overall_sum / len(final_run_scores) if final_run_scores else 0.0
        
        # Statistics
        model_results['statistics'] = stats.compute_statistics(model_results['run_scores'])
        
        # Per-task stats
        task_statistics = {}
        for task_name in config.GLUE_SUITE:
            task_run_scores = [
                model_results['glue_results'][f"{task_name}_run_{rid}"]['primary_score']
                for rid in range(num_runs)
                if f"{task_name}_run_{rid}" in model_results['glue_results'] and 'error' not in model_results['glue_results'][f"{task_name}_run_{rid}"]
            ]
            if task_run_scores:
                task_statistics[task_name] = {
                    'statistics': stats.compute_statistics(task_run_scores),
                    'num_successful_runs': len(task_run_scores)
                }
        model_results['task_statistics'] = task_statistics
        
        results['models'][model_name] = model_results
        
        # Save checkpoint after model update
        with open(checkpoint_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    # Overall
    results['summary'] = reporter.create_summary(results)
    results['statistical_analysis'] = stats.compute_overall_statistics(results)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Modular GLUE Evaluation Pipeline")
    parser.add_argument('--num_runs', type=int, default=3, help='Number of runs')
    parser.add_argument('--seed', type=int, default=config.DEFAULT_PARAMS['random_seed'], help='Random seed')
    parser.add_argument('--max_samples', type=int, default=config.DEFAULT_PARAMS['max_samples'], help='Max samples per task')
    parser.add_argument('--batch_size', type=int, default=1, help='Inference batch size (1 = disabled)')
    parser.add_argument('--output_dir', default=config.DEFAULT_PARAMS['output_dir'], help='Output directory')
    args = parser.parse_args()
    
    print("🚀 MODULAR GLUE EVALUATION PIPELINE")
    print("=" * 60)
    
    model_configs = load_model_configs()
    
    results = run_evaluation(
        model_configs,
        args.num_runs,
        args.seed,
        args.max_samples,
        args.batch_size
    )
    
    # Report
    reporter.print_summary(results)
    json_path = reporter.save_results(results, output_dir=args.output_dir)
    md_path = reporter.save_markdown_report(results, output_dir=args.output_dir)
    
    print(f"\n📁 JSON: {json_path}")
    print(f"📄 MD: {md_path}")
    print("\n✅ Evaluation completed!")
    return 0

if __name__ == "__main__":
    exit(main())
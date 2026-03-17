#!/usr/bin/env python3
"""
Simple GLUE Evaluation Pipeline
A streamlined evaluation pipeline focused only on GLUE benchmark tasks.
"""

import json
import os
import time
import argparse
import torch
import numpy as np
from datetime import datetime
import joblib
from typing import Dict, List, Any, Tuple, Optional
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForSequenceClassification, AutoModelForCausalLM
from neuroplastic_llama import NeuroplasticLlama
from neuroplastic_lib.multi_task_error_integration import BiologicalTaskCoordinator
from datasets import load_dataset, load_from_disk
import evaluate
import logging
from scipy import stats
from tqdm import tqdm
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score
from scipy.stats import pearsonr, spearmanr

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        return json.JSONEncoder.default(self, obj)

class EvaluationConfig:
    """Configuration for the evaluation pipeline."""
    def __init__(self):
        self.model_name = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
        self.glue_tasks = ['cola', 'mrpc', 'qqp', 'sst2', 'stsb', 'wnli', 'rte', 'qnli'] # Excluded mnli/ax
        self.batch_size = 1 # Set to 1 for Banked Skip speedup and high precision
        self.max_samples = 1000 # Default to None for full evaluation
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.output_dir = "./llama3_neuroplastic/evaluation_results"
        self.adapter_bottleneck = 256 # Deprecated in SCA-v2 (kept for compatibility)
        self.sca_block_size = 32
        self.sca_block_rank = 4
        self.sca_top_k = 3
        self.sca_sigma = 1.0
        self.sca_refractory_steps = 100
        self.sca_inhibition_lambda = 0.0
        self.sca_use_cuda = False
        self.checkpoint_dir = None
        self.num_runs = 1
        self.random_seed = 42
        
        # Model configurations to compare
        self.model_configs = [
            {
                'name': 'Pure-Llama-3.1-8B',
                'neuroplasticity_enabled': False
            },
            {
                'name': 'NeuroplasticLlama-3.1-8B',
                'neuroplasticity_enabled': True
            }
        ]

class SimpleGLUEEvaluationPipeline:
    """
    A simple evaluation pipeline focused only on GLUE benchmark tasks.
    """
    
    def __init__(self, config: EvaluationConfig):
        """
        Initialize the evaluation pipeline.
        
        Args:
            config: Configuration object for the evaluation pipeline
        """
        self.config = config
        self.models = {}
        self.tokenizers = {}
        self.probes = {} # Dictionary to store loaded joblib probes
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Check for resumable file
        self.resume_path = self._find_resume_file()
        if self.resume_path:
            logger.info(f"â™»ï¸  Found resumable results at: {self.resume_path}")
            self._load_resume_state()
        else:
            self.results = {}
        
        # Set random seed for reproducibility
        np.random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.random_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # Comprehensive GLUE benchmark suite for publication-ready evaluation
        self.glue_suite = {
            'cola': {'metric': 'matthews_correlation', 'description': 'Sentence acceptability'},
            'sst2': {'metric': 'accuracy', 'description': 'Sentiment analysis'},
            'mrpc': {'metric': 'f1', 'description': 'Paraphrase detection'},
            'qqp': {'metric': 'accuracy', 'description': 'Question similarity'},
            'stsb': {'metric': 'pearson', 'description': 'Semantic similarity'},
            'qnli': {'metric': 'accuracy', 'description': 'Question-entailment'},
            'rte': {'metric': 'accuracy', 'description': 'Textual entailment'},
            'wnli': {'metric': 'accuracy', 'description': 'Word-level natural language inference'}
        }
        
        logger.info(f"Initialized GLUE evaluation pipeline for model: {self.config.model_name}")
        logger.info(f"Statistical analysis: {self.config.num_runs} runs, random seed: {self.config.random_seed}")
    
    def _clean_text(self, text: str) -> str:
        """Clean text of problematic characters and whitespace."""
        if not isinstance(text, str):
            text = str(text)
        # Remove strange characters and normalize whitespace
        text = text.encode('ascii', 'ignore').decode('ascii')
        text = " ".join(text.split())
        return text

    def _find_resume_file(self) -> Optional[str]:
        """Find the most recent partial result file."""
        try:
            results_dir = self.config.output_dir
            if not os.path.exists(results_dir):
                return None
            
            files = [os.path.join(results_dir, f) for f in os.listdir(results_dir) 
                    if f.startswith("glue_evaluation_results_") and f.endswith(".json")]
            
            if not files:
                return None
            
            # Get latest file that is substantial (ignore crashes/empty starts < 5KB)
            valid_files = [f for f in files if os.path.getsize(f) > 5120]
            
            if not valid_files:
                logger.warning("No substantial resume files found (all < 5KB). Starting fresh.")
                return None
                
            latest_file = max(valid_files, key=os.path.getmtime)
            
            # Trust the latest substantial file
            logger.info(f"Found most recent valid resume file: {latest_file} ({os.path.getsize(latest_file)} bytes)")
            return latest_file
        except Exception as e:
            logger.warning(f"Error checking for resume file: {e}")
            return None

    def _load_resume_state(self):
        """Load state from resume file."""
        try:
            with open(self.resume_path, 'r') as f:
                self.results = json.load(f)
            # Restore timestamp to keep appending to same file
            self.timestamp = self.results.get('timestamp', self.timestamp)
            logger.info(f"Loaded {len(self.results.get('models', {}))} models from previous session.")
        except Exception as e:
            logger.error(f"Failed to load resume file: {e}")
            self.results = {}

    def compute_statistics(self, scores: List[float]) -> Dict[str, float]:
        """
        Compute statistical measures for a list of scores.
        
        Args:
            scores: List of scores from multiple runs
            
        Returns:
            Dictionary with statistical measures
        """
        if len(scores) < 2:
            return {
                'mean': scores[0] if scores else 0.0,
                'std': 0.0,
                'ci_95': 0.0,
                'min': scores[0] if scores else 0.0,
                'max': scores[0] if scores else 0.0,
                'median': scores[0] if scores else 0.0
            }
        
        scores_array = np.array(scores)
        mean = np.mean(scores_array)
        std = np.std(scores_array, ddof=1)  # Sample standard deviation
        sem = std / np.sqrt(len(scores))  # Standard error of mean
        ci_95 = sem * stats.t.ppf(0.975, len(scores) - 1)  # 95% CI
        median = np.median(scores_array)
        
        return {
            'mean': float(mean),
            'std': float(std),
            'ci_95': float(ci_95),
            'min': float(np.min(scores_array)),
            'max': float(np.max(scores_array)),
            'median': float(median),
            'n_runs': len(scores)
        }
    
    def compute_overall_statistics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute overall statistical analysis across all models.
        """
        analysis = {
            'model_comparisons': {},
            'statistical_significance': {},
            'performance_gaps': {}
        }
        
        model_names = list(results['models'].keys())
        if len(model_names) >= 2:
            for i, model1 in enumerate(model_names):
                for j, model2 in enumerate(model_names[i+1:], i+1):
                    model1_scores = results['models'][model1]['run_scores']
                    model2_scores = results['models'][model2]['run_scores']
                    
                    if len(model1_scores) >= 2 and len(model2_scores) >= 2:
                        t_stat, p_value = stats.ttest_ind(model1_scores, model2_scores)
                        analysis['statistical_significance'][f"{model1}_vs_{model2}"] = {
                            't_statistic': float(t_stat),
                            'p_value': float(p_value),
                            'significant': bool(p_value < 0.05)
                        }
                        
                        mean1 = np.mean(model1_scores)
                        mean2 = np.mean(model2_scores)
                        improvement = ((mean2 - mean1) / mean1) * 100 if mean1 != 0 else 0
                        analysis['performance_gaps'][f"{model1}_vs_{model2}"] = {
                            'model1_mean': float(mean1),
                            'model2_mean': float(mean2),
                            'improvement_percent': float(improvement),
                            'absolute_difference': float(mean2 - mean1)
                        }
                    
                    analysis['model_comparisons'][f"{model1}_vs_{model2}"] = {
                        'model1': model1,
                        'model2': model2,
                        'model1_stats': results['models'][model1]['statistics'],
                        'model2_stats': results['models'][model2]['statistics']
                    }
        
        return analysis
    
    def _load_single_model(self, config: Dict[str, Any]) -> bool:
        """
        Load a single model configuration into memory.
        """
        model_name = config['name']
        logger.info(f"Loading {model_name}...")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        import gc
        gc.collect()
        time.sleep(1)
            
        try:
            logger.info(f"Loading tokenizer for {self.config.model_name}")
            tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # CRITICAL FIX: Ensure model config knows about pad token
            # This prevents standard models from outputting garbage if they rely on it
            if tokenizer.pad_token_id is not None:
                # Ensure the model config knows about the pad token to prevent issues
                # with internal attention masking or generation if applicable.
                pass # The model instance isn't created yet; we handle it below.
            
            neuro_enabled = config.get('neuroplasticity_enabled', True)
            checkpoint_dir = config.get('checkpoint_dir') or self.config.checkpoint_dir
            if checkpoint_dir and neuro_enabled:
                logger.info(f"Loading neuroplastic checkpoint from {checkpoint_dir}")
                model = NeuroplasticLlama.from_pretrained(
                    checkpoint_dir,
                    neuroplasticity_enabled=neuro_enabled,
                )
            else:
                model = NeuroplasticLlama(
                    model_name=self.config.model_name,
                    num_tasks=len(self.config.glue_tasks),
                    adapter_bottleneck=self.config.adapter_bottleneck,
                    neuroplasticity_enabled=neuro_enabled,
                    sca_block_size=self.config.sca_block_size,
                    sca_block_rank=self.config.sca_block_rank,
                    sca_top_k=self.config.sca_top_k,
                    sca_sigma=self.config.sca_sigma,
                    sca_refractory_steps=self.config.sca_refractory_steps,
                    sca_inhibition_lambda=self.config.sca_inhibition_lambda,
                    sca_use_cuda=self.config.sca_use_cuda,
                )
            
            self.models[model_name] = model
            self.tokenizers[model_name] = tokenizer
            
            # CRITICAL FIX: Explicitly sync pad_token_id to model config
            if tokenizer.pad_token_id is not None:
                if hasattr(model, 'config'):
                    model.config.pad_token_id = tokenizer.pad_token_id
                if hasattr(model, 'model') and hasattr(model.model, 'config'):
                    model.model.config.pad_token_id = tokenizer.pad_token_id
            
            logger.info(f"Successfully loaded {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.models[model_name] = None
            self.tokenizers[model_name] = None
            return False
    
    def _unload_model(self, model_name: str):
        """Unload model from memory to free resources."""
        if model_name in self.models:
            del self.models[model_name]
        if model_name in self.tokenizers:
            del self.tokenizers[model_name]
            
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"Unloaded {model_name} and cleared cache")
    
    def evaluate_glue_task(self, model_name: str, task_name: str, run_id: int = 0, cache_file: str = None) -> Dict[str, Any]:
        """
        Evaluate a model on a GLUE task with error tracking and intra-task checkpointing.
        """
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        
        # PROBE LOADING LOGIC (Global for this task)
        # Check if we have a pre-trained probe for this task
        probe_key = f"{task_name}_T1.5" # Default to our critical threshold
        if probe_key not in self.probes:
             # Look in root/probes/ or current_dir/probes/
             possible_paths = [
                 f"probes/llama_probe_{task_name}_T1.5.joblib",
                 f"../probes/llama_probe_{task_name}_T1.5.joblib",
                 f"probes/llama_probe_{task_name}_T1.0.joblib"
             ]
             for p in possible_paths:
                 if os.path.exists(p):
                     logger.info(f"ðŸŽ¯ Loading trained probe from {p}")
                     self.probes[probe_key] = joblib.load(p)
                     break
             
             if probe_key not in self.probes:
                 logger.info(f"No pre-trained probe found for {task_name}. Using model default head. For probe-based evaluation without saved probe files, use run_llama_benchmark_1000.py with --checkpoint-dir to train a probe on the task split.")
                 self.probes[probe_key] = None

        if model is None or tokenizer is None:
            return {'error': 'Model or tokenizer not loaded'}
        
        try:
            dataset = load_dataset('glue', task_name)
            if task_name == 'mnli':
                test_data = dataset['validation_matched']
            elif task_name == 'ax':
                test_data = dataset['test']
            elif 'validation' in dataset:
                test_data = dataset['validation']
            else:
                test_data = next(iter(dataset.values())) # fallback
            
            metric_info = self.glue_suite[task_name]
            metric_name = metric_info['metric']
            
            try:
                # Use 'glue' config for robust loading
                if task_name in ['stsb', 'mrpc', 'qqp', 'mnli', 'qnli', 'rte', 'wnli', 'cola', 'sst2']:
                    metric = evaluate.load("glue", task_name)
                else:
                    metric = evaluate.load(metric_name)
            except Exception as e:
                logger.warning(f"Could not load metric '{metric_name}' for {task_name}: {e}")
                if task_name == 'stsb':
                    metric_name = 'mse'
                    metric = evaluate.load(metric_name)
                else:
                    raise
            
            predictions = []
            references = []
            processing_errors = []
            start_offset = 0

            # RESUME LOGIC (Intra-Task)
            if cache_file and os.path.exists(cache_file):
                try:
                    import json
                    with open(cache_file, 'r') as f:
                        cached_data = json.load(f)
                    predictions = cached_data.get('predictions', [])
                    references = cached_data.get('references', [])
                    start_offset = len(predictions)
                    logger.info(f"ðŸ”„ Resuming {task_name} from sample {start_offset} (loaded {len(predictions)} saved results)")
                except Exception as e:
                    logger.warning(f"Failed to load cache file {cache_file}: {e}")
            
            max_samples = self.config.max_samples
            run_seed = self.config.random_seed + run_id
            indices = self._get_random_samples(test_data, max_samples, run_seed)
            total_samples = len(indices)

            # Slice indices to process only remaining
            remaining_indices = indices[start_offset:]

            if not remaining_indices and indices:
                 logger.info(f"Task {task_name} already fully completed in cache!")
                 # We need to compute metrics on the loaded predictions/references
                 # Logic follows after the loop

            logger.info(f"Processing {len(remaining_indices)} remaining samples ({total_samples} total) for {task_name}")

            batch_size = self.config.batch_size if hasattr(self.config, 'batch_size') else 1
            
            # Wrap with tqdm for progress visibility
            for i in tqdm(range(0, len(remaining_indices), batch_size), desc=f"Processing {task_name}", unit="sample"):
                batch_indices = remaining_indices[i:i + batch_size]
                batch_texts = []
                batch_references = []
                
                for idx in batch_indices:
                    try:
                        sample = test_data[idx]
                        if task_name == 'cola' or task_name == 'sst2':
                            text = sample['sentence']
                        elif task_name in ['mrpc', 'stsb', 'rte', 'wnli']:
                            text = f"{sample['sentence1']} [SEP] {sample['sentence2']}"
                        elif task_name == 'qqp':
                            text = f"{sample['question1']} [SEP] {sample['question2']}"
                        elif task_name == 'mnli' or task_name == 'ax':
                            text = f"{sample['premise']} [SEP] {sample['hypothesis']}"
                        elif task_name == 'qnli':
                            text = f"{sample['question']} [SEP] {sample['sentence']}"
                        else:
                            text = sample.get('sentence', str(sample))
                        
                        text = self._clean_text(text)
                        batch_texts.append(text)
                        batch_references.append(sample['label'])
                    except Exception as e:
                        processing_errors.append({'sample_index': idx, 'error': str(e)})
                        continue
                
                if not batch_texts:
                    continue
                
                try:
                    inputs = tokenizer(batch_texts, truncation=True, max_length=512, padding=True, return_tensors="pt")
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        if task_name == 'stsb' and hasattr(model, 'predict_regression'):
                             batch_preds = model.predict_regression(**inputs)
                             predictions.extend(batch_preds.view(-1).cpu().tolist())
                        elif hasattr(model, 'predict_classification') and task_name not in ['mnli', 'stsb', 'ax']:
                             # Check if we have a trained probe to use
                             trained_probe = self.probes.get(probe_key)
                             
                             if trained_probe and hasattr(model, 'forward'):
                                 # Get hidden states and use trained probe
                                 outputs = model.forward(
                                     input_ids=inputs['input_ids'], 
                                     attention_mask=inputs['attention_mask'],
                                     output_hidden_states=True
                                 )
                                 last_hidden = outputs.hidden_states[-1]
                                 # Pool last token (BS=1)
                                 seq_len = inputs['attention_mask'].sum(dim=1) - 1
                                 seq_len = torch.clamp(seq_len, min=0)
                                 pooled = last_hidden[0, seq_len[0]].float().cpu().numpy().reshape(1, -1)
                                 pred = trained_probe.predict(pooled)[0]
                                 predictions.append(pred)
                             else:
                                 # Fallback to model's built-in head
                                 batch_preds = model.predict_classification(**inputs)
                                 predictions.extend(batch_preds.view(-1).cpu().tolist())
                        else:
                             outputs = model(**inputs)
                             logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                             if task_name == 'stsb':
                                 predictions.extend(logits.view(-1).cpu().tolist())
                             else:
                                 probs = torch.argmax(logits, dim=-1)
                                 if len(probs.shape) > 1: probs = probs[:, -1]
                                 predictions.extend(probs.view(-1).cpu().tolist())
                    
                    references.extend(batch_references)
                    del inputs
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    
                except Exception as sample_error:
                    processing_errors.append({'sample_index': i, 'error': str(sample_error)})

                # Aggressive GC/Cache Clear
                if i > 0 and i % 200 == 0:
                    try:
                        import gc
                        if 'inputs' in locals(): del inputs
                        if 'outputs' in locals(): del outputs
                        if 'logits' in locals(): del logits
                    except: pass
                    import gc
                    gc.collect()
                    if torch.cuda.is_available(): torch.cuda.empty_cache()

                # INCREMENTAL SAVE (Intra-Task Checkpoint)
                if cache_file and i > 0 and i % 500 == 0:
                    try:
                        import json
                        with open(cache_file, 'w') as f:
                            json.dump({'predictions': predictions, 'references': references}, f)
                    except Exception as e:
                        logger.warning(f"Failed to save partial cache: {e}")
            
            # Cleanup cache on success
            if cache_file and os.path.exists(cache_file):
                try: os.remove(cache_file)
                except: pass
            try:
                if metric_name == 'mse':
                    mse = np.mean((np.array(predictions) - np.array(references)) ** 2)
                    results = {'mse': mse}
                else:
                    results = metric.compute(predictions=predictions, references=references)
            except Exception as e:
                logger.error(f"Metric computation failed for {task_name}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                results = {'score': 0.0, 'metric_error': str(e)}
            
            primary_score = 0.0
            if isinstance(results, dict):
                if 'accuracy' in results: primary_score = results['accuracy']
                elif 'f1' in results: primary_score = results['f1']
                elif 'matthews_correlation' in results: primary_score = results['matthews_correlation']
                elif 'pearson' in results: primary_score = results['pearson']
                elif 'mse' in results: primary_score = 1.0 / (1.0 + results['mse'])
                elif 'score' in results: primary_score = results['score']
            
            return {
                'task': task_name,
                'metric': metric_name,
                'description': metric_info['description'],
                'results': results,
                'primary_score': primary_score,
                'num_samples': total_samples,
                'num_errors': len(processing_errors),
                'processing_errors': processing_errors[:5],
                'model_name': model_name,
                'run_id': run_id
            }
            
        except Exception as e:
            logger.error(f"Error evaluating {task_name}: {e}")
            return {'error': str(e), 'run_id': run_id}

    def _get_random_samples(self, dataset, num_samples: int = None, seed: int = 42) -> List[int]:
        rs = np.random.RandomState(seed)
        num_available = len(dataset)
        actual_num = min(num_samples, num_available) if num_samples is not None else num_available
        indices = rs.choice(num_available, actual_num, replace=False)
        return indices.tolist()
    
    def run_glue_evaluation(self) -> Dict[str, Any]:
        """
        Run GLUE evaluation for the configured Llama-3 models with statistical analysis.
        """
        logger.info(f"Starting GLUE evaluation across {len(self.config.model_configs)} models...")
        
        results = {
            'timestamp': self.timestamp,
            'num_runs': self.config.num_runs,
            'random_seed': self.config.random_seed,
            'models': self.results.get('models', {}),
            'summary': {},
            'statistical_analysis': {}
        }
        
        for config in self.config.model_configs:
            model_name = config['name']
            if not self._load_single_model(config): continue
                    
            logger.info(f"Evaluating {model_name}...")
            
            if model_name in results['models']:
                model_results = results['models'][model_name]
            else:
                model_results = {
                    'model_name': model_name,
                    'base_model': self.config.model_name,
                    'glue_results': {},
                    'overall_score': 0.0,
                    'successful_evaluations': 0,
                    'total_evaluations': 0,
                    'run_scores': []
                }
            
            for run_id in range(self.config.num_runs):
                logger.info(f"  Run {run_id + 1}/{self.config.num_runs}")
                run_scores = []
                run_successful = 0
                
                for task_name in self.config.glue_tasks:
                    result_key = f"{task_name}_run_{run_id}"
                    
                    if result_key in model_results['glue_results']:
                        existing = model_results['glue_results'][result_key]
                        if 'error' not in existing and 'primary_score' in existing:
                            if not (isinstance(existing['primary_score'], float) and np.isnan(existing['primary_score'])):
                                logger.info(f"    Skipping {task_name} (already done)")
                                run_scores.append(existing['primary_score'])
                                run_successful += 1
                                continue

                    # Intra-task caching
                    cache_dir = os.path.join(self.config.output_dir, "partial_cache")
                    os.makedirs(cache_dir, exist_ok=True)
                    sanitized_model = model_name.replace("/", "_").replace("\\", "_")
                    cache_file = os.path.join(cache_dir, f"{sanitized_model}_{task_name}_run{run_id}_partial.json")

                    logger.info(f"    Running {task_name}...")
                    result = self.evaluate_glue_task(model_name, task_name, run_id, cache_file=cache_file)
                    model_results['glue_results'][result_key] = result
                    if 'error' not in result:
                        run_successful += 1
                        run_scores.append(result['primary_score'])
                    
                    results['models'][model_name] = model_results
                    self.results = results
                    self.save_results()
                
                run_overall_score = sum(run_scores) / len(run_scores) if run_scores else 0.0
                if run_id < len(model_results['run_scores']):
                    model_results['run_scores'][run_id] = run_overall_score
                else:
                    model_results['run_scores'].append(run_overall_score)
            
            model_results['successful_evaluations'] = len([k for k, v in model_results['glue_results'].items() if 'error' not in v and not (isinstance(v.get('primary_score'), float) and np.isnan(v.get('primary_score')))])
            model_results['total_evaluations'] = len(self.config.glue_tasks) * self.config.num_runs
            model_results['overall_score'] = sum(model_results['run_scores']) / len(model_results['run_scores']) if model_results['run_scores'] else 0.0
            model_results['statistics'] = self.compute_statistics(model_results['run_scores'])
            
            results['models'][model_name] = model_results
            self.results = results
            self.save_results()
            self._unload_model(model_name)
            
        results['summary'] = self.create_summary(results)
        results['statistical_analysis'] = self.compute_overall_statistics(results)
        self.results = results
        self.save_results()
        
        logger.info("GLUE evaluation completed!")
        return results

    def create_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        summary = {
            'total_models': len(results['models']),
            'successful_models': 0,
            'failed_models': 0,
            'best_model': None,
            'best_score': 0.0,
            'rankings': [],
            'average_scores': {},
            'success_rates': {},
            'statistical_summary': {}
        }
        
        model_scores = []
        for model_name, model_results in results['models'].items():
            score = model_results.get('overall_score', 0.0)
            success_rate = model_results.get('successful_evaluations', 0) / max(1, model_results.get('total_evaluations', 1))
            statistics = model_results.get('statistics', {})
            model_scores.append((model_name, score, success_rate, statistics))
            summary['average_scores'][model_name] = score
            summary['success_rates'][model_name] = success_rate
            summary['statistical_summary'][model_name] = statistics
            if score > 0: summary['successful_models'] += 1
            else: summary['failed_models'] += 1
        
        model_scores.sort(key=lambda x: x[1], reverse=True)
        for i, (model_name, score, success_rate, statistics) in enumerate(model_scores):
            summary['rankings'].append({'rank': i + 1, 'model_name': model_name, 'score': score, 'success_rate': success_rate, 'statistics': statistics})
            if i == 0:
                summary['best_model'] = model_name
                summary['best_score'] = score
        
        if 'statistical_analysis' in results:
            sig = results['statistical_analysis'].get('statistical_significance', {})
            gaps = results['statistical_analysis'].get('performance_gaps', {})
            summary['statistical_insights'] = {
                'significant_differences': len([k for k, v in sig.items() if v.get('significant', False)]),
                'total_comparisons': len(sig),
                'best_improvement': max([v.get('improvement_percent', 0) for v in gaps.values()], default=0)
            }
        return summary
    
    def save_results(self, output_path: str = None) -> str:
        if output_path is None:
            output_path = os.path.join(self.config.output_dir, f"glue_evaluation_results_{self.timestamp}.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        logger.info(f"Results saved to: {output_path}")
        return output_path
    
    def print_summary(self):
        if not self.results: return
        print("\n" + "="*80)
        print("GLUE EVALUATION SUMMARY")
        print("="*80)
        print(f"Timestamp: {self.results['timestamp']}")
        for model_name, model_results in self.results['models'].items():
            print(f"\nðŸ¤– {model_name}:")
            print(f"   Overall Score: {model_results.get('overall_score', 0.0):.4f}")
            if 'statistics' in model_results:
                stats = model_results['statistics']
                print(f"   Mean: {stats.get('mean', 0.0):.4f} Â± {stats.get('std', 0.0):.4f}")
        print("\n" + "="*80)

def main():
    config = EvaluationConfig()
    parser = argparse.ArgumentParser(description="Run GLUE evaluation for NeuroplasticLlama.")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Optional SCA-v2 model directory (contains config.json and neuroplastic_llama_sca_v2.bin).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on samples per task.",
    )
    args = parser.parse_args()
    if args.checkpoint_dir:
        config.checkpoint_dir = args.checkpoint_dir
    if args.max_samples is not None:
        config.max_samples = args.max_samples
    pipeline = SimpleGLUEEvaluationPipeline(config)
    results = pipeline.run_glue_evaluation()
    pipeline.print_summary()
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())


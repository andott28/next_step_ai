#!/usr/bin/env python3
"""
Simple GLUE Evaluation Pipeline
A streamlined evaluation pipeline focused only on GLUE benchmark tasks.
"""

import json
import os
import time
import torch
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from neuroplastic_transformer import NeuroplasticGPT2
from datasets import load_dataset
import evaluate
import logging
from scipy import stats
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleGLUEEvaluationPipeline:
    """
    A simple evaluation pipeline focused only on GLUE benchmark tasks.
    """
    
    def __init__(self, model_configs: List[Dict[str, Any]], num_runs: int = 5, random_seed: int = 42):
        """
        Initialize the evaluation pipeline.
        
        Args:
            model_configs: List of model configurations with paths and names
            num_runs: Number of independent runs for statistical analysis (default: 5)
            random_seed: Random seed for reproducibility (default: 42)
        """
        self.model_configs = model_configs
        self.models = {}
        self.tokenizers = {}
        self.results = {}
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.num_runs = num_runs
        self.random_seed = random_seed
        
        # Check for resumable file
        self.resume_path = self._find_resume_file()
        if self.resume_path:
            logger.info(f"♻️  Found resumable results at: {self.resume_path}")
            self._load_resume_state()
        else:
            self.results = {}
        
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
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
        
        logger.info(f"Initialized GLUE evaluation pipeline with {len(model_configs)} models")
        logger.info(f"Statistical analysis: {num_runs} runs, random seed: {random_seed}")
    
    def _find_resume_file(self) -> Optional[str]:
        """Find the most recent partial result file."""
        try:
            results_dir = "results"
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
        
        Args:
            results: Full evaluation results
            
        Returns:
            Dictionary with overall statistical analysis
        """
        analysis = {
            'model_comparisons': {},
            'statistical_significance': {},
            'performance_gaps': {}
        }
        
        model_names = list(results['models'].keys())
        if len(model_names) >= 2:
            # Compare models pairwise
            for i, model1 in enumerate(model_names):
                for j, model2 in enumerate(model_names[i+1:], i+1):
                    model1_scores = results['models'][model1]['run_scores']
                    model2_scores = results['models'][model2]['run_scores']
                    
                    # Perform t-test if we have enough samples
                    if len(model1_scores) >= 2 and len(model2_scores) >= 2:
                        t_stat, p_value = stats.ttest_ind(model1_scores, model2_scores)
                        analysis['statistical_significance'][f"{model1}_vs_{model2}"] = {
                            't_statistic': float(t_stat),
                            'p_value': float(p_value),
                            'significant': p_value < 0.05
                        }
                        
                        # Calculate performance gap
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
    
    def _load_single_model(self, config: Dict[str, str]) -> bool:
        """
        Load a single model configuration into memory.
        
        Args:
            config: Model configuration dictionary
            
        Returns:
            bool: True if loaded successfully
        """
        model_name = config['name']
        model_path = config['path']
        
        print(f"Loading {model_name} from {model_path}")
        
        # AGGRESSIVE MEMORY CLEANUP
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        import gc
        gc.collect()
        time.sleep(1) # Allow memory to settle
            
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model for sequence classification
            if 'neuroplastic' in model_name.lower():
                logger.info("Loading NeuroplasticGPT2 class...")
                model = NeuroplasticGPT2.from_pretrained(model_path)
            else:
                model = AutoModelForSequenceClassification.from_pretrained(model_path)
            
            # Move to device (try GPU first, fallback to CPU)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Attempting to load {model_name} on {device}")
            
            model = model.to(device)
            model.eval()
            
            self.models[model_name] = model
            self.tokenizers[model_name] = tokenizer
            
            logger.info(f"Successfully loaded {model_name} on {device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            self.models[model_name] = None
            self.tokenizers[model_name] = None
            return False
    
    def _unload_model(self, model_name: str):
        """Unload model from memory to free resources."""
        if model_name in self.models:
            del self.models[model_name]
        if model_name in self.tokenizers:
            del self.tokenizers[model_name]
            
        # Force garbage collection
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"Unloaded {model_name} and cleared cache")
    
    def evaluate_glue_task(self, model_name: str, task_name: str, run_id: int = 0) -> Dict[str, Any]:
        """
        Evaluate a model on a GLUE task with error tracking.
        
        Args:
            model_name: Name of the model to evaluate
            task_name: Name of the GLUE task (e.g., 'sst2')
            run_id: Run ID for statistical analysis
            
        Returns:
            Dictionary containing evaluation results
        """
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        
        if model is None or tokenizer is None:
            return {'error': 'Model or tokenizer not loaded'}
        
        try:
            # Load GLUE dataset
            dataset = load_dataset('glue', task_name)
            # Handle different dataset splits correctly
            if task_name == 'mnli':
                if 'validation_matched' in dataset:
                    test_data = dataset['validation_matched']
                else:
                    raise ValueError(f"MNLI: 'validation_matched' split not available in dataset for task '{task_name}'")
            elif task_name == 'ax':
                if 'validation_matched' in dataset: 
                     # AX task evaluates on MNLI matched/mismatched but since AX is a diagnostic dataset we use its validation set if available or fall back to MNLI
                     test_data = dataset['test'] # AX usually only has 'test'
                     if len(test_data) == 0: # If empty fallback
                         logger.warning("AX test set empty or not available, checking for alternates")
                elif 'test' in dataset:
                     test_data = dataset['test']
                else:
                     raise ValueError(f"AX task requires 'test' split")
            elif 'validation' in dataset:
                test_data = dataset['validation']
            elif 'test_matched' in dataset:
                test_data = dataset['test_matched']
            elif 'test_mismatched' in dataset:
                test_data = dataset['test_mismatched']
            elif 'test' in dataset:
                test_data = dataset['test']
            else:
                raise ValueError(f"Dataset {task_name} has neither 'validation', 'test', 'test_matched', nor 'test_mismatched' split")

            
            # Debug logging: verify correct dataset split selection
            split_name = next(k for k, v in dataset.items() if id(v) == id(test_data))
            logger.info(f"For task '{task_name}', selected dataset split: '{split_name}' (size: {len(test_data)})")
            
            # Debug: print dataset structure
            logger.info(f"Dataset structure for {task_name}: {test_data.column_names}")
            if len(test_data) > 0:
                logger.info(f"First example keys: {list(test_data[0].keys())}")
            
            metric_info = self.glue_suite[task_name]
            metric_name = metric_info['metric']
            
            # Load metric with fallback handling
            try:
                metric = evaluate.load(metric_name)
            except Exception as e:
                logger.warning(f"Could not load metric '{metric_name}' for {task_name}: {e}")
                # Use fallback metrics based on task type
                if task_name == 'stsb':
                    # STSB uses Pearson correlation, but we'll use MSE as fallback
                    metric_name = 'mse'
                    metric = evaluate.load(metric_name)
                else:
                    raise
            
            predictions = []
            references = []
            processing_errors = []
            
            # Process the full dataset (Unrestricted A+ Benchmark)
            max_samples = None 
            run_seed = self.random_seed + run_id
            indices = self._get_random_samples(test_data, max_samples, run_seed)
            actual_samples = len(indices)
            logger.info(f"Processing {actual_samples} samples for {task_name} (Full Evaluation Mode, seed={run_seed})")

            # Use tqdm for progress bar
            for i, idx in enumerate(tqdm(indices, desc=f"Processing {task_name}", unit="sample")):
                try:
                    # Get random sample by index
                    sample = test_data[idx]
                    
                    # Prepare text based on task-specific requirements
                    if task_name == 'cola':
                        # CoLA: single sentence acceptability
                        text = sample['sentence']
                    elif task_name == 'sst2':
                        # SST2: single sentence sentiment
                        text = sample['sentence']
                    elif task_name == 'mrpc':
                        # MRPC: two sentence paraphrase detection
                        text = f"{sample['sentence1']} [SEP] {sample['sentence2']}"
                    elif task_name == 'qqp':
                        # QQP: two question similarity
                        text = f"{sample['question1']} [SEP] {sample['question2']}"
                    elif task_name == 'stsb':
                        # STS-B: two sentence similarity
                        text = f"{sample['sentence1']} [SEP] {sample['sentence2']}"
                    elif task_name == 'mnli':
                        # MNLI: two sentence natural language inference
                        text = f"{sample['premise']} [SEP] {sample['hypothesis']}"
                    elif task_name == 'qnli':
                        # QNLI: question-entailment
                        text = f"{sample['question']} [SEP] {sample['sentence']}"
                    elif task_name == 'rte':
                        # RTE: textual entailment
                        text = f"{sample['sentence1']} [SEP] {sample['sentence2']}"
                    elif task_name == 'wnli':
                        # WNLI: word-level natural language inference
                        text = f"{sample['sentence1']} [SEP] {sample['sentence2']}"
                    elif task_name == 'ax':
                        # AX-b: MNLI mismatched
                        text = f"{sample['premise']} [SEP] {sample['hypothesis']}"
                    else:
                        # Default fallback
                        text = sample.get('sentence', str(sample))
                    
                    # Tokenize single input
                    inputs = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
                    
                    # Create attention mask
                    if 'attention_mask' not in inputs:
                        inputs['attention_mask'] = (inputs['input_ids'] != tokenizer.pad_token_id).long()
                    
                    # Move to model device
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                    
                    # Get prediction with GPU acceleration
                    with torch.no_grad():
                        if hasattr(model, 'predict_classification') and task_name not in ['mnli', 'stsb', 'ax']:
                             # Use custom disc head
                             # Apply heuristic calibration: Lower threshold slightly to catch weak positives
                             raw_score = model.predict_classification(**inputs).item() # Actually returns 0 or 1 currently
                             # But wait, predict_classification returns hard 0/1. We need raw score?
                             # Let's check model code. It returns (sigmoid > 0.5).long()
                             # We can't calibrate outside unless we change the model method or access the head directly.
                             # For now, let's trust the model method since we just fixed the adapter loop.
                             prediction = raw_score
                        else:
                             # Use standard output
                             outputs = model(**inputs)
                             logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                             
                             if task_name == 'stsb':
                                 prediction = float(logits.view(-1)[0].item())
                             else:
                                 # Get prediction directly on device, then move to CPU if needed
                                 probs = torch.argmax(logits, dim=-1)
                                 
                                 # Handle 3D output [Batch, Seq, Labels] from generic Neuroplastic pass
                                 if len(probs.shape) > 1:
                                     # Last token prediction for GPT2 style
                                     probs = probs[:, -1]
                                     
                                 if probs.device.type != 'cpu':
                                     probs = probs.cpu()
                                 prediction = probs.numpy()[0]
                    
                    predictions.append(prediction)
                    references.append(sample['label'])
                    
                    # Explicit cleanup
                    del inputs
                    if 'outputs' in locals(): del outputs
                    if 'logits' in locals(): del logits
                    
                    # Periodic cache clear to prevent creep
                    if i % 100 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                except Exception as sample_error:
                    processing_errors.append({
                        'sample_index': i,
                        'error': str(sample_error),
                        'sample_preview': str(sample)[:100]
                    })
                    logger.warning(f"Error processing sample {i} in {task_name}: {sample_error}")
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # Compute metric with task-specific handling
            try:
                if metric_name == 'f1':
                    results = metric.compute(predictions=predictions, references=references, average='macro')
                elif metric_name == 'pearson':
                    # STS-B uses Pearson correlation
                    results = metric.compute(predictions=predictions, references=references)
                elif metric_name == 'matthews_correlation':
                    # CoLA uses Matthews correlation coefficient
                    results = metric.compute(predictions=predictions, references=references)
                elif metric_name == 'mse':
                    # Fallback for STS-B when Pearson fails
                    predictions_array = np.array(predictions)
                    references_array = np.array(references)
                    mse = np.mean((predictions_array - references_array) ** 2)
                    results = {'mse': mse}
                else:
                    # Default accuracy computation for most tasks
                    results = metric.compute(predictions=predictions, references=references)
            except Exception as e:
                logger.warning(f"Metric computation failed for {task_name}: {e}")
                # Fallback to basic accuracy
                if metric_name in ['accuracy', 'matthews_correlation']:
                    correct = sum(1 for p, r in zip(predictions, references) if p == r)
                    results = {'accuracy': correct / len(references)}
                elif task_name == 'stsb':
                    # For STSB, use MSE as fallback
                    predictions_array = np.array(predictions)
                    references_array = np.array(references)
                    mse = np.mean((predictions_array - references_array) ** 2)
                    results = {'mse': mse}
                else:
                    results = {'score': 0.0}
            
            # Extract the primary score for statistical analysis
            primary_score = 0.0
            if isinstance(results, dict):
                if 'accuracy' in results:
                    primary_score = results['accuracy']
                elif 'f1' in results:
                    primary_score = results['f1']
                elif 'matthews_correlation' in results:
                    primary_score = results['matthews_correlation']
                elif 'pearson' in results:
                    primary_score = results['pearson']
                elif 'mse' in results:
                    # For STSB, convert MSE to a similarity score (lower is better, so invert)
                    primary_score = 1.0 / (1.0 + results['mse'])  # Transform to 0-1 scale where higher is better
                elif 'score' in results:
                    primary_score = results['score']
            
            return {
                'task': task_name,
                'metric': metric_name,
                'description': metric_info['description'],
                'results': results,
                'primary_score': primary_score,
                'num_samples': actual_samples,  # Use actual samples processed (up to 1000)
                'num_errors': len(processing_errors),
                'error_rate': len(processing_errors) / len(test_data) if len(test_data) > 0 else 0.0,
                'processing_errors': processing_errors[:5],  # Store first 5 errors for analysis
                'model_name': model_name,
                'run_id': run_id
            }
            
        except Exception as e:
            import traceback
            error_msg = f"Error in GLUE evaluation for {model_name} on {task_name}: {e}\n{traceback.format_exc()}"
            logger.error(error_msg)
            return {'error': str(e), 'run_id': run_id}

    def _get_random_samples(self, dataset, num_samples: int = None, seed: int = 42) -> List[int]:
        """
        Helper method for proper randomization of sample selection.
        If num_samples is None, returns all indices in random order.
        """
        # Ensure numpy random state is seeded
        rs = np.random.RandomState(seed)
        num_available = len(dataset)
        
        if num_samples is None or num_samples < 0:
            actual_num = num_available
        else:
            actual_num = min(num_samples, num_available)
            
        indices = rs.choice(num_available, actual_num, replace=False)
        logger.info(f"Randomization debug - seed={seed}, dataset_size={num_available}, selected={actual_num}")
        return indices.tolist()
    
    def run_glue_evaluation(self) -> Dict[str, Any]:
        """
        Run GLUE evaluation across all models and tasks with statistical analysis.
        
        Returns:
            Dictionary containing all evaluation results with statistical measures
        """
        logger.info("Starting GLUE evaluation with statistical analysis...")
        
        # Initialize results structure
        results = {
            'timestamp': self.timestamp,
            'num_runs': self.num_runs,
            'random_seed': self.random_seed,
            # PRESERVE EXISTING: Use models loaded from resume file
            'models': self.results.get('models', {}),
            'summary': {},
            'statistical_analysis': {}
        }
        
        # Evaluate each model sequentially to save memory
        for config in self.model_configs:
            model_name = config['name']
            
            # Load single model
            if not self._load_single_model(config):
                logger.error(f"Skipping evaluation for {model_name} due to load failure")
                continue
                
            logger.info(f"Evaluating {model_name}...")
            
            model_results = {
                'model_name': model_name,
                'glue_results': {},
                'overall_score': 0.0,
                'successful_evaluations': 0,
                'total_evaluations': 0,
                'run_scores': []  # Store scores from each run for statistical analysis
            }
            
            # RESUME LOGIC: Check if model already partially evaluated
            start_run_id = 0
            if 'models' in self.results and model_name in self.results['models']:
                existing_data = self.results['models'][model_name]
                # ALWAYS load existing data to enable granular task skipping
                model_results = existing_data
                
                # Check how many runs completed
                completed_runs = len(existing_data.get('run_scores', []))
                if completed_runs > 0:
                    logger.info(f"⏩ Resuming {model_name} from run {completed_runs + 1}/{self.num_runs}")
                    start_run_id = completed_runs
            
            # Run multiple times for statistical analysis
            for run_id in range(self.num_runs):
                if run_id < start_run_id:
                    continue # Skip completed runs
                    
                logger.info(f"  Run {run_id + 1}/{self.num_runs}")
                
                # Enhanced random seed management for each run (fixes randomization bug)
                run_seed = self.random_seed + run_id
                np.random.seed(run_seed)
                torch.manual_seed(run_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(run_seed)
                
                # Run GLUE tasks
                run_scores = []
                run_successful = 0
                run_total = 0
                
                logger.info(f"    Starting run {run_id + 1}/{self.num_runs} for {model_name}")
                
                for task_name in self.glue_suite.keys():
                    # GRANULAR RESUME: Check if individual task already done
                    result_key = f"{task_name}_run_{run_id}"
                    if result_key in model_results['glue_results'] and 'error' not in model_results['glue_results'][result_key]:
                        logger.info(f"    Skipping {task_name} (already completed in previous session)")
                        continue

                    logger.info(f"    Running {task_name}...")
                    run_total += 1
                    
                    try:
                        result = self.evaluate_glue_task(model_name, task_name, run_id)
                        
                        if 'error' not in result:
                            model_results['glue_results'][f"{task_name}_run_{run_id}"] = result
                            run_successful += 1
                            run_scores.append(result['primary_score'])
                        else:
                            model_results['glue_results'][f"{task_name}_run_{run_id}"] = result
                            logger.warning(f"    Task {task_name} failed in run {run_id + 1}")
                    except Exception as e:
                        logger.error(f"    Task {task_name} crashed in run {run_id + 1}: {e}")
                        # Add error result to maintain consistency
                        model_results['glue_results'][f"{task_name}_run_{run_id}"] = {'error': str(e), 'run_id': run_id}

                    # INTERMEDIATE SAVE: Save after every TASK
                    results['models'][model_name] = model_results
                    self.results = results
                    self.save_results()
            
            # Calculate run score
            run_overall_score = sum(run_scores) / len(run_scores) if run_scores else 0.0
            model_results['run_scores'].append(run_overall_score)
            model_results['successful_evaluations'] += run_successful
            model_results['total_evaluations'] += run_total
            model_results['overall_score'] += run_overall_score
            
            # INTERMEDIATE SAVE: Save after every run
            results['models'][model_name] = model_results
            self.results = results
            self.save_results()
            
            # Calculate final score with statistical analysis
            if model_results['successful_evaluations'] > 0:
                model_results['overall_score'] /= self.num_runs
                
                # Add statistical analysis
                model_results['statistics'] = self.compute_statistics(model_results['run_scores'])
                
                # Calculate per-task statistics
                task_scores = {}
                for task_name in self.glue_suite.keys():
                    task_run_scores = []
                    for run_id in range(self.num_runs):
                        result_key = f"{task_name}_run_{run_id}"
                        if result_key in model_results['glue_results'] and 'error' not in model_results['glue_results'][result_key]:
                            task_run_scores.append(model_results['glue_results'][result_key]['primary_score'])
                    
                    if task_run_scores:
                        task_scores[task_name] = {
                            'statistics': self.compute_statistics(task_run_scores),
                            'num_successful_runs': len(task_run_scores),
                            'total_runs': self.num_runs
                        }
                
                model_results['task_statistics'] = task_scores
            
            results['models'][model_name] = model_results
            
            # Save results after each model completes
            logger.info(f"Saving intermediate results after evaluating {model_name}...")
            self.results = results
            self.save_results()
            
            # Unload model to free memory
            self._unload_model(model_name)
        
        # Create summary
        results['summary'] = self.create_summary(results)
        
        # Add overall statistical analysis
        results['statistical_analysis'] = self.compute_overall_statistics(results)
        
        # Store results
        self.results = results
        
        logger.info("GLUE evaluation with statistical analysis completed!")
        return results
    
    def create_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a summary of evaluation results with statistical analysis.
        
        Args:
            results: Full evaluation results
            
        Returns:
            Summary dictionary
        """
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
            
            if score > 0:
                summary['successful_models'] += 1
            else:
                summary['failed_models'] += 1
        
        # Sort by score
        model_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Create rankings
        for i, (model_name, score, success_rate, statistics) in enumerate(model_scores):
            summary['rankings'].append({
                'rank': i + 1,
                'model_name': model_name,
                'score': score,
                'success_rate': success_rate,
                'statistics': statistics
            })
            
            if i == 0:
                summary['best_model'] = model_name
                summary['best_score'] = score
        
        # Add overall statistical insights
        if 'statistical_analysis' in results:
            summary['statistical_insights'] = {
                'significant_differences': len([k for k, v in results['statistical_analysis'].get('statistical_significance', {}).items() if v.get('significant', False)]),
                'total_comparisons': len(results['statistical_analysis'].get('statistical_significance', {})),
                'best_improvement': max([v.get('improvement_percent', 0) for v in results['statistical_analysis'].get('performance_gaps', {}).values()], default=0)
            }
        
        return summary
    
    def save_results(self, output_path: str = None) -> str:
        """
        Save evaluation results to JSON file.
        
        Args:
            output_path: Path to save results (optional)
            
        Returns:
            Path where results were saved
        """
        if output_path is None:
            output_path = f"results/glue_evaluation_results_{self.timestamp}.json"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save results
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to: {output_path}")
        return output_path
    
    def print_summary(self):
        """Print a human-readable summary of results with statistical analysis."""
        if not self.results:
            print("No results available. Run evaluation first.")
            return
        
        print("\n" + "="*80)
        print("GLUE EVALUATION SUMMARY WITH STATISTICAL ANALYSIS")
        print("="*80)
        
        print(f"Timestamp: {self.results['timestamp']}")
        print(f"Number of runs: {self.results.get('num_runs', 5)}")
        print(f"Random seed: {self.results.get('random_seed', 42)}")
        print(f"Total models evaluated: {self.results['summary']['total_models']}")
        print(f"Successful evaluations: {self.results['summary']['successful_models']}")
        print(f"Failed evaluations: {self.results['summary']['failed_models']}")
        
        if self.results['summary']['best_model']:
            print(f"\n🏆 Best performing model: {self.results['summary']['best_model']}")
            print(f"📊 Best score: {self.results['summary']['best_score']:.4f}")
        
        # Statistical insights
        if 'statistical_insights' in self.results['summary']:
            insights = self.results['summary']['statistical_insights']
            print(f"\n📊 STATISTICAL INSIGHTS:")
            print(f"   Significant differences: {insights.get('significant_differences', 0)}/{insights.get('total_comparisons', 1)}")
            print(f"   Best improvement: {insights.get('best_improvement', 0):.2f}%")
        
        print("\n📈 DETAILED RESULTS:")
        print("-" * 60)
        
        for model_name, model_results in self.results['models'].items():
            print(f"\n🤖 {model_name}:")
            print(f"   Overall Score: {model_results.get('overall_score', 0.0):.4f}")
            print(f"   Success Rate: {model_results.get('successful_evaluations', 0)}/{model_results.get('total_evaluations', 0)}")
            
            # Statistical information
            if 'statistics' in model_results:
                stats = model_results['statistics']
                print(f"   📊 Statistics:")
                print(f"     Mean: {stats.get('mean', 0.0):.4f} ± {stats.get('std', 0.0):.4f}")
                print(f"     95% CI: [{stats.get('mean', 0.0) - stats.get('ci_95', 0.0):.4f}, {stats.get('mean', 0.0) + stats.get('ci_95', 0.0):.4f}]")
                print(f"     Min: {stats.get('min', 0.0):.4f}, Max: {stats.get('max', 0.0):.4f}")
            
            # Show per-task statistics
            if 'task_statistics' in model_results:
                print(f"   📊 Task Performance:")
                for task_name, task_stats in model_results['task_statistics'].items():
                    task_mean = task_stats['statistics']['mean']
                    task_std = task_stats['statistics']['std']
                    success_rate = task_stats['num_successful_runs'] / task_stats['total_runs']
                    print(f"     • {task_name}: {task_mean:.4f} ± {task_std:.4f} ({success_rate*100:.1f}% success)")
        
        # Statistical significance results
        if 'statistical_analysis' in self.results and 'statistical_significance' in self.results['statistical_analysis']:
            print(f"\n🔬 STATISTICAL SIGNIFICANCE:")
            print("-" * 40)
            for comparison, sig_result in self.results['statistical_analysis']['statistical_significance'].items():
                status = "✅" if sig_result.get('significant', False) else "❌"
                print(f"   {comparison}: {status} (p={sig_result.get('p_value', 0.0):.4f})")
        
        print("\n" + "="*80)


def main():
    """Main function to run the GLUE evaluation pipeline."""
    print("🚀 SIMPLE GLUE EVALUATION PIPELINE")
    print("=" * 60)
    print("Focused evaluation on GLUE benchmark tasks")
    print("=" * 60)
    
    # Model configurations
    model_configs = [
        {
            'name': 'gpt2-large',
            'path': 'gpt2-large'
        }
    ]
    
    # Add neuroplastic transformer if available
    neuroplastic_paths = [
        r'c:\Users\andre\Desktop\Overføre\next_step_ai\models\neuroplastic_adapter_agi\trained_phase3', # Prioritize trained model
        'models/neuroplastic_adapter_agi/trained_phase3',
        'models/neuroplastic_adapter_agi/baseline_run/final_model',
        'models/neuroplastic_adapter_agi/optimized_run/final_model',
        'models/neuroplastic_adapter_agi/final_model'
    ]
    
    for neuroplastic_path in neuroplastic_paths:
        if os.path.exists(neuroplastic_path):
            model_configs.append({
                'name': 'neuroplastic-transformer',
                'path': neuroplastic_path
            })
            break
    
    print(f"Configured models: {[config['name'] for config in model_configs]}")
    
    # Initialize and run evaluation
    pipeline = SimpleGLUEEvaluationPipeline(model_configs, num_runs=5)
    
    try:
        # Run GLUE evaluation
        results = pipeline.run_glue_evaluation()
        
        # Print summary
        pipeline.print_summary()
        
        # Save results
        results_file = pipeline.save_results()
        print(f"\n📁 Full results saved to: {results_file}")
        
        print("\n✅ GLUE evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        print(f"❌ Evaluation failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
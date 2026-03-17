
import os
import sys
import logging
from datetime import datetime

# Add the llama3_neuroplastic directory to the path to import the local pipeline
sys.path.append(os.path.join(os.getcwd(), 'llama3_neuroplastic'))
from llama_evaluation_pipeline import SimpleGLUEEvaluationPipeline, EvaluationConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_gains_verification():
    print("Running Fast Verification of Neuroplastic Gains (Low Noise)")
    print("-" * 60)
    
    config = EvaluationConfig()
    config.batch_size = 4
    config.max_samples = 100 # Fast statistically significant check
    config.num_runs = 1
    # Test MRPC (Previous Big Gain) and SST-2 (Baseline Baseline) and STS-B (Semantic)
    config.glue_tasks = ['mrpc', 'sst2', 'stsb'] 
    config.output_dir = "results_gains_check"
    
    # Initialize pipeline
    # This will use the GLOBAL default noise (0.0005) set in multi_task_error_integration.py
    pipeline = SimpleGLUEEvaluationPipeline(config)
    
    results = pipeline.run_glue_evaluation()
    
    print("\nVerification Complete!")
    print("-" * 60)
    
    # Compare Results
    models = results.get('models', {})
    if len(models) >= 2:
        model_names = list(models.keys()) # Expected: Pure, Neuroplastic
        print(f"{'Task':<10} | {model_names[0]:<25} | {model_names[1]:<25} | {'Delta':<10}")
        print("-" * 75)
        
        for task in config.glue_tasks:
            s1 = models[model_names[0]]['glue_results'].get(f"{task}_run_0", {}).get('primary_score', 0.0)
            s2 = models[model_names[1]]['glue_results'].get(f"{task}_run_0", {}).get('primary_score', 0.0)
            delta = ((s2 - s1) / (s1 + 1e-9)) * 100
            print(f"{task:<10} | {s1:<25.4f} | {s2:<25.4f} | {delta:>+9.1f}%")

if __name__ == "__main__":
    run_gains_verification()

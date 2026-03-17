
import os
import sys
import logging
import json
import argparse
from datetime import datetime

# Add the llama3_neuroplastic directory to the path to import the local pipeline
sys.path.append(os.path.join(os.getcwd(), 'llama3_neuroplastic'))
from llama_evaluation_pipeline import SimpleGLUEEvaluationPipeline, EvaluationConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_cola_check(checkpoint_dir=None):
    print("Running Quick CoLA Recovery Check (Neuroplastic Model)")
    print("-" * 60)
    
    config = EvaluationConfig()
    config.batch_size = 4
    config.max_samples = 50 # Fast check
    config.num_runs = 1 
    config.glue_tasks = ['cola'] # ONLY CoLA
    config.output_dir = "results_cola_check"
    config.checkpoint_dir = checkpoint_dir
    
    # Initialize pipeline
    # NOTE: The pipeline will now use the updated default noise of 0.0005
    # or roughly that if it reloads the module.
    pipeline = SimpleGLUEEvaluationPipeline(config)
    
    # Run evaluation
    results = pipeline.run_glue_evaluation()
    
    print("\nCoLA Check Complete!")
    
    # Extract Score
    models = results.get('models', {})
    for model_name, data in models.items():
        score = data['glue_results'].get('cola_run_0', {}).get('primary_score', 0.0)
        print(f"Model: {model_name} | CoLA Score (N=50): {score}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run quick CoLA check for NeuroplasticLlama.")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Optional SCA-v2 model directory to evaluate.",
    )
    args = parser.parse_args()
    run_cola_check(checkpoint_dir=args.checkpoint_dir)

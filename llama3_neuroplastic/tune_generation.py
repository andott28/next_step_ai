
import torch
from transformers import AutoTokenizer
from neuroplastic_llama import NeuroplasticLlama
import os

MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"

def tune_generation():
    print(f"Loading {MODEL_NAME}...")
    
    # Enable Neuroplasticity
    neuro_model = NeuroplasticLlama(
        model_name=MODEL_NAME,
        num_tasks=8, 
        adapter_bottleneck=256,
        neuroplasticity_enabled=True,
        kv_int4_quantization=True,
        kv_cpu_offload=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    prompt_text = "Explain the concept of quantum entanglement in a poetic, metaphorical way."
    formatted_prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{prompt_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    inputs = tokenizer([formatted_prompt], return_tensors = "pt").to("cuda")
    
    generation_kwargs = {
        "max_new_tokens": 100, 
        "use_cache": True,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "pad_token_id": tokenizer.eos_token_id
    }
    
    # SWEEP CONFIG
    noise_levels = [0.0, 0.0001, 0.0005, 0.001, 0.005]
    
    output_file = os.path.join("llama3_neuroplastic", "tuning_results.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("NEUROPLASTIC LLAMA-3 TUNING RESULTS\n")
        f.write("=====================================\n\n")
    
    for noise in noise_levels:
        print(f"\n[Testing Noise Level: {noise}]")
        
        # Set noise manually for task 0
        if hasattr(neuro_model, 'task_coordinator'):
            neuro_model.task_coordinator.task_integrator.task_biological_params['task_0'] = {
                'noise_intensity': noise,
                'firing_dynamics_enabled': True
            }
        
        try:
            outputs = neuro_model.generate(**inputs, **generation_kwargs)
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Print preview
            preview = text.replace("\n", " ")[:100] + "..."
            print(f"Result: {preview}")
            
            # Save
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(f"--- NOISE LEVEL: {noise} ---\n")
                f.write(text + "\n\n")
                
        except Exception as e:
            print(f"Error at noise {noise}: {e}")

    print(f"\nTuning results written to {output_file}")

if __name__ == "__main__":
    tune_generation()

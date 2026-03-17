
import torch
from transformers import AutoTokenizer
from neuroplastic_llama import NeuroplasticLlama
import os

# Configuration matching the working pipeline
MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"

def generate_sample():
    print(f"Loading {MODEL_NAME}...")

    try:
        neuro_model = NeuroplasticLlama(
            model_name=MODEL_NAME,
            num_tasks=8, 
            adapter_bottleneck=256,
            neuroplasticity_enabled=False,
            kv_int4_quantization=True,
            kv_cpu_offload=True,
        )
    except Exception as e:
        print(f"FAILED to load model: {e}")
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    prompt_text = "Explain the concept of quantum entanglement in a poetic, metaphorical way."
    
    formatted_prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{prompt_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    inputs = tokenizer([formatted_prompt], return_tensors = "pt").to("cuda")
    
    generation_kwargs = {
        "max_new_tokens": 150, 
        "use_cache": True,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "pad_token_id": tokenizer.eos_token_id
    }
    
    print("\nGenerating (Baseline)...")
    try:
        outputs = neuro_model.generate(**inputs, **generation_kwargs)
        
        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        output_file = os.path.join("llama3_neuroplastic", "final_generation.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(generated_text)
            
        print(f"\nWritten result to {output_file}")
        print("\n" + "="*80)
        print(generated_text)
        print("="*80)
        
    except Exception as e:
        print(f"\nGeneration FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    generate_sample()


import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, GPT2LMHeadModel
from neuroplastic_transformer import NeuroplasticGPT2
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def get_embeddings_batched(model, tokenizer, texts, device, batch_size=16): # Small batch size for GPT-2 Large
    embeddings = []
    
    # Process in batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Computing Embeddings", leave=False):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize batch
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
        
        with torch.no_grad():
            # Call the model wrapper directly to ensure Neuroplastic components (adapters/noise) are active
            # output_hidden_states=True ensures we get the tuple containing the final modified state
            outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], output_hidden_states=True)
            
            # The NeuroplasticGPT2 now appends the modified state as the LAST element of hidden_states
            # For GPT-2 Baseline, it's also the last element
            hidden = outputs.hidden_states[-1]
            
            # Mask padding
            mask = inputs['attention_mask'].unsqueeze(-1).expand(hidden.size()).float()
            sum_embeddings = torch.sum(hidden * mask, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            pooled = sum_embeddings / sum_mask
            
            embeddings.append(pooled.cpu().numpy())
            
        # GPU Cleanup
        del inputs, outputs, hidden, mask
    
    return np.vstack(embeddings)

def main():
    print("🧠 Visualizing Semantic Attractors (Full QQP Paraphrases)")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained('gpt2-large')
    tokenizer.pad_token = tokenizer.eos_token
    
    # 1. Load Models
    print("\n--- Loading Models ---")
    print("Loading Baseline GPT-2 Large...")
    baseline_model = GPT2LMHeadModel.from_pretrained('gpt2-large').to(device)
    baseline_model.eval()

    print("Loading Neuroplastic Transformer...")
    neuro_path = r'c:\Users\andre\Desktop\Overføre\next_step_ai\models\neuroplastic_adapter_agi\trained_phase3'
    if not os.path.exists(neuro_path):
        print(f"Error: Neuroplastic model not found at {neuro_path}")
        return
    neuro_model = NeuroplasticGPT2.from_pretrained(neuro_path).to(device)
    neuro_model.eval()

    # 2. Get QQP Paraphrases (FULL SET)
    print("\n--- Loading QQP Data ---")
    dataset = load_dataset("glue", "qqp", split='validation')
    
    # Filter for ACTUAL paraphrases (label=1)
    paraphrase_data = [d for d in dataset if d['label'] == 1]
    # User requested full validation set for scientific validity
    print(f"Processing FULL set of {len(paraphrase_data)} paraphrase pairs.")
    
    # 3. Compute Similarities
    q1_texts = [d['question1'] for d in paraphrase_data]
    q2_texts = [d['question2'] for d in paraphrase_data]

    print("\n--- Processing Baseline (GPT-2) ---")
    b_q1 = get_embeddings_batched(baseline_model, tokenizer, q1_texts, device)
    b_q2 = get_embeddings_batched(baseline_model, tokenizer, q2_texts, device)
    
    print("\n--- Processing Neuroplastic ---")
    n_q1 = get_embeddings_batched(neuro_model, tokenizer, q1_texts, device)
    n_q2 = get_embeddings_batched(neuro_model, tokenizer, q2_texts, device)

    # Calculate Cosine Similarities
    def cos_sim_batch(a, b):
        # Optimized batch cosine similarity
        norm_a = np.linalg.norm(a, axis=1)
        norm_b = np.linalg.norm(b, axis=1)
        dot = np.sum(a * b, axis=1)
        return dot / (norm_a * norm_b)

    baseline_sims = cos_sim_batch(b_q1, b_q2)
    neuro_sims = cos_sim_batch(n_q1, n_q2)

    # 4. Generate Visualization
    print("\n--- Generating Visualization ---")
    plt.figure(figsize=(10, 6))
    
    # Plotting boxplot comparison
    plt.subplot(1, 2, 1)
    plt.boxplot([baseline_sims, neuro_sims], labels=['GPT-2 Baseline', 'Neuroplastic'])
    plt.title(f"Cos-Sim Distribution (N={len(baseline_sims)})")
    plt.ylabel("Cosine Similarity")
    plt.grid(True, alpha=0.3)
    
    # Plotting Histogram instead of scatter (Scatter is too messy for 15k points)
    plt.subplot(1, 2, 2)
    plt.hist(baseline_sims, bins=50, alpha=0.5, label='Baseline', color='gray', density=True)
    plt.hist(neuro_sims, bins=50, alpha=0.5, label='Neuroplastic', color='blue', density=True)
    plt.title("Similarity Density Distribution")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Density")
    plt.legend()
    
    plt.tight_layout()
    output_path = 'semantic_attractor_comparison.png'
    plt.savefig(output_path, dpi=300)
    print(f"✅ Visualization saved to {output_path}")

    print("\n" + "="*50)
    print(" SEMANTIC ATTRACTOR REPORT (FULL VALIDATION SET)")
    print("="*50)
    print(f"Pairs Processed:                {len(baseline_sims)}")
    print(f"Average Baseline Similarity:    {np.mean(baseline_sims):.4f}")
    print(f"Average Neuroplastic Similarity: {np.mean(neuro_sims):.4f}")
    delta = np.mean(neuro_sims) - np.mean(baseline_sims)
    print(f"Net Semantic Delta:             {delta:+.4f}")
    print("="*50)

if __name__ == "__main__":
    main()

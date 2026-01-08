import os
# Set HuggingFace cache directory to cephfs when on HPC to avoid quota issues
# MUST be done before importing transformers
if os.path.exists('/cephfs/store/gr-mc2473/eszt2'):
    os.environ['HF_HOME'] = '/cephfs/store/gr-mc2473/eszt2/hf_cache'
    os.environ['TRANSFORMERS_CACHE'] = '/cephfs/store/gr-mc2473/eszt2/hf_cache'
    os.environ['HF_DATASETS_CACHE'] = '/cephfs/store/gr-mc2473/eszt2/hf_cache/datasets'
    print("Running on HPC - using cephfs cache directory")

import torch
import json
import math
import time
import random
import argparse

from load_qwen import load_qwen_model
from datasets import load_dataset

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run Experiment 5 Ablation: Identity function intervention on MLP layers')
parser.add_argument('--max-chars', type=int, default=None,
                    help='Optional: Maximum characters to use from dataset (default: use full dataset)')
args = parser.parse_args()

# Start timing
start_time = time.time()

# Determine device - use CUDA on HPC, MPS on Mac
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# Load the model
model, tokenizer = load_qwen_model(device=device)

# Load the dataset
wikitext = load_dataset("Salesforce/wikitext", "wikitext-2-v1")
train_data = wikitext["train"]["text"]
train_text = "\n".join([t for t in train_data if t.strip()])

val_data = wikitext["validation"]["text"]
val_text = "\n".join([t for t in val_data if t.strip()])

# OPTIONAL: Truncate data for faster testing (set to None for full experiment)
MAX_CHARS = args.max_chars

if MAX_CHARS is not None:
    # Use random sampling for better representation
    random.seed(290402)  # For reproducibility

    def sample_text_random(text, max_chars):
        if len(text) <= max_chars:
            return text
        # Random starting position
        max_start = len(text) - max_chars
        start_pos = random.randint(0, max_start)
        return text[start_pos:start_pos + max_chars]

    train_text = sample_text_random(train_text, MAX_CHARS)
    val_text = sample_text_random(val_text, MAX_CHARS)

# Which MLP layers to replace with identity
layers = [7, 14, 21]

# SETUP THE EXPERIMENT
base_experiment_folder = "experiment5_ablation"

layers_str = "_".join(str(x) for x in layers)

if MAX_CHARS:
    results_location = f"{base_experiment_folder}/layers_{layers_str}/max_chars{MAX_CHARS}"
else:
    results_location = f"{base_experiment_folder}/layers_{layers_str}"

os.makedirs(base_experiment_folder, exist_ok=True)
os.makedirs(results_location, exist_ok=True)

# Create results dictionary
experimental_results = {"layers": layers}


def get_perplexity(model, tokenizer, text, experimental_results, max_length=1024, dataset_name="train"):
    # Tokenize the full training text
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings['input_ids'][0]

    experimental_results["max_length"] = max_length
    experimental_results[f"num_{dataset_name}_tokens"] = len(input_ids)

    num_chunks = (len(input_ids) + max_length - 1) // max_length

    print(f"Processing {len(input_ids)} tokens in {num_chunks} chunks of {max_length} tokens each")

    # Track loss for perplexity calculation
    total_loss = 0.0
    total_tokens = 0

    for i in range(0, len(input_ids), max_length):
        chunk = input_ids[i:i+max_length].unsqueeze(0).to(model.device)

        with torch.no_grad():
            outputs = model(input_ids=chunk, labels=chunk, use_cache=False)

            # Accumulate loss
            total_loss += outputs.loss.item() * chunk.size(1)
            total_tokens += chunk.size(1)

        if (i // max_length + 1) % 10 == 0:
            print(f"Processed chunk {i // max_length + 1}/{num_chunks}")

    # Calculate perplexity
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    print(f"Perplexity on {dataset_name} set: {perplexity:.4f}")

    return perplexity


# Run baseline (no intervention)
print("Running baseline (no intervention)")
experimental_results["baseline_perplexity_train"] = get_perplexity(model, tokenizer, train_text, experimental_results)
experimental_results["baseline_perplexity_val"] = get_perplexity(model, tokenizer, val_text, experimental_results, dataset_name="val")


# Create identity intervention hook
def make_mlp_identity_intervention_hook(layer_num):
    """
    Create a hook function that replaces MLP output with identity (returns input as output)
    """
    def mlp_identity_intervention_hook(module, input, output):
        # Simply return the input as the output (identity function)
        return input[0]

    return mlp_identity_intervention_hook


# Register intervention hooks for all layers
intervention_handles = []
for layer in layers:
    handle = model.model.layers[layer].mlp.register_forward_hook(make_mlp_identity_intervention_hook(layer))
    intervention_handles.append(handle)

print("Running forward pass with identity intervention")

experimental_results["intervened_perplexity_train"] = get_perplexity(model, tokenizer, train_text, experimental_results)
experimental_results["intervened_perplexity_val"] = get_perplexity(model, tokenizer, val_text, experimental_results, dataset_name="val")

# Remove all intervention hooks when done
for handle in intervention_handles:
    handle.remove()

# Add runtime to results
experimental_results["total_runtime_minutes"] = (time.time() - start_time)/60

# Save experimental results
results_file = f'{results_location}/experimental_results.json'
with open(results_file, 'w') as f:
    json.dump(experimental_results, f, indent=2)

print(f"\n{'='*60}")
print(f"EXPERIMENT 5 ABLATION COMPLETE")
print(f"{'='*60}")
print(f"\nExperimental Setup:")
print(f"  Layers intervened: {layers}")
print(f"  Intervention: Identity function (MLP output = MLP input)")
print(f"\nBaseline Perplexity:")
print(f"  Train: {experimental_results['baseline_perplexity_train']:.4f}")
print(f"  Val:   {experimental_results['baseline_perplexity_val']:.4f}")
print(f"\nIntervened Perplexity:")
print(f"  Train: {experimental_results['intervened_perplexity_train']:.4f}")
print(f"  Val:   {experimental_results['intervened_perplexity_val']:.4f}")
print(f"\nPerplexity Change:")
print(f"  Train: {experimental_results['intervened_perplexity_train'] - experimental_results['baseline_perplexity_train']:.4f}")
print(f"  Val:   {experimental_results['intervened_perplexity_val'] - experimental_results['baseline_perplexity_val']:.4f}")
print(f"\nResults saved to: {results_file}")

# Calculate and display total runtime
end_time = time.time()
total_time = end_time - start_time
hours = int(total_time // 3600)
minutes = int((total_time % 3600) // 60)
seconds = total_time % 60

print(f"\nTotal Runtime: {hours}h {minutes}m {seconds:.2f}s")
print(f"{'='*60}")

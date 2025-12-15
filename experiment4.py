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
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA
import numpy as np
import pickle
import math
import time
import random
import argparse

from load_qwen import load_qwen_model
from datasets import load_dataset

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run Experiment 4: PCA intervention on MLP layers')
parser.add_argument('pca_comps_I', type=int,
                    help='Number of PCA components for input')
parser.add_argument('pca_comps_O', type=int,
                    help='Number of PCA components for output')
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

#load the model
model, tokenizer = load_qwen_model(device=device)

#load the dataset
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

#which mlp layers to mess with
layers = [7, 14, 21]

#how many pca comps to use for I/O
pca_comps_I = args.pca_comps_I
pca_comps_O = args.pca_comps_O

#SETUP THE EXPERIMENT
base_experiment_folder = "experiment4_testing"

layers_str = "_".join(str(x) for x in layers)

if MAX_CHARS:
    results_location = f"{base_experiment_folder}/layers_{layers_str}/max_chars{MAX_CHARS}/pca_comps_I{pca_comps_I}_O{pca_comps_O}"

else:
    results_location = f"{base_experiment_folder}/layers_{layers_str}/pca_comps_I{pca_comps_I}_O{pca_comps_O}"


os.makedirs(base_experiment_folder, exist_ok=True)
os.makedirs(results_location, exist_ok=True)


#create storage for MLP I/O
mlp_activations = {}
handles = []

#create results dictionary 
experimental_results = {"layers": layers,
                        "pca_comps_I": pca_comps_I,
                        "pca_comps_O": pca_comps_O
                        }


def make_mlp_hook(layer_num):
    """Create a hook function for a specific layer"""
    def mlp_hook(module, input, output):
        mlp_activations[f"layer{layer_num}"]['inputs'].append(input[0].detach().cpu())
        mlp_activations[f"layer{layer_num}"]['outputs'].append(output.detach().cpu())
    return mlp_hook

# Only run below code if we don't have the information already
# Store baseline data in the base experiment folder (shared across all setups)
if MAX_CHARS:
    path = f"{base_experiment_folder}/layers_{layers_str}/max_chars{MAX_CHARS}/mlp_activations.pt"
else:
    path = f"{base_experiment_folder}/layers_{layers_str}/mlp_activations.pt"


def get_perplexity(model, tokenizer, text, experimental_results, max_length = 1024, dataset_name = "train"):
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
            outputs = model(input_ids=chunk, labels=chunk, use_cache = False)

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

if os.path.exists(path):
    print("Not running baseline - we already have the activations saved.")
else:
    print("Running baseline and collecting activations")

    for layer in layers:
        # Create storage
        mlp_activations[f"layer{layer}"] = {'inputs': [], 'outputs': []}
        
        #register hook
        handle = model.model.layers[layer].mlp.register_forward_hook(make_mlp_hook(layer))
        handles.append(handle)

    print("Running model no intervention")

    experimental_results["baseline_perplexity_train"]=get_perplexity(model, tokenizer, train_text, experimental_results)

    # Remove the hook when done
    for handle in handles:
        handle.remove()

    experimental_results["baseline_perplexity_val"]=get_perplexity(model, tokenizer, val_text, experimental_results, dataset_name="val")

    # Save activations for each layer separately
    activations_to_save = {}
    for layer in layers:
        # Concatenate along the token dimension (assuming batch_size=1, we concatenate sequence lengths)
        inputs_tensor = torch.cat([x.squeeze(0) for x in mlp_activations[f'layer{layer}']['inputs']], dim=0)
        outputs_tensor = torch.cat([x.squeeze(0) for x in mlp_activations[f'layer{layer}']['outputs']], dim=0)

        activations_to_save[f'layer{layer}'] = {
            'inputs': inputs_tensor,
            'outputs': outputs_tensor
        }
        print(f"Layer {layer} - Inputs: {inputs_tensor.shape}, Outputs: {outputs_tensor.shape}")

    torch.save(activations_to_save, f'{base_experiment_folder}/layers_{layers_str}/mlp_activations.pt')

#Fit the PCA model if we haven't done it already

# Load the saved activations from base experiment folder
data = torch.load(f"{base_experiment_folder}/layers_{layers_str}/mlp_activations.pt")

# Dictionary to store PCA models for each layer
pca_models = {}

def train_pca(X, pca_comps):

    pca = PCA(n_components = pca_comps, whiten = False, random_state= 290402)
    pca.fit(X)

    return pca, pca.explained_variance_ratio_.sum()

for layer in layers:
    # Initialize storage for this layer's PCA models
    pca_models[layer] = {}
    experimental_results[f"pca_layer{layer}"] = {}

    # Train/load PCA for OUTPUTS
    pca_output_path = f'{results_location}/pca_output_layer{layer}.pkl'
    if os.path.exists(pca_output_path):
        print(f"Loading existing OUTPUT PCA model for layer {layer} from {pca_output_path}")
        with open(pca_output_path, 'rb') as f:
            pca_models[layer]['output'] = pickle.load(f)
        print(f"  Output PCA loaded. Explained variance ratio sum: {pca_models[layer]['output'].explained_variance_ratio_.sum():.4f}")
        experimental_results[f"pca_layer{layer}"]["output_explained_var_ratio"] = float(pca_models[layer]['output'].explained_variance_ratio_.sum())
    else:
        print(f"Training OUTPUT PCA for layer {layer}")
        X_output = data[f'layer{layer}']['outputs'].numpy()
        print(f"  Output shape: {X_output.shape}")

        pca_output, explained_var_ratio = train_pca(X_output, pca_comps_O)

        with open(pca_output_path, 'wb') as f:
            pickle.dump(pca_output, f)
        print(f"  Output PCA saved. Explained variance: {explained_var_ratio:.4f}")

        pca_models[layer]['output'] = pca_output
        experimental_results[f"pca_layer{layer}"]["output_explained_var_ratio"] = float(explained_var_ratio)

    # Train/load PCA for INPUTS
    pca_input_path = f'{results_location}/pca_input_layer{layer}.pkl'
    if os.path.exists(pca_input_path):
        print(f"Loading existing INPUT PCA model for layer {layer} from {pca_input_path}")
        with open(pca_input_path, 'rb') as f:
            pca_models[layer]['input'] = pickle.load(f)
        print(f"  Input PCA loaded. Explained variance ratio sum: {pca_models[layer]['input'].explained_variance_ratio_.sum():.4f}")
        experimental_results[f"pca_layer{layer}"]["input_explained_var_ratio"] = float(pca_models[layer]['input'].explained_variance_ratio_.sum())
    else:
        print(f"Training INPUT PCA for layer {layer}")
        X_input = data[f'layer{layer}']['inputs'].numpy()
        print(f"  Input shape: {X_input.shape}")

        pca_input, explained_var_ratio = train_pca(X_input, pca_comps_I)

        with open(pca_input_path, 'wb') as f:
            pickle.dump(pca_input, f)
        print(f"  Input PCA saved. Explained variance: {explained_var_ratio:.4f}")

        pca_models[layer]['input'] = pca_input
        experimental_results[f"pca_layer{layer}"]["input_explained_var_ratio"] = float(explained_var_ratio)

# Create intervention hooks for multiple layers
def make_mlp_input_intervention_hook(layer_num):
    """
    Create a PRE-hook function for PCA intervention on MLP input
    """
    def mlp_input_intervention_hook(module, input):
        # Store original device and dtype for input
        original_input = input[0]  # Input is a tuple, get first element
        original_input_device = original_input.device
        original_input_dtype = original_input.dtype
        original_input_shape = original_input.shape

        # Transform input to numpy for PCA
        X_input = original_input.detach().cpu().numpy()

        # Reshape for PCA if needed (batch_size * seq_len, hidden_dim)
        X_input_reshaped = X_input.reshape(-1, X_input.shape[-1])

        # Apply PCA transform and inverse (reconstruction) on INPUT
        Z_input = pca_models[layer_num]['input'].transform(X_input_reshaped)
        X_input_hat = pca_models[layer_num]['input'].inverse_transform(Z_input)

        # Reshape back to original shape
        X_input_hat = X_input_hat.reshape(original_input_shape)

        # Convert back to torch tensor
        X_input_hat_tensor = torch.from_numpy(X_input_hat).to(device=original_input_device, dtype=original_input_dtype)

        # Return modified input as a tuple
        return (X_input_hat_tensor,)
    return mlp_input_intervention_hook

def make_mlp_output_intervention_hook(layer_num):
    """
    Create a hook function for PCA intervention on MLP output
    """
    def mlp_output_intervention_hook(module, input, output):
        # Store original device and dtype for output
        original_output_device = output.device
        original_output_dtype = output.dtype
        original_output_shape = output.shape

        # Transform output to numpy for PCA
        X_output = output.detach().cpu().numpy()

        # Reshape for PCA if needed (batch_size * seq_len, hidden_dim)
        X_output_reshaped = X_output.reshape(-1, X_output.shape[-1])

        # Apply PCA transform and inverse (reconstruction) on OUTPUT
        Z_output = pca_models[layer_num]['output'].transform(X_output_reshaped)
        X_output_hat = pca_models[layer_num]['output'].inverse_transform(Z_output)

        # Reshape back to original shape
        X_output_hat = X_output_hat.reshape(original_output_shape)

        # Convert back to torch tensor
        X_output_hat_tensor = torch.from_numpy(X_output_hat).to(device=original_output_device, dtype=original_output_dtype)

        return X_output_hat_tensor
    return mlp_output_intervention_hook

# Register intervention hooks for all layers
intervention_handles = []
for layer in layers:
    # Pre-hook for input intervention
    pre_handle = model.model.layers[layer].mlp.register_forward_pre_hook(make_mlp_input_intervention_hook(layer))
    intervention_handles.append(pre_handle)

    # Forward hook for output intervention
    post_handle = model.model.layers[layer].mlp.register_forward_hook(make_mlp_output_intervention_hook(layer))
    intervention_handles.append(post_handle)

print("Running forward pass with PCA intervention")

experimental_results["intervened_perplexity_train"]=get_perplexity(model, tokenizer, train_text, experimental_results)
experimental_results["intervened_perplexity_val"]=get_perplexity(model, tokenizer, val_text, experimental_results, dataset_name="val")

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
print(f"EXPERIMENT 4 COMPLETE")
print(f"{'='*60}")
print(f"\nExperimental Setup:")
print(f"  Layers intervened: {layers}")
print(f"  PCA components (input): {pca_comps_I}")
print(f"  PCA components (output): {pca_comps_O}")
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
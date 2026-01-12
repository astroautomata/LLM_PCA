"""
Script to load and interact with hf models
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set HuggingFace cache directory to cephfs when on HPC to avoid quota issues
# Check if we're on the HPC by looking for the cephfs directory
if os.path.exists('/cephfs/store/gr-mc2473/eszt2'):
    os.environ['HF_HOME'] = '/cephfs/store/gr-mc2473/eszt2/hf_cache'
    os.environ['TRANSFORMERS_CACHE'] = '/cephfs/store/gr-mc2473/eszt2/hf_cache'
    os.environ['HF_DATASETS_CACHE'] = '/cephfs/store/gr-mc2473/eszt2/hf_cache/datasets'
    print("Running on HPC - using cephfs cache directory")

def load_hf_model(device="mps", name = "Qwen/Qwen2.5-1.5B-Instruct"):
    """
    Load Qwen2.5-1.5B-Instruct model and tokenizer

    Args:
        device: Device to load model on ("mps" for Mac GPU, "cpu", or "cuda")

    Returns:
        model, tokenizer
    """
    model_name = name

    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Configure loading options for Mac
    model_kwargs = {
        "dtype": torch.float16 if device == "mps" else torch.float32,
    }

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )

    # Move to device
    model = model.to(device)

    print("Model loaded successfully!")
    print(f"Model device: {device}")
    print(f"Model dtype: {model.dtype}")

    return model, tokenizer
# LLM PCA + Symbolic Regression Experiments

This repository contains the code for the case study in Section 4.1 of the ICML paper **"SymTorch: A PyTorch Framework for Symbolic Distillation of Deep Neural Networks"**.

## Paper Relevance

**IMPORTANT:** The experiments relevant to the paper are:
- **`experiment4.py`** - PCA sensitivity analysis (Figure 4)
- **`experiment5.py`** - PCA + Symbolic Regression intervention (Table 1: PCA+SymTorch)
- **`experiment5_ablation.py`** - Control experiment with identity function (Table 1: Control)
- **`experiment7.py`** - Inference speed benchmarking across models (Figure 5, Table 5)

The other experiment files (`experiment.py`, `experiment2.py`, `experiment3.py`) are earlier exploratory work and are **not** used in the paper.

## SymTorch Version Note

The code in this repository uses an **earlier version of SymTorch** from commit [`13b9925`](https://github.com/elizabethsztan/SymTorch/commit/13b9925e464d529c6baddeb8cd06572e013707e3). This version predates the implementation of native PyTorch model serialization, so the saving and loading mechanisms for symbolic models are different from the current SymTorch release. Specifically:
- Symbolic models are saved using `save_model()` with `save_pytorch=False, save_regressors=True`
- Models are loaded using `SymbolicModel.load_model()` with the original callable function passed as `mlp_architecture`

---

## Experiment Descriptions

### Paper Experiments

#### `experiment4.py` - PCA Sensitivity Analysis
**Corresponds to:** Figure 4 in the paper

Performs PCA compression and reconstruction on MLP layer activations to measure the sensitivity of model perplexity to dimensionality reduction.

- **Model:** Qwen2.5-1.5B-Instruct
- **Layers intervened:** 7, 14, 21
- **Dataset:** WikiText-2-v1
- **Method:** Applies separate PCA models to MLP inputs (pre-hook) and outputs (post-hook)
- **Metrics:** Perplexity on train/validation sets, explained variance ratio

**Usage:**
```bash
python experiment4.py <pca_comps_input> <pca_comps_output> [--max-chars N]
# Example: python experiment4.py 32 8 --max-chars 750000
```

#### `experiment5.py` - PCA + Symbolic Regression
**Corresponds to:** Table 1 (PCA+SymTorch) in the paper

Replaces MLP layers with symbolic surrogates: MLP inputs are reduced via PCA, passed through a symbolic regression model, then reconstructed via inverse PCA.

- **Model:** Qwen2.5-1.5B-Instruct
- **Layers intervened:** 7, 14, 21
- **Dataset:** WikiText-2-v1
- **SR Training:** 6,000 samples per layer, 5,000 iterations
- **Metrics:** Baseline vs intervened perplexity

**Usage:**
```bash
python experiment5.py <pca_comps_input> <pca_comps_output> [--max-chars N]
# Example: python experiment5.py 32 8 --max-chars 750000
```

#### `experiment5_ablation.py` - Identity Function Control
**Corresponds to:** Table 1 (Control) in the paper

Ablation study that replaces MLP outputs with an identity function (MLP output = MLP input). This serves as a control to measure the importance of MLP computations.

- **Model:** Qwen2.5-1.5B-Instruct
- **Layers intervened:** 7, 14, 21
- **Dataset:** WikiText-2-v1
- **Metrics:** Baseline vs identity intervention perplexity

**Usage:**
```bash
python experiment5_ablation.py [--max-chars N]
```

#### `experiment7.py` - Inference Benchmarking
**Corresponds to:** Figure 5 and Table 5 in the paper

Benchmarks inference speed and perplexity across multiple LLM architectures, comparing:
1. Baseline (unmodified model)
2. Skip MLP (identity function)
3. Symbolic (PCA + symbolic regression)

- **Models tested:** Qwen2.5 (0.5B-7B), LLaMA-3.1/3.2 (1B-8B), SmolLM2/3, TinyLlama, and more
- **Devices:** CPU, CUDA, MPS
- **Metrics:** Latency (avg, p95), throughput (tokens/sec), perplexity

**Usage:**
```bash
python experiment7.py <results_dir> [--model_name MODEL] [--max-chars N] [--num-iterations N]
# Example: python experiment7.py experiment5/layers_7_14_21/max_chars750000/pca_comps_I32_O8
# For other models: python experiment7.py <results_dir> --model_name "meta-llama/Llama-3.2-1B-Instruct"
```

### Exploratory Experiments (Not in Paper)

#### `experiment.py` - Initial Proof of Concept
Early-stage single-layer PCA intervention on layer 19 only. Tests PCA reconstruction on MLP outputs with custom prompts.

#### `experiment2.py` - Multi-Layer Grid Search
Grid search over PCA components across 7 layers (3, 7, 11, 15, 19, 23, 27). Tests various configurations and measures perplexity on Harry Potter text.

#### `experiment3.py` - Early Symbolic Regression Test
Proof-of-concept combining PCA with symbolic regression on layer 3 only. Uses 32 PCA components for both inputs and outputs.

---

## Results Folders

### Paper Results

| Folder | Description |
|--------|-------------|
| `experiment4_testing/` | PCA sensitivity analysis results. Contains `experimental_results.json` and PCA models for each input/output component configuration |
| `experiment5/` | PCA + Symbolic regression results. Contains PCA models, symbolic models (`*_metadata.pkl`, `*_regressors*.pkl`), and `experimental_results.json` |
| `experiment5_ablation/` | Identity function ablation results. Contains `experimental_results.json` with baseline vs identity perplexity |
| `experiment7_results/` | Benchmarking results. Subfolders per model-device combination with `baseline_benchmark.json`, `skip_mlp_benchmark.json`, `symbolic_benchmark.json` |
| `experiment7_results_runpod/` | Benchmarking results from RunPod GPU infrastructure |

### Exploratory Results

| Folder | Description |
|--------|-------------|
| `experiment_results/` | Early single-layer PCA results with MLP activations |
| `experiment2/` | Grid search results with various PCA configurations |
| `experiment2_3layers/` | 3-layer variant of experiment2 |
| `experiment3/` | Early symbolic regression proof-of-concept |
| `figures/` | Generated plots (`benchmark_comparison.png`, `explained_var_ratio.png`, `pca_results.png`) |
| `model_example_outputs/` | Qualitative text generation comparisons between baseline, skip MLP, and symbolic |
| `logs/` | HPC Slurm job logs |

---

## File Structure

```
LLM_PCA/
├── experiment.py          # Early proof-of-concept (not in paper)
├── experiment2.py         # Grid search exploration (not in paper)
├── experiment3.py         # Early SR test (not in paper)
├── experiment4.py         # PCA sensitivity (Figure 4)
├── experiment5.py         # PCA + SR (Table 1)
├── experiment5_ablation.py # Identity ablation (Table 1)
├── experiment7.py         # Benchmarking (Figure 5, Table 5)
├── load_qwen.py           # Qwen model loading utility
├── load_other_models.py   # Generic HuggingFace model loader
├── torch_pca.py           # PyTorch-native PCA wrapper for GPU
├── requirements.txt       # Python dependencies
├── train_prompts.json     # Training prompts for text generation
├── test_prompts.json      # Test prompts
└── [results folders]/     # Experimental outputs
```

---

## Requirements

```
datasets==4.4.1
numpy==2.4.1
scikit_learn==1.8.0
torch==2.9.1
torch_symbolic==2.1.2
transformers==4.57.3
```

Install dependencies:
```bash
pip install -r requirements.txt
```

**Note:** You also need to install SymTorch from the specific commit mentioned above, or use a compatible version.

---

## Reproducing Paper Results

### Table 1 Results

```bash
# PCA+MLP (32 input, 8 output components)
python experiment4.py 32 8 --max-chars 750000

# PCA+SymTorch
python experiment5.py 32 8 --max-chars 750000

# Control (identity)
python experiment5_ablation.py --max-chars 750000
```

### Figure 4 (PCA Sensitivity Heatmap)

Run experiment4.py with various input/output PCA component combinations:
```bash
for i in 16 32 64 128; do
  for o in 8 16 32 64 128 256; do
    python experiment4.py $i $o --max-chars 750000
  done
done
```

### Figure 5 (Throughput vs Perplexity)

```bash
# Symbolic Qwen model
python experiment7.py experiment5/layers_7_14_21/max_chars750000/pca_comps_I32_O8

# Other models for comparison
python experiment7.py <results_dir> --model_name "meta-llama/Llama-3.2-1B-Instruct"
python experiment7.py <results_dir> --model_name "HuggingFaceTB/SmolLM2-1.7B-Instruct"
# ... etc for other models
```

---

## Citation

If you use this code, please cite the SymTorch paper:

```bibtex
@inproceedings{symtorch2025,
  title={SymTorch: A PyTorch Framework for Symbolic Distillation of Deep Neural Networks},
  author={Anonymous},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2025}
}
```

"""
Experiment 7: Benchmark symbolic model performance on WikiText

This script loads symbolic models trained by experiment5.py and benchmarks their
performance with switch_to_symbolic() OFF vs ON, on both CPU and GPU.

Metrics measured:
- Inference time (forward pass latency)
- Perplexity on WikiText validation set
- Performance on CPU vs GPU (if available)

Usage: python experiment7.py <results_dir>

MODELS THAT WE USE IN THIS BENCHMARKING:

"Qwen/Qwen2.5-0.5B-Instruct"
"Qwen/Qwen2.5-1.5B-Instruct"
"Qwen/Qwen2.5-3B-Instruct"
"meta-llama/Llama-3.2-1B-Instruct"
"meta-llama/Llama-3.2-3B-Instruct"
"HuggingFaceTB/SmolLM3-3B-Instruct"

"""
import argparse
import contextlib
import json
import math
import os
import pickle
import random
import sys
import time

import datasets
import symtorch
import torch

import torch_pca


@contextlib.contextmanager
def block_print():
    original_stdout = sys.stdout
    try:
        sys.stdout = open(os.devnull, 'w')
        yield
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout


def benchmark_inference(
    model,
    tokenizer,
    text,
    max_length=1024,
    device="cpu",
    num_warmup=5,
    num_iterations=20,
):
    model.eval()  # IMPORTANT

    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings["input_ids"][:, :max_length].to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_ids=input_ids, use_cache=False)

    if device == "cuda":
        torch.cuda.synchronize()

    times = []

    # Benchmark
    for _ in range(num_iterations):
        if device == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        else:
            start = time.perf_counter()

        with torch.no_grad():
            _ = model(input_ids=input_ids, use_cache=False)

        if device == "cuda":
            end.record()
            torch.cuda.synchronize()
            elapsed_ms = start.elapsed_time(end)
        else:
            elapsed_ms = (time.perf_counter() - start) * 1000

        times.append(elapsed_ms)

    avg_ms = sum(times) / len(times)
    tokens_per_sec = (input_ids.size(1) * num_iterations) / (sum(times) / 1000)

    return {
        "avg_latency_ms": avg_ms,
        "p95_latency_ms": sorted(times)[int(0.95 * len(times))],
        "tokens_per_second": tokens_per_sec,
        "total_time_sec": sum(times) / 1000,
        "num_iterations": num_iterations,
        "sequence_length": input_ids.size(1),
    }


def get_perplexity(model, tokenizer, text, max_length=1024, device="cpu"):
    """Calculate perplexity on text (sanity check)."""
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings['input_ids'][0]

    num_chunks = (len(input_ids) + max_length - 1) // max_length

    total_loss = 0.0
    total_tokens = 0

    for i in range(0, len(input_ids), max_length):
        chunk = input_ids[i:i+max_length].unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_ids=chunk, labels=chunk, use_cache=False)
            total_loss += outputs.loss.item() * chunk.size(1)
            total_tokens += chunk.size(1)

        if (i // max_length + 1) % 10 == 0:
            print(f"Processed chunk {i // max_length + 1}/{num_chunks}")

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    return perplexity


def load_pca_models(results_dir, layers, device, dtype=torch.float32):
    ret = {}
    for layer in layers:
        ret[layer] = {}
        for pca_type in ("input", "output"):
            path = os.path.join(results_dir, f"pca_{pca_type}_layer{layer}.pkl")
            with open(path, "rb") as fd:
                ret[layer][pca_type] = torch_pca.TorchPCA(
                    pickle.load(fd), device=device, dtype=dtype
                )
    return ret


def load_symbolic_models(results_dir, layers, n_pca_inputs, n_pca_outputs):
    ret = {}
    for layer in layers:
        ret[layer] = {}
        path = os.path.join(
            results_dir,
            f"symbolic_model_I{n_pca_inputs}_O{n_pca_outputs}_layer{layer}"
        )
        with block_print():  # XXX: This might hide errors.
            model = symtorch.SymbolicModel.load_model(path)
            model.switch_to_symbolic()
        ret[layer]["symbolic"] = model
    return ret


def load_models(results_dir, device):
    from load_qwen import load_qwen_model

    metadata = load_experimental_results(results_dir)
    layers = metadata["layers"]
    n_pca_inputs = metadata["pca_comps_I"]
    n_pca_outputs = metadata["pca_comps_O"]

    model, tokenizer = load_qwen_model(device=device)

    # Get model's dtype to ensure consistency
    model_dtype = next(model.parameters()).dtype

    pca_models = load_pca_models(results_dir, layers, device=device, dtype=model_dtype)
    sym_models = load_symbolic_models(
        results_dir, layers, n_pca_inputs, n_pca_outputs
    )
    aux_models = {l: pca_models[l] | sym_models[l] for l in layers}

    return model, tokenizer, aux_models

from load_other_models import load_hf_model


def to_neural(model, layers):
    for layer in layers:
        curr_mlp = model.model.layers[layer].mlp
        if hasattr(curr_mlp, "base_mlp"):
            model.model.layers[layer].mlp = curr_mlp.base_mlp


class SkipMLP(torch.nn.Module):
    def __init__(self, base_mlp):
        super().__init__()
        # NOTE: Unused since we totally rewrite the forward pass.
        self.base_mlp = base_mlp

    def forward(self, x):
        return x  # Do nothing on the forward pass.


def to_skip(model, layers):
    to_neural(model, layers)  # Reset if needed.
    for layer in layers:
        base_mlp = model.model.layers[layer].mlp
        model.model.layers[layer].mlp = SkipMLP(base_mlp)
        # TODO: torch.compile optional?
        #  model.model.layers[layer].mlp = torch.compile(
        #      model.model.layers[layer].mlp
        #  )


class SymbolicMLP(torch.nn.Module):
    def __init__(self, base_mlp, input_pca, output_pca, sr_model):
        super().__init__()
        
        self.base_mlp = base_mlp # NOTE: Unused since we totally rewrite the forward pass.
        self.input_pca = input_pca
        self.output_pca = output_pca
        self.sr_model = sr_model
        # self.out_features = self.base_mlp.down_proj.out_features

    def forward(self, x):
        assert x.shape[0] == 1
        return self.output_pca.inverse_transform(
            self.sr_model(self.input_pca.transform(x[0]))
        )


def to_symbolic(model, aux_models, layers):
    to_neural(model, layers)  # Reset if needed.
    for layer in layers:
        base_mlp = model.model.layers[layer].mlp
        model.model.layers[layer].mlp = SymbolicMLP(
            base_mlp,
            input_pca=aux_models[layer]["input"],
            output_pca=aux_models[layer]["output"],
            sr_model=aux_models[layer]["symbolic"]
        )
        # TODO: torch.compile optional?


def run_benchmarks(
    model, tokenizer,
    train_text, val_text,
    device, num_warmup, num_iterations, 
    compute_perplexity = True,
):

    if compute_perplexity:
        print(f"  Computing perplexity (sanity check)...")
        train_ppl = get_perplexity(model, tokenizer, train_text, device=device)
        print(f"    Train Perplexity : {train_ppl:.4f}")
        val_ppl = get_perplexity(model, tokenizer, val_text, device=device)
        print(f"    Val Perplexity   : {val_ppl:.4f}")

    print(f"  Benchmarking inference speed...")
    rslt = benchmark_inference(
        model, tokenizer, val_text,
        device=device,
        num_warmup=num_warmup,
        num_iterations=num_iterations,
    )
    print(f"    Avg latency : {rslt['avg_latency_ms']:.2f} ms")
    print(f"    P95 latency : {rslt['p95_latency_ms']:.2f} ms")
    print(f"    Throughput  : {rslt['tokens_per_second']:.1f} tokens/sec\n")

    if compute_perplexity:
        rslt['train_perplexity'] = train_ppl
        rslt['val_perplexity'] = val_ppl

    return rslt


def load_wikitext(max_chars, seed, verbose=True):
    wikitext = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-v1")
    train_data = wikitext["train"]["text"]
    train_text = "\n".join([t for t in train_data if t.strip()])

    val_data = wikitext["validation"]["text"]
    val_text = "\n".join([t for t in val_data if t.strip()])

    if max_chars is not None:
        # Use random sampling for better representation
        random.seed(seed)

        def sample_text_random(text):
            if len(text) <= max_chars:
                return text
            # Random starting position
            max_start = len(text) - max_chars
            start_pos = random.randint(0, max_start)
            return text[start_pos:start_pos + max_chars]

        train_text = sample_text_random(train_text)
        val_text = sample_text_random(val_text)

    if not verbose:
        return train_text, val_text
    print(f"Training text length   : {len(train_text)} characters")
    print(f"Validation text length : {len(val_text)} characters")
    return train_text, val_text


def load_experimental_results(results_dir, verbose=False):
    results_file = os.path.join(results_dir, "experimental_results.json")
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")

    with open(results_file, 'r') as f:
        exp_results = json.load(f)

    if not verbose:
        return exp_results

    layers = exp_results['layers']
    pca_comps_I = exp_results['pca_comps_I']
    pca_comps_O = exp_results['pca_comps_O']
    base_train_ppl = exp_results["baseline_perplexity_train"]
    base_val_ppl = exp_results["baseline_perplexity_val"]
    int_train_ppl = exp_results["intervened_perplexity_train"]
    int_val_ppl = exp_results["intervened_perplexity_val"]
    print(f"{'='*60}")
    print(f"EXPERIMENT 7: Symbolic Model Benchmarking")
    print(f"{'='*60}")
    print(f"Results location: {results_dir}")
    print(f"Layers: {layers}")
    print(f"PCA components: I={pca_comps_I}, O={pca_comps_O}")
    print(f"Baseline Perplexity")
    print(f"  Train : {base_train_ppl:.4f}")
    print(f"  Val   : {base_val_ppl:.4f}")
    print(f"Intervened Perplexity")
    print(f"  Train : {int_train_ppl:.4f}")
    print(f"  Val   : {int_val_ppl:.4f}")
    print(f"{'='*60}\n")
    return exp_results


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'results_dir', help='Path to experiment5 results directory'
    )
    parser.add_argument(
        '--max-chars', type=int, default=750000,
        help='Max characters from WikiText for benchmarking (default: None)'
    )
    parser.add_argument(
        '--num-iterations', type=int, default=100,
        help='Number of benchmark iterations (default: 100)'
    )
    parser.add_argument(
        '--num-warmup', type=int, default=5,
        help='Number of warmup iterations (default: 5)'
    )

    parser.add_argument(
        '--model_name', type = str, default='original' # this runs the benchmarking for the hybrid symbolic model and usual qwen2.5-1.5bn
    )

    parser.add_argument(
        '--no-compute-ppl', action='store_true',
        help='Skip perplexity computation (only benchmark speed)'
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--cpu-only', action='store_true', help='Only benchmark on CPU'
    )
    group.add_argument(
        '--gpu-only', action='store_true', help='Only benchmark on GPU'
    )
    parser.add_argument('--seed', type=int, default=290402)
    return parser, parser.parse_args()


def main():
    parser, args = parse_args()

    model_name = args.model_name

    # Only load metadata for 'original' model
    if model_name == "original":
        metadata = load_experimental_results(args.results_dir, verbose=True)

    devices = []
    if not args.gpu_only:
        devices.append("cpu")
    if not args.cpu_only and torch.cuda.is_available():
        devices.append("cuda")
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    if not args.cpu_only and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append("mps")
        print(f"MPS available")
    if not devices:
        parser.error("no devices available for benchmarking")
    print(f"Testing on {devices}\n")

    # Only load dataset if computing perplexity
    compute_ppl = not args.no_compute_ppl

    if compute_ppl:
        print("Loading WikiText dataset...")
        train_text, val_text = load_wikitext(args.max_chars, args.seed)
        print()
    else:
        # Still need val_text for inference benchmarking, but can use smaller sample
        print("Loading WikiText dataset (validation only)...")
        train_text, val_text = load_wikitext(args.max_chars, args.seed, verbose=False)
        train_text = None  # Don't need training text if not computing perplexity

    # Create experiment7_results directory if it doesn't exist
    experiment7_dir = "experiment7_results"
    os.makedirs(experiment7_dir, exist_ok=True)

    # Generate timestamp for unique filenames
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    for device in devices:
        print(f"{'='*60}")
        print(f"Benchmarking on {device.upper()}")
        print(f"{'='*60}\n")

        print(f"Loading models on {device}")

        if model_name == "original":
            # Create subfolder for this benchmark run
            run_dir = os.path.join(experiment7_dir, f"original_{device}_{timestamp}")
            os.makedirs(run_dir, exist_ok=True)

            model, tokenizer, aux_models = load_models(args.results_dir, device)
            print()

            # Baseline
            print(f"[{device.upper()}] BASELINE (no intervention)")
            res = run_benchmarks(
                model, tokenizer,
                train_text, val_text,
                device, args.num_warmup, args.num_iterations, compute_ppl
            )

            # Save benchmark results
            results_file = os.path.join(run_dir, "baseline_benchmark.json")
            benchmark_data = {k: v for k, v in res.items() if k not in ['train_perplexity', 'val_perplexity']}
            with open(results_file, 'w') as f:
                json.dump(benchmark_data, f, indent=2)
            print(f"Benchmark results saved to {results_file}")

            # Save perplexity results if computed
            if compute_ppl:
                ppl_file = os.path.join(run_dir, "baseline_perplexity.json")
                ppl_data = {
                    'train_perplexity': res['train_perplexity'],
                    'val_perplexity': res['val_perplexity']
                }
                with open(ppl_file, 'w') as f:
                    json.dump(ppl_data, f, indent=2)
                print(f"Perplexity results saved to {ppl_file}")
            print()

            # Skip MLP
            print(f"[{device.upper()}] SKIP MLP (forward pass does nothing)")
            to_skip(model, metadata["layers"])
            res = run_benchmarks(
                model, tokenizer,
                train_text, val_text,
                device, args.num_warmup, args.num_iterations, compute_ppl
            )

            results_file = os.path.join(run_dir, "skip_mlp_benchmark.json")
            benchmark_data = {k: v for k, v in res.items() if k not in ['train_perplexity', 'val_perplexity']}
            with open(results_file, 'w') as f:
                json.dump(benchmark_data, f, indent=2)
            print(f"Benchmark results saved to {results_file}")

            if compute_ppl:
                ppl_file = os.path.join(run_dir, "skip_mlp_perplexity.json")
                ppl_data = {
                    'train_perplexity': res['train_perplexity'],
                    'val_perplexity': res['val_perplexity']
                }
                with open(ppl_file, 'w') as f:
                    json.dump(ppl_data, f, indent=2)
                print(f"Perplexity results saved to {ppl_file}")
            print()

            # Symbolic
            print(f"[{device.upper()}] SYMBOLIC (with symbolic intervention)")
            to_symbolic(model, aux_models, metadata["layers"])
            res = run_benchmarks(
                model, tokenizer,
                train_text, val_text,
                device, args.num_warmup, args.num_iterations, compute_ppl
            )

            results_file = os.path.join(run_dir, "symbolic_benchmark.json")
            benchmark_data = {k: v for k, v in res.items() if k not in ['train_perplexity', 'val_perplexity']}
            with open(results_file, 'w') as f:
                json.dump(benchmark_data, f, indent=2)
            print(f"Benchmark results saved to {results_file}")

            if compute_ppl:
                ppl_file = os.path.join(run_dir, "symbolic_perplexity.json")
                ppl_data = {
                    'train_perplexity': res['train_perplexity'],
                    'val_perplexity': res['val_perplexity']
                }
                with open(ppl_file, 'w') as f:
                    json.dump(ppl_data, f, indent=2)
                print(f"Perplexity results saved to {ppl_file}")
            print()

        else:
            # Create subfolder for this model
            safe_model_name = model_name.replace('/', '_')
            run_dir = os.path.join(experiment7_dir, f"{safe_model_name}_{device}_{timestamp}")
            os.makedirs(run_dir, exist_ok=True)

            model, tokenizer = load_hf_model(device, model_name)
            res = run_benchmarks(
                model, tokenizer,
                train_text, val_text,
                device, args.num_warmup, args.num_iterations, compute_ppl
            )

            # Save benchmark results
            results_file = os.path.join(run_dir, "benchmark.json")
            benchmark_data = {k: v for k, v in res.items() if k not in ['train_perplexity', 'val_perplexity']}
            with open(results_file, 'w') as f:
                json.dump(benchmark_data, f, indent=2)
            print(f"Benchmark results saved to {results_file}")

            # Save perplexity results if computed
            if compute_ppl:
                ppl_file = os.path.join(run_dir, "perplexity.json")
                ppl_data = {
                    'train_perplexity': res['train_perplexity'],
                    'val_perplexity': res['val_perplexity']
                }
                with open(ppl_file, 'w') as f:
                    json.dump(ppl_data, f, indent=2)
                print(f"Perplexity results saved to {ppl_file}")
            print()





if __name__ == "__main__":
    main()

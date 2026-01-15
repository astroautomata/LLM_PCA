
import os
import json
import torch
from experiment7 import *

def generate_text(model, tokenizer, prompt, max_new_tokens=100, device="mps"):
    """Generate text for a given prompt."""
    model.eval()  # Set to evaluation mode

    # Format as chat message
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize
    inputs = tokenizer([text], return_tensors="pt").to(device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode - return the full generated text including prompt
    # (This matches the behavior of experiment2.py which produced good results)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_text

def main():

    # You change this stuff
    prompt = "Create a short story that includes fruits and vegetables but make sure not to include bananas in this story."
    prompt_name = "no_bananas"
    max_tokens = 500

    saving_dir = "model_example_outputs"
    results_dir = "experiment5/layers_7_14_21/max_chars750000/pca_comps_I32_O8"
    device = "mps"

    os.makedirs(saving_dir, exist_ok=True)

    metadata = load_experimental_results(results_dir, verbose=True)

    model, tokenizer, aux_models = load_models(results_dir, device)

    #run baseline
    response = generate_text(model, tokenizer, prompt, max_new_tokens= max_tokens)
    with open(f'{saving_dir}/baseline_{prompt_name}.json', 'w') as f:
        json.dump(response, f, indent=2)

    #run skip mlp
    to_skip(model, metadata["layers"])
    response = generate_text(model, tokenizer, prompt, max_new_tokens= max_tokens)
    with open(f'{saving_dir}/skip_mlp_{prompt_name}.json', 'w') as f:
        json.dump(response, f, indent=2)

    #run symbolic model
    to_symbolic(model, aux_models, metadata["layers"])
    response = generate_text(model, tokenizer, prompt, max_new_tokens= max_tokens)
    with open(f'{saving_dir}/symbolic_{prompt_name}.json', 'w') as f:
        json.dump(response, f, indent=2)


if __name__ == "__main__":
    main()

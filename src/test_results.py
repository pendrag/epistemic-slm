import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import torch.nn.functional as F

from generate_dataset import add_i_dont_know_choice

def load_config(config_path):
    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        raise ValueError("Config file must .yaml or .yml")
    
def test_model(dataset, model, tokenizer):
    """Evaluate results on the preference dataset.
    Returns: original answer, correct answer, is_correct"""
    results = []
    for example in tqdm(dataset):
        question = example["question"]

        prompt = f"Question: {question}\nChoices:\n"
        for label, text in zip(example["choices"]["label"], example["choices"]["text"]):
            prompt += f"{label}. {text}\n"
        prompt += "Answer:"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=1, 
                return_dict_in_generate=True, 
                output_scores=True, 
                do_sample=False
            )

        decoded_output = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

        predicted_answer = decoded_output.split("Answer:")[-1].strip()[0] if "Answer:" in decoded_output and decoded_output.split("Answer:")[-1].strip() else example["idk_label"]

        # logits for generated step (the next token after prompt)
        logits = outputs.scores[0][0]  # shape: (1, vocab_size)
        choices = ["A", "B", "C", "D", "E", "F"]
        token_ids = torch.tensor([tokenizer.convert_tokens_to_ids(t) for t in choices]).to(model.device)
        selected_logits = logits[token_ids]
        # Calculate probabilities with softmax
        restricted_vocab_probs = torch.nn.functional.softmax(selected_logits, dim=0)

        # Convert to Python dict safely without calling .item() on the full tensor
        choice_logprobs = {choices[i]: restricted_vocab_probs[i].item() for i in range(len(choices))}

        # Compute probs for all vocabulary
        all_vocab_probs = F.softmax(logits, dim=0)
        selected_probs = all_vocab_probs[token_ids]
        # Mapping tokens to their probabilities
        token_prob_dict = {choices[i]: selected_probs[i].item() for i in range(len(choices))}

        # Get top 10 tokens and their probabilities
        top_probs, top_indices = torch.topk(all_vocab_probs, 10)
        top_tokens = [tokenizer.decode([idx]).strip() for idx in top_indices]

        top_10 = {top_tokens[i]: top_probs[i].item() for i in range(len(top_tokens))}

        results.append({
            "prompt": prompt,
            "predicted_answer": predicted_answer,
            "correct_answer": example["answerKey"],
            "idk_answer": example["idk_label"],
            "is_correct": predicted_answer.lower() == example["answerKey"].lower(),
            "relative_probs": choice_logprobs,
            "absolute_probs": token_prob_dict,
            "top_10_tokens": top_10
        })

    return results

def main(config_path):
    cfg = load_config(config_path)

    dataset = load_dataset(cfg["qa_dataset"], split="validation")

    # Select only 10 samples for testing
    dataset = dataset.select(range(2))

    dataset = dataset.map(
        add_i_dont_know_choice,
        batched=False, 
        batch_size=cfg.get("batch_size", 8)
    )

    print("Added 'I don't know' choice to dataset.")
    tokenizer = AutoTokenizer.from_pretrained(cfg['base_model_name'])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    print("Testing model on dataset...")
    results = test_model(dataset, model, tokenizer)

    # Save results
    import json
    with open(os.path.join(cfg["output_dir"], f"{cfg['model_name'].split('/')[-1]}_results.json"), "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        sys.exit(1)
    config_path = sys.argv[1]
    main(config_path)
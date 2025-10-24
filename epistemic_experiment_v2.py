# -*- coding: utf-8 -*-

import os
import json
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import matplotlib.pyplot as plt
from tqdm import tqdm

print(torch.version.cuda)
print(torch.cuda.get_device_name(0))

# Configuration dictionary
config = {
    "model": {
        "name": "models/HuggingFaceTB/SmolLM2-1.7B-Instruct",
    },
    "lora_target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
    ],
    "dataset": {
        "name": "sentence-transformers/natural-questions",
        "max_train_samples": 10000,
        "test_fraction": 0.1,
    },
    "training": {
        "epochs": 3,
        "batch_size": 16,
        "gradient_accumulation_steps": 0,
        "learning_rate": 5e-5,
        "warmup_steps": 100,
    },
    "paths": {
        "output_dir": "./epistemic_experiment_results"
    }
}

os.makedirs(config["paths"]["output_dir"], exist_ok=True)

# Logger utility
class Logger:
    def info(self, *args):
        print("[INFO]", *args)

logger = Logger()

def load_qa_dataset(config):
    """Load 'sentence-transformers/natural-questions' and create train/test split."""
    dataset_name = "datasets/natural-questions"
    max_samples = config.get("max_train_samples", 10000)
    test_fraction = config.get("test_fraction", 0.1)

    logger.info(f"Loading dataset {dataset_name}")
    ds = load_dataset("parquet", data_files="datasets/pair/train-00000-of-00001.parquet")["train"]  # only train split available

    # Limit to max_samples if needed
    ds = ds.select(range(min(len(ds), max_samples)))

    # Shuffle and split into train and test
    ds = ds.shuffle(seed=42)
    split_idx = int(len(ds) * (1 - test_fraction))
    train_ds = ds.select(range(split_idx))
    test_ds = ds.select(range(split_idx, len(ds)))

    # Format to standard dict including only question and answer fields
    def format_example(ex):
        return {"question": ex["query"], "answer": ex["answer"]}

    train_ds = train_ds.map(format_example, remove_columns=train_ds.column_names)
    test_ds = test_ds.map(format_example, remove_columns=test_ds.column_names)

    logger.info(f"Created train/test split with {len(train_ds)} train and {len(test_ds)} test samples")
    return train_ds, test_ds

def load_tokenizer_and_model(model_name):
    logger.info(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    return tokenizer, model

def generate_responses(model, dataset, tokenizer):
    logger.info("Generating responses from model...")
    responses = []
    device = next(model.parameters()).device

    for item in tqdm(dataset, desc="Generating responses"):
        question = item["question"]
        prompt = f"Question: {question}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = decoded[len(prompt):].strip()
        responses.append(response)
    return responses

def evaluate_responses(responses, ground_truth):
    logger.info("Evaluating response correctness...")
    correct_flags = []
    for response, truth in zip(responses, ground_truth):
        response_lower = response.lower()
        truth_lower = truth.lower()
        truth_words = [w for w in truth_lower.split() if len(w) > 2]
        is_correct = any(word in response_lower for word in truth_words) if truth_words else False
        correct_flags.append(is_correct)
    accuracy = sum(correct_flags) / len(correct_flags)
    logger.info(f"Model accuracy: {accuracy:.3f}")
    return correct_flags

def create_instruction_dataset(train_dataset, output_dir, responses=None, correct_flags=None):
    logger.info("Creating instruction dataset with epistemic humility...")

    instruction_path = os.path.join(output_dir, "instruction_dataset")
    if os.path.exists(instruction_path):
        print(f"Instruction dataset encontrado en {instruction_path}, cargando...")
        instr_dataset = Dataset.load_from_disk(instruction_path)
    else:
        data = []
        for item, response, correct in zip(train_dataset, responses, correct_flags):
            target_answer = item["answer"] if correct else "I don't know"
            data.append({
                "instruction": item["question"],
                "input": "",
                "output": target_answer,
                "original_response": response,
                "was_correct": correct,
            })
        instr_dataset = Dataset.from_list(data)
        instr_dataset.save_to_disk(os.path.join(output_dir, "instruction_dataset"))
        logger.info(f"Instruction dataset saved with {len(instr_dataset)} samples")
    return instr_dataset

def prepare_training_data(instruction_dataset, tokenizer):
    def format_example(example):
        prompt = f"Question: {example['instruction']}\nAnswer: {example['output']}"
        return {"text": prompt}
    
    print(instruction_dataset.column_names)  # Así sabes si existe 'instruction'
    print(instruction_dataset[0])           # Así inspeccionas los datos directamente
    formatted = instruction_dataset.map(format_example)
    tokenized = formatted.map(
        lambda examples: tokenizer(examples["text"], truncation=True, padding=True, max_length=512),
        batched=True
    )
    return tokenized

def get_precision_settings():
    if not torch.cuda.is_available():
        return {"fp16": False, "bf16": False}
    if torch.cuda.is_bf16_supported():
        return {"fp16": False, "bf16": True}
    return {"fp16": False, "bf16": False}

def fine_tune_model(tokenized_dataset, tokenizer, cfg):
    logger.info("Starting LoRA fine-tuning...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name"], 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=cfg.get("lora_target_modules"),
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=os.path.join(cfg["paths"]["output_dir"], "finetuned_model"),
        overwrite_output_dir=True,
        num_train_epochs=cfg["training"]["epochs"],
        per_device_train_batch_size=cfg["training"]["batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        warmup_steps=cfg["training"]["warmup_steps"],
        learning_rate=cfg["training"]["learning_rate"],
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        prediction_loss_only=True,
        report_to="none",
        remove_unused_columns=True,
        **get_precision_settings(),
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    trainer.train()

    finetuned_path = os.path.join(cfg["paths"]["output_dir"], "finetuned_model")
    model.save_pretrained(finetuned_path)
    tokenizer.save_pretrained(finetuned_path)
    logger.info(f"LoRA adapters saved to {finetuned_path}")
    return finetuned_path

def compare_models(test_dataset, tokenizer, finetuned_model_path):
    logger.info("Evaluating original and fine-tuned models on test set...")
    original_model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"],
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    finetuned_model = AutoModelForCausalLM.from_pretrained(
        finetuned_model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    original_model.to(device)
    finetuned_model.to(device)

    original_responses = generate_responses(original_model, test_dataset, tokenizer)
    finetuned_responses = generate_responses(finetuned_model, test_dataset, tokenizer)
    ground_truth = [item["answer"] for item in test_dataset]

    original_correct = evaluate_responses(original_responses, ground_truth)
    finetuned_correct = evaluate_responses(finetuned_responses, ground_truth)

    original_idk_count = sum("don't know" in r.lower() for r in original_responses)
    finetuned_idk_count = sum("don't know" in r.lower() for r in finetuned_responses)

    finetuned_idk_on_wrong = 0
    finetuned_idk_on_correct = 0
    for orig_corr, ft_resp in zip(original_correct, finetuned_responses):
        if "don't know" in ft_resp.lower():
            if not orig_corr:
                finetuned_idk_on_wrong += 1
            else:
                finetuned_idk_on_correct += 1

    results = {
        "original_accuracy": sum(original_correct) / len(original_correct),
        "finetuned_accuracy": sum(finetuned_correct) / len(finetuned_correct),
        "original_idk_count": original_idk_count,
        "finetuned_idk_count": finetuned_idk_count,
        "finetuned_idk_on_wrong": finetuned_idk_on_wrong,
        "finetuned_idk_on_correct": finetuned_idk_on_correct,
        "total_originally_wrong": sum(1 for c in original_correct if not c),
        "responses": {
            "original": original_responses,
            "finetuned": finetuned_responses,
            "ground_truth": ground_truth,
            "original_correct": original_correct,
            "finetuned_correct": finetuned_correct,
        },
    }
    return results

def analyze_results(results):
    logger.info("Analyzing epistemic humility results...")
    total_wrong = results["total_originally_wrong"]
    idk_wrong = results["finetuned_idk_on_wrong"]
    idk_correct = results["finetuned_idk_on_correct"]

    precision = idk_wrong / (idk_wrong + idk_correct) if (idk_wrong + idk_correct) > 0 else 0
    recall = idk_wrong / total_wrong if total_wrong > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    accuracy_change = results["finetuned_accuracy"] - results["original_accuracy"]
    idk_increase = results["finetuned_idk_count"] - results["original_idk_count"]

    analysis = {
        "epistemic_humility_precision": precision,
        "epistemic_humility_recall": recall,
        "epistemic_humility_f1": f1,
        "accuracy_change": accuracy_change,
        "idk_increase": idk_increase,
        "conclusions": [],
    }

    if recall > 0.5:
        analysis["conclusions"].append(
            "Model successfully learned to say 'I don't know' for many previously incorrect answers."
        )
    else:
        analysis["conclusions"].append(
            "Model showed limited improvement in expressing uncertainty for wrong answers."
        )
    if precision > 0.7:
        analysis["conclusions"].append(
            "Model shows good precision in epistemic humility - mostly says 'I don't know' appropriately."
        )
    else:
        analysis["conclusions"].append(
            "Model may over-generalize 'I don't know' to questions it can answer."
        )
    if accuracy_change > 0:
        analysis["conclusions"].append("Fine-tuning improved overall accuracy.")
    elif accuracy_change < -0.05:
        analysis["conclusions"].append("Fine-tuning reduced accuracy - may need tuning.")

    return analysis

def create_visualizations(results, analysis, output_dir):
    logger.info("Creating visualizations...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Accuracy comparison
    axes[0, 0].bar(["Original", "Fine-tuned"], [results["original_accuracy"], results["finetuned_accuracy"]], color=["blue", "orange"])
    axes[0, 0].set_title("Model Accuracy Comparison")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].set_ylim(0, 1)

    # "I don't know" counts
    axes[0, 1].bar(["Original", "Fine-tuned"], [results["original_idk_count"], results["finetuned_idk_count"]], color=["blue", "orange"])
    axes[0, 1].set_title("\"I Don't Know\" Response Counts")
    axes[0, 1].set_ylabel("Count")

    # Breakdown of IDK usage in fine-tuned
    categories = ["IDK on Wrong", "IDK on Correct"]
    values = [results["finetuned_idk_on_wrong"], results["finetuned_idk_on_correct"]]
    axes[1, 0].bar(categories, values, color=["green", "red"])
    axes[1, 0].set_title("\"I Don't Know\" Usage in Fine-tuned Model")
    axes[1, 0].set_ylabel("Count")

    # Epistemic humility precision/recall/f1
    metrics = ["Precision", "Recall", "F1"]
    metric_vals = [
        analysis["epistemic_humility_precision"],
        analysis["epistemic_humility_recall"],
        analysis["epistemic_humility_f1"],
    ]
    axes[1, 1].bar(metrics, metric_vals, color=["purple", "brown", "pink"])
    axes[1, 1].set_title("Epistemic Humility Metrics")
    axes[1, 1].set_ylabel("Score")
    axes[1, 1].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "results_visualization.png"), dpi=300, bbox_inches="tight")
    plt.close()

def save_results(results, analysis, cfg):
    logger.info("Saving results and report...")
    summary_results = {k: v for k, v in results.items() if k != "responses"}
    summary = {
        "results": summary_results,
        "analysis": analysis,
        "config": cfg,
    }
    with open(os.path.join(cfg["paths"]["output_dir"], "experiment_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(cfg["paths"]["output_dir"], "detailed_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    report = f"""# Epistemic Humility Experiment Report

## Experiment Configuration

- Model: {cfg['model']['name']}
- Dataset: {cfg['dataset']['name']}
- Training samples: {cfg['dataset']['max_train_samples']}
- Test samples: {cfg['dataset']['test_fraction']}

## Results Summary

### Accuracy Metrics

- Original model accuracy: {results['original_accuracy']:.3f}
- Fine-tuned model accuracy: {results['finetuned_accuracy']:.3f}
- Accuracy change: {analysis['accuracy_change']:.3f}

### Epistemic Humility Metrics

- Precision: {analysis['epistemic_humility_precision']:.3f}
- Recall: {analysis['epistemic_humility_recall']:.3f}
- F1 Score: {analysis['epistemic_humility_f1']:.3f}

### "I Don't Know" Response Analysis

- Original model "IDK" responses: {results['original_idk_count']}
- Fine-tuned model "IDK" responses: {results['finetuned_idk_count']}
- "IDK" on originally wrong answers: {results['finetuned_idk_on_wrong']}
- "IDK" on originally correct answers: {results['finetuned_idk_on_correct']}

### Key Insights

"""
    for conclusion in analysis["conclusions"]:
        report += f"- {conclusion}\n"
    
    with open(os.path.join(cfg["paths"]["output_dir"], "experiment_report.md"), "w") as f:
        f.write(report)

def main():
    # Load datasets
    train_ds, test_ds = load_qa_dataset(config)

    # Load original model and tokenizer
    tokenizer, original_model = load_tokenizer_and_model(config["model"]["name"])

    instruction_path = os.path.join(config["paths"]["output_dir"], "instruction_dataset")
    if os.path.exists(instruction_path):
        instruction_dataset = create_instruction_dataset(train_ds, config["paths"]["output_dir"])
    else:
        # Generate responses on training data
        train_responses = generate_responses(original_model, train_ds, tokenizer)

        # Evaluate original responses
        ground_truth_train = [item["answer"] for item in train_ds]
        correct_flags = evaluate_responses(train_responses, ground_truth_train)

        # Create instruction dataset with replaced wrong answers as "I don't know"
        instruction_dataset = create_instruction_dataset(train_ds, config["paths"]["output_dir"], train_responses, correct_flags)

    # Prepare and tokenize training data for finetuning
    tokenized_training_data = prepare_training_data(instruction_dataset, tokenizer)

    # Fine-tune model with LoRA
    finetuned_model_path = fine_tune_model(tokenized_training_data, tokenizer, config)

    # Evaluate both original and fine-tuned models on test dataset
    results = compare_models(test_ds, tokenizer, finetuned_model_path)

    # Analyze results for epistemic humility
    analysis = analyze_results(results)

    # Create visualizations of results
    create_visualizations(results, analysis, config["paths"]["output_dir"])

    # Save results and detailed report
    save_results(results, analysis, config)

if __name__ == "__main__":
    main()

import os
import torch
import datetime
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, EarlyStoppingCallback
from peft import LoraConfig, TaskType
from trl import DPOConfig, DPOTrainer

from generate_dataset import add_i_dont_know_choice, generate_dpo_dataset

def load_config(config_path):
    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        raise ValueError("Config file must .yaml or .yml")


def train_dpo(cfg, train_dataset, eval_dataset, model, tokenizer):
    """Train model on a preference dataset using DPO."""

    # Configure wandb for offline mode if specified
    if cfg.get("use_wandb"):
        os.environ["WANDB_MODE"] = "offline"
        os.environ["WANDB_PROJECT"] = f"epistemic_humility_{cfg['training_type']}"
        os.environ["WANDB_NAME"] = f"{cfg['train_model_name']}_lr{cfg['learning_rate']}_beta{cfg['dpo_beta']}"

    dpo_config = DPOConfig(
        output_dir=os.path.join(cfg["output_dir"], f"{cfg['train_model_name']}_lr{cfg['learning_rate']}_beta{cfg['dpo_beta']}"),
        learning_rate=cfg["learning_rate"],
        beta=cfg["dpo_beta"],
        push_to_hub=False,
        max_length=cfg.get("max_length", 512),
        max_prompt_length=256,
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        gradient_checkpointing=False,
        metric_for_best_model="loss",
        greater_is_better=False,
        num_train_epochs=cfg["num_train_epochs"],
        eval_strategy="steps",
        eval_steps=cfg["eval_steps"],
        save_strategy="steps",
        save_steps=cfg["save_steps"],
        logging_steps=cfg["logging_steps"],
        load_best_model_at_end=True,
        save_total_limit=3,
        report_to="wandb" if cfg.get("use_wandb") else None,
    )

    if cfg.get("training_type") == "lora":
        print("Using LoRA for parameter-efficient fine-tuning.")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
    else:
        lora_config = None

    trainer = DPOTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        args=dpo_config,
        peft_config=lora_config,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=cfg["early_stopping_patience"])
        ],
    )
    trainer.train()

    return model

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Implement your metric calculation here, e.g. accuracy
    # Suppose predictions and labels are already processed to compare correct option.
    correct = sum([pred == label for pred, label in zip(predictions, labels)])
    return {"accuracy": correct / len(labels)}

def main(config_path):
    cfg = load_config(config_path)
    
    print("Loadig model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

    # Load dataset if it exists, else generate it
    if os.path.exists(cfg["preference_dataset"]):
        print("Loading existing preference dataset...")
        dpo_dataset = Dataset.load_from_disk(cfg["preference_dataset"])
    else:
        print("Generating preference dataset...")
        dataset = load_dataset(cfg["qa_dataset"])   # Load a multi-answer QA dataset
        dataset.pop("test", None)  # Remove test split if exists

        dataset = dataset.map(
            add_i_dont_know_choice, 
            batched=False,
        )
        print("Updated dataset with 'i don't know' choice")

        model = AutoModelForCausalLM.from_pretrained(
            cfg["model_name"],
            dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        
        preference_results = generate_dpo_dataset(dataset, model, tokenizer)
        print(f"Generated {len(preference_results)} preference data points.")
        print("Sample preference result:", preference_results[0])

        # Save results as a dataset
        print("Saving results dataset...")
        dpo_dataset = Dataset.from_list(preference_results)
        dpo_dataset.save_to_disk(cfg["preference_dataset"])

    # Split dataset into train and eval based on test_split parameter
    eval_split = cfg.get("eval_split", 0.1)  # Default to 10% if not specified
    print(f"Splitting dataset with eval_split={eval_split}")
    
    split_dataset = dpo_dataset.train_test_split(test_size=eval_split, seed=cfg.get("seed", 42))
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")

    # Train DPO model if not already trained
    train_model_name = f"{cfg['training_type']}_model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    cfg["train_model_name"] = train_model_name

    if os.path.exists(os.path.join(cfg['output_dir'], train_model_name)): # This never happens (datetime based name), replace with a 'do_train' flag
        print("Loading existing DPO trained model...")
        dpo_model = AutoModelForCausalLM.from_pretrained(
            os.path.join(cfg['output_dir'], train_model_name),
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        print("Training DPO model...")
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model_name"],
            dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        dpo_model = train_dpo(cfg, train_dataset, eval_dataset, model, tokenizer)
        print("DPO training completed.")

        # Save the trained model to output directory
        dpo_model.save_pretrained(os.path.join(cfg['output_dir'], f"{train_model_name}_lr{cfg['learning_rate']}_beta{cfg['dpo_beta']}"))

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python train_model.py <config_path>")
        sys.exit(1)
    config_path = sys.argv[1]
    main(config_path)
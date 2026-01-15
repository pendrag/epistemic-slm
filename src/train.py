import os
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback
from datasets import load_from_disk
from peft import LoraConfig
from utils import load_config

def compute_metrics(eval_preds):
    """Calculate the score based on the model's predictions."""
    import numpy as np
    logits, labels = eval_preds
    preds = np.argmax(logits, axis=-1)

    # Scoring is computed as a test-like exam where each correct prediction yields one point,
    # each incorrect prediction yields -1 point, and unanswered (i don't know) questions yield 0 points.
    score = 0
    for pred, label in zip(preds, labels):
        if label == -100:
            continue
        elif pred == label:
            score += 1
        else:
            score -= 1
    return {"score": score}

def main(cfg_path):
    cfg = load_config(cfg_path)

    if cfg['training_type'] == 'lora':
        model_name = cfg['model_name'].split('/')[-1]
        learning_rate = cfg['lora_lr']
        output_dir = f"{cfg['output_dir']}/{model_name}-DPO_lr{learning_rate}_beta{cfg['dpo_beta']}_r{cfg['lora_r']}_alpha{cfg['lora_alpha']}_do{cfg['lora_dropout']}"
    else:
        learning_rate = cfg['learning_rate']
        output_dir = f"{cfg['output_dir']}/{model_name}-DPO_lr{learning_rate}_beta{cfg['dpo_beta']}"

    print(*(f"{param}: {value}" for param, value in cfg.items()), sep="\n")

    # Configure wandb for offline mode if specified
    if cfg.get("use_wandb"):
        os.environ["WANDB_MODE"] = "offline"
        os.environ["WANDB_PROJECT"] = f"epistemic_humility_{cfg['training_type']}"
        os.environ["WANDB_NAME"] = f"{model_name}-DPO_lr{learning_rate}_beta{cfg['dpo_beta']}_r{cfg['lora_r']}_alpha{cfg['lora_alpha']}_do{cfg['lora_dropout']}"

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(cfg['model_name'], device_map="auto" if not cfg['use_accelerate'] else None)
    tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'])
    tokenizer.pad_token = tokenizer.eos_token

    train_preference_ds = load_from_disk(f"{cfg['output_ds']}/train/preference_dataset")
    val_preference_ds = load_from_disk(f"{cfg['output_ds']}/validation/preference_dataset")

    # Shuffle ds
    train_preference_ds = train_preference_ds.shuffle(seed=42)
    val_preference_ds = val_preference_ds.shuffle(seed=42)

    # Set up LoRA
    peft_config = LoraConfig(
        r=cfg['lora_r'],
        lora_alpha=cfg['lora_alpha'],
        lora_dropout=cfg['lora_dropout'],
        target_modules=cfg['lora_target_modules'],
    )

    # Set up DPO
    training_args = DPOConfig(
        output_dir=output_dir,
        beta=cfg['dpo_beta'], # Higher means less deviation from the reference model
        per_device_train_batch_size=cfg['per_device_train_batch_size'],
        per_device_eval_batch_size=cfg['per_device_eval_batch_size'],
        learning_rate=learning_rate,
        num_train_epochs=cfg['num_train_epochs'],
        do_train=True,
        gradient_checkpointing=False, # Accelerate requirement
        ddp_find_unused_parameters=False,   # Accelerate requirement
        logging_steps=cfg['logging_steps'],
        logging_first_step=True,
        save_steps=cfg['save_steps'],
        eval_steps=cfg['eval_steps'],
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        warmup_steps=cfg.get('warmup_steps', 0),
        report_to="wandb" if cfg.get("use_wandb") else None,
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_preference_ds,
        eval_dataset=val_preference_ds,
        peft_config=peft_config if cfg['training_type'] == 'lora' else None,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=cfg["early_stopping_patience"])
        ],
    )

    trainer.train()
    trainer.save_model(output_dir + "/best_model")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        sys.exit(1)
    config_path = sys.argv[1]
    main(config_path)


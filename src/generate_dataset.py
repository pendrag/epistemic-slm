import torch
import random
from transformers import pipeline
from tqdm import tqdm

def add_i_dont_know_choice(example):
    """Add 'I don't know' option and shuffle all choices."""
    # Store the original correct answer text BEFORE adding "I don't know"
    original_answer_index = example["choices"]["label"].index(example["answerKey"])
    correct_answer_text = example["choices"]["text"][original_answer_index]
    
    # Add "I don't know" as option F
    example["choices"]["label"].append("F")
    example["choices"]["text"].append("i don't know")
    
    # Shuffle only the text options, keeping labels A-F in order
    texts = example["choices"]["text"].copy()
    random.shuffle(texts)
    example["choices"]["text"] = texts
    
    # Find where the correct answer ended up after shuffling
    new_correct_index = texts.index(correct_answer_text)
    example["answerKey"] = example["choices"]["label"][new_correct_index]
    
    # Find where "I don't know" ended up after shuffling
    idk_index = texts.index("i don't know")
    example["idk_label"] = example["choices"]["label"][idk_index]
    
    return example

def generate_dpo_dataset(dataset, model, tokenizer):
    """Generate a preference dataset for DPO training based on a multi-answer QA dataset."""
    results = []
    
    for example in tqdm(dataset["train"], desc="Generating DPO dataset"):
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
                do_sample=True
            )
        
        # Get the most probable token among the possible choices (A-F)
        logits = outputs.scores[0][0]
        choice_labels = example["choices"]["label"]
        choice_token_ids = [tokenizer.convert_tokens_to_ids(label) for label in choice_labels]
        choice_logits = logits[choice_token_ids]
        predicted_index = torch.argmax(choice_logits).item()
        predicted_answer = choice_labels[predicted_index]
        # Format dataset for DPO (prompt, rejected, chosen)
        is_correct = predicted_answer.lower() == example["answerKey"].lower()

        ##### Alternative decoding approach #####
        # decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # predicted_answer = decoded_output.split("Answer:")[-1]

        # # Extract the predicted answer label
        # if predicted_answer != "":
        #     predicted_answer = predicted_answer.strip()[0]
        # else:
        #     print("Warning: Model did not generate an answer:", decoded_output)
        #     predicted_answer = example["idk_label"]  # Default to "I don't know" label
        
        # # Format dataset for DPO (prompt, rejected, chosen)
        # is_correct = predicted_answer.lower() == example["answerKey"].lower()

        results.append({
            "prompt": prompt,
            "rejected": example["idk_label"] if is_correct else predicted_answer,
            "chosen": example["answerKey"] if is_correct else example["idk_label"]
        })

    return results

def generate_dpo_dataset_batched(dataset, model, tokenizer, batch_size=8):
    """Generate a preference dataset for DPO training based on a multi-answer QA dataset."""
    
    # Create a text-generation pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_new_tokens=8
    )
    
    # Set padding side to left for better generation
    pipe.tokenizer.padding_side = "left"
    
    # Create prompts dataset
    def format_prompt(example):
        question = example["question"]
        prompt = f"Question: {question}\nChoices:\n"
        
        for label, text in zip(example["choices"]["label"], example["choices"]["text"]):
            prompt += f"{label}. {text}\n"
        
        prompt += "Answer:"
        example["prompt"] = prompt
        return example
    
    # Add prompts to dataset
    dataset_with_prompts = dataset["train"].map(format_prompt)
    
    # Run batched inference on the entire dataset
    print(f"Generating predictions with batch_size={batch_size}...")
    
    results = []
    prompts = dataset_with_prompts["prompt"]
    
    # Use pipeline with batching
    with torch.no_grad():
        outputs = pipe(prompts, batch_size=batch_size, return_full_text=False)
    
    # Process results
    for output, example in tqdm(zip(outputs, dataset_with_prompts), 
                                 total=len(dataset_with_prompts), 
                                 desc="Processing results"):
        predicted_answer = output[0]["generated_text"].strip()
        
        # Extract the predicted answer label
        if predicted_answer != "":
            predicted_answer = predicted_answer[0]
        else:
            print("Warning: Model did not generate an answer")
            predicted_answer = example["idk_label"]
        
        # Format dataset for DPO (prompt, rejected, chosen)
        is_correct = predicted_answer.lower() == example["answerKey"].lower()
        
        results.append({
            "prompt": example["prompt"],
            "rejected": example["idk_label"] if is_correct else predicted_answer,
            "chosen": example["answerKey"] if is_correct else example["idk_label"]
        })
    
    return results
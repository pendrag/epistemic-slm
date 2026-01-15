import os
import random
import torch
from datasets import load_dataset
from torch.functional import F
from transformers import AutoModelForCausalLM, AutoTokenizer
from itertools import permutations

from utils import load_config

def remove_choices(example):
    """Randomly remove choices so there are only two choices left. Correct answer CANNOT be removed."""
    choices = example['choices']['text']
    labels = example['choices']['label']
    answerKey = example['answerKey']
    answerText = example['choices']['text'][example['choices']['label'].index(answerKey)]

    choices.remove(answerText)
    labels.remove(answerKey)

    # Randomly select one choice and remove the rest
    choice_to_keep = random.choice(choices)

    example["choices"]["text"] = [choice_to_keep, answerText]
    example["choices"]["label"] = ["A", "B"]
    example["answerKey"] = "B"

    return example

def add_i_dont_know(example):
    example['choices']['text'].append("i don't know")
    IDK_label = "C" if len(example['choices']['label']) == 2 else "F"
    example['choices']['label'].append(IDK_label) # Set to the last letter in the choices
    return example

def permute_choices_and_update_answer(batch):
    all_ids = []
    all_questions = []
    all_question_concepts = []
    all_choices_labels = []
    all_choices_texts = []
    all_answer_keys = []

    for i in range(len(batch['id'])):
        choices_dict = batch['choices'][i]
        choices_text_list = choices_dict['text']
        labels = choices_dict['label']
        correct_label = batch['answerKey'][i]
        correct_answer = choices_text_list[labels.index(correct_label)]

        perms = list(permutations(choices_text_list))
        for perm in perms:
            new_answer_key = labels[perm.index(correct_answer)]

            all_ids.append(batch['id'][i])
            all_questions.append(batch['question'][i])
            all_question_concepts.append(batch['question_concept'][i])
            all_choices_labels.append(labels)        # length matches label count (always 3)
            all_choices_texts.append(list(perm))
            all_answer_keys.append(new_answer_key)

    return {
        'id': all_ids,
        'question': all_questions,
        'question_concept': all_question_concepts,
        'choices_label': all_choices_labels,
        'choices_text': all_choices_texts,
        'answerKey': all_answer_keys
    }

def format_prompt_chat_template(example):
    question = example['question']
    choices = example['choices_text']
    labels = example['choices_label']
    # TODO (optional): Include detailed instructions about the task in the prompt
    prompt = f"Question: {question}\nChoices:\n"
    for label, choice in zip(labels, choices):
        prompt += f"{label}. {choice}\n"
    prompt += "Answer:\n"    # The \n at the end is very important! Space or nothing at all underperforms notably
    return {"prompt": prompt}

def get_choice_probs(batch, label2id, model, tokenizer):
    SYSTEM = {
        "role": "system",
        "content": "You are a helpful assistant that answers multianwer questions by returning ONLY the letter associated with your choice. If you don't know the answer or you are not completely sure about it, you should choose the 'i don't know' answer."
    }
    messages = [[SYSTEM, {"role": "user", "content": prompt}] for prompt in batch['prompt']]
    model_inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        padding=True,
        return_dict=True
    ).to(model.device)

    # Tokenize the prompts (already batched)
    # model_inputs = tokenizer(batch['prompt'], return_tensors="pt", padding=True).to(model.device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=1, do_sample=False, return_dict_in_generate=True, output_scores=True)

    # Calculate all label probabilities
    batch_probs = F.softmax(generated_ids.scores[0], dim=-1)  # [batch_size, vocab_size]
    
    # We split top_10 into two separate lists (tokens and values)
    top_10_tokens_list = []
    top_10_values_list = []
    choice_probs = []

    for probs in batch_probs:
        topk = torch.topk(probs, k=10)
        tokens = [tokenizer.decode([idx]) for idx in topk.indices.tolist()]
        values = topk.values.tolist()
        
        top_10_tokens_list.append(tokens)
        top_10_values_list.append(values)

        # Get choice probabilities
        
        choice_probs.append({label: probs[id].item() for label, id in label2id.items()})
    
    # Store them
    batch['top_10_tokens'] = top_10_tokens_list
    batch['top_10_probs'] = top_10_values_list
    batch['choice_probs'] = choice_probs

    # Get most probable choice from choice_probs
    batch['prediction_label'] = [max(probs, key=probs.get) for probs in choice_probs]
    batch['prediction_text'] = [batch['choices_text'][i][batch['choices_label'][i].index(pred)] for i, pred in enumerate(batch['prediction_label'])]
    return batch

def calculate_average_prediction_confidence(batch):
    choice_probs = batch['choice_probs']
    prediction_probs = [choice_probs[i][pred] for i, pred in enumerate(batch['prediction_label'])]
    batch['averaged_prediction_confidence'] = [sum(prediction_probs)/len(prediction_probs)] * 6
    return batch

def main(cfg_path):
    cfg = load_config(cfg_path)

    if cfg['eval_dpo_model']:
        model_path = cfg['dpo_model']
        model_name = model_path + "/checkpoint-3800"
    else:
        model_path = cfg['model_name']
        model_name = model_path
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left") # Without padding_side=left the outputs logits differ when batch encoded
    tokenizer.pad_token = tokenizer.eos_token # Most LLMs don't have a pad token by default   

    dataset = load_dataset(cfg['base_ds'])
    dataset.pop('test')

    label2id = {label: tokenizer(label).input_ids[-1] for label in ["A", "B", "C"]}

    for ds_name, split in dataset.items():
        # If folder 'ds_name' does not exist, create it
        if not os.path.exists(ds_name):
            os.makedirs(ds_name)

        # ds = split.select(range(10)) # For testing purposes
        ds = split
        ds = ds.map(remove_choices, load_from_cache_file=False)
        ds = ds.map(add_i_dont_know)
        ds = ds.map(permute_choices_and_update_answer, batched=True, batch_size=6, load_from_cache_file=False, remove_columns=ds.column_names)
        ds = ds.map(format_prompt_chat_template, load_from_cache_file=False)
        
        ds = ds.map(
            get_choice_probs,
            batched=True,
            batch_size=6,
            fn_kwargs={"label2id": label2id, "model": model, "tokenizer": tokenizer},
            load_from_cache_file=False
        )
        ds = ds.map(calculate_average_prediction_confidence, batched=True, batch_size=6, load_from_cache_file=False) # BS is set to the number of permutations

        if not os.path.exists(f"{cfg['output_ds']}/{model_path.split('/')[-1]}"):
            os.makedirs(f"{cfg['output_ds']}/{model_path.split('/')[-1]}")
        ds.save_to_disk(f"{cfg['output_ds']}/{model_path.split('/')[-1]}/{ds_name}/evaluated_ds")
        ds.to_json(f"{cfg['output_ds']}/{model_path.split('/')[-1]}/{ds_name}/evaluated_ds.json")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        sys.exit(1)
    config_path = sys.argv[1]
    main(config_path)
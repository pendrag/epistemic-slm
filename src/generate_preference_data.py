import os
import pandas as pd
from datasets import load_from_disk
from utils import (
    load_config,
)

def generate_preference_dataset(batch, permutations=6):
    """Generate a preference dataset from a given dataset."""

    # Get gold text
    answerKey_id = batch["choices_label"][0].index(batch["answerKey"][0])
    answer_text = batch["choices_text"][0][answerKey_id]

    answer_labels = []
    idk_labels = []
    for i in range(permutations):
        # Get answer label
        answer_label = batch["choices_label"][i][batch["choices_text"][i].index(answer_text)]
        answer_labels.append(answer_label)

        # Get IDK label
        IDK_label = batch["choices_label"][i][batch["choices_text"][i].index("i don't know")]
        idk_labels.append(IDK_label)

    predictions = batch["prediction_text"]
    # If the model is consistent (i.e. all predicted answers are identical)
    if len(set(predictions)) == 1:
        chosen = batch["prediction_label"]
        # If all predictions are "I don't know", set correct labels as rejected
        if batch["prediction_label"] == idk_labels:
            rejected = answer_labels
        else:
            rejected = idk_labels
    
    # If NOT all predictions are equal (i.e. the model is not consistent)
    else:
        chosen = idk_labels
        rejected = []
        for idk, pred, ans in zip(idk_labels, batch['prediction_label'], answer_labels):
            if pred == idk:
                # If the prediction is "I don't know", reject the answer label
                rejected.append(ans)
            else:
                rejected.append(pred)

    return {
        "prompt": batch["prompt"],
        "chosen": chosen,
        "rejected": rejected
    }

def main(cfg_path):
    cfg = load_config(cfg_path)

    # Load dataset
    train_ds = load_from_disk(f"{cfg['output_ds']}/train/evaluated_ds")
    val_ds = load_from_disk(f"{cfg['output_ds']}/validation/evaluated_ds")

    # Generate preference datasets
    train_preference_ds = train_ds.map(generate_preference_dataset, batched=True, batch_size=6, remove_columns=train_ds.column_names)
    val_preference_ds = val_ds.map(generate_preference_dataset, batched=True, batch_size=6, remove_columns=val_ds.column_names)

    # Save datasets
    if not os.path.exists(f"{cfg['output_ds']}"):
        os.makedirs(f"{cfg['output_ds']}")
    train_preference_ds.save_to_disk(f"{cfg['output_ds']}/train/preference_dataset")
    train_preference_ds.to_json(f"{cfg['output_ds']}/train/preference_dataset.json")
    val_preference_ds.save_to_disk(f"{cfg['output_ds']}/validation/preference_dataset")
    val_preference_ds.to_json(f"{cfg['output_ds']}/validation/preference_dataset.json")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        sys.exit(1)
    config_path = sys.argv[1]
    main(config_path)
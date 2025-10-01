# Epistemic Humility Experiment for Generative Transformer Models

The code in this folder implements an experiment to teach generative transformer models (such as DialoGPT or GPT-2) to exhibit epistemic humility—that is, to respond with "I don't know" when they lack sufficient evidence to answer a question correctly. The main script, epistemic_experiment.py, automates the process of:

1. Loading a Q&A dataset (e.g., SQuAD)
1. Evaluating the model's initial answers
1. Creating a fine-tuning dataset where incorrect answers are replaced with "I don't know"
1. Fine-tuning the model on this new dataset
1. Comparing the original and fine-tuned models' performance, especially their ability to admit uncertainty

Configuration is managed via YAML files like config.yaml, and results are saved and visualized for analysis. The goal is to improve the model's ability to recognize and communicate its own uncertainty.

**How to launch the experiment**

1. Install dependencies: 
```bash
pip install -r requirements.txt
```

2. Configure the experiment by editing ``epistemic_experiment.py`` as needed.

3. Run the main script:
```bash
python epistemic_experiment.py 
```
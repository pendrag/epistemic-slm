#!/usr/bin/env python3
"""
Epistemic Humility Experiment for Generative Transformer Models
Teaches models to respond "I don't know" when they lack sufficient evidence.
"""

import os
import json
import torch
import pandas as pd
import numpy as np
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Any
import argparse
import logging
from tqdm import tqdm
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EpistemicHumilityExperiment:
    """Main class for running the epistemic humility experiment."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config['model']['name']
        self.dataset_name = config['dataset']['name']
        self.output_dir = config['paths']['output_dir']
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.original_model = None
        self.finetuned_model = None
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_qa_dataset(self) -> Tuple[Dataset, Dataset]:
        """
        Step 1 & 2: Load and split Q&A dataset
        """
        logger.info(f"Loading dataset: {self.dataset_name}")
        
        if self.dataset_name == "squad":
            dataset = load_dataset("squad")
            train_data = dataset['train']
            test_data = dataset['validation']
            
            # Convert to simple Q&A format
            train_qa = []
            test_qa = []
            
            for item in train_data:
                if item['answers']['text']:  # Only include items with answers
                    train_qa.append({
                        'question': item['question'],
                        'answer': item['answers']['text'][0]
                    })
                    
            for item in test_data:
                if item['answers']['text']:
                    test_qa.append({
                        'question': item['question'],
                        'answer': item['answers']['text'][0]
                    })
                    
        elif self.dataset_name == "natural_questions":
            dataset = load_dataset("natural_questions")
            # Process natural questions format
            train_qa = []
            test_qa = []
            
            for i, item in enumerate(dataset['train']):
                if i >= 10000:  # Limit for computational efficiency
                    break
                if item['annotations']['short_answers']:
                    train_qa.append({
                        'question': item['question']['text'],
                        'answer': item['annotations']['short_answers'][0]['text']
                    })
                    
            for i, item in enumerate(dataset['validation']):
                if i >= 2000:
                    break
                if item['annotations']['short_answers']:
                    test_qa.append({
                        'question': item['question']['text'],
                        'answer': item['annotations']['short_answers'][0]['text']
                    })
        else:
            # Custom dataset loading
            raise NotImplementedError(f"Dataset {self.dataset_name} not implemented")
            
        # Limit dataset size for experimentation
        max_train = min(len(train_qa), self.config['dataset']['max_train_samples'])
        max_test = min(len(test_qa), self.config['dataset']['max_test_samples'])
        
        train_dataset = Dataset.from_list(train_qa[:max_train])
        test_dataset = Dataset.from_list(test_qa[:max_test])
        
        logger.info(f"Loaded {len(train_dataset)} training and {len(test_dataset)} test samples")
        return train_dataset, test_dataset
    
    def load_model(self, model_path: str = None) -> AutoModelForCausalLM:
        """Load the generative model"""
        path = model_path or self.model_name
        logger.info(f"Loading model: {path}")
        
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        return model
    
    def generate_responses(self, model: AutoModelForCausalLM, dataset: Dataset) -> List[str]:
        """
        Step 3: Generate responses from the model for all questions
        """
        logger.info("Generating responses from model...")
        
        responses = []
        device = next(model.parameters()).device
        
        for item in tqdm(dataset, desc="Generating responses"):
            question = item['question']
            
            # Format prompt
            prompt = f"Question: {question}\nAnswer:"
            
            # Tokenize
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            ).to(device)
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = full_response[len(prompt):].strip()
            responses.append(response)
            
        return responses
    
    def evaluate_responses(self, responses: List[str], ground_truth: List[str]) -> List[bool]:
        """
        Evaluate if responses are correct (simple string matching for now)
        You might want to use more sophisticated evaluation metrics
        """
        logger.info("Evaluating response correctness...")
        
        correct_flags = []
        
        for response, truth in zip(responses, ground_truth):
            # Simple evaluation - check if key terms from ground truth appear in response
            response_lower = response.lower()
            truth_lower = truth.lower()
            
            # Basic heuristic: if any word from ground truth (>2 chars) appears in response
            truth_words = [word for word in truth_lower.split() if len(word) > 2]
            is_correct = any(word in response_lower for word in truth_words) if truth_words else False
            
            correct_flags.append(is_correct)
            
        accuracy = sum(correct_flags) / len(correct_flags)
        logger.info(f"Original model accuracy: {accuracy:.3f}")
        
        return correct_flags
    
    def create_instruction_dataset(self, 
                                 train_dataset: Dataset, 
                                 responses: List[str], 
                                 correct_flags: List[bool]) -> Dataset:
        """
        Step 4: Create instruction dataset for fine-tuning
        """
        logger.info("Creating instruction dataset...")
        
        instruction_data = []
        
        for i, (item, response, is_correct) in enumerate(zip(train_dataset, responses, correct_flags)):
            question = item['question']
            ground_truth = item['answer']
            
            if is_correct:
                # Model was correct - use ground truth answer
                target_answer = ground_truth
            else:
                # Model was wrong - teach to say "I don't know"
                target_answer = "I don't know"
            
            instruction_data.append({
                'instruction': question,
                'input': '',
                'output': target_answer,
                'original_response': response,
                'was_correct': is_correct
            })
        
        instruction_dataset = Dataset.from_list(instruction_data)
        
        # Save dataset
        instruction_dataset.save_to_disk(os.path.join(self.output_dir, 'instruction_dataset'))
        
        # Log statistics
        wrong_count = sum(1 for flag in correct_flags if not flag)
        logger.info(f"Created {len(instruction_dataset)} instruction samples")
        logger.info(f"Teaching 'I don't know' for {wrong_count} wrong answers")
        logger.info(f"Reinforcing correct answers for {len(instruction_dataset) - wrong_count} samples")
        
        return instruction_dataset
    
    def prepare_training_data(self, instruction_dataset: Dataset) -> Dataset:
        """Prepare data for training with proper formatting"""
        
        def format_example(example):
            prompt = f"Question: {example['instruction']}\nAnswer: {example['output']}"
            return {'text': prompt}
        
        formatted_dataset = instruction_dataset.map(format_example)
        return formatted_dataset
    
    def tokenize_function(self, examples):
        """Tokenize examples for training"""
        return self.tokenizer(
            examples['text'],
            truncation=True,
            padding=True,
            max_length=512
        )
    
    def fine_tune_model(self, instruction_dataset: Dataset) -> str:
        """
        Step 5: Fine-tune the model using the instruction dataset
        """
        logger.info("Starting fine-tuning...")
        
        # Prepare training data
        formatted_dataset = self.prepare_training_data(instruction_dataset)
        tokenized_dataset = formatted_dataset.map(self.tokenize_function, batched=True)
        
        # Load fresh model for fine-tuning
        model = self.load_model()
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=os.path.join(self.output_dir, 'finetuned_model'),
            overwrite_output_dir=True,
            num_train_epochs=self.config['training']['epochs'],
            per_device_train_batch_size=self.config['training']['batch_size'],
            gradient_accumulation_steps=self.config['training']['gradient_accumulation_steps'],
            warmup_steps=self.config['training']['warmup_steps'],
            learning_rate=self.config['training']['learning_rate'],
            fp16=torch.cuda.is_available(),
            logging_steps=50,
            save_strategy="epoch",
            evaluation_strategy="no",
            save_total_limit=2,
            prediction_loss_only=True,
            remove_unused_columns=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        
        # Train
        trainer.train()
        
        # Save model
        finetuned_path = os.path.join(self.output_dir, 'finetuned_model')
        trainer.save_model(finetuned_path)
        
        return finetuned_path
    
    def compare_models(self, test_dataset: Dataset, finetuned_model_path: str) -> Dict[str, Any]:
        """
        Step 6: Compare original and fine-tuned models
        """
        logger.info("Comparing original and fine-tuned models...")
        
        # Load models
        original_model = self.load_model()
        finetuned_model = self.load_model(finetuned_model_path)
        
        # Generate responses from both models
        logger.info("Generating responses from original model...")
        original_responses = self.generate_responses(original_model, test_dataset)
        
        logger.info("Generating responses from fine-tuned model...")
        finetuned_responses = self.generate_responses(finetuned_model, test_dataset)
        
        # Evaluate both
        ground_truth = [item['answer'] for item in test_dataset]
        original_correct = self.evaluate_responses(original_responses, ground_truth)
        finetuned_correct = self.evaluate_responses(finetuned_responses, ground_truth)
        
        # Count "I don't know" responses
        original_idk_count = sum(1 for r in original_responses if "don't know" in r.lower())
        finetuned_idk_count = sum(1 for r in finetuned_responses if "don't know" in r.lower())
        
        # Analyze where fine-tuned model says "I don't know"
        finetuned_idk_on_wrong = 0
        finetuned_idk_on_correct = 0
        
        for i, (orig_correct, ft_response) in enumerate(zip(original_correct, finetuned_responses)):
            if "don't know" in ft_response.lower():
                if not orig_correct:  # Original was wrong
                    finetuned_idk_on_wrong += 1
                else:  # Original was correct
                    finetuned_idk_on_correct += 1
        
        results = {
            'original_accuracy': sum(original_correct) / len(original_correct),
            'finetuned_accuracy': sum(finetuned_correct) / len(finetuned_correct),
            'original_idk_count': original_idk_count,
            'finetuned_idk_count': finetuned_idk_count,
            'finetuned_idk_on_wrong': finetuned_idk_on_wrong,
            'finetuned_idk_on_correct': finetuned_idk_on_correct,
            'total_originally_wrong': sum(1 for c in original_correct if not c),
            'responses': {
                'original': original_responses,
                'finetuned': finetuned_responses,
                'ground_truth': ground_truth,
                'original_correct': original_correct,
                'finetuned_correct': finetuned_correct
            }
        }
        
        return results
    
    def analyze_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 7: Analyze results and extract conclusions
        """
        logger.info("Analyzing results...")
        
        # Calculate key metrics
        total_wrong_original = results['total_originally_wrong']
        idk_on_wrong = results['finetuned_idk_on_wrong']
        idk_on_correct = results['finetuned_idk_on_correct']
        
        # Epistemic humility metrics
        humility_precision = idk_on_wrong / (idk_on_wrong + idk_on_correct) if (idk_on_wrong + idk_on_correct) > 0 else 0
        humility_recall = idk_on_wrong / total_wrong_original if total_wrong_original > 0 else 0
        humility_f1 = 2 * (humility_precision * humility_recall) / (humility_precision + humility_recall) if (humility_precision + humility_recall) > 0 else 0
        
        analysis = {
            'epistemic_humility_precision': humility_precision,
            'epistemic_humility_recall': humility_recall,
            'epistemic_humility_f1': humility_f1,
            'accuracy_change': results['finetuned_accuracy'] - results['original_accuracy'],
            'idk_increase': results['finetuned_idk_count'] - results['original_idk_count'],
            'conclusions': []
        }
        
        # Generate conclusions
        if humility_recall > 0.5:
            analysis['conclusions'].append("Model successfully learned to express uncertainty for many questions it originally answered incorrectly.")
        else:
            analysis['conclusions'].append("Model showed limited improvement in expressing uncertainty for incorrect answers.")
            
        if humility_precision > 0.7:
            analysis['conclusions'].append("Model demonstrates good precision in epistemic humility - mostly says 'I don't know' appropriately.")
        else:
            analysis['conclusions'].append("Model may be over-generalizing 'I don't know' responses to questions it could answer correctly.")
            
        if analysis['accuracy_change'] > 0:
            analysis['conclusions'].append("Fine-tuning improved overall accuracy.")
        elif analysis['accuracy_change'] < -0.05:
            analysis['conclusions'].append("Fine-tuning significantly reduced overall accuracy - may need adjustment.")
        
        return analysis
    
    def create_visualizations(self, results: Dict[str, Any], analysis: Dict[str, Any]):
        """Create visualizations of the results"""
        logger.info("Creating visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Accuracy comparison
        accuracies = [results['original_accuracy'], results['finetuned_accuracy']]
        axes[0, 0].bar(['Original', 'Fine-tuned'], accuracies, color=['blue', 'orange'])
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim(0, 1)
        
        # 2. "I don't know" response counts
        idk_counts = [results['original_idk_count'], results['finetuned_idk_count']]
        axes[0, 1].bar(['Original', 'Fine-tuned'], idk_counts, color=['blue', 'orange'])
        axes[0, 1].set_title('"I Don\'t Know" Response Counts')
        axes[0, 1].set_ylabel('Count')
        
        # 3. Epistemic humility breakdown
        categories = ['IDK on Wrong', 'IDK on Correct']
        values = [results['finetuned_idk_on_wrong'], results['finetuned_idk_on_correct']]
        axes[1, 0].bar(categories, values, color=['green', 'red'])
        axes[1, 0].set_title('Fine-tuned Model: "I Don\'t Know" Usage')
        axes[1, 0].set_ylabel('Count')
        
        # 4. Epistemic humility metrics
        metrics = ['Precision', 'Recall', 'F1']
        metric_values = [
            analysis['epistemic_humility_precision'],
            analysis['epistemic_humility_recall'],
            analysis['epistemic_humility_f1']
        ]
        axes[1, 1].bar(metrics, metric_values, color=['purple', 'brown', 'pink'])
        axes[1, 1].set_title('Epistemic Humility Metrics')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'results_visualization.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self, results: Dict[str, Any], analysis: Dict[str, Any]):
        """Save all results to files"""
        logger.info("Saving results...")
        
        # Remove large response data for summary
        summary_results = {k: v for k, v in results.items() if k != 'responses'}
        
        # Save summary
        summary = {
            'results': summary_results,
            'analysis': analysis,
            'config': self.config
        }
        
        with open(os.path.join(self.output_dir, 'experiment_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save detailed results
        with open(os.path.join(self.output_dir, 'detailed_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create human-readable report
        self.create_report(results, analysis)
    
    def create_report(self, results: Dict[str, Any], analysis: Dict[str, Any]):
        """Create a human-readable report"""
        report = f"""
# Epistemic Humility Experiment Report

## Experiment Configuration
- Model: {self.config['model']['name']}
- Dataset: {self.config['dataset']['name']}
- Training samples: {self.config['dataset']['max_train_samples']}
- Test samples: {self.config['dataset']['max_test_samples']}

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
        
        for conclusion in analysis['conclusions']:
            report += f"- {conclusion}\n"
        
        with open(os.path.join(self.output_dir, 'experiment_report.md'), 'w') as f:
            f.write(report)
    
    def run_full_experiment(self):
        """Run the complete experiment pipeline"""
        logger.info("Starting full epistemic humility experiment...")
        
        # Step 1-2: Load dataset
        train_dataset, test_dataset = self.load_qa_dataset()
        
        # Step 3: Generate responses from original model
        original_model = self.load_model()
        train_responses = self.generate_responses(original_model, train_dataset)
        
        # Step 4: Evaluate and create instruction dataset
        ground_truth = [item['answer'] for item in train_dataset]
        correct_flags = self.evaluate_responses(train_responses, ground_truth)
        instruction_dataset = self.create_instruction_dataset(train_dataset, train_responses, correct_flags)
        
        # Step 5: Fine-tune model
        finetuned_model_path = self.fine_tune_model(instruction_dataset)
        
        # Step 6: Compare models
        results = self.compare_models(test_dataset, finetuned_model_path)
        
        # Step 7: Analyze results
        analysis = self.analyze_results(results)
        
        # Create visualizations and save results
        self.create_visualizations(results, analysis)
        self.save_results(results, analysis)
        
        logger.info("Experiment completed successfully!")
        logger.info(f"Results saved to: {self.output_dir}")
        
        return results, analysis

def create_default_config():
    """Create default configuration"""
    return {
        'model': {
            'name': 'microsoft/DialoGPT-small'  # Small model for testing
        },
        'dataset': {
            'name': 'squad',
            'max_train_samples': 1000,
            'max_test_samples': 200
        },
        'training': {
            'epochs': 3,
            'batch_size': 4,
            'gradient_accumulation_steps': 2,
            'learning_rate': 5e-5,
            'warmup_steps': 100
        },
        'paths': {
            'output_dir': './epistemic_experiment_results'
        }
    }

def main():
    parser = argparse.ArgumentParser(description='Run Epistemic Humility Experiment')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--model', type=str, default='microsoft/DialoGPT-small', help='Model name')
    parser.add_argument('--dataset', type=str, default='squad', help='Dataset name')
    parser.add_argument('--output_dir', type=str, default='./epistemic_experiment_results', help='Output directory')
    
    args = parser.parse_args()
    
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = create_default_config()
        config['model']['name'] = args.model
        config['dataset']['name'] = args.dataset
        config['paths']['output_dir'] = args.output_dir
    
    # Run experiment
    experiment = EpistemicHumilityExperiment(config)
    results, analysis = experiment.run_full_experiment()
    
    # Print key results
    print("\n" + "="*50)
    print("EXPERIMENT COMPLETED")
    print("="*50)
    print(f"Epistemic Humility F1 Score: {analysis['epistemic_humility_f1']:.3f}")
    print(f"Accuracy Change: {analysis['accuracy_change']:.3f}")
    print(f"Results saved to: {config['paths']['output_dir']}")
    
if __name__ == "__main__":
    main()

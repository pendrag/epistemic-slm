# axolotl_integration.py - Integration with Axolotl for advanced training
import os
import yaml
import json
import subprocess
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class AxolotlIntegration:
    """Integration class for using Axolotl for fine-tuning"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.axolotl_config_path = config.get('axolotl', {}).get('config_path', './axolotl_config.yaml')
        
    def create_axolotl_config(self, instruction_dataset_path: str, model_name: str, output_dir: str):
        """Create Axolotl configuration file"""
        
        axolotl_config = {
            'base_model': model_name,
            'model_type': 'AutoModelForCausalLM',
            'tokenizer_type': 'AutoTokenizer',
            
            'load_in_8bit': False,
            'load_in_4bit': True,
            'strict': False,
            
            # Dataset configuration
            'datasets': [
                {
                    'path': instruction_dataset_path,
                    'type': 'alpaca',
                    'conversation': {
                        'field': 'text'
                    }
                }
            ],
            
            # Training configuration
            'dataset_prepared_path': f'{output_dir}/prepared_dataset',
            'val_set_size': 0.05,
            'output_dir': output_dir,
            
            'adapter': 'lora',
            'lora_model_dir': f'{output_dir}/lora',
            
            'sequence_len': 512,
            'sample_packing': True,
            'pad_to_sequence_len': True,
            
            'lora_r': 32,
            'lora_alpha': 16,
            'lora_dropout': 0.05,
            'lora_target_modules': ['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'down_proj', 'up_proj'],
            
            'wandb_project': 'epistemic_humility',
            'wandb_entity': '',
            'wandb_watch': '',
            'wandb_name': 'epistemic_experiment',
            'wandb_log_model': '',
            
            # Training parameters
            'gradient_accumulation_steps': self.config['training']['gradient_accumulation_steps'],
            'micro_batch_size': self.config['training']['batch_size'],
            'num_epochs': self.config['training']['epochs'],
            'optimizer': 'adamw_bnb_8bit',
            'lr_scheduler': 'cosine',
            'learning_rate': self.config['training']['learning_rate'],
            'train_on_inputs': False,
            'group_by_length': False,
            'bf16': True,
            'fp16': False,
            'tf32': False,
            
            'gradient_checkpointing': True,
            'early_stopping_patience': '',
            'resume_from_checkpoint': '',
            'local_rank': '',
            
            'logging_steps': 1,
            'xformers_attention': '',
            'flash_attention': True,
            
            'loss_watchdog_threshold': 5.0,
            'loss_watchdog_patience': 3,
            
            'warmup_steps': self.config['training']['warmup_steps'],
            'evals_per_epoch': 4,
            'eval_table_size': '',
            'eval_max_new_tokens': 128,
            'saves_per_epoch': 1,
            'debug': '',
            'deepspeed': '',
            'weight_decay': 0.0,
            'fsdp': '',
            'fsdp_config': '',
            'special_tokens': {},
            'tokens': []
        }
        
        # Save Axolotl config
        with open(self.axolotl_config_path, 'w') as f:
            yaml.dump(axolotl_config, f, default_flow_style=False)
            
        logger.info(f"Axolotl config saved to: {self.axolotl_config_path}")
        return self.axolotl_config_path
    
    def prepare_data_for_axolotl(self, instruction_dataset_path: str) -> str:
        """Convert instruction dataset to Axolotl format"""
        from datasets import load_from_disk
        
        dataset = load_from_disk(instruction_dataset_path)
        
        # Convert to Alpaca format
        alpaca_data = []
        for item in dataset:
            alpaca_item = {
                'instruction': item['instruction'],
                'input': item.get('input', ''),
                'output': item['output']
            }
            alpaca_data.append(alpaca_item)
        
        # Save as JSON
        alpaca_path = instruction_dataset_path + '_alpaca.json'
        with open(alpaca_path, 'w') as f:
            json.dump(alpaca_data, f, indent=2)
            
        return alpaca_path
    
    def run_axolotl_training(self) -> str:
        """Run Axolotl training"""
        logger.info("Starting Axolotl training...")
        
        try:
            # Run Axolotl
            cmd = f"python -m axolotl.cli.train {self.axolotl_config_path}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Axolotl training failed: {result.stderr}")
                raise RuntimeError(f"Axolotl training failed: {result.stderr}")
            
            logger.info("Axolotl training completed successfully")
            
            # Return path to trained model
            with open(self.axolotl_config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            return config['output_dir']
            
        except Exception as e:
            logger.error(f"Error running Axolotl: {e}")
            raise

# enhanced_evaluation.py - Advanced evaluation methods
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import numpy as np
from typing import List, Tuple, Dict
import re

class AdvancedEvaluator:
    """Advanced evaluation methods for response quality"""
    
    def __init__(self, use_semantic_similarity: bool = True):
        self.use_semantic_similarity = use_semantic_similarity
        
        if use_semantic_similarity:
            try:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            except:
                logger.warning("Could not load sentence transformer, falling back to basic evaluation")
                self.use_semantic_similarity = False
    
    def evaluate_response_quality(self, response: str, ground_truth: str, threshold: float = 0.5) -> Dict[str, float]:
        """Evaluate response quality using multiple metrics"""
        
        results = {
            'exact_match': 0.0,
            'token_overlap': 0.0,
            'semantic_similarity': 0.0,
            'is_correct': False
        }
        
        # Exact match
        if response.lower().strip() == ground_truth.lower().strip():
            results['exact_match'] = 1.0
        
        # Token overlap (F1)
        response_tokens = set(self._tokenize(response.lower()))
        truth_tokens = set(self._tokenize(ground_truth.lower()))
        
        if len(response_tokens) > 0 and len(truth_tokens) > 0:
            intersection = response_tokens & truth_tokens
            precision = len(intersection) / len(response_tokens)
            recall = len(intersection) / len(truth_tokens)
            results['token_overlap'] = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Semantic similarity
        if self.use_semantic_similarity:
            try:
                embeddings = self.sentence_model.encode([response, ground_truth])
                similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
                results['semantic_similarity'] = max(0.0, similarity)
            except:
                results['semantic_similarity'] = 0.0
        
        # Overall correctness decision
        if self.use_semantic_similarity:
            results['is_correct'] = results['semantic_similarity'] > threshold
        else:
            results['is_correct'] = results['token_overlap'] > threshold
        
        return results
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        return re.findall(r'\b\w+\b', text.lower())
    
    def evaluate_epistemic_humility(self, responses: List[str], original_correct: List[bool]) -> Dict[str, float]:
        """Evaluate epistemic humility specifically"""
        
        idk_responses = []
        for response in responses:
            is_idk = any(phrase in response.lower() for phrase in [
                "don't know", "not sure", "uncertain", "unsure", 
                "i'm not certain", "i don't have enough information",
                "cannot determine", "unable to answer"
            ])
            idk_responses.append(is_idk)
        
        # Calculate metrics
        tp = sum(1 for idk, correct in zip(idk_responses, original_correct) if idk and not correct)  # IDK on wrong
        fp = sum(1 for idk, correct in zip(idk_responses, original_correct) if idk and correct)     # IDK on correct
        fn = sum(1 for idk, correct in zip(idk_responses, original_correct) if not idk and not correct) # No IDK on wrong
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }

# experiment_runner.py - Main runner with all integrations
import argparse
import yaml
import logging
from pathlib import Path

class ExperimentRunner:
    """Main experiment runner with all integrations"""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.setup_logging()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file or create default"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            return self._create_default_config()
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration"""
        return {
            'model': {'name': 'microsoft/DialoGPT-small'},
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
            'evaluation': {
                'use_semantic_similarity': False,
                'similarity_threshold': 0.5
            },
            'paths': {
                'output_dir': './epistemic_experiment_results',
                'cache_dir': './cache'
            },
            'axolotl': {
                'use_axolotl': False,
                'config_path': './axolotl_config.yaml'
            }
        }
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(Path(self.config['paths']['output_dir']) / 'experiment.log'),
                logging.StreamHandler()
            ]
        )
    
    def run_experiment(self):
        """Run the complete experiment with all features"""
        logger.info("Starting enhanced epistemic humility experiment...")
        
        # Create enhanced experiment class
        class EnhancedEpistemicExperiment(EpistemicHumilityExperiment):
            def __init__(self, config, evaluator=None, axolotl_integration=None):
                super().__init__(config)
                self.evaluator = evaluator or AdvancedEvaluator(
                    config.get('evaluation', {}).get('use_semantic_similarity', False)
                )
                self.axolotl = axolotl_integration
            
            def evaluate_responses(self, responses: List[str], ground_truth: List[str]) -> List[bool]:
                """Enhanced response evaluation"""
                logger.info("Evaluating responses with advanced metrics...")
                
                correct_flags = []
                detailed_results = []
                
                threshold = self.config.get('evaluation', {}).get('similarity_threshold', 0.5)
                
                for response, truth in zip(responses, ground_truth):
                    eval_result = self.evaluator.evaluate_response_quality(response, truth, threshold)
                    correct_flags.append(eval_result['is_correct'])
                    detailed_results.append(eval_result)
                
                # Log detailed statistics
                avg_token_overlap = np.mean([r['token_overlap'] for r in detailed_results])
                avg_semantic_sim = np.mean([r['semantic_similarity'] for r in detailed_results])
                
                logger.info(f"Average token overlap F1: {avg_token_overlap:.3f}")
                logger.info(f"Average semantic similarity: {avg_semantic_sim:.3f}")
                logger.info(f"Accuracy: {np.mean(correct_flags):.3f}")
                
                return correct_flags
            
            def fine_tune_model(self, instruction_dataset: Dataset) -> str:
                """Fine-tune with Axolotl if configured"""
                if self.config.get('axolotl', {}).get('use_axolotl', False) and self.axolotl:
                    logger.info("Using Axolotl for fine-tuning...")
                    
                    # Prepare data for Axolotl
                    dataset_path = instruction_dataset.save_to_disk(
                        os.path.join(self.output_dir, 'instruction_dataset')
                    )
                    alpaca_path = self.axolotl.prepare_data_for_axolotl(dataset_path)
                    
                    # Create Axolotl config
                    self.axolotl.create_axolotl_config(
                        alpaca_path, 
                        self.model_name, 
                        os.path.join(self.output_dir, 'axolotl_output')
                    )
                    
                    # Run training
                    return self.axolotl.run_axolotl_training()
                else:
                    # Use standard training
                    return super().fine_tune_model(instruction_dataset)
        
        # Initialize components
        evaluator = AdvancedEvaluator(
            self.config.get('evaluation', {}).get('use_semantic_similarity', False)
        )
        
        axolotl_integration = None
        if self.config.get('axolotl', {}).get('use_axolotl', False):
            axolotl_integration = AxolotlIntegration(self.config)
        
        # Run enhanced experiment
        experiment = EnhancedEpistemicExperiment(self.config, evaluator, axolotl_integration)
        results, analysis = experiment.run_full_experiment()
        
        return results, analysis

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Enhanced Epistemic Humility Experiment')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--use-axolotl', action='store_true', help='Use Axolotl for training')
    parser.add_argument('--use-semantic', action='store_true', help='Use semantic similarity evaluation')
    
    args = parser.parse_args()
    
    # Load or create config
    runner = ExperimentRunner(args.config)
    
    # Override config with command line args
    if args.use_axolotl:
        runner.config['axolotl']['use_axolotl'] = True
    if args.use_semantic:
        runner.config['evaluation']['use_semantic_similarity'] = True
    
    # Run experiment
    results, analysis = runner.run_experiment()
    
    # Print results
    print("\n" + "="*60)
    print("ENHANCED EXPERIMENT COMPLETED")
    print("="*60)
    print(f"Epistemic Humility F1 Score: {analysis['epistemic_humility_f1']:.3f}")
    print(f"Precision: {analysis['epistemic_humility_precision']:.3f}")
    print(f"Recall: {analysis['epistemic_humility_recall']:.3f}")
    print(f"Accuracy Change: {analysis['accuracy_change']:.3f}")
    print(f"Results saved to: {runner.config['paths']['output_dir']}")
    print("="*60)

if __name__ == "__main__":
    main()
            
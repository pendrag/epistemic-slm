#!/usr/bin/env python3
"""
Complete Epistemic Humility Experiment Runner
Run this script to execute the full experiment with sensible defaults.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def setup_experiment_environment():
    """Setup the experiment environment"""
    print("Setting up Epistemic Humility Experiment...")
    
    # Create directories
    directories = ['results', 'cache', 'logs', 'data']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/experiment.log'),
            logging.StreamHandler()
        ]
    )
    print("✓ Logging configured")

def create_default_config():
    """Create a default configuration for the experiment"""
    return {
        'model': {
            'name': 'microsoft/DialoGPT-small'  # Small model for quick testing
        },
        'dataset': {
            'name': 'squad',
            'max_train_samples': 500,  # Reduced for quick execution
            'max_test_samples': 100
        },
        'training': {
            'epochs': 2,  # Reduced epochs for quick testing
            'batch_size': 2,  # Small batch size for memory efficiency
            'gradient_accumulation_steps': 4,
            'learning_rate': 5e-5,
            'warmup_steps': 50
        },
        'evaluation': {
            'use_semantic_similarity': False,  # Disabled by default for simplicity
            'similarity_threshold': 0.5
        },
        'paths': {
            'output_dir': './results/epistemic_experiment',
            'cache_dir': './cache'
        },
        'axolotl': {
            'use_axolotl': False  # Disabled by default
        }
    }

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'torch', 'transformers', 'datasets', 'numpy', 
        'pandas', 'matplotlib', 'seaborn', 'scikit-learn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} (missing)")
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Please install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def run_quick_test():
    """Run a quick test experiment"""
    print("\n" + "="*50)
    print("RUNNING QUICK TEST EXPERIMENT")
    print("="*50)
    
    # Import after environment setup
    from epistemic_experiment import EpistemicHumilityExperiment
    
    # Quick test configuration
    config = {
        'model': {'name': 'microsoft/DialoGPT-small'},
        'dataset': {
            'name': 'squad',
            'max_train_samples': 50,  # Very small for testing
            'max_test_samples': 10
        },
        'training': {
            'epochs': 1,
            'batch_size': 1,
            'gradient_accumulation_steps': 2,
            'learning_rate': 5e-5,
            'warmup_steps': 5
        },
        'paths': {
            'output_dir': './results/quick_test'
        }
    }
    
    # Run experiment
    experiment = EpistemicHumilityExperiment(config)
    results, analysis = experiment.run_full_experiment()
    
    # Print results
    print("\n" + "="*40)
    print("QUICK TEST RESULTS")
    print("="*40)
    print(f"Epistemic Humility F1: {analysis.get('epistemic_humility_f1', 0):.3f}")
    print(f"Accuracy Change: {analysis.get('accuracy_change', 0):+.3f}")
    print(f"Results saved to: {config['paths']['output_dir']}")
    
    return results, analysis

def run_standard_experiment():
    """Run the standard experiment"""
    print("\n" + "="*50)
    print("RUNNING STANDARD EXPERIMENT")
    print("="*50)
    
    from epistemic_experiment import EpistemicHumilityExperiment
    
    config = create_default_config()
    
    print(f"Configuration:")
    print(f"- Model: {config['model']['name']}")
    print(f"- Dataset: {config['dataset']['name']}")
    print(f"- Train samples: {config['dataset']['max_train_samples']}")
    print(f"- Test samples: {config['dataset']['max_test_samples']}")
    print(f"- Epochs: {config['training']['epochs']}")
    
    # Run experiment
    experiment = EpistemicHumilityExperiment(config)
    results, analysis = experiment.run_full_experiment()
    
    # Print comprehensive results
    print("\n" + "="*50)
    print("EXPERIMENT COMPLETED")
    print("="*50)
    print(f"📊 Epistemic Humility Metrics:")
    print(f"   F1 Score: {analysis.get('epistemic_humility_f1', 0):.3f}")
    print(f"   Precision: {analysis.get('epistemic_humility_precision', 0):.3f}")
    print(f"   Recall: {analysis.get('epistemic_humility_recall', 0):.3f}")
    
    print(f"\n📈 Model Performance:")
    print(f"   Original Accuracy: {results.get('original_accuracy', 0):.3f}")
    print(f"   Fine-tuned Accuracy: {results.get('finetuned_accuracy', 0):.3f}")
    print(f"   Accuracy Change: {analysis.get('accuracy_change', 0):+.3f}")
    
    print(f"\n🤔 'I Don't Know' Responses:")
    print(f"   Original: {results.get('original_idk_count', 0)}")
    print(f"   Fine-tuned: {results.get('finetuned_idk_count', 0)}")
    print(f"   On wrong answers: {results.get('finetuned_idk_on_wrong', 0)}")
    print(f"   On correct answers: {results.get('finetuned_idk_on_correct', 0)}")
    
    print(f"\n📁 Results saved to: {config['paths']['output_dir']}")
    print(f"📄 View the report: {config['paths']['output_dir']}/experiment_report.md")
    
    return results, analysis

def run_advanced_experiment():
    """Run experiment with advanced features"""
    print("\n" + "="*50)
    print("RUNNING ADVANCED EXPERIMENT")
    print("="*50)
    
    try:
        from experiment_runner import ExperimentRunner
        from utils import AdvancedEvaluator
        
        config = create_default_config()
        config['evaluation']['use_semantic_similarity'] = True
        config['dataset']['max_train_samples'] = 1000
        config['dataset']['max_test_samples'] = 200
        config['paths']['output_dir'] = './results/advanced_experiment'
        
        print("Advanced features enabled:")
        print("✓ Semantic similarity evaluation")
        print("✓ Enhanced analysis")
        
        runner = ExperimentRunner()
        runner.config = config
        results, analysis = runner.run_experiment()
        
        print("\n🎯 Advanced Results:")
        print(f"   Epistemic Humility F1: {analysis.get('epistemic_humility_f1', 0):.3f}")
        print(f"   Enhanced accuracy evaluation used")
        print(f"   Results in: {config['paths']['output_dir']}")
        
        return results, analysis
        
    except ImportError as e:
        print(f"Advanced features not available: {e}")
        print("Falling back to standard experiment...")
        return run_standard_experiment()

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Complete Epistemic Humility Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiment.py --quick          # Quick test (5 min)
  python run_experiment.py --standard       # Standard experiment (30 min)
  python run_experiment.py --advanced       # Advanced with semantic similarity (60 min)
  python run_experiment.py --check          # Check requirements only
        """
    )
    
    parser.add_argument('--quick', action='store_true', 
                       help='Run quick test experiment (5 minutes)')
    parser.add_argument('--standard', action='store_true', 
                       help='Run standard experiment (30 minutes)')
    parser.add_argument('--advanced', action='store_true', 
                       help='Run advanced experiment with semantic similarity (60 minutes)')
    parser.add_argument('--check', action='store_true', 
                       help='Check requirements only')
    
    args = parser.parse_args()
    
    # Setup environment
    setup_experiment_environment()
    
    # Check requirements
    print("\n📋 Checking Requirements...")
    if not check_requirements():
        print("\n❌ Please install missing packages before running the experiment.")
        sys.exit(1)
    
    if args.check:
        print("\n✅ All requirements satisfied!")
        return
    
    # Determine which experiment to run
    if args.quick:
        results, analysis = run_quick_test()
    elif args.advanced:
        results, analysis = run_advanced_experiment()
    elif args.standard:
        results, analysis = run_standard_experiment()
    else:
        # Default to standard if no option specified
        print("\nNo experiment type specified. Running standard experiment...")
        print("Use --help to see all options")
        results, analysis = run_standard_experiment()
    
    # Final summary
    print("\n" + "="*60)
    print("🎉 EXPERIMENT SUMMARY")
    print("="*60)
    
    # Interpret results
    humility_f1 = analysis.get('epistemic_humility_f1', 0)
    accuracy_change = analysis.get('accuracy_change', 0)
    
    if humility_f1 > 0.6:
        humility_assessment = "🟢 EXCELLENT - Model learned strong epistemic humility"
    elif humility_f1 > 0.4:
        humility_assessment = "🟡 GOOD - Model showed moderate epistemic humility"
    elif humility_f1 > 0.2:
        humility_assessment = "🟠 FAIR - Model showed limited epistemic humility"
    else:
        humility_assessment = "🔴 POOR - Model failed to learn epistemic humility"
    
    if accuracy_change > 0.05:
        accuracy_assessment = "🟢 Model accuracy improved significantly"
    elif accuracy_change > -0.02:
        accuracy_assessment = "🟡 Model accuracy remained stable"
    else:
        accuracy_assessment = "🔴 Model accuracy decreased significantly"
    
    print(f"Epistemic Humility: {humility_assessment}")
    print(f"Accuracy Impact: {accuracy_assessment}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if humility_f1 < 0.4:
        print("- Try training with more epochs or different learning rate")
        print("- Consider using a larger model or more training data")
        print("- Experiment with different 'I don't know' training strategies")
    
    if accuracy_change < -0.05:
        print("- Reduce learning rate to preserve original knowledge")
        print("- Use more balanced training data")
        print("- Consider regularization techniques")
    
    if humility_f1 > 0.5 and accuracy_change > -0.02:
        print("- Excellent results! Consider scaling to larger models")
        print("- Try more challenging datasets")
        print("- Experiment with confidence-based IDK thresholds")
    
    print(f"\n📊 Next Steps:")
    print("- Review detailed report for insights")
    print("- Analyze response patterns in results")
    print("- Consider running batch experiments with different configurations")
    
    return results, analysis

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Experiment failed with error: {e}")
        print("Check the logs for detailed error information")
        sys.exit(1)
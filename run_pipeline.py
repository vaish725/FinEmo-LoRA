"""
Complete FinEmo-LoRA Pipeline Runner
Execute the entire training pipeline from data collection to evaluation
"""

import argparse
import subprocess
import sys
from pathlib import Path
import yaml

def load_config():
    """Load project configuration"""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_command(command, description):
    """
    Run a shell command and handle errors
    
    Args:
        command (str): Command to execute
        description (str): Description of the step
    """
    print("\n" + "=" * 80)
    print(f"Step: {description}")
    print("=" * 80)
    print(f"Command: {command}\n")
    
    result = subprocess.run(command, shell=True)
    
    if result.returncode != 0:
        print(f"\nError: Step '{description}' failed with exit code {result.returncode}")
        sys.exit(1)
    
    print(f"\nStep '{description}' completed successfully!")

def main():
    parser = argparse.ArgumentParser(description='FinEmo-LoRA Pipeline Runner')
    parser.add_argument('--skip-download', action='store_true', 
                       help='Skip data download steps')
    parser.add_argument('--skip-annotation', action='store_true',
                       help='Skip annotation step (use existing annotations)')
    parser.add_argument('--skip-stage1', action='store_true',
                       help='Skip Stage 1 training (use existing model)')
    parser.add_argument('--skip-stage2', action='store_true',
                       help='Skip Stage 2 training (use existing model)')
    parser.add_argument('--skip-evaluation', action='store_true',
                       help='Skip evaluation step')
    parser.add_argument('--annotation-samples', type=int, default=5000,
                       help='Number of samples to annotate (default: 5000)')
    parser.add_argument('--stage', type=str, default='all',
                       choices=['all', 'evaluate', 'download', 'annotation', 'stage1', 'stage2', 'train'],
                       help='Run only a specific stage: evaluate, download, annotation, stage1, stage2, train (stage1+stage2), or all')
    
    args = parser.parse_args()

    # If a specific --stage is provided, map it to the existing skip flags
    if args.stage != 'all':
        if args.stage == 'evaluate':
            # Only run evaluation
            args.skip_download = True
            args.skip_annotation = True
            args.skip_stage1 = True
            args.skip_stage2 = True
            args.skip_evaluation = False
        elif args.stage == 'download':
            # Only download datasets
            args.skip_download = False
            args.skip_annotation = True
            args.skip_stage1 = True
            args.skip_stage2 = True
            args.skip_evaluation = True
        elif args.stage == 'annotation':
            args.skip_download = True
            args.skip_annotation = False
            args.skip_stage1 = True
            args.skip_stage2 = True
            args.skip_evaluation = True
        elif args.stage == 'stage1':
            args.skip_download = True
            args.skip_annotation = True
            args.skip_stage1 = False
            args.skip_stage2 = True
            args.skip_evaluation = True
        elif args.stage == 'stage2':
            args.skip_download = True
            args.skip_annotation = True
            args.skip_stage1 = True
            args.skip_stage2 = False
            args.skip_evaluation = True
        elif args.stage == 'train':
            # Run both stage1 and stage2 training
            args.skip_download = True
            args.skip_annotation = True
            args.skip_stage1 = False
            args.skip_stage2 = False
            args.skip_evaluation = True
    
    print("=" * 80)
    print("FinEmo-LoRA Complete Pipeline")
    print("=" * 80)
    print("\nThis script will execute the entire training pipeline:")
    print("  1. Download datasets")
    print("  2. Preprocess GoEmotions")
    print("  3. Annotate financial text with GPT-4")
    print("  4. Stage 1 training (GoEmotions)")
    print("  5. Stage 2 training (Financial emotions)")
    print("  6. Evaluation")
    print("\nThis may take 8-15 hours and will use GPU resources.")
    print("\nMake sure you have:")
    print("  - OPENAI_API_KEY in .env file")
    print("  - HF_TOKEN in .env file")
    print("  - Kaggle API configured (optional, for SEntFiN)")
    print("  - CUDA GPU available")
    
    response = input("\nContinue? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Pipeline cancelled.")
        sys.exit(0)
    
    # Step 1: Download datasets
    if not args.skip_download:
        run_command(
            "python scripts/data_collection/download_goemotions.py",
            "Download GoEmotions dataset"
        )
        
        run_command(
            "python scripts/data_collection/download_fingpt.py",
            "Download FinGPT dataset"
        )
        
        # SEntFiN is optional
        print("\nNote: SEntFiN download skipped (optional). Run manually if needed:")
        print("  python scripts/data_collection/download_sentfin.py")
    else:
        print("\nSkipping data download (--skip-download)")
    
    # Step 2: Preprocess GoEmotions
    if not args.skip_download:
        run_command(
            "python scripts/data_collection/preprocess_goemotions.py",
            "Preprocess GoEmotions (map to 6 emotions)"
        )
    
    # Step 3: Annotate financial text
    if not args.skip_annotation:
        print("\n" + "=" * 80)
        print("Step: Annotate Financial Text with GPT-4")
        print("=" * 80)
        print(f"Annotating {args.annotation_samples} samples...")
        print("Note: This will incur OpenAI API costs (~$20-40 for 5000 samples)")
        
        # Create annotation script
        annotation_cmd = f"""python -c "
from scripts.annotation.llm_annotator import annotate_dataset

annotate_dataset(
    input_path='data/raw/fingpt/train.csv',
    output_path='data/annotated/fingpt_annotated.csv',
    text_column='text',
    max_samples={args.annotation_samples},
    batch_size=50
)
" """
        
        run_command(annotation_cmd, "Annotate financial text")
    else:
        print("\nSkipping annotation (--skip-annotation)")
    
    # Step 4: Stage 1 Training
    if not args.skip_stage1:
        run_command(
            "python scripts/training/train_stage1_goemotions.py",
            "Stage 1 Training - GoEmotions (4-8 hours)"
        )
    else:
        print("\nSkipping Stage 1 training (--skip-stage1)")
    
    # Step 5: Stage 2 Training
    if not args.skip_stage2:
        run_command(
            "python scripts/training/train_stage2_financial.py",
            "Stage 2 Training - Financial Emotions (2-4 hours)"
        )
    else:
        print("\nSkipping Stage 2 training (--skip-stage2)")
    
    # Step 6: Evaluation
    if not args.skip_evaluation:
        run_command(
            "python scripts/evaluation/evaluate.py",
            "Model Evaluation"
        )
    else:
        print("\nSkipping evaluation (--skip-evaluation)")
    
    # Complete
    print("\n" + "=" * 80)
    print("Pipeline Complete!")
    print("=" * 80)
    print("\nResults are available in:")
    print("  - results/              (evaluation metrics, confusion matrix)")
    print("  - models/finemo-lora-final/  (final trained model)")
    print("  - logs/                 (training logs)")
    
    print("\nTo view training logs with TensorBoard:")
    print("  tensorboard --logdir=models/checkpoints")
    
    print("\nNext steps:")
    print("  1. Review evaluation results")
    print("  2. Test model on sample texts")
    print("  3. Experiment with different configurations")

if __name__ == "__main__":
    main()


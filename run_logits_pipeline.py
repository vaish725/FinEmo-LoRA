"""
Complete Logits-based Classification Pipeline
Fast and simple approach - completes in hours instead of days!
"""

import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description='FinEmo-LoRA Logits-based Pipeline - Fast Emotion Classification'
    )
    parser.add_argument('--annotation-samples', type=int, default=2000,
                       help='Number of samples to annotate (default: 2000)')
    parser.add_argument('--skip-download', action='store_true',
                       help='Skip data download')
    parser.add_argument('--skip-annotation', action='store_true',
                       help='Skip annotation (use existing data)')
    parser.add_argument('--skip-extraction', action='store_true',
                       help='Skip feature extraction (use cached features)')
    parser.add_argument('--classifier', type=str, default='mlp',
                       choices=['mlp', 'xgboost', 'svm', 'random_forest'],
                       help='Classifier type (default: mlp)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("FinEmo-LoRA: Logits-based Classification Pipeline")
    print("=" * 80)
    print("\nThis pipeline uses LLM embeddings + lightweight classifier")
    print("Much faster than fine-tuning - completes in a few hours!")
    print()
    print("Pipeline steps:")
    print("  1. Download FinGPT dataset (5 min)")
    print("  2. Annotate with GPT-4 (30-60 min for 2000 samples)")
    print("  3. Extract features from FinBERT (10-20 min)")
    print("  4. Train MLP classifier (5-15 min)")
    print("  5. Evaluate and generate results (5 min)")
    print("\nTotal time: ~1-2 hours")
    
    response = input("\nContinue? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Pipeline cancelled.")
        sys.exit(0)
    
    # Import here to avoid slow imports at startup
    print("\nImporting required modules...")
    from scripts.data_collection.download_fingpt import download_fingpt_dataset
    from scripts.annotation.llm_annotator import annotate_dataset
    from scripts.feature_extraction.extract_features import extract_and_save_features
    from scripts.classifier.train_classifier import train_and_evaluate
    import yaml
    
    # Load config
    config_path = Path("config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Step 1: Download dataset
    if not args.skip_download:
        print("\n" + "=" * 80)
        print("STEP 1: Download FinGPT Dataset")
        print("=" * 80)
        try:
            download_fingpt_dataset(config)
        except Exception as e:
            print(f"\nError downloading dataset: {e}")
            print("Please check your internet connection and try again.")
            sys.exit(1)
    else:
        print("\nSkipping download (--skip-download)")
    
    # Step 2: Annotate financial text
    if not args.skip_annotation:
        print("\n" + "=" * 80)
        print("STEP 2: Annotate Financial Text with GPT-4")
        print("=" * 80)
        print(f"\nAnnotating {args.annotation_samples} samples...")
        print(f"Estimated cost: ${args.annotation_samples * 0.008:.2f} - ${args.annotation_samples * 0.016:.2f}")
        
        try:
            annotate_dataset(
                input_path='data/raw/fingpt/train.csv',
                output_path='data/annotated/fingpt_annotated.csv',
                text_column='text',  # Adjust if needed
                max_samples=args.annotation_samples,
                batch_size=50
            )
        except Exception as e:
            print(f"\nError during annotation: {e}")
            print("\nPossible issues:")
            print("  - OPENAI_API_KEY not set in .env")
            print("  - Insufficient API credits")
            print("  - Column name 'text' not found (check actual column name)")
            sys.exit(1)
    else:
        print("\nSkipping annotation (--skip-annotation)")
    
    # Step 3: Extract features
    if not args.skip_extraction:
        print("\n" + "=" * 80)
        print("STEP 3: Extract Features from Pre-trained LLM")
        print("=" * 80)
        print(f"\nUsing model: {config['model']['selected']}")
        
        try:
            extract_and_save_features(
                input_file='data/annotated/fingpt_annotated_high_confidence.csv',
                output_file='data/features/train_features.npy',
                text_column='text'
            )
        except Exception as e:
            print(f"\nError during feature extraction: {e}")
            print("\nPossible issues:")
            print("  - Annotated file not found")
            print("  - GPU out of memory (try reducing batch_size in config.yaml)")
            print("  - HF_TOKEN not set (for Llama)")
            sys.exit(1)
    else:
        print("\nSkipping feature extraction (--skip-extraction)")
    
    # Step 4: Train classifier
    print("\n" + "=" * 80)
    print("STEP 4: Train Classifier")
    print("=" * 80)
    print(f"\nTraining {args.classifier} classifier...")
    
    try:
        model, label_encoder, accuracy, f1 = train_and_evaluate(
            features_file='data/features/train_features.npy',
            labels_file='data/annotated/fingpt_annotated_high_confidence.csv',
            classifier_type=args.classifier
        )
    except Exception as e:
        print(f"\nError during training: {e}")
        print("\nPossible issues:")
        print("  - Features file not found")
        print("  - Labels file not found")
        print("  - Insufficient training data")
        sys.exit(1)
    
    # Complete!
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)
    print(f"\nFinal Results:")
    print(f"  Classifier: {args.classifier}")
    print(f"  Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Validation F1-Score: {f1:.4f}")
    print(f"\nModel saved in: models/classifiers/")
    print(f"Confusion matrix saved in: models/classifiers/")
    print()
    print("Next steps:")
    print("  1. Review confusion matrix visualization")
    print("  2. Test model on new financial texts")
    print("  3. Experiment with different classifiers:")
    print("     python run_logits_pipeline.py --skip-download --skip-annotation \\")
    print("       --skip-extraction --classifier xgboost")
    print()
    print("To test the model interactively:")
    print("  python scripts/evaluation/inference_demo.py")

if __name__ == "__main__":
    main()


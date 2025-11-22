"""
Retrain Classifier with Cleaned Dataset and Class Weights
Extracts features and trains classifier on fingpt_annotated_v2.csv

Usage:
    python scripts/training/retrain_with_cleaned_data.py [--classifier mlp|xgboost]
"""

import sys
import argparse
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from feature_extraction.extract_features import extract_and_save_features
from classifier.train_classifier import train_and_evaluate

def retrain_pipeline(
    annotated_file: str = "data/annotated/fingpt_annotated_v2.csv",
    classifier_type: str = "mlp"
):
    """
    Complete retraining pipeline with cleaned data
    
    Args:
        annotated_file: Path to cleaned annotated dataset
        classifier_type: Type of classifier (mlp or xgboost)
    """
    print("=" * 80)
    print("RETRAINING CLASSIFIER WITH CLEANED DATA")
    print("=" * 80)
    
    # Step 1: Extract features
    print("\nSTEP 1: Extracting features from cleaned dataset")
    print("-" * 80)
    
    features_file = "data/features/train_features_v2.npy"
    
    extract_and_save_features(
        input_file=annotated_file,
        output_file=features_file,
        text_column='text'
    )
    
    print(f"\n✓ Features saved to: {features_file}")
    
    # Step 2: Train classifier with class weights
    print("\nSTEP 2: Training classifier with class-weighted loss")
    print("-" * 80)
    
    model, label_encoder, accuracy, f1 = train_and_evaluate(
        features_file=features_file,
        labels_file=annotated_file,
        classifier_type=classifier_type
    )
    
    print("\n" + "=" * 80)
    print("RETRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nFinal Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Macro F1: {f1:.4f}")
    
    print("\nNext steps:")
    print("  1. Check confusion matrix in models/classifiers/")
    print("  2. Compare to baseline (63.33% accuracy, 0.33 F1)")
    print("  3. Run full evaluation: python scripts/evaluation/run_full_evaluation.py")
    
    return model, label_encoder, accuracy, f1

def main():
    parser = argparse.ArgumentParser(description='Retrain classifier with cleaned data')
    parser.add_argument('--annotated', type=str,
                       default='data/annotated/fingpt_annotated_v2.csv',
                       help='Path to cleaned annotated dataset')
    parser.add_argument('--classifier', type=str,
                       default='mlp',
                       choices=['mlp', 'xgboost'],
                       help='Classifier type')
    
    args = parser.parse_args()
    
    # Check file exists
    if not Path(args.annotated).exists():
        print(f"Error: Annotated file not found: {args.annotated}")
        print("Have you run merge_cleaned_annotations.py?")
        return
    
    # Retrain
    retrain_pipeline(args.annotated, args.classifier)

if __name__ == "__main__":
    main()

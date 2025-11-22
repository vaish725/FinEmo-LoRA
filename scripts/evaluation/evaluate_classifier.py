"""
Fast CPU-friendly Classifier Evaluation Script
Evaluates trained MLP/XGBoost classifiers on test data
No GPU required - runs on CPU in seconds!

Usage:
    python scripts/evaluation/evaluate_classifier.py --classifier models/classifiers/mlp_20251103_200252.pkl
"""

import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

# Import PyTorch for MLP models (if available)
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not found. Cannot load PyTorch-based classifiers.")

# Import the MLP class definition if available
try:
    from scripts.classifier.train_classifier import MLPClassifierPyTorch
except ImportError:
    MLPClassifierPyTorch = None

def load_classifier(classifier_path: str):
    """
    Load trained classifier from pickle file
    
    Args:
        classifier_path: Path to pickled classifier
        
    Returns:
        Loaded classifier model
    """
    print(f"Loading classifier from: {classifier_path}")
    with open(classifier_path, 'rb') as f:
        classifier = pickle.load(f)
    print("Classifier loaded successfully!")
    return classifier

def load_features_and_labels(features_path: str, metadata_path: str = None):
    """
    Load extracted features and labels
    
    Args:
        features_path: Path to .npy file with features
        metadata_path: Optional path to JSON with labels/metadata
        
    Returns:
        tuple: (features, labels, emotion_labels)
    """
    print(f"\nLoading features from: {features_path}")
    features = np.load(features_path)
    print(f"Features shape: {features.shape}")
    
    # Try to load metadata/labels
    if metadata_path is None:
        # Try to find corresponding JSON file
        json_path = Path(features_path).with_suffix('.json')
        if json_path.exists():
            metadata_path = str(json_path)
    
    labels = None
    emotion_labels = None
    
    if metadata_path and Path(metadata_path).exists():
        print(f"Loading metadata from: {metadata_path}")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        labels = metadata.get('labels', None)
        emotion_labels = metadata.get('emotion_labels', None)
        
        if labels:
            print(f"Found {len(labels)} labels")
    
    return features, labels, emotion_labels

def evaluate_classifier(classifier, features, labels, emotion_labels, output_dir: str = "results"):
    """
    Evaluate classifier and generate metrics
    
    Args:
        classifier: Trained classifier model
        features: Feature matrix (N x D)
        labels: Ground truth labels
        emotion_labels: List of emotion label names
        output_dir: Directory to save results
    """
    print("\n" + "=" * 80)
    print("Evaluating Classifier")
    print("=" * 80)
    
    # Make predictions
    print("\nGenerating predictions...")
    predictions = classifier.predict(features)
    
    # Get probabilities if available
    try:
        probabilities = classifier.predict_proba(features)
        has_probs = True
    except:
        probabilities = None
        has_probs = False
    
    # Compute metrics
    accuracy = accuracy_score(labels, predictions)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=list(range(len(emotion_labels))) if emotion_labels else None,
        average=None,
        zero_division=0
    )
    
    # Macro averages
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    print(f"\nMacro-averaged Metrics:")
    print(f"  Precision: {macro_precision:.4f}")
    print(f"  Recall: {macro_recall:.4f}")
    print(f"  F1-Score: {macro_f1:.4f}")
    
    # Per-class metrics table
    print(f"\nPer-Class Metrics:")
    print("-" * 80)
    print(f"{'Emotion':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 80)
    
    for i in range(len(precision)):
        emotion = emotion_labels[i] if emotion_labels else f"Class_{i}"
        print(f"{emotion:<15} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<10}")
    
    print("-" * 80)
    
    # Detailed classification report
    print("\nDetailed Classification Report:")
    print("-" * 80)
    target_names = emotion_labels if emotion_labels else [f"Class_{i}" for i in range(len(precision))]
    report = classification_report(labels, predictions, target_names=target_names, zero_division=0)
    print(report)
    
    # Confusion Matrix
    cm = confusion_matrix(labels, predictions)
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save metrics
    metrics_dict = {
        'timestamp': timestamp,
        'overall_accuracy': float(accuracy),
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'macro_f1': float(macro_f1),
        'per_class_metrics': {
            (emotion_labels[i] if emotion_labels else f"Class_{i}"): {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1[i]),
                'support': int(support[i])
            }
            for i in range(len(precision))
        }
    }
    
    metrics_file = output_path / f"classifier_metrics_{timestamp}.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"\nMetrics saved to: {metrics_file}")
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, target_names, output_path, timestamp)
    
    print("\n" + "=" * 80)
    print("Evaluation Complete!")
    print("=" * 80)
    
    return metrics_dict

def plot_confusion_matrix(cm, class_names, output_dir, timestamp):
    """
    Plot and save confusion matrix
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        output_dir: Directory to save plot
        timestamp: Timestamp string
    """
    print("\nGenerating confusion matrix plot...")
    
    plt.figure(figsize=(10, 8))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Proportion'}
    )
    
    plt.title('Classifier Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Emotion', fontsize=12)
    plt.xlabel('Predicted Emotion', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save plot
    plot_file = output_dir / f"confusion_matrix_{timestamp}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to: {plot_file}")
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained classifier (CPU-friendly)')
    parser.add_argument('--classifier', type=str, required=True,
                       help='Path to pickled classifier (.pkl file)')
    parser.add_argument('--features', type=str, default='data/features/train_features.npy',
                       help='Path to features (.npy file)')
    parser.add_argument('--metadata', type=str, default=None,
                       help='Path to metadata JSON (optional, auto-detected if not provided)')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("FinEmo-LoRA Classifier Evaluation (CPU-friendly)")
    print("=" * 80)
    print(f"\nClassifier: {args.classifier}")
    print(f"Features: {args.features}")
    print(f"Output: {args.output_dir}")
    
    # Load classifier
    classifier = load_classifier(args.classifier)
    
    # Load features and labels
    features, labels, emotion_labels = load_features_and_labels(args.features, args.metadata)
    
    if labels is None:
        print("\nError: No labels found in metadata. Cannot evaluate.")
        print("Please ensure the features JSON file contains 'labels' field.")
        return
    
    if emotion_labels is None:
        print("\nWarning: No emotion_labels found. Using generic class names.")
        emotion_labels = [f"Class_{i}" for i in range(len(set(labels)))]
    
    # Evaluate
    metrics = evaluate_classifier(classifier, features, labels, emotion_labels, args.output_dir)
    
    print("\nTo view results:")
    print(f"  - Metrics: {args.output_dir}/classifier_metrics_*.json")
    print(f"  - Confusion matrix: {args.output_dir}/confusion_matrix_*.png")

if __name__ == "__main__":
    main()

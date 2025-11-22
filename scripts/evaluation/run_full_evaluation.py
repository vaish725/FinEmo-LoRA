"""
Complete End-to-End Evaluation (CPU-Friendly)
Loads annotated data → Extracts features → Runs classifier → Generates metrics

This script is optimized for CPU and does NOT require GPU!

Usage:
    python scripts/evaluation/run_full_evaluation.py --classifier models/classifiers/mlp_20251103_200252.pkl
"""

import os
import sys
import argparse
import pickle
import yaml
import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

# Import transformers for feature extraction
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers not available")

try:
    from scripts.classifier.train_classifier import MLPClassifierPyTorch
except ImportError:
    MLPClassifierPyTorch = None

def load_config():
    """Load project configuration"""
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_annotated_data(data_path: str):
    """
    Load annotated dataset
    
    Args:
        data_path: Path to CSV file with annotations
        
    Returns:
        DataFrame with text and emotion labels
    """
    print(f"Loading annotated data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples")
    
    # Check required columns
    if 'text' not in df.columns or 'emotion' not in df.columns:
        raise ValueError("CSV must contain 'text' and 'emotion' columns")
    
    # Show emotion distribution
    print("\nEmotion distribution:")
    print(df['emotion'].value_counts())
    
    return df

def extract_features_cpu(texts, model_name="distilbert-base-uncased", device="cpu", batch_size=16):
    """
    Extract features from texts using a pre-trained model (CPU-optimized)
    
    Args:
        texts: List of text strings
        model_name: HuggingFace model name
        device: Device to use (cpu, mps, or cuda)
        batch_size: Batch size for processing
        
    Returns:
        numpy array of features (N x feature_dim)
    """
    if not HAS_TRANSFORMERS:
        raise RuntimeError("transformers library is required for feature extraction")
    
    print(f"\nExtracting features using {model_name} on {device}...")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Move to device
    if device == "mps" and torch.backends.mps.is_available():
        model = model.to("mps")
    elif device == "cuda" and torch.cuda.is_available():
        model = model.to("cuda")
    else:
        model = model.to("cpu")
        device = "cpu"
    
    model.eval()
    
    print(f"Model loaded on {device}")
    
    # Extract features in batches
    all_features = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting features"):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Extract features
        with torch.no_grad():
            outputs = model(**inputs)
            
            # Use mean pooling of last hidden state
            hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)
            attention_mask = inputs['attention_mask'].unsqueeze(-1)  # (batch, seq_len, 1)
            
            # Mean pooling
            masked_hidden = hidden_states * attention_mask
            sum_hidden = masked_hidden.sum(dim=1)
            sum_mask = attention_mask.sum(dim=1)
            features = sum_hidden / sum_mask  # (batch, hidden_dim)
            
            # L2 normalization
            features = features / features.norm(dim=1, keepdim=True)
            
            all_features.append(features.cpu().numpy())
    
    # Concatenate all batches
    all_features = np.vstack(all_features)
    
    print(f"Extracted features shape: {all_features.shape}")
    
    return all_features

def evaluate_classifier(classifier, features, labels, emotion_labels, output_dir: str = "results"):
    """
    Evaluate classifier and generate comprehensive metrics
    
    Args:
        classifier: Trained classifier model
        features: Feature matrix (N x D)
        labels: Ground truth labels (encoded as integers)
        emotion_labels: List of emotion label names
        output_dir: Directory to save results
        
    Returns:
        Dictionary of metrics
    """
    print("\n" + "=" * 80)
    print("Evaluating Classifier")
    print("=" * 80)
    
    # Make predictions
    print("\nGenerating predictions...")
    
    # Handle PyTorch models
    if isinstance(classifier, torch.nn.Module):
        classifier.eval()
        device = next(classifier.parameters()).device
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).to(device)
            outputs = classifier(features_tensor)
            predictions = outputs.argmax(dim=1).cpu().numpy()
    else:
        # Scikit-learn models
        predictions = classifier.predict(features)
    
    # Compute metrics
    accuracy = accuracy_score(labels, predictions)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=list(range(len(emotion_labels))),
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
    
    for i, emotion in enumerate(emotion_labels):
        print(f"{emotion:<15} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<10}")
    
    print("-" * 80)
    
    # Detailed classification report
    print("\nDetailed Classification Report:")
    print("-" * 80)
    report = classification_report(labels, predictions, target_names=emotion_labels, zero_division=0)
    print(report)
    
    # Confusion Matrix
    cm = confusion_matrix(labels, predictions, labels=list(range(len(emotion_labels))))
    
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
            emotion_labels[i]: {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1[i]),
                'support': int(support[i])
            }
            for i in range(len(emotion_labels))
        }
    }
    
    metrics_file = output_path / f"evaluation_metrics_{timestamp}.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"\nMetrics saved to: {metrics_file}")
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, emotion_labels, output_path, timestamp)
    
    return metrics_dict

def plot_confusion_matrix(cm, class_names, output_dir, timestamp):
    """Plot and save confusion matrix"""
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
    
    plt.title('FinEmo-LoRA Confusion Matrix', fontsize=14, fontweight='bold')
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
    parser = argparse.ArgumentParser(description='Complete evaluation pipeline (CPU-friendly)')
    parser.add_argument('--classifier', type=str, required=True,
                       help='Path to trained classifier (.pkl file)')
    parser.add_argument('--data', type=str, default='data/annotated/fingpt_annotated.csv',
                       help='Path to annotated CSV file')
    parser.add_argument('--model', type=str, default='distilbert-base-uncased',
                       help='Feature extraction model (default: distilbert-base-uncased)')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'mps', 'cuda', 'auto'],
                       help='Device to use for feature extraction')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for feature extraction')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Fraction of data to use for testing')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("FinEmo-LoRA Complete Evaluation Pipeline (CPU-Friendly)")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Classifier: {args.classifier}")
    print(f"  Data: {args.data}")
    print(f"  Feature model: {args.model}")
    print(f"  Device: {args.device}")
    print(f"  Test size: {args.test_size}")
    print(f"  Output: {args.output_dir}")
    
    # Load config
    config = load_config()
    emotion_labels = config['emotion_labels']
    
    # Load data
    df = load_annotated_data(args.data)
    
    # Split into train/test
    print(f"\nSplitting data (test_size={args.test_size}, seed={args.seed})...")
    train_df, test_df = train_test_split(df, test_size=args.test_size, random_state=args.seed, stratify=df['emotion'])
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(emotion_labels)
    
    test_labels_encoded = label_encoder.transform(test_df['emotion'].tolist())
    
    # Extract features from test set
    test_features = extract_features_cpu(
        test_df['text'].tolist(),
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size
    )
    
    # Load classifier
    print(f"\nLoading classifier from: {args.classifier}")
    with open(args.classifier, 'rb') as f:
        classifier_obj = pickle.load(f)
    
    # Handle both dict and direct model formats
    if isinstance(classifier_obj, dict):
        classifier = classifier_obj['model']
        label_encoder_saved = classifier_obj.get('label_encoder', None)
        print(f"Loaded classifier type: {classifier_obj.get('classifier_type', 'unknown')}")
        print(f"Feature dim: {classifier_obj.get('feature_dim', 'unknown')}")
    else:
        classifier = classifier_obj
        label_encoder_saved = None
    
    print("Classifier loaded successfully!")
    
    # Evaluate
    metrics = evaluate_classifier(
        classifier,
        test_features,
        test_labels_encoded,
        emotion_labels,
        args.output_dir
    )
    
    print("\n" + "=" * 80)
    print("Evaluation Complete!")
    print("=" * 80)
    print(f"\nResults saved to: {args.output_dir}/")
    print(f"  - Metrics JSON: evaluation_metrics_*.json")
    print(f"  - Confusion matrix: confusion_matrix_*.png")
    
    return metrics

if __name__ == "__main__":
    main()

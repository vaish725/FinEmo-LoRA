"""
Model Evaluation Script
Evaluates the FinEmo-LoRA model on test set with comprehensive metrics:
- Per-class Precision, Recall, F1-score
- Overall Accuracy
- Confusion Matrix
- Error Analysis
"""

import os
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from dotenv import load_dotenv

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from tqdm import tqdm

# Load environment variables
load_dotenv()

def load_config():
    """
    Load the project configuration file
    
    Returns:
        dict: Configuration dictionary
    """
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_model_for_evaluation(config: dict, model_type: str = "final"):
    """
    Load the trained model for evaluation
    
    Args:
        config (dict): Project configuration
        model_type (str): "final" for Stage 2 model, "stage1" for Stage 1 model
        
    Returns:
        tuple: (model, tokenizer)
    """
    print("=" * 80)
    print("Loading Model for Evaluation")
    print("=" * 80)
    
    if model_type == "final":
        model_dir = Path(config['training']['final_model_dir'])
    elif model_type == "stage1":
        model_dir = Path(config['paths']['models']) / "stage1_final"
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    if not model_dir.exists():
        raise FileNotFoundError(f"Model not found at {model_dir}")
    
    print(f"\nLoading model from: {model_dir}")
    
    # Check if merged model exists
    merged_dir = model_dir / "merged"
    if merged_dir.exists():
        print("Loading merged model (LoRA adapters integrated)...")
        load_dir = merged_dir
        is_peft = False
    else:
        print("Loading model with LoRA adapters...")
        load_dir = model_dir
        is_peft = True
    
    # Get HuggingFace token
    hf_token = os.getenv('HF_TOKEN')
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(load_dir),
        token=hf_token,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model
    # For evaluation, we can use full precision or fp16
    if is_peft:
        # Load base model then LoRA adapters
        selected_model = config['model']['selected']
        base_model_name = config['model']['base_models'][selected_model]['name']
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            token=hf_token,
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        model = PeftModel.from_pretrained(base_model, str(load_dir))
    else:
        # Load merged model directly
        model = AutoModelForCausalLM.from_pretrained(
            str(load_dir),
            device_map="auto",
            token=hf_token,
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
    
    model.eval()  # Set to evaluation mode
    
    print("Model loaded successfully!")
    return model, tokenizer

def predict_emotion(text: str, model, tokenizer, emotion_labels: list) -> tuple:
    """
    Predict emotion for a single text
    
    Args:
        text (str): Input text
        model: Trained model
        tokenizer: Tokenizer
        emotion_labels (list): List of emotion labels
        
    Returns:
        tuple: (predicted_emotion, confidence)
    """
    # Create prompt
    prompt = f"Classify the economic emotion in this financial text: {text}\n\nEmotion:"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,  # Emotion label should be short
            temperature=0.1,  # Low temperature for consistent predictions
            do_sample=False,  # Greedy decoding
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract emotion from generated text
    # The emotion should appear after "Emotion:"
    if "Emotion:" in generated_text:
        predicted_text = generated_text.split("Emotion:")[-1].strip()
    else:
        predicted_text = generated_text[len(prompt):].strip()
    
    # Match to one of our emotion labels (case-insensitive)
    predicted_text_lower = predicted_text.lower()
    predicted_emotion = None
    
    for emotion in emotion_labels:
        if emotion.lower() in predicted_text_lower:
            predicted_emotion = emotion
            break
    
    # If no match, take the first word
    if predicted_emotion is None:
        first_word = predicted_text.split()[0] if predicted_text.split() else ""
        # Check if it's close to any emotion
        for emotion in emotion_labels:
            if first_word.lower() == emotion.lower():
                predicted_emotion = emotion
                break
        
        if predicted_emotion is None:
            # Default to most common emotion or "uncertainty"
            predicted_emotion = "uncertainty"
    
    # For now, confidence is set to 1.0 (can be enhanced with logits analysis)
    confidence = 1.0
    
    return predicted_emotion, confidence

def evaluate_model(config: dict, model_type: str = "final"):
    """
    Comprehensive model evaluation
    
    Args:
        config (dict): Project configuration
        model_type (str): Which model to evaluate
    """
    print("\n" + "=" * 80)
    print("FinEmo-LoRA Model Evaluation")
    print("=" * 80)
    
    # Load model
    model, tokenizer = load_model_for_evaluation(config, model_type)
    
    # Load test dataset
    test_file = Path(config['paths']['data_processed']) / "financial_emotion_splits" / "test.csv"
    
    if not test_file.exists():
        raise FileNotFoundError(
            f"Test dataset not found at {test_file}. "
            "Please run Stage 2 training first to create data splits."
        )
    
    print(f"\nLoading test dataset from: {test_file}")
    test_df = pd.read_csv(test_file)
    print(f"Test samples: {len(test_df)}")
    
    # Get emotion labels
    emotion_labels = config['emotion_labels']
    print(f"\nEmotion labels: {emotion_labels}")
    
    # Make predictions
    print("\nGenerating predictions...")
    predictions = []
    confidences = []
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating"):
        text = row['text']
        pred_emotion, confidence = predict_emotion(text, model, tokenizer, emotion_labels)
        predictions.append(pred_emotion)
        confidences.append(confidence)
    
    # Add predictions to dataframe
    test_df['predicted_emotion'] = predictions
    test_df['prediction_confidence'] = confidences
    
    # Get ground truth
    true_labels = test_df['emotion'].tolist()
    pred_labels = test_df['predicted_emotion'].tolist()
    
    # Compute metrics
    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    
    # Overall accuracy
    accuracy = accuracy_score(true_labels, pred_labels)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels,
        pred_labels,
        labels=emotion_labels,
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
    report = classification_report(true_labels, pred_labels, labels=emotion_labels, zero_division=0)
    print(report)
    
    # Confusion Matrix
    cm = confusion_matrix(true_labels, pred_labels, labels=emotion_labels)
    
    # Save results
    results_dir = Path(config['paths']['results'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save predictions
    predictions_file = results_dir / f"predictions_{model_type}_{timestamp}.csv"
    test_df.to_csv(predictions_file, index=False)
    print(f"\nPredictions saved to: {predictions_file}")
    
    # Save metrics
    metrics_dict = {
        'model_type': model_type,
        'timestamp': timestamp,
        'overall_accuracy': float(accuracy),
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'macro_f1': float(macro_f1),
        'per_class_metrics': {
            emotion: {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1[i]),
                'support': int(support[i])
            }
            for i, emotion in enumerate(emotion_labels)
        }
    }
    
    import json
    metrics_file = results_dir / f"metrics_{model_type}_{timestamp}.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"Metrics saved to: {metrics_file}")
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, emotion_labels, results_dir, model_type, timestamp)
    
    # Error analysis
    error_analysis(test_df, results_dir, model_type, timestamp)
    
    print("\n" + "=" * 80)
    print("Evaluation Complete!")
    print("=" * 80)
    
    return metrics_dict

def plot_confusion_matrix(cm, emotion_labels, results_dir, model_type, timestamp):
    """
    Plot and save confusion matrix
    
    Args:
        cm: Confusion matrix
        emotion_labels: List of emotion labels
        results_dir: Directory to save plot
        model_type: Model type identifier
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
        xticklabels=emotion_labels,
        yticklabels=emotion_labels,
        cbar_kws={'label': 'Proportion'}
    )
    
    plt.title(f'FinEmo-LoRA Confusion Matrix ({model_type})', fontsize=14, fontweight='bold')
    plt.ylabel('True Emotion', fontsize=12)
    plt.xlabel('Predicted Emotion', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save plot
    plot_file = results_dir / f"confusion_matrix_{model_type}_{timestamp}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to: {plot_file}")
    
    plt.close()

def error_analysis(test_df, results_dir, model_type, timestamp):
    """
    Perform error analysis on misclassified samples
    
    Args:
        test_df: Test dataframe with predictions
        results_dir: Directory to save analysis
        model_type: Model type identifier
        timestamp: Timestamp string
    """
    print("\nPerforming error analysis...")
    
    # Find misclassified samples
    errors_df = test_df[test_df['emotion'] != test_df['predicted_emotion']].copy()
    
    if len(errors_df) == 0:
        print("No errors found - perfect classification!")
        return
    
    print(f"Misclassified samples: {len(errors_df)} / {len(test_df)} ({len(errors_df)/len(test_df)*100:.2f}%)")
    
    # Most common error patterns
    errors_df['error_type'] = errors_df['emotion'] + ' -> ' + errors_df['predicted_emotion']
    error_counts = errors_df['error_type'].value_counts()
    
    print("\nMost common error patterns:")
    for error_type, count in error_counts.head(10).items():
        print(f"  {error_type}: {count} times")
    
    # Save error samples
    errors_file = results_dir / f"error_analysis_{model_type}_{timestamp}.csv"
    errors_df.to_csv(errors_file, index=False)
    print(f"\nError analysis saved to: {errors_file}")

def main():
    """
    Main function to run evaluation
    """
    # Load configuration
    config = load_config()
    
    # Evaluate the final model
    print("Evaluating FinEmo-LoRA final model...")
    evaluate_model(config, model_type="final")

if __name__ == "__main__":
    main()


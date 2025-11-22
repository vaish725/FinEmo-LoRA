"""
Classifier Training Script for Logits-based Approach
Trains a lightweight classifier on LLM-extracted features
Much faster than fine-tuning - trains in minutes instead of hours!
"""

import os
import yaml
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

# Classifier imports
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb

# PyTorch for MLP (more flexible)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight

def load_config():
    """Load project configuration"""
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class MLPClassifierPyTorch(nn.Module):
    """
    Multi-Layer Perceptron classifier using PyTorch
    More flexible than sklearn's MLPClassifier
    """
    
    def __init__(self, input_dim, hidden_layers, num_classes, dropout=0.3):
        """
        Initialize MLP classifier
        
        Args:
            input_dim (int): Input feature dimension
            hidden_layers (list): List of hidden layer sizes
            num_classes (int): Number of output classes
            dropout (float): Dropout rate
        """
        super(MLPClassifierPyTorch, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass"""
        return self.model(x)

def train_mlp_pytorch(X_train, y_train, X_val, y_val, config, num_classes):
    """
    Train MLP classifier using PyTorch
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        config: Classifier configuration
        num_classes: Number of classes
        
    Returns:
        Trained model
    """
    print("\n" + "=" * 80)
    print("Training MLP Classifier (PyTorch)")
    print("=" * 80)
    
    mlp_config = config['types']['mlp']
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.LongTensor(y_val).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=mlp_config['batch_size'],
        shuffle=True
    )
    
    # Initialize model
    model = MLPClassifierPyTorch(
        input_dim=X_train.shape[1],
        hidden_layers=mlp_config['hidden_layers'],
        num_classes=num_classes,
        dropout=mlp_config['dropout']
    ).to(device)
    
    print(f"\nModel architecture:")
    print(model)
    print(f"\nTrainable parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Compute class weights to handle imbalance
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    
    print(f"\nClass weights (to handle imbalance):")
    for i, weight in enumerate(class_weights):
        print(f"  Class {i}: {weight:.3f}")
    
    # Loss and optimizer with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=mlp_config['learning_rate'])
    
    # Training loop
    epochs = mlp_config['epochs']
    patience = mlp_config['early_stopping_patience']
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    print(f"\nTraining for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()
            val_losses.append(val_loss)
            
            # Calculate accuracy
            _, predicted = torch.max(val_outputs, 1)
            val_acc = (predicted == y_val_tensor).float().mean().item()
            val_accuracies.append(val_acc)
        
        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val Acc: {val_acc:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    print(f"\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    return model, train_losses, val_losses, val_accuracies

def train_sklearn_classifier(X_train, y_train, classifier_type, config):
    """
    Train a scikit-learn classifier
    
    Args:
        X_train: Training features
        y_train: Training labels
        classifier_type: Type of classifier (xgboost, svm, random_forest)
        config: Classifier configuration
        
    Returns:
        Trained classifier
    """
    print(f"\n" + "=" * 80)
    print(f"Training {classifier_type.upper()} Classifier")
    print("=" * 80)
    
    clf_config = config['types'][classifier_type]
    
    if classifier_type == 'xgboost':
        classifier = xgb.XGBClassifier(
            n_estimators=clf_config['n_estimators'],
            max_depth=clf_config['max_depth'],
            learning_rate=clf_config['learning_rate'],
            subsample=clf_config['subsample'],
            colsample_bytree=clf_config['colsample_bytree'],
            random_state=config['seed']
        )
    
    elif classifier_type == 'svm':
        classifier = SVC(
            kernel=clf_config['kernel'],
            C=clf_config['C'],
            gamma=clf_config['gamma'],
            random_state=config['seed'],
            probability=True  # Enable probability estimates
        )
    
    elif classifier_type == 'random_forest':
        classifier = RandomForestClassifier(
            n_estimators=clf_config['n_estimators'],
            max_depth=clf_config['max_depth'],
            min_samples_split=clf_config['min_samples_split'],
            random_state=config['seed']
        )
    
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")
    
    print(f"\nTraining {classifier_type}...")
    classifier.fit(X_train, y_train)
    
    print("Training complete!")
    
    return classifier

def train_and_evaluate(features_file: str, labels_file: str, classifier_type: str = None):
    """
    Main training and evaluation pipeline
    
    Args:
        features_file (str): Path to .npy file with features
        labels_file (str): Path to CSV file with labels
        classifier_type (str): Type of classifier (None = use config default)
    """
    print("=" * 80)
    print("Classifier Training Pipeline")
    print("=" * 80)
    
    # Load configuration
    config = load_config()
    classifier_config = config['classifier']
    
    if classifier_type is None:
        classifier_type = classifier_config['selected']
    
    print(f"\nClassifier type: {classifier_type}")
    
    # Load features
    print(f"\nLoading features from: {features_file}")
    X = np.load(features_file)
    print(f"Features loaded: {X.shape}")
    
    # Load labels
    print(f"\nLoading labels from: {labels_file}")
    df = pd.read_csv(labels_file)
    
    if 'emotion' not in df.columns:
        raise ValueError("Labels file must have 'emotion' column")
    
    y = df['emotion'].values
    print(f"Labels loaded: {len(y)}")
    
    # Check dimensions match
    if len(X) != len(y):
        raise ValueError(f"Feature count ({len(X)}) != label count ({len(y)})")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"\nClasses: {list(label_encoder.classes_)}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    
    # Class distribution
    print("\nClass distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  {label}: {count} ({count/len(y)*100:.1f}%)")
    
    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded,
        test_size=0.2,
        stratify=y_encoded,
        random_state=classifier_config['seed']
    )
    
    print(f"\nTrain samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Train classifier
    if classifier_type == 'mlp':
        model, train_losses, val_losses, val_accs = train_mlp_pytorch(
            X_train, y_train, X_val, y_val,
            classifier_config, len(label_encoder.classes_)
        )
        
        # Convert model to CPU for saving
        model = model.cpu()
        
    else:
        model = train_sklearn_classifier(
            X_train, y_train, classifier_type, classifier_config
        )
    
    # Evaluate on validation set
    print("\n" + "=" * 80)
    print("Validation Set Evaluation")
    print("=" * 80)
    
    if classifier_type == 'mlp':
        # PyTorch model prediction
        model.eval()
        with torch.no_grad():
            X_val_tensor = torch.FloatTensor(X_val)
            outputs = model(X_val_tensor)
            _, y_pred = torch.max(outputs, 1)
            y_pred = y_pred.numpy()
    else:
        # sklearn classifier prediction
        y_pred = model.predict(X_val)
    
    # Calculate metrics
    accuracy = accuracy_score(y_val, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_val, y_pred, average='macro'
    )
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision (macro): {precision:.4f}")
    print(f"  Recall (macro): {recall:.4f}")
    print(f"  F1-Score (macro): {f1:.4f}")
    
    # Detailed classification report
    print("\nPer-Class Metrics:")
    print(classification_report(
        y_val, y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    ))
    
    # Save model and artifacts
    output_dir = Path(classifier_config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{classifier_type}_{timestamp}"
    
    # Save model
    model_path = output_dir / f"{model_name}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'label_encoder': label_encoder,
            'classifier_type': classifier_type,
            'feature_dim': X.shape[1],
            'config': classifier_config
        }, f)
    
    print(f"\nModel saved to: {model_path}")
    
    # Save confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    plot_confusion_matrix(cm, label_encoder.classes_, output_dir, model_name)
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    
    return model, label_encoder, accuracy, f1

def plot_confusion_matrix(cm, class_names, output_dir, model_name):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    
    # Normalize
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Proportion'}
    )
    
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Emotion', fontsize=12)
    plt.xlabel('Predicted Emotion', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save
    plot_path = output_dir / f"confusion_matrix_{model_name}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to: {plot_path}")

def main():
    """Example usage"""
    print("\nExample Usage:")
    print("-" * 80)
    print("from scripts.classifier.train_classifier import train_and_evaluate")
    print()
    print("# Train MLP classifier on extracted features")
    print("train_and_evaluate(")
    print("    features_file='data/features/train_features.npy',")
    print("    labels_file='data/annotated/fingpt_annotated_high_confidence.csv',")
    print("    classifier_type='mlp'  # Options: mlp, xgboost, svm, random_forest")
    print(")")
    print("-" * 80)

if __name__ == "__main__":
    main()


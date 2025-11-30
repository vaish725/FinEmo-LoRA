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
from sklearn.neural_network import MLPClassifier as SklearnMLP
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

class ImprovedMLPClassifier(nn.Module):
    """
    Deeper MLP with optional batch normalization for better performance
    Targets 75-80% accuracy with larger datasets
    """
    def __init__(self, input_dim=768, hidden_dims=[512, 384, 256, 128], 
                 num_classes=6, dropout=0.4, use_batchnorm=False):
        super(ImprovedMLPClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

def train_mlp_pytorch(X_train, y_train, X_val, y_val, config, num_classes, use_improved=False):
    """
    Train MLP classifier using PyTorch
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        config: Classifier configuration
        num_classes: Number of classes
        use_improved: Use improved architecture with BatchNorm
        
    Returns:
        Trained model
    """
    print("\n" + "=" * 80)
    print(f"Training {'Improved ' if use_improved else ''}MLP Classifier (PyTorch)")
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
    if use_improved:
        # Disable BatchNorm on CPU to avoid segfault on macOS
        use_bn = device.type == 'cuda'
        model = ImprovedMLPClassifier(
            input_dim=X_train.shape[1],
            hidden_dims=[512, 384, 256, 128],
            num_classes=num_classes,
            dropout=0.4,
            use_batchnorm=use_bn
        ).to(device)
        if not use_bn:
            print("\nNote: BatchNorm disabled on CPU (prevents segfault on macOS)")
    else:
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
    # Use standard CrossEntropyLoss without class weights on macOS to avoid segfault
    import platform
    if platform.system() == 'Darwin' and device.type == 'cpu':
        print("\nNote: Using unweighted loss on macOS CPU to avoid segfault")
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    if use_improved:
        # Use standard Adam instead of AdamW on macOS CPU
        if platform.system() == 'Darwin' and device.type == 'cpu':
            optimizer = optim.Adam(model.parameters(), lr=0.0005)
        else:
            optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=mlp_config['learning_rate'])
        scheduler = None
    
    # Training loop
    epochs = mlp_config['epochs']
    patience = mlp_config['early_stopping_patience'] if not use_improved else 15
    
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
        
        # Learning rate scheduling (improved model only)
        if scheduler is not None:
            scheduler.step(val_loss)
        
        # Print progress every 5 epochs (or 10 for improved)
        print_freq = 10 if use_improved else 5
        if (epoch + 1) % print_freq == 0:
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

def train_sklearn_mlp(X_train, y_train, use_improved=False):
    """
    Train sklearn MLP (safer on macOS, no segfault issues)
    """
    print("\n" + "=" * 80)
    print(f"Training {'Improved ' if use_improved else ''}MLP (sklearn)")
    print("=" * 80)
    
    if use_improved:
        # Deeper architecture matching improved PyTorch version
        model = SklearnMLP(
            hidden_layer_sizes=(512, 384, 256, 128),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size=32,
            learning_rate='adaptive',
            learning_rate_init=0.0005,
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=15,
            random_state=42,
            verbose=True
        )
    else:
        model = SklearnMLP(
            hidden_layer_sizes=(512, 256, 128),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size=32,
            learning_rate='adaptive',
            max_iter=100,
            early_stopping=True,
            validation_fraction=0.2,
            random_state=42,
            verbose=True
        )
    
    print("\nTraining sklearn MLP...")
    model.fit(X_train, y_train)
    print("\nTraining complete!")
    
    return model

def train_sklearn_classifier(X_train, y_train, X_val, y_val, classifier_type, config, use_improved=False):
    """
    Train a scikit-learn classifier
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        classifier_type: Type of classifier (xgboost, svm, random_forest)
        config: Classifier configuration
        use_improved: Use improved hyperparameters
        
    Returns:
        Trained classifier
    """
    print(f"\n" + "=" * 80)
    print(f"Training {'Improved ' if use_improved else ''}{classifier_type.upper()} Classifier")
    print("=" * 80)
    
    clf_config = config['types'][classifier_type]
    
    if classifier_type == 'xgboost':
        if use_improved:
            # Compute class weights
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(y_train),
                y=y_train
            )
            sample_weights = np.array([class_weights[y] for y in y_train])
            
            classifier = xgb.XGBClassifier(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=config['seed'],
                eval_metric='mlogloss',
                early_stopping_rounds=20
            )
            
            print("\nTraining with improved hyperparameters and sample weighting...")
            classifier.fit(
                X_train, y_train,
                sample_weight=sample_weights,
                eval_set=[(X_val, y_val)],
                verbose=20
            )
            return classifier
        else:
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
    
    if not use_improved or classifier_type != 'xgboost':
        print(f"\nTraining {classifier_type}...")
        classifier.fit(X_train, y_train)
        print("Training complete!")
    
    return classifier

def ensemble_predictions(models, X, model_types):
    """
    Ensemble predictions from multiple models using majority voting
    
    Args:
        models: List of trained models
        X: Input features
        model_types: List of model types ('mlp', 'xgboost', etc.)
        
    Returns:
        numpy array of ensemble predictions
    """
    predictions = []
    
    for model, model_type in zip(models, model_types):
        if model_type == 'mlp':
            # Check if PyTorch or sklearn model
            if isinstance(model, nn.Module):
                # PyTorch model
                model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X)
                    outputs = model(X_tensor)
                    _, pred = torch.max(outputs, 1)
                    predictions.append(pred.numpy())
            else:
                # sklearn MLP
                predictions.append(model.predict(X))
        else:
            predictions.append(model.predict(X))
    
    # Majority voting
    predictions = np.array(predictions)
    ensemble_pred = []
    for i in range(predictions.shape[1]):
        votes = predictions[:, i]
        ensemble_pred.append(np.bincount(votes).argmax())
    
    return np.array(ensemble_pred)

def train_and_evaluate(features_file: str, labels_file: str, classifier_type: str = None, 
                      use_improved: bool = False, use_ensemble: bool = False):
    """
    Main training and evaluation pipeline
    
    Args:
        features_file (str): Path to .npy file with features
        labels_file (str): Path to CSV file with labels
        classifier_type (str): Type of classifier (None = use config default)
        use_improved (bool): Use improved architectures/hyperparameters
        use_ensemble (bool): Train ensemble of MLP + XGBoost
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
    
    # Train classifier(s)
    if use_ensemble:
        print("\n" + "=" * 80)
        print("ENSEMBLE MODE: Training MLP + XGBoost")
        print("=" * 80)
        
        # Use sklearn MLP on macOS to avoid segfault
        import platform
        if platform.system() == 'Darwin':
            print("\nUsing sklearn MLP (avoids PyTorch segfault on macOS)")
            mlp_model = train_sklearn_mlp(X_train, y_train, use_improved=use_improved)
        else:
            mlp_model, _, _, _ = train_mlp_pytorch(
                X_train, y_train, X_val, y_val,
                classifier_config, len(label_encoder.classes_),
                use_improved=use_improved
            )
            mlp_model = mlp_model.cpu()
        
        # Train XGBoost
        xgb_model = train_sklearn_classifier(
            X_train, y_train, X_val, y_val,
            'xgboost', classifier_config,
            use_improved=use_improved
        )
        
        # Store both models
        model = {'mlp': mlp_model, 'xgboost': xgb_model}
        classifier_type = 'ensemble'
        
    elif classifier_type == 'mlp':
        # Use sklearn MLP on macOS to avoid segfault
        import platform
        if platform.system() == 'Darwin':
            print("\nUsing sklearn MLP (avoids PyTorch segfault on macOS)")
            model = train_sklearn_mlp(X_train, y_train, use_improved=use_improved)
        else:
            model, train_losses, val_losses, val_accs = train_mlp_pytorch(
                X_train, y_train, X_val, y_val,
                classifier_config, len(label_encoder.classes_),
                use_improved=use_improved
            )
            # Convert model to CPU for saving
            model = model.cpu()
        
    else:
        model = train_sklearn_classifier(
            X_train, y_train, X_val, y_val,
            classifier_type, classifier_config,
            use_improved=use_improved
        )
    
    # Evaluate on validation set
    print("\n" + "=" * 80)
    print("Validation Set Evaluation")
    print("=" * 80)
    
    if classifier_type == 'ensemble':
        # Ensemble prediction
        y_pred = ensemble_predictions(
            [model['mlp'], model['xgboost']],
            X_val,
            ['mlp', 'xgboost']
        )
    elif classifier_type == 'mlp':
        # Check if PyTorch or sklearn model
        if isinstance(model, nn.Module):
            # PyTorch model prediction
            model.eval()
            with torch.no_grad():
                X_val_tensor = torch.FloatTensor(X_val)
                outputs = model(X_val_tensor)
                _, y_pred = torch.max(outputs, 1)
                y_pred = y_pred.numpy()
        else:
            # sklearn MLP prediction
            y_pred = model.predict(X_val)
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
    improved_tag = "_improved" if use_improved else ""
    model_name = f"{classifier_type}{improved_tag}_{timestamp}"
    
    # Save model
    model_path = output_dir / f"{model_name}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'label_encoder': label_encoder,
            'classifier_type': classifier_type,
            'feature_dim': X.shape[1],
            'config': classifier_config,
            'improved': use_improved,
            'ensemble': use_ensemble
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
    import argparse
    
    parser = argparse.ArgumentParser(description='Train classifier on extracted features')
    parser.add_argument('--features', type=str, required=True,
                       help='Path to features .npy file')
    parser.add_argument('--labels', type=str, required=True,
                       help='Path to labels CSV file')
    parser.add_argument('--classifier', type=str, default='mlp',
                       choices=['mlp', 'xgboost', 'svm', 'random_forest'],
                       help='Classifier type')
    parser.add_argument('--improved', action='store_true',
                       help='Use improved architecture/hyperparameters')
    parser.add_argument('--ensemble', action='store_true',
                       help='Train ensemble of MLP + XGBoost')
    
    args = parser.parse_args()
    
    train_and_evaluate(
        features_file=args.features,
        labels_file=args.labels,
        classifier_type=args.classifier,
        use_improved=args.improved,
        use_ensemble=args.ensemble
    )

if __name__ == "__main__":
    main()


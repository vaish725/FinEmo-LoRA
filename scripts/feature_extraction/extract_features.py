"""
Feature Extraction Script for Logits-based Classification
Extracts embeddings/features from pre-trained LLM (frozen) to use as inputs for classifier
This is much faster than fine-tuning - no training of the LLM itself!
"""

import os
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM
)

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
        return yaml.safe_load(f)

class FeatureExtractor:
    """
    Extracts features from pre-trained language models
    Models remain frozen - no training required!
    """
    
    def __init__(self, config: dict):
        """
        Initialize feature extractor
        
        Args:
            config (dict): Project configuration
        """
        self.config = config
        self.selected_model = config['model']['selected']
        self.model_config = config['model']['feature_extractors'][self.selected_model]
        
        # Feature extraction settings
        self.pooling_strategy = config['model']['feature_extraction']['pooling_strategy']
        self.layer = config['model']['feature_extraction']['layer']
        self.normalize = config['model']['feature_extraction']['normalize']
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.model, self.tokenizer = self._load_model()
        
    def _load_model(self):
        """
        Load pre-trained model and tokenizer (frozen for feature extraction)
        
        Returns:
            tuple: (model, tokenizer)
        """
        print("=" * 80)
        print("Loading Pre-trained Model for Feature Extraction")
        print("=" * 80)
        
        model_name = self.model_config['name']
        model_type = self.model_config['type']
        
        print(f"\nModel: {model_name}")
        print(f"Type: {model_type}")
        
        # Get HuggingFace token if needed
        hf_token = os.getenv('HF_TOKEN')
        
        # Load tokenizer
        print("\nLoading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token,
            trust_remote_code=True
        )
        
        # Set padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        print(f"Tokenizer loaded: {type(tokenizer).__name__}")
        
        # Load model based on type
        print("\nLoading model...")
        
        if model_type == "encoder":
            # BERT-style encoder models (e.g., FinBERT)
            model = AutoModel.from_pretrained(
                model_name,
                token=hf_token,
                trust_remote_code=True
            )
        else:
            # Causal LM models (e.g., Llama, Phi-2)
            # Optional: use quantization for memory efficiency
            if 'quantization' in self.model_config:
                from transformers import BitsAndBytesConfig
                
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    token=hf_token,
                    trust_remote_code=True
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    token=hf_token,
                    trust_remote_code=True
                )
        
        # Move to device (if not using device_map)
        if not hasattr(model, 'hf_device_map'):
            model = model.to(self.device)
        
        # Set to evaluation mode (no training!)
        model.eval()
        
        print(f"Model loaded successfully!")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
        print("\nNote: Model is FROZEN - no training will occur")
        
        return model, tokenizer
    
    def _pool_embeddings(self, hidden_states, attention_mask):
        """
        Pool token embeddings into a single vector
        
        Args:
            hidden_states: Hidden states from model (batch_size, seq_len, hidden_dim)
            attention_mask: Attention mask (batch_size, seq_len)
            
        Returns:
            torch.Tensor: Pooled embeddings (batch_size, hidden_dim)
        """
        if self.pooling_strategy == "mean":
            # Mean pooling (average of all tokens, excluding padding)
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            return sum_embeddings / sum_mask
            
        elif self.pooling_strategy == "cls":
            # CLS token (first token)
            return hidden_states[:, 0, :]
            
        elif self.pooling_strategy == "max":
            # Max pooling
            return torch.max(hidden_states, dim=1)[0]
            
        elif self.pooling_strategy == "last":
            # Last non-padding token
            batch_size = hidden_states.shape[0]
            sequence_lengths = attention_mask.sum(dim=1) - 1
            return hidden_states[range(batch_size), sequence_lengths, :]
        
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
    
    def extract_features(self, texts: list):
        """
        Extract features from a list of texts
        
        Args:
            texts (list): List of text strings
            
        Returns:
            numpy.ndarray: Feature matrix (n_samples, feature_dim)
        """
        batch_size = self.model_config['batch_size']
        max_length = self.model_config['max_length']
        
        all_features = []
        
        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting features"):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Extract features (no gradient computation)
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                
                # Get hidden states from specified layer
                hidden_states = outputs.hidden_states[self.layer]
                
                # Pool embeddings
                features = self._pool_embeddings(hidden_states, inputs['attention_mask'])
                
                # Normalize if specified
                if self.normalize:
                    features = torch.nn.functional.normalize(features, p=2, dim=1)
                
                # Move to CPU and convert to numpy
                features = features.cpu().numpy()
                all_features.append(features)
        
        # Concatenate all batches
        all_features = np.vstack(all_features)
        
        return all_features

def extract_and_save_features(input_file: str, output_file: str, text_column: str = 'text'):
    """
    Main function to extract features from a dataset and save them
    
    Args:
        input_file (str): Path to input CSV file with texts
        output_file (str): Path to save extracted features
        text_column (str): Name of column containing text
    """
    print("=" * 80)
    print("Feature Extraction Pipeline")
    print("=" * 80)
    
    # Load configuration
    config = load_config()
    
    # Load dataset
    print(f"\nLoading dataset from: {input_file}")
    df = pd.read_csv(input_file)
    
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in dataset")
    
    print(f"Dataset loaded: {len(df)} samples")
    
    # Initialize feature extractor
    extractor = FeatureExtractor(config)
    
    # Extract features
    print("\nExtracting features...")
    texts = df[text_column].tolist()
    features = extractor.extract_features(texts)
    
    print(f"\nFeatures extracted!")
    print(f"Feature shape: {features.shape}")
    print(f"Feature dimension: {features.shape[1]}")
    
    # Save features
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as numpy array
    np.save(output_path, features)
    print(f"\nFeatures saved to: {output_path}")
    
    # Also save a metadata file
    metadata = {
        'n_samples': len(features),
        'feature_dim': features.shape[1],
        'model': config['model']['selected'],
        'model_name': config['model']['feature_extractors'][config['model']['selected']]['name'],
        'pooling': config['model']['feature_extraction']['pooling_strategy'],
        'normalized': config['model']['feature_extraction']['normalize']
    }
    
    metadata_path = output_path.with_suffix('.json')
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")
    
    return features

def main():
    """
    CLI for feature extraction
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Extract features from text data using pre-trained LLM (frozen)'
    )
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input CSV file')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to output .npy file')
    parser.add_argument('--text-column', type=str, default='text',
                       help='Name of text column in CSV (default: text)')
    
    args = parser.parse_args()
    
    extract_and_save_features(
        input_file=args.input,
        output_file=args.output,
        text_column=args.text_column
    )

if __name__ == "__main__":
    main()


"""
GoEmotions Dataset Download Script
Downloads the GoEmotions dataset from Hugging Face for transfer learning
This dataset has 27 fine-grained emotions that we'll map to our 6 economic emotions
"""

import os
import yaml
from datasets import load_dataset
import pandas as pd
from pathlib import Path
import json

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

def download_goemotions_dataset(config):
    """
    Download and save GoEmotions dataset
    
    Args:
        config (dict): Project configuration
    """
    print("=" * 80)
    print("GoEmotions Dataset Download")
    print("=" * 80)
    
    # Get dataset name from config
    dataset_name = config['datasets']['goemotions']['name']
    print(f"\nDownloading dataset: {dataset_name}")
    
    try:
        # Load dataset from Hugging Face
        # GoEmotions has a specific structure with emotion labels
        dataset = load_dataset(dataset_name, "simplified")
        
        print(f"\nDataset loaded successfully!")
        print(f"Available splits: {list(dataset.keys())}")
        
        # Create output directory
        output_dir = Path(config['paths']['data_raw']) / "goemotions"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get emotion labels
        emotion_labels = dataset['train'].features['labels'].feature.names
        print(f"\nEmotion labels in GoEmotions ({len(emotion_labels)}):")
        print(emotion_labels)
        
        # Save emotion mapping configuration
        mapping_path = output_dir / "emotion_mapping.json"
        mapping_info = {
            "original_emotions": emotion_labels,
            "target_emotions": config['emotion_labels'],
            "mapping": config['datasets']['goemotions']['emotion_mapping']
        }
        with open(mapping_path, 'w') as f:
            json.dump(mapping_info, f, indent=2)
        print(f"\nEmotion mapping saved to: {mapping_path}")
        
        # Save each split as CSV
        for split_name, split_data in dataset.items():
            df = split_data.to_pandas()
            
            # Convert label indices to emotion names
            def labels_to_emotions(label_indices):
                """Convert list of label indices to emotion names"""
                return [emotion_labels[idx] for idx in label_indices]
            
            df['emotion_names'] = df['labels'].apply(labels_to_emotions)
            
            output_path = output_dir / f"{split_name}.csv"
            df.to_csv(output_path, index=False)
            
            print(f"\nSplit: {split_name}")
            print(f"  - Samples: {len(df)}")
            print(f"  - Columns: {list(df.columns)}")
            print(f"  - Saved to: {output_path}")
            
            # Display sample
            if len(df) > 0:
                print(f"\nSample from {split_name}:")
                sample_df = df[['text', 'emotion_names']].head(3)
                for idx, row in sample_df.iterrows():
                    print(f"  Text: {row['text'][:80]}...")
                    print(f"  Emotions: {row['emotion_names']}")
                    print()
        
        print("=" * 80)
        print("GoEmotions dataset download completed successfully!")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\nError downloading dataset: {e}")
        print("\nNote: Make sure you have an internet connection and datasets library is installed")
        return False

def main():
    """
    Main function to execute the download process
    """
    # Load configuration
    config = load_config()
    
    # Download dataset
    success = download_goemotions_dataset(config)
    
    if success:
        print("\nNext steps:")
        print("  1. Review the downloaded data in data/raw/goemotions/")
        print("  2. Check emotion_mapping.json for the 27->6 emotion mapping")
        print("  3. Use this dataset for Stage 1 transfer learning")
        print("  4. The model will learn general emotion understanding here")
    else:
        print("\nPlease resolve the errors and try again.")

if __name__ == "__main__":
    main()


"""
GoEmotions Preprocessing Script
Maps GoEmotions' 27 emotions to our 6-emotion taxonomy for transfer learning
Prepares the dataset in the format required for LoRA fine-tuning
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import json
from collections import Counter

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

def map_emotions_to_target(emotion_list: list, emotion_mapping: dict) -> str:
    """
    Map a list of GoEmotions emotions to our target 6-emotion taxonomy
    
    Args:
        emotion_list (list): List of emotion strings from GoEmotions
        emotion_mapping (dict): Mapping from target emotions to GoEmotions emotions
        
    Returns:
        str: Mapped target emotion, or None if no clear mapping
    """
    if not emotion_list or len(emotion_list) == 0:
        return None
    
    # Reverse the mapping for lookup: goemotions_emotion -> target_emotion
    reverse_mapping = {}
    for target_emotion, goemotions_list in emotion_mapping.items():
        for goemotion in goemotions_list:
            if goemotion not in reverse_mapping:
                reverse_mapping[goemotion] = []
            reverse_mapping[goemotion].append(target_emotion)
    
    # Find all possible target emotions for the given emotion list
    target_candidates = []
    for emotion in emotion_list:
        if emotion in reverse_mapping:
            target_candidates.extend(reverse_mapping[emotion])
    
    if not target_candidates:
        return None
    
    # If multiple target emotions, pick the most common one
    # This handles cases where multiple GoEmotions map to different targets
    target_counts = Counter(target_candidates)
    most_common_emotion = target_counts.most_common(1)[0][0]
    
    return most_common_emotion

def preprocess_goemotions(config: dict, output_format: str = 'instruction'):
    """
    Preprocess GoEmotions dataset and map to target taxonomy
    
    Args:
        config (dict): Project configuration
        output_format (str): Format for output - 'instruction' or 'classification'
    """
    print("=" * 80)
    print("GoEmotions Preprocessing for Transfer Learning")
    print("=" * 80)
    
    # Load emotion mapping
    emotion_mapping = config['datasets']['goemotions']['emotion_mapping']
    target_emotions = config['emotion_labels']
    
    print("\nEmotion Mapping Configuration:")
    for target, sources in emotion_mapping.items():
        print(f"  {target} <- {sources}")
    
    # Input and output paths
    raw_data_dir = Path(config['paths']['data_raw']) / "goemotions"
    processed_data_dir = Path(config['paths']['data_processed']) / "goemotions"
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each split
    for split in ['train', 'validation', 'test']:
        input_file = raw_data_dir / f"{split}.csv"
        
        if not input_file.exists():
            print(f"\nWarning: {input_file} not found, skipping {split} split")
            continue
        
        print(f"\nProcessing {split} split...")
        df = pd.read_csv(input_file)
        
        print(f"  Original samples: {len(df)}")
        
        # Parse emotion_names if it's a string representation of a list
        if 'emotion_names' in df.columns:
            import ast
            df['emotion_names'] = df['emotion_names'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
        
        # Map emotions to target taxonomy
        df['target_emotion'] = df['emotion_names'].apply(
            lambda x: map_emotions_to_target(x, emotion_mapping)
        )
        
        # Filter out samples that don't map to any target emotion
        df_mapped = df[df['target_emotion'].notna()].copy()
        print(f"  Mapped samples: {len(df_mapped)}")
        print(f"  Unmapped (filtered): {len(df) - len(df_mapped)}")
        
        # Show emotion distribution
        print(f"\n  Emotion distribution in {split}:")
        emotion_counts = df_mapped['target_emotion'].value_counts()
        for emotion, count in emotion_counts.items():
            percentage = (count / len(df_mapped)) * 100
            print(f"    {emotion}: {count} ({percentage:.1f}%)")
        
        # Format for training
        if output_format == 'instruction':
            # Instruction-following format for LLM fine-tuning
            df_mapped['prompt'] = df_mapped['text'].apply(
                lambda x: f"Classify the emotion in this text: {x}\n\nEmotion:"
            )
            df_mapped['completion'] = df_mapped['target_emotion']
            
        elif output_format == 'classification':
            # Simple classification format
            df_mapped['text'] = df_mapped['text']
            df_mapped['label'] = df_mapped['target_emotion']
        
        # Save processed data
        output_file = processed_data_dir / f"{split}_mapped.csv"
        df_mapped.to_csv(output_file, index=False)
        print(f"\n  Saved to: {output_file}")
    
    # Generate summary statistics
    print("\n" + "=" * 80)
    print("Preprocessing Summary")
    print("=" * 80)
    
    summary = {
        'target_emotions': target_emotions,
        'emotion_mapping': emotion_mapping,
        'output_format': output_format,
        'splits_processed': []
    }
    
    for split in ['train', 'validation', 'test']:
        output_file = processed_data_dir / f"{split}_mapped.csv"
        if output_file.exists():
            df = pd.read_csv(output_file)
            split_summary = {
                'split': split,
                'samples': len(df),
                'emotion_distribution': df['target_emotion'].value_counts().to_dict()
            }
            summary['splits_processed'].append(split_summary)
            
            print(f"\n{split.upper()} Split:")
            print(f"  Samples: {len(df)}")
            for emotion, count in df['target_emotion'].value_counts().items():
                print(f"  {emotion}: {count}")
    
    # Save summary
    summary_file = processed_data_dir / "preprocessing_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_file}")
    
    print("\n" + "=" * 80)
    print("GoEmotions preprocessing completed!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Use this preprocessed data for Stage 1 transfer learning")
    print("  2. The model will learn to classify general emotions")
    print("  3. Then fine-tune on financial emotion data in Stage 2")

def main():
    """
    Main function to execute preprocessing
    """
    # Load configuration
    config = load_config()
    
    # Preprocess GoEmotions
    preprocess_goemotions(config, output_format='instruction')
    
    print("\nTo use this preprocessed data:")
    print("  - Training data: data/processed/goemotions/train_mapped.csv")
    print("  - Validation data: data/processed/goemotions/validation_mapped.csv")
    print("  - Test data: data/processed/goemotions/test_mapped.csv")

if __name__ == "__main__":
    main()


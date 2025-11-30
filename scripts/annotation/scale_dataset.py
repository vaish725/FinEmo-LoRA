"""
Scale Dataset to Target Balance

Uses GPT-4 to annotate additional samples, focusing on rare classes
to improve overall balance and model performance.
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from annotation.llm_annotator import annotate_dataset

def analyze_current_distribution(csv_path, target_per_class=200):
    """
    Analyze current dataset distribution
    
    Args:
        csv_path: Path to current annotated dataset
        target_per_class: Target number of samples per class
        
    Returns:
        Dictionary of needed samples per emotion and total needed
    """
    df = pd.read_csv(csv_path)
    
    print("=" * 80)
    print("CURRENT DATASET ANALYSIS")
    print("=" * 80)
    
    emotion_counts = df['emotion'].value_counts().sort_index()
    total = len(df)
    
    print(f"\nTotal samples: {total}")
    print("\nEmotion distribution:")
    for emotion, count in emotion_counts.items():
        percentage = (count / total) * 100
        print(f"  {emotion:<15} {count:>4} ({percentage:>5.1f}%)")
    
    print(f"\nTarget: {target_per_class} samples per class")
    print("\nSamples needed per emotion:")
    
    needed = {}
    for emotion in ['anxiety', 'excitement', 'fear', 'hope', 'optimism', 'uncertainty']:
        current = emotion_counts.get(emotion, 0)
        need = max(0, target_per_class - current)
        needed[emotion] = need
        status = "✅ OK" if need == 0 else f"📈 +{need}"
        print(f"  {emotion:<15} {current:>4} current → {target_per_class:>4} target  {status}")
    
    total_needed = sum(needed.values())
    current_min = emotion_counts.min()
    current_max = emotion_counts.max()
    new_min = min(target_per_class, emotion_counts.min())
    new_max = max(target_per_class, emotion_counts.max())
    
    print(f"\nTotal additional samples needed: {total_needed}")
    print(f"Current imbalance: {current_max / current_min:.1f}:1")
    print(f"After scaling: {new_max / new_min:.1f}:1")
    
    return needed, total_needed

def sample_raw_data_for_annotation(raw_csv_path, num_samples, output_path):
    """
    Sample unannotated data from raw dataset
    
    Args:
        raw_csv_path: Path to raw FinGPT data
        num_samples: Number of samples to extract
        output_path: Where to save sampled data for annotation
    """
    print("\n" + "=" * 80)
    print("SAMPLING RAW DATA FOR ANNOTATION")
    print("=" * 80)
    
    df = pd.read_csv(raw_csv_path)
    
    # Try common column names for text
    text_column = None
    for col in ['text', 'input', 'sentence', 'content', 'headline', 'news']:
        if col in df.columns:
            text_column = col
            break
    
    if text_column is None:
        print(f"Available columns: {list(df.columns)}")
        raise ValueError("Could not find text column in raw data")
    
    print(f"\nRaw dataset size: {len(df)}")
    print(f"Text column: '{text_column}'")
    
    # Sample random subset
    if len(df) > num_samples:
        sampled = df.sample(n=num_samples, random_state=42)
    else:
        sampled = df
    
    # Keep only text column and create clean CSV
    sampled = sampled[[text_column]].copy()
    sampled.rename(columns={text_column: 'text'}, inplace=True)
    
    # Remove duplicates and very short texts
    sampled = sampled.drop_duplicates(subset=['text'])
    sampled = sampled[sampled['text'].str.len() > 20]
    
    # Save
    sampled.to_csv(output_path, index=False)
    
    print(f"\nSampled {len(sampled)} texts for annotation")
    print(f"Saved to: {output_path}")
    
    return output_path

def merge_annotations(original_csv, new_csv, output_csv):
    """
    Merge original and newly annotated data
    
    Args:
        original_csv: Path to original annotated dataset
        new_csv: Path to newly annotated data
        output_csv: Path to save merged dataset
    """
    print("\n" + "=" * 80)
    print("MERGING DATASETS")
    print("=" * 80)
    
    df_original = pd.read_csv(original_csv)
    df_new = pd.read_csv(new_csv)
    
    print(f"\nOriginal dataset: {len(df_original)} samples")
    print(f"New annotations: {len(df_new)} samples")
    
    # Combine
    df_merged = pd.concat([df_original, df_new], ignore_index=True)
    
    # Remove duplicates based on text
    df_merged = df_merged.drop_duplicates(subset=['text'], keep='first')
    
    print(f"Merged (after deduplication): {len(df_merged)} samples")
    
    # Show new distribution
    print("\nNew emotion distribution:")
    emotion_counts = df_merged['emotion'].value_counts().sort_index()
    for emotion, count in emotion_counts.items():
        percentage = (count / len(df_merged)) * 100
        print(f"  {emotion:<15} {count:>4} ({percentage:>5.1f}%)")
    
    # Save
    df_merged.to_csv(output_csv, index=False)
    print(f"\nMerged dataset saved to: {output_csv}")
    
    return df_merged

def main():
    """
    Main scaling pipeline
    """
    parser = argparse.ArgumentParser(description='Scale dataset with GPT-4 annotations')
    parser.add_argument('--input', type=str, 
                        default='data/annotated/fingpt_annotated_scaled.csv',
                        help='Input annotated dataset')
    parser.add_argument('--output', type=str,
                        default='data/annotated/fingpt_annotated_balanced.csv',
                        help='Output path for balanced dataset')
    parser.add_argument('--raw-data', type=str,
                        default='data/raw/fingpt/train.csv',
                        help='Raw FinGPT data for sampling')
    parser.add_argument('--target-per-class', type=int, default=200,
                        help='Target number of samples per class')
    
    args = parser.parse_args()
    
    # Paths
    project_root = Path(__file__).parent.parent.parent
    current_annotated = project_root / args.input
    raw_data = project_root / args.raw_data
    output_path = project_root / args.output
    
    # Step 1: Analyze current distribution
    needed, total_needed = analyze_current_distribution(current_annotated, args.target_per_class)
    
    if total_needed == 0:
        print("\n✅ Dataset already has target number of samples!")
        return
    
    # Add buffer for low-confidence annotations that will be filtered out
    buffer_multiplier = 1.2
    samples_to_annotate = int(total_needed * buffer_multiplier)
    
    print(f"\nWill annotate {samples_to_annotate} samples (with {buffer_multiplier}x buffer)")
    
    # Step 2: Sample raw data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_raw_sample = project_root / 'data' / 'annotated' / f'temp_raw_sample_{timestamp}.csv'
    
    print(f"\nSampling {samples_to_annotate} texts from raw data...")
    sample_raw_data_for_annotation(raw_data, samples_to_annotate, temp_raw_sample)
    
    # Step 3: Annotate with GPT-4
    temp_annotated = project_root / 'data' / 'annotated' / f'temp_new_annotations_{timestamp}.csv'
    
    print("\n" + "=" * 80)
    print("STARTING GPT-4 ANNOTATION")
    print("=" * 80)
    print("\nThis will take approximately:")
    print(f"  Time: {samples_to_annotate * 1.5 / 60:.1f}-{samples_to_annotate * 2 / 60:.1f} minutes")
    print(f"  Cost: ~${samples_to_annotate * 0.002:.2f} (at $0.002 per sample)")
    
    input("\nPress Enter to continue or Ctrl+C to abort...")
    
    annotate_dataset(
        input_path=str(temp_raw_sample),
        output_path=str(temp_annotated),
        text_column='text',
        max_samples=samples_to_annotate,
        batch_size=50
    )
    
    # Step 4: Merge with original
    merge_annotations(current_annotated, temp_annotated, output_path)
    
    print("\n" + "=" * 80)
    print("✅ SCALING COMPLETE")
    print("=" * 80)
    print(f"\nNew dataset: {output_path}")
    print("\nNext steps:")
    print("  1. Extract features: python3 scripts/feature_extraction/extract_features.py \\")
    print(f"                        --input {output_path} \\")
    print(f"                        --output data/features/train_features_balanced.npy")
    print("  2. Train classifier: python3 scripts/classifier/train_classifier.py \\")
    print("                        --features data/features/train_features_balanced.npy \\")
    print(f"                        --labels {output_path}")

if __name__ == "__main__":
    main()

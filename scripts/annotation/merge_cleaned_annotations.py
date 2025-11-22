"""
Merge Cleaned Annotations Back Into Full Dataset
Replaces optimism samples with cleaned/relabeled versions

Usage:
    python scripts/annotation/merge_cleaned_annotations.py
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

def merge_cleaned_annotations(
    original_file: str = "data/annotated/fingpt_annotated.csv",
    cleaned_file: str = "cleaned_optimism_final.csv",
    output_file: str = "data/annotated/fingpt_annotated_v2.csv"
):
    """
    Merge cleaned optimism annotations back into full dataset
    
    Args:
        original_file: Path to original annotated dataset
        cleaned_file: Path to cleaned optimism samples
        output_file: Path to save merged dataset
    """
    print("=" * 80)
    print("MERGING CLEANED ANNOTATIONS")
    print("=" * 80)
    
    # Load datasets
    print(f"\nLoading original dataset: {original_file}")
    df_original = pd.read_csv(original_file)
    print(f"Original samples: {len(df_original)}")
    print(f"Original optimism samples: {len(df_original[df_original['emotion'] == 'optimism'])}")
    
    print(f"\nLoading cleaned optimism: {cleaned_file}")
    df_cleaned = pd.read_csv(cleaned_file)
    print(f"Cleaned samples: {len(df_cleaned)}")
    print(f"Cleaned optimism: {len(df_cleaned[df_cleaned['emotion'] == 'optimism'])}")
    print(f"Changed to uncertainty: {len(df_cleaned[df_cleaned['emotion'] == 'uncertainty'])}")
    
    # Remove old optimism samples from original
    df_no_optimism = df_original[df_original['emotion'] != 'optimism'].copy()
    print(f"\nAfter removing old optimism: {len(df_no_optimism)} samples")
    
    # Add cleaned samples back
    df_merged = pd.concat([df_no_optimism, df_cleaned], ignore_index=True)
    print(f"After adding cleaned samples: {len(df_merged)} samples")
    
    # Show emotion distribution
    print("\n" + "=" * 80)
    print("EMOTION DISTRIBUTION COMPARISON")
    print("=" * 80)
    
    print("\nORIGINAL:")
    original_dist = df_original['emotion'].value_counts().sort_index()
    for emotion, count in original_dist.items():
        print(f"  {emotion:<15} {count:>4} ({count/len(df_original)*100:>5.1f}%)")
    
    print("\nMERGED (V2):")
    merged_dist = df_merged['emotion'].value_counts().sort_index()
    for emotion, count in merged_dist.items():
        change = count - original_dist.get(emotion, 0)
        sign = "+" if change > 0 else ""
        print(f"  {emotion:<15} {count:>4} ({count/len(df_merged)*100:>5.1f}%) [{sign}{change}]")
    
    # Calculate class balance improvement
    original_imbalance = original_dist.max() / original_dist.min()
    merged_imbalance = merged_dist.max() / merged_dist.min()
    
    print(f"\nClass imbalance ratio:")
    print(f"  Original: {original_imbalance:.1f}x")
    print(f"  Merged:   {merged_imbalance:.1f}x")
    if merged_imbalance < original_imbalance:
        improvement = ((original_imbalance - merged_imbalance) / original_imbalance) * 100
        print(f"  ✓ Improvement: {improvement:.1f}% more balanced")
    
    # Save merged dataset
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df_merged.to_csv(output_path, index=False)
    print(f"\n✓ Merged dataset saved to: {output_file}")
    
    # Create backup of original
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = f"data/annotated/fingpt_annotated_backup_{timestamp}.csv"
    df_original.to_csv(backup_file, index=False)
    print(f"✓ Original backed up to: {backup_file}")
    
    # Save summary stats
    summary = {
        'timestamp': timestamp,
        'original_samples': len(df_original),
        'merged_samples': len(df_merged),
        'optimism_original': int(original_dist.get('optimism', 0)),
        'optimism_merged': int(merged_dist.get('optimism', 0)),
        'imbalance_original': float(original_imbalance),
        'imbalance_merged': float(merged_imbalance),
        'emotion_distribution': merged_dist.to_dict()
    }
    
    import json
    summary_file = f"data/annotated/merge_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary saved to: {summary_file}")
    
    print("\n" + "=" * 80)
    print("MERGE COMPLETE!")
    print("=" * 80)
    print(f"\nNext step: Retrain classifier with {output_file}")
    
    return df_merged

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Merge cleaned annotations into full dataset')
    parser.add_argument('--original', type=str, 
                       default='data/annotated/fingpt_annotated.csv',
                       help='Original annotated dataset')
    parser.add_argument('--cleaned', type=str,
                       default='cleaned_optimism_final.csv',
                       help='Cleaned optimism samples')
    parser.add_argument('--output', type=str,
                       default='data/annotated/fingpt_annotated_v2.csv',
                       help='Output file for merged dataset')
    
    args = parser.parse_args()
    
    # Check files exist
    if not Path(args.original).exists():
        print(f"Error: Original file not found: {args.original}")
        return
    
    if not Path(args.cleaned).exists():
        print(f"Error: Cleaned file not found: {args.cleaned}")
        return
    
    # Merge
    merge_cleaned_annotations(args.original, args.cleaned, args.output)

if __name__ == "__main__":
    main()

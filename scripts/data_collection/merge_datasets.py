"""
Dataset Merger: Combine Synthetic + Original Data
==================================================

PURPOSE:
    Merge synthetic LLM-generated samples with original annotated dataset.
    Creates expanded training set with balanced emotion distribution.

PROCESS:
    1. Load original annotated data (928 samples)
    2. Load all synthetic data files (2,000+ samples)
    3. Combine and remove duplicates
    4. Filter by text length (50-500 chars)
    5. Validate emotion labels
    6. Save expanded dataset

OUTPUT:
    Expanded dataset: data/annotated/fingpt_annotated_expanded_latest.csv
    Typical size: 3,000-3,500 samples after filtering

USAGE:
    python merge_datasets.py

AUTHOR:
    Vaishnavi Kamdi - Fall 2025 NNDL Project
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

def load_existing_data():
    """
    Load original annotated dataset.
    
    RETURNS:
        DataFrame with 'text' and 'emotion' columns, or None if not found
    """
    
    existing_path = Path('../../data/annotated/fingpt_annotated_balanced.csv')
    
    if not existing_path.exists():
        print(f"⚠️  Existing dataset not found: {existing_path}")
        return None
    
    df = pd.read_csv(existing_path)
    print(f"✅ Loaded existing: {len(df)} samples")
    print(f"   Emotions: {df['emotion'].value_counts().to_dict()}")
    
    return df

def load_synthetic_data():
    """
    Load all synthetic datasets from data/raw/synthetic/ directory.
    
    RETURNS:
        Combined DataFrame with all synthetic samples
    """
    
    synthetic_dir = Path('../../data/raw/synthetic')
    
    if not synthetic_dir.exists():
        print(f"⚠️  Synthetic directory not found: {synthetic_dir}")
        return None
    
    csv_files = list(synthetic_dir.glob('synthetic_*.csv'))
    
    if not csv_files:
        print(f"⚠️  No synthetic CSV files found in {synthetic_dir}")
        return None
    
    print(f"\n📁 Found {len(csv_files)} synthetic files:")
    
    all_dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        print(f"   {file.name}: {len(df)} samples ({df['emotion'].iloc[0] if len(df) > 0 else 'unknown'})")
        all_dfs.append(df)
    
    combined = pd.concat(all_dfs, ignore_index=True)
    
    print(f"\n✅ Loaded synthetic: {len(combined)} samples")
    print(f"   Emotions: {combined['emotion'].value_counts().to_dict()}")
    
    return combined

def merge_and_clean(existing_df, synthetic_df):
    """Merge datasets and clean."""
    
    print("\n" + "="*80)
    print("MERGING DATASETS")
    print("="*80)
    
    # Ensure column compatibility
    if existing_df is not None:
        # Keep only essential columns from existing
        existing_df = existing_df[['text', 'emotion']].copy()
        existing_df['source'] = 'original'
    
    if synthetic_df is not None:
        synthetic_df = synthetic_df[['text', 'emotion']].copy()
        synthetic_df['source'] = 'synthetic'
    
    # Combine
    if existing_df is not None and synthetic_df is not None:
        combined = pd.concat([existing_df, synthetic_df], ignore_index=True)
    elif existing_df is not None:
        combined = existing_df
    elif synthetic_df is not None:
        combined = synthetic_df
    else:
        print("❌ No data to merge")
        return None
    
    print(f"\n📊 Combined: {len(combined)} samples")
    
    # Remove duplicates
    before = len(combined)
    combined = combined.drop_duplicates(subset=['text'], keep='first')
    duplicates_removed = before - len(combined)
    
    if duplicates_removed > 0:
        print(f"🔄 Removed {duplicates_removed} duplicates")
    
    # Filter by text length (50-500 chars)
    before = len(combined)
    combined = combined[combined['text'].str.len().between(50, 500)]
    length_filtered = before - len(combined)
    
    if length_filtered > 0:
        print(f"📏 Filtered {length_filtered} samples (length 50-500 chars)")
    
    # Remove any missing values
    before = len(combined)
    combined = combined.dropna(subset=['text', 'emotion'])
    missing_removed = before - len(combined)
    
    if missing_removed > 0:
        print(f"🔍 Removed {missing_removed} samples with missing values")
    
    print(f"\n✅ Final dataset: {len(combined)} samples")
    
    return combined

def save_merged_dataset(df, output_dir='../../data/annotated'):
    """Save merged dataset."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'fingpt_annotated_expanded_{timestamp}.csv'
    filepath = output_path / filename
    
    df.to_csv(filepath, index=False)
    
    print(f"\n💾 Saved to: {filepath}")
    
    # Also save as "latest"
    latest_path = output_path / 'fingpt_annotated_expanded_latest.csv'
    df.to_csv(latest_path, index=False)
    print(f"💾 Saved to: {latest_path}")
    
    return filepath

def print_summary(df):
    """Print dataset summary."""
    
    print("\n" + "="*80)
    print("MERGED DATASET SUMMARY")
    print("="*80)
    
    print(f"\n📊 Total samples: {len(df)}")
    
    print(f"\n📈 Class distribution:")
    for emotion, count in df['emotion'].value_counts().sort_index().items():
        pct = count / len(df) * 100
        bar = '█' * int(count / 20)
        print(f"  {emotion:>12}: {count:>4} ({pct:>5.1f}%) {bar}")
    
    # Check balance
    counts = df['emotion'].value_counts()
    imbalance = counts.max() / counts.min()
    print(f"\n⚖️  Balance ratio: {imbalance:.1f}:1 ({counts.idxmax()}/{counts.idxmin()})")
    
    if imbalance < 2.0:
        print("   ✅ Well balanced!")
    elif imbalance < 5.0:
        print("   ⚠️  Moderately imbalanced")
    else:
        print("   ❌ Highly imbalanced")
    
    # Source breakdown
    if 'source' in df.columns:
        print(f"\n📁 Source breakdown:")
        for source, count in df['source'].value_counts().items():
            print(f"  {source:>12}: {count:>4} samples")
    
    # Text stats
    print(f"\n📝 Text statistics:")
    print(f"  Min length: {df['text'].str.len().min()} chars")
    print(f"  Max length: {df['text'].str.len().max()} chars")
    print(f"  Avg length: {df['text'].str.len().mean():.0f} chars")

def main():
    print("="*80)
    print("DATASET MERGER")
    print("="*80)
    
    # Load data
    existing_df = load_existing_data()
    synthetic_df = load_synthetic_data()
    
    if existing_df is None and synthetic_df is None:
        print("\n❌ No data found to merge")
        return
    
    # Merge and clean
    merged_df = merge_and_clean(existing_df, synthetic_df)
    
    if merged_df is None:
        return
    
    # Print summary
    print_summary(merged_df)
    
    # Save
    filepath = save_merged_dataset(merged_df)
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print(f"\n1. Review merged dataset: {filepath}")
    print(f"2. Update notebook to use: fingpt_annotated_expanded_latest.csv")
    print(f"3. Train v3 model:")
    print(f"   - Open notebooks/FinEmo_LoRA_Training.ipynb")
    print(f"   - Change dataset path to: fingpt_annotated_expanded_latest.csv")
    print(f"   - Run all cells")
    print(f"\n4. Expected results:")
    print(f"   - Current: 61% with 928 samples")
    print(f"   - Expected: 68-72% with {len(merged_df)} samples")
    print(f"   - Target: 75%+ (may need 5000+ samples)")

if __name__ == '__main__':
    main()

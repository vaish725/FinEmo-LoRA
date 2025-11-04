"""
Alternative Financial Dataset Download Script
Uses publicly available financial sentiment dataset as alternative to FinGPT
"""

import pandas as pd
from pathlib import Path
import yaml

def load_config():
    """Load project configuration"""
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def download_alternative_dataset(config):
    """
    Download alternative financial dataset
    Using Twitter Financial News Sentiment dataset (publicly available)
    """
    print("=" * 80)
    print("Alternative Financial Dataset Download")
    print("=" * 80)
    
    from datasets import load_dataset
    
    print("\nDownloading: zeroshot/twitter-financial-news-sentiment")
    print("This is a publicly available financial sentiment dataset")
    
    try:
        # Load dataset
        dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")
        
        print(f"\nDataset loaded successfully!")
        print(f"Available splits: {list(dataset.keys())}")
        
        # Create output directory
        output_dir = Path(config['paths']['data_raw']) / "fingpt"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save train split
        if 'train' in dataset:
            df = dataset['train'].to_pandas()
            output_path = output_dir / "train.csv"
            df.to_csv(output_path, index=False)
            
            print(f"\nTrain split:")
            print(f"  - Samples: {len(df)}")
            print(f"  - Columns: {list(df.columns)}")
            print(f"  - Saved to: {output_path}")
            
            # Display sample
            print(f"\nSample data:")
            print(df.head(2))
        
        print("\n" + "=" * 80)
        print("Dataset download completed successfully!")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\nError: {e}")
        return False

def main():
    config = load_config()
    download_alternative_dataset(config)

if __name__ == "__main__":
    main()


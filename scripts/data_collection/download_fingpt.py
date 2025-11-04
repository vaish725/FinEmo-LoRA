"""
FinGPT Dataset Download Script
Downloads the FinGPT sentiment training dataset from Hugging Face
This dataset provides financial text that we'll re-annotate with our 6-emotion taxonomy
"""

import os
import yaml
from datasets import load_dataset
import pandas as pd
from pathlib import Path

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

def download_fingpt_dataset(config):
    """
    Download and save FinGPT sentiment dataset
    
    Args:
        config (dict): Project configuration
    """
    print("=" * 80)
    print("FinGPT Dataset Download")
    print("=" * 80)
    
    # Get dataset name from config
    dataset_name = config['datasets']['fingpt']['name']
    print(f"\nDownloading dataset: {dataset_name}")
    
    try:
        # Load dataset from Hugging Face
        # Try multiple possible dataset names
        try:
            dataset = load_dataset(dataset_name)
        except:
            # Fallback to alternative financial sentiment dataset
            print(f"  Primary dataset not accessible, trying alternative...")
            dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")
        
        print(f"\nDataset loaded successfully!")
        print(f"Available splits: {list(dataset.keys())}")
        
        # Create output directory
        output_dir = Path(config['paths']['data_raw']) / "fingpt"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each split as CSV
        for split_name, split_data in dataset.items():
            df = split_data.to_pandas()
            output_path = output_dir / f"{split_name}.csv"
            df.to_csv(output_path, index=False)
            
            print(f"\nSplit: {split_name}")
            print(f"  - Samples: {len(df)}")
            print(f"  - Columns: {list(df.columns)}")
            print(f"  - Saved to: {output_path}")
            
            # Display sample
            if len(df) > 0:
                print(f"\nSample from {split_name}:")
                print(df.head(2))
        
        print("\n" + "=" * 80)
        print("FinGPT dataset download completed successfully!")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\nError downloading dataset: {e}")
        print("\nNote: If the dataset path is incorrect, please check Hugging Face Hub")
        print("      and update the dataset name in config.yaml")
        return False

def main():
    """
    Main function to execute the download process
    """
    # Load configuration
    config = load_config()
    
    # Download dataset
    success = download_fingpt_dataset(config)
    
    if success:
        print("\nNext steps:")
        print("  1. Review the downloaded data in data/raw/fingpt/")
        print("  2. Run the annotation pipeline to add emotion labels")
        print("  3. Check that the text column contains financial news/content")
    else:
        print("\nPlease resolve the errors and try again.")

if __name__ == "__main__":
    main()


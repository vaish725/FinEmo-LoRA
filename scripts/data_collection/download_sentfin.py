"""
SEntFiN 1.0 Dataset Download Script
Downloads the SEntFiN dataset from Kaggle
This dataset provides entity-aware financial sentiment that we'll re-annotate with emotions
"""

import os
import yaml
import subprocess
from pathlib import Path
import pandas as pd
import shutil

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

def check_kaggle_credentials():
    """
    Check if Kaggle API credentials are configured
    
    Returns:
        bool: True if credentials exist, False otherwise
    """
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    
    if not kaggle_json.exists():
        print("\nKaggle API credentials not found!")
        print("\nTo set up Kaggle API:")
        print("  1. Go to https://www.kaggle.com/settings/account")
        print("  2. Scroll to 'API' section and click 'Create New Token'")
        print("  3. Download kaggle.json")
        print("  4. Place it in ~/.kaggle/kaggle.json")
        print("  5. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False
    
    return True

def download_sentfin_dataset(config):
    """
    Download and save SEntFiN dataset from Kaggle
    
    Args:
        config (dict): Project configuration
    """
    print("=" * 80)
    print("SEntFiN 1.0 Dataset Download")
    print("=" * 80)
    
    # Check credentials first
    if not check_kaggle_credentials():
        return False
    
    # Get dataset name from config
    dataset_name = config['datasets']['sentfin']['kaggle_dataset']
    print(f"\nDownloading dataset: {dataset_name}")
    
    try:
        # Create output directory
        output_dir = Path(config['paths']['data_raw']) / "sentfin"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Download using Kaggle API
        print("\nDownloading from Kaggle...")
        subprocess.run([
            "kaggle", "datasets", "download",
            "-d", dataset_name,
            "-p", str(output_dir),
            "--unzip"
        ], check=True)
        
        print("\nDataset downloaded successfully!")
        
        # List downloaded files
        files = list(output_dir.glob("*.csv"))
        print(f"\nDownloaded files ({len(files)}):")
        
        for file_path in files:
            print(f"  - {file_path.name}")
            
            # Load and display sample
            try:
                df = pd.read_csv(file_path, nrows=1000)  # Load first 1000 rows for inspection
                print(f"    Samples: {len(df)}")
                print(f"    Columns: {list(df.columns)}")
                
                # Display sample
                if len(df) > 0:
                    print(f"\n    Sample from {file_path.name}:")
                    print(df.head(2))
                    
            except Exception as e:
                print(f"    Could not read file: {e}")
        
        print("\n" + "=" * 80)
        print("SEntFiN dataset download completed successfully!")
        print("=" * 80)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\nError downloading dataset: {e}")
        print("\nPlease check:")
        print("  1. Kaggle API is installed: pip install kaggle")
        print("  2. Dataset name is correct in config.yaml")
        print("  3. You have accepted the dataset terms on Kaggle website")
        return False
        
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return False

def main():
    """
    Main function to execute the download process
    """
    # Load configuration
    config = load_config()
    
    # Download dataset
    success = download_sentfin_dataset(config)
    
    if success:
        print("\nNext steps:")
        print("  1. Review the downloaded data in data/raw/sentfin/")
        print("  2. Identify the text column for annotation")
        print("  3. Run the annotation pipeline to add emotion labels")
    else:
        print("\nPlease resolve the errors and try again.")

if __name__ == "__main__":
    main()


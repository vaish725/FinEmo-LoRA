"""
Stage 2 Training: Fine-tuning on Financial Emotion Dataset
Fine-tune the Stage 1 model on LLM-annotated financial text with 6-emotion taxonomy
This specializes the model for economic emotion interpretation
"""

import os
import yaml
import torch
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import PeftModel
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split

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
        config = yaml.safe_load(f)
    return config

def load_stage1_model(config: dict):
    """
    Load the Stage 1 trained model with LoRA adapters
    
    Args:
        config (dict): Project configuration
        
    Returns:
        tuple: (model, tokenizer)
    """
    print("=" * 80)
    print("Loading Stage 1 Model")
    print("=" * 80)
    
    stage1_model_dir = Path(config['paths']['models']) / "stage1_final"
    
    if not stage1_model_dir.exists():
        raise FileNotFoundError(
            f"Stage 1 model not found at {stage1_model_dir}. "
            "Please complete Stage 1 training first."
        )
    
    print(f"\nLoading model from: {stage1_model_dir}")
    
    # Get HuggingFace token if needed
    hf_token = os.getenv('HF_TOKEN')
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(stage1_model_dir),
        token=hf_token,
        trust_remote_code=True
    )
    
    # Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model (this will be a PeftModel with LoRA adapters)
    # We continue training from the Stage 1 checkpoint
    from transformers import BitsAndBytesConfig
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load base model first
    selected_model = config['model']['selected']
    model_config = config['model']['base_models'][selected_model]
    base_model_name = model_config['name']
    
    print(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token,
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    
    # Load LoRA adapters from Stage 1
    print(f"Loading LoRA adapters from Stage 1...")
    model = PeftModel.from_pretrained(base_model, str(stage1_model_dir))
    
    print(f"\nStage 1 model loaded successfully!")
    print(f"  Parameters: {model.num_parameters() / 1e9:.2f}B")
    
    return model, tokenizer

def prepare_financial_dataset(config: dict, tokenizer):
    """
    Load and prepare the annotated financial emotion dataset
    
    Args:
        config (dict): Project configuration
        tokenizer: Tokenizer for encoding
        
    Returns:
        dict: Dataset dictionary with train/val/test splits
    """
    print("\n" + "=" * 80)
    print("Loading Financial Emotion Dataset")
    print("=" * 80)
    
    annotated_dir = Path(config['paths']['data_annotated'])
    
    # Look for high-confidence annotated data
    # This assumes you've run the annotation pipeline
    annotated_files = list(annotated_dir.glob("*_high_confidence.csv"))
    
    if not annotated_files:
        # Fallback: look for any annotated files
        annotated_files = list(annotated_dir.glob("*_annotated.csv"))
    
    if not annotated_files:
        raise FileNotFoundError(
            f"No annotated data found in {annotated_dir}. "
            "Please run the annotation pipeline first."
        )
    
    print(f"\nFound annotated files:")
    for f in annotated_files:
        print(f"  - {f.name}")
    
    # Load and combine all annotated data
    dfs = []
    for file in annotated_files:
        df = pd.read_csv(file)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal annotated samples: {len(combined_df)}")
    
    # Filter by confidence if available
    if 'confidence' in combined_df.columns:
        min_conf = config['annotation']['min_confidence']
        combined_df = combined_df[combined_df['confidence'] >= min_conf]
        print(f"After confidence filtering (>= {min_conf}): {len(combined_df)}")
    
    # Check required columns
    if 'text' not in combined_df.columns or 'emotion' not in combined_df.columns:
        raise ValueError("Annotated data must have 'text' and 'emotion' columns")
    
    # Show emotion distribution
    print("\nEmotion distribution:")
    emotion_counts = combined_df['emotion'].value_counts()
    for emotion, count in emotion_counts.items():
        percentage = (count / len(combined_df)) * 100
        print(f"  {emotion}: {count} ({percentage:.1f}%)")
    
    # Create train/val/test splits
    test_size = config['evaluation']['test_size']
    val_size = config['evaluation']['validation_size']
    
    # First split: separate test set
    train_val_df, test_df = train_test_split(
        combined_df,
        test_size=test_size,
        stratify=combined_df['emotion'],
        random_state=config['training']['seed']
    )
    
    # Second split: separate validation set from train
    val_ratio = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio,
        stratify=train_val_df['emotion'],
        random_state=config['training']['seed']
    )
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_df)}")
    print(f"  Validation: {len(val_df)}")
    print(f"  Test: {len(test_df)}")
    
    # Save splits for reproducibility
    splits_dir = Path(config['paths']['data_processed']) / "financial_emotion_splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(splits_dir / "train.csv", index=False)
    val_df.to_csv(splits_dir / "validation.csv", index=False)
    test_df.to_csv(splits_dir / "test.csv", index=False)
    print(f"\nSplits saved to: {splits_dir}")
    
    # Create Hugging Face datasets
    dataset_dict = load_dataset('csv', data_files={
        'train': str(splits_dir / "train.csv"),
        'validation': str(splits_dir / "validation.csv"),
        'test': str(splits_dir / "test.csv")
    })
    
    # Tokenization function
    def preprocess_function(examples):
        """
        Preprocess examples into instruction format
        """
        # Create prompts
        prompts = [
            f"Classify the economic emotion in this financial text: {text}\n\nEmotion: {emotion}{tokenizer.eos_token}"
            for text, emotion in zip(examples['text'], examples['emotion'])
        ]
        
        # Tokenize
        model_inputs = tokenizer(
            prompts,
            max_length=config['training']['stage2']['max_seq_length'],
            truncation=True,
            padding='max_length',
            return_tensors=None
        )
        
        # Set labels for causal language modeling
        model_inputs['labels'] = model_inputs['input_ids'].copy()
        
        return model_inputs
    
    # Apply tokenization
    print("\nTokenizing dataset...")
    tokenized_datasets = dataset_dict.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset_dict['train'].column_names,
        desc="Tokenizing"
    )
    
    print("Dataset preparation complete!")
    
    return tokenized_datasets

def train_stage2(config: dict):
    """
    Execute Stage 2 training: Fine-tuning on financial emotions
    
    Args:
        config (dict): Project configuration
    """
    print("\n" + "=" * 80)
    print("STAGE 2 TRAINING: Fine-tuning on Financial Emotions")
    print("=" * 80)
    
    # Set random seed
    torch.manual_seed(config['training']['seed'])
    
    # Load Stage 1 model
    model, tokenizer = load_stage1_model(config)
    
    # Prepare financial dataset
    tokenized_datasets = prepare_financial_dataset(config, tokenizer)
    
    # Configure training arguments
    stage2_config = config['training']['stage2']
    
    output_dir = Path(config['training']['output_dir']) / "stage2_financial"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=stage2_config['epochs'],
        per_device_train_batch_size=stage2_config['batch_size'],
        per_device_eval_batch_size=stage2_config['batch_size'],
        gradient_accumulation_steps=stage2_config['gradient_accumulation_steps'],
        learning_rate=stage2_config['learning_rate'],
        warmup_steps=stage2_config['warmup_steps'],
        logging_steps=config['training']['logging_steps'],
        evaluation_strategy=config['training']['evaluation_strategy'],
        save_strategy=config['training']['save_strategy'],
        fp16=True if config['training']['mixed_precision'] == 'fp16' else False,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=2,
        seed=config['training']['seed'],
    )
    
    print("\n" + "=" * 80)
    print("Training Configuration")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Epochs: {stage2_config['epochs']}")
    print(f"Batch size: {stage2_config['batch_size']}")
    print(f"Gradient accumulation: {stage2_config['gradient_accumulation_steps']}")
    print(f"Effective batch size: {stage2_config['batch_size'] * stage2_config['gradient_accumulation_steps']}")
    print(f"Learning rate: {stage2_config['learning_rate']}")
    print(f"Max sequence length: {stage2_config['max_seq_length']}")
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        tokenizer=tokenizer,
    )
    
    # Start training
    print("\n" + "=" * 80)
    print("Starting Stage 2 Training...")
    print("=" * 80)
    
    start_time = datetime.now()
    trainer.train()
    end_time = datetime.now()
    
    training_duration = end_time - start_time
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Duration: {training_duration}")
    
    # Save final model
    final_model_dir = Path(config['training']['final_model_dir'])
    final_model_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving final FinEmo-LoRA model to: {final_model_dir}")
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))
    
    # Merge LoRA adapters into base model for easier deployment (optional)
    print("\nMerging LoRA adapters with base model...")
    merged_model = model.merge_and_unload()
    merged_dir = final_model_dir / "merged"
    merged_dir.mkdir(exist_ok=True)
    merged_model.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))
    print(f"Merged model saved to: {merged_dir}")
    
    print("\n" + "=" * 80)
    print("Stage 2 Training Complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Evaluate model on test set using scripts/evaluation/evaluate.py")
    print("  2. Generate confusion matrix and per-class metrics")
    print("  3. Test model on sample financial texts")
    print(f"  4. Final model saved at: {final_model_dir}")

def main():
    """
    Main function to execute Stage 2 training
    """
    # Load configuration
    config = load_config()
    
    # Check prerequisites
    stage1_model_dir = Path(config['paths']['models']) / "stage1_final"
    if not stage1_model_dir.exists():
        print("Error: Stage 1 model not found")
        print("Please complete Stage 1 training first:")
        print("  python scripts/training/train_stage1_goemotions.py")
        return
    
    annotated_dir = Path(config['paths']['data_annotated'])
    if not annotated_dir.exists() or not list(annotated_dir.glob("*.csv")):
        print("Error: Annotated financial data not found")
        print("Please run the annotation pipeline first:")
        print("  python scripts/annotation/llm_annotator.py")
        return
    
    # Start training
    train_stage2(config)

if __name__ == "__main__":
    main()


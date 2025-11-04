"""
Stage 1 Training: Transfer Learning on GoEmotions
Fine-tune base model (Llama 3.1 8B or Phi-2) on mapped GoEmotions dataset
This helps the model learn general emotion classification before financial domain adaptation
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
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from datasets import load_dataset
import pandas as pd

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

def load_model_and_tokenizer(config: dict):
    """
    Load base model and tokenizer with quantization
    
    Args:
        config (dict): Project configuration
        
    Returns:
        tuple: (model, tokenizer)
    """
    print("=" * 80)
    print("Loading Base Model and Tokenizer")
    print("=" * 80)
    
    # Get selected model configuration
    selected_model = config['model']['selected']
    model_config = config['model']['base_models'][selected_model]
    model_name = model_config['name']
    
    print(f"\nSelected model: {selected_model}")
    print(f"Model name: {model_name}")
    
    # Get HuggingFace token if needed (for gated models like Llama)
    hf_token = os.getenv('HF_TOKEN')
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token,
        trust_remote_code=True
    )
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"Tokenizer loaded: {type(tokenizer).__name__}")
    print(f"  Vocab size: {len(tokenizer)}")
    print(f"  Padding token: {tokenizer.pad_token}")
    
    # Configure 4-bit quantization for memory efficiency (QLoRA)
    from transformers import BitsAndBytesConfig
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # Enable 4-bit quantization
        bnb_4bit_quant_type="nf4",  # Normal float 4-bit
        bnb_4bit_compute_dtype=torch.float16,  # Compute dtype
        bnb_4bit_use_double_quant=True,  # Nested quantization for more memory savings
    )
    
    # Load model with quantization
    print("\nLoading model with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",  # Automatically distribute across available GPUs
        token=hf_token,
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    
    # Prepare model for training (gradient checkpointing, etc.)
    model = prepare_model_for_kbit_training(model)
    
    print(f"Model loaded: {type(model).__name__}")
    print(f"  Parameters: {model.num_parameters() / 1e9:.2f}B")
    print(f"  Device: {model.device}")
    
    return model, tokenizer

def setup_lora(model, config: dict):
    """
    Configure and apply LoRA to the model
    
    Args:
        model: Base model
        config (dict): Project configuration
        
    Returns:
        model: Model with LoRA adapters
    """
    print("\n" + "=" * 80)
    print("Configuring LoRA")
    print("=" * 80)
    
    lora_config = config['model']['lora']
    
    # Create LoRA configuration
    peft_config = LoraConfig(
        r=lora_config['r'],  # LoRA rank
        lora_alpha=lora_config['lora_alpha'],  # LoRA scaling
        target_modules=lora_config['target_modules'],  # Which layers to adapt
        lora_dropout=lora_config['lora_dropout'],
        bias=lora_config['bias'],
        task_type=lora_config['task_type']
    )
    
    print("\nLoRA Configuration:")
    print(f"  Rank (r): {lora_config['r']}")
    print(f"  Alpha: {lora_config['lora_alpha']}")
    print(f"  Target modules: {lora_config['target_modules']}")
    print(f"  Dropout: {lora_config['lora_dropout']}")
    
    # Apply LoRA to model
    model = get_peft_model(model, peft_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_percentage = (trainable_params / total_params) * 100
    
    print(f"\nTrainable Parameters:")
    print(f"  Trainable: {trainable_params:,} ({trainable_percentage:.2f}%)")
    print(f"  Total: {total_params:,}")
    print(f"  Memory efficiency: {100 - trainable_percentage:.2f}% reduction")
    
    return model

def load_and_prepare_dataset(config: dict, tokenizer):
    """
    Load preprocessed GoEmotions dataset and prepare for training
    
    Args:
        config (dict): Project configuration
        tokenizer: Tokenizer for encoding
        
    Returns:
        dict: Dataset dictionary with train/val splits
    """
    print("\n" + "=" * 80)
    print("Loading and Preparing Dataset")
    print("=" * 80)
    
    data_dir = Path(config['paths']['data_processed']) / "goemotions"
    
    # Load train and validation splits
    train_file = data_dir / "train_mapped.csv"
    val_file = data_dir / "validation_mapped.csv"
    
    if not train_file.exists() or not val_file.exists():
        raise FileNotFoundError(
            "Preprocessed GoEmotions data not found. "
            "Please run preprocess_goemotions.py first."
        )
    
    print(f"\nLoading data from: {data_dir}")
    
    # Load as Hugging Face datasets
    dataset_dict = load_dataset('csv', data_files={
        'train': str(train_file),
        'validation': str(val_file)
    })
    
    print(f"Train samples: {len(dataset_dict['train'])}")
    print(f"Validation samples: {len(dataset_dict['validation'])}")
    
    # Tokenization function
    def preprocess_function(examples):
        """
        Preprocess examples by combining prompt and completion
        Format: [prompt] [completion] [EOS]
        """
        # Combine prompt and completion
        texts = [
            f"{prompt} {completion}{tokenizer.eos_token}"
            for prompt, completion in zip(examples['prompt'], examples['completion'])
        ]
        
        # Tokenize
        model_inputs = tokenizer(
            texts,
            max_length=config['training']['stage1']['max_seq_length'],
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

def train_stage1(config: dict):
    """
    Execute Stage 1 training: Transfer learning on GoEmotions
    
    Args:
        config (dict): Project configuration
    """
    print("\n" + "=" * 80)
    print("STAGE 1 TRAINING: Transfer Learning on GoEmotions")
    print("=" * 80)
    
    # Set random seed for reproducibility
    torch.manual_seed(config['training']['seed'])
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)
    
    # Setup LoRA
    model = setup_lora(model, config)
    
    # Load and prepare dataset
    tokenized_datasets = load_and_prepare_dataset(config, tokenizer)
    
    # Configure training arguments
    stage1_config = config['training']['stage1']
    
    output_dir = Path(config['training']['output_dir']) / "stage1_goemotions"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=stage1_config['epochs'],
        per_device_train_batch_size=stage1_config['batch_size'],
        per_device_eval_batch_size=stage1_config['batch_size'],
        gradient_accumulation_steps=stage1_config['gradient_accumulation_steps'],
        learning_rate=stage1_config['learning_rate'],
        warmup_steps=stage1_config['warmup_steps'],
        logging_steps=config['training']['logging_steps'],
        evaluation_strategy=config['training']['evaluation_strategy'],
        save_strategy=config['training']['save_strategy'],
        fp16=True if config['training']['mixed_precision'] == 'fp16' else False,
        report_to=["tensorboard"],  # Log to tensorboard
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=2,  # Keep only 2 best checkpoints
        seed=config['training']['seed'],
    )
    
    print("\n" + "=" * 80)
    print("Training Configuration")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Epochs: {stage1_config['epochs']}")
    print(f"Batch size: {stage1_config['batch_size']}")
    print(f"Gradient accumulation: {stage1_config['gradient_accumulation_steps']}")
    print(f"Effective batch size: {stage1_config['batch_size'] * stage1_config['gradient_accumulation_steps']}")
    print(f"Learning rate: {stage1_config['learning_rate']}")
    print(f"Max sequence length: {stage1_config['max_seq_length']}")
    
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
    print("Starting Training...")
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
    final_model_dir = Path(config['paths']['models']) / "stage1_final"
    final_model_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving final model to: {final_model_dir}")
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))
    
    print("\n" + "=" * 80)
    print("Stage 1 Training Complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Review training logs in: logs/")
    print("  2. Evaluate model on GoEmotions test set")
    print("  3. Proceed to Stage 2: Fine-tuning on financial emotion data")
    print(f"  4. Model saved at: {final_model_dir}")

def main():
    """
    Main function to execute Stage 1 training
    """
    # Load configuration
    config = load_config()
    
    # Check if preprocessed data exists
    data_dir = Path(config['paths']['data_processed']) / "goemotions"
    if not data_dir.exists():
        print("Error: Preprocessed GoEmotions data not found")
        print("Please run: python scripts/data_collection/preprocess_goemotions.py")
        return
    
    # Start training
    train_stage1(config)

if __name__ == "__main__":
    main()


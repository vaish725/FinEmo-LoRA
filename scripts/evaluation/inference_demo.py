"""
Inference Demo Script
Test the trained FinEmo-LoRA model on sample financial texts
"""

import os
import yaml
import torch
from pathlib import Path
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load environment variables
load_dotenv()

def load_config():
    """Load project configuration"""
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_model_for_inference():
    """Load the trained FinEmo-LoRA model"""
    config = load_config()
    
    # Try to load merged model first (faster)
    model_dir = Path(config['training']['final_model_dir'])
    merged_dir = model_dir / "merged"
    
    if merged_dir.exists():
        print(f"Loading merged model from: {merged_dir}")
        model_path = merged_dir
        is_peft = False
    elif model_dir.exists():
        print(f"Loading LoRA model from: {model_dir}")
        model_path = model_dir
        is_peft = True
    else:
        raise FileNotFoundError(
            "Trained model not found. Please complete training first."
        )
    
    hf_token = os.getenv('HF_TOKEN')
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        token=hf_token,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    if is_peft:
        selected_model = config['model']['selected']
        base_model_name = config['model']['base_models'][selected_model]['name']
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            token=hf_token,
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        model = PeftModel.from_pretrained(base_model, str(model_path))
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            device_map="auto",
            token=hf_token,
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
    
    model.eval()
    return model, tokenizer

def predict_emotion(text, model, tokenizer, emotion_labels):
    """
    Predict emotion for a given text
    
    Args:
        text (str): Financial text to analyze
        model: Trained model
        tokenizer: Tokenizer
        emotion_labels (list): List of possible emotions
        
    Returns:
        str: Predicted emotion
    """
    # Create prompt
    prompt = f"Classify the economic emotion in this financial text: {text}\n\nEmotion:"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract emotion
    if "Emotion:" in generated_text:
        prediction = generated_text.split("Emotion:")[-1].strip()
    else:
        prediction = generated_text[len(prompt):].strip()
    
    # Match to emotion label
    prediction_lower = prediction.lower()
    for emotion in emotion_labels:
        if emotion.lower() in prediction_lower:
            return emotion
    
    # Default if no match
    return prediction.split()[0] if prediction.split() else "unknown"

def main():
    """Main demo function"""
    print("=" * 80)
    print("FinEmo-LoRA Inference Demo")
    print("=" * 80)
    
    # Load model
    print("\nLoading model...")
    model, tokenizer = load_model_for_inference()
    print("Model loaded successfully!")
    
    # Get emotion labels
    config = load_config()
    emotion_labels = config['emotion_labels']
    
    print(f"\nEmotion labels: {', '.join(emotion_labels)}")
    
    # Sample financial texts for testing
    sample_texts = [
        "Apple's stock surged 15% after announcing record-breaking quarterly earnings, exceeding analyst expectations.",
        
        "Investors are growing increasingly worried about the Federal Reserve's aggressive interest rate hikes and potential recession.",
        
        "Markets remain uncertain as geopolitical tensions continue, with traders struggling to predict the next move.",
        
        "Tech startups are buzzing with excitement following the breakthrough in AI technology that could revolutionize the industry.",
        
        "Economists express cautious optimism about economic recovery, citing improving employment figures and consumer spending.",
        
        "Bank failures and credit concerns are sparking widespread panic among investors, triggering massive sell-offs."
    ]
    
    # Predict emotions for sample texts
    print("\n" + "=" * 80)
    print("Sample Predictions")
    print("=" * 80)
    
    for i, text in enumerate(sample_texts, 1):
        print(f"\n{i}. Text:")
        print(f"   {text}")
        
        emotion = predict_emotion(text, model, tokenizer, emotion_labels)
        print(f"   Predicted Emotion: {emotion}")
    
    # Interactive mode
    print("\n" + "=" * 80)
    print("Interactive Mode")
    print("=" * 80)
    print("Enter financial texts to classify (type 'quit' to exit):\n")
    
    while True:
        user_input = input("Your text: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nExiting demo. Goodbye!")
            break
        
        if not user_input:
            continue
        
        emotion = predict_emotion(user_input, model, tokenizer, emotion_labels)
        print(f"Predicted Emotion: {emotion}\n")

if __name__ == "__main__":
    main()


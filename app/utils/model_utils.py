"""
Model loading and inference utilities
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import streamlit as st
from pathlib import Path

# Emotion labels
EMOTIONS = ['anxiety', 'excitement', 'fear', 'hope', 'optimism', 'uncertainty']

# Emotion descriptions
EMOTION_DESCRIPTIONS = {
    'anxiety': 'Nervousness, worry, or concern about financial outcomes',
    'excitement': 'Enthusiasm, joy, or positive anticipation in markets',
    'fear': 'Strong concern, panic, or apprehension about financial risks',
    'hope': 'Optimistic expectation for positive financial developments',
    'optimism': 'Confidence and positive outlook on market conditions',
    'uncertainty': 'Ambiguity, confusion, or lack of clarity in financial context'
}

# Emotion colors for visualization
EMOTION_COLORS = {
    'anxiety': '#FFA500',      # Orange
    'excitement': '#FFD700',   # Gold
    'fear': '#FF4500',         # Red-Orange
    'hope': '#32CD32',         # Lime Green
    'optimism': '#00CED1',     # Dark Turquoise
    'uncertainty': '#9370DB'   # Medium Purple
}

@st.cache_resource
def load_model(model_path: str, model_name: str = "distilbert-base-uncased"):
    """
    Load LoRA model from saved checkpoint
    
    Args:
        model_path: Path to LoRA adapter checkpoint
        model_name: Base model name
        
    Returns:
        model, tokenizer
    """
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load base model
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=6,
            id2label={i: emotion for i, emotion in enumerate(EMOTIONS)},
            label2id={emotion: i for i, emotion in enumerate(EMOTIONS)}
        )
        
        # Load LoRA adapters
        model = PeftModel.from_pretrained(base_model, model_path)
        model.eval()
        
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def predict_emotion(text: str, model, tokenizer, return_probs: bool = True):
    """
    Predict emotion for a single text
    
    Args:
        text: Input text
        model: Loaded LoRA model
        tokenizer: Tokenizer
        return_probs: Return probability distribution
        
    Returns:
        predicted_emotion, confidence, probabilities (if return_probs=True)
    """
    # Tokenize
    inputs = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
    # Get prediction
    predicted_idx = torch.argmax(probs, dim=-1).item()
    confidence = probs[0][predicted_idx].item()
    predicted_emotion = EMOTIONS[predicted_idx]
    
    if return_probs:
        prob_dict = {emotion: probs[0][i].item() for i, emotion in enumerate(EMOTIONS)}
        return predicted_emotion, confidence, prob_dict
    else:
        return predicted_emotion, confidence

def predict_batch(texts: list, model, tokenizer):
    """
    Predict emotions for multiple texts
    
    Args:
        texts: List of input texts
        model: Loaded LoRA model
        tokenizer: Tokenizer
        
    Returns:
        List of (text, predicted_emotion, confidence) tuples
    """
    results = []
    
    for text in texts:
        emotion, confidence = predict_emotion(text, model, tokenizer, return_probs=False)
        results.append((text, emotion, confidence))
    
    return results

def get_model_info():
    """
    Return model information for display
    """
    return {
        "Base Model": "DistilBERT-base-uncased",
        "Total Parameters": "67.7M",
        "Trainable Parameters": "742,662 (1.1%)",
        "LoRA Rank": "8",
        "LoRA Alpha": "16",
        "Training Dataset": "1,152 samples (enhanced)",
        "Validation Accuracy": "61.0%",
        "Macro F1": "0.61",
        "Training Time": "~60 minutes (T4 GPU)",
        "Model Size": "2.8 MB (adapters only)"
    }

def get_example_texts():
    """
    Return example financial texts for testing
    """
    return {
        "Positive Market": "The stock market rallied today with tech stocks leading gains. Investors are optimistic about the economic recovery.",
        "Market Crash": "Markets plunged today as recession fears intensified. Investors are panic selling amid economic uncertainty.",
        "Uncertain Outlook": "The Federal Reserve's decision remains unclear. Analysts are divided on the potential impact on inflation.",
        "Hopeful Recovery": "Early signs suggest the economy may be recovering faster than expected. This could be a turning point.",
        "Anxiety Rising": "Growing concerns about supply chain disruptions are making investors nervous about Q4 earnings.",
        "Exciting Innovation": "The breakthrough in renewable energy technology has generated tremendous excitement among investors."
    }

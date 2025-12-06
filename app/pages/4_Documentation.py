"""
Documentation page - Complete project guide and methodology
"""

import streamlit as st
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.model_utils import EMOTIONS, EMOTION_DESCRIPTIONS

# Page config
st.title(" Documentation")
st.markdown("Complete guide to the FinEmo-LoRA system and methodology.")

# Table of contents
st.markdown("""
**Quick Navigation:**
-  [Project Overview](#project-overview)
-  [Model Architecture](#model-architecture)
-  [Training Methodology](#training-methodology)
-  [Usage Guide](#usage-guide)
-  [API Reference](#api-reference)
-  [Performance Analysis](#performance-analysis)
-  [Future Work](#future-work)
""")

st.markdown("---")

# Project Overview
st.header(" Project Overview")

st.markdown("""
**FinEmo-LoRA** is an end-to-end financial emotion detection system that classifies financial texts 
into 6 emotion categories using parameter-efficient fine-tuning with LoRA (Low-Rank Adaptation).

### Problem Statement
Traditional sentiment analysis provides only positive/negative/neutral classifications, missing nuanced 
emotional signals crucial for financial decision-making. This project addresses three key challenges:

1. **Multi-class emotion detection** in financial domain
2. **Severe class imbalance** with rare emotions (hope, fear, excitement)
3. **Parameter efficiency** for deployment and fine-tuning

### Solution
A two-stage training approach combining:
- Transfer learning from general emotion data (GoEmotions)
- Domain adaptation with targeted minority sampling
- LoRA for parameter-efficient fine-tuning (only 1.1% trainable parameters)
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### Key Features
    -  **6 Financial Emotions**: Anxiety, Excitement, Fear, Hope, Optimism, Uncertainty
    -  **76.8% Accuracy**: Exceeds 46.3% baseline by +30.5pp
    -  **3.8MB Model**: Efficient LoRA adapters only
    -  **Fast Inference**: DistilBERT base (3x faster than BERT)
    """)

with col2:
    st.markdown("""
    ### Applications
    -  **Market Analysis**: Detect investor sentiment shifts
    -  **News Monitoring**: Track emotional tone in financial news
    -  **Social Media**: Analyze retail investor discussions
    -  **Alert Systems**: Trigger warnings on fear/anxiety spikes
    """)

# Model Architecture
st.markdown("---")
st.header(" Model Architecture")

# Visual Architecture Diagram
st.markdown("### Architecture Diagram")

# Create a visual block diagram using columns and styled containers
st.markdown("""
<style>
.arch-box {
    background: #ffffff;
    border: 2px solid #333333;
    padding: 20px;
    border-radius: 8px;
    text-align: center;
    color: #333333;
    font-weight: 600;
    margin: 8px 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.arch-box-highlight {
    background: #f5f5f5;
    border: 3px solid #000000;
    padding: 20px;
    border-radius: 8px;
    text-align: center;
    color: #000000;
    font-weight: 700;
    margin: 8px 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.15);
}
.arch-arrow {
    text-align: center;
    font-size: 28px;
    color: #333333;
    margin: 8px 0;
    font-weight: bold;
}
.arch-small {
    font-size: 0.85em;
    font-weight: 400;
    color: #666666;
    margin-top: 5px;
}
</style>
""", unsafe_allow_html=True)

col_left, col_mid, col_right = st.columns([1, 2, 1])

with col_mid:
    st.markdown('<div class="arch-box">Input: Financial Text</div>', unsafe_allow_html=True)
    st.markdown('<div class="arch-arrow">↓</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="arch-box">Tokenizer<br/><span class="arch-small">(DistilBERT Tokenizer)</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="arch-arrow">↓</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="arch-box">DistilBERT Base<br/><span class="arch-small">6 Layers | 768 Dims | 67.7M Params</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="arch-arrow">↓</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="arch-box-highlight">LoRA Adapters<br/><span class="arch-small">r=8, α=16 | 742K Params (1.1%)</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="arch-arrow">↓</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="arch-box">Classification Head<br/><span class="arch-small">6 Emotion Classes</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="arch-arrow">↓</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="arch-box">Output: Emotion + Confidence</div>', unsafe_allow_html=True)

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### Base Model: DistilBERT
    **Specifications:**
    - 6 Transformer layers
    - 768 hidden dimensions
    - 12 attention heads
    - 67.7M total parameters
    - 66M pre-trained (frozen)
    - ~3x faster than BERT-base
    
    **Why DistilBERT?**
    - Retains 97% of BERT's performance
    - Smaller memory footprint
    - Faster inference for production
    - Proven on financial texts
    """)

with col2:
    st.markdown("""
    ### LoRA Configuration
    **Adapter Settings:**
    - Rank: **r = 8**
    - Alpha: **α = 16**
    - Target modules: **q_lin, v_lin** (query & value projections)
    - Trainable params: **742,662** (1.1% of total)
    - Dropout: **0.1**
    - Adapter size: **3.8MB**
    
    **Advantages:**
    - No base model modification
    - Swap adapters for different tasks
    - Fast fine-tuning (minutes vs hours)
    - Minimal storage per task
    """)

# Two-stage training
st.markdown("""
### Two-Stage Training Pipeline

**Stage 1: Transfer Learning (GoEmotions)**
- **Purpose**: Learn general emotion patterns from 10K diverse samples
- **Dataset**: GoEmotions (27 emotions → 6 financial emotions)
- **Epochs**: 3
- **Learning Rate**: 2e-4
- **Result**: Foundation for emotion understanding

**Stage 2: Domain Adaptation (Financial Data)**
- **Purpose**: Specialize for financial texts with targeted sampling
- **Dataset**: FinGPT + synthetic data expansion (3,472 samples)
- **Augmentation**: SMOTE (k=5) for rare classes
- **Epochs**: 10
- **Learning Rate**: 1e-4
- **Result**: 76.8% accuracy, 0.74 F1-score
""")

# Training Methodology
st.markdown("---")
st.header(" Training Methodology")

st.markdown("""
### Data Collection Strategy

The breakthrough came from **targeted minority sampling** instead of random sampling:
""")

st.markdown("""
#### 1. Keyword-Based Filtering
Domain-specific keywords for each emotion class:

- **Anxiety**: "worried", "concerned", "nervous", "uncertain outlook", "volatility concerns"
- **Excitement**: "thrilled", "enthusiastic", "bullish momentum", "breakthrough innovation"
- **Fear**: "crash", "panic", "devastating", "collapse", "severe downturn"
- **Hope**: "recovery ahead", "turnaround expected", "optimistic signs", "potential rebound"
- **Optimism**: "confident", "positive outlook", "strong growth", "favorable conditions"
- **Uncertainty**: "unclear", "ambiguous", "mixed signals", "unpredictable", "remains to be seen"

#### 2. Data Augmentation
**SMOTE (Synthetic Minority Over-sampling Technique)**
- k=5 nearest neighbors
- Applied to rare classes: fear, hope, excitement
- Balanced dataset: 25-62 samples per class

#### 3. Quality Control
- Manual annotation review
- High-confidence filtering (>0.7 confidence)
- Domain expert validation
""")

# Emotion definitions
st.subheader(" Emotion Definitions")

for emotion in EMOTIONS:
    with st.expander(f"**{emotion.capitalize()}**"):
        st.markdown(f"**Definition:** {EMOTION_DESCRIPTIONS[emotion]}")
        
        # Example keywords
        keywords = {
            'anxiety': ['worried', 'nervous', 'concerned', 'anxious', 'jittery'],
            'excitement': ['excited', 'thrilled', 'enthusiastic', 'eager', 'pumped'],
            'fear': ['afraid', 'scared', 'terrified', 'panic', 'dread'],
            'hope': ['hopeful', 'optimistic', 'promising', 'encouraging', 'upbeat'],
            'optimism': ['positive', 'confident', 'bullish', 'favorable', 'strong'],
            'uncertainty': ['unclear', 'uncertain', 'ambiguous', 'unknown', 'unpredictable']
        }
        
        st.markdown(f"**Keywords:** {', '.join(keywords[emotion])}")

# Usage Guide
st.markdown("---")
st.header(" Usage Guide")

st.markdown("""
### Getting Started

**Single Prediction** ( Prediction page)
- Enter or select a financial text
- Get instant emotion classification
- View confidence scores
- See detailed probabilities
""")

# API Reference
st.markdown("---")
st.header(" API Reference")

st.markdown("""
### Model Loading

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

# Load tokenizer and base model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
base_model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=6
)

# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, "models/finemo_lora_v2_best")
model.eval()
```

### Single Prediction

```python
import torch

def predict_emotion(text):
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
    
    # Get prediction
    pred_idx = torch.argmax(probs, dim=-1).item()
    confidence = probs[0][pred_idx].item()
    
    emotions = ['anxiety', 'excitement', 'fear', 'hope', 'optimism', 'uncertainty']
    
    return emotions[pred_idx], confidence

# Example usage
text = "The stock market rallied on positive earnings reports."
emotion, confidence = predict_emotion(text)
print(f"Emotion: {emotion}, Confidence: {confidence:.2%}")
```

### Batch Prediction

```python
def predict_batch(texts, batch_size=32):
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize batch
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
        
        # Get predictions
        pred_indices = torch.argmax(probs, dim=-1)
        confidences = probs[range(len(pred_indices)), pred_indices]
        
        for pred_idx, conf in zip(pred_indices, confidences):
            results.append({
                'emotion': emotions[pred_idx.item()],
                'confidence': conf.item()
            })
    
    return results
```
""")

# Performance Analysis
st.markdown("---")
st.header(" Performance Analysis")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### LoRA v2 Final Results
    
    **Overall Metrics:**
    - Accuracy: **76.8%**
    - Macro F1: **0.74**
    - Dataset: **3,472 samples**
    
    **Key Improvements:**
    - Hope: 0% → **95%** (+95pp)
    - Fear: 0% → **50%** (+50pp)
    - Excitement: 5% → **79%** (+74pp)
    """)

with col2:
    st.markdown("""
    ### Key Achievements
    
     Exceeded 46.3% baseline by +30.5pp  
     Eliminated zero-recall classes  
     Massive improvement in rare emotions  
     Two-stage LoRA training approach  
     Efficient 3.8MB adapter model  
     Fast inference (<100ms per sample)  
    """)

# Confusion Matrix
st.markdown("### Confusion Matrix")
confusion_matrix_path = Path(__file__).parent.parent.parent / "models" / "finemo_lora_final_v2_full_dataset" / "confusion_matrix_lora_v2.png"

if confusion_matrix_path.exists():
    st.image(str(confusion_matrix_path), caption="LoRA v2 Confusion Matrix - Final Model Performance", use_container_width=True)
else:
    st.info("Confusion matrix visualization will be displayed here once available.")

# Future Work
st.markdown("---")
st.header(" Future Work")

st.markdown("""
### Planned Improvements

1. **Model Enhancements**
   - Experiment with larger LoRA ranks (r=16, r=32)
   - Try different base models (RoBERTa, FinBERT)
   - Ensemble methods for higher accuracy
   - Multi-task learning with sentiment

2. **Data Expansion**
   - Collect more fear/hope samples (target: 100+ each)
   - Add temporal context (news sequences)
   - Include market data correlation
   - Annotate confidence levels

3. **Feature Engineering**
   - Add market indicators as features
   - Incorporate entity recognition
   - Time-series emotion tracking
   - Multi-modal inputs (text + charts)
""")

# Contact and Citation
st.markdown("---")
st.header(" Contact & Citation")

st.markdown("""
### Project Information

**Author:** Vaishnavi Kamdi  
**Institution:** George Washington University  
**Course:** Neural Networks and Deep Learning  
**Date:** Fall 2025

### Repository

GitHub: [vaish725/FinEmo-LoRA](https://github.com/vaish725/FinEmo-LoRA)

### Acknowledgments

- **Datasets**: GoEmotions, FinGPT
- **Models**: HuggingFace Transformers, DistilBERT
- **Framework**: PyTorch, PEFT (LoRA)
- **Visualization**: Streamlit, Plotly
""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>Built with  using Streamlit and HuggingFace Transformers</p>
    <p>FinEmo-LoRA v2 | Parameter-Efficient Financial Emotion Detection</p>
</div>
""", unsafe_allow_html=True)

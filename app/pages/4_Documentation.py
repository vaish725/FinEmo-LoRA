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
st.title("📚 Documentation")
st.markdown("Complete guide to the FinEmo-LoRA system and methodology.")

# Table of contents
st.markdown("""
**Quick Navigation:**
- 📖 [Project Overview](#project-overview)
- 🏗️ [Model Architecture](#model-architecture)
- 🔬 [Training Methodology](#training-methodology)
- 📊 [Usage Guide](#usage-guide)
- 🔧 [API Reference](#api-reference)
- 📈 [Performance Analysis](#performance-analysis)
- 🚀 [Future Work](#future-work)
""")

st.markdown("---")

# Project Overview
st.header("📖 Project Overview")

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
    - 🎯 **6 Financial Emotions**: Anxiety, Excitement, Fear, Hope, Optimism, Uncertainty
    - 🚀 **61% Accuracy**: Exceeds 55.6% baseline
    - 💾 **2.8MB Model**: Efficient LoRA adapters only
    - ⚡ **Fast Inference**: DistilBERT base (3x faster than BERT)
    """)

with col2:
    st.markdown("""
    ### Applications
    - 📈 **Market Analysis**: Detect investor sentiment shifts
    - 📰 **News Monitoring**: Track emotional tone in financial news
    - 💬 **Social Media**: Analyze retail investor discussions
    - 🔔 **Alert Systems**: Trigger warnings on fear/anxiety spikes
    """)

# Model Architecture
st.markdown("---")
st.header("🏗️ Model Architecture")

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
    - Adapter size: **2.8MB**
    
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
- **Dataset**: FinGPT + targeted minority sampling (1,152 samples)
- **Augmentation**: SMOTE (k=5) for rare classes
- **Epochs**: 10
- **Learning Rate**: 1e-4
- **Result**: 61% accuracy, 0.61 F1-score
""")

# Training Methodology
st.markdown("---")
st.header("🔬 Training Methodology")

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
st.subheader("🎯 Emotion Definitions")

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
st.header("📊 Usage Guide")

st.markdown("""
### Getting Started

1. **Single Prediction** (🔮 Prediction page)
   - Enter or select a financial text
   - Get instant emotion classification
   - View confidence scores
   - See detailed probabilities

2. **Batch Processing** (📊 Batch Analysis page)
   - Upload CSV with financial texts
   - Process unlimited samples
   - Download results with confidence scores
   - View emotion distribution charts

3. **Model Comparison** (📈 Comparison page)
   - Compare v1 vs v2 performance
   - Analyze per-class improvements
   - Review training configurations
   - Understand methodology evolution

### CSV Format for Batch Processing

```csv
text
"The market rallied on positive economic data."
"Investors remain uncertain about future prospects."
"Fear gripped Wall Street as indices plummeted."
```

**Requirements:**
- Column named `text` or `content`
- One financial text per row
- UTF-8 encoding recommended
""")

# API Reference
st.markdown("---")
st.header("🔧 API Reference")

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
model = PeftModel.from_pretrained(base_model, "path/to/finemo-lora-final-v2")
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
st.header("📈 Performance Analysis")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### LoRA v2 Final Results
    
    **Overall Metrics:**
    - Accuracy: **61.0%**
    - Macro F1: **0.61**
    - Weighted F1: **0.64**
    
    **Per-Class F1:**
    - Anxiety: 0.56
    - Excitement: 0.42
    - Fear: 0.75
    - Hope: 0.80
    - Optimism: 0.77
    - Uncertainty: 0.67
    """)

with col2:
    st.markdown("""
    ### Key Achievements
    
    ✅ Exceeded 55.6% baseline by +5.4pp  
    ✅ Eliminated zero-performance classes  
    ✅ Doubled F1-score from v1 (0.29 → 0.61)  
    ✅ Balanced performance across all emotions  
    ✅ Efficient 2.8MB adapter model  
    ✅ Fast inference (<100ms per sample)  
    """)

# Future Work
st.markdown("---")
st.header("🚀 Future Work")

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

4. **Production Deployment**
   - REST API with FastAPI
   - Real-time news feed processing
   - Streaming analytics dashboard
   - Confidence calibration
   - A/B testing framework

5. **Research Directions**
   - Emotion causality analysis
   - Cross-market emotion transfer
   - Explainable AI for predictions
   - Emotion dynamics over time
""")

# Contact and Citation
st.markdown("---")
st.header("📧 Contact & Citation")

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
    <p>Built with ❤️ using Streamlit and HuggingFace Transformers</p>
    <p>FinEmo-LoRA v2 | Parameter-Efficient Financial Emotion Detection</p>
</div>
""", unsafe_allow_html=True)

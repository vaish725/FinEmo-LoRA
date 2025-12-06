"""
FinEmo-LoRA Dashboard - Main Application
Simplified version that actually works!
"""

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="FinEmo-LoRA Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# # Sidebar
# with st.sidebar:
#     st.title(" FinEmo-LoRA")
#     st.markdown("**Financial Emotion Detection**")
#     st.markdown("---")
    
#     st.markdown("###  Performance")
#     st.metric("Accuracy", "76.8%", "+24.1pp")
#     st.metric("Macro F1", "0.74", "+159%")
    
#     st.markdown("---")
#     st.caption("Vaishnavi Kamdi")
#     st.caption("GWU NNDL - Fall 2025")

# Main content
st.title(" FinEmo-LoRA Dashboard")
st.markdown("### Financial Emotion Detection with LoRA")

# Metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Accuracy", "76.8%")
with col2:
    st.metric("F1", "0.74")
with col3:
    st.metric("Samples", "3,472", "Full dataset")
with col4:
    st.metric("Model Size", "3.8 MB")

st.markdown("---")

# Overview
col1, col2 = st.columns(2)

with col1:
    st.subheader(" Supported Emotions")
    emotions = {
        'anxiety': ('🟠', 'Nervousness, worry about outcomes'),
        'excitement': ('🟡', 'Enthusiasm, positive anticipation'),
        'fear': ('🔴', 'Panic, strong apprehension'),
        'hope': ('🟢', 'Optimistic expectation'),
        'optimism': ('🟢', 'Confidence, positive outlook'),
        'uncertainty': ('🟣', 'Ambiguity, confusion')
    }
    
    for emotion, (emoji, desc) in emotions.items():
        st.markdown(f"**{emoji} {emotion.upper()}** - {desc}")

with col2:
    st.subheader(" Training Method")
    st.markdown("""
    **Approach:**
    - Two-stage LoRA fine-tuning
    - Stage 1: GoEmotions (10K samples)
    - Stage 2: FinGPT (3,472 samples)
    - SMOTE oversampling (k=5)
    
    **Configuration:**
    - LoRA rank: r=8, α=16
    - Only 1.1% parameters trained
    - Training time: ~60 mins (T4 GPU)
    """)

st.markdown("---")

# Quick stats
st.subheader(" Model Architecture")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **Base Model**
    - DistilBERT-base-uncased
    - 67.7M parameters
    - 6 layers, 768 dims
    """)

with col2:
    st.markdown("""
    **LoRA Config**
    - Rank: r=8, α=16
    - 742K trainable params (1.1%)
    """)

with col3:
    st.markdown("""
    **Training**
    - Stage 1: GoEmotions (10K)
    - Stage 2: FinGPT (3,472)
    - Time: ~60 mins on T4
    """)

# st.success(" Dashboard loaded successfully! Check the sidebar for navigation.")

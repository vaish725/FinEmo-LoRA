"""
FinEmo-LoRA Dashboard - Main Application
Simplified version that actually works!
"""

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="FinEmo-LoRA Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
with st.sidebar:
    st.title("📈 FinEmo-LoRA")
    st.markdown("**Financial Emotion Detection**")
    st.markdown("---")
    
    st.markdown("### 🎯 Performance")
    st.metric("Accuracy", "61.0%", "+8.3pp")
    st.metric("Macro F1", "0.61", "+114%")
    
    st.markdown("---")
    st.markdown("### 🏆 Key Wins")
    st.markdown("✅ Hope: 0% → 82%")
    st.markdown("✅ Fear: 0% → 76%")
    st.markdown("✅ Excitement: 5% → 39%")
    
    st.markdown("---")
    st.caption("Vaishnavi Kamdi")
    st.caption("GWU NNDL - Fall 2025")

# Main content
st.title("📈 FinEmo-LoRA Dashboard")
st.markdown("### Real-time Financial Emotion Detection with LoRA")

# Metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Accuracy", "61.0%", "+8.3pp from v1")
with col2:
    st.metric("Macro F1", "0.61", "+114%")
with col3:
    st.metric("Samples", "1,152", "+224")
with col4:
    st.metric("Model Size", "2.8 MB")

st.markdown("---")

# Overview
col1, col2 = st.columns(2)

with col1:
    st.subheader("🎯 Supported Emotions")
    emotions = {
        'anxiety': ('🟠', 'Nervousness, worry about outcomes'),
        'excitement': ('🟡', 'Enthusiasm, positive anticipation'),
        'fear': ('🔴', 'Panic, strong apprehension'),
        'hope': ('🟢', 'Optimistic expectation'),
        'optimism': ('🔵', 'Confidence, positive outlook'),
        'uncertainty': ('🟣', 'Ambiguity, confusion')
    }
    
    for emotion, (emoji, desc) in emotions.items():
        st.markdown(f"**{emoji} {emotion.upper()}** - {desc}")

with col2:
    st.subheader("📊 v1 vs v2 Comparison")
    st.markdown("""
    **Major Improvements:**
    - Hope recall: **0% → 82%** (+82pp) 🚀
    - Fear recall: **0% → 76%** (+76pp) 🚀  
    - Excitement recall: **5% → 39%** (+34pp) ⬆️
    - Overall accuracy: **52.7% → 61.0%** (+8.3pp) ✅
    
    **Method:**
    - Targeted minority sampling (224 samples)
    - Two-stage LoRA training
    - SMOTE with k=5
    - Cost: Only $1.13!
    """)

st.markdown("---")

# Navigation instructions
st.info("""
👈 **Use the sidebar to navigate:**
- Create additional pages in the `pages/` folder
- Name them like `1_🔮_Prediction.py`, `2_📊_Batch.py`, etc.
- They'll appear automatically in the sidebar!

**For now, this home page shows you the key metrics. Let me create the other pages...**
""")

# Quick stats
st.subheader("📈 Model Architecture")
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
    - Target: q_lin, v_lin
    """)

with col3:
    st.markdown("""
    **Training**
    - Stage 1: GoEmotions (10K)
    - Stage 2: FinGPT (1,152)
    - Time: ~60 mins on T4
    """)

st.success("✅ Dashboard loaded successfully! Check the sidebar for navigation.")

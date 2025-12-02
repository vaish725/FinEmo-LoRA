"""
Model comparison page - Compare LoRA v1 vs v2 performance
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.model_utils import EMOTIONS, EMOTION_COLORS

# Page config
st.title("📈 Model Comparison: v1 vs v2")
st.markdown("Compare performance improvements between LoRA v1 and v2 models.")

# Performance data
v1_data = {
    'anxiety': {'precision': 0.27, 'recall': 0.36, 'f1': 0.31, 'support': 47},
    'excitement': {'precision': 0.09, 'recall': 0.05, 'f1': 0.06, 'support': 19},
    'fear': {'precision': 0.00, 'recall': 0.00, 'f1': 0.00, 'support': 11},
    'hope': {'precision': 0.00, 'recall': 0.00, 'f1': 0.00, 'support': 5},
    'optimism': {'precision': 0.60, 'recall': 0.66, 'f1': 0.63, 'support': 76},
    'uncertainty': {'precision': 0.64, 'recall': 0.79, 'f1': 0.71, 'support': 58}
}

v2_data = {
    'anxiety': {'precision': 0.52, 'recall': 0.59, 'f1': 0.56, 'support': 29},
    'excitement': {'precision': 0.45, 'recall': 0.39, 'f1': 0.42, 'support': 28},
    'fear': {'precision': 0.74, 'recall': 0.76, 'f1': 0.75, 'support': 25},
    'hope': {'precision': 0.79, 'recall': 0.82, 'f1': 0.80, 'support': 28},
    'optimism': {'precision': 0.67, 'recall': 0.90, 'f1': 0.77, 'support': 62},
    'uncertainty': {'precision': 0.56, 'recall': 0.83, 'f1': 0.67, 'support': 59}
}

# Overall metrics
st.subheader("📊 Overall Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style='padding: 20px; border-radius: 10px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;'>
        <h4 style='margin: 0;'>Accuracy</h4>
        <p style='margin: 10px 0 0 0; font-size: 0.9rem;'>v1: 52.7%</p>
        <p style='margin: 5px 0; font-size: 2rem; font-weight: bold;'>v2: 61.0%</p>
        <p style='margin: 0; font-size: 1.2rem; color: #90EE90;'>↑ +8.3pp</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style='padding: 20px; border-radius: 10px; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white;'>
        <h4 style='margin: 0;'>Macro F1</h4>
        <p style='margin: 10px 0 0 0; font-size: 0.9rem;'>v1: 0.29</p>
        <p style='margin: 5px 0; font-size: 2rem; font-weight: bold;'>v2: 0.61</p>
        <p style='margin: 0; font-size: 1.2rem; color: #90EE90;'>↑ +114%</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style='padding: 20px; border-radius: 10px; background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white;'>
        <h4 style='margin: 0;'>Training Data</h4>
        <p style='margin: 10px 0 0 0; font-size: 0.9rem;'>v1: 928</p>
        <p style='margin: 5px 0; font-size: 2rem; font-weight: bold;'>v2: 1,152</p>
        <p style='margin: 0; font-size: 1.2rem; color: #90EE90;'>↑ +224</p>
    </div>
    """, unsafe_allow_html=True)

# Per-class comparison
st.markdown("---")
st.subheader("🎯 Per-Class Performance Comparison")

# Prepare data for plotting
metrics_df = []
for emotion in EMOTIONS:
    metrics_df.append({
        'Emotion': emotion,
        'Metric': 'Precision',
        'v1': v1_data[emotion]['precision'],
        'v2': v2_data[emotion]['precision'],
        'Improvement': v2_data[emotion]['precision'] - v1_data[emotion]['precision']
    })
    metrics_df.append({
        'Emotion': emotion,
        'Metric': 'Recall',
        'v1': v1_data[emotion]['recall'],
        'v2': v2_data[emotion]['recall'],
        'Improvement': v2_data[emotion]['recall'] - v1_data[emotion]['recall']
    })
    metrics_df.append({
        'Emotion': emotion,
        'Metric': 'F1-Score',
        'v1': v1_data[emotion]['f1'],
        'v2': v2_data[emotion]['f1'],
        'Improvement': v2_data[emotion]['f1'] - v1_data[emotion]['f1']
    })

metrics_df = pd.DataFrame(metrics_df)

# Metric selector
selected_metric = st.selectbox(
    "Select metric to compare:",
    ['F1-Score', 'Precision', 'Recall'],
    index=0
)

# Filter data
metric_data = metrics_df[metrics_df['Metric'] == selected_metric]

# Side-by-side comparison
fig = go.Figure()

fig.add_trace(go.Bar(
    name='LoRA v1',
    x=metric_data['Emotion'],
    y=metric_data['v1'],
    marker_color='lightblue',
    text=metric_data['v1'].apply(lambda x: f'{x:.2f}'),
    textposition='outside'
))

fig.add_trace(go.Bar(
    name='LoRA v2',
    x=metric_data['Emotion'],
    y=metric_data['v2'],
    marker_color='lightgreen',
    text=metric_data['v2'].apply(lambda x: f'{x:.2f}'),
    textposition='outside'
))

fig.update_layout(
    title=f'{selected_metric} Comparison by Emotion',
    xaxis_title='Emotion',
    yaxis_title=selected_metric,
    barmode='group',
    height=400,
    yaxis=dict(range=[0, 1])
)

st.plotly_chart(fig, use_container_width=True)

# Improvement heatmap
st.subheader("🔥 Performance Improvement Heatmap")

improvement_matrix = []
for emotion in EMOTIONS:
    improvement_matrix.append([
        v2_data[emotion]['precision'] - v1_data[emotion]['precision'],
        v2_data[emotion]['recall'] - v1_data[emotion]['recall'],
        v2_data[emotion]['f1'] - v1_data[emotion]['f1']
    ])

fig = go.Figure(data=go.Heatmap(
    z=improvement_matrix,
    x=['Precision', 'Recall', 'F1-Score'],
    y=EMOTIONS,
    colorscale='RdYlGn',
    zmid=0,
    text=[[f'{val:.2f}' for val in row] for row in improvement_matrix],
    texttemplate='%{text}',
    textfont={"size": 12},
    colorbar=dict(title="Improvement")
))

fig.update_layout(
    title='Performance Improvements (v2 - v1)',
    height=400
)

st.plotly_chart(fig, use_container_width=True)

# Detailed metrics table
st.markdown("---")
st.subheader("📋 Detailed Metrics Table")

# Create comparison dataframe
comparison_data = []
for emotion in EMOTIONS:
    comparison_data.append({
        'Emotion': emotion.capitalize(),
        'v1 Precision': f"{v1_data[emotion]['precision']:.2f}",
        'v2 Precision': f"{v2_data[emotion]['precision']:.2f}",
        'Δ Precision': f"{v2_data[emotion]['precision'] - v1_data[emotion]['precision']:+.2f}",
        'v1 Recall': f"{v1_data[emotion]['recall']:.2f}",
        'v2 Recall': f"{v2_data[emotion]['recall']:.2f}",
        'Δ Recall': f"{v2_data[emotion]['recall'] - v1_data[emotion]['recall']:+.2f}",
        'v1 F1': f"{v1_data[emotion]['f1']:.2f}",
        'v2 F1': f"{v2_data[emotion]['f1']:.2f}",
        'Δ F1': f"{v2_data[emotion]['f1'] - v1_data[emotion]['f1']:+.2f}",
    })

comparison_df = pd.DataFrame(comparison_data)
st.dataframe(comparison_df, use_container_width=True)

# Key improvements section
st.markdown("---")
st.subheader("🌟 Key Improvements")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### Biggest Gains:
    - **Hope**: 0.00 → 0.80 F1 (+0.80) 🚀
    - **Fear**: 0.00 → 0.75 F1 (+0.75) 🚀
    - **Excitement**: 0.06 → 0.42 F1 (+0.36) 📈
    - **Anxiety**: 0.31 → 0.56 F1 (+0.25) 📈
    
    **Why?** Targeted minority sampling and better data curation eliminated zero-performance classes.
    """)

with col2:
    st.markdown("""
    ### Consistent Performance:
    - **Optimism**: 0.63 → 0.77 F1 (+0.14) 📊
    - **Uncertainty**: 0.71 → 0.67 F1 (-0.04) 📊
    
    **Note:** Slight drop in uncertainty likely due to better balance across all classes 
    rather than over-fitting to dominant classes.
    """)

# Training configuration comparison
st.markdown("---")
st.subheader("⚙️ Training Configuration")

config_col1, config_col2 = st.columns(2)

with config_col1:
    st.markdown("""
    ### LoRA v1
    - **Data**: 928 samples (random selection)
    - **Balance**: Severe imbalance (fear: 11, hope: 5)
    - **Augmentation**: None
    - **Epochs**: 10
    - **Learning Rate**: 1e-4
    - **Result**: 52.7% accuracy, 0.29 F1
    """)

with config_col2:
    st.markdown("""
    ### LoRA v2
    - **Data**: 1,152 samples (targeted sampling)
    - **Balance**: Improved (fear: 25, hope: 28)
    - **Augmentation**: SMOTE (k=5)
    - **Epochs**: 10
    - **Learning Rate**: 1e-4
    - **Result**: 61.0% accuracy, 0.61 F1 ✨
    """)

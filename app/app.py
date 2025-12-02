"""
FinEmo-LoRA: Financial Emotion Detection Dashboard
End-to-end application for real-time emotion classification in financial texts
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Page configuration
st.set_page_config(
    page_title="FinEmo-LoRA Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #667eea;
    }
    .success-badge {
        background-color: #28a745;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 10px;
        padding: 10px 25px;
    }
    .stButton>button:hover {
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/emotions.png", width=80)
    st.title("FinEmo-LoRA")
    st.markdown("**Financial Emotion Detection**")
    st.markdown("---")
    
    # Navigation
    page = st.radio(
        "Navigate to:",
        ["🏠 Home", "🔮 Single Prediction", "📊 Batch Analysis", 
         "📈 Model Comparison", "📚 Documentation"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 🎯 Model Performance")
    st.metric("Accuracy", "61.0%", "+8.3pp from v1")
    st.metric("Macro F1", "0.61", "+114% vs v1")
    
    st.markdown("---")
    st.markdown("### 🏆 Achievements")
    st.markdown("✅ Target exceeded (55.6%)")
    st.markdown("✅ Hope: 0% → 82%")
    st.markdown("✅ Fear: 0% → 76%")
    st.markdown("✅ Excitement: 5% → 39%")
    
    st.markdown("---")
    st.caption("Developed by Vaishnavi Kamdi")
    st.caption("GWU NNDL - Fall 2025")

# Main content area
try:
    if page == "🏠 Home":
        from pages import home
        home.show()
    elif page == "🔮 Single Prediction":
        from pages import prediction
        prediction.show()
    elif page == "📊 Batch Analysis":
        from pages import batch_analysis
        batch_analysis.show()
    elif page == "📈 Model Comparison":
        from pages import comparison
        comparison.show()
    elif page == "📚 Documentation":
        from pages import documentation
        documentation.show()
except Exception as e:
    st.error(f"Error loading page: {e}")
    st.exception(e)

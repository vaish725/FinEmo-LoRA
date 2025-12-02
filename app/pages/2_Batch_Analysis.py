"""
Batch analysis page - Upload CSV files for bulk emotion classification
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import io

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.model_utils import load_model, predict_batch, EMOTION_COLORS

# Page config
st.title("📊 Batch Analysis")
st.markdown("Upload a CSV file with financial texts for bulk emotion classification.")

# Model loading
model_path = Path(__file__).parent.parent.parent / "models" / "finemo-lora-final-v2"

if not model_path.exists():
    st.error(f"⚠️ Model not found at: {model_path}")
    st.stop()

# Load model with caching
@st.cache_resource
def load_emotion_model():
    """Load the LoRA model and tokenizer"""
    return load_model(str(model_path))

with st.spinner("Loading LoRA v2 model..."):
    model, tokenizer = load_emotion_model()

if model is None:
    st.error("Failed to load model.")
    st.stop()

st.success("✅ Model loaded successfully!")

# Instructions
with st.expander("📋 CSV Format Instructions"):
    st.markdown("""
    **Required CSV format:**
    - Must contain a column named `text` or `content` with financial texts
    - Each row should contain one text sample
    - Additional columns will be preserved in the output
    
    **Example CSV:**
    ```
    text
    "The stock market rallied today..."
    "Investors are concerned about..."
    "Economic indicators show positive signs..."
    ```
    
    **Download sample CSV:**
    """)
    
    # Create sample CSV
    sample_data = pd.DataFrame({
        'text': [
            "The stock market rallied today with tech stocks leading gains.",
            "Markets plunged as recession fears intensified among investors.",
            "The Federal Reserve's decision remains unclear to analysts.",
            "Early signs suggest economic recovery may be faster than expected."
        ]
    })
    
    csv_buffer = io.StringIO()
    sample_data.to_csv(csv_buffer, index=False)
    
    st.download_button(
        label="⬇️ Download Sample CSV",
        data=csv_buffer.getvalue(),
        file_name="sample_financial_texts.csv",
        mime="text/csv"
    )

# File upload
uploaded_file = st.file_uploader(
    "Upload CSV file:",
    type=['csv'],
    help="Upload a CSV file with a 'text' or 'content' column"
)

if uploaded_file is not None:
    try:
        # Load CSV
        df = pd.read_csv(uploaded_file)
        
        # Find text column
        text_column = None
        for col in ['text', 'content', 'Text', 'Content', 'message', 'Message']:
            if col in df.columns:
                text_column = col
                break
        
        if text_column is None:
            st.error("❌ No 'text' or 'content' column found in CSV. Please check your file format.")
            st.info(f"Available columns: {', '.join(df.columns)}")
            st.stop()
        
        # Display preview
        st.subheader("📄 File Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.info(f"📊 Total samples: **{len(df)}** | Text column: **{text_column}**")
        
        # Process button
        if st.button("🚀 Classify All Texts", type="primary"):
            with st.spinner(f"Processing {len(df)} texts..."):
                # Get predictions
                texts = df[text_column].tolist()
                results = predict_batch(model, tokenizer, texts)
                
                # Add results to dataframe
                df['predicted_emotion'] = results['emotions']
                df['confidence'] = results['confidences']
                
            st.success(f"✅ Processed {len(df)} texts successfully!")
            
            # Display results
            st.subheader("📊 Results")
            
            # Emotion distribution
            col1, col2 = st.columns(2)
            
            with col1:
                emotion_counts = df['predicted_emotion'].value_counts()
                fig = px.pie(
                    values=emotion_counts.values,
                    names=emotion_counts.index,
                    title="Emotion Distribution",
                    color=emotion_counts.index,
                    color_discrete_map=EMOTION_COLORS
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                avg_conf = df.groupby('predicted_emotion')['confidence'].mean().sort_values()
                fig = px.bar(
                    x=avg_conf.values,
                    y=avg_conf.index,
                    orientation='h',
                    title="Average Confidence by Emotion",
                    labels={'x': 'Average Confidence', 'y': 'Emotion'},
                    color=avg_conf.index,
                    color_discrete_map=EMOTION_COLORS
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Results table
            st.subheader("📋 Detailed Results")
            
            # Add color coding
            def highlight_emotion(row):
                color = EMOTION_COLORS.get(row['predicted_emotion'], '#cccccc')
                return [f'background-color: {color}20' if col == 'predicted_emotion' else '' for col in row.index]
            
            styled_df = df.style.apply(highlight_emotion, axis=1)
            st.dataframe(styled_df, use_container_width=True)
            
            # Download results
            csv_output = io.StringIO()
            df.to_csv(csv_output, index=False)
            
            st.download_button(
                label="⬇️ Download Results CSV",
                data=csv_output.getvalue(),
                file_name=f"emotion_analysis_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            # Summary statistics
            with st.expander("📈 Summary Statistics"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Samples", len(df))
                    st.metric("Unique Emotions", df['predicted_emotion'].nunique())
                
                with col2:
                    st.metric("Avg Confidence", f"{df['confidence'].mean():.2%}")
                    st.metric("Min Confidence", f"{df['confidence'].min():.2%}")
                
                with col3:
                    st.metric("Max Confidence", f"{df['confidence'].max():.2%}")
                    most_common = df['predicted_emotion'].mode()[0]
                    st.metric("Most Common", most_common)
    
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.info("Please ensure your CSV file is properly formatted.")
else:
    # Show helpful message
    st.info("👆 Upload a CSV file to get started!")
    
    st.markdown("""
    ### Features:
    - 🚀 Bulk processing of unlimited texts
    - 📊 Interactive visualization of emotion distribution
    - 📈 Confidence scores for each prediction
    - 💾 Downloadable results in CSV format
    - 🎯 Automatic text column detection
    """)

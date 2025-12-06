"""
Single Text Prediction Page
Place this file as: pages/1__Prediction.py
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

st.set_page_config(page_title="Prediction", page_icon="", layout="wide")

st.title(" Single Text Prediction")
st.markdown("Analyze financial text for emotion classification in real-time.")

# Model loading section
model_path = Path(__file__).parent.parent.parent / "models" / "finemo_lora_v2_best"

if not model_path.exists():
    st.error(f" Model not found at: {model_path}")
    st.info("Please ensure the LoRA v2 model files are in the `models/finemo-lora-final-v2/` directory")
    st.stop()

#Load model (cached)
@st.cache_resource
def load_emotion_model():
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from peft import PeftModel
        import torch
        
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        base_model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=6
        )
        model = PeftModel.from_pretrained(base_model, str(model_path))
        model.eval()
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

with st.spinner("Loading LoRA v2 model..."):
    model, tokenizer = load_emotion_model()

if model is None:
    st.error("Failed to load model")
    st.stop()

st.success(" Model loaded successfully!")

# Example texts
examples = {
    "Positive Market": "The stock market rallied today with tech stocks leading gains. Investors are optimistic about the economic recovery.",
    "Market Crash": "Markets plunged today as recession fears intensified. Investors are panic selling amid economic uncertainty.",
    "Uncertain Outlook": "The Federal Reserve's decision remains unclear. Analysts are divided on the potential impact on inflation.",
    "Hopeful Recovery": "Early signs suggest the economy may be recovering faster than expected. This could be a turning point.",
    "Anxiety Rising": "Growing concerns about supply chain disruptions are making investors nervous about Q4 earnings.",
    "Exciting Innovation": "The breakthrough in renewable energy technology has generated tremendous excitement among investors."
}

# Input method
input_method = st.radio("Choose input:", ["Type text", "Use examples"], horizontal=True)

if input_method == "Use examples":
    selected = st.selectbox("Select example:", list(examples.keys()))
    text_input = examples[selected]
    st.text_area("Text:", text_input, height=100, disabled=True)
else:
    text_input = st.text_area(
        "Enter financial text:",
        placeholder="e.g., The stock market rallied today...",
        height=150
    )

# Predict button
if st.button(" Analyze Emotion", type="primary"):
    if not text_input.strip():
        st.warning("Please enter some text")
    else:
        with st.spinner("Analyzing..."):
            import torch
            
            # Tokenize
            inputs = tokenizer(text_input, return_tensors="pt", max_length=128, truncation=True, padding='max_length')
            
            # Predict
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)[0]
            
            emotions = ['anxiety', 'excitement', 'fear', 'hope', 'optimism', 'uncertainty']
            colors = {'anxiety': '#FFA500', 'excitement': '#FFD700', 'fear': '#FF4500', 
                     'hope': '#32CD32', 'optimism': '#00CED1', 'uncertainty': '#9370DB'}
            
            predicted_idx = torch.argmax(probs).item()
            predicted_emotion = emotions[predicted_idx]
            confidence = probs[predicted_idx].item()
            
            # Display results
            st.markdown("---")
            st.subheader(" Results")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                color = colors[predicted_emotion]
                st.markdown(f"""
                <div style='padding: 30px; background-color: {color}22; border-left: 5px solid {color}; border-radius: 10px;'>
                    <h2 style='color: {color}; margin: 0;'>{predicted_emotion.upper()}</h2>
                    <h1 style='color: {color}; margin: 10px 0;'>{confidence:.1%}</h1>
                    <p>Confidence Score</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.metric("Predicted Emotion", predicted_emotion.title())
                st.metric("Confidence", f"{confidence:.1%}")
            
            # Probability distribution
            st.markdown("### All Probabilities")
            for i, (emotion, prob) in enumerate(zip(emotions, probs)):
                st.progress(float(prob), text=f"{emotion}: {prob:.1%}")

st.markdown("---")
st.info(" **Tip:** The model works best with financial news, market analysis, and economic commentary.")

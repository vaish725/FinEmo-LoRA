#  FinEmo-LoRA Dashboard

Interactive web application for financial emotion detection using LoRA-enhanced DistilBERT.

![Dashboard Preview](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit)
![Accuracy](https://img.shields.io/badge/Accuracy-76.8%25-success?style=for-the-badge)
![Model](https://img.shields.io/badge/Model-LoRA--v2-blue?style=for-the-badge)

##  Features

###  Home Dashboard
- Performance metrics (76.8% accuracy)
- Model architecture overview
- Training methodology details
- Supported emotions list

###  Single Prediction
- Real-time emotion classification
- Confidence scores
- Probability distributions
- Example financial texts

###  Batch Analysis
- CSV file upload for bulk processing
- Downloadable results (CSV/Excel)
- Emotion distribution analytics
- Per-emotion insights

###  Documentation
- Complete project overview
- Model architecture details
- Training methodology
- Usage examples

##  Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/vaish725/FinEmo-LoRA.git
cd FinEmo-LoRA

# Install dashboard dependencies
pip install -r requirements_app.txt
```

### 2. Verify Model Files

Ensure the LoRA v2 model is in the correct location:
```
FinEmo-LoRA/
 models/
    finemo_lora_v2_best/
        adapter_model.safetensors
        adapter_config.json
        tokenizer.json
        vocab.txt
        ...
```

### 3. Run the Dashboard

```bash
# Navigate to app directory
cd app

# Launch Streamlit
streamlit run Home.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`

##  Project Structure

```
app/
 Home.py                         # Main Streamlit application
 pages/
    __init__.py
    1_Prediction.py             # Single text prediction
    2_Batch_Analysis.py         # Batch CSV processing
    4_Documentation.py          # Documentation page
 utils/
     __init__.py
     model_utils.py              # Model loading & inference
```

##  Supported Emotions

| Emotion | Description | Use Case |
|---------|-------------|----------|
| **Anxiety** | Nervousness, worry about outcomes | Risk assessment |
| **Excitement** | Enthusiasm, positive anticipation | Market sentiment |
| **Fear** | Panic, strong apprehension | Crisis detection |
| **Hope** | Optimistic expectation | Recovery signals |
| **Optimism** | Positive outlook | Bullish sentiment |
| **Uncertainty** | Ambiguity, confusion | Market volatility |

##  Usage Examples

### Single Text Analysis

1. Navigate to ** Single Prediction**
2. Enter or select financial text
3. Click **Analyze Emotion**
4. View results with confidence scores

### Batch Processing

1. Navigate to ** Batch Analysis**
2. Prepare CSV with `text` column
3. Upload file
4. Click **Analyze All Texts**
5. Download results

**Sample CSV Format:**
```csv
text
"The stock market rallied today with strong gains..."
"Investors are concerned about rising inflation..."
"Economic recovery shows promising early signs..."
```

##  Performance Metrics

### Overall (LoRA v2)
- **Accuracy**: 76.8% (+24.1pp from baseline)
- **Macro F1**: 0.74 (+159% improvement)
- **Model Size**: 3.8 MB (adapters only)
- **Dataset**: 3,472 samples
- **Inference Speed**: ~50ms per text

### Key Improvements
| Emotion | Baseline | v2 | Improvement |
|---------|----------|-------|-------------|
| Hope | 0% | **95%** | +95pp  |
| Fear | 0% | **50%** | +50pp  |
| Excitement | 5% | **79%** | +74pp  |

##  Advanced Configuration

### Custom Model Path

Edit `app/pages/1_Prediction.py` to use a different model:

```python
model_path = Path(__file__).parent.parent.parent / "models" / "your_model_name"
```

### Batch Size Adjustment

For large CSV files, process in batches by modifying `batch_analysis.py`:

```python
BATCH_SIZE = 100  # Process 100 texts at a time
```

##  Troubleshooting

### Model Not Found Error
- Ensure model files are in `models/finemo_lora_v2_best/`
- Check file permissions
- Verify `adapter_model.safetensors` exists

### Memory Issues
- Reduce batch size for large CSV files
- Close other applications
- Use smaller model variants if available

### Slow Inference
- Ensure GPU is available: `torch.cuda.is_available()`
- Reduce max_length in tokenizer (default: 128)
- Process smaller batches

##  API Integration

Use the model programmatically:

```python
from app.utils.model_utils import load_model, predict_emotion

# Load model
model, tokenizer = load_model("models/finemo_lora_v2_best")

# Predict
text = "Markets surged on positive earnings reports"
emotion, confidence, probs = predict_emotion(
    text, model, tokenizer, return_probs=True
)

print(f"Emotion: {emotion} ({confidence:.1%} confidence)")
print(f"Probabilities: {probs}")
```

##  Customization

### Change Color Theme

Edit `Home.py` CSS section:

```python
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #YOUR_COLOR_1 0%, #YOUR_COLOR_2 100%);
    }
</style>
""", unsafe_allow_html=True)
```

### Add New Emotions

1. Update `EMOTIONS` list in `utils/model_utils.py`
2. Retrain model with new labels
3. Update `EMOTION_COLORS` and `EMOTION_DESCRIPTIONS`

##  Dashboard Pages

- **Home**: Performance metrics and model architecture
- **Prediction**: Single text emotion analysis
- **Batch Analysis**: Process multiple texts from CSV
- **Documentation**: Complete project guide

##  Contributing

Contributions welcome! Areas for improvement:

- Multi-label classification support
- Additional visualization options
- Performance optimizations
- New emotion categories
- REST API endpoint

##  License

This project is part of academic work at George Washington University.

##  Author

**Vaishnavi Kamdi**
- Course: NNDL - Fall 2025, GWU
- GitHub: [@vaish725](https://github.com/vaish725)

##  Acknowledgments

- DistilBERT & LoRA papers
- Hugging Face Transformers
- Streamlit framework
- FinGPT & GoEmotions datasets

---

**Built with  using Streamlit, PyTorch, and Hugging Face**

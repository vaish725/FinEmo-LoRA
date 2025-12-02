# 📈 FinEmo-LoRA Dashboard

Interactive web application for real-time financial emotion detection using LoRA-enhanced DistilBERT.

![Dashboard Preview](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit)
![Accuracy](https://img.shields.io/badge/Accuracy-61.0%25-success?style=for-the-badge)
![Model](https://img.shields.io/badge/Model-LoRA--v2-blue?style=for-the-badge)

## 🌟 Features

### 🏠 Home Dashboard
- Real-time performance metrics
- Model architecture overview
- Per-class performance visualization
- Quick start guide

### 🔮 Single Prediction
- Real-time emotion classification
- Confidence scores with visual gauges
- Probability distribution charts
- Example financial texts library

### 📊 Batch Analysis
- CSV file upload for bulk processing
- Downloadable results (CSV/Excel)
- Emotion distribution analytics
- Per-emotion insights and samples

### 📈 Model Comparison
- v1 vs v2 side-by-side comparison
- Interactive performance charts
- Improvement heatmaps
- Cost-benefit analysis

### 📚 Documentation
- Complete project overview
- Model architecture details
- Training methodology
- API reference
- Usage examples

## 🚀 Quick Start

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
├── models/
│   └── finemo-lora-final-v2/
│       ├── adapter_model.safetensors
│       ├── adapter_config.json
│       ├── tokenizer.json
│       └── ...
```

### 3. Run the Dashboard

```bash
# Navigate to app directory
cd app

# Launch Streamlit
streamlit run app.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`

## 📁 Project Structure

```
app/
├── app.py                          # Main Streamlit application
├── pages/
│   ├── __init__.py
│   ├── home.py                     # Home dashboard
│   ├── prediction.py               # Single text prediction
│   ├── batch_analysis.py           # Batch CSV processing
│   ├── comparison.py               # Model comparison
│   └── documentation.py            # Documentation page
└── utils/
    ├── __init__.py
    └── model_utils.py              # Model loading & inference
```

## 🎯 Supported Emotions

| Emotion | Description | Use Case |
|---------|-------------|----------|
| **Anxiety** | Nervousness, worry about outcomes | Risk assessment |
| **Excitement** | Enthusiasm, positive anticipation | Market sentiment |
| **Fear** | Panic, strong apprehension | Crisis detection |
| **Hope** | Optimistic expectation | Recovery signals |
| **Optimism** | Positive outlook | Bullish sentiment |
| **Uncertainty** | Ambiguity, confusion | Market volatility |

## 💡 Usage Examples

### Single Text Analysis

1. Navigate to **🔮 Single Prediction**
2. Enter or select financial text
3. Click **Analyze Emotion**
4. View results with confidence scores

### Batch Processing

1. Navigate to **📊 Batch Analysis**
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

## 📊 Performance Metrics

### Overall (LoRA v2)
- **Accuracy**: 61.0% (+8.3pp from v1)
- **Macro F1**: 0.61 (+114% from v1)
- **Model Size**: 2.8 MB (adapters only)
- **Inference Speed**: ~50ms per text

### Per-Class Recall
| Emotion | v1 | v2 | Improvement |
|---------|----|----|-------------|
| Hope | 0% | **82%** | +82pp 🚀 |
| Fear | 0% | **76%** | +76pp 🚀 |
| Excitement | 5% | **39%** | +34pp ⬆️ |
| Anxiety | 36% | **59%** | +23pp ✅ |
| Optimism | 66% | **90%** | +24pp ✅ |
| Uncertainty | 79% | **83%** | +4pp ✅ |

## 🛠️ Advanced Configuration

### Custom Model Path

Edit `app/pages/prediction.py` to use a different model:

```python
model_path = Path("path/to/your/model")
```

### Batch Size Adjustment

For large CSV files, process in batches by modifying `batch_analysis.py`:

```python
BATCH_SIZE = 100  # Process 100 texts at a time
```

## 🐛 Troubleshooting

### Model Not Found Error
- Ensure model files are in `models/finemo-lora-final-v2/`
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

## 📝 API Integration

Use the model programmatically:

```python
from app.utils.model_utils import load_model, predict_emotion

# Load model
model, tokenizer = load_model("models/finemo-lora-final-v2")

# Predict
text = "Markets surged on positive earnings reports"
emotion, confidence, probs = predict_emotion(
    text, model, tokenizer, return_probs=True
)

print(f"Emotion: {emotion} ({confidence:.1%} confidence)")
print(f"Probabilities: {probs}")
```

## 🎨 Customization

### Change Color Theme

Edit `app.py` CSS section:

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

## 📖 Documentation

- **Home Page**: Overview and quick stats
- **Single Prediction**: Real-time emotion analysis
- **Batch Analysis**: Process multiple texts
- **Model Comparison**: v1 vs v2 metrics
- **Documentation**: Full project guide

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- Multi-label classification support
- Additional visualization options
- Performance optimizations
- New emotion categories
- REST API endpoint

## 📄 License

This project is part of academic work at George Washington University.

## 👤 Author

**Vaishnavi Kamdi**
- Course: NNDL - Fall 2025, GWU
- GitHub: [@vaish725](https://github.com/vaish725)

## 🙏 Acknowledgments

- DistilBERT & LoRA papers
- Hugging Face Transformers
- Streamlit framework
- FinGPT & GoEmotions datasets

---

**Built with ❤️ using Streamlit, PyTorch, and Hugging Face**

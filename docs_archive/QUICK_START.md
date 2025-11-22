# FinEmo-LoRA Quick Start Guide

Get started with FinEmo-LoRA in 5 simple steps.

---

## Prerequisites

- Python 3.10+
- CUDA GPU with 16GB+ VRAM (for training)
- 50GB+ free disk space
- OpenAI API key (for annotation)
- Hugging Face account (for Llama access)

---

## 1. Installation (5 minutes)

```bash
# Navigate to project directory
cd "/Users/vaishnavikamdi/Documents/GWU/Classes/Fall 2025/NNDL/FinEmo-LoRA"

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 2. Setup API Keys (2 minutes)

```bash
# Copy environment template
cp .env.example .env

# Edit .env file and add your keys:
# OPENAI_API_KEY=sk-...
# HF_TOKEN=hf_...
```

Get your keys:
- **OpenAI**: https://platform.openai.com/api-keys
- **Hugging Face**: https://huggingface.co/settings/tokens (need to accept Llama license first)

---

## 3. Download and Prepare Data (30 minutes)

```bash
# Download GoEmotions (for transfer learning)
python scripts/data_collection/download_goemotions.py

# Preprocess GoEmotions (map 27 emotions to 6)
python scripts/data_collection/preprocess_goemotions.py

# Download FinGPT financial sentiment data
python scripts/data_collection/download_fingpt.py
```

---

## 4. Annotate Financial Text (1-2 hours)

```bash
# Annotate financial texts with 6-emotion taxonomy using GPT-4
python -c "
from scripts.annotation.llm_annotator import annotate_dataset

annotate_dataset(
    input_path='data/raw/fingpt/train.csv',
    output_path='data/annotated/fingpt_annotated.csv',
    text_column='text',  # Check actual column name in your data
    max_samples=5000,    # Start with 5000 samples
    batch_size=50
)
"
```

**Cost Estimate**: ~$20-40 for 5000 samples with GPT-4

**Tip**: Start with 500 samples for testing:
```python
max_samples=500  # Costs ~$2-4
```

---

## 5. Train the Model (6-12 hours)

### Stage 1: Emotion Understanding

```bash
# Transfer learning on GoEmotions (4-8 hours)
python scripts/training/train_stage1_goemotions.py
```

### Stage 2: Financial Domain Adaptation

```bash
# Fine-tune on financial emotions (2-4 hours)
python scripts/training/train_stage2_financial.py
```

---

## 6. Evaluate Results (10 minutes)

```bash
# Run comprehensive evaluation
python scripts/evaluation/evaluate.py

# View results
ls results/
# - predictions_*.csv
# - metrics_*.json
# - confusion_matrix_*.png
# - error_analysis_*.csv
```

---

## Monitor Training

### TensorBoard (Real-time Monitoring)

```bash
# Open new terminal
tensorboard --logdir=models/checkpoints

# Open browser to: http://localhost:6006
```

---

## Quick Test (Without Full Training)

Want to test the pipeline without full training? Use these reduced settings:

Edit `config.yaml`:
```yaml
training:
  stage1:
    epochs: 1  # Reduced from 3
  stage2:
    epochs: 2  # Reduced from 5
```

And annotate fewer samples:
```python
max_samples=500  # Instead of 5000
```

This completes in ~2-3 hours total.

---

## Troubleshooting

### Out of Memory?
- Reduce `batch_size` to 4 or 2 in `config.yaml`
- Increase `gradient_accumulation_steps` to 16 or 32

### API Rate Limits?
- Reduce `batch_size` in annotation (try 10-20)
- Add delays between batches

### Dataset Column Names?
- Check actual column names: `head data/raw/fingpt/train.csv`
- Update `text_column` parameter accordingly

---

## Expected Timeline

| Step | Time | GPU Required |
|------|------|-------------|
| Installation | 5 min | No |
| API Setup | 2 min | No |
| Data Download | 30 min | No |
| Annotation | 1-2 hrs | No |
| Stage 1 Training | 4-8 hrs | Yes |
| Stage 2 Training | 2-4 hrs | Yes |
| Evaluation | 10 min | Yes |
| **Total** | **8-15 hrs** | Yes (for training) |

---

## Next Steps

After completing the quick start:

1. **Review Results**: Analyze confusion matrix and per-class metrics
2. **Validate Quality**: Run validation pipeline on annotation sample
3. **Experiment**: Try different hyperparameters or base models
4. **Deploy**: Use the model for inference on new financial text

See full README.md for detailed documentation.

---

## Need Help?

- Check README.md for detailed documentation
- Review config.yaml for all configuration options
- Open an issue for bugs or questions

Happy training!


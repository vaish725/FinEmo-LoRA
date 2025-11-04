# FinEmo-LoRA: Logits-based Economic Emotion Classification

**A Fast, Practical Approach to Fine-Grained Emotion Interpretation from Financial News**

Using LLM embeddings + lightweight classifier instead of fine-tuning  
**Completes in 1-2 hours instead of 8-15 hours!**

**Author:** Vaishnavi Kamdi  
**Course:** CSCI 6907 Neural Networks and Deep Learning  
**Institution:** George Washington University  
**Professor:** Joel Klein

---

## Overview

Traditional financial sentiment analysis oversimplifies market reactions into Positive/Negative/Neutral. This project classifies financial text into 6 nuanced economic emotions:

**Target Emotions:** `anxiety`, `excitement`, `optimism`, `fear`, `uncertainty`, `hope`

### Approach: Logits-based Classification

Instead of fine-tuning an LLM (which takes days), we:

1. **Extract features** from a frozen pre-trained LLM (FinBERT/Llama/Phi-2)
2. **Train a lightweight classifier** (MLP/XGBoost) on these features
3. **Achieve comparable results** in a fraction of the time

**Benefits:**
- ✅ **Fast**: 1-2 hours total instead of 8-15 hours
- ✅ **Memory efficient**: No GPU needed for training classifier
- ✅ **Easy to debug**: Simpler pipeline with fewer moving parts
- ✅ **Flexible**: Easy to experiment with different classifiers
- ✅ **Practical**: Suitable for 1-month solo project timeline

---

## Quick Start (1-2 hours)

### Prerequisites

```bash
# Python 3.10+, GPU optional (CPU works fine)
cd "/Users/vaishnavikamdi/Documents/GWU/Classes/Fall 2025/NNDL/FinEmo-LoRA"

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Edit .env and add:
# OPENAI_API_KEY=sk-...
# HF_TOKEN=hf_...  (optional, only if using Llama)
```

### Run Complete Pipeline

```bash
# Run the entire pipeline with one command!
python run_logits_pipeline.py --annotation-samples 2000
```

That's it! The pipeline will:
1. Download FinGPT dataset
2. Annotate 2000 samples with GPT-4 (~$8-16)
3. Extract features using FinBERT
4. Train MLP classifier
5. Generate results and confusion matrix

**Total time: 1-2 hours**  
**Total cost: $8-16**

---

## Step-by-Step Guide

If you prefer manual control:

### Step 1: Download Data (5 min)

```bash
python scripts/data_collection/download_fingpt.py
```

### Step 2: Annotate with GPT-4 (30-60 min)

```python
from scripts.annotation.llm_annotator import annotate_dataset

annotate_dataset(
    input_path='data/raw/fingpt/train.csv',
    output_path='data/annotated/fingpt_annotated.csv',
    text_column='text',
    max_samples=2000,  # Start small!
    batch_size=50
)
```

**Cost estimate**: ~$8-16 for 2000 samples

### Step 3: Extract Features (10-20 min)

```python
from scripts.feature_extraction.extract_features import extract_and_save_features

extract_and_save_features(
    input_file='data/annotated/fingpt_annotated_high_confidence.csv',
    output_file='data/features/train_features.npy',
    text_column='text'
)
```

This extracts 768-dimensional embeddings from FinBERT (or your chosen LLM).

### Step 4: Train Classifier (5-15 min)

```python
from scripts.classifier.train_classifier import train_and_evaluate

train_and_evaluate(
    features_file='data/features/train_features.npy',
    labels_file='data/annotated/fingpt_annotated_high_confidence.csv',
    classifier_type='mlp'  # Options: mlp, xgboost, svm, random_forest
)
```

### Step 5: Review Results

Check `models/classifiers/` for:
- Trained model (.pkl file)
- Confusion matrix (PNG visualization)
- Per-class metrics

---

## Project Structure

```
FinEmo-LoRA/
├── config.yaml                          # Configuration (updated for logits approach)
├── run_logits_pipeline.py               # One-command pipeline runner
│
├── scripts/
│   ├── data_collection/
│   │   └── download_fingpt.py           # Download financial text
│   ├── annotation/
│   │   └── llm_annotator.py             # GPT-4 annotation
│   ├── feature_extraction/
│   │   └── extract_features.py          # Extract LLM embeddings
│   └── classifier/
│       └── train_classifier.py          # Train lightweight classifier
│
├── data/
│   ├── raw/                             # Downloaded datasets
│   ├── annotated/                       # GPT-4 annotated data
│   └── features/                        # Extracted embeddings (.npy files)
│
└── models/
    └── classifiers/                     # Trained classifiers + results
```

---

## Configuration

All settings in `config.yaml`:

### Choose Feature Extraction Model

```yaml
model:
  selected: "finbert"  # Options: finbert, phi, llama
  
  feature_extractors:
    finbert:              # Recommended: Fast, finance-specific
      name: "ProsusAI/finbert"
      batch_size: 32
    
    phi:                  # Alternative: Larger, more general
      name: "microsoft/phi-2"
      batch_size: 16
    
    llama:                # Alternative: Most powerful but slowest
      name: "meta-llama/Meta-Llama-3.1-8B"
      batch_size: 8
```

### Choose Classifier

```yaml
classifier:
  selected: "mlp"  # Options: mlp, xgboost, svm, random_forest
```

---

## Experiment with Different Configurations

Once you have features extracted, try different classifiers instantly:

```bash
# Try XGBoost
python run_logits_pipeline.py --skip-download --skip-annotation \
  --skip-extraction --classifier xgboost

# Try SVM
python run_logits_pipeline.py --skip-download --skip-annotation \
  --skip-extraction --classifier svm

# Try Random Forest
python run_logits_pipeline.py --skip-download --skip-annotation \
  --skip-extraction --classifier random_forest
```

This lets you compare multiple approaches for your report!

---

## Expected Results

Based on similar financial NLP tasks:

| Metric | Expected Range |
|--------|----------------|
| Overall Accuracy | 65-80% |
| Macro F1-Score | 0.60-0.75 |
| Training Time | 5-15 minutes |
| Feature Extraction Time | 10-20 minutes |

**Note:** Results depend on:
- Quality of GPT-4 annotations
- Amount of training data
- Class balance
- Choice of feature extractor

---

## Methodology Justification

### Why Logits-based Instead of Fine-tuning?

**Practical Advantages:**
- ✅ **Time**: 1-2 hours vs 8-15 hours
- ✅ **Complexity**: Simpler to debug and understand
- ✅ **Resources**: Works on CPU, doesn't need powerful GPU
- ✅ **Iteration**: Fast experimentation with different classifiers

**Academic Validity:**
- ✅ Still uses state-of-the-art LLMs (FinBERT/Llama/Phi-2)
- ✅ Demonstrates understanding of transfer learning
- ✅ Common approach in production ML systems
- ✅ Allows comparison of multiple classifiers

**Performance:**
- ✅ Often achieves 85-95% of fine-tuned performance
- ✅ Especially effective with strong feature extractors (FinBERT)
- ✅ Can surpass fine-tuning with limited data

---

## Timeline for 1-Month Project

### Week 1: Data & Baseline (5-10 hours)
- **Days 1-2**: Setup + download datasets (2 hours)
- **Days 3-4**: Annotate 1000 samples for quick test (3 hours)
- **Days 5-6**: Extract features + train first classifier (2 hours)
- **Day 7**: Baseline results + iteration (3 hours)

### Week 2: Refinement (8-12 hours)
- **Days 1-2**: Annotate remaining samples to reach 2000 (4 hours)
- **Days 3-4**: Experiment with different feature extractors (3 hours)
- **Days 5-6**: Experiment with different classifiers (2 hours)
- **Day 7**: Error analysis + improvements (3 hours)

### Week 3: Final Model (5-8 hours)
- **Days 1-2**: Train final model on best configuration (2 hours)
- **Days 3-4**: Comprehensive evaluation (2 hours)
- **Days 5-7**: Create visualizations + prepare results (4 hours)

### Week 4: Report & Polish (15-20 hours)
- **Days 1-3**: Write comprehensive report (12 hours)
- **Days 4-5**: Create presentation (optional) (4 hours)
- **Days 6-7**: Final review + polish (4 hours)

**Buffer:** 5-8 hours for unexpected issues

---

## Cost Breakdown

| Item | Cost |
|------|------|
| GPT-4 annotation (2000 samples) | $8-16 |
| Compute (can use free tier) | $0 |
| **Total** | **$8-16** |

**Budget options:**
- Start with 500 samples ($2-4) for testing
- Scale to 2000 samples ($8-16) for final model
- Can go up to 5000 samples ($20-40) if needed

---

## Deliverables for Project

1. ✅ **Working Classifier** with 6-emotion taxonomy
2. ✅ **Evaluation Metrics** (Accuracy, F1, Precision, Recall per class)
3. ✅ **Confusion Matrix** visualization
4. ✅ **Comparison** of multiple classifiers (MLP vs XGBoost vs SVM)
5. ✅ **Error Analysis** identifying common mistakes
6. ✅ **Code Repository** with clear documentation
7. ✅ **Report** discussing methodology, results, and insights

---

## Troubleshooting

### Out of Memory During Feature Extraction

Reduce batch size in `config.yaml`:
```yaml
model:
  feature_extractors:
    finbert:
      batch_size: 16  # Reduce from 32
```

### Annotation Fails

Check:
- `OPENAI_API_KEY` is set in `.env`
- API has sufficient credits
- Column name matches actual data (might not be 'text')

### Low Classification Performance

Try:
- Annotate more samples (1000 → 2000 → 3000)
- Use FinBERT instead of generic LLM
- Try XGBoost instead of MLP
- Check class balance in data

---

## Research Context

This approach is based on:

1. **Feature-based Transfer Learning** - Common in production ML
2. **Financial Domain Embeddings** - FinBERT for finance-specific understanding
3. **LLM-assisted Annotation** - Using GPT-4 for high-quality labels
4. **Lightweight Classifiers** - Efficient and interpretable

**Key Papers:**
- FinBERT: Financial Domain Pre-training
- Sentence Transformers for embedding extraction
- Transfer learning with frozen features

---

## Future Extensions

After completing the base project, you could explore:

1. **Ensemble Methods**: Combine multiple classifiers
2. **Active Learning**: Strategically select samples for annotation
3. **Multi-label Classification**: Detect multiple emotions
4. **Comparison with Fine-tuning**: Implement LoRA as comparison (if time permits)
5. **Real-time System**: Deploy as API for live classification

---

## Contact

**Vaishnavi Kamdi**  
George Washington University  
CSCI 6907 Neural Networks and Deep Learning

---

## Acknowledgments

- **Professor Joel Klein** for suggesting the logits-based approach
- **Hugging Face** for pre-trained models
- **OpenAI** for GPT-4 API
- **FinBERT Team** for financial domain embeddings

---

**Ready to start?** Run:

```bash
python run_logits_pipeline.py --annotation-samples 2000
```

Good luck with your project! 🚀


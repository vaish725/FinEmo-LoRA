# CPU-Friendly Evaluation Guide for FinEmo-LoRA

## Overview
Your FinEmo-LoRA project uses a **logits-based approach** (feature extraction + lightweight classifier), which is perfect for CPU-only environments. **You do NOT need GPU or LLaMA models for evaluation!**

## What We've Set Up

### 1. CPU-Optimized Configuration
- **Model**: DistilBERT (CPU-friendly, 768-dim features)
- **Classifier**: MLP PyTorch model (trained, pickled)
- **Device**: CPU (configurable to MPS on Apple Silicon)
- **Dependencies**: All installed in `.venv`

### 2. New Evaluation Scripts

#### Fast Evaluation (End-to-End)
```bash
# Activate venv
source .venv/bin/activate

# Run complete evaluation (extracts features + evaluates classifier)
python scripts/evaluation/run_full_evaluation.py \
    --classifier models/classifiers/mlp_20251103_200252.pkl \
    --data data/annotated/fingpt_annotated.csv \
    --device cpu \
    --test-size 0.3
```

**What it does:**
1. Loads annotated financial texts
2. Extracts features using DistilBERT (CPU)
3. Runs MLP classifier
4. Generates metrics + confusion matrix

**Current Performance (Nov 21, 2025):**
- Overall Accuracy: **63.33%**
- Best classes: uncertainty (81% F1), hope (67% F1)
- Weak classes: excitement, optimism, fear (need more training data)

## Running Without GPU

### Option 1: Your Current Setup (RECOMMENDED)
✅ **Feature extraction** with DistilBERT/FinBERT on CPU  
✅ **Lightweight classifier** (MLP/XGBoost) on CPU  
✅ **Fast inference** (seconds, not hours)

This is what you have now and it works great!

### Option 2: Apple Silicon (MPS)
If you have M1/M2 Mac, use MPS for faster feature extraction:
```bash
python scripts/evaluation/run_full_evaluation.py \
    --classifier models/classifiers/mlp_20251103_200252.pkl \
    --device mps
```

### Option 3: Cloud GPU (if needed)
For training or heavy workloads:
- Google Colab (free GPU)
- AWS/GCP spot instances
- Hugging Face Spaces

## Required Credentials

### Mandatory for Your Workflow
1. **OPENAI_API_KEY** (for annotation) - ✅ You have this
2. **HF_TOKEN** (for downloading models from Hugging Face) - Get this next

### Optional
3. **Kaggle API** (only for SEntFiN dataset) - Skip unless needed

### How to Get HF_TOKEN
1. Go to https://huggingface.co/settings/tokens
2. Create a new token (read access)
3. Add to `.env` file:
   ```
   OPENAI_API_KEY=sk-...
   HF_TOKEN=hf_...
   ```

## Virtual Environment Setup

### Create and Activate
```bash
# One-time setup
python3 -m venv .venv

# Activate (run this every session)
source .venv/bin/activate

# Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# macOS-specific: Install OpenMP for XGBoost
brew install libomp
```

### Always Use venv
```bash
# Activate first
source .venv/bin/activate

# Then run any Python command
python scripts/evaluation/run_full_evaluation.py --help
python run_pipeline.py --stage evaluate
```

## Pipeline Stages

### What Each Stage Does
1. **Download** - Get datasets (GoEmotions, FinGPT, SEntFiN)
2. **Annotation** - Label financial text with GPT-4 (uses OPENAI_API_KEY)
3. **Stage 1 Training** - Train on GoEmotions (emotion understanding)
4. **Stage 2 Training** - Fine-tune on financial emotions
5. **Evaluation** - Test classifier on held-out data

### Run Specific Stages
```bash
# Only evaluation
python run_pipeline.py --stage evaluate

# Only annotation
python run_pipeline.py --stage annotation

# Train both stages
python run_pipeline.py --stage train
```

## Current Project Status

### ✅ Completed
- [x] Environment setup (venv, dependencies)
- [x] CPU-friendly configuration
- [x] Unified requirements.txt
- [x] Working evaluation pipeline
- [x] MLP classifier trained (63.33% accuracy)
- [x] Confusion matrix generation

### 🔄 Next Steps (Priority Order)
1. **Error Analysis** (todo #3)
   - Review confusion matrix in `results/`
   - Identify why excitement/optimism/fear have low F1
   - Sample misclassified examples

2. **Improve Low-Confidence Annotations** (todo #4)
   - Check `data/annotated/fingpt_annotated_low_confidence.csv`
   - Re-annotate ambiguous examples
   - Run `scripts/annotation/validate_annotations.py`

3. **Retrain if Needed** (todo #5)
   - If annotation quality improves, retrain classifier
   - Experiment with hyperparameters (hidden layers, dropout)
   - Try XGBoost: `models/classifiers/xgboost_20251103_200426.pkl`

4. **Results Write-up** (todo #6)
   - Document metrics and findings
   - Add confusion matrix to README
   - Prepare presentation slides

## Quick Commands Reference

### Evaluation
```bash
# Activate venv
source .venv/bin/activate

# Run evaluation (CPU)
python scripts/evaluation/run_full_evaluation.py \
    --classifier models/classifiers/mlp_20251103_200252.pkl \
    --device cpu

# Try XGBoost classifier
python scripts/evaluation/run_full_evaluation.py \
    --classifier models/classifiers/xgboost_20251103_200426.pkl \
    --device cpu
```

### View Results
```bash
# See metrics
cat results/evaluation_metrics_*.json

# View confusion matrix
open results/confusion_matrix_*.png
```

### Feature Extraction
```bash
# Extract features from new data
python scripts/feature_extraction/extract_features.py \
    --input data/annotated/fingpt_annotated.csv \
    --output data/features/test_features.npy \
    --model distilbert-base-uncased \
    --device cpu
```

## Troubleshooting

### Issue: Module not found
```bash
# Ensure venv is activated
source .venv/bin/activate

# Reinstall requirements
pip install -r requirements.txt
```

### Issue: XGBoost fails on macOS
```bash
# Install OpenMP
brew install libomp
```

### Issue: Out of memory
```bash
# Reduce batch size
python scripts/evaluation/run_full_evaluation.py \
    --batch-size 8 \
    --device cpu
```

### Issue: Slow feature extraction
- Use MPS if on Apple Silicon: `--device mps`
- Or reduce test set size: `--test-size 0.1`
- Or use cached features if available

## Performance Notes

### Current Metrics (200 samples, 30% test split)
```
Overall Accuracy: 63.33%

Per-Class F1 Scores:
  uncertainty: 0.81 (25 samples) ✅ Good
  hope:        0.67 (17 samples) ✅ OK
  anxiety:     0.50 (7 samples)  ⚠️  Needs work
  excitement:  0.00 (6 samples)  ❌ Low data
  optimism:    0.00 (3 samples)  ❌ Low data
  fear:        0.00 (2 samples)  ❌ Low data
```

### Recommendations
1. **Collect more data** for excitement, optimism, fear
2. **Validate annotations** for low-confidence samples
3. **Balance dataset** (currently skewed toward uncertainty/optimism)
4. **Try XGBoost** (may perform better on small datasets)

## FAQ

**Q: Do I need LLaMA or GPU?**  
A: No! Your pipeline uses feature extraction (DistilBERT) + classifier. This is CPU-friendly.

**Q: What is llama-cpp-python for?**  
A: Only if you want to run large LLaMA models locally. Not needed for your current approach.

**Q: Can I use Apple Silicon (M1/M2)?**  
A: Yes! Use `--device mps` for faster feature extraction.

**Q: How long does evaluation take?**  
A: ~10-30 seconds on CPU for 60 test samples.

**Q: What if I want to retrain?**  
A: Run `python scripts/classifier/train_classifier.py` (see script for options).

**Q: Where are results saved?**  
A: `results/` directory (metrics JSON + confusion matrix PNG).

## Summary

You're in the **evaluation and refinement stage**. Your setup is now fully CPU-friendly and working. Focus on:

1. Analyzing errors (confusion matrix)
2. Improving annotations (low-confidence samples)
3. Retraining with better data
4. Writing up results

No GPU required! 🎉

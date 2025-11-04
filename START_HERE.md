# 🚀 START HERE: FinEmo-LoRA Logits-based Pipeline

**Welcome!** Your complete logits-based emotion classification project is ready.

---

## What You Have

✅ **Fast, practical approach** using LLM embeddings + lightweight classifier  
✅ **1-2 hour pipeline** instead of 8-15 hours  
✅ **Complete for 1-month solo project**  
✅ **~$8-16 total cost** (GPT-4 annotation)  

---

## Your Datasets (APPROVED!)

The three datasets you chose **ARE PERFECT** with the logits-based approach:

1. **FinGPT** → Financial text (re-annotate with GPT-4)
2. **SEntFiN** → More financial text (re-annotate with GPT-4)  
3. **GoEmotions** → Not used in logits approach (optional for analysis)

---

## Quick Start (Choose One)

### Option 1: One-Command Pipeline (Easiest)

```bash
# 1. Setup
cd "/Users/vaishnavikamdi/Documents/GWU/Classes/Fall 2025/NNDL/FinEmo-LoRA"
python -m venv venv
source venv/bin/activate
pip install -r requirements_logits.txt

# 2. Configure .env
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 3. Run everything!
python run_logits_pipeline.py --annotation-samples 2000
```

Done! Results in `models/classifiers/`

---

### Option 2: Step-by-Step (More Control)

```bash
# 1. Setup (same as above)
# ...

# 2. Download data
python scripts/data_collection/download_fingpt.py

# 3. Annotate (30-60 min, ~$8-16)
python -c "
from scripts.annotation.llm_annotator import annotate_dataset
annotate_dataset(
    input_path='data/raw/fingpt/train.csv',
    output_path='data/annotated/fingpt_annotated.csv',
    text_column='text',
    max_samples=2000,
    batch_size=50
)
"

# 4. Extract features (10-20 min)
python -c "
from scripts.feature_extraction.extract_features import extract_and_save_features
extract_and_save_features(
    input_file='data/annotated/fingpt_annotated_high_confidence.csv',
    output_file='data/features/train_features.npy',
    text_column='text'
)
"

# 5. Train classifier (5-15 min)
python -c "
from scripts.classifier.train_classifier import train_and_evaluate
train_and_evaluate(
    features_file='data/features/train_features.npy',
    labels_file='data/annotated/fingpt_annotated_high_confidence.csv',
    classifier_type='mlp'
)
"
```

---

## What's Different from LoRA?

| Aspect | LoRA (Original) | Logits-based (NEW) |
|--------|-----------------|-------------------|
| Training Time | 6-12 hours | 30-90 minutes |
| GPU Required | Yes (powerful) | Optional (any) |
| Complexity | High | Medium |
| Cost | Higher | Lower |
| Performance | Slightly higher | Comparable |
| Timeline | Risky for 1 month | Safe for 1 month |

**Professor's feedback**: Use logits approach → You're following his recommendation!

---

## Key Files to Know

### Documentation
- **`README_LOGITS.md`** → Complete guide (read this!)
- **`TIMELINE_1MONTH.md`** → Week-by-week plan
- **`START_HERE.md`** → This file

### Code
- **`run_logits_pipeline.py`** → Run entire pipeline
- **`config.yaml`** → Configure everything here
- **`scripts/feature_extraction/extract_features.py`** → Extract LLM embeddings
- **`scripts/classifier/train_classifier.py`** → Train classifier

### Old Files (Ignore These)
- `scripts/training/train_stage1_goemotions.py` → Was for LoRA
- `scripts/training/train_stage2_financial.py` → Was for LoRA
- `README.md` → Was for LoRA approach
- `run_pipeline.py` → Was for LoRA approach

---

## Configuration Options

Edit `config.yaml`:

### Choose LLM for Feature Extraction
```yaml
model:
  selected: "finbert"  # Options: finbert, phi, llama
```

**Recommendation**: Start with `finbert` (fastest, finance-specific)

### Choose Classifier
```yaml
classifier:
  selected: "mlp"  # Options: mlp, xgboost, svm, random_forest
```

**Recommendation**: Start with `mlp`, then try `xgboost`

### Annotation Settings
```yaml
annotation:
  target_samples: 2000  # Reduce to 1000 for faster testing
  min_confidence: 0.7   # Keep high-quality annotations only
```

---

## Recommended Workflow

### Week 1: Get Baseline
1. Test with 500 samples
2. Get working end-to-end pipeline
3. Baseline results: ~60-70% accuracy expected

### Week 2: Scale Up
1. Annotate full 2000 samples
2. Try FinBERT, Phi-2 for features
3. Try MLP, XGBoost classifiers
4. Target: 65-80% accuracy

### Week 3: Final Model
1. Train final model on best config
2. Complete evaluation
3. Generate all visualizations

### Week 4: Report
1. Write 10-15 page report
2. Document methodology
3. Present results

---

## Cost Management

### Testing Phase (Week 1)
- 500 samples = $2-4
- Test everything works

### Production Phase (Week 2)
- 2000 samples = $8-16
- Final dataset

### Total: $10-20

**Budget tip**: Start with 100 samples (free tier) to verify pipeline works!

---

## Expected Results

Based on similar projects:

- **Accuracy**: 65-80%
- **Macro F1**: 0.60-0.75
- **Best emotions**: optimism, fear (clear signals)
- **Hardest emotions**: anxiety vs fear, hope vs optimism

---

## Troubleshooting

### "Column 'text' not found"
Check actual column name in data:
```bash
head data/raw/fingpt/train.csv
```
Update `text_column` parameter accordingly.

### "OPENAI_API_KEY not found"
Add to `.env` file:
```
OPENAI_API_KEY=sk-...
```

### Out of memory
Reduce batch size in `config.yaml`:
```yaml
model:
  feature_extractors:
    finbert:
      batch_size: 16  # Reduce from 32
```

---

## Email Your Professor

Use this template:

```
Subject: Dataset Approach for FinEmo-LoRA Project

Dear Professor Klein,

Thank you for suggesting the logits-based classification approach.

I've implemented the pipeline using:
- FinGPT + SEntFiN datasets (re-annotated with GPT-4 for my 6-emotion taxonomy)
- LLM embeddings extraction (testing FinBERT, Phi-2)
- Lightweight classifier (MLP, XGBoost comparison)

This approach is much more feasible for the 1-month timeline and allows me to:
- Complete experiments in 1-2 hours instead of days
- Easily compare multiple classifiers
- Focus on methodology and analysis

I plan to annotate 2000 samples (~$8-16) and compare multiple 
feature extractors and classifiers in my report.

Please let me know if this approach aligns with your expectations.

Best regards,
Vaishnavi Kamdi
```

---

## Next Steps

1. **Right now**: Setup environment and run test with 100 samples
2. **This week**: Get baseline results
3. **Next week**: Scale to full dataset
4. **Week 3**: Final experiments
5. **Week 4**: Write report

---

## Need Help?

1. **Documentation**: Read `README_LOGITS.md`
2. **Timeline**: Check `TIMELINE_1MONTH.md`
3. **Configuration**: See `config.yaml` with comments
4. **Code**: All scripts have detailed comments

---

## You're Ready! 

```bash
# Test the pipeline right now (5 minutes, free):
python run_logits_pipeline.py --annotation-samples 100
```

This will:
1. Download data
2. Annotate 100 samples (~$0.40)
3. Extract features
4. Train classifier
5. Show results

If this works, scale to 2000 samples for your final project!

**Good luck!** 🎉


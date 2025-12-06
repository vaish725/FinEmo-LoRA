# FinEmo-LoRA: Financial Emotion Detection Using Parameter-Efficient Fine-Tuning

**Neural Networks and Deep Learning - Final Project**  
**December 2025**

[![Accuracy](https://img.shields.io/badge/Accuracy-76.8%25-success)](notebooks/FinEmo_LoRA_Training.ipynb)
[![Improvement](https://img.shields.io/badge/Improvement-+30.5pp-blue)](notebooks/FinEmo_LoRA_Training.ipynb)
[![LoRA](https://img.shields.io/badge/Parameters-0.3%25_trainable-orange)](notebooks/FinEmo_LoRA_Training.ipynb)
[![Cost](https://img.shields.io/badge/Cost-$1.63-green)](SUBMISSION_GUIDE.md)

A state-of-the-art emotion detection system for financial texts using **LoRA (Low-Rank Adaptation)** - achieving **76.8% accuracy** with only **0.3% trainable parameters**.

---

## 👤 Author

- **Vaishnavi Kamdi** - GitHub: [@vaish725](https://github.com/vaish725)

**Course:** CSCI 4/6366 Neural Networks and Deep Learning  
**Institution:** George Washington University  
**Semester:** Fall 2025

---

## 📊 Key Results

| Metric | Baseline | v1 | **v2 (Final)** |
|--------|----------|-----|----------------|
| **Accuracy** | 46.3% | 52.7% | **76.8%** ✅ |
| **Hope Recall** | N/A | 0% | **95%** 🚀 |
| **Fear Recall** | N/A | 0% | **50%** ✅ |
| **Excitement Recall** | N/A | 5% | **79%** 🎯 |
| **Training Time** | 30 min | 50 min | **60 min** |
| **Trainable Params** | 66M | 200K | **200K (0.3%)** |
| **Cost** | $0 | $0.50 | **$1.63** 💰 |

**Achievement**: Exceeded 75% accuracy target by **1.8 percentage points**!

---

## 🎯 Project Overview

Traditional financial sentiment analysis oversimplifies emotions into Positive/Negative/Neutral. This project develops a nuanced 6-class emotion classifier using parameter-efficient fine-tuning:

**Emotions**: `anxiety`, `excitement`, `fear`, `hope`, `optimism`, `uncertainty`

**Innovation**: Two-stage LoRA training pipeline combining general emotion knowledge (GoEmotions) with financial domain expertise (FinGPT).

---

## 🚀 Quick Start

### Option 1: View Final Presentation (Recommended)

Open **[`notebooks/FinEmo_LoRA_FINAL_PRESENTATION.ipynb`](notebooks/FinEmo_LoRA_FINAL_PRESENTATION.ipynb)** for:
- Complete project walkthrough
- Architecture diagrams
- Results analysis
- Live inference demo

### Option 2: Run Training (Google Colab)

1. Upload [`notebooks/FinEmo_LoRA_Training.ipynb`](notebooks/FinEmo_LoRA_Training.ipynb) to Google Colab
2. Change runtime to GPU (T4 or better)
3. Upload `data/annotated/fingpt_annotated_enhanced.csv`
4. Run all cells (60 min)
5. Achieves **76.8% accuracy**

### Option 3: Run Inference Only

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

# Load model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
base_model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=6
)
model = PeftModel.from_pretrained(base_model, "./models/finemo_lora_final_v2")

# Predict
text = "Strong Q4 guidance suggests revenue growth ahead."
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
outputs = model(**inputs)
prediction = outputs.logits.argmax().item()
# Output: 3 (hope) with 94% confidence
```

### Option 2: Local Training (macOS/Linux)

```bash
# Run complete pipeline
python3 run_pipeline.py --pipeline logits --improved

# Or train manually
python3 scripts/classifier/train_classifier.py \
  --features data/features/train_features_scaled.npy \
  --labels data/annotated/fingpt_annotated_scaled.csv \
  --classifier xgboost --improved
```

**Note:** Deep MLP training will fail on macOS due to PyTorch bug. Use XGBoost or Colab instead.

---

## Dataset Sources

All datasets used in this project are publicly available:

### 1. FinGPT Sentiment Dataset
- **Source:** [FinGPT GitHub Repository](https://github.com/FinancialDiets/FINGPT)
- **Description:** Financial news headlines and social media posts with sentiment labels
- **License:** Apache 2.0
- **Usage:** Primary source for financial text to be annotated with emotions

### 2. SEntFiN (Sentiment in Financial News)
- **Source:** [SEntFiN Kaggle Dataset](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news)
- **Description:** Entity-aware sentiment analysis for financial news
- **License:** CC BY-SA 4.0
- **Usage:** Additional financial domain text for training

### 3. GoEmotions
- **Source:** [Google Research GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions)
- **Paper:** [GoEmotions: A Dataset of Fine-Grained Emotions](https://arxiv.org/abs/2005.00547)
- **Description:** 58k Reddit comments labeled with 27 emotions
- **License:** Apache 2.0
- **Usage:** Transfer learning for general emotion understanding (Stage 1)

---

## Methodology & Approach

### Two-Track Exploration

This project explores **two complementary approaches** to financial emotion classification:

#### Track 1: Logits-Based Classification (Current Implementation)
- **Architecture:** Frozen DistilBERT feature extractor + lightweight MLP/XGBoost classifier
- **Advantages:** Fast training (minutes), CPU-friendly, interpretable
- **Status:**  Complete - Achieved 63.33% accuracy on initial data
- **Challenge Identified:** Annotation quality issues causing performance ceiling

#### Track 2: LoRA Fine-Tuning (Planned)
- **Architecture:** Two-stage LoRA fine-tuning (Llama 3.1 8B or Phi-2)
- **Stage 1:** Transfer learning on GoEmotions (27 emotions → 6 economic emotions)
- **Stage 2:** Financial domain adaptation on annotated FinGPT data
- **Advantages:** Higher capacity, end-to-end learning, state-of-the-art potential
- **Status:**  Pipeline ready, awaiting cleaned dataset

### Current Focus: Data Quality Improvement

**Discovery:** Error analysis revealed that ~60-70% of "optimism" labels were neutral factual statements (e.g., "Company acquires assets for €420 million").

**Action Taken:**
1. Created automated review tool (`review_annotations.py`) with sentiment heuristics
2. Cleaned 83 optimism samples → 54 true optimism + 18 relabeled to uncertainty
3. Merged cleaned data into `fingpt_annotated_v2.csv` (189 samples)
4. Retrained with class-weighted loss to handle imbalance

**Next Steps:**
- Analyze performance on cleaned data vs. original
- Consider 3-class taxonomy (negative/positive/uncertainty) for small dataset
- Scale up annotation to 500+ samples for 6-class classification

---

## Project Structure

```
FinEmo-LoRA/
 config.yaml                         # Main configuration file
 requirements.txt                    # Python dependencies
 .env.example                        # Environment variables template
 README.md                           # This file

 data/
    raw/                           # Raw downloaded datasets
       fingpt/                    # FinGPT sentiment data (200 samples)
    annotated/                     # LLM-annotated financial text
       fingpt_annotated.csv       # Original GPT-4 annotations
       fingpt_annotated_v2.csv    # Cleaned annotations (189 samples)
    features/                      # Extracted DistilBERT features
        train_features.npy         # Original features (200x768)
        train_features_v2.npy      # Cleaned features (189x768)

 models/
    classifiers/                   # Trained lightweight classifiers
       mlp_*.pkl                  # MLP PyTorch models
       xgboost_*.pkl              # XGBoost models
       confusion_matrix_*.png     # Confusion matrices
    finemo-lora-final/             # (Future) LoRA fine-tuned model

 scripts/
    data_collection/               # Dataset download & preprocessing
       download_fingpt.py         # Download FinGPT data
       download_sentfin.py        # Download SEntFiN (Kaggle)
       download_goemotions.py     # Download GoEmotions
    annotation/                    # Annotation quality control
       llm_annotator.py           # GPT-4 annotation with confidence
       review_annotations.py      # Automated annotation cleaning
       merge_cleaned_annotations.py # Merge cleaned data
    feature_extraction/            # Feature extraction pipeline
       extract_features.py        # DistilBERT feature extraction
    classifier/                    # Lightweight classifier training
       train_classifier.py        # MLP/XGBoost training
    training/                      # (Future) LoRA training pipelines
       train_stage1_goemotions.py # Stage 1: Emotion understanding
       train_stage2_financial.py  # Stage 2: Financial adaptation
       retrain_with_cleaned_data.py # Retrain with cleaned data
    evaluation/                    # Evaluation framework
        evaluate.py                # Comprehensive evaluation
        run_full_evaluation.py     # CPU-friendly evaluation

 results/                           # Evaluation results
    evaluation_20251103/           # Latest evaluation results

 logs/                              # Training logs
 notebooks/                         # Analysis notebooks
 config.yaml                        # Central configuration
 requirements.txt                   # Python dependencies
 run_pipeline.py                    # Main orchestration script
 README.md                          # This file
```

---

## Installation & Setup

### Prerequisites

- **Python 3.10+** (tested on Python 3.13.7)
- **CPU or GPU:** Current implementation runs on CPU (no GPU required)
- **For LoRA training (future):** CUDA-capable GPU with 16GB+ VRAM recommended

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/vaish725/FinEmo-LoRA.git
   cd FinEmo-LoRA
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download sample data**
   ```bash
   # Download FinGPT dataset
   python scripts/data_collection/download_fingpt.py
   ```

5. **Run evaluation on existing model**
   ```bash
   # Evaluate pre-trained classifier
   python run_pipeline.py --stage evaluate
   ```

---

## Initial Work & Current Results

### Completed Components

#### 1. Data Pipeline ( Complete)
```bash
# Download financial news data
python scripts/data_collection/download_fingpt.py
```
- Downloaded 200 FinGPT samples for initial experiments
- Implemented data loaders and preprocessing utilities

#### 2. GPT-4 Annotation System ( Complete)
```python
from scripts.annotation.llm_annotator import LLMAnnotator

annotator = LLMAnnotator(api_key="your-key")
annotator.annotate_batch(texts, batch_size=50)
```
- Created prompt engineering system for 6-emotion classification
- Implemented confidence scoring and batch processing
- Generated 200 annotated samples with reasoning

**Example Annotation:**
```
Text: "Federal Reserve signals potential rate hikes amid inflation concerns"
Emotion: anxiety
Confidence: 0.85
Reasoning: "Text conveys worry about future economic conditions due to potential 
           rate increases and inflation."
```

#### 3. Feature-Based Classifier ( Complete)
```bash
# Train lightweight classifier
python scripts/classifier/train_classifier.py
```

**Architecture:**
- **Feature Extractor:** DistilBERT (768-dimensional embeddings, frozen)
- **Classifier:** 3-layer MLP (768 → 512 → 256 → 128 → 6 classes)
- **Training:** Cross-entropy loss with class weights
- **Inference:** CPU-friendly (3-4 seconds per batch of 32)

**Results (Initial Model):**
```
Accuracy: 63.33%
Macro F1: 0.33

Per-Class F1 Scores:
  anxiety:      0.57
  excitement:   0.00  
  fear:         0.00  
  hope:         1.00
  optimism:     0.00  
  uncertainty:  0.42
```

#### 4. Error Analysis & Data Quality Review ( Complete)

**Key Finding:** Manual inspection revealed **annotation quality issues**:
- 60-70% of "optimism" labels were neutral factual statements
- Example: "Company acquires €420M in assets" (factual, not emotional)
- GPT-4 confused business-positive context with emotional optimism

**Solution Implemented:**
```bash
# Automated annotation cleaning
python scripts/annotation/review_annotations.py --auto

# Results:
# - Original: 83 optimism samples
# - Cleaned: 54 true optimism + 18 relabeled to uncertainty + 11 removed
```

#### 5. Dataset Cleaning & Retraining ( In Progress)
```bash
# Merge cleaned annotations
python scripts/annotation/merge_cleaned_annotations.py

# Retrain with cleaned data
python scripts/training/retrain_with_cleaned_data.py --classifier mlp
```

**Cleaned Dataset Stats:**
```
Total samples: 189 (down from 200)
Class distribution:
  uncertainty:  76 (40.2%)  ↑ +18
  optimism:     54 (28.6%)  ↓ -29
  anxiety:      22 (11.6%)
  excitement:   19 (10.1%)
  fear:         12 (6.3%)
  hope:          6 (3.2%)

Class imbalance: 12.7x (improved from 13.8x)
```

### Current Challenges

1. **Small Dataset:** 189 samples insufficient for robust 6-class classification
2. **Class Imbalance:** 12.7x ratio between most/least frequent classes
3. **Rare Classes:** Hope (6), fear (12), excitement (19) have very few samples

### Next Steps

**Short-term (Before Final Deliverable):**
1.  Analyze retraining results on cleaned data
2.  Consider 3-class taxonomy (negative/positive/uncertainty) for better performance
3. ⏳ Scale up annotation to 500+ samples
4. ⏳ Implement Stage 1 LoRA training on GoEmotions

**Long-term (Post-Course):**
- Complete two-stage LoRA fine-tuning pipeline
- Expand dataset to 5000+ samples
- Benchmark against FinBERT and other financial NLP models

---

## Technical Details

### Emotion Taxonomy (6 Economic Emotions)

| Emotion | Definition | Example |
|---------|------------|---------|
| **Anxiety** | Worry about uncertain economic outcomes | "Inflation concerns weigh on consumer spending" |
| **Excitement** | Enthusiasm about positive developments | "Tech stocks surge on breakthrough AI earnings" |
| **Optimism** | Positive long-term economic outlook | "Analysts predict robust growth for next quarter" |
| **Fear** | Strong concern about negative outcomes | "Market panic triggers mass sell-off" |
| **Uncertainty** | Lack of clarity about economic direction | "Fed officials remain divided on policy path" |
| **Hope** | Desire for positive future outcomes | "Investors await potential recovery signals" |

### Architecture Comparison

#### Current: Logits-Based Classifier
```
Input Text → DistilBERT (frozen) → 768-dim features → MLP → 6 emotions
                                                       ↓
                                            Class-weighted loss
```
- **Pros:** Fast training, CPU-friendly, interpretable
- **Cons:** Limited capacity, can't learn text patterns

#### Planned: LoRA Fine-Tuning
```
Stage 1: GoEmotions (58k samples, 27 emotions)
         ↓
    Llama 3.1 8B + LoRA (r=16) → Emotion-aware base
         ↓
Stage 2: Financial text (500+ samples, 6 emotions)
         ↓
    Further LoRA tuning → FinEmo-LoRA
```
- **Pros:** End-to-end learning, higher capacity, SOTA potential
- **Cons:** GPU required, longer training (6-12 hours)

### Configuration (`config.yaml`)

Key settings that control the project:

```yaml
# Emotion labels
emotion_labels:
  - anxiety
  - excitement  
  - optimism
  - fear
  - uncertainty
  - hope

# Model selection
model:
  selected: "distilbert"  # Options: distilbert, llama, phi
  device: "cpu"           # cpu, cuda, or mps (Apple Silicon)

# Classifier training
classifier:
  selected: "mlp"         # Options: mlp, xgboost
  types:
    mlp:
      hidden_layers: [512, 256, 128]
      dropout: 0.3
      learning_rate: 0.001
      epochs: 50
      batch_size: 16

# Annotation
annotation:
  llm_provider: "openai"
  model: "gpt-4o"
  min_confidence: 0.7
```

---

## Key Files

### Python Scripts (`scripts/` directory)

1. **Data Collection**
   - `scripts/data_collection/download_fingpt.py` - Download FinGPT dataset
   - Implements automated data fetching and preprocessing

2. **Annotation System**
   - `scripts/annotation/llm_annotator.py` - GPT-4-based emotion annotation
   - `scripts/annotation/review_annotations.py` - Automated quality control
   - `scripts/annotation/merge_cleaned_annotations.py` - Dataset merging

3. **Feature Extraction**
   - `scripts/feature_extraction/extract_features.py` - DistilBERT feature extraction

4. **Model Training**
   - `scripts/classifier/train_classifier.py` - MLP/XGBoost classifier training
   - `scripts/training/retrain_with_cleaned_data.py` - Retraining pipeline

5. **Evaluation**
   - `scripts/evaluation/run_full_evaluation.py` - Complete evaluation pipeline
   - Generates confusion matrices, per-class metrics, error analysis

### Notebooks (`notebooks/` directory)

- **Data exploration notebooks** - Analysis of emotion distributions, text lengths
- **Error analysis notebooks** - Visualization of misclassifications
- *(Note: Notebooks in development, scripts are primary deliverables)*

### Main Orchestration

- `run_pipeline.py` - Main script coordinating all stages
- `config.yaml` - Central configuration file

### Example Execution

```bash
# Complete pipeline (download → annotate → train → evaluate)
python run_pipeline.py --stage all

# Run only evaluation on existing model
python run_pipeline.py --stage evaluate
```

---

## References & Related Work

### Research Papers

1. **GoEmotions: A Dataset of Fine-Grained Emotions**  
   Demszky et al., 2020 | [arXiv:2005.00547](https://arxiv.org/abs/2005.00547)
   - Inspiration for fine-grained emotion taxonomy

2. **LoRA: Low-Rank Adaptation of Large Language Models**  
   Hu et al., 2021 | [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
   - Foundation for parameter-efficient fine-tuning approach

3. **FNSPID: A Comprehensive Financial News Dataset**  
   [arXiv:2402.06698](https://arxiv.org/pdf/2402.06698)
   - Comprehensive financial NLP dataset overview

4. **Parameter-efficient fine tuning of LLaMa for sentiment analysis**  
   [Research Link](https://www.henryszhang.com/static/media/35_Parameter_Efficient_Fine_Tu.a9386bf67cdb60c44871.pdf)
   - LoRA application to financial sentiment

### GitHub Resources

- **FinGPT**: https://github.com/FinancialDiets/FINGPT
- **Hugging Face Transformers**: https://github.com/huggingface/transformers
- **PEFT (Parameter-Efficient Fine-Tuning)**: https://github.com/huggingface/peft

---

## Acknowledgments

- **Professor Joel Klein** - Course instruction and project guidance
- **Google Research** - GoEmotions dataset
- **OpenAI** - GPT-4 API for annotation
- **Hugging Face** - Transformers library and model hosting
- **FinGPT Team** - Financial NLP resources

---

## License

This project is for academic purposes as part of CSCI 6366 coursework at the George Washington University.

---

## Contact

**Vaishnavi Kamdi**  
GitHub: [@vaish725](https://github.com/vaish725)  
Course: CSCI 6366 Neural Networks and Deep Learning  
Instructor: Professor Joel Klein ([@jdk514](https://github.com/jdk514))  
George Washington University  
Fall 2025  
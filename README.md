# FinEmo-LoRA: Financial Emotion Detection Using Parameter-Efficient Fine-Tuning

**Neural Networks and Deep Learning - Final Project**  
**December 2025**


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

---

## 🎯 Project Overview

Traditional financial sentiment analysis oversimplifies emotions into Positive/Negative/Neutral. This project develops a nuanced 6-class emotion classifier using parameter-efficient fine-tuning:

**Emotions**: `anxiety`, `excitement`, `fear`, `hope`, `optimism`, `uncertainty`

**Innovation**: Two-stage LoRA training pipeline combining general emotion knowledge (GoEmotions) with financial domain expertise (FinGPT).

---

## 🚀 Quick Start

### Option 1: Interactive Dashboard (Recommended)

Launch the Streamlit dashboard for live predictions:

```bash
# Clone and setup
git clone https://github.com/vaish725/FinEmo-LoRA.git
cd FinEmo-LoRA
pip install -r requirements.txt

# Launch dashboard
cd app && streamlit run Home.py
```

**Features:**
- Real-time emotion prediction
- Confidence scores & probability distributions
- Complete project documentation
- Model architecture diagrams

### Option 2: View Final Presentation

Open **[`notebooks/FinEmo_LoRA_FINAL_PRESENTATION.ipynb`](notebooks/FinEmo_LoRA_FINAL_PRESENTATION.ipynb)** for:
- Complete project walkthrough
- Training methodology
- Results analysis with visualizations
- Live inference examples

### Option 3: Run Training (Google Colab - 60 min)

Reproduce the **76.8% accuracy** result:

1. Upload [`notebooks/FinEmo_LoRA_Training.ipynb`](notebooks/FinEmo_LoRA_Training.ipynb) to Google Colab
2. Change runtime to **GPU (T4 or better)**
3. Upload dataset: `data/annotated/fingpt_annotated_expanded_latest.csv`
4. Run all cells (Stage 1: 30 min, Stage 2: 30 min)
5. Download trained model from Colab

**Note:** Free T4 GPU on Google Colab is sufficient!

### Option 4: Run Inference Only

Use the pre-trained model for predictions:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

# Load model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
base_model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=6
)
model = PeftModel.from_pretrained(base_model, "./models/finemo_lora_v2_best")

# Predict
text = "Strong Q4 guidance suggests revenue growth ahead."
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
outputs = model(**inputs)
prediction = outputs.logits.argmax().item()

emotions = ['anxiety', 'excitement', 'fear', 'hope', 'optimism', 'uncertainty']
print(f"Emotion: {emotions[prediction]}")  # Output: hope (94% confidence)
```

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

### Final Implementation: LoRA Fine-Tuning

This project successfully implemented **two-stage LoRA fine-tuning** achieving **76.8% accuracy**:

#### Final Architecture (v2 - Implemented)
- **Base Model:** DistilBERT (67.7M parameters, frozen)
- **LoRA Adapters:** r=8, α=16 (742K parameters, 1.1% trainable)
- **Stage 1:** Transfer learning on GoEmotions 10K subset (general emotion understanding)
- **Stage 2:** Financial domain adaptation on 3,472 annotated FinGPT samples
- **Training:** 60 minutes on Google Colab T4 GPU
- **Results:** 76.8% accuracy, 0.74 F1-score (exceeded 75% target)

### Dataset Evolution & Scale-Up

**Initial Phase (v0 - Baseline):**
- 200 samples with GPT-4 annotations
- Logits-based classifier: 46.3% accuracy
- Challenge: Severe class imbalance (13.8:1 ratio)

**v1 Enhancement (Balanced Dataset):**
- 1,152 samples (928 original + 224 targeted minority samples)
- Class imbalance improved to 2.6:1
- Logits-based classifier: 52.7% accuracy
- Hope/Fear/Excitement recall significantly improved

**v2 Scale-Up (Final Dataset - 3,472 samples):**
- **Method:** GPT-4 synthetic data generation (6 emotions × ~500 samples each) + Original annotations
- **Synthetic Generation:**
  - Generated 2,631 synthetic samples across all emotions
  - Cost: ~$1.63 for complete generation
  - Per-emotion: anxiety (360), excitement (387), fear (493), hope (981), optimism (190), uncertainty (220)
  - Merged with 928 original annotated samples → 3,472 total after filtering
- **Final Composition:**
  - anxiety: 487 samples (14.1%)
  - excitement: 478 samples (13.8%)
  - fear: 541 samples (15.6%)
  - hope: 1,003 samples (28.9%)
  - optimism: 480 samples (13.8%)
  - uncertainty: 478 samples (13.8%)
- **Balance:** Near-perfect distribution (2.1:1 ratio max, down from 13.8:1)
- **Quality Control:** 
  - Multi-stage validation with confidence scoring
  - Text length filtering (50-500 characters)
  - Duplicate removal
  - GPT-4 confidence threshold (>0.7)
- **Result:** **76.8% accuracy** with LoRA fine-tuning (30.5pp improvement from baseline)

**Key Improvements (v2 vs Baseline):**
- Hope recall: 0% → **95%** (+95pp)
- Fear recall: 0% → **50%** (+50pp)
- Excitement recall: 5% → **79%** (+74pp)
- Overall accuracy: 46.3% → **76.8%** (+30.5pp)

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
       fingpt/                    # FinGPT sentiment data
       synthetic/                 # GPT-4 generated synthetic samples
    annotated/                     # LLM-annotated financial text
       fingpt_annotated_expanded_latest.csv  # Final v2 dataset (3,472 samples)
       fingpt_annotated_enhanced.csv         # Balanced v1 dataset (1,152 samples)
       fingpt_annotated.csv                  # Original annotations (928 samples)
    features/                      # Extracted DistilBERT features (for logits approach)
        train_features.npy         # Cached features

 models/
    finemo_lora_v2_best/           # Final LoRA model (76.8% accuracy)
       adapter_model.safetensors  # LoRA adapters (3.8MB)
       adapter_config.json        # LoRA configuration
       tokenizer files...         # DistilBERT tokenizer
    finemo_lora_final_v2_full_dataset/  # Full training artifacts
       confusion_matrix_lora_v2.png     # Performance visualization
    classifiers/                   # Logits-based classifiers (baseline)
       mlp_*.pkl                  # MLP PyTorch models
       xgboost_*.pkl              # XGBoost models

 scripts/
    data_collection/               # Dataset download & preprocessing
       download_fingpt.py         # Download FinGPT data
       download_goemotions.py     # Download GoEmotions
       generate_synthetic_samples.py  # GPT-4 synthetic data generation
       merge_datasets.py          # Merge original + synthetic data
       preprocess_goemotions.py   # GoEmotions preprocessing
    annotation/                    # Annotation pipeline
       llm_annotator.py           # GPT-4 annotation with confidence
       validate_annotations.py    # Annotation validation
       scale_dataset.py           # Dataset scaling utilities
    feature_extraction/            # Feature extraction (for logits approach)
       extract_features.py        # DistilBERT feature extraction
    classifier/                    # Logits-based classifier training
       train_classifier.py        # MLP/XGBoost training (baseline)
    training/                      # LoRA training pipelines
       train_stage1_goemotions.py # Stage 1: General emotion learning
       train_stage2_financial.py  # Stage 2: Financial adaptation
    evaluation/                    # Evaluation framework
        evaluate.py                # LoRA model evaluation

 app/                               # Streamlit Dashboard
    Home.py                        # Main dashboard page
    pages/                         # Dashboard pages
       1_Prediction.py             # Single text prediction
       4_Documentation.py          # Complete documentation
    utils/                         # Dashboard utilities
       model_utils.py              # Model loading functions

 notebooks/                         # Jupyter Notebooks
    FinEmo_LoRA_Training.ipynb     # v2 training notebook (76.8%)
    FinEmo_LoRA_FINAL_PRESENTATION.ipynb  # Final presentation

 results/                           # Evaluation results
 logs/                              # Training logs
 config.yaml                        # Central configuration
 requirements.txt                   # Complete dependencies
 run_pipeline.py                    # Pipeline orchestration (reference)
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

## Implementation Timeline & Results

### Phase 1: Baseline Development (Complete)
**Goal:** Establish baseline performance with simple approach

**Approach:**
- Logits-based classifier (frozen DistilBERT + MLP)
- Initial 928 GPT-4 annotated samples
- CPU-friendly, fast training

**Results:**
- Accuracy: **46.3%**
- Major issue: Severe class imbalance (13.8:1 ratio)
- Hope/Fear/Excitement recall: 0-5%

### Phase 2: Data Enhancement (Complete)
**Goal:** Address class imbalance with targeted sampling

**Approach:**
```bash
# Generate minority class samples
python scripts/data_collection/generate_synthetic_samples.py --emotion hope --count 500

# Merge with original data
python scripts/data_collection/merge_datasets.py
```

**Dataset v1 (Enhanced):**
- Total: 1,152 samples (928 + 224 targeted minority)
- Improved balance: 2.6:1 ratio
- Logits accuracy: **52.7%** (+6.4pp)

### Phase 3: Large-Scale Dataset Creation (Complete)
**Goal:** Scale to 3,000+ samples for robust training

**Synthetic Data Generation Process:**
```bash
# Generated ~500 samples per emotion using GPT-4
python scripts/data_collection/generate_synthetic_samples.py --emotion hope --count 500
python scripts/data_collection/generate_synthetic_samples.py --emotion fear --count 500
python scripts/data_collection/generate_synthetic_samples.py --emotion anxiety --count 500
python scripts/data_collection/generate_synthetic_samples.py --emotion excitement --count 500
python scripts/data_collection/generate_synthetic_samples.py --emotion optimism --count 500
python scripts/data_collection/generate_synthetic_samples.py --emotion uncertainty --count 500

# Result: 2,631 synthetic samples generated
# - anxiety: 360 samples
# - excitement: 387 samples  
# - fear: 493 samples
# - hope: 981 samples (2 batches)
# - optimism: 190 samples
# - uncertainty: 220 samples

# Merge with original 928 annotated samples
python scripts/data_collection/merge_datasets.py
```

**Quality Control Pipeline:**
- Text length filtering (50-500 characters)
- Duplicate removal (exact text matching)
- GPT-4 confidence threshold (>0.7)
- Manual spot-checking of synthetic samples
- Format validation

**Dataset v2 (Final - 3,472 samples after filtering):**
```
anxiety:      487 (14.1%)
excitement:   478 (13.8%)
fear:         541 (15.6%)
hope:       1,003 (28.9%)
optimism:     480 (13.8%)
uncertainty:  478 (13.8%)

Balance: 2.1:1 ratio (near-perfect, down from 13.8:1)
Cost: $1.63 total for all synthetic generation
```

### Phase 4: LoRA Fine-Tuning (Complete)
**Goal:** Achieve >75% accuracy with parameter-efficient training

**Training Pipeline:**
```python
# Stage 1: General emotion understanding (60 min)
# - GoEmotions 10K subset
# - DistilBERT + LoRA (r=8, α=16)
# - Learn 27 → 6 emotion mapping

# Stage 2: Financial adaptation (60 min)
# - 3,472 financial samples
# - Continue from Stage 1 checkpoint
# - Specialize for economic emotions
```

**Final Results (v2):**
```
Overall Accuracy: 76.8% (+30.5pp from baseline)
F1-Score: 0.74

Per-Class Performance:
  hope:         95% recall (+95pp)  🚀
  excitement:   79% recall (+74pp)  🎯
  optimism:     81% recall (+81pp)  ✅
  fear:         50% recall (+50pp)  ✅
  anxiety:      77% recall (+20pp)
  uncertainty:  68% recall (+26pp)

Model Size: 3.8MB (LoRA adapters only)
Trainable: 742K params (1.1% of base model)
```

### Phase 5: Dashboard Deployment (Complete)
**Goal:** Interactive demo for presentation

**Features:**
```bash
# Launch dashboard
cd app && streamlit run Home.py
```

- Real-time single text prediction
- Confidence scores & probability distributions
- Complete documentation
- Professional black/white architecture diagrams

### Key Achievements

✅ **Exceeded Target:** 76.8% accuracy (target was 75%)  
✅ **Solved Class Imbalance:** All emotions >50% recall  
✅ **Parameter Efficient:** Only 1.1% parameters trained  
✅ **Fast Training:** 60 minutes on free Google Colab T4  
✅ **Low Cost:** $1.63 total for synthetic data generation  
✅ **Production Ready:** Streamlit dashboard deployed
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

## Key Files & Usage

### Primary Training Notebook

**`notebooks/FinEmo_LoRA_Training.ipynb`** - Complete v2 training pipeline
- Two-stage LoRA fine-tuning (Stage 1: GoEmotions, Stage 2: Financial)
- Achieves **76.8% accuracy** on Google Colab T4
- Includes data loading, training, evaluation, and model saving
- **This is the main deliverable** - reproduces final results

### Dashboard Application

**`app/`** - Streamlit interactive demo
- `Home.py` - Main dashboard with metrics
- `pages/1_Prediction.py` - Single text prediction
- `pages/4_Documentation.py` - Complete documentation
- Launch: `cd app && streamlit run Home.py`

### Python Scripts (`scripts/` directory)

1. **Data Collection**
   - `download_fingpt.py` - FinGPT dataset download
   - `download_goemotions.py` - GoEmotions dataset download
   - `generate_synthetic_samples.py` - GPT-4 synthetic data generation ($1.63)
   - `merge_datasets.py` - Merge original + synthetic datasets

2. **Annotation System**
   - `llm_annotator.py` - GPT-4 emotion annotation with confidence scoring
   - `validate_annotations.py` - Annotation validation
   - `scale_dataset.py` - Dataset scaling utilities

3. **Feature Extraction (for logits baseline)**
   - `extract_features.py` - DistilBERT feature extraction

4. **Model Training**
   - `train_classifier.py` - Baseline logits classifier (MLP/XGBoost)
   - `train_stage1_goemotions.py` - Stage 1 LoRA training (reference)
   - `train_stage2_financial.py` - Stage 2 LoRA training (reference)

5. **Evaluation**
   - `evaluate.py` - LoRA model evaluation

### Example Usage

```bash
# Generate synthetic data (costs ~$1.63 for full dataset)
python scripts/data_collection/generate_synthetic_samples.py \
  --emotion hope --count 500 --model gpt-4

# Merge all synthetic + original datasets
python scripts/data_collection/merge_datasets.py

# Launch interactive dashboard
cd app && streamlit run Home.py

# Train in Google Colab (recommended)
# Upload notebooks/FinEmo_LoRA_Training.ipynb to Colab
# Upload data/annotated/fingpt_annotated_expanded_latest.csv
# Change runtime to GPU (T4) and run all cells
```

**Note:** The final v2 model (76.8%) was trained in `notebooks/FinEmo_LoRA_Training.ipynb` on Google Colab, not via command-line scripts.

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
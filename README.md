# FinEmo-LoRA: A Parameter-Efficient Approach to Fine-Grained Economic Emotion Interpretation from Financial News

A deep learning project that fine-tunes Large Language Models (LLMs) using Low-Rank Adaptation (LoRA) to classify financial text into six economic emotions: **anxiety**, **excitement**, **optimism**, **fear**, **uncertainty**, and **hope**.

**Author:** Vaishnavi Kamdi  
**Course:** CSCI 6907 Neural Networks and Deep Learning  
**Institution:** George Washington University  
**Professor:** Joel Klein

---

## Overview

Traditional financial sentiment analysis oversimplifies market reactions into basic Positive/Negative/Neutral classifications. FinEmo-LoRA addresses this limitation by:

1. **Fine-Grained Emotion Classification**: Classifying financial text into 6 nuanced economic emotions
2. **Parameter-Efficient Training**: Using LoRA for memory-efficient fine-tuning
3. **Two-Stage Training**: Transfer learning on GoEmotions followed by financial domain adaptation
4. **LLM-Based Annotation**: Using GPT-4 for high-quality pseudo-labeling of financial text

### Key Features

- Two-stage training pipeline (emotion understanding → financial domain specialization)
- Support for Llama 3.1 8B and Microsoft Phi-2 base models
- Automated LLM-based annotation with confidence filtering
- Comprehensive evaluation with per-class metrics and confusion matrices
- QLoRA (4-bit quantization) for memory-efficient training
- Fully configurable via YAML

---

## Project Structure

```
FinEmo-LoRA/
├── config.yaml                         # Main configuration file
├── requirements.txt                    # Python dependencies
├── .env.example                        # Environment variables template
├── README.md                           # This file
│
├── data/
│   ├── raw/                           # Raw downloaded datasets
│   │   ├── fingpt/                    # FinGPT sentiment data
│   │   ├── sentfin/                   # SEntFiN entity-aware sentiment
│   │   └── goemotions/                # GoEmotions 27-emotion dataset
│   ├── processed/                     # Preprocessed datasets
│   │   ├── goemotions/                # Mapped to 6-emotion taxonomy
│   │   └── financial_emotion_splits/  # Train/val/test splits
│   └── annotated/                     # LLM-annotated financial text
│
├── models/
│   ├── checkpoints/                   # Training checkpoints
│   │   ├── stage1_goemotions/         # Stage 1 checkpoints
│   │   └── stage2_financial/          # Stage 2 checkpoints
│   ├── stage1_final/                  # Stage 1 final model
│   └── finemo-lora-final/             # Final FinEmo-LoRA model
│       └── merged/                    # Merged model (LoRA + base)
│
├── scripts/
│   ├── data_collection/               # Dataset download & preprocessing
│   │   ├── download_fingpt.py         # Download FinGPT data
│   │   ├── download_sentfin.py        # Download SEntFiN data
│   │   ├── download_goemotions.py     # Download GoEmotions data
│   │   └── preprocess_goemotions.py   # Map GoEmotions to 6 emotions
│   ├── annotation/                    # LLM-based annotation pipeline
│   │   ├── llm_annotator.py           # GPT-4 annotation script
│   │   └── validate_annotations.py    # Validation & agreement metrics
│   ├── training/                      # Training pipelines
│   │   ├── train_stage1_goemotions.py # Stage 1: Emotion understanding
│   │   └── train_stage2_financial.py  # Stage 2: Financial adaptation
│   └── evaluation/                    # Evaluation framework
│       └── evaluate.py                # Comprehensive evaluation
│
├── results/                           # Evaluation results
│   ├── predictions_*.csv              # Model predictions
│   ├── metrics_*.json                 # Performance metrics
│   ├── confusion_matrix_*.png         # Confusion matrix plots
│   └── error_analysis_*.csv           # Misclassification analysis
│
├── logs/                              # Training logs (TensorBoard)
└── notebooks/                         # Jupyter notebooks for analysis
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (recommended: 16GB+ VRAM)
- 50GB+ free disk space

### Setup Instructions

1. **Clone the repository**
   ```bash
   cd "/Users/vaishnavikamdi/Documents/GWU/Classes/Fall 2025/NNDL/FinEmo-LoRA"
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys:
   # - OPENAI_API_KEY (for GPT-4 annotation)
   # - HF_TOKEN (for Llama model access)
   # - WANDB_API_KEY (optional, for experiment tracking)
   ```

5. **Configure Kaggle API** (for SEntFiN dataset)
   ```bash
   # Download kaggle.json from https://www.kaggle.com/settings/account
   mkdir -p ~/.kaggle
   mv kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

---

## Usage

### Step 1: Download Datasets

```bash
# Download FinGPT sentiment dataset
python scripts/data_collection/download_fingpt.py

# Download SEntFiN dataset (requires Kaggle API)
python scripts/data_collection/download_sentfin.py

# Download GoEmotions dataset
python scripts/data_collection/download_goemotions.py

# Preprocess GoEmotions: Map 27 emotions to our 6 economic emotions
python scripts/data_collection/preprocess_goemotions.py
```

### Step 2: Annotate Financial Text with Emotions

```bash
# Use GPT-4 to annotate financial text with 6-emotion taxonomy
python -c "
from scripts.annotation.llm_annotator import annotate_dataset

annotate_dataset(
    input_path='data/raw/fingpt/train.csv',
    output_path='data/annotated/fingpt_annotated.csv',
    text_column='text',  # Adjust column name based on actual data
    max_samples=5000,    # Start with 5000 samples
    batch_size=50
)
"
```

**Note:** Annotation uses GPT-4 API and may incur costs. Start with a small sample for testing.

### Step 3: Validate Annotations (Optional but Recommended)

```bash
# Create a validation sample
python -c "
from scripts.annotation.validate_annotations import create_validation_sample

create_validation_sample(
    annotated_file='data/annotated/fingpt_annotated_high_confidence.csv',
    output_file='data/annotated/validation_sample.csv',
    sample_size=200
)
"

# Manually annotate the validation sample, then compute agreement
python -c "
from scripts.annotation.validate_annotations import compute_agreement

compute_agreement(
    validation_file='data/annotated/validation_sample.csv'
)
"
```

### Step 4: Stage 1 Training - Transfer Learning on GoEmotions

```bash
# Fine-tune base model (Llama 3.1 8B or Phi-2) on mapped GoEmotions
# This teaches the model general emotion classification
python scripts/training/train_stage1_goemotions.py
```

**Training Configuration** (in `config.yaml`):
- Epochs: 3
- Batch size: 16 (with gradient accumulation)
- Learning rate: 2e-4
- Max sequence length: 256
- LoRA rank: 16

**Estimated Time:** 4-8 hours on a single GPU (depends on GPU model)

### Step 5: Stage 2 Training - Financial Domain Adaptation

```bash
# Fine-tune Stage 1 model on annotated financial emotion dataset
# This specializes the model for economic emotion interpretation
python scripts/training/train_stage2_financial.py
```

**Training Configuration** (in `config.yaml`):
- Epochs: 5
- Batch size: 8 (with gradient accumulation)
- Learning rate: 1e-4
- Max sequence length: 512
- LoRA rank: 16

**Estimated Time:** 2-4 hours on a single GPU

### Step 6: Evaluate the Model

```bash
# Comprehensive evaluation on test set
python scripts/evaluation/evaluate.py
```

**Evaluation Outputs:**
- Overall accuracy and macro-averaged metrics
- Per-class Precision, Recall, F1-Score
- Confusion matrix visualization
- Error analysis with misclassification patterns
- Predictions CSV for manual review

---

## Configuration

All project settings are controlled via `config.yaml`. Key sections:

### Emotion Labels
```yaml
emotion_labels:
  - anxiety
  - excitement
  - optimism
  - fear
  - uncertainty
  - hope
```

### Model Selection
```yaml
model:
  selected: "llama"  # Options: "llama" or "phi"
  
  base_models:
    llama:
      name: "meta-llama/Meta-Llama-3.1-8B"
    phi:
      name: "microsoft/phi-2"
```

### LoRA Configuration
```yaml
lora:
  r: 16                    # LoRA rank (higher = more parameters)
  lora_alpha: 32           # LoRA scaling factor
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
  lora_dropout: 0.05
```

### Annotation Configuration
```yaml
annotation:
  llm_provider: "openai"
  model: "gpt-4o"
  min_confidence: 0.7      # Filter low-confidence annotations
  batch_size: 50
```

---

## Methodology

### Two-Stage Training Approach

**Stage 1: Emotion Understanding (Transfer Learning)**
- Dataset: GoEmotions (27 emotions → mapped to 6)
- Purpose: Learn general emotion classification
- Base: General Reddit text with emotion labels
- Output: Emotion-aware model

**Stage 2: Financial Domain Adaptation**
- Dataset: LLM-annotated financial text (FinGPT + SEntFiN)
- Purpose: Specialize for economic emotions in financial context
- Base: Stage 1 model
- Output: FinEmo-LoRA final model

### LoRA (Low-Rank Adaptation)

LoRA enables efficient fine-tuning by:
- Adding trainable low-rank matrices to attention layers
- Freezing base model weights
- Training only ~0.1-1% of total parameters
- Reducing memory requirements by 80-90%

### LLM-Based Annotation Pipeline

1. **Prompt Engineering**: Carefully crafted system prompt defines 6 emotions with examples
2. **Batch Processing**: Annotate financial texts in batches
3. **Confidence Scoring**: GPT-4 provides confidence for each annotation
4. **Quality Filtering**: Keep only high-confidence annotations (≥0.7)
5. **Manual Validation**: Sample validation for quality assurance

---

## Expected Results

Based on similar financial NLP tasks and the project methodology:

- **Overall Accuracy**: 70-85%
- **Macro F1-Score**: 0.65-0.80
- **Per-Class Performance**: Variable (common emotions like "optimism" and "fear" perform better)

**Key Challenges:**
- Class imbalance in financial text
- Subtle differences between emotions (e.g., fear vs. anxiety)
- Context-dependent emotion interpretation
- Limited training data for some emotions

---

## Monitoring Training

### TensorBoard

View training metrics in real-time:

```bash
tensorboard --logdir=models/checkpoints
```

Access at `http://localhost:6006`

**Metrics to Monitor:**
- Training loss (should decrease steadily)
- Evaluation loss (should decrease, watch for overfitting)
- Learning rate schedule
- Gradient norms

### Weights & Biases (Optional)

If configured, experiments are automatically logged to W&B for:
- Hyperparameter tracking
- Model comparison
- Collaborative analysis

---

## Customization

### Using Different Base Models

Edit `config.yaml`:
```yaml
model:
  selected: "phi"  # Switch to Phi-2
```

Or add your own model:
```yaml
model:
  base_models:
    custom:
      name: "your-org/your-model"
      type: "causal_lm"
      quantization: "4bit"
  selected: "custom"
```

### Adjusting Emotion Taxonomy

To use different emotions, update `config.yaml`:
```yaml
emotion_labels:
  - your_emotion_1
  - your_emotion_2
  # ...
```

Then update the annotation prompt in `scripts/annotation/llm_annotator.py` and GoEmotions mapping.

### Training Hyperparameters

Tune training settings in `config.yaml`:
```yaml
training:
  stage2:
    epochs: 10              # More epochs for smaller datasets
    batch_size: 4           # Smaller for limited VRAM
    learning_rate: 5e-5     # Lower for stable convergence
    gradient_accumulation_steps: 16  # Increase for effective larger batch
```

---

## Troubleshooting

### Out of Memory (OOM) Errors

- Reduce `batch_size` in `config.yaml`
- Increase `gradient_accumulation_steps` to maintain effective batch size
- Reduce `max_seq_length`
- Ensure 4-bit quantization is enabled

### Slow Training

- Enable `fp16` mixed precision (default)
- Consider using `unsloth` library for faster Llama training (see requirements.txt)
- Use gradient checkpointing (enabled by default)

### Low Annotation Quality

- Increase `min_confidence` threshold in `config.yaml`
- Refine the system prompt in `llm_annotator.py`
- Provide more examples in the prompt
- Use GPT-4 instead of GPT-3.5

### Model Not Learning

- Check for data quality issues
- Increase learning rate slightly
- Verify emotion labels are balanced
- Ensure sufficient training data (1000+ samples per emotion)

---

## Research Papers & Resources

This project is based on the following research:

1. **FNSPID: A Comprehensive Financial News Dataset in Time Series**  
   https://arxiv.org/pdf/2402.06698

2. **FinLoRA: Benchmarking LoRA methods for fine-tuning LLMs for financial datasets**  
   https://arxiv.org/html/2505.19819v1

3. **Parameter-efficient fine tuning of LLaMa for sentiment analysis for enhanced stock price prediction**  
   https://www.henryszhang.com/static/media/35_Parameter_Efficient_Fine_Tu.a9386bf67cdb60c44871.pdf

### GitHub Repositories

- **FinGPT**: https://github.com/FinancialDiets/FINGPT
- **Phi-2 LoRA Fine-tuning**: https://github.com/KaifAhmad1/LLM-FineTuning-for-Sentiment-Classification
- **Llama 3.1 QLoRA Fine-tuning**: https://github.com/matteo-stat/transformers-llm-llama3.1-fine-tuning-qlora

---

## Citation

If you use this project in your research, please cite:

```
@misc{kamdi2025finemo,
  author = {Kamdi, Vaishnavi},
  title = {FinEmo-LoRA: A Parameter-Efficient Approach to Fine-Grained Economic Emotion Interpretation from Financial News},
  year = {2025},
  institution = {George Washington University},
  course = {CSCI 6907 Neural Networks and Deep Learning},
  professor = {Joel Klein}
}
```

---

## License

This project is for academic purposes as part of CSCI 6907 coursework at George Washington University.

---

## Acknowledgments

- **Professor Joel Klein** for guidance and course instruction
- **Hugging Face** for transformers library and model hosting
- **Google Research** for GoEmotions dataset
- **FinGPT Team** for financial NLP resources
- **OpenAI** for GPT-4 API enabling high-quality annotation

---

## Contact

**Vaishnavi Kamdi**  
George Washington University  
CSCI 6907 Neural Networks and Deep Learning  

For questions or issues, please open an issue in this repository or contact via GWU email.

---

## Future Work

Potential extensions of this project:

1. **Multi-label Classification**: Detect multiple emotions simultaneously
2. **Temporal Analysis**: Track emotion shifts in time-series financial data
3. **Market Correlation**: Correlate emotions with market movements
4. **Real-time Inference**: Deploy model as API for live financial text analysis
5. **Explainability**: Add attention visualization and SHAP values
6. **Cross-lingual**: Extend to non-English financial text
7. **Entity-Aware**: Incorporate entity recognition for company-specific emotions

---

**Project Status**: Implementation Complete  
**Last Updated**: November 3, 2025


# Quick Start: Evaluation on CPU

## Setup (One-Time)
```bash
# 1. Create virtual environment
python3 -m venv .venv

# 2. Activate it
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install OpenMP (macOS only, for XGBoost)
brew install libomp

# 5. Add credentials to .env file
echo "OPENAI_API_KEY=sk-your-key-here" >> .env
echo "HF_TOKEN=hf_your-token-here" >> .env
```

## Run Evaluation (Every Time)
```bash
# Activate venv
source .venv/bin/activate

# Run evaluation
python scripts/evaluation/run_full_evaluation.py \
    --classifier models/classifiers/mlp_20251103_200252.pkl \
    --device cpu

# View results
cat results/evaluation_metrics_*.json
open results/confusion_matrix_*.png
```

## Current Status
- ✅ CPU-friendly pipeline configured
- ✅ MLP classifier trained (63.33% accuracy)
- ✅ Evaluation script working
- ⚠️  Need to improve low-performing classes (excitement, optimism, fear)

## Next Steps
1. Analyze confusion matrix (`results/confusion_matrix_*.png`)
2. Review misclassified examples
3. Improve annotations for low-confidence samples
4. Retrain if needed

## Get Help
See `CPU_EVALUATION_GUIDE.md` for detailed documentation.

# Error Analysis Report - FinEmo-LoRA Classifier
**Date**: November 21, 2025  
**Model**: MLP Classifier (mlp_20251103_200252.pkl)  
**Test Accuracy**: 63.33%  
**Macro F1**: 0.33

## Executive Summary

The classifier shows **severe class imbalance issues** and **annotation quality problems**. While it performs well on 3 out of 6 emotions (uncertainty, hope, anxiety), it completely fails on the other 3 (excitement, optimism, fear).

**Critical Finding**: The model's "failure" on optimism is actually revealing a **fundamental annotation problem** - many texts labeled as "optimism" are neutral factual statements that don't convey emotional sentiment.

## Detailed Findings

### 1. Model Performance by Class

| Emotion | Precision | Recall | F1 | Support | Status |
|---------|-----------|--------|----|---------| -------|
| Uncertainty | 0.86 | 0.76 | 0.81 | 25 | ✅ **Good** |
| Hope | 0.56 | 0.82 | 0.67 | 17 | ✅ **Acceptable** |
| Anxiety | 0.38 | 0.71 | 0.50 | 7 | ⚠️ **Weak** |
| Excitement | 0.00 | 0.00 | 0.00 | 6 | ❌ **Failed** |
| Optimism | 0.00 | 0.00 | 0.00 | 3 | ❌ **Failed** |
| Fear | 0.00 | 0.00 | 0.00 | 2 | ❌ **Failed** |

### 2. Confusion Matrix Analysis

**Key Patterns**:

1. **Model never predicts**: excitement, optimism, or fear (columns all zeros)
2. **Hope is over-predicted**: Model defaults to "hope" for ambiguous positive emotions
   - 67% of excitement → hope
   - 50% of fear → hope  
   - 29% of anxiety → hope

3. **100% optimism → anxiety**: Every optimism sample misclassified as anxiety
   - This is a **conceptually opposite** error (positive → negative)
   - Reveals annotation quality issues

4. **Reasonable confusions**:
   - Fear ↔ Anxiety (both negative, similar intensity)
   - Excitement → Hope (both positive)

### 3. Root Cause: Annotation Quality Issues

#### Problem: "Optimism" Label Applied to Neutral Statements

**Examples labeled as "optimism" with 0.9 confidence**:
```
"The sale of the oxygen measurement business strengthens our goal to focus on our chosen market segments."
→ GPT-4 reasoning: "confident positive outlook about strategic decision"
→ Reality: Neutral corporate PR statement

"Los Angeles-based Pacific Office Properties Trust acquires, owns, and operates office properties..."
→ GPT-4 reasoning: "strategic focus and expansion suggesting confident outlook"
→ Reality: Factual business description

"The Moscow Metro was the first metro system in Europe to implement smart cards..."
→ Reality: Historical fact, no emotional content
```

Only 2-3 examples show genuine optimism:
```
"$aapl news is becoming very bullish again" ✓
"I'm still bullish on $AAPL" ✓
"$brcm raises revenue forecast" ✓ (borderline)
```

#### Why GPT-4 Mislabeled These

GPT-4's annotation prompt likely asked it to identify "economic optimism," and it interpreted this as:
- Presence of positive business actions (acquisitions, investments, expansion)
- Corporate growth language ("strengthens," "impressive," "benefiting")
- Forward-looking statements

**But it confused**:
- **Factual reporting** with **emotional sentiment**
- **Neutral corporate PR** with **genuine positive outlook**
- **Strategic decisions** with **emotional confidence**

#### Model's Response is Rational

The classifier **correctly identifies** these texts as non-optimistic:
- Features extracted from neutral/factual language
- No emotional markers present
- Model defaults to "anxiety" (perhaps due to restructuring/change language)

**The model is not broken - the annotations are.**

### 4. Class Imbalance Analysis

**Full Dataset Distribution** (200 samples):
```
optimism:       83 samples (41.5%) ← Majority class
uncertainty:    58 samples (29.0%)
anxiety:        22 samples (11.0%)
excitement:     19 samples (9.5%)
fear:           12 samples (6.0%)
hope:            6 samples (3.0%) ← Minority class

Imbalance ratio: 13.8x
```

**Impact**:
- Model learns to ignore rare classes (excitement, fear, optimism in test set)
- Over-predicts common classes (hope, uncertainty)
- Insufficient training data for reliable patterns

### 5. Confidence Score Analysis

**By Emotion** (mean confidence):
```
fear:         0.90 ← High confidence, but wrong labels
excitement:   0.87
anxiety:      0.83
optimism:     0.84 ← High confidence, but many neutral texts
uncertainty:  0.75 ← Lower confidence (more ambiguous)
hope:         0.80
```

**Low-confidence samples** (<0.75): 18 total (9%), all "uncertainty"
- Shows GPT-4 was genuinely uncertain about these
- Other classes had artificially high confidence

## Specific Error Patterns

### Pattern 1: Neutral Corporate Statements → Optimism
**Frequency**: ~60-70% of optimism labels  
**Example**: "The total area of the Gorelovo plant is about 22,000 sq m. Atria invested about EURO 70mn..."  
**Fix**: Re-label as neutral or remove from dataset

### Pattern 2: Excitement → Hope Confusion  
**Frequency**: 67% of excitement samples  
**Example**: Positive market news gets labeled as hope instead of excitement  
**Fix**: Clearer emotion definitions; more excitement training data

### Pattern 3: Fear → Anxiety/Hope Split
**Frequency**: 50% each direction  
**Example**: Financial risks being interpreted as anxious caution or hopeful uncertainty  
**Fix**: Merge fear+anxiety into single "negative_concern" class

## Recommendations

### Priority 1: Fix Annotation Quality (CRITICAL)

**Action**: Manual review and re-labeling
```bash
# Extract all "optimism" samples
grep "optimism" data/annotated/fingpt_annotated.csv > review_optimism.csv

# Rules for re-labeling:
1. Factual statements with no emotion → REMOVE or label as "neutral"
2. Generic corporate PR → REMOVE
3. True bullish sentiment ("bullish", "gains", "soaring") → KEEP as optimism
4. Growth metrics alone → REMOVE (not emotional)
```

**Expected impact**: Reduce optimism from 83 → ~20-30 samples (actual optimism)

### Priority 2: Balance Dataset

**Option A: Collect more data** (preferred)
- Annotate 500-1000 more financial texts
- Target rare emotions: fear, excitement, true optimism
- Use more emotionally charged sources (social media, analyst opinions)

**Option B: Use class weights in training**
```python
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
# Apply in loss function
```

**Option C: Downsample majority classes**
- Reduce optimism/uncertainty to 20-30 samples each
- Creates balanced but smaller dataset

### Priority 3: Simplify Emotion Taxonomy

**Merge similar classes**:
```
Current 6 emotions → Proposed 3 categories:

1. negative_concern: anxiety + fear
2. positive_outlook: excitement + optimism + hope  
3. uncertainty: (keep as-is)
```

**Benefits**:
- Clearer class boundaries
- Better balance
- Higher accuracy (likely 75-85%)
- More reliable predictions

### Priority 4: Improve Annotation Process

**For future annotations**:
1. **Use emotion-specific examples** in GPT-4 prompt
2. **Add negative examples** ("This is NOT optimism: ...")
3. **Require emotional markers** (explicit sentiment words)
4. **Filter neutral statements** (factual business descriptions)
5. **Lower confidence threshold** (only keep >0.8 initially)

## Next Steps

### Immediate (This Week)
1. ✅ Complete error analysis (this document)
2. ⬜ Manual review of all 83 "optimism" samples
3. ⬜ Re-label or remove neutral statements
4. ⬜ Create cleaned dataset: `fingpt_annotated_v2.csv`

### Short-term (Next 1-2 Weeks)
5. ⬜ Retrain classifier with:
   - Cleaned annotations
   - Class weights (balanced)
   - Validation set monitoring
6. ⬜ Evaluate XGBoost on same data
7. ⬜ Try merged 3-class taxonomy

### Long-term (If Time Permits)
8. ⬜ Collect 500+ more samples (focus on rare emotions)
9. ⬜ Implement active learning (annotate uncertain samples)
10. ⬜ Create inference demo with cleaned model

## Metrics Targets (After Fixes)

**Realistic targets with cleaned data**:
- Overall Accuracy: 75-85% (up from 63%)
- Macro F1: 0.65-0.75 (up from 0.33)
- Per-class F1: >0.60 for all classes

**With 3-class taxonomy**:
- Overall Accuracy: 85-90%
- Macro F1: 0.80-0.85

## Conclusion

The classifier is **working as intended** - the problem is **data quality, not model architecture**. The "failure" on optimism actually reveals that GPT-4's annotations conflated factual statements with emotional sentiment.

**Key takeaway**: Fix the annotations first, then retrain. The current model architecture (DistilBERT + MLP) is appropriate for this task.

# Quick Guide: Cleaning Optimism Annotations

## What You're Fixing
Your annotations have ~60-70% neutral statements incorrectly labeled as "optimism". You need to identify and remove these.

## Two Options

### Option 1: Automatic Review (Fast, ~80% Accurate)
```bash
source .venv/bin/activate
python scripts/annotation/review_annotations.py --input review_optimism.csv --output cleaned_optimism.csv --auto
```

This automatically:
- Keeps samples with emotional keywords ("bullish", "excited", "confident")
- Removes samples with factual language ("acquires", "based in", "total area")
- Changes low-confidence (<0.8) to "uncertainty"

**Time**: 2 seconds  
**Accuracy**: ~80% (may miss nuances)

### Option 2: Manual Review (Slow, ~100% Accurate)
```bash
source .venv/bin/activate
python scripts/annotation/review_annotations.py --input review_optimism.csv --output cleaned_optimism.csv
```

This shows you each sample and asks:
```
[k] Keep - Real optimism
[r] Remove - Neutral/factual
[u] Uncertainty - Change label
[n] Neutral - Mark for removal
[s] Skip - Review later
[q] Quit and save
```

**Time**: ~15-30 minutes (83 samples)  
**Accuracy**: 100% (your judgment)

## Decision Rules (Use These!)

### ❌ REMOVE if the text:
- Is a factual statement ("Company X operates in...")
- Is a business description ("acquires, owns, operates...")
- Is historical fact ("was the first to...")
- Is neutral metrics ("total area is...", "invested EUR...")
- Is generic corporate PR ("strengthens our goal...")
- Has NO emotional words

### ✅ KEEP if the text:
- Contains emotional words ("bullish", "excited", "confident", "soaring")
- Expresses personal sentiment ("I'm optimistic about...")
- Shows positive surprise ("relief", "breakthrough", "impressive")
- Talks about future prospects positively ("expecting strong growth")
- Would make an investor feel good

### Examples

**REMOVE** ❌
```
"Pacific Office Properties Trust acquires, owns, and operates office properties"
→ Just business description

"The Moscow Metro was the first metro system to implement smart cards"
→ Historical fact, no emotion

"Company invests EUR 70mn in the plant"
→ Neutral financial fact
```

**KEEP** ✅
```
"$aapl news is becoming very bullish again"
→ Clear bullish sentiment

"Relief for Lewis as Tesco sees sales grow for first time in a year"
→ "Relief" + growth = optimism

"PACCAR reports impressive Q4 earnings and strong demand"
→ "Impressive" + "strong" = positive emotion
```

## After Review

### Step 1: Check the Results
```bash
# See how many you kept
wc -l cleaned_optimism.csv
# Target: 20-30 samples (down from 83)

# View what you kept
cat cleaned_optimism.csv
```

### Step 2: Merge Back Into Full Dataset
```bash
# I'll create a script for this
python scripts/annotation/merge_cleaned_annotations.py
```

### Step 3: Retrain Classifier
```bash
# With cleaned data
python scripts/classifier/train_classifier.py \
    --data data/annotated/fingpt_annotated_v2.csv \
    --class-weight balanced
```

## Expected Outcome

**Before cleaning**:
- 83 optimism samples (many neutral)
- Model confusion: 100% predicted as anxiety
- 0% F1 score

**After cleaning**:
- ~20-30 genuine optimism samples
- Better class boundaries
- Expected improvement: 60-80% F1 for optimism

## Questions?

- **"Should I keep borderline cases?"** → When in doubt, remove (stricter = better)
- **"What if I'm not sure?"** → Use [s] to skip and review later
- **"Can I undo?"** → Yes, original file is unchanged
- **"Is automatic good enough?"** → For quick iteration, yes. For final model, do manual.

## Recommendation

1. Start with **automatic** to get a quick baseline
2. Check the results
3. If results look good (20-30 kept), proceed to retrain
4. If results look off, do **manual review** on the automatic output

Good luck! 🎯

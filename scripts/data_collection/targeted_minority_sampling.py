"""
Targeted Minority Class Sampling for Financial Emotion Detection

This script collects samples likely to contain rare emotions (hope, fear, excitement)
using keyword filtering and sends them for LLM annotation.

Target: 200-300 samples focused on minority classes
Estimated cost: $5-7 (at $0.003/sample)
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.annotation.llm_annotator import EmotionAnnotator


class TargetedMinoritySampler:
    """Sample financial texts likely to contain rare emotions"""
    
    # Keyword patterns for each minority emotion
    EMOTION_KEYWORDS = {
        'hope': {
            'primary': [
                r'\bhope\b', r'\bhopeful\b', r'\bhoping\b', r'\bhopes\b',
                r'\bconfident\b', r'\bconfidence\b', r'\boptimistic about\b',
                r'\bexpect.*improve', r'\bexpect.*growth', r'\bexpect.*gain',
                r'\blooking forward\b', r'\bencouraging signs\b',
                r'\bpromising outlook\b', r'\bpositive momentum\b',
                r'\bupside potential\b', r'\bturnaround expected\b',
                r'\brecovery.*expected\b', r'\brecovery.*anticipated\b',
                r'\bbullish.*long.term\b', r'\bstrongly believe\b',
                r'\bwell positioned\b', r'\bpoised for growth\b'
            ],
            'secondary': [
                r'\banticipate.*positive\b', r'\bfavorable outlook\b',
                r'\bimprovement.*ahead\b', r'\bbright.*future\b',
                r'\bupbeat\b', r'\bconstructive\b', r'\bencouraging\b'
            ]
        },
        'fear': {
            'primary': [
                r'\bfear\b', r'\bfeared\b', r'\bfearing\b', r'\bfears\b',
                r'\bworried\b', r'\bworry\b', r'\bworries\b', r'\bworrying\b',
                r'\bconcerned\b', r'\bconcern\b', r'\bconcerns\b',
                r'\bbearish\b', r'\bdownside risk\b', r'\bmajor risk\b',
                r'\bserious threat\b', r'\balarm\b', r'\balarming\b',
                r'\bpanic\b', r'\bfrightened\b', r'\bscared\b',
                r'\bdread\b', r'\bapprehensive\b', r'\bnervous about\b',
                r'\brisk.*significant\b', r'\brisk.*substantial\b',
                r'\bthreat.*imminent\b', r'\bvulnerable to\b'
            ],
            'secondary': [
                r'\bcautious outlook\b', r'\bwarning.*downturn\b',
                r'\bpessimistic\b', r'\bnegative outlook\b',
                r'\bdownside.*likely\b', r'\bexposed to risk\b',
                r'\bat risk of\b', r'\bthreatened by\b'
            ]
        },
        'excitement': {
            'primary': [
                r'\bexcited\b', r'\bexcitement\b', r'\bexciting\b',
                r'\bthrilled\b', r'\benthusiastic\b', r'\benthusiasm\b',
                r'\beager\b', r'\beagerly\b', r'\bbreakthrough\b',
                r'\bexceptional\b', r'\bremarkable growth\b',
                r'\bsurge\b', r'\bsoar\b', r'\bsoaring\b', r'\brally\b',
                r'\bimpressive gains\b', r'\bstrong momentum\b',
                r'\bbullish.*immediately\b', r'\bexplosive growth\b',
                r'\brecord.*high\b', r'\ball.time high\b'
            ],
            'secondary': [
                r'\bvery positive\b', r'\bhighly optimistic\b',
                r'\bstrongly recommend\b', r'\bgreat opportunity\b',
                r'\bsignificant upside\b', r'\bhuge potential\b'
            ]
        }
    }
    
    def __init__(self, raw_data_path: str, existing_annotations_path: str = None):
        """
        Initialize the sampler
        
        Args:
            raw_data_path: Path to FinGPT train.csv
            existing_annotations_path: Path to existing annotations to avoid duplicates
        """
        self.raw_data_path = Path(raw_data_path)
        self.existing_annotations_path = Path(existing_annotations_path) if existing_annotations_path else None
        
        print("Loading raw data...")
        self.raw_df = pd.read_csv(self.raw_data_path)
        print(f"Loaded {len(self.raw_df)} raw samples")
        
        # Load existing annotations to avoid duplicates
        self.existing_texts = set()
        if self.existing_annotations_path and self.existing_annotations_path.exists():
            existing_df = pd.read_csv(self.existing_annotations_path)
            self.existing_texts = set(existing_df['text'].values)
            print(f"Loaded {len(self.existing_texts)} existing annotations")
        
        # Compile regex patterns for efficiency
        self.compiled_patterns = {}
        for emotion, patterns in self.EMOTION_KEYWORDS.items():
            self.compiled_patterns[emotion] = {
                'primary': [re.compile(p, re.IGNORECASE) for p in patterns['primary']],
                'secondary': [re.compile(p, re.IGNORECASE) for p in patterns['secondary']]
            }
    
    def score_text(self, text: str, emotion: str) -> tuple:
        """
        Score how likely a text contains a specific emotion
        
        Returns:
            (score, primary_matches, secondary_matches)
        """
        text_lower = text.lower()
        
        primary_matches = 0
        secondary_matches = 0
        
        patterns = self.compiled_patterns[emotion]
        
        for pattern in patterns['primary']:
            if pattern.search(text):
                primary_matches += 1
        
        for pattern in patterns['secondary']:
            if pattern.search(text):
                secondary_matches += 1
        
        # Scoring: primary keywords worth 3 points, secondary worth 1 point
        score = (primary_matches * 3) + (secondary_matches * 1)
        
        return score, primary_matches, secondary_matches
    
    def filter_by_length(self, text: str, min_words: int = 10, max_words: int = 150) -> bool:
        """Filter texts by word count"""
        word_count = len(text.split())
        return min_words <= word_count <= max_words
    
    def sample_for_emotion(self, emotion: str, target_count: int, min_score: int = 3) -> pd.DataFrame:
        """
        Sample texts likely to contain a specific emotion
        
        Args:
            emotion: Target emotion ('hope', 'fear', 'excitement')
            target_count: Number of samples to collect
            min_score: Minimum keyword score to include
        
        Returns:
            DataFrame with sampled texts and scores
        """
        print(f"\n{'='*80}")
        print(f"Sampling for emotion: {emotion.upper()}")
        print(f"{'='*80}")
        
        # Score all texts
        scores = []
        for idx, row in self.raw_df.iterrows():
            text = row['input']
            
            # Skip if already annotated
            if text in self.existing_texts:
                continue
            
            # Skip if wrong length
            if not self.filter_by_length(text):
                continue
            
            score, primary_matches, secondary_matches = self.score_text(text, emotion)
            
            if score >= min_score:
                scores.append({
                    'text': text,
                    'score': score,
                    'primary_matches': primary_matches,
                    'secondary_matches': secondary_matches,
                    'emotion_target': emotion
                })
        
        print(f"Found {len(scores)} candidate texts with score >= {min_score}")
        
        if len(scores) == 0:
            print(f"⚠️ No texts found! Try lowering min_score or checking keywords.")
            return pd.DataFrame()
        
        # Sort by score (highest first) and sample
        scores_df = pd.DataFrame(scores).sort_values('score', ascending=False)
        
        # Take top N samples
        sampled = scores_df.head(target_count)
        
        print(f"\nSampled {len(sampled)} texts:")
        print(f"  Score range: {sampled['score'].min():.1f} - {sampled['score'].max():.1f}")
        print(f"  Avg primary matches: {sampled['primary_matches'].mean():.1f}")
        print(f"  Avg secondary matches: {sampled['secondary_matches'].mean():.1f}")
        
        # Show examples
        print(f"\nTop 3 examples:")
        for i, row in sampled.head(3).iterrows():
            print(f"\n  [{i}] Score: {row['score']:.0f} (primary={row['primary_matches']}, secondary={row['secondary_matches']})")
            print(f"  Text: {row['text'][:150]}...")
        
        return sampled
    
    def sample_balanced_minorities(
        self,
        hope_count: int = 100,
        fear_count: int = 100,
        excitement_count: int = 50,
        min_score: int = 3
    ) -> pd.DataFrame:
        """
        Sample balanced set of minority emotion candidates
        
        Args:
            hope_count: Target samples for hope
            fear_count: Target samples for fear
            excitement_count: Target samples for excitement
            min_score: Minimum keyword score
        
        Returns:
            Combined DataFrame with all sampled texts
        """
        print("="*80)
        print("TARGETED MINORITY SAMPLING")
        print("="*80)
        print(f"\nTarget distribution:")
        print(f"  Hope:       {hope_count} samples")
        print(f"  Fear:       {fear_count} samples")
        print(f"  Excitement: {excitement_count} samples")
        print(f"  Total:      {hope_count + fear_count + excitement_count} samples")
        print(f"\nMinimum keyword score: {min_score}")
        
        all_samples = []
        
        # Sample each emotion
        for emotion, count in [('hope', hope_count), ('fear', fear_count), ('excitement', excitement_count)]:
            sampled = self.sample_for_emotion(emotion, count, min_score)
            if len(sampled) > 0:
                all_samples.append(sampled)
        
        if not all_samples:
            print("\n⚠️ No samples collected!")
            return pd.DataFrame()
        
        # Combine all samples
        combined_df = pd.concat(all_samples, ignore_index=True)
        
        print(f"\n{'='*80}")
        print(f"SAMPLING COMPLETE")
        print(f"{'='*80}")
        print(f"Total samples collected: {len(combined_df)}")
        print(f"\nBreakdown by target emotion:")
        print(combined_df['emotion_target'].value_counts())
        
        return combined_df
    
    def save_samples(self, samples_df: pd.DataFrame, output_path: str):
        """Save sampled texts for annotation"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save just the texts for annotation
        samples_df[['text', 'emotion_target', 'score', 'primary_matches', 'secondary_matches']].to_csv(
            output_path, index=False
        )
        print(f"\n✅ Saved samples to: {output_path}")


def annotate_samples(samples_path: str, output_path: str, batch_size: int = 50):
    """
    Annotate the sampled texts using LLM
    
    Args:
        samples_path: Path to sampled texts CSV
        output_path: Path to save annotations
        batch_size: Number of samples per API batch
    """
    print("="*80)
    print("LLM ANNOTATION")
    print("="*80)
    
    # Load samples
    samples_df = pd.read_csv(samples_path)
    texts = samples_df['text'].tolist()
    
    print(f"\nAnnotating {len(texts)} samples...")
    print(f"Estimated cost: ${len(texts) * 0.003:.2f}")
    
    # Load config and initialize annotator
    import yaml
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    annotator = EmotionAnnotator(config)
    
    # Annotate in batches
    all_annotations = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        print(f"\nProcessing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1} ({len(batch)} samples)...")
        
        try:
            annotations = annotator.annotate_batch(batch)
            all_annotations.extend(annotations)
            
            print(f"  ✅ Annotated {len(annotations)} samples")
            
            # Show distribution in this batch
            batch_emotions = [a['emotion'] for a in annotations]
            from collections import Counter
            emotion_counts = Counter(batch_emotions)
            print(f"  Emotion distribution: {dict(emotion_counts)}")
            
        except Exception as e:
            print(f"  ❌ Error in batch: {e}")
            continue
    
    # Convert to DataFrame and save
    annotations_df = pd.DataFrame(all_annotations)
    
    # Add target emotion and keyword scores from original samples
    annotations_df = annotations_df.merge(
        samples_df[['text', 'emotion_target', 'score', 'primary_matches', 'secondary_matches']],
        on='text',
        how='left'
    )
    
    annotations_df.to_csv(output_path, index=False)
    
    print(f"\n{'='*80}")
    print(f"ANNOTATION COMPLETE")
    print(f"{'='*80}")
    print(f"Total annotated: {len(annotations_df)}")
    print(f"\nFinal emotion distribution:")
    print(annotations_df['emotion'].value_counts())
    print(f"\nTarget vs Actual:")
    comparison = annotations_df.groupby(['emotion_target', 'emotion']).size().unstack(fill_value=0)
    print(comparison)
    print(f"\n✅ Saved annotations to: {output_path}")
    
    # Calculate success rate for each target
    print(f"\nKeyword targeting effectiveness:")
    for target in annotations_df['emotion_target'].unique():
        subset = annotations_df[annotations_df['emotion_target'] == target]
        matched = len(subset[subset['emotion'] == target])
        total = len(subset)
        print(f"  {target}: {matched}/{total} ({100*matched/total:.1f}%) correctly predicted")


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Targeted minority class sampling')
    parser.add_argument('--raw-data', type=str, 
                       default='data/raw/fingpt/train.csv',
                       help='Path to raw FinGPT data')
    parser.add_argument('--existing-annotations', type=str,
                       default='data/annotated/fingpt_annotated_balanced.csv',
                       help='Path to existing annotations (to avoid duplicates)')
    parser.add_argument('--output-samples', type=str,
                       default='data/processed/minority_samples_targeted.csv',
                       help='Path to save sampled texts')
    parser.add_argument('--output-annotations', type=str,
                       default='data/annotated/minority_annotations_targeted.csv',
                       help='Path to save final annotations')
    parser.add_argument('--hope-count', type=int, default=100,
                       help='Target hope samples')
    parser.add_argument('--fear-count', type=int, default=100,
                       help='Target fear samples')
    parser.add_argument('--excitement-count', type=int, default=50,
                       help='Target excitement samples')
    parser.add_argument('--min-score', type=int, default=3,
                       help='Minimum keyword score')
    parser.add_argument('--sample-only', action='store_true',
                       help='Only sample, do not annotate')
    parser.add_argument('--annotate-only', action='store_true',
                       help='Only annotate existing samples')
    
    args = parser.parse_args()
    
    # Step 1: Sample texts
    if not args.annotate_only:
        sampler = TargetedMinoritySampler(
            raw_data_path=args.raw_data,
            existing_annotations_path=args.existing_annotations
        )
        
        samples_df = sampler.sample_balanced_minorities(
            hope_count=args.hope_count,
            fear_count=args.fear_count,
            excitement_count=args.excitement_count,
            min_score=args.min_score
        )
        
        if len(samples_df) > 0:
            sampler.save_samples(samples_df, args.output_samples)
        else:
            print("❌ No samples collected. Exiting.")
            return
    
    # Step 2: Annotate samples
    if not args.sample_only:
        print("\n" + "="*80)
        response = input("Proceed with LLM annotation? This will cost ~$0.75-2.00. (y/n): ")
        if response.lower() == 'y':
            annotate_samples(
                samples_path=args.output_samples if not args.annotate_only else args.output_samples,
                output_path=args.output_annotations,
                batch_size=50
            )
        else:
            print("Annotation cancelled.")


if __name__ == '__main__':
    main()

"""
Interactive Annotation Review Tool
Helps manually review and clean annotation quality issues

Usage:
    python scripts/annotation/review_annotations.py --input review_optimism.csv --output cleaned_optimism.csv
"""

import argparse
import pandas as pd
from pathlib import Path

def review_annotations_interactive(input_file: str, output_file: str, emotion: str = "optimism"):
    """
    Interactive review of annotations
    
    Args:
        input_file: CSV file with annotations to review
        output_file: Output file for cleaned annotations
        emotion: Emotion being reviewed
    """
    df = pd.read_csv(input_file)
    
    print("=" * 80)
    print(f"Annotation Review Tool - {emotion.upper()}")
    print("=" * 80)
    print(f"\nTotal samples to review: {len(df)}")
    print("\nInstructions:")
    print("  For each sample, decide:")
    print("    [k] Keep - Genuine emotional content")
    print("    [r] Remove - Neutral/factual statement")
    print("    [u] Uncertainty - Change to 'uncertainty'")
    print("    [n] Neutral - Mark for removal (no emotion)")
    print("    [s] Skip - Review later")
    print("    [q] Quit and save")
    print("\n" + "=" * 80)
    
    decisions = []
    
    for idx, row in df.iterrows():
        print(f"\n[Sample {idx + 1}/{len(df)}]")
        print("-" * 80)
        print(f"Text: {row['text']}")
        print(f"Current Label: {row['emotion']}")
        print(f"Confidence: {row['confidence']}")
        if 'reasoning' in row:
            print(f"Reasoning: {row['reasoning']}")
        print("-" * 80)
        
        while True:
            choice = input("\nDecision [k/r/u/n/s/q]: ").strip().lower()
            
            if choice == 'k':
                decisions.append('keep')
                print("✓ Keeping as optimism")
                break
            elif choice == 'r':
                decisions.append('remove')
                print("✗ Marked for removal")
                break
            elif choice == 'u':
                decisions.append('uncertainty')
                print("→ Changed to uncertainty")
                break
            elif choice == 'n':
                decisions.append('neutral')
                print("→ Marked as neutral (remove)")
                break
            elif choice == 's':
                decisions.append('skip')
                print("⊘ Skipped for now")
                break
            elif choice == 'q':
                print("\nQuitting and saving progress...")
                decisions.append('skip')  # Mark current as skip
                save_results(df, decisions, output_file, idx + 1)
                return
            else:
                print("Invalid choice. Please enter k, r, u, n, s, or q")
    
    # Save all results
    save_results(df, decisions, output_file, len(df))

def save_results(df, decisions, output_file, num_reviewed):
    """Save review results"""
    # Pad decisions if needed
    while len(decisions) < num_reviewed:
        decisions.append('skip')
    
    # Add decision column
    df_reviewed = df.iloc[:num_reviewed].copy()
    df_reviewed['review_decision'] = decisions[:num_reviewed]
    
    # Create cleaned dataset
    kept = df_reviewed[df_reviewed['review_decision'] == 'keep']
    changed = df_reviewed[df_reviewed['review_decision'] == 'uncertainty'].copy()
    changed['emotion'] = 'uncertainty'
    
    cleaned = pd.concat([kept, changed])
    
    # Save
    cleaned.to_csv(output_file, index=False)
    
    # Stats
    print("\n" + "=" * 80)
    print("REVIEW SUMMARY")
    print("=" * 80)
    print(f"Total reviewed: {num_reviewed}")
    print(f"Kept as {df['emotion'].iloc[0]}: {sum(1 for d in decisions if d == 'keep')}")
    print(f"Changed to uncertainty: {sum(1 for d in decisions if d == 'uncertainty')}")
    print(f"Marked for removal: {sum(1 for d in decisions if d in ['remove', 'neutral'])}")
    print(f"Skipped: {sum(1 for d in decisions if d == 'skip')}")
    print(f"\nCleaned data saved to: {output_file}")
    print(f"Samples in cleaned dataset: {len(cleaned)}")

def auto_review_heuristics(input_file: str, output_file: str, strict: bool = False):
    """
    Automatic heuristic-based review (faster but less accurate)
    
    Rules:
    - Keep if: contains emotional keywords
    - Remove if: purely factual, no emotional markers
    - Strict mode: Only keep samples with strong emotional indicators
    """
    df = pd.read_csv(input_file)
    
    # Emotional keywords for optimism
    positive_keywords = [
        'bullish', 'optimis', 'excit', 'soar', 'surg', 'rally',
        'upbeat', 'confident', 'positive outlook', 'gains',
        'strong demand', 'impressive', 'relief', 'breakthrough'
    ]
    
    # Strong emotional indicators (for strict mode)
    strong_emotional = [
        'bullish', 'very bullish', 'extremely', 'soaring', 'surging',
        'breakthrough', 'excited', 'optimistic', 'confident'
    ]
    
    # Neutral/factual indicators
    neutral_indicators = [
        'acquires', 'owns', 'operates', 'located', 'based in',
        'was the first', 'total area', 'the company', 'provides',
        'offers services', 'euro', 'usd', 'sq m', 'announced',
        'reported', 'said', 'percent', 'million', 'invest'
    ]
    
    def score_sample(text):
        text_lower = text.lower()
        positive_score = sum(1 for kw in positive_keywords if kw in text_lower)
        strong_score = sum(1 for kw in strong_emotional if kw in text_lower)
        neutral_score = sum(1 for kw in neutral_indicators if kw in text_lower)
        return positive_score, strong_score, neutral_score
    
    decisions = []
    for _, row in df.iterrows():
        pos_score, strong_score, neu_score = score_sample(row['text'])
        
        if strict:
            # Strict mode: only keep if strong emotional indicators present
            if strong_score > 0 and neu_score == 0:
                decisions.append('keep')
            elif neu_score > 0:
                decisions.append('remove')
            elif row['confidence'] < 0.8:
                decisions.append('uncertainty')
            else:
                decisions.append('remove')  # Default to remove if unclear
        else:
            # Normal mode: more lenient
            if pos_score > 0 and neu_score == 0:
                decisions.append('keep')
            elif neu_score > pos_score:
                decisions.append('remove')
            elif row['confidence'] < 0.8:
                decisions.append('uncertainty')
            else:
                decisions.append('keep')  # Default to keep if unclear
    
    df['review_decision'] = decisions
    
    # Create cleaned dataset
    kept = df[df['review_decision'] == 'keep']
    changed = df[df['review_decision'] == 'uncertainty'].copy()
    changed['emotion'] = 'uncertainty'
    
    cleaned = pd.concat([kept, changed])
    cleaned.to_csv(output_file, index=False)
    
    print("=" * 80)
    print("AUTOMATIC REVIEW SUMMARY")
    print("=" * 80)
    print(f"Total samples: {len(df)}")
    print(f"Kept: {sum(1 for d in decisions if d == 'keep')}")
    print(f"Changed to uncertainty: {sum(1 for d in decisions if d == 'uncertainty')}")
    print(f"Marked for removal: {sum(1 for d in decisions if d == 'remove')}")
    print(f"\nCleaned data saved to: {output_file}")
    print(f"Samples in cleaned dataset: {len(cleaned)}")
    print("\nNote: Automatic review may miss nuances. Consider manual review for critical samples.")

def main():
    parser = argparse.ArgumentParser(description='Review and clean annotations')
    parser.add_argument('--input', type=str, default='review_optimism.csv',
                       help='Input CSV file to review')
    parser.add_argument('--output', type=str, default='cleaned_optimism.csv',
                       help='Output CSV file for cleaned annotations')
    parser.add_argument('--auto', action='store_true',
                       help='Use automatic heuristic-based review instead of interactive')
    parser.add_argument('--strict', action='store_true',
                       help='Use strict mode (only keep samples with strong emotional indicators)')
    parser.add_argument('--emotion', type=str, default='optimism',
                       help='Emotion being reviewed')
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"Error: Input file {args.input} not found")
        return
    
    if args.auto:
        mode = "STRICT" if args.strict else "NORMAL"
        print(f"Running automatic heuristic-based review ({mode} mode)...")
        auto_review_heuristics(args.input, args.output, strict=args.strict)
    else:
        print("Starting interactive review...")
        review_annotations_interactive(args.input, args.output, args.emotion)

if __name__ == "__main__":
    main()

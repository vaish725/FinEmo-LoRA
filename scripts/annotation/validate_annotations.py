"""
Annotation Validation Script
Helps with manual validation and quality assessment of LLM-generated annotations
Prepares validation sample and provides inter-annotator agreement metrics
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from sklearn.metrics import cohen_kappa_score, classification_report
import random

def load_config():
    """
    Load the project configuration file
    
    Returns:
        dict: Configuration dictionary
    """
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_validation_sample(annotated_file: str, output_file: str, sample_size: int = 200):
    """
    Create a stratified sample for manual validation
    
    Args:
        annotated_file (str): Path to annotated CSV file
        output_file (str): Path to save validation sample
        sample_size (int): Number of samples to extract
    """
    print("=" * 80)
    print("Creating Validation Sample")
    print("=" * 80)
    
    # Load annotated data
    df = pd.read_csv(annotated_file)
    print(f"\nLoaded {len(df)} annotated samples")
    
    # Stratified sampling: get equal samples from each emotion
    config = load_config()
    emotions = config['emotion_labels']
    
    samples_per_emotion = sample_size // len(emotions)
    validation_samples = []
    
    print(f"\nCreating stratified sample ({samples_per_emotion} per emotion):")
    
    for emotion in emotions:
        emotion_df = df[df['emotion'] == emotion]
        
        if len(emotion_df) >= samples_per_emotion:
            sample = emotion_df.sample(n=samples_per_emotion, random_state=42)
        else:
            print(f"  Warning: Only {len(emotion_df)} samples for '{emotion}', using all")
            sample = emotion_df
        
        validation_samples.append(sample)
        print(f"  {emotion}: {len(sample)} samples")
    
    # Combine samples
    validation_df = pd.concat(validation_samples, ignore_index=True)
    
    # Add columns for manual annotation
    validation_df['manual_emotion'] = ''
    validation_df['manual_confidence'] = ''
    validation_df['notes'] = ''
    
    # Shuffle the samples
    validation_df = validation_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save validation sample
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    validation_df.to_csv(output_path, index=False)
    
    print(f"\nValidation sample saved to: {output_path}")
    print(f"Total validation samples: {len(validation_df)}")
    print("\nNext steps:")
    print("  1. Open the validation file in a spreadsheet editor")
    print("  2. Fill in 'manual_emotion' column with your annotation")
    print("  3. Fill in 'manual_confidence' (Low/Medium/High)")
    print("  4. Add notes for disagreements or edge cases")
    print("  5. Save the file and run compute_agreement() function")

def compute_agreement(validation_file: str):
    """
    Compute inter-annotator agreement between LLM and manual annotations
    
    Args:
        validation_file (str): Path to validation file with manual annotations
    """
    print("=" * 80)
    print("Computing Inter-Annotator Agreement")
    print("=" * 80)
    
    # Load validation data
    df = pd.read_csv(validation_file)
    
    # Filter rows where manual annotation is provided
    df_annotated = df[df['manual_emotion'].notna() & (df['manual_emotion'] != '')]
    
    if len(df_annotated) == 0:
        print("\nNo manual annotations found!")
        print("Please fill in the 'manual_emotion' column in the validation file")
        return
    
    print(f"\nFound {len(df_annotated)} manually annotated samples")
    
    # Get LLM and manual annotations
    llm_annotations = df_annotated['emotion'].tolist()
    manual_annotations = df_annotated['manual_emotion'].tolist()
    
    # Compute agreement metrics
    
    # 1. Overall agreement (accuracy)
    agreement = sum(1 for llm, manual in zip(llm_annotations, manual_annotations) 
                   if llm == manual) / len(llm_annotations)
    
    print(f"\nOverall Agreement: {agreement:.2%}")
    
    # 2. Cohen's Kappa (accounts for chance agreement)
    kappa = cohen_kappa_score(manual_annotations, llm_annotations)
    print(f"Cohen's Kappa: {kappa:.3f}")
    
    # Interpret kappa
    if kappa < 0:
        interpretation = "Poor (worse than chance)"
    elif kappa < 0.2:
        interpretation = "Slight"
    elif kappa < 0.4:
        interpretation = "Fair"
    elif kappa < 0.6:
        interpretation = "Moderate"
    elif kappa < 0.8:
        interpretation = "Substantial"
    else:
        interpretation = "Almost Perfect"
    
    print(f"Kappa Interpretation: {interpretation}")
    
    # 3. Per-class metrics
    print("\nPer-Emotion Agreement:")
    config = load_config()
    emotions = config['emotion_labels']
    
    for emotion in emotions:
        emotion_indices = [i for i, manual in enumerate(manual_annotations) if manual == emotion]
        
        if len(emotion_indices) == 0:
            continue
        
        emotion_agreement = sum(1 for i in emotion_indices 
                               if llm_annotations[i] == manual_annotations[i]) / len(emotion_indices)
        
        print(f"  {emotion}: {emotion_agreement:.2%} ({len(emotion_indices)} samples)")
    
    # 4. Detailed classification report
    print("\nDetailed Classification Report:")
    print("(LLM predictions vs Manual labels)")
    print("-" * 80)
    report = classification_report(manual_annotations, llm_annotations, 
                                   labels=emotions, zero_division=0)
    print(report)
    
    # 5. Common disagreements
    print("\nMost Common Disagreements:")
    disagreements = []
    for llm, manual in zip(llm_annotations, manual_annotations):
        if llm != manual:
            disagreements.append(f"{llm} -> {manual}")
    
    if disagreements:
        from collections import Counter
        disagreement_counts = Counter(disagreements)
        for disagreement, count in disagreement_counts.most_common(10):
            print(f"  {disagreement}: {count} times")
    else:
        print("  No disagreements found - perfect agreement!")
    
    # 6. Confidence analysis
    if 'confidence' in df_annotated.columns:
        print("\nConfidence Analysis:")
        
        # Agreement by confidence level
        high_conf = df_annotated[df_annotated['confidence'] >= 0.8]
        med_conf = df_annotated[(df_annotated['confidence'] >= 0.6) & 
                                (df_annotated['confidence'] < 0.8)]
        low_conf = df_annotated[df_annotated['confidence'] < 0.6]
        
        for conf_name, conf_df in [("High (>=0.8)", high_conf), 
                                    ("Medium (0.6-0.8)", med_conf), 
                                    ("Low (<0.6)", low_conf)]:
            if len(conf_df) > 0:
                conf_agreement = sum(1 for llm, manual in 
                                    zip(conf_df['emotion'], conf_df['manual_emotion'])
                                    if llm == manual) / len(conf_df)
                print(f"  {conf_name}: {conf_agreement:.2%} agreement ({len(conf_df)} samples)")
    
    print("\n" + "=" * 80)
    print("Validation Complete!")
    print("=" * 80)
    
    # Recommendations based on results
    print("\nRecommendations:")
    if agreement >= 0.85:
        print("  - Excellent agreement! LLM annotations are high quality")
        print("  - Proceed with using the annotated dataset for training")
    elif agreement >= 0.70:
        print("  - Good agreement, but some room for improvement")
        print("  - Consider refining the annotation prompt")
        print("  - Focus on emotions with low agreement")
    else:
        print("  - Agreement is below ideal threshold")
        print("  - Review and refine the system prompt")
        print("  - Consider providing more examples in the prompt")
        print("  - May need to increase confidence threshold filter")

def main():
    """
    Example usage of validation functions
    """
    print("\nValidation Workflow:")
    print("-" * 80)
    print("Step 1: Create validation sample")
    print("  create_validation_sample(")
    print("      annotated_file='data/annotated/fingpt_annotated.csv',")
    print("      output_file='data/annotated/validation_sample.csv',")
    print("      sample_size=200")
    print("  )")
    print()
    print("Step 2: Manually annotate the validation sample")
    print("  - Open validation_sample.csv")
    print("  - Fill in manual_emotion column")
    print()
    print("Step 3: Compute agreement metrics")
    print("  compute_agreement(")
    print("      validation_file='data/annotated/validation_sample.csv'")
    print("  )")
    print("-" * 80)

if __name__ == "__main__":
    main()


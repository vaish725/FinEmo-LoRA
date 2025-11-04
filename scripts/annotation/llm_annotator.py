"""
LLM-Based Emotion Annotation Pipeline
Uses GPT-4 (or other LLMs) to annotate financial text with our 6-emotion taxonomy:
{anxiety, excitement, optimism, fear, uncertainty, hope}

This creates high-quality pseudo-labels for training FinEmo-LoRA
"""

import os
import yaml
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from tqdm import tqdm
import time

# Load environment variables
load_dotenv()

# LLM API imports
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI library not installed. Install with: pip install openai")

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

class EmotionAnnotator:
    """
    LLM-based emotion annotator for financial text
    Supports GPT-4 and other providers
    """
    
    def __init__(self, config: dict):
        """
        Initialize the annotator with configuration
        
        Args:
            config (dict): Project configuration
        """
        self.config = config
        self.emotion_labels = config['emotion_labels']
        self.annotation_config = config['annotation']
        
        # Initialize LLM client
        self.client = self._initialize_llm_client()
        
        # Create system prompt for emotion annotation
        self.system_prompt = self._create_system_prompt()
        
    def _initialize_llm_client(self):
        """
        Initialize the appropriate LLM API client based on config
        
        Returns:
            Client object for the selected LLM provider
        """
        provider = self.annotation_config['llm_provider']
        
        if provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI library not installed")
            
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            
            return OpenAI(api_key=api_key)
        
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    def _create_system_prompt(self) -> str:
        """
        Create the system prompt that defines the annotation task
        
        Returns:
            str: System prompt for the LLM
        """
        emotions_str = ", ".join(self.emotion_labels)
        
        prompt = f"""You are an expert financial text analyst specializing in emotion detection.

Your task is to analyze financial news text and identify the PRIMARY economic emotion expressed.

EMOTION TAXONOMY (choose exactly ONE):
- anxiety: Uneasiness about future market conditions, but not extreme
- excitement: High energy positive emotion about opportunities or gains
- optimism: Confident positive outlook about future performance
- fear: Intense worry or alarm about losses or negative outcomes
- uncertainty: Confusion or doubt about market direction or outcomes
- hope: Aspiration or wish for positive outcomes, with some doubt

ANNOTATION GUIDELINES:
1. Focus on the OVERALL emotional tone, not individual words
2. Consider the financial context and implications
3. Distinguish between similar emotions (e.g., fear vs anxiety, optimism vs hope)
4. If multiple emotions are present, choose the DOMINANT one
5. Provide a confidence score (0.0 to 1.0) based on how clear the emotion is

RESPONSE FORMAT (JSON only):
{{
  "emotion": "one of [{emotions_str}]",
  "confidence": 0.85,
  "reasoning": "Brief explanation of why this emotion was chosen"
}}

IMPORTANT: 
- Respond ONLY with valid JSON
- Confidence must be between 0.0 and 1.0
- Emotion must be exactly one of the listed options
- Keep reasoning concise (1-2 sentences)"""
        
        return prompt
    
    def annotate_single_text(self, text: str) -> Dict:
        """
        Annotate a single text with emotion label
        
        Args:
            text (str): Financial text to annotate
            
        Returns:
            dict: Annotation result with emotion, confidence, and reasoning
        """
        try:
            # Create user prompt with the text
            user_prompt = f"Analyze the following financial text and identify the primary emotion:\n\n{text}"
            
            # Call LLM API
            response = self.client.chat.completions.create(
                model=self.annotation_config['model'],
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.annotation_config['temperature'],
                max_tokens=self.annotation_config['max_tokens'],
                response_format={"type": "json_object"}  # Force JSON output
            )
            
            # Parse response
            result = json.loads(response.choices[0].message.content)
            
            # Validate emotion label
            if result['emotion'] not in self.emotion_labels:
                print(f"Warning: Invalid emotion '{result['emotion']}' returned. Skipping.")
                return None
            
            # Validate confidence
            if not (0.0 <= result['confidence'] <= 1.0):
                print(f"Warning: Invalid confidence {result['confidence']}. Setting to 0.5.")
                result['confidence'] = 0.5
            
            return result
            
        except Exception as e:
            print(f"Error annotating text: {e}")
            return None
    
    def annotate_batch(self, texts: List[str], batch_id: str = "") -> List[Dict]:
        """
        Annotate a batch of texts with progress tracking
        
        Args:
            texts (List[str]): List of texts to annotate
            batch_id (str): Identifier for this batch (for logging)
            
        Returns:
            List[Dict]: List of annotation results
        """
        results = []
        
        print(f"\nAnnotating batch {batch_id} ({len(texts)} texts)")
        
        for idx, text in enumerate(tqdm(texts, desc="Annotating")):
            # Annotate single text
            result = self.annotate_single_text(text)
            
            if result:
                result['text'] = text
                result['batch_id'] = batch_id
                result['index'] = idx
                results.append(result)
            
            # Rate limiting: small delay between API calls
            time.sleep(0.1)
        
        return results
    
    def filter_by_confidence(self, annotations: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Filter annotations by confidence threshold
        
        Args:
            annotations (List[Dict]): List of annotation results
            
        Returns:
            Tuple[List[Dict], List[Dict]]: (high_confidence, low_confidence) annotations
        """
        threshold = self.annotation_config['min_confidence']
        
        high_confidence = [a for a in annotations if a['confidence'] >= threshold]
        low_confidence = [a for a in annotations if a['confidence'] < threshold]
        
        return high_confidence, low_confidence

def annotate_dataset(input_path: str, output_path: str, text_column: str, 
                     max_samples: int = None, batch_size: int = 50):
    """
    Main function to annotate an entire dataset
    
    Args:
        input_path (str): Path to input CSV file
        output_path (str): Path to save annotated CSV
        text_column (str): Name of the column containing text to annotate
        max_samples (int): Maximum number of samples to annotate (None = all)
        batch_size (int): Number of samples per batch
    """
    print("=" * 80)
    print("LLM-Based Emotion Annotation Pipeline")
    print("=" * 80)
    
    # Load configuration
    config = load_config()
    
    # Initialize annotator
    print("\nInitializing LLM annotator...")
    annotator = EmotionAnnotator(config)
    
    # Load dataset
    print(f"\nLoading dataset from: {input_path}")
    df = pd.read_csv(input_path)
    
    if text_column not in df.columns:
        print(f"Error: Column '{text_column}' not found in dataset")
        print(f"Available columns: {list(df.columns)}")
        return
    
    print(f"Dataset loaded: {len(df)} samples")
    
    # Limit samples if specified
    if max_samples:
        df = df.head(max_samples)
        print(f"Limiting to first {max_samples} samples")
    
    # Get texts to annotate
    texts = df[text_column].tolist()
    
    # Annotate in batches
    all_annotations = []
    num_batches = (len(texts) + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(texts))
        batch_texts = texts[start_idx:end_idx]
        
        batch_results = annotator.annotate_batch(
            batch_texts, 
            batch_id=f"{batch_idx+1}/{num_batches}"
        )
        all_annotations.extend(batch_results)
    
    # Convert to DataFrame
    annotations_df = pd.DataFrame(all_annotations)
    
    # Filter by confidence
    high_conf, low_conf = annotator.filter_by_confidence(all_annotations)
    
    print("\n" + "=" * 80)
    print("Annotation Summary")
    print("=" * 80)
    print(f"Total annotations: {len(all_annotations)}")
    print(f"High confidence (>= {config['annotation']['min_confidence']}): {len(high_conf)}")
    print(f"Low confidence (< {config['annotation']['min_confidence']}): {len(low_conf)}")
    
    # Emotion distribution
    print("\nEmotion Distribution:")
    emotion_counts = annotations_df['emotion'].value_counts()
    for emotion, count in emotion_counts.items():
        percentage = (count / len(annotations_df)) * 100
        print(f"  {emotion}: {count} ({percentage:.1f}%)")
    
    # Save results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save all annotations
    annotations_df.to_csv(output_path, index=False)
    print(f"\nAll annotations saved to: {output_path}")
    
    # Save high-confidence subset
    high_conf_path = output_path.parent / f"{output_path.stem}_high_confidence.csv"
    high_conf_df = pd.DataFrame(high_conf)
    high_conf_df.to_csv(high_conf_path, index=False)
    print(f"High-confidence annotations saved to: {high_conf_path}")
    
    # Save low-confidence for manual review
    low_conf_path = output_path.parent / f"{output_path.stem}_low_confidence.csv"
    low_conf_df = pd.DataFrame(low_conf)
    low_conf_df.to_csv(low_conf_path, index=False)
    print(f"Low-confidence annotations (for manual review) saved to: {low_conf_path}")
    
    print("\n" + "=" * 80)
    print("Annotation pipeline completed!")
    print("=" * 80)

def main():
    """
    Example usage of the annotation pipeline
    """
    # This is an example - adjust paths and column names for your actual data
    
    print("\nExample Usage:")
    print("-" * 80)
    print("from scripts.annotation.llm_annotator import annotate_dataset")
    print()
    print("# Annotate FinGPT data")
    print("annotate_dataset(")
    print("    input_path='data/raw/fingpt/train.csv',")
    print("    output_path='data/annotated/fingpt_annotated.csv',")
    print("    text_column='text',  # Adjust column name")
    print("    max_samples=1000,  # Start small for testing")
    print("    batch_size=50")
    print(")")
    print("-" * 80)
    print()
    print("To use this script:")
    print("  1. Set up your .env file with OPENAI_API_KEY")
    print("  2. Download datasets using data_collection scripts")
    print("  3. Call annotate_dataset() with your data paths")
    print("  4. Review high_confidence and low_confidence outputs")

if __name__ == "__main__":
    main()


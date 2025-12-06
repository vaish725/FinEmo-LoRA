"""
Synthetic Financial Emotion Sample Generator
=============================================

PURPOSE:
    Generate realistic financial texts expressing specific emotions using LLM.
    This is the fastest way to expand training data (2000+ samples in 1-2 hours).

APPROACH:
    Uses GPT-4o-mini API to generate diverse, realistic financial texts for each emotion.
    Much faster and cheaper than manual annotation (~$0.01 per sample).

USAGE:
    python generate_synthetic_samples.py --emotion hope --count 500 --model gpt-4o-mini
    python generate_synthetic_samples.py --emotion fear --count 500 --model gpt-4o-mini

COST:
    ~$10-20 for 2000 samples using GPT-4o-mini

AUTHOR:
    Vaishnavi Kamdi - Fall 2025 NNDL Project
"""

import pandas as pd
import json
from pathlib import Path
import argparse
from tqdm import tqdm
import time

# Example templates for each emotion
EMOTION_TEMPLATES = {
    'hope': [
        "Stock analysis showing potential recovery",
        "Optimistic outlook on company turnaround",
        "Positive catalyst for stock rebound",
        "Potential breakthrough in earnings",
        "Recovery signs in market sector"
    ],
    'fear': [
        "Concerns about market crash",
        "Worry about company bankruptcy",
        "Anxiety over declining stock",
        "Warning signs for portfolio",
        "Risk of significant losses"
    ],
    'excitement': [
        "Incredible gains on investment",
        "Amazing stock performance",
        "Explosive growth in sector",
        "Breakthrough financial news",
        "Massive rally in market"
    ],
    'anxiety': [
        "Uncertain about investment decision",
        "Nervous about market volatility",
        "Stressed about portfolio performance",
        "Worried about holding position",
        "Uneasy about market direction"
    ],
    'optimism': [
        "Confident in long-term growth",
        "Strong fundamentals support price",
        "Positive outlook for investment",
        "Solid company performance",
        "Good prospects for sector"
    ],
    'uncertainty': [
        "Mixed signals from market",
        "Unclear direction for stock",
        "Conflicting indicators",
        "Hard to predict outcome",
        "Market showing confusion"
    ]
}

GENERATION_PROMPT = """Generate {count} realistic financial text samples expressing {emotion}.

Emotion definition: {definition}

Requirements:
- Length: 50-300 characters (like a social media post or news headline)
- Style: Mix of: retail investor comments, news snippets, analyst takes
- Realistic: Use real company types (tech, pharma, EV, etc.) but keep names generic
- Varied: Different sentence structures and vocabulary
- Natural: Sound like real people discussing finance

Examples of {emotion}:
{examples}

Return ONLY a JSON array of strings:
["text 1", "text 2", "text 3", ...]
"""

EMOTION_DEFINITIONS = {
    'hope': "Expectation of positive future outcomes, optimism about recovery or growth potential. Not confident yet, but hopeful.",
    'fear': "Worry about negative outcomes, anxiety about losses, concern about crashes or declines.",
    'excitement': "High energy positive emotion, enthusiasm about gains, explosive movement, incredible performance.",
    'anxiety': "Nervous uncertainty, stress without clear direction, worried about decisions.",
    'optimism': "Confident positive outlook, belief in success, strong fundamentals.",
    'uncertainty': "Confusion, lack of clarity, mixed signals, hard to predict."
}

EMOTION_EXAMPLES = {
    'hope': [
        "I'm hoping this stock can recover after earnings. The fundamentals are still solid.",
        "Could see a turnaround if they hit their guidance. Fingers crossed.",
        "This might be the bottom. Hoping for a bounce back soon."
    ],
    'fear': [
        "Terrified this will drop another 20%. Should I sell now or wait?",
        "Warning signs everywhere. Market crash incoming?",
        "Scared to hold through earnings. What if they miss badly?"
    ],
    'excitement': [
        "Holy gains! This stock is going to the moon!",
        "This is insane! Up 40% in one week. LFG!",
        "Incredible rally! Best trade of the year!"
    ],
    'anxiety': [
        "Not sure if I should sell before earnings. Feeling nervous.",
        "Market volatility has me stressed. What's the right move?",
        "Anxious about this position. Hold or cut losses?"
    ],
    'optimism': [
        "Strong fundamentals will drive this higher long-term.",
        "Confident this will grow steadily. Good company.",
        "Solid earnings report. This should continue upward."
    ],
    'uncertainty': [
        "Hard to say where this goes next. Mixed signals.",
        "Market is unclear. Could go either way honestly.",
        "Not sure what to make of this. Conflicting data."
    ]
}

def generate_with_openai(emotion, count, api_key, batch_size=50):
    """
    Generate synthetic financial texts using OpenAI's GPT-4o-mini API.
    
    PARAMETERS:
        emotion (str): Target emotion ('hope', 'fear', 'excitement', etc.)
        count (int): Total number of samples to generate
        api_key (str): OpenAI API key for authentication
        batch_size (int): Number of samples per API call (default: 50)
    
    RETURNS:
        list: Generated text samples, or None if error occurs
    
    APPROACH:
        - Uses GPT-4o-mini with JSON mode for structured output
        - Generates in batches to handle large requests
        - Temperature 0.9 for diverse, creative outputs
        - Fallback regex parsing if JSON.loads fails
    """
    
    try:
        from openai import OpenAI
    except ImportError:
        print("❌ ERROR: Install OpenAI library: pip install openai")
        return None
    
    # Initialize OpenAI client with API key
    client = OpenAI(api_key=api_key)
    
    prompt = GENERATION_PROMPT.format(
        count=batch_size,
        emotion=emotion,
        definition=EMOTION_DEFINITIONS[emotion],
        examples='\n'.join(f"- {ex}" for ex in EMOTION_EXAMPLES[emotion])
    )
    
    all_samples = []
    batches = (count + batch_size - 1) // batch_size
    
    for i in tqdm(range(batches), desc=f"Generating {emotion} samples"):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a financial text generator. Generate realistic financial social media posts and comments. Return ONLY a JSON array."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.9  # Higher temperature for variety
            )
            
            content = response.choices[0].message.content
            
            # Try to extract JSON from response
            try:
                result = json.loads(content)
            except:
                # Try to find JSON array in text
                import re
                json_match = re.search(r'\[.*\]', content, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    result = []
            
            # Handle different response formats
            if isinstance(result, dict) and 'samples' in result:
                samples = result['samples']
            elif isinstance(result, dict) and 'texts' in result:
                samples = result['texts']
            elif isinstance(result, list):
                samples = result
            else:
                samples = list(result.values())[0] if result else []
            
            all_samples.extend(samples[:batch_size])
            
        except Exception as e:
            print(f"\nError in batch {i+1}: {e}")
        
        time.sleep(1)  # Rate limiting
    
    return all_samples[:count]

def generate_with_anthropic(emotion, count, api_key, batch_size=50):
    """Generate samples using Anthropic Claude."""
    
    try:
        import anthropic
    except ImportError:
        print("Install Anthropic: pip install anthropic")
        return None
    
    client = anthropic.Anthropic(api_key=api_key)
    
    prompt = GENERATION_PROMPT.format(
        count=batch_size,
        emotion=emotion,
        definition=EMOTION_DEFINITIONS[emotion],
        examples='\n'.join(f"- {ex}" for ex in EMOTION_EXAMPLES[emotion])
    )
    
    all_samples = []
    batches = (count + batch_size - 1) // batch_size
    
    for i in tqdm(range(batches), desc=f"Generating {emotion} samples"):
        try:
            message = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            text = message.content[0].text
            # Try to parse JSON
            samples = json.loads(text)
            
            if isinstance(samples, dict):
                samples = list(samples.values())[0]
            
            all_samples.extend(samples[:batch_size])
            
        except Exception as e:
            print(f"\nError in batch {i+1}: {e}")
        
        time.sleep(1)
    
    return all_samples[:count]

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic financial emotion samples')
    parser.add_argument('--emotion', type=str, required=True,
                       choices=['hope', 'fear', 'excitement', 'anxiety', 'optimism', 'uncertainty'],
                       help='Target emotion')
    parser.add_argument('--count', type=int, default=500,
                       help='Number of samples to generate')
    parser.add_argument('--model', type=str, default='gpt-4',
                       choices=['gpt-4', 'claude'],
                       help='LLM model to use')
    parser.add_argument('--api-key', type=str, default=None,
                       help='API key (or set env var)')
    parser.add_argument('--output', type=str, default='../../data/raw/synthetic',
                       help='Output directory')
    
    args = parser.parse_args()
    
    print("="*80)
    print("SYNTHETIC SAMPLE GENERATOR")
    print("="*80)
    print(f"\nEmotion: {args.emotion}")
    print(f"Count: {args.count}")
    print(f"Model: {args.model}")
    
    # Get API key
    if args.api_key:
        api_key = args.api_key
    else:
        import os
        if args.model == 'gpt-4':
            api_key = os.environ.get('OPENAI_API_KEY')
        else:
            api_key = os.environ.get('ANTHROPIC_API_KEY')
    
    if not api_key:
        print(f"\n❌ Error: No API key found")
        print(f"Set {args.model.upper()}_API_KEY environment variable")
        return
    
    # Generate samples
    print(f"\n🤖 Generating {args.count} {args.emotion} samples...")
    
    if args.model == 'gpt-4':
        samples = generate_with_openai(args.emotion, args.count, api_key)
    else:
        samples = generate_with_anthropic(args.emotion, args.count, api_key)
    
    if not samples:
        print("❌ No samples generated")
        return
    
    # Create dataframe
    df = pd.DataFrame({
        'text': samples,
        'emotion': args.emotion,
        'source': f'synthetic_{args.model}',
        'validated': True  # Pre-validated since we specified the emotion
    })
    
    # Save
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'synthetic_{args.emotion}_{timestamp}.csv'
    filepath = output_path / filename
    
    df.to_csv(filepath, index=False)
    
    print("\n" + "="*80)
    print("GENERATION COMPLETE")
    print("="*80)
    print(f"\n💾 Saved to: {filepath}")
    print(f"✅ Generated {len(df)} {args.emotion} samples")
    print(f"\n📊 Sample preview:")
    for i, text in enumerate(df['text'].head(3), 1):
        print(f"   {i}. {text}")
    
    print(f"\n💰 Estimated cost: ${len(samples) * 0.01:.2f}")
    
    print("\n🎯 NEXT STEPS:")
    print(f"1. Review samples in: {filepath}")
    print(f"2. Generate more emotions: python generate_synthetic_samples.py --emotion fear --count 500")
    print(f"3. Merge all: python merge_datasets.py")

if __name__ == '__main__':
    main()

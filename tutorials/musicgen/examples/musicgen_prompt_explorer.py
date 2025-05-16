#!/usr/bin/env python3
# musicgen_prompt_explorer.py
import torch
import os
import argparse
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

def explore_prompts(model_size="small", duration=5.0, output_dir="prompt_samples"):
    """
    Generate music using different prompting techniques.
    
    Args:
        model_size (str): Size of model to use
        duration (float): Length of each sample in seconds
        output_dir (str): Directory to save samples
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define different prompting techniques
    prompts = {
        # Basic genre prompts
        "simple_genre_rock": "Rock music",
        "simple_genre_jazz": "Jazz music",
        "simple_genre_classical": "Classical music",
        
        # Instrument-focused prompts
        "instruments_piano": "Solo piano piece",
        "instruments_guitar": "Acoustic guitar fingerpicking",
        "instruments_orchestra": "Full orchestra with strings and brass",
        
        # Emotional prompts
        "emotion_happy": "Happy and uplifting music",
        "emotion_sad": "Sad and melancholic music",
        "emotion_epic": "Epic and powerful music",
        
        # Detailed descriptive prompts
        "detailed_electronic": "An electronic music track with arpeggiated synthesizers, deep bass, and a driving four-on-the-floor beat",
        "detailed_ambient": "A peaceful ambient soundscape with soft pads, gentle piano notes, and subtle field recordings of a forest",
        "detailed_orchestral": "An orchestral piece with soaring string melodies, dramatic brass swells, and dynamic percussion",
        
        # Style reference prompts
        "style_80s": "1980s synthwave track with retro synthesizers and drum machines",
        "style_baroque": "Baroque classical music in the style of Bach with complex counterpoint",
        "style_jazz_fusion": "Jazz fusion in the style of Weather Report with virtuosic solos and complex chord progressions",
    }
    
    # Determine device
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS (Metal) for generation")
    elif torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA for generation")
    else:
        device = "cpu"
        print("Using CPU for generation (this will be slow)")
    
    # Load model
    print(f"Loading MusicGen {model_size} model...")
    model = MusicGen.get_pretrained(model_size)
    model.to(device)
    
    # Set generation parameters
    model.set_generation_params(
        duration=duration,
        temperature=1.0,
        top_k=250,
        top_p=0.0,
    )
    
    # Generate music for each prompt
    for prompt_name, prompt in prompts.items():
        print(f"Generating '{prompt_name}': {prompt}")
        
        # Generate
        wav = model.generate([prompt])
        
        # Save
        output_path = os.path.join(output_dir, prompt_name)
        audio_write(
            output_path,
            wav[0].cpu(),
            model.sample_rate,
            strategy="loudness"
        )
        
        print(f"Saved to {output_path}.wav")
    
    print("Prompt exploration complete! All samples saved to", output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate music samples using different prompt techniques')
    parser.add_argument('--model', type=str, default='small', choices=['small', 'medium', 'large'], 
                        help='Model size to use (default: small)')
    parser.add_argument('--duration', type=float, default=5.0, 
                        help='Duration in seconds for each sample (default: 5.0)')
    parser.add_argument('--output', type=str, default='prompt_samples', 
                        help='Output directory for samples (default: prompt_samples)')
    
    args = parser.parse_args()
    
    explore_prompts(
        model_size=args.model,
        duration=args.duration,
        output_dir=args.output
    )
#!/usr/bin/env python3
# musicgen_genre_explorer.py
import torch
import os
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

def explore_genres(model_size="medium", duration=10.0, output_dir="genre_samples"):
    """
    Generate samples across different musical genres.
    
    Args:
        model_size (str): Size of model to use
        duration (float): Length of each sample in seconds
        output_dir (str): Directory to save samples
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define genre prompts
    genres = {
        "classical": "A classical orchestra playing a beautiful symphony with strings and woodwinds",
        "jazz": "A jazz quartet with piano, bass, drums, and saxophone playing a smooth jazz piece",
        "rock": "An energetic rock song with electric guitars, bass, and drums",
        "electronic": "An electronic dance music track with synthesizers and a strong beat",
        "ambient": "A peaceful ambient soundscape with soft pads and atmospheric textures",
        "hiphop": "A hip hop beat with boom bap drums and a catchy sample",
        "folk": "An acoustic folk song with guitar and soft vocals",
        "latin": "A latin jazz piece with congas, piano, and brass section",
        "metal": "A heavy metal song with distorted guitars, aggressive drums, and power chords",
        "funk": "A funky groove with slap bass, rhythm guitar, and brass stabs"
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
    
    # Generate music for each genre
    for genre_name, prompt in genres.items():
        print(f"Generating {genre_name} music...")
        
        # Generate
        wav = model.generate([prompt])
        
        # Save
        output_path = os.path.join(output_dir, f"{genre_name}_sample")
        audio_write(
            output_path,
            wav[0].cpu(),
            model.sample_rate,
            strategy="loudness"
        )
        
        print(f"Saved {genre_name} sample to {output_path}.wav")
    
    print("Genre exploration complete! All samples saved to", output_dir)

if __name__ == "__main__":
    explore_genres(model_size="medium", duration=10.0)
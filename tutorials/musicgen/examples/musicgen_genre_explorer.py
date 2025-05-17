#!/usr/bin/env python3
# musicgen_genre_explorer.py - Generate samples across different musical genres
import torch
import os
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

def explore_genres(model_size="medium", duration=10.0, output_dir="genre_samples"):
    """
    Generate audio samples across a diverse range of musical genres using MusicGen.
    
    This script demonstrates MusicGen's versatility across musical styles by
    generating examples of ten different genres from classical to funk. It's useful
    for comparing how the model handles different musical conventions, instrumentation,
    and stylistic elements.
    
    Args:
        model_size (str): Size of MusicGen model to use ("small", "medium", or "large")
                         Larger models provide better quality but use more memory
        duration (float): Length of each sample in seconds (recommended: 8-15 seconds)
        output_dir (str): Directory where all genre samples will be saved
                         
    Returns:
        None: Files are saved to the specified output directory
        
    Note:
        - This can be resource-intensive as it generates 10 audio samples sequentially
        - Runtime depends on hardware, but expect 1-15 minutes for all samples
        - The medium model offers a good balance of quality and speed
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define genre prompts with detailed musical descriptions
    # Each prompt includes instrumentation and stylistic elements specific to the genre
    genres = {
        "classical": "A classical orchestra playing a beautiful symphony with strings, woodwinds, and subtle timpani, with a flowing melody in a major key",
        "jazz": "A jazz quartet with piano, upright bass, brushed drums, and saxophone playing a smooth jazz piece with complex chord progressions and improvisation",
        "rock": "An energetic rock song with distorted electric guitars, bass, drums, and vocals, with a catchy chorus and powerful bridge",
        "electronic": "An electronic dance music track with synthesizers, driving beat at 128 BPM, build-ups, and a euphoric drop with arpeggiated synthesizers",
        "ambient": "A peaceful ambient soundscape with soft synthesizer pads, atmospheric textures, subtle piano notes, and gentle evolution over time",
        "hiphop": "A hip hop beat with boom bap drums, a deep bass line, vinyl crackle, and a sample of an old soul record, with space for vocals",
        "folk": "An acoustic folk song with fingerpicked guitar, soft vocals, gentle harmonica, and subtle string accompaniment telling a story",
        "latin": "A latin jazz piece with congas, timbales, piano montunos, and a brass section playing a lively salsa rhythm with call and response elements",
        "metal": "A heavy metal song with down-tuned distorted guitars, double bass drum patterns, aggressive vocals, and technical guitar solos",
        "funk": "A funky groove with slap bass, clean rhythm guitar playing sixteenth notes, tight drums, brass stabs, and a wah-wah pedal guitar solo"
    }
    
    # Determine the best available compute device
    # MusicGen benefits significantly from GPU acceleration
    if torch.backends.mps.is_available():
        device = "mps"  # Apple Silicon GPU
        print("Using MPS (Metal) for generation")
    elif torch.cuda.is_available():
        device = "cuda"  # NVIDIA GPU
        print("Using CUDA for generation")
    else:
        device = "cpu"  # Fallback to CPU
        print("Using CPU for generation (this will be slow)")
    
    # Load the MusicGen model
    print(f"Loading MusicGen {model_size} model...")
    model = MusicGen.get_pretrained(model_size)
    model.to(device)
    
    # Configure generation parameters
    # We use standard settings that work well across genres
    model.set_generation_params(
        duration=duration,        # Same duration for all samples
        temperature=1.0,          # Standard creativity level
        top_k=250,                # Diverse but controlled sampling
        top_p=0.0,                # Disable nucleus sampling
    )
    
    # Track overall progress
    total_genres = len(genres)
    completed = 0
    
    # Generate a sample for each genre
    print(f"Starting generation of {total_genres} genre samples...")
    for genre_name, prompt in genres.items():
        completed += 1
        print(f"[{completed}/{total_genres}] Generating {genre_name} music...")
        
        # Generate the audio for this genre
        wav = model.generate([prompt])
        
        # Save the audio file with genre-specific name
        output_path = os.path.join(output_dir, f"{genre_name}_sample")
        audio_write(
            output_path,             # Path without extension
            wav[0].cpu(),            # First (and only) audio in batch, moved to CPU
            model.sample_rate,       # Sample rate (32kHz for MusicGen)
            strategy="loudness"      # Normalize for consistent volume across genres
        )
        
        print(f"Saved {genre_name} sample to {output_path}.wav")
    
    print(f"Genre exploration complete! All {total_genres} samples saved to {output_dir}")
    print("Try listening to the samples to compare how MusicGen handles different musical styles.")

if __name__ == "__main__":
    # You can customize these parameters based on your needs
    explore_genres(
        model_size="medium",   # Options: "small", "medium", "large"
        duration=10.0,         # Length in seconds (10s is a good sample length)
        output_dir="genre_samples"  # Output folder for all generated files
    )
    
    # Uncomment to generate longer samples with the large model:
    # explore_genres(model_size="large", duration=15.0, output_dir="genre_samples_large")
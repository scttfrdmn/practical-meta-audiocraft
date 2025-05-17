#!/usr/bin/env python3
# audiogen_basic.py - Basic example of using Meta's AudioGen model for sound generation
import torch
import torchaudio
from audiocraft.models import AudioGen
from audiocraft.data.audio import audio_write
import time
import os

def generate_sound(
    prompt,
    duration=5.0,
    model_size="medium",
    output_dir="sound_output",
    temperature=1.0
):
    """
    Generate environmental sounds and audio effects based on a text description.
    
    This function demonstrates the basic usage of Meta's AudioGen model to create
    realistic sound effects and ambient audio from textual descriptions. AudioGen
    is trained on environmental sounds rather than music, making it ideal for
    creating sound effects, ambient backgrounds, and audio environments.
    
    Args:
        prompt (str): Detailed text description of the sound to generate
                     (e.g., "Rain falling on a tin roof with thunder in the distance")
        duration (float): Length of audio to generate in seconds (recommended: 3-10s)
        model_size (str): Size of model to use - only "medium" or "large" are available
                         (larger provides better quality but uses more memory)
        output_dir (str): Directory where generated audio will be saved
        temperature (float): Controls randomness/variability of generation
                           (higher = more random/creative, lower = more deterministic)
    
    Returns:
        str: Path to the generated audio file (.wav format)
        
    Notes:
        - AudioGen works best with detailed, descriptive prompts about sounds
        - Unlike MusicGen, AudioGen is specialized for non-musical audio
        - The model can generate multiple sounds simultaneously (e.g., "rain and thunder")
        - Higher temperature values (e.g., 1.0-1.2) often produce more interesting variations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating sound for prompt: '{prompt}'")
    print(f"Using model size: {model_size}, duration: {duration}s")
    
    start_time = time.time()
    
    # Determine the best available device for computation
    # AudioGen benefits significantly from GPU acceleration
    if torch.backends.mps.is_available():
        device = "mps"  # Apple Silicon GPU
        print("Using MPS (Metal) for generation")
    elif torch.cuda.is_available():
        device = "cuda"  # NVIDIA GPU
        print("Using CUDA for generation")
    else:
        device = "cpu"  # Fallback to CPU
        print("Using CPU for generation (this will be slow)")
    
    # Load the AudioGen model
    # Unlike MusicGen, AudioGen only has "medium" and "large" sizes
    model = AudioGen.get_pretrained(model_size)
    model.to(device)  # Move model to the appropriate compute device
    
    # Configure generation parameters
    # These settings control how the model generates audio
    model.set_generation_params(
        duration=duration,        # How long the generated audio will be
        temperature=temperature,  # Controls randomness/creativity
        top_k=250,                # Sample from top 250 token predictions
        top_p=0.0,                # Disable nucleus sampling (use top_k instead)
    )
    
    # Generate the sound based on the text prompt
    # This is where the actual generation happens
    wav = model.generate([prompt])  # Model expects a list of prompts
    
    # Create a filesystem-safe filename based on the prompt
    # This truncates long prompts and replaces special characters
    safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt)[:30]
    filename = f"{safe_prompt}_{model_size}_{duration}s"
    
    # Save the generated audio as a WAV file
    # audio_write handles proper formatting and normalization
    output_path = os.path.join(output_dir, filename)
    audio_write(
        output_path,              # Path without extension
        wav[0].cpu(),             # Audio tensor (first in batch)
        model.sample_rate,        # Sample rate (32kHz for AudioGen)
        strategy="loudness",      # Normalize loudness for consistent volume
    )
    
    # Report performance metrics
    generation_time = time.time() - start_time
    print(f"Generation completed in {generation_time:.2f} seconds")
    print(f"Audio saved to {output_path}.wav")
    
    return f"{output_path}.wav"

if __name__ == "__main__":
    # Example prompts demonstrating different types of sounds AudioGen can create
    # Each prompt is detailed and descriptive for best results
    prompts = [
        # Weather and nature sounds
        "Heavy rain falling on a metal roof with occasional thunder in the distance",
        
        # Fire and ambient sounds
        "Crackling campfire with wood popping and occasional owl hoots in the forest",
        
        # Urban environments
        "Busy city street with cars passing, people talking, and sirens in the distance",
        
        # Natural environments
        "Forest ambience with birds chirping, leaves rustling, and a stream flowing nearby",
        
        # Mechanical sounds
        "Old mechanical clock ticking with gears turning and a soft bell chime every few seconds"
    ]
    
    # Generate sound using the first prompt
    # To try different sounds, change the index (0-4) to use a different prompt
    # or modify the prompts list with your own descriptions
    generate_sound(
        prompt=prompts[0],          # Using the first example prompt
        duration=5.0,               # 5 seconds of audio
        model_size="medium",        # Use "medium" or "large" (large = better quality)
        temperature=1.0             # Default creativity level
    )
    
    # Uncomment to generate all example sounds at once:
    # for i, prompt in enumerate(prompts):
    #     print(f"\nGenerating example {i+1}/{len(prompts)}")
    #     generate_sound(prompt=prompt, duration=5.0)
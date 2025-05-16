#!/usr/bin/env python3
# musicgen_basic.py
import torch
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import time
import os

def generate_music(
    prompt,
    duration=10.0,
    model_size="small",
    output_dir="music_output",
    temperature=1.0
):
    """
    Generate music based on a text prompt.
    
    Args:
        prompt (str): Text description of the music to generate
        duration (float): Length of audio in seconds (max 30 seconds)
        model_size (str): Size of model to use ("small", "medium", or "large")
        output_dir (str): Directory to save output files
        temperature (float): Controls randomness (higher = more random)
    
    Returns:
        str: Path to the generated audio file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating music for prompt: '{prompt}'")
    print(f"Using model size: {model_size}, duration: {duration}s")
    
    start_time = time.time()
    
    # Determine device (MPS for Mac, CUDA for NVIDIA, CPU otherwise)
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS (Metal) for generation")
    elif torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA for generation")
    else:
        device = "cpu"
        print("Using CPU for generation (this will be slow)")
    
    # Load the model
    model = MusicGen.get_pretrained(model_size)
    model.to(device)
    
    # Set generation parameters
    model.set_generation_params(
        duration=duration,
        temperature=temperature,
        top_k=250,
        top_p=0.0,
    )
    
    # Generate music
    wav = model.generate([prompt])
    
    # Create a filename based on the prompt
    safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt)[:30]
    filename = f"{safe_prompt}_{model_size}_{duration}s"
    
    # Save the audio file
    output_path = os.path.join(output_dir, filename)
    audio_write(
        output_path, 
        wav[0].cpu(), 
        model.sample_rate, 
        strategy="loudness",
    )
    
    generation_time = time.time() - start_time
    print(f"Generation completed in {generation_time:.2f} seconds")
    print(f"Audio saved to {output_path}.wav")
    
    return f"{output_path}.wav"

if __name__ == "__main__":
    # Example prompts to try
    prompts = [
        "An upbeat electronic dance track with a catchy melody and energetic rhythm",
        "A peaceful piano solo with gentle melodies",
        "An orchestral film score with dramatic strings and brass",
        "A jazz fusion piece with funky bass and smooth saxophone",
        "A lo-fi hip hop beat with relaxing atmosphere"
    ]
    
    # Generate music for the first prompt
    # Change the index to try different prompts
    generate_music(
        prompt=prompts[0],
        duration=10.0,  # 10 seconds
        model_size="small",  # Use "small", "medium", or "large"
        temperature=1.0  # Default creativity level
    )
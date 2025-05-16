#!/usr/bin/env python3
# musicgen_temperature_explorer.py
import torch
import os
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

def explore_temperatures(prompt, model_size="medium", duration=5.0, output_dir="temperature_samples"):
    """
    Generate variations of the same prompt at different temperatures.
    
    Args:
        prompt (str): Text description of music to generate
        model_size (str): Size of model to use
        duration (float): Length of each sample in seconds
        output_dir (str): Directory to save samples
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define temperatures to explore
    temperatures = [0.3, 0.5, 0.7, 1.0, 1.3, 1.6]
    
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
    
    # Generate music at each temperature
    for temp in temperatures:
        print(f"Generating with temperature {temp}...")
        
        # Set generation parameters
        model.set_generation_params(
            duration=duration,
            temperature=temp,
            top_k=250,
            top_p=0.0,
        )
        
        # Generate
        wav = model.generate([prompt])
        
        # Save
        output_path = os.path.join(output_dir, f"temp_{temp:.1f}")
        audio_write(
            output_path,
            wav[0].cpu(),
            model.sample_rate,
            strategy="loudness"
        )
        
        print(f"Saved temperature {temp} sample to {output_path}.wav")
    
    print("Temperature exploration complete! All samples saved to", output_dir)

if __name__ == "__main__":
    # Choose a prompt that works well for temperature exploration
    prompt = "A melodic piano piece with gentle strings in the background"
    explore_temperatures(prompt, model_size="medium", duration=5.0)
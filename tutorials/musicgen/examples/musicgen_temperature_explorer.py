#!/usr/bin/env python3
# musicgen_temperature_explorer.py - Explore how temperature affects music generation
import torch
import os
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

def explore_temperatures(prompt, model_size="medium", duration=5.0, output_dir="temperature_samples"):
    """
    Generate variations of the same prompt at different temperature settings.
    
    This script demonstrates how the temperature parameter affects MusicGen's output.
    Lower temperatures produce more predictable, consistent results that closely follow
    common patterns in the training data. Higher temperatures produce more creative,
    varied, and sometimes experimental results.
    
    Args:
        prompt (str): Text description of music to generate - same prompt used for all variations
        model_size (str): Size of model to use ("small", "medium", or "large")
        duration (float): Length of each sample in seconds
        output_dir (str): Directory to save all temperature variation samples
        
    Returns:
        None: Audio files are saved to the specified output directory
        
    Note:
        Temperature is one of the most important parameters for controlling generation:
        - Very low temp (0.1-0.4): Most deterministic, sometimes repetitive
        - Low temp (0.5-0.7): Consistent and coherent but with some variation
        - Medium temp (0.8-1.2): Good balance of predictability and creativity
        - High temp (1.3-1.8): More experimental, unexpected, and diverse outputs
        - Very high temp (1.9+): Often chaotic, may produce unusual sounds
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define a range of temperatures to explore
    # We use a wide range to clearly demonstrate the effect
    temperatures = [0.3, 0.5, 0.7, 1.0, 1.3, 1.6]
    
    # Determine the best available compute device
    if torch.backends.mps.is_available():
        device = "mps"  # Apple Silicon GPU
        print("Using MPS (Metal) for generation")
    elif torch.cuda.is_available():
        device = "cuda"  # NVIDIA GPU
        print("Using CUDA for generation")
    else:
        device = "cpu"  # Fallback to CPU
        print("Using CPU for generation (this will be slow)")
    
    # Load the model - we only need to do this once for all temperatures
    print(f"Loading MusicGen {model_size} model...")
    model = MusicGen.get_pretrained(model_size)
    model.to(device)
    
    # Save the prompt to a text file for reference
    prompt_file = os.path.join(output_dir, "prompt.txt")
    with open(prompt_file, 'w') as f:
        f.write(f"Prompt: {prompt}\n\n")
        f.write("Temperature values and their effects:\n")
        f.write("- Very low (0.1-0.4): Most deterministic, sometimes repetitive\n")
        f.write("- Low (0.5-0.7): Consistent and coherent but with some variation\n")
        f.write("- Medium (0.8-1.2): Good balance of predictability and creativity\n")
        f.write("- High (1.3-1.8): More experimental, unexpected, and diverse\n")
        f.write("- Very high (1.9+): Often chaotic, may produce unusual sounds\n")
    
    print(f"Using prompt: '{prompt}'")
    print(f"Generating {len(temperatures)} variations with different temperatures...")
    
    # Generate music at each temperature setting
    for i, temp in enumerate(temperatures):
        print(f"[{i+1}/{len(temperatures)}] Generating with temperature {temp:.1f}...")
        
        # Set generation parameters - only temperature changes between runs
        model.set_generation_params(
            duration=duration,       # Same duration for all samples
            temperature=temp,        # This is the parameter we're exploring
            top_k=250,               # Keep top_k consistent for fair comparison
            top_p=0.0,               # Disable nucleus sampling
        )
        
        # Generate audio with this temperature
        wav = model.generate([prompt])
        
        # Save with temperature value in filename
        output_path = os.path.join(output_dir, f"temp_{temp:.1f}")
        audio_write(
            output_path,
            wav[0].cpu(),
            model.sample_rate,
            strategy="loudness"
        )
        
        print(f"Saved temperature {temp:.1f} sample to {output_path}.wav")
    
    # Create a README file with instructions for comparing samples
    readme_file = os.path.join(output_dir, "README.txt")
    with open(readme_file, 'w') as f:
        f.write("TEMPERATURE EXPLORATION SAMPLES\n")
        f.write("==============================\n\n")
        f.write(f"Prompt used: \"{prompt}\"\n\n")
        f.write("How to compare these samples:\n")
        f.write("1. Listen to them in order from lowest temperature to highest\n")
        f.write("2. Note how the music becomes more varied and less predictable\n")
        f.write("3. The sweet spot is often between 0.7-1.3 for most applications\n")
        f.write("4. For creative exploration, try temperatures above 1.3\n")
        f.write("5. For consistent, reliable outputs, use temperatures below 0.7\n")
    
    print(f"\nTemperature exploration complete! All samples saved to {output_dir}")
    print("Listen to the samples in sequence to hear how temperature affects generation.")
    print("Check the README.txt file for tips on how to compare the samples.")

if __name__ == "__main__":
    # Choose a prompt that works well for temperature exploration
    # Simpler prompts often show temperature differences more clearly
    prompt = "A melodic piano piece with gentle strings in the background"
    
    explore_temperatures(
        prompt=prompt,
        model_size="medium",         # "medium" is a good balance of quality and speed
        duration=5.0,                # 5 seconds is enough to hear the differences
        output_dir="temperature_samples"  # Where to save the output files
    )
    
    # Other interesting prompts to try (uncomment to use):
    
    # Electronic music (shows rhythm/beat variations well):
    # prompt = "An electronic music track with a strong beat and synthesizer melody"
    # explore_temperatures(prompt, output_dir="temp_electronic")
    
    # Orchestral (shows compositional complexity differences):
    # prompt = "An orchestral piece with dramatic strings and brass"
    # explore_temperatures(prompt, output_dir="temp_orchestral")
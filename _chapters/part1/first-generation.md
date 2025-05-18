---
layout: chapter
title: "Chapter 4: Your First Audio Generation"
# Copyright © 2025 Scott Friedman.
# Licensed under Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.
difficulty: beginner
estimated_time: 2 hours
scenario:
  quote: "I've set up AudioCraft successfully, and I understand the basic concepts. Now I'm ready to actually generate some audio! I want to create a simple example that works reliably, learn how to save and play the results, and start experimenting with different prompts and settings."
  persona: "Sam Patel"
  role: "Digital Media Student"
next_steps:
  - title: "Basic Music Generation"
    url: "/chapters/part2/basic-music/"
    description: "Dive deeper into music generation with MusicGen"
  - title: "Basic Sound Effect Generation"
    url: "/chapters/part3/basic-sound-effects/"
    description: "Create sound effects and environments with AudioGen"
  - title: "Prompt Engineering for Music"
    url: "/chapters/part2/prompt-engineering/"
    description: "Learn techniques for crafting effective music prompts"
further_reading:
  - title: "AudioCraft GitHub Examples"
    url: "https://github.com/facebookresearch/audiocraft/tree/main/examples"
  - title: "PyTorch Audio Documentation"
    url: "https://pytorch.org/audio/stable/index.html"
  - title: "IPython Audio Playback"
    url: "https://ipython.readthedocs.io/en/stable/api/generated/IPython.display.html#IPython.display.Audio"
---

# Chapter 4: Your First Audio Generation

## The Challenge

You've installed AudioCraft and understand its architecture in theory, but how do you actually use it to generate audio? Creating your first successful generation requires bringing together everything we've learned so far: setting up the right environment, loading the appropriate model, crafting an effective prompt, configuring generation parameters, and handling the output.

Many beginners struggle with this first step - they may encounter errors when loading models, generate audio that doesn't match their expectations, or be unsure how to save and play the generated content. Getting this foundational workflow right is crucial before moving on to more complex applications.

In this chapter, we'll create a complete, working audio generation script, explaining each step along the way. We'll explore both MusicGen and AudioGen to generate different types of audio, and establish a solid foundation for your future audio generation projects.

## Learning Objectives

By the end of this chapter, you'll be able to:

- Create a complete script for generating audio with AudioCraft
- Generate both music and sound effects from text descriptions
- Configure generation parameters for different results
- Save generated audio in various formats
- Play generated audio directly in different environments
- Implement basic error handling for robust generation
- Begin experimenting with your own prompts and settings

## Your First Complete Generation Script

Let's dive right in with a complete script for generating music, which we'll then break down step by step:

```python
# first_generation.py - Your first AudioCraft generation script
import torch
import os
import time
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

def generate_music(
    prompt,
    duration=10.0,
    model_size="small",
    output_dir="generated_audio",
    temperature=1.0,
    top_k=250,
    seed=None
):
    """
    Generate music using AudioCraft's MusicGen model.
    
    Args:
        prompt (str): Text description of the music to generate
        duration (float): Length of audio in seconds (max 30)
        model_size (str): Size of model to use ('small', 'medium', or 'large')
        output_dir (str): Directory to save the generated audio
        temperature (float): Controls randomness (0.5-1.5 recommended)
        top_k (int): Controls diversity via top-k sampling
        seed (int, optional): Random seed for reproducible generation
    
    Returns:
        str: Path to the generated audio file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating music for prompt: '{prompt}'")
    start_time = time.time()
    
    # Set random seed if provided (for reproducibility)
    if seed is not None:
        torch.manual_seed(seed)
        print(f"Using random seed: {seed}")
    
    # Determine the appropriate device
    if torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA for generation")
    elif torch.backends.mps.is_available():
        device = "mps"  # Apple Silicon
        print("Using MPS (Metal) for generation")
    else:
        device = "cpu"
        print("Using CPU for generation (this will be slower)")
    
    try:
        # Load the model
        print(f"Loading MusicGen {model_size} model...")
        model = MusicGen.get_pretrained(model_size)
        model.to(device)
        
        # Set generation parameters
        model.set_generation_params(
            duration=duration,
            temperature=temperature,
            top_k=top_k,
            top_p=0.0,  # Disable nucleus sampling
            cfg_coef=3.0  # Classifier-free guidance coefficient
        )
        
        # Generate the audio
        print("Generating audio...")
        wav = model.generate([prompt])  # Batch of 1 prompt
        
        # Create a filename based on the prompt
        safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt)[:30]
        filename = f"music_{safe_prompt}_{model_size}"
        
        # Save the audio file
        output_path = os.path.join(output_dir, filename)
        audio_write(
            output_path,          # Path without extension
            wav[0].cpu(),         # First (and only) audio in batch
            model.sample_rate,    # Sample rate (typically 32kHz)
            strategy="loudness",  # Normalize audio levels
        )
        
        generation_time = time.time() - start_time
        print(f"Generation completed in {generation_time:.2f} seconds")
        print(f"Audio saved to {output_path}.wav")
        
        return f"{output_path}.wav"
        
    except Exception as e:
        print(f"Error generating audio: {str(e)}")
        return None
    finally:
        # Clean up to free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    # Example prompts to try
    example_prompts = [
        "An upbeat electronic dance track with a catchy melody and energetic rhythm",
        "A peaceful piano piece with gentle strings in the background",
        "A rock anthem with powerful electric guitar and driving drums",
        "A jazz fusion track with complex chord progressions and saxophone solo",
        "An ambient soundscape with ethereal pads and subtle textures"
    ]
    
    # Choose the first prompt (change the index to try different prompts)
    prompt_index = 0
    chosen_prompt = example_prompts[prompt_index]
    
    # Generate music with the chosen prompt
    output_file = generate_music(
        prompt=chosen_prompt,
        duration=10.0,  # 10 seconds
        model_size="small",  # Use "small", "medium", or "large"
        temperature=1.0  # Standard creativity level
    )
    
    if output_file:
        print(f"Success! Generated audio saved to: {output_file}")
        
        # Provide instructions for playing the audio
        print("\nTo play the generated audio:")
        print("1. Use a media player to open the WAV file")
        print("2. Or in Python, you can use:")
        print("   from IPython.display import Audio")
        print(f"   Audio(filename='{output_file}')")
```

Save this script as `first_generation.py` and run it from your command line:

```bash
python first_generation.py
```

If everything is set up correctly, it will generate a 10-second music clip based on the first example prompt and save it to the `generated_audio` directory.

## Breaking Down the Script

Let's understand each component of our generation script:

### 1. Imports and Dependencies

```python
import torch
import os
import time
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
```

These imports provide:
- `torch`: PyTorch for tensor operations and device management
- `os`: Operating system utilities for file handling
- `time`: Time measurement for performance tracking
- `MusicGen`: The music generation model from AudioCraft
- `audio_write`: Utility for saving audio files with proper formatting

### 2. The Main Generation Function

```python
def generate_music(
    prompt,
    duration=10.0,
    model_size="small",
    output_dir="generated_audio",
    temperature=1.0,
    top_k=250,
    seed=None
):
```

This function encapsulates the complete generation process with configurable parameters:
- `prompt`: Text description of the music to generate
- `duration`: Length in seconds (max 30 seconds per generation)
- `model_size`: Model size ('small', 'medium', or 'large')
- `output_dir`: Directory to save generated audio
- `temperature`: Controls randomness/creativity 
- `top_k`: Controls diversity via top-k sampling
- `seed`: Optional random seed for reproducible results

### 3. Device Selection

```python
# Determine the appropriate device
if torch.cuda.is_available():
    device = "cuda"
    print("Using CUDA for generation")
elif torch.backends.mps.is_available():
    device = "mps"  # Apple Silicon
    print("Using MPS (Metal) for generation")
else:
    device = "cpu"
    print("Using CPU for generation (this will be slower)")
```

This code automatically selects the best available hardware:
- NVIDIA GPUs with CUDA
- Apple Silicon with Metal
- CPU as a fallback

The device selection is critical for performance - generation can be 10-20x faster on GPUs compared to CPUs.

### 4. Loading the Model

```python
# Load the model
print(f"Loading MusicGen {model_size} model...")
model = MusicGen.get_pretrained(model_size)
model.to(device)
```

This loads the pre-trained model of the specified size and moves it to the appropriate device. The first time you run this, it will download the model weights from the internet.

### 5. Setting Generation Parameters

```python
# Set generation parameters
model.set_generation_params(
    duration=duration,
    temperature=temperature,
    top_k=top_k,
    top_p=0.0,  # Disable nucleus sampling
    cfg_coef=3.0  # Classifier-free guidance coefficient
)
```

These parameters control the generation process:
- `duration`: Length of audio to generate
- `temperature`: Randomness in the generation (higher = more random)
- `top_k`: Number of most likely tokens to consider
- `top_p`: Nucleus sampling threshold (0.0 disables it)
- `cfg_coef`: How closely to follow the text prompt

### 6. Generating the Audio

```python
# Generate the audio
print("Generating audio...")
wav = model.generate([prompt])  # Batch of 1 prompt
```

This is where the magic happens! The model takes our prompt and generates audio that matches the description. Note that `generate()` expects a list of prompts, even if we're only using one.

### 7. Saving the Output

```python
# Create a filename based on the prompt
safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt)[:30]
filename = f"music_{safe_prompt}_{model_size}"

# Save the audio file
output_path = os.path.join(output_dir, filename)
audio_write(
    output_path,          # Path without extension
    wav[0].cpu(),         # First (and only) audio in batch
    model.sample_rate,    # Sample rate (typically 32kHz)
    strategy="loudness",  # Normalize audio levels
)
```

This code:
1. Creates a safe filename from the prompt (removing special characters)
2. Uses `audio_write()` to save the audio as a WAV file
3. Moves the tensor from GPU to CPU before saving
4. Uses loudness normalization for consistent volume

### 8. Memory Management

```python
finally:
    # Clean up to free memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

This cleanup step is important to free GPU memory after generation, especially if you're running multiple generations or working with limited resources.

## Your First Sound Effect Generation

Now let's adapt our script to generate sound effects with AudioGen:

```python
# first_sound_effect.py - Generate sound effects with AudioGen
import torch
import os
import time
from audiocraft.models import AudioGen  # Changed from MusicGen
from audiocraft.data.audio import audio_write

def generate_sound_effect(
    prompt,
    duration=5.0,
    model_size="medium",  # AudioGen only has 'medium' and 'large'
    output_dir="generated_audio",
    temperature=1.0,
    top_k=250,
    seed=None
):
    """
    Generate sound effects using AudioCraft's AudioGen model.
    
    Args:
        prompt (str): Text description of the sound to generate
        duration (float): Length of audio in seconds (max 30)
        model_size (str): Size of model to use ('medium' or 'large')
        output_dir (str): Directory to save the generated audio
        temperature (float): Controls randomness (0.5-1.5 recommended)
        top_k (int): Controls diversity via top-k sampling
        seed (int, optional): Random seed for reproducible generation
    
    Returns:
        str: Path to the generated audio file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating sound effect for prompt: '{prompt}'")
    start_time = time.time()
    
    # Set random seed if provided (for reproducibility)
    if seed is not None:
        torch.manual_seed(seed)
        print(f"Using random seed: {seed}")
    
    # Determine the appropriate device
    if torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA for generation")
    elif torch.backends.mps.is_available():
        device = "mps"  # Apple Silicon
        print("Using MPS (Metal) for generation")
    else:
        device = "cpu"
        print("Using CPU for generation (this will be slower)")
    
    try:
        # Load the model
        print(f"Loading AudioGen {model_size} model...")
        model = AudioGen.get_pretrained(model_size)
        model.to(device)
        
        # Set generation parameters
        model.set_generation_params(
            duration=duration,
            temperature=temperature,
            top_k=top_k,
            top_p=0.0,  # Disable nucleus sampling
            cfg_coef=3.0  # Classifier-free guidance coefficient
        )
        
        # Generate the audio
        print("Generating audio...")
        wav = model.generate([prompt])  # Batch of 1 prompt
        
        # Create a filename based on the prompt
        safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt)[:30]
        filename = f"sound_{safe_prompt}_{model_size}"
        
        # Save the audio file
        output_path = os.path.join(output_dir, filename)
        audio_write(
            output_path,          # Path without extension
            wav[0].cpu(),         # First (and only) audio in batch
            model.sample_rate,    # Sample rate (typically 32kHz)
            strategy="loudness",  # Normalize audio levels
        )
        
        generation_time = time.time() - start_time
        print(f"Generation completed in {generation_time:.2f} seconds")
        print(f"Audio saved to {output_path}.wav")
        
        return f"{output_path}.wav"
        
    except Exception as e:
        print(f"Error generating audio: {str(e)}")
        return None
    finally:
        # Clean up to free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    # Example prompts to try - tailored for sound effects
    example_prompts = [
        "Heavy rain falling on a metal roof with occasional thunder in the distance",
        "Crackling campfire with wood popping and occasional owl hoots",
        "Busy city street with cars passing, people talking, and sirens in the distance",
        "Forest ambience with birds chirping, leaves rustling, and a stream flowing",
        "Old mechanical clock ticking with gears turning and a soft bell chime"
    ]
    
    # Choose the first prompt (change the index to try different prompts)
    prompt_index = 0
    chosen_prompt = example_prompts[prompt_index]
    
    # Generate sound effect with the chosen prompt
    output_file = generate_sound_effect(
        prompt=chosen_prompt,
        duration=5.0,  # 5 seconds
        model_size="medium",  # Use "medium" or "large"
        temperature=1.0  # Standard creativity level
    )
    
    if output_file:
        print(f"Success! Generated audio saved to: {output_file}")
```

The key differences in this script are:
1. We import `AudioGen` instead of `MusicGen`
2. AudioGen only offers 'medium' and 'large' model sizes (no 'small')
3. Default duration is shorter (5s vs 10s) as sound effects are often briefer
4. Example prompts are focused on sound effects rather than music
5. Filename prefix is 'sound_' instead of 'music_'

Save this as `first_sound_effect.py` and run it to generate your first sound effect.

## Playing Generated Audio

Once you've generated audio, you'll want to listen to it. Here are several methods for playing your generated audio:

### 1. Using External Media Players

The simplest approach is to open the generated WAV file with your system's media player:
- Windows: Windows Media Player, VLC
- macOS: QuickTime, VLC
- Linux: VLC, mpv, etc.

### 2. Playing Audio in Python Scripts

You can add audio playback directly to your scripts using libraries like `playsound`:

```python
# Install with: pip install playsound
from playsound import playsound

# After generating audio
playsound(output_file)
```

### 3. Playing Audio in Jupyter Notebooks

For Jupyter notebook users, IPython provides a convenient Audio widget:

```python
from IPython.display import Audio

# After generating audio
Audio(filename=output_file)
```

This creates an interactive audio player directly in your notebook.

### 4. Creating a Simple Playback Script

Here's a utility script to play any audio file:

```python
# play_audio.py - Simple audio player for generated files
import sys
import os
import argparse
from playsound import playsound

def play_audio(file_path):
    """Play an audio file."""
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return False
    
    print(f"Playing {file_path}...")
    playsound(file_path)
    print("Playback complete.")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play audio files")
    parser.add_argument("file", help="Path to the audio file to play")
    args = parser.parse_args()
    
    play_audio(args.file)
```

Use it like this: `python play_audio.py generated_audio/your_file.wav`

## Customizing Your Generations

Now that you have the basic generation working, let's explore how to customize your results by adjusting parameters and prompts.

### Adjusting Generation Parameters

```python
# Parameter experimentation example
def explore_parameters(prompt, output_dir="parameter_exploration"):
    """Generate variations of audio using different parameters."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Template for parameter description
    def param_name(temp, topk):
        return f"temp{temp}_topk{topk}"
    
    # List of parameters to try
    temperatures = [0.5, 1.0, 1.5]
    top_ks = [50, 250, 500]
    
    # Generate with each parameter combination
    results = {}
    for temp in temperatures:
        for topk in top_ks:
            print(f"\nGenerating with temperature={temp}, top_k={topk}")
            
            output_file = generate_music(
                prompt=prompt,
                temperature=temp,
                top_k=topk,
                output_dir=output_dir,
                # Use parameter values in filename
                filename_prefix=param_name(temp, topk)
            )
            
            results[param_name(temp, topk)] = output_file
            
    return results
```

This function generates audio with different temperature and top-k combinations, allowing you to compare how parameters affect the output.

### Experiment with Different Prompts

```python
# Prompt experimentation example
def explore_prompts(prompts, output_dir="prompt_exploration"):
    """Generate audio using different variations of a base prompt."""
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    for i, prompt in enumerate(prompts):
        print(f"\nPrompt {i+1}/{len(prompts)}: {prompt}")
        
        output_file = generate_music(
            prompt=prompt,
            output_dir=output_dir,
            # Use consistent parameters
            temperature=1.0,
            top_k=250
        )
        
        results[f"prompt_{i+1}"] = output_file
    
    return results
```

This function lets you compare how different prompt formulations affect the generated audio.

## Creating a Unified Generator

Let's create a more versatile script that can generate both music and sound effects:

```python
# unified_generator.py - Generate both music and sound effects
import torch
import os
import time
import argparse
from audiocraft.models import MusicGen, AudioGen
from audiocraft.data.audio import audio_write

def generate_audio(
    prompt,
    model_type="music",  # "music" or "sound"
    duration=None,  # Default depends on model_type
    model_size=None,  # Default depends on model_type
    output_dir="generated_audio",
    temperature=1.0,
    top_k=250,
    seed=None
):
    """
    Generate audio using AudioCraft models.
    
    Args:
        prompt (str): Text description of the audio to generate
        model_type (str): Type of model to use ("music" or "sound")
        duration (float, optional): Length of audio in seconds
        model_size (str, optional): Size of model to use
        output_dir (str): Directory to save the generated audio
        temperature (float): Controls randomness (0.5-1.5 recommended)
        top_k (int): Controls diversity via top-k sampling
        seed (int, optional): Random seed for reproducible generation
    
    Returns:
        str: Path to the generated audio file
    """
    # Set defaults based on model_type
    if model_type not in ["music", "sound"]:
        raise ValueError("model_type must be 'music' or 'sound'")
    
    # Default duration
    if duration is None:
        duration = 10.0 if model_type == "music" else 5.0
    
    # Default model size
    if model_size is None:
        model_size = "small" if model_type == "music" else "medium"
    
    # Validate model size
    if model_type == "music" and model_size not in ["small", "medium", "large"]:
        raise ValueError("For music, model_size must be 'small', 'medium', or 'large'")
    if model_type == "sound" and model_size not in ["medium", "large"]:
        raise ValueError("For sound effects, model_size must be 'medium' or 'large'")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating {model_type} for prompt: '{prompt}'")
    start_time = time.time()
    
    # Set random seed if provided (for reproducibility)
    if seed is not None:
        torch.manual_seed(seed)
        print(f"Using random seed: {seed}")
    
    # Determine the appropriate device
    if torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA for generation")
    elif torch.backends.mps.is_available():
        device = "mps"  # Apple Silicon
        print("Using MPS (Metal) for generation")
    else:
        device = "cpu"
        print("Using CPU for generation (this will be slower)")
    
    try:
        # Load the appropriate model
        print(f"Loading {'MusicGen' if model_type == 'music' else 'AudioGen'} {model_size} model...")
        
        if model_type == "music":
            model = MusicGen.get_pretrained(model_size)
        else:  # model_type == "sound"
            model = AudioGen.get_pretrained(model_size)
            
        model.to(device)
        
        # Set generation parameters
        model.set_generation_params(
            duration=duration,
            temperature=temperature,
            top_k=top_k,
            top_p=0.0,  # Disable nucleus sampling
            cfg_coef=3.0  # Classifier-free guidance coefficient
        )
        
        # Generate the audio
        print("Generating audio...")
        wav = model.generate([prompt])  # Batch of 1 prompt
        
        # Create a filename based on the prompt
        safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt)[:30]
        filename = f"{model_type}_{safe_prompt}_{model_size}"
        
        # Save the audio file
        output_path = os.path.join(output_dir, filename)
        audio_write(
            output_path,          # Path without extension
            wav[0].cpu(),         # First (and only) audio in batch
            model.sample_rate,    # Sample rate (typically 32kHz)
            strategy="loudness",  # Normalize audio levels
        )
        
        generation_time = time.time() - start_time
        print(f"Generation completed in {generation_time:.2f} seconds")
        print(f"Audio saved to {output_path}.wav")
        
        return f"{output_path}.wav"
        
    except Exception as e:
        print(f"Error generating audio: {str(e)}")
        return None
    finally:
        # Clean up to free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Generate music or sound effects with AudioCraft")
    parser.add_argument("--type", choices=["music", "sound"], default="music",
                        help="Type of audio to generate (music or sound)")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Text description of the audio to generate")
    parser.add_argument("--duration", type=float, default=None,
                        help="Length of audio in seconds")
    parser.add_argument("--model", choices=["small", "medium", "large"], default=None,
                        help="Model size to use")
    parser.add_argument("--temp", type=float, default=1.0,
                        help="Temperature parameter (0.5-1.5 recommended)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible generation")
    parser.add_argument("--output", type=str, default="generated_audio",
                        help="Output directory for generated audio")
    
    args = parser.parse_args()
    
    # Generate audio based on command-line arguments
    output_file = generate_audio(
        prompt=args.prompt,
        model_type=args.type,
        duration=args.duration,
        model_size=args.model,
        temperature=args.temp,
        seed=args.seed,
        output_dir=args.output
    )
    
    if output_file:
        print(f"Success! Generated audio saved to: {output_file}")
```

This unified script can be used from the command line like this:

```bash
# Generate music
python unified_generator.py --type music --prompt "An upbeat electronic track with a catchy melody" --duration 15.0 --model medium

# Generate a sound effect
python unified_generator.py --type sound --prompt "Heavy rain falling on a metal roof with thunder" --duration 8.0 --model medium
```

## Common Issues and Troubleshooting

When generating your first audio, you might encounter some common issues. Here's how to troubleshoot them:

### Out of Memory Errors

**Problem**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Use a smaller model:
   ```python
   model = MusicGen.get_pretrained('small')  # Instead of 'medium' or 'large'
   ```

2. Generate shorter audio:
   ```python
   model.set_generation_params(duration=5.0)  # Instead of longer durations
   ```

3. Clear cache between generations:
   ```python
   torch.cuda.empty_cache()
   ```

### Slow Generation

**Problem**: Generation takes a very long time.

**Solutions**:
1. Check if you're actually using GPU acceleration:
   ```python
   print(f"Using device: {device}")
   ```

2. For first-time runs, note that model downloading adds time:
   ```python
   # The first run includes download time
   # Subsequent runs will be faster
   ```

3. On CPU, use the smallest model possible:
   ```python
   model = MusicGen.get_pretrained('small')
   ```

### Disappointing Audio Quality

**Problem**: Generated audio doesn't match expectations.

**Solutions**:
1. Improve your prompt with more details:
   ```python
   # Instead of:
   prompt = "Electronic music"
   
   # Try:
   prompt = "An upbeat electronic track with a catchy synth melody, driving bass, and energetic drum patterns"
   ```

2. Try a larger model:
   ```python
   model = MusicGen.get_pretrained('large')  # Better quality but more resources
   ```

3. Experiment with temperature:
   ```python
   # For more varied/creative results:
   model.set_generation_params(temperature=1.3)
   
   # For more consistent/focused results:
   model.set_generation_params(temperature=0.7)
   ```

## A Complete Example: Interactive Audio Generation

Let's finish with a more interactive example that lets you experiment with different settings:

```python
# interactive_generator.py - Interactive audio generation
import torch
import os
import time
import argparse
from audiocraft.models import MusicGen, AudioGen
from audiocraft.data.audio import audio_write

def generate_audio_interactive():
    """Run an interactive session for generating audio."""
    print("\n=== AudioCraft Interactive Generator ===\n")
    
    # Get model type
    while True:
        model_type = input("Generate [m]usic or [s]ound effects? (m/s): ").lower()
        if model_type in ['m', 'music']:
            model_type = "music"
            break
        elif model_type in ['s', 'sound']:
            model_type = "sound"
            break
        print("Please enter 'm' for music or 's' for sound effects.")
    
    # Get model size
    valid_sizes = ["small", "medium", "large"] if model_type == "music" else ["medium", "large"]
    while True:
        print(f"\nAvailable model sizes for {model_type}: {', '.join(valid_sizes)}")
        model_size = input(f"Model size [{valid_sizes[0]}]: ").lower() or valid_sizes[0]
        if model_size in valid_sizes:
            break
        print(f"Please enter a valid model size: {', '.join(valid_sizes)}")
    
    # Get prompt
    prompt = input("\nEnter a description of the audio to generate: ")
    while not prompt.strip():
        prompt = input("Prompt cannot be empty. Please enter a description: ")
    
    # Get duration
    while True:
        duration_input = input("\nDuration in seconds (1-30) [10]: ") or "10"
        try:
            duration = float(duration_input)
            if 1 <= duration <= 30:
                break
            print("Duration must be between 1 and 30 seconds.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Get temperature
    while True:
        temp_input = input("\nTemperature (0.1-2.0) [1.0]: ") or "1.0"
        try:
            temperature = float(temp_input)
            if 0.1 <= temperature <= 2.0:
                break
            print("Temperature must be between 0.1 and 2.0.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Ask about random seed
    use_seed = input("\nUse random seed for reproducibility? (y/n) [n]: ").lower() == 'y'
    seed = None
    if use_seed:
        while True:
            seed_input = input("Enter seed (integer) [42]: ") or "42"
            try:
                seed = int(seed_input)
                break
            except ValueError:
                print("Please enter a valid integer.")
    
    # Set up output directory
    output_dir = input("\nOutput directory [generated_audio]: ") or "generated_audio"
    
    # Determine the appropriate device
    if torch.cuda.is_available():
        device = "cuda"
        print("\nUsing CUDA for generation")
    elif torch.backends.mps.is_available():
        device = "mps"  # Apple Silicon
        print("\nUsing MPS (Metal) for generation")
    else:
        device = "cpu"
        print("\nUsing CPU for generation (this will be slower)")
    
    # Generate the audio
    print(f"\nGenerating {model_type} with the following parameters:")
    print(f"- Prompt: '{prompt}'")
    print(f"- Model: {model_size}")
    print(f"- Duration: {duration} seconds")
    print(f"- Temperature: {temperature}")
    print(f"- Seed: {seed if seed is not None else 'None (random)'}")
    print(f"- Output directory: {output_dir}")
    
    start = input("\nPress Enter to start generation...")
    
    start_time = time.time()
    
    try:
        # Set seed if provided
        if seed is not None:
            torch.manual_seed(seed)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load the appropriate model
        print(f"Loading {'MusicGen' if model_type == 'music' else 'AudioGen'} {model_size} model...")
        
        if model_type == "music":
            model = MusicGen.get_pretrained(model_size)
        else:  # model_type == "sound"
            model = AudioGen.get_pretrained(model_size)
            
        model.to(device)
        
        # Set generation parameters
        model.set_generation_params(
            duration=duration,
            temperature=temperature,
            top_k=250,
            top_p=0.0,
            cfg_coef=3.0
        )
        
        # Generate the audio
        print("Generating audio...")
        wav = model.generate([prompt])
        
        # Create a filename based on the prompt
        safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt)[:30]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{model_type}_{safe_prompt}_{model_size}_{timestamp}"
        
        # Save the audio file
        output_path = os.path.join(output_dir, filename)
        audio_write(
            output_path,
            wav[0].cpu(),
            model.sample_rate,
            strategy="loudness",
        )
        
        generation_time = time.time() - start_time
        print(f"\nGeneration completed in {generation_time:.2f} seconds")
        print(f"Audio saved to {output_path}.wav")
        
        # Ask about playing the audio
        try_play = input("\nWould you like to try playing the audio? (y/n) [y]: ").lower() != 'n'
        if try_play:
            try:
                # Try different methods to play audio
                try:
                    from IPython.display import Audio, display
                    print("Playing audio with IPython...")
                    display(Audio(f"{output_path}.wav"))
                    print("If you don't hear anything, your environment may not support IPython audio playback.")
                except (ImportError, TypeError):
                    try:
                        from playsound import playsound
                        print("Playing audio with playsound...")
                        playsound(f"{output_path}.wav")
                    except ImportError:
                        print("\nCouldn't play audio automatically.")
                        print(f"Please open the file manually: {output_path}.wav")
            except Exception as e:
                print(f"Error playing audio: {str(e)}")
                print(f"Please open the file manually: {output_path}.wav")
        
        # Ask about generating another
        another = input("\nGenerate another? (y/n) [y]: ").lower() != 'n'
        if another:
            # Clean up to free memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True
        else:
            print("\nThank you for using AudioCraft Interactive Generator!")
            return False
        
    except Exception as e:
        print(f"\nError generating audio: {str(e)}")
        
        # Ask about trying again
        try_again = input("\nWould you like to try again? (y/n) [y]: ").lower() != 'n'
        if try_again:
            # Clean up to free memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True
        else:
            print("\nExiting AudioCraft Interactive Generator.")
            return False
    finally:
        # Clean up to free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    # Run interactive session in a loop
    continue_session = True
    while continue_session:
        continue_session = generate_audio_interactive()
```

This interactive script gives you a user-friendly way to experiment with different settings and quickly iterate on your audio generation.

## Hands-on Challenge

Now it's your turn to experiment with what you've learned. Try the following challenge:

### Challenge: Theme-Based Audio Generator

Create a script that:
1. Takes a theme (e.g., "nature", "urban", "space") as input
2. Generates multiple audio files (both music and sound effects) related to that theme
3. Creates a simple HTML page that displays all the generated audio with playback controls
4. Uses different generation parameters for each audio piece to create variety

This challenge will reinforce your understanding of audio generation while creating a useful tool for theme-based projects.

## Key Takeaways

- AudioCraft provides two main models: MusicGen for music and AudioGen for sound effects
- The basic workflow is similar for both: load model, set parameters, generate, save
- Generation parameters like temperature and top-k significantly affect the output
- Hardware acceleration (GPU) makes generation much faster
- Detailed, specific prompts produce better results than vague ones
- Error handling and memory management are important for robust generation

## Next Steps

Now that you've successfully generated your first audio with AudioCraft, you're ready to explore more specialized topics:

- [Basic Music Generation](/chapters/part2/basic-music/): Dive deeper into music generation with MusicGen
- [Basic Sound Effect Generation](/chapters/part3/basic-sound-effects/): Create sound effects and environments with AudioGen
- [Prompt Engineering for Music](/chapters/part2/prompt-engineering/): Learn techniques for crafting effective music prompts

## Further Reading

- [AudioCraft GitHub Examples](https://github.com/facebookresearch/audiocraft/tree/main/examples)
- [PyTorch Audio Documentation](https://pytorch.org/audio/stable/index.html)
- [IPython Audio Playback](https://ipython.readthedocs.io/en/stable/api/generated/IPython.display.html#IPython.display.Audio)
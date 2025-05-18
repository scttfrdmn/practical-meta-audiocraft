---
layout: chapter
title: "Chapter X: [Chapter Title]"
# Copyright © 2025 Scott Friedman.
# Licensed under Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.
difficulty: beginner|intermediate|advanced
estimated_time: X hours
---

> "I need to create [specific audio type] for my [application/project], but I'm not sure how to generate audio that sounds [desired quality]." — *[Persona Name], [Persona Role]*

# Chapter X: [Chapter Title]

## The Challenge

[Real-world problem description in 2-3 paragraphs. Explain the challenges that practitioners face when trying to accomplish this audio generation task.]

In this chapter, you'll learn how to [main learning objective] by [approach]. We'll walk through the entire process from [starting point] to [end result], with practical examples you can run on your own system.

## Learning Objectives

By the end of this chapter, you'll be able to:

- [Specific skill or knowledge #1]
- [Specific skill or knowledge #2]
- [Specific skill or knowledge #3]
- [Specific skill or knowledge #4]

## Prerequisites

Before proceeding, ensure you have:
- [Prerequisite #1]
- [Prerequisite #2]
- [Prerequisite #3]

## Key Concepts

### [Concept Name]

[Explain the key concept in 2-3 paragraphs with analogies and clear language. Connect it to the chapter's learning objectives.]

```python
# Simple example illustrating the concept
```

### [Another Concept Name]

[Explain another key concept in 2-3 paragraphs with analogies and clear language. Connect it to the chapter's learning objectives.]

```python
# Simple example illustrating the concept
```

## Solution Walkthrough

### 1. [First Step]

Let's begin by [first step description]. This is crucial because [reason].

```python
# Code for first step with detailed comments
import torch
from audiocraft.models import MusicGen  # Import the model we'll be using

# Set up our environment
if torch.backends.mps.is_available():
    device = "mps"  # Apple Silicon GPU
    print("Using MPS (Metal) for generation")
elif torch.cuda.is_available():
    device = "cuda"  # NVIDIA GPU
    print("Using CUDA for generation")
else:
    device = "cpu"  # Fallback to CPU
    print("Using CPU for generation (this will be slow)")
```

### 2. [Second Step]

Now that we've [first step], we can [second step]. This involves [explanation].

```python
# Code for second step with detailed comments
# Load the model
model = MusicGen.get_pretrained('small')
model.to(device)  # Move model to the appropriate device

# Configure generation parameters
model.set_generation_params(
    duration=10.0,       # Length of audio to generate in seconds
    temperature=1.0,     # Controls randomness/creativity
    top_k=250,           # Limits sampling to top 250 token predictions
    top_p=0.0,           # Disables nucleus sampling
)
```

### 3. [Third Step]

With [previous steps] in place, we can now [third step]. This is where [explanation].

```python
# Code for third step with detailed comments
# Define our prompt
prompt = "An upbeat electronic track with a catchy melody and energetic rhythm"

# Generate the audio
wav = model.generate([prompt])  # Note: model.generate expects a list of prompts
```

### 4. [Fourth Step]

Finally, we [fourth step] to [complete the solution].

```python
# Code for fourth step with detailed comments
from audiocraft.data.audio import audio_write
import os

# Create output directory
output_dir = "music_output"
os.makedirs(output_dir, exist_ok=True)

# Save the generated audio
output_path = os.path.join(output_dir, "my_generated_music")
audio_write(
    output_path,         # Path without extension
    wav[0].cpu(),        # Audio tensor (first in batch)
    model.sample_rate,   # Sample rate (32kHz for MusicGen)
    strategy="loudness", # Normalize loudness for consistent volume
)

print(f"Audio saved to {output_path}.wav")
```

## Complete Implementation

Let's put everything together into a complete, runnable example:

```python
# complete_example.py - [Brief description of what this file does]
import torch
import os
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import time

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
    
    # Determine the best device for computation
    if torch.backends.mps.is_available():
        device = "mps"  # Apple Silicon GPU
        print("Using MPS (Metal) for generation")
    elif torch.cuda.is_available():
        device = "cuda"  # NVIDIA GPU
        print("Using CUDA for generation")
    else:
        device = "cpu"  # Fallback to CPU
        print("Using CPU for generation (this will be slow)")
    
    # Load the model
    model = MusicGen.get_pretrained(model_size)
    model.to(device)
    
    # Configure generation parameters
    model.set_generation_params(
        duration=duration,       # Length of audio to generate in seconds
        temperature=temperature, # Controls randomness/creativity
        top_k=250,               # Limits sampling to top 250 token predictions
        top_p=0.0,               # Disables nucleus sampling
    )
    
    # Generate music
    wav = model.generate([prompt])  # Model expects a list of prompts
    
    # Create a filename based on the prompt
    safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt)[:30]
    filename = f"{safe_prompt}_{model_size}_{duration}s"
    
    # Save the audio file
    output_path = os.path.join(output_dir, filename)
    audio_write(
        output_path, 
        wav[0].cpu(), 
        model.sample_rate, 
        strategy="loudness",  # Normalize for consistent volume
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
        duration=10.0,       # 10 seconds
        model_size="small",  # Use "small", "medium", or "large"
        temperature=1.0      # Default creativity level
    )
```

## Variations and Customizations

Let's explore some variations of our solution to address different needs or preferences.

### Variation 1: Batch Processing Multiple Prompts

If you want to generate multiple music pieces from different prompts in sequence:

```python
def batch_generate(prompts, model_size="small", duration=10.0):
    """Generate multiple music pieces from a list of prompts."""
    results = []
    for i, prompt in enumerate(prompts):
        print(f"\nGenerating {i+1}/{len(prompts)}: {prompt[:30]}...")
        output_path = generate_music(
            prompt=prompt,
            model_size=model_size,
            duration=duration
        )
        results.append(output_path)
    
    return results

# Usage
batch_generate(prompts, model_size="medium", duration=15.0)
```

### Variation 2: Temperature Exploration

If you want to explore how different temperature values affect the same prompt:

```python
def explore_temperatures(prompt, temperatures=[0.5, 1.0, 1.5], model_size="small"):
    """Generate variations of the same prompt with different temperature settings."""
    results = []
    for temp in temperatures:
        print(f"\nGenerating with temperature={temp}:")
        output_path = generate_music(
            prompt=prompt,
            temperature=temp,
            model_size=model_size
        )
        results.append(output_path)
    
    return results

# Usage
explore_temperatures(
    "A cinematic orchestral piece with strings and brass",
    temperatures=[0.3, 0.7, 1.0, 1.5]
)
```

## Common Pitfalls and Troubleshooting

### Problem: Out of Memory Errors

When running on GPUs, you might encounter out-of-memory errors, especially with larger models.

**Solution**: 
- Use a smaller model size ('small' instead of 'medium' or 'large')
- Reduce the generation duration
- Ensure no other GPU-intensive applications are running
- Clear GPU cache between generations:

```python
# Clear GPU cache between generations
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

### Problem: Slow Generation

**Solution**:
- The first generation is always slower due to model loading
- Use GPU acceleration if available
- Pre-load the model once and reuse it for multiple generations:

```python
# Pre-load the model once
model = MusicGen.get_pretrained('small')
model.to(device)

# Reuse for multiple generations
for prompt in prompts:
    wav = model.generate([prompt])
    # Save output...
```

### Problem: Low Quality or Undesirable Output

**Solution**:
- Improve your prompt with more specific descriptions
- Adjust the temperature parameter (higher for more creativity, lower for more predictable results)
- Try different model sizes for better quality

## Hands-on Challenge

Now it's your turn to experiment with what you've learned. Try the following challenge:

### Challenge: Genre-Specific Music Generator

Create a script that:
1. Defines at least 5 different music genres with detailed prompts
2. Generates a sample for each genre
3. Creates a simple HTML page that lists all generated samples with playback controls
4. Adds controls for adjusting temperature and duration

### Bonus Challenge

Experiment with combining AudioCraft's MusicGen with a visualization library to create music visualizations that respond to the generated audio.

## Key Takeaways

- [Key concept #1 recap]
- [Key concept #2 recap] 
- [Key concept #3 recap]
- [Key concept #4 recap]

## Next Steps

Now that you've mastered [this chapter's topic], you're ready to explore:

- [Related chapter/topic #1]: Learn how to [brief description]
- [Related chapter/topic #2]: Discover techniques for [brief description]
- [Related chapter/topic #3]: Explore advanced methods for [brief description]

## Further Reading

- [Resource #1]
- [Resource #2]
- [Resource #3]
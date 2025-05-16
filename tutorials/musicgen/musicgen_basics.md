# MusicGen Basics: Music Generation with AudioCraft

This tutorial introduces Meta's MusicGen model and how to use it for generating music from text descriptions. We'll explore the basic functionality and parameters that control the generation process.

## Introduction to MusicGen

MusicGen is a single model that generates high-quality music from textual descriptions such as "an upbeat pop song with catchy melodies" or "a soft jazz piece with piano and saxophone." It was trained on a large dataset of licensed music and can produce impressive results across various musical genres and styles.

Key features of MusicGen:
- Generate music from text descriptions
- Control the duration of generated audio
- Adjust parameters to control the generation process
- Condition on melody inputs to guide the music creation

## Getting Started

Before we begin, make sure you have AudioCraft installed. If not, refer to the [Getting Started Guide](../getting-started/README.md).

Let's first create a simple script to generate music from a text prompt:

```python
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
```

Save this script as `musicgen_basic.py` and run it to generate your first piece of music with MusicGen!

## Understanding the Key Parameters

Let's break down the important parameters that control MusicGen's output:

### Model Size

MusicGen comes in three sizes:
- **small**: Fastest generation, lowest memory usage, but less musical complexity
- **medium**: Balanced generation time and quality
- **large**: Highest quality output but slower and requires more memory

Choose based on your hardware capabilities and quality requirements.

### Duration

The `duration` parameter controls how long the generated audio will be, in seconds. MusicGen officially supports up to 30 seconds of generation. For longer pieces, you'll need to implement techniques for extending generation (covered in advanced tutorials).

### Temperature

Temperature (between 0.0 and 2.0) controls the randomness of the generation:
- **Lower values** (0.1-0.5): More predictable, focused output
- **Medium values** (0.6-1.0): Balanced creativity
- **Higher values** (1.1-2.0): More experimental, diverse output

### Top-k and Top-p Sampling

These parameters control the diversity of the generation:

- **top_k**: Limits sampling to the k most likely tokens at each step
- **top_p**: Uses nucleus sampling to dynamically select the most likely tokens (set to 0.0 to disable)

In our example, we set `top_k=250` and disabled top-p sampling, which works well for many cases.

## Experimenting with Prompts

The text prompt is the most important input for guiding MusicGen. Here are some strategies for crafting effective prompts:

### Be Specific About Genre and Style

```python
# Specific genre prompts
genres = [
    "A classical piano sonata in the style of Mozart",
    "A heavy metal track with distorted guitars and double bass drums",
    "A reggae song with laid-back rhythm and prominent bass",
    "A techno track with pulsing synthesizers and four-on-the-floor beat",
    "A bluegrass tune with fiddle and banjo"
]
```

### Specify Instruments

```python
# Instrument-focused prompts
instruments = [
    "A guitar solo with virtuosic playing and melodic phrases",
    "A drum and bass track with complex percussion patterns",
    "A string quartet playing a melancholic piece",
    "A synthesizer lead melody over a deep bass line",
    "An accordion folk tune with traditional rhythms"
]
```

### Describe Mood and Emotion

```python
# Emotional prompts
moods = [
    "A triumphant and uplifting orchestral piece",
    "A melancholic and introspective piano melody",
    "An energetic and joyful pop song",
    "A tense and suspenseful soundtrack for a thriller movie",
    "A peaceful and serene ambient track for meditation"
]
```

### Combine Descriptions for Better Results

```python
# Combined prompts for more specific guidance
combined = [
    "An energetic electronic dance track with arpeggiated synthesizers, punchy drums, and a catchy melody in the key of A minor",
    "A calm acoustic guitar ballad with gentle finger picking, soft piano in the background, and a melancholic mood",
    "A jazz fusion piece with complex chord progressions, virtuosic saxophone solo, and syncopated rhythm section"
]
```

## Creating a Genre Exploration Script

Let's create a script that explores different musical genres:

```python
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
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    # Load model
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

if __name__ == "__main__":
    explore_genres(model_size="medium", duration=10.0)
```

This script will generate samples across 10 different musical genres, letting you hear how MusicGen interprets various genre descriptions.

## Temperature Exploration

Let's also create a script to explore how temperature affects generation:

```python
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
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    # Load model
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

if __name__ == "__main__":
    # Choose a prompt that works well for temperature exploration
    prompt = "A melodic piano piece with gentle strings in the background"
    explore_temperatures(prompt, model_size="medium", duration=5.0)
```

This script generates variations of the same prompt at different temperature settings, allowing you to hear how this parameter affects creativity and variation.

## Exercises

1. **Genre Blending**: Try creating prompts that blend multiple genres (e.g., "A jazz-rock fusion with elements of electronic music").

2. **Parameter Exploration**: Modify the temperature explorer script to test different `top_k` values while keeping temperature constant.

3. **Instrument Focus**: Create a script similar to the genre explorer but focusing on different solo instruments.

4. **Mood Generator**: Build a script that generates music for different emotional moods and compares the results.

## Next Steps

Now that you understand the basics of MusicGen, you can:

1. Experiment with different prompts and parameters
2. Try using different model sizes to see quality differences
3. Explore melody conditioning in the [Advanced MusicGen](musicgen_advanced.md) tutorial
4. Learn how to create longer pieces with [Extended Generation Techniques](musicgen_extended.md)

## Conclusion

MusicGen provides an impressive capability to generate music from text descriptions. By understanding how to craft effective prompts and adjust generation parameters, you can create a wide variety of musical outputs. As you continue with these tutorials, you'll discover more advanced techniques for fine-tuning and controlling the generation process.
# MusicGen Tutorial

This tutorial covers the usage of MusicGen, Meta's AI model for music generation from text prompts.

## What is MusicGen?

MusicGen is a single model that generates high-quality music from textual descriptions, such as "rock song with a strong guitar and energetic drums" or "calm piano melody with strings in the background."

## Features

- Generate up to 30 seconds of music from text descriptions
- Control melody by providing a reference audio tune
- Set inference parameters for customized generation
- Choose from different pre-trained models (small, medium, large)

## Tutorial Contents

### Getting Started
- [MusicGen Basics](musicgen_basics.md) - Learn the fundamentals of music generation
- [Basic Music Generation Example](examples/musicgen_basic.py) - Simple script to generate music

### Intermediate Tutorials
- [Building a MusicGen Web Interface](musicgen_web_interface.md) - Create an interactive UI for music generation
- [Temperature and Prompting Techniques](examples/README.md) - Explore different parameters and prompt styles

### Advanced Tutorials
- [Advanced MusicGen Techniques](musicgen_advanced.md) - Melody conditioning, extended generation, and more
- [Melody Conditioning Example](examples/musicgen_melody_conditioning.py) - Generate music from reference melodies

### Example Scripts
- [Genre Explorer](examples/musicgen_genre_explorer.py) - Generate music in different genres
- [Temperature Explorer](examples/musicgen_temperature_explorer.py) - Experiment with creativity settings
- [Prompt Explorer](examples/musicgen_prompt_explorer.py) - Test different prompt techniques
- [Web Interface](examples/musicgen_web_app.py) - Interactive Gradio interface for music generation

## Hardware Requirements

MusicGen models have different memory requirements:
- **small**: ~2GB VRAM, suitable for most computers
- **medium**: ~4GB VRAM, recommended for better quality
- **large**: ~8GB VRAM, highest quality output

## Common Parameters

- **temperature**: Controls randomness (0.3-2.0)
- **top_k**: Controls diversity by limiting token selection (50-1000)
- **top_p**: Nucleus sampling parameter (0.0-1.0)
- **duration**: Length of generated audio in seconds (1-30)
- **cfg_coef**: Classifier-free guidance scale (1.0-10.0)

## Example Usage

```python
import torch
from audiocraft.models import MusicGen

# Load model
model = MusicGen.get_pretrained('small')

# Set parameters
model.set_generation_params(
    duration=10.0,
    temperature=1.0
)

# Generate from text prompt
wav = model.generate(["Upbeat electronic track with melodic synths and driving rhythm"])

# Save audio (refer to examples for full saving code)
```

## Advanced Techniques

### Melody Conditioning

```python
# Load melody
melody, sr = torchaudio.load("melody.wav")

# Generate with melody conditioning
wav = model.generate_with_chroma(["Orchestral arrangement"], melody)
```

### Continuation Generation

```python
# Generate continuation from an audio segment
continuation = model.generate_continuation(
    prompt_or_tokens=["Continue this melody"],
    prompt_sample=audio_segment,
    sample_rate=32000
)
```

## Getting Help

If you encounter issues with MusicGen, check the [Troubleshooting](../getting-started/README.md#common-issues-and-solutions) section or visit the [AudioCraft GitHub repository](https://github.com/facebookresearch/audiocraft).
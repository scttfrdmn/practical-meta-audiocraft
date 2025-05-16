# Basic Usage: Your First Steps with AudioCraft

This tutorial will guide you through your first steps with AudioCraft, creating a simple "Hello World" example of audio generation.

## Prerequisites

- AudioCraft successfully installed ([see installation guides](README.md))
- Basic familiarity with Python and PyTorch
- A text editor or Python IDE

## Understanding the AudioCraft API

AudioCraft provides several key models, but we'll begin with MusicGen, which is the easiest to get started with.

The basic workflow is:
1. Load a pretrained model
2. Set generation parameters
3. Generate audio with text prompts
4. Save or play the resulting audio

## Your First Audio Generation

Let's create a simple script that generates a short piece of music from a text description:

```python
# first_generation.py
import torch
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import time

def generate_music(prompt, duration=5.0, output_path="output"):
    """
    Generate music based on a text prompt.
    
    Args:
        prompt (str): Text description of the music to generate
        duration (float): Length of audio in seconds
        output_path (str): Directory to save audio files
    """
    print(f"Generating '{prompt}'...")
    start_time = time.time()
    
    # Check if MPS (Mac Metal) is available, otherwise use CUDA or CPU
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS (Metal) for generation")
    elif torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA for generation")
    else:
        device = "cpu"
        print("Using CPU for generation (this will be slow)")
    
    # Load the model - 'small' is fastest, 'medium' or 'large' for higher quality
    model = MusicGen.get_pretrained('small')
    model.to(device)
    
    # Set generation parameters
    model.set_generation_params(
        duration=duration,  # seconds
        temperature=1.0,    # Controls randomness (higher = more random)
        top_k=250,          # Samples only from the k most likely tokens
        top_p=0.0,          # Nucleus sampling (0.0 = disabled)
    )
    
    # Generate audio from text prompt
    wav = model.generate([prompt])  # Generate a single sample
    
    # Save the generated audio
    audio_write(
        f"{output_path}/{prompt.replace(' ', '_')[:20]}", 
        wav[0].cpu(),       # Move back to CPU for saving
        model.sample_rate,
        strategy="loudness"  # Normalize loudness
    )
    
    generation_time = time.time() - start_time
    print(f"Generation completed in {generation_time:.2f} seconds")
    print(f"Audio saved to {output_path}/{prompt.replace(' ', '_')[:20]}.wav")
    
    return wav

if __name__ == "__main__":
    # Create a simple prompt
    prompt = "Happy acoustic guitar melody with piano"
    
    # Generate the music
    generate_music(prompt)
```

### Running the Example

1. Create a new Python file called `first_generation.py` with the code above
2. Create an `output` directory in the same location
3. Run the script with: `python first_generation.py`

The script will:
- Check for available hardware acceleration (Metal, CUDA, or CPU)
- Load the smallest MusicGen model
- Generate approximately 5 seconds of music based on the prompt
- Save the result to the output directory as a WAV file

## Understanding the Parameters

Let's break down some key parameters you can adjust:

### Model Size
```python
model = MusicGen.get_pretrained('small')  # Options: 'small', 'medium', 'large'
```
- `small`: Fastest, lowest quality, requires less memory
- `medium`: Balanced quality and speed
- `large`: Highest quality, slowest, requires more memory

### Generation Parameters
```python
model.set_generation_params(
    duration=duration,
    temperature=1.0,
    top_k=250,
    top_p=0.0,
)
```

- `duration`: Length of audio in seconds
- `temperature`: Controls randomness (higher values = more creative/chaotic results)
- `top_k`: Controls diversity by limiting to the top k most likely tokens
- `top_p`: Nucleus sampling (alternative to top_k, set to 0 to disable)

## Experimenting with Prompts

Try modifying the prompt to generate different kinds of music:

```python
# Different prompt ideas to try
prompts = [
    "Epic orchestral soundtrack with dramatic drums",
    "Gentle piano melody with soft background strings",
    "Upbeat electronic dance music with synths",
    "Jazz trio with walking bass and brushed drums",
    "Lo-fi hip hop beats to relax/study to"
]
```

## Batch Generation

You can generate multiple audio samples at once:

```python
# Generate multiple samples in a batch
prompts = [
    "Heavy metal guitar riff",
    "Relaxing ambient soundscape",
    "Jazzy piano solo"
]

wavs = model.generate(prompts)  # Generates 3 different audio samples

# Save each generated audio
for idx, (prompt, wav) in enumerate(zip(prompts, wavs)):
    audio_write(
        f"output/batch_{idx}_{prompt.replace(' ', '_')[:10]}", 
        wav.cpu(), 
        model.sample_rate,
        strategy="loudness"
    )
```

## Next Steps

Now that you've created your first audio generation with AudioCraft, you can:

1. Experiment with different prompts to see how they affect the output
2. Try different model sizes to see the quality vs. speed tradeoff
3. Adjust the generation parameters to control the output
4. Move on to the [MusicGen tutorial](../musicgen/README.md) for more advanced music generation techniques

## Troubleshooting

### Common Issues

- **Out of Memory Errors**: Try using a smaller model size or shorter duration
- **Slow Generation**: First generation is always slower due to model loading
- **Unexpected Results**: Refine your prompt to be more specific about instruments, tempo, mood, etc.

### Memory Usage

The approximate memory requirements are:
- `small` model: ~2GB VRAM
- `medium` model: ~4GB VRAM
- `large` model: ~8GB VRAM

For Apple Silicon Macs, unified memory is shared between the CPU and GPU, so ensure you have enough system memory available.
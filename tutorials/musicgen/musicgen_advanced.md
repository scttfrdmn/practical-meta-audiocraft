# Advanced MusicGen Techniques

This tutorial explores advanced techniques for music generation with MusicGen, including melody conditioning, extended generation, and fine-grained control of the generation process.

## Prerequisites

Before proceeding with this tutorial, you should:
- Have completed the [Basic MusicGen Tutorial](musicgen_basics.md)
- Be familiar with PyTorch and audio processing concepts
- Have sufficient GPU resources (at least 4GB VRAM recommended)

## Melody Conditioning

One of MusicGen's most powerful features is the ability to condition the generation on a reference melody. This allows you to provide a melodic structure that the model will follow while generating the rest of the music.

### How Melody Conditioning Works

MusicGen uses a chroma representation of the reference audio to extract melodic information. The chroma features capture the prominence of each pitch class (C, C#, D, etc.) over time, providing the model with melodic guidance while allowing it to generate complementary elements like harmony, rhythm, and instrumentation.

### Basic Melody Conditioning Example

Here's a simple example of melody conditioning:

```python
# melody_conditioning_basic.py
import torch
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import os

def generate_with_melody(
    text_prompt,
    melody_path,
    output_path="melody_output",
    duration=10.0,
    model_size="medium",
    temperature=1.0
):
    """
    Generate music conditioned on both text and melody.
    
    Args:
        text_prompt (str): Text description of the music to generate
        melody_path (str): Path to the melody audio file
        output_path (str): Directory to save output files
        duration (float): Length of audio in seconds
        model_size (str): Size of model to use
        temperature (float): Controls randomness (higher = more random)
    """
    os.makedirs(output_path, exist_ok=True)
    
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
    
    # Load the melody file
    print(f"Loading melody from: {melody_path}")
    melody_waveform, melody_sample_rate = torchaudio.load(melody_path)
    
    # Ensure the melody is mono (if it's stereo, take the first channel)
    if melody_waveform.shape[0] > 1:
        melody_waveform = melody_waveform[0:1]
    
    # Generate with text and melody conditioning
    print(f"Generating with text prompt: '{text_prompt}'")
    wav = model.generate_with_chroma(
        descriptions=[text_prompt],
        melody_wavs=melody_waveform.unsqueeze(0),  # Add batch dimension
        melody_sample_rate=melody_sample_rate,
    )
    
    # Save the generated audio
    filename = f"melody_conditioned_{model_size}"
    output_file = os.path.join(output_path, filename)
    audio_write(
        output_file,
        wav[0].cpu(),
        model.sample_rate,
        strategy="loudness"
    )
    
    print(f"Generated audio saved to {output_file}.wav")
    return f"{output_file}.wav"

if __name__ == "__main__":
    # Example usage
    text_prompt = "An orchestral arrangement with strings, brass, and woodwinds"
    melody_path = "path/to/your/melody.mp3"  # Replace with a real melody file path
    
    generate_with_melody(
        text_prompt=text_prompt,
        melody_path=melody_path,
        duration=15.0,
        model_size="medium"
    )
```

### Tips for Effective Melody Conditioning

1. **Choose Clear Melodies**: Reference audio with a distinct, audible melody works best.
2. **Shorter Durations**: Start with shorter melody segments (5-10 seconds) for more predictable results.
3. **Complementary Prompts**: Make your text prompt complement the melody, describing instrumentation and style.
4. **Match Genres**: For best results, use text prompts that match the genre of your melody.
5. **Clean References**: Use clean, high-quality reference audio without background noise.

### Advanced Options for Melody Conditioning

MusicGen provides additional parameters for fine-tuning the melody conditioning:

```python
# Melody conditioning with fine-tuning
wav = model.generate_with_chroma(
    descriptions=[text_prompt],
    melody_wavs=melody_waveform.unsqueeze(0),
    melody_sample_rate=melody_sample_rate,
    chroma_coefficient=0.8,  # Controls how strictly to follow the melody (0.0-1.0)
    progress=True,           # Show progress bar
)
```

The `chroma_coefficient` parameter controls how closely the generated music follows the reference melody. Higher values (closer to 1.0) enforce stricter adherence to the melody, while lower values allow more creative freedom.

## Extended Generation Techniques

MusicGen officially supports generating up to 30 seconds of audio. For longer pieces, we need to employ extended generation techniques.

### Continuation Generation

One approach for creating longer pieces is to generate music in segments, where each new segment continues from the previous one:

```python
# extended_generation.py
import torch
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import os
import numpy as np

def generate_extended_music(
    prompt,
    total_duration=60.0,
    segment_duration=15.0,
    output_path="extended_output",
    model_size="medium",
    temperature=1.0,
    crossfade_duration=2.0
):
    """
    Generate an extended music piece by creating and concatenating multiple segments.
    
    Args:
        prompt (str): Text description of the music to generate
        total_duration (float): Total desired duration in seconds
        segment_duration (float): Duration of each segment in seconds
        output_path (str): Directory to save output files
        model_size (str): Size of model to use
        temperature (float): Controls randomness (higher = more random)
        crossfade_duration (float): Duration of crossfade between segments
    """
    os.makedirs(output_path, exist_ok=True)
    
    # Determine device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    # Load the model
    model = MusicGen.get_pretrained(model_size)
    model.to(device)
    
    # Set generation parameters
    model.set_generation_params(
        duration=segment_duration,
        temperature=temperature,
        top_k=250,
        top_p=0.0,
    )
    
    # Calculate number of segments needed
    num_segments = int(np.ceil(total_duration / segment_duration))
    print(f"Generating {num_segments} segments for a total of approximately {total_duration} seconds")
    
    # Generate the first segment
    print(f"Generating segment 1/{num_segments} with prompt: '{prompt}'")
    wav = model.generate([prompt])
    
    # Convert to numpy for easier manipulation
    all_audio = wav[0].cpu().numpy()
    
    # Save the first segment
    audio_write(
        os.path.join(output_path, "segment_1"),
        wav[0].cpu(),
        model.sample_rate,
        strategy="loudness"
    )
    
    # Generate subsequent segments with continuation
    for i in range(1, num_segments):
        # Get the end of the previous segment to use as a prompt
        continuation_samples = int(5.0 * model.sample_rate)  # Use last 5 seconds as conditioning
        audio_prompt = torch.tensor(all_audio[-continuation_samples:]).unsqueeze(0).to(device)
        
        print(f"Generating segment {i+1}/{num_segments} (continuation)")
        
        # Generate continuation
        wav = model.generate_continuation(
            prompt_or_tokens=None,  # No text prompt for continuation
            prompt_sample=audio_prompt,
            prompt_sample_rate=model.sample_rate,
        )
        
        segment_audio = wav[0].cpu().numpy()
        
        # Save individual segment
        audio_write(
            os.path.join(output_path, f"segment_{i+1}"),
            wav[0].cpu(),
            model.sample_rate,
            strategy="loudness"
        )
        
        # Apply crossfade between segments
        crossfade_samples = int(crossfade_duration * model.sample_rate)
        
        # Create crossfade weights
        fade_out = np.linspace(1.0, 0.0, crossfade_samples)
        fade_in = np.linspace(0.0, 1.0, crossfade_samples)
        
        # Apply crossfade
        all_audio[-crossfade_samples:] = all_audio[-crossfade_samples:] * fade_out + segment_audio[:crossfade_samples] * fade_in
        
        # Append the new segment (excluding the crossfaded part)
        all_audio = np.concatenate([all_audio, segment_audio[crossfade_samples:]])
    
    # Convert final audio back to tensor
    final_audio = torch.tensor(all_audio)
    
    # Save the complete extended audio
    output_file = os.path.join(output_path, f"extended_{int(total_duration)}s")
    torchaudio.save(
        f"{output_file}.wav",
        final_audio.unsqueeze(0),  # Add channel dimension
        model.sample_rate,
    )
    
    print(f"Extended audio generation complete. Total duration: {len(all_audio) / model.sample_rate:.2f} seconds")
    print(f"Final audio saved to {output_file}.wav")
    return f"{output_file}.wav"

if __name__ == "__main__":
    # Example usage
    prompt = "Ambient electronic music with evolving synthesizer pads and gentle rhythm"
    
    generate_extended_music(
        prompt=prompt,
        total_duration=60.0,    # 1 minute
        segment_duration=15.0,  # 15 seconds per segment
        model_size="medium",
        temperature=0.8         # Slightly lower temperature for continuity
    )
```

This technique generates music in segments and uses each previous segment to condition the generation of the next, creating a coherent extended piece.

### Tips for Extended Generation

1. **Use Lower Temperature**: A slightly lower temperature (0.7-0.9) helps maintain consistency between segments.
2. **Overlapping Segments**: Use a longer crossfade for smoother transitions (2-4 seconds).
3. **Same Prompt Throughout**: For stylistic consistency, use the same text prompt for all segments.
4. **Check for Repetition**: If the music becomes too repetitive, try increasing the temperature for later segments.
5. **Monitor Memory Usage**: Extended generation can consume significant memory over time, so monitor system resources.

## Advanced Parameter Tuning

MusicGen offers several parameters that allow fine-grained control over the generation process.

### Classifier-Free Guidance

Classifier-Free Guidance (CFG) is a technique that enhances the model's adherence to the text prompt:

```python
# Set generation parameters with CFG
model.set_generation_params(
    duration=10.0,
    temperature=1.0,
    top_k=250,
    top_p=0.0,
    cfg_coef=3.0  # Classifier-free guidance scale
)
```

The `cfg_coef` parameter controls how strongly the generation follows the text description:
- **Lower values (1.0-2.0)**: More creative, potentially less aligned with the prompt
- **Medium values (3.0-5.0)**: Good balance between adherence and creativity
- **Higher values (6.0-10.0)**: Strictly follows the prompt, potentially less creative

### Sampling Parameters

Fine-tuning the sampling parameters can significantly impact the quality and diversity of generated music:

```python
model.set_generation_params(
    duration=10.0,
    temperature=0.9,    # Controls randomness
    top_k=50,           # Limits token selection to top k options
    top_p=0.9,          # Uses nucleus sampling with probability threshold
    repetition_penalty=1.2  # Penalizes repeated tokens
)
```

- **top_k**: Limits token selection to the k most likely tokens. Lower values (50-100) create more focused outputs, while higher values (250-1000) allow more diversity.
- **top_p**: Nucleus sampling parameter that dynamically selects tokens whose cumulative probability exceeds the threshold p. Values around 0.9 work well when enabled.
- **repetition_penalty**: Discourages the model from repeating the same patterns by penalizing previously generated tokens.

### Experimenting with Parameters

To understand how different parameters affect generation, create a parameter exploration script:

```python
# parameter_explorer.py
import torch
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import os

def explore_parameters(prompt, output_dir="parameter_exploration"):
    """Explore different generation parameters with the same prompt."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Parameter combinations to explore
    parameters = [
        # Temperature variations
        {"temp": 0.5, "top_k": 250, "top_p": 0.0, "cfg": 3.0},
        {"temp": 1.0, "top_k": 250, "top_p": 0.0, "cfg": 3.0},
        {"temp": 1.5, "top_k": 250, "top_p": 0.0, "cfg": 3.0},
        
        # CFG variations
        {"temp": 1.0, "top_k": 250, "top_p": 0.0, "cfg": 1.0},
        {"temp": 1.0, "top_k": 250, "top_p": 0.0, "cfg": 5.0},
        {"temp": 1.0, "top_k": 250, "top_p": 0.0, "cfg": 8.0},
        
        # Sampling strategy variations
        {"temp": 1.0, "top_k": 50, "top_p": 0.0, "cfg": 3.0},
        {"temp": 1.0, "top_k": 0, "top_p": 0.9, "cfg": 3.0},
        {"temp": 1.0, "top_k": 50, "top_p": 0.9, "cfg": 3.0},
    ]
    
    # Determine device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    # Load model (use "small" for faster exploration)
    model = MusicGen.get_pretrained("small")
    model.to(device)
    
    # Generate with each parameter set
    for i, params in enumerate(parameters):
        print(f"Generating sample {i+1}/{len(parameters)} with parameters: {params}")
        
        # Set parameters
        model.set_generation_params(
            duration=5.0,  # Short duration for exploration
            temperature=params["temp"],
            top_k=params["top_k"],
            top_p=params["top_p"],
            cfg_coef=params["cfg"]
        )
        
        # Generate
        wav = model.generate([prompt])
        
        # Save with parameter information in filename
        filename = f"temp{params['temp']}_topk{params['top_k']}_topp{params['top_p']}_cfg{params['cfg']}"
        output_path = os.path.join(output_dir, filename)
        audio_write(output_path, wav[0].cpu(), model.sample_rate)
        
        print(f"Saved to {output_path}.wav")

if __name__ == "__main__":
    prompt = "An electronic dance track with driving beat and synthesizer melody"
    explore_parameters(prompt)
```

This script will generate multiple samples with different parameter combinations, allowing you to hear how each parameter affects the output.

## Combining Advanced Techniques

For the most sophisticated music generation, you can combine melody conditioning with parameter tuning and extended generation:

```python
# advanced_music_project.py
import torch
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import os
import numpy as np

def generate_advanced_music_project(
    text_prompt,
    melody_path,
    total_duration=60.0,
    segment_duration=15.0,
    output_dir="advanced_project",
    model_size="large",
    temperature=0.9,
    cfg_coef=3.5,
    crossfade_duration=3.0
):
    """
    Create an advanced music project using melody conditioning, parameter tuning,
    and extended generation techniques.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    # Load the model
    model = MusicGen.get_pretrained(model_size)
    model.to(device)
    
    # Load the melody file
    melody_waveform, melody_sample_rate = torchaudio.load(melody_path)
    if melody_waveform.shape[0] > 1:
        melody_waveform = melody_waveform[0:1]  # Convert to mono if needed
    
    # Calculate number of segments needed
    num_segments = int(np.ceil(total_duration / segment_duration))
    print(f"Generating {num_segments} segments for a total of approximately {total_duration} seconds")
    
    # Set parameters for the first segment
    model.set_generation_params(
        duration=segment_duration,
        temperature=temperature,
        top_k=250,
        top_p=0.0,
        cfg_coef=cfg_coef
    )
    
    # Generate the first segment with melody conditioning
    print(f"Generating segment 1/{num_segments} with melody conditioning")
    wav = model.generate_with_chroma(
        descriptions=[text_prompt],
        melody_wavs=melody_waveform.unsqueeze(0),
        melody_sample_rate=melody_sample_rate
    )
    
    # Convert to numpy for easier manipulation
    all_audio = wav[0].cpu().numpy()
    
    # Save the first segment
    audio_write(
        os.path.join(output_dir, "segment_1"),
        wav[0].cpu(),
        model.sample_rate,
        strategy="loudness"
    )
    
    # Generate subsequent segments with continuation and gradually increasing temperature
    for i in range(1, num_segments):
        # Slightly increase temperature for each segment to add variation
        current_temperature = min(temperature + (i * 0.05), 1.3)
        
        # Set parameters for this segment
        model.set_generation_params(
            duration=segment_duration,
            temperature=current_temperature,
            top_k=250,
            top_p=0.0,
            cfg_coef=cfg_coef
        )
        
        # Get the end of the previous segment to use as a prompt
        continuation_samples = int(5.0 * model.sample_rate)  # Use last 5 seconds as conditioning
        audio_prompt = torch.tensor(all_audio[-continuation_samples:]).unsqueeze(0).to(device)
        
        print(f"Generating segment {i+1}/{num_segments} (continuation, temp={current_temperature:.2f})")
        
        # Generate continuation with the original text prompt for guidance
        wav = model.generate_continuation(
            prompt_or_tokens=[text_prompt],  # Use text prompt for guidance
            prompt_sample=audio_prompt,
            prompt_sample_rate=model.sample_rate,
        )
        
        segment_audio = wav[0].cpu().numpy()
        
        # Save individual segment
        audio_write(
            os.path.join(output_dir, f"segment_{i+1}"),
            wav[0].cpu(),
            model.sample_rate,
            strategy="loudness"
        )
        
        # Apply crossfade between segments
        crossfade_samples = int(crossfade_duration * model.sample_rate)
        
        # Create crossfade weights
        fade_out = np.linspace(1.0, 0.0, crossfade_samples)
        fade_in = np.linspace(0.0, 1.0, crossfade_samples)
        
        # Apply crossfade
        all_audio[-crossfade_samples:] = all_audio[-crossfade_samples:] * fade_out + segment_audio[:crossfade_samples] * fade_in
        
        # Append the new segment (excluding the crossfaded part)
        all_audio = np.concatenate([all_audio, segment_audio[crossfade_samples:]])
    
    # Convert final audio back to tensor
    final_audio = torch.tensor(all_audio)
    
    # Save the complete extended audio
    output_file = os.path.join(output_dir, f"advanced_project_{int(total_duration)}s")
    torchaudio.save(
        f"{output_file}.wav",
        final_audio.unsqueeze(0),
        model.sample_rate,
    )
    
    # Apply post-processing: 
    # 1. Normalize audio levels
    normalized_audio = final_audio / (torch.max(torch.abs(final_audio)) + 1e-8)
    
    # 2. Add fade-in and fade-out
    fade_samples = int(3.0 * model.sample_rate)  # 3-second fades
    if len(normalized_audio) > 2 * fade_samples:
        fade_in = torch.linspace(0.0, 1.0, fade_samples)
        fade_out = torch.linspace(1.0, 0.0, fade_samples)
        
        normalized_audio[:fade_samples] *= fade_in
        normalized_audio[-fade_samples:] *= fade_out
    
    # Save the processed final version
    output_file_processed = os.path.join(output_dir, f"advanced_project_processed_{int(total_duration)}s")
    torchaudio.save(
        f"{output_file_processed}.wav",
        normalized_audio.unsqueeze(0),
        model.sample_rate,
    )
    
    print(f"Advanced music project complete. Total duration: {len(all_audio) / model.sample_rate:.2f} seconds")
    print(f"Final audio saved to {output_file_processed}.wav")
    return f"{output_file_processed}.wav"

if __name__ == "__main__":
    text_prompt = "An epic orchestral piece with evolving sections, starting gentle and building to a powerful climax"
    melody_path = "path/to/your/melody.mp3"  # Replace with actual path
    
    generate_advanced_music_project(
        text_prompt=text_prompt,
        melody_path=melody_path,
        total_duration=120.0,  # 2 minutes
        model_size="large"     # Use large model for highest quality
    )
```

This advanced script combines melody conditioning, extended generation with crossfades, temperature variation, and post-processing into a comprehensive music generation pipeline.

## Performance Optimization

When working with advanced techniques, performance optimization becomes important:

### Memory Management

```python
# Clear GPU memory after generation
def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()

# Use in your generation loop
for i in range(num_segments):
    # Generate segment
    # ...
    
    # Clear memory
    clear_gpu_memory()
```

### Batch Processing

For generating multiple variations efficiently:

```python
# Generate multiple variations in a single batch
prompts = [
    "Epic orchestral music with brass and strings",
    "Epic orchestral music with choir and percussion",
    "Epic orchestral music with solo violin and piano"
]

# Generate all variations at once
wavs = model.generate(prompts)

# Save all variations
for i, wav in enumerate(wavs):
    audio_write(f"variation_{i+1}", wav.cpu(), model.sample_rate)
```

## Exercises

1. **Melody Transformation**: Take a simple melody (like a piano recording of "Twinkle Twinkle Little Star") and use melody conditioning with different style prompts (jazz, rock, orchestral, electronic).

2. **Extended Composition**: Create a 2-minute piece that evolves between different moods by changing the text prompt for each segment.

3. **Parameter Exploration**: Systematically explore how different values of `cfg_coef` affect generation with the same prompt and melody.

4. **Genre Fusion**: Use melody conditioning and carefully crafted prompts to blend two distinct genres (e.g., "Electronic dance music with classical orchestral elements").

## Next Steps

After mastering these advanced techniques, explore:

- Fine-tuning the MusicGen model on your own data
- Building a web interface for your generation pipeline
- Creating an audio post-processing chain for mastering
- Integrating MusicGen with live performance tools

## Conclusion

The advanced techniques covered in this tutorial allow you to push MusicGen to its full potential, creating sophisticated, expressive, and extended musical compositions. By combining melody conditioning, parameter tuning, and extended generation strategies, you can achieve results that go well beyond basic text-to-music generation.

Remember that generating high-quality music still requires careful prompt engineering, parameter tuning, and often manual selection of the best results. MusicGen is a powerful tool, but making the most of it requires both technical understanding and musical intuition.
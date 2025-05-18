---
layout: chapter
title: "Melody Conditioning for AI Music Generation"
difficulty: intermediate
copyright: "Copyright © 2025 Scott Friedman. This work is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License."
estimated_time: 60
---

> *"I have these amazing melody ideas I've recorded on my phone and some piano sketches, but I'm not a skilled producer or arranger. I wish I could feed these melodies into an AI and have it create fully produced tracks while keeping my original musical ideas intact."* 
> 
> — Jamie Rodriguez, songwriter and composer

# Chapter 8: Melody Conditioning for AI Music Generation

## The Challenge

You've mastered generating music from text prompts and optimizing parameters, but sometimes you already have a specific melody in mind. Perhaps you've recorded a catchy tune on your phone, played a simple motif on piano, or have an existing melody from a royalty-free source that you want to transform.

The challenge is to use MusicGen's melody conditioning capabilities to create fully arranged, richly produced music that follows your specific melodic ideas while leveraging your text prompts for stylistic direction.

## Learning Objectives

By the end of this chapter, you will be able to:

- Prepare and preprocess audio files for effective melody conditioning
- Generate music that follows a reference melody using MusicGen
- Combine text and melody conditioning for precise creative control
- Apply different stylistic variations to the same melodic material
- Build a complete melody transformation pipeline for your audio content
- Troubleshoot common issues with melody conditioning

## Prerequisites

- Basic understanding of MusicGen (Chapter 5)
- Experience with prompt engineering for music (Chapter 6)
- Familiarity with parameter optimization (Chapter 7)
- Python environment with AudioCraft installed
- Basic understanding of audio formats and processing

## Key Concepts: Understanding Melody Conditioning

### What is Melody Conditioning?

Melody conditioning is one of MusicGen's most powerful features. It allows you to provide a reference audio file containing a melody, which the model will then use as a structural framework for generation. The generated music will follow the melodic contour, rhythm, and overall musical structure of your reference, while applying the style specified in your text prompt.

This creates a powerful combination:
- **Melody (audio input)**: Controls the musical structure and melodic content
- **Text prompt**: Controls the genre, instrumentation, and stylistic elements

The result is a unique form of human-AI collaboration where you provide the core musical idea and MusicGen handles the arrangement and production.

### How Melody Conditioning Works

Behind the scenes, MusicGen processes your melody reference through several steps:

1. **Audio preprocessing**: The input audio is converted to mono, resampled to the model's required sample rate, and normalized.

2. **Chroma extraction**: The model extracts chromagram features from the audio, which represent pitch content over time, capturing the essential melodic information.

3. **Conditioning**: During generation, these chroma features condition the model along with the text prompt, guiding it to create music that follows both the melodic structure and the requested style.

This process differs from the standard text-only generation in that the model is now constrained to follow a specific musical path rather than freely interpreting just the text prompt.

## Preparing Audio for Melody Conditioning

Not all audio files work equally well for melody conditioning. Here are the key requirements and best practices:

### Audio Format Requirements

```python
def prepare_melody_for_conditioning(melody_file, target_sr=32000, plot=False, output_dir=None):
    """
    Prepare an audio file for use as melody conditioning in MusicGen.
    
    Args:
        melody_file (str): Path to the melody audio file
        target_sr (int): Target sample rate (32000 Hz for MusicGen)
        plot (bool): Whether to plot and save the waveform for visualization
        output_dir (str): Directory to save processed audio and plots
        
    Returns:
        torch.Tensor: Processed audio tensor ready for melody conditioning
        
    Note:
        Melody conditioning works best with:
        - Clean, monophonic melodies (single notes, not chords)
        - 3-15 seconds of audio (longer can work but may lose coherence)
        - Minimal background noise or accompaniment
        - Consistent tempo and clear note onsets
    """
    import torch
    import torchaudio
    import torchaudio.transforms as T
    import os
    import matplotlib.pyplot as plt
    
    # Create output directory if specified and doesn't exist
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading melody file: {melody_file}")
    
    # Load audio file
    waveform, sample_rate = torchaudio.load(melody_file)
    
    # Step 1: Convert to mono by averaging channels if stereo
    if waveform.shape[0] > 1:
        print("Converting stereo to mono (melody conditioning requires mono)")
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Step 2: Resample to target sample rate if needed
    if sample_rate != target_sr:
        print(f"Resampling from {sample_rate}Hz to {target_sr}Hz")
        resampler = T.Resample(sample_rate, target_sr)
        waveform = resampler(waveform)
    
    # Step 3: Normalize audio to prevent extreme volumes
    # Using peak normalization to maintain dynamics
    peak = torch.abs(waveform).max()
    if peak > 1.0:
        print(f"Normalizing audio (peak value: {peak:.2f})")
        waveform = waveform / peak
    
    # Optionally plot the waveform for visualization
    if plot and output_dir is not None:
        print("Creating waveform visualization")
        plt.figure(figsize=(10, 4))
        plt.plot(waveform[0].numpy())
        plt.title("Melody Waveform")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "melody_waveform.png")
        plt.savefig(plot_path)
        print(f"Saved waveform plot to {plot_path}")
        plt.close()
    
    # Optionally save the processed audio
    if output_dir is not None:
        output_path = os.path.join(output_dir, "processed_melody.wav")
        torchaudio.save(
            output_path, 
            waveform, 
            target_sr,
            encoding="PCM_S", 
            bits_per_sample=16
        )
        print(f"Saved processed audio to {output_path}")
    
    # Return the processed waveform
    return waveform
```

This function handles the key preprocessing steps required for effective melody conditioning:

1. **Conversion to mono**: MusicGen's melody conditioning expects mono audio
2. **Resampling to 32kHz**: The model requires this exact sample rate
3. **Normalization**: Ensures audio levels are appropriate
4. **Optional visualization**: Helps you understand the audio input

### Optimal Melody Characteristics

For best results with melody conditioning, your input audio should have these characteristics:

1. **Monophonic content**: Single notes work better than chords or complex harmonies
2. **Clean recording**: Minimal background noise or other instruments
3. **Clear note onsets**: Well-defined beginnings of notes
4. **Moderate length**: 5-15 seconds generally works best (longer clips may lose coherence)
5. **Consistent tempo**: Regular rhythm helps the model follow the structure

### Recording Tips for Melody Conditioning

If you're creating your own melody recordings specifically for conditioning:

1. Record in a quiet environment with minimal reverb
2. Use a simple instrument (piano, guitar, voice) playing one note at a time
3. Play with clear articulation and consistent timing
4. Keep the melody relatively simple (8-16 bars)
5. Leave some space between phrases

## Basic Melody Conditioning Implementation

Let's create a basic implementation of melody conditioning with MusicGen:

```python
# melody_generator.py
import torch
import torchaudio
import os
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import matplotlib.pyplot as plt

def generate_from_melody(
    melody_file, 
    prompt, 
    model_size="medium", 
    duration=None,
    output_dir="melody_generations",
    plot_waveform=False
):
    """
    Generate music that follows a reference melody with MusicGen.
    
    Args:
        melody_file (str): Path to the melody audio file
        prompt (str): Text description of desired musical style
        model_size (str): Size of MusicGen model ("small", "medium", or "large")
        duration (float): Optional duration override (uses melody length if None)
        output_dir (str): Directory to save generated audio and visualizations
        plot_waveform (bool): Whether to plot and save melody waveform
        
    Returns:
        str: Path to the generated audio file
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine the device to use
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS (Metal) for generation")
    elif torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA for generation")
    else:
        device = "cpu"
        print("Using CPU for generation (this will be slow)")
    
    # Load and process the melody file
    print(f"Processing melody: {melody_file}")
    waveform, sample_rate = torchaudio.load(melody_file)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample to 32kHz (MusicGen's expected sample rate)
    if sample_rate != 32000:
        print(f"Resampling from {sample_rate}Hz to 32000Hz")
        resampler = torchaudio.transforms.Resample(sample_rate, 32000)
        waveform = resampler(waveform)
        sample_rate = 32000
    
    # Plot waveform if requested
    if plot_waveform:
        plt.figure(figsize=(10, 4))
        plt.plot(waveform[0].numpy())
        plt.title("Melody Waveform")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "melody_waveform.png"))
        plt.close()
    
    # Calculate melody duration in seconds
    melody_duration = waveform.shape[1] / sample_rate
    print(f"Melody duration: {melody_duration:.2f} seconds")
    
    # Use melody duration if no explicit duration is provided
    if duration is None:
        duration = melody_duration
        print(f"Using melody length as generation duration: {duration:.2f}s")
    
    # Load MusicGen model
    print(f"Loading MusicGen {model_size} model...")
    model = MusicGen.get_pretrained(model_size)
    model.to(device)
    
    # Set generation parameters
    model.set_generation_params(
        duration=duration,
        temperature=1.0,
        top_k=250,
        top_p=0.0,
        cfg_coef=3.0
    )
    
    # Move melody tensor to the same device as the model
    melody = waveform.to(device)
    
    # Save information about the generation
    info_file = os.path.join(output_dir, "generation_info.txt")
    with open(info_file, "w") as f:
        f.write(f"Melody File: {melody_file}\n")
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Model Size: {model_size}\n")
        f.write(f"Melody Duration: {melody_duration:.2f}s\n")
        f.write(f"Generation Duration: {duration:.2f}s\n")
    
    # Generate music conditioned on both text and melody
    print(f"Generating music with prompt: '{prompt}'")
    wav = model.generate_with_chroma(
        descriptions=[prompt],
        chroma=melody.unsqueeze(0)  # Add batch dimension
    )
    
    # Save the generated audio
    output_path = os.path.join(output_dir, "melody_generation")
    audio_write(
        output_path,
        wav[0].cpu(),
        model.sample_rate,
        strategy="loudness"
    )
    
    print(f"Generation complete! Output saved to {output_path}.wav")
    return f"{output_path}.wav"

if __name__ == "__main__":
    # Example usage
    melody_file = "path/to/your/melody.wav"  # Replace with your melody file
    prompt = "A cinematic orchestral arrangement with strings, brass, and dramatic percussion"
    
    generate_from_melody(
        melody_file=melody_file,
        prompt=prompt,
        model_size="medium",
        output_dir="orchestral_arrangement",
        plot_waveform=True
    )
```

This implementation provides a solid foundation for melody conditioning. The key aspects are:

1. **Audio preprocessing**: Converting to mono, resampling to 32kHz
2. **Using melody duration**: Automatically adapting the generation length to match the input melody
3. **Melody device handling**: Moving the melody tensor to the same device as the model
4. **Using `generate_with_chroma`**: The special method for melody-conditioned generation

## Advanced Techniques: Style Variation Generator

One powerful application of melody conditioning is creating multiple stylistic variations of the same melody. Let's implement a system that takes a melody and generates it in several different musical styles:

```python
# melody_style_explorer.py
import torch
import torchaudio
import os
import json
from datetime import datetime
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

class MelodyStyleExplorer:
    """
    A tool for generating multiple stylistic variations of a melody.
    
    This class allows you to take a single melody and render it in various
    musical styles, genres, and arrangements using MusicGen's melody
    conditioning capabilities.
    """
    
    def __init__(self, model_size="medium", device=None):
        """Initialize the style explorer with a MusicGen model."""
        # Determine device automatically if not specified
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
                print("Using MPS (Metal) for generation")
            elif torch.cuda.is_available():
                device = "cuda"
                print("Using CUDA for generation")
            else:
                device = "cpu"
                print("Using CPU for generation (this will be slow)")
                
        self.device = device
        self.model_size = model_size
        
        # Load the model
        print(f"Loading MusicGen {model_size} model...")
        self.model = MusicGen.get_pretrained(model_size)
        self.model.to(device)
        
        # Predefined style categories and prompts
        self.style_presets = {
            "orchestral": [
                "A grand orchestral arrangement with soaring strings and brass",
                "A delicate chamber orchestra piece with strings and woodwinds",
                "An epic cinematic orchestral theme with full percussion and brass stabs"
            ],
            "electronic": [
                "An EDM track with driving beat and synthesizer leads",
                "A chillwave electronic piece with ambient pads and glitchy beats",
                "A synthwave track with 80s drum machines and analog synth arpeggios"
            ],
            "jazz": [
                "A smooth jazz arrangement with saxophone, piano and brushed drums",
                "A bebop jazz combo with walking bass, piano and trumpet",
                "A jazz fusion piece with electric piano, synth bass and tight drums"
            ],
            "rock": [
                "A rock band arrangement with electric guitars, bass and drums",
                "An indie rock track with jangly guitars and laid-back groove",
                "A hard rock version with distorted guitars and powerful drums"
            ],
            "folk": [
                "An acoustic folk arrangement with guitars, mandolin and light percussion",
                "A Celtic folk interpretation with fiddle, flute and acoustic guitar",
                "A folk ballad with fingerpicked guitar and subtle string accompaniment"
            ]
        }
        
    def process_melody(self, melody_file):
        """
        Process a melody file for conditioning.
        
        Args:
            melody_file (str): Path to the melody audio file
            
        Returns:
            tuple: Processed melody tensor and its duration in seconds
        """
        print(f"Processing melody: {melody_file}")
        waveform, sample_rate = torchaudio.load(melody_file)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample to 32kHz (MusicGen's required sample rate)
        if sample_rate != 32000:
            print(f"Resampling from {sample_rate}Hz to 32000Hz")
            resampler = torchaudio.transforms.Resample(sample_rate, 32000)
            waveform = resampler(waveform)
            sample_rate = 32000
        
        # Calculate duration
        duration = waveform.shape[1] / sample_rate
        
        # Move to appropriate device
        waveform = waveform.to(self.device)
        
        return waveform, duration
    
    def generate_style_variations(self, melody_file, styles=None, custom_prompts=None, 
                                  duration=None, output_dir=None, generation_params=None):
        """
        Generate multiple style variations of a melody.
        
        Args:
            melody_file (str): Path to the melody audio file
            styles (list): List of style categories to use (from self.style_presets)
            custom_prompts (list): Additional custom prompts to use
            duration (float): Duration override (uses melody length if None)
            output_dir (str): Directory to save all variations
            generation_params (dict): Override default generation parameters
            
        Returns:
            dict: Information about the generated variations
        """
        # Process the melody file
        melody, melody_duration = self.process_melody(melody_file)
        
        # Use melody duration if no override is provided
        if duration is None:
            duration = melody_duration
            print(f"Using melody length as generation duration: {duration:.2f}s")
        
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_dir is None:
            melody_name = os.path.splitext(os.path.basename(melody_file))[0]
            output_dir = f"melody_variations_{melody_name}_{timestamp}"
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the processed melody for reference
        melody_path = os.path.join(output_dir, "original_melody.wav")
        torchaudio.save(
            melody_path,
            melody.cpu(),
            32000
        )
        
        # Collect all prompts to use
        all_prompts = []
        
        # Add prompts from requested style categories
        if styles:
            for style in styles:
                if style in self.style_presets:
                    all_prompts.extend(self.style_presets[style])
                else:
                    print(f"Warning: Style '{style}' not found in presets")
        
        # Add any custom prompts
        if custom_prompts:
            all_prompts.extend(custom_prompts)
        
        # If no prompts were specified, use a default selection
        if not all_prompts:
            # Take one prompt from each style category
            for style, prompts in self.style_presets.items():
                all_prompts.append(prompts[0])
        
        # Set generation parameters
        default_params = {
            'duration': duration,
            'temperature': 1.0,
            'top_k': 250,
            'top_p': 0.0,
            'cfg_coef': 3.0
        }
        
        # Override with any provided parameters
        if generation_params:
            default_params.update(generation_params)
        
        self.model.set_generation_params(**default_params)
        
        # Prepare melody with batch dimension for generation
        melody_input = melody.unsqueeze(0)  # Add batch dimension
        
        # Track generation results
        generation_results = {
            'melody_file': melody_file,
            'processed_melody': melody_path,
            'melody_duration': melody_duration,
            'generation_duration': duration,
            'model_size': self.model_size,
            'generation_params': default_params,
            'variations': []
        }
        
        # Generate each variation
        total_prompts = len(all_prompts)
        for i, prompt in enumerate(all_prompts):
            print(f"[{i+1}/{total_prompts}] Generating variation: '{prompt}'")
            
            # Create a prompt-based filename (simplified version of the prompt)
            prompt_filename = prompt.lower()
            for char in [',', '.', "'", '"', '!', '?', ':', ';', '/', '\\']:
                prompt_filename = prompt_filename.replace(char, '')
            prompt_filename = prompt_filename.replace(' ', '_')[:50]  # Limit length
            
            output_filename = f"{i+1:02d}_{prompt_filename}"
            output_path = os.path.join(output_dir, output_filename)
            
            # Generate with melody conditioning
            wav = self.model.generate_with_chroma(
                descriptions=[prompt],
                chroma=melody_input
            )
            
            # Save the audio
            audio_write(
                output_path,
                wav[0].cpu(),
                self.model.sample_rate,
                strategy="loudness"
            )
            
            # Record information about this variation
            generation_results['variations'].append({
                'prompt': prompt,
                'output_file': f"{output_path}.wav"
            })
            
            print(f"Saved to {output_path}.wav")
        
        # Save generation metadata
        metadata_path = os.path.join(output_dir, "generation_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(generation_results, f, indent=2)
        
        # Create a helpful README
        readme_path = os.path.join(output_dir, "README.txt")
        with open(readme_path, 'w') as f:
            f.write("MELODY STYLE VARIATIONS\n")
            f.write("======================\n\n")
            f.write(f"Original melody: {melody_file}\n")
            f.write(f"Processed for MusicGen: {melody_path}\n")
            f.write(f"Generation duration: {duration:.2f} seconds\n\n")
            f.write("Style variations:\n\n")
            
            for i, variation in enumerate(generation_results['variations']):
                f.write(f"{i+1:02d}. {variation['prompt']}\n")
                f.write(f"    File: {os.path.basename(variation['output_file'])}\n\n")
            
            f.write("\nGeneration parameters:\n")
            for param, value in default_params.items():
                f.write(f"- {param}: {value}\n")
            
            f.write("\nListen to each variation to compare how MusicGen interprets\n")
            f.write("the same melody in different musical styles and arrangements.\n")
        
        print(f"\nGenerated {total_prompts} style variations in {output_dir}")
        return generation_results
    
    def create_comparative_styles(self, melody_file, output_dir=None):
        """
        Generate a standard set of comparative styles covering major genres.
        
        This is a convenience method that generates one variation in each
        major style category for easy comparison.
        """
        # Create custom prompt list with one from each category
        comparison_prompts = []
        style_descriptions = []
        
        for style, prompts in self.style_presets.items():
            comparison_prompts.append(prompts[0])
            style_descriptions.append(style)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_dir is None:
            melody_name = os.path.splitext(os.path.basename(melody_file))[0]
            output_dir = f"melody_style_comparison_{melody_name}_{timestamp}"
        
        results = self.generate_style_variations(
            melody_file=melody_file,
            custom_prompts=comparison_prompts,
            output_dir=output_dir
        )
        
        # Add style categories to the README
        readme_path = os.path.join(output_dir, "README.txt")
        with open(readme_path, 'r') as f:
            content = f.read()
        
        # Add style categories to the output
        style_content = "Style categories:\n\n"
        for i, style in enumerate(style_descriptions):
            style_content += f"{i+1:02d}. {style.capitalize()}\n"
        
        with open(readme_path, 'w') as f:
            f.write(content + "\n" + style_content)
        
        return results
    
    def batch_process_melodies(self, melody_files, style, output_base_dir="batch_generations"):
        """
        Process multiple melody files with the same style prompt.
        
        Useful for processing a collection of melodies with consistent styling.
        """
        os.makedirs(output_base_dir, exist_ok=True)
        
        # Choose a prompt for the style
        if style in self.style_presets:
            prompt = self.style_presets[style][0]
        else:
            prompt = style  # Use the style parameter directly as a prompt
        
        results = []
        
        for melody_file in melody_files:
            melody_name = os.path.splitext(os.path.basename(melody_file))[0]
            output_dir = os.path.join(output_base_dir, melody_name)
            
            print(f"Processing melody: {melody_name}")
            result = self.generate_style_variations(
                melody_file=melody_file,
                custom_prompts=[prompt],
                output_dir=output_dir
            )
            
            results.append(result)
        
        # Create a summary file
        summary_path = os.path.join(output_base_dir, "batch_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"BATCH PROCESSING SUMMARY\n")
            f.write(f"=======================\n\n")
            f.write(f"Style prompt: \"{prompt}\"\n\n")
            f.write(f"Processed melodies:\n\n")
            
            for i, result in enumerate(results):
                f.write(f"{i+1}. {os.path.basename(result['melody_file'])}\n")
                f.write(f"   Output: {result['variations'][0]['output_file']}\n\n")
        
        return results

# Example usage
if __name__ == "__main__":
    explorer = MelodyStyleExplorer(model_size="medium")
    
    # Generate variations in all preset styles
    # explorer.generate_style_variations(
    #     melody_file="path/to/your/melody.wav",
    #     styles=["orchestral", "electronic", "jazz", "rock", "folk"],
    #     output_dir="complete_style_exploration"
    # )
    
    # Or generate the standard comparison set
    explorer.create_comparative_styles(
        melody_file="path/to/your/melody.wav",
        output_dir="melody_style_comparison"
    )
```

This advanced implementation builds on our basic approach and adds several powerful features:

1. **Style presets**: Predefined genre categories with specific prompts
2. **Batch processing**: Process multiple melodies with the same style
3. **Comparative analysis**: Generate one example in each major style
4. **Detailed output**: README files and metadata to track generations
5. **Style variation**: Multiple prompts within the same genre category

## Melody Conditioning with Parameter Optimization

To get the best results from melody conditioning, we need to optimize generation parameters. Different parameter settings can dramatically affect how closely the output follows the melody versus how creative it gets with the arrangement.

Here's a guide to parameter tuning specifically for melody conditioning:

| Parameter | Effect on Melody Conditioning | Recommended Setting |
|-----------|--------------------------------|---------------------|
| temperature | Controls how strictly the melody is followed | 0.6-0.8 for accurate following, 1.0-1.3 for creative interpretation |
| cfg_coef | Balances melody following vs. prompt adherence | 2.0-2.5 for melody-dominant, 3.0-3.5 for balanced approach |
| top_k | Affects diversity in the arrangement | 150-250 for most cases, higher for more experimental arrangements |
| duration | Controls generation length | Match to melody length for best results |

For most melody conditioning applications, we recommend a balanced approach:

```python
# Balanced melody conditioning parameters
model.set_generation_params(
    duration=melody_duration,  # Match to melody length
    temperature=0.9,           # Slightly conservative for better melody following
    top_k=200,                 # Standard diversity
    top_p=0.0,                 # Disable nucleus sampling
    cfg_coef=2.8               # Slightly reduced to prioritize melody
)
```

## Complete Implementation: Melody Transformation Pipeline

Let's create a complete melody transformation pipeline that combines all the concepts we've learned. This pipeline will:

1. Process and prepare melodies from various input formats
2. Apply multiple stylistic variations
3. Optimize parameters for different melody types
4. Provide a comprehensive API for melody-based music generation

```python
# melody_transformation_pipeline.py
import torch
import torchaudio
import os
import json
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

class MelodyTransformationPipeline:
    """
    End-to-end pipeline for transforming melody ideas into fully produced music.
    
    This class provides a comprehensive workflow for processing melodies,
    transforming them with various styles, and generating high-quality
    musical outputs using MusicGen's melody conditioning capabilities.
    """
    
    def __init__(self, model_size="medium", device=None):
        """Initialize the melody transformation pipeline."""
        # Determine device automatically if not specified
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
                print("Using MPS (Metal) for generation")
            elif torch.cuda.is_available():
                device = "cuda"
                print("Using CUDA for generation")
            else:
                device = "cpu"
                print("Using CPU for generation (this will be slow)")
                
        self.device = device
        self.model_size = model_size
        
        # Load the model
        print(f"Loading MusicGen {model_size} model...")
        self.model = MusicGen.get_pretrained(model_size)
        self.model.to(device)
        
        # Define parameter presets optimized for melody conditioning
        self.parameter_presets = {
            "accurate": {
                'temperature': 0.7,
                'top_k': 150,
                'top_p': 0.0,
                'cfg_coef': 2.5
            },
            "balanced": {
                'temperature': 0.9,
                'top_k': 200,
                'top_p': 0.0,
                'cfg_coef': 2.8
            },
            "creative": {
                'temperature': 1.2,
                'top_k': 250,
                'top_p': 0.0,
                'cfg_coef': 2.2
            }
        }
        
        # Define style presets with prompts optimized for melody conditioning
        self.style_presets = {
            "orchestral": {
                "cinematic": "A cinematic orchestral arrangement with dramatic strings, brass, and percussion that builds to an epic climax",
                "classical": "A classical orchestra arrangement with strings, woodwinds, and brass in a traditional symphonic style",
                "chamber": "A delicate chamber orchestra arrangement with string quartet, piano, and light woodwinds"
            },
            "electronic": {
                "edm": "An upbeat electronic dance track with synthesizers, driving beat, and modern production",
                "ambient": "An ambient electronic piece with atmospheric pads, subtle beats, and ethereal textures",
                "synthwave": "A retro synthwave track with 80s drum machines, analog synth arpeggios, and neon atmosphere"
            },
            "band": {
                "rock": "A rock band arrangement with electric guitars, bass, drums, and energetic performance",
                "jazz": "A jazz ensemble with piano, upright bass, drums, saxophone, and trumpet playing in a smooth style",
                "acoustic": "An intimate acoustic arrangement with guitar, piano, light percussion, and warm atmosphere"
            }
        }
        
    def process_melody(self, melody_file, output_dir=None, analyze=False):
        """
        Process a melody file for conditioning.
        
        Args:
            melody_file (str): Path to the melody audio file
            output_dir (str): Directory to save processed files
            analyze (bool): Whether to perform and save melody analysis
            
        Returns:
            tuple: Processed melody tensor and its duration in seconds
        """
        # Create output directory if needed
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
        
        print(f"Processing melody: {melody_file}")
        waveform, sample_rate = torchaudio.load(melody_file)
        
        # Save original audio information
        melody_info = {
            'original_file': melody_file,
            'original_sample_rate': sample_rate,
            'original_channels': waveform.shape[0],
            'original_duration': waveform.shape[1] / sample_rate
        }
        
        # Processing steps with detailed logging
        processing_steps = []
        
        # Step 1: Convert to mono if stereo
        if waveform.shape[0] > 1:
            processing_steps.append("Converting stereo to mono (averaged channels)")
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Step 2: Resample to 32kHz (MusicGen's required sample rate)
        if sample_rate != 32000:
            processing_steps.append(f"Resampling from {sample_rate}Hz to 32000Hz")
            resampler = torchaudio.transforms.Resample(sample_rate, 32000)
            waveform = resampler(waveform)
            sample_rate = 32000
        
        # Step 3: Normalize audio to reasonable level
        peak = torch.abs(waveform).max()
        if peak > 1.0:
            processing_steps.append(f"Peak normalizing (original peak: {peak:.2f})")
            waveform = waveform / peak
        
        # Calculate duration
        duration = waveform.shape[1] / sample_rate
        melody_info['processed_duration'] = duration
        melody_info['processing_steps'] = processing_steps
        
        # Perform melody analysis if requested
        if analyze and output_dir is not None:
            melody_info['analysis'] = self._analyze_melody(waveform[0], sample_rate, output_dir)
        
        # Save the processed melody if output directory specified
        if output_dir is not None:
            processed_path = os.path.join(output_dir, "processed_melody.wav")
            torchaudio.save(
                processed_path,
                waveform,
                sample_rate
            )
            melody_info['processed_file'] = processed_path
            
            # Save information about the melody
            info_path = os.path.join(output_dir, "melody_info.json")
            with open(info_path, 'w') as f:
                json.dump(melody_info, f, indent=2)
                
            print(f"Processed melody saved to {processed_path}")
        
        # Move to appropriate device
        waveform = waveform.to(self.device)
        
        return waveform, duration, melody_info
    
    def _analyze_melody(self, waveform, sample_rate, output_dir):
        """
        Perform analysis on the melody to extract useful features.
        
        This includes waveform visualization, spectral analysis, and
        volume envelope detection.
        """
        analysis_results = {}
        
        # 1. Create waveform visualization
        plt.figure(figsize=(10, 4))
        plt.plot(waveform.cpu().numpy())
        plt.title("Melody Waveform")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.tight_layout()
        waveform_path = os.path.join(output_dir, "melody_waveform.png")
        plt.savefig(waveform_path)
        plt.close()
        analysis_results['waveform_plot'] = waveform_path
        
        # 2. Compute and visualize spectrogram
        spec = torchaudio.transforms.Spectrogram()(waveform.cpu())
        spec_db = torchaudio.transforms.AmplitudeToDB()(spec)
        
        plt.figure(figsize=(10, 4))
        plt.imshow(spec_db.numpy()[0], aspect='auto', origin='lower')
        plt.title("Melody Spectrogram")
        plt.xlabel("Time Frame")
        plt.ylabel("Frequency Bin")
        plt.colorbar(format="%+2.0f dB")
        plt.tight_layout()
        spec_path = os.path.join(output_dir, "melody_spectrogram.png")
        plt.savefig(spec_path)
        plt.close()
        analysis_results['spectrogram_plot'] = spec_path
        
        # 3. Compute amplitude envelope
        window_size = int(sample_rate * 0.03)  # 30ms windows
        hop_length = window_size // 2
        
        # Compute RMS energy
        waveform_numpy = waveform.cpu().numpy()
        n_frames = 1 + (len(waveform_numpy) - window_size) // hop_length
        envelope = np.zeros(n_frames)
        
        for i in range(n_frames):
            start = i * hop_length
            end = start + window_size
            frame = waveform_numpy[start:end]
            envelope[i] = np.sqrt(np.mean(frame**2))
        
        # Plot envelope
        plt.figure(figsize=(10, 4))
        time_axis = np.arange(len(envelope)) * hop_length / sample_rate
        plt.plot(time_axis, envelope)
        plt.title("Amplitude Envelope")
        plt.xlabel("Time (s)")
        plt.ylabel("RMS Amplitude")
        plt.grid(True)
        plt.tight_layout()
        envelope_path = os.path.join(output_dir, "melody_envelope.png")
        plt.savefig(envelope_path)
        plt.close()
        analysis_results['envelope_plot'] = envelope_path
        
        # Calculate some basic statistics
        analysis_results['stats'] = {
            'peak_amplitude': float(np.max(np.abs(waveform_numpy))),
            'rms_level': float(np.sqrt(np.mean(waveform_numpy**2))),
            'crest_factor': float(np.max(np.abs(waveform_numpy)) / np.sqrt(np.mean(waveform_numpy**2))),
            'zero_crossings': int(np.sum(np.abs(np.diff(np.signbit(waveform_numpy))))),
        }
        
        return analysis_results
    
    def transform_melody(self, melody_file, prompt, interpretation="balanced", 
                         duration=None, output_dir=None):
        """
        Transform a melody with a specific prompt and interpretation style.
        
        Args:
            melody_file (str): Path to the melody audio file
            prompt (str): Text description of desired musical style
            interpretation (str): How to interpret the melody: "accurate", 
                                 "balanced", or "creative"
            duration (float): Optional duration override (uses melody length if None)
            output_dir (str): Directory to save generation
            
        Returns:
            dict: Information about the generated music
        """
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_dir is None:
            melody_name = os.path.splitext(os.path.basename(melody_file))[0]
            output_dir = f"melody_transform_{melody_name}_{timestamp}"
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Process the melody file
        melody, melody_duration, melody_info = self.process_melody(
            melody_file, 
            output_dir=output_dir,
            analyze=True
        )
        
        # Use melody duration if no explicit duration is provided
        if duration is None:
            duration = melody_duration
            print(f"Using melody length as generation duration: {duration:.2f}s")
        
        # Get parameters based on interpretation style
        if interpretation in self.parameter_presets:
            params = self.parameter_presets[interpretation].copy()
        else:
            # Default to balanced if invalid interpretation specified
            print(f"Warning: Unknown interpretation '{interpretation}', using balanced")
            params = self.parameter_presets["balanced"].copy()
        
        # Set duration in parameters
        params['duration'] = duration
        
        # Set generation parameters
        self.model.set_generation_params(**params)
        
        # Save information about the generation
        generation_info = {
            'melody_file': melody_file,
            'prompt': prompt,
            'interpretation': interpretation,
            'parameters': params,
            'model_size': self.model_size,
            'timestamp': timestamp
        }
        
        # Add melody info (excluding redundant model attributes)
        generation_info['melody_info'] = melody_info
        
        # Generate music conditioned on both text and melody
        print(f"Generating with prompt: '{prompt}'")
        print(f"Using {interpretation} interpretation with parameters: {params}")
        
        # Prepare melody with batch dimension for generation
        melody_input = melody.unsqueeze(0)  # Add batch dimension
        
        # Generate with melody conditioning
        wav = self.model.generate_with_chroma(
            descriptions=[prompt],
            chroma=melody_input
        )
        
        # Save the generated audio
        output_path = os.path.join(output_dir, "melody_transformation")
        audio_write(
            output_path,
            wav[0].cpu(),
            self.model.sample_rate,
            strategy="loudness"
        )
        
        # Update generation info with output path
        generation_info['output_file'] = f"{output_path}.wav"
        
        # Save generation metadata
        metadata_path = os.path.join(output_dir, "generation_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(generation_info, f, indent=2)
        
        print(f"Generation complete! Output saved to {output_path}.wav")
        return generation_info
    
    def transform_with_style_variations(self, melody_file, style_category, 
                                       interpretation="balanced", output_dir=None):
        """
        Transform a melody using all prompts from a style category.
        
        Args:
            melody_file (str): Path to the melody audio file
            style_category (str): Category from self.style_presets
            interpretation (str): Interpretation style for parameters
            output_dir (str): Directory to save all variations
            
        Returns:
            list: Information about all generated variations
        """
        # Check if style category exists
        if style_category not in self.style_presets:
            print(f"Error: Style category '{style_category}' not found")
            print(f"Available categories: {list(self.style_presets.keys())}")
            return None
            
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_dir is None:
            melody_name = os.path.splitext(os.path.basename(melody_file))[0]
            output_dir = f"melody_style_{style_category}_{melody_name}_{timestamp}"
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Process the melody once for all variations
        melody_dir = os.path.join(output_dir, "original_melody")
        os.makedirs(melody_dir, exist_ok=True)
        
        melody, melody_duration, melody_info = self.process_melody(
            melody_file, 
            output_dir=melody_dir,
            analyze=True
        )
        
        # Get style variations from the selected category
        styles = self.style_presets[style_category]
        
        # Generate each style variation
        results = []
        
        for style_name, prompt in styles.items():
            print(f"Generating {style_category} - {style_name} variation")
            
            # Create subdirectory for this variation
            variation_dir = os.path.join(output_dir, f"{style_name}")
            os.makedirs(variation_dir, exist_ok=True)
            
            # Create symbolic link to original melody analysis
            # for link_file in os.listdir(melody_dir):
            #     os.symlink(
            #         os.path.join(melody_dir, link_file),
            #         os.path.join(variation_dir, link_file)
            #     )
            
            # Transform with this prompt
            result = self.transform_melody(
                melody_file=melody_file,
                prompt=prompt,
                interpretation=interpretation,
                output_dir=variation_dir
            )
            
            results.append({
                'style_name': style_name,
                'prompt': prompt,
                'output_file': result['output_file']
            })
        
        # Create a comparison README
        readme_path = os.path.join(output_dir, "README.txt")
        with open(readme_path, 'w') as f:
            f.write(f"{style_category.upper()} STYLE VARIATIONS\n")
            f.write("="*len(f"{style_category.upper()} STYLE VARIATIONS")+"\n\n")
            f.write(f"Original melody: {melody_file}\n")
            f.write(f"Interpretation style: {interpretation}\n\n")
            f.write("Generated variations:\n\n")
            
            for result in results:
                f.write(f"- {result['style_name']}\n")
                f.write(f"  Prompt: \"{result['prompt']}\"\n")
                f.write(f"  Output: {os.path.basename(result['output_file'])}\n\n")
            
            f.write("\nListen to each variation to compare how different prompts\n")
            f.write(f"within the {style_category} category affect the arrangement of\n")
            f.write("the same melody.\n")
        
        return results
    
    def explore_interpretations(self, melody_file, prompt, output_dir=None):
        """
        Generate variations with different interpretation parameters.
        
        This method generates the same melody and prompt with different
        parameter settings to demonstrate the spectrum from accurate melody
        reproduction to creative interpretation.
        """
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_dir is None:
            melody_name = os.path.splitext(os.path.basename(melody_file))[0]
            output_dir = f"melody_interpretations_{melody_name}_{timestamp}"
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate with each interpretation style
        results = []
        
        for interpretation in ["accurate", "balanced", "creative"]:
            print(f"Generating with {interpretation} interpretation")
            
            # Create subdirectory for this interpretation
            interp_dir = os.path.join(output_dir, interpretation)
            
            # Transform with this interpretation
            result = self.transform_melody(
                melody_file=melody_file,
                prompt=prompt,
                interpretation=interpretation,
                output_dir=interp_dir
            )
            
            results.append({
                'interpretation': interpretation,
                'parameters': result['parameters'],
                'output_file': result['output_file']
            })
        
        # Create a comparison README
        readme_path = os.path.join(output_dir, "README.txt")
        with open(readme_path, 'w') as f:
            f.write("MELODY INTERPRETATION COMPARISON\n")
            f.write("===============================\n\n")
            f.write(f"Original melody: {melody_file}\n")
            f.write(f"Prompt: \"{prompt}\"\n\n")
            f.write("Interpretation styles:\n\n")
            
            for result in results:
                f.write(f"- {result['interpretation'].capitalize()}\n")
                f.write(f"  Parameters: temperature={result['parameters']['temperature']}, ")
                f.write(f"cfg_coef={result['parameters']['cfg_coef']}, ")
                f.write(f"top_k={result['parameters']['top_k']}\n")
                f.write(f"  Output: {os.path.basename(result['output_file'])}\n\n")
            
            f.write("\nInterpretation spectrum:\n")
            f.write("- Accurate: Closely follows the melody with minimal creative license\n")
            f.write("- Balanced: Maintains the core melody while adding appropriate accompaniment\n")
            f.write("- Creative: Uses the melody as inspiration for a more elaborate composition\n")
        
        return results
    
    def create_custom_parameter_preset(self, name, temperature, top_k, top_p, cfg_coef, description=""):
        """Create and save a custom parameter preset for melody conditioning."""
        preset = {
            'temperature': temperature,
            'top_k': top_k,
            'top_p': top_p,
            'cfg_coef': cfg_coef
        }
        
        # Add to parameter presets
        self.parameter_presets[name] = preset
        
        # Return the new preset
        return {
            'name': name,
            'parameters': preset,
            'description': description
        }

# Example usage
if __name__ == "__main__":
    pipeline = MelodyTransformationPipeline(model_size="medium")
    
    # Example: Transform a melody with orchestral styling
    # pipeline.transform_melody(
    #     melody_file="path/to/your/melody.wav",
    #     prompt="A cinematic orchestral arrangement with strings, brass, and dramatic percussion",
    #     interpretation="balanced",
    #     output_dir="orchestral_transformation"
    # )
    
    # Example: Generate variations within a style category
    pipeline.transform_with_style_variations(
        melody_file="path/to/your/melody.wav",
        style_category="orchestral",
        interpretation="balanced",
        output_dir="orchestral_variations"
    )
    
    # Example: Compare different interpretation parameters
    # pipeline.explore_interpretations(
    #     melody_file="path/to/your/melody.wav",
    #     prompt="A jazz ensemble with piano, bass, drums, and saxophone",
    #     output_dir="interpretation_comparison"
    # )
```

This comprehensive pipeline provides everything you need for melody-based music generation:

1. **Melody preprocessing**: Thorough audio preparation for optimal results
2. **Melody analysis**: Visualizations and statistics to understand your input
3. **Style variations**: Predefined style categories with specialized prompts
4. **Interpretation control**: Parameter presets for different levels of melody adherence
5. **Thorough documentation**: Detailed output information and README files

## Common Pitfalls and Solutions

| Pitfall | Symptom | Solution |
|---------|---------|----------|
| Low-quality melody recording | Generated output has artifacts or ignores melody | Clean up recording, remove background noise, use higher sample rate |
| Melody is too complex | Output loses structure, doesn't follow melody well | Simplify the melody, use monophonic instruments, avoid dense chords |
| Melody is too long | Model loses coherence after 20-30 seconds | Split longer melodies into sections, generate separately |
| Too much specificity in prompt | Model ignores melody to follow prompt details | Simplify prompt, focus on style not specific instruments |
| Too little specificity in prompt | Output follows melody but lacks character | Add more style details to prompt, specify genre and mood |
| Temperature too high | Melody structure gets lost in randomness | Lower temperature to 0.6-0.8 for better melody following |
| Temperature too low | Output is repetitive, lacks development | Increase temperature to 0.9-1.1 for more variation |

### Troubleshooting Guide

If your melody conditioning isn't working as expected:

1. **Melody isn't recognized**: 
   - Ensure audio is mono, not stereo
   - Check that sample rate is exactly 32kHz after processing
   - Make sure melody is clearly audible and distinct

2. **Melody is recognized but not followed well**:
   - Try "accurate" interpretation parameters
   - Use simpler prompts that don't compete with melody
   - Ensure your melody has clear note onsets

3. **Output is boring or too simple**:
   - Try "creative" interpretation parameters
   - Use more descriptive prompts with specific style elements
   - Increase temperature slightly

4. **Output ignores parts of the melody**:
   - Shorten melody if it's over 15 seconds
   - Make sure all parts of melody are similar volume
   - Try using a more prominent instrument for the melody

## Hands-on Challenge: Style Transformation Suite

Now it's time to apply what you've learned to create a comprehensive style transformation for a melody:

1. Select or record a simple melody (8-15 seconds, monophonic)
2. Process the melody for optimal conditioning
3. Create at least three stylistic variations in different genres
4. Experiment with different interpretation parameters
5. Create a custom parameter preset optimized for your specific melody
6. Compare the results and analyze how well each variation preserves the melody

**Bonus Challenge**: Create a "production-ready" version by:
1. Generating multiple variations with different seeds
2. Selecting the best generation
3. Post-processing with EQ and reverb
4. Extending the arrangement by generating multiple sections

## Key Takeaways

- Melody conditioning provides a powerful way to maintain creative control while leveraging AI
- The combination of text prompts and melody conditioning offers precise creative direction
- Audio preprocessing is critical for effective melody conditioning
- Different parameter settings affect how strictly the model follows the melody
- Style variations allow exploring different genres while preserving your core musical ideas
- Melody conditioning works best with clean, monophonic melodies
- The interpretation spectrum from "accurate" to "creative" offers flexibility in output style

## Next Steps

Now that you've mastered melody conditioning, you're ready to move on to more advanced MusicGen features:

- **Chapter 9: Batch Processing and Automation** - Scale up your music generation workflow
- **Chapter 10: Introduction to AudioGen** - Move beyond music to sound effect generation

## Further Reading

- [AudioCraft GitHub: Melody Conditioning](https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md#using-melody-conditioning)
- [MusicGen: Simple and Controllable Music Generation](https://arxiv.org/abs/2306.05284)
- [Chromagram Feature Extraction](https://librosa.org/doc/main/generated/librosa.feature.chroma_stft.html)
- [Audio Preprocessing Techniques](https://towardsdatascience.com/audio-deep-learning-preprocessing-268852326ca5)
- [AI Assisted Music Production](https://arxiv.org/abs/2206.09358)
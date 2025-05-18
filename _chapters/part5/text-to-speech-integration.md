---
layout: chapter
title: "Chapter 19: Text-to-Speech Integration"
# Copyright © 2025 Scott Friedman.
# Licensed under Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.
difficulty: advanced
estimated_time: 3 hours
---

> "I need to create voice-driven experiences with background music and sound effects, but integrating separate systems is causing sync issues and inconsistent quality. I'm spending too much time manually assembling audio components instead of focusing on content." — *Maya Rodriguez, Interactive Media Producer*

# Chapter 19: Text-to-Speech Integration

## The Challenge

While AudioCraft excels at generating music and environmental sounds, it doesn't natively handle voice synthesis. Yet many practical applications—from interactive experiences to educational content—require a combination of spoken narration, background music, and sound effects. Developers and content creators typically end up cobbling together multiple disconnected systems, resulting in workflow inefficiencies, synchronization problems, and inconsistent audio quality.

The ideal solution integrates high-quality Text-to-Speech (TTS) capabilities with AudioCraft's generative audio systems within a unified framework that handles timing, mixing, and production automatically. This integration should preserve the expressive capabilities of modern TTS systems while leveraging AudioCraft's strengths in creating rich environmental sounds and music.

In this chapter, you'll learn how to build a complete integrated audio narrative pipeline that combines advanced TTS systems with AudioCraft models. We'll show you how to select the right TTS technology for your needs, synchronize audio elements with precision, and create production-ready audio experiences that blend voice, music, and sound effects seamlessly.

## Learning Objectives

By the end of this chapter, you'll be able to:

- Evaluate and implement different TTS systems based on their strengths and compatibility with AudioCraft
- Design a unified pipeline that integrates TTS with music and sound effect generation
- Synchronize voice narration with background elements using automated timeline management
- Control emotion, pacing, and style in voice narration to complement generated audio
- Create complete audio narratives with professional quality mixing and mastering

## Prerequisites

Before proceeding, ensure you have:
- Completed the chapters on basic MusicGen and AudioGen usage
- Completed Chapter 18 on building a complete audio pipeline
- Understanding of audio processing concepts (sample rates, mixing, normalization)
- Familiarity with Python and API integration

## Key Concepts

### Modern TTS Technologies

Text-to-Speech has evolved dramatically in recent years, moving from robotic-sounding voices to highly natural speech synthesis with emotional expression. Modern TTS systems leverage deep learning to generate speech that's increasingly difficult to distinguish from human recordings. These systems vary in their approach, capabilities, and integration requirements.

The ideal TTS system for AudioCraft integration should provide:
1. High-quality, natural-sounding voices
2. Emotional expression and prosody control
3. Reasonable generation speed
4. Straightforward API or local integration
5. Voice variety or voice cloning capabilities

Several TTS systems meet these criteria to varying degrees. Bark by Suno AI offers exceptional expressivity through its innovative prompt format. ElevenLabs provides industry-leading voice quality with fine-grained control. XTTS/YourTTS offers voice cloning with multilingual support. Tortoise TTS provides high-quality results with open architecture, and Facebook's MMS offers integration potential with other Meta AI tools.

```python
# Conceptual comparison of TTS systems
tts_options = {
    "Bark": {
        "quality": "Very Good",
        "expressivity": "Excellent",
        "speed": "Moderate",
        "integration": "Local (pip install)",
        "voice_variety": "Good (prompt-based)"
    },
    "ElevenLabs": {
        "quality": "Excellent",
        "expressivity": "Very Good",
        "speed": "Fast (API)",
        "integration": "API only",
        "voice_variety": "Excellent (100+ voices + cloning)"
    },
    "XTTS/YourTTS": {
        "quality": "Good",
        "expressivity": "Moderate",
        "speed": "Moderate",
        "integration": "Local (pip install)",
        "voice_variety": "Good (cloning-based)"
    }
}
```

### Audio Narrative Architecture

An effective audio narrative architecture combines voice synthesis with background elements in a way that maintains synchronization, balances audio levels, and creates a cohesive listening experience. The architecture has several key components:

1. **Narrative Script**: A structured representation of the audio experience, including voice content, timing information, and background elements
2. **Voice Generation**: Converting text to expressive speech
3. **Background Generation**: Creating music and environmental sounds
4. **Timeline Management**: Aligning all audio elements in time
5. **Audio Mixing**: Combining elements with appropriate levels and processing

This architecture enables a declarative approach to creating audio narratives, where you specify what you want rather than manually executing each production step.

```python
# Conceptual example of a narrative script
narrative_script = [
    {"type": "voice", "text": "Welcome to our adventure", "emotion": "excited", "start_time": 0.0},
    {"type": "music", "description": "Upbeat adventure music", "start_time": 0.5, "duration": 30.0},
    {"type": "sfx", "description": "Birds chirping", "start_time": 3.0, "duration": 10.0},
    {"type": "voice", "text": "Listen to the birds!", "emotion": "happy", "start_time": 5.0},
]
```

## Solution Walkthrough

### 1. Setting Up TTS Integration

Let's begin by setting up a basic integration between Bark TTS and AudioCraft. Bark provides expressive voice synthesis and can be installed locally, making it a good starting point.

```python
# tts_audiocraft_integration.py - Basic integration between Bark TTS and AudioCraft
import torch
import numpy as np
import torchaudio
import scipy.io.wavfile as wavfile
import os
from audiocraft.models import MusicGen, AudioGen
from audiocraft.data.audio import audio_write
from bark import SAMPLE_RATE, generate_audio, preload_models

def setup_device():
    """Set up the appropriate device for AudioCraft models"""
    if torch.backends.mps.is_available():
        device = "mps"  # Apple Silicon GPU
        print("Using MPS (Metal) for generation")
    elif torch.cuda.is_available():
        device = "cuda"  # NVIDIA GPU
        print("Using CUDA for generation")
    else:
        device = "cpu"  # Fallback to CPU
        print("Using CPU for generation (this will be slow)")
    return device

def generate_voice(text, output_path="voice_output.wav"):
    """Generate speech using Bark TTS"""
    print(f"Generating voice narration: '{text[:50]}...'")
    
    # Generate speech using Bark
    preload_models()  # Takes time on first run
    speech_array = generate_audio(text)
    
    # Save as WAV file
    wavfile.write(output_path, SAMPLE_RATE, speech_array)
    print(f"Voice saved to {output_path}")
    
    return speech_array, SAMPLE_RATE
```

### 2. Adding Music and Sound Effect Generation

Now, let's add functions to generate background music and sound effects using AudioCraft models:

```python
def generate_background_music(prompt, duration=10.0, output_path="music_output.wav"):
    """Generate background music using MusicGen"""
    print(f"Generating background music: '{prompt}'")
    
    # Set up device
    device = setup_device()
    
    # Load MusicGen model
    model = MusicGen.get_pretrained('small')
    model.to(device)
    
    # Set generation parameters
    model.set_generation_params(
        duration=duration,
        temperature=0.9,
        top_k=250,
        top_p=0.0,
    )
    
    # Generate music
    wav = model.generate([prompt])
    
    # Save the music
    audio_write(
        output_path.replace('.wav', ''),
        wav[0].cpu(),
        model.sample_rate,
        strategy="loudness"
    )
    print(f"Music saved to {output_path}")
    
    return wav[0].cpu().numpy(), model.sample_rate

def generate_sound_effect(prompt, duration=5.0, output_path="sfx_output.wav"):
    """Generate sound effect using AudioGen"""
    print(f"Generating sound effect: '{prompt}'")
    
    # Set up device
    device = setup_device()
    
    # Load AudioGen model
    model = AudioGen.get_pretrained('medium')
    model.to(device)
    
    # Set generation parameters
    model.set_generation_params(
        duration=duration,
        temperature=1.0,
        top_k=250,
        top_p=0.0,
    )
    
    # Generate sound effect
    wav = model.generate([prompt])
    
    # Save the sound effect
    audio_write(
        output_path.replace('.wav', ''),
        wav[0].cpu(),
        model.sample_rate,
        strategy="loudness"
    )
    print(f"Sound effect saved to {output_path}")
    
    return wav[0].cpu().numpy(), model.sample_rate
```

### 3. Building the Audio Mixing System

Next, we need a system to combine voice, music, and sound effects into a single cohesive audio file:

```python
def combine_audio_tracks(voice_path, music_path, sfx_path=None, output_path="combined_output.wav"):
    """Combine voice, music, and optional sound effect into a single audio file"""
    print("Combining audio tracks...")
    
    # Load the audio files
    voice, voice_sr = torchaudio.load(voice_path)
    music, music_sr = torchaudio.load(music_path)
    
    # Convert to mono if stereo
    voice = voice[0] if voice.shape[0] > 1 else voice
    music = music[0] if music.shape[0] > 1 else music
    
    # Resample to the same sample rate (use the voice sample rate)
    if music_sr != voice_sr:
        music = torchaudio.functional.resample(music, music_sr, voice_sr)
    
    # Make all audio tracks the same length (use the voice length)
    voice_length = voice.shape[0]
    
    # Trim or pad music
    if music.shape[0] > voice_length:
        music = music[:voice_length]
    else:
        padding = torch.zeros(voice_length - music.shape[0])
        music = torch.cat([music, padding])
    
    # Initialize the combined audio with voice and music
    voice_weight = 1.0
    music_weight = 0.3  # Adjust as needed
    
    combined = (voice * voice_weight) + (music * music_weight)
    
    # Add sound effects if provided
    if sfx_path:
        sfx, sfx_sr = torchaudio.load(sfx_path)
        sfx = sfx[0] if sfx.shape[0] > 1 else sfx
        
        if sfx_sr != voice_sr:
            sfx = torchaudio.functional.resample(sfx, sfx_sr, voice_sr)
        
        # Trim or pad sfx
        if sfx.shape[0] > voice_length:
            sfx = sfx[:voice_length]
        else:
            padding = torch.zeros(voice_length - sfx.shape[0])
            sfx = torch.cat([sfx, padding])
        
        # Add to the mix
        sfx_weight = 0.5  # Adjust as needed
        combined = combined + (sfx * sfx_weight)
    
    # Normalize to prevent clipping
    max_val = torch.max(torch.abs(combined))
    if max_val > 1.0:
        combined = combined / max_val
    
    # Save the combined audio
    torchaudio.save(output_path, combined.unsqueeze(0), voice_sr)
    print(f"Combined audio saved to {output_path}")
    
    return output_path
```

### 4. Creating an Integrated Narrative Pipeline

Finally, let's integrate everything into a timeline-based narrative pipeline that can handle a script with precise timing:

```python
def create_timed_audio_experience(script, output_path="timed_output.wav"):
    """
    Creates a synchronized audio experience from a timed script
    
    Script format:
    [
        {"type": "voice", "text": "Welcome to the jungle", "start_time": 0.0},
        {"type": "sfx", "description": "jungle ambience", "start_time": 0.5, "duration": 10.0},
        {"type": "music", "description": "tribal drums", "start_time": 2.0, "duration": 8.0}
    ]
    """
    # Sort script by start time
    script.sort(key=lambda x: x["start_time"])
    
    # Determine total duration
    total_duration = 0
    for item in script:
        end_time = item["start_time"]
        if "duration" in item:
            end_time += item["duration"]
        total_duration = max(total_duration, end_time)
    
    # Add a small buffer at the end
    total_duration += 2.0
    
    # Initialize an empty audio canvas (44.1kHz sample rate)
    sample_rate = 44100
    total_samples = int(total_duration * sample_rate)
    canvas = np.zeros(total_samples)
    
    # Process each script item
    for i, item in enumerate(script):
        start_sample = int(item["start_time"] * sample_rate)
        
        if item["type"] == "voice":
            # Generate voice
            voice_path = f"temp_voice_{i}.wav"
            voice_array, voice_sr = generate_voice(item["text"], voice_path)
            voice_resampled = resample_audio(voice_array, voice_sr, sample_rate)
            
            # Add to canvas
            end_sample = min(start_sample + len(voice_resampled), total_samples)
            canvas[start_sample:end_sample] += voice_resampled[:end_sample-start_sample] * 1.0  # Full volume
            
        elif item["type"] == "sfx":
            # Generate sound effect
            sfx_path = f"temp_sfx_{i}.wav"
            sfx_array, sfx_sr = generate_sound_effect(
                item["description"], 
                item.get("duration", 5.0),
                sfx_path
            )
            sfx_resampled = resample_audio(sfx_array, sfx_sr, sample_rate)
            
            # Add to canvas
            end_sample = min(start_sample + len(sfx_resampled), total_samples)
            canvas[start_sample:end_sample] += sfx_resampled[:end_sample-start_sample] * 0.5  # Half volume
            
        elif item["type"] == "music":
            # Generate music
            music_path = f"temp_music_{i}.wav"
            music_array, music_sr = generate_background_music(
                item["description"],
                item.get("duration", 10.0),
                music_path
            )
            music_resampled = resample_audio(music_array, music_sr, sample_rate)
            
            # Add to canvas
            end_sample = min(start_sample + len(music_resampled), total_samples)
            canvas[start_sample:end_sample] += music_resampled[:end_sample-start_sample] * 0.3  # Lower volume
    
    # Normalize to prevent clipping
    max_amplitude = np.max(np.abs(canvas))
    if max_amplitude > 1.0:
        canvas = canvas / max_amplitude
    
    # Save the final audio
    wavfile.write(output_path, sample_rate, canvas.astype(np.float32))
    print(f"Timed audio experience saved to {output_path}")
    
    # Clean up temporary files
    for i in range(len(script)):
        for prefix in ["temp_voice_", "temp_sfx_", "temp_music_"]:
            temp_path = f"{prefix}{i}.wav"
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    return output_path

def resample_audio(audio_array, orig_sr, target_sr):
    """Resample audio to the target sample rate"""
    # Convert to torch tensor for resampling
    if isinstance(audio_array, np.ndarray):
        audio_tensor = torch.from_numpy(audio_array).float()
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
    else:
        audio_tensor = audio_array
        
    # Resample
    resampled = torchaudio.functional.resample(audio_tensor, orig_sr, target_sr)
    
    # Return as numpy array
    if resampled.dim() > 1:
        resampled = resampled.squeeze(0)
    return resampled.numpy()
```

## Complete Implementation

Let's put everything together into a complete, runnable example:

```python
#!/usr/bin/env python3
# audio_narrative_pipeline.py - Complete TTS and AudioCraft integration pipeline
import torch
import numpy as np
import torchaudio
import scipy.io.wavfile as wavfile
import os
from audiocraft.models import MusicGen, AudioGen
from audiocraft.data.audio import audio_write
from bark import SAMPLE_RATE, generate_audio, preload_models

class AudioNarrativePipeline:
    """
    A comprehensive pipeline for creating audio narratives with voice, music, and sound effects.
    
    This class integrates text-to-speech technology with AudioCraft's music and sound
    generation capabilities to create complete audio experiences from structured scripts.
    
    Key features:
    - Voice generation with emotion control
    - Music and sound effect generation
    - Timeline-based composition
    - Automatic audio mixing and normalization
    - Memory-efficient processing
    
    Example usage:
        pipeline = AudioNarrativePipeline()
        script = [
            {"type": "voice", "text": "Welcome to our journey", "start_time": 0.0},
            {"type": "music", "description": "Ambient music", "start_time": 0.5, "duration": 20.0}
        ]
        pipeline.create_narrative(script, "my_narrative.wav")
    """
    def __init__(self, use_gpu=True, voice_system="bark"):
        """
        Initialize the audio narrative pipeline.
        
        Args:
            use_gpu (bool): Whether to use GPU acceleration if available
            voice_system (str): TTS system to use ('bark', 'elevenlabs', etc.)
        """
        # Determine device for AudioCraft models
        self.device = self._setup_device() if use_gpu else "cpu"
        self.voice_system = voice_system
        
        # Store models for reuse
        self.models = {}
        
        # Track if TTS models are preloaded
        self.tts_initialized = False
        
        print(f"AudioNarrativePipeline initialized using {self.voice_system} for voice and device: {self.device}")
    
    def _setup_device(self):
        """Determine the best available compute device"""
        if torch.backends.mps.is_available():
            return "mps"  # Apple Silicon GPU
        elif torch.cuda.is_available():
            return "cuda"  # NVIDIA GPU
        return "cpu"  # Fallback to CPU
    
    def _ensure_tts_initialized(self):
        """Ensure TTS models are loaded"""
        if not self.tts_initialized and self.voice_system == "bark":
            print("Initializing Bark TTS models (this may take a moment)...")
            preload_models()
            self.tts_initialized = True
    
    def _get_audiocraft_model(self, model_type, model_size):
        """
        Get or load an AudioCraft model.
        
        Args:
            model_type (str): 'music' or 'audio'
            model_size (str): Model size ('small', 'medium', 'large')
            
        Returns:
            The loaded model
        """
        model_key = f"{model_type}_{model_size}"
        
        if model_key not in self.models:
            print(f"Loading {model_type} model ({model_size})...")
            
            if model_type == "music":
                model = MusicGen.get_pretrained(model_size)
            elif model_type == "audio":
                model = AudioGen.get_pretrained(model_size)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
                
            model.to(self.device)
            self.models[model_key] = model
            
        return self.models[model_key]
    
    def generate_voice(self, text, emotion=None, output_path="voice_output.wav"):
        """
        Generate voice narration from text.
        
        Args:
            text (str): Text to synthesize
            emotion (str, optional): Emotion to express (used with supported TTS systems)
            output_path (str): Path to save the generated audio
            
        Returns:
            tuple: (audio_array, sample_rate)
        """
        self._ensure_tts_initialized()
        
        print(f"Generating voice: '{text[:50]}...'")
        
        if self.voice_system == "bark":
            # Format prompt with emotion if provided
            if emotion:
                formatted_text = f"[emotion: {emotion}] {text}"
            else:
                formatted_text = text
                
            # Generate speech
            speech_array = generate_audio(formatted_text)
            
            # Save output
            wavfile.write(output_path, SAMPLE_RATE, speech_array)
            return speech_array, SAMPLE_RATE
            
        # Additional voice systems could be added here
        else:
            raise ValueError(f"Unsupported voice system: {self.voice_system}")
    
    def generate_music(self, prompt, duration=10.0, temperature=0.9, output_path="music_output.wav"):
        """
        Generate background music using MusicGen.
        
        Args:
            prompt (str): Text description of the music
            duration (float): Length in seconds
            temperature (float): Creativity control parameter
            output_path (str): Path to save the generated audio
            
        Returns:
            tuple: (audio_array, sample_rate)
        """
        print(f"Generating music: '{prompt}'")
        
        # Get the MusicGen model
        model = self._get_audiocraft_model("music", "small")
        
        # Configure generation parameters
        model.set_generation_params(
            duration=duration,
            temperature=temperature,
            top_k=250,
            top_p=0.0,
        )
        
        # Generate music
        with torch.no_grad():
            wav = model.generate([prompt])
        
        # Get sample rate
        sample_rate = model.sample_rate
        
        # Save the output
        audio_write(
            output_path.replace('.wav', ''),
            wav[0].cpu(),
            sample_rate,
            strategy="loudness"
        )
        
        return wav[0].cpu().numpy(), sample_rate
    
    def generate_sound_effect(self, prompt, duration=5.0, temperature=1.0, output_path="sfx_output.wav"):
        """
        Generate sound effects using AudioGen.
        
        Args:
            prompt (str): Text description of the sound
            duration (float): Length in seconds
            temperature (float): Creativity control parameter
            output_path (str): Path to save the generated audio
            
        Returns:
            tuple: (audio_array, sample_rate)
        """
        print(f"Generating sound effect: '{prompt}'")
        
        # Get the AudioGen model
        model = self._get_audiocraft_model("audio", "medium")
        
        # Configure generation parameters
        model.set_generation_params(
            duration=duration,
            temperature=temperature,
            top_k=250,
            top_p=0.0,
        )
        
        # Generate sound effect
        with torch.no_grad():
            wav = model.generate([prompt])
        
        # Get sample rate
        sample_rate = model.sample_rate
        
        # Save the output
        audio_write(
            output_path.replace('.wav', ''),
            wav[0].cpu(),
            sample_rate,
            strategy="loudness"
        )
        
        return wav[0].cpu().numpy(), sample_rate
    
    def mix_audio_files(self, audio_files, weights=None, output_path="mixed_output.wav"):
        """
        Mix multiple audio files with specified weights.
        
        Args:
            audio_files (list): List of audio file paths
            weights (list, optional): Corresponding weights for each file
            output_path (str): Path to save the mixed audio
            
        Returns:
            str: Path to the mixed audio file
        """
        if not audio_files:
            raise ValueError("No audio files provided for mixing")
        
        # Default to equal weights if not specified
        if weights is None:
            weights = [1.0 / len(audio_files)] * len(audio_files)
        
        # Validate weights
        if len(weights) != len(audio_files):
            raise ValueError("Number of weights must match number of audio files")
        
        # Load all audio files
        loaded_audio = []
        sample_rates = []
        
        for path in audio_files:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Audio file not found: {path}")
                
            # Load the audio
            waveform, sr = torchaudio.load(path)
            loaded_audio.append(waveform)
            sample_rates.append(sr)
        
        # Use the first sample rate as reference
        reference_sr = sample_rates[0]
        
        # Convert all audio to the same format (mono, same sample rate)
        processed_audio = []
        
        for i, (audio, sr) in enumerate(zip(loaded_audio, sample_rates)):
            # Convert to mono if stereo
            if audio.shape[0] > 1:
                audio = audio[0].unsqueeze(0)
                
            # Resample if needed
            if sr != reference_sr:
                audio = torchaudio.functional.resample(audio, sr, reference_sr)
                
            processed_audio.append(audio)
        
        # Find the longest audio length
        max_length = max(audio.shape[1] for audio in processed_audio)
        
        # Mix audio with weights
        mixed_audio = torch.zeros(1, max_length)
        
        for i, (audio, weight) in enumerate(zip(processed_audio, weights)):
            # Pad shorter audio with silence
            if audio.shape[1] < max_length:
                padding = torch.zeros(1, max_length - audio.shape[1])
                audio = torch.cat([audio, padding], dim=1)
            
            # Add weighted audio to the mix
            mixed_audio += audio * weight
        
        # Normalize to prevent clipping
        max_amplitude = torch.max(torch.abs(mixed_audio))
        if max_amplitude > 1.0:
            mixed_audio = mixed_audio / max_amplitude
        
        # Save the mixed audio
        torchaudio.save(output_path, mixed_audio, reference_sr)
        print(f"Mixed audio saved to {output_path}")
        
        return output_path
    
    def create_narrative(self, script, output_path="narrative_output.wav", cleanup_temp=True):
        """
        Create a complete audio narrative from a script.
        
        Args:
            script (list): List of script elements with timing and content
            output_path (str): Path to save the final narrative
            cleanup_temp (bool): Whether to remove temporary files after processing
            
        Returns:
            str: Path to the final narrative audio file
        """
        # Sort script by start time
        script.sort(key=lambda x: x.get("start_time", 0))
        
        # Determine total duration
        total_duration = 0
        for item in script:
            end_time = item.get("start_time", 0)
            if "duration" in item:
                end_time += item["duration"]
            total_duration = max(total_duration, end_time)
        
        # Add a small buffer at the end
        total_duration += 2.0
        print(f"Creating narrative with total duration: {total_duration:.1f} seconds")
        
        # Initialize an empty audio canvas (44.1kHz sample rate)
        sample_rate = 44100
        total_samples = int(total_duration * sample_rate)
        canvas = np.zeros(total_samples)
        
        # Keep track of temporary files
        temp_files = []
        
        # Process each script item
        for i, item in enumerate(script):
            start_time = item.get("start_time", 0)
            start_sample = int(start_time * sample_rate)
            
            if item["type"] == "voice":
                # Generate voice
                temp_path = f"temp_voice_{i}.wav"
                voice_array, voice_sr = self.generate_voice(
                    item["text"],
                    item.get("emotion"),
                    temp_path
                )
                temp_files.append(temp_path)
                
                # Resample if needed
                voice_resampled = self._resample_audio(voice_array, voice_sr, sample_rate)
                
                # Add to canvas
                end_sample = min(start_sample + len(voice_resampled), total_samples)
                canvas[start_sample:end_sample] += voice_resampled[:end_sample-start_sample] * 1.0  # Full volume
                
            elif item["type"] == "music":
                # Generate music
                temp_path = f"temp_music_{i}.wav"
                music_array, music_sr = self.generate_music(
                    item["description"],
                    item.get("duration", 10.0),
                    item.get("temperature", 0.9),
                    temp_path
                )
                temp_files.append(temp_path)
                
                # Resample if needed
                music_resampled = self._resample_audio(music_array, music_sr, sample_rate)
                
                # Add to canvas
                end_sample = min(start_sample + len(music_resampled), total_samples)
                canvas[start_sample:end_sample] += music_resampled[:end_sample-start_sample] * item.get("volume", 0.3)
                
            elif item["type"] == "sfx":
                # Generate sound effect
                temp_path = f"temp_sfx_{i}.wav"
                sfx_array, sfx_sr = self.generate_sound_effect(
                    item["description"],
                    item.get("duration", 5.0),
                    item.get("temperature", 1.0),
                    temp_path
                )
                temp_files.append(temp_path)
                
                # Resample if needed
                sfx_resampled = self._resample_audio(sfx_array, sfx_sr, sample_rate)
                
                # Add to canvas
                end_sample = min(start_sample + len(sfx_resampled), total_samples)
                canvas[start_sample:end_sample] += sfx_resampled[:end_sample-start_sample] * item.get("volume", 0.5)
        
        # Normalize to prevent clipping
        max_amplitude = np.max(np.abs(canvas))
        if max_amplitude > 1.0:
            canvas = canvas / max_amplitude
        
        # Save the final audio
        wavfile.write(output_path, sample_rate, canvas.astype(np.float32))
        print(f"Audio narrative saved to {output_path}")
        
        # Clean up temporary files if requested
        if cleanup_temp:
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            print(f"Cleaned up {len(temp_files)} temporary files")
        
        return output_path
    
    def _resample_audio(self, audio_array, orig_sr, target_sr):
        """
        Resample audio to the target sample rate.
        
        Args:
            audio_array: Audio data
            orig_sr: Original sample rate
            target_sr: Target sample rate
            
        Returns:
            Resampled audio as numpy array
        """
        # Skip if already at target rate
        if orig_sr == target_sr:
            return audio_array
            
        # Convert to torch tensor for resampling
        if isinstance(audio_array, np.ndarray):
            audio_tensor = torch.from_numpy(audio_array).float()
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
        else:
            audio_tensor = audio_array
            
        # Resample
        resampled = torchaudio.functional.resample(audio_tensor, orig_sr, target_sr)
        
        # Return as numpy array
        if resampled.dim() > 1:
            resampled = resampled.squeeze(0)
        return resampled.numpy()

    def free_memory(self):
        """Release memory used by models"""
        self.models = {}
        self.tts_initialized = False
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear CUDA cache if using GPU
        if self.device == "cuda":
            torch.cuda.empty_cache()
            
        print("Memory released")

# Example usage
if __name__ == "__main__":
    # Create the pipeline
    pipeline = AudioNarrativePipeline()
    
    # Define a narrative script
    forest_journey_script = [
        # Introduction with excited voice
        {"type": "voice", "text": "Welcome to our journey through the magical forest!", 
         "emotion": "excited", "start_time": 0.0},
        
        # Background music starts slightly after voice
        {"type": "music", "description": "Peaceful fantasy music with soft flutes and mystical bells", 
         "start_time": 1.0, "duration": 20.0, "volume": 0.3},
        
        # Forest ambience fades in
        {"type": "sfx", "description": "Forest ambience with birds chirping and leaves rustling", 
         "start_time": 3.0, "duration": 17.0, "volume": 0.4},
        
        # Next voice segment
        {"type": "voice", "text": "Listen carefully to the sounds of nature all around us.", 
         "emotion": "calm", "start_time": 6.0},
        
        # Special sound effect
        {"type": "sfx", "description": "Magical shimmer sound with wind chimes", 
         "start_time": 10.0, "duration": 3.0, "volume": 0.5},
        
        # Final voice segment
        {"type": "voice", "text": "Oh! Did you hear that? I think we've discovered something magical!", 
         "emotion": "surprised", "start_time": 12.0}
    ]
    
    # Create the narrative
    pipeline.create_narrative(
        script=forest_journey_script,
        output_path="magical_forest_journey.wav"
    )
    
    # Release memory
    pipeline.free_memory()
    
    print("Narrative creation complete!")
```

## Variations and Customizations

Let's explore some variations of our solution to address different needs or preferences.

### Variation 1: Using ElevenLabs for Higher Quality Voices

ElevenLabs provides exceptionally high-quality voice synthesis. Here's how to integrate it into our pipeline:

```python
import requests
import os

class ElevenLabsVoiceGenerator:
    """Voice generation using ElevenLabs API"""
    
    def __init__(self, api_key=None):
        """
        Initialize the ElevenLabs voice generator.
        
        Args:
            api_key (str, optional): API key for ElevenLabs. If not provided,
                                    looks for ELEVEN_API_KEY environment variable.
        """
        # Get API key from environment if not provided
        self.api_key = api_key or os.environ.get("ELEVEN_API_KEY")
        if not self.api_key:
            raise ValueError("ElevenLabs API key required. Set ELEVEN_API_KEY environment variable or pass as parameter.")
    
    def generate_voice(self, text, voice_id="21m00Tcm4TlvDq8ikWAM", output_path="elevenlabs_output.wav",
                      stability=0.5, similarity_boost=0.75):
        """
        Generate speech using ElevenLabs API.
        
        Args:
            text (str): Text to convert to speech
            voice_id (str): ElevenLabs voice ID
            output_path (str): Path to save output audio
            stability (float): Voice stability (0.0-1.0)
            similarity_boost (float): Voice similarity boost (0.0-1.0)
            
        Returns:
            tuple: (Path to saved audio, sample rate)
        """
        print(f"Generating speech with ElevenLabs: '{text[:50]}...'")
        
        # API endpoint
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        
        # Request headers
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key
        }
        
        # Request data
        data = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": stability,
                "similarity_boost": similarity_boost
            }
        }
        
        # Make API call
        response = requests.post(url, json=data, headers=headers)
        
        # Check for success
        if response.status_code == 200:
            # Save audio file
            with open(output_path, "wb") as file:
                file.write(response.content)
            print(f"ElevenLabs audio saved to {output_path}")
            
            # Get audio info using torchaudio
            info = torchaudio.info(output_path)
            sample_rate = info.sample_rate
            
            return output_path, sample_rate
        else:
            # Handle error
            error_msg = f"ElevenLabs API Error: {response.status_code}"
            try:
                error_details = response.json()
                error_msg += f" - {error_details}"
            except:
                pass
            raise Exception(error_msg)
    
    def generate_voice_with_emotion(self, text, emotion="neutral", voice_id="21m00Tcm4TlvDq8ikWAM", 
                                   output_path="elevenlabs_emotion_output.wav"):
        """
        Generate speech with emotional expression.
        
        Args:
            text (str): Text to convert to speech
            emotion (str): Desired emotion (neutral, happy, sad, angry, etc.)
            voice_id (str): ElevenLabs voice ID
            output_path (str): Path to save output audio
            
        Returns:
            tuple: (Path to saved audio, sample rate)
        """
        # Map emotions to voice settings
        emotion_settings = {
            "neutral": {"stability": 0.5, "similarity_boost": 0.75},
            "happy": {"stability": 0.3, "similarity_boost": 0.65},
            "excited": {"stability": 0.2, "similarity_boost": 0.7},
            "sad": {"stability": 0.7, "similarity_boost": 0.8},
            "angry": {"stability": 0.3, "similarity_boost": 0.6},
            "fear": {"stability": 0.4, "similarity_boost": 0.7},
            "surprise": {"stability": 0.2, "similarity_boost": 0.6}
        }
        
        # Get settings for the emotion (default to neutral if not found)
        settings = emotion_settings.get(emotion.lower(), emotion_settings["neutral"])
        
        # Add emotion context to the text for better expression
        emotion_prefixes = {
            "happy": "😊 [Happily] ",
            "excited": "🤩 [Excitedly] ",
            "sad": "😢 [Sadly] ",
            "angry": "😠 [Angrily] ",
            "fear": "😨 [Fearfully] ",
            "surprise": "😲 [Surprised] ",
            "neutral": ""
        }
        
        # Add emotion prefix if available
        prefix = emotion_prefixes.get(emotion.lower(), "")
        enhanced_text = prefix + text
        
        # Generate speech with emotion-specific settings
        return self.generate_voice(
            text=enhanced_text,
            voice_id=voice_id,
            output_path=output_path,
            stability=settings["stability"],
            similarity_boost=settings["similarity_boost"]
        )

# To use in the main pipeline:
# 1. Replace the generate_voice method in AudioNarrativePipeline
# 2. Initialize the ElevenLabsVoiceGenerator in the constructor
```

### Variation 2: Advanced Script Format with Scene-Based Organization

For more complex narratives, a scene-based script format can be more manageable:

```python
def create_narrative_from_scenes(self, scenes, output_path="narrative_with_scenes.wav"):
    """
    Create an audio narrative from a scene-based script format.
    
    Args:
        scenes (list): List of scene dictionaries, each containing:
            - name: Scene name
            - description: Scene description
            - elements: List of audio elements in the scene
            - transitions: Optional transition specifications
        output_path (str): Path to save the final narrative
        
    Returns:
        str: Path to the final narrative audio file
    """
    # Convert scene-based format to flat timeline
    timeline = []
    current_time = 0
    
    for scene in scenes:
        scene_name = scene["name"]
        print(f"Processing scene: {scene_name}")
        
        # Add scene elements to timeline
        scene_elements = scene["elements"]
        scene_duration = 0
        
        for element in scene_elements:
            # Calculate element start time (relative to scene start + specified offset)
            element_start = current_time + element.get("offset", 0)
            
            # Add element to timeline with absolute start time
            timeline_element = element.copy()
            timeline_element["start_time"] = element_start
            timeline.append(timeline_element)
            
            # Track scene duration based on elements
            element_end = element_start
            if "duration" in element:
                element_end += element["duration"]
            scene_duration = max(scene_duration, element_end - current_time)
        
        # Add transition if specified
        if "transition" in scene:
            transition = scene["transition"]
            timeline.append({
                "type": "sfx",
                "description": transition.get("description", "Smooth transition effect"),
                "start_time": current_time + scene_duration - transition.get("overlap", 1.0),
                "duration": transition.get("duration", 2.0),
                "volume": transition.get("volume", 0.6)
            })
        
        # Update current time for next scene
        current_time += scene_duration
    
    # Create narrative from the flattened timeline
    return self.create_narrative(timeline, output_path)

# Example usage:
scenes = [
    {
        "name": "Forest Introduction",
        "description": "Introducing the magical forest setting",
        "elements": [
            {"type": "voice", "text": "Welcome to the magical forest!", "emotion": "excited", "offset": 0.0},
            {"type": "music", "description": "Magical forest theme with flutes", "offset": 0.5, "duration": 15.0, "volume": 0.3},
            {"type": "sfx", "description": "Forest ambience with birds", "offset": 2.0, "duration": 12.0, "volume": 0.4}
        ],
        "transition": {
            "description": "Magical shimmer transition",
            "duration": 3.0,
            "overlap": 2.0
        }
    },
    {
        "name": "Mysterious Discovery",
        "description": "Characters discover something in the forest",
        "elements": [
            {"type": "voice", "text": "What's that glowing behind the trees?", "emotion": "surprised", "offset": 0.5},
            {"type": "sfx", "description": "Mysterious magical glow sound", "offset": 2.0, "duration": 5.0, "volume": 0.5},
            {"type": "music", "description": "Suspenseful mysterious music with strings", "offset": 0.0, "duration": 20.0, "volume": 0.25}
        ]
    }
]

# pipeline.create_narrative_from_scenes(scenes, "forest_adventure.wav")
```

## Common Pitfalls and Troubleshooting

### Problem: TTS Voice Quality Issues

TTS systems can sometimes produce speech with unnatural pronunciation, especially for unusual words or names.

**Solution**:
- Use phonetic spelling or SSML markup for difficult words
- Try different voice models for the specific content type
- Break long sentences into shorter phrases with appropriate pauses:

```python
def enhance_speech_text(text):
    """Enhance text for better TTS quality"""
    # Replace difficult words with phonetic versions or SSML
    replacements = {
        "AudioCraft": "Audio Craft",
        "MusicGen": "Music Gen",
        "ElevenLabs": "Eleven Labs"
    }
    
    for word, replacement in replacements.items():
        text = text.replace(word, replacement)
    
    # Add pauses at punctuation for more natural speech
    text = text.replace(".", ".[pause]")
    text = text.replace("!", "![pause]")
    text = text.replace("?", "?[pause]")
    text = text.replace(",", ",[micropause]")
    
    return text

# Usage
enhanced_text = enhance_speech_text("Welcome to AudioCraft, the remarkable AI tool for music generation!")
```

### Problem: Audio Synchronization Issues

When creating complex narratives, timing can be challenging because TTS generation produces variable-length outputs.

**Solution**:
- First generate all TTS segments to determine their actual durations
- Adjust the timeline based on actual durations rather than estimates
- Use padding or trimming to ensure precise alignment:

```python
def create_precisely_timed_narrative(self, script, output_path="precise_narrative.wav"):
    """Create narrative with precise timing based on pre-generated TTS"""
    # First, generate all voice segments to get actual durations
    voice_segments = []
    for item in script:
        if item["type"] == "voice":
            # Generate voice
            temp_path = f"temp_voice_{len(voice_segments)}.wav"
            self.generate_voice(item["text"], item.get("emotion"), temp_path)
            
            # Get duration
            info = torchaudio.info(temp_path)
            duration = info.num_frames / info.sample_rate
            
            # Store segment info
            voice_segments.append({
                "path": temp_path,
                "duration": duration,
                "start_time": item["start_time"],
                "original_idx": script.index(item)
            })
    
    # Now adjust timeline based on actual durations
    adjusted_script = script.copy()
    for i, segment in enumerate(voice_segments):
        original_idx = segment["original_idx"]
        actual_duration = segment["duration"]
        
        # Update script with actual path and duration
        adjusted_script[original_idx]["audio_path"] = segment["path"]
        adjusted_script[original_idx]["actual_duration"] = actual_duration
    
    # Process the adjusted script
    # [Implementation would use pre-generated audio files and adjust other elements as needed]
```

### Problem: Memory Management with Multiple Models

Loading multiple large models simultaneously can cause out-of-memory errors.

**Solution**:
- Implement staged generation where only one model is loaded at a time
- Use model unloading to free resources after generation
- Process audio in smaller batches:

```python
def staged_generation(self, script, output_path="staged_output.wav"):
    """Generate audio in stages to manage memory usage"""
    # Temporary output paths
    voice_outputs = []
    music_outputs = []
    sfx_outputs = []
    
    # Stage 1: Generate all voice segments
    self._ensure_tts_initialized()
    print("Stage 1: Generating voice segments...")
    for i, item in enumerate(script):
        if item["type"] == "voice":
            temp_path = f"temp_voice_{i}.wav"
            self.generate_voice(item["text"], item.get("emotion"), temp_path)
            voice_outputs.append({"path": temp_path, "start_time": item["start_time"]})
    
    # Free TTS resources
    self.tts_initialized = False
    import gc
    gc.collect()
    
    # Stage 2: Generate all music
    print("Stage 2: Generating music...")
    model = self._get_audiocraft_model("music", "small")
    for i, item in enumerate(script):
        if item["type"] == "music":
            temp_path = f"temp_music_{i}.wav"
            duration = item.get("duration", 10.0)
            
            # Generate with MusicGen
            model.set_generation_params(duration=duration, temperature=item.get("temperature", 0.9))
            with torch.no_grad():
                wav = model.generate([item["description"]])
            
            # Save output
            audio_write(temp_path.replace('.wav', ''), wav[0].cpu(), model.sample_rate)
            music_outputs.append({"path": temp_path, "start_time": item["start_time"]})
    
    # Free music model
    del self.models["music_small"]
    gc.collect()
    
    # Stage 3: Generate all sound effects
    print("Stage 3: Generating sound effects...")
    model = self._get_audiocraft_model("audio", "medium")
    for i, item in enumerate(script):
        if item["type"] == "sfx":
            temp_path = f"temp_sfx_{i}.wav"
            duration = item.get("duration", 5.0)
            
            # Generate with AudioGen
            model.set_generation_params(duration=duration, temperature=item.get("temperature", 1.0))
            with torch.no_grad():
                wav = model.generate([item["description"]])
            
            # Save output
            audio_write(temp_path.replace('.wav', ''), wav[0].cpu(), model.sample_rate)
            sfx_outputs.append({"path": temp_path, "start_time": item["start_time"]})
    
    # Free sound effect model
    del self.models["audio_medium"]
    gc.collect()
    
    # Final stage: Mix everything
    print("Final stage: Assembling and mixing...")
    # [Implementation would assemble all generated audio files]
```

## Hands-on Challenge

Now it's your turn to experiment with what you've learned. Try the following challenge:

### Challenge: Interactive Storytelling System

Create an interactive storytelling system that:
1. Takes a story script with branching narrative options
2. Generates audio for all possible narrative paths
3. Provides an interface for making choices at decision points
4. Seamlessly transitions between narrative segments based on choices
5. Includes dynamic background music that adapts to the story context

### Bonus Challenge

Implement an emotion analysis system that identifies the emotional tone of text and automatically selects appropriate background music and sound effects based on the detected emotion.

## Key Takeaways

- Modern TTS systems can be effectively integrated with AudioCraft to create complete audio experiences
- A unified pipeline architecture simplifies the creation of complex audio narratives
- Timeline-based composition enables precise control over audio elements
- Memory management is crucial when working with multiple generative models
- Emotion and context awareness enhance the realism and impact of audio narratives

## Next Steps

Now that you've mastered TTS integration with AudioCraft, you're ready to explore:

- **Interactive Audio Systems**: Build responsive audio experiences that adapt to user input and context
- **Real-time Generation**: Explore techniques for generating audio on-demand with minimal latency
- **Multi-modal Integration**: Combine audio generation with other AI modalities like image and video

## Further Reading

- [Bark TTS GitHub Repository](https://github.com/suno-ai/bark) - Expressive text-to-speech system
- [ElevenLabs Documentation](https://docs.elevenlabs.io/) - Advanced voice synthesis platform
- [SSML Specification](https://www.w3.org/TR/speech-synthesis11/) - Speech Synthesis Markup Language standard
- [AudioCraft Documentation](https://github.com/facebookresearch/audiocraft) - Official documentation for Meta's audio generation tools
- [Audio Production Techniques](https://www.soundonsound.com/techniques) - Professional audio mixing and production guides
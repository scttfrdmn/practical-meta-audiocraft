---
layout: chapter
title: "Chapter 18: Building a Complete Audio Pipeline"
# Copyright © 2025 Scott Friedman.
# Licensed under Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.
difficulty: advanced
estimated_time: 3 hours
---

> "I need to create different types of audio for our game - from background music to environmental sounds and sound effects. Right now I'm using separate scripts and processes for each, and it's becoming a maintenance nightmare." — *Eliana Chen, Game Audio Director*

# Chapter 18: Building a Complete Audio Pipeline

## The Challenge

Modern interactive applications rarely require just one type of audio. Games, virtual reality experiences, and multimedia applications often need a combination of music, ambient sounds, and sound effects to create a rich audio experience. Managing multiple audio generation tools for different purposes quickly becomes unwieldy, with inconsistent interfaces, separate configuration settings, and no unified way to combine outputs.

Audio engineers and developers need a way to streamline this process—a single, coherent pipeline that can handle diverse audio generation needs while providing a consistent interface. The ideal solution should allow for switching between different models, managing resources efficiently, and simplifying the process of combining various audio elements into cohesive soundscapes.

In this chapter, you'll learn how to build a comprehensive audio pipeline that unifies AudioCraft's generation models into a single, flexible system. We'll walk through the entire process from designing the architecture to implementing a full-featured pipeline that can generate and mix different types of audio seamlessly.

## Learning Objectives

By the end of this chapter, you'll be able to:

- Design and implement a unified pipeline that consolidates MusicGen and AudioGen functionality
- Create a resource-efficient system for loading and managing different audio generation models
- Develop mixing and processing capabilities for creating layered audio compositions
- Build a flexible API that makes complex audio generation tasks simple and consistent
- Implement best practices for memory management and performance optimization

## Prerequisites

Before proceeding, ensure you have:
- Completed the chapters on basic MusicGen and AudioGen usage
- Familiarity with Python class design and object-oriented programming
- Understanding of audio processing concepts like mixing and normalization
- Working AudioCraft installation with access to both MusicGen and AudioGen models

## Key Concepts

### Unified Pipeline Architecture

A unified audio pipeline consolidates multiple audio generation tools behind a consistent interface, simplifying the developer experience and reducing complexity. Instead of managing separate systems for music and sound effects—each with their own quirks and configuration patterns—a unified pipeline provides a single access point with standardized methods.

This architecture offers several advantages. First, it reduces cognitive load by providing consistent patterns for different audio generation tasks. Second, it enables more efficient resource management by intelligently loading models only when needed. Third, it facilitates the creation of complex audio compositions by providing built-in tools for combining different audio elements.

```python
# Conceptual example of a unified pipeline
pipeline = TextToAudioPipeline()

# Generate music the same way you generate sound effects
music = pipeline.generate("Ambient electronic music", model_type="music")
sfx = pipeline.generate("Water splashing", model_type="audio")

# Mix them together with a unified interface
mixed = pipeline.mix_audio([music, sfx], weights=[0.7, 0.4])
```

### On-Demand Resource Management

Audio generation models, particularly larger variants, require significant memory. Loading all possible models at startup would be inefficient and might exceed available resources on many systems. Instead, an effective pipeline implements on-demand resource management, loading models only when needed and potentially unloading them when they're no longer in use.

This approach is particularly important for production environments or systems with limited resources. By intelligently managing model loading, we can work with multiple model types even on hardware that couldn't simultaneously hold all models in memory.

```python
# Conceptual example of on-demand resource management
pipeline = TextToAudioPipeline()

# Model is loaded only when first requested
music = pipeline.generate("Jazz piano solo", model_type="music")

# Different model loaded only when needed
sfx = pipeline.generate("Thunder cracking", model_type="audio")

# Memory can be freed when a model is no longer needed
pipeline.unload_model("music")
```

## Solution Walkthrough

### 1. Designing the Pipeline Interface

Let's begin by designing a clean, intuitive interface for our audio pipeline. The interface should abstract away the differences between models while providing access to their specific capabilities.

```python
# text_to_audio_pipeline.py - Design of our pipeline interface
import torch
import os
from audiocraft.models import MusicGen, AudioGen
from audiocraft.data.audio import audio_write

class TextToAudioPipeline:
    """
    A unified pipeline for generating different types of audio from text descriptions.
    
    This class provides a high-level interface to Meta's AudioCraft models,
    allowing seamless switching between music generation (MusicGen) and
    sound effects generation (AudioGen).
    """
    def __init__(self, use_gpu=True):
        """
        Initialize the text-to-audio pipeline with hardware detection.
        
        Args:
            use_gpu (bool): Whether to use GPU acceleration if available.
        """
        # Determine optimal computing device based on hardware and user preference
        self.device = self._get_device() if use_gpu else "cpu"
        
        # Dictionary to store loaded models (loaded on demand to save memory)
        self.models = {}
        
        # Will be set when the first model is loaded (same for all models)
        self.sample_rate = None
        
        print(f"TextToAudioPipeline initialized using device: {self.device}")
    
    def _get_device(self):
        """
        Determine the best available compute device for the current hardware.
        
        Returns:
            str: Device identifier ("mps" for Apple Silicon, "cuda" for NVIDIA, or "cpu")
        """
        if torch.backends.mps.is_available():
            return "mps"  # Apple Silicon GPU (M1/M2/M3)
        elif torch.cuda.is_available():
            return "cuda"  # NVIDIA GPU
        return "cpu"  # Fallback to CPU
```

### 2. Implementing Model Management

Now that we've set up our basic pipeline structure, we need to implement model loading and management. This involves loading models on demand and configuring them appropriately.

```python
def load_model(self, model_type, model_size):
    """
    Load a specific audio generation model into memory.
    
    This method handles loading the appropriate model type and size,
    moving it to the correct device, and storing it for later use.
    Models are loaded on demand to conserve memory.
    
    Args:
        model_type (str): Type of model to load - either "music" (MusicGen) 
                        or "audio" (AudioGen)
        model_size (str): Size/variant of the model to load:
                        - MusicGen: "small", "medium", or "large"
                        - AudioGen: "medium" or "large" only
                        
    Raises:
        ValueError: If invalid model type or size is specified
    """
    # Validate model type
    if model_type not in ["music", "audio"]:
        raise ValueError("Model type must be 'music' or 'audio'")
    
    # Validate model size based on model type
    if model_type == "music" and model_size not in ["small", "medium", "large"]:
        raise ValueError("MusicGen size must be 'small', 'medium', or 'large'")
    
    if model_type == "audio" and model_size not in ["medium", "large"]:
        raise ValueError("AudioGen size must be 'medium' or 'large'")
    
    print(f"Loading {model_type} model ({model_size})...")
    
    # Load appropriate model based on type
    if model_type == "music":
        model = MusicGen.get_pretrained(model_size)
        self.models["music"] = model
    else:
        model = AudioGen.get_pretrained(model_size)
        self.models["audio"] = model
    
    # Move model to the appropriate device (GPU/CPU)
    model.to(self.device)
    
    # Store sample rate (same for all AudioCraft models: 32kHz)
    self.sample_rate = model.sample_rate
    
    print(f"{model_type.capitalize()} model ({model_size}) loaded successfully!")

def unload_model(self, model_type):
    """
    Remove a model from memory to free up resources.
    
    Args:
        model_type (str): Type of model to unload ("music" or "audio")
        
    Returns:
        bool: True if model was unloaded, False if it wasn't loaded
    """
    if model_type in self.models:
        # Remove reference to the model
        del self.models[model_type]
        
        # Force garbage collection to free memory
        import gc
        gc.collect()
        
        # Clear CUDA cache if using NVIDIA GPU
        if self.device == "cuda":
            torch.cuda.empty_cache()
            
        print(f"{model_type.capitalize()} model unloaded successfully")
        return True
    
    return False
```

### 3. Building the Generation Interface

With model management in place, we can now implement the core generation functionality that will handle both music and sound effects creation through a single interface.

```python
def generate(self, prompt, model_type, duration=5.0, temperature=1.0, top_k=250, top_p=0.0, cfg_coef=3.0):
    """
    Generate audio from a text prompt using the specified model.
    
    This method handles the actual generation process, applying the
    specified parameters and returning the resulting audio as a tensor.
    
    Args:
        prompt (str): Text prompt describing the audio to generate
        model_type (str): Type of model to use ("music" or "audio")
        duration (float): Length of audio to generate in seconds
        temperature (float): Controls randomness/creativity of generation
                          (higher = more random, lower = more deterministic)
        top_k (int): Limits sampling to the top k most likely tokens
        top_p (float): Nucleus sampling threshold (0.0 to disable)
        cfg_coef (float): Classifier-free guidance scale (1.0-10.0)
        
    Returns:
        torch.Tensor: Generated audio tensor (mono, 32kHz sample rate)
        
    Raises:
        ValueError: If the specified model type has not been loaded
    """
    # Check if the requested model is loaded
    if model_type not in self.models:
        raise ValueError(f"{model_type.capitalize()} model not loaded. Call load_model first.")
    
    # Get the appropriate model
    model = self.models[model_type]
    
    # Configure generation parameters
    model.set_generation_params(
        duration=duration,      # Target duration in seconds
        temperature=temperature, # Creativity/randomness control
        top_k=top_k,            # Sample from top k predictions
        top_p=top_p,            # Nucleus sampling threshold
        cfg_coef=cfg_coef       # Classifier-free guidance scale
    )
    
    print(f"Generating {model_type} with prompt: '{prompt}'")
    
    # Generate audio (with gradient tracking disabled for efficiency)
    with torch.no_grad():
        wav = model.generate([prompt])  # Model expects a batch of prompts
    
    # Return the first (and only) item in the batch, moved to CPU for compatibility
    return wav[0].cpu()
```

### 4. Adding Audio Utility Functions

Finally, we need to add utilities for saving and manipulating audio. These functions will handle common tasks like file saving and mixing multiple audio sources together.

```python
def save_audio(self, audio_tensor, filename, output_dir="audio_output"):
    """
    Save an audio tensor to a WAV file with proper formatting.
    
    Args:
        audio_tensor (torch.Tensor): Audio data to save
        filename (str): Base filename (without extension)
        output_dir (str): Directory to save the file in (created if doesn't exist)
        
    Returns:
        str: Complete path to the saved WAV file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct full output path
    output_path = os.path.join(output_dir, filename)
    
    # Write audio file using AudioCraft's utility
    audio_write(
        output_path,              # Path without extension (.wav added automatically)
        audio_tensor,             # Audio data tensor
        self.sample_rate,         # Sample rate (32kHz for AudioCraft models)
        strategy="loudness"       # Normalize loudness for consistent volume
    )
    
    print(f"Audio saved to {output_path}.wav")
    return f"{output_path}.wav"

def mix_audio(self, audio_tensors, weights=None):
    """
    Mix multiple audio tensors together with optional weighting.
    
    This method allows combining multiple generated audio segments into
    a single composite audio. For example, mixing background music with
    sound effects or layering multiple environmental sounds.
    
    Args:
        audio_tensors (list): List of audio tensors to mix together
        weights (list, optional): Relative weights for each audio tensor.
                                If not provided, equal weighting is used.
        
    Returns:
        torch.Tensor: Mixed audio tensor
        
    Raises:
        ValueError: If no audio tensors are provided or weights don't match
    """
    # Validate inputs
    if not audio_tensors:
        raise ValueError("No audio tensors provided for mixing")
    
    # Default to equal weights if not specified
    if weights is None:
        weights = [1.0 / len(audio_tensors)] * len(audio_tensors)
    
    # Verify weights match the number of audio tensors
    if len(weights) != len(audio_tensors):
        raise ValueError("Number of weights must match number of audio tensors")
    
    # Convert all tensors to numpy arrays for easier manipulation
    import numpy as np
    audio_arrays = [tensor.numpy() for tensor in audio_tensors]
    
    # Find the longest audio length (all must be padded to this length)
    max_length = max(array.shape[0] for array in audio_arrays)
    
    # Initialize output array with zeros (silence)
    mixed_audio = np.zeros(max_length)
    
    # Mix each audio with its corresponding weight
    for i, (array, weight) in enumerate(zip(audio_arrays, weights)):
        # Create padded version of this audio (filled with zeros/silence)
        padded = np.zeros(max_length)
        
        # Copy actual audio data into the padded array (leaving zeros at the end)
        padded[:array.shape[0]] = array
        
        # Add weighted audio to the mix
        mixed_audio += padded * weight
    
    # Normalize to prevent clipping in the final mixed audio
    # This ensures the maximum amplitude is within the valid range (-1 to 1)
    if np.max(np.abs(mixed_audio)) > 1.0:
        mixed_audio = mixed_audio / np.max(np.abs(mixed_audio))
    
    # Convert numpy array back to PyTorch tensor for compatibility
    return torch.from_numpy(mixed_audio).to(torch.float32)
```

## Complete Implementation

Let's put everything together into a complete, runnable example:

```python
#!/usr/bin/env python3
# text_to_audio_pipeline.py - Unified pipeline for AudioCraft's generation models
import torch
import torchaudio
import numpy as np
import os
from audiocraft.models import MusicGen, AudioGen
from audiocraft.data.audio import audio_write

class TextToAudioPipeline:
    """
    A unified pipeline for generating different types of audio from text descriptions.
    
    This class provides a high-level interface to Meta's AudioCraft models,
    allowing seamless switching between music generation (MusicGen) and
    sound effects generation (AudioGen). It also includes utilities for
    saving audio and mixing multiple generated audio segments together.
    
    Key features:
    - Single interface for multiple audio generation models
    - On-demand model loading to save memory
    - Device management for optimal performance
    - Audio mixing capabilities for creating layered soundscapes
    - Consistent audio saving functionality
    
    Example usage:
        pipeline = TextToAudioPipeline()
        pipeline.load_model("music", "small")
        music = pipeline.generate("Ambient piano with soft pads", "music")
        pipeline.save_audio(music, "ambient_piano")
    """
    def __init__(self, use_gpu=True):
        """
        Initialize the text-to-audio pipeline with hardware detection.
        
        Args:
            use_gpu (bool): Whether to use GPU acceleration if available.
                           Set to False to force CPU usage even if a GPU is present.
        """
        # Determine optimal computing device based on hardware and user preference
        self.device = self._get_device() if use_gpu else "cpu"
        
        # Dictionary to store loaded models (loaded on demand to save memory)
        self.models = {}
        
        # Will be set when the first model is loaded (same for all models)
        self.sample_rate = None
        
        print(f"TextToAudioPipeline initialized using device: {self.device}")
    
    def _get_device(self):
        """
        Determine the best available compute device for the current hardware.
        
        Returns:
            str: Device identifier ("mps" for Apple Silicon, "cuda" for NVIDIA, or "cpu")
            
        Note:
            MPS (Metal Performance Shaders) support is available on macOS with Apple Silicon.
            CUDA support requires an NVIDIA GPU with appropriate drivers.
        """
        if torch.backends.mps.is_available():
            return "mps"  # Apple Silicon GPU (M1/M2/M3)
        elif torch.cuda.is_available():
            return "cuda"  # NVIDIA GPU
        return "cpu"  # Fallback to CPU
    
    def load_model(self, model_type, model_size):
        """
        Load a specific audio generation model into memory.
        
        This method handles loading the appropriate model type and size,
        moving it to the correct device, and storing it for later use.
        Models are loaded on demand to conserve memory.
        
        Args:
            model_type (str): Type of model to load - either "music" (MusicGen) 
                            or "audio" (AudioGen)
            model_size (str): Size/variant of the model to load:
                            - MusicGen: "small", "medium", or "large"
                            - AudioGen: "medium" or "large" only
                            
        Raises:
            ValueError: If invalid model type or size is specified
            
        Note:
            Larger models produce higher quality output but require more memory
            and take longer to generate.
        """
        # Validate model type
        if model_type not in ["music", "audio"]:
            raise ValueError("Model type must be 'music' or 'audio'")
        
        # Validate model size based on model type
        if model_type == "music" and model_size not in ["small", "medium", "large"]:
            raise ValueError("MusicGen size must be 'small', 'medium', or 'large'")
        
        if model_type == "audio" and model_size not in ["medium", "large"]:
            raise ValueError("AudioGen size must be 'medium' or 'large'")
        
        print(f"Loading {model_type} model ({model_size})...")
        
        # Load appropriate model based on type
        if model_type == "music":
            model = MusicGen.get_pretrained(model_size)
            self.models["music"] = model
        else:
            model = AudioGen.get_pretrained(model_size)
            self.models["audio"] = model
        
        # Move model to the appropriate device (GPU/CPU)
        model.to(self.device)
        
        # Store sample rate (same for all AudioCraft models: 32kHz)
        self.sample_rate = model.sample_rate
        
        print(f"{model_type.capitalize()} model ({model_size}) loaded successfully!")
    
    def unload_model(self, model_type):
        """
        Remove a model from memory to free up resources.
        
        Args:
            model_type (str): Type of model to unload ("music" or "audio")
            
        Returns:
            bool: True if model was unloaded, False if it wasn't loaded
        """
        if model_type in self.models:
            # Remove reference to the model
            del self.models[model_type]
            
            # Force garbage collection to free memory
            import gc
            gc.collect()
            
            # Clear CUDA cache if using NVIDIA GPU
            if self.device == "cuda":
                torch.cuda.empty_cache()
                
            print(f"{model_type.capitalize()} model unloaded successfully")
            return True
        
        return False
    
    def generate(self, prompt, model_type, duration=5.0, temperature=1.0, top_k=250, top_p=0.0, cfg_coef=3.0):
        """
        Generate audio from a text prompt using the specified model.
        
        This method handles the actual generation process, applying the
        specified parameters and returning the resulting audio as a tensor.
        
        Args:
            prompt (str): Text prompt describing the audio to generate
            model_type (str): Type of model to use ("music" or "audio")
            duration (float): Length of audio to generate in seconds
            temperature (float): Controls randomness/creativity of generation
                              (higher = more random, lower = more deterministic)
            top_k (int): Limits sampling to the top k most likely tokens
            top_p (float): Nucleus sampling threshold (0.0 to disable)
            cfg_coef (float): Classifier-free guidance scale (1.0-10.0)
            
        Returns:
            torch.Tensor: Generated audio tensor (mono, 32kHz sample rate)
            
        Raises:
            ValueError: If the specified model type has not been loaded
            
        Note:
            - MusicGen works best with prompts describing musical elements and styles
            - AudioGen works best with prompts describing environmental sounds and effects
        """
        # Check if the requested model is loaded
        if model_type not in self.models:
            raise ValueError(f"{model_type.capitalize()} model not loaded. Call load_model first.")
        
        # Get the appropriate model
        model = self.models[model_type]
        
        # Configure generation parameters
        model.set_generation_params(
            duration=duration,        # Target duration in seconds
            temperature=temperature,  # Creativity/randomness control
            top_k=top_k,              # Sample from top k predictions
            top_p=top_p,              # Nucleus sampling threshold
            cfg_coef=cfg_coef         # Classifier-free guidance scale
        )
        
        print(f"Generating {model_type} with prompt: '{prompt}'")
        
        # Generate audio (with gradient tracking disabled for efficiency)
        with torch.no_grad():
            wav = model.generate([prompt])  # Model expects a batch of prompts
        
        # Return the first (and only) item in the batch, moved to CPU for compatibility
        return wav[0].cpu()
    
    def save_audio(self, audio_tensor, filename, output_dir="audio_output"):
        """
        Save an audio tensor to a WAV file with proper formatting.
        
        Args:
            audio_tensor (torch.Tensor): Audio data to save
            filename (str): Base filename (without extension)
            output_dir (str): Directory to save the file in (created if doesn't exist)
            
        Returns:
            str: Complete path to the saved WAV file
            
        Note:
            This method uses AudioCraft's audio_write utility which handles
            proper formatting and loudness normalization for optimal playback.
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Construct full output path
        output_path = os.path.join(output_dir, filename)
        
        # Write audio file using AudioCraft's utility
        audio_write(
            output_path,              # Path without extension (.wav added automatically)
            audio_tensor,             # Audio data tensor
            self.sample_rate,         # Sample rate (32kHz for AudioCraft models)
            strategy="loudness"       # Normalize loudness for consistent volume
        )
        
        print(f"Audio saved to {output_path}.wav")
        return f"{output_path}.wav"
    
    def mix_audio(self, audio_tensors, weights=None):
        """
        Mix multiple audio tensors together with optional weighting.
        
        This method allows combining multiple generated audio segments into
        a single composite audio. For example, mixing background music with
        sound effects or layering multiple environmental sounds.
        
        Args:
            audio_tensors (list): List of audio tensors to mix together
            weights (list, optional): Relative weights for each audio tensor.
                                    If not provided, equal weighting is used.
            
        Returns:
            torch.Tensor: Mixed audio tensor
            
        Raises:
            ValueError: If no audio tensors are provided or weights don't match
            
        Note:
            - Audio segments of different lengths will be handled by padding shorter ones
            - Audio is automatically normalized to prevent clipping
            - Default mixing is equal parts if weights aren't specified
        """
        # Validate inputs
        if not audio_tensors:
            raise ValueError("No audio tensors provided for mixing")
        
        # Default to equal weights if not specified
        if weights is None:
            weights = [1.0 / len(audio_tensors)] * len(audio_tensors)
        
        # Verify weights match the number of audio tensors
        if len(weights) != len(audio_tensors):
            raise ValueError("Number of weights must match number of audio tensors")
        
        # Convert all tensors to numpy arrays for easier manipulation
        audio_arrays = [tensor.numpy() for tensor in audio_tensors]
        
        # Find the longest audio length (all must be padded to this length)
        max_length = max(array.shape[0] for array in audio_arrays)
        
        # Initialize output array with zeros (silence)
        mixed_audio = np.zeros(max_length)
        
        # Mix each audio with its corresponding weight
        for i, (array, weight) in enumerate(zip(audio_arrays, weights)):
            # Create padded version of this audio (filled with zeros/silence)
            padded = np.zeros(max_length)
            
            # Copy actual audio data into the padded array (leaving zeros at the end)
            padded[:array.shape[0]] = array
            
            # Add weighted audio to the mix
            mixed_audio += padded * weight
        
        # Normalize to prevent clipping in the final mixed audio
        # This ensures the maximum amplitude is within the valid range (-1 to 1)
        if np.max(np.abs(mixed_audio)) > 1.0:
            mixed_audio = mixed_audio / np.max(np.abs(mixed_audio))
        
        # Convert numpy array back to PyTorch tensor for compatibility
        return torch.from_numpy(mixed_audio).to(torch.float32)

# Example usage
if __name__ == "__main__":
    """
    Demonstration of the TextToAudioPipeline's capabilities.
    
    This example shows how to:
    1. Initialize the pipeline
    2. Load different model types
    3. Generate both music and sound effects
    4. Save individual audio files
    5. Create a mixed composition by combining elements
    
    The result is a layered soundscape with background music and environmental effects.
    """
    print("=== AudioCraft Text-to-Audio Pipeline Demo ===")
    
    # Create the pipeline with automatic device selection
    pipeline = TextToAudioPipeline()
    
    # Load both music and sound generation models
    # We use smaller models for faster generation, but you can use larger ones for better quality
    print("\n[1] Loading generation models...")
    pipeline.load_model("music", "small")  # MusicGen for musical elements
    pipeline.load_model("audio", "medium")  # AudioGen for sound effects
    
    # Generate background music track using MusicGen
    # Lower temperature (0.7) for more consistent, less random results
    print("\n[2] Generating background music...")
    music_prompt = "Gentle ambient music with soft piano and synthesizer pads, peaceful and calm"
    music_audio = pipeline.generate(
        prompt=music_prompt, 
        model_type="music", 
        duration=10.0,      # 10 seconds of audio
        temperature=0.7     # Lower temperature for more predictable music
    )
    
    # Generate environmental sound effects using AudioGen
    # Higher temperature (1.0) for more varied and natural sounds
    print("\n[3] Generating environmental sounds...")
    sfx_prompt = "Forest ambience with birds chirping, leaves rustling in a gentle breeze, and a distant stream"
    sfx_audio = pipeline.generate(
        prompt=sfx_prompt, 
        model_type="audio", 
        duration=10.0,      # Same duration as music for easy mixing
        temperature=1.0     # Standard temperature for natural variation
    )
    
    # Save both individual audio tracks for reference
    print("\n[4] Saving individual audio tracks...")
    pipeline.save_audio(music_audio, "ambient_music")
    pipeline.save_audio(sfx_audio, "forest_ambience")
    
    # Create a mixed composition by combining music and sound effects
    # We weight music at 70% and sound effects at 50% for a balanced mix
    # where the music forms a foundation with the nature sounds layered on top
    print("\n[5] Creating mixed soundscape...")
    mixed_audio = pipeline.mix_audio(
        audio_tensors=[music_audio, sfx_audio],
        weights=[0.7, 0.5]  # Music at 70%, SFX at 50% - these don't need to sum to 1.0
    )
    
    # Save the final mixed composition
    print("\n[6] Saving final composition...")
    output_path = pipeline.save_audio(mixed_audio, "ambient_forest_scene")
    
    print(f"\nDemo complete! Final mixed audio saved to: {output_path}")
    print("Try experimenting with different prompts, model sizes, and mix weights!")
```

## Variations and Customizations

Let's explore some variations of our solution to address different needs or preferences.

### Variation 1: Memory-Optimized Pipeline

For systems with limited memory, we can create a variation that automatically unloads models when switching between them:

```python
class MemoryOptimizedPipeline(TextToAudioPipeline):
    """
    A memory-optimized version of the audio pipeline that automatically
    unloads models when switching between them to minimize memory usage.
    """
    
    def generate(self, prompt, model_type, model_size="small", duration=5.0, temperature=1.0):
        """
        Generate audio, loading the required model and unloading any others.
        
        This overridden method handles model loading and unloading automatically,
        ensuring only one model is in memory at a time.
        
        Args:
            prompt (str): Text prompt describing the audio
            model_type (str): Type of model to use ("music" or "audio")
            model_size (str): Size of model to load
            duration (float): Length of audio in seconds
            temperature (float): Creativity control parameter
            
        Returns:
            torch.Tensor: Generated audio
        """
        # First, unload any models that aren't the one we need
        for loaded_type in list(self.models.keys()):
            if loaded_type != model_type:
                self.unload_model(loaded_type)
        
        # Check if we need to load the requested model
        if model_type not in self.models:
            # For AudioGen, default to medium if small requested (small not available)
            if model_type == "audio" and model_size == "small":
                model_size = "medium"
                
            self.load_model(model_type, model_size)
        
        # Now generate using the standard method
        return super().generate(prompt, model_type, duration, temperature)

# Usage example
memory_pipeline = MemoryOptimizedPipeline()
music = memory_pipeline.generate("Epic orchestral theme", "music", "small")
# The music model is automatically unloaded when we switch to audio
sfx = memory_pipeline.generate("Thunderstorm with heavy rain", "audio", "medium")
```

### Variation 2: Scene Composition System

We can extend our pipeline to support scene-based composition, automatically generating and mixing multiple elements:

```python
def generate_scene(self, scene_description):
    """
    Generate a complete audio scene from a structured description.
    
    This method takes a dictionary describing a scene with multiple
    audio elements and generates each one, then mixes them together
    with the specified weights.
    
    Args:
        scene_description (dict): A dictionary containing scene elements:
            {
                "name": "Scene name",
                "duration": 10.0,
                "elements": [
                    {
                        "type": "music",
                        "model_size": "small",
                        "prompt": "Ambient underscore",
                        "weight": 0.7,
                        "temperature": 0.8
                    },
                    {
                        "type": "audio",
                        "model_size": "medium", 
                        "prompt": "Rain and thunder",
                        "weight": 0.5,
                        "temperature": 1.0
                    }
                ]
            }
            
    Returns:
        torch.Tensor: The fully mixed audio scene
        
    Note:
        This automatically manages all required models and generates
        elements at the same duration for proper mixing.
    """
    scene_name = scene_description.get("name", "Unnamed Scene")
    duration = scene_description.get("duration", 10.0)
    elements = scene_description.get("elements", [])
    
    if not elements:
        raise ValueError("Scene must contain at least one audio element")
    
    print(f"Generating scene: {scene_name}")
    
    # Generate each element
    generated_audio = []
    weights = []
    
    for i, element in enumerate(elements):
        element_type = element["type"]
        model_size = element.get("model_size", "small" if element_type == "music" else "medium")
        prompt = element["prompt"]
        weight = element.get("weight", 1.0)
        temperature = element.get("temperature", 1.0)
        
        print(f"Generating element {i+1}/{len(elements)}: {prompt}")
        
        # Load the appropriate model if needed
        if element_type not in self.models:
            self.load_model(element_type, model_size)
        
        # Generate this element
        audio = self.generate(
            prompt=prompt,
            model_type=element_type,
            duration=duration,
            temperature=temperature
        )
        
        generated_audio.append(audio)
        weights.append(weight)
        
        # Optionally save individual elements
        if element.get("save", False):
            safe_name = "".join(c if c.isalnum() else "_" for c in prompt)[:20]
            self.save_audio(audio, f"{scene_name}_{safe_name}")
    
    # Mix all elements together
    print(f"Mixing {len(generated_audio)} elements...")
    mixed_scene = self.mix_audio(generated_audio, weights)
    
    # Save the complete scene
    output_path = self.save_audio(mixed_scene, scene_name)
    
    return mixed_scene

# Usage example
rain_scene = {
    "name": "rainy_forest",
    "duration": 15.0,
    "elements": [
        {
            "type": "music",
            "prompt": "Soft piano music with melancholic mood",
            "weight": 0.6,
            "temperature": 0.7
        },
        {
            "type": "audio",
            "prompt": "Heavy rain on forest leaves",
            "weight": 0.8,
            "temperature": 1.0
        },
        {
            "type": "audio",
            "prompt": "Distant thunder rumbling",
            "weight": 0.4,
            "temperature": 1.2
        }
    ]
}

pipeline = TextToAudioPipeline()
rain_forest_audio = pipeline.generate_scene(rain_scene)
```

## Common Pitfalls and Troubleshooting

### Problem: Memory Management Issues

When working with multiple models, especially larger ones, you might encounter memory limitations.

**Solution**: 
- Implement a memory management strategy that unloads models when not in use
- Load only one model at a time if memory is constrained
- Use smaller model variants when possible
- Consider a deferred execution pattern where you generate one element at a time:

```python
# Sequence generations to minimize memory usage
pipeline = TextToAudioPipeline()

# Generate and save music first
pipeline.load_model("music", "small")
music = pipeline.generate("Ambient music", "music")
pipeline.save_audio(music, "ambient_track")
pipeline.unload_model("music")

# Then generate and save sound effects
pipeline.load_model("audio", "medium")
sfx = pipeline.generate("Ocean waves", "audio")
pipeline.save_audio(sfx, "ocean_waves")

# Load saved files for mixing instead of keeping tensors in memory
import torchaudio
music, sr1 = torchaudio.load("audio_output/ambient_track.wav")
sfx, sr2 = torchaudio.load("audio_output/ocean_waves.wav")

# Mix them together
mixed = pipeline.mix_audio([music[0], sfx[0]])
pipeline.save_audio(mixed, "mixed_composition")
```

### Problem: Inconsistent Audio Lengths

When mixing audio sources of different durations, you might get unexpected results.

**Solution**:
- Our mixing function automatically handles different lengths by padding shorter ones with silence
- You can explicitly set the same duration for all generations to ensure consistency
- For more control, use audio processing tools to trim or extend clips before mixing:

```python
def trim_audio(audio_tensor, target_length):
    """Trim or pad an audio tensor to a specific length."""
    current_length = audio_tensor.shape[0]
    
    if current_length > target_length:
        # Trim to target length
        return audio_tensor[:target_length]
    elif current_length < target_length:
        # Pad with zeros (silence)
        padding = torch.zeros(target_length - current_length)
        return torch.cat([audio_tensor, padding])
    else:
        # Already the right length
        return audio_tensor
```

### Problem: Getting Realistic Sound Combinations

Generating and mixing different audio elements doesn't always result in natural-sounding combinations.

**Solution**:
- Adjust the mix weights carefully based on the prominence each element should have
- Consider the frequency content of each element when setting weights (bass-heavy elements may need lower weights)
- Use post-processing like equalization or compression to help elements blend:

```python
def apply_simple_eq(mixed_audio, bass_boost=0.0, treble_cut=0.0):
    """Apply simple equalization to shape the frequency balance."""
    import numpy as np
    from scipy import signal
    
    # Convert to numpy for signal processing
    audio_np = mixed_audio.numpy()
    
    # Apply bass boost if requested
    if bass_boost > 0:
        # Simple low-shelf filter
        b, a = signal.butter(2, 300 / (self.sample_rate/2), 'lowshelf', gain=bass_boost)
        audio_np = signal.lfilter(b, a, audio_np)
    
    # Apply treble cut if requested
    if treble_cut > 0:
        # Simple high-shelf filter
        b, a = signal.butter(2, 3000 / (self.sample_rate/2), 'highshelf', gain=-treble_cut)
        audio_np = signal.lfilter(b, a, audio_np)
    
    # Return as tensor
    return torch.from_numpy(audio_np).to(torch.float32)
```

## Hands-on Challenge

Now it's your turn to experiment with what you've learned. Try the following challenge:

### Challenge: Interactive Scene Generator

Create an application that:
1. Provides a UI for defining audio scenes with multiple elements
2. Allows users to specify model types, prompts, and mix weights
3. Generates each element and provides a real-time preview
4. Saves the final mixed scene and its component parts
5. Implements memory-efficient generation for low-resource systems

### Bonus Challenge

Extend the pipeline to support continuous generation for longer compositions. Implement a system that can generate audio in chunks and seamlessly stitch them together for extended playback.

## Key Takeaways

- Unified pipeline architectures simplify audio generation by providing consistent interfaces across multiple model types
- On-demand resource management is crucial for efficient operation, especially with large models
- Combining different audio elements through properly weighted mixing creates rich, layered soundscapes
- Memory management strategies enable working with multiple models even on resource-constrained systems
- Scene-based composition approaches help organize complex audio generation workflows

## Next Steps

Now that you've mastered building a complete audio pipeline, you're ready to explore:

- **Text-to-Speech Integration**: Learn how to combine AudioCraft with TTS systems for narrated audio experiences
- **Interactive Audio Systems**: Discover techniques for creating responsive audio that adapts to user input
- **Research Extensions**: Explore cutting-edge techniques for extending AudioCraft's capabilities

## Further Reading

- [AudioCraft GitHub Repository](https://github.com/facebookresearch/audiocraft) - Official source code and documentation
- [Meta AI Blog: AudioCraft](https://ai.meta.com/blog/audiocraft-musicgen-audiogen-encodec-generative-ai-audio/) - Technical details and research insights
- [Digital Signal Processing for Audio Applications](https://www.dspguide.com/) - In-depth guide to audio processing techniques
- [Game Audio Implementation](https://www.routledge.com/Game-Audio-Implementation-A-Practical-Guide-to-Using-the-Unreal-Engine/Stevens-Raybould/p/book/9781138777248) - Advanced techniques for game audio pipelines
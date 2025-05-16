#!/usr/bin/env python3
# text_to_audio_pipeline.py
import torch
import torchaudio
import numpy as np
import os
from audiocraft.models import MusicGen, AudioGen
from audiocraft.data.audio import audio_write

class TextToAudioPipeline:
    """
    A unified pipeline for generating various types of audio from text.
    """
    def __init__(self, use_gpu=True):
        """
        Initialize the text-to-audio pipeline.
        
        Args:
            use_gpu (bool): Whether to use GPU acceleration if available
        """
        self.device = self._get_device() if use_gpu else "cpu"
        self.models = {}
        self.sample_rate = None
        print(f"Using device: {self.device}")
    
    def _get_device(self):
        """Determine the best available device."""
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    def load_model(self, model_type, model_size):
        """
        Load a specific audio generation model.
        
        Args:
            model_type (str): Type of model ("music" or "audio")
            model_size (str): Size of model ("small", "medium", or "large")
        """
        if model_type not in ["music", "audio"]:
            raise ValueError("Model type must be 'music' or 'audio'")
        
        if model_type == "music" and model_size not in ["small", "medium", "large"]:
            raise ValueError("MusicGen size must be 'small', 'medium', or 'large'")
        
        if model_type == "audio" and model_size not in ["medium", "large"]:
            raise ValueError("AudioGen size must be 'medium' or 'large'")
        
        print(f"Loading {model_type} model ({model_size})...")
        
        if model_type == "music":
            model = MusicGen.get_pretrained(model_size)
            self.models["music"] = model
        else:
            model = AudioGen.get_pretrained(model_size)
            self.models["audio"] = model
        
        model.to(self.device)
        
        # Store sample rate (same for all models)
        self.sample_rate = model.sample_rate
        
        print(f"{model_type.capitalize()} model loaded successfully!")
    
    def generate(self, prompt, model_type, duration=5.0, temperature=1.0):
        """
        Generate audio from a text prompt using the specified model.
        
        Args:
            prompt (str): Text prompt describing the audio to generate
            model_type (str): Type of model to use ("music" or "audio")
            duration (float): Length of audio in seconds
            temperature (float): Controls randomness (higher = more random)
            
        Returns:
            torch.Tensor: Generated audio tensor
        """
        if model_type not in self.models:
            raise ValueError(f"{model_type.capitalize()} model not loaded. Call load_model first.")
        
        model = self.models[model_type]
        
        # Set generation parameters
        model.set_generation_params(
            duration=duration,
            temperature=temperature,
            top_k=250,
            top_p=0.0,
        )
        
        print(f"Generating {model_type} with prompt: '{prompt}'")
        
        # Generate audio
        with torch.no_grad():
            wav = model.generate([prompt])
        
        # Move to CPU
        return wav[0].cpu()
    
    def save_audio(self, audio_tensor, filename, output_dir="audio_output"):
        """
        Save an audio tensor to a file.
        
        Args:
            audio_tensor (torch.Tensor): Audio tensor to save
            filename (str): Base filename (without extension)
            output_dir (str): Directory to save to
        """
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        
        audio_write(
            output_path,
            audio_tensor,
            self.sample_rate,
            strategy="loudness"
        )
        
        print(f"Audio saved to {output_path}.wav")
        return f"{output_path}.wav"
    
    def mix_audio(self, audio_tensors, weights=None):
        """
        Mix multiple audio tensors together.
        
        Args:
            audio_tensors (list): List of audio tensors to mix
            weights (list, optional): Weights for each audio tensor
            
        Returns:
            torch.Tensor: Mixed audio tensor
        """
        if not audio_tensors:
            raise ValueError("No audio tensors provided for mixing")
        
        # Default to equal weights if not specified
        if weights is None:
            weights = [1.0 / len(audio_tensors)] * len(audio_tensors)
        
        if len(weights) != len(audio_tensors):
            raise ValueError("Number of weights must match number of audio tensors")
        
        # Convert all tensors to numpy for easier handling
        audio_arrays = [tensor.numpy() for tensor in audio_tensors]
        
        # Find the longest audio
        max_length = max(array.shape[0] for array in audio_arrays)
        
        # Initialize output array
        mixed_audio = np.zeros(max_length)
        
        # Mix each audio with its weight
        for i, (array, weight) in enumerate(zip(audio_arrays, weights)):
            # Pad shorter audios with silence
            padded = np.zeros(max_length)
            padded[:array.shape[0]] = array
            
            # Add to mix with weight
            mixed_audio += padded * weight
        
        # Normalize to prevent clipping
        if np.max(np.abs(mixed_audio)) > 1.0:
            mixed_audio = mixed_audio / np.max(np.abs(mixed_audio))
        
        # Convert back to tensor
        return torch.from_numpy(mixed_audio).to(torch.float32)

# Example usage
if __name__ == "__main__":
    # Create the pipeline
    pipeline = TextToAudioPipeline()
    
    # Load models
    pipeline.load_model("music", "small")
    pipeline.load_model("audio", "medium")
    
    # Generate music
    music_prompt = "Gentle ambient music with soft piano and synthesizer pads"
    music_audio = pipeline.generate(music_prompt, "music", duration=10.0, temperature=0.7)
    
    # Generate sound effects
    sfx_prompt = "Forest ambience with birds chirping and leaves rustling"
    sfx_audio = pipeline.generate(sfx_prompt, "audio", duration=10.0, temperature=1.0)
    
    # Save individual tracks
    pipeline.save_audio(music_audio, "ambient_music")
    pipeline.save_audio(sfx_audio, "forest_ambience")
    
    # Mix them together with music at 70% and SFX at 50%
    mixed_audio = pipeline.mix_audio([music_audio, sfx_audio], weights=[0.7, 0.5])
    pipeline.save_audio(mixed_audio, "ambient_forest_scene")
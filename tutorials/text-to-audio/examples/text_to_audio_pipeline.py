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
    
    def generate(self, prompt, model_type, duration=5.0, temperature=1.0):
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
            top_k=250,                # Sample from top 250 predictions
            top_p=0.0,                # Disable nucleus sampling
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
    
    # Additional usage ideas (commented out):
    #
    # # Generate multiple variations of the same prompt with different temperatures
    # variations = []
    # for temp in [0.5, 0.8, 1.1]:
    #     audio = pipeline.generate("Dramatic orchestral melody", "music", temperature=temp)
    #     pipeline.save_audio(audio, f"orchestra_temp_{temp}")
    #     variations.append(audio)
    #
    # # Create a city soundscape with multiple layered sounds
    # city_base = pipeline.generate("City street ambience with traffic", "audio")
    # cafe_sounds = pipeline.generate("Cafe conversation and coffee shop sounds", "audio")
    # construction = pipeline.generate("Distant construction work and jackhammer", "audio")
    # 
    # city_mix = pipeline.mix_audio(
    #     [city_base, cafe_sounds, construction],
    #     weights=[1.0, 0.6, 0.3]  # Main street sounds with quieter cafe and very quiet construction
    # )
    # pipeline.save_audio(city_mix, "complex_city_scene")
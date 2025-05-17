#!/usr/bin/env python3
# musicgen_melody_conditioning.py - Example of using MusicGen with melody conditioning
import os
import torch
import torchaudio
import matplotlib.pyplot as plt
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import argparse
import time

def load_and_process_melody(melody_file, target_sr=32000, plot=False, output_dir=None):
    """
    Load and process a melody file for conditioning MusicGen generation.
    
    This function handles several important preprocessing steps for melody conditioning:
    1. Loading the audio file in various formats (mp3, wav, etc.)
    2. Converting stereo files to mono (required for melody conditioning)
    3. Resampling to 32kHz (MusicGen's expected sample rate)
    4. Optionally visualizing the waveform for analysis
    
    Args:
        melody_file (str): Path to the melody audio file (wav, mp3, etc.)
        target_sr (int): Target sample rate (MusicGen expects exactly 32kHz)
        plot (bool): Whether to plot and save the waveform visualization
        output_dir (str): Directory to save the waveform plot
        
    Returns:
        torch.Tensor: Processed melody tensor ready for MusicGen's generate_with_chroma method
    
    Note:
        The returned tensor is in the shape expected by MusicGen: [1, num_samples]
        Melody conditioning is sensitive to audio quality, so high-quality input files work best
    """
    print(f"Loading melody from {melody_file}")
    
    # Load the audio file using torchaudio
    # This supports various audio formats including WAV, MP3, FLAC, etc.
    melody, sr = torchaudio.load(melody_file)
    
    # If stereo, convert to mono by averaging channels
    # MusicGen's melody conditioning requires mono audio
    if melody.shape[0] > 1:
        print(f"Converting {melody.shape[0]} channels to mono")
        melody = melody.mean(dim=0, keepdim=True)
    
    # Resample if needed to exactly 32kHz
    # MusicGen models are trained on 32kHz audio and expect this sample rate
    if sr != target_sr:
        print(f"Resampling from {sr}Hz to {target_sr}Hz")
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        melody = resampler(melody)
    
    # Optionally create a visualization of the waveform
    # This is useful for understanding the melody's structure
    if plot and output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        plt.figure(figsize=(10, 4))
        plt.plot(melody[0].numpy())
        plt.title("Melody Waveform")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plot_path = os.path.join(output_dir, "melody_waveform.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved waveform plot to {plot_path}")
    
    return melody

def generate_with_melody(
    prompt,
    melody_file,
    output_dir="melody_conditioned",
    model_size="medium",
    duration=10.0,
    temperature=1.0,
    top_k=250,
    top_p=0.0,
    save_original=True,
    plot_waveform=True
):
    """
    Generate music based on both a text prompt and a reference melody.
    
    This function demonstrates melody conditioning, a powerful feature of MusicGen
    that allows you to guide generation with a reference melody. The model will create
    music that follows the melodic contour and rhythm of your input while adopting
    the style described in your text prompt.
    
    Args:
        prompt (str): Text description of the desired musical style (e.g., "orchestral", "electronic")
        melody_file (str): Path to the melody audio file to use as a reference
        output_dir (str): Directory to save output files and visualizations
        model_size (str): Size of MusicGen model to use ("small", "medium", or "large")
        duration (float): Length of audio to generate in seconds
        temperature (float): Controls randomness (higher = more random, lower = more deterministic)
        top_k (int): Number of top tokens to sample from at each step (higher = more diversity)
        top_p (float): Nucleus sampling parameter (0.0 disables nucleus sampling)
        save_original (bool): Whether to save a copy of the original melody for comparison
        plot_waveform (bool): Whether to create a visualization of the input melody's waveform
        
    Returns:
        torch.Tensor: Generated audio as a tensor
        
    Note:
        - Melody conditioning works best with clean, monophonic melodies (single notes, not chords)
        - The model tries to follow the rhythm and contour, but not necessarily exact notes
        - Higher temperature values produce more creative interpretations of the melody
        - Medium and large models tend to produce better melody conditioning results than small
    """
    # Create output directory for all generated files
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine the best available compute device for the user's hardware
    # MusicGen benefits significantly from GPU acceleration
    if torch.backends.mps.is_available():
        device = "mps"  # Apple Silicon GPU
        print("Using MPS (Metal) for generation")
    elif torch.cuda.is_available():
        device = "cuda"  # NVIDIA GPU
        print("Using CUDA for generation")
    else:
        device = "cpu"  # Fallback to CPU
        print("Using CPU for generation (this will be slow)")
    
    # Load the MusicGen model - timing how long it takes to inform the user
    # Model loading can take significant time, especially for larger models
    print(f"Loading MusicGen {model_size} model...")
    start_time = time.time()
    model = MusicGen.get_pretrained(model_size)
    model.to(device)
    print(f"Model loaded in {time.time() - start_time:.2f} seconds")
    
    # Configure how the model will generate audio
    # These parameters control the behavior of the generation algorithm
    model.set_generation_params(
        duration=duration,            # How long the generated audio will be
        temperature=temperature,      # How random/creative the generation is
        top_k=top_k,                  # Restricts sampling to top k tokens
        top_p=top_p,                  # Nucleus sampling (0.0 = disabled)
    )
    
    # Process the melody file to make it compatible with MusicGen
    # This handles format conversion, resampling, and mono conversion
    melody = load_and_process_melody(
        melody_file, 
        plot=plot_waveform, 
        output_dir=output_dir
    )
    
    # Optionally save the processed melody for reference
    # This is useful for comparing the original against the generated version
    if save_original:
        torchaudio.save(
            os.path.join(output_dir, "original_melody.wav"), 
            melody, 
            32000  # MusicGen's standard sample rate
        )
    
    # Begin the generation process with melody conditioning
    print(f"Generating with melody conditioning for prompt: '{prompt}'")
    start_time = time.time()
    
    # Move the melody tensor to the same device as the model
    # This is essential for proper operation and prevents CUDA/MPS errors
    melody = melody.to(device)
    
    # Generate audio with both text and melody conditioning
    # The generate_with_chroma method combines both condition types
    # Note: melody needs an extra dimension since the model expects a batch
    wav = model.generate_with_chroma([prompt], melody.unsqueeze(0))
    
    # Report generation time for performance tracking
    generation_time = time.time() - start_time
    print(f"Generation completed in {generation_time:.2f} seconds")
    
    # Create a filename based on the prompt (making it filesystem-safe)
    safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt)[:30]
    output_path = os.path.join(output_dir, f"melody_conditioned_{safe_prompt}")
    
    # Save the generated audio using audiocraft's utility function
    # This ensures proper formatting and loudness normalization
    audio_write(output_path, wav[0].cpu(), model.sample_rate)
    
    print(f"Generated audio saved to {output_path}.wav")
    return wav

def generate_style_variations(
    melody_file,
    styles,
    output_dir="style_variations",
    model_size="medium",
    duration=10.0,
    temperature=1.0,
    top_k=250,
    top_p=0.0
):
    """
    Generate multiple style variations of the same melody in a batch process.
    
    This function demonstrates how to create different stylistic interpretations
    of the same melody. It allows you to explore how MusicGen's melody conditioning
    can be combined with different text prompts to create diverse arrangements
    while preserving the core melodic elements.
    
    Args:
        melody_file (str): Path to the melody audio file to use as a reference
        styles (dict): Dictionary mapping style names to text prompts 
                      (e.g., {"jazz": "A jazz quartet...", "rock": "A rock band..."})
        output_dir (str): Directory to save all generated outputs
        model_size (str): Size of MusicGen model to use ("small", "medium", or "large")
        duration (float): Length of audio to generate in seconds
        temperature (float): Controls randomness/creativity of the generation
        top_k (int): Number of top tokens to sample from at each step
        top_p (float): Nucleus sampling parameter (0.0 disables nucleus sampling)
        
    Example usage:
        styles = {
            "orchestral": "A cinematic orchestral arrangement",
            "electronic": "An electronic dance track with synthesizers",
            "jazz": "A jazz arrangement with piano and saxophone"
        }
        generate_style_variations("melody.wav", styles)
    
    This will produce multiple WAV files, each containing a different stylistic
    interpretation of the same input melody.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine the best available device for computation
    # Using GPU acceleration is highly recommended for multiple generations
    if torch.backends.mps.is_available():
        device = "mps"  # Apple Silicon GPU
        print("Using MPS (Metal) for generation")
    elif torch.cuda.is_available():
        device = "cuda"  # NVIDIA GPU
        print("Using CUDA for generation")
    else:
        device = "cpu"  # Fallback to CPU
        print("Using CPU for generation (this will be slow)")
    
    # Load the MusicGen model
    # We only load the model once for all style variations to save time
    print(f"Loading MusicGen {model_size} model...")
    model = MusicGen.get_pretrained(model_size)
    model.to(device)
    
    # Configure generation parameters 
    # These settings apply to all style variations
    model.set_generation_params(
        duration=duration,        # Same duration for all variations
        temperature=temperature,  # Controls randomness
        top_k=top_k,              # Limits token sampling to top k
        top_p=top_p,              # Nucleus sampling threshold
    )
    
    # Load and process the melody file, creating visualization
    # We use the same melody for all style variations
    melody = load_and_process_melody(
        melody_file, 
        plot=True,  # Always create a waveform visualization for reference
        output_dir=output_dir
    )
    
    # Save the original processed melody for comparison
    torchaudio.save(
        os.path.join(output_dir, "original_melody.wav"), 
        melody, 
        32000  # MusicGen's standard sample rate
    )
    
    # Move melody tensor to the appropriate device
    melody = melody.to(device)
    
    # Generate a variation for each style in the dictionary
    # This is the main loop that creates all the different arrangements
    for style_name, prompt in styles.items():
        print(f"Generating {style_name} version with prompt: '{prompt}'")
        start_time = time.time()
        
        # Generate the audio using both the melody and the style-specific prompt
        # The model combines the melodic structure with the described style
        wav = model.generate_with_chroma([prompt], melody.unsqueeze(0))
        
        # Track and report performance
        generation_time = time.time() - start_time
        print(f"Generated in {generation_time:.2f} seconds")
        
        # Save this specific style variation
        output_path = os.path.join(output_dir, style_name)
        audio_write(output_path, wav[0].cpu(), model.sample_rate)
        
        print(f"Saved {style_name} version to {output_path}.wav")

def main():
    """
    Main function to parse command line arguments and run the melody conditioning script.
    
    This function sets up command-line argument handling to make the script easy to use
    from the terminal. It supports both single-style generation and multi-style exploration
    modes through the --variations flag.
    
    Command-line arguments:
        --melody: Path to melody file (required)
        --prompt: Text prompt describing the desired style
        --output: Directory to save outputs
        --model: Model size to use (small/medium/large)
        --duration: Duration in seconds
        --temperature: Controls randomness
        --variations: Flag to generate multiple style variations
        
    Example usage:
        # Generate a single style with a custom prompt:
        python musicgen_melody_conditioning.py --melody piano_melody.wav --prompt "A jazzy arrangement"
        
        # Generate multiple style variations from the same melody:
        python musicgen_melody_conditioning.py --melody guitar_melody.wav --variations --output guitar_variations
    """
    # Create command-line argument parser with descriptive help text
    parser = argparse.ArgumentParser(
        description="Generate music with melody conditioning using MusicGen",
        epilog="Example: python musicgen_melody_conditioning.py --melody piano.wav --variations"
    )
    
    # Define all command-line arguments with helpful descriptions
    parser.add_argument(
        "--melody", 
        type=str, 
        required=True, 
        help="Path to melody file (WAV, MP3, etc.)"
    )
    parser.add_argument(
        "--prompt", 
        type=str, 
        default="A cinematic orchestral arrangement with strings, brass and epic percussion", 
        help="Text prompt describing the desired music style"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="melody_output", 
        help="Output directory for generated audio files"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="medium", 
        choices=["small", "medium", "large"], 
        help="Model size to use (larger = better quality but slower)"
    )
    parser.add_argument(
        "--duration", 
        type=float, 
        default=10.0, 
        help="Duration of generated audio in seconds"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=1.0, 
        help="Temperature controls randomness (higher = more diverse)"
    )
    parser.add_argument(
        "--variations", 
        action="store_true", 
        help="Generate multiple style variations of the same melody"
    )
    
    # Parse the command-line arguments
    args = parser.parse_args()
    
    # Run in the appropriate mode based on user's request
    if args.variations:
        # When --variations flag is used, generate multiple style variations
        # Each style is defined by a name and a detailed text prompt
        print(f"Generating multiple style variations for melody: {args.melody}")
        
        # Define different style variations with descriptive prompts
        # These cover a range of distinct musical genres for comparison
        styles = {
            "orchestral": "A cinematic orchestral arrangement with strings, brass and epic percussion",
            "electronic": "An electronic dance track with synthesizers and driving beat",
            "jazz": "A jazz arrangement with piano, upright bass, and brushed drums",
            "rock": "A rock band arrangement with distorted guitars and powerful drums",
            "folk": "An acoustic folk arrangement with guitar, fiddle, and light percussion"
        }
        
        # Call the function to generate all style variations
        generate_style_variations(
            args.melody,
            styles,
            output_dir=args.output,
            model_size=args.model,
            duration=args.duration,
            temperature=args.temperature
        )
    else:
        # Generate a single version with the user's specified prompt
        print(f"Generating a single version with prompt: '{args.prompt}'")
        
        # Call the function to generate with melody conditioning
        generate_with_melody(
            args.prompt,
            args.melody,
            output_dir=args.output,
            model_size=args.model,
            duration=args.duration,
            temperature=args.temperature
        )

if __name__ == "__main__":
    main()
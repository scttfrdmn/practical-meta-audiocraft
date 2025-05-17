#!/usr/bin/env python3
# musicgen_web_app.py - Web interface for MusicGen audio generation using Gradio
import os
import torch
import gradio as gr
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import torchaudio
import time
import uuid
import json
import numpy as np
from scipy.signal import hann

# Create output directory for saving generated audio files
OUTPUTS_DIR = "music_generations"
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Global variable to store the model - this avoids reloading the model for each generation
model = None

def load_model(model_size):
    """
    Load the MusicGen model of specified size and move it to the appropriate device.
    
    This function implements the singleton pattern - it will only load a new model
    if the requested size differs from the currently loaded model.
    
    Args:
        model_size (str): Size of model to load ('small', 'medium', or 'large')
        
    Returns:
        str: Status message indicating successful model loading
    """
    global model
    
    print(f"Loading MusicGen {model_size} model...")
    
    # Determine the best available device for inference
    # MPS for Apple Silicon Macs, CUDA for NVIDIA GPUs, CPU as fallback
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS (Metal) for generation")
    elif torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA for generation")
    else:
        device = "cpu"
        print("Using CPU for generation (this will be slow)")
    
    # Load the pretrained model of specified size
    model = MusicGen.get_pretrained(model_size)
    model.to(device)  # Move model to the selected device
    
    return f"MusicGen {model_size} model loaded successfully on {device}"

def load_and_process_melody(melody_file):
    """
    Load and process a melody file for conditioning.
    
    This function loads an audio file, converts it to mono if needed,
    and resamples it to the 32kHz rate expected by MusicGen.
    
    Args:
        melody_file (str): Path to the melody audio file
        
    Returns:
        torch.Tensor: Processed melody tensor, or None if no file provided
    """
    if melody_file is None:
        return None
    
    # Load the audio file using torchaudio
    melody_wav, sr = torchaudio.load(melody_file)
    
    # If stereo (2+ channels), convert to mono by averaging channels
    if melody_wav.shape[0] > 1:
        melody_wav = melody_wav.mean(dim=0, keepdim=True)
    
    # Resample if needed (MusicGen always expects 32kHz sample rate)
    if sr != 32000:
        resampler = torchaudio.transforms.Resample(sr, 32000)
        melody_wav = resampler(melody_wav)
    
    return melody_wav

def generate_music(
    prompt, 
    duration, 
    temperature, 
    top_k, 
    top_p, 
    model_size="small",
    melody_file=None,
    autoplay=True
):
    """
    Generate music based on input parameters and return audio.
    
    This is the main generation function that handles both standard
    text-to-music generation and melody-conditioned generation.
    
    Args:
        prompt (str): Text description of the music to generate
        duration (float): Length of audio in seconds (1-30)
        temperature (float): Controls randomness (0.1-2.0)
        top_k (int): Number of top tokens to sample from
        top_p (float): Nucleus sampling parameter
        model_size (str): Size of model to use (small, medium, large)
        melody_file (str, optional): Path to melody file for conditioning
        autoplay (bool): Whether to autoplay audio in the UI
        
    Returns:
        tuple: (audio_file_path, generation_info)
    """
    global model
    
    # Load model if not already loaded or if model size changed
    # This optimizes performance by avoiding unnecessary model loading
    if model is None or model.name != model_size:
        load_model(model_size)
    
    # Set generation parameters that control the output characteristics
    model.set_generation_params(
        duration=float(duration),       # Length of the generated audio
        temperature=float(temperature), # Randomness/creativity control
        top_k=int(top_k),               # Limits token pool to top k options
        top_p=float(top_p),             # Dynamic token limiting (nucleus sampling)
    )
    
    # Start timing to measure generation performance
    start_time = time.time()
    
    # Process melody if provided for melody conditioning
    melody = None
    if melody_file is not None:
        melody = load_and_process_melody(melody_file)
    
    # Generate music - two different approaches based on whether we have a melody
    if melody is not None:
        # With melody conditioning - generates music that follows the melody
        print(f"Generating music with melody conditioning for: '{prompt}'")
        wav = model.generate_with_chroma([prompt], melody.unsqueeze(0).to(model.device))
    else:
        # Standard text-to-music generation
        print(f"Generating music for: '{prompt}'")
        wav = model.generate([prompt])
    
    # Calculate and log generation time for performance tracking
    generation_time = time.time() - start_time
    print(f"Generation completed in {generation_time:.2f} seconds")
    
    # Create a unique filename for this generation to avoid overwriting
    filename = f"{str(uuid.uuid4())[:8]}"
    output_path = os.path.join(OUTPUTS_DIR, filename)
    
    # Save the generated audio using audiocraft's utility
    audio_write(
        output_path, 
        wav[0].cpu(),       # Move tensor back to CPU for saving
        model.sample_rate,  # Use the model's sample rate (32kHz)
        strategy="loudness", # Apply loudness normalization
    )
    
    # Save metadata alongside audio for future reference
    # This allows tracking generation parameters for reproducibility
    metadata = {
        "prompt": prompt,
        "model_size": model_size,
        "parameters": {
            "duration": duration,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "melody_used": melody_file is not None
    }
    
    # Write metadata to JSON file with same base name as audio
    with open(f"{output_path}.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Return both the audio file path and generation information
    return (
        f"{output_path}.wav", 
        f"Generated in {generation_time:.2f}s with {model_size} model\nPrompt: {prompt}"
    )

def generate_variations(prompt, duration, temperature, top_k, top_p, model_size, num_variations):
    """
    Generate multiple variations with the same parameters.
    
    This function creates multiple audio samples from the same prompt and parameters,
    allowing for comparison of different outputs.
    
    Args:
        prompt (str): Text description of the music to generate
        duration (float): Length of audio in seconds
        temperature (float): Controls randomness
        top_k (int): Number of top tokens to sample from
        top_p (float): Nucleus sampling parameter
        model_size (str): Size of model to use
        num_variations (int): Number of variations to generate
        
    Returns:
        list: Paths to the generated audio files
    """
    global model
    
    # Load model if needed (implementation of singleton pattern)
    if model is None or model.name != model_size:
        load_model(model_size)
    
    # Set generation parameters
    model.set_generation_params(
        duration=float(duration),
        temperature=float(temperature),
        top_k=int(top_k),
        top_p=float(top_p),
    )
    
    # Generate multiple variations
    outputs = []
    
    for i in range(int(num_variations)):
        print(f"Generating variation {i+1}/{num_variations} for: '{prompt}'")
        # Each call to generate will produce a different output due to sampling
        wav = model.generate([prompt])
        
        # Create a unique filename for this variation
        filename = f"{str(uuid.uuid4())[:8]}_var{i+1}"
        output_path = os.path.join(OUTPUTS_DIR, filename)
        
        # Save the audio file
        audio_write(
            output_path, 
            wav[0].cpu(), 
            model.sample_rate, 
            strategy="loudness",
        )
        
        outputs.append(f"{output_path}.wav")
    
    # Fill in missing outputs with None if fewer than 4 variations
    # This ensures a consistent return structure for the UI
    while len(outputs) < 4:
        outputs.append(None)
    
    return outputs

def stitch_audio_segments(segment_paths, crossfade_duration=1.0):
    """
    Stitch multiple audio segments together with crossfading.
    
    This function combines multiple audio files into a single longer piece,
    applying crossfades between segments for smooth transitions.
    
    Args:
        segment_paths (list): List of paths to audio files to combine
        crossfade_duration (float): Length of crossfade in seconds
        
    Returns:
        tuple: (tensor of combined audio, sample rate)
    """
    segments = []
    sample_rate = None
    
    # Load all segments into memory
    for path in segment_paths:
        if path:  # Only process valid paths
            wav, sr = torchaudio.load(path)
            segments.append(wav.squeeze().numpy())  # Convert to numpy for processing
            if sample_rate is None:
                sample_rate = sr  # Use the first file's sample rate
    
    # Return early if no valid segments
    if not segments:
        return None, None
    
    # Calculate crossfade length in samples
    crossfade_samples = int(crossfade_duration * sample_rate)
    
    # Create crossfade window using Hann function
    # Hann window provides a smooth bell curve for natural-sounding fades
    window = hann(2 * crossfade_samples)
    fade_in = window[:crossfade_samples]   # First half for fade in
    fade_out = window[crossfade_samples:]  # Second half for fade out
    
    # Stitch segments with crossfading
    result = segments[0]  # Start with the first segment
    
    # Process each subsequent segment
    for i in range(1, len(segments)):
        # Ensure current result is long enough for crossfade
        if len(result) >= crossfade_samples:
            # Apply fade out to the end of the current result
            result[-crossfade_samples:] *= fade_out
            
            # Apply fade in to the beginning of the next segment
            next_segment = segments[i].copy()  # Copy to avoid modifying original
            next_segment[:crossfade_samples] *= fade_in
            
            # Create an overlap region where both segments contribute
            overlap = np.zeros(crossfade_samples)
            overlap += result[-crossfade_samples:]  # Add fade-out portion
            overlap += next_segment[:crossfade_samples]  # Add fade-in portion
            
            # Combine segments: first segment (minus fade region) + overlap + second segment
            result = np.concatenate([result[:-crossfade_samples], overlap, next_segment[crossfade_samples:]])
        else:
            # If the first segment is too short for crossfade, just concatenate
            result = np.concatenate([result, segments[i]])
    
    # Convert back to torch tensor for audiocraft compatibility
    return torch.tensor(result), sample_rate

def update_segments_list():
    """
    Update the list of available segments for composition.
    
    This function scans the output directory for generated audio files
    and returns them sorted by creation time (newest first).
    
    Returns:
        gr.CheckboxGroup.update: Updated checkbox group with audio files
    """
    # Get all wav files in the output directory
    files = [f for f in os.listdir(OUTPUTS_DIR) if f.endswith('.wav')]
    # Sort by modification time, newest first
    files.sort(key=lambda x: os.path.getmtime(os.path.join(OUTPUTS_DIR, x)), reverse=True)
    # Limit to most recent 10 files to avoid cluttering the UI
    files = files[:10]
    # Create full paths for the files
    file_paths = [os.path.join(OUTPUTS_DIR, f) for f in files]
    # Update the checkbox group component
    return gr.CheckboxGroup.update(choices=file_paths, value=[])

def create_composition(segment_paths, crossfade_duration):
    """
    Create a composition from multiple segments.
    
    This function takes selected audio segments and combines them
    into a single longer composition with crossfades.
    
    Args:
        segment_paths (list): List of paths to audio segments
        crossfade_duration (float): Length of crossfade in seconds
        
    Returns:
        str: Path to the composed audio file
    """
    if not segment_paths:
        return None
    
    print(f"Creating composition with {len(segment_paths)} segments")
    
    # Stitch segments together using the helper function
    composed_audio, sr = stitch_audio_segments(segment_paths, crossfade_duration)
    
    if composed_audio is None:
        return None
    
    # Create a unique filename for the composition
    composition_filename = f"composition_{str(uuid.uuid4())[:8]}"
    composition_path = os.path.join(OUTPUTS_DIR, composition_filename)
    
    # Convert to numpy for saving if it's a tensor
    composed_audio_np = composed_audio.numpy() if isinstance(composed_audio, torch.Tensor) else composed_audio
    
    # Save using torchaudio
    torchaudio.save(
        f"{composition_path}.wav",
        torch.tensor(composed_audio_np).unsqueeze(0),  # Add channel dimension
        sr
    )
    
    print(f"Composition saved to {composition_path}.wav")
    return f"{composition_path}.wav"

def create_interface():
    """
    Create the Gradio interface.
    
    This function builds the complete web UI with all interactive elements,
    including input controls, generation buttons, and audio players.
    
    Returns:
        gr.Blocks: The configured Gradio interface
    """
    # Define the interface using Gradio's Blocks API for custom layout
    with gr.Blocks(title="MusicGen Web Interface") as interface:
        gr.Markdown("# MusicGen Web Interface")
        gr.Markdown("Generate music from text descriptions using Meta's MusicGen model")
        
        with gr.Row():
            with gr.Column(scale=3):
                # Input section
                prompt_input = gr.Textbox(
                    label="Music Description",
                    placeholder="Describe the music you want to generate...",
                    lines=3,
                    value="An upbeat electronic dance track with a catchy melody and energetic rhythm"
                )
                
                with gr.Row():
                    with gr.Column():
                        model_dropdown = gr.Dropdown(
                            label="Model Size",
                            choices=["small", "medium", "large"],
                            value="small"
                        )
                        
                        duration_slider = gr.Slider(
                            label="Duration (seconds)",
                            minimum=1,
                            maximum=30,
                            value=10,
                            step=1
                        )
                    
                    with gr.Column():
                        temperature_slider = gr.Slider(
                            label="Temperature",
                            minimum=0.1,
                            maximum=2.0,
                            value=1.0,
                            step=0.1
                        )
                        
                        top_k_slider = gr.Slider(
                            label="Top-K",
                            minimum=50,
                            maximum=1000,
                            value=250,
                            step=50
                        )
                        
                        top_p_slider = gr.Slider(
                            label="Top-P",
                            minimum=0.0,
                            maximum=1.0,
                            value=0.0,
                            step=0.05
                        )
                
                melody_input = gr.Audio(
                    label="Optional Melody Reference (will be used to condition the generation)",
                    type="filepath"
                )
                
                generate_btn = gr.Button("Generate Music", variant="primary")
            
            with gr.Column(scale=2):
                # Output section
                audio_output = gr.Audio(type="filepath", label="Generated Music")
                info_output = gr.Textbox(label="Generation Info", lines=2)
                
                # Variations section
                with gr.Row():
                    variation_count = gr.Slider(
                        label="Number of Variations to Generate",
                        minimum=1,
                        maximum=4,
                        value=1,
                        step=1
                    )
                    
                    generate_variations_btn = gr.Button("Generate Variations", variant="secondary")
                
                # History section
                gr.Markdown("### Generation History")
                
                with gr.Row():
                    history_audio1 = gr.Audio(type="filepath", label="Sample 1", visible=False)
                    history_audio2 = gr.Audio(type="filepath", label="Sample 2", visible=False)
                
                with gr.Row():
                    history_audio3 = gr.Audio(type="filepath", label="Sample 3", visible=False)
                    history_audio4 = gr.Audio(type="filepath", label="Sample 4", visible=False)
        
        # Presets section
        with gr.Accordion("Prompt Presets", open=False):
            gr.Markdown("Click on any preset to use it")
            
            with gr.Row():
                preset1 = gr.Button("Electronic Dance")
                preset2 = gr.Button("Piano Ballad")
                preset3 = gr.Button("Epic Orchestral")
                preset4 = gr.Button("Jazz Quartet")
            
            with gr.Row():
                preset5 = gr.Button("Lo-fi Hip Hop")
                preset6 = gr.Button("Rock Band")
                preset7 = gr.Button("Ambient Soundscape")
                preset8 = gr.Button("Folk Acoustic")
        
        # Composition section
        with gr.Accordion("Composition Tool", open=False):
            gr.Markdown("### Create longer compositions by stitching multiple segments")
            
            with gr.Row():
                composition_segments = gr.CheckboxGroup(
                    label="Segments to Include",
                    choices=[],
                    value=[]
                )
                
                crossfade_slider = gr.Slider(
                    label="Crossfade Duration (seconds)",
                    minimum=0.5,
                    maximum=3.0,
                    value=1.0,
                    step=0.1
                )
            
            create_composition_btn = gr.Button("Create Composition")
            composition_output = gr.Audio(type="filepath", label="Composed Music")
        
        # Event handlers for generation
        generate_btn.click(
            fn=generate_music,
            inputs=[
                prompt_input,
                duration_slider,
                temperature_slider,
                top_k_slider,
                top_p_slider,
                model_dropdown,
                melody_input
            ],
            outputs=[audio_output, info_output]
        )
        
        # History tracking system
        current_index = 0
        history_outputs = [history_audio1, history_audio2, history_audio3, history_audio4]
        
        def update_history(audio_path, info):
            nonlocal current_index
            # Make this sample's history element visible
            history_outputs[current_index].visible = True
            # Update index for next generation (circular)
            current_index = (current_index + 1) % len(history_outputs)
            # Return all updated history elements
            return [audio_path] + [gr.update(visible=True) for _ in range(len(history_outputs))]
        
        generate_btn.click(
            fn=update_history,
            inputs=[audio_output, info_output],
            outputs=[history_outputs[0]] + history_outputs
        )
        
        # Connect variations button
        generate_variations_btn.click(
            fn=generate_variations,
            inputs=[
                prompt_input,
                duration_slider,
                temperature_slider,
                top_k_slider,
                top_p_slider,
                model_dropdown,
                variation_count
            ],
            outputs=history_outputs
        )
        
        # Connect updating segments list
        generate_btn.click(fn=update_segments_list, inputs=None, outputs=[composition_segments])
        generate_variations_btn.click(fn=update_segments_list, inputs=None, outputs=[composition_segments])
        
        # Connect composition button
        create_composition_btn.click(
            fn=create_composition,
            inputs=[composition_segments, crossfade_slider],
            outputs=[composition_output]
        )
        
        # Preset event handlers
        preset_prompts = {
            preset1: "An upbeat electronic dance track with a catchy melody and energetic rhythm",
            preset2: "A soft piano ballad with emotional melody and subtle string accompaniment",
            preset3: "An epic orchestral piece with powerful brass, dramatic percussion, and sweeping strings",
            preset4: "A jazz quartet with piano, bass, drums, and saxophone playing a smooth improvisational piece",
            preset5: "A lo-fi hip hop beat with relaxed drums, atmospheric samples, and a chill mood",
            preset6: "A rock band with electric guitars, bass, and energetic drums playing a memorable chord progression",
            preset7: "An ambient soundscape with ethereal pads, subtle textures, and a peaceful atmosphere",
            preset8: "A folk acoustic song with fingerpicked guitar, warm vocals, and natural feeling"
        }
        
        # Connect preset buttons to update the prompt
        for preset_btn, preset_text in preset_prompts.items():
            preset_btn.click(
                fn=lambda x: x,
                inputs=[gr.Textbox(value=preset_text, visible=False)],
                outputs=[prompt_input]
            )
    
    return interface

# Launch the interface
if __name__ == "__main__":
    print("Starting MusicGen Web Interface...")
    print(f"Output files will be saved to: {os.path.abspath(OUTPUTS_DIR)}")
    
    # Create the Gradio interface
    interface = create_interface()
    
    # Launch the interface
    interface.launch(share=False)  # Set share=True to generate a public URL
# Building a MusicGen Web Interface: Interactive Music Generation

This tutorial will guide you through creating a web-based interface for MusicGen that allows users to interactively generate and compare music samples. We'll use Gradio to create a simple but powerful UI that lets users adjust parameters and hear results in real-time.

## Introduction

A web interface makes MusicGen more accessible and practical for various users, from musicians to content creators. In this tutorial, you'll learn how to:

1. Create an interactive web interface with Gradio
2. Implement parameter controls for fine-tuning generation
3. Create a system for comparing and saving generated samples
4. Deploy your interface locally or share it temporarily

## Prerequisites

- AudioCraft successfully installed (see [Getting Started](../getting-started/README.md))
- Basic knowledge of Python and MusicGen (see [MusicGen Basics](musicgen_basics.md))
- Additional requirements: Gradio library (`pip install gradio`)

## Setting Up the Project

First, let's create a new Python script for our web application:

```python
# musicgen_web_app.py
import os
import torch
import gradio as gr
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import time
import uuid

# Create output directory
OUTPUTS_DIR = "music_generations"
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Global variable to store the model
model = None

def load_model(model_size):
    """
    Load the MusicGen model of specified size.
    """
    global model
    
    print(f"Loading MusicGen {model_size} model...")
    
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
    
    # Load model
    model = MusicGen.get_pretrained(model_size)
    model.to(device)
    
    return f"MusicGen {model_size} model loaded successfully on {device}"

def generate_music(
    prompt, 
    duration, 
    temperature, 
    top_k, 
    top_p, 
    model_size="small",
    autoplay=True
):
    """
    Generate music based on input parameters and return audio.
    """
    global model
    
    # Load model if not already loaded or if model size changed
    if model is None or model.name != model_size:
        load_model(model_size)
    
    # Set generation parameters
    model.set_generation_params(
        duration=float(duration),
        temperature=float(temperature),
        top_k=int(top_k),
        top_p=float(top_p),
    )
    
    # Start timing
    start_time = time.time()
    
    # Generate music
    print(f"Generating music for: '{prompt}'")
    wav = model.generate([prompt])
    
    # Calculate generation time
    generation_time = time.time() - start_time
    print(f"Generation completed in {generation_time:.2f} seconds")
    
    # Create a unique filename
    filename = f"{str(uuid.uuid4())[:8]}"
    output_path = os.path.join(OUTPUTS_DIR, filename)
    
    # Save the audio file
    audio_write(
        output_path, 
        wav[0].cpu(), 
        model.sample_rate, 
        strategy="loudness",
    )
    
    # Return audio file path and generation info
    return (
        f"{output_path}.wav", 
        f"Generated in {generation_time:.2f}s with {model_size} model\nPrompt: {prompt}"
    )

def create_interface():
    """
    Create the Gradio interface.
    """
    # Define the interface
    with gr.Blocks(title="MusicGen Web Interface") as interface:
        gr.Markdown("# MusicGen Web Interface")
        gr.Markdown("Generate music from text descriptions using Meta's MusicGen")
        
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
                
                generate_btn = gr.Button("Generate Music", variant="primary")
            
            with gr.Column(scale=2):
                # Output section
                audio_output = gr.Audio(type="filepath", label="Generated Music")
                info_output = gr.Textbox(label="Generation Info", lines=2)
                
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
        
        # Event handlers for generation
        generate_btn.click(
            fn=generate_music,
            inputs=[
                prompt_input,
                duration_slider,
                temperature_slider,
                top_k_slider,
                top_p_slider,
                model_dropdown
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
            outputs=[history_outputs[0]] + [history_output for history_output in history_outputs]
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
    interface = create_interface()
    interface.launch(share=False)  # Set share=True to generate a public URL
```

Save this file as `musicgen_web_app.py` and run it with `python musicgen_web_app.py`. This will start a local web server with your MusicGen interface.

## Understanding the Interface Components

Our web application has several key components:

### 1. Model Management

We've created a global variable to store the MusicGen model and a function to load models of different sizes. This allows users to switch between models without restarting the application:

```python
def load_model(model_size):
    global model
    
    print(f"Loading MusicGen {model_size} model...")
    
    # Determine device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    # Load model
    model = MusicGen.get_pretrained(model_size)
    model.to(device)
    
    return f"MusicGen {model_size} model loaded successfully on {device}"
```

### 2. Music Generation Function

The `generate_music` function handles parameter processing and music generation:

```python
def generate_music(prompt, duration, temperature, top_k, top_p, model_size="small", autoplay=True):
    global model
    
    # Load model if needed
    if model is None or model.name != model_size:
        load_model(model_size)
    
    # Set generation parameters
    model.set_generation_params(
        duration=float(duration),
        temperature=float(temperature),
        top_k=int(top_k),
        top_p=float(top_p),
    )
    
    # Generate music
    wav = model.generate([prompt])
    
    # Save and return the audio
    # ...
```

### 3. Gradio Interface

We use Gradio's `Blocks` API to create a flexible, multi-component interface:

```python
with gr.Blocks(title="MusicGen Web Interface") as interface:
    # Input section with text prompt and parameter sliders
    # ...
    
    # Output section with audio player and generation info
    # ...
    
    # History section to compare multiple generations
    # ...
    
    # Preset section with common prompt templates
    # ...
```

### 4. History Tracking

The application keeps track of recent generations for easy comparison:

```python
def update_history(audio_path, info):
    nonlocal current_index
    # Make this sample's history element visible
    history_outputs[current_index].visible = True
    # Update index for next generation (circular)
    current_index = (current_index + 1) % len(history_outputs)
    # Return all updated history elements
    return [audio_path] + [gr.update(visible=True) for _ in range(len(history_outputs))]
```

## Enhancements to the Basic Interface

Now let's enhance our application with more advanced features:

### 1. Adding Melody Conditioning

Let's add the ability to condition the generation on a melody by uploading a reference audio file:

```python
# Add to the imports at the top
import torchaudio

# Add this function
def load_and_process_melody(melody_file):
    """
    Load and process a melody file for conditioning.
    """
    if melody_file is None:
        return None
    
    # Load the audio file
    melody_wav, sr = torchaudio.load(melody_file)
    
    # If stereo, convert to mono
    if melody_wav.shape[0] > 1:
        melody_wav = melody_wav.mean(dim=0, keepdim=True)
    
    # Resample if needed (MusicGen expects 32kHz)
    if sr != 32000:
        resampler = torchaudio.transforms.Resample(sr, 32000)
        melody_wav = resampler(melody_wav)
    
    return melody_wav

# Modify the generate_music function to accept a melody parameter
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
    global model
    
    # Load model if needed
    if model is None or model.name != model_size:
        load_model(model_size)
    
    # Set generation parameters
    model.set_generation_params(
        duration=float(duration),
        temperature=float(temperature),
        top_k=int(top_k),
        top_p=float(top_p),
    )
    
    # Process melody if provided
    melody = None
    if melody_file is not None:
        melody = load_and_process_melody(melody_file)
    
    # Generate music
    if melody is not None:
        # With melody conditioning
        print(f"Generating music with melody conditioning for: '{prompt}'")
        wav = model.generate_with_chroma([prompt], melody.unsqueeze(0).to(model.device))
    else:
        # Without melody conditioning
        print(f"Generating music for: '{prompt}'")
        wav = model.generate([prompt])
    
    # Save and return the audio
    # ... (rest of function remains the same)
```

Add the melody input component to the interface:

```python
# In the interface creation function, add this to the input section
melody_input = gr.Audio(
    label="Optional Melody Reference (will be used to condition the generation)",
    type="filepath"
)

# Update the generate_btn.click function inputs to include melody_input
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
```

### 2. Adding Multi-Generation Comparison

Let's add a feature to generate multiple variations with the same parameters for comparison:

```python
# Add to the interface
with gr.Row():
    variation_count = gr.Slider(
        label="Number of Variations to Generate",
        minimum=1,
        maximum=4,
        value=1,
        step=1
    )
    
    generate_variations_btn = gr.Button("Generate Variations", variant="secondary")

# Add this function
def generate_variations(prompt, duration, temperature, top_k, top_p, model_size, num_variations):
    """
    Generate multiple variations with the same parameters.
    """
    global model
    
    # Load model if needed
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
        wav = model.generate([prompt])
        
        # Create a unique filename
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
    while len(outputs) < 4:
        outputs.append(None)
    
    return outputs

# Connect the variations button
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
    outputs=[history_audio1, history_audio2, history_audio3, history_audio4]
)
```

### 3. Adding Audio Metadata Saving

Let's add a feature to save the generation parameters with the audio files for future reference:

```python
# Add to the imports
import json

# Modify the generate_music function to save metadata
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
    # ... (existing code)
    
    # Create a unique filename
    filename = f"{str(uuid.uuid4())[:8]}"
    output_path = os.path.join(OUTPUTS_DIR, filename)
    
    # Save the audio file
    audio_write(
        output_path, 
        wav[0].cpu(), 
        model.sample_rate, 
        strategy="loudness",
    )
    
    # Save metadata
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
    
    with open(f"{output_path}.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # ... (rest of function)
```

## Deploying Your Web Interface

You can deploy your MusicGen web interface in several ways:

### 1. Local Deployment

The simplest way to run the interface is locally:

```bash
python musicgen_web_app.py
```

This starts a local web server, typically at http://127.0.0.1:7860.

### 2. Temporary Public URL

To share your interface with others temporarily, set `share=True` when launching:

```python
interface.launch(share=True)  # Generates a temporary public URL
```

This creates a temporary URL that's valid for 72 hours.

### 3. Persistent Deployment

For persistent deployment, you can use services like:

- **Hugging Face Spaces**: Free hosting for Gradio apps
- **Streamlit**: Another platform supporting web apps
- **Your own server**: Deploy on AWS, Google Cloud, etc.

## Example: Creating a MusicGen Composition Tool

Let's expand our web interface into a composition tool by enabling the creation of longer music sequences:

```python
# Add this function
def stitch_audio_segments(segment_paths, crossfade_duration=1.0):
    """
    Stitch multiple audio segments together with crossfading.
    """
    import numpy as np
    from scipy.signal import hann
    
    segments = []
    sample_rate = None
    
    # Load all segments
    for path in segment_paths:
        if path:
            wav, sr = torchaudio.load(path)
            segments.append(wav.squeeze().numpy())
            if sample_rate is None:
                sample_rate = sr
    
    if not segments:
        return None, None
    
    # Calculate crossfade samples
    crossfade_samples = int(crossfade_duration * sample_rate)
    
    # Create crossfade window
    window = hann(2 * crossfade_samples)
    fade_in = window[:crossfade_samples]
    fade_out = window[crossfade_samples:]
    
    # Stitch segments with crossfading
    result = segments[0]
    
    for i in range(1, len(segments)):
        # Ensure current result is long enough for crossfade
        if len(result) >= crossfade_samples:
            # Apply fade out to the end of the current result
            result[-crossfade_samples:] *= fade_out
            
            # Apply fade in to the beginning of the next segment
            next_segment = segments[i].copy()
            next_segment[:crossfade_samples] *= fade_in
            
            # Overlap and add
            overlap = np.zeros(crossfade_samples)
            overlap += result[-crossfade_samples:]
            overlap += next_segment[:crossfade_samples]
            
            # Combine
            result = np.concatenate([result[:-crossfade_samples], overlap, next_segment[crossfade_samples:]])
        else:
            # If the first segment is too short, just concatenate
            result = np.concatenate([result, segments[i]])
    
    return torch.tensor(result), sample_rate

# Add this to the interface
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

# Function to update the segments checkbox group
def update_segments_list():
    # Get all wav files in the output directory
    files = [f for f in os.listdir(OUTPUTS_DIR) if f.endswith('.wav')]
    files.sort(key=lambda x: os.path.getmtime(os.path.join(OUTPUTS_DIR, x)), reverse=True)
    # Limit to most recent 10 files
    files = files[:10]
    # Create full paths
    file_paths = [os.path.join(OUTPUTS_DIR, f) for f in files]
    # Update the checkbox
    return gr.CheckboxGroup.update(choices=file_paths, value=[])

# Connect the update function
generate_btn.click(fn=update_segments_list, inputs=None, outputs=[composition_segments])
generate_variations_btn.click(fn=update_segments_list, inputs=None, outputs=[composition_segments])

# Function to create composition
def create_composition(segment_paths, crossfade_duration):
    if not segment_paths:
        return None
    
    print(f"Creating composition with {len(segment_paths)} segments")
    
    # Stitch segments together
    composed_audio, sr = stitch_audio_segments(segment_paths, crossfade_duration)
    
    if composed_audio is None:
        return None
    
    # Save the composition
    composition_filename = f"composition_{str(uuid.uuid4())[:8]}"
    composition_path = os.path.join(OUTPUTS_DIR, composition_filename)
    
    # Convert to numpy and save
    composed_audio_np = composed_audio.numpy() if isinstance(composed_audio, torch.Tensor) else composed_audio
    
    # Save using torchaudio
    torchaudio.save(
        f"{composition_path}.wav",
        torch.tensor(composed_audio_np).unsqueeze(0),
        sr
    )
    
    print(f"Composition saved to {composition_path}.wav")
    return f"{composition_path}.wav"

# Connect the composition button
create_composition_btn.click(
    fn=create_composition,
    inputs=[composition_segments, crossfade_slider],
    outputs=[composition_output]
)
```

## Exercises

Here are some exercises to enhance your MusicGen web application:

1. **Style Transfer Interface**: Create a tab in the UI that lets users upload a reference audio file and attempt to generate music in a similar style.

2. **Batch Processing**: Add a feature to process a list of prompts in batch mode, generating all samples at once.

3. **Custom Presets System**: Expand the presets to include both prompt text and parameter settings, allowing users to save and load their favorite configurations.

4. **Audio Visualization**: Add visual representations of the generated audio using waveform or spectrogram displays.

5. **Multi-Model Comparison**: Create a side-by-side comparison feature that generates the same prompt with different model sizes.

## Conclusion

You've now created a powerful web interface for MusicGen that enables interactive music generation. This interface makes AudioCraft more accessible and practical for various use cases, from music composition to sound design.

With the features we've implemented, users can:
- Generate music with detailed control over parameters
- Compare variations side by side
- Use melody conditioning for more guided generation
- Create longer compositions by stitching segments

Try extending this application with additional features from the exercises to create an even more powerful music generation tool.

## Next Steps

1. Explore the [Advanced MusicGen](musicgen_advanced.md) tutorial for more techniques
2. Learn about [AudioGen](../audiogen/README.md) for sound effect generation
3. Integrate your web interface with other tools using our [API Integration](../advanced-techniques/api_integration.md) guide
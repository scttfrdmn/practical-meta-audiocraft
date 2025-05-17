# Building Web Interfaces with Gradio

This tutorial guides you through creating interactive web interfaces for AudioCraft using Gradio. Gradio allows you to build beautiful, user-friendly interfaces for your audio generation models with minimal code.

## Prerequisites

- AudioCraft successfully installed
- Basic understanding of Python
- Familiarity with web concepts (optional)

## Installation

Install Gradio alongside AudioCraft:

```bash
pip install gradio
```

## Why Gradio for AudioCraft?

Gradio offers several advantages for AudioCraft applications:

1. **Simplicity**: Create UIs with just a few lines of code
2. **Audio Support**: Built-in components for audio playback and visualization
3. **Interactive**: Real-time parameter adjustment and generation
4. **Shareable**: Easily share demos with others via temporary links
5. **Customizable**: Flexible layouts and styling options

## Basic AudioCraft Interface

Let's start with a simple MusicGen interface:

```python
# simple_musicgen_interface.py
import gradio as gr
import torch
from audiocraft.models import MusicGen

# Load model
model = MusicGen.get_pretrained('small')

# Detect device and move model
if torch.cuda.is_available():
    device = "cuda"
    model.to(device)
elif torch.backends.mps.is_available():
    device = "mps"
    model.to(device)
else:
    device = "cpu"
    
print(f"Using device: {device}")

def generate_music(prompt, duration, model_size, temperature=1.0):
    """Generate music based on text prompt."""
    # Load appropriate model if changed
    global model
    if model.name != model_size:
        print(f"Loading {model_size} model...")
        model = MusicGen.get_pretrained(model_size)
        model.to(device)
        
    # Set generation parameters
    model.set_generation_params(
        duration=duration,
        temperature=temperature,
    )
    
    # Generate audio
    print(f"Generating: '{prompt}'")
    wav = model.generate([prompt])
    
    # Return audio for playback
    return (model.sample_rate, wav[0].cpu().numpy())

# Create the Gradio interface
demo = gr.Interface(
    fn=generate_music,
    inputs=[
        gr.Textbox(
            label="Music Description", 
            placeholder="Enter a description of the music you want to generate",
            value="A gentle piano melody with ambient pads in the background"
        ),
        gr.Slider(minimum=1, maximum=30, value=5, step=1, label="Duration (seconds)"),
        gr.Radio(["small", "medium", "large"], label="Model Size", value="small"),
        gr.Slider(minimum=0.1, maximum=2.0, value=1.0, step=0.1, label="Temperature")
    ],
    outputs=gr.Audio(label="Generated Music", type="numpy"),
    title="MusicGen: Text-to-Music Generation",
    description="Generate music from text descriptions using Meta's MusicGen model."
)

# Launch the interface
if __name__ == "__main__":
    demo.launch()
```

Save this as `simple_musicgen_interface.py` and run:

```bash
python simple_musicgen_interface.py
```

This will start a local web server (typically at http://127.0.0.1:7860) with your interface.

## Advanced Gradio Interface

Let's create a more comprehensive interface with additional features:

```python
# advanced_musicgen_interface.py
import gradio as gr
import torch
import torchaudio
import os
import time
import uuid
import matplotlib.pyplot as plt
import numpy as np
from audiocraft.models import MusicGen

# Create output directory for generations
OUTPUT_DIR = "music_generations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set up global model cache
models = {}

def get_model(model_size):
    """Load and cache model of specified size."""
    if model_size not in models:
        print(f"Loading {model_size} model...")
        models[model_size] = MusicGen.get_pretrained(model_size)
        
        # Determine device
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
            
        models[model_size].to(device)
        print(f"Model loaded on {device}")
    
    return models[model_size]

def visualize_waveform(audio_data, sample_rate):
    """Create waveform visualization of audio."""
    plt.figure(figsize=(10, 2))
    plt.plot(audio_data)
    plt.title("Waveform")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    
    # Create a unique filename
    filename = f"{uuid.uuid4()}_waveform.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    
    return filepath

def visualize_spectrogram(audio_data, sample_rate):
    """Create spectrogram visualization of audio."""
    plt.figure(figsize=(10, 4))
    
    # Calculate the spectrogram
    n_fft = 2048
    hop_length = 512
    spec = np.abs(librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length))
    spec_db = librosa.amplitude_to_db(spec, ref=np.max)
    
    # Plot the spectrogram
    librosa.display.specshow(spec_db, sr=sample_rate, hop_length=hop_length, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Spectrogram")
    
    # Create a unique filename
    filename = f"{uuid.uuid4()}_spectrogram.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    
    return filepath

def process_melody(melody_file):
    """Process a melody file for conditioning."""
    if melody_file is None:
        return None, None
        
    # Load melody file
    melody, sr = torchaudio.load(melody_file)
    
    # Convert to mono if stereo
    if melody.shape[0] > 1:
        melody = melody.mean(dim=0, keepdim=True)
    
    # Resample if needed
    if sr != 32000:
        resampler = torchaudio.transforms.Resample(sr, 32000)
        melody = resampler(melody)
    
    # Return the processed melody and sample rate
    return melody, 32000

def generate_with_params(
    prompt, 
    duration, 
    model_size, 
    use_melody,
    melody_file,
    temperature, 
    top_k, 
    top_p,
    cfg_coef,
    seed
):
    """Generate music with all parameters."""
    # Set random seed if provided
    if seed is not None and seed > 0:
        torch.manual_seed(seed)
    
    # Get appropriate model
    model = get_model(model_size)
    
    # Set generation parameters
    model.set_generation_params(
        duration=duration,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        cfg_coef=cfg_coef
    )
    
    # Start timing
    start_time = time.time()
    
    # Determine if we're using melody conditioning
    if use_melody and melody_file is not None:
        # Process melody file
        melody, _ = process_melody(melody_file)
        
        # Move melody to same device as model
        device = next(model.parameters()).device
        melody = melody.to(device)
        
        # Generate with melody conditioning
        print(f"Generating with melody conditioning: '{prompt}'")
        wav = model.generate_with_chroma([prompt], melody.unsqueeze(0))
    else:
        # Generate without melody
        print(f"Generating: '{prompt}'")
        wav = model.generate([prompt])
    
    # Calculate generation time
    generation_time = time.time() - start_time
    print(f"Generated in {generation_time:.2f} seconds")
    
    # Save the file for later reference
    filename = f"{uuid.uuid4()}"
    filepath = os.path.join(OUTPUT_DIR, f"{filename}.wav")
    torchaudio.save(filepath, wav[0].cpu(), model.sample_rate)
    
    # Get audio data as numpy array for visualization
    audio_data = wav[0].cpu().numpy()
    
    # Create visualizations
    waveform_img = visualize_waveform(audio_data, model.sample_rate)
    spectrogram_img = None  # Only compute if needed
    
    # Return results
    generation_info = f"Generated in {generation_time:.2f}s with {model_size} model\nPrompt: {prompt}\nSeed: {seed if seed else 'random'}"
    
    return (
        (model.sample_rate, audio_data),  # Audio for playback
        waveform_img,                     # Waveform visualization
        filepath,                         # File path for download
        generation_info                   # Generation info
    )

# Presets for different musical styles
style_presets = {
    "Electronic": "An electronic dance track with a catchy melody, driving beat, and synthesizer pads",
    "Orchestral": "An epic orchestral piece with sweeping strings, brass fanfare, and dramatic percussion",
    "Acoustic": "A gentle acoustic guitar melody with piano accompaniment and soft strings in the background",
    "Jazz": "A cool jazz piece with piano, upright bass, brushed drums, and saxophone solo",
    "Lo-Fi": "A lo-fi hip hop beat with relaxed drums, mellow piano, and atmospheric samples",
    "Rock": "An energetic rock track with distorted electric guitars, driving drums, and powerful vocals",
    "Ambient": "A peaceful ambient soundscape with ethereal pads, subtle textures, and gentle evolution",
    "8-bit": "An 8-bit video game soundtrack with chiptune sounds, simple melody, and nostalgic feeling"
}

# Create the Gradio interface with tabs and more features
with gr.Blocks(title="AudioCraft MusicGen Studio") as demo:
    gr.Markdown("# AudioCraft MusicGen Studio")
    gr.Markdown("Generate music from text descriptions using Meta's MusicGen model")
    
    with gr.Tabs():
        # Simple generation tab
        with gr.TabItem("Basic Generation"):
            with gr.Row():
                with gr.Column(scale=4):
                    prompt_input = gr.Textbox(
                        label="Music Description", 
                        placeholder="Enter a description of the music you want to generate",
                        value="A gentle piano melody with ambient pads in the background",
                        lines=3
                    )
                    
                with gr.Column(scale=1):
                    style_preset = gr.Dropdown(
                        choices=list(style_presets.keys()),
                        label="Style Preset",
                        value=None
                    )
            
            with gr.Row():
                with gr.Column(scale=1):
                    model_size = gr.Radio(
                        ["small", "medium", "large"], 
                        label="Model Size", 
                        value="small",
                        info="Larger models produce better quality but take longer"
                    )
                    duration = gr.Slider(
                        minimum=1, maximum=30, value=5, step=1, 
                        label="Duration (seconds)"
                    )
                    
                with gr.Column(scale=1):
                    temperature = gr.Slider(
                        minimum=0.1, maximum=2.0, value=1.0, step=0.1, 
                        label="Temperature",
                        info="Higher = more creative, Lower = more coherent"
                    )
                    seed = gr.Number(
                        label="Random Seed", 
                        value=0,
                        info="Set to 0 for random generation each time"
                    )
            
            generate_btn = gr.Button("Generate Music", variant="primary")
            
            with gr.Row():
                with gr.Column(scale=2):
                    audio_output = gr.Audio(
                        label="Generated Music", 
                        type="numpy"
                    )
                    generation_info = gr.Textbox(
                        label="Generation Info", 
                        interactive=False
                    )
                    
                with gr.Column(scale=1):
                    waveform_output = gr.Image(
                        label="Waveform", 
                        type="filepath"
                    )
                    file_output = gr.File(
                        label="Download Generated Audio"
                    )
            
        # Advanced generation tab
        with gr.TabItem("Advanced Options"):
            with gr.Row():
                with gr.Column(scale=2):
                    advanced_prompt = gr.Textbox(
                        label="Music Description", 
                        placeholder="Enter a detailed description of the music you want to generate",
                        value="A complex jazz fusion composition with intricate piano chords, syncopated drums, and a walking bass line",
                        lines=3
                    )
                    
                with gr.Column(scale=1):
                    adv_model_size = gr.Radio(
                        ["small", "medium", "large"], 
                        label="Model Size", 
                        value="medium"
                    )
                    adv_duration = gr.Slider(
                        minimum=1, maximum=30, value=10, step=1, 
                        label="Duration (seconds)"
                    )
            
            with gr.Row():
                with gr.Column():
                    use_melody = gr.Checkbox(
                        label="Use Melody Conditioning",
                        value=False
                    )
                    melody_input = gr.Audio(
                        label="Melody Reference (optional)",
                        type="filepath",
                        visible=False
                    )
                    
                with gr.Column():
                    adv_temperature = gr.Slider(
                        minimum=0.1, maximum=2.0, value=1.0, step=0.1, 
                        label="Temperature"
                    )
                    top_k = gr.Slider(
                        minimum=50, maximum=1000, value=250, step=10, 
                        label="Top-K",
                        info="Number of highest probability tokens to consider"
                    )
                    top_p = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.0, step=0.05, 
                        label="Top-P", 
                        info="Nucleus sampling: 0.0 to disable, >0 for dynamic token selection"
                    )
                    cfg_coef = gr.Slider(
                        minimum=1.0, maximum=10.0, value=3.0, step=0.1,
                        label="Classifier-Free Guidance",
                        info="How strictly to follow the prompt (higher = more faithful)"
                    )
                    adv_seed = gr.Number(
                        label="Random Seed", 
                        value=42
                    )
            
            adv_generate_btn = gr.Button("Generate Music", variant="primary")
            
            with gr.Row():
                with gr.Column(scale=2):
                    adv_audio_output = gr.Audio(
                        label="Generated Music", 
                        type="numpy"
                    )
                    adv_generation_info = gr.Textbox(
                        label="Generation Info", 
                        interactive=False
                    )
                    
                with gr.Column(scale=1):
                    adv_waveform_output = gr.Image(
                        label="Waveform", 
                        type="filepath"
                    )
                    adv_file_output = gr.File(
                        label="Download Generated Audio"
                    )
        
        # History and comparison tab
        with gr.TabItem("History & Compare"):
            gr.Markdown("### Generation History")
            gr.Markdown("Your recent generations will appear here for comparison.")
            
            with gr.Row():
                history_audio1 = gr.Audio(label="Sample 1", visible=False, type="filepath")
                history_audio2 = gr.Audio(label="Sample 2", visible=False, type="filepath")
            
            with gr.Row():
                history_audio3 = gr.Audio(label="Sample 3", visible=False, type="filepath")
                history_audio4 = gr.Audio(label="Sample 4", visible=False, type="filepath")
    
    # Set up event handlers
    
    # Apply style presets
    def apply_preset(preset_name):
        if preset_name in style_presets:
            return style_presets[preset_name]
        return ""
    
    style_preset.change(apply_preset, inputs=style_preset, outputs=prompt_input)
    
    # Toggle melody input visibility
    use_melody.change(
        lambda x: gr.update(visible=x),
        inputs=use_melody,
        outputs=melody_input
    )
    
    # Basic generation tab events
    generate_btn.click(
        generate_with_params,
        inputs=[
            prompt_input, duration, model_size, 
            gr.Checkbox(value=False, visible=False),  # Hidden use_melody
            gr.Audio(visible=False),                  # Hidden melody_file
            temperature, 
            gr.Slider(value=250, visible=False),      # Hidden top_k
            gr.Slider(value=0.0, visible=False),      # Hidden top_p
            gr.Slider(value=3.0, visible=False),      # Hidden cfg_coef
            seed
        ],
        outputs=[audio_output, waveform_output, file_output, generation_info]
    )
    
    # Advanced generation tab events
    adv_generate_btn.click(
        generate_with_params,
        inputs=[
            advanced_prompt, adv_duration, adv_model_size,
            use_melody, melody_input, adv_temperature,
            top_k, top_p, cfg_coef, adv_seed
        ],
        outputs=[adv_audio_output, adv_waveform_output, adv_file_output, adv_generation_info]
    )
    
    # History tracking function
    current_index = 0
    history_outputs = [history_audio1, history_audio2, history_audio3, history_audio4]
    
    def update_history(audio, waveform, file_path, info):
        nonlocal current_index
        history_outputs[current_index].visible = True
        history_outputs[current_index].value = file_path
        current_index = (current_index + 1) % len(history_outputs)
        return [audio, waveform, file_path, info] + [gr.update(visible=True) for _ in history_outputs]
    
    # Connect history updates to both generation buttons
    generate_btn.click(
        update_history,
        inputs=[audio_output, waveform_output, file_output, generation_info],
        outputs=[audio_output, waveform_output, file_output, generation_info] + history_outputs
    )
    
    adv_generate_btn.click(
        update_history,
        inputs=[adv_audio_output, adv_waveform_output, adv_file_output, adv_generation_info],
        outputs=[adv_audio_output, adv_waveform_output, adv_file_output, adv_generation_info] + history_outputs
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()
```

Save this as `advanced_musicgen_interface.py` and run:

```bash
python advanced_musicgen_interface.py
```

This advanced interface includes:
- Tabbed layout for basic and advanced options
- Style presets for quick generation
- Melody conditioning support
- Audio waveform visualization
- Generation history for comparisons
- Advanced parameter controls

## Deployment Options

### Local Sharing

To temporarily share your Gradio app with others:

```python
demo.launch(share=True)
```

This will generate a public URL that's valid for 72 hours.

### Hosted Deployment

You can deploy your Gradio interface to Hugging Face Spaces:

1. Create a GitHub repository with your Gradio app
2. Include `requirements.txt`:
   ```
   gradio>=3.50.2
   audiocraft
   torch>=2.1.0
   torchaudio>=2.1.0
   matplotlib
   ```
3. Create a `README.md` file with app documentation
4. Create a Hugging Face Space using the "Gradio" option
5. Connect it to your GitHub repository

### Embedding in Existing Web Applications

You can embed Gradio interfaces in other web applications:

```python
# Create your interface
demo = gr.Interface(...)

# Get the HTML for embedding
html_embed = demo.get_embed_html()

# Now you can insert html_embed into your existing web page
```

## Customizing the Interface

### Custom Layout

Gradio's Blocks API allows for complete layout customization:

```python
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=2):
            # Left column content
            prompt_input = gr.Textbox(...)
            parameter_slider = gr.Slider(...)
        
        with gr.Column(scale=1):
            # Right column content
            image_output = gr.Image(...)
    
    with gr.Row():
        # Bottom row content
        audio_output = gr.Audio(...)
```

### Custom CSS

Add custom styling to your Gradio app:

```python
custom_css = """
.gradio-container {
    background-color: #f5f5f5;
}
.main-header {
    font-family: 'Arial', sans-serif;
    font-size: 2.5rem;
    color: #2c3e50;
}
"""

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# MusicGen Studio", elem_classes=["main-header"])
    # Rest of your interface
```

### Custom JavaScript

For interactive features, you can add custom JavaScript:

```python
custom_js = """
function updateCounter(seconds) {
    const counterElem = document.getElementById('generation-counter');
    counterElem.textContent = seconds;
}
"""

with gr.Blocks(js=custom_js) as demo:
    # Your interface components
```

## Performance Optimization

### Model Caching

Cache models to avoid reloading between generations:

```python
# Global model cache
models = {}

def get_model(model_size):
    if model_size not in models:
        models[model_size] = MusicGen.get_pretrained(model_size)
    return models[model_size]
```

### Progress Indicators

Add progress indicators for long-running generations:

```python
def generate_with_progress(prompt, duration, model_size):
    progress = gr.Progress(track_tqdm=True)
    
    # Load model
    progress(0.1, desc="Loading model...")
    model = get_model(model_size)
    
    # Prepare for generation
    progress(0.3, desc="Setting up generation...")
    model.set_generation_params(duration=duration)
    
    # Generate audio
    progress(0.5, desc="Generating audio...")
    wav = model.generate([prompt])
    
    # Process output
    progress(0.9, desc="Processing output...")
    # ... processing code ...
    
    progress(1.0, desc="Complete!")
    return (model.sample_rate, wav[0].cpu().numpy())
```

### Background Processing

For time-consuming operations, use background processing with `gr.FlaggingCallback`:

```python
def process_in_background(prompt, duration, model_size):
    # This function will run in the background
    # ... generation code ...
    return result

demo = gr.Interface(
    fn=process_in_background,
    inputs=[...],
    outputs=[...],
    flagging_callback=gr.FlaggingCallback()
)
```

## Building a Two-Model Interface

Let's create an interface that demonstrates both MusicGen and AudioGen side-by-side:

```python
# dual_model_interface.py
import gradio as gr
import torch
from audiocraft.models import MusicGen, AudioGen

# Cache for models
models = {
    "MusicGen": {},
    "AudioGen": {}
}

def get_model(model_type, model_size):
    """Load and cache models."""
    if model_size not in models[model_type]:
        if model_type == "MusicGen":
            models[model_type][model_size] = MusicGen.get_pretrained(model_size)
        else:
            models[model_type][model_size] = AudioGen.get_pretrained(model_size)
        
        # Move to appropriate device
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
            
        models[model_type][model_size].to(device)
        
    return models[model_type][model_size]

def generate_audio(model_type, prompt, duration, model_size, temperature):
    """Generate audio with either MusicGen or AudioGen."""
    model = get_model(model_type, model_size)
    
    # Set generation parameters
    model.set_generation_params(
        duration=duration,
        temperature=temperature
    )
    
    # Generate audio
    print(f"Generating {model_type} audio: '{prompt}'")
    wav = model.generate([prompt])
    
    # Return audio for playback
    return (model.sample_rate, wav[0].cpu().numpy())

# Create the dual interface
with gr.Blocks(title="AudioCraft: MusicGen vs AudioGen") as demo:
    gr.Markdown("# AudioCraft: MusicGen vs AudioGen")
    gr.Markdown("Compare music generation and sound effect generation from the same prompt")
    
    with gr.Row():
        prompt_input = gr.Textbox(
            label="Audio Description", 
            placeholder="Enter a description of the audio you want to generate",
            value="Thunderstorm with heavy rain and distant thunder",
            lines=3
        )
    
    with gr.Row():
        with gr.Column():
            music_model_size = gr.Radio(
                ["small", "medium", "large"], 
                label="MusicGen Model Size", 
                value="small"
            )
            music_temperature = gr.Slider(
                minimum=0.1, maximum=2.0, value=1.0, step=0.1, 
                label="Temperature"
            )
        
        with gr.Column():
            audio_model_size = gr.Radio(
                ["small", "medium", "large"], 
                label="AudioGen Model Size", 
                value="small"
            )
            audio_temperature = gr.Slider(
                minimum=0.1, maximum=2.0, value=1.0, step=0.1, 
                label="Temperature"
            )
    
    duration = gr.Slider(
        minimum=1, maximum=30, value=5, step=1, 
        label="Duration (seconds)"
    )
    
    generate_btn = gr.Button("Generate Both", variant="primary")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### MusicGen Output")
            gr.Markdown("Optimized for music generation")
            music_output = gr.Audio(label="Generated Music", type="numpy")
            
        with gr.Column():
            gr.Markdown("### AudioGen Output")
            gr.Markdown("Optimized for sound effects and environmental audio")
            audio_output = gr.Audio(label="Generated Sound", type="numpy")
    
    # Event handler for generation
    generate_btn.click(
        lambda p, d, ms, mt: generate_audio("MusicGen", p, d, ms, mt),
        inputs=[prompt_input, duration, music_model_size, music_temperature],
        outputs=music_output
    )
    
    generate_btn.click(
        lambda p, d, ms, mt: generate_audio("AudioGen", p, d, ms, mt),
        inputs=[prompt_input, duration, audio_model_size, audio_temperature],
        outputs=audio_output
    )

# Launch the demo
if __name__ == "__main__":
    demo.launch()
```

This interface allows users to compare MusicGen and AudioGen outputs from the same prompt, helping them understand the differences between the models.

## Example Prompt Libraries

Adding a built-in prompt library helps users get started:

```python
# Add these prompt libraries to your interface
music_prompts = {
    "Electronic": [
        "A pulsing electronic dance track with driving beats and arpeggiated synthesizers",
        "A futuristic synthwave track with retro 80s feel and modern production",
        "A melodic techno piece with hypnotic rhythm and evolving textures"
    ],
    "Orchestral": [
        "An epic orchestral piece with soaring strings and powerful brass",
        "A delicate classical composition featuring piano and string quartet",
        "A dramatic film score with tense percussion and emotional strings"
    ],
    "Ambient": [
        "A peaceful ambient soundscape with gentle pads and subtle textures",
        "A meditative drone piece with evolving timbres and spatial effects",
        "A calm atmospheric track with field recordings and soft synthesizers"
    ]
}

sound_prompts = {
    "Nature": [
        "Forest ambience with birds, wind in trees, and distant stream",
        "Ocean waves crashing on a rocky shore with seagulls",
        "Thunderstorm with heavy rain, wind, and rolling thunder"
    ],
    "Urban": [
        "Busy city street with traffic, conversations, and construction",
        "Subway station with train arrivals, announcements, and footsteps",
        "Cafe ambience with coffee machines, quiet conversations, and background music"
    ],
    "Mechanical": [
        "Old mechanical clock with ticking, gears, and chimes",
        "Factory machinery with rhythmic metallic sounds and steam",
        "Vintage car engine starting, idling, and accelerating"
    ]
}

# Then add dropdown or button components to select these prompts
```

## Conclusion

This tutorial has shown you how to build interactive web interfaces for AudioCraft using Gradio. From a simple one-function interface to advanced applications with tabs, visualization, and multiple models, Gradio makes it easy to showcase audio generation capabilities.

Key takeaways include:
- Gradio provides a quick way to create UIs for AudioCraft models
- Advanced interfaces can include visualizations and audio comparison
- A good interface should include presets and examples for users
- Models can be cached to improve performance
- Gradio apps can be easily shared and deployed

By creating intuitive interfaces, you make AI audio generation accessible to users without technical knowledge of the underlying models.

## Next Steps

- Check out the [React Integration Guide](react_integration.md) for more advanced web applications
- See [Deployment Strategies](docker_containerization.md) for hosting your Gradio app
- Learn about [API Development](rest_api.md) to serve AudioCraft models to multiple clients
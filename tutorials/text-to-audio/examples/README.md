# Text-to-Audio Integration Examples

This directory contains example scripts for building integrated text-to-audio pipelines using AudioCraft's MusicGen and AudioGen models together.

## Available Examples

1. **text_to_audio_pipeline.py** - Base pipeline for combining different audio generation models
2. **audio_scene_generator.py** - Creates complete audio scenes with multiple components
3. **dynamic_audio_environment.py** - Generates parameterized environmental audio
4. **text_to_audio_web.py** - Creates a web interface for text-to-audio generation (requires Gradio)

## Running the Examples

First, make sure you have AudioCraft installed. Then, you can run any example:

```bash
# Run the basic pipeline
python text_to_audio_pipeline.py

# Generate a complete audio scene
python audio_scene_generator.py --scene 0 --output my_scenes

# Create a dynamic environment
python dynamic_audio_environment.py --type forest --mood calm --instrument piano --time morning --weather clear

# Launch the web interface (requires Gradio)
# pip install gradio
# python text_to_audio_web.py
```

## Command Line Options

Many scripts support command line arguments:

```bash
# Generate a cafe scene
python audio_scene_generator.py --scene 1

# Create a space environment with specific parameters
python dynamic_audio_environment.py --type space --mood mysterious --instrument "synthesizer pads" --duration 20.0
```

## Output

Examples save generated audio files to different directories:
- `audio_output/` - Basic pipeline outputs
- `scenes/` - Complete audio scenes
- `environments/` - Dynamic environmental audio
- `web_output/` - Files generated through the web interface

## Requirements

- Python 3.9+
- PyTorch 2.1.0+
- AudioCraft library
- torchaudio
- Gradio (optional, for web interface)

## Core Concepts

These examples demonstrate several important concepts:
1. **Multi-model integration** - Using MusicGen and AudioGen together
2. **Audio mixing** - Combining multiple generated audio elements
3. **Parameterized generation** - Creating dynamic audio based on parameters
4. **Component-based design** - Building audio scenes from individual elements

## Extension Ideas

Some ways to extend these examples:
- Add audio post-processing effects (reverb, EQ, compression)
- Create audio transitions between different scenes
- Integrate TTS for narrated content
- Build more complex web applications
- Add real-time audio generation capabilities
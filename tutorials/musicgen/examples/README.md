# MusicGen Examples

This directory contains example scripts for exploring Meta's MusicGen audio generation capabilities.

## Available Examples

1. **musicgen_basic.py** - Basic music generation from text prompts
2. **musicgen_genre_explorer.py** - Generates music across different genres
3. **musicgen_temperature_explorer.py** - Explores how temperature affects generation
4. **musicgen_prompt_explorer.py** - Demonstrates different prompt techniques

## Running the Examples

First, make sure you have AudioCraft installed. Then, you can run any example:

```bash
# Basic example
python musicgen_basic.py

# Genre exploration
python musicgen_genre_explorer.py

# Temperature exploration
python musicgen_temperature_explorer.py

# Prompt techniques exploration
python musicgen_prompt_explorer.py
```

## Command Line Options

Some scripts support command line arguments:

```bash
# Run with custom parameters (prompt explorer)
python musicgen_prompt_explorer.py --model medium --duration 8.0 --output my_samples
```

## Output

All examples save generated audio files to dedicated output directories:
- `music_output/` - Basic examples
- `genre_samples/` - Genre explorations
- `temperature_samples/` - Temperature variations
- `prompt_samples/` - Prompt technique samples

## Requirements

- Python 3.9+
- PyTorch 2.1.0+
- AudioCraft library
- Sufficient memory for model loading (varies by model size)

## Notes

- Most examples use the "small" model by default, which requires less memory
- First generation is always slower due to model loading
- Using GPU acceleration (CUDA) or Apple Silicon (MPS) is highly recommended
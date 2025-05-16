# AudioGen Examples

This directory contains example scripts for exploring Meta's AudioGen sound generation capabilities.

## Available Examples

1. **audiogen_basic.py** - Basic sound effect generation from text prompts
2. **audiogen_category_explorer.py** - Generates sounds across different categories
3. **audiogen_ambient_generator.py** - Creates ambient soundscapes for different environments
4. **audiogen_sound_combiner.py** - Demonstrates combining multiple sound elements

## Running the Examples

First, make sure you have AudioCraft installed. Then, you can run any example:

```bash
# Basic example
python audiogen_basic.py

# Category exploration
python audiogen_category_explorer.py

# Ambient soundscape generation
python audiogen_ambient_generator.py --environment forest --duration 15.0

# Sound combination
python audiogen_sound_combiner.py
```

## Command Line Options

Some scripts support command line arguments:

```bash
# Generate an ocean ambient soundscape for 20 seconds
python audiogen_ambient_generator.py --environment ocean --duration 20.0 --model medium --temp 0.7 --output my_ambiences
```

## Output

All examples save generated audio files to dedicated output directories:
- `sound_output/` - Basic examples
- `sound_categories/` - Category explorations
- `ambient_soundscapes/` - Ambient environment sounds
- `combined_sounds/` - Layered sound combinations
  - `combined_sounds/elements/` - Individual sound elements before mixing

## Requirements

- Python 3.9+
- PyTorch 2.1.0+
- AudioCraft library
- Sufficient memory for model loading

## Notes

- AudioGen has only "medium" and "large" model sizes (no "small" option)
- First generation is always slower due to model loading
- Using GPU acceleration (CUDA) or Apple Silicon (MPS) is highly recommended
- For best results with ambient generation, use durations of 10 seconds or longer
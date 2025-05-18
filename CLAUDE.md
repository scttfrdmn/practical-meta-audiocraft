# AudioCraft Tutorial Project - Claude Code Guide

This document provides guidance for Claude Code when working with the AudioCraft tutorial project, including key commands, conventions, and important notes.

## Project Overview

This repository contains comprehensive tutorials for Meta's AudioCraft framework, covering music generation (MusicGen), sound effects (AudioGen), and deployment strategies. The tutorials are organized in a progressive learning path from basic to advanced.

## Repository Structure

- `/tutorials/` - Main tutorial content
  - `/getting-started/` - Installation and basic usage guides
  - `/musicgen/` - MusicGen-specific tutorials
  - `/audiogen/` - AudioGen-specific tutorials
  - `/deployment/` - Deployment and integration guides
  - `troubleshooting.md` - Common issues and solutions
  - `glossary.md` - Terminology reference
- `README.md` - Project overview
- `TUTORIAL_OVERVIEW.md` - Complete tutorial structure and navigation
- `CLAUDE.md` - This file

## Code Comments Enhancement Plan

To ensure that all code examples in this repository are educational and easy to follow, we've implemented a comprehensive commenting strategy:

### Comment Enhancement Goals

1. **Educational Value**: Comments should explain the "why" behind code choices, not just the "what"
2. **Accessibility**: Make examples approachable for beginners while still covering advanced concepts
3. **Clarity**: Use clear, concise language with consistent terminology
4. **Completeness**: Cover all key elements of each script, especially custom parameters
5. **Cross-platform**: Note hardware-specific considerations for different environments

### Comment Structure Guidelines

For each example script, include:

1. **File Header**: Brief description of the script's purpose
2. **Function/Class Documentation**:
   - Comprehensive docstrings explaining purpose, arguments, and return values
   - Usage examples where appropriate
   - Notes about limitations or edge cases
3. **Section Comments**:
   - Brief explanation of each logical section of code
   - Parameter explanations, especially for ML-specific settings
4. **Key Line Comments**:
   - Insights on important or non-obvious lines of code
   - Hardware-specific considerations
5. **Implementation Notes**:
   - Alternative approaches that could be taken
   - Performance considerations
   - Memory usage tips

### Enhanced Scripts

The following scripts have received enhanced comments:

1. **Melody Conditioning** (`/tutorials/musicgen/examples/musicgen_melody_conditioning.py`):
   - Detailed explanation of melody conditioning process
   - Audio format requirements and preprocessing steps
   - Optimal parameter settings for melody-conditioned generation
   - Command-line interface documentation

2. **Basic AudioGen** (`/tutorials/audiogen/examples/audiogen_basic.py`):
   - Clear distinction between AudioGen and MusicGen capabilities
   - Sound prompt engineering guidance
   - Hardware and performance considerations
   - Example prompt categorization (environmental, mechanical, etc.)

3. **Text-to-Audio Pipeline** (`/tutorials/text-to-audio/examples/text_to_audio_pipeline.py`):
   - Unified pipeline architecture explanation
   - Model loading and memory management strategies
   - Audio mixing and processing techniques
   - Advanced use cases and extensions

4. **Genre Explorer** (`/tutorials/musicgen/examples/musicgen_genre_explorer.py`):
   - Musical genre characteristics in prompts
   - Generation parameter optimization for different genres
   - File organization and batch processing
   - Progress tracking for better user experience

5. **Temperature Explorer** (`/tutorials/musicgen/examples/musicgen_temperature_explorer.py`):
   - Detailed explanation of temperature's effect on generation
   - Reference documentation generation
   - Optimal temperature ranges for different use cases
   - Comparative analysis approach

6. **Audio Scene Generator** (`/tutorials/text-to-audio/examples/audio_scene_generator.py`):
   - Scene composition strategies
   - Declarative scene description format
   - Component mixing techniques
   - Automated documentation for generated scenes

### Comment Style Example

```python
# Load and process the melody file for conditioning
# This handles format conversion, resampling, and mono conversion
melody = load_and_process_melody(
    melody_file, 
    plot=plot_waveform,    # Create visualization of input melody 
    output_dir=output_dir  # Where to save visualization and processed audio
)

# Move the melody tensor to the same device as the model
# This is essential for proper operation and prevents CUDA/MPS errors
melody = melody.to(device)

# Generate audio with both text and melody conditioning
# The generate_with_chroma method combines both condition types
# Note: melody needs an extra dimension since the model expects a batch
wav = model.generate_with_chroma([prompt], melody.unsqueeze(0))
```

## Key Commands

### Environment Setup

```bash
# Create Python environment
conda create -n audiocraft python=3.9
conda activate audiocraft

# Install PyTorch with MPS support (Mac)
pip install torch==2.1.0 torchaudio==2.1.0

# Install PyTorch with CUDA support (Windows/Linux)
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install AudioCraft
pip install -U audiocraft

# Install additional dependencies
pip install matplotlib jupyter gradio
```

### Basic AudioCraft Usage

```python
import torch
from audiocraft.models import MusicGen

# Determine device
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Load model
model = MusicGen.get_pretrained('small')  # Options: 'small', 'medium', 'large'
model.to(device)

# Set generation parameters
model.set_generation_params(
    duration=10.0,       # Duration in seconds (max 30)
    temperature=1.0,     # Controls randomness (0.1-2.0)
    top_k=250,           # Controls diversity
    top_p=0.0,           # Nucleus sampling (0.0 to disable)
    cfg_coef=3.0         # Classifier-free guidance scale (1.0-10.0)
)

# Generate audio
prompt = "An upbeat electronic track with a catchy melody"
wav = model.generate([prompt])

# Save the audio
from audiocraft.data.audio import audio_write
audio_write("output", wav[0].cpu(), model.sample_rate)
```

## Important Technical Notes

### Hardware Requirements

- **GPU Acceleration**: CUDA (NVIDIA) or MPS (Apple Silicon) strongly recommended
- **Memory Requirements**:
  - small model: ~2GB VRAM
  - medium model: ~4GB VRAM
  - large model: ~8GB VRAM

### Performance Considerations

- First generation is always slower due to model loading
- Expected generation times for 10 seconds of audio:
  - CUDA: 5-15 seconds
  - MPS (Metal): 15-45 seconds
  - CPU: 1-5 minutes

### Common Issues

- Out of memory errors: Use smaller model or shorter duration
- Slow generation: First run includes model loading time
- Audio quality issues: Use larger models and detailed prompts

## Code Style Guidelines

1. **Comprehensive Error Handling**:
   ```python
   try:
       # Code that might fail
   except Exception as e:
       print(f"Error: {str(e)}")
       # Appropriate fallback
   ```

2. **Device Handling**:
   ```python
   # Standard device detection pattern
   if torch.backends.mps.is_available():
       device = "mps"
   elif torch.cuda.is_available():
       device = "cuda"
   else:
       device = "cpu"
   ```

3. **Parameter Validation**:
   ```python
   # Always validate parameters
   if duration < 1.0 or duration > 30.0:
       raise ValueError("Duration must be between 1 and 30 seconds")
   ```

4. **Resource Cleanup**:
   ```python
   # Clear memory after usage
   import gc
   torch.cuda.empty_cache()
   gc.collect()
   ```

## Documentation Standards

When creating or updating tutorials:

1. Start with clear prerequisites
2. Include both basic and advanced usage examples
3. Provide troubleshooting tips for common errors
4. Link to related tutorials for further learning
5. Explain parameters and their effects in detail
6. Include sample outputs or expected behavior

## Testing Requirements

Test all code examples with:
- Multiple model sizes
- Different generation parameters
- All supported hardware configurations
- Edge cases (very short/long prompts)

## Implementation Checklist

- [x] Check for required imports
- [x] Validate user inputs
- [x] Detect hardware and configure appropriately
- [x] Include proper error handling
- [x] Implement cleanup to avoid memory leaks
- [x] Add clear documentation for each function
- [x] Test on supported platforms

## Additional Resources

- [AudioCraft GitHub Repository](https://github.com/facebookresearch/audiocraft)
- [Meta AI Blog: AudioCraft](https://ai.meta.com/blog/audiocraft-musicgen-audiogen-encodec-generative-ai-audio/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Gradio Documentation](https://www.gradio.app/docs/)

## Working with this Project

When adding to or modifying tutorials:
1. Ensure consistency with existing content
2. Follow the progressive learning path
3. Test examples thoroughly
4. Update related documentation
5. Maintain cross-platform compatibility
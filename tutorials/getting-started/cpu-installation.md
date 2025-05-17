# CPU-Only Installation Guide for AudioCraft

This guide walks you through setting up Meta's AudioCraft framework on systems without GPU acceleration, using CPU-only processing. While generation will be slower, this approach makes AudioCraft accessible on almost any modern computer.

## System Requirements

- Modern multi-core CPU (4+ cores recommended)
- 8GB RAM minimum (16GB+ recommended)
- Python 3.9 (recommended for AudioCraft)
- 5GB+ free disk space for models and environment
- Any operating system (Windows, macOS, Linux)

## Installation Steps

### 1. Set Up a Python Environment

We recommend using a virtual environment to avoid package conflicts.

#### Using Conda (Recommended)

```bash
# Install Miniconda (if not already installed)
# Visit https://docs.conda.io/en/latest/miniconda.html for OS-specific installers

# Create and activate a new environment
conda create -n audiocraft python=3.9
conda activate audiocraft
```

#### Using venv (Alternative)

```bash
# Create a virtual environment
python -m venv audiocraft-env

# Activate the environment
# On Windows
audiocraft-env\Scripts\activate
# On macOS/Linux
source audiocraft-env/bin/activate
```

### 2. Install PyTorch (CPU Version)

```bash
# Using pip (all platforms)
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu

# Or using conda
conda install pytorch==2.1.0 torchaudio==2.1.0 cpuonly -c pytorch
```

### 3. Install AudioCraft

```bash
# Install from PyPI
pip install audiocraft

# Or install the latest version from GitHub
# pip install git+https://github.com/facebookresearch/audiocraft.git
```

### 4. Install Additional Dependencies

```bash
# Install ffmpeg for audio processing
# On Windows (using conda)
conda install ffmpeg -c conda-forge

# On macOS (using Homebrew)
brew install ffmpeg

# On Linux (Ubuntu/Debian)
sudo apt update
sudo apt install -y ffmpeg
```

## Verifying Installation

Create a simple test script to ensure AudioCraft works correctly:

```python
# test_audiocraft_cpu.py
import torch
from audiocraft.models import MusicGen
import time

# Confirm we're using CPU
print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

# Time the model loading
start_time = time.time()
print("Loading MusicGen small model...")
model = MusicGen.get_pretrained('small')
print(f"Model loaded in {time.time() - start_time:.2f} seconds")

# Set shorter generation length for testing
model.set_generation_params(
    duration=3.0  # 3 seconds for quick testing
)

# Generate a simple test
prompt = "A gentle piano melody"
print(f"Generating: '{prompt}'")

start_time = time.time()
wav = model.generate([prompt])
generation_time = time.time() - start_time

print(f"Generation completed in {generation_time:.2f} seconds")
print(f"Audio shape: {wav.shape}, Sample rate: {model.sample_rate}")
print("Installation verified successfully!")
```

Run this script with `python test_audiocraft_cpu.py`. A successful test will load the model, generate a short audio clip, and report timing information.

## CPU Optimization Strategies

When running AudioCraft on CPU, consider these optimizations:

### 1. Model Size Selection

Always use the `small` model size for CPU inference:

```python
model = MusicGen.get_pretrained('small')  # or AudioGen.get_pretrained('small')
```

### 2. Shorter Generation Lengths

```python
model.set_generation_params(
    duration=5.0,  # Keep under 10 seconds for reasonable generation times
)
```

### 3. Enable Intel MKL Optimizations (if available)

On Intel CPUs, ensure Intel MKL (Math Kernel Library) is being used by PyTorch:

```bash
# Set environment variables for better performance with PyTorch + MKL
# Windows (in Command Prompt before running Python)
set MKL_NUM_THREADS=4
set OMP_NUM_THREADS=4

# Linux/macOS
export MKL_NUM_THREADS=4
export OMP_NUM_THREADS=4
```

Adjust the number of threads to match your CPU core count.

### 4. Batch Processing for Multiple Generations

If you need to create multiple audio samples, use batch processing for better efficiency:

```python
# Generate multiple samples at once
prompts = [
    "Soft piano melody",
    "Gentle acoustic guitar",
    "Calm ambient music"
]

# Single batch generation is more efficient than multiple individual generations
wavs = model.generate(prompts)
```

## Expected Performance

The generation time on CPU varies significantly depending on your hardware:

| CPU Type | Generation Time (5 seconds audio) | Model Loading Time |
|----------|----------------------------------|-------------------|
| Laptop (dual-core) | 3-5 minutes | 30-60 seconds |
| Desktop (quad-core) | 1-3 minutes | 20-40 seconds |
| Performance CPU (8+ cores) | 30-90 seconds | 10-30 seconds |

Note that the first generation is always slower due to model initialization.

## Memory Optimization

CPU-only generation can still be memory-intensive. If you encounter memory issues:

1. **Close other memory-intensive applications**
2. **Use the small model size only**
3. **Generate shorter audio clips**
4. **Clear memory between generations**:

```python
import gc

# After generation
del model
gc.collect()

# Then reload the model if needed
model = MusicGen.get_pretrained('small')
```

## Running AudioCraft in Google Colab (Alternative)

If CPU-only generation is too slow on your system, consider using Google Colab with its free GPU:

1. Visit [Google Colab](https://colab.research.google.com/)
2. Create a new notebook
3. Enable GPU: Runtime > Change runtime type > Hardware accelerator > GPU
4. Install and run AudioCraft:

```python
!pip install audiocraft

import torch
from audiocraft.models import MusicGen

# Check for GPU
print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

# Load model and generate
model = MusicGen.get_pretrained('small')
model.set_generation_params(duration=5.0)
wav = model.generate(["Upbeat electronic music with a catchy melody"])

# Play in Colab
from IPython.display import Audio
Audio(wav.cpu().numpy()[0], rate=model.sample_rate)
```

## Troubleshooting

### Common Issues

1. **Memory Errors**
   - Reduce model size to 'small'
   - Generate shorter audio clips
   - Close other applications
   - Use a system with more RAM

2. **Slow Generation**
   - CPU generation is inherently slower
   - Use the 'small' model size
   - Keep audio duration short (<10 seconds)
   - Consider cloud alternatives or Google Colab

3. **Missing Modules or Import Errors**
   - Ensure you're in the correct virtual environment
   - Reinstall dependencies with `pip install -r requirements.txt`
   - Verify PyTorch installation: `python -c "import torch; print(torch.__version__)"`

4. **Audio Quality Issues**
   - The 'small' model produces lower quality audio
   - Try different prompts for better results
   - Post-process audio with an audio editor if needed

## Next Steps

Now that you have AudioCraft installed with CPU support:

1. Try the [Basic Usage Tutorial](basic-usage.md) with shorter generation times
2. Explore [MusicGen](../musicgen/README.md) or [AudioGen](../audiogen/README.md) capabilities
3. Consider using a cloud service with GPU support for faster generation

## Getting Help

If you encounter issues, check:
- [AudioCraft GitHub Issues](https://github.com/facebookresearch/audiocraft/issues)
- [PyTorch Forums](https://discuss.pytorch.org/)

Remember that while CPU generation is slower, it's still viable for experimentation and learning how AudioCraft works, particularly with the 'small' model size and shorter audio clips.
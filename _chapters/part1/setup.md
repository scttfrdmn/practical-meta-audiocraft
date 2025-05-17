---
layout: chapter
title: "Chapter 2: Setting Up Your Environment"
# Copyright © 2025 Scott Friedman.
# Licensed under Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.
difficulty: beginner
estimated_time: 2 hours
scenario:
  quote: "I'm excited to try generating AI audio, but I've hit a roadblock with the setup. I'm not sure which version of Python to use, how to handle GPU support, or why I'm getting dependency errors. I just want a reliable environment where I can experiment without fighting with installation issues."
  persona: "Jordan Kim"
  role: "Game Developer"
next_steps:
  - title: "Understanding AudioCraft Architecture"
    url: "/chapters/part1/architecture/"
    description: "Learn how AudioCraft's components work together"
  - title: "Your First Audio Generation"
    url: "/chapters/part1/first-generation/"
    description: "Generate your first AI audio with AudioCraft"
further_reading:
  - title: "PyTorch Installation Guide"
    url: "https://pytorch.org/get-started/locally/"
  - title: "Conda Environment Management"
    url: "https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html"
  - title: "CUDA Toolkit Documentation"
    url: "https://docs.nvidia.com/cuda/"
  - title: "Metal Performance Shaders Documentation"
    url: "https://developer.apple.com/documentation/metalperformanceshaders"
---

# Chapter 2: Setting Up Your Environment

## The Challenge

Setting up a proper development environment for AI audio generation can be surprisingly challenging. Unlike standard Python applications, machine learning frameworks like AudioCraft require specific versions of PyTorch, CUDA drivers for NVIDIA GPUs, or Metal Performance Shaders for Apple Silicon. Getting these dependencies right is crucial for performance—the difference between waiting 30 seconds or 10 minutes for a generation can be simply having the right environment configuration.

While AudioCraft itself can be installed with a single pip command, ensuring it runs efficiently requires careful setup of the underlying machine learning infrastructure. The process varies significantly between operating systems and hardware configurations, making it easy to fall into compatibility pitfalls.

In this chapter, we'll establish a robust, efficient environment for AudioCraft that will serve as the foundation for all our future work. We'll cover multiple hardware configurations (CPU, NVIDIA GPU, and Apple Silicon) across different operating systems, ensuring you get optimal performance from your specific setup.

## Learning Objectives

By the end of this chapter, you'll be able to:

- Create an isolated Python environment for AudioCraft
- Install the correct PyTorch version for your hardware
- Set up AudioCraft with all dependencies
- Configure GPU acceleration (if applicable)
- Verify your environment works correctly
- Troubleshoot common installation issues

## Prerequisites

Before we begin, make sure you have:

- Administrative access to your computer (for installing software)
- At least 5GB of free disk space
- Internet connection for downloading packages
- Basic familiarity with command line operations

## Environment Setup Strategies

There are multiple ways to set up a Python environment for AudioCraft. We'll cover three main approaches:

### 1. Conda Environment (Recommended)

Using Conda provides the most reliable isolation and dependency management across all platforms. This approach:

- Creates a fully isolated environment
- Handles complex dependencies automatically
- Works consistently across platforms
- Makes it easy to switch between projects

### 2. Virtual Environment with venv

Python's built-in venv module provides a lighter-weight alternative:

- Creates a basic isolated environment
- Requires more manual dependency management
- Uses less disk space than Conda
- Works well for simpler setups

### 3. Direct Installation

While not recommended for most users, direct installation is the simplest approach:

- Installs packages in your system Python
- May cause conflicts with other projects
- Requires less initial setup
- More likely to encounter dependency issues

We'll focus primarily on the Conda approach as it offers the best balance of reliability and ease of use, while noting differences for the other methods where relevant.

## Hardware-Specific Setup

AudioCraft's performance varies significantly based on your hardware. Let's explore the setup process for different configurations.

### Setting Up for NVIDIA GPUs (Windows/Linux)

NVIDIA GPUs with CUDA support offer the fastest generation times for AudioCraft. Here's how to set up your environment for NVIDIA hardware:

```python
# 1. Install Miniconda (if not already installed)
# Download from https://docs.conda.io/en/latest/miniconda.html

# 2. Create a new Conda environment
conda create -n audiocraft python=3.9
conda activate audiocraft

# 3. Install PyTorch with CUDA support
# For CUDA 11.8 (most common)
conda install pytorch==2.1.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# 4. Install AudioCraft
pip install audiocraft
```

#### Important NVIDIA-Specific Considerations:

1. **CUDA Version**: Ensure your NVIDIA driver supports the CUDA version you install. You can check your driver's supported CUDA version with `nvidia-smi`.

2. **Memory Requirements**: AudioCraft models have different VRAM requirements:
   - Small model: ~2GB VRAM
   - Medium model: ~4GB VRAM
   - Large model: ~8GB VRAM

3. **Windows-Specific Note**: On Windows, you may need to install Visual C++ Build Tools if you encounter compilation errors.

### Setting Up for Apple Silicon (M1/M2/M3/M4 Macs)

Apple Silicon Macs use Metal Performance Shaders (MPS) for GPU acceleration:

```python
# 1. Install Miniconda (if not already installed)
# Download from https://docs.conda.io/en/latest/miniconda.html

# 2. Create a new Conda environment
conda create -n audiocraft python=3.9
conda activate audiocraft

# 3. Install PyTorch with MPS support
pip install torch==2.1.0 torchaudio==2.1.0

# 4. Install AudioCraft
pip install audiocraft
```

#### Important Apple Silicon Considerations:

1. **Metal Device**: PyTorch automatically uses the Metal device when available. You'll access it with `device = "mps"` in your code.

2. **Performance**: While not as fast as high-end NVIDIA GPUs, M-series chips provide significant acceleration compared to CPU. Expect:
   - M1: 2-3x faster than CPU
   - M2: 3-4x faster than CPU
   - M3/M4: 4-5x faster than CPU

3. **Memory Sharing**: MPS uses shared memory with your system RAM, so ensure you have sufficient free memory (at least 8GB recommended, 16GB preferred).

### Setting Up for CPU-Only (All Platforms)

If you don't have a compatible GPU, you can still run AudioCraft on CPU:

```python
# 1. Install Miniconda (if not already installed)
# Download from https://docs.conda.io/en/latest/miniconda.html

# 2. Create a new Conda environment
conda create -n audiocraft python=3.9
conda activate audiocraft

# 3. Install PyTorch CPU version
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu

# 4. Install AudioCraft
pip install audiocraft
```

#### Important CPU Considerations:

1. **Performance**: CPU generation is significantly slower:
   - Small model: 1-2 minutes per 10-second clip
   - Medium model: 3-5 minutes per 10-second clip
   - Large model: 8-15 minutes per 10-second clip

2. **Memory Usage**: CPU generation uses more system RAM:
   - Small model: ~4GB RAM
   - Medium model: ~8GB RAM
   - Large model: ~16GB RAM

3. **Optimization**: The smaller models are strongly recommended for CPU-only setups.

## Step-by-Step Installation Guide

Now let's walk through a complete installation process that will work for most users.

### 1. Installing Miniconda

Miniconda gives us a lightweight Python distribution with the Conda package manager:

#### Windows:
1. Download the Miniconda installer from [the official site](https://docs.conda.io/en/latest/miniconda.html)
2. Run the installer and follow the prompts
3. Select "Add Miniconda to PATH" when prompted
4. Open a new Command Prompt or PowerShell window

#### macOS:
1. Download the Miniconda installer from [the official site](https://docs.conda.io/en/latest/miniconda.html)
2. Open Terminal and navigate to the download location
3. Run `bash Miniconda3-latest-MacOSX-arm64.sh` (for Apple Silicon) or `bash Miniconda3-latest-MacOSX-x86_64.sh` (for Intel)
4. Follow the prompts and let it initialize Conda
5. Close and reopen Terminal

#### Linux:
1. Download the Miniconda installer from [the official site](https://docs.conda.io/en/latest/miniconda.html)
2. Open a terminal and navigate to the download location
3. Run `bash Miniconda3-latest-Linux-x86_64.sh`
4. Follow the prompts and let it initialize Conda
5. Close and reopen your terminal

### 2. Creating a Conda Environment

Now we'll create an isolated environment for AudioCraft:

```bash
# Create a new environment with Python 3.9
conda create -n audiocraft python=3.9

# Activate the environment
conda activate audiocraft
```

You should see `(audiocraft)` at the beginning of your command prompt, indicating the environment is active.

### 3. Installing PyTorch

Next, we'll install PyTorch with the appropriate configuration for your hardware:

#### For NVIDIA GPUs:
```bash
# Check your CUDA version first (if you have the NVIDIA toolkit installed)
nvcc --version

# For CUDA 11.8
conda install pytorch==2.1.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# For CUDA 12.1
conda install pytorch==2.1.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```

#### For Apple Silicon:
```bash
pip install torch==2.1.0 torchaudio==2.1.0
```

#### For CPU-only:
```bash
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
```

### 4. Installing AudioCraft

With PyTorch properly installed, we can now install AudioCraft:

```bash
pip install audiocraft
```

This command installs the latest stable version of AudioCraft and its dependencies.

### 5. Installing Additional Helpful Packages

Let's add a few packages that will be useful throughout this book:

```bash
# For visualization and audio playback
pip install matplotlib jupyterlab ipywidgets

# For web interfaces (we'll use this in later chapters)
pip install gradio

# For audio processing
pip install librosa soundfile
```

## Testing Your Installation

Let's verify everything works correctly by running a simple test:

```python
# Create a file named test_audiocraft.py with the following content

import torch
import torchaudio
from audiocraft.models import MusicGen
import time

def test_installation():
    # Check PyTorch installation
    print(f"PyTorch version: {torch.__version__}")
    print(f"Torchaudio version: {torchaudio.__version__}")
    
    # Check device availability
    if torch.cuda.is_available():
        device = "cuda"
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("MPS (Metal Performance Shaders) available")
    else:
        device = "cpu"
        print("Using CPU (no GPU acceleration available)")
    
    # Load a small model to test
    print("Loading MusicGen 'small' model (this might take a moment)...")
    start_time = time.time()
    model = MusicGen.get_pretrained('small')
    model.to(device)
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds")
    
    # Generate a tiny sample
    print("Generating a short test sample...")
    start_time = time.time()
    model.set_generation_params(duration=1.0)  # Just 1 second for testing
    wav = model.generate(["A short test melody"])
    generation_time = time.time() - start_time
    print(f"Generation completed in {generation_time:.2f} seconds")
    print("Installation test successful!")
    
    return {
        "device": device,
        "load_time": load_time,
        "generation_time": generation_time
    }

if __name__ == "__main__":
    results = test_installation()
    print("\nPerformance Summary:")
    print(f"Device: {results['device']}")
    print(f"Model load time: {results['load_time']:.2f} seconds")
    print(f"Generation time: {results['generation_time']:.2f} seconds")
```

Run this script to test your installation:

```bash
python test_audiocraft.py
```

You should see output showing your PyTorch version, available devices, and timing information for loading the model and generating a sample.

### Expected Output

The output will vary depending on your hardware, but here's what you might expect:

#### On an NVIDIA GPU system:
```
PyTorch version: 2.1.0
Torchaudio version: 2.1.0
CUDA available: NVIDIA GeForce RTX 3080
CUDA version: 11.8
Loading MusicGen 'small' model (this might take a moment)...
Model loaded in 5.43 seconds
Generating a short test sample...
Generation completed in 3.21 seconds
Installation test successful!

Performance Summary:
Device: cuda
Model load time: 5.43 seconds
Generation time: 3.21 seconds
```

#### On an Apple Silicon Mac:
```
PyTorch version: 2.1.0
Torchaudio version: 2.1.0
MPS (Metal Performance Shaders) available
Loading MusicGen 'small' model (this might take a moment)...
Model loaded in 7.82 seconds
Generating a short test sample...
Generation completed in 6.35 seconds
Installation test successful!

Performance Summary:
Device: mps
Model load time: 7.82 seconds
Generation time: 6.35 seconds
```

#### On a CPU-only system:
```
PyTorch version: 2.1.0
Torchaudio version: 2.1.0
Using CPU (no GPU acceleration available)
Loading MusicGen 'small' model (this might take a moment)...
Model loaded in 12.47 seconds
Generating a short test sample...
Generation completed in 26.18 seconds
Installation test successful!

Performance Summary:
Device: cpu
Model load time: 12.47 seconds
Generation time: 26.18 seconds
```

## Common Installation Issues and Solutions

Let's troubleshoot some common issues you might encounter during installation:

### CUDA Not Found

**Problem**: You have an NVIDIA GPU, but PyTorch doesn't recognize it.

**Solutions**:
1. Check that your NVIDIA driver is up-to-date:
   ```bash
   nvidia-smi
   ```

2. Ensure you installed the correct CUDA version:
   ```bash
   python -c "import torch; print(torch.version.cuda)"
   ```

3. Try reinstalling PyTorch with explicit CUDA version:
   ```bash
   pip uninstall -y torch torchaudio
   conda install pytorch==2.1.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
   ```

### MPS Device Not Available on Apple Silicon

**Problem**: You have an M-series Mac, but PyTorch doesn't recognize the MPS device.

**Solutions**:
1. Check that you're using Python 3.9 or newer:
   ```bash
   python --version
   ```

2. Ensure you have PyTorch 2.0.0 or newer:
   ```bash
   python -c "import torch; print(torch.__version__)"
   ```

3. Reinstall PyTorch:
   ```bash
   pip uninstall -y torch torchaudio
   pip install torch==2.1.0 torchaudio==2.1.0
   ```

### Out of Memory Errors

**Problem**: You get CUDA out of memory errors or system crashes.

**Solutions**:
1. Use a smaller model:
   ```python
   model = MusicGen.get_pretrained('small')  # Instead of 'medium' or 'large'
   ```

2. Generate shorter audio segments:
   ```python
   model.set_generation_params(duration=5.0)  # Instead of 10.0 or higher
   ```

3. Close other GPU-intensive applications

4. Clear cache between generations:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

### Dependencies Conflict

**Problem**: You get errors about conflicting dependencies.

**Solutions**:
1. Start with a fresh environment:
   ```bash
   conda create -n audiocraft_new python=3.9
   conda activate audiocraft_new
   # Reinstall following the steps above
   ```

2. Install dependencies in the correct order:
   ```bash
   # Install PyTorch first
   conda install pytorch==2.1.0 torchaudio==2.1.0 -c pytorch
   
   # Then AudioCraft
   pip install audiocraft
   ```

## Environment Management Best Practices

To keep your AudioCraft environment healthy, follow these best practices:

### Activating and Deactivating

Always activate your environment before working with AudioCraft:

```bash
# Activate
conda activate audiocraft

# Deactivate when done
conda deactivate
```

### Keeping Track of Dependencies

Save your environment configuration to recreate it later:

```bash
# Export environment
conda env export > audiocraft_environment.yml

# Recreate environment from file
conda env create -f audiocraft_environment.yml
```

### Updating Safely

Update packages cautiously to avoid breaking changes:

```bash
# Update AudioCraft only
pip install --upgrade audiocraft

# Update PyTorch safely
pip install --upgrade torch==2.1.0 torchaudio==2.1.0
```

## Hardware-Specific Optimizations

To get the best performance from your specific hardware:

### NVIDIA GPU Optimizations

1. **Precision Settings**: Lower precision can be faster:
   ```python
   # Use mixed precision (faster with minimal quality loss)
   model = model.to(torch.float16)
   ```

2. **Memory Management**: Monitor GPU memory:
   ```python
   # Check available memory
   free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
   print(f"Free CUDA memory: {free_memory / 1024**3:.2f} GB")
   ```

### Apple Silicon Optimizations

1. **Memory Sharing**: Be mindful of system memory usage as MPS shares RAM:
   ```python
   # Free up memory between generations
   import gc
   torch.mps.empty_cache()
   gc.collect()
   ```

2. **Battery Considerations**: GPU acceleration uses more power:
   ```python
   # For battery-sensitive scenarios, consider CPU for small models
   if battery_saving_mode:
       device = "cpu"
   else:
       device = "mps" if torch.backends.mps.is_available() else "cpu"
   ```

### CPU Optimizations

1. **Thread Control**: Manage CPU thread usage:
   ```python
   # Limit threads for better system responsiveness
   import torch
   torch.set_num_threads(4)  # Adjust based on your CPU
   ```

2. **Memory Optimization**: Pre-load and reuse models:
   ```python
   # Load once, use multiple times
   model = MusicGen.get_pretrained('small')
   
   # Generate multiple outputs with the same model
   for prompt in prompts:
       wav = model.generate([prompt])
       # Save or process each result
   ```

## Complete Installation Script

Here's a comprehensive script that automates the installation process for any platform:

```python
#!/usr/bin/env python3
# audiocraft_setup.py - Complete AudioCraft environment setup script

import os
import sys
import platform
import subprocess
import shutil

def run_command(command):
    """Run a shell command and print output."""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, check=False)
    return result.returncode == 0

def detect_system():
    """Detect operating system and hardware."""
    system = platform.system()
    machine = platform.machine()
    
    is_windows = system == "Windows"
    is_macos = system == "Darwin"
    is_linux = system == "Linux"
    
    is_apple_silicon = is_macos and machine == "arm64"
    
    # Check for NVIDIA GPU
    has_nvidia = False
    if is_windows or is_linux:
        nvidia_smi = shutil.which("nvidia-smi")
        has_nvidia = nvidia_smi is not None and run_command("nvidia-smi")
    
    return {
        "is_windows": is_windows,
        "is_macos": is_macos,
        "is_linux": is_linux,
        "is_apple_silicon": is_apple_silicon,
        "has_nvidia": has_nvidia
    }

def setup_environment():
    """Set up the AudioCraft environment based on detected system."""
    sys_info = detect_system()
    
    print("\n=== System Detection ===")
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"Machine Architecture: {platform.machine()}")
    print(f"NVIDIA GPU Available: {'Yes' if sys_info['has_nvidia'] else 'No'}")
    if sys_info['is_apple_silicon']:
        print("Apple Silicon Mac detected")
    
    # Check if conda is available
    conda_available = shutil.which("conda") is not None
    if not conda_available:
        print("\nConda not found. Please install Miniconda from:")
        print("https://docs.conda.io/en/latest/miniconda.html")
        return False
    
    # Create conda environment
    env_name = input("\nEnter environment name (default: audiocraft): ").strip() or "audiocraft"
    
    if run_command(f"conda env list | grep {env_name}"):
        action = input(f"Environment '{env_name}' already exists. [r]ecreate, [u]se existing, or [c]ancel? ").lower()
        if action == 'c':
            return False
        elif action == 'r':
            run_command(f"conda env remove -n {env_name} -y")
        # else use existing
    
    if not run_command(f"conda info --envs | grep {env_name}"):
        print(f"\nCreating new environment: {env_name}")
        run_command(f"conda create -n {env_name} python=3.9 -y")
    
    # Get activation command
    if sys_info['is_windows']:
        activate_cmd = f"conda activate {env_name}"
    else:
        activate_cmd = f"source activate {env_name}"
    
    # Install PyTorch based on hardware
    print("\n=== Installing PyTorch ===")
    
    if sys_info['has_nvidia']:
        print("Installing PyTorch with CUDA support")
        torch_install = f"{activate_cmd} && conda install pytorch==2.1.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y"
    elif sys_info['is_apple_silicon']:
        print("Installing PyTorch with Metal support for Apple Silicon")
        torch_install = f"{activate_cmd} && pip install torch==2.1.0 torchaudio==2.1.0"
    else:
        print("Installing PyTorch for CPU")
        torch_install = f"{activate_cmd} && pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu"
    
    run_command(torch_install)
    
    # Install AudioCraft
    print("\n=== Installing AudioCraft ===")
    run_command(f"{activate_cmd} && pip install audiocraft")
    
    # Install additional packages
    print("\n=== Installing Additional Packages ===")
    run_command(f"{activate_cmd} && pip install matplotlib jupyterlab ipywidgets gradio librosa soundfile")
    
    # Create test script
    print("\n=== Creating Test Script ===")
    with open("test_audiocraft.py", "w") as f:
        f.write("""
import torch
import torchaudio
from audiocraft.models import MusicGen
import time

def test_installation():
    # Check PyTorch installation
    print(f"PyTorch version: {torch.__version__}")
    print(f"Torchaudio version: {torchaudio.__version__}")
    
    # Check device availability
    if torch.cuda.is_available():
        device = "cuda"
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("MPS (Metal Performance Shaders) available")
    else:
        device = "cpu"
        print("Using CPU (no GPU acceleration available)")
    
    # Load a small model to test
    print("Loading MusicGen 'small' model (this might take a moment)...")
    start_time = time.time()
    model = MusicGen.get_pretrained('small')
    model.to(device)
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds")
    
    # Generate a tiny sample
    print("Generating a short test sample...")
    start_time = time.time()
    model.set_generation_params(duration=1.0)  # Just 1 second for testing
    wav = model.generate(["A short test melody"])
    generation_time = time.time() - start_time
    print(f"Generation completed in {generation_time:.2f} seconds")
    print("Installation test successful!")
    
    return {
        "device": device,
        "load_time": load_time,
        "generation_time": generation_time
    }

if __name__ == "__main__":
    results = test_installation()
    print("\\nPerformance Summary:")
    print(f"Device: {results['device']}")
    print(f"Model load time: {results['load_time']:.2f} seconds")
    print(f"Generation time: {results['generation_time']:.2f} seconds")
""")
    
    # Create activation script
    if sys_info['is_windows']:
        with open("activate_audiocraft.bat", "w") as f:
            f.write(f"@echo off\ncall conda activate {env_name}\necho AudioCraft environment activated. Run 'python test_audiocraft.py' to test.")
        print("\nCreated activation script: activate_audiocraft.bat")
    else:
        with open("activate_audiocraft.sh", "w") as f:
            f.write(f"#!/bin/bash\nsource activate {env_name}\necho AudioCraft environment activated. Run 'python test_audiocraft.py' to test.")
        run_command("chmod +x activate_audiocraft.sh")
        print("\nCreated activation script: activate_audiocraft.sh")
    
    print("\n=== Setup Complete ===")
    print(f"To activate the environment:  {activate_cmd}")
    print("To test the installation:     python test_audiocraft.py")
    
    return True

if __name__ == "__main__":
    print("=== AudioCraft Environment Setup ===")
    setup_environment()
```

You can download this script, save it as `audiocraft_setup.py`, and run it with:

```bash
python audiocraft_setup.py
```

It will detect your system configuration, create an appropriate environment, and set up AudioCraft with the correct dependencies for your hardware.

## Key Takeaways

- The right environment setup is crucial for efficient AudioCraft usage
- Conda provides the most reliable isolation for AudioCraft dependencies
- Hardware-specific configurations can dramatically impact performance
- Testing your installation verifies everything works correctly
- Common installation issues have straightforward solutions
- Different hardware configurations require different optimization strategies

## Next Steps

Now that you have a properly configured environment for AudioCraft, you're ready to explore:

- [Understanding AudioCraft Architecture](/chapters/part1/architecture/): Learn how AudioCraft's components work together
- [Your First Audio Generation](/chapters/part1/first-generation/): Generate your first AI audio with AudioCraft

## Further Reading

- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [Conda Environment Management](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [Metal Performance Shaders Documentation](https://developer.apple.com/documentation/metalperformanceshaders)
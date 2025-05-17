# Setting Up AudioCraft on Windows and Linux with NVIDIA GPUs

This guide will help you set up Meta's AudioCraft framework on Windows and Linux systems with NVIDIA GPUs using CUDA for hardware acceleration.

## System Requirements

- NVIDIA GPU with CUDA support (GTX 1060 6GB or better recommended)
- CUDA Toolkit 11.8 or newer
- Python 3.9 (recommended for AudioCraft)
- At least 8GB system RAM (16GB+ recommended)
- 5GB+ free disk space for models and environment

## Installation Steps

### 1. Install CUDA Toolkit

First, ensure you have the NVIDIA CUDA Toolkit installed.

#### Windows

1. Download the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) from NVIDIA's website
2. Run the installer and follow the prompts
3. Verify installation by running `nvcc --version` in Command Prompt

#### Linux (Ubuntu/Debian)

```bash
# Install CUDA toolkit
sudo apt update
sudo apt install -y nvidia-cuda-toolkit

# Verify installation
nvcc --version
```

### 2. Set Up a Conda Environment

We recommend using Conda to manage your Python environment.

#### Windows

1. Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. Open the Anaconda Prompt from the Start menu
3. Create and activate a new environment:

```bash
conda create -n audiocraft python=3.9
conda activate audiocraft
```

#### Linux

```bash
# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Follow the prompts to complete installation

# Create and activate environment
conda create -n audiocraft python=3.9
conda activate audiocraft
```

### 3. Install PyTorch with CUDA Support

```bash
# For CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Or using pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4. Install AudioCraft

```bash
# Install AudioCraft from PyPI
pip install audiocraft

# Or install directly from GitHub for the latest version
# pip install git+https://github.com/facebookresearch/audiocraft.git
```

### 5. Install Additional Dependencies

```bash
# Install FFmpeg
# Windows (using conda)
conda install ffmpeg -c conda-forge

# Linux
sudo apt update
sudo apt install -y ffmpeg
```

## Verifying CUDA Support

Create a simple Python script to verify that CUDA is working:

```python
# test_cuda.py
import torch

# Check if CUDA is available
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    # Get device count
    print(f"Device count: {torch.cuda.device_count()}")
    
    # Get device name
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    
    # Create a tensor on GPU
    cuda_device = torch.device("cuda")
    x = torch.ones(1, device=cuda_device)
    print(f"Tensor on CUDA device: {x}")
else:
    print("CUDA not available. Check your PyTorch installation and GPU drivers.")
```

Save this as `test_cuda.py` and run it with `python test_cuda.py`. If everything is set up correctly, you should see that CUDA is available and information about your GPU.

## Performance Optimization for NVIDIA GPUs

### Memory Management

NVIDIA GPUs have dedicated VRAM, which is used for processing models. To optimize performance:

1. **Monitor GPU Memory**: Use `nvidia-smi` to monitor VRAM usage
2. **Batch Size Adjustment**: Start with smaller batch sizes and gradually increase
3. **Model Size Selection**: Choose appropriate model sizes based on your GPU's VRAM

### Recommended Settings for Different GPUs

| GPU Model | VRAM | MusicGen Model Size | AudioGen Model Size | Max Batch Size | Notes |
|-----------|------|---------------------|---------------------|----------------|-------|
| GTX 1060/1660 | 6GB | small | small | 1-2 | Limit to 10-15 second generations |
| RTX 2060/3060 | 8GB | small/medium | small/medium | 2-4 | Good for standard usage |
| RTX 3070/3080 | 10GB+ | medium/large | medium/large | 4-8 | Can handle longer generations |
| RTX 3090/4090 | 24GB | large/xlarge | large | 8-16 | Can handle complex generations |

### CUDA Configuration Tips

1. **Mixed Precision**: Use mixed precision to reduce memory usage and increase speed:

```python
# Enable mixed precision
from torch.cuda.amp import autocast

with autocast():
    # Your generation code here
    wav = model.generate(prompts)
```

2. **Memory Cleanup**: Clear CUDA cache between generations to avoid memory fragmentation:

```python
import torch
import gc

# After generation
torch.cuda.empty_cache()
gc.collect()
```

## Windows-Specific Notes

### Setting Up the Environment

1. **Path Length Limitation**: Windows has a path length limit that can sometimes cause issues with deep directory structures. Consider installing in a short path like `C:\Conda`.

2. **Antivirus Interference**: Some antivirus software may slow down Python or block certain operations. Consider adding exceptions for your Python environment.

### Audio Playback

On Windows, you can use the `sounddevice` library to play generated audio directly:

```python
pip install sounddevice

# Then in your code
import sounddevice as sd

# Play audio (assuming 'wav' is your generated audio and 'sr' is the sample rate)
sd.play(wav.cpu().numpy(), sr)
sd.wait()  # Wait for audio to finish playing
```

## Linux-Specific Notes

### GPU Driver Installation

Make sure you have the latest NVIDIA drivers installed:

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y nvidia-driver-535  # Replace with latest version

# Verify installation
nvidia-smi
```

### Audio Setup

Ensure you have the necessary audio libraries for playback:

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y libasound2-dev portaudio19-dev libportaudio2 libportaudiocpp0

# Install Python audio libraries
pip install sounddevice
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory Errors**
   - Reduce batch size or model size
   - Try shorter generation lengths
   - Close other GPU-intensive applications
   - Use mixed precision training

2. **CUDA Not Found**
   - Verify NVIDIA drivers are installed: `nvidia-smi`
   - Check PyTorch CUDA is working: `python -c "import torch; print(torch.cuda.is_available())"`
   - Reinstall PyTorch with the correct CUDA version

3. **Slow Generation Times**
   - First generation is always slower due to model loading
   - Try running in batches instead of one-by-one
   - Use a smaller model size for faster results

### Platform-Specific Issues

#### Windows

- **DLL Not Found Errors**: Make sure your PATH includes the CUDA bin directory
- **VSCode issues**: If using VSCode, make sure to select the correct Python interpreter

#### Linux

- **Permission Issues**: Ensure you have proper permissions for NVIDIA devices: `ls -l /dev/nvidia*`
- **Missing Libraries**: Install necessary libs with `apt install nvidia-cuda-toolkit`

## Getting Help

If you encounter issues, check:
- [AudioCraft GitHub Issues](https://github.com/facebookresearch/audiocraft/issues)
- [PyTorch Forums](https://discuss.pytorch.org/)
- [NVIDIA Developer Forums](https://forums.developer.nvidia.com/)

## Next Steps

Now that you have AudioCraft running on your Windows or Linux system with CUDA acceleration, proceed to our [MusicGen tutorial](../musicgen/README.md) or [AudioGen tutorial](../audiogen/README.md) to start generating audio!
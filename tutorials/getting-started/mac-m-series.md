# Setting Up AudioCraft on Apple Silicon Macs (M-Series)

This guide will help you set up Meta's AudioCraft framework on Apple Silicon Macs, including M1, M2, M3, and M4 chips. We'll leverage the Metal Performance Shaders (MPS) backend for GPU acceleration.

## System Requirements

- macOS 12.3 or newer
- Apple Silicon Mac (M1, M2, M3, or M4 series)
- Python 3.9 (recommended for AudioCraft)
- Miniforge or Miniconda (recommended for Apple Silicon)

## Installation Steps

### 1. Set Up Conda for Apple Silicon

First, download and install Miniforge, which provides Conda optimized for Apple Silicon:

```bash
# Download Miniforge
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh"
bash Miniforge3-MacOSX-arm64.sh
```

### 2. Create and Activate a Conda Environment

```bash
# Create a new environment with Python 3.9
conda create -n audiocraft python=3.9
conda activate audiocraft
```

### 3. Install PyTorch with Metal Support

```bash
# Install PyTorch 2.1.0 with MPS support
pip install torch==2.1.0 torchaudio==2.1.0
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
# Install FFmpeg (for audio processing)
conda install ffmpeg -c conda-forge
```

## Verifying Metal GPU Support

Create a simple Python script to verify that MPS (Metal Performance Shaders) is working:

```python
import torch

# Check if MPS is available
print(f"MPS available: {torch.backends.mps.is_available()}")
if torch.backends.mps.is_available():
    # Create an MPS device
    mps_device = torch.device("mps")
    
    # Create a tensor on MPS device
    x = torch.ones(1, device=mps_device)
    print(f"Tensor on MPS device: {x}")
    
    # You'll see output at this point if MPS is working correctly
else:
    print("MPS device not found. Check your PyTorch installation.")
```

Save this as `test_mps.py` and run it with `python test_mps.py`. If everything is set up correctly, you should see that MPS is available and a tensor created on the MPS device.

## Performance Optimization for M-Series Macs

### Memory Management

Apple Silicon Macs have unified memory architecture, which means CPU and GPU share the same memory. To optimize performance:

1. **Monitor Memory Usage**: Use Activity Monitor to keep an eye on memory pressure.
2. **Batch Size Adjustment**: Start with smaller batch sizes and gradually increase until you find the optimal balance.
3. **Model Size Selection**: Choose appropriate model sizes (small, medium, large) based on your Mac's capabilities.

### Recommended Settings for Different Mac Models

| Mac Model | MusicGen Model Size | AudioGen Model Size | Batch Size | Notes |
|-----------|---------------------|---------------------|------------|-------|
| M1/M2     | small/medium        | small/medium        | 1-4        | Limit generation duration to 30 seconds |
| M3        | medium/large        | medium/large        | 2-8        | Can handle longer generations |
| M4 Max    | large/xlarge        | large               | 4-16       | Can handle complex generations |

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Solution: Reduce batch size or model size
   - Try closing other applications to free up memory

2. **Slow Generation Times**
   - Metal initialization has some overhead
   - First generation is always slower; subsequent generations are faster
   - Try running in batches instead of one-by-one

3. **PyTorch Version Issues**
   - Make sure you're using PyTorch 2.1.0 or newer
   - MPS requires macOS 12.3+

### Getting Help

If you encounter issues, check:
- [AudioCraft GitHub Issues](https://github.com/facebookresearch/audiocraft/issues)
- [PyTorch Forums](https://discuss.pytorch.org/)

## Next Steps

Now that you have AudioCraft running on your Apple Silicon Mac, proceed to our [MusicGen tutorial](../musicgen/README.md) or [AudioGen tutorial](../audiogen/README.md) to start generating audio!
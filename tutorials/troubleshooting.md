# AudioCraft Troubleshooting & FAQs

This guide addresses common issues, questions, and solutions when working with Meta's AudioCraft framework. Whether you're encountering installation problems, generation errors, or looking for ways to optimize performance, you'll find helpful information here.

## Table of Contents

- [Installation Issues](#installation-issues)
- [AudioCraft Model Loading Issues](#audiocraft-model-loading-issues)
- [Generation Problems](#generation-problems)
- [Memory and Performance](#memory-and-performance)
- [Platform-Specific Issues](#platform-specific-issues)
- [Audio Quality Issues](#audio-quality-issues)
- [Integration Challenges](#integration-challenges)
- [Common Error Messages](#common-error-messages)

## Installation Issues

### PyTorch Installation Errors

**Problem**: Errors when installing PyTorch with CUDA support.

**Solution**: 
1. Visit [PyTorch's official installation page](https://pytorch.org/get-started/locally/) for the correct command for your system.
2. For Mac users with Apple Silicon, use:
   ```bash
   pip install torch==2.1.0 torchaudio==2.1.0
   ```
3. For CUDA users, specify the CUDA version that matches your drivers:
   ```bash
   # Example for CUDA 11.8
   pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
   ```

### AudioCraft Installation Failures

**Problem**: `pip install audiocraft` fails with dependency errors.

**Solution**:
1. Ensure you have the latest pip: `pip install --upgrade pip`
2. Install in a clean virtual environment:
   ```bash
   python -m venv audiocraft_env
   source audiocraft_env/bin/activate  # On Windows: audiocraft_env\Scripts\activate
   pip install audiocraft
   ```
3. If still failing, install from source:
   ```bash
   git clone https://github.com/facebookresearch/audiocraft.git
   cd audiocraft
   pip install -e .
   ```

### FFmpeg Missing

**Problem**: Errors about missing FFmpeg during audio processing.

**Solution**:
- **On macOS**: `brew install ffmpeg`
- **On Ubuntu/Debian**: `sudo apt-get install ffmpeg`
- **On Windows**: Download from [FFmpeg website](https://ffmpeg.org/download.html) or use `conda install -c conda-forge ffmpeg`

## AudioCraft Model Loading Issues

### Models Fail to Download

**Problem**: Error when trying to load pretrained models.

**Solutions**:
1. **Check internet connection** and firewall settings.
2. **Manually download models** from Hugging Face:
   - Visit [facebook/MusicGen](https://huggingface.co/facebook/MusicGen) or [facebook/AudioGen](https://huggingface.co/facebook/AudioGen)
   - Download the model files to the default cache location:
     - Linux/Mac: `~/.cache/huggingface/hub`
     - Windows: `C:\Users\<username>\.cache\huggingface\hub`
3. **Set custom cache directory**:
   ```python
   import os
   os.environ["HF_HOME"] = "/path/to/custom/directory"
   ```

### Out of Memory When Loading Models

**Problem**: OOM errors when loading larger models.

**Solutions**:
1. Use a smaller model size (`small` instead of `medium` or `large`).
2. Load the model in CPU mode first, then move specific components to GPU:
   ```python
   import torch
   from audiocraft.models import MusicGen
   
   # Load on CPU first
   model = MusicGen.get_pretrained('medium', device="cpu")
   
   # Move only necessary components to GPU
   model.lm = model.lm.to("cuda")  # Move only the language model to GPU
   ```
3. Reduce batch size and ensure no other large models are loaded.

## Generation Problems

### Generation Hangs or is Very Slow

**Problem**: Audio generation takes extremely long or seems to hang.

**Solutions**:
1. The **first generation is always slowest** due to model loading and compilation.
2. **Reduce duration** parameter for faster generation.
3. **Use a smaller model size** (e.g., `small` instead of `medium` or `large`).
4. **Check your hardware**: CPU generation can be 10-20x slower than GPU.
5. If using Apple Silicon, **ensure MPS is being used**:
   ```python
   import torch
   print(f"MPS available: {torch.backends.mps.is_available()}")
   ```

### Unexpected Generation Results

**Problem**: Audio output doesn't match the prompt or sounds wrong.

**Solutions**:
1. **Improve your prompt**: Be more specific about instruments, style, tempo, mood.
2. **Adjust temperature**: Lower (0.3-0.7) for more predictable results, higher for more creativity.
3. **Try different model sizes**: Larger models generally produce better quality results.
4. **Check your audio output settings**: Sample rate mismatches can cause distortion.

## Memory and Performance

### Out of Memory During Generation

**Problem**: CUDA out of memory or similar errors during generation.

**Solutions**:
1. **Reduce batch size** (generate one at a time).
2. **Use a smaller model size** (switch from `large` to `medium` or `small`).
3. **Decrease the generation duration**.
4. **Free unused memory** before generation:
   ```python
   import torch
   torch.cuda.empty_cache()  # For CUDA
   ```
5. **Monitor GPU memory** with `nvidia-smi` (CUDA) or Activity Monitor (Mac).

### Slow Generation on Good Hardware

**Problem**: Generation is slower than expected despite good GPU.

**Solutions**:
1. **Check if GPU is actually being used**:
   ```python
   print(f"Device being used: {next(model.parameters()).device}")
   ```
2. **Ensure PyTorch was installed with GPU support**:
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"MPS available: {torch.backends.mps.is_available()}")
   ```
3. **Update GPU drivers** to the latest version.
4. **Optimize batch processing**: Generate multiple outputs in one batch rather than one at a time.

## Platform-Specific Issues

### Mac (Apple Silicon) Issues

**Problem**: Not utilizing GPU acceleration on M1/M2/M3/M4 Mac.

**Solutions**:
1. **Install PyTorch with MPS support**:
   ```bash
   pip install torch==2.1.0 torchaudio==2.1.0
   ```
2. **Explicitly use MPS device**:
   ```python
   device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
   model.to(device)
   ```
3. **Ensure macOS version is 12.3+** (required for MPS support).
4. **Be aware of MPS limitations**: Some operations might fall back to CPU.

### Windows-Specific Issues

**Problem**: Various installation or runtime errors on Windows.

**Solutions**:
1. **Use Anaconda/Miniconda environment** instead of plain pip.
2. **Install Visual C++ Redistributable** if getting DLL errors.
3. **Use Python 3.9 or 3.10** for best compatibility.
4. **Path length issues**: Install in a directory with a shorter path.

### Linux GPU Acceleration Problems

**Problem**: CUDA not working properly on Linux.

**Solutions**:
1. **Verify CUDA installation**: `nvcc --version`
2. **Ensure PyTorch CUDA version matches system CUDA**:
   ```python
   import torch
   print(torch.version.cuda)
   ```
3. **Check NVIDIA drivers**: `nvidia-smi`
4. **Set environment variables** if needed:
   ```bash
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
   ```

## Audio Quality Issues

### Poor Audio Quality

**Problem**: Generated audio has low quality, artifacts, or distortion.

**Solutions**:
1. **Use a larger model size** (`medium` or `large`).
2. **Adjust generation parameters**:
   ```python
   model.set_generation_params(
       duration=10.0,
       temperature=0.8,  # Try between 0.5-1.0
       top_k=250,
       top_p=0.0,
   )
   ```
3. **Use specific audio formats** in prompts (e.g., "high-quality stereo recording").
4. **Apply post-processing**: Consider using audio enhancement tools or filters.

### Abrupt Audio Endings

**Problem**: Generated audio cuts off abruptly.

**Solutions**:
1. **Increase duration parameter**.
2. **Apply fadeout effect**:
   ```python
   # Apply a simple fadeout to the end of the audio
   def apply_fadeout(audio, fade_samples=3000):
       fade = torch.linspace(1.0, 0.0, fade_samples)
       audio[-fade_samples:] *= fade
       return audio
   
   # Apply fadeout before saving
   generated_audio = apply_fadeout(generated_audio)
   ```
3. **Try prompt engineering**: Add terms like "with proper ending" or "with fade out ending".

## Integration Challenges

### Combining with Other Libraries

**Problem**: Errors when integrating AudioCraft with other audio or ML libraries.

**Solutions**:
1. **Check for device mismatches**:
   ```python
   # Ensure all tensors are on the same device
   tensor1 = tensor1.to(device)
   tensor2 = tensor2.to(device)
   ```
2. **Handle different audio formats**: Convert sample rates and channels as needed:
   ```python
   import torchaudio.functional as F
   
   # Resample audio if necessary
   if source_sr != target_sr:
       audio = F.resample(audio, source_sr, target_sr)
   ```
3. **Convert between libraries**: Use NumPy as an intermediate format:
   ```python
   # Convert from AudioCraft output to librosa format
   audio_np = audio_tensor.cpu().numpy()
   ```

### Web Integration Issues

**Problem**: Difficulties embedding AudioCraft in web applications.

**Solutions**:
1. **Use Gradio for quick web interfaces**:
   ```python
   import gradio as gr
   
   def generate(prompt):
       # AudioCraft generation code here
       return output_path
   
   interface = gr.Interface(
       fn=generate,
       inputs=gr.Textbox(label="Prompt"),
       outputs=gr.Audio(label="Generated Audio"),
   )
   interface.launch()
   ```
2. **For production**: Create a REST API with Flask or FastAPI.
3. **Consider compute requirements**: Web servers might need GPU access.

## Common Error Messages

### "RuntimeError: CUDA out of memory"

**Solutions**:
1. Use a smaller model.
2. Reduce the duration of generated audio.
3. Generate one sample at a time instead of batches.
4. Close other GPU-intensive applications.

### "RuntimeError: MPS backend out of memory"

**Solutions**:
1. Same as CUDA out of memory, plus:
2. Restart your Python session (MPS memory management is less robust).
3. Consider using `device="cpu"` if the model is too large for your Mac's memory.

### "ModuleNotFoundError: No module named 'audiocraft'"

**Solutions**:
1. Check installation: `pip list | grep audiocraft`
2. Reinstall: `pip install -U audiocraft`
3. Make sure you're in the correct virtual environment.

### "AssertionError: generation with duration > X not supported by model"

**Solutions**:
1. Reduce the requested duration to be within model limits.
2. For longer generations, look at the advanced tutorials on combining multiple generations.

### "RuntimeError: Expected all tensors to be on the same device"

**Solutions**:
1. Explicitly move all tensors to the same device:
   ```python
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   tensor1 = tensor1.to(device)
   tensor2 = tensor2.to(device)
   ```
2. Check your code for device inconsistencies.

## Additional Resources

- [AudioCraft GitHub Repository](https://github.com/facebookresearch/audiocraft)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Hugging Face Audio Models](https://huggingface.co/models?pipeline_tag=audio)
- [Mac Metal Developer Guide](https://developer.apple.com/metal/pytorch/)

## Reporting New Issues

If you encounter issues not covered in this guide, please:

1. Check the [GitHub Issues](https://github.com/facebookresearch/audiocraft/issues) to see if it's a known problem.
2. Provide detailed information when reporting new issues:
   - OS and hardware details
   - AudioCraft and PyTorch versions
   - Complete error messages
   - Minimal code example to reproduce the problem
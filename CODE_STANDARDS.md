# Code Standards for Practical Meta AudioCraft

Copyright © 2025 Scott Friedman.  
Licensed under Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.

This document defines the coding standards, documentation guidelines, and best practices for all code examples in the "Practical Meta AudioCraft" book. Consistent code style helps readers focus on concepts rather than being distracted by stylistic differences between chapters.

## File Structure Standards

### 1. File Header

All Python files should begin with a standardized header:

```python
#!/usr/bin/env python3
# filename.py - Brief description of what this file does
# Part of "Practical Meta AudioCraft" - Chapter X: Chapter Title
```

### 2. Import Organization

Imports should be grouped and ordered as follows:

```python
# Standard library imports
import os
import time
import argparse
from pathlib import Path

# Third-party imports
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt

# AudioCraft imports
from audiocraft.models import MusicGen, AudioGen
from audiocraft.data.audio import audio_write
from audiocraft.utils.notebook import display_audio
```

### 3. Main Function Structure

Files that can be run as scripts should use the standard `if __name__ == "__main__":` pattern:

```python
def main():
    """Main function that runs when the script is executed directly."""
    # Main functionality here
    pass

if __name__ == "__main__":
    main()
```

## Code Commenting Standards

### 1. Function Docstrings

All functions should have detailed docstrings following this format:

```python
def function_name(param1, param2, optional_param=default_value):
    """
    Brief description of what the function does.
    
    Args:
        param1 (type): Description of param1
        param2 (type): Description of param2
        optional_param (type, optional): Description of optional_param. Defaults to default_value.
    
    Returns:
        return_type: Description of return value
        
    Raises:
        ExceptionType: When/why this exception might be raised
        
    Notes:
        Additional information, caveats, or implementation details
    
    Example:
        >>> result = function_name("example", 42)
        >>> print(result)
        Expected output
    """
```

### 2. Class Docstrings

Classes should have detailed docstrings following this format:

```python
class ClassName:
    """
    Brief description of the class purpose.
    
    Detailed description of the class, its behavior, and usage.
    
    Attributes:
        attr1 (type): Description of attr1
        attr2 (type): Description of attr2
    
    Notes:
        Any additional information or caveats
    
    Example:
        >>> obj = ClassName(param)
        >>> obj.method()
        Expected output
    """
```

### 3. Section Comments

Use section comments to divide logical parts of the code:

```python
# ----- Device Configuration -----
# Determine the best available device for computation
if torch.backends.mps.is_available():
    device = "mps"  # Apple Silicon GPU
    print("Using MPS (Metal) for generation")
elif torch.cuda.is_available():
    device = "cuda"  # NVIDIA GPU
    print("Using CUDA for generation")
else:
    device = "cpu"  # Fallback to CPU
    print("Using CPU for generation (this will be slow)")
```

### 4. Inline Comments

Use inline comments to explain non-obvious code:

```python
# Create a filesystem-safe filename from the prompt
# This truncates long prompts and replaces special characters
safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt)[:30]
```

## Code Style Guidelines

### 1. Variable Naming

- Use descriptive variable names that clearly indicate purpose
- Follow Python conventions:
  - `snake_case` for variables and functions
  - `PascalCase` for classes
  - `UPPER_CASE` for constants

```python
# Good examples
model_size = "medium"                  # Clear variable name
sample_rate = 32000                    # Technical parameter
AUDIO_FORMATS = ["wav", "mp3", "ogg"]  # Constant
```

### 2. Error Handling

All examples should include proper error handling:

```python
try:
    # Operation that might fail
    model = MusicGen.get_pretrained(model_size)
    model.to(device)
except RuntimeError as e:
    if "CUDA out of memory" in str(e):
        print(f"ERROR: Not enough GPU memory for {model_size} model.")
        print("Try using a smaller model size ('small' instead of 'medium' or 'large')")
        print("Or reduce the duration of the generated audio")
    else:
        print(f"ERROR: Failed to load model: {str(e)}")
    # Provide fallback or exit gracefully
```

### 3. Parameter Validation

Validate input parameters at the beginning of functions:

```python
def generate_music(prompt, duration=10.0, model_size="small", temperature=1.0):
    """Generate music with parameter validation."""
    # Validate parameters
    if not prompt or not isinstance(prompt, str):
        raise ValueError("Prompt must be a non-empty string")
        
    if duration <= 0 or duration > 30:
        raise ValueError("Duration must be between 0 and 30 seconds")
    
    if model_size not in ["small", "medium", "large"]:
        raise ValueError("Model size must be 'small', 'medium', or 'large'")
    
    if temperature <= 0 or temperature > 2.0:
        raise ValueError("Temperature must be between 0 and 2.0")
    
    # Function implementation...
```

### 4. Resource Cleanup

Include proper resource cleanup in examples:

```python
def generate_with_cleanup():
    """Generate audio with proper cleanup."""
    try:
        # Generation code...
        pass
    finally:
        # Clean up resources
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Other cleanup as needed
```

## Hardware Awareness Standards

### 1. Device Detection

Always include standardized device detection:

```python
def get_device():
    """Determine the best available device for computation."""
    if torch.backends.mps.is_available():
        return "mps"  # Apple Silicon GPU
    elif torch.cuda.is_available():
        return "cuda"  # NVIDIA GPU
    else:
        return "cpu"  # Fallback to CPU
```

### 2. Performance Considerations

Include hardware-specific performance notes:

```python
def generate_with_performance_notes(model_size="small"):
    """Generate audio with performance considerations."""
    device = get_device()
    
    # Provide performance expectations
    expected_times = {
        "cuda": {"small": "5-10s", "medium": "10-20s", "large": "20-40s"},
        "mps": {"small": "10-20s", "medium": "30-60s", "large": "60-120s"},
        "cpu": {"small": "60-120s", "medium": "180-300s", "large": "300-600s"}
    }
    
    device_name = "GPU (CUDA)" if device == "cuda" else "Apple Silicon (MPS)" if device == "mps" else "CPU"
    print(f"Using {device_name} for generation")
    print(f"Expected generation time: {expected_times.get(device, expected_times['cpu']).get(model_size, 'unknown')}")
    
    # Generation code...
```

### 3. Memory Management

Include memory management guidance:

```python
def check_memory_requirements(model_size="small"):
    """Check if the system meets memory requirements."""
    memory_required = {
        "small": 2,  # GB
        "medium": 4, # GB
        "large": 8   # GB
    }
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # Convert to GB
        if gpu_memory < memory_required[model_size]:
            print(f"WARNING: Your GPU has {gpu_memory:.1f}GB memory, but {model_size} model typically requires {memory_required[model_size]}GB")
            print("Generation may fail or be very slow. Consider using a smaller model.")
    else:
        import psutil
        system_memory = psutil.virtual_memory().total / 1024**3  # Convert to GB
        if system_memory < memory_required[model_size] * 1.5:  # CPU needs more memory than GPU
            print(f"WARNING: Your system has {system_memory:.1f}GB memory, but {model_size} model on CPU typically requires {memory_required[model_size] * 1.5}GB")
            print("Generation may fail or be very slow. Consider using a smaller model.")
```

## Reusable Components

### 1. Standard Utility Functions

Include a set of standard utility functions that can be reused across examples:

```python
def save_audio(wav_tensor, filename, sample_rate, output_dir="output"):
    """
    Save audio tensor to file with proper directory creation and error handling.
    
    Args:
        wav_tensor (torch.Tensor): Audio tensor to save
        filename (str): Base filename (without extension)
        sample_rate (int): Sample rate of the audio
        output_dir (str): Directory to save the file
    
    Returns:
        str: Full path to the saved audio file
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create full output path
    output_path = os.path.join(output_dir, filename)
    
    try:
        # Save audio file
        audio_write(
            output_path,
            wav_tensor.cpu(),
            sample_rate,
            strategy="loudness"
        )
        return f"{output_path}.wav"
    except Exception as e:
        print(f"Error saving audio: {str(e)}")
        return None
```

### 2. Configuration Templates

Include standard configuration templates:

```python
def get_default_generation_params(model_type="music", duration=10.0):
    """
    Get default generation parameters for AudioCraft models.
    
    Args:
        model_type (str): Either "music" for MusicGen or "audio" for AudioGen
        duration (float): Duration in seconds
    
    Returns:
        dict: Default parameters for the specified model type
    """
    # Base parameters common to both models
    params = {
        "duration": duration,
        "temperature": 1.0,
        "top_k": 250,
        "top_p": 0.0,
    }
    
    # Model-specific adjustments
    if model_type == "audio":
        # AudioGen often works better with slightly higher temperature
        params["temperature"] = 1.1
    elif model_type == "music":
        # Add MusicGen-specific parameters
        params["cfg_coef"] = 3.0
    
    return params
```

## Documentation Patterns

### 1. Example Usage Patterns

Each code file should include example usage patterns:

```python
def example_function():
    """Example function documented with usage patterns."""
    pass

# ----- Example Usage -----
# Basic usage:
# result = example_function()
#
# Advanced usage:
# result = example_function(additional_param=value)
```

### 2. Practical Considerations Section

Include a "Practical Considerations" section in comments:

```python
# ----- Practical Considerations -----
# 1. Memory Usage: This example requires approximately 4GB of GPU memory with the 'medium' model
# 2. Generation Time: Expect 10-30 seconds on GPU, 1-3 minutes on CPU
# 3. Quality vs. Speed: 'small' model is faster but produces less detailed audio
# 4. Prompt Effectiveness: Detailed, specific prompts produce better results than vague ones
```

## Code Example Templates

### 1. Basic Generate Function Template

```python
def generate_audio(
    model_type,      # "music" or "audio"
    prompt,          # Text description
    duration=10.0,   # Duration in seconds
    model_size="small",  # Model size
    temperature=1.0, # Creativity control
    output_dir="output"  # Output directory
):
    """
    Generate audio using either MusicGen or AudioGen.
    
    Args:
        model_type (str): Either "music" or "audio"
        prompt (str): Text description of what to generate
        duration (float): Length of audio in seconds
        model_size (str): Size of model to use
        temperature (float): Controls randomness/creativity
        output_dir (str): Directory to save output
    
    Returns:
        str: Path to the generated audio file
    """
    # Validate parameters
    if model_type not in ["music", "audio"]:
        raise ValueError("model_type must be either 'music' or 'audio'")
        
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine device
    device = get_device()
    print(f"Using device: {device}")
    
    try:
        # Load appropriate model
        if model_type == "music":
            model = MusicGen.get_pretrained(model_size)
            print(f"Loaded MusicGen {model_size} model")
        else:  # model_type == "audio"
            model = AudioGen.get_pretrained(model_size if model_size != "small" else "medium")
            print(f"Loaded AudioGen {model_size} model")
            
        # Move model to device
        model.to(device)
        
        # Set generation parameters
        model.set_generation_params(
            duration=duration,
            temperature=temperature,
            top_k=250,
            top_p=0.0,
        )
        
        # Generate audio
        print(f"Generating {duration}s of audio from prompt: '{prompt}'")
        wav = model.generate([prompt])
        
        # Create filename
        safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt)[:30]
        filename = f"{model_type}_{safe_prompt}_{model_size}"
        
        # Save audio file
        output_path = os.path.join(output_dir, filename)
        audio_write(
            output_path,
            wav[0].cpu(),
            model.sample_rate,
            strategy="loudness"
        )
        
        print(f"Audio saved to {output_path}.wav")
        return f"{output_path}.wav"
        
    except Exception as e:
        print(f"Error generating audio: {str(e)}")
        return None
    finally:
        # Clean up
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
```

### 2. Interactive Example Template

```python
def interactive_generation():
    """Run an interactive audio generation session."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Interactive audio generation")
    parser.add_argument("--type", type=str, choices=["music", "audio"], default="music",
                        help="Type of audio to generate: 'music' or 'audio'")
    parser.add_argument("--model", type=str, choices=["small", "medium", "large"], default="small",
                        help="Model size to use")
    parser.add_argument("--output", type=str, default="generated_audio",
                        help="Output directory")
    args = parser.parse_args()
    
    print(f"=== Interactive {args.type.capitalize()} Generation ===")
    print(f"Using {args.model} model, saving to {args.output}/")
    print("Type 'exit' to quit")
    
    while True:
        # Get prompt
        prompt = input("\nEnter a description: ")
        if prompt.lower() == 'exit':
            break
            
        # Get duration
        try:
            duration = float(input("Enter duration in seconds (5-30): "))
            duration = max(1.0, min(30.0, duration))  # Clamp to valid range
        except ValueError:
            duration = 10.0
            print(f"Using default duration: {duration}s")
            
        # Get temperature
        try:
            temperature = float(input("Enter temperature (0.1-2.0): "))
            temperature = max(0.1, min(2.0, temperature))  # Clamp to valid range
        except ValueError:
            temperature = 1.0
            print(f"Using default temperature: {temperature}")
            
        # Generate
        generate_audio(
            model_type=args.type,
            prompt=prompt,
            duration=duration,
            model_size=args.model,
            temperature=temperature,
            output_dir=args.output
        )
```

## Example Quality Checklist

Before including any code example in the book, ensure it meets the following criteria:

1. [ ] Follows all coding standards defined in this document
2. [ ] Includes comprehensive error handling
3. [ ] Validates input parameters
4. [ ] Has complete and accurate documentation
5. [ ] Demonstrates hardware awareness
6. [ ] Includes proper resource cleanup
7. [ ] Provides practical considerations
8. [ ] Features example usage patterns
9. [ ] Has been tested on multiple hardware configurations
10. [ ] Includes realistic performance expectations

## Additional Resources

- [PEP 8 - Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [AudioCraft GitHub Repository](https://github.com/facebookresearch/audiocraft)
- [PyTorch Documentation Style](https://pytorch.org/docs/stable/)
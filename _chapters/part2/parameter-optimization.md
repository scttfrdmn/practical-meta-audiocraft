---
layout: chapter
title: "Parameter Optimization for Music Generation"
difficulty: intermediate
copyright: "Copyright © 2025 Scott Friedman. This work is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License."
estimated_time: 45
---

> *"My music generation AI system is working, but the results are way too random and weird sounding. Sometimes they're too repetitive, other times they're chaotic. There must be some way to control this better. I've read about temperature, top-k, and other parameters, but I have no idea how to set them for consistent, high-quality results."* 
> 
> — Mia Chen, music technology startup founder

# Chapter 7: Parameter Optimization for Music Generation

## The Challenge

You've mastered prompt engineering, but your music generation results still vary widely in quality. Sometimes they're uninspired and repetitive, other times they're chaotic and disorganized. The same prompt can yield dramatically different results with each generation.

In this chapter, we'll tackle the challenge of finding optimal parameter settings for MusicGen to produce consistently high-quality outputs that match your creative intent. We'll systematically explore how each generation parameter affects the output and develop strategies for finding the right balance between predictability and creativity.

## Learning Objectives

By the end of this chapter, you will be able to:

- Understand how each MusicGen generation parameter affects the output quality and characteristics
- Develop systematic testing methodologies to find optimal parameter combinations
- Create parameter presets tailored to different musical genres and applications
- Implement a parameter optimization workflow for your music generation pipeline
- Diagnose and fix common parameter-related issues in AI-generated music

## Prerequisites

- Basic understanding of MusicGen (Chapter 5)
- Experience with prompt engineering for music (Chapter 6)
- Python environment with AudioCraft installed

## Key Concepts: The Generation Parameter Space

MusicGen, like other generative models, offers several parameters that control the generation process. These parameters don't affect what content is generated (that's the prompt's job) but rather how the model navigates its probability space to create that content.

Let's first understand the core parameters available in MusicGen:

### Temperature

Temperature controls the randomness of the generation process. It directly scales the logits (unnormalized probabilities) before sampling, affecting how likely the model is to choose lower-probability options.

- **Low temperature (0.1 - 0.6)**: More deterministic, favors high-probability outputs, often more repetitive but coherent
- **Medium temperature (0.7 - 1.2)**: Balanced randomness, good mix of predictability and variation
- **High temperature (1.3 - 2.0)**: More chaotic and experimental, explores unusual combinations

Temperature is the most important parameter for controlling generation diversity, acting as a "creativity knob" that balances between safe, predictable outputs and wild, experimental ones.

### Top-k Sampling

Top-k sampling restricts the model to only consider the k most likely tokens at each generation step. This parameter sets a hard limit on the diversity of choices:

- **Low k (50 - 100)**: More coherent but limited in variety
- **Medium k (200 - 300)**: Good balance for most applications
- **High k (500+)**: More diverse outputs, potentially less coherent

### Top-p (Nucleus) Sampling

Top-p sampling, also called nucleus sampling, dynamically selects the smallest set of tokens whose cumulative probability exceeds the threshold p:

- **Low p (0.1 - 0.3)**: Very conservative sampling, more predictable
- **Medium p (0.4 - 0.7)**: Balanced approach for most applications
- **High p (0.8 - 0.95)**: More unexpected choices, higher diversity

Note: When top_p is set to 0.0, nucleus sampling is disabled in MusicGen.

### Classifier-Free Guidance (cfg_coef)

This parameter controls how strictly the model follows the conditioning (your prompt). Higher values make the model adhere more closely to the prompt, potentially at the cost of quality:

- **Low cfg_coef (1.0 - 2.0)**: More freedom to deviate from the prompt, often more musical
- **Medium cfg_coef (2.5 - 3.5)**: Good balance between prompt adherence and quality
- **High cfg_coef (4.0 - 10.0)**: Strongly adheres to prompt, sometimes at cost of naturalness

## Understanding Parameter Interactions

These parameters interact in complex ways. For example:

1. A high temperature with a low top-k might still produce conservative outputs
2. A low temperature with a high top-p might produce more diverse results than expected
3. A high cfg_coef can compensate for a high-temperature setting by keeping generation more aligned with the prompt

The key is finding the right balance for your specific application, which requires systematic testing.

## Systematic Parameter Testing

Let's implement a parameter exploration system that allows us to understand how different settings affect our output:

```python
# parameter_explorer.py
import torch
import os
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import json
from datetime import datetime

class ParameterExplorer:
    """
    A tool for systematic exploration of MusicGen parameters.
    
    This class helps identify optimal parameter combinations by
    generating multiple samples with different parameter settings
    and organizing the results for easy comparison.
    """
    
    def __init__(self, model_size="medium", device=None):
        """Initialize the parameter explorer with a MusicGen model."""
        # Determine device automatically if not specified
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
                print("Using MPS (Metal) for generation")
            elif torch.cuda.is_available():
                device = "cuda"
                print("Using CUDA for generation")
            else:
                device = "cpu"
                print("Using CPU for generation (this will be slow)")
                
        self.device = device
        self.model_size = model_size
        
        # Load the model
        print(f"Loading MusicGen {model_size} model...")
        self.model = MusicGen.get_pretrained(model_size)
        self.model.to(device)
        
        # Track experiment history
        self.experiment_history = []
        
    def explore_single_parameter(self, prompt, parameter_name, parameter_values, 
                                 duration=5.0, output_dir=None):
        """
        Explore the effect of varying a single parameter while keeping others constant.
        
        Args:
            prompt (str): Text description of the music to generate
            parameter_name (str): Parameter to vary ('temperature', 'top_k', 'top_p', or 'cfg_coef')
            parameter_values (list): List of values to test for the parameter
            duration (float): Length of each sample in seconds
            output_dir (str): Directory to save samples, defaults to parameter name + timestamp
            
        Returns:
            str: Path to output directory containing the generated samples
        """
        # Setup default parameters
        default_params = {
            'duration': duration,
            'temperature': 1.0,
            'top_k': 250,
            'top_p': 0.0,
            'cfg_coef': 3.0
        }
        
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_dir is None:
            output_dir = f"{parameter_name}_exploration_{timestamp}"
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save experiment information
        experiment_info = {
            'prompt': prompt,
            'parameter_varied': parameter_name,
            'values_tested': parameter_values,
            'default_parameters': default_params,
            'timestamp': timestamp,
            'model_size': self.model_size
        }
        
        # Save experiment metadata
        with open(os.path.join(output_dir, "experiment_info.json"), 'w') as f:
            json.dump(experiment_info, f, indent=2)
            
        # Save prompt for reference
        with open(os.path.join(output_dir, "prompt.txt"), 'w') as f:
            f.write(f"Prompt: {prompt}\n\n")
            f.write(f"Parameter explored: {parameter_name}\n")
            f.write(f"Values tested: {parameter_values}\n\n")
            f.write("Default parameters:\n")
            for param, value in default_params.items():
                if param != parameter_name:  # Don't list the parameter we're varying
                    f.write(f"- {param}: {value}\n")
        
        # Track overall progress
        total_tests = len(parameter_values)
        completed = 0
        
        # Generate samples for each parameter value
        for value in parameter_values:
            completed += 1
            print(f"[{completed}/{total_tests}] Testing {parameter_name}={value}...")
            
            # Set generation parameters, with our test parameter modified
            generation_params = default_params.copy()
            generation_params[parameter_name] = value
            
            self.model.set_generation_params(**generation_params)
            
            # Generate audio with these parameters
            wav = self.model.generate([prompt])
            
            # Save with parameter value in filename
            output_path = os.path.join(output_dir, f"{parameter_name}_{value}")
            audio_write(
                output_path,
                wav[0].cpu(),
                self.model.sample_rate,
                strategy="loudness"
            )
            
            print(f"Saved sample with {parameter_name}={value} to {output_path}.wav")
        
        # Create a README with listening instructions
        self._create_readme(output_dir, parameter_name, prompt)
        
        # Add to experiment history
        self.experiment_history.append(experiment_info)
        
        print(f"\nExploration complete! All {total_tests} samples saved to {output_dir}")
        return output_dir
    
    def explore_parameter_grid(self, prompt, parameter_grid, duration=5.0, output_dir=None):
        """
        Explore combinations of parameters using a grid search approach.
        
        Args:
            prompt (str): Text description of the music to generate
            parameter_grid (dict): Dictionary mapping parameter names to lists of values
                                  e.g., {'temperature': [0.5, 1.0], 'top_k': [100, 250]}
            duration (float): Length of each sample in seconds
            output_dir (str): Directory to save samples, defaults to 'parameter_grid' + timestamp
            
        Returns:
            str: Path to output directory containing the generated samples
        """
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_dir is None:
            output_dir = f"parameter_grid_{timestamp}"
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Default parameters
        default_params = {
            'duration': duration,
            'temperature': 1.0,
            'top_k': 250,
            'top_p': 0.0,
            'cfg_coef': 3.0
        }
        
        # Update with parameters from grid
        for param in parameter_grid:
            if param in default_params:
                # Remove from defaults since we'll be varying it
                default_params.pop(param)
        
        # Save experiment information
        experiment_info = {
            'prompt': prompt,
            'parameter_grid': parameter_grid,
            'default_parameters': default_params,
            'timestamp': timestamp,
            'model_size': self.model_size
        }
        
        # Save metadata
        with open(os.path.join(output_dir, "experiment_info.json"), 'w') as f:
            json.dump(experiment_info, f, indent=2)
        
        # Generate all parameter combinations
        import itertools
        
        # Get all parameter names and their possible values from the grid
        param_names = list(parameter_grid.keys())
        param_values = [parameter_grid[param] for param in param_names]
        
        # Generate all combinations of parameter values
        combinations = list(itertools.product(*param_values))
        
        # Track progress
        total_combinations = len(combinations)
        print(f"Testing {total_combinations} parameter combinations...")
        
        # Generate samples for each parameter combination
        for i, combination in enumerate(combinations):
            # Create parameter dictionary for this combination
            params = {
                'duration': duration,
            }
            
            # Add fixed parameters
            for param, value in default_params.items():
                params[param] = value
                
            # Add grid parameters for this combination
            for j, param in enumerate(param_names):
                params[param] = combination[j]
            
            # Create descriptive filename based on parameters
            filename_parts = []
            for j, param in enumerate(param_names):
                # Format value to 1 decimal place if it's a float
                value = combination[j]
                if isinstance(value, float):
                    value_str = f"{value:.1f}"
                else:
                    value_str = str(value)
                filename_parts.append(f"{param[:3]}{value_str}")
            
            filename = "_".join(filename_parts)
            
            print(f"[{i+1}/{total_combinations}] Testing combination: {filename}")
            
            # Set generation parameters
            self.model.set_generation_params(**params)
            
            # Generate audio
            wav = self.model.generate([prompt])
            
            # Save the audio
            output_path = os.path.join(output_dir, filename)
            audio_write(
                output_path,
                wav[0].cpu(),
                self.model.sample_rate,
                strategy="loudness"
            )
            
            # Also save parameter info in a JSON file
            with open(f"{output_path}.json", 'w') as f:
                json.dump(params, f, indent=2)
            
            print(f"Saved sample to {output_path}.wav")
        
        # Create README with instructions
        with open(os.path.join(output_dir, "README.txt"), 'w') as f:
            f.write("PARAMETER GRID EXPLORATION\n")
            f.write("==========================\n\n")
            f.write(f"Prompt: \"{prompt}\"\n\n")
            f.write("Parameters explored:\n")
            for param, values in parameter_grid.items():
                f.write(f"- {param}: {values}\n")
            f.write("\nFixed parameters:\n")
            for param, value in default_params.items():
                f.write(f"- {param}: {value}\n")
            f.write("\nFilename format explains the parameters used:\n")
            f.write("Example: tem1.0_top250.wav means temperature=1.0, top_k=250\n\n")
            f.write("How to compare samples:\n")
            f.write("1. Listen systematically to isolate the effect of each parameter\n")
            f.write("2. Each sample has a JSON file with the exact parameters used\n")
            f.write("3. Take notes on which combinations work best for your use case\n")
        
        # Add to experiment history
        self.experiment_history.append(experiment_info)
        
        print(f"\nGrid exploration complete! All {total_combinations} samples saved to {output_dir}")
        return output_dir
    
    def load_best_params(self, json_path):
        """
        Load parameter settings from a JSON file.
        
        This is useful for reusing parameter combinations you've identified as optimal.
        """
        with open(json_path, 'r') as f:
            params = json.load(f)
        
        # Remove non-generation parameters
        if 'duration' in params:
            duration = params.pop('duration')
        else:
            duration = 10.0
            
        print(f"Loaded parameters from {json_path}")
        print(f"Parameters: {params}")
        
        return params, duration
    
    def save_preset(self, name, params, description=""):
        """Save a parameter preset for future use."""
        preset = {
            'name': name,
            'parameters': params,
            'description': description,
            'model_size': self.model_size,
            'created': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Create presets directory if it doesn't exist
        os.makedirs("parameter_presets", exist_ok=True)
        
        # Save preset
        preset_path = os.path.join("parameter_presets", f"{name}.json")
        with open(preset_path, 'w') as f:
            json.dump(preset, f, indent=2)
            
        print(f"Saved preset '{name}' to {preset_path}")
        return preset_path
    
    def _create_readme(self, output_dir, parameter_name, prompt):
        """Create a README file with guidance on parameter effects."""
        with open(os.path.join(output_dir, "README.txt"), 'w') as f:
            f.write(f"{parameter_name.upper()} EXPLORATION SAMPLES\n")
            f.write("="*len(f"{parameter_name.upper()} EXPLORATION SAMPLES")+"\n\n")
            f.write(f"Prompt used: \"{prompt}\"\n\n")
            
            f.write("How to interpret these samples:\n\n")
            
            if parameter_name == "temperature":
                f.write("TEMPERATURE controls randomness in generation:\n")
                f.write("- Lower values (0.1-0.7): More predictable, sometimes repetitive\n")
                f.write("- Medium values (0.8-1.2): Good balance of consistency and creativity\n")
                f.write("- Higher values (1.3+): More experimental, unexpected, sometimes chaotic\n")
                f.write("\nRecommendation: Start with 0.7-1.0 for most applications.\n")
                f.write("Use lower values for more consistent, genre-typical results.\n")
                f.write("Use higher values for creative exploration.\n")
                
            elif parameter_name == "top_k":
                f.write("TOP_K limits how many options the model considers at each step:\n")
                f.write("- Lower values (50-100): More focused but potentially repetitive\n")
                f.write("- Medium values (150-300): Good balance for most music\n")
                f.write("- Higher values (400+): Greater variety but potentially less coherent\n")
                f.write("\nRecommendation: 250 is a good default. Increase for more\n")
                f.write("experimental results, decrease for more coherent, safer output.\n")
                
            elif parameter_name == "top_p":
                f.write("TOP_P (nucleus sampling) dynamically limits token selection:\n")
                f.write("- Note: In MusicGen, top_p=0.0 means nucleus sampling is disabled\n")
                f.write("- Lower values (0.1-0.3): Very conservative, highly predictable\n")
                f.write("- Medium values (0.4-0.7): Balanced approach\n")
                f.write("- Higher values (0.8-0.95): Diverse but potentially less coherent\n")
                f.write("\nRecommendation: Either use 0.0 (disabled) and rely on top_k,\n")
                f.write("or try values around 0.9 with top_k=0 for a different sampling approach.\n")
                
            elif parameter_name == "cfg_coef":
                f.write("CFG_COEF (classifier-free guidance) controls prompt adherence:\n")
                f.write("- Lower values (1-2): More musical but may stray from prompt\n")
                f.write("- Medium values (2.5-3.5): Good balance for most use cases\n")
                f.write("- Higher values (4-10): Strictly follows prompt, sometimes at quality cost\n")
                f.write("\nRecommendation: 3.0 is a good default. Increase if model ignores\n")
                f.write("important aspects of your prompt. Decrease if output sounds forced or unnatural.\n")

def explore_temperature():
    """Example function to explore temperature parameter."""
    explorer = ParameterExplorer(model_size="medium")
    
    prompt = "A smooth jazz piece with piano, saxophone, and light drums"
    
    # Test a range of temperature values
    temperatures = [0.3, 0.5, 0.7, 1.0, 1.3, 1.6, 1.9]
    
    explorer.explore_single_parameter(
        prompt=prompt,
        parameter_name="temperature",
        parameter_values=temperatures,
        duration=5.0,
        output_dir="temperature_test"
    )

def explore_parameter_combinations():
    """Example function to explore combinations of parameters."""
    explorer = ParameterExplorer(model_size="medium")
    
    prompt = "An electronic track with a driving beat and ambient synth pads"
    
    # Define a parameter grid to explore
    parameter_grid = {
        'temperature': [0.7, 1.0, 1.3],
        'top_k': [100, 250],
        'cfg_coef': [2.0, 3.0]
    }
    
    # This will test all combinations: 3 temperatures × 2 top_k values × 2 cfg values = 12 combinations
    explorer.explore_parameter_grid(
        prompt=prompt,
        parameter_grid=parameter_grid,
        duration=5.0,
        output_dir="parameter_grid_test"
    )

if __name__ == "__main__":
    # Uncomment the function you want to run
    explore_temperature()
    # explore_parameter_combinations()
```

This comprehensive system allows you to:

1. Test individual parameters in isolation to understand their effects
2. Explore parameter combinations to find optimal settings
3. Save and reuse successful parameter presets
4. Document your findings for future reference

## Genre-Specific Parameter Optimization

Different musical genres often benefit from different parameter settings. Here are some starting points based on extensive testing:

### Classical Music
```python
classical_params = {
    'temperature': 0.7,     # Lower temperature for structured, coherent development
    'top_k': 250,           # Standard top_k works well
    'top_p': 0.0,           # Disable nucleus sampling
    'cfg_coef': 3.0         # Standard guidance
}
```

### Electronic Music
```python
electronic_params = {
    'temperature': 1.1,     # Slightly higher for interesting textures and sounds
    'top_k': 300,           # Higher diversity for electronic elements
    'top_p': 0.0,           # Disable nucleus sampling
    'cfg_coef': 2.5         # Slightly lower to allow creative sound design
}
```

### Jazz
```python
jazz_params = {
    'temperature': 1.2,     # Higher temperature for improvisational feel
    'top_k': 350,           # Higher diversity for jazz explorations
    'top_p': 0.0,           # Disable nucleus sampling
    'cfg_coef': 2.0         # Lower guidance to allow "jazz improvisation"
}
```

### Rock/Pop
```python
rock_pop_params = {
    'temperature': 0.9,     # Balanced temperature for structured but engaging music
    'top_k': 200,           # Moderate diversity
    'top_p': 0.0,           # Disable nucleus sampling
    'cfg_coef': 3.5         # Higher guidance for clear genre adherence
}
```

### Ambient/Atmospheric
```python
ambient_params = {
    'temperature': 1.0,     # Balanced for evolving textures
    'top_k': 250,           # Standard diversity
    'top_p': 0.0,           # Disable nucleus sampling
    'cfg_coef': 2.8         # Moderate guidance
}
```

## Parameter Optimization Workflow

Let's develop a systematic workflow for finding optimal parameters for your specific needs:

1. **Start with baselines**: Begin with the genre-specific presets above as starting points
2. **Isolate and test**: Use the `explore_single_parameter` method to understand each parameter in isolation
3. **Narrow down ranges**: Identify promising value ranges for each parameter
4. **Grid search**: Use `explore_parameter_grid` to test combinations within those promising ranges
5. **Evaluate results**: Create a consistent rating system for generated outputs
6. **Fine-tune**: Make small adjustments to the best combinations
7. **Create presets**: Save your optimal settings as presets for different use cases

## Common Parameter-Related Issues and Solutions

| Issue | Probable Parameter Cause | Solution |
|-------|--------------------------|----------|
| Output is too random/chaotic | Temperature too high | Reduce temperature to 0.6-0.8 |
| Output is too repetitive | Temperature too low | Increase temperature to 1.0-1.2 |
| Output ignores aspects of the prompt | cfg_coef too low | Increase cfg_coef to 3.5-4.5 |
| Output sounds forced and unnatural | cfg_coef too high | Decrease cfg_coef to 2.0-2.5 |
| Output lacks variety between generations | Sampling too restrictive | Increase top_k or experiment with top_p |
| Output is too generic | Temperature too low, top_k too low | Increase both for more creative results |

## Complete Implementation: Parameter Optimization Suite

Let's create a complete parameter optimization tool that builds on our ParameterExplorer class and adds a rating system to help you track which settings work best:

```python
# parameter_optimization_suite.py
import torch
import os
import json
import csv
from datetime import datetime
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import matplotlib.pyplot as plt
import numpy as np

class ParameterOptimizationSuite:
    """
    A comprehensive suite for optimizing MusicGen parameters and tracking results.
    
    Features:
    - Systematic parameter exploration
    - Result rating and tracking
    - Visualization of parameter effects
    - Preset management
    """
    
    def __init__(self, model_size="medium", device=None, results_dir="parameter_results"):
        """Initialize the optimization suite."""
        # Determine device automatically if not specified
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
                print("Using MPS (Metal) for generation")
            elif torch.cuda.is_available():
                device = "cuda"
                print("Using CUDA for generation")
            else:
                device = "cpu"
                print("Using CPU for generation (this will be slow)")
                
        self.device = device
        self.model_size = model_size
        self.results_dir = results_dir
        
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        # Results tracking
        self.results_file = os.path.join(results_dir, "parameter_results.csv")
        self._initialize_results_tracking()
        
        # Load the model
        print(f"Loading MusicGen {model_size} model...")
        self.model = MusicGen.get_pretrained(model_size)
        self.model.to(device)
        
        # Standard presets
        self.standard_presets = {
            "balanced": {
                'temperature': 1.0,
                'top_k': 250,
                'top_p': 0.0,
                'cfg_coef': 3.0
            },
            "creative": {
                'temperature': 1.3,
                'top_k': 350,
                'top_p': 0.0,
                'cfg_coef': 2.0
            },
            "conservative": {
                'temperature': 0.7,
                'top_k': 150,
                'top_p': 0.0,
                'cfg_coef': 3.5
            }
        }
        
    def _initialize_results_tracking(self):
        """Initialize or verify the results CSV file."""
        if not os.path.exists(self.results_file):
            with open(self.results_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'prompt', 'temperature', 'top_k', 'top_p', 'cfg_coef',
                    'duration', 'output_file', 'rating', 'notes'
                ])
    
    def generate_with_params(self, prompt, params, duration=10.0, output_dir=None):
        """
        Generate a sample with specific parameters and track the result.
        
        Args:
            prompt (str): Text description of the music to generate
            params (dict): Generation parameters (temperature, top_k, etc.)
            duration (float): Length of sample in seconds
            output_dir (str): Directory to save sample
            
        Returns:
            str: Path to the generated audio file
        """
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_dir is None:
            output_dir = os.path.join(self.results_dir, f"sample_{timestamp}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Set generation parameters
        generation_params = {
            'duration': duration,
            'temperature': params.get('temperature', 1.0),
            'top_k': params.get('top_k', 250),
            'top_p': params.get('top_p', 0.0),
            'cfg_coef': params.get('cfg_coef', 3.0)
        }
        
        self.model.set_generation_params(**generation_params)
        
        # Generate audio
        print(f"Generating sample with parameters: {generation_params}")
        wav = self.model.generate([prompt])
        
        # Save the audio
        output_filename = f"sample_{timestamp}"
        output_path = os.path.join(output_dir, output_filename)
        audio_write(
            output_path,
            wav[0].cpu(),
            self.model.sample_rate,
            strategy="loudness"
        )
        
        # Save parameters and prompt
        with open(f"{output_path}.json", 'w') as f:
            json.dump({
                'prompt': prompt,
                'parameters': generation_params,
                'timestamp': timestamp,
                'model_size': self.model_size
            }, f, indent=2)
        
        # Log to results CSV
        with open(self.results_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                prompt,
                generation_params['temperature'],
                generation_params['top_k'],
                generation_params['top_p'],
                generation_params['cfg_coef'],
                duration,
                f"{output_path}.wav",
                "", # Empty rating to be filled later
                ""  # Empty notes to be filled later
            ])
        
        print(f"Generated and saved to {output_path}.wav")
        return f"{output_path}.wav"
    
    def rate_sample(self, file_path, rating, notes=""):
        """
        Rate a generated sample and add notes.
        
        Args:
            file_path (str): Path to the audio file
            rating (int): Rating from 1-10
            notes (str): Optional notes about the quality
        """
        # Read current CSV
        rows = []
        updated = False
        
        with open(self.results_file, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader)
            rows.append(header)
            
            for row in reader:
                if row[7] == file_path:
                    row[8] = str(rating)
                    row[9] = notes
                    updated = True
                rows.append(row)
        
        if not updated:
            print(f"Warning: Could not find {file_path} in results log")
            return False
            
        # Write updated CSV
        with open(self.results_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
            
        print(f"Updated rating for {file_path} to {rating}/10")
        return True
    
    def generate_parameter_series(self, prompt, parameter, values, duration=5.0):
        """
        Generate a series of samples varying one parameter.
        
        Args:
            prompt (str): Text description of the music
            parameter (str): Parameter to vary
            values (list): List of values to test
            duration (float): Length of each sample in seconds
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(self.results_dir, f"{parameter}_series_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Default parameters
        params = {
            'temperature': 1.0,
            'top_k': 250,
            'top_p': 0.0,
            'cfg_coef': 3.0
        }
        
        # Generate for each value
        for value in values:
            test_params = params.copy()
            test_params[parameter] = value
            
            output_name = f"{parameter}_{value}"
            sample_dir = os.path.join(output_dir, output_name)
            
            self.generate_with_params(
                prompt=prompt,
                params=test_params,
                duration=duration,
                output_dir=sample_dir
            )
        
        print(f"Generated {len(values)} variations in {output_dir}")
        return output_dir
    
    def analyze_results(self):
        """
        Analyze parameter ratings to identify optimal settings.
        
        Returns:
            dict: Analysis of parameter effects on ratings
        """
        # Read results CSV
        data = []
        with open(self.results_file, 'r', newline='') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            
            for row in reader:
                # Skip entries without ratings
                if not row[8]:
                    continue
                    
                data.append({
                    'timestamp': row[0],
                    'prompt': row[1],
                    'temperature': float(row[2]),
                    'top_k': int(row[3]),
                    'top_p': float(row[4]),
                    'cfg_coef': float(row[5]),
                    'duration': float(row[6]),
                    'output_file': row[7],
                    'rating': int(row[8]),
                    'notes': row[9]
                })
        
        if not data:
            print("No rated samples found. Rate some samples first.")
            return None
            
        # Calculate average rating for each parameter value
        analysis = {
            'temperature': {},
            'top_k': {},
            'cfg_coef': {},
            'top_p': {},
            'overall_best': None,
            'parameter_correlations': {}
        }
        
        # For each parameter, group by value and calculate average rating
        for param in ['temperature', 'top_k', 'top_p', 'cfg_coef']:
            param_values = {}
            
            for entry in data:
                value = entry[param]
                if value not in param_values:
                    param_values[value] = []
                param_values[value].append(entry['rating'])
            
            # Calculate averages
            for value, ratings in param_values.items():
                avg_rating = sum(ratings) / len(ratings)
                analysis[param][value] = {
                    'avg_rating': avg_rating,
                    'sample_count': len(ratings)
                }
        
        # Find overall best-rated sample
        best_sample = max(data, key=lambda x: x['rating'])
        analysis['overall_best'] = {
            'prompt': best_sample['prompt'],
            'parameters': {
                'temperature': best_sample['temperature'],
                'top_k': best_sample['top_k'],
                'top_p': best_sample['top_p'],
                'cfg_coef': best_sample['cfg_coef']
            },
            'rating': best_sample['rating'],
            'output_file': best_sample['output_file']
        }
        
        # Calculate correlations between parameters and ratings
        import numpy as np
        ratings = np.array([entry['rating'] for entry in data])
        
        for param in ['temperature', 'top_k', 'top_p', 'cfg_coef']:
            param_values = np.array([entry[param] for entry in data])
            correlation = np.corrcoef(param_values, ratings)[0, 1]
            analysis['parameter_correlations'][param] = correlation
        
        return analysis
    
    def visualize_parameter_effects(self):
        """Generate plots showing how parameters affect ratings."""
        analysis = self.analyze_results()
        if not analysis:
            return
            
        # Create output directory for plots
        plots_dir = os.path.join(self.results_dir, "parameter_plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot for each parameter
        for param in ['temperature', 'top_k', 'cfg_coef']:
            if not analysis[param]:
                continue
                
            plt.figure(figsize=(10, 6))
            
            # Extract data for plotting
            values = []
            ratings = []
            counts = []
            
            for value, data in analysis[param].items():
                values.append(value)
                ratings.append(data['avg_rating'])
                counts.append(data['sample_count'])
            
            # Sort by parameter value
            sorted_data = sorted(zip(values, ratings, counts))
            values = [x[0] for x in sorted_data]
            ratings = [x[1] for x in sorted_data]
            counts = [x[2] for x in sorted_data]
            
            # Plot average ratings
            plt.bar(range(len(values)), ratings, tick_label=[str(v) for v in values])
            plt.xlabel(param)
            plt.ylabel("Average Rating (1-10)")
            plt.title(f"Effect of {param} on Music Quality")
            
            # Add sample counts
            for i, count in enumerate(counts):
                plt.text(i, 0.5, f"n={count}", ha='center')
            
            # Add correlation coefficient
            correlation = analysis['parameter_correlations'][param]
            plt.figtext(0.5, 0.01, f"Correlation: {correlation:.2f}", ha='center')
            
            # Save plot
            plt.savefig(os.path.join(plots_dir, f"{param}_effect.png"))
            plt.close()
        
        # Create overall recommendations summary
        with open(os.path.join(plots_dir, "parameter_recommendations.txt"), 'w') as f:
            f.write("PARAMETER OPTIMIZATION RECOMMENDATIONS\n")
            f.write("====================================\n\n")
            
            f.write("Best overall parameters found:\n")
            best = analysis['overall_best']
            f.write(f"From prompt: \"{best['prompt']}\"\n")
            for param, value in best['parameters'].items():
                f.write(f"- {param}: {value}\n")
            f.write(f"Rating: {best['rating']}/10\n\n")
            
            f.write("Parameter-specific recommendations:\n")
            for param in ['temperature', 'top_k', 'cfg_coef']:
                if not analysis[param]:
                    continue
                    
                # Find best value for this parameter
                best_value = max(analysis[param].items(), 
                                key=lambda x: x[1]['avg_rating'])
                
                f.write(f"- {param}: Best value = {best_value[0]} ")
                f.write(f"(avg rating: {best_value[1]['avg_rating']:.1f}/10)\n")
            
            # Add correlation insights
            f.write("\nParameter correlations with quality:\n")
            for param, corr in analysis['parameter_correlations'].items():
                direction = "higher is better" if corr > 0.1 else (
                    "lower is better" if corr < -0.1 else "minimal effect")
                f.write(f"- {param}: {corr:.2f} ({direction})\n")
        
        print(f"Visualization complete. Plots and recommendations saved to {plots_dir}")
        return plots_dir
    
    def get_preset(self, preset_name):
        """Get parameters from a standard or saved preset."""
        # Check standard presets first
        if preset_name in self.standard_presets:
            return self.standard_presets[preset_name]
            
        # Check saved presets
        preset_path = os.path.join("parameter_presets", f"{preset_name}.json")
        if os.path.exists(preset_path):
            with open(preset_path, 'r') as f:
                preset = json.load(f)
            return preset['parameters']
            
        print(f"Preset '{preset_name}' not found")
        return None
    
    def save_preset(self, name, params, description=""):
        """Save a parameter preset for future use."""
        preset = {
            'name': name,
            'parameters': params,
            'description': description,
            'model_size': self.model_size,
            'created': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Create presets directory if it doesn't exist
        os.makedirs("parameter_presets", exist_ok=True)
        
        # Save preset
        preset_path = os.path.join("parameter_presets", f"{name}.json")
        with open(preset_path, 'w') as f:
            json.dump(preset, f, indent=2)
            
        print(f"Saved preset '{name}' to {preset_path}")
        return preset_path
    
    def export_optimal_parameters(self):
        """
        Export the current optimal parameters based on analysis.
        """
        analysis = self.analyze_results()
        if not analysis or not analysis['overall_best']:
            print("Insufficient data for optimization. Rate more samples first.")
            return None
            
        optimal_params = analysis['overall_best']['parameters']
        
        # Save as a preset
        timestamp = datetime.now().strftime("%Y%m%d")
        preset_name = f"optimal_params_{timestamp}"
        
        return self.save_preset(
            preset_name,
            optimal_params,
            f"Optimized parameters based on {len(analysis['temperature'])} samples"
        )

# Example usage
def main():
    suite = ParameterOptimizationSuite(model_size="medium")
    
    # Generate samples with different parameters
    prompt = "A cinematic orchestral piece with dramatic strings and brass"
    
    # Generate with standard presets
    for preset_name in ["balanced", "creative", "conservative"]:
        preset_params = suite.get_preset(preset_name)
        suite.generate_with_params(
            prompt=prompt,
            params=preset_params,
            duration=8.0
        )
    
    # Try temperature variations
    suite.generate_parameter_series(
        prompt=prompt,
        parameter="temperature",
        values=[0.5, 0.8, 1.0, 1.2, 1.5],
        duration=5.0
    )
    
    print("Generated samples with various parameters.")
    print("Listen to the samples and rate them using suite.rate_sample(file_path, rating, notes)")
    print("Then run suite.analyze_results() and suite.visualize_parameter_effects()")

if __name__ == "__main__":
    main()
```

This complete implementation gives you everything you need to systematically find the best parameters for your use case, including:

1. A way to generate samples with different parameter settings
2. A rating system to track which settings work best
3. Analysis tools to identify optimal parameters
4. Visualization of parameter effects on music quality
5. A preset system for saving and reusing successful parameter combinations

## Common Pitfalls and Solutions

### 1. Focusing Too Much on Single Parameters

**Pitfall**: Testing parameters in isolation without considering their interactions.

**Solution**: Always test parameter combinations after understanding individual effects. The grid search approach in our ParameterOptimizationSuite handles this systematically.

### 2. Over-Optimizing for a Specific Prompt

**Pitfall**: Finding parameters that work well for one prompt but don't generalize.

**Solution**: Test your parameter settings with multiple different prompts in the same genre or style to ensure they're robust.

### 3. Ignoring Hardware Constraints

**Pitfall**: Finding optimal parameters that work on your development machine but cause out-of-memory errors in production.

**Solution**: Consider memory usage in your optimization. Higher top_k values and larger models use more memory. Set a realistic memory budget and optimize within those constraints.

### 4. Relying Too Heavily on Temperature

**Pitfall**: Using temperature as the only control knob for generation quality.

**Solution**: Explore the full parameter space. Sometimes adjusting cfg_coef or top_k can solve problems more effectively than changing temperature.

### 5. Not Documenting Your Findings

**Pitfall**: Finding great parameter combinations but forgetting which settings led to which results.

**Solution**: Use the rating and notes features in our ParameterOptimizationSuite to keep track of what works and why.

## Hands-on Challenge: Build Your Own Parameter Presets

Now that you understand how MusicGen parameters affect generation, it's time to create your own presets:

1. Choose a musical genre you want to optimize for
2. Using the ParameterOptimizationSuite, test at least three variations of each parameter
3. Rate each sample on a scale of 1-10
4. Analyze your results to identify trends
5. Create and save at least three presets:
   - A "safe" preset for reliable, consistent results
   - A "balanced" preset for everyday use
   - A "creative" preset for experimental exploration
6. Test your presets with three different prompts to ensure they generalize well

**Bonus Challenge**: Create a hybrid approach that varies parameters based on prompt content. For example, increase temperature for "experimental" prompts, decrease it for "traditional" prompts.

## Key Takeaways

- MusicGen parameters offer fine-grained control over the generation process
- Temperature is the most impactful parameter, controlling randomness and creativity
- Different genres and applications benefit from different parameter settings
- Systematic parameter testing leads to more consistent, high-quality results
- Parameter optimization is an iterative process that improves with feedback
- Creating parameter presets can save time and improve workflow efficiency
- The interaction between parameters is often more important than individual settings

## Next Steps

Now that you've mastered parameter optimization, you're ready to move on to more advanced MusicGen features:

- **Chapter 8: Melody Conditioning** - Learn how to condition MusicGen on existing melodies
- **Chapter 9: Batch Processing and Automation** - Scale up your music generation workflow

## Further Reading

- [AudioCraft GitHub: Generation Parameters](https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md#generation-parameters)
- [Sampling Strategies in Language Models](https://huggingface.co/blog/how-to-generate)
- [Classifier-Free Guidance Explained](https://arxiv.org/abs/2207.12598)
- [Parameter Optimization for Creative AI](https://arxiv.org/abs/2205.05674)
- [MusicGen: Simple and Controllable Music Generation](https://arxiv.org/abs/2306.05284)
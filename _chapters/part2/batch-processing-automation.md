---
layout: chapter
title: "Batch Processing and Automation for AI Audio Generation"
difficulty: intermediate
copyright: "Copyright © 2025 Scott Friedman. This work is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License."
estimated_time: 60
---

> *"I need to create variations of 50 different musical themes, each in 3 different styles with 2 different parameter settings. That's 300 audio files! Generating them one by one would take forever, not to mention keeping everything organized. There must be a way to automate this process and handle it efficiently."* 
> 
> — Alex Winters, game audio director

# Chapter 9: Batch Processing and Automation for AI Audio Generation

## The Challenge

You've mastered generating individual music pieces with MusicGen and you understand how to optimize parameters and use melody conditioning. But real-world projects often require generating dozens, hundreds, or even thousands of audio samples efficiently. Manual generation quickly becomes impractical at scale.

In this chapter, we'll tackle the challenge of building scalable, automated audio generation systems. We'll create frameworks for batch processing, implement organizational structures for the outputs, and develop efficient workflows that can handle large-scale audio generation tasks.

## Learning Objectives

By the end of this chapter, you will be able to:

- Implement batch processing systems for generating multiple audio samples
- Create frameworks for systematic variation experiments
- Build audio generation pipelines with configurable workflows
- Develop organizational structures for managing large collections of generated audio
- Automate the documentation process for generated content
- Optimize system resources for efficient large-scale generation
- Implement progress tracking and error handling for long-running processes

## Prerequisites

- Basic understanding of MusicGen and AudioGen (Chapters 5-8)
- Experience with Python programming and file operations
- Familiarity with parameter optimization (Chapter 7)
- Python environment with AudioCraft installed

## Key Concepts: Approaches to Batch Processing

Before diving into implementation, let's explore different approaches to batch processing for AI audio generation:

### 1. Sequential Batch Processing

The simplest approach is to process items one after another in a sequential manner:

```python
def generate_sequential_batch(prompts, output_dir, model_type="music", model_size="small", duration=5.0):
    """
    Generate multiple audio samples one after another.
    
    Args:
        prompts (list): List of text prompts to generate
        output_dir (str): Directory to save all outputs
        model_type (str): "music" or "audio"
        model_size (str): Model size to use
        duration (float): Duration for all samples
        
    Returns:
        list: Paths to all generated files
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the appropriate model
    if model_type == "music":
        model = MusicGen.get_pretrained(model_size)
    else:
        model = AudioGen.get_pretrained(model_size)
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    model.to(device)
    
    # Set generation parameters
    model.set_generation_params(
        duration=duration,
        temperature=1.0,
        top_k=250,
        top_p=0.0,
    )
    
    # Track generated files
    generated_files = []
    
    # Generate each prompt and save the output
    total_prompts = len(prompts)
    for i, prompt in enumerate(prompts):
        print(f"[{i+1}/{total_prompts}] Generating: '{prompt}'")
        
        # Create a safe filename from the prompt
        filename = f"{i+1:03d}_" + "".join(c if c.isalnum() or c == "_" else "_" for c in prompt[:50])
        
        # Generate audio
        wav = model.generate([prompt])
        
        # Save the audio
        output_path = os.path.join(output_dir, filename)
        audio_write(
            output_path,
            wav[0].cpu(),
            model.sample_rate,
            strategy="loudness"
        )
        
        # Add to generated files list
        generated_files.append(f"{output_path}.wav")
        print(f"Saved to {output_path}.wav")
    
    print(f"Batch generation complete. Generated {len(generated_files)} files.")
    return generated_files
```

This approach is straightforward but doesn't take advantage of potential parallel processing opportunities.

### 2. Parameterized Batch Processing

A more flexible approach is to generate variations across multiple parameters:

```python
def generate_parameter_variations(prompt, output_dir, parameters_to_vary):
    """
    Generate variations of a single prompt with different parameter settings.
    
    Args:
        prompt (str): Base text prompt to use for all variations
        output_dir (str): Directory to save outputs
        parameters_to_vary (dict): Dictionary mapping parameter names to lists of values
            Example: {"temperature": [0.5, 1.0, 1.5], "duration": [5.0, 10.0]}
            
    Returns:
        list: Paths to all generated files
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model (using small for faster iteration)
    model = MusicGen.get_pretrained("small")
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    model.to(device)
    
    # Create all possible parameter combinations
    import itertools
    
    # Get parameter names and values
    param_names = list(parameters_to_vary.keys())
    param_values = [parameters_to_vary[param] for param in param_names]
    
    # Generate all combinations
    combinations = list(itertools.product(*param_values))
    
    # Default parameters (will be overridden by specific variations)
    default_params = {
        "temperature": 1.0,
        "top_k": 250,
        "top_p": 0.0,
        "cfg_coef": 3.0
    }
    
    # Track generated files
    generated_files = []
    
    # Generate each parameter combination
    total_combinations = len(combinations)
    for i, combination in enumerate(combinations):
        # Create parameter dictionary for this combination
        params = default_params.copy()
        
        # Add specific parameters for this combination
        for j, param_name in enumerate(param_names):
            params[param_name] = combination[j]
        
        # Create descriptive filename
        filename_parts = [f"{param_names[j]}_{combination[j]}" for j in range(len(param_names))]
        filename = "_".join(filename_parts)
        
        print(f"[{i+1}/{total_combinations}] Generating with parameters: {filename}")
        
        # Set generation parameters
        generation_params = {k: v for k, v in params.items() if k != "duration"}
        model.set_generation_params(
            duration=params.get("duration", 5.0),
            **generation_params
        )
        
        # Generate audio
        wav = model.generate([prompt])
        
        # Save the audio
        output_path = os.path.join(output_dir, filename)
        audio_write(
            output_path,
            wav[0].cpu(),
            model.sample_rate,
            strategy="loudness"
        )
        
        # Save parameter metadata
        with open(f"{output_path}.json", "w") as f:
            json.dump({
                "prompt": prompt,
                "parameters": params
            }, f, indent=2)
        
        # Add to generated files list
        generated_files.append(f"{output_path}.wav")
        print(f"Saved to {output_path}.wav")
    
    # Create summary file
    with open(os.path.join(output_dir, "variations_summary.txt"), "w") as f:
        f.write(f"PARAMETER VARIATIONS SUMMARY\n")
        f.write("=========================\n\n")
        f.write(f"Base prompt: \"{prompt}\"\n\n")
        f.write("Parameters varied:\n")
        for param, values in parameters_to_vary.items():
            f.write(f"- {param}: {values}\n")
        f.write(f"\nTotal variations: {total_combinations}\n")
    
    print(f"Parameter variation complete. Generated {len(generated_files)} files.")
    return generated_files
```

This approach allows for systematic exploration of parameter spaces, which is valuable for experimentation and finding optimal settings.

### 3. Pipeline-based Batch Processing

For more complex workflows, a pipeline architecture provides better modularity and reusability:

```python
class AudioGenerationPipeline:
    """
    A modular pipeline for batch processing audio generation tasks.
    
    This class encapsulates the entire generation workflow with distinct stages:
    1. Pipeline initialization (models, devices)
    2. Batch configuration (parameters, outputs)
    3. Generation execution
    4. Post-processing
    5. Documentation
    """
    
    def __init__(self, model_type="music", model_size="small", device=None):
        """Initialize the generation pipeline with models and device settings."""
        self.model_type = model_type
        self.model_size = model_size
        
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
        
        # Load the appropriate model
        print(f"Loading {model_type} model ({model_size})...")
        if model_type == "music":
            self.model = MusicGen.get_pretrained(model_size)
        else:
            self.model = AudioGen.get_pretrained(model_size)
        
        self.model.to(device)
        
        # Initialize state variables
        self.batch_config = None
        self.generated_files = []
        self.metadata = {}
    
    def configure_batch(self, prompts, output_dir, generation_params=None, batch_name=None):
        """
        Configure the batch generation parameters.
        
        Args:
            prompts (list or dict): List of prompts or dictionary mapping IDs to prompts
            output_dir (str): Directory to save outputs
            generation_params (dict): Generation parameters
            batch_name (str): Optional name for this batch
        """
        # Convert prompts to dictionary if it's a list
        if isinstance(prompts, list):
            self.prompts = {f"prompt_{i+1:03d}": prompt for i, prompt in enumerate(prompts)}
        else:
            self.prompts = prompts
        
        # Store output directory
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Store batch name
        self.batch_name = batch_name or f"batch_{int(time.time())}"
        
        # Set default generation parameters if not specified
        self.generation_params = generation_params or {
            "duration": 5.0,
            "temperature": 1.0,
            "top_k": 250,
            "top_p": 0.0,
            "cfg_coef": 3.0
        }
        
        # Store configuration for metadata
        self.batch_config = {
            "batch_name": self.batch_name,
            "model_type": self.model_type,
            "model_size": self.model_size,
            "device": self.device,
            "prompt_count": len(self.prompts),
            "generation_params": self.generation_params,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        print(f"Batch configured: {len(self.prompts)} prompts, output to {output_dir}")
        return self
    
    def execute(self, progressive_save=True):
        """
        Execute the batch generation process.
        
        Args:
            progressive_save (bool): Whether to save files as they're generated
                                   or all at once at the end
                                   
        Returns:
            list: Paths to generated files
        """
        if not self.batch_config:
            raise ValueError("Batch not configured. Call configure_batch first.")
        
        # Set generation parameters
        self.model.set_generation_params(
            duration=self.generation_params.get("duration", 5.0),
            temperature=self.generation_params.get("temperature", 1.0),
            top_k=self.generation_params.get("top_k", 250),
            top_p=self.generation_params.get("top_p", 0.0),
            cfg_coef=self.generation_params.get("cfg_coef", 3.0)
        )
        
        # Prepare to store results
        self.generated_files = []
        self.metadata = {
            "config": self.batch_config,
            "generations": {}
        }
        
        # Track progress
        total_prompts = len(self.prompts)
        start_time = time.time()
        
        print(f"Starting batch generation of {total_prompts} audio samples...")
        
        # Generate each prompt
        for i, (prompt_id, prompt) in enumerate(self.prompts.items()):
            # Create safe filename
            safe_id = "".join(c if c.isalnum() or c == "_" else "_" for c in prompt_id)
            
            # Display progress
            print(f"[{i+1}/{total_prompts}] Generating {prompt_id}: '{prompt}'")
            
            # Generate audio
            generation_start = time.time()
            wav = self.model.generate([prompt])
            generation_time = time.time() - generation_start
            
            # Create output path
            output_path = os.path.join(self.output_dir, safe_id)
            
            # Save the audio if progressive saving is enabled
            if progressive_save:
                audio_write(
                    output_path,
                    wav[0].cpu(),
                    self.model.sample_rate,
                    strategy="loudness"
                )
                print(f"Saved to {output_path}.wav ({generation_time:.2f}s)")
            
            # Store generation information
            self.generated_files.append({
                "prompt_id": prompt_id,
                "prompt": prompt,
                "output_path": f"{output_path}.wav",
                "wav_tensor": wav[0] if not progressive_save else None
            })
            
            # Store metadata
            self.metadata["generations"][prompt_id] = {
                "prompt": prompt,
                "output_file": f"{output_path}.wav",
                "generation_time": generation_time
            }
        
        # Save any remaining files if not using progressive save
        if not progressive_save:
            print("Saving all generated files...")
            for item in self.generated_files:
                audio_write(
                    os.path.splitext(item["output_path"])[0],
                    item["wav_tensor"].cpu(),
                    self.model.sample_rate,
                    strategy="loudness"
                )
                # Clear tensor reference to free memory
                item["wav_tensor"] = None
        
        # Calculate statistics
        total_time = time.time() - start_time
        avg_time = total_time / total_prompts
        
        # Add statistics to metadata
        self.metadata["statistics"] = {
            "total_generation_time": total_time,
            "average_per_sample": avg_time,
            "total_samples": total_prompts
        }
        
        print(f"Batch generation complete!")
        print(f"Total time: {total_time:.2f}s, Average per sample: {avg_time:.2f}s")
        
        # Return list of just the output paths for convenience
        return [item["output_path"] for item in self.generated_files]
    
    def create_documentation(self):
        """
        Create comprehensive documentation for the batch generation.
        
        This includes:
        - README with batch overview
        - CSV listing of all generated files
        - JSON metadata file with all generation details
        """
        if not self.metadata:
            raise ValueError("No generation metadata available. Run execute first.")
        
        # Save complete metadata as JSON
        metadata_path = os.path.join(self.output_dir, "batch_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)
        
        # Create CSV listing for easy spreadsheet import
        csv_path = os.path.join(self.output_dir, "generation_listing.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(["Prompt ID", "Prompt", "Output File", "Generation Time (s)"])
            # Write data
            for prompt_id, data in self.metadata["generations"].items():
                writer.writerow([
                    prompt_id,
                    data["prompt"],
                    os.path.basename(data["output_file"]),
                    f"{data['generation_time']:.2f}"
                ])
        
        # Create README with batch overview
        readme_path = os.path.join(self.output_dir, "README.txt")
        with open(readme_path, "w") as f:
            f.write(f"BATCH GENERATION: {self.batch_name}\n")
            f.write("=" * (len(self.batch_name) + 18) + "\n\n")
            
            # Configuration summary
            f.write("CONFIGURATION:\n")
            f.write(f"- Model: {self.model_type.capitalize()}-{self.model_size}\n")
            f.write(f"- Device: {self.device}\n")
            f.write(f"- Generation date: {self.batch_config['timestamp']}\n")
            f.write(f"- Total prompts: {len(self.prompts)}\n\n")
            
            # Generation parameters
            f.write("GENERATION PARAMETERS:\n")
            for param, value in self.generation_params.items():
                f.write(f"- {param}: {value}\n")
            f.write("\n")
            
            # Statistics
            stats = self.metadata["statistics"]
            f.write("STATISTICS:\n")
            f.write(f"- Total generation time: {stats['total_generation_time']:.2f} seconds\n")
            f.write(f"- Average time per sample: {stats['average_per_sample']:.2f} seconds\n")
            f.write(f"- Total samples generated: {stats['total_samples']}\n\n")
            
            # Files guide
            f.write("FILES:\n")
            f.write("- *.wav: Generated audio files\n")
            f.write("- batch_metadata.json: Complete generation metadata\n")
            f.write("- generation_listing.csv: Spreadsheet-compatible listing of all files\n")
            f.write("- README.txt: This file\n")
        
        print(f"Documentation created:")
        print(f"- {readme_path}")
        print(f"- {csv_path}")
        print(f"- {metadata_path}")
        
        return {
            "readme": readme_path,
            "csv": csv_path,
            "metadata": metadata_path
        }
```

The pipeline approach offers several advantages:
- Modular design with clear separation of concerns
- Easier to extend and customize for different workflows
- Better organization with built-in documentation generation
- Cleaner error handling and resource management

## Implementing Systematic Variation Experiments

One common use case for batch processing is conducting systematic variation experiments to explore how different parameters and prompts affect generation results. Let's implement a framework specifically designed for this purpose:

```python
# variation_experiments.py
import os
import json
import time
import csv
from datetime import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt
from audiocraft.models import MusicGen, AudioGen
from audiocraft.data.audio import audio_write

class VariationExperiment:
    """
    Framework for conducting systematic audio generation variation experiments.
    
    This class allows you to explore how different parameters and prompt
    formulations affect generation results through structured experiments.
    
    Key features:
    - Experiment definition with control variables and experimental variables
    - Automated generation of all experiment conditions
    - Structured output organization
    - Comprehensive experiment documentation
    - Built-in visualization capabilities
    """
    
    def __init__(self, name, model_type="music", model_size="small", base_output_dir="experiments"):
        """
        Initialize a new variation experiment.
        
        Args:
            name (str): Name of the experiment
            model_type (str): Type of model to use ("music" or "audio")
            model_size (str): Size of model to use
            base_output_dir (str): Base directory for all experiments
        """
        self.name = name
        self.model_type = model_type
        self.model_size = model_size
        
        # Create timestamp for this experiment
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create a unique experiment directory
        safe_name = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
        self.experiment_dir = os.path.join(base_output_dir, f"{safe_name}_{self.timestamp}")
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Initialize model
        self._initialize_model()
        
        # Experiment definition (to be set by configure methods)
        self.control_variables = {}
        self.experiment_variables = {}
        self.experiment_design = None
        
        # Results storage
        self.results = {}
    
    def _initialize_model(self):
        """Load the appropriate model and move to the best available device."""
        # Determine the best available device
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        print(f"Loading {self.model_type} model ({self.model_size}) on {self.device}...")
        
        # Load the appropriate model
        if self.model_type == "music":
            self.model = MusicGen.get_pretrained(self.model_size)
        else:
            self.model = AudioGen.get_pretrained(self.model_size)
        
        self.model.to(self.device)
    
    def set_control_variables(self, **kwargs):
        """
        Set control variables that remain constant throughout the experiment.
        
        Args:
            **kwargs: Control variables as keyword arguments
                     (e.g., duration=5.0, top_k=250)
        """
        self.control_variables = kwargs
        return self
    
    def set_experiment_variables(self, **kwargs):
        """
        Set experiment variables that will be systematically varied.
        
        Args:
            **kwargs: Variables as keyword arguments mapping to lists of values
                     (e.g., temperature=[0.5, 1.0, 1.5], prompt=["A", "B", "C"])
        """
        self.experiment_variables = kwargs
        return self
    
    def design_full_factorial(self):
        """
        Create a full factorial design testing all combinations of experiment variables.
        
        This creates all possible combinations of the experiment variables,
        resulting in a comprehensive but potentially large experiment.
        """
        import itertools
        
        # Verify we have experiment variables
        if not self.experiment_variables:
            raise ValueError("No experiment variables defined. Use set_experiment_variables first.")
        
        # Get variable names and values
        var_names = list(self.experiment_variables.keys())
        var_values = [self.experiment_variables[var] for var in var_names]
        
        # Generate all combinations
        combinations = list(itertools.product(*var_values))
        
        # Create experiment design
        self.experiment_design = []
        for i, combination in enumerate(combinations):
            condition = {
                "condition_id": f"condition_{i+1:03d}",
                "variables": {var_names[j]: combination[j] for j in range(len(var_names))}
            }
            self.experiment_design.append(condition)
        
        print(f"Designed full factorial experiment with {len(self.experiment_design)} conditions")
        return self
    
    def design_one_factor_at_a_time(self, base_values=None):
        """
        Create a one-factor-at-a-time design.
        
        This varies each experiment variable independently while keeping others
        at their base values, resulting in a more manageable experiment size.
        
        Args:
            base_values (dict): Base values for each variable when not being varied
                               If not provided, first value of each variable is used
        """
        # Verify we have experiment variables
        if not self.experiment_variables:
            raise ValueError("No experiment variables defined. Use set_experiment_variables first.")
        
        # Use first value as base value if not specified
        if base_values is None:
            base_values = {var: values[0] for var, values in self.experiment_variables.items()}
        
        # Create experiment design
        self.experiment_design = []
        condition_counter = 1
        
        # Add baseline condition with all variables at base values
        baseline = {
            "condition_id": f"baseline",
            "variables": base_values.copy()
        }
        self.experiment_design.append(baseline)
        
        # For each variable, create conditions varying only that variable
        for var_name, values in self.experiment_variables.items():
            for value in values:
                # Skip if this is the base value (already in baseline condition)
                if value == base_values.get(var_name):
                    continue
                
                # Create a condition varying just this variable
                variables = base_values.copy()
                variables[var_name] = value
                
                condition = {
                    "condition_id": f"condition_{condition_counter:03d}_{var_name}_{value}",
                    "variables": variables
                }
                self.experiment_design.append(condition)
                condition_counter += 1
        
        print(f"Designed one-factor-at-a-time experiment with {len(self.experiment_design)} conditions")
        return self
    
    def run_experiment(self):
        """
        Execute the designed experiment by generating audio for all conditions.
        
        Returns:
            dict: Experiment results
        """
        # Verify we have a design
        if not self.experiment_design:
            raise ValueError("No experiment design. Use a design_* method first.")
        
        # Create results structure
        self.results = {
            "experiment_name": self.name,
            "timestamp": self.timestamp,
            "model_type": self.model_type,
            "model_size": self.model_size,
            "device": self.device,
            "control_variables": self.control_variables,
            "experiment_variables": self.experiment_variables,
            "conditions": {}
        }
        
        # Create conditions directory
        conditions_dir = os.path.join(self.experiment_dir, "conditions")
        os.makedirs(conditions_dir, exist_ok=True)
        
        # Set control variables in generation parameters
        base_params = self.control_variables.copy()
        
        # Run each experimental condition
        total_conditions = len(self.experiment_design)
        start_time = time.time()
        
        for i, condition in enumerate(self.experiment_design):
            condition_id = condition["condition_id"]
            variables = condition["variables"]
            
            print(f"[{i+1}/{total_conditions}] Running condition: {condition_id}")
            
            # Create directory for this condition
            condition_dir = os.path.join(conditions_dir, condition_id)
            os.makedirs(condition_dir, exist_ok=True)
            
            # Save condition details
            with open(os.path.join(condition_dir, "condition.json"), "w") as f:
                json.dump({
                    "condition_id": condition_id,
                    "variables": variables,
                    "control_variables": base_params
                }, f, indent=2)
            
            # Extract prompt and other generation parameters
            prompt = variables.get("prompt", "")
            generation_params = base_params.copy()
            
            # Update generation parameters with experimental variables
            for var_name, value in variables.items():
                if var_name != "prompt":  # Handle prompt separately
                    generation_params[var_name] = value
            
            # Set duration separately as it's a special parameter
            duration = generation_params.pop("duration", 5.0)
            
            # Set generation parameters
            self.model.set_generation_params(
                duration=duration,
                **{k: v for k, v in generation_params.items() 
                   if k in ["temperature", "top_k", "top_p", "cfg_coef"]}
            )
            
            # Generate audio
            start_generate = time.time()
            wav = self.model.generate([prompt])
            generation_time = time.time() - start_generate
            
            # Save the audio
            output_path = os.path.join(condition_dir, "output")
            audio_write(
                output_path,
                wav[0].cpu(),
                self.model.sample_rate,
                strategy="loudness"
            )
            
            # Record results
            self.results["conditions"][condition_id] = {
                "variables": variables,
                "output_file": f"{output_path}.wav",
                "generation_time": generation_time
            }
            
            print(f"  Generated in {generation_time:.2f}s")
        
        # Calculate total experiment time
        total_time = time.time() - start_time
        self.results["total_experiment_time"] = total_time
        self.results["average_generation_time"] = total_time / total_conditions
        
        # Save complete results
        results_path = os.path.join(self.experiment_dir, "experiment_results.json")
        with open(results_path, "w") as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Experiment complete! {total_conditions} conditions in {total_time:.2f}s")
        print(f"Results saved to {results_path}")
        
        return self.results
    
    def create_experiment_report(self):
        """
        Create a comprehensive report of the experiment.
        
        This includes README, CSV listing, and visualizations.
        
        Returns:
            dict: Paths to report files
        """
        if not self.results:
            raise ValueError("No results available. Run the experiment first.")
        
        report_files = {}
        
        # Create README with experiment overview
        readme_path = os.path.join(self.experiment_dir, "README.txt")
        with open(readme_path, "w") as f:
            f.write(f"EXPERIMENT: {self.name}\n")
            f.write("=" * (len(self.name) + 12) + "\n\n")
            
            # Basic info
            f.write("OVERVIEW:\n")
            f.write(f"- Date: {self.timestamp}\n")
            f.write(f"- Model: {self.model_type.capitalize()}-{self.model_size}\n")
            f.write(f"- Device: {self.device}\n")
            f.write(f"- Conditions: {len(self.results['conditions'])}\n")
            f.write(f"- Total time: {self.results['total_experiment_time']:.2f}s\n\n")
            
            # Control variables
            f.write("CONTROL VARIABLES:\n")
            for var, value in self.control_variables.items():
                f.write(f"- {var}: {value}\n")
            f.write("\n")
            
            # Experiment variables
            f.write("EXPERIMENT VARIABLES:\n")
            for var, values in self.experiment_variables.items():
                f.write(f"- {var}: {values}\n")
            f.write("\n")
            
            # Directory structure
            f.write("DIRECTORY STRUCTURE:\n")
            f.write("- conditions/: Contains subdirectories for each experimental condition\n")
            f.write("- conditions/*/output.wav: Generated audio for each condition\n")
            f.write("- conditions/*/condition.json: Details of each condition\n")
            f.write("- experiment_results.json: Complete experiment results\n")
            f.write("- conditions.csv: CSV listing of all conditions\n")
            if "prompt" in self.experiment_variables:
                f.write("- prompt_comparison.png: Visualization of prompt variations\n")
            if "temperature" in self.experiment_variables:
                f.write("- temperature_comparison.png: Visualization of temperature variations\n")
        
        report_files["readme"] = readme_path
        
        # Create CSV listing
        csv_path = os.path.join(self.experiment_dir, "conditions.csv")
        with open(csv_path, "w", newline="") as f:
            # Determine all possible variable columns
            all_vars = set()
            for condition in self.results["conditions"].values():
                all_vars.update(condition["variables"].keys())
            
            # Create CSV writer with dynamic columns
            writer = csv.writer(f)
            header = ["Condition ID"] + list(all_vars) + ["Output File", "Generation Time (s)"]
            writer.writerow(header)
            
            # Write each condition
            for condition_id, data in self.results["conditions"].items():
                row = [condition_id]
                
                # Add each variable's value (or blank if not used in this condition)
                for var in all_vars:
                    row.append(data["variables"].get(var, ""))
                
                # Add output file and generation time
                row.append(os.path.basename(data["output_file"]))
                row.append(f"{data['generation_time']:.2f}")
                
                writer.writerow(row)
        
        report_files["csv"] = csv_path
        
        # Create visualizations if applicable
        if "temperature" in self.experiment_variables:
            self._create_temperature_visualization()
            report_files["temperature_viz"] = os.path.join(
                self.experiment_dir, "temperature_comparison.png")
        
        if "prompt" in self.experiment_variables and len(self.experiment_variables["prompt"]) <= 10:
            self._create_prompt_comparison()
            report_files["prompt_viz"] = os.path.join(
                self.experiment_dir, "prompt_comparison.png")
        
        print(f"Experiment report created!")
        return report_files
    
    def _create_temperature_visualization(self):
        """Create visualization of generation time vs temperature."""
        if "temperature" not in self.experiment_variables:
            return
        
        # Collect temperature and generation time data
        temps = []
        times = []
        
        for condition in self.results["conditions"].values():
            if "temperature" in condition["variables"]:
                temps.append(condition["variables"]["temperature"])
                times.append(condition["generation_time"])
        
        if not temps:
            return
        
        # Create the visualization
        plt.figure(figsize=(10, 6))
        plt.scatter(temps, times, alpha=0.7, s=100)
        plt.plot(temps, times, 'b--', alpha=0.5)
        
        plt.title("Effect of Temperature on Generation Time")
        plt.xlabel("Temperature")
        plt.ylabel("Generation Time (seconds)")
        plt.grid(True, alpha=0.3)
        
        # Add annotations
        for i, (temp, time) in enumerate(zip(temps, times)):
            plt.annotate(
                f"{time:.2f}s", 
                (temp, time),
                xytext=(5, 5),
                textcoords="offset points"
            )
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.experiment_dir, "temperature_comparison.png"))
        plt.close()
    
    def _create_prompt_comparison(self):
        """Create visual comparison of different prompts."""
        if "prompt" not in self.experiment_variables:
            return
        
        # Limit to reasonable number of prompts
        if len(self.experiment_variables["prompt"]) > 10:
            return
        
        # Collect prompt and generation time data
        prompts = []
        times = []
        
        for condition in self.results["conditions"].values():
            if "prompt" in condition["variables"]:
                # Get shortened prompt for display
                prompt = condition["variables"]["prompt"]
                if len(prompt) > 30:
                    prompt = prompt[:27] + "..."
                
                prompts.append(prompt)
                times.append(condition["generation_time"])
        
        if not prompts:
            return
        
        # Create the visualization
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(prompts)), times, alpha=0.7)
        
        plt.title("Generation Time by Prompt")
        plt.xlabel("Prompt")
        plt.ylabel("Generation Time (seconds)")
        plt.xticks(range(len(prompts)), prompts, rotation=45, ha="right")
        plt.grid(True, alpha=0.3, axis="y")
        
        # Add time labels
        for bar, time in zip(bars, times):
            plt.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.1,
                f"{time:.2f}s",
                ha="center"
            )
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.experiment_dir, "prompt_comparison.png"))
        plt.close()
```

This framework provides a scientific approach to exploring variations:

1. Precise control of experimental variables
2. Multiple experimental design options (full factorial, one-factor-at-a-time)
3. Comprehensive documentation with visualizations
4. Clear organization of experimental conditions

## Building an Audio Generation Automation System

Now let's create a complete automation system for large-scale audio generation:

```python
# audio_generation_automation.py
import os
import json
import time
import csv
import threading
import queue
from datetime import datetime
import torch
from audiocraft.models import MusicGen, AudioGen
from audiocraft.data.audio import audio_write

class GenerationJob:
    """
    Represents a single generation job with all required parameters.
    
    This class encapsulates all the information needed to generate a single
    audio sample, making it easy to queue and track jobs in a batch system.
    """
    
    def __init__(self, job_id, prompt, output_dir, **params):
        """
        Initialize a generation job.
        
        Args:
            job_id (str): Unique identifier for this job
            prompt (str): Text prompt for generation
            output_dir (str): Directory to save output
            **params: Generation parameters (duration, temperature, etc.)
        """
        self.job_id = job_id
        self.prompt = prompt
        self.output_dir = output_dir
        self.params = params
        
        # Set default parameter values if not provided
        self.params.setdefault("duration", 5.0)
        self.params.setdefault("temperature", 1.0)
        self.params.setdefault("top_k", 250)
        self.params.setdefault("top_p", 0.0)
        self.params.setdefault("cfg_coef", 3.0)
        
        # Track job status
        self.status = "queued"  # queued, running, completed, failed
        self.start_time = None
        self.end_time = None
        self.output_file = None
        self.error = None
    
    def to_dict(self):
        """Convert job to dictionary representation for serialization."""
        return {
            "job_id": self.job_id,
            "prompt": self.prompt,
            "output_dir": self.output_dir,
            "params": self.params,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.end_time - self.start_time if self.start_time and self.end_time else None,
            "output_file": self.output_file,
            "error": self.error
        }


class GenerationWorker:
    """
    Worker that processes generation jobs from a queue.
    
    This worker runs in a separate thread to generate audio samples
    based on job specifications from a queue.
    """
    
    def __init__(self, worker_id, job_queue, results_queue, model_type, model_size, device):
        """
        Initialize a generation worker.
        
        Args:
            worker_id (int): Unique identifier for this worker
            job_queue (Queue): Queue to get jobs from
            results_queue (Queue): Queue to put results in
            model_type (str): Type of model to use ("music" or "audio")
            model_size (str): Size of model to use
            device (str): Device to run model on
        """
        self.worker_id = worker_id
        self.job_queue = job_queue
        self.results_queue = results_queue
        self.model_type = model_type
        self.model_size = model_size
        self.device = device
        
        # Initialize model
        if model_type == "music":
            self.model = MusicGen.get_pretrained(model_size)
        else:
            self.model = AudioGen.get_pretrained(model_size)
        
        self.model.to(device)
        
        # Control flags
        self.running = False
        self.thread = None
    
    def start(self):
        """Start the worker thread."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._worker_loop)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        """Stop the worker thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
    
    def _worker_loop(self):
        """Main worker loop that processes jobs from the queue."""
        print(f"Worker {self.worker_id} started on {self.device} using {self.model_type}-{self.model_size}")
        
        while self.running:
            try:
                # Get job from queue with timeout to allow checking running flag
                job = self.job_queue.get(timeout=1.0)
                
                # Process the job
                self._process_job(job)
                
                # Mark job as done in queue
                self.job_queue.task_done()
                
            except queue.Empty:
                # No jobs available, continue loop
                continue
            except Exception as e:
                # Log error but keep worker alive
                print(f"Worker {self.worker_id} encountered error: {str(e)}")
                continue
        
        print(f"Worker {self.worker_id} stopped")
    
    def _process_job(self, job):
        """
        Process a single generation job.
        
        Args:
            job (GenerationJob): Job to process
        """
        try:
            print(f"Worker {self.worker_id} processing job {job.job_id}")
            
            # Update job status
            job.status = "running"
            job.start_time = time.time()
            
            # Ensure output directory exists
            os.makedirs(job.output_dir, exist_ok=True)
            
            # Extract parameters
            duration = job.params.get("duration", 5.0)
            
            # Set generation parameters
            self.model.set_generation_params(
                duration=duration,
                temperature=job.params.get("temperature", 1.0),
                top_k=job.params.get("top_k", 250),
                top_p=job.params.get("top_p", 0.0),
                cfg_coef=job.params.get("cfg_coef", 3.0)
            )
            
            # Generate audio
            wav = self.model.generate([job.prompt])
            
            # Save output
            output_path = os.path.join(job.output_dir, job.job_id)
            audio_write(
                output_path,
                wav[0].cpu(),
                self.model.sample_rate,
                strategy="loudness"
            )
            
            # Update job status
            job.status = "completed"
            job.end_time = time.time()
            job.output_file = f"{output_path}.wav"
            
            # Put result in results queue
            self.results_queue.put(job)
            
            print(f"Worker {self.worker_id} completed job {job.job_id} ({job.end_time - job.start_time:.2f}s)")
            
        except Exception as e:
            # Update job status on error
            job.status = "failed"
            job.end_time = time.time()
            job.error = str(e)
            
            # Put failed job in results queue
            self.results_queue.put(job)
            
            print(f"Worker {self.worker_id} failed job {job.job_id}: {str(e)}")


class GenerationAutomationSystem:
    """
    Complete system for automating large-scale audio generation tasks.
    
    This system manages the entire generation workflow:
    - Loading and managing jobs from various sources
    - Running multiple workers to process jobs concurrently
    - Tracking job progress and results
    - Organizing and documenting output
    - Handling errors and retries
    """
    
    def __init__(self, base_output_dir="automated_generation"):
        """
        Initialize the automation system.
        
        Args:
            base_output_dir (str): Base directory for all generation output
        """
        self.base_output_dir = base_output_dir
        os.makedirs(base_output_dir, exist_ok=True)
        
        # Create session timestamp and directory
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.session_dir = os.path.join(base_output_dir, self.session_id)
        os.makedirs(self.session_dir, exist_ok=True)
        
        # Create log directory
        self.log_dir = os.path.join(self.session_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create output directory
        self.output_dir = os.path.join(self.session_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize queues
        self.job_queue = queue.Queue()
        self.results_queue = queue.Queue()
        
        # Initialize workers list
        self.workers = []
        
        # Job tracking
        self.jobs = {}
        self.completed_jobs = {}
        self.failed_jobs = {}
        
        # Status flags
        self.running = False
        self.result_thread = None
    
    def add_job(self, prompt, params=None, job_id=None):
        """
        Add a single generation job to the queue.
        
        Args:
            prompt (str): Text prompt for generation
            params (dict): Generation parameters
            job_id (str): Optional job ID (generated if not provided)
            
        Returns:
            str: Job ID
        """
        # Generate job ID if not provided
        if job_id is None:
            job_id = f"job_{len(self.jobs) + 1:06d}"
        
        # Create job output directory
        job_output_dir = os.path.join(self.output_dir, job_id)
        
        # Create the job
        job = GenerationJob(
            job_id=job_id,
            prompt=prompt,
            output_dir=job_output_dir,
            **(params or {})
        )
        
        # Add to tracking
        self.jobs[job_id] = job
        
        # Add to queue if system is running
        if self.running:
            self.job_queue.put(job)
        
        return job_id
    
    def add_jobs_from_list(self, prompts, common_params=None):
        """
        Add multiple jobs from a list of prompts.
        
        Args:
            prompts (list): List of prompt strings
            common_params (dict): Parameters to use for all jobs
            
        Returns:
            list: List of job IDs
        """
        job_ids = []
        for i, prompt in enumerate(prompts):
            job_id = f"job_{len(self.jobs) + 1:06d}"
            self.add_job(prompt, params=common_params, job_id=job_id)
            job_ids.append(job_id)
        
        return job_ids
    
    def add_jobs_from_csv(self, csv_file, prompt_column="prompt", params_columns=None):
        """
        Add jobs from a CSV file.
        
        Args:
            csv_file (str): Path to CSV file
            prompt_column (str): Name of column containing prompts
            params_columns (list): Names of columns containing parameters
            
        Returns:
            list: List of job IDs
        """
        job_ids = []
        
        with open(csv_file, newline="") as f:
            reader = csv.DictReader(f)
            
            for i, row in enumerate(reader):
                # Extract prompt
                if prompt_column not in row:
                    print(f"Warning: Row {i} missing prompt column '{prompt_column}'")
                    continue
                
                prompt = row[prompt_column]
                
                # Extract parameters if specified
                params = {}
                if params_columns:
                    for param in params_columns:
                        if param in row and row[param]:
                            # Try to convert numeric values appropriately
                            try:
                                value = float(row[param])
                                # Convert to int if it's a whole number
                                if value.is_integer():
                                    value = int(value)
                                params[param] = value
                            except ValueError:
                                # Keep as string if not numeric
                                params[param] = row[param]
                
                # Add the job
                job_id = f"job_{len(self.jobs) + 1:06d}"
                self.add_job(prompt, params=params, job_id=job_id)
                job_ids.append(job_id)
        
        return job_ids
    
    def add_jobs_from_variations(self, base_prompt, variations, common_params=None):
        """
        Add jobs from text variations of a base prompt.
        
        Args:
            base_prompt (str): Base prompt template with {var} placeholders
            variations (dict): Dictionary mapping variables to lists of values
            common_params (dict): Parameters to use for all jobs
            
        Returns:
            list: List of job IDs
        """
        import itertools
        
        job_ids = []
        
        # Get all variable names and possible values
        var_names = list(variations.keys())
        var_values = [variations[var] for var in var_names]
        
        # Generate all combinations
        combinations = list(itertools.product(*var_values))
        
        # Generate a job for each combination
        for i, combination in enumerate(combinations):
            # Create variable dictionary for this combination
            vars_dict = {var_names[j]: combination[j] for j in range(len(var_names))}
            
            # Format the prompt with these variables
            try:
                prompt = base_prompt.format(**vars_dict)
                
                # Add the job
                job_id = f"job_{len(self.jobs) + 1:06d}"
                self.add_job(prompt, params=common_params, job_id=job_id)
                job_ids.append(job_id)
                
            except KeyError as e:
                print(f"Warning: Missing variable in prompt template: {e}")
                continue
        
        return job_ids
    
    def start_workers(self, num_workers=1, model_type="music", model_size="small", devices=None):
        """
        Start worker threads to process the job queue.
        
        Args:
            num_workers (int): Number of worker threads to start
            model_type (str): Type of model to use
            model_size (str): Size of model to use
            devices (list): List of devices to use. If None, will automatically select.
                          If fewer devices than workers, will cycle through available devices.
            
        Returns:
            bool: True if workers started successfully
        """
        # Determine devices if not specified
        if devices is None:
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                # Use all available CUDA devices
                devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
            elif torch.backends.mps.is_available():
                # Use MPS (Apple Silicon)
                devices = ["mps"]
            else:
                # Fall back to CPU
                devices = ["cpu"]
        
        # Make sure we have at least one device
        if not devices:
            devices = ["cpu"]
        
        print(f"Starting {num_workers} workers on devices: {devices}")
        
        # Create and start workers
        for i in range(num_workers):
            # Select device (cycling through available devices)
            device = devices[i % len(devices)]
            
            # Create worker
            worker = GenerationWorker(
                worker_id=i,
                job_queue=self.job_queue,
                results_queue=self.results_queue,
                model_type=model_type,
                model_size=model_size,
                device=device
            )
            
            # Add to workers list
            self.workers.append(worker)
            
            # Start the worker
            worker.start()
        
        # Start result processing thread
        self.running = True
        self.result_thread = threading.Thread(target=self._process_results)
        self.result_thread.daemon = True
        self.result_thread.start()
        
        # Add all tracked jobs to the queue
        for job in self.jobs.values():
            if job.status == "queued":
                self.job_queue.put(job)
        
        return True
    
    def _process_results(self):
        """Process completed jobs from the results queue."""
        while self.running:
            try:
                # Get result from queue with timeout
                job = self.results_queue.get(timeout=1.0)
                
                # Process the result
                if job.status == "completed":
                    self.completed_jobs[job.job_id] = job
                elif job.status == "failed":
                    self.failed_jobs[job.job_id] = job
                
                # Mark as done in queue
                self.results_queue.task_done()
                
                # Save intermediate progress periodically
                if len(self.completed_jobs) % 10 == 0:
                    self._save_progress()
                
            except queue.Empty:
                # No results available, continue loop
                continue
            except Exception as e:
                # Log error but keep thread alive
                print(f"Error processing results: {str(e)}")
                continue
    
    def _save_progress(self):
        """Save current progress to log file."""
        log_path = os.path.join(self.log_dir, "progress.json")
        
        progress = {
            "session_id": self.session_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_jobs": len(self.jobs),
            "completed_jobs": len(self.completed_jobs),
            "failed_jobs": len(self.failed_jobs),
            "queued_jobs": len(self.jobs) - len(self.completed_jobs) - len(self.failed_jobs)
        }
        
        with open(log_path, "w") as f:
            json.dump(progress, f, indent=2)
    
    def wait_until_done(self, progress_interval=5.0):
        """
        Wait until all jobs are processed.
        
        Args:
            progress_interval (float): Interval in seconds to print progress updates
            
        Returns:
            dict: Summary of processing results
        """
        if not self.running:
            print("System is not running. Start workers first.")
            return None
        
        last_update = time.time()
        total_jobs = len(self.jobs)
        
        # Wait for queue to be empty
        while not self.job_queue.empty() or self.job_queue.unfinished_tasks > 0:
            # Print progress update at specified interval
            if time.time() - last_update > progress_interval:
                completed = len(self.completed_jobs)
                failed = len(self.failed_jobs)
                remaining = total_jobs - completed - failed
                
                print(f"Progress: {completed}/{total_jobs} completed, {failed} failed, {remaining} remaining")
                last_update = time.time()
            
            # Small sleep to prevent CPU spinning
            time.sleep(0.5)
        
        # Final progress update
        completed = len(self.completed_jobs)
        failed = len(self.failed_jobs)
        print(f"All jobs processed: {completed}/{total_jobs} completed, {failed} failed")
        
        # Create summary
        summary = {
            "session_id": self.session_id,
            "total_jobs": total_jobs,
            "completed_jobs": completed,
            "failed_jobs": failed,
            "session_dir": self.session_dir
        }
        
        return summary
    
    def stop(self):
        """Stop all workers and the result processing thread."""
        self.running = False
        
        # Stop workers
        for worker in self.workers:
            worker.stop()
        
        # Wait for result thread to finish
        if self.result_thread:
            self.result_thread.join(timeout=1.0)
        
        # Clear workers list
        self.workers = []
        
        print("System stopped")
        return True
    
    def create_final_report(self):
        """
        Create a comprehensive final report of the generation session.
        
        Returns:
            dict: Paths to report files
        """
        report_files = {}
        
        # Create summary file
        summary_path = os.path.join(self.session_dir, "summary.json")
        
        summary = {
            "session_id": self.session_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_jobs": len(self.jobs),
            "completed_jobs": len(self.completed_jobs),
            "failed_jobs": len(self.failed_jobs),
            "completion_rate": len(self.completed_jobs) / len(self.jobs) if self.jobs else 0
        }
        
        # Calculate statistics if we have completed jobs
        if self.completed_jobs:
            # Calculate timing statistics
            durations = []
            for job in self.completed_jobs.values():
                if job.start_time and job.end_time:
                    durations.append(job.end_time - job.start_time)
            
            if durations:
                summary["timing"] = {
                    "average_duration": sum(durations) / len(durations),
                    "min_duration": min(durations),
                    "max_duration": max(durations),
                    "total_processing_time": sum(durations)
                }
        
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        report_files["summary"] = summary_path
        
        # Create CSV listings of jobs
        csv_path = os.path.join(self.session_dir, "jobs.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                "Job ID", "Status", "Prompt", "Duration (s)", 
                "Output File", "Error", "Temperature", "Top-k", "Top-p"
            ])
            
            # Write all jobs
            for job in self.jobs.values():
                duration = None
                if job.start_time and job.end_time:
                    duration = job.end_time - job.start_time
                
                writer.writerow([
                    job.job_id,
                    job.status,
                    job.prompt[:50] + "..." if len(job.prompt) > 50 else job.prompt,
                    f"{duration:.2f}" if duration else "",
                    os.path.basename(job.output_file) if job.output_file else "",
                    job.error or "",
                    job.params.get("temperature", ""),
                    job.params.get("top_k", ""),
                    job.params.get("top_p", "")
                ])
        
        report_files["csv"] = csv_path
        
        # Create README with session overview
        readme_path = os.path.join(self.session_dir, "README.txt")
        with open(readme_path, "w") as f:
            f.write(f"GENERATION SESSION: {self.session_id}\n")
            f.write("=" * (len(self.session_id) + 19) + "\n\n")
            
            # Basic statistics
            f.write("SESSION SUMMARY:\n")
            f.write(f"- Total jobs: {summary['total_jobs']}\n")
            f.write(f"- Completed: {summary['completed_jobs']} ")
            f.write(f"({summary['completion_rate']*100:.1f}%)\n")
            f.write(f"- Failed: {summary['failed_jobs']}\n")
            
            if "timing" in summary:
                f.write("\nTIMING INFORMATION:\n")
                f.write(f"- Average generation time: {summary['timing']['average_duration']:.2f}s\n")
                f.write(f"- Minimum time: {summary['timing']['min_duration']:.2f}s\n")
                f.write(f"- Maximum time: {summary['timing']['max_duration']:.2f}s\n")
                f.write(f"- Total processing time: {summary['timing']['total_processing_time']:.2f}s\n")
            
            f.write("\nDIRECTORY STRUCTURE:\n")
            f.write("- output/: Contains all generated audio files\n")
            f.write("- output/{job_id}/: Job-specific output directories\n")
            f.write("- logs/: Log files and progress tracking\n")
            f.write("- summary.json: Complete session summary\n")
            f.write("- jobs.csv: CSV listing of all jobs\n")
            
            f.write("\nNOTES:\n")
            f.write("- Failed jobs can be found in jobs.csv with status 'failed'\n")
            f.write("- Each job output directory contains the generated audio\n")
        
        report_files["readme"] = readme_path
        
        print(f"Final report created:")
        for name, path in report_files.items():
            print(f"- {name}: {path}")
        
        return report_files
```

This automation system provides everything needed for large-scale production:

1. **Comprehensive job management**: Add jobs individually, from lists, CSV files, or by generating variations
2. **Multi-worker processing**: Uses multiple workers across available devices
3. **Progress tracking**: Live updates on job status
4. **Error handling**: Robust error tracking and reporting
5. **Detailed documentation**: Comprehensive reports and logs

## Automated Testing and Quality Control

For production applications, it's important to have automated testing and quality control. Here's a framework for implementing this:

```python
# audio_quality_control.py
import os
import numpy as np
import torch
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

class AudioQualityControl:
    """
    Automated quality control system for generated audio.
    
    This class provides methods to assess the quality of generated audio
    and detect common issues such as silence, clipping, and repetition.
    """
    
    def __init__(self):
        """Initialize the quality control system."""
        pass
    
    def check_audio(self, audio_tensor, sample_rate=32000):
        """
        Perform comprehensive quality checks on an audio tensor.
        
        Args:
            audio_tensor (torch.Tensor): Audio tensor to check
            sample_rate (int): Sample rate of the audio
            
        Returns:
            dict: Quality assessment results
        """
        # Convert to numpy for analysis if needed
        if isinstance(audio_tensor, torch.Tensor):
            audio_np = audio_tensor.numpy()
        else:
            audio_np = audio_tensor
        
        # Initialize results dictionary
        results = {
            "passed": True,
            "issues": [],
            "metrics": {}
        }
        
        # Check for silence
        silence_check = self.check_silence(audio_np)
        if not silence_check["passed"]:
            results["passed"] = False
            results["issues"].append(f"Silence: {silence_check['details']}")
        results["metrics"]["silence"] = silence_check
        
        # Check for clipping
        clipping_check = self.check_clipping(audio_np)
        if not clipping_check["passed"]:
            results["passed"] = False
            results["issues"].append(f"Clipping: {clipping_check['details']}")
        results["metrics"]["clipping"] = clipping_check
        
        # Check for repetition
        repetition_check = self.check_repetition(audio_np, sample_rate)
        if not repetition_check["passed"]:
            results["passed"] = False
            results["issues"].append(f"Repetition: {repetition_check['details']}")
        results["metrics"]["repetition"] = repetition_check
        
        # Check for imbalance
        balance_check = self.check_spectral_balance(audio_np, sample_rate)
        if not balance_check["passed"]:
            results["passed"] = False
            results["issues"].append(f"Spectral imbalance: {balance_check['details']}")
        results["metrics"]["spectral_balance"] = balance_check
        
        return results
    
    def check_silence(self, audio, threshold_db=-60, max_silent_percentage=30):
        """
        Check if audio contains too much silence.
        
        Args:
            audio (numpy.ndarray): Audio data
            threshold_db (float): Silence threshold in dB
            max_silent_percentage (float): Maximum allowed percentage of silence
            
        Returns:
            dict: Check results
        """
        # Convert dB threshold to amplitude
        threshold_amp = 10 ** (threshold_db / 20)
        
        # Calculate RMS amplitude in short windows
        window_size = 1024
        hop_size = window_size // 2
        
        n_windows = (len(audio) - window_size) // hop_size + 1
        silent_windows = 0
        
        for i in range(n_windows):
            start = i * hop_size
            end = start + window_size
            window = audio[start:end]
            rms = np.sqrt(np.mean(window ** 2))
            
            if rms < threshold_amp:
                silent_windows += 1
        
        # Calculate percentage of silent windows
        silent_percentage = (silent_windows / n_windows) * 100 if n_windows > 0 else 0
        
        # Check if silence exceeds threshold
        passed = silent_percentage <= max_silent_percentage
        
        return {
            "passed": passed,
            "silent_percentage": silent_percentage,
            "details": f"{silent_percentage:.1f}% of audio is silent (threshold: {max_silent_percentage}%)"
        }
    
    def check_clipping(self, audio, threshold=0.99, max_clip_percentage=1):
        """
        Check if audio contains clipping.
        
        Args:
            audio (numpy.ndarray): Audio data
            threshold (float): Clipping threshold (0.0-1.0)
            max_clip_percentage (float): Maximum allowed percentage of clipping
            
        Returns:
            dict: Check results
        """
        # Count samples above threshold
        clipped_samples = np.sum(np.abs(audio) > threshold)
        clip_percentage = (clipped_samples / len(audio)) * 100
        
        # Check if clipping exceeds threshold
        passed = clip_percentage <= max_clip_percentage
        
        return {
            "passed": passed,
            "clip_percentage": clip_percentage,
            "details": f"{clip_percentage:.2f}% of samples show clipping (threshold: {max_clip_percentage}%)"
        }
    
    def check_repetition(self, audio, sample_rate, segment_duration=1.0, similarity_threshold=0.95):
        """
        Check if audio contains excessive repetition.
        
        Args:
            audio (numpy.ndarray): Audio data
            sample_rate (int): Sample rate of the audio
            segment_duration (float): Duration of segments to compare in seconds
            similarity_threshold (float): Similarity threshold for repetition detection
            
        Returns:
            dict: Check results
        """
        # Calculate segment size in samples
        segment_size = int(segment_duration * sample_rate)
        
        # If audio is too short for analysis, return passed
        if len(audio) < segment_size * 2:
            return {
                "passed": True,
                "repetition_score": 0.0,
                "details": "Audio too short for repetition analysis"
            }
        
        # Divide audio into segments
        num_segments = len(audio) // segment_size
        segments = [audio[i*segment_size:(i+1)*segment_size] for i in range(num_segments)]
        
        # Calculate cross-correlation between consecutive segments
        max_correlation = 0
        for i in range(num_segments - 1):
            correlation = np.corrcoef(segments[i], segments[i+1])[0, 1]
            max_correlation = max(max_correlation, correlation)
        
        # Check if repetition exceeds threshold
        passed = max_correlation < similarity_threshold
        
        return {
            "passed": passed,
            "repetition_score": max_correlation,
            "details": f"Maximum segment correlation: {max_correlation:.2f} (threshold: {similarity_threshold})"
        }
    
    def check_spectral_balance(self, audio, sample_rate, min_db_range=30):
        """
        Check spectral balance of audio.
        
        Args:
            audio (numpy.ndarray): Audio data
            sample_rate (int): Sample rate of the audio
            min_db_range (float): Minimum required dynamic range in dB
            
        Returns:
            dict: Check results
        """
        # Calculate spectrum
        n_fft = 2048
        hop_length = n_fft // 4
        
        # Compute spectrogram
        audio_tensor = torch.from_numpy(audio).float()
        spec = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            power=2
        )(audio_tensor)
        
        # Convert to dB scale (avoid log of zero)
        spec_db = 10 * torch.log10(spec + 1e-10)
        
        # Calculate average spectrum across time
        avg_spectrum = torch.mean(spec_db, dim=1).numpy()
        
        # Calculate spectrum range in dB
        db_range = np.max(avg_spectrum) - np.min(avg_spectrum)
        
        # Check if spectral range is sufficient
        passed = db_range >= min_db_range
        
        # Calculate spectrum tilt (high frequency vs low frequency energy)
        low_energy = np.mean(avg_spectrum[:len(avg_spectrum)//4])
        high_energy = np.mean(avg_spectrum[3*len(avg_spectrum)//4:])
        tilt = low_energy - high_energy
        
        return {
            "passed": passed,
            "db_range": db_range,
            "spectral_tilt": tilt,
            "details": f"Spectral range: {db_range:.1f} dB (threshold: {min_db_range} dB), Tilt: {tilt:.1f} dB"
        }
    
    def batch_check_audio_files(self, audio_files):
        """
        Check quality of multiple audio files.
        
        Args:
            audio_files (list): List of audio file paths
            
        Returns:
            dict: Check results for each file
        """
        results = {}
        
        for file_path in audio_files:
            try:
                # Load audio file
                waveform, sample_rate = torchaudio.load(file_path)
                
                # Convert to mono if needed
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                # Check audio quality
                check_result = self.check_audio(waveform[0], sample_rate)
                
                # Store results
                results[file_path] = check_result
                
            except Exception as e:
                # Handle errors
                results[file_path] = {
                    "passed": False,
                    "issues": [f"Error processing file: {str(e)}"],
                    "metrics": {}
                }
        
        return results
```

This quality control system can be integrated into your batch processing workflow to automatically detect issues with generated audio, such as:

1. **Silence detection**: Identifies audio with excessive silent segments
2. **Clipping detection**: Finds samples with amplitude distortion
3. **Repetition detection**: Detects repetitive patterns that might indicate model issues
4. **Spectral balance**: Ensures audio has appropriate frequency distribution

## Hands-on Challenge: Build a Music Generation Factory

Now it's time to apply everything you've learned to create a complete music generation factory:

1. Create a comprehensive pipeline that:
   - Takes a CSV file with music generation specifications (prompts, parameters)
   - Processes batches of generations efficiently
   - Implements quality control to filter out problematic generations
   - Creates well-organized output with comprehensive documentation

2. Extend the pipeline with:
   - A retry mechanism for failed generations
   - A results evaluation system that rates generations
   - Parallel processing across multiple available devices

3. Create a reporting system that:
   - Generates summary statistics on generation quality
   - Creates visualizations of key metrics
   - Identifies optimal parameter settings based on quality scores

**Bonus Challenge**: Implement a web dashboard that monitors generation progress and displays results in real-time.

## Key Takeaways

- Batch processing is essential for scaling AI audio generation to production levels
- Well-designed pipelines with clear separation of concerns improve maintainability
- Systematic experimentation helps identify optimal parameters for different use cases
- Proper organization and documentation are crucial when working with large volumes of generated content
- Quality control systems help maintain consistent output quality
- Resource optimization is important for efficient large-scale generation

## Next Steps

Now that you've mastered MusicGen and learned how to scale your generation processes, you're ready to move on to the next section of our tutorial:

- **Chapter 10: Introduction to AudioGen** - Shift focus from music to sound effects generation
- **Chapter 11: Sound Effect Generation Techniques** - Learn specialized techniques for SFX
- **Chapter 12: Audio Scene Composition** - Combine multiple sound elements into cohesive scenes
- **Chapter 13: Sound Design Workflows** - Build professional sound design pipelines

## Further Reading

- [Large-Scale Audio Generation with AudioCraft](https://ai.meta.com/blog/audiocraft-musicgen-audiogen-encodec-generative-ai-audio/)
- [Model Parallelism for Efficient Generation](https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html)
- [Scientific Experimentation with AI Systems](https://arxiv.org/abs/2104.01778)
- [Systematic Parameter Optimization](https://proceedings.mlr.press/v119/turner20a/turner20a.pdf)
- [Quality Control for Audio Generation](https://arxiv.org/abs/2206.01283)
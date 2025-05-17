---
layout: chapter
title: "Audio Scene Composition"
difficulty: intermediate
copyright: "Copyright © 2025 Scott Friedman. This work is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License."
estimated_time: 60
---

> *"We're creating an immersive VR experience set in different biomes across the world. Each environment needs a complete, cohesive soundscape that blends ambient sounds, wildlife, weather, and subtle music. We want users to feel truly present in these spaces, and the audio has to adjust based on time of day and weather conditions. Creating all these audio environments manually would take months of work."* 
> 
> — Sophia Ramirez, VR experience producer

# Chapter 12: Audio Scene Composition

## The Challenge

Generating individual sound effects is powerful, but real-world applications often require complete audio scenes—cohesive soundscapes that combine multiple audio elements to create immersive environments. These environments need to feel natural, balanced, and appropriate for their context, whether it's a rainforest at dawn, a cyberpunk cityscape at night, or an alien world with exotic phenomena.

In this chapter, we'll explore techniques for composing complete audio scenes by combining multiple AI-generated elements into coherent, multi-layered soundscapes.

## Learning Objectives

By the end of this chapter, you will be able to:

- Create multi-layered audio scenes that combine different sound categories
- Implement dynamic audio environments that respond to parameter changes
- Balance and mix multiple audio elements for natural, cohesive soundscapes
- Design flexible scene templates for different environment types
- Apply spatial audio concepts to create immersive 3D soundscapes
- Develop scene transition systems for smooth audio changes
- Export and integrate composed audio scenes into various applications

## Prerequisites

- Understanding of AudioGen basics (Chapter 10)
- Experience with sound effect generation techniques (Chapter 11)
- Python environment with AudioCraft installed
- Basic understanding of audio mixing concepts

## Key Concepts: Audio Scene Architecture

### Understanding Audio Scene Layers

Complete audio scenes typically consist of multiple layers, each serving a different purpose:

1. **Ambient Background**: Continuous, relatively uniform sounds that establish the base atmosphere (room tone, environmental ambience)
2. **Environmental Elements**: Distinctive sounds that characterize the environment (weather, machinery, nature)
3. **Dynamic Events**: Occasional, non-continuous sounds that add interest and variation (animals, vehicles, interactions)
4. **Musical Elements**: Optional background music or tonal elements that enhance mood
5. **Narrative Elements**: Sounds that support the story or context (dialogue, key sound objects)

These layers combine to create a complete sonic environment that feels alive and immersive.

### Scene Composition Principles

When composing audio scenes, several key principles help create effective results:

1. **Proper Layering**: Each element should occupy its appropriate space in the mix
2. **Frequency Balance**: Different elements should occupy different frequency ranges
3. **Dynamic Spacing**: Sounds should be timed to avoid overwhelming the listener
4. **Contextual Relevance**: All elements should make sense in the environment
5. **Emotional Coherence**: The emotional tone should be consistent across elements

## Building a Scene Composition System

Let's implement a complete scene composition system:

```python
# audio_scene_composer.py
import torch
import os
import json
from datetime import datetime
import random
import numpy as np
from audiocraft.models import MusicGen, AudioGen
from audiocraft.data.audio import audio_write

class AudioSceneComposer:
    """
    System for composing complete audio scenes by combining multiple
    AI-generated elements into cohesive soundscapes.
    """
    
    def __init__(self, music_model_size="small", audio_model_size="medium", device=None):
        """Initialize the scene composer with music and audio models."""
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
        
        # Load the models
        print(f"Loading MusicGen {music_model_size} model...")
        self.music_model = MusicGen.get_pretrained(music_model_size)
        self.music_model.to(device)
        
        print(f"Loading AudioGen {audio_model_size} model...")
        self.audio_model = AudioGen.get_pretrained(audio_model_size)
        self.audio_model.to(device)
        
        # Store sample rate
        self.sample_rate = self.music_model.sample_rate  # Same for both models
        
        # Track generated elements
        self.scene_elements = {}
    
    def generate_scene_element(self, element_id, prompt, model_type, duration=5.0, temperature=1.0):
        """
        Generate a single scene element.
        
        Args:
            element_id (str): Identifier for this element
            prompt (str): Description of the sound to generate
            model_type (str): Type of model to use ("music" or "audio")
            duration (float): Duration in seconds
            temperature (float): Generation temperature
            
        Returns:
            torch.Tensor: Generated audio tensor
        """
        # Choose the appropriate model
        if model_type == "music":
            model = self.music_model
        else:
            model = self.audio_model
        
        # Set generation parameters
        model.set_generation_params(
            duration=duration,
            temperature=temperature,
            top_k=250,
            top_p=0.0,
        )
        
        print(f"Generating {model_type} element '{element_id}'...")
        print(f"Prompt: '{prompt}'")
        
        # Generate the audio
        wav = model.generate([prompt])
        audio_tensor = wav[0].cpu()
        
        # Store element info
        self.scene_elements[element_id] = {
            "prompt": prompt,
            "model_type": model_type,
            "duration": duration,
            "temperature": temperature,
            "audio": audio_tensor
        }
        
        return audio_tensor
    
    def mix_scene(self, element_ids=None, volumes=None, output_path=None):
        """
        Mix multiple scene elements into a cohesive scene.
        
        Args:
            element_ids (list): List of element IDs to include (uses all if None)
            volumes (dict): Dictionary mapping element IDs to volume levels (0.0-1.0)
            output_path (str): Path to save the final scene audio
            
        Returns:
            torch.Tensor: Mixed scene audio tensor
        """
        # Use all elements if none specified
        if element_ids is None:
            element_ids = list(self.scene_elements.keys())
        
        # Verify elements exist
        for element_id in element_ids:
            if element_id not in self.scene_elements:
                raise ValueError(f"Element '{element_id}' not found. Generate it first.")
        
        # Default to equal volumes if not specified
        if volumes is None:
            volumes = {element_id: 1.0 for element_id in element_ids}
        
        # Ensure all elements have volume settings
        for element_id in element_ids:
            if element_id not in volumes:
                volumes[element_id] = 1.0
        
        print(f"Mixing scene with {len(element_ids)} elements:")
        for element_id in element_ids:
            print(f"- {element_id} (volume: {volumes[element_id]:.2f})")
        
        # Find the longest element
        max_length = max(self.scene_elements[element_id]["audio"].shape[0] for element_id in element_ids)
        
        # Create empty output array
        scene_audio = torch.zeros(max_length)
        
        # Mix elements
        for element_id in element_ids:
            element = self.scene_elements[element_id]
            audio = element["audio"]
            volume = volumes[element_id]
            
            # Apply volume and add to scene (up to element length)
            scene_audio[:audio.shape[0]] += audio * volume
        
        # Normalize if needed to prevent clipping
        if torch.max(torch.abs(scene_audio)) > 1.0:
            scene_audio = scene_audio / torch.max(torch.abs(scene_audio))
        
        # Save the scene if output path is provided
        if output_path:
            # Create directory if needed
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save the audio
            audio_write(
                output_path,
                scene_audio,
                self.sample_rate,
                strategy="loudness"
            )
            print(f"Scene saved to {output_path}.wav")
        
        return scene_audio
    
    def save_element(self, element_id, output_dir="scene_elements"):
        """
        Save a generated scene element to disk.
        
        Args:
            element_id (str): ID of the element to save
            output_dir (str): Directory to save the element
            
        Returns:
            str: Path to the saved file
        """
        if element_id not in self.scene_elements:
            raise ValueError(f"Element '{element_id}' not found. Generate it first.")
        
        os.makedirs(output_dir, exist_ok=True)
        
        element = self.scene_elements[element_id]
        output_path = os.path.join(output_dir, element_id)
        
        # Save the audio
        audio_write(
            output_path,
            element["audio"],
            self.sample_rate,
            strategy="loudness"
        )
        
        # Save metadata
        with open(f"{output_path}.json", 'w') as f:
            metadata = {
                "element_id": element_id,
                "prompt": element["prompt"],
                "model_type": element["model_type"],
                "duration": element["duration"],
                "temperature": element["temperature"],
            }
            json.dump(metadata, f, indent=2)
        
        print(f"Saved element '{element_id}' to {output_path}.wav")
        return f"{output_path}.wav"
    
    def save_all_elements(self, output_dir="scene_elements"):
        """Save all generated scene elements to disk."""
        paths = {}
        for element_id in self.scene_elements:
            path = self.save_element(element_id, output_dir)
            paths[element_id] = path
        return paths
    
    def create_scene_from_template(self, template, parameters=None, output_dir=None):
        """
        Create a complete scene based on a scene template.
        
        Args:
            template (dict): Scene template with elements, mixing rules, etc.
            parameters (dict): Parameters to customize the scene
            output_dir (str): Directory to save scene files
            
        Returns:
            dict: Information about the generated scene
        """
        # Default parameter values if not specified
        if parameters is None:
            parameters = {}
        
        # Get scene name and create output directory
        scene_name = template.get("name", "unnamed_scene")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_dir is None:
            output_dir = f"scene_{scene_name.replace(' ', '_').lower()}_{timestamp}"
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "elements"), exist_ok=True)
        
        print(f"Creating scene: {scene_name}")
        
        # Process each element in the template
        elements = template.get("elements", {})
        volumes = template.get("volumes", {})
        generated_elements = []
        
        for element_id, element_config in elements.items():
            # Get base configuration
            model_type = element_config.get("model_type", "audio")
            duration = element_config.get("duration", 10.0)
            temperature = element_config.get("temperature", 1.0)
            
            # Format the prompt with parameters
            prompt_template = element_config.get("prompt", "")
            try:
                # Only format if parameters are provided in the prompt template
                if "{" in prompt_template and "}" in prompt_template:
                    prompt = prompt_template.format(**parameters)
                else:
                    prompt = prompt_template
            except KeyError as e:
                # Handle missing parameters
                print(f"Warning: Missing parameter {e} for element '{element_id}'")
                prompt = prompt_template
            
            # Generate the element
            self.generate_scene_element(
                element_id=element_id,
                prompt=prompt,
                model_type=model_type,
                duration=duration,
                temperature=temperature
            )
            
            # Save individual element
            self.save_element(element_id, os.path.join(output_dir, "elements"))
            generated_elements.append(element_id)
        
        # Mix and save the full scene
        scene_path = os.path.join(output_dir, f"{scene_name.replace(' ', '_').lower()}")
        self.mix_scene(
            element_ids=generated_elements,
            volumes=volumes,
            output_path=scene_path
        )
        
        # Save scene configuration
        scene_config = {
            "name": scene_name,
            "template": template,
            "parameters": parameters,
            "elements": {eid: self.scene_elements[eid]["prompt"] for eid in generated_elements},
            "volumes": volumes,
            "timestamp": timestamp,
            "output_file": f"{scene_path}.wav"
        }
        
        with open(os.path.join(output_dir, "scene_config.json"), 'w') as f:
            json.dump(scene_config, f, indent=2)
        
        # Create README
        with open(os.path.join(output_dir, "README.txt"), 'w') as f:
            f.write(f"AUDIO SCENE: {scene_name}\n")
            f.write("=" * (len(scene_name) + 13) + "\n\n")
            f.write("Scene elements:\n\n")
            
            for element_id in generated_elements:
                element = self.scene_elements[element_id]
                f.write(f"{element_id}:\n")
                f.write(f"  Prompt: \"{element['prompt']}\"\n")
                f.write(f"  Model: {element['model_type']}\n")
                f.write(f"  Duration: {element['duration']}s\n")
                f.write(f"  Temperature: {element['temperature']}\n")
                f.write(f"  Volume in mix: {volumes.get(element_id, 1.0):.2f}\n\n")
            
            f.write(f"Complete scene file: {os.path.basename(scene_path)}.wav\n")
        
        print(f"Scene '{scene_name}' created successfully!")
        print(f"Scene files saved to {output_dir}")
        
        return scene_config
```

This scene composer provides a flexible foundation for creating multi-layered audio scenes. It handles:

1. Generating individual scene elements with appropriate models
2. Mixing elements with customizable volume levels
3. Saving both individual elements and complete scenes
4. Creating scenes from templates with parameters

## Scene Templates for Different Environments

Different environments require different approaches to scene composition. Let's create a library of scene templates:

```python
# scene_template_library.py

# Natural Environment Templates
natural_templates = {
    "forest": {
        "name": "Forest Environment",
        "elements": {
            "ambient_background": {
                "model_type": "audio",
                "prompt": "Continuous forest ambient background with light wind through leaves and distant bird calls",
                "duration": 15.0,
                "temperature": 0.7
            },
            "weather": {
                "model_type": "audio",
                "prompt": "{weather} conditions in a forest with appropriate sounds and intensity",
                "duration": 15.0,
                "temperature": 0.9
            },
            "wildlife": {
                "model_type": "audio",
                "prompt": "Forest wildlife sounds with {wildlife_type} being active and occasional movement in underbrush",
                "duration": 15.0,
                "temperature": 1.0
            },
            "music": {
                "model_type": "music",
                "prompt": "Subtle {mood} ambient music inspired by forest environments, with soft {instrument} and natural tones",
                "duration": 15.0,
                "temperature": 0.8
            }
        },
        "volumes": {
            "ambient_background": 0.7,
            "weather": 0.6,
            "wildlife": 0.8,
            "music": 0.4
        }
    },
    
    "ocean": {
        "name": "Ocean Environment",
        "elements": {
            "waves": {
                "model_type": "audio",
                "prompt": "{wave_intensity} ocean waves breaking on {shore_type} shore with natural water movement and rhythm",
                "duration": 15.0,
                "temperature": 0.7
            },
            "wind": {
                "model_type": "audio",
                "prompt": "{wind_intensity} sea breeze with appropriate air movement and subtle effects on the environment",
                "duration": 15.0,
                "temperature": 0.8
            },
            "wildlife": {
                "model_type": "audio",
                "prompt": "Coastal wildlife with {seabird_type} calls, movement, and occasional water interaction",
                "duration": 15.0,
                "temperature": 1.0
            },
            "music": {
                "model_type": "music",
                "prompt": "Peaceful {mood} ambient music inspired by ocean sounds, with flowing {instrument} and maritime qualities",
                "duration": 15.0,
                "temperature": 0.8
            }
        },
        "volumes": {
            "waves": 0.8,
            "wind": 0.6,
            "wildlife": 0.5,
            "music": 0.35
        }
    }
}

# Urban Environment Templates
urban_templates = {
    "city_street": {
        "name": "City Street",
        "elements": {
            "traffic": {
                "model_type": "audio",
                "prompt": "{traffic_intensity} city traffic with vehicle movement, engines, and occasional horns in a {city_type} setting",
                "duration": 15.0,
                "temperature": 0.8
            },
            "people": {
                "model_type": "audio",
                "prompt": "{crowd_density} pedestrian activity with footsteps, conversations, and movement on city streets",
                "duration": 15.0,
                "temperature": 0.9
            },
            "urban_elements": {
                "model_type": "audio",
                "prompt": "Urban environment sounds like construction, shops, and city infrastructure during {time_of_day}",
                "duration": 15.0,
                "temperature": 1.0
            },
            "music": {
                "model_type": "music",
                "prompt": "Urban {mood} background music with {instrument} that complements a city environment",
                "duration": 15.0,
                "temperature": 0.8
            }
        },
        "volumes": {
            "traffic": 0.7,
            "people": 0.6,
            "urban_elements": 0.5,
            "music": 0.3
        }
    },
    
    "cafe": {
        "name": "Cafe Environment",
        "elements": {
            "ambient_background": {
                "model_type": "audio",
                "prompt": "Interior cafe ambience with {crowd_level} customer activity and general restaurant sounds",
                "duration": 15.0,
                "temperature": 0.7
            },
            "service_sounds": {
                "model_type": "audio",
                "prompt": "Cafe service sounds with coffee machines, dishware, and staff activity in a {cafe_type} establishment",
                "duration": 15.0,
                "temperature": 0.9
            },
            "music": {
                "model_type": "music",
                "prompt": "{genre} music playing through cafe speakers with appropriate volume and acoustics for a cafe environment",
                "duration": 15.0,
                "temperature": 0.7
            },
            "outside_ambience": {
                "model_type": "audio",
                "prompt": "Muffled outside sounds entering a cafe, including {outside_elements} with subtle indoor filtering",
                "duration": 15.0,
                "temperature": 0.8
            }
        },
        "volumes": {
            "ambient_background": 0.8,
            "service_sounds": 0.6,
            "music": 0.5,
            "outside_ambience": 0.3
        }
    }
}

# Fantasy/Sci-Fi Environment Templates
fictional_templates = {
    "spaceship": {
        "name": "Spaceship Environment",
        "elements": {
            "engine_core": {
                "model_type": "audio",
                "prompt": "Spaceship {engine_type} engine core with mechanical and energy hum at {power_level} power",
                "duration": 15.0,
                "temperature": 0.7
            },
            "systems": {
                "model_type": "audio",
                "prompt": "Spacecraft systems with computers, life support, and {tech_level} technology operating in background",
                "duration": 15.0,
                "temperature": 0.8
            },
            "crew_activity": {
                "model_type": "audio",
                "prompt": "{activity_level} crew movement and operations throughout the spacecraft with appropriate sounds",
                "duration": 15.0,
                "temperature": 0.9
            },
            "music": {
                "model_type": "music",
                "prompt": "Sci-fi ambient {mood} music with {instrument} creating a {atmosphere_type} spaceship atmosphere",
                "duration": 15.0,
                "temperature": 0.8
            }
        },
        "volumes": {
            "engine_core": 0.6,
            "systems": 0.7,
            "crew_activity": 0.5,
            "music": 0.4
        }
    },
    
    "magical_realm": {
        "name": "Magical Realm",
        "elements": {
            "ambient_magic": {
                "model_type": "audio",
                "prompt": "Ambient magical environment with {magic_type} energy, mystical resonances, and enchanted atmosphere",
                "duration": 15.0,
                "temperature": 0.8
            },
            "creatures": {
                "model_type": "audio",
                "prompt": "Magical creatures with {creature_type} sounds, movements, and activities in an enchanted realm",
                "duration": 15.0,
                "temperature": 1.0
            },
            "environment": {
                "model_type": "audio",
                "prompt": "Fantasy {environment_type} with magical plants, structures, and phenomena creating unique sounds",
                "duration": 15.0,
                "temperature": 0.9
            },
            "music": {
                "model_type": "music",
                "prompt": "Ethereal {mood} fantasy music with {instrument} and magical qualities appropriate for an enchanted realm",
                "duration": 15.0,
                "temperature": 0.8
            }
        },
        "volumes": {
            "ambient_magic": 0.7,
            "creatures": 0.6,
            "environment": 0.8,
            "music": 0.5
        }
    }
}

# Combined template library
scene_templates = {
    **natural_templates,
    **urban_templates,
    **fictional_templates
}
```

These templates provide structured frameworks for different environment types, with:
1. Appropriate element types for each environment
2. Balanced volume mixing recommendations
3. Parameter placeholders for customization
4. Duration and temperature settings optimized for each element type

## Implementing a Dynamic Audio Environment System

Let's create a dynamic audio environment system that can respond to parameter changes in real-time:

```python
# dynamic_audio_environment.py
import torch
import os
import json
import time
from datetime import datetime
from audio_scene_composer import AudioSceneComposer
from scene_template_library import scene_templates

class DynamicAudioEnvironment:
    """
    System for creating parametric audio environments that can
    transition between different states dynamically.
    """
    
    def __init__(self):
        """Initialize the dynamic audio environment system."""
        self.composer = AudioSceneComposer()
        
        # Available parameter options
        self.parameter_options = {
            # Weather parameters
            "weather": ["clear", "light_rain", "heavy_rain", "storm", "windy", "foggy", "snowy"],
            "wind_intensity": ["none", "light", "moderate", "strong", "gale"],
            "precipitation": ["none", "drizzle", "rain", "downpour", "hail", "snow", "sleet"],
            
            # Time parameters
            "time_of_day": ["dawn", "morning", "midday", "afternoon", "evening", "night", "midnight"],
            "daylight": ["bright", "overcast", "dim", "dark"],
            
            # Nature parameters
            "wildlife_type": ["birds", "insects", "mammals", "reptiles", "mixed"],
            "wildlife_activity": ["quiet", "sparse", "moderate", "active", "intense"],
            "vegetation": ["sparse", "moderate", "dense", "lush"],
            
            # Urban parameters
            "traffic_intensity": ["none", "light", "moderate", "heavy", "congested"],
            "crowd_density": ["empty", "sparse", "moderate", "busy", "crowded"],
            "city_type": ["small_town", "suburb", "downtown", "metropolis", "industrial"],
            
            # Mood parameters
            "mood": ["peaceful", "tense", "mysterious", "cheerful", "melancholic", "epic", "intimate"],
            "atmosphere_type": ["calm", "threatening", "magical", "technological", "ancient", "futuristic"],
            
            # Music parameters
            "genre": ["ambient", "classical", "electronic", "jazz", "folk"],
            "instrument": ["piano", "strings", "synth", "guitar", "woodwinds", "percussion"],
            
            # Sci-fi parameters
            "tech_level": ["primitive", "contemporary", "advanced", "futuristic", "alien"],
            "power_level": ["low", "standard", "high", "critical", "overload"],
            "activity_level": ["abandoned", "minimal", "normal", "busy", "emergency"]
        }
        
        # Current environment state
        self.current_template = None
        self.current_parameters = {}
        self.current_scene_info = None
        self.last_generation_time = None
    
    def initialize_environment(self, template_name, parameters=None, output_dir=None):
        """
        Initialize an audio environment using a template and parameters.
        
        Args:
            template_name (str): Name of the template to use
            parameters (dict): Custom parameters for the environment
            output_dir (str): Directory to save environment files
            
        Returns:
            dict: Information about the generated environment
        """
        # Verify template exists
        if template_name not in scene_templates:
            available = list(scene_templates.keys())
            raise ValueError(f"Unknown template: {template_name}. Available: {available}")
        
        # Get the template
        template = scene_templates[template_name]
        self.current_template = template
        
        # Initialize default parameters if none provided
        if parameters is None:
            parameters = {}
        
        # Fill in missing parameters with defaults
        self._fill_default_parameters(parameters)
        self.current_parameters = parameters
        
        # Create output directory with timestamp if not provided
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"environment_{template_name}_{timestamp}"
        
        # Generate the initial environment
        print(f"Initializing {template_name} environment with parameters:")
        for key, value in parameters.items():
            print(f"- {key}: {value}")
        
        start_time = time.time()
        self.current_scene_info = self.composer.create_scene_from_template(
            template=template,
            parameters=parameters,
            output_dir=output_dir
        )
        self.last_generation_time = time.time() - start_time
        
        return self.current_scene_info
    
    def update_environment(self, parameters, output_dir=None):
        """
        Update an existing environment with new parameters.
        
        Args:
            parameters (dict): New or updated parameters
            output_dir (str): Directory to save updated environment
            
        Returns:
            dict: Information about the updated environment
        """
        if self.current_template is None:
            raise ValueError("No environment initialized. Call initialize_environment first.")
        
        # Update parameters
        updated_parameters = self.current_parameters.copy()
        updated_parameters.update(parameters)
        self.current_parameters = updated_parameters
        
        # Create output directory with timestamp if not provided
        if output_dir is None:
            template_name = self.current_template.get("name", "environment").replace(" ", "_").lower()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"{template_name}_update_{timestamp}"
        
        # Generate the updated environment
        print(f"Updating environment with new parameters:")
        for key, value in parameters.items():
            print(f"- {key}: {value}")
        
        start_time = time.time()
        self.current_scene_info = self.composer.create_scene_from_template(
            template=self.current_template,
            parameters=updated_parameters,
            output_dir=output_dir
        )
        self.last_generation_time = time.time() - start_time
        
        return self.current_scene_info
    
    def transition_to_template(self, template_name, target_parameters=None, crossfade_duration=5.0, output_dir=None):
        """
        Create a transition between the current environment and a new template.
        
        This creates a crossfade between the current environment and a new one,
        useful for smooth scene changes.
        
        Args:
            template_name (str): Name of the template to transition to
            target_parameters (dict): Parameters for the target environment
            crossfade_duration (float): Duration of crossfade in seconds
            output_dir (str): Directory to save transition files
            
        Returns:
            dict: Information about the transition
        """
        if self.current_template is None:
            # If no current environment, just initialize the new one
            return self.initialize_environment(template_name, target_parameters, output_dir)
        
        # Verify template exists
        if template_name not in scene_templates:
            available = list(scene_templates.keys())
            raise ValueError(f"Unknown template: {template_name}. Available: {available}")
        
        # Get the target template
        target_template = scene_templates[template_name]
        
        # Initialize target parameters if none provided
        if target_parameters is None:
            target_parameters = {}
        
        # Fill in missing parameters with defaults
        target_parameters_filled = self.current_parameters.copy()  # Start with current parameters
        self._fill_default_parameters(target_parameters_filled, template_name)
        target_parameters_filled.update(target_parameters)  # Override with provided parameters
        
        # Create output directory with timestamp if not provided
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"transition_{self.current_template.get('name', 'source').replace(' ', '_')}_{target_template.get('name', 'target').replace(' ', '_')}_{timestamp}"
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate the target environment
        print(f"Generating target environment ({target_template.get('name')}) for transition...")
        target_composer = AudioSceneComposer()  # Create a new composer for the target environment
        
        # Generate the target scene
        target_output_dir = os.path.join(output_dir, "target_environment")
        target_scene_info = target_composer.create_scene_from_template(
            template=target_template,
            parameters=target_parameters_filled,
            output_dir=target_output_dir
        )
        
        # Create the transition (crossfade)
        print(f"Creating {crossfade_duration}s crossfade transition...")
        
        # Load source and target audio
        source_audio, sr = torchaudio.load(self.current_scene_info['output_file'])
        target_audio, sr = torchaudio.load(target_scene_info['output_file'])
        
        # Make sure both are mono for simplicity
        if source_audio.shape[0] > 1:
            source_audio = torch.mean(source_audio, dim=0, keepdim=True)
        if target_audio.shape[0] > 1:
            target_audio = torch.mean(target_audio, dim=0, keepdim=True)
        
        # Calculate crossfade points
        crossfade_samples = int(crossfade_duration * sr)
        total_length = source_audio.shape[1] + target_audio.shape[1] - crossfade_samples
        
        # Create output buffer
        transition_audio = torch.zeros(1, total_length)
        
        # Copy source audio to the beginning
        transition_audio[0, :source_audio.shape[1]] += source_audio[0]
        
        # Create linear crossfade weights
        fade_in = torch.linspace(0, 1, crossfade_samples)
        fade_out = torch.linspace(1, 0, crossfade_samples)
        
        # Apply crossfade
        crossfade_start = source_audio.shape[1] - crossfade_samples
        for i in range(crossfade_samples):
            pos = crossfade_start + i
            transition_audio[0, pos] = source_audio[0, pos] * fade_out[i] + target_audio[0, i] * fade_in[i]
        
        # Copy remaining target audio
        remaining_start = source_audio.shape[1]
        remaining_target_start = crossfade_samples
        remaining_length = target_audio.shape[1] - crossfade_samples
        
        transition_audio[0, remaining_start:] += target_audio[0, remaining_target_start:]
        
        # Save the transition
        transition_path = os.path.join(output_dir, "environment_transition")
        torchaudio.save(
            f"{transition_path}.wav",
            transition_audio,
            sr
        )
        
        print(f"Transition created and saved to {transition_path}.wav")
        
        # Save transition info
        transition_info = {
            'source_template': self.current_template.get('name'),
            'target_template': target_template.get('name'),
            'source_parameters': self.current_parameters,
            'target_parameters': target_parameters_filled,
            'crossfade_duration': crossfade_duration,
            'output_file': f"{transition_path}.wav"
        }
        
        with open(os.path.join(output_dir, "transition_info.json"), 'w') as f:
            json.dump(transition_info, f, indent=2)
        
        # Update current state
        self.current_template = target_template
        self.current_parameters = target_parameters_filled
        self.current_scene_info = target_scene_info
        
        return transition_info
    
    def _fill_default_parameters(self, parameters, template_name=None):
        """
        Fill in default parameter values for missing parameters.
        
        This first checks what parameters are needed by the template,
        then fills in defaults for any missing ones.
        """
        template = self.current_template
        if template_name is not None:
            template = scene_templates.get(template_name)
            if template is None:
                return
        
        if template is None:
            return
        
        # Identify needed parameters from template prompts
        needed_params = set()
        elements = template.get("elements", {})
        
        for element_config in elements.values():
            prompt_template = element_config.get("prompt", "")
            
            # Find all parameters in the template (between { and })
            import re
            params_in_template = re.findall(r'\{([^}]+)\}', prompt_template)
            needed_params.update(params_in_template)
        
        # Fill in defaults for missing parameters
        for param in needed_params:
            if param not in parameters:
                # If we have options for this parameter, use the first one as default
                if param in self.parameter_options:
                    parameters[param] = self.parameter_options[param][0]
                else:
                    # Otherwise use a placeholder
                    parameters[param] = "default"

# Example usage
if __name__ == "__main__":
    # Create the dynamic environment system
    env = DynamicAudioEnvironment()
    
    # Initialize a forest environment
    env.initialize_environment(
        template_name="forest",
        parameters={
            "weather": "clear",
            "wildlife_type": "birds",
            "mood": "peaceful",
            "instrument": "piano"
        },
        output_dir="forest_morning"
    )
    
    # Update to a rainy afternoon
    env.update_environment(
        parameters={
            "weather": "light_rain",
            "time_of_day": "afternoon",
            "mood": "melancholic"
        },
        output_dir="forest_rainy_afternoon"
    )
    
    # Transition to a city environment in the evening
    env.transition_to_template(
        template_name="city_street",
        target_parameters={
            "time_of_day": "evening",
            "traffic_intensity": "moderate",
            "crowd_density": "busy",
            "mood": "mysterious"
        },
        crossfade_duration=3.0,
        output_dir="forest_to_city_transition"
    )
```

This dynamic environment system provides powerful capabilities:

1. **Parameter-Driven Generation**: Create environments by specifying high-level parameters
2. **Dynamic Updates**: Modify environments by changing parameter values
3. **Smooth Transitions**: Create crossfades between different environment types
4. **Consistent State**: Maintain environmental context across generations

## Spatial Audio Scene Creation

For immersive applications like VR and games, spatial audio significantly increases realism. Let's implement basic spatial audio capabilities:

```python
# spatial_audio_scene.py
import torch
import numpy as np
import os
from audio_scene_composer import AudioSceneComposer

class SpatialAudioScene:
    """
    Create immersive 3D audio scenes with spatial positioning of elements.
    
    This class extends the AudioSceneComposer to add spatial positioning
    capabilities, allowing elements to be placed in 3D space and simulating
    distance-based attenuation and basic directional audio.
    """
    
    def __init__(self):
        """Initialize the spatial audio scene creator."""
        self.composer = AudioSceneComposer()
        
        # Track spatial properties of elements
        self.element_positions = {}  # {element_id: [x, y, z]}
        self.listener_position = [0, 0, 0]  # Default listener at center
    
    def add_element(self, element_id, prompt, model_type, position, duration=5.0, temperature=1.0):
        """
        Add an audio element with spatial position.
        
        Args:
            element_id (str): Identifier for this element
            prompt (str): Description of the sound to generate
            model_type (str): Type of model to use ("music" or "audio")
            position (list): [x, y, z] position in 3D space
            duration (float): Duration in seconds
            temperature (float): Generation temperature
            
        Returns:
            torch.Tensor: Generated audio tensor
        """
        # Generate the audio using the composer
        audio = self.composer.generate_scene_element(
            element_id=element_id,
            prompt=prompt,
            model_type=model_type,
            duration=duration,
            temperature=temperature
        )
        
        # Store the position
        self.element_positions[element_id] = position
        
        return audio
    
    def set_listener_position(self, position):
        """Set the listener's position in 3D space."""
        self.listener_position = position
    
    def render_spatial_scene(self, output_path=None):
        """
        Render the scene with spatial audio processing.
        
        This applies basic distance-based attenuation and stereo panning
        to create a simple spatial audio experience.
        
        Args:
            output_path (str): Path to save the spatially-rendered scene
            
        Returns:
            torch.Tensor: Stereo (2-channel) audio tensor with spatial processing
        """
        if not self.composer.scene_elements:
            raise ValueError("No elements in scene. Add elements first.")
        
        # Get all element IDs
        element_ids = list(self.composer.scene_elements.keys())
        
        # Find the longest element
        max_length = max(self.composer.scene_elements[element_id]["audio"].shape[0] for element_id in element_ids)
        
        # Create stereo output buffer
        stereo_scene = torch.zeros(2, max_length)
        
        # Process each element with spatial properties
        for element_id in element_ids:
            if element_id not in self.element_positions:
                print(f"Warning: No position for element '{element_id}', using default center position")
                position = [0, 0, 0]
            else:
                position = self.element_positions[element_id]
            
            # Get original mono audio
            mono_audio = self.composer.scene_elements[element_id]["audio"]
            
            # Calculate distance from listener
            dx = position[0] - self.listener_position[0]
            dy = position[1] - self.listener_position[1]
            dz = position[2] - self.listener_position[2]
            distance = np.sqrt(dx**2 + dy**2 + dz**2)
            
            # Distance attenuation (inverse square law, clamped for safety)
            min_distance = 0.1  # Prevent division by zero
            max_distance = 100.0  # Maximum effective distance
            distance = max(min_distance, min(distance, max_distance))
            
            # Calculate gain using inverse square law with min gain
            min_gain = 0.01  # Minimum gain (-40dB)
            gain = min(1.0, 1.0 / (distance * distance))
            gain = max(min_gain, gain)
            
            # Calculate stereo panning based on azimuth angle
            # This is a simplified model - proper HRTF would be more realistic
            azimuth = np.arctan2(dy, dx)  # Azimuth angle in radians
            
            # Convert azimuth to pan value (-1 to 1, where -1 is full left, 1 is full right)
            pan = np.sin(azimuth)
            
            # Apply distance gain and panning
            left_gain = gain * (1.0 - max(0, pan))
            right_gain = gain * (1.0 + min(0, pan))
            
            # Ensure we don't exceed the buffer length
            audio_length = mono_audio.shape[0]
            
            # Add to stereo output
            stereo_scene[0, :audio_length] += mono_audio * left_gain
            stereo_scene[1, :audio_length] += mono_audio * right_gain
        
        # Normalize if needed to prevent clipping
        if torch.max(torch.abs(stereo_scene)) > 1.0:
            stereo_scene = stereo_scene / torch.max(torch.abs(stereo_scene))
        
        # Save the scene if output path is provided
        if output_path:
            # Create directory if needed
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save the audio
            import torchaudio
            torchaudio.save(
                f"{output_path}.wav",
                stereo_scene,
                self.composer.sample_rate
            )
            print(f"Spatial scene saved to {output_path}.wav")
        
        return stereo_scene
    
    def create_ambisonic_scene(self, order=1, output_path=None):
        """
        Create a first-order Ambisonic (B-format) representation of the scene.
        
        Ambisonics is a full-sphere surround sound format that can be decoded
        to various speaker layouts or binauralized for headphones.
        
        Args:
            order (int): Ambisonic order (1 = first-order B-format)
            output_path (str): Path to save the Ambisonic scene
            
        Returns:
            torch.Tensor: Ambisonic audio tensor (4 channels for first-order)
        """
        if order != 1:
            raise ValueError("Currently only first-order Ambisonics (order=1) is supported")
        
        if not self.composer.scene_elements:
            raise ValueError("No elements in scene. Add elements first.")
        
        # Get all element IDs
        element_ids = list(self.composer.scene_elements.keys())
        
        # Find the longest element
        max_length = max(self.composer.scene_elements[element_id]["audio"].shape[0] for element_id in element_ids)
        
        # Create B-format output buffer (4 channels: W, X, Y, Z)
        b_format = torch.zeros(4, max_length)
        
        # Process each element with Ambisonic encoding
        for element_id in element_ids:
            if element_id not in self.element_positions:
                print(f"Warning: No position for element '{element_id}', using default center position")
                position = [0, 0, 0]
            else:
                position = self.element_positions[element_id]
            
            # Get original mono audio
            mono_audio = self.composer.scene_elements[element_id]["audio"]
            
            # Calculate direction vector relative to listener
            dx = position[0] - self.listener_position[0]
            dy = position[1] - self.listener_position[1]
            dz = position[2] - self.listener_position[2]
            
            # Calculate distance
            distance = np.sqrt(dx**2 + dy**2 + dz**2)
            
            # Distance attenuation (inverse square law, clamped for safety)
            min_distance = 0.1  # Prevent division by zero
            max_distance = 100.0  # Maximum effective distance
            distance = max(min_distance, min(distance, max_distance))
            
            # Calculate gain using inverse square law with min gain
            min_gain = 0.01  # Minimum gain (-40dB)
            gain = min(1.0, 1.0 / (distance * distance))
            gain = max(min_gain, gain)
            
            # Normalize direction vector
            if distance > 0:
                dx /= distance
                dy /= distance
                dz /= distance
            else:
                dx, dy, dz = 0, 0, 0
            
            # First-order Ambisonic encoding
            # W = omni-directional component
            # X = front-back component
            # Y = left-right component
            # Z = up-down component
            
            # Apply encoding and distance gain
            audio_length = mono_audio.shape[0]
            
            # W = mono signal (omni-directional, scaled by 0.707)
            b_format[0, :audio_length] += mono_audio * gain * 0.707
            
            # X = front-back (X direction)
            b_format[1, :audio_length] += mono_audio * gain * dx
            
            # Y = left-right (Y direction)
            b_format[2, :audio_length] += mono_audio * gain * dy
            
            # Z = up-down (Z direction)
            b_format[3, :audio_length] += mono_audio * gain * dz
        
        # Save the Ambisonic scene if output path is provided
        if output_path:
            # Create directory if needed
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save the audio
            import torchaudio
            torchaudio.save(
                f"{output_path}.wav",
                b_format,
                self.composer.sample_rate
            )
            print(f"Ambisonic scene saved to {output_path}.wav")
        
        return b_format
```

This spatial audio implementation offers several key capabilities:

1. **3D Positioning**: Place audio elements anywhere in 3D space
2. **Distance Attenuation**: Simulate distance-based volume falloff
3. **Stereo Panning**: Create directional audio for headphone listening
4. **Ambisonic Support**: Generate first-order Ambisonic audio for VR/3D applications

## Hands-on Challenge: Interactive Weather System

Now it's time to apply everything you've learned to create an interactive weather system for a game or VR experience:

1. Create a base environment (forest, ocean, city, etc.) with the audio scene composer
2. Implement at least 3 different weather conditions (clear, rain, storm, snow, etc.)
3. Design a transition system that can smoothly blend between different weather states
4. Make your system parameter-driven, allowing control over intensity, time of day, and mood
5. Export your environments in a format suitable for integration into a game or VR project

**Bonus Challenge**: Create an interactive demo that allows users to control weather parameters in real-time, hearing the audio environment change dynamically.

## Key Takeaways

- Complete audio scenes require multiple layered elements for realism and immersion
- Scene templates provide structured frameworks for different environment types
- Dynamic parameters allow environments to respond to changing conditions
- Proper mixing and balancing are essential for natural-sounding scenes
- Spatial audio capabilities significantly increase immersion for 3D applications
- Transitions between different audio states require careful blending techniques

## Next Steps

Now that you've mastered audio scene composition, you're ready to explore more advanced production workflows:

- **Chapter 13: Sound Design Workflows** - Develop efficient pipelines for audio production
- **Chapter 14: Integration with Unity and Unreal** - Implement AI audio in game engines
- **Chapter 15: Interactive Audio Systems** - Create responsive audio that reacts to user input

## Further Reading

- [The Art of Mixing: A Visual Guide to Recording](https://www.amazon.com/Art-Mixing-Visual-Recording-Engineering/dp/1285424867)
- [Game Audio Implementation](https://www.taylorfrancis.com/books/mono/10.4324/9781315707525/game-audio-implementation-richard-stevens-dave-raybould)
- [Designing Sound for Animation](https://www.routledge.com/Designing-Sound-for-Animation/Beauchamp/p/book/9780240824987)
- [Spatial Audio with Unity and Wwise](https://blog.audiokinetic.com/en/spatial-audio-in-unity-with-wwise/)
- [Foundations of Ambisonic Audio](https://www.cambridge.org/core/books/foundations-of-ambisonics/4E3C64F3D2C9D35B4A4ABFCC0DA6ED04)
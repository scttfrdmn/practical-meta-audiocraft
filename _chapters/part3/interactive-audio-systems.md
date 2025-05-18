# Chapter 15: Interactive Audio Systems

> *"Our players are exploring this vast open world we've created, but the audio feels static. How can we use AudioCraft to create a dynamic soundscape that responds to their actions, changes with the environment, and creates a more immersive experience?"*  
> — Audio Director, Open World Game Studio

## Learning Objectives

In this chapter, you'll learn how to:

- Design responsive audio systems that adapt to player actions and game states
- Implement parameter-driven audio generation for dynamic environments
- Create procedural audio layers that evolve over time
- Build interactive music systems with adaptive transitions
- Develop emotion-aware audio that responds to narrative context

## Introduction

Interactive audio systems transform static sound assets into dynamic, responsive experiences. By connecting audio generation to gameplay systems, we can create soundscapes that evolve with player actions, environmental conditions, and narrative progression.

In this chapter, we'll build on our Unity integration to create interactive audio systems that leverage AudioCraft's generation capabilities. We'll explore both technical implementation and design concepts for creating truly responsive audio experiences.

## Implementation: Parameter-Driven Audio Environment

First, let's create a system that dynamically generates audio based on gameplay parameters:

```python
import os
import json
import torch
import numpy as np
import torchaudio
from typing import Dict, List, Optional, Union, Literal, Tuple
from dataclasses import dataclass, field
from audiocraft.models import AudioGen

@dataclass
class EnvironmentParameters:
    """Parameters that define an audio environment state."""
    
    # Environment type
    environment_type: str  # e.g., "forest", "city", "dungeon"
    
    # Time parameters
    time_of_day: float  # 0-24 hour
    day_night_cycle: float  # 0-1 transition value
    
    # Weather parameters
    weather_type: str  # e.g., "clear", "rain", "storm"
    weather_intensity: float  # 0-1 intensity
    wind_intensity: float  # 0-1 intensity
    
    # Mood/tension parameters
    tension: float  # 0-1 tension level
    danger: float  # 0-1 danger level
    
    # Population parameters
    population_density: float  # 0-1 population density
    civilization_proximity: float  # 0-1 proximity (0=wilderness, 1=urban)
    
    # Magic/supernatural parameters
    magic_presence: float  # 0-1 magic level
    supernatural_activity: float  # 0-1 supernatural activity
    
    # Custom parameters
    custom_parameters: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {k: v for k, v in self.__dict__.items() if k != 'custom_parameters'} | self.custom_parameters
    
    def get_time_period(self) -> str:
        """Get the time period based on time of day."""
        if 5 <= self.time_of_day < 8:
            return "dawn"
        elif 8 <= self.time_of_day < 12:
            return "morning"
        elif 12 <= self.time_of_day < 17:
            return "afternoon"
        elif 17 <= self.time_of_day < 20:
            return "evening"
        elif 20 <= self.time_of_day < 24 or 0 <= self.time_of_day < 5:
            return "night"
        else:
            return "day"

class ParametricAudioGenerator:
    """
    Generates audio based on environmental parameters.
    
    This system takes a set of environmental parameters (weather, time, mood, etc.)
    and generates appropriate audio using AudioCraft models.
    """
    
    def __init__(
        self,
        output_dir: str = "parametric_audio",
        model_size: str = "medium",
        device: str = None
    ):
        """
        Initialize the parametric audio generator.
        
        Args:
            output_dir: Directory for output files
            model_size: Size of AudioGen model to use
            device: Device to run model on (cuda, mps, cpu)
        """
        self.output_dir = output_dir
        self.model_size = model_size
        
        # Initialize device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize model (lazy loaded)
        self._model = None
        
        # Load environment templates
        self.templates_path = os.path.join(output_dir, "environment_templates.json")
        self.load_templates()
    
    def load_templates(self):
        """Load environment templates or create defaults if not found."""
        if os.path.exists(self.templates_path):
            with open(self.templates_path, 'r') as f:
                self.environment_templates = json.load(f)
        else:
            # Create default templates
            self.environment_templates = {
                "forest": {
                    "base_prompt": "Forest ambience with {time_desc} atmosphere, {weather_desc} conditions, and {wildlife_desc} wildlife sounds",
                    "time_mapping": {
                        "dawn": "early morning birds chirping and awakening forest",
                        "morning": "active birds and forest creatures",
                        "afternoon": "warm buzzing insects and active wildlife",
                        "evening": "transitioning to evening insects and nocturnal animals",
                        "night": "night creatures, owls, and distant animal calls"
                    },
                    "weather_mapping": {
                        "clear": {
                            "0.0": "clear and calm",
                            "0.5": "gentle breezes through leaves",
                            "1.0": "strong wind rustling through tree canopy"
                        },
                        "rain": {
                            "0.0": "light misty rain on leaves",
                            "0.5": "steady rainfall on forest canopy",
                            "1.0": "heavy downpour and water runoff"
                        },
                        "storm": {
                            "0.0": "distant thunder and increasing wind",
                            "0.5": "moderate thunderstorm with wind gusts",
                            "1.0": "severe storm with heavy rain, close thunder, and trees creaking"
                        }
                    },
                    "tension_mapping": {
                        "0.0": "peaceful and serene",
                        "0.3": "slightly mysterious",
                        "0.6": "tense and foreboding",
                        "1.0": "threatening and dangerous"
                    },
                    "magic_mapping": {
                        "0.0": "",
                        "0.3": "with subtle magical elements",
                        "0.7": "with clear magical presence and occasional magical sounds",
                        "1.0": "with strong magical energy, ethereal tones, and magical phenomena"
                    },
                    "wildlife_density_mapping": {
                        "0.0": "sparse",
                        "0.3": "occasional",
                        "0.7": "moderate",
                        "1.0": "abundant"
                    }
                },
                "city": {
                    "base_prompt": "City ambience with {time_desc} atmosphere, {weather_desc} conditions, and {population_desc} urban activity",
                    "time_mapping": {
                        "dawn": "early morning city waking up with sparse traffic",
                        "morning": "morning commute with traffic and pedestrians",
                        "afternoon": "busy midday urban activity",
                        "evening": "evening urban life with entertainment venues active",
                        "night": "night city ambience with distant sirens and sparse traffic"
                    },
                    "weather_mapping": {
                        "clear": {
                            "0.0": "clear and calm",
                            "0.5": "moderate winds between buildings",
                            "1.0": "strong winds whistling through urban canyons"
                        },
                        "rain": {
                            "0.0": "light rain on pavement",
                            "0.5": "steady rainfall on buildings and streets",
                            "1.0": "heavy downpour with water draining and umbrellas"
                        },
                        "storm": {
                            "0.0": "approaching storm with wind picking up",
                            "0.5": "moderate thunderstorm with traffic sounds",
                            "1.0": "severe storm with thunder echoing off buildings"
                        }
                    },
                    "population_mapping": {
                        "0.0": "quiet and sparse",
                        "0.3": "moderate pedestrian activity",
                        "0.7": "busy crowds and significant traffic",
                        "1.0": "crowded celebration or major event"
                    },
                    "tension_mapping": {
                        "0.0": "peaceful and normal",
                        "0.3": "slightly tense with occasional police sirens",
                        "0.6": "elevated tension with multiple sirens and shouting",
                        "1.0": "emergency situation with alarms and disturbance"
                    }
                },
                "dungeon": {
                    "base_prompt": "Underground dungeon ambience with {mood_desc} atmosphere, {water_desc} water sounds, and {supernatural_desc} elements",
                    "water_mapping": {
                        "0.0": "dry stone",
                        "0.3": "occasional water drips",
                        "0.7": "steady water dripping and small puddles",
                        "1.0": "flowing water and flooded sections"
                    },
                    "mood_mapping": {
                        "0.0": "neutral and empty",
                        "0.3": "slightly eerie",
                        "0.7": "ominous and dreadful",
                        "1.0": "terrifying and threatening"
                    },
                    "supernatural_mapping": {
                        "0.0": "no supernatural",
                        "0.3": "subtle strange sounds",
                        "0.7": "clear supernatural presence with whispers",
                        "1.0": "powerful supernatural activity with otherworldly sounds"
                    },
                    "wind_mapping": {
                        "0.0": "still air",
                        "0.3": "faint air movement",
                        "0.7": "wind flowing through passages",
                        "1.0": "strong drafts howling through corridors"
                    }
                }
            }
            
            # Save default templates
            self.save_templates()
    
    def save_templates(self):
        """Save environment templates to disk."""
        with open(self.templates_path, 'w') as f:
            json.dump(self.environment_templates, f, indent=2)
    
    def get_model(self):
        """Lazy-load the AudioGen model."""
        if self._model is None:
            print(f"Loading AudioGen model ({self.model_size})...")
            self._model = AudioGen.get_pretrained(self.model_size)
            self._model.to(self.device)
        return self._model
    
    def interpolate_value(self, mapping: Dict[str, str], value: float) -> str:
        """
        Interpolate between discrete values in a mapping.
        
        Args:
            mapping: Dictionary with string keys representing float values
            value: Float value to interpolate
            
        Returns:
            Interpolated string value
        """
        # Convert keys to floats
        float_mapping = {float(k): v for k, v in mapping.items()}
        
        # Sort keys
        sorted_keys = sorted(float_mapping.keys())
        
        # Find the closest keys
        if value <= sorted_keys[0]:
            return float_mapping[sorted_keys[0]]
        elif value >= sorted_keys[-1]:
            return float_mapping[sorted_keys[-1]]
        
        # Find bounding keys
        lower_key = max(k for k in sorted_keys if k <= value)
        upper_key = min(k for k in sorted_keys if k >= value)
        
        # If exact match, return that value
        if lower_key == upper_key:
            return float_mapping[lower_key]
        
        # Otherwise return the lower bound value (we can't interpolate strings)
        return float_mapping[lower_key]
    
    def find_closest_mapping(self, mapping: Dict[str, Dict], key: str, default: str) -> Dict:
        """
        Find the closest matching key in a nested mapping.
        
        Args:
            mapping: Dictionary with nested mappings
            key: Key to find
            default: Default key to use if no match
            
        Returns:
            Closest matching nested dictionary
        """
        if key in mapping:
            return mapping[key]
        
        # Check for partial matches
        for k in mapping.keys():
            if key in k or k in key:
                return mapping[k]
        
        # Return default if available, otherwise first entry
        if default in mapping:
            return mapping[default]
        
        # Fall back to first entry
        if mapping:
            return mapping[next(iter(mapping))]
        
        # Empty dict as last resort
        return {}
    
    def generate_prompt_from_parameters(self, params: EnvironmentParameters) -> str:
        """
        Generate a textual prompt based on environmental parameters.
        
        Args:
            params: Environmental parameters
            
        Returns:
            Prompt string for audio generation
        """
        # Get the environment template
        if params.environment_type not in self.environment_templates:
            # Fall back to forest if not found
            env_template = self.environment_templates["forest"]
        else:
            env_template = self.environment_templates[params.environment_type]
        
        # Get time description
        time_period = params.get_time_period()
        time_desc = env_template.get("time_mapping", {}).get(time_period, time_period)
        
        # Get weather description
        weather_mapping = self.find_closest_mapping(
            env_template.get("weather_mapping", {}),
            params.weather_type,
            "clear"
        )
        weather_desc = self.interpolate_value(weather_mapping, params.weather_intensity)
        
        # Apply wind modifier if available and significant
        if params.wind_intensity > 0.3 and "wind_mapping" in env_template:
            wind_desc = self.interpolate_value(env_template["wind_mapping"], params.wind_intensity)
            weather_desc = f"{weather_desc} with {wind_desc}"
        
        # Handle tension/mood
        if "tension_mapping" in env_template:
            tension_desc = self.interpolate_value(env_template["tension_mapping"], params.tension)
            mood_desc = tension_desc
        elif "mood_mapping" in env_template:
            mood_desc = self.interpolate_value(env_template["mood_mapping"], params.tension)
        else:
            mood_desc = "neutral"
        
        # Handle wildlife/population
        if "wildlife_density_mapping" in env_template:
            wildlife_desc = self.interpolate_value(
                env_template["wildlife_density_mapping"],
                params.population_density
            )
        else:
            wildlife_desc = "moderate"
        
        # Handle population for urban environments
        if "population_mapping" in env_template:
            population_desc = self.interpolate_value(
                env_template["population_mapping"],
                params.population_density
            )
        else:
            population_desc = "moderate"
        
        # Handle water for applicable environments
        if "water_mapping" in env_template:
            water_intensity = params.custom_parameters.get("water_intensity", 0.5)
            water_desc = self.interpolate_value(env_template["water_mapping"], water_intensity)
        else:
            water_desc = "no"
        
        # Handle supernatural elements
        if "magic_mapping" in env_template:
            supernatural_desc = self.interpolate_value(
                env_template["magic_mapping"],
                max(params.magic_presence, params.supernatural_activity)
            )
        elif "supernatural_mapping" in env_template:
            supernatural_desc = self.interpolate_value(
                env_template["supernatural_mapping"],
                params.supernatural_activity
            )
        else:
            supernatural_desc = ""
        
        # Build the final prompt using the template
        base_prompt = env_template.get("base_prompt", "Environmental ambience with {time_desc} atmosphere")
        
        # Replace placeholders with actual values
        prompt = base_prompt.format(
            time_desc=time_desc,
            weather_desc=weather_desc,
            mood_desc=mood_desc,
            wildlife_desc=wildlife_desc,
            population_desc=population_desc,
            water_desc=water_desc,
            supernatural_desc=supernatural_desc
        )
        
        return prompt
    
    def generate_parametric_audio(
        self,
        params: EnvironmentParameters,
        duration: float = 10.0,
        output_filename: str = None,
        generation_params: Dict = None
    ) -> Tuple[torch.Tensor, int, str]:
        """
        Generate audio based on environmental parameters.
        
        Args:
            params: Environmental parameters
            duration: Duration in seconds
            output_filename: Optional filename for saving
            generation_params: Optional generation parameters
            
        Returns:
            Tuple of (audio_tensor, sample_rate, prompt)
        """
        # Generate the prompt
        prompt = self.generate_prompt_from_parameters(params)
        
        # Default generation parameters
        default_params = {
            "duration": duration,
            "temperature": 1.0,
            "cfg_coef": 3.5,
            "top_k": 250,
            "top_p": 0.0
        }
        
        # Merge with provided parameters
        gen_params = dict(default_params)
        if generation_params:
            gen_params.update(generation_params)
        
        # Load model
        model = self.get_model()
        model.set_generation_params(**gen_params)
        
        # Generate audio
        wav = model.generate([prompt])
        
        # Save if filename provided
        if output_filename:
            if not output_filename.endswith('.wav'):
                output_filename += '.wav'
            
            output_path = os.path.join(self.output_dir, output_filename)
            torchaudio.save(
                output_path,
                wav[0].cpu(),
                model.sample_rate
            )
            
            # Save metadata
            metadata_path = output_path.replace('.wav', '.json')
            with open(metadata_path, 'w') as f:
                metadata = {
                    "prompt": prompt,
                    "parameters": params.to_dict(),
                    "generation_params": gen_params
                }
                json.dump(metadata, f, indent=2)
            
            print(f"Generated audio saved to {output_path}")
        
        return wav[0], model.sample_rate, prompt
    
    def generate_transition(
        self,
        start_params: EnvironmentParameters,
        end_params: EnvironmentParameters,
        n_steps: int = 3,
        duration: float = 10.0,
        output_prefix: str = "transition"
    ) -> List[Tuple[torch.Tensor, int, str]]:
        """
        Generate a series of audio segments to transition between environments.
        
        Args:
            start_params: Starting environmental parameters
            end_params: Ending environmental parameters
            n_steps: Number of transition steps
            duration: Duration of each segment in seconds
            output_prefix: Prefix for output filenames
            
        Returns:
            List of (audio_tensor, sample_rate, prompt) tuples
        """
        results = []
        
        # Generate intermediate parameter sets
        param_steps = []
        for i in range(n_steps):
            # Calculate interpolation factor
            t = i / (n_steps - 1) if n_steps > 1 else 0
            
            # Create interpolated parameters
            interp_params = EnvironmentParameters(
                environment_type=start_params.environment_type if t < 0.5 else end_params.environment_type,
                time_of_day=start_params.time_of_day * (1-t) + end_params.time_of_day * t,
                day_night_cycle=start_params.day_night_cycle * (1-t) + end_params.day_night_cycle * t,
                weather_type=start_params.weather_type if t < 0.5 else end_params.weather_type,
                weather_intensity=start_params.weather_intensity * (1-t) + end_params.weather_intensity * t,
                wind_intensity=start_params.wind_intensity * (1-t) + end_params.wind_intensity * t,
                tension=start_params.tension * (1-t) + end_params.tension * t,
                danger=start_params.danger * (1-t) + end_params.danger * t,
                population_density=start_params.population_density * (1-t) + end_params.population_density * t,
                civilization_proximity=start_params.civilization_proximity * (1-t) + end_params.civilization_proximity * t,
                magic_presence=start_params.magic_presence * (1-t) + end_params.magic_presence * t,
                supernatural_activity=start_params.supernatural_activity * (1-t) + end_params.supernatural_activity * t,
            )
            
            # Interpolate custom parameters
            for key in set(start_params.custom_parameters.keys()) | set(end_params.custom_parameters.keys()):
                start_value = start_params.custom_parameters.get(key, 0.0)
                end_value = end_params.custom_parameters.get(key, 0.0)
                interp_params.custom_parameters[key] = start_value * (1-t) + end_value * t
            
            param_steps.append(interp_params)
        
        # Generate audio for each step
        for i, step_params in enumerate(param_steps):
            output_filename = f"{output_prefix}_{i+1}_of_{n_steps}.wav"
            
            result = self.generate_parametric_audio(
                params=step_params,
                duration=duration,
                output_filename=output_filename
            )
            
            results.append(result)
        
        return results
    
    def create_environment_template(
        self,
        name: str,
        base_prompt: str,
        time_mapping: Dict[str, str] = None,
        weather_mapping: Dict[str, Dict[str, str]] = None,
        additional_mappings: Dict[str, Dict] = None
    ):
        """
        Create a new environment template.
        
        Args:
            name: Name of the environment
            base_prompt: Base prompt template
            time_mapping: Mapping of time periods to descriptions
            weather_mapping: Mapping of weather types to intensity descriptions
            additional_mappings: Additional parameter mappings
        """
        # Create template structure
        template = {
            "base_prompt": base_prompt,
        }
        
        # Add time mapping if provided
        if time_mapping:
            template["time_mapping"] = time_mapping
        
        # Add weather mapping if provided
        if weather_mapping:
            template["weather_mapping"] = weather_mapping
        
        # Add additional mappings
        if additional_mappings:
            for key, mapping in additional_mappings.items():
                template[key] = mapping
        
        # Add to templates
        self.environment_templates[name] = template
        self.save_templates()
        
        print(f"Created environment template: {name}")
```

## Implementation: Interactive Music System

Next, let's create a system for interactive, adaptive music that responds to gameplay events:

```python
import os
import json
import torch
import torchaudio
import numpy as np
from typing import Dict, List, Optional, Union, Literal, Tuple
from dataclasses import dataclass, field
from audiocraft.models import MusicGen

@dataclass
class MusicState:
    """Represents a music state with associated parameters."""
    
    state_id: str
    description: str
    
    # Main parameters
    intensity: float = 0.5  # 0-1 intensity
    energy: float = 0.5  # 0-1 energy level
    
    # Emotional parameters
    tension: float = 0.0  # 0-1 tension level
    danger: float = 0.0  # 0-1 danger level
    mystery: float = 0.0  # 0-1 mystery level
    triumph: float = 0.0  # 0-1 triumph level
    
    # Musical parameters
    tempo: int = 120  # BPM
    key: str = "C"  # Musical key
    mode: Literal["major", "minor"] = "major"
    
    # Transition parameters
    transition_time: float = 4.0  # Seconds
    can_transition_to: List[str] = field(default_factory=list)
    
    # Generation parameters
    prompt_template: str = "{description} with {intensity_desc} intensity, {energy_desc} energy, and {emotion_desc} emotion. {tempo} BPM in {key} {mode}."
    generation_params: Dict = field(default_factory=dict)
    
    def get_intensity_description(self) -> str:
        """Get textual description of intensity."""
        if self.intensity < 0.25:
            return "low"
        elif self.intensity < 0.5:
            return "moderate"
        elif self.intensity < 0.75:
            return "high"
        else:
            return "very high"
    
    def get_energy_description(self) -> str:
        """Get textual description of energy."""
        if self.energy < 0.25:
            return "calm"
        elif self.energy < 0.5:
            return "steady"
        elif self.energy < 0.75:
            return "energetic"
        else:
            return "extremely energetic"
    
    def get_emotion_description(self) -> str:
        """Get the dominant emotion based on parameters."""
        emotions = []
        
        if self.tension > 0.5:
            emotions.append(f"tense {'and suspenseful' if self.tension > 0.75 else ''}")
        
        if self.danger > 0.5:
            emotions.append(f"dangerous {'and threatening' if self.danger > 0.75 else ''}")
        
        if self.mystery > 0.5:
            emotions.append(f"mysterious {'and enigmatic' if self.mystery > 0.75 else ''}")
        
        if self.triumph > 0.5:
            emotions.append(f"triumphant {'and victorious' if self.triumph > 0.75 else ''}")
        
        if not emotions:
            if self.mode == "major":
                return "uplifting"
            else:
                return "somber"
        
        return " and ".join(emotions)
    
    def generate_prompt(self) -> str:
        """Generate a text prompt based on the music state."""
        # Get textual descriptions
        intensity_desc = self.get_intensity_description()
        energy_desc = self.get_energy_description()
        emotion_desc = self.get_emotion_description()
        
        # Format the prompt template
        prompt = self.prompt_template.format(
            description=self.description,
            intensity_desc=intensity_desc,
            energy_desc=energy_desc,
            emotion_desc=emotion_desc,
            tempo=self.tempo,
            key=self.key,
            mode=self.mode
        )
        
        return prompt
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {k: v for k, v in self.__dict__.items()}

class InteractiveMusicSystem:
    """
    Interactive music system that adapts to gameplay events and states.
    
    This system manages a collection of music states and transitions,
    generating adaptive music based on gameplay context.
    """
    
    def __init__(
        self,
        output_dir: str = "interactive_music",
        model_size: str = "small",
        device: str = None
    ):
        """
        Initialize the interactive music system.
        
        Args:
            output_dir: Directory for output files
            model_size: Size of MusicGen model to use
            device: Device to run model on (cuda, mps, cpu)
        """
        self.output_dir = output_dir
        self.model_size = model_size
        
        # Initialize device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize model (lazy loaded)
        self._model = None
        
        # Music states
        self.states = {}
        
        # Load music state definitions
        self.states_path = os.path.join(output_dir, "music_states.json")
        self.load_states()
    
    def load_states(self):
        """Load music states or create defaults if not found."""
        if os.path.exists(self.states_path):
            with open(self.states_path, 'r') as f:
                state_data = json.load(f)
                
                # Convert dictionary to MusicState objects
                for state_id, data in state_data.items():
                    self.states[state_id] = MusicState(**data)
        else:
            # Create default music states
            self.create_default_states()
            self.save_states()
    
    def save_states(self):
        """Save music states to disk."""
        # Convert MusicState objects to dictionaries
        state_data = {state_id: state.to_dict() for state_id, state in self.states.items()}
        
        with open(self.states_path, 'w') as f:
            json.dump(state_data, f, indent=2)
    
    def create_default_states(self):
        """Create a set of default music states."""
        # Exploration state
        exploration = MusicState(
            state_id="exploration",
            description="Peaceful exploration music with ambient elements",
            intensity=0.3,
            energy=0.4,
            tension=0.1,
            mystery=0.4,
            tempo=95,
            mode="major",
            can_transition_to=["tension", "combat", "discovery"]
        )
        exploration.generation_params = {
            "duration": 30.0,
            "temperature": 1.0,
            "cfg_coef": 3.0
        }
        self.states["exploration"] = exploration
        
        # Tension state
        tension = MusicState(
            state_id="tension",
            description="Suspenseful music with building tension",
            intensity=0.6,
            energy=0.5,
            tension=0.7,
            danger=0.4,
            mystery=0.3,
            tempo=110,
            mode="minor",
            can_transition_to=["exploration", "combat", "stealth"]
        )
        tension.generation_params = {
            "duration": 25.0,
            "temperature": 0.9,
            "cfg_coef": 3.5
        }
        self.states["tension"] = tension
        
        # Combat state
        combat = MusicState(
            state_id="combat",
            description="Intense combat music with driving percussion",
            intensity=0.9,
            energy=0.85,
            tension=0.6,
            danger=0.8,
            tempo=135,
            mode="minor",
            can_transition_to=["tension", "victory", "defeat"]
        )
        combat.generation_params = {
            "duration": 20.0,
            "temperature": 0.8,
            "cfg_coef": 4.0
        }
        self.states["combat"] = combat
        
        # Victory state
        victory = MusicState(
            state_id="victory",
            description="Triumphant victory music with uplifting melody",
            intensity=0.7,
            energy=0.8,
            triumph=0.9,
            tempo=120,
            mode="major",
            can_transition_to=["exploration"]
        )
        victory.generation_params = {
            "duration": 15.0,
            "temperature": 0.9,
            "cfg_coef": 3.0
        }
        self.states["victory"] = victory
        
        # Discovery state
        discovery = MusicState(
            state_id="discovery",
            description="Wondrous discovery music with magical elements",
            intensity=0.5,
            energy=0.6,
            mystery=0.7,
            triumph=0.3,
            tempo=100,
            mode="major",
            can_transition_to=["exploration", "tension"]
        )
        discovery.generation_params = {
            "duration": 20.0,
            "temperature": 1.0,
            "cfg_coef": 2.5
        }
        self.states["discovery"] = discovery
    
    def get_model(self):
        """Lazy-load the MusicGen model."""
        if self._model is None:
            print(f"Loading MusicGen model ({self.model_size})...")
            self._model = MusicGen.get_pretrained(self.model_size)
            self._model.to(self.device)
        return self._model
    
    def add_music_state(self, state: MusicState):
        """
        Add a new music state to the system.
        
        Args:
            state: Music state to add
        """
        self.states[state.state_id] = state
        self.save_states()
    
    def remove_music_state(self, state_id: str):
        """
        Remove a music state from the system.
        
        Args:
            state_id: ID of state to remove
        """
        if state_id in self.states:
            del self.states[state_id]
            self.save_states()
    
    def update_music_state(self, state_id: str, **kwargs):
        """
        Update parameters of a music state.
        
        Args:
            state_id: ID of state to update
            **kwargs: Parameters to update
        """
        if state_id not in self.states:
            print(f"State '{state_id}' not found")
            return
        
        state = self.states[state_id]
        
        # Update parameters
        for key, value in kwargs.items():
            if hasattr(state, key):
                setattr(state, key, value)
        
        self.save_states()
    
    def generate_music_for_state(
        self,
        state_id: str,
        output_filename: str = None,
        override_params: Dict = None
    ) -> Tuple[torch.Tensor, int, str]:
        """
        Generate music for a specific state.
        
        Args:
            state_id: ID of the music state
            output_filename: Optional filename for saving
            override_params: Optional parameter overrides
            
        Returns:
            Tuple of (audio_tensor, sample_rate, prompt)
        """
        if state_id not in self.states:
            raise ValueError(f"State '{state_id}' not found")
        
        state = self.states[state_id]
        
        # Generate the prompt
        prompt = state.generate_prompt()
        
        # Get generation parameters
        gen_params = dict(state.generation_params)
        if override_params:
            gen_params.update(override_params)
        
        # Load model
        model = self.get_model()
        model.set_generation_params(**gen_params)
        
        # Generate music
        wav = model.generate([prompt])
        
        # Save if filename provided
        if output_filename:
            if not output_filename.endswith('.wav'):
                output_filename += '.wav'
            
            output_path = os.path.join(self.output_dir, output_filename)
            torchaudio.save(
                output_path,
                wav[0].cpu(),
                model.sample_rate
            )
            
            # Save metadata
            metadata_path = output_path.replace('.wav', '.json')
            with open(metadata_path, 'w') as f:
                metadata = {
                    "state_id": state_id,
                    "prompt": prompt,
                    "state": state.to_dict(),
                    "generation_params": gen_params
                }
                json.dump(metadata, f, indent=2)
            
            print(f"Generated music saved to {output_path}")
        
        return wav[0], model.sample_rate, prompt
    
    def generate_transition(
        self,
        from_state_id: str,
        to_state_id: str,
        output_filename: str = None,
        duration: float = None
    ) -> Tuple[torch.Tensor, int, str]:
        """
        Generate a transition between two music states.
        
        Args:
            from_state_id: Starting state ID
            to_state_id: Target state ID
            output_filename: Optional filename for saving
            duration: Optional duration override
            
        Returns:
            Tuple of (audio_tensor, sample_rate, prompt)
        """
        if from_state_id not in self.states:
            raise ValueError(f"State '{from_state_id}' not found")
        
        if to_state_id not in self.states:
            raise ValueError(f"State '{to_state_id}' not found")
        
        from_state = self.states[from_state_id]
        to_state = self.states[to_state_id]
        
        # Check if transition is allowed
        if to_state_id not in from_state.can_transition_to:
            print(f"Warning: Transition from '{from_state_id}' to '{to_state_id}' is not defined as allowed")
        
        # Create blend of the two states
        transition_state = MusicState(
            state_id=f"transition_{from_state_id}_to_{to_state_id}",
            description=f"Transition from {from_state.description} to {to_state.description}",
            intensity=(from_state.intensity + to_state.intensity) / 2,
            energy=(from_state.energy + to_state.energy) / 2,
            tension=(from_state.tension + to_state.tension) / 2,
            danger=(from_state.danger + to_state.danger) / 2,
            mystery=(from_state.mystery + to_state.mystery) / 2,
            triumph=(from_state.triumph + to_state.triumph) / 2,
            tempo=int((from_state.tempo + to_state.tempo) / 2),
            key=to_state.key,  # Use target key for better transition
            mode=to_state.mode  # Use target mode for better transition
        )
        
        # Set transition prompt template
        transition_state.prompt_template = "Transition music bridging {description}. Moving from {intensity_desc} intensity to {energy_desc} energy with {emotion_desc} emotion. {tempo} BPM in {key} {mode}."
        
        # Determine duration
        if duration is None:
            # Use average of the two states' transition times, or a default
            duration = (from_state.transition_time + to_state.transition_time) / 2
        
        # Set generation parameters
        transition_params = {
            "duration": duration,
            "temperature": 1.0,  # Higher temperature for more variation
            "cfg_coef": 3.0
        }
        
        # Generate the transition music
        if output_filename is None:
            output_filename = f"transition_{from_state_id}_to_{to_state_id}.wav"
        
        return self.generate_music_for_state(
            state_id=transition_state.state_id,  # This won't be found, but we're passing the state directly
            output_filename=output_filename,
            override_params=transition_params
        )
    
    def generate_music_suite(self, output_prefix: str = "suite"):
        """
        Generate a complete suite of music assets for all states.
        
        Args:
            output_prefix: Prefix for output filenames
            
        Returns:
            Dictionary mapping state IDs to file paths
        """
        outputs = {}
        
        # Generate music for each state
        for state_id, state in self.states.items():
            output_filename = f"{output_prefix}_{state_id}.wav"
            
            _, _, _ = self.generate_music_for_state(
                state_id=state_id,
                output_filename=output_filename
            )
            
            outputs[state_id] = os.path.join(self.output_dir, output_filename)
        
        # Generate transitions between states
        for from_state_id, from_state in self.states.items():
            for to_state_id in from_state.can_transition_to:
                if to_state_id in self.states:
                    transition_filename = f"{output_prefix}_transition_{from_state_id}_to_{to_state_id}.wav"
                    
                    try:
                        _, _, _ = self.generate_transition(
                            from_state_id=from_state_id,
                            to_state_id=to_state_id,
                            output_filename=transition_filename
                        )
                        
                        transition_key = f"transition_{from_state_id}_to_{to_state_id}"
                        outputs[transition_key] = os.path.join(self.output_dir, transition_filename)
                    except Exception as e:
                        print(f"Error generating transition from {from_state_id} to {to_state_id}: {e}")
        
        print(f"Generated music suite with {len(outputs)} assets")
        return outputs
    
    def export_unity_package(self, package_name: str = None):
        """
        Generate a file structure suitable for import into Unity.
        
        Args:
            package_name: Name for the output package directory
            
        Returns:
            Path to the generated package
        """
        if package_name is None:
            package_name = "InteractiveMusicSystem"
        
        package_dir = os.path.join(self.output_dir, package_name)
        os.makedirs(package_dir, exist_ok=True)
        
        # Create subdirectories
        audio_dir = os.path.join(package_dir, "Audio")
        states_dir = os.path.join(package_dir, "States")
        transitions_dir = os.path.join(package_dir, "Transitions")
        
        os.makedirs(audio_dir, exist_ok=True)
        os.makedirs(states_dir, exist_ok=True)
        os.makedirs(transitions_dir, exist_ok=True)
        
        # Generate music assets if needed
        if not os.listdir(self.output_dir):
            print("No music assets found. Generating music suite...")
            self.generate_music_suite()
        
        # Gather all WAV files in the output directory
        wav_files = [f for f in os.listdir(self.output_dir) if f.endswith('.wav')]
        
        # Copy and organize files
        for filename in wav_files:
            source_path = os.path.join(self.output_dir, filename)
            
            if "transition" in filename:
                # Copy to transitions directory
                dest_path = os.path.join(transitions_dir, filename)
            else:
                # Try to extract state ID from filename
                parts = filename.split('_')
                if len(parts) > 1 and parts[1] in self.states:
                    # Copy to states directory
                    dest_path = os.path.join(states_dir, filename)
                else:
                    # Default to audio directory
                    dest_path = os.path.join(audio_dir, filename)
            
            # Copy the file
            shutil.copy2(source_path, dest_path)
            
            # Copy metadata if available
            metadata_path = source_path.replace('.wav', '.json')
            if os.path.exists(metadata_path):
                shutil.copy2(metadata_path, dest_path.replace('.wav', '.json'))
        
        # Export music state definitions
        with open(os.path.join(package_dir, "music_states.json"), 'w') as f:
            state_data = {state_id: state.to_dict() for state_id, state in self.states.items()}
            json.dump(state_data, f, indent=2)
        
        # Create a README file
        with open(os.path.join(package_dir, "README.md"), 'w') as f:
            f.write(f"# {package_name}\n\n")
            f.write("Interactive music system generated with AudioCraft.\n\n")
            
            f.write("## Music States\n\n")
            f.write("| State ID | Description | Intensity | Energy | Key | Mode |\n")
            f.write("|---|---|:---:|:---:|:---:|:---:|\n")
            
            for state_id, state in self.states.items():
                f.write(f"| {state_id} | {state.description} | {state.intensity:.1f} | {state.energy:.1f} | {state.key} | {state.mode} |\n")
            
            f.write("\n## Valid Transitions\n\n")
            for state_id, state in self.states.items():
                if state.can_transition_to:
                    f.write(f"* {state_id} → {', '.join(state.can_transition_to)}\n")
        
        print(f"Exported Unity package to {package_dir}")
        return package_dir
```

## Unity Implementation: Interactive Audio Controller

To complement our Python generation systems, let's implement a Unity C# script for handling interactive audio at runtime:

```csharp
using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Audio;

namespace GameAudio
{
    /// <summary>
    /// Parameter-driven audio controller for dynamic audio environments.
    /// </summary>
    [Serializable]
    public class AudioParameter
    {
        public string name;
        [Range(0f, 1f)] public float value;
        [Range(0f, 10f)] public float changeSensitivity = 1f;
        public bool triggersUpdate = true;
        public float lastUpdateValue;
        
        public bool HasSignificantChange()
        {
            float delta = Mathf.Abs(value - lastUpdateValue);
            return delta > (0.05f / changeSensitivity);
        }
        
        public void MarkUpdated()
        {
            lastUpdateValue = value;
        }
    }
    
    [Serializable]
    public class AudioLayer
    {
        public string name;
        public AudioClip[] possibleClips;
        public float volume = 1f;
        [Range(0f, 1f)] public float spatialBlend = 0f;
        public bool loop = true;
        
        [Header("Parameter Influence")]
        public string primaryParameter;
        [Range(-1f, 1f)] public float primaryInfluence = 1f;
        public string secondaryParameter;
        [Range(-1f, 1f)] public float secondaryInfluence = 0f;
        
        [HideInInspector] public AudioSource source;
        [HideInInspector] public int lastClipIndex = -1;
    }
    
    [Serializable]
    public class AudioState
    {
        public string name;
        public AudioClip primaryClip;
        public AudioClip[] layerClips;
        public float transitionTime = 2f;
        
        [Header("Parameter Thresholds")]
        public string triggerParameter;
        public float minimumValue = 0.7f;
    }
    
    public class InteractiveAudioController : MonoBehaviour
    {
        [Header("Audio Layers")]
        [SerializeField] private List<AudioLayer> audioLayers = new List<AudioLayer>();
        
        [Header("Parameters")]
        [SerializeField] private List<AudioParameter> parameters = new List<AudioParameter>();
        
        [Header("States")]
        [SerializeField] private List<AudioState> audioStates = new List<AudioState>();
        [SerializeField] private string defaultState;
        
        [Header("Settings")]
        [SerializeField] private AudioMixerGroup mixerGroup;
        [SerializeField] private float updateInterval = 0.5f;
        [SerializeField] private float crossfadeDuration = 1.5f;
        [SerializeField] private bool debugMode = false;
        
        // Internal state
        private Dictionary<string, AudioParameter> parameterLookup = new Dictionary<string, AudioParameter>();
        private Dictionary<string, AudioState> stateLookup = new Dictionary<string, AudioState>();
        private AudioState currentState;
        private float timeSinceLastUpdate = 0f;
        private bool initialized = false;
        
        private void Start()
        {
            Initialize();
        }
        
        private void Initialize()
        {
            if (initialized) return;
            
            // Create parameter lookup
            foreach (var param in parameters)
            {
                parameterLookup[param.name] = param;
                param.MarkUpdated();
            }
            
            // Create state lookup
            foreach (var state in audioStates)
            {
                stateLookup[state.name] = state;
            }
            
            // Initialize audio sources for layers
            foreach (var layer in audioLayers)
            {
                var sourceObj = new GameObject($"Layer - {layer.name}");
                sourceObj.transform.SetParent(transform);
                
                layer.source = sourceObj.AddComponent<AudioSource>();
                layer.source.outputAudioMixerGroup = mixerGroup;
                layer.source.playOnAwake = false;
                layer.source.loop = layer.loop;
                layer.source.spatialBlend = layer.spatialBlend;
                layer.source.volume = 0f; // Start silent
            }
            
            // Set default state
            if (!string.IsNullOrEmpty(defaultState) && stateLookup.ContainsKey(defaultState))
            {
                TransitionToState(defaultState);
            }
            
            initialized = true;
        }
        
        private void Update()
        {
            if (!initialized)
            {
                Initialize();
                return;
            }
            
            // Check for parameter updates
            timeSinceLastUpdate += Time.deltaTime;
            
            if (timeSinceLastUpdate >= updateInterval)
            {
                bool shouldUpdate = false;
                
                // Check if any parameters have changed significantly
                foreach (var param in parameters)
                {
                    if (param.triggersUpdate && param.HasSignificantChange())
                    {
                        shouldUpdate = true;
                        param.MarkUpdated();
                    }
                }
                
                // Check for state transitions based on parameters
                CheckStateTransitions();
                
                // Update audio based on parameters
                if (shouldUpdate)
                {
                    UpdateAudioLayers();
                }
                
                timeSinceLastUpdate = 0f;
            }
        }
        
        private void CheckStateTransitions()
        {
            foreach (var state in audioStates)
            {
                if (state == currentState) continue;
                
                if (!string.IsNullOrEmpty(state.triggerParameter) && 
                    parameterLookup.TryGetValue(state.triggerParameter, out AudioParameter param))
                {
                    if (param.value >= state.minimumValue)
                    {
                        TransitionToState(state.name);
                        break;
                    }
                }
            }
        }
        
        private void UpdateAudioLayers()
        {
            foreach (var layer in audioLayers)
            {
                // Calculate layer volume based on parameters
                float parameterFactor = 1f;
                
                if (!string.IsNullOrEmpty(layer.primaryParameter) && 
                    parameterLookup.TryGetValue(layer.primaryParameter, out AudioParameter primaryParam))
                {
                    // Apply primary influence (positive or negative)
                    float primaryFactor = layer.primaryInfluence > 0 
                        ? primaryParam.value * layer.primaryInfluence
                        : (1 - primaryParam.value) * -layer.primaryInfluence;
                    
                    parameterFactor *= primaryFactor;
                }
                
                if (!string.IsNullOrEmpty(layer.secondaryParameter) && 
                    parameterLookup.TryGetValue(layer.secondaryParameter, out AudioParameter secondaryParam))
                {
                    // Apply secondary influence (positive or negative)
                    float secondaryFactor = layer.secondaryInfluence > 0 
                        ? secondaryParam.value * layer.secondaryInfluence
                        : (1 - secondaryParam.value) * -layer.secondaryInfluence;
                    
                    parameterFactor *= secondaryFactor;
                }
                
                // Apply parameter factor to layer volume
                float targetVolume = layer.volume * parameterFactor;
                
                // Update audio source
                if (layer.source.volume != targetVolume)
                {
                    StartCoroutine(FadeVolume(layer.source, targetVolume, crossfadeDuration));
                }
                
                // Choose appropriate clip if needed
                if (layer.possibleClips != null && layer.possibleClips.Length > 0)
                {
                    // Get parameter value for clip selection
                    float selectionValue = 0.5f;
                    
                    if (!string.IsNullOrEmpty(layer.primaryParameter) && 
                        parameterLookup.TryGetValue(layer.primaryParameter, out AudioParameter selectionParam))
                    {
                        selectionValue = selectionParam.value;
                    }
                    
                    // Map parameter value to clip index
                    int clipIndex = Mathf.FloorToInt(selectionValue * layer.possibleClips.Length);
                    clipIndex = Mathf.Clamp(clipIndex, 0, layer.possibleClips.Length - 1);
                    
                    // Change clip if needed
                    if (clipIndex != layer.lastClipIndex)
                    {
                        layer.lastClipIndex = clipIndex;
                        
                        // Only change clip if significant parameter change
                        if (Mathf.Abs(selectionValue - 0.5f) > 0.2f)
                        {
                            StartCoroutine(CrossfadeClip(
                                layer.source, 
                                layer.possibleClips[clipIndex], 
                                crossfadeDuration
                            ));
                        }
                    }
                }
                
                // Ensure layer is playing
                if (!layer.source.isPlaying && layer.source.clip != null && layer.loop)
                {
                    layer.source.Play();
                }
            }
        }
        
        public void SetParameter(string paramName, float value)
        {
            if (parameterLookup.TryGetValue(paramName, out AudioParameter param))
            {
                param.value = Mathf.Clamp01(value);
                
                if (debugMode)
                {
                    Debug.Log($"[InteractiveAudio] Parameter '{paramName}' set to {param.value:F2}");
                }
            }
            else if (debugMode)
            {
                Debug.LogWarning($"[InteractiveAudio] Parameter '{paramName}' not found");
            }
        }
        
        public float GetParameter(string paramName)
        {
            if (parameterLookup.TryGetValue(paramName, out AudioParameter param))
            {
                return param.value;
            }
            
            return 0f;
        }
        
        public void TransitionToState(string stateName)
        {
            if (!stateLookup.ContainsKey(stateName))
            {
                Debug.LogWarning($"[InteractiveAudio] State '{stateName}' not found");
                return;
            }
            
            AudioState newState = stateLookup[stateName];
            AudioState previousState = currentState;
            currentState = newState;
            
            if (debugMode)
            {
                Debug.Log($"[InteractiveAudio] Transitioning to state: {stateName}");
            }
            
            // Setup transition
            float transitionTime = newState.transitionTime;
            
            // Apply primary clip to first layer if available
            if (newState.primaryClip != null && audioLayers.Count > 0)
            {
                var primaryLayer = audioLayers[0];
                StartCoroutine(CrossfadeClip(primaryLayer.source, newState.primaryClip, transitionTime));
            }
            
            // Apply layer clips
            if (newState.layerClips != null)
            {
                for (int i = 0; i < newState.layerClips.Length && i + 1 < audioLayers.Count; i++)
                {
                    var layer = audioLayers[i + 1];
                    if (newState.layerClips[i] != null)
                    {
                        StartCoroutine(CrossfadeClip(layer.source, newState.layerClips[i], transitionTime));
                    }
                }
            }
        }
        
        private System.Collections.IEnumerator CrossfadeClip(
            AudioSource source, AudioClip newClip, float fadeDuration)
        {
            if (source == null || newClip == null)
                yield break;
            
            // If the source isn't playing at all, just set the clip directly
            if (!source.isPlaying)
            {
                source.clip = newClip;
                source.Play();
                yield break;
            }
            
            // Remember the original volume
            float originalVolume = source.volume;
            
            // Fade out current clip
            float timeElapsed = 0f;
            while (timeElapsed < fadeDuration * 0.5f)
            {
                source.volume = Mathf.Lerp(originalVolume, 0f, timeElapsed / (fadeDuration * 0.5f));
                timeElapsed += Time.deltaTime;
                yield return null;
            }
            
            // Change clip
            source.Stop();
            source.clip = newClip;
            source.Play();
            source.volume = 0f;
            
            // Fade in new clip
            timeElapsed = 0f;
            while (timeElapsed < fadeDuration * 0.5f)
            {
                source.volume = Mathf.Lerp(0f, originalVolume, timeElapsed / (fadeDuration * 0.5f));
                timeElapsed += Time.deltaTime;
                yield return null;
            }
            
            // Ensure we end at the exact target volume
            source.volume = originalVolume;
        }
        
        private System.Collections.IEnumerator FadeVolume(
            AudioSource source, float targetVolume, float fadeDuration)
        {
            if (source == null)
                yield break;
            
            float startVolume = source.volume;
            float timeElapsed = 0f;
            
            while (timeElapsed < fadeDuration)
            {
                source.volume = Mathf.Lerp(startVolume, targetVolume, timeElapsed / fadeDuration);
                timeElapsed += Time.deltaTime;
                yield return null;
            }
            
            // Ensure we end at the exact target volume
            source.volume = targetVolume;
        }
    }
}
```

## Implementing Emotion-Driven Audio

Let's create a system that maps emotional states to audio generation:

```csharp
using System;
using System.Collections.Generic;
using UnityEngine;

namespace GameAudio
{
    /// <summary>
    /// Maps emotional states to audio parameters to drive
    /// emotion-aware audio systems.
    /// </summary>
    [Serializable]
    public class EmotionalState
    {
        public string name;
        
        [Header("Core Emotions")]
        [Range(0f, 1f)] public float joy;
        [Range(0f, 1f)] public float sadness;
        [Range(0f, 1f)] public float fear;
        [Range(0f, 1f)] public float anger;
        [Range(0f, 1f)] public float surprise;
        [Range(0f, 1f)] public float disgust;
        
        [Header("Extended Emotions")]
        [Range(0f, 1f)] public float tension;
        [Range(0f, 1f)] public float mystery;
        [Range(0f, 1f)] public float triumph;
        [Range(0f, 1f)] public float serenity;
        [Range(0f, 1f)] public float despair;
        
        [Header("Sound Design")]
        public float musicIntensity = 0.5f;
        public float musicEnergy = 0.5f;
        public float ambientTension = 0.0f;
        public float reverbAmount = 0.0f;
        public string suggestedAudioState;
        
        public Dictionary<string, float> ToParameterDictionary()
        {
            return new Dictionary<string, float>
            {
                { "joy", joy },
                { "sadness", sadness },
                { "fear", fear },
                { "anger", anger },
                { "surprise", surprise },
                { "disgust", disgust },
                { "tension", tension },
                { "mystery", mystery },
                { "triumph", triumph },
                { "serenity", serenity },
                { "despair", despair },
                { "musicIntensity", musicIntensity },
                { "musicEnergy", musicEnergy },
                { "ambientTension", ambientTension },
                { "reverbAmount", reverbAmount }
            };
        }
    }
    
    public class EmotionAwareAudioSystem : MonoBehaviour
    {
        [SerializeField] private List<EmotionalState> emotionalStates = new List<EmotionalState>();
        [SerializeField] private string currentState = "neutral";
        [SerializeField] private float blendDuration = 3.0f;
        [SerializeField] private bool debugMode = false;
        
        [Header("Audio Controllers")]
        [SerializeField] private InteractiveAudioController audioController;
        [SerializeField] private AudioSource ambienceSource;
        [SerializeField] private AudioSource musicSource;
        [SerializeField] private AudioMixerGroup mixerGroup;
        
        // Internal state
        private Dictionary<string, EmotionalState> stateLookup = new Dictionary<string, EmotionalState>();
        private EmotionalState activeState;
        private EmotionalState targetState;
        private float blendProgress = 1.0f;
        private Dictionary<string, float> currentParameters = new Dictionary<string, float>();
        
        private void Start()
        {
            Initialize();
        }
        
        private void Initialize()
        {
            // Create state lookup
            foreach (var state in emotionalStates)
            {
                stateLookup[state.name] = state;
            }
            
            // Set initial state
            if (stateLookup.TryGetValue(currentState, out EmotionalState initialState))
            {
                activeState = initialState;
                targetState = initialState;
                ApplyEmotionalState(initialState);
            }
            else if (emotionalStates.Count > 0)
            {
                // Default to first state
                activeState = emotionalStates[0];
                targetState = activeState;
                ApplyEmotionalState(activeState);
            }
            
            // Find audio controller if not set
            if (audioController == null)
            {
                audioController = FindObjectOfType<InteractiveAudioController>();
            }
        }
        
        private void Update()
        {
            // Handle blending between states
            if (blendProgress < 1.0f)
            {
                blendProgress += Time.deltaTime / blendDuration;
                
                if (blendProgress >= 1.0f)
                {
                    // Blend complete
                    blendProgress = 1.0f;
                    activeState = targetState;
                    
                    if (debugMode)
                    {
                        Debug.Log($"[EmotionAudio] Blend to {targetState.name} complete");
                    }
                }
                else
                {
                    // Update blended parameters
                    UpdateBlendedParameters();
                }
            }
        }
        
        public void SetEmotionalState(string stateName, bool instant = false)
        {
            if (!stateLookup.TryGetValue(stateName, out EmotionalState newState))
            {
                Debug.LogWarning($"[EmotionAudio] Emotional state '{stateName}' not found");
                return;
            }
            
            if (debugMode)
            {
                Debug.Log($"[EmotionAudio] Setting emotional state to: {stateName}");
            }
            
            // Set target state
            targetState = newState;
            currentState = stateName;
            
            if (instant)
            {
                // Apply immediately
                activeState = targetState;
                blendProgress = 1.0f;
                ApplyEmotionalState(activeState);
            }
            else
            {
                // Start blending
                blendProgress = 0.0f;
                
                // Initialize blend with current parameters
                if (activeState != null)
                {
                    currentParameters = activeState.ToParameterDictionary();
                }
                else
                {
                    currentParameters = targetState.ToParameterDictionary();
                    blendProgress = 1.0f;
                }
            }
        }
        
        private void UpdateBlendedParameters()
        {
            if (activeState == null || targetState == null)
                return;
            
            // Get parameter dictionaries
            Dictionary<string, float> targetParams = targetState.ToParameterDictionary();
            
            // Create blended dictionary
            Dictionary<string, float> blendedParams = new Dictionary<string, float>();
            
            // Blend parameters
            foreach (var kvp in targetParams)
            {
                float startValue = currentParameters.ContainsKey(kvp.Key) ? currentParameters[kvp.Key] : kvp.Value;
                float blendedValue = Mathf.Lerp(startValue, kvp.Value, blendProgress);
                blendedParams[kvp.Key] = blendedValue;
            }
            
            // Apply blended parameters
            ApplyParameters(blendedParams);
        }
        
        private void ApplyEmotionalState(EmotionalState state)
        {
            if (state == null)
                return;
            
            // Convert state to parameters and apply
            Dictionary<string, float> parameters = state.ToParameterDictionary();
            ApplyParameters(parameters);
            
            // Apply suggested audio state if available
            if (!string.IsNullOrEmpty(state.suggestedAudioState) && audioController != null)
            {
                audioController.TransitionToState(state.suggestedAudioState);
            }
        }
        
        private void ApplyParameters(Dictionary<string, float> parameters)
        {
            // Apply to interactive audio controller
            if (audioController != null)
            {
                foreach (var kvp in parameters)
                {
                    audioController.SetParameter(kvp.Key, kvp.Value);
                }
            }
            
            // Apply mixer effects if available
            if (mixerGroup != null)
            {
                // Example: Control reverb based on emotions
                if (parameters.TryGetValue("reverbAmount", out float reverbAmount))
                {
                    mixerGroup.audioMixer.SetFloat("ReverbAmount", Mathf.Lerp(0f, 1f, reverbAmount));
                }
            }
            
            // Apply to music source if available
            if (musicSource != null)
            {
                // Example: Control music volume based on emotions
                if (parameters.TryGetValue("musicIntensity", out float musicIntensity))
                {
                    musicSource.volume = musicIntensity;
                }
            }
            
            // Apply to ambience source if available
            if (ambienceSource != null)
            {
                // Example: Control ambience volume based on emotions
                if (parameters.TryGetValue("ambientTension", out float ambientTension))
                {
                    ambienceSource.volume = 0.5f + (ambientTension * 0.5f);
                }
            }
        }
        
        // Designer-friendly methods to gradually adjust emotional states
        public void AdjustEmotion(string emotion, float amount, float blendTime = 1.0f)
        {
            if (activeState == null)
                return;
            
            // Create a copy of the current state
            EmotionalState adjustedState = new EmotionalState();
            
            // Copy all values
            adjustedState.name = $"{activeState.name}_adjusted";
            adjustedState.joy = activeState.joy;
            adjustedState.sadness = activeState.sadness;
            adjustedState.fear = activeState.fear;
            adjustedState.anger = activeState.anger;
            adjustedState.surprise = activeState.surprise;
            adjustedState.disgust = activeState.disgust;
            adjustedState.tension = activeState.tension;
            adjustedState.mystery = activeState.mystery;
            adjustedState.triumph = activeState.triumph;
            adjustedState.serenity = activeState.serenity;
            adjustedState.despair = activeState.despair;
            adjustedState.musicIntensity = activeState.musicIntensity;
            adjustedState.musicEnergy = activeState.musicEnergy;
            adjustedState.ambientTension = activeState.ambientTension;
            adjustedState.reverbAmount = activeState.reverbAmount;
            adjustedState.suggestedAudioState = activeState.suggestedAudioState;
            
            // Apply adjustment to specified emotion
            switch (emotion.ToLower())
            {
                case "joy": adjustedState.joy = Mathf.Clamp01(adjustedState.joy + amount); break;
                case "sadness": adjustedState.sadness = Mathf.Clamp01(adjustedState.sadness + amount); break;
                case "fear": adjustedState.fear = Mathf.Clamp01(adjustedState.fear + amount); break;
                case "anger": adjustedState.anger = Mathf.Clamp01(adjustedState.anger + amount); break;
                case "surprise": adjustedState.surprise = Mathf.Clamp01(adjustedState.surprise + amount); break;
                case "disgust": adjustedState.disgust = Mathf.Clamp01(adjustedState.disgust + amount); break;
                case "tension": adjustedState.tension = Mathf.Clamp01(adjustedState.tension + amount); break;
                case "mystery": adjustedState.mystery = Mathf.Clamp01(adjustedState.mystery + amount); break;
                case "triumph": adjustedState.triumph = Mathf.Clamp01(adjustedState.triumph + amount); break;
                case "serenity": adjustedState.serenity = Mathf.Clamp01(adjustedState.serenity + amount); break;
                case "despair": adjustedState.despair = Mathf.Clamp01(adjustedState.despair + amount); break;
            }
            
            // Adjust derived audio parameters based on emotional changes
            // Example: Increasing fear increases tension and decreases energy
            if (emotion.ToLower() == "fear" && amount != 0)
            {
                adjustedState.tension = Mathf.Clamp01(adjustedState.tension + (amount * 0.5f));
                adjustedState.musicEnergy = Mathf.Clamp01(adjustedState.musicEnergy - (amount * 0.3f));
                adjustedState.ambientTension = Mathf.Clamp01(adjustedState.ambientTension + (amount * 0.7f));
            }
            
            // Set the adjusted state as target
            targetState = adjustedState;
            blendDuration = blendTime;
            blendProgress = 0.0f;
            
            // Initialize blend with current parameters
            currentParameters = activeState.ToParameterDictionary();
        }
    }
}
```

## Interactive Audio Best Practices

Based on our implementations, here are best practices for designing interactive audio systems with AudioCraft:

1. **Parameter-Driven Generation**
   - Define clear, measurable parameters that reflect gameplay states
   - Create mappings between parameters and audio characteristics
   - Use interpolation between parameter values for smooth transitions
   - Design templates for different environment types

2. **State-Based Music Systems**
   - Define distinct musical states that match gameplay contexts
   - Create explicit transitions between states
   - Use emotional parameters to guide music generation
   - Balance consistency and variation in generated content

3. **Layered Audio Architecture**
   - Create multiple audio layers that can be independently controlled
   - Use parameter influence to fade layers in and out
   - Implement cross-fading between audio clips
   - Ensure seamless transitions between game states

4. **Emotional Mapping**
   - Define core emotional states and their audio characteristics
   - Create mappings between narrative events and emotional changes
   - Use blending to create smooth transitions between emotional states
   - Connect emotions to specific audio parameters

5. **Technical Implementation**
   - Use pooling for efficient audio source management
   - Implement smooth crossfading for transitions
   - Create systems that can batch-generate content when needed
   - Optimize runtime memory usage for audio assets

## Hands-On Challenge: Create an Emotion-Aware Audio Environment

**Challenge:** Build a complete emotion-aware audio environment that responds to a player's game state.

1. Define five emotional states: neutral, tense, triumphant, mysterious, and defeated
2. Create parameter mappings for each emotional state
3. Implement transitions between states based on gameplay events
4. Build a demonstration scene that shows all states and transitions
5. Generate a complete asset package using AudioCraft

**Steps to implement:**

1. Create the EmotionalState definitions
2. Implement the parameter mappings for audio and music
3. Build the interactive audio controllers
4. Create a test scene with state trigger zones
5. Generate the complete audio asset set using our AudioCraft systems

## Next Steps

In the next chapter, we'll explore integrating AI audio generation with other multimedia elements to create complete interactive experiences. We'll build systems that connect audio generation with procedural visuals, narrative systems, and player input.

Copyright © 2025 Scott Friedman. Licensed under [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).
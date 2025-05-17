# Chapter 13: Sound Design Workflows

> *"Our game needs a complete audio update - we need ambient background soundscapes, interactive environmental elements, UI sounds, and character effects. And we need to be able to iterate on everything quickly as the design evolves. Can we build a workflow that helps us manage all these audio assets efficiently with AudioCraft?"*  
> — Lead Sound Designer, Indie Game Studio

## Learning Objectives

In this chapter, you'll learn how to:

- Design and implement end-to-end sound design workflows using AudioCraft
- Structure modular audio generation pipelines for efficient asset creation
- Organize project architectures for collaborative sound design
- Create metadata-driven workflows for sound asset management
- Automate repetitive sound design tasks with batch processing

## Introduction

Sound design is rarely a one-off task. Professional sound designers build comprehensive workflows that let them efficiently generate, organize, refine, and manage audio assets. In this chapter, we'll explore how to create structured workflows that leverage AudioCraft's capabilities for both music and sound effect generation.

A well-designed audio workflow provides several advantages:

1. **Efficiency**: Automate repetitive tasks and batch process similar audio needs
2. **Consistency**: Ensure all audio assets follow the same quality standards and aesthetic direction
3. **Iteration**: Quickly generate variations and alternatives for stakeholder feedback
4. **Organization**: Maintain clear relationships between assets, their purposes, and their metadata
5. **Scalability**: Handle growing audio needs as projects expand in scope

Let's explore how to build robust sound design pipelines that integrate AudioCraft's capabilities into professional-grade workflows.

## Implementation: Building a Text-to-Audio Pipeline

The foundation of an efficient sound design workflow is a pipeline architecture that can handle different audio generation tasks in a consistent way. Let's implement a unified pipeline that can work with both MusicGen and AudioGen:

```python
import os
import json
import torch
import torchaudio
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Literal
from audiocraft.models import MusicGen, AudioGen

@dataclass
class AssetMetadata:
    """Data structure for storing audio asset metadata."""
    
    # Basic information
    asset_id: str
    asset_type: Literal["music", "sfx", "ambience", "voice", "ui"]
    description: str
    
    # Generation parameters
    prompt: str
    model_used: str
    generation_params: Dict
    
    # Organization
    tags: List[str]
    category: str
    subcategory: Optional[str] = None
    
    # Project information
    project: str
    version: str
    created_at: str = None
    
    # Additional information
    notes: Optional[str] = None
    
    def __post_init__(self):
        """Set created_at timestamp if not provided."""
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {k: v for k, v in self.__dict__.items()}

class AudioPipeline:
    """
    A unified pipeline for generating, processing, and managing audio assets.
    
    This pipeline provides a consistent interface for working with both music
    and sound effect generation, handling the entire process from generation
    to organization and metadata management.
    """
    
    def __init__(
        self,
        base_output_dir: str = "audio_assets",
        music_model_size: str = "small",
        audio_model_size: str = "medium",
        device: str = None
    ):
        """
        Initialize the audio pipeline.
        
        Args:
            base_output_dir: Base directory for all generated assets
            music_model_size: Size of MusicGen model to use
            audio_model_size: Size of AudioGen model to use
            device: Device to run models on (cuda, mps, cpu)
        """
        self.base_output_dir = base_output_dir
        self.music_model_size = music_model_size
        self.audio_model_size = audio_model_size
        
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
            
        # Create models (lazy-loaded on first use)
        self._music_model = None
        self._audio_model = None
        
        # Create output directories
        os.makedirs(base_output_dir, exist_ok=True)
        
        # Asset database
        self.asset_db_path = os.path.join(base_output_dir, "asset_database.json")
        self.load_asset_database()
    
    def load_asset_database(self):
        """Load existing asset database or create a new one."""
        if os.path.exists(self.asset_db_path):
            with open(self.asset_db_path, 'r') as f:
                self.asset_database = json.load(f)
        else:
            self.asset_database = {}
            self.save_asset_database()
    
    def save_asset_database(self):
        """Save the asset database to disk."""
        with open(self.asset_db_path, 'w') as f:
            json.dump(self.asset_database, f, indent=2)
    
    def get_music_model(self):
        """Lazy-load the MusicGen model."""
        if self._music_model is None:
            print(f"Loading MusicGen model ({self.music_model_size})...")
            self._music_model = MusicGen.get_pretrained(self.music_model_size)
            self._music_model.to(self.device)
        return self._music_model
    
    def get_audio_model(self):
        """Lazy-load the AudioGen model."""
        if self._audio_model is None:
            print(f"Loading AudioGen model ({self.audio_model_size})...")
            self._audio_model = AudioGen.get_pretrained(self.audio_model_size)
            self._audio_model.to(self.device)
        return self._audio_model
    
    def generate_asset(
        self,
        prompt: str,
        asset_type: Literal["music", "sfx", "ambience", "voice", "ui"],
        project: str,
        category: str,
        tags: List[str],
        duration: float = 5.0,
        temperature: float = 1.0,
        cfg_coef: float = 3.0,
        top_k: int = 250,
        top_p: float = 0.0,
        version: str = "1.0",
        subcategory: Optional[str] = None,
        notes: Optional[str] = None,
        melody_conditioning: Optional[torch.Tensor] = None
    ) -> str:
        """
        Generate a new audio asset and save it with metadata.
        
        Args:
            prompt: Text description of the desired audio
            asset_type: Type of audio asset to generate
            project: Project name for organization
            category: Main category for the asset
            tags: List of tags for searching and filtering
            duration: Duration in seconds
            temperature: Controls randomness (1.0 = standard)
            cfg_coef: Guidance scale (higher = more prompt adherence)
            top_k: Controls diversity with k-sampling
            top_p: Controls diversity with nucleus sampling
            version: Version string for the asset
            subcategory: Optional subcategory
            notes: Optional notes about the asset
            melody_conditioning: Optional melody conditioning tensor
        
        Returns:
            asset_id: Unique identifier for the generated asset
        """
        # Create unique asset ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        asset_id = f"{project}_{asset_type}_{timestamp}"
        
        # Create asset directory
        asset_dir = os.path.join(self.base_output_dir, project, asset_type, category)
        if subcategory:
            asset_dir = os.path.join(asset_dir, subcategory)
        os.makedirs(asset_dir, exist_ok=True)
        
        # Set generation parameters
        generation_params = {
            "duration": duration,
            "temperature": temperature,
            "cfg_coef": cfg_coef,
            "top_k": top_k,
            "top_p": top_p
        }
        
        # Generate audio based on asset type
        if asset_type == "music":
            model = self.get_music_model()
            model.set_generation_params(**generation_params)
            
            if melody_conditioning is not None:
                melody = melody_conditioning.to(self.device)
                wav = model.generate_with_chroma([prompt], melody.unsqueeze(0))
            else:
                wav = model.generate([prompt])
            
            model_used = f"MusicGen-{self.music_model_size}"
            sample_rate = model.sample_rate
            
        else:  # All non-music asset types use AudioGen
            model = self.get_audio_model()
            model.set_generation_params(**generation_params)
            wav = model.generate([prompt])
            model_used = f"AudioGen-{self.audio_model_size}"
            sample_rate = model.sample_rate
        
        # Create file paths
        audio_path = os.path.join(asset_dir, f"{asset_id}.wav")
        metadata_path = os.path.join(asset_dir, f"{asset_id}.json")
        
        # Save audio file
        torchaudio.save(
            audio_path,
            wav[0].cpu(),
            sample_rate
        )
        
        # Create and save metadata
        metadata = AssetMetadata(
            asset_id=asset_id,
            asset_type=asset_type,
            description=prompt,
            prompt=prompt,
            model_used=model_used,
            generation_params=generation_params,
            tags=tags,
            category=category,
            subcategory=subcategory,
            project=project,
            version=version,
            notes=notes
        )
        
        # Save metadata to file
        with open(metadata_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        # Add to asset database
        self.asset_database[asset_id] = {
            "metadata": metadata.to_dict(),
            "file_path": audio_path,
            "metadata_path": metadata_path
        }
        self.save_asset_database()
        
        print(f"Generated asset {asset_id} and saved to {audio_path}")
        return asset_id
    
    def search_assets(
        self,
        asset_type: Optional[str] = None,
        project: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        text_search: Optional[str] = None
    ) -> List[Dict]:
        """
        Search for assets matching the given criteria.
        
        Args:
            asset_type: Filter by asset type
            project: Filter by project
            category: Filter by category
            tags: Filter by tags (asset must have ALL specified tags)
            text_search: Search in prompt and description
            
        Returns:
            List of matching asset metadata
        """
        results = []
        
        for asset_id, asset_info in self.asset_database.items():
            metadata = asset_info["metadata"]
            match = True
            
            # Apply filters
            if asset_type and metadata["asset_type"] != asset_type:
                match = False
            
            if project and metadata["project"] != project:
                match = False
                
            if category and metadata["category"] != category:
                match = False
            
            if tags:
                if not all(tag in metadata["tags"] for tag in tags):
                    match = False
            
            if text_search:
                text_search = text_search.lower()
                if (text_search not in metadata["prompt"].lower() and 
                    text_search not in metadata["description"].lower()):
                    match = False
            
            if match:
                results.append(asset_info)
        
        return results
    
    def batch_generate(
        self,
        generation_specs: List[Dict],
        parallel: bool = False
    ) -> List[str]:
        """
        Generate multiple assets in a batch.
        
        Args:
            generation_specs: List of dictionaries with generation parameters
            parallel: Whether to process in parallel (requires sufficient memory)
            
        Returns:
            List of generated asset IDs
        """
        asset_ids = []
        
        if parallel:
            # Future implementation: parallel processing
            raise NotImplementedError("Parallel processing not yet implemented")
        else:
            # Sequential processing
            for spec in generation_specs:
                asset_id = self.generate_asset(**spec)
                asset_ids.append(asset_id)
        
        return asset_ids

class ProjectSoundDesigner:
    """
    High-level interface for managing sound design for a specific project.
    
    This class provides project-specific workflow tools that build on the
    AudioPipeline to streamline common sound design tasks.
    """
    
    def __init__(
        self,
        project_name: str,
        output_dir: str = "projects",
        music_model_size: str = "small",
        audio_model_size: str = "medium",
        device: str = None
    ):
        """
        Initialize a project sound designer.
        
        Args:
            project_name: Name of the project
            output_dir: Base directory for project output
            music_model_size: Size of MusicGen model to use
            audio_model_size: Size of AudioGen model to use
            device: Device to run models on (cuda, mps, cpu)
        """
        self.project_name = project_name
        self.project_dir = os.path.join(output_dir, project_name)
        os.makedirs(self.project_dir, exist_ok=True)
        
        # Initialize audio pipeline
        self.pipeline = AudioPipeline(
            base_output_dir=self.project_dir,
            music_model_size=music_model_size,
            audio_model_size=audio_model_size,
            device=device
        )
        
        # Project sound definition file
        self.sound_def_path = os.path.join(self.project_dir, "sound_definitions.json")
        self.load_sound_definitions()
    
    def load_sound_definitions(self):
        """Load existing sound definitions or create new ones."""
        if os.path.exists(self.sound_def_path):
            with open(self.sound_def_path, 'r') as f:
                self.sound_definitions = json.load(f)
        else:
            # Initialize with default categories
            self.sound_definitions = {
                "music": {},
                "sfx": {
                    "ui": {},
                    "environment": {},
                    "character": {},
                    "interaction": {}
                },
                "ambience": {}
            }
            self.save_sound_definitions()
    
    def save_sound_definitions(self):
        """Save sound definitions to disk."""
        with open(self.sound_def_path, 'w') as f:
            json.dump(self.sound_definitions, f, indent=2)
    
    def define_sound(
        self,
        sound_id: str,
        asset_type: Literal["music", "sfx", "ambience", "voice", "ui"],
        category: str,
        description: str,
        prompt_template: str,
        tags: List[str],
        default_params: Dict = None,
        subcategory: str = None
    ):
        """
        Define a sound for the project.
        
        Args:
            sound_id: Unique identifier for this sound
            asset_type: Type of audio asset
            category: Main category
            description: Description of the sound's purpose
            prompt_template: Base prompt template (can include {variables})
            tags: List of tags
            default_params: Default generation parameters
            subcategory: Optional subcategory
        """
        # Create default parameters if not provided
        if default_params is None:
            if asset_type == "music":
                default_params = {
                    "duration": 30.0,
                    "temperature": 1.0,
                    "cfg_coef": 3.0
                }
            else:
                default_params = {
                    "duration": 5.0,
                    "temperature": 0.8,
                    "cfg_coef": 5.0
                }
        
        # Create sound definition
        sound_def = {
            "sound_id": sound_id,
            "asset_type": asset_type,
            "category": category,
            "subcategory": subcategory,
            "description": description,
            "prompt_template": prompt_template,
            "tags": tags,
            "default_params": default_params,
            "variations": []
        }
        
        # Add to sound definitions
        if subcategory:
            if subcategory not in self.sound_definitions[asset_type][category]:
                self.sound_definitions[asset_type][category][subcategory] = {}
            self.sound_definitions[asset_type][category][subcategory][sound_id] = sound_def
        else:
            if category not in self.sound_definitions[asset_type]:
                self.sound_definitions[asset_type][category] = {}
            self.sound_definitions[asset_type][category][sound_id] = sound_def
        
        self.save_sound_definitions()
        return sound_def
    
    def generate_sound(
        self,
        sound_id: str,
        template_vars: Dict = None,
        override_params: Dict = None
    ) -> str:
        """
        Generate a sound from a predefined sound definition.
        
        Args:
            sound_id: ID of the sound to generate
            template_vars: Variables to fill into the prompt template
            override_params: Parameters to override defaults
            
        Returns:
            Asset ID of the generated sound
        """
        # Find sound definition
        sound_def = self._find_sound_definition(sound_id)
        if not sound_def:
            raise ValueError(f"Sound definition '{sound_id}' not found")
        
        # Process template variables
        template_vars = template_vars or {}
        prompt = sound_def["prompt_template"]
        for var_name, var_value in template_vars.items():
            prompt = prompt.replace(f"{{{var_name}}}", var_value)
        
        # Merge default params with overrides
        params = dict(sound_def["default_params"])
        if override_params:
            params.update(override_params)
        
        # Generate the asset
        asset_id = self.pipeline.generate_asset(
            prompt=prompt,
            asset_type=sound_def["asset_type"],
            project=self.project_name,
            category=sound_def["category"],
            subcategory=sound_def["subcategory"],
            tags=sound_def["tags"],
            **params
        )
        
        # Add to variations in sound definition
        variation_info = {
            "asset_id": asset_id,
            "prompt": prompt,
            "template_vars": template_vars,
            "params": params,
            "created_at": datetime.now().isoformat()
        }
        sound_def["variations"].append(variation_info)
        self.save_sound_definitions()
        
        return asset_id
    
    def _find_sound_definition(self, sound_id):
        """Helper to find a sound definition by ID."""
        # Search through the nested structure
        for asset_type, categories in self.sound_definitions.items():
            if isinstance(categories, dict):
                for category, items in categories.items():
                    if isinstance(items, dict):
                        # Check if it's a mapping of sound_ids or subcategories
                        if sound_id in items:
                            return items[sound_id]
                        
                        # Check subcategories
                        for subcategory, sounds in items.items():
                            if isinstance(sounds, dict) and sound_id in sounds:
                                return sounds[sound_id]
        
        return None
    
    def generate_variations(
        self,
        sound_id: str,
        n_variations: int = 3,
        parameter_ranges: Dict = None,
        template_var_options: Dict = None
    ) -> List[str]:
        """
        Generate multiple variations of a sound.
        
        Args:
            sound_id: ID of the sound to generate
            n_variations: Number of variations to generate
            parameter_ranges: Ranges for parameters to vary
            template_var_options: Options for template variables
            
        Returns:
            List of asset IDs for the generated variations
        """
        sound_def = self._find_sound_definition(sound_id)
        if not sound_def:
            raise ValueError(f"Sound definition '{sound_id}' not found")
        
        asset_ids = []
        
        import random
        
        for i in range(n_variations):
            # Create parameter overrides from ranges
            override_params = {}
            if parameter_ranges:
                for param, value_range in parameter_ranges.items():
                    if isinstance(value_range, list) and len(value_range) == 2:
                        min_val, max_val = value_range
                        if isinstance(min_val, float):
                            override_params[param] = min_val + random.random() * (max_val - min_val)
                        elif isinstance(min_val, int):
                            override_params[param] = random.randint(min_val, max_val)
            
            # Select template variables
            template_vars = {}
            if template_var_options:
                for var_name, options in template_var_options.items():
                    template_vars[var_name] = random.choice(options)
            
            # Generate the variation
            asset_id = self.generate_sound(
                sound_id=sound_id,
                template_vars=template_vars,
                override_params=override_params
            )
            
            asset_ids.append(asset_id)
        
        return asset_ids

    def export_project_plan(self, output_file=None):
        """
        Export a project sound design plan.
        
        This creates a readable document showing all sound definitions
        and their relationships for project planning.
        """
        if output_file is None:
            output_file = os.path.join(self.project_dir, f"{self.project_name}_sound_plan.md")
        
        with open(output_file, 'w') as f:
            f.write(f"# Sound Design Plan: {self.project_name}\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Write out each category
            for asset_type, categories in self.sound_definitions.items():
                f.write(f"## {asset_type.upper()}\n\n")
                
                if isinstance(categories, dict):
                    for category, items in categories.items():
                        f.write(f"### {category}\n\n")
                        
                        if isinstance(items, dict):
                            # Check if it's a mapping of sound_ids or subcategories
                            subcategories = False
                            
                            for key, value in items.items():
                                if isinstance(value, dict) and "sound_id" not in value:
                                    subcategories = True
                            
                            if subcategories:
                                # Process subcategories
                                for subcategory, sounds in items.items():
                                    f.write(f"#### {subcategory}\n\n")
                                    if isinstance(sounds, dict):
                                        f.write("| Sound ID | Description | Prompt Template |\n")
                                        f.write("|---|---|---|\n")
                                        
                                        for sound_id, sound_def in sounds.items():
                                            if isinstance(sound_def, dict) and "sound_id" in sound_def:
                                                f.write(f"| {sound_def['sound_id']} | {sound_def['description']} | {sound_def['prompt_template']} |\n")
                                    f.write("\n")
                            else:
                                # Process direct sound IDs
                                f.write("| Sound ID | Description | Prompt Template |\n")
                                f.write("|---|---|---|\n")
                                
                                for sound_id, sound_def in items.items():
                                    if isinstance(sound_def, dict) and "sound_id" in sound_def:
                                        f.write(f"| {sound_def['sound_id']} | {sound_def['description']} | {sound_def['prompt_template']} |\n")
                                f.write("\n")
        
        print(f"Project plan exported to {output_file}")
        return output_file
```

## Building a Sound Library Workflow

Let's create a practical example of using our AudioPipeline for a game project, creating a structured sound design workflow:

```python
import os
import torch
from typing import Dict, List

# Helper function to clear memory after generation
def clear_memory():
    """Free up CUDA memory after generation."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()

def setup_game_sound_library(game_name, output_dir="game_projects"):
    """
    Set up a complete sound library structure for a game project.
    
    Args:
        game_name: Name of the game project
        output_dir: Base directory for game projects
    
    Returns:
        ProjectSoundDesigner: Configured sound designer for the project
    """
    # Create project sound designer
    designer = ProjectSoundDesigner(
        project_name=game_name,
        output_dir=output_dir,
        music_model_size="small",
        audio_model_size="medium"
    )
    
    # Define music categories
    designer.define_sound(
        sound_id="main_theme", 
        asset_type="music",
        category="themes",
        description="Main theme music for the game",
        prompt_template="Epic orchestral main theme for a {game_type} game. {mood} and {instrumentation} with memorable melody.",
        tags=["theme", "orchestral", "main"]
    )
    
    designer.define_sound(
        sound_id="combat_music", 
        asset_type="music",
        category="gameplay",
        description="Dynamic combat music",
        prompt_template="Intense combat music for {enemy_type} battle. {genre} style with {intensity} energy.",
        tags=["combat", "action", "intense"],
        default_params={"duration": 45.0, "temperature": 0.9, "cfg_coef": 3.5}
    )
    
    designer.define_sound(
        sound_id="exploration_ambient", 
        asset_type="music",
        category="gameplay",
        description="Ambient exploration music",
        prompt_template="Ambient exploration music for {environment} area. Peaceful and {mood}.",
        tags=["ambient", "exploration", "peaceful"],
        default_params={"duration": 60.0, "temperature": 1.0, "cfg_coef": 2.5}
    )
    
    # Define ambient sound categories
    designer.define_sound(
        sound_id="forest_ambience", 
        asset_type="ambience",
        category="environments",
        description="Forest ambient background",
        prompt_template="Daytime forest ambience with {density} bird calls, {wind_level} wind in leaves, and distant {wildlife}.",
        tags=["forest", "nature", "ambient"],
        default_params={"duration": 30.0, "temperature": 0.7, "cfg_coef": 5.0}
    )
    
    designer.define_sound(
        sound_id="dungeon_ambience", 
        asset_type="ambience",
        category="environments",
        description="Spooky dungeon ambient background",
        prompt_template="Dark dungeon ambient atmosphere with {water_presence} water drips, {wind_type} wind through corridors, and {tension} tension elements.",
        tags=["dungeon", "spooky", "ambient"],
        default_params={"duration": 30.0, "temperature": 0.7, "cfg_coef": 5.0}
    )
    
    # Define sound effect categories
    designer.define_sound(
        sound_id="sword_swing", 
        asset_type="sfx",
        category="character",
        subcategory="weapons",
        description="Sword swing sound effect",
        prompt_template="Sword swing whoosh effect, {sword_size} blade cutting through air with {speed} speed",
        tags=["weapon", "sword", "swing"],
        default_params={"duration": 1.5, "temperature": 0.6, "cfg_coef": 6.0}
    )
    
    designer.define_sound(
        sound_id="fireball_cast", 
        asset_type="sfx",
        category="character",
        subcategory="magic",
        description="Fireball spell casting sound",
        prompt_template="Fire magic spell casting sound, {intensity} energy building up with {element} energy release",
        tags=["magic", "fire", "spell"],
        default_params={"duration": 2.0, "temperature": 0.7, "cfg_coef": 5.0}
    )
    
    designer.define_sound(
        sound_id="footsteps_stone", 
        asset_type="sfx",
        category="character",
        subcategory="movement",
        description="Footsteps on stone surface",
        prompt_template="Character footsteps on stone surface, {speed} pace, {weight} character, {footwear} footwear",
        tags=["footsteps", "stone", "movement"],
        default_params={"duration": 3.0, "temperature": 0.5, "cfg_coef": 6.0}
    )
    
    designer.define_sound(
        sound_id="ui_button_click", 
        asset_type="sfx",
        category="ui",
        description="UI button click sound",
        prompt_template="User interface button {click_type} sound, {tone} tone, {material} material feeling",
        tags=["ui", "button", "click"],
        default_params={"duration": 0.5, "temperature": 0.5, "cfg_coef": 5.0}
    )
    
    # Generate project plan
    designer.export_project_plan()
    
    return designer

def generate_game_sound_library_sample(designer, n_samples=1):
    """
    Generate sample sounds for the game library.
    
    Args:
        designer: ProjectSoundDesigner instance
        n_samples: Number of samples to generate for each sound
    
    Returns:
        Dict: Generated asset IDs by sound ID
    """
    generated_assets = {}
    
    # Generate main theme variations
    generated_assets["main_theme"] = designer.generate_variations(
        sound_id="main_theme",
        n_variations=n_samples,
        parameter_ranges={"temperature": [0.7, 1.2]},
        template_var_options={
            "game_type": ["fantasy RPG", "epic adventure", "action RPG"],
            "mood": ["heroic", "mysterious", "triumphant"],
            "instrumentation": ["brass and strings", "full orchestra with choir", "orchestral with ethnic instruments"]
        }
    )
    clear_memory()
    
    # Generate combat music
    generated_assets["combat_music"] = designer.generate_variations(
        sound_id="combat_music",
        n_variations=n_samples,
        template_var_options={
            "enemy_type": ["boss", "horde", "dragon"],
            "genre": ["orchestral", "electronic", "hybrid"],
            "intensity": ["high", "medium", "extreme"]
        }
    )
    clear_memory()
    
    # Generate forest ambience
    generated_assets["forest_ambience"] = designer.generate_variations(
        sound_id="forest_ambience",
        n_variations=n_samples,
        template_var_options={
            "density": ["sparse", "moderate", "dense"],
            "wind_level": ["gentle", "moderate", "strong"],
            "wildlife": ["squirrels", "deer", "distant wolves"]
        }
    )
    clear_memory()
    
    # Generate sword effects
    generated_assets["sword_swing"] = designer.generate_variations(
        sound_id="sword_swing",
        n_variations=n_samples,
        template_var_options={
            "sword_size": ["short", "medium", "large"],
            "speed": ["fast", "very fast", "lightning fast"]
        }
    )
    clear_memory()
    
    # Generate UI sounds
    generated_assets["ui_button_click"] = designer.generate_variations(
        sound_id="ui_button_click",
        n_variations=n_samples,
        template_var_options={
            "click_type": ["click", "tap", "press"],
            "tone": ["bright", "warm", "digital"],
            "material": ["wooden", "metallic", "crystal"]
        }
    )
    clear_memory()
    
    return generated_assets

if __name__ == "__main__":
    # Create a complete game sound library
    game_designer = setup_game_sound_library("Epic Fantasy RPG")
    
    # Generate example sounds
    asset_ids = generate_game_sound_library_sample(game_designer, n_samples=1)
    
    # Print summary
    print(f"Generated {sum(len(v) for v in asset_ids.values())} sound assets")
    for sound_id, ids in asset_ids.items():
        print(f"- {sound_id}: {len(ids)} variations")
```

## Advanced Workflow: Scene-Based Generation

For more complex projects like games, films, or interactive experiences, organizing sound generation around scenes can be more effective. Let's implement a scene-based workflow:

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Tuple

@dataclass
class SoundSceneDefinition:
    """Definition of a complete audio scene for a game, film, or interactive experience."""
    
    scene_id: str
    scene_name: str
    description: str
    
    # Scene components
    ambience: Dict[str, Dict] = field(default_factory=dict)
    music: Optional[Dict] = None
    sound_effects: Dict[str, Dict] = field(default_factory=dict)
    
    # Scene properties
    duration: float = 30.0
    loop: bool = True
    transition_in: Optional[str] = None
    transition_out: Optional[str] = None
    
    # Scene organization
    tags: List[str] = field(default_factory=list)
    parent_scene: Optional[str] = None
    child_scenes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {k: v for k, v in self.__dict__.items()}

class SceneBasedWorkflow:
    """
    Workflow for organizing sound design around scenes or environments.
    
    This workflow is ideal for games, films, or interactive experiences
    where groups of sounds need to be organized by location, event, or
    narrative moment.
    """
    
    def __init__(
        self,
        project_name: str,
        output_dir: str = "projects",
        music_model_size: str = "small", 
        audio_model_size: str = "medium",
        device: str = None
    ):
        """
        Initialize a scene-based workflow.
        
        Args:
            project_name: Name of the project
            output_dir: Base directory for output
            music_model_size: Size of MusicGen model
            audio_model_size: Size of AudioGen model
            device: Device to run on
        """
        self.project_name = project_name
        self.project_dir = os.path.join(output_dir, project_name)
        os.makedirs(self.project_dir, exist_ok=True)
        
        # Initialize audio pipeline
        self.pipeline = AudioPipeline(
            base_output_dir=os.path.join(self.project_dir, "assets"),
            music_model_size=music_model_size,
            audio_model_size=audio_model_size,
            device=device
        )
        
        # Scene definitions
        self.scenes_dir = os.path.join(self.project_dir, "scenes")
        os.makedirs(self.scenes_dir, exist_ok=True)
        self.scenes = {}
    
    def create_scene(
        self,
        scene_id: str,
        scene_name: str,
        description: str,
        tags: List[str] = None,
        parent_scene: str = None,
        duration: float = 30.0,
        loop: bool = True
    ) -> SoundSceneDefinition:
        """
        Create a new scene definition.
        
        Args:
            scene_id: Unique identifier for the scene
            scene_name: Human-readable name
            description: Description of the scene
            tags: List of tags for organization
            parent_scene: Optional parent scene ID
            duration: Default duration for the scene
            loop: Whether the scene should loop
            
        Returns:
            SoundSceneDefinition object for the scene
        """
        scene = SoundSceneDefinition(
            scene_id=scene_id,
            scene_name=scene_name,
            description=description,
            tags=tags or [],
            parent_scene=parent_scene,
            duration=duration,
            loop=loop
        )
        
        # Link to parent scene if specified
        if parent_scene and parent_scene in self.scenes:
            if scene_id not in self.scenes[parent_scene].child_scenes:
                self.scenes[parent_scene].child_scenes.append(scene_id)
        
        # Save scene definition
        self.scenes[scene_id] = scene
        self._save_scene(scene)
        
        return scene
    
    def _save_scene(self, scene: SoundSceneDefinition):
        """Save a scene definition to disk."""
        scene_path = os.path.join(self.scenes_dir, f"{scene.scene_id}.json")
        with open(scene_path, 'w') as f:
            json.dump(scene.to_dict(), f, indent=2)
    
    def load_scene(self, scene_id: str) -> SoundSceneDefinition:
        """Load a scene definition from disk."""
        scene_path = os.path.join(self.scenes_dir, f"{scene_id}.json")
        if not os.path.exists(scene_path):
            raise ValueError(f"Scene '{scene_id}' not found")
        
        with open(scene_path, 'r') as f:
            scene_data = json.load(f)
        
        scene = SoundSceneDefinition(**scene_data)
        self.scenes[scene_id] = scene
        return scene
    
    def add_ambience_to_scene(
        self,
        scene_id: str,
        ambience_id: str,
        prompt: str,
        weight: float = 1.0,
        parameters: Dict = None
    ):
        """
        Add an ambience layer to a scene.
        
        Args:
            scene_id: ID of the scene to modify
            ambience_id: ID for this ambient sound
            prompt: Text prompt for generation
            weight: Mix weight relative to other ambiences
            parameters: Generation parameters
            
        Returns:
            Updated scene definition
        """
        if scene_id not in self.scenes:
            raise ValueError(f"Scene '{scene_id}' not found")
        
        scene = self.scenes[scene_id]
        
        # Default parameters for ambience
        default_params = {
            "duration": scene.duration,
            "temperature": 0.7,
            "cfg_coef": 5.0,
            "top_k": 250
        }
        
        # Merge with provided parameters
        if parameters:
            default_params.update(parameters)
        
        # Add ambience to scene
        scene.ambience[ambience_id] = {
            "prompt": prompt,
            "parameters": default_params,
            "weight": weight,
            "asset_id": None  # Will be populated when generated
        }
        
        self._save_scene(scene)
        return scene
    
    def add_music_to_scene(
        self,
        scene_id: str,
        prompt: str,
        parameters: Dict = None
    ):
        """
        Add music to a scene.
        
        Args:
            scene_id: ID of the scene to modify
            prompt: Text prompt for generation
            parameters: Generation parameters
            
        Returns:
            Updated scene definition
        """
        if scene_id not in self.scenes:
            raise ValueError(f"Scene '{scene_id}' not found")
        
        scene = self.scenes[scene_id]
        
        # Default parameters for music
        default_params = {
            "duration": scene.duration,
            "temperature": 1.0,
            "cfg_coef": 3.0,
            "top_k": 250
        }
        
        # Merge with provided parameters
        if parameters:
            default_params.update(parameters)
        
        # Add music to scene
        scene.music = {
            "prompt": prompt,
            "parameters": default_params,
            "asset_id": None  # Will be populated when generated
        }
        
        self._save_scene(scene)
        return scene
    
    def add_sound_effect_to_scene(
        self,
        scene_id: str,
        sfx_id: str,
        prompt: str,
        category: str,
        parameters: Dict = None
    ):
        """
        Add a sound effect to a scene.
        
        Args:
            scene_id: ID of the scene to modify
            sfx_id: ID for this sound effect
            prompt: Text prompt for generation
            category: Category for organization
            parameters: Generation parameters
            
        Returns:
            Updated scene definition
        """
        if scene_id not in self.scenes:
            raise ValueError(f"Scene '{scene_id}' not found")
        
        scene = self.scenes[scene_id]
        
        # Default parameters for sound effects
        default_params = {
            "duration": 3.0,  # Short duration for SFX
            "temperature": 0.6,
            "cfg_coef": 6.0,
            "top_k": 50
        }
        
        # Merge with provided parameters
        if parameters:
            default_params.update(parameters)
        
        # Add sound effect to scene
        scene.sound_effects[sfx_id] = {
            "prompt": prompt,
            "category": category,
            "parameters": default_params,
            "asset_id": None  # Will be populated when generated
        }
        
        self._save_scene(scene)
        return scene
    
    def generate_scene_assets(self, scene_id: str) -> Dict:
        """
        Generate all assets for a scene.
        
        Args:
            scene_id: ID of the scene to generate
            
        Returns:
            Dictionary mapping component IDs to asset IDs
        """
        if scene_id not in self.scenes:
            self.load_scene(scene_id)
        
        scene = self.scenes[scene_id]
        asset_ids = {}
        
        # Generate ambience layers
        for ambience_id, ambience_def in scene.ambience.items():
            asset_id = self.pipeline.generate_asset(
                prompt=ambience_def["prompt"],
                asset_type="ambience",
                project=self.project_name,
                category=scene_id,
                tags=["ambience", scene_id],
                **ambience_def["parameters"]
            )
            
            # Update scene definition with asset ID
            scene.ambience[ambience_id]["asset_id"] = asset_id
            asset_ids[f"ambience_{ambience_id}"] = asset_id
            
            # Clear memory after generation
            clear_memory()
        
        # Generate music if defined
        if scene.music:
            asset_id = self.pipeline.generate_asset(
                prompt=scene.music["prompt"],
                asset_type="music",
                project=self.project_name,
                category=scene_id,
                tags=["music", scene_id],
                **scene.music["parameters"]
            )
            
            # Update scene definition with asset ID
            scene.music["asset_id"] = asset_id
            asset_ids["music"] = asset_id
            
            # Clear memory after generation
            clear_memory()
        
        # Generate sound effects
        for sfx_id, sfx_def in scene.sound_effects.items():
            asset_id = self.pipeline.generate_asset(
                prompt=sfx_def["prompt"],
                asset_type="sfx",
                project=self.project_name,
                category=sfx_def["category"],
                subcategory=scene_id,
                tags=["sfx", scene_id, sfx_def["category"]],
                **sfx_def["parameters"]
            )
            
            # Update scene definition with asset ID
            scene.sound_effects[sfx_id]["asset_id"] = asset_id
            asset_ids[f"sfx_{sfx_id}"] = asset_id
            
            # Clear memory after generation
            clear_memory()
        
        # Save updated scene definition
        self._save_scene(scene)
        
        return asset_ids
    
    def export_scene_graph(self, output_file=None):
        """
        Export a graph visualization of all scenes.
        
        Returns:
            Path to the generated graph file
        """
        if output_file is None:
            output_file = os.path.join(self.project_dir, f"{self.project_name}_scene_graph.md")
        
        with open(output_file, 'w') as f:
            f.write(f"# Scene Graph: {self.project_name}\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Write main scene list
            f.write("## Scenes\n\n")
            f.write("| Scene ID | Name | Description | Assets |\n")
            f.write("|---|---|---|---|\n")
            
            for scene_id, scene in self.scenes.items():
                # Count assets
                asset_count = len(scene.ambience)
                asset_count += 1 if scene.music else 0
                asset_count += len(scene.sound_effects)
                
                f.write(f"| {scene.scene_id} | {scene.scene_name} | {scene.description} | {asset_count} |\n")
            
            f.write("\n## Scene Hierarchy\n\n")
            
            # Find root scenes (no parent)
            root_scenes = [scene_id for scene_id, scene in self.scenes.items() if not scene.parent_scene]
            
            # Write scene hierarchy using indentation
            for root_scene_id in root_scenes:
                self._write_scene_hierarchy(f, root_scene_id, 0)
        
        print(f"Scene graph exported to {output_file}")
        return output_file
    
    def _write_scene_hierarchy(self, file, scene_id, level):
        """Recursively write scene hierarchy with proper indentation."""
        scene = self.scenes[scene_id]
        indent = "  " * level
        file.write(f"{indent}- {scene.scene_name} ({scene_id})\n")
        
        # Write child scenes
        for child_id in scene.child_scenes:
            if child_id in self.scenes:
                self._write_scene_hierarchy(file, child_id, level + 1)

def create_game_scene_workflow():
    """
    Create a complete scene-based workflow for a game project.
    
    Returns:
        SceneBasedWorkflow: Configured workflow
    """
    # Create workflow
    workflow = SceneBasedWorkflow(
        project_name="Fantasy RPG Scene Demo",
        music_model_size="small",
        audio_model_size="medium"
    )
    
    # Create main environment scenes
    forest_scene = workflow.create_scene(
        scene_id="forest",
        scene_name="Enchanted Forest",
        description="Mystical forest environment with magical elements",
        tags=["environment", "outdoor", "forest"],
        duration=45.0
    )
    
    castle_scene = workflow.create_scene(
        scene_id="castle",
        scene_name="Ancient Castle",
        description="Imposing stone castle with gothic architecture",
        tags=["environment", "indoor", "dungeon"],
        duration=45.0
    )
    
    village_scene = workflow.create_scene(
        scene_id="village",
        scene_name="Medieval Village",
        description="Bustling village with markets and townspeople",
        tags=["environment", "outdoor", "settlement"],
        duration=60.0
    )
    
    # Create sub-scenes (state variations)
    workflow.create_scene(
        scene_id="forest_night",
        scene_name="Enchanted Forest - Night",
        description="Mystical forest at night with eerie atmosphere",
        tags=["environment", "outdoor", "forest", "night"],
        parent_scene="forest",
        duration=45.0
    )
    
    workflow.create_scene(
        scene_id="castle_battle",
        scene_name="Castle Under Siege",
        description="Castle during battle with chaos and fighting",
        tags=["environment", "indoor", "battle"],
        parent_scene="castle",
        duration=60.0
    )
    
    # Add ambient sounds to forest scene
    workflow.add_ambience_to_scene(
        scene_id="forest",
        ambience_id="forest_base",
        prompt="Daytime forest environment with birds chirping, leaves rustling, and occasional wildlife sounds",
        weight=1.0
    )
    
    workflow.add_ambience_to_scene(
        scene_id="forest",
        ambience_id="magic_atmosphere",
        prompt="Subtle magical atmosphere with soft wind chimes, gentle sparkles, and ethereal tones",
        weight=0.6,
        parameters={"temperature": 0.8}
    )
    
    # Add music to forest scene
    workflow.add_music_to_scene(
        scene_id="forest",
        prompt="Peaceful fantasy music for an enchanted forest. Ethereal flutes, soft strings, and harps with a mystical quality. Calm and magical.",
        parameters={
            "duration": 60.0,
            "temperature": 1.0,
            "cfg_coef": 3.0
        }
    )
    
    # Add sound effects to forest scene
    workflow.add_sound_effect_to_scene(
        scene_id="forest",
        sfx_id="magic_sparkle",
        category="magic",
        prompt="Magical sparkle effect with gentle chimes and crystalline sounds",
        parameters={"duration": 3.0}
    )
    
    workflow.add_sound_effect_to_scene(
        scene_id="forest",
        sfx_id="branch_break",
        category="environment",
        prompt="Tree branch breaking and falling, with leaves rustling and wood cracking",
        parameters={"duration": 2.0}
    )
    
    # Add ambient sounds to castle scene
    workflow.add_ambience_to_scene(
        scene_id="castle",
        ambience_id="castle_interior",
        prompt="Stone castle interior ambience with distant echoes, wind through corridors, and occasional creaking wood",
        weight=1.0
    )
    
    workflow.add_ambience_to_scene(
        scene_id="castle",
        ambience_id="dungeon_drips",
        prompt="Water dripping in stone dungeon with occasional distant moans and creaking chains",
        weight=0.7,
        parameters={"temperature": 0.6}
    )
    
    # Add music to castle scene
    workflow.add_music_to_scene(
        scene_id="castle",
        prompt="Dark, ominous castle theme. Slow, brooding strings and haunting choir. Gothic atmosphere with occasional brass stabs for tension.",
        parameters={
            "duration": 60.0,
            "temperature": 0.9,
            "cfg_coef": 3.5
        }
    )
    
    # Generate scene graph
    workflow.export_scene_graph()
    
    return workflow

if __name__ == "__main__":
    # Create scene workflow
    scene_workflow = create_game_scene_workflow()
    
    # Generate assets for forest scene
    forest_assets = scene_workflow.generate_scene_assets("forest")
    print(f"Generated {len(forest_assets)} assets for forest scene")
```

## Sound Design Workflow Best Practices

Based on our implementations, here are key best practices for effective sound design workflows with AudioCraft:

1. **Pipeline Architecture**
   - Create unified pipelines that can handle both music and sound effects
   - Establish consistent interfaces for all audio generation tasks
   - Build reusable components that can be combined in different ways

2. **Metadata Management**
   - Maintain comprehensive metadata for all generated assets
   - Include generation parameters to enable reproducing similar sounds
   - Use standardized tags for effective searching and filtering

3. **Organization Strategies**
   - Group assets by project, type, category, and subcategory
   - Create logical hierarchies for complex projects
   - Use scene-based organization for spatial or narrative contexts

4. **Parameterization**
   - Create prompt templates with variable placeholders
   - Define parameter ranges for systematic variation
   - Store optimal parameters for different sound types

5. **Batch Processing**
   - Generate variations in batches to compare alternatives
   - Use memory management techniques to handle large batches
   - Implement caching for frequently used models

6. **Project Planning**
   - Create sound definition documents before generation
   - Establish a common vocabulary for sound descriptions
   - Document all sound assets and their relationships

7. **Resource Optimization**
   - Load models on demand to minimize memory usage
   - Clear memory after large generation tasks
   - Batch similar generation tasks for efficiency

## Hands-On Challenge: Create a Game Audio Pipeline

**Challenge:** Create a complete audio generation pipeline for a game project with the following requirements:

1. A structured workflow that organizes sounds by game areas and events
2. Systematic generation of variants for each sound type
3. A metadata system that links sounds to their game contexts
4. Memory-efficient generation that works on modest hardware
5. Documentation for sound designers to understand the organization

**Steps to complete:**

1. Extend the `ProjectSoundDesigner` class with game-specific organization
2. Create a template system for standardized asset naming
3. Implement variation generation with parameter exploration
4. Add metadata fields for game integration (e.g., triggering events)
5. Create reporting tools to document the audio design system

## Next Steps

In the next chapter, we'll explore integrating AudioCraft-generated content with game engines like Unity. We'll build systems that connect our sound design workflows to interactive environments, creating dynamic sound systems that respond to player actions and game states.

Copyright © 2025 Scott Friedman. Licensed under [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).
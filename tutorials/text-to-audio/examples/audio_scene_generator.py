#!/usr/bin/env python3
# audio_scene_generator.py - Create layered audio scenes with multiple components
import json
import os
import torch
import argparse
import sys

# Add the parent directory to the path to import our custom pipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from examples.text_to_audio_pipeline import TextToAudioPipeline

def generate_audio_scene(scene_description, output_dir="scenes"):
    """
    Generate a complete audio scene from a structured JSON description.
    
    This function creates complex, layered audio environments by generating
    multiple audio components (music and sound effects) and mixing them together.
    It follows a declarative approach where the scene is defined as a data structure
    with multiple components, each with their own generation parameters.
    
    Args:
        scene_description (dict): Scene description containing:
            - name: Name of the scene
            - components: List of audio components, each with:
                - name: Component name
                - type: "music" or "audio" (sound effects)
                - prompt: Text description
                - duration: Length in seconds
                - temperature: Randomness parameter
                - weight: Mixing weight in final scene
                
        output_dir (str): Base directory to save output files
        
    Returns:
        str: Path to the directory containing all generated files
        
    Example scene_description:
    {
        "name": "Peaceful Forest",
        "components": [
            {
                "name": "background_music",
                "type": "music",
                "prompt": "Gentle ambient music with piano", 
                "duration": 15.0,
                "temperature": 0.7,
                "weight": 0.6
            },
            {
                "name": "forest_sounds",
                "type": "audio",
                "prompt": "Forest with birds and stream",
                "duration": 15.0,
                "temperature": 1.0,
                "weight": 0.8
            }
        ]
    }
    """
    # Create base output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a specific directory for this scene based on its name
    scene_name = scene_description.get("name", "unnamed_scene")
    safe_name = "".join(c if c.isalnum() or c == "_" else "_" for c in scene_name)
    scene_dir = os.path.join(output_dir, safe_name)
    os.makedirs(scene_dir, exist_ok=True)
    
    print(f"Generating scene: {scene_name}")
    
    # Save the scene description as JSON for reference
    with open(os.path.join(scene_dir, "scene_description.json"), "w") as f:
        json.dump(scene_description, f, indent=2)
    
    # Initialize our audio pipeline
    pipeline = TextToAudioPipeline()
    
    # Analyze the scene to determine which models we need to load
    # This prevents loading models we won't use
    models_needed = set()
    for component in scene_description.get("components", []):
        models_needed.add(component["type"])
    
    # Load only the required models with appropriate sizes
    print("Loading required models...")
    for model_type in models_needed:
        # Use smaller music model for faster generation
        size = "small" if model_type == "music" else "medium"
        pipeline.load_model(model_type, size)
    
    # Track all generated component audios and their mixing weights
    component_audios = []
    component_weights = []
    component_count = len(scene_description.get("components", []))
    
    # Generate each audio component in the scene
    for i, component in enumerate(scene_description.get("components", [])):
        # Extract component parameters
        component_type = component["type"]  # "music" or "audio"
        prompt = component["prompt"]  # Text description
        duration = component.get("duration", 10.0)  # Default 10 seconds
        temperature = component.get("temperature", 1.0)  # Default 1.0
        weight = component.get("weight", 1.0)  # Default equal weighting
        
        # Display progress
        print(f"Generating component {i+1}/{component_count}: {component_type}")
        print(f"Prompt: '{prompt}'")
        
        # Generate audio for this component
        audio = pipeline.generate(prompt, component_type, duration, temperature)
        
        # Save this individual component
        component_name = component.get("name", f"{component_type}_{i+1}")
        safe_component_name = "".join(c if c.isalnum() or c == "_" else "_" for c in component_name)
        component_path = pipeline.save_audio(audio, safe_component_name, scene_dir)
        print(f"Saved component to {component_path}")
        
        # Add to our collection for later mixing
        component_audios.append(audio)
        component_weights.append(weight)
    
    # Mix all components together if we have any
    if component_audios:
        print(f"Mixing {len(component_audios)} components together...")
        
        # Create the mixed audio scene
        mixed_audio = pipeline.mix_audio(component_audios, component_weights)
        
        # Save the final mixed scene
        final_path = pipeline.save_audio(mixed_audio, f"{safe_name}_complete", scene_dir)
        
        # Create a README with scene details
        with open(os.path.join(scene_dir, "README.txt"), "w") as f:
            f.write(f"AUDIO SCENE: {scene_name}\n")
            f.write("=" * (len(scene_name) + 13) + "\n\n")
            f.write("Components:\n")
            
            for i, component in enumerate(scene_description.get("components", [])):
                f.write(f"{i+1}. {component.get('name', f'Component {i+1}')}\n")
                f.write(f"   Type: {component['type']}\n")
                f.write(f"   Prompt: \"{component['prompt']}\"\n")
                f.write(f"   Weight in mix: {component.get('weight', 1.0)}\n\n")
            
            f.write(f"Final mixed scene saved as: {os.path.basename(final_path)}\n")
        
        print(f"Scene generation complete! All files saved to {scene_dir}")
        return scene_dir
    else:
        print("No components to mix! Scene creation failed.")
        return None

# Example scene descriptions
# These demonstrate how to structure scene data for different environments
scenes = [
    {
        "name": "Peaceful Forest Morning",
        "components": [
            {
                "name": "background_music",
                "type": "music",
                "prompt": "Gentle ambient music with soft piano and nature-inspired melodies, peaceful and calm", 
                "duration": 15.0,
                "temperature": 0.7,  # Lower temperature for more consistent, gentle music
                "weight": 0.6        # Music at 60% volume in the final mix
            },
            {
                "name": "forest_ambience",
                "type": "audio",
                "prompt": "Forest ambience with birds chirping, gentle breeze through leaves, and distant stream flowing over rocks",
                "duration": 15.0,
                "temperature": 1.0,  # Standard temperature for natural variation in ambience
                "weight": 0.8        # Nature sounds at 80% volume (dominant element)
            }
        ]
    },
    {
        "name": "Busy City Cafe",
        "components": [
            {
                "name": "jazz_music",
                "type": "music",
                "prompt": "Smooth jazz cafe music with piano, upright bass, and soft brush drums, relaxed tempo",
                "duration": 15.0,
                "temperature": 0.8,  # Moderate temperature for coherent jazz with some variation
                "weight": 0.5        # Music at 50% volume (background element)
            },
            {
                "name": "cafe_ambience",
                "type": "audio",
                "prompt": "Busy cafe with conversations, coffee machines hissing, cups clinking, and chairs moving occasionally",
                "duration": 15.0,
                "temperature": 1.0,  # Standard temperature for natural ambience
                "weight": 0.7        # Cafe sounds at 70% volume (primary ambience)
            },
            {
                "name": "outside_traffic",
                "type": "audio",
                "prompt": "Distant city traffic and occasional car horns through windows, muffled by glass",
                "duration": 15.0,
                "temperature": 0.9,  # Slight randomization
                "weight": 0.3        # Traffic at 30% volume (subtle background element)
            }
        ]
    },
    # Users can add their own scene descriptions here!
]

if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Generate layered audio scenes from multiple components",
        epilog="Example: python audio_scene_generator.py --scene 1 --output my_scenes"
    )
    
    parser.add_argument(
        "--scene", 
        type=int, 
        default=0, 
        choices=range(len(scenes)),
        help=f"Scene index to generate (0: Forest, 1: Cafe) [0-{len(scenes)-1}]"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default="scenes",
        help="Output directory where scene files will be saved"
    )
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Generate the selected scene
    scene_dir = generate_audio_scene(scenes[args.scene], args.output)
    
    if scene_dir:
        print("\nSuccessfully created audio scene!")
        print(f"All audio files are in: {scene_dir}")
        print("The complete mixed scene has '_complete' in the filename.")
        print("\nCreate your own scenes by modifying the 'scenes' list in this script!")
    else:
        print("Scene generation failed. Check for errors above.")
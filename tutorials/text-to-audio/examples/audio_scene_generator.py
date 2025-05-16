#!/usr/bin/env python3
# audio_scene_generator.py
import json
import os
import torch
import argparse
import sys

# Add the parent directory to the path to import the pipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from examples.text_to_audio_pipeline import TextToAudioPipeline

def generate_audio_scene(scene_description, output_dir="scenes"):
    """
    Generate a complete audio scene from a structured description.
    
    Args:
        scene_description (dict): Scene description with components
        output_dir (str): Directory to save output files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output scene directory
    scene_name = scene_description.get("name", "unnamed_scene")
    safe_name = "".join(c if c.isalnum() or c == "_" else "_" for c in scene_name)
    scene_dir = os.path.join(output_dir, safe_name)
    os.makedirs(scene_dir, exist_ok=True)
    
    # Initialize the pipeline
    pipeline = TextToAudioPipeline()
    
    # Load required models based on the scene components
    models_needed = set()
    for component in scene_description.get("components", []):
        models_needed.add(component["type"])
    
    for model_type in models_needed:
        size = "small" if model_type == "music" else "medium"
        pipeline.load_model(model_type, size)
    
    # Generate each component
    component_audios = []
    component_weights = []
    
    for i, component in enumerate(scene_description.get("components", [])):
        component_type = component["type"]
        prompt = component["prompt"]
        duration = component.get("duration", 10.0)
        temperature = component.get("temperature", 1.0)
        weight = component.get("weight", 1.0)
        
        print(f"Generating component {i+1}/{len(scene_description.get('components', []))}: {component_type}")
        
        # Generate audio for this component
        audio = pipeline.generate(prompt, component_type, duration, temperature)
        
        # Save individual component
        component_name = component.get("name", f"{component_type}_{i+1}")
        safe_component_name = "".join(c if c.isalnum() or c == "_" else "_" for c in component_name)
        pipeline.save_audio(audio, safe_component_name, scene_dir)
        
        # Add to the list for mixing
        component_audios.append(audio)
        component_weights.append(weight)
    
    # Mix all components together
    if component_audios:
        print("Mixing components together...")
        mixed_audio = pipeline.mix_audio(component_audios, component_weights)
        
        # Save the mixed scene
        pipeline.save_audio(mixed_audio, f"{safe_name}_complete", scene_dir)
        
        print(f"Scene generation complete! Files saved to {scene_dir}")

# Example scene descriptions
scenes = [
    {
        "name": "Peaceful Forest Morning",
        "components": [
            {
                "name": "background_music",
                "type": "music",
                "prompt": "Gentle ambient music with soft piano and nature-inspired melodies", 
                "duration": 15.0,
                "temperature": 0.7,
                "weight": 0.6
            },
            {
                "name": "forest_ambience",
                "type": "audio",
                "prompt": "Forest ambience with birds chirping, gentle breeze through leaves, and distant stream",
                "duration": 15.0,
                "temperature": 1.0,
                "weight": 0.8
            }
        ]
    },
    {
        "name": "Busy City Cafe",
        "components": [
            {
                "name": "jazz_music",
                "type": "music",
                "prompt": "Smooth jazz cafe music with piano, bass, and soft drums",
                "duration": 15.0,
                "temperature": 0.8,
                "weight": 0.5
            },
            {
                "name": "cafe_ambience",
                "type": "audio",
                "prompt": "Busy cafe with conversations, coffee machines, cups clinking, and chairs moving",
                "duration": 15.0,
                "temperature": 1.0,
                "weight": 0.7
            },
            {
                "name": "outside_traffic",
                "type": "audio",
                "prompt": "Distant city traffic and occasional car horns through windows",
                "duration": 15.0,
                "temperature": 0.9,
                "weight": 0.3
            }
        ]
    }
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate audio scenes")
    parser.add_argument("--scene", type=int, default=0, choices=[0, 1],
                        help="Scene index to generate (0: Forest, 1: Cafe)")
    parser.add_argument("--output", type=str, default="scenes",
                        help="Output directory")
    
    args = parser.parse_args()
    
    # Generate the selected scene
    generate_audio_scene(scenes[args.scene], args.output)
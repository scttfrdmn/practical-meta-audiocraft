#!/usr/bin/env python3
# audiogen_category_explorer.py
import torch
import os
from audiocraft.models import AudioGen
from audiocraft.data.audio import audio_write

def explore_sound_categories(model_size="medium", duration=5.0, output_dir="sound_categories"):
    """
    Generate samples across different sound categories.
    
    Args:
        model_size (str): Size of model to use
        duration (float): Length of each sample in seconds
        output_dir (str): Directory to save samples
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define category prompts
    categories = {
        "nature_rain": "Heavy rain falling on leaves with occasional thunder in the distance",
        "nature_wind": "Strong wind blowing through trees with leaves rustling",
        "nature_water": "Gentle stream flowing over rocks with water bubbling and splashing",
        "nature_fire": "Crackling campfire with wood popping and embers hissing",
        
        "urban_traffic": "Busy city intersection with cars honking, engines revving, and traffic signals beeping",
        "urban_construction": "Construction site with jackhammers, power drills, and workers shouting",
        "urban_cafe": "Busy cafe with coffee machines, quiet conversations, and clinking dishes",
        "urban_subway": "Subway train arriving at station with screeching brakes and announcement",
        
        "household_kitchen": "Kitchen sounds with cutting vegetables, water running, and microwave beeping",
        "household_bathroom": "Shower running with water spray hitting tiles and drain gurgling",
        "household_livingroom": "Living room with TV playing, clock ticking, and occasional footsteps",
        "household_yard": "Lawnmower running with grass cutting and birds in background",
        
        "animals_birds": "Forest birds chirping, calling, and fluttering their wings",
        "animals_dogs": "Dogs barking, panting, and playing with toys",
        "animals_insects": "Crickets chirping and buzzing insects flying around",
        "animals_farm": "Farm animals with cows mooing, chickens clucking, and sheep bleating",
        
        "mechanical_engine": "Car engine starting, idling, and revving",
        "mechanical_factory": "Factory machinery with conveyor belts, pneumatic presses, and motors",
        "mechanical_tools": "Power tools with electric drill, circular saw, and sander",
        "mechanical_office": "Office equipment with printers, keyboards typing, and telephones ringing"
    }
    
    # Determine device
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS (Metal) for generation")
    elif torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA for generation")
    else:
        device = "cpu"
        print("Using CPU for generation (this will be slow)")
    
    # Load model
    print(f"Loading AudioGen {model_size} model...")
    model = AudioGen.get_pretrained(model_size)
    model.to(device)
    
    # Set generation parameters
    model.set_generation_params(
        duration=duration,
        temperature=1.0,
        top_k=250,
        top_p=0.0,
    )
    
    # Generate sound for each category
    for category_name, prompt in categories.items():
        print(f"Generating {category_name} sound...")
        
        # Generate
        wav = model.generate([prompt])
        
        # Save
        output_path = os.path.join(output_dir, category_name)
        audio_write(
            output_path,
            wav[0].cpu(),
            model.sample_rate,
            strategy="loudness"
        )
        
        print(f"Saved {category_name} sample to {output_path}.wav")
    
    print("Category exploration complete! All samples saved to", output_dir)

if __name__ == "__main__":
    explore_sound_categories(model_size="medium", duration=5.0)
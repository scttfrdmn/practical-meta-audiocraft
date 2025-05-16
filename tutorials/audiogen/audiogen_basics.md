# AudioGen Basics: Sound Effect Generation with AudioCraft

This tutorial introduces Meta's AudioGen model and how to use it for generating environmental sounds and effects from text descriptions. We'll explore the basic functionality and parameters that control the generation process.

## Introduction to AudioGen

AudioGen is a text-to-audio model designed to generate realistic sound effects and environmental audio from textual descriptions. Unlike MusicGen, which focuses on musical content, AudioGen specializes in creating non-musical sounds such as:

- Environmental sounds (rain, wind, forest ambience)
- Sound effects (footsteps, door creaks, explosions)
- Machine sounds (engines, appliances, vehicles)
- Animal sounds (birds, dogs, insects)
- Human non-speech sounds (laughter, coughing, clapping)

AudioGen was trained on a diverse dataset of sound effects and can generate high-quality audio across many categories of sounds.

## Getting Started

Before we begin, make sure you have AudioCraft installed. If not, refer to the [Getting Started Guide](../getting-started/README.md).

Let's first create a simple script to generate a sound effect from a text prompt:

```python
# audiogen_basic.py
import torch
import torchaudio
from audiocraft.models import AudioGen
from audiocraft.data.audio import audio_write
import time
import os

def generate_sound(
    prompt,
    duration=5.0,
    model_size="medium",
    output_dir="sound_output",
    temperature=1.0
):
    """
    Generate a sound effect based on a text prompt.
    
    Args:
        prompt (str): Text description of the sound to generate
        duration (float): Length of audio in seconds
        model_size (str): Size of model to use ("medium" or "large")
        output_dir (str): Directory to save output files
        temperature (float): Controls randomness (higher = more random)
    
    Returns:
        str: Path to the generated audio file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating sound for prompt: '{prompt}'")
    print(f"Using model size: {model_size}, duration: {duration}s")
    
    start_time = time.time()
    
    # Determine device (MPS for Mac, CUDA for NVIDIA, CPU otherwise)
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS (Metal) for generation")
    elif torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA for generation")
    else:
        device = "cpu"
        print("Using CPU for generation (this will be slow)")
    
    # Load the model
    model = AudioGen.get_pretrained(model_size)
    model.to(device)
    
    # Set generation parameters
    model.set_generation_params(
        duration=duration,
        temperature=temperature,
        top_k=250,
        top_p=0.0,
    )
    
    # Generate sound
    wav = model.generate([prompt])
    
    # Create a filename based on the prompt
    safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt)[:30]
    filename = f"{safe_prompt}_{model_size}_{duration}s"
    
    # Save the audio file
    output_path = os.path.join(output_dir, filename)
    audio_write(
        output_path, 
        wav[0].cpu(), 
        model.sample_rate, 
        strategy="loudness",
    )
    
    generation_time = time.time() - start_time
    print(f"Generation completed in {generation_time:.2f} seconds")
    print(f"Audio saved to {output_path}.wav")
    
    return f"{output_path}.wav"

if __name__ == "__main__":
    # Example prompts to try
    prompts = [
        "Heavy rain falling on a metal roof with occasional thunder",
        "Crackling campfire with wood popping and occasional owl hoots",
        "Busy city street with cars passing, people talking, and sirens in the distance",
        "Forest ambience with birds chirping, leaves rustling, and a stream flowing",
        "Old mechanical clock ticking with gears turning and a soft bell chime"
    ]
    
    # Generate sound for the first prompt
    # Change the index to try different prompts
    generate_sound(
        prompt=prompts[0],
        duration=5.0,  # 5 seconds
        model_size="medium",  # Use "medium" or "large"
        temperature=1.0  # Default creativity level
    )
```

Save this script as `audiogen_basic.py` and run it to generate your first sound effect with AudioGen!

## Understanding the Key Parameters

Like MusicGen, AudioGen has several parameters that control its output:

### Model Size

AudioGen comes in two sizes:
- **medium**: Balanced generation time and quality
- **large**: Highest quality output but slower and requires more memory

Note that unlike MusicGen, AudioGen does not have a "small" model.

### Duration

The `duration` parameter controls how long the generated audio will be, in seconds. For sound effects, shorter durations (3-5 seconds) often work well for discrete sounds, while longer durations (10-30 seconds) are better for ambient backgrounds.

### Temperature

Temperature (between 0.0 and 2.0) controls the randomness of the generation:
- **Lower values** (0.1-0.5): More predictable, consistent sounds
- **Medium values** (0.6-1.0): Balanced variability
- **Higher values** (1.1-2.0): More experimental, diverse sounds

### Top-k and Top-p Sampling

These parameters control the diversity of the generation:

- **top_k**: Limits sampling to the k most likely tokens at each step
- **top_p**: Uses nucleus sampling to dynamically select the most likely tokens (set to 0.0 to disable)

## Crafting Effective Sound Prompts

Creating good prompts for AudioGen is crucial for getting the desired results. Here are some strategies:

### Be Specific and Descriptive

Instead of vague descriptions, use detailed and specific language:

```python
# Vague vs. Specific prompts
vague_prompts = [
    "Rain",
    "Wind",
    "Animal sounds",
    "Machine"
]

specific_prompts = [
    "Heavy rain falling on a metal roof with water dripping from gutters",
    "Strong wind howling through pine trees on a mountain ridge",
    "A pack of wolves howling in the distance with occasional barks",
    "Industrial machine with rhythmic metallic clanking and hydraulic hisses"
]
```

### Include Sound Qualities

Describe acoustic characteristics of the sound:

```python
# Sound quality descriptors
quality_prompts = [
    "Deep, resonant drum beats echoing in a cave",
    "High-pitched, metallic scraping sounds",
    "Soft, muffled footsteps on carpet",
    "Sharp, staccato tapping on a wooden table"
]
```

### Specify Environment and Context

Include details about where the sound is occurring:

```python
# Environmental context
environment_prompts = [
    "Waves crashing on a rocky shore with seagulls in the distance",
    "Bustling coffee shop with espresso machines, quiet conversations, and clinking cups",
    "Empty subway station with occasional train announcements and distant footsteps",
    "Jungle at night with insects chirping, frogs croaking, and leaves rustling"
]
```

### Layer Multiple Sound Elements

Combine different sound elements for complex soundscapes:

```python
# Layered sound prompts
layered_prompts = [
    "Crackling campfire with occasional wood pops, crickets chirping, and a gentle breeze",
    "Office ambience with keyboard typing, printer running, and quiet conversation",
    "Thunderstorm with heavy rain, wind, thunder cracks, and water dripping",
    "Kitchen sounds with chopping vegetables, sizzling pan, and timer beeping"
]
```

## Creating a Sound Category Explorer

Let's create a script that explores different categories of sounds:

```python
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

if __name__ == "__main__":
    explore_sound_categories(model_size="medium", duration=5.0)
```

This script will generate samples across 20 different sound categories, giving you a comprehensive exploration of what AudioGen can create.

## Creating an Ambient Soundscape Generator

Let's create a script that generates longer ambient soundscapes:

```python
# audiogen_ambient_generator.py
import torch
import os
import argparse
from audiocraft.models import AudioGen
from audiocraft.data.audio import audio_write

def generate_ambient_soundscape(
    environment,
    duration=15.0,
    model_size="medium",
    temperature=0.8,
    output_dir="ambient_soundscapes"
):
    """
    Generate an ambient soundscape for a specific environment.
    
    Args:
        environment (str): Type of environment to generate
        duration (float): Length of audio in seconds
        model_size (str): Size of model to use
        temperature (float): Controls randomness
        output_dir (str): Directory to save output
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define environment prompts
    environments = {
        "forest": "Peaceful forest ambient sounds with birds chirping, leaves rustling in the breeze, and a distant stream flowing",
        "ocean": "Calm ocean waves gently rolling onto a sandy beach with seagulls calling in the distance",
        "city": "Urban city ambient sound with distant traffic, people talking as they walk past, and occasional sirens",
        "cafe": "Cozy cafe ambience with quiet conversations, coffee machines steaming, and soft music in the background",
        "rain": "Gentle rainfall ambient with raindrops hitting windows and roof, creating a peaceful atmosphere",
        "night": "Night time ambient sounds with crickets chirping, occasional owl hoots, and gentle wind through trees",
        "river": "River ambient sounds with water flowing over rocks, fish occasionally jumping, and light wind in reeds",
        "office": "Office ambient sounds with keyboard typing, quiet conversations, printing, and occasional phone rings",
        "mountain": "High mountain ambient with strong wind gusts, distant eagle calls, and rocks occasionally falling",
        "spaceship": "Science fiction spaceship ambient with humming engines, electronic beeps, and occasional mechanical sounds"
    }
    
    # Check if the environment is valid
    if environment not in environments:
        print(f"Environment '{environment}' not found. Available environments:")
        for env_name in environments.keys():
            print(f"- {env_name}")
        return None
    
    prompt = environments[environment]
    print(f"Generating ambient soundscape: {environment}")
    print(f"Using prompt: '{prompt}'")
    
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
    model = AudioGen.get_pretrained(model_size)
    model.to(device)
    
    # Set generation parameters
    model.set_generation_params(
        duration=duration,
        temperature=temperature,
        top_k=250,
        top_p=0.0,
    )
    
    # Generate soundscape
    wav = model.generate([prompt])
    
    # Save the audio file
    output_path = os.path.join(output_dir, f"{environment}_ambient_{duration}s")
    audio_write(
        output_path,
        wav[0].cpu(),
        model.sample_rate,
        strategy="loudness"
    )
    
    print(f"Soundscape saved to {output_path}.wav")
    return f"{output_path}.wav"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate ambient soundscapes')
    parser.add_argument('--environment', type=str, default='forest',
                        help='Type of environment to generate (forest, ocean, city, etc.)')
    parser.add_argument('--duration', type=float, default=15.0,
                        help='Duration in seconds (default: 15.0)')
    parser.add_argument('--model', type=str, default='medium', choices=['medium', 'large'],
                        help='Model size to use (default: medium)')
    parser.add_argument('--output', type=str, default='ambient_soundscapes',
                        help='Output directory (default: ambient_soundscapes)')
    
    args = parser.parse_args()
    
    generate_ambient_soundscape(
        environment=args.environment,
        duration=args.duration,
        model_size=args.model,
        output_dir=args.output
    )
```

This script creates longer ambient soundscapes for different environments, perfect for background audio in various scenarios.

## Creating a Sound Effect Combination Tool

One of the powerful applications of AudioGen is creating complex sound combinations. Let's build a script for that:

```python
# audiogen_sound_combiner.py
import torch
import torchaudio
import os
import numpy as np
from audiocraft.models import AudioGen
from audiocraft.data.audio import audio_write

def generate_sound_combination(
    sound_elements,
    duration=8.0, 
    model_size="medium",
    output_dir="combined_sounds"
):
    """
    Generate multiple sound elements and combine them into a single audio file.
    
    Args:
        sound_elements (dict): Dictionary of sound element names and their prompts
        duration (float): Length of each sound element in seconds
        model_size (str): Size of model to use
        output_dir (str): Directory to save outputs
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "elements"), exist_ok=True)
    
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
    model = AudioGen.get_pretrained(model_size)
    model.to(device)
    
    # Set generation parameters
    model.set_generation_params(
        duration=duration,
        temperature=1.0,
        top_k=250,
        top_p=0.0,
    )
    
    # Generate each sound element
    generated_sounds = {}
    for name, prompt in sound_elements.items():
        print(f"Generating sound element: {name}")
        print(f"Prompt: '{prompt}'")
        
        # Generate
        wav = model.generate([prompt])
        audio_array = wav[0].cpu().numpy()
        
        # Save individual element
        element_path = os.path.join(output_dir, "elements", name)
        audio_write(
            element_path,
            wav[0].cpu(),
            model.sample_rate,
            strategy="loudness"
        )
        
        generated_sounds[name] = {
            "audio": audio_array,
            "path": f"{element_path}.wav"
        }
        
        print(f"Saved element to {element_path}.wav")
    
    # Combine sounds
    # Get the sample rate from the model
    sample_rate = model.sample_rate
    
    # Initialize an empty audio array
    max_length = max(sound["audio"].shape[0] for sound in generated_sounds.values())
    combined_audio = np.zeros(max_length)
    
    # Mix all sounds together
    for name, sound in generated_sounds.items():
        # Normalize the volume of each sound element
        normalized = sound["audio"] / (len(generated_sounds) * 1.5)  # Prevent clipping
        
        # Add to the combined audio
        combined_audio[:normalized.shape[0]] += normalized
    
    # Ensure the combined audio doesn't clip
    if np.max(np.abs(combined_audio)) > 1.0:
        combined_audio = combined_audio / np.max(np.abs(combined_audio))
    
    # Convert back to torch tensor
    combined_tensor = torch.from_numpy(combined_audio).to(torch.float32)
    
    # Save the combined sound
    combination_name = "_".join(sound_elements.keys())[:30]
    output_path = os.path.join(output_dir, f"combined_{combination_name}")
    torchaudio.save(
        f"{output_path}.wav",
        combined_tensor.unsqueeze(0),  # Add channel dimension
        sample_rate
    )
    
    print(f"Combined sound saved to {output_path}.wav")
    return f"{output_path}.wav"

if __name__ == "__main__":
    # Example sound elements for a rainy cafe scene
    sound_elements = {
        "rain": "Heavy rain falling on windows and roof with occasional thunder",
        "cafe": "Inside cafe ambience with quiet conversations and cups clinking",
        "music": "Soft jazz music playing from speakers with piano and saxophone",
        "traffic": "Occasional traffic sounds from outside, muffled by the rain"
    }
    
    generate_sound_combination(
        sound_elements=sound_elements,
        duration=8.0,
        model_size="medium"
    )
```

This script generates multiple sound elements separately and then combines them into a layered soundscape, giving you more control over the final result.

## Exercises

1. **Sound Scene Creator**: Try creating prompts for specific scenes, like "thunderstorm at sea" or "busy airport terminal."

2. **Sound Design Challenge**: Generate sound effects that could be used in a video game or film, such as "alien spaceship landing" or "magical portal opening."

3. **Sound Variety**: Generate the same sound description multiple times with different temperature settings to see how the results vary.

4. **Ambient Loop**: Try to create ambient sounds that could work well as seamless loops.

## Next Steps

Now that you understand the basics of AudioGen, you can:

1. Experiment with different prompts and parameters
2. Explore sound design for different applications
3. Create complex soundscapes by combining multiple generations
4. Use the generated sounds in your creative projects

## Conclusion

AudioGen provides powerful capabilities for generating realistic sound effects and ambient soundscapes from text descriptions. By understanding how to craft effective prompts and adjust generation parameters, you can create a wide variety of high-quality audio content for use in games, films, applications, or any other creative project requiring sound design.
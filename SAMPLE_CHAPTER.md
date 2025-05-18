---
layout: chapter
title: "Chapter 10: Sound Effect Generation with AudioGen"
# Copyright © 2025 Scott Friedman.
# Licensed under Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.
difficulty: beginner
estimated_time: 2 hours
---

> "I'm working on an indie game, and I need dozens of environmental sound effects - rain, wind, footsteps, machinery. I don't have the budget for a professional sound library, and recording everything myself would take months." — *Maya Chen, Indie Game Developer*

# Chapter 10: Sound Effect Generation with AudioGen

## The Challenge

Creating high-quality, diverse sound effects is a persistent challenge across many creative industries. Professional sound libraries can be expensive, recording sounds requires specialized equipment and environments, and creating synthetic sounds often demands expertise with digital audio workstations. This creates a significant barrier for indie game developers, small film productions, and digital content creators who need custom sound effects but lack the resources for traditional approaches.

AudioGen, part of Meta's AudioCraft framework, offers an AI-powered solution to this challenge by generating realistic sound effects from text descriptions. In this chapter, we'll explore how to leverage AudioGen to create custom environmental sounds, ambient effects, and sound design elements through simple text prompts.

## Learning Objectives

By the end of this chapter, you'll be able to:

- Generate realistic environmental sounds and effects using text prompts
- Craft effective prompts that produce the desired audio characteristics
- Optimize AudioGen's parameters for different types of sound effects
- Create and save a library of custom sound effects for your projects
- Understand key differences between AudioGen and MusicGen models

## Prerequisites

Before proceeding, ensure you have:
- Completed the setup instructions in Chapter 2
- Basic understanding of the AudioCraft framework from Chapter 3
- Python environment with AudioCraft installed and functioning
- Basic familiarity with audio concepts (sample rate, duration, etc.)

## Key Concepts

### Understanding AudioGen vs. MusicGen

While both models are part of the AudioCraft framework, AudioGen and MusicGen serve distinct purposes and were trained on different datasets. MusicGen specializes in creating musical compositions with instruments, rhythm, and melody, while AudioGen focuses on non-musical sounds like environmental effects, mechanical noises, and ambient soundscapes.

Think of MusicGen as a composer who creates structured musical pieces, while AudioGen is more like a sound designer who crafts the ambient sounds for a film scene or game environment. The distinction is important because prompts that work well for MusicGen may not be effective for AudioGen, and vice versa.

```python
# MusicGen example (musical content)
music_prompt = "An upbeat electronic track with a catchy melody"

# AudioGen example (non-musical content)
sound_prompt = "Heavy rain falling on a metal roof with occasional thunder"
```

### Sound Effect Categories

AudioGen can generate a wide range of sound effects that generally fall into several categories:

1. **Environmental**: Natural sounds like rain, wind, water, fire
2. **Urban**: City sounds, traffic, construction, crowds
3. **Mechanical**: Machines, engines, tools, electronics
4. **Biological**: Animal sounds, human non-speech sounds
5. **Abstract**: Synthetic sounds, textures, atmospheres

Understanding these categories helps you craft prompts that align with AudioGen's strengths and training data. For example, AudioGen excels at creating realistic rain sounds but may struggle with very specific musical instruments (which would be MusicGen's strength).

```python
# Example prompts for different sound categories
environmental = "Forest ambience with birds chirping, leaves rustling, and a stream flowing"
urban = "Busy coffee shop with espresso machines, quiet conversations, and clinking cups"
mechanical = "Old mechanical clock ticking with gears turning and occasional chimes"
biological = "Dogs barking and playing in a dog park with panting and paws on grass"
abstract = "Ethereal atmospheric sound with shimmering textures and deep resonances"
```

## Solution Walkthrough

### 1. Setting Up AudioGen

Let's begin by importing the necessary libraries and setting up AudioGen. Unlike MusicGen, which comes in three sizes (small, medium, large), AudioGen only offers "medium" and "large" variants.

```python
import torch
import torchaudio
import os
import time
from audiocraft.models import AudioGen
from audiocraft.data.audio import audio_write

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

# Load the AudioGen model - note: only "medium" and "large" are available
model_size = "medium"  # Use "medium" for faster generation or "large" for higher quality
model = AudioGen.get_pretrained(model_size)
model.to(device)  # Move model to the appropriate compute device
```

### 2. Configuring Generation Parameters

Now that we have AudioGen loaded, we need to set the generation parameters that control the sound output. For sound effects, we typically use shorter durations (3-15 seconds) compared to music, and we might want to experiment with different temperature values to get varied results.

```python
# Configure generation parameters
duration = 5.0  # Sound effects often work well with shorter durations (in seconds)
temperature = 1.0  # Controls randomness/creativity of the generation

model.set_generation_params(
    duration=duration,        # How long the generated audio will be
    temperature=temperature,  # Controls randomness/creativity
    top_k=250,                # Sample from top 250 token predictions
    top_p=0.0,                # Disable nucleus sampling (use top_k instead)
)
```

### 3. Crafting and Using Sound Prompts

The prompt is the most critical element for getting good results with AudioGen. Let's create a function that allows us to generate sound effects from text descriptions:

```python
def generate_sound_effect(prompt, output_dir="sound_effects"):
    """Generate a sound effect based on a text description."""
    print(f"Generating sound: '{prompt}'")
    start_time = time.time()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate the sound based on the text prompt
    wav = model.generate([prompt])  # Model expects a list of prompts
    
    # Create a filesystem-safe filename based on the prompt
    safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt)[:30]
    filename = f"{safe_prompt}_{model_size}_{duration}s"
    
    # Save the generated audio as a WAV file
    output_path = os.path.join(output_dir, filename)
    audio_write(
        output_path,              # Path without extension
        wav[0].cpu(),             # Audio tensor (first in batch)
        model.sample_rate,        # Sample rate (32kHz for AudioGen)
        strategy="loudness",      # Normalize loudness for consistent volume
    )
    
    generation_time = time.time() - start_time
    print(f"Generation completed in {generation_time:.2f} seconds")
    print(f"Audio saved to {output_path}.wav")
    
    return f"{output_path}.wav"
```

### 4. Creating a Sound Effect Library

Now, let's use our function to generate a variety of sound effects across different categories. This will help us build a small sound effect library that could be used in projects:

```python
# Define example prompts for different sound categories
sound_prompts = {
    # Weather and nature
    "rain_heavy": "Heavy rain falling on a metal roof with water dripping and occasional thunder",
    "wind_forest": "Strong wind blowing through forest trees with leaves rustling and branches creaking",
    "ocean_waves": "Ocean waves crashing on a rocky shore with seagulls calling in the distance",
    "river_stream": "Gentle river flowing over rocks with water bubbling and splashing",
    
    # Urban environments
    "city_traffic": "Busy city intersection with car horns, engines revving, and distant sirens",
    "cafe_ambience": "Busy coffee shop with espresso machines, quiet conversations, and clinking cups",
    "construction": "Construction site with jackhammers, power drills, and workers shouting",
    "subway_station": "Underground subway station with train arriving, brakes screeching, and station announcements",
    
    # Mechanical and electronic
    "old_clock": "Old mechanical clock ticking with gears turning and a soft bell chime every few seconds",
    "factory_machinery": "Factory manufacturing floor with conveyor belts, pneumatic presses, and robotic arms",
    "old_computer": "Vintage computer with whirring fan, clicking hard drive, and electronic processing sounds",
    "sci_fi_spaceship": "Futuristic spaceship engine room with humming reactors, beeping consoles, and steam venting"
}

# Generate and save each sound effect
for sound_name, prompt in sound_prompts.items():
    output_path = generate_sound_effect(prompt, output_dir="sound_library")
    print(f"Created {sound_name} effect: {output_path}\n")
```

## Complete Implementation

Let's put everything together into a complete, runnable example that demonstrates generating sound effects with AudioGen:

```python
# audiogen_sound_library.py - Generate a library of sound effects using AudioGen
import torch
import os
import time
import argparse
from audiocraft.models import AudioGen
from audiocraft.data.audio import audio_write

def generate_sound_effect(
    prompt,
    duration=5.0,
    model_size="medium",
    output_dir="sound_effects",
    temperature=1.0
):
    """
    Generate environmental sounds and audio effects based on a text description.
    
    Args:
        prompt (str): Detailed text description of the sound to generate
        duration (float): Length of audio to generate in seconds (recommended: 3-10s)
        model_size (str): Size of model to use - only "medium" or "large" are available
        output_dir (str): Directory where generated audio will be saved
        temperature (float): Controls randomness/variability of generation
                           (higher = more random/creative, lower = more deterministic)
    
    Returns:
        str: Path to the generated audio file (.wav format)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating sound for prompt: '{prompt}'")
    print(f"Using model size: {model_size}, duration: {duration}s")
    
    start_time = time.time()
    
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
    
    # Load the AudioGen model
    model = AudioGen.get_pretrained(model_size)
    model.to(device)  # Move model to the appropriate compute device
    
    # Configure generation parameters
    model.set_generation_params(
        duration=duration,        # How long the generated audio will be
        temperature=temperature,  # Controls randomness/creativity
        top_k=250,                # Sample from top 250 token predictions
        top_p=0.0,                # Disable nucleus sampling (use top_k instead)
    )
    
    # Generate the sound based on the text prompt
    wav = model.generate([prompt])  # Model expects a list of prompts
    
    # Create a filesystem-safe filename based on the prompt
    safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt)[:30]
    filename = f"{safe_prompt}_{model_size}_{duration}s"
    
    # Save the generated audio as a WAV file
    output_path = os.path.join(output_dir, filename)
    audio_write(
        output_path,              # Path without extension
        wav[0].cpu(),             # Audio tensor (first in batch)
        model.sample_rate,        # Sample rate (32kHz for AudioGen)
        strategy="loudness",      # Normalize loudness for consistent volume
    )
    
    generation_time = time.time() - start_time
    print(f"Generation completed in {generation_time:.2f} seconds")
    print(f"Audio saved to {output_path}.wav")
    
    return f"{output_path}.wav"

def create_sound_library(category=None, output_dir="sound_library"):
    """
    Generate a library of sound effects across different categories.
    
    Args:
        category (str, optional): Specific category to generate, or None for all categories
        output_dir (str): Directory to save the sound effects
    """
    # Define sound effect prompts organized by category
    sound_categories = {
        "weather": {
            "rain_light": "Light rainfall on leaves with occasional drips and gentle wind",
            "rain_heavy": "Heavy rain pouring down with water splashing in puddles and thunder rumbling",
            "wind_gentle": "Gentle breeze rustling leaves with occasional bird chirps",
            "wind_storm": "Howling wind of a severe storm with debris flying and trees creaking",
            "thunder_storm": "Thunderstorm with heavy rain, cracking thunder, and strong winds"
        },
        
        "nature": {
            "forest_day": "Daytime forest with birds chirping, leaves rustling, and a distant stream",
            "forest_night": "Nighttime forest with crickets, occasional owl hoots, and gentle breeze",
            "ocean_waves": "Ocean waves crashing on a sandy beach with seagulls in the distance",
            "river_stream": "Small stream flowing over rocks with water bubbling and splashing gently",
            "jungle": "Dense jungle with exotic birds calling, insects buzzing, and leaves rustling"
        },
        
        "urban": {
            "city_traffic": "Busy city street with cars passing, horns honking, and people talking",
            "cafe": "Coffee shop ambience with espresso machines, quiet conversations, and cups clinking",
            "restaurant": "Busy restaurant with utensils on plates, conversations, and kitchen sounds",
            "construction": "Construction site with jackhammers, power tools, and workers calling out",
            "subway": "Subway train arriving at station with screeching brakes and announcements"
        },
        
        "mechanical": {
            "clock": "Old mechanical clock with ticking gears, pendulum swinging, and occasional chimes",
            "factory": "Factory machinery with conveyor belts, hydraulic presses, and metal clanking",
            "engine_small": "Small engine running with mechanical parts whirring and occasional misfires",
            "engine_large": "Large industrial engine with powerful rhythmic pounding and steam releases",
            "typewriter": "Manual typewriter with keys clacking, carriage moving, and bell dinging"
        },
        
        "household": {
            "kitchen": "Kitchen sounds with chopping vegetables, water running, and pots simmering",
            "bathroom": "Bathroom with shower running, water draining, and fan humming",
            "living_room": "Living room with clock ticking, pages turning, and occasional footsteps",
            "laundry": "Laundry room with washing machine spinning, clothes tumbling, and water filling",
            "garden": "Garden with lawn mower running, birds chirping, and sprinkler spraying"
        }
    }
    
    # Create the main output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine which categories to generate
    categories_to_generate = [category] if category else sound_categories.keys()
    
    for category_name in categories_to_generate:
        if category_name not in sound_categories:
            print(f"Category '{category_name}' not found. Available categories:")
            for available_category in sound_categories.keys():
                print(f"- {available_category}")
            return
        
        # Create category subdirectory
        category_dir = os.path.join(output_dir, category_name)
        os.makedirs(category_dir, exist_ok=True)
        
        print(f"\n===== Generating {category_name.upper()} SOUNDS =====\n")
        
        # Generate each sound in the category
        for sound_name, prompt in sound_categories[category_name].items():
            output_path = generate_sound_effect(
                prompt=prompt,
                duration=5.0,
                output_dir=category_dir
            )
            print(f"Created {sound_name}: {output_path}\n")
            
            # Clear GPU cache between generations if using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a library of sound effects with AudioGen")
    parser.add_argument("--category", type=str, default=None, 
                        help="Specific category to generate (weather, nature, urban, mechanical, household)")
    parser.add_argument("--output", type=str, default="sound_library",
                        help="Output directory for sound effects")
    parser.add_argument("--model", type=str, default="medium", choices=["medium", "large"],
                        help="Model size to use (medium or large)")
    
    args = parser.parse_args()
    
    create_sound_library(
        category=args.category,
        output_dir=args.output
    )
```

## Variations and Customizations

Let's explore some variations of our solution to address different needs or preferences.

### Variation 1: Ambient Soundscape Generator

Instead of short, focused sound effects, you might want to create longer ambient soundscapes for background audio in games, videos, or meditation apps:

```python
def generate_ambient_soundscape(
    environment,
    duration=15.0,
    model_size="medium",
    temperature=0.8,
    output_dir="ambient_soundscapes"
):
    """
    Generate a longer ambient soundscape for a specific environment.
    
    Args:
        environment (str): Type of environment or specific description
        duration (float): Length of audio in seconds (10-30 seconds recommended)
        model_size (str): Size of model to use
        temperature (float): Controls randomness
        output_dir (str): Directory to save output
    """
    # Create prompt with ambient-specific language
    ambient_prompt = f"Continuous ambient background sound of {environment} with subtle variations and natural evolution of elements"
    
    # Lower temperature for more consistent ambient backgrounds
    return generate_sound_effect(
        prompt=ambient_prompt,
        duration=duration,
        model_size=model_size,
        temperature=temperature,
        output_dir=output_dir
    )

# Example usage
environments = [
    "peaceful forest with distant birds and gentle wind",
    "calm ocean waves on a beach with occasional seagulls",
    "light rainfall on a window with distant thunder",
    "busy coffee shop with quiet conversations and kitchen sounds",
    "space station with humming equipment and occasional beeps"
]

for env in environments:
    generate_ambient_soundscape(env, duration=20.0)
```

### Variation 2: Sound Effect Layering

For more complex sound design, you might want to generate and layer multiple sound elements:

```python
import numpy as np

def generate_layered_soundscape(elements, output_dir="layered_sounds"):
    """
    Generate multiple sound elements and layer them into a single audio file.
    
    Args:
        elements (dict): Dictionary of sound element descriptions
        output_dir (str): Directory to save outputs
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "elements"), exist_ok=True)
    
    # Load model once to avoid reloading for each element
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        
    model = AudioGen.get_pretrained("medium")
    model.to(device)
    sample_rate = model.sample_rate
    
    # Generate each sound element
    generated_sounds = {}
    for name, description in elements.items():
        print(f"Generating sound element: {name}")
        print(f"Description: '{description}'")
        
        # Set generation parameters
        model.set_generation_params(
            duration=8.0,          # Length of each element
            temperature=1.0,       # Standard creativity level
            top_k=250,             # Diverse but controlled sampling
            top_p=0.0,             # Disable nucleus sampling
        )
        
        # Generate the sound
        wav = model.generate([description])
        audio_array = wav[0].cpu().numpy()
        
        # Save individual element
        element_path = os.path.join(output_dir, "elements", name)
        audio_write(
            element_path,
            wav[0].cpu(),
            sample_rate,
            strategy="loudness"
        )
        
        generated_sounds[name] = {
            "audio": audio_array,
            "path": f"{element_path}.wav"
        }
        
        print(f"Saved element to {element_path}.wav")
    
    # Combine sounds
    # Initialize an empty audio array at the proper length
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
    scene_name = "_".join(elements.keys())[:30]
    output_path = os.path.join(output_dir, f"scene_{scene_name}")
    torchaudio.save(
        f"{output_path}.wav",
        combined_tensor.unsqueeze(0),  # Add channel dimension
        sample_rate
    )
    
    print(f"Combined soundscape saved to {output_path}.wav")
    return f"{output_path}.wav"

# Example usage
rainy_cafe_scene = {
    "rain": "Heavy rain falling on windows and roof with occasional thunder",
    "cafe": "Inside cafe ambience with quiet conversations and cups clinking",
    "music": "Soft jazz music playing from speakers, very quiet in the background",
    "traffic": "Occasional traffic sounds from outside, muffled by the rain"
}

generate_layered_soundscape(rainy_cafe_scene)
```

## Common Pitfalls and Troubleshooting

### Problem: Inconsistent Sound Quality

Sometimes AudioGen might produce sounds with artifacts or inconsistencies, especially for complex sound descriptions.

**Solution**: 
- Break down complex sounds into individual elements and layer them (as in Variation 2)
- Use more specific, detailed prompts
- Try different temperature settings (lower for more consistent results)
- Use the "large" model for higher quality output, if your hardware can handle it

### Problem: Missing Sound Elements

You might notice that sometimes AudioGen misses certain elements from your prompt.

**Solution**:
- Prioritize the most important sound elements earlier in your prompt
- Try to limit prompts to 2-3 key sound elements
- Generate separate sounds for complex scenes and combine them manually
- Emphasize specific elements with descriptive language (e.g., "prominent", "clearly audible")

### Problem: Unrealistic Sound Textures

Some generated sounds might not sound entirely realistic, especially for specific mechanical or electronic sounds.

**Solution**:
- Focus on the general sound character rather than specifics
- Combine with real recorded samples when needed
- Post-process the audio with effects to enhance realism
- Try multiple generations with the same prompt to get variations

## Hands-on Challenge

Now it's your turn to experiment with what you've learned. Try the following challenge:

### Challenge: Interactive Sound Effect Generator

Create a script that:
1. Presents a menu of sound categories (weather, nature, mechanical, etc.)
2. Allows the user to input custom text descriptions for sounds
3. Generates the sound effect and plays it back
4. Gives the option to save successful generations to a sound library
5. Implements parameter controls for duration and temperature

### Bonus Challenge

Create a "sound scene composer" that allows you to:
1. Generate multiple sound elements with different parameters
2. Adjust the volume level of each element
3. Apply basic effects (reverb, EQ, etc.) to individual elements
4. Mix the elements together with proper timing
5. Export the final composed scene

## Key Takeaways

- AudioGen specializes in non-musical sounds and effects, complementing MusicGen's focus on music
- Detailed, specific prompts with clear sound descriptions produce the best results
- Sound generation benefits from proper categorization (environmental, mechanical, etc.)
- Complex soundscapes can be created by layering multiple generated elements
- Parameter tuning (especially temperature) significantly affects the character of generated sounds

## Next Steps

Now that you've mastered basic sound effect generation with AudioGen, you're ready to explore:

- **Chapter 11: Crafting Sound Effect Prompts**: Learn advanced techniques for optimizing sound descriptions
- **Chapter 12: Building Sound Libraries**: Discover methods for organizing and categorizing generated sounds
- **Chapter 13: Creating Complex Soundscapes**: Explore advanced techniques for layering and composition
- **Chapter 18: Building a Complete Audio Pipeline**: Combine sound effects with music and speech

## Further Reading

- [Meta AI AudioCraft Blog Post](https://ai.meta.com/blog/audiocraft-musicgen-audiogen-encodec-generative-ai-audio/)
- [Sound Design Fundamentals](https://www.soundonsound.com/techniques/sound-design-basics)
- [Film Sound Theory](https://www.filmindependent.org/blog/know-the-score-the-role-of-sound-design-in-filmmaking/)
- [Game Audio Resources](https://www.gamasutra.com/blogs/PeterGofton/20160620/275070/Sound_Design_Deep_Dive_Creating_the_Sound_for_Alien_Isolation.php)
---
layout: chapter
title: "Chapter 5: Basic Music Generation"
# Copyright © 2025 Scott Friedman.
# Licensed under Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.
difficulty: beginner
estimated_time: 2 hours
scenario:
  quote: "I need to create background music for my indie game project, but I have no musical training and can't afford to hire a composer. I want to generate custom tracks that match specific scenes and moods, but I'm overwhelmed by all the options in MusicGen."
  persona: "Dev Harrison"
  role: "Indie Game Developer"
next_steps:
  - title: "Prompt Engineering for Music"
    url: "/chapters/part2/prompt-engineering/"
    description: "Learn advanced techniques for crafting effective music prompts"
  - title: "Parameter Optimization"
    url: "/chapters/part2/parameter-optimization/"
    description: "Fine-tune generation parameters for better results"
  - title: "Melody Conditioning"
    url: "/chapters/part2/melody-conditioning/"
    description: "Guide music generation with reference melodies"
further_reading:
  - title: "MusicGen Research Paper"
    url: "https://arxiv.org/abs/2306.05284"
    description: "Simple and Controllable Music Generation"
  - title: "MusicGen Demo on Hugging Face"
    url: "https://huggingface.co/spaces/facebook/MusicGen"
  - title: "Prompt Engineering Guide"
    url: "https://www.promptingguide.ai/"
    description: "General techniques for effective prompting"
---

# Chapter 5: Basic Music Generation

## The Challenge

Creating original music for your projects has traditionally required musical training or hiring a composer. Even with stock music libraries, finding the perfect track that matches your specific vision can be frustrating and time-consuming. You might need music that shifts in tone at precise moments, follows particular themes, or perfectly matches your visual content.

For indie developers, content creators, and digital artists, music creation has often been a bottleneck. Without the budget for professional composers or the time to learn music production, many settle for generic stock music that doesn't fully realize their creative vision.

In this chapter, we'll explore how to use MusicGen to generate custom music tracks from simple text descriptions. We'll build on what you learned in the first audio generation chapter, diving deeper into the specific capabilities, techniques, and best practices for music generation. You'll learn how to reliably generate high-quality music that matches your creative intent.

## Learning Objectives

By the end of this chapter, you'll be able to:

- Generate music in specific genres and styles using effective prompts
- Control mood, instrumentation, and musical structure through text descriptions
- Customize generation parameters for different musical outcomes
- Create variations of the same musical concept
- Develop a systematic approach to music generation for projects
- Build a reusable music generation module for your applications

## Crafting Effective Music Prompts

The quality of music you generate depends significantly on the prompt you provide. Unlike general text generation, music prompts benefit from specific elements that guide the AI toward successful generation.

### The Anatomy of an Effective Music Prompt

An effective music prompt typically includes several key components:

1. **Genre or Style**: The overall musical category (e.g., "jazz," "electronic," "orchestral")
2. **Mood/Emotion**: The feeling the music should convey (e.g., "uplifting," "melancholic," "tense")
3. **Instrumentation**: Key instruments to include (e.g., "piano," "synth bass," "strings")
4. **Tempo/Rhythm**: Speed and rhythmic characteristics (e.g., "fast-paced," "steady beat," "waltz rhythm")
5. **Structure**: Any desired musical elements (e.g., "building chorus," "quiet intro," "dramatic bridge")

Here's a template for constructing effective prompts:

```
A [genre/style] piece with [instrumentation] featuring [mood/emotion] and [tempo/rhythm characteristics]. The music has [structural elements] and [additional details].
```

### Examples of Effective Prompts

Let's look at several well-crafted prompts and what makes them effective:

**Basic prompt:**
```
"Electronic music"
```

**Improved prompt:**
```
"An upbeat electronic dance track with a catchy synth melody, driving bass, and energetic drums. The music builds tension before dropping into a euphoric chorus."
```

The improved version provides:
- Genre/style: "electronic dance track"
- Instrumentation: "synth melody," "driving bass," "energetic drums"
- Mood: "upbeat," "euphoric"
- Structure: "builds tension before dropping into a chorus"

Let's look at another example:

**Basic prompt:**
```
"Sad piano music"
```

**Improved prompt:**
```
"A melancholic piano solo in a minor key with gentle, flowing notes and occasional string accompaniment. The piece evokes a feeling of nostalgia with emotional crescendos and delicate, quiet passages."
```

The improved version includes:
- Genre/style: "piano solo"
- Technical details: "minor key"
- Playing style: "gentle, flowing notes"
- Additional instrumentation: "occasional string accompaniment"
- Mood: "melancholic," "nostalgia," "emotional"
- Structure: "crescendos," "delicate, quiet passages"

## A Complete MusicGen Application

Now let's build a comprehensive music generation application that incorporates these prompt design principles. This application will:

1. Generate music based on detailed prompts
2. Save the results with appropriate metadata
3. Support different generation parameters
4. Include error handling and performance monitoring

```python
# musicgen_application.py - Complete music generation application
import torch
import os
import time
import json
import argparse
from datetime import datetime
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

class MusicGenerator:
    """A comprehensive music generation application using MusicGen."""
    
    def __init__(self, model_size="medium", output_dir="generated_music"):
        """
        Initialize the music generator.
        
        Args:
            model_size (str): Size of model to use ('small', 'medium', or 'large')
            output_dir (str): Directory to save generated music
        """
        self.model_size = model_size
        self.output_dir = output_dir
        self.model = None
        self.device = self._get_device()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up metadata recording
        self.metadata_path = os.path.join(output_dir, "generation_metadata.json")
        self.metadata = self._load_metadata()
    
    def _get_device(self):
        """Determine the best available device."""
        if torch.cuda.is_available():
            device = "cuda"
            print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = "mps"
            print("Using MPS (Metal) device")
        else:
            device = "cpu"
            print("Using CPU (generation will be slower)")
        return device
    
    def _load_metadata(self):
        """Load existing metadata or create new metadata file."""
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print("Error reading metadata file. Creating new metadata.")
                return {"generations": []}
        else:
            return {"generations": []}
    
    def _save_metadata(self):
        """Save metadata to disk."""
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def load_model(self):
        """Load the MusicGen model."""
        if self.model is not None:
            print("Model already loaded")
            return
        
        print(f"Loading MusicGen {self.model_size} model...")
        start_time = time.time()
        
        try:
            self.model = MusicGen.get_pretrained(self.model_size)
            self.model.to(self.device)
            load_time = time.time() - start_time
            print(f"Model loaded in {load_time:.2f} seconds")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def generate(self, 
                 prompt, 
                 duration=10.0, 
                 temperature=1.0, 
                 top_k=250, 
                 top_p=0.0, 
                 cfg_coef=3.0,
                 seed=None,
                 filename=None):
        """
        Generate music from a text prompt.
        
        Args:
            prompt (str): Text description of the music to generate
            duration (float): Length of audio in seconds (max 30)
            temperature (float): Controls randomness (higher = more random)
            top_k (int): Number of highest probability tokens to consider
            top_p (float): Nucleus sampling threshold (0.0 to disable)
            cfg_coef (float): Classifier-free guidance scale (how closely to follow text)
            seed (int, optional): Random seed for reproducible results
            filename (str, optional): Custom filename, if None will be auto-generated
            
        Returns:
            str: Path to the generated audio file
        """
        # Ensure model is loaded
        if self.model is None:
            self.load_model()
        
        # Set seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            print(f"Using random seed: {seed}")
        
        # Set generation parameters
        self.model.set_generation_params(
            duration=duration,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            cfg_coef=cfg_coef
        )
        
        print(f"Generating music for prompt: '{prompt}'")
        print(f"Parameters: duration={duration}s, temperature={temperature}, cfg_coef={cfg_coef}")
        
        # Record start time for performance measurement
        start_time = time.time()
        
        try:
            # Generate music
            wav = self.model.generate([prompt])
            
            # Create a filename
            if filename is None:
                safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt)[:30]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"music_{safe_prompt}_{timestamp}"
            
            # Save the audio file
            output_path = os.path.join(self.output_dir, filename)
            audio_write(
                output_path,
                wav[0].cpu(),
                self.model.sample_rate,
                strategy="loudness",
            )
            
            # Calculate performance metrics
            generation_time = time.time() - start_time
            print(f"Generation completed in {generation_time:.2f} seconds")
            print(f"Audio saved to {output_path}.wav")
            
            # Record metadata
            generation_info = {
                "prompt": prompt,
                "model_size": self.model_size,
                "duration": duration,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "cfg_coef": cfg_coef,
                "seed": seed,
                "filename": f"{output_path}.wav",
                "generation_time": generation_time,
                "timestamp": datetime.now().isoformat()
            }
            self.metadata["generations"].append(generation_info)
            self._save_metadata()
            
            return f"{output_path}.wav"
            
        except Exception as e:
            print(f"Error generating music: {str(e)}")
            return None
        finally:
            # Clean up
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def generate_variations(self, prompt, count=3, **kwargs):
        """
        Generate multiple variations of the same prompt.
        
        Args:
            prompt (str): Base prompt to use
            count (int): Number of variations to generate
            **kwargs: Additional parameters to pass to generate()
            
        Returns:
            list: Paths to the generated audio files
        """
        results = []
        base_temp = kwargs.get('temperature', 1.0)
        
        for i in range(count):
            # Slightly adjust temperature for variety
            variation_temp = base_temp * (0.9 + 0.2 * (i / count))
            
            # Create variation-specific filename
            filename = kwargs.get('filename')
            if filename:
                var_filename = f"{filename}_variation_{i+1}"
            else:
                var_filename = None
            
            # Generate the variation
            print(f"\nGenerating variation {i+1}/{count} (temperature: {variation_temp:.2f})")
            result = self.generate(
                prompt=prompt, 
                temperature=variation_temp,
                filename=var_filename,
                **{k: v for k, v in kwargs.items() if k != 'temperature' and k != 'filename'}
            )
            
            if result:
                results.append(result)
        
        return results
    
    def generate_for_project(self, project_name, descriptions):
        """
        Generate a set of music tracks for a project.
        
        Args:
            project_name (str): Name of the project
            descriptions (dict): Dictionary of track names and their prompts
            
        Returns:
            dict: Track names and their file paths
        """
        project_dir = os.path.join(self.output_dir, project_name)
        os.makedirs(project_dir, exist_ok=True)
        
        results = {}
        
        for track_name, prompt in descriptions.items():
            print(f"\nGenerating '{track_name}' for project '{project_name}'")
            
            # Generate the track
            output_path = self.generate(
                prompt=prompt,
                filename=os.path.join(project_name, track_name)
            )
            
            if output_path:
                results[track_name] = output_path
        
        # Create a project summary
        with open(os.path.join(project_dir, "project_info.txt"), "w") as f:
            f.write(f"Project: {project_name}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("Tracks:\n")
            for track_name, prompt in descriptions.items():
                f.write(f"\n{track_name}:\n")
                f.write(f"Prompt: {prompt}\n")
                f.write(f"File: {results.get(track_name, 'Generation failed')}\n")
        
        return results

def main():
    """Run the music generator from the command line."""
    parser = argparse.ArgumentParser(description="Generate music with MusicGen")
    
    parser.add_argument("--prompt", type=str, help="Text description of music to generate")
    parser.add_argument("--model", type=str, default="medium", choices=["small", "medium", "large"],
                        help="Model size to use (default: medium)")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Duration in seconds (default: 10.0)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature parameter (default: 1.0)")
    parser.add_argument("--variations", type=int, default=1,
                        help="Number of variations to generate (default: 1)")
    parser.add_argument("--output", type=str, default="generated_music",
                        help="Output directory (default: generated_music)")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    if not args.prompt:
        # Interactive mode if no prompt provided
        print("=== MusicGen Interactive Mode ===")
        args.prompt = input("Enter your music description: ")
    
    generator = MusicGenerator(model_size=args.model, output_dir=args.output)
    
    if args.variations > 1:
        generator.generate_variations(
            prompt=args.prompt,
            count=args.variations,
            duration=args.duration,
            temperature=args.temperature,
            seed=args.seed
        )
    else:
        generator.generate(
            prompt=args.prompt,
            duration=args.duration,
            temperature=args.temperature,
            seed=args.seed
        )

if __name__ == "__main__":
    main()
```

This script provides a robust framework for music generation that you can use directly or integrate into your projects.

## Using the MusicGenerator Class

The `MusicGenerator` class offers several ways to generate music, each suited to different needs:

### 1. Basic Single-Track Generation

```python
from musicgen_application import MusicGenerator

# Create a generator instance
generator = MusicGenerator(model_size="medium")

# Generate a single track
output_file = generator.generate(
    prompt="An upbeat electronic track with a catchy melody and energetic rhythm",
    duration=15.0,
    temperature=1.0
)

print(f"Generated music saved to: {output_file}")
```

### 2. Generating Multiple Variations

```python
# Generate variations of the same concept
variations = generator.generate_variations(
    prompt="A peaceful piano piece with gentle melodies and soft string accompaniment",
    count=3,
    duration=10.0,
    temperature=1.0  # Base temperature that will be slightly adjusted for each variation
)

print("Generated variations:")
for i, file_path in enumerate(variations):
    print(f"Variation {i+1}: {file_path}")
```

### 3. Project-Based Generation

```python
# Generate a set of tracks for a game project
game_soundtrack = {
    "main_theme": "An epic orchestral piece with dramatic brass, soaring strings, and powerful percussion. The music conveys a sense of adventure and heroism.",
    
    "battle_music": "An intense, fast-paced orchestral track with driving percussion, staccato strings, and powerful brass. The music builds tension with rhythmic patterns and minor key harmonies.",
    
    "peaceful_exploration": "A gentle ambient piece with soft flutes, harp arpeggios, and sustained string pads. The music creates a calm, peaceful atmosphere with occasional bird calls and nature sounds.",
    
    "victory_fanfare": "A triumphant brass-heavy orchestral piece with celebratory percussion and uplifting major key progressions. The music conveys accomplishment and joy."
}

project_files = generator.generate_for_project(
    project_name="fantasy_game_soundtrack",
    descriptions=game_soundtrack
)

print("\nGame soundtrack generated:")
for track_name, file_path in project_files.items():
    print(f"{track_name}: {file_path}")
```

## Music Generation for Different Genres

MusicGen can generate a wide variety of musical genres and styles. Here's how to approach generation for some popular genres:

### Orchestral/Cinematic Music

Orchestral music benefits from detailed instrumentation and emotional descriptions:

```python
orchestral_prompt = "A grand orchestral piece with soaring strings, powerful brass fanfares, and thunderous percussion. The music builds from a gentle, mysterious introduction to an epic, heroic climax with full orchestra."

generator.generate(
    prompt=orchestral_prompt,
    duration=20.0,  # Longer duration for orchestral development
    temperature=0.9,  # Slightly lower temperature for coherence
    cfg_coef=4.0  # Higher adherence to text prompt
)
```

### Electronic/Dance Music

Electronic music prompts should specify rhythm, tempo, and synthesizer characteristics:

```python
edm_prompt = "An energetic electronic dance track at 128 BPM with pulsing synthesizer bass, arpeggiated lead synths, and a four-on-the-floor kick drum pattern. The track builds tension with a filtered breakdown before dropping into an euphoric chorus with wide, layered synth chords."

generator.generate(
    prompt=edm_prompt,
    duration=15.0,
    temperature=1.1,  # Slightly higher temperature for creativity
    cfg_coef=3.0
)
```

### Jazz

Jazz generation benefits from mentioning instrumentation, tempo, and stylistic elements:

```python
jazz_prompt = "A smooth jazz piece with a walking bass line, brushed drums, piano comping, and a mellow saxophone playing the lead melody. The music has a relaxed swing feel with complex chord progressions and occasional improvised solos."

generator.generate(
    prompt=jazz_prompt,
    duration=15.0,
    temperature=1.2,  # Higher temperature for improvisation-like qualities
    cfg_coef=2.5  # Lower cfg for more musical freedom
)
```

### Ambient/Atmospheric

Ambient music descriptions should focus on texture, mood, and sonic qualities:

```python
ambient_prompt = "A serene ambient soundscape with evolving synthesizer pads, gentle piano notes echoing in the distance, and subtle atmospheric textures. The music creates a floating, weightless feeling with slow harmonic progressions and no percussion."

generator.generate(
    prompt=ambient_prompt,
    duration=25.0,  # Longer duration for ambient evolution
    temperature=0.8,  # Lower temperature for consistency
    cfg_coef=3.5
)
```

## Parameter Tuning for Different Musical Outcomes

The generation parameters significantly impact the music produced. Here's a guide to adjusting parameters for specific musical goals:

### Temperature

The temperature parameter controls randomness and creativity:

- **Low temperature (0.3-0.7)**: More predictable, coherent, and focused music. Good for when you need consistent, structured pieces.

- **Medium temperature (0.8-1.2)**: Balanced creativity and coherence. This is the sweet spot for most generations.

- **High temperature (1.3-2.0)**: More experimental, surprising, and varied music. Can introduce unexpected elements and progressions.

### Classifier-Free Guidance (cfg_coef)

The cfg_coef parameter controls how closely the generation follows your text prompt:

- **Low cfg_coef (1.0-2.0)**: Looser interpretation of the prompt, potentially more musically interesting but less controlled.

- **Medium cfg_coef (2.5-4.0)**: Balanced adherence to the prompt and musical quality.

- **High cfg_coef (4.5-7.0)**: Stricter adherence to the prompt, especially useful for very specific musical requirements.

### Practical Examples

Here's how to adjust these parameters for specific goals:

```python
# For highly consistent, structured music (e.g., for looping game background)
generator.generate(
    prompt="A gentle, repeating piano melody with soft string accompaniment in a major key",
    temperature=0.5,
    cfg_coef=5.0
)

# For creative variations with unexpected elements (e.g., experimental music)
generator.generate(
    prompt="A fusion of jazz and electronic elements with unorthodox rhythms and harmonies",
    temperature=1.8,
    cfg_coef=2.0
)

# For balanced, production-ready music (e.g., for video background)
generator.generate(
    prompt="An uplifting pop track with acoustic guitar, piano, and light percussion",
    temperature=1.0,
    cfg_coef=3.0
)
```

## Handling Common Music Generation Issues

When generating music, you might encounter certain issues. Here's how to address them:

### Problem: Repetitive or Stuck Patterns

**Solution**: Increase temperature to introduce more variation:

```python
# If music is too repetitive
generator.generate(
    prompt=prompt,
    temperature=1.4,  # Higher temperature
    top_k=500  # Consider more token options
)
```

### Problem: Incoherent or Chaotic Music

**Solution**: Lower temperature and increase cfg_coef for more structure:

```python
# If music lacks coherence
generator.generate(
    prompt=prompt,
    temperature=0.7,  # Lower temperature
    cfg_coef=4.5  # Stronger adherence to prompt
)
```

### Problem: Missing Specified Instruments

**Solution**: Make instrumentation more prominent in your prompt:

```python
# Before: "A jazz piece with saxophone"

# After: "A jazz piece featuring a prominent saxophone lead playing the melody, accompanied by piano, bass, and drums"
```

### Problem: Weak Musical Structure

**Solution**: Explicitly describe the desired structure in your prompt:

```python
# Before: "An electronic dance track"

# After: "An electronic dance track with a clear intro, verse with subtle elements, building pre-chorus, and energetic chorus with a drop. The track has a distinct breakdown in the middle before returning to the main chorus."
```

## Music Generation for Specific Moods

Mood is a critical aspect of music for projects. Here's how to generate music for different emotional states:

### Uplifting/Positive

```python
uplifting_prompt = "An uplifting and inspirational orchestral piece with soaring strings, bright brass, and positive chord progressions in a major key. The music conveys optimism, achievement, and triumph over adversity."

generator.generate(prompt=uplifting_prompt)
```

### Tense/Suspenseful

```python
suspense_prompt = "A tense orchestral piece with tremolo strings, occasional dissonant brass stabs, and a steady, heartbeat-like percussion. The music builds suspense with minor keys, chromatic movements, and moments of unsettling silence."

generator.generate(prompt=suspense_prompt)
```

### Sad/Emotional

```python
emotional_prompt = "A melancholic piano piece with emotional string accompaniment in a minor key. The music expresses sadness and longing through slow, deliberate notes, gentle crescendos, and heartfelt melodic phrases."

generator.generate(prompt=emotional_prompt)
```

### Mysterious/Magical

```python
mysterious_prompt = "A mysterious and magical soundscape with ethereal harps, celesta, soft choir, and shimmering percussion. The music creates a sense of wonder and discovery through unusual scales and airy textures."

generator.generate(prompt=mysterious_prompt)
```

## Case Study: Creating a Game Soundtrack

Let's walk through a real-world example of creating a complete soundtrack for an indie game. We'll generate several tracks for different game contexts:

```python
from musicgen_application import MusicGenerator

def create_game_soundtrack(game_title, output_dir):
    """Create a complete game soundtrack."""
    
    # Initialize generator with larger model for quality
    generator = MusicGenerator(model_size="large", output_dir=output_dir)
    
    # Define soundtrack requirements
    soundtrack = {
        "main_theme": "An epic orchestral theme with heroic brass, powerful strings, and dramatic percussion. The music conveys adventure and courage with a memorable, hummable main melody that builds to a powerful climax.",
        
        "exploration": "A peaceful, ambient piece with gentle harp arpeggios, soft woodwinds, and sustained string pads. The music creates a sense of wonder and discovery with a relaxed tempo and major key harmonies.",
        
        "battle": "An intense, driving orchestral piece with aggressive percussion, staccato strings, and bold brass hits. The music creates tension and excitement with a fast tempo, minor key, and rhythmic ostinatos.",
        
        "sad_moment": "A melancholic piece with solo piano and emotional string accompaniment. The music conveys sadness and reflection with a slow tempo, minor key, and expressive melodic phrases.",
        
        "victory": "A triumphant orchestral fanfare with celebratory brass, uplifting strings, and triumphant percussion. The music expresses achievement and joy with major key harmonies and resolving chord progressions."
    }
    
    # Generate each track with appropriate parameters
    results = {}
    
    # Main theme - longer duration, balanced parameters
    results["main_theme"] = generator.generate(
        prompt=soundtrack["main_theme"],
        duration=25.0,
        temperature=0.9,
        cfg_coef=3.5,
        filename=f"{game_title}/main_theme"
    )
    
    # Exploration - longer duration, lower temperature for consistency
    results["exploration"] = generator.generate(
        prompt=soundtrack["exploration"],
        duration=20.0,
        temperature=0.7,
        cfg_coef=3.0,
        filename=f"{game_title}/exploration"
    )
    
    # Battle - standard duration, higher temperature for intensity
    results["battle"] = generator.generate(
        prompt=soundtrack["battle"],
        duration=15.0,
        temperature=1.1,
        cfg_coef=3.2,
        filename=f"{game_title}/battle"
    )
    
    # Sad moment - gentle, with high adherence to prompt
    results["sad_moment"] = generator.generate(
        prompt=soundtrack["sad_moment"],
        duration=15.0,
        temperature=0.8,
        cfg_coef=4.0,
        filename=f"{game_title}/sad_moment"
    )
    
    # Victory - short and impactful
    results["victory"] = generator.generate(
        prompt=soundtrack["victory"],
        duration=10.0,
        temperature=0.9,
        cfg_coef=3.5,
        filename=f"{game_title}/victory"
    )
    
    # Generate variations of the exploration track for different areas
    variations = generator.generate_variations(
        prompt=soundtrack["exploration"] + " Variation with slightly different instrumentation but maintaining the same mood and theme.",
        count=2,
        duration=20.0,
        temperature=0.8,
        filename=f"{game_title}/exploration_variation"
    )
    
    for i, path in enumerate(variations):
        results[f"exploration_variation_{i+1}"] = path
    
    # Create a summary file
    summary_path = os.path.join(output_dir, game_title, "soundtrack_info.txt")
    with open(summary_path, "w") as f:
        f.write(f"{game_title} - Game Soundtrack\n")
        f.write("=" * 50 + "\n\n")
        
        for track_name, prompt in soundtrack.items():
            f.write(f"Track: {track_name}\n")
            f.write(f"Prompt: {prompt}\n")
            f.write(f"File: {results.get(track_name, 'Generation failed')}\n\n")
            
        f.write("Additional Variations:\n")
        for i in range(len(variations)):
            f.write(f"Exploration Variation {i+1}: {results.get(f'exploration_variation_{i+1}', 'Generation failed')}\n")
    
    return results

# Create a complete game soundtrack
soundtrack_files = create_game_soundtrack(
    game_title="Dragon_Quest_Adventures",
    output_dir="game_soundtracks"
)

print("Soundtrack generation complete!")
```

This case study demonstrates how to approach a real-world music generation task by:
- Breaking down requirements into specific tracks
- Crafting detailed prompts for each context
- Adjusting parameters to suit each musical need
- Creating variations for additional content
- Documenting the generation process

## Integrating MusicGen into Applications

You can integrate MusicGen into various application types:

### Web Application Integration

```python
import gradio as gr
from musicgen_application import MusicGenerator

# Initialize the generator
generator = MusicGenerator(model_size="medium")

def generate_music_for_web(prompt, duration, temperature):
    """Generate music for web interface."""
    try:
        output_file = generator.generate(
            prompt=prompt,
            duration=float(duration),
            temperature=float(temperature)
        )
        
        # Return audio file path for Gradio to display
        return output_file
    except Exception as e:
        return f"Error: {str(e)}"

# Create a simple web interface
demo = gr.Interface(
    fn=generate_music_for_web,
    inputs=[
        gr.Textbox(label="Music Description", lines=3, placeholder="Describe the music you want to generate..."),
        gr.Slider(minimum=5, maximum=30, value=10, step=5, label="Duration (seconds)"),
        gr.Slider(minimum=0.5, maximum=1.5, value=1.0, step=0.1, label="Temperature (creativity)")
    ],
    outputs=gr.Audio(label="Generated Music"),
    title="MusicGen - AI Music Generator",
    description="Generate custom music from text descriptions."
)

# Launch the web interface
demo.launch()
```

### Game Engine Integration

```python
# Example of a module that could be integrated with a game engine

class GameMusicManager:
    """Manages dynamic music generation for games."""
    
    def __init__(self):
        """Initialize the music manager."""
        self.generator = MusicGenerator(model_size="medium")
        self.loaded_model = False
        self.current_track = None
        self.music_cache = {}
    
    def initialize(self):
        """Initialize the model (call during game loading)."""
        self.generator.load_model()
        self.loaded_model = True
    
    def get_track(self, context, force_regenerate=False):
        """
        Get a music track for a game context.
        
        Args:
            context (str): Game context (e.g., "battle", "exploration")
            force_regenerate (bool): If True, regenerate even if cached
        
        Returns:
            str: Path to the audio file
        """
        # Check if we have this context cached
        if context in self.music_cache and not force_regenerate:
            return self.music_cache[context]
        
        # Define prompts for different game contexts
        prompts = {
            "main_menu": "An epic orchestral theme with heroic brass and strings, creating a sense of adventure and excitement for the start of a journey.",
            "exploration": "A peaceful ambient piece with gentle harp arpeggios and soft strings, creating a sense of wonder during exploration.",
            "battle": "An intense orchestral piece with driving percussion and minor key strings, building tension for combat sequences.",
            "victory": "A triumphant brass fanfare with uplifting strings and celebratory percussion.",
            "game_over": "A somber piano piece with minor key harmonies, conveying defeat and reflection."
        }
        
        # Get the appropriate prompt
        if context in prompts:
            prompt = prompts[context]
        else:
            prompt = f"Background music for {context} in a fantasy game"
        
        # Generate the track
        output_file = self.generator.generate(
            prompt=prompt,
            duration=20.0,  # Longer for game background
            temperature=0.9,
            filename=f"game_music/{context}"
        )
        
        # Cache the result
        if output_file:
            self.music_cache[context] = output_file
        
        return output_file
    
    def clean_up(self):
        """Free resources when game exits."""
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
```

## Hands-on Challenge

Now it's your turn to apply what you've learned. Try this challenge to deepen your understanding:

### Challenge: Mood-Adaptive Music Generator

Create a script that:
1. Takes a mood parameter (e.g., "happy," "sad," "tense," "peaceful")
2. Automatically generates appropriate prompt additions based on the mood
3. Adjusts generation parameters to suit the mood (e.g., lower temperature for sad music)
4. Generates variations for each mood
5. Creates a simple HTML page that showcases the different mood-based compositions

This challenge will reinforce your understanding of prompt engineering and parameter tuning for different musical characteristics.

## Key Takeaways

- Detailed prompts with genre, instrumentation, mood, and structure produce better results
- Different musical styles require different approaches to prompt engineering
- Generation parameters like temperature and cfg_coef significantly impact musical output
- Project-based generation creates cohesive sets of music for specific applications
- Structured frameworks make music generation more reliable and reusable

## Next Steps

Now that you understand the basics of MusicGen, you're ready to explore more advanced techniques:

- [Prompt Engineering for Music](/chapters/part2/prompt-engineering/): Learn advanced techniques for crafting effective music prompts
- [Parameter Optimization](/chapters/part2/parameter-optimization/): Fine-tune generation parameters for better results
- [Melody Conditioning](/chapters/part2/melody-conditioning/): Guide music generation with reference melodies

## Further Reading

- [MusicGen Research Paper](https://arxiv.org/abs/2306.05284): Simple and Controllable Music Generation
- [MusicGen Demo on Hugging Face](https://huggingface.co/spaces/facebook/MusicGen)
- [Prompt Engineering Guide](https://www.promptingguide.ai/): General techniques for effective prompting
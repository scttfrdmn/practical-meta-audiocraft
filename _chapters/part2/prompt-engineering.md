---
layout: chapter
title: "Chapter 6: Prompt Engineering for Music"
# Copyright © 2025 Scott Friedman.
# Licensed under Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.
difficulty: intermediate
estimated_time: 2.5 hours
scenario:
  quote: "I've been using MusicGen for my projects, but I'm frustrated with the inconsistent results. Sometimes I get exactly what I want, but other times the music is completely off. I need to figure out a more reliable approach to crafting prompts so I can consistently get the kind of music I'm looking for."
  persona: "Jamie Nguyen"
  role: "Independent Filmmaker"
next_steps:
  - title: "Parameter Optimization"
    url: "/chapters/part2/parameter-optimization/"
    description: "Fine-tune generation parameters for better results"
  - title: "Melody Conditioning"
    url: "/chapters/part2/melody-conditioning/"
    description: "Guide music generation with reference melodies"
  - title: "Advanced Music Generation Techniques"
    url: "/chapters/part2/advanced-techniques/"
    description: "Learn sophisticated approaches to music generation"
further_reading:
  - title: "Music Theory Basics"
    url: "https://www.musictheory.net/"
    description: "Learn musical terminology for more precise prompts"
  - title: "MusicGen Research Paper"
    url: "https://arxiv.org/abs/2306.05284"
    description: "Simple and Controllable Music Generation"
  - title: "Prompt Engineering Guide"
    url: "https://www.promptingguide.ai/"
    description: "General techniques for effective prompting"
---

# Chapter 6: Prompt Engineering for Music

## The Challenge

You've learned the basics of music generation with MusicGen, but you've discovered that the quality and relevance of the output heavily depends on how you formulate your prompts. Sometimes the generated music captures your vision perfectly, while other times it feels disconnected from what you described. This inconsistency can be frustrating, especially when you have specific musical needs for a project.

Creating truly effective prompts requires more than just basic descriptions. You need a systematic approach that leverages MusicGen's training and capabilities while avoiding common pitfalls. Different musical styles, moods, and contexts demand different prompt strategies, and knowing how to adapt your approach can be the difference between a generic result and a perfect musical match.

In this chapter, we'll develop advanced prompt engineering techniques specifically for music generation. You'll learn how to analyze and deconstruct music into promptable elements, build prompts with deliberate structure, and create specialized prompts for different musical contexts. By the end, you'll have a reliable framework for crafting prompts that consistently produce the music you need.

## Learning Objectives

By the end of this chapter, you'll be able to:

- Apply systematic prompt engineering techniques for music generation
- Analyze and deconstruct music into promptable components
- Create structured, effective prompts for different musical styles and contexts
- Use specialized music terminology to guide generation more precisely
- Develop prompt templates for consistent results across multiple generations
- Test and refine prompts through systematic experimentation
- Troubleshoot and improve problematic prompts

## Understanding the MusicGen Prompt Space

Before diving into specific techniques, it's important to understand how MusicGen "thinks" about music based on its training. Unlike traditional composition software that works with notes, scales, and explicit musical elements, MusicGen operates in a higher-level semantic space where text descriptions are connected to audio patterns.

### How MusicGen Interprets Prompts

MusicGen was trained on a large dataset of music paired with text descriptions. During this training, it learned to associate certain words and phrases with specific musical patterns, textures, and structures. This means that the prompt vocabulary you use directly activates particular "regions" of the model's learned space.

Think of it like this: When you use the word "jazz" in a prompt, you're not just giving the model a general category - you're activating a complex network of associations that includes:

- Typical jazz instrumentation (piano, bass, drums, saxophone)
- Common jazz chord progressions and harmonies
- Characteristic rhythmic patterns and swing feel
- Production aesthetics associated with jazz recordings

The same applies to descriptive terms like "upbeat," "melancholic," or "epic" - each word triggers specific patterns the model has learned. Understanding this connection between language and music is key to crafting effective prompts.

### The Hierarchy of Musical Elements in Prompts

When constructing prompts, it helps to understand the hierarchy of influence that different elements have on the generated music:

1. **Genre and Style**: The strongest influence, setting the fundamental character
2. **Instrumentation**: Strongly shapes the sound palette and texture
3. **Mood/Emotion**: Guides harmonic choices and expressive elements
4. **Tempo/Rhythm**: Establishes the energy and movement
5. **Structure**: Influences how the music develops over time
6. **Technical Elements**: Specific musical techniques or production details

In general, elements higher in this hierarchy have more influence on the output. For example, specifying "jazz" (genre) has a stronger effect than specifying "with a bridge section" (structure).

## Systematic Prompt Engineering Framework

Let's develop a systematic framework for constructing effective music prompts:

### 1. The GENRE-MOOD-INST-TECH Framework

A reliable structure for building prompts is the GENRE-MOOD-INST-TECH framework:

- **GENRE**: The musical style or genre
- **MOOD**: The emotional character or atmosphere
- **INST**: Key instrumentation or sonic elements
- **TECH**: Technical aspects like tempo, key, or structure

Here's how to apply this framework:

```
A [GENRE] piece with [MOOD] atmosphere, featuring [INST] with [TECH] characteristics.
```

**Example**:
```
A jazz fusion piece with energetic and playful atmosphere, featuring electric piano, slap bass, and saxophone with complex chord progressions and a moderate tempo.
```

### 2. The Mood-First Approach

For projects where emotion is the primary concern, you can use a mood-first approach:

```
A [MOOD] piece that creates a feeling of [EMOTION DETAILS], using [INST] in a [GENRE] style with [TECH] elements.
```

**Example**:
```
A melancholic piece that creates a feeling of gentle nostalgia and reflection, using piano and subtle string accompaniment in a contemporary classical style with a slow tempo and minor key harmonies.
```

### 3. The Cinematic Structure Approach

For film-like music with narrative development:

```
A [GENRE] piece that begins with [SECTION 1 DESCRIPTION], then develops into [SECTION 2 DESCRIPTION], before concluding with [SECTION 3 DESCRIPTION]. The music features [INST] and conveys [MOOD].
```

**Example**:
```
An orchestral piece that begins with quiet, mysterious strings and subtle percussion, then develops into a tense section with brass stabs and driving rhythms, before concluding with a triumphant, full orchestral resolution. The music features french horns, timpani, and string sections, and conveys a journey from uncertainty to victory.
```

## Music-Specific Vocabulary for Prompts

Using precise musical terminology can significantly improve your prompts. Here's a guide to effective musical vocabulary:

### Instrumentation Terminology

Instead of vague terms like "instruments," use specific instrument names:

**Basic**:
- Piano, guitar, drums, bass
- Strings, brass, woodwinds, percussion
- Synthesizer, electronic drums

**More Specific**:
- Upright piano, acoustic guitar, drum kit, double bass
- Violin section, trumpet ensemble, flute solo, timpani
- Analog synthesizer, TR-808 drum machine

**Examples**:
```
# Vague prompt
"A piece with keyboard and drums"

# Improved prompt
"A piece featuring a bright grand piano with jazz drum kit playing with brushes"
```

### Music Theory Elements

Incorporating music theory concepts can guide the harmonic and structural qualities:

**Harmonic Terms**:
- Major/minor key
- Modal (Dorian, Lydian, etc.)
- Chromatic, dissonant, consonant
- Chord progressions, arpeggios

**Rhythm Terms**:
- Tempo descriptors (allegro, adagio, moderato)
- Time signatures (4/4, 3/4, 6/8)
- Rhythmic patterns (syncopated, steady, polyrhythmic)

**Examples**:
```
# Basic prompt
"A happy sounding classical piece"

# Improved prompt
"A classical piece in D major with a moderate tempo (allegro moderato), featuring flowing arpeggios and a waltz-like 3/4 time signature"
```

### Production and Mixing Terms

For modern music, production aesthetics matter:

- Reverb, delay, echo
- Compressed, distorted, clean
- Lo-fi, hi-fi, vintage, modern
- Stereo field, panning, depth

**Examples**:
```
# Basic prompt
"An electronic track"

# Improved prompt
"An electronic track with heavily compressed drums, a wide stereo field for synthesizer pads, and vintage tape saturation for warmth"
```

## Genre-Specific Prompt Techniques

Different musical genres respond to different prompt strategies:

### Orchestral/Film Music Prompts

Orchestral music benefits from narrative structure and specific instrument sections:

```python
orchestral_prompt = """
A dynamic orchestral piece that begins with soft, mysterious string passages and delicate harp arpeggios. 
The music gradually builds tension with french horns and woodwinds joining, introducing a noble theme. 
The middle section intensifies with percussion and brass, creating a sense of conflict and drama,
before resolving into a triumphant finale with full orchestra playing in a major key.
The piece features prominent brass fanfares, timpani rolls, and soaring violin melodies.
"""
```

**Key techniques**:
- Describe the narrative arc (beginning, middle, end)
- Specify instrument sections rather than just "orchestra"
- Include dynamics (soft, building, intense, etc.)
- Reference thematic elements (noble theme, conflict, resolution)

### Electronic Music Prompts

Electronic music responds well to technical and production details:

```python
electronic_prompt = """
A high-energy electronic dance track with a driving four-on-the-floor kick at 128 BPM.
The track features a punchy, sidechained bass synth, arpeggiated lead synthesizers with moderate resonance,
and atmospheric pad sounds with long reverb tails. The arrangement follows a clear build-up structure
with a filtered breakdown in the middle, leading to an euphoric drop with wide, detuned synth chords.
The production has a modern, clean character with tight, compressed drums and deep sub-bass.
"""
```

**Key techniques**:
- Specify BPM (beats per minute)
- Describe synth characteristics (arpeggiated, resonance, detuned)
- Include production elements (sidechained, reverb, compressed)
- Detail the arrangement structure (build-up, breakdown, drop)

### Jazz Prompts

Jazz generation benefits from references to playing styles and harmonic approaches:

```python
jazz_prompt = """
A cool jazz piece in the style of Miles Davis' modal period, featuring a muted trumpet playing a relaxed,
melodic line over sophisticated chord voicings from a piano. The rhythm section maintains a laid-back swing feel,
with a walking bass line and brushed drums providing a subtle, responsive foundation. The harmonic language
uses extended chords and occasional modal passages, while maintaining a cohesive, bluesy sensibility throughout.
The piece includes space for brief improvisational solos from the trumpet and piano.
"""
```

**Key techniques**:
- Reference specific artists or eras for stylistic guidance
- Describe playing techniques (muted trumpet, brushed drums)
- Include jazz-specific concepts (modal, walking bass, swing)
- Mention improvisational elements

### Pop/Rock Prompts

Pop and rock benefit from structural clarity and production references:

```python
pop_prompt = """
An upbeat pop rock song with a catchy, radio-friendly chorus featuring bright electric guitars and a driving rhythm section.
The verses have a more restrained energy with clean electric guitar arpeggios and a four-on-the-floor kick pattern.
The pre-chorus builds tension with increasing dynamics and added synthesizer pads, leading to an anthemic chorus
with distorted power chords and a memorable melodic hook. The bridge section introduces a new perspective with
a half-time feel and atmospheric elements before returning to the final chorus with added harmonies and energy.
"""
```

**Key techniques**:
- Clearly define song sections (verse, chorus, bridge)
- Contrast elements between sections
- Specify guitar tones and playing styles
- Include structural development (building, increasing, etc.)

## Advanced Prompt Construction Techniques

Beyond the basic frameworks, here are advanced techniques to further improve your results:

### 1. Reference Artist Technique

MusicGen has been trained on diverse musical styles and can often recognize artist references:

```python
artist_reference_prompt = """
A neo-soul track in the style of D'Angelo, with warm Rhodes electric piano, deep pocket drumming,
and layered vocal harmonies. The music has an organic, slightly loose feel with complex jazz-influenced
chord progressions and a gentle head-nodding groove around 85 BPM.
"""
```

This technique works best when the artist has a distinctive, well-known sound. Be careful not to simply ask for a specific artist's song, as this might not work well. Instead, describe the characteristic elements of their style.

### 2. Era and Production Reference

Referencing specific time periods or production eras can help achieve consistent aesthetic results:

```python
era_reference_prompt = """
A 1980s synthwave track with gated reverb drums, analog synthesizer arpeggios, and retro production aesthetics.
The music evokes the sound of vintage science fiction soundtracks with warm analog pads, punchy bass synths,
and a cinematic atmosphere reminiscent of early electronic film scores.
"""
```

This works especially well for genres that have strong associations with particular time periods or production techniques.

### 3. Emotional Trajectory Mapping

For pieces where emotional development is important, map the emotional journey:

```python
emotional_trajectory_prompt = """
A piano-led instrumental that begins with a sense of uncertainty and gentle melancholy,
gradually building confidence and warmth as new elements are introduced. The middle section
introduces hopeful string elements that lift the piece emotionally, leading to a sense
of peaceful resolution and acceptance in the final section, where the piano returns to
the initial themes but transforms them into a major key with confident expression.
"""
```

This technique is particularly effective for narrative contexts like film music or game soundtracks.

### 4. Technical Constraint Injection

Sometimes explicitly including technical constraints helps focus the generation:

```python
technical_constraint_prompt = """
A folk-inspired acoustic piece in 6/8 time signature and D minor, featuring fingerpicked
guitar with occasional hammer-ons and pull-offs. The melody stays primarily within a
middle octave range, with subtle dynamic variations rather than dramatic changes.
The piece maintains a consistent tempo around 70 BPM without tempo changes.
"""
```

This is useful when you need the music to fit specific technical requirements, like suitable background music for voice-over that needs to stay in a certain dynamic range.

## Prompt Testing and Refinement Process

Crafting the perfect prompt often requires experimentation and refinement. Here's a systematic approach:

### 1. The A/B Testing Method

Start with a basic prompt and create variations that test one element at a time:

```python
# Base prompt
base_prompt = "An upbeat electronic track with melodic synths and driving rhythm"

# A/B test - Genre specificity 
variation_a = "An upbeat synthwave track with melodic synths and driving rhythm"
variation_b = "An upbeat electronic dance music track with melodic synths and driving rhythm"

# A/B test - Instrumentation detail
variation_c = "An upbeat electronic track with arpeggiated analog synthesizers, warm pad sounds, and driving 808 drum patterns"
variation_d = "An upbeat electronic track with melodic lead synthesizers, atmospheric textures, and driving drum machine rhythms"
```

Generate music with each variation and compare the results to understand which changes produce the desired effects.

### 2. The Iterative Expansion Method

Start with a minimal prompt and progressively add detail based on results:

**Iteration 1**: "A jazz piano trio"

**Iteration 2**: "A jazz piano trio with a relaxed swing feel and bluesy harmony"

**Iteration 3**: "A jazz piano trio with a relaxed swing feel and bluesy harmony, featuring a conversational interplay between piano and upright bass, with subtle brushed drums maintaining the groove"

This approach helps you identify which details actually matter for your desired outcome.

### 3. The Subtraction Method

Start with a detailed prompt and remove elements to see what's essential:

**Original**: "A cinematic orchestral piece with dramatic string passages, powerful brass fanfares, thunderous percussion, and a heroic theme in a major key, building to an emotional climax with full orchestra"

**Subtraction 1**: "A cinematic orchestral piece with dramatic string passages, powerful brass, and a heroic theme building to an emotional climax" (Removed percussion details and key specification)

**Subtraction 2**: "A cinematic orchestral piece with dramatic strings and brass, building to an emotional climax" (Further simplification)

This helps identify which elements are most influential and which can be removed without affecting the core result.

## Implementing a Prompt Engineering Workflow

Let's implement a systematic prompt engineering workflow that incorporates these techniques:

```python
# prompt_engineering_workflow.py
import torch
import os
import json
import time
from datetime import datetime
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

class MusicPromptEngineer:
    """A system for developing and testing music generation prompts."""
    
    def __init__(self, model_size="medium", output_dir="prompt_experiments"):
        """Initialize the prompt engineering system."""
        self.model_size = model_size
        self.output_dir = output_dir
        self.model = None
        self.device = self._get_device()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create experiment tracking file
        self.experiments_file = os.path.join(output_dir, "experiments.json")
        self.experiments = self._load_experiments()
    
    def _get_device(self):
        """Determine the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _load_experiments(self):
        """Load existing experiment data or create new file."""
        if os.path.exists(self.experiments_file):
            try:
                with open(self.experiments_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {"experiments": []}
        else:
            return {"experiments": []}
    
    def _save_experiments(self):
        """Save experiment data to disk."""
        with open(self.experiments_file, 'w') as f:
            json.dump(self.experiments, f, indent=2)
    
    def load_model(self):
        """Load the MusicGen model."""
        if self.model is not None:
            return
        
        self.model = MusicGen.get_pretrained(self.model_size)
        self.model.to(self.device)
    
    def run_experiment(self, experiment_name, prompts, generation_params=None):
        """
        Run a prompt engineering experiment with multiple prompts.
        
        Args:
            experiment_name (str): Name for this experiment
            prompts (dict): Dictionary of prompt name -> prompt text
            generation_params (dict, optional): Parameters for generation
            
        Returns:
            dict: Experiment results with paths to generated files
        """
        self.load_model()
        
        # Use default parameters if none provided
        if generation_params is None:
            generation_params = {
                "duration": 10.0,
                "temperature": 1.0,
                "top_k": 250,
                "top_p": 0.0,
                "cfg_coef": 3.0
            }
        
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = os.path.join(self.output_dir, f"{experiment_name}_{timestamp}")
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Set generation parameters
        self.model.set_generation_params(**generation_params)
        
        # Run experiment for each prompt
        results = {}
        for prompt_name, prompt_text in prompts.items():
            print(f"Generating for prompt: {prompt_name}")
            
            # Generate music
            try:
                wav = self.model.generate([prompt_text])
                
                # Save audio
                filename = f"{prompt_name}"
                output_path = os.path.join(experiment_dir, filename)
                audio_write(
                    output_path,
                    wav[0].cpu(),
                    self.model.sample_rate,
                    strategy="loudness",
                )
                
                # Record result
                results[prompt_name] = f"{output_path}.wav"
                print(f"Generated: {output_path}.wav")
                
            except Exception as e:
                results[prompt_name] = f"Error: {str(e)}"
                print(f"Error with prompt '{prompt_name}': {str(e)}")
                
            # Clean up between generations
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Save experiment details
        experiment_data = {
            "name": experiment_name,
            "timestamp": timestamp,
            "prompts": prompts,
            "generation_params": generation_params,
            "results": results,
            "model_size": self.model_size
        }
        
        self.experiments["experiments"].append(experiment_data)
        self._save_experiments()
        
        # Create HTML report
        self._create_experiment_report(experiment_dir, experiment_data)
        
        return results
    
    def _create_experiment_report(self, experiment_dir, experiment_data):
        """Create an HTML report for the experiment."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Prompt Experiment: {experiment_data['name']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
                h1, h2 {{ color: #333; }}
                .prompt-container {{ margin-bottom: 30px; border: 1px solid #ddd; padding: 20px; border-radius: 5px; }}
                .prompt-text {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; white-space: pre-wrap; }}
                .parameters {{ background-color: #f0f8ff; padding: 10px; border-radius: 5px; margin: 15px 0; }}
                audio {{ width: 100%; margin-top: 10px; }}
            </style>
        </head>
        <body>
            <h1>Prompt Experiment: {experiment_data['name']}</h1>
            <p>Timestamp: {experiment_data['timestamp']}</p>
            <p>Model: MusicGen {experiment_data['model_size']}</p>
            
            <div class="parameters">
                <h3>Generation Parameters:</h3>
                <ul>
        """
        
        # Add parameters
        for param, value in experiment_data['generation_params'].items():
            html_content += f"<li><strong>{param}:</strong> {value}</li>\n"
        
        html_content += """
                </ul>
            </div>
            
            <h2>Prompts and Results:</h2>
        """
        
        # Add each prompt and result
        for prompt_name, prompt_text in experiment_data['prompts'].items():
            result_path = experiment_data['results'].get(prompt_name, "Generation failed")
            audio_element = ""
            
            if not result_path.startswith("Error"):
                # Create relative path for HTML
                rel_path = os.path.basename(result_path)
                audio_element = f'<audio controls src="{rel_path}"></audio>'
            
            html_content += f"""
            <div class="prompt-container">
                <h3>Prompt: {prompt_name}</h3>
                <div class="prompt-text">{prompt_text}</div>
                {audio_element}
                <p>Result: {result_path}</p>
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        # Write HTML file
        with open(os.path.join(experiment_dir, "report.html"), "w") as f:
            f.write(html_content)
    
    def ab_test_prompts(self, base_prompt, variations, experiment_name="A/B_Test"):
        """
        Run an A/B test experiment with variations on a base prompt.
        
        Args:
            base_prompt (str): The baseline prompt
            variations (dict): Dictionary of variation name -> variation prompt
            experiment_name (str): Name for this experiment
            
        Returns:
            dict: Experiment results
        """
        prompts = {"base": base_prompt}
        prompts.update(variations)
        
        return self.run_experiment(experiment_name, prompts)
    
    def genre_exploration(self, template, genres, experiment_name="Genre_Exploration"):
        """
        Explore different genres using the same prompt template.
        
        Args:
            template (str): Prompt template with {genre} placeholder
            genres (list): List of genres to test
            experiment_name (str): Name for this experiment
            
        Returns:
            dict: Experiment results
        """
        prompts = {
            f"genre_{genre}": template.format(genre=genre)
            for genre in genres
        }
        
        return self.run_experiment(experiment_name, prompts)
    
    def detail_level_test(self, content, experiment_name="Detail_Level_Test"):
        """
        Test different levels of detail for the same musical content.
        
        Args:
            content (dict): Dictionary with 'minimal', 'moderate', and 'detailed' prompts
            experiment_name (str): Name for this experiment
            
        Returns:
            dict: Experiment results
        """
        # Ensure we have the expected levels
        expected_levels = ['minimal', 'moderate', 'detailed']
        prompts = {level: content.get(level, "Missing prompt") for level in expected_levels}
        
        return self.run_experiment(experiment_name, prompts)

def main():
    """Run example prompt engineering workflows."""
    engineer = MusicPromptEngineer(model_size="medium")
    
    # Example 1: A/B testing different aspects of a prompt
    ab_test_example = engineer.ab_test_prompts(
        base_prompt="An electronic dance track with a catchy melody",
        variations={
            "more_specific_genre": "A synthwave electronic dance track with a catchy melody",
            "more_instruments": "An electronic dance track with a catchy synth melody, pulsing bass, and four-on-the-floor kick drum",
            "more_structure": "An electronic dance track with a catchy melody, starting with a minimal intro, building to an energetic drop, and ending with a stripped-back outro",
            "more_technical": "An electronic dance track with a catchy melody in F minor at 128 BPM with side-chained compression on the bass"
        }
    )
    
    # Example 2: Genre exploration
    genre_template = "A {genre} piece with a contemplative mood, featuring piano and subtle orchestral elements"
    genre_example = engineer.genre_exploration(
        template=genre_template,
        genres=["classical", "jazz", "ambient", "cinematic", "folk"]
    )
    
    # Example 3: Testing level of detail
    detail_test = engineer.detail_level_test({
        "minimal": "A piano solo with strings",
        "moderate": "A melancholic piano solo with soft string accompaniment in a minor key",
        "detailed": "A melancholic piano solo with expressive rubato timing and delicate touch, accompanied by soft, legato string sections playing sustained chords in A minor. The piece has a gentle dynamic range with occasional swells in the strings to highlight emotional moments, and maintains a slow, contemplative tempo throughout."
    })
    
    print("\nExperiments completed! Review the results in the output directory.")

if __name__ == "__main__":
    main()
```

This comprehensive prompt engineering system allows you to:
- Run controlled experiments with different prompt variations
- Test genre explorations using the same template
- Evaluate the effect of different levels of detail
- Generate HTML reports to compare results

## Genre-Specific Prompt Templates

Let's create a library of effective prompt templates for different genres. These templates include placeholders that you can customize for your specific needs:

### Orchestral/Film Music Template

```
An orchestral {mood} piece in a {style} style. The music features {primary_instruments} with supporting {secondary_instruments}. The piece begins with {intro_description}, develops into {middle_description}, and concludes with {finale_description}. The overall atmosphere conveys a sense of {emotion}.
```

**Example**:
```
An orchestral epic piece in a modern cinematic style. The music features french horns and timpani with supporting string sections and choir. The piece begins with quiet, mysterious string passages, develops into a powerful and driving action sequence, and concludes with a triumphant, heroic finale. The overall atmosphere conveys a sense of adventure and victory.
```

### Electronic Music Template

```
A {subgenre} electronic track at {tempo} BPM. The production features {key_elements} with {processing_technique}. The arrangement starts with {intro}, builds with {buildup_elements}, and peaks with {climax_description}. The overall sound has a {sound_character} quality with {additional_details}.
```

**Example**:
```
A deep house electronic track at 124 BPM. The production features a subby bass and filtered synth chords with sidechain compression. The arrangement starts with atmospheric pads and a minimal kick-hat pattern, builds with gradually opening filters and rising percussion layers, and peaks with a full frequency chord progression and vocal samples. The overall sound has a warm, analog quality with subtle tape saturation on the drum bus.
```

### Jazz Template

```
A {jazz_style} jazz piece featuring {lead_instrument} with {rhythm_section_description}. The performance has a {feel} feel with {harmonic_approach} harmonies. The piece includes {additional_elements} and demonstrates {playing_characteristics}.
```

**Example**:
```
A cool jazz piece featuring muted trumpet with a rhythm section of upright bass, piano, and brushed drums. The performance has a relaxed, late-night feel with extended harmonies and modal passages. The piece includes tasteful improvised solos and demonstrates restrained dynamics with occasional flourishes of technical skill.
```

### Ambient/Atmospheric Template

```
An ambient {mood} soundscape with {texture_elements} creating a {atmosphere} atmosphere. The piece evolves slowly through {progression_description} with {detail_elements} adding depth and interest. The frequency spectrum emphasizes {frequency_focus} with {space_description} spatial characteristics.
```

**Example**:
```
An ambient meditative soundscape with layered synthesizer pads and processed field recordings creating a serene and otherworldly atmosphere. The piece evolves slowly through gradual timbre modulations and subtle harmonic shifts with occasional piano notes and distant bird calls adding depth and interest. The frequency spectrum emphasizes mid and high frequencies with wide, expansive spatial characteristics.
```

## Prompt Analysis Case Studies

Let's analyze some real-world examples to understand what makes them effective:

### Case Study 1: Effective Orchestral Prompt

**Prompt**:
```
A majestic orchestral fantasy theme with a heroic brass melody, soaring string sections, and powerful percussion. The piece begins with a mysterious introduction featuring harp arpeggios and soft woodwinds before the main theme enters with french horns. The middle section builds intensity with timpani rolls and full string sections, developing the melodic themes with counterpoint. The piece concludes with a triumphant resolution bringing all orchestral sections together in a grand finale.
```

**Analysis**:

1. **Genre Clarity**: "Orchestral fantasy theme" immediately establishes the style
2. **Instrumentation Specificity**: Names specific instruments (french horns, harp, timpani) rather than just saying "orchestra"
3. **Structural Guidance**: Clear three-part structure (mysterious intro → building middle → triumphant conclusion)
4. **Musical Techniques**: Mentions specific musical elements (arpeggios, counterpoint, resolution)
5. **Emotional Direction**: Clear emotional journey from "mysterious" to "triumphant"

### Case Study 2: Problematic Pop Prompt

**Original Prompt**:
```
A pop song with vocals and catchy tune.
```

**Problems**:
1. Too vague - "pop" spans many substyles
2. No instrumentation details
3. No structural elements
4. MusicGen actually doesn't produce lyrics/vocals effectively
5. "Catchy" is subjective and not descriptive

**Improved Prompt**:
```
An upbeat modern pop instrumental in the style of recent chart hits, featuring a bright piano and synthesizer melody, energetic drum programming with trap hi-hat patterns, and a deep 808 bass. The arrangement follows a clear verse-chorus structure with a distinctive hook that repeats throughout the chorus sections. The bridge introduces a filter sweep and half-time feel before the final chorus.
```

**Improvements**:
1. Specified "instrumental" to avoid vocal generation issues
2. Added time reference ("modern" and "recent chart hits")
3. Detailed specific instruments and production elements
4. Included specific structural elements
5. Added distinctive production techniques (filter sweep, half-time)

## Testing Prompts Across Parameters

Different prompts may require different generation parameters. Let's explore this relationship:

```python
def test_prompt_parameter_matrix(prompt, experiment_name="Parameter_Matrix_Test"):
    """Test a prompt across a matrix of parameters."""
    engineer = MusicPromptEngineer(model_size="medium")
    
    # Define parameter combinations to test
    temperatures = [0.5, 1.0, 1.5]
    cfg_coefs = [1.5, 3.0, 5.0]
    
    all_prompts = {}
    
    # Create a version for each parameter combination
    for temp in temperatures:
        for cfg in cfg_coefs:
            prompt_name = f"temp_{temp}_cfg_{cfg}"
            all_prompts[prompt_name] = prompt
    
    # Create parameter sets for each combination
    param_sets = {}
    for prompt_name in all_prompts.keys():
        temp = float(prompt_name.split("_")[1])
        cfg = float(prompt_name.split("_")[3])
        
        param_sets[prompt_name] = {
            "duration": 10.0,
            "temperature": temp,
            "top_k": 250,
            "top_p": 0.0,
            "cfg_coef": cfg
        }
    
    # Run a customized experiment with different parameters per prompt
    engineer.load_model()
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(engineer.output_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Run experiment for each prompt with its specific parameters
    results = {}
    for prompt_name, prompt_text in all_prompts.items():
        print(f"Generating for {prompt_name}")
        
        # Set specific parameters for this prompt
        params = param_sets[prompt_name]
        engineer.model.set_generation_params(**params)
        
        # Generate music
        try:
            wav = engineer.model.generate([prompt_text])
            
            # Save audio
            output_path = os.path.join(experiment_dir, prompt_name)
            audio_write(
                output_path,
                wav[0].cpu(),
                engineer.model.sample_rate,
                strategy="loudness",
            )
            
            # Record result
            results[prompt_name] = f"{output_path}.wav"
            print(f"Generated: {output_path}.wav")
            
        except Exception as e:
            results[prompt_name] = f"Error: {str(e)}"
            print(f"Error with {prompt_name}: {str(e)}")
            
        # Clean up between generations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Save experiment details
    experiment_data = {
        "name": experiment_name,
        "timestamp": timestamp,
        "base_prompt": prompt,
        "parameter_sets": param_sets,
        "results": results,
        "model_size": engineer.model_size
    }
    
    engineer.experiments["experiments"].append(experiment_data)
    engineer._save_experiments()
    
    # Create HTML report
    create_parameter_matrix_report(experiment_dir, experiment_data)
    
    return results

def create_parameter_matrix_report(experiment_dir, experiment_data):
    """Create an HTML report for the parameter matrix experiment."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Parameter Matrix Test: {experiment_data['name']}</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
            h1, h2 {{ color: #333; }}
            .prompt-box {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin: 20px 0; white-space: pre-wrap; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: center; }}
            th {{ background-color: #f2f2f2; }}
            audio {{ width: 100%; }}
            .temp-low {{ background-color: #e6f7ff; }}
            .temp-med {{ background-color: #ffffff; }}
            .temp-high {{ background-color: #fff0f0; }}
        </style>
    </head>
    <body>
        <h1>Parameter Matrix Test</h1>
        <p>Experiment: {experiment_data['name']}</p>
        <p>Timestamp: {experiment_data['timestamp']}</p>
        <p>Model: MusicGen {experiment_data['model_size']}</p>
        
        <h2>Base Prompt:</h2>
        <div class="prompt-box">{experiment_data['base_prompt']}</div>
        
        <h2>Results Matrix (Temperature × CFG Coefficient):</h2>
        <table>
            <tr>
                <th>CFG / Temp</th>
    """
    
    # Extract unique temperatures and cfg values
    temperatures = sorted(set([p["temperature"] for p in experiment_data["parameter_sets"].values()]))
    cfg_values = sorted(set([p["cfg_coef"] for p in experiment_data["parameter_sets"].values()]))
    
    # Create header row with temperatures
    for temp in temperatures:
        html_content += f"<th>{temp}</th>\n"
    
    html_content += "</tr>\n"
    
    # Create rows for each CFG value
    for cfg in cfg_values:
        html_content += f"<tr><th>{cfg}</th>\n"
        
        for temp in temperatures:
            # Find the matching result
            prompt_name = f"temp_{temp}_cfg_{cfg}"
            result_path = experiment_data["results"].get(prompt_name, "Missing")
            
            temp_class = "temp-med"
            if temp < 1.0:
                temp_class = "temp-low"
            elif temp > 1.0:
                temp_class = "temp-high"
            
            if "Error" in result_path or result_path == "Missing":
                html_content += f'<td class="{temp_class}">Error</td>\n'
            else:
                rel_path = os.path.basename(result_path)
                html_content += f'<td class="{temp_class}"><audio controls src="{rel_path}"></audio></td>\n'
        
        html_content += "</tr>\n"
    
    html_content += """
        </table>
        
        <h2>Parameter Details:</h2>
        <ul>
    """
    
    # Add parameter descriptions
    html_content += f"<li><strong>Temperature</strong>: Controls randomness/creativity (Low: {min(temperatures)} / High: {max(temperatures)})</li>\n"
    html_content += f"<li><strong>CFG Coefficient</strong>: Controls adherence to prompt (Low: {min(cfg_values)} / High: {max(cfg_values)})</li>\n"
    
    html_content += """
        </ul>
        <p>Lower temperature = more consistent, predictable output</p>
        <p>Higher temperature = more varied, potentially creative output</p>
        <p>Lower CFG = more musical freedom, less adherence to prompt</p>
        <p>Higher CFG = stricter adherence to prompt, potentially at expense of musical quality</p>
    </body>
    </html>
    """
    
    # Write HTML file
    with open(os.path.join(experiment_dir, "parameter_matrix_report.html"), "w") as f:
        f.write(html_content)
```

## Creating a Prompt Library

Organizing your successful prompts can save time and ensure consistency. Here's how to create a reusable prompt library:

```python
class MusicPromptLibrary:
    """A library of reusable music generation prompts."""
    
    def __init__(self, library_file="prompt_library.json"):
        """Initialize the prompt library."""
        self.library_file = library_file
        self.prompts = self._load_library()
    
    def _load_library(self):
        """Load existing library or create new one."""
        if os.path.exists(self.library_file):
            try:
                with open(self.library_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {"categories": {}, "templates": {}}
        else:
            return {"categories": {}, "templates": {}}
    
    def _save_library(self):
        """Save library to disk."""
        with open(self.library_file, 'w') as f:
            json.dump(self.prompts, f, indent=2)
    
    def add_prompt(self, name, prompt_text, category=None, tags=None, notes=None, parameters=None):
        """
        Add a prompt to the library.
        
        Args:
            name (str): Name for this prompt
            prompt_text (str): The prompt text
            category (str, optional): Category for organization
            tags (list, optional): List of tags for searching
            notes (str, optional): Additional notes about this prompt
            parameters (dict, optional): Recommended generation parameters
        """
        # Create entry
        entry = {
            "text": prompt_text,
            "category": category,
            "tags": tags or [],
            "notes": notes or "",
            "parameters": parameters or {},
            "created": datetime.now().isoformat()
        }
        
        # Add to library
        self.prompts.setdefault("categories", {})
        if category:
            self.prompts["categories"].setdefault(category, [])
            self.prompts["categories"][category].append(name)
        
        # Store the prompt
        self.prompts[name] = entry
        self._save_library()
    
    def add_template(self, name, template_text, placeholders=None, example=None, notes=None):
        """
        Add a prompt template with placeholders.
        
        Args:
            name (str): Template name
            template_text (str): Template with {placeholders}
            placeholders (dict, optional): Description of each placeholder
            example (dict, optional): Example values for placeholders
            notes (str, optional): Usage notes
        """
        # Create entry
        entry = {
            "text": template_text,
            "placeholders": placeholders or {},
            "example": example or {},
            "notes": notes or "",
            "created": datetime.now().isoformat()
        }
        
        # Store the template
        self.prompts.setdefault("templates", {})
        self.prompts["templates"][name] = entry
        self._save_library()
    
    def get_prompt(self, name):
        """Get a prompt by name."""
        return self.prompts.get(name, {}).get("text", "Prompt not found")
    
    def get_template(self, name):
        """Get a template by name."""
        return self.prompts.get("templates", {}).get(name, {}).get("text", "Template not found")
    
    def fill_template(self, template_name, **kwargs):
        """
        Fill a template with provided values.
        
        Args:
            template_name (str): Name of the template
            **kwargs: Values for placeholders
            
        Returns:
            str: Filled template
        """
        template = self.get_template(template_name)
        try:
            return template.format(**kwargs)
        except KeyError as e:
            missing = str(e).strip("'")
            return f"Error: Missing required placeholder '{missing}'"
    
    def search(self, query, in_tags=True, in_text=True, category=None):
        """
        Search the prompt library.
        
        Args:
            query (str): Search term
            in_tags (bool): Search in tags
            in_text (bool): Search in prompt text
            category (str, optional): Limit to a specific category
            
        Returns:
            list: Matching prompt names
        """
        results = []
        
        for name, entry in self.prompts.items():
            # Skip categories and templates entries
            if name in ["categories", "templates"]:
                continue
                
            # Skip if category doesn't match
            if category and entry.get("category") != category:
                continue
            
            # Check tags
            if in_tags and query.lower() in [tag.lower() for tag in entry.get("tags", [])]:
                results.append(name)
                continue
                
            # Check text
            if in_text and query.lower() in entry.get("text", "").lower():
                results.append(name)
                
        return results
    
    def list_categories(self):
        """List all categories in the library."""
        return list(self.prompts.get("categories", {}).keys())
    
    def list_templates(self):
        """List all templates in the library."""
        return list(self.prompts.get("templates", {}).keys())
    
    def export_html(self, output_file="prompt_library.html"):
        """Export the library as an HTML document."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Music Prompt Library</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
                h1, h2, h3 { color: #333; }
                .category { margin-bottom: 40px; }
                .prompt { margin-bottom: 30px; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }
                .prompt-text { background-color: #f5f5f5; padding: 15px; border-radius: 5px; white-space: pre-wrap; margin: 10px 0; }
                .template { margin-bottom: 30px; border: 1px solid #4d94ff; padding: 15px; border-radius: 5px; background-color: #f0f7ff; }
                .template-text { background-color: #ffffff; padding: 15px; border-radius: 5px; white-space: pre-wrap; margin: 10px 0; }
                .tags { margin-top: 10px; }
                .tag { display: inline-block; background-color: #e6e6e6; padding: 3px 8px; border-radius: 10px; margin-right: 5px; font-size: 0.8em; }
                .parameters { background-color: #f9f9f9; padding: 10px; border-radius: 5px; margin-top: 10px; }
                .notes { font-style: italic; margin-top: 10px; }
                .placeholders { margin-top: 10px; }
                .placeholder { margin-bottom: 5px; }
            </style>
        </head>
        <body>
            <h1>Music Prompt Library</h1>
            <p>Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
            
            <h2>Templates</h2>
            <div class="templates">
        """
        
        # Add templates
        for template_name, template in self.prompts.get("templates", {}).items():
            html_content += f"""
            <div class="template">
                <h3>{template_name}</h3>
                <div class="template-text">{template.get('text', '')}</div>
            """
            
            # Add placeholders
            if template.get("placeholders"):
                html_content += "<div class='placeholders'><strong>Placeholders:</strong><ul>"
                for ph_name, ph_desc in template.get("placeholders", {}).items():
                    html_content += f"<li class='placeholder'><code>{ph_name}</code>: {ph_desc}</li>"
                html_content += "</ul></div>"
            
            # Add example
            if template.get("example"):
                example_text = template.get("text", "").format(**template.get("example", {}))
                html_content += f"""
                <div class='example'>
                    <strong>Example:</strong>
                    <div class='prompt-text'>{example_text}</div>
                    <strong>Values:</strong><ul>
                """
                for k, v in template.get("example", {}).items():
                    html_content += f"<li><code>{k}</code>: {v}</li>"
                html_content += "</ul></div>"
            
            # Add notes
            if template.get("notes"):
                html_content += f"<div class='notes'><strong>Notes:</strong> {template.get('notes', '')}</div>"
                
            html_content += "</div>"
        
        html_content += """
            </div>
            
            <h2>Prompts by Category</h2>
        """
        
        # Add prompts by category
        for category, prompt_names in self.prompts.get("categories", {}).items():
            html_content += f"""
            <div class="category">
                <h3>{category}</h3>
            """
            
            for name in prompt_names:
                prompt = self.prompts.get(name, {})
                
                html_content += f"""
                <div class="prompt">
                    <h4>{name}</h4>
                    <div class="prompt-text">{prompt.get('text', '')}</div>
                """
                
                # Add tags
                if prompt.get("tags"):
                    html_content += "<div class='tags'>"
                    for tag in prompt.get("tags", []):
                        html_content += f"<span class='tag'>{tag}</span>"
                    html_content += "</div>"
                
                # Add parameters
                if prompt.get("parameters"):
                    html_content += "<div class='parameters'><strong>Recommended Parameters:</strong><ul>"
                    for param, value in prompt.get("parameters", {}).items():
                        html_content += f"<li><code>{param}</code>: {value}</li>"
                    html_content += "</ul></div>"
                
                # Add notes
                if prompt.get("notes"):
                    html_content += f"<div class='notes'><strong>Notes:</strong> {prompt.get('notes', '')}</div>"
                
                html_content += "</div>"
                
            html_content += "</div>"
        
        html_content += """
        </body>
        </html>
        """
        
        # Write HTML file
        with open(output_file, "w") as f:
            f.write(html_content)
            
        return output_file
```

This prompt library system allows you to:
- Save successful prompts with metadata
- Organize prompts by categories and tags
- Create reusable templates with placeholders
- Search your prompt collection
- Export a browsable HTML version of your library

## Hands-on Challenge

Now it's your turn to apply what you've learned. Try this multi-step challenge to explore prompt engineering in depth:

### Challenge: Genre Transformation Prompts

1. Choose a specific musical piece or style as a reference
2. Create prompts to transform this reference into at least three different genres
3. For each genre transformation:
   - Create three variations of the prompt with different levels of detail
   - Test each prompt with at least two different temperature settings
4. Compare the results and document which prompt strategies were most effective for each genre
5. Create a final "optimized" prompt for each genre transformation

This challenge will help you develop intuition for how different prompt elements influence generation across genres.

## Key Takeaways

- Effective music prompts follow specific frameworks (GENRE-MOOD-INST-TECH)
- Different musical genres respond better to different prompt structures
- A systematic approach to testing and refining prompts yields better results
- Technical and music-specific vocabulary significantly improves generations
- Prompt libraries and templates provide consistency across projects
- Parameter settings interact with prompts in complex ways

## Next Steps

Now that you've mastered prompt engineering for music, you're ready to explore:

- [Parameter Optimization](/chapters/part2/parameter-optimization/): Fine-tune generation parameters for better results
- [Melody Conditioning](/chapters/part2/melody-conditioning/): Guide music generation with reference melodies
- [Advanced Music Generation Techniques](/chapters/part2/advanced-techniques/): Learn sophisticated approaches to music generation

## Further Reading

- [Music Theory Basics](https://www.musictheory.net/): Learn musical terminology for more precise prompts
- [MusicGen Research Paper](https://arxiv.org/abs/2306.05284): Simple and Controllable Music Generation
- [Prompt Engineering Guide](https://www.promptingguide.ai/): General techniques for effective prompting
#!/usr/bin/env python3
# dynamic_audio_environment.py
import os
import torch
import argparse
import sys

# Add the parent directory to the path to import the pipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from examples.text_to_audio_pipeline import TextToAudioPipeline

class DynamicAudioEnvironment:
    """
    A system that generates atmospheric audio environments that can change
    dynamically based on parameters.
    """
    def __init__(self):
        """Initialize the dynamic audio environment."""
        self.pipeline = TextToAudioPipeline()
        self.pipeline.load_model("music", "small")
        self.pipeline.load_model("audio", "medium")
        
        # Define base prompts for different environment types
        self.environment_templates = {
            "forest": {
                "music": "Peaceful {mood} ambient music inspired by forest sounds with {instrument}",
                "ambient": "Forest environment with {time_of_day} atmosphere, {weather} conditions, and {wildlife} sounds",
            },
            "ocean": {
                "music": "{mood} ambient music with {instrument}, reminiscent of ocean waves and sea breeze",
                "ambient": "Ocean shore with {weather} conditions, {time_of_day} atmosphere, and {wildlife} sounds",
            },
            "city": {
                "music": "Urban {mood} background music with {instrument} for a city environment",
                "ambient": "{time_of_day} city sounds with {weather} conditions, {traffic_level} traffic, and {crowd_level} crowd noise",
            },
            "space": {
                "music": "Ethereal {mood} space-themed ambient music with {instrument}",
                "ambient": "Science fiction space environment with {spacecraft} sounds, {activity} noises, and distant {events}",
            }
        }
        
        # Parameter options
        self.parameters = {
            "mood": ["calm", "mysterious", "uplifting", "melancholic", "tense", "hopeful"],
            "instrument": ["piano", "synthesizer pads", "strings", "flute", "guitar", "digital textures"],
            "time_of_day": ["morning", "afternoon", "evening", "night", "dawn", "dusk"],
            "weather": ["clear", "rainy", "windy", "stormy", "foggy", "sunny"],
            "wildlife": ["birds", "insects", "small animals", "frogs", "distant animals", "minimal wildlife"],
            "traffic_level": ["light", "moderate", "heavy", "occasional", "distant", "none"],
            "crowd_level": ["quiet", "moderate", "busy", "sparse", "loud", "none"],
            "spacecraft": ["engine hum", "control room beeps", "airlock", "life support systems", "computers"],
            "activity": ["crew movement", "mechanical work", "communication signals", "docking procedures"],
            "events": ["asteroid impacts", "space debris", "alien signals", "warp drive", "distant explosions"]
        }
    
    def generate_environment(self, env_type, params=None, duration=15.0, output_dir="environments"):
        """
        Generate a complete audio environment.
        
        Args:
            env_type (str): Environment type ('forest', 'ocean', 'city', 'space')
            params (dict): Parameters to customize the environment
            duration (float): Length of audio in seconds
            output_dir (str): Directory to save output files
        """
        if env_type not in self.environment_templates:
            raise ValueError(f"Unknown environment type: {env_type}")
        
        # Use default parameters if none provided
        if params is None:
            params = {}
        
        # Fill in default parameters for any missing ones
        for param in self.parameters:
            if param not in params:
                # Use first option as default
                params[param] = self.parameters[param][0]
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Format the prompts with parameters
        music_template = self.environment_templates[env_type]["music"]
        ambient_template = self.environment_templates[env_type]["ambient"]
        
        music_prompt = music_template.format(**params)
        ambient_prompt = ambient_template.format(**params)
        
        print(f"Generating {env_type} environment with:")
        print(f"Music prompt: '{music_prompt}'")
        print(f"Ambient prompt: '{ambient_prompt}'")
        
        # Generate both components
        music_audio = self.pipeline.generate(music_prompt, "music", duration, temperature=0.7)
        ambient_audio = self.pipeline.generate(ambient_prompt, "audio", duration, temperature=1.0)
        
        # Save individual components
        music_file = self.pipeline.save_audio(music_audio, f"{env_type}_music", output_dir)
        ambient_file = self.pipeline.save_audio(ambient_audio, f"{env_type}_ambient", output_dir)
        
        # Mix them together
        mixed_audio = self.pipeline.mix_audio([music_audio, ambient_audio], weights=[0.6, 0.8])
        mixed_file = self.pipeline.save_audio(mixed_audio, f"{env_type}_environment", output_dir)
        
        return {
            "music": music_file,
            "ambient": ambient_file,
            "mixed": mixed_file
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate dynamic audio environments')
    parser.add_argument('--type', type=str, default='forest', 
                        choices=['forest', 'ocean', 'city', 'space'],
                        help='Environment type')
    parser.add_argument('--mood', type=str, default='calm',
                        help='Mood of the music')
    parser.add_argument('--instrument', type=str, default='piano',
                        help='Primary instrument for music')
    parser.add_argument('--time', type=str, default='morning',
                        help='Time of day')
    parser.add_argument('--weather', type=str, default='clear',
                        help='Weather conditions')
    parser.add_argument('--duration', type=float, default=15.0,
                        help='Duration in seconds')
    
    args = parser.parse_args()
    
    # Create the environment generator
    env_generator = DynamicAudioEnvironment()
    
    # Generate the environment with specified parameters
    params = {
        "mood": args.mood,
        "instrument": args.instrument,
        "time_of_day": args.time,
        "weather": args.weather
    }
    
    env_generator.generate_environment(
        args.type,
        params,
        duration=args.duration
    )
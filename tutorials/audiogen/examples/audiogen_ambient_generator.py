#!/usr/bin/env python3
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
    print(f"Loading AudioGen {model_size} model...")
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
    parser.add_argument('--temp', type=float, default=0.8,
                        help='Temperature value (default: 0.8)')
    parser.add_argument('--output', type=str, default='ambient_soundscapes',
                        help='Output directory (default: ambient_soundscapes)')
    
    args = parser.parse_args()
    
    generate_ambient_soundscape(
        environment=args.environment,
        duration=args.duration,
        model_size=args.model,
        temperature=args.temp,
        output_dir=args.output
    )
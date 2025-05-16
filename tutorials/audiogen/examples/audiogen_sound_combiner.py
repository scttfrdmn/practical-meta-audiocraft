#!/usr/bin/env python3
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
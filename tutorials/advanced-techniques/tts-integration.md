# Integrating Advanced TTS with AudioCraft

This tutorial covers how to integrate expressive Text-to-Speech (TTS) systems with AudioCraft to create comprehensive audio experiences that combine voice narration, sound effects, and music.

## Introduction

While AudioCraft excels at generating music and environmental sounds, it doesn't specifically focus on voice synthesis. By combining AudioCraft with modern TTS systems, we can create rich audio productions with:

- Expressive narration with emotion control
- Multiple character voices
- Perfect synchronization of speech, sound effects, and music
- Dynamically responsive audio environments

## Choosing a TTS System

Several open-source TTS systems offer voice variety and prosody control:

### 1. XTTS/YourTTS
- **Strengths**: Voice cloning, emotion control, multilingual
- **GitHub**: [Coqui TTS](https://github.com/coqui-ai/TTS)
- **Installation**: `pip install TTS`
- **Voice Control**: Provides speaker embeddings for voice cloning

### 2. Bark by Suno AI
- **Strengths**: High expressivity, multiple voices, semantic tokens
- **GitHub**: [Bark](https://github.com/suno-ai/bark)
- **Installation**: `pip install git+https://github.com/suno-ai/bark.git`
- **Voice Control**: Extensive prompt format for controlling emotion, pace

### 3. ElevenLabs
- **Strengths**: Exceptional quality, voice cloning, emotion control, multilingual
- **API**: [ElevenLabs API](https://api.elevenlabs.io/docs)
- **Installation**: `pip install elevenlabs`
- **Voice Control**: Fine-grained control through API parameters

### 4. Tortoise TTS
- **Strengths**: High quality, multi-voice with fine control
- **GitHub**: [Tortoise TTS](https://github.com/neonbjb/tortoise-tts)
- **Installation**: `pip install tortoise-tts`
- **Voice Control**: CVVP conditioning for voice cloning

### 5. Facebook's MMS (Massively Multilingual Speech)
- **Strengths**: Extensive language support, Meta integration
- **GitHub**: [fairseq](https://github.com/facebookresearch/fairseq)
- **Installation**: Complex, requires fairseq

## Basic Integration: Bark TTS with AudioCraft

Let's start with a simple example using Bark TTS with AudioCraft:

```python
# combined_audio_generation.py
import torch
import numpy as np
import torchaudio
import scipy.io.wavfile as wavfile
from audiocraft.models import MusicGen, AudioGen
from audiocraft.data.audio import audio_write
from bark import SAMPLE_RATE, generate_audio, preload_models

def generate_voice(text, output_path="voice_output.wav"):
    """Generate speech using Bark TTS"""
    print("Generating voice narration...")
    
    # Generate speech using Bark
    preload_models()  # Takes time on first run
    speech_array = generate_audio(text)
    
    # Save as WAV file
    wavfile.write(output_path, SAMPLE_RATE, speech_array)
    print(f"Voice saved to {output_path}")
    
    return speech_array, SAMPLE_RATE

def generate_background_music(prompt, duration=10.0, output_path="music_output.wav"):
    """Generate background music using MusicGen"""
    print(f"Generating background music: '{prompt}'")
    
    # Determine device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    
    # Load MusicGen model
    model = MusicGen.get_pretrained('small')
    model.to(device)
    
    # Set generation parameters
    model.set_generation_params(
        duration=duration,
        temperature=0.9,
        top_k=250,
        top_p=0.0,
    )
    
    # Generate music
    wav = model.generate([prompt])
    
    # Save the music
    audio_write(
        output_path.replace('.wav', ''),
        wav[0].cpu(),
        model.sample_rate,
        strategy="loudness"
    )
    print(f"Music saved to {output_path}")
    
    return wav[0].cpu().numpy(), model.sample_rate

def generate_sound_effect(prompt, duration=5.0, output_path="sfx_output.wav"):
    """Generate sound effect using AudioGen"""
    print(f"Generating sound effect: '{prompt}'")
    
    # Determine device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    # Load AudioGen model
    model = AudioGen.get_pretrained('medium')
    model.to(device)
    
    # Set generation parameters
    model.set_generation_params(
        duration=duration,
        temperature=1.0,
        top_k=250,
        top_p=0.0,
    )
    
    # Generate sound effect
    wav = model.generate([prompt])
    
    # Save the sound effect
    audio_write(
        output_path.replace('.wav', ''),
        wav[0].cpu(),
        model.sample_rate,
        strategy="loudness"
    )
    print(f"Sound effect saved to {output_path}")
    
    return wav[0].cpu().numpy(), model.sample_rate

def combine_audio_tracks(voice_path, music_path, sfx_path, output_path="combined_output.wav"):
    """Combine voice, music, and sound effect into a single audio file"""
    print("Combining audio tracks...")
    
    # Load the audio files
    voice, voice_sr = torchaudio.load(voice_path)
    music, music_sr = torchaudio.load(music_path)
    sfx, sfx_sr = torchaudio.load(sfx_path)
    
    # Convert to mono if stereo
    voice = voice[0] if voice.shape[0] > 1 else voice
    music = music[0] if music.shape[0] > 1 else music
    sfx = sfx[0] if sfx.shape[0] > 1 else sfx
    
    # Resample to the same sample rate (use the voice sample rate)
    if music_sr != voice_sr:
        music = torchaudio.functional.resample(music, music_sr, voice_sr)
    if sfx_sr != voice_sr:
        sfx = torchaudio.functional.resample(sfx, sfx_sr, voice_sr)
    
    # Make all audio tracks the same length (use the voice length)
    voice_length = voice.shape[0]
    
    # Trim or pad music
    if music.shape[0] > voice_length:
        music = music[:voice_length]
    else:
        padding = torch.zeros(voice_length - music.shape[0])
        music = torch.cat([music, padding])
    
    # Trim or pad sfx
    if sfx.shape[0] > voice_length:
        sfx = sfx[:voice_length]
    else:
        padding = torch.zeros(voice_length - sfx.shape[0])
        sfx = torch.cat([sfx, padding])
    
    # Mix the audio (voice at full volume, music and sfx at reduced volume)
    voice_weight = 1.0
    music_weight = 0.3  # Adjust as needed
    sfx_weight = 0.5    # Adjust as needed
    
    combined = (voice * voice_weight) + (music * music_weight) + (sfx * sfx_weight)
    
    # Normalize to prevent clipping
    max_val = torch.max(torch.abs(combined))
    if max_val > 1.0:
        combined = combined / max_val
    
    # Save the combined audio
    torchaudio.save(output_path, combined.unsqueeze(0), voice_sr)
    print(f"Combined audio saved to {output_path}")

if __name__ == "__main__":
    # Define our content
    narration_text = """
    [laughter] Welcome to our adventure! Today we're exploring the ancient forest.
    Can you hear the birds singing? [pause] Listen carefully to the sounds of nature.
    """
    
    music_prompt = "Mysterious ambient forest music with soft flutes and gentle strings"
    sfx_prompt = "Forest ambience with birds chirping, wind in leaves, and distant water stream"
    
    # Generate each audio component
    voice_array, voice_sr = generate_voice(narration_text, "voice_output.wav")
    generate_background_music(music_prompt, 15.0, "music_output.wav")
    generate_sound_effect(sfx_prompt, 15.0, "sfx_output.wav")
    
    # Combine all audio tracks
    combine_audio_tracks("voice_output.wav", "music_output.wav", "sfx_output.wav", "forest_adventure.wav")
    
    print("Complete! Your audio narrative has been created.")
```

## Advanced Voice Control with Bark

Bark TTS offers powerful voice customization through its prompt format. Here's an example:

```python
# Customized voice generation with Bark
from bark import generate_audio, SAMPLE_RATE
import scipy.io.wavfile as wavfile

# Voice prompt with control markers
prompt = """
[speaker: female_child, accent: American, emotion: excited]
Wow! Look at all the butterflies in this meadow!
[emotion: curious]
I wonder which one is the most colorful?
[emotion: happy, pace: slow]
This has been the best adventure ever!
"""

# Generate and save audio
audio_array = generate_audio(prompt)
wavfile.write("child_excited.wav", SAMPLE_RATE, audio_array)
```

## Synchronizing Audio Elements

For more precise synchronization, you'll need timestamps for your narration and sound effects:

```python
def create_timed_audio_experience(script, output_path="timed_output.wav"):
    """
    Creates a synchronized audio experience from a timed script
    
    Script format:
    [
        {"type": "voice", "text": "Welcome to the jungle", "start_time": 0.0},
        {"type": "sfx", "description": "jungle ambience", "start_time": 0.5, "duration": 10.0},
        {"type": "music", "description": "tribal drums", "start_time": 2.0, "duration": 8.0}
    ]
    """
    # Sort script by start time
    script.sort(key=lambda x: x["start_time"])
    
    # Determine total duration
    total_duration = 0
    for item in script:
        end_time = item["start_time"]
        if "duration" in item:
            end_time += item["duration"]
        total_duration = max(total_duration, end_time)
    
    # Add a small buffer at the end
    total_duration += 2.0
    
    # Initialize an empty audio canvas (44.1kHz sample rate)
    sample_rate = 44100
    total_samples = int(total_duration * sample_rate)
    canvas = np.zeros(total_samples)
    
    # Process each script item
    for item in script:
        start_sample = int(item["start_time"] * sample_rate)
        
        if item["type"] == "voice":
            # Generate voice
            voice_array, voice_sr = generate_voice(item["text"], "temp_voice.wav")
            voice_resampled = resample_audio(voice_array, voice_sr, sample_rate)
            
            # Add to canvas
            end_sample = min(start_sample + len(voice_resampled), total_samples)
            canvas[start_sample:end_sample] += voice_resampled[:end_sample-start_sample] * 1.0  # Full volume
            
        elif item["type"] == "sfx":
            # Generate sound effect
            sfx_array, sfx_sr = generate_sound_effect(
                item["description"], 
                item.get("duration", 5.0),
                "temp_sfx.wav"
            )
            sfx_resampled = resample_audio(sfx_array, sfx_sr, sample_rate)
            
            # Add to canvas
            end_sample = min(start_sample + len(sfx_resampled), total_samples)
            canvas[start_sample:end_sample] += sfx_resampled[:end_sample-start_sample] * 0.5  # Half volume
            
        elif item["type"] == "music":
            # Generate music
            music_array, music_sr = generate_background_music(
                item["description"],
                item.get("duration", 10.0),
                "temp_music.wav"
            )
            music_resampled = resample_audio(music_array, music_sr, sample_rate)
            
            # Add to canvas
            end_sample = min(start_sample + len(music_resampled), total_samples)
            canvas[start_sample:end_sample] += music_resampled[:end_sample-start_sample] * 0.3  # Lower volume
    
    # Normalize to prevent clipping
    max_amplitude = np.max(np.abs(canvas))
    if max_amplitude > 1.0:
        canvas = canvas / max_amplitude
    
    # Save the final audio
    wavfile.write(output_path, sample_rate, canvas.astype(np.float32))
    print(f"Timed audio experience saved to {output_path}")

def resample_audio(audio_array, orig_sr, target_sr):
    """Resample audio to the target sample rate"""
    # Convert to torch tensor for resampling
    if isinstance(audio_array, np.ndarray):
        audio_tensor = torch.from_numpy(audio_array).float()
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
    else:
        audio_tensor = audio_array
        
    # Resample
    resampled = torchaudio.functional.resample(audio_tensor, orig_sr, target_sr)
    
    # Return as numpy array
    if resampled.dim() > 1:
        resampled = resampled.squeeze(0)
    return resampled.numpy()

# Example usage
script = [
    {"type": "voice", "text": "Welcome to our magical journey through the enchanted forest.", "start_time": 0.0},
    {"type": "music", "description": "Soft magical fantasy music with harps and flutes", "start_time": 0.5, "duration": 30.0},
    {"type": "sfx", "description": "Forest ambience with birds and leaves", "start_time": 3.0, "duration": 20.0},
    {"type": "voice", "text": "Listen to the birds singing in the trees.", "start_time": 7.0},
    {"type": "sfx", "description": "Magical shimmer sound effect", "start_time": 12.0, "duration": 3.0},
    {"type": "voice", "text": "Oh look! A magical creature appears before us!", "start_time": 13.5}
]

create_timed_audio_experience(script, "enchanted_forest_journey.wav")
```

## Building an Audio Narrative Pipeline

For more complex projects, create a modular audio narrative pipeline:

1. **Script Parsing**: Convert a structured script to audio cues
2. **Asset Generation**: Generate all required audio assets
3. **Timeline Assembly**: Assemble assets on a precise timeline
4. **Mixing and Mastering**: Balance levels and apply effects

## Integrating Other TTS Systems

### ElevenLabs API Integration

[ElevenLabs](https://elevenlabs.io/) offers one of the most advanced commercially available TTS systems with exceptional voice quality, emotion control, and multilingual capabilities. Their API is straightforward to integrate with AudioCraft:

```python
import requests
import json
import os
from pathlib import Path

def generate_voice_elevenlabs(
    text, 
    voice_id="21m00Tcm4TlvDq8ikWAM", # Rachel voice (default)
    api_key=None,
    model_id="eleven_monolingual_v1",
    output_path="elevenlabs_output.wav"
):
    """
    Generate voice using ElevenLabs API
    
    Args:
        text (str): Text to convert to speech
        voice_id (str): ElevenLabs voice ID
            Some popular voices:
            - "21m00Tcm4TlvDq8ikWAM" (Rachel - conversational)
            - "AZnzlk1XvdvUeBnXmlld" (Domi - narrative)
            - "EXAVITQu4vr4xnSDxMaL" (Bella - children's stories)
            - "MF3mGyEYCl7XYWbV9V6O" (Adam - professional)
        api_key (str): Your ElevenLabs API key, if None will look for ELEVEN_API_KEY env var
        model_id (str): ElevenLabs model ID
        output_path (str): Path to save output audio file
        
    Returns:
        str: Path to output audio file
    """
    # Get API key from environment if not provided
    if api_key is None:
        api_key = os.environ.get("ELEVEN_API_KEY")
        if api_key is None:
            raise ValueError("ElevenLabs API key not found. Please provide api_key or set ELEVEN_API_KEY environment variable.")
    
    # Prepare request data
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": api_key
    }
    
    data = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {
            "stability": 0.5,         # Lower for more expressive, higher for more consistent
            "similarity_boost": 0.75,  # How much to prioritize sounding like reference
            "style": 0.0,             # Experimental; control speaking style (-1.0 to 1.0)
            "use_speaker_boost": True  # Enhances clarity and quality
        }
    }
    
    print(f"Generating speech with ElevenLabs API: {text[:50]}...")
    
    # Make API call
    response = requests.post(url, json=data, headers=headers)
    
    if response.status_code == 200:
        # Save audio file
        with open(output_path, "wb") as file:
            file.write(response.content)
        print(f"Audio saved to {output_path}")
        return output_path
    else:
        error_msg = f"ElevenLabs API Error: {response.status_code}"
        try:
            error_details = response.json()
            error_msg += f" - {error_details.get('detail', {}).get('message', 'Unknown error')}"
        except:
            pass
        print(error_msg)
        raise Exception(error_msg)

def generate_voice_elevenlabs_with_emotion(
    text,
    emotion="neutral", 
    voice_id="21m00Tcm4TlvDq8ikWAM",
    api_key=None,
    output_path="elevenlabs_emotion_output.wav"
):
    """
    Generate voice with specific emotion using ElevenLabs API
    
    Args:
        text (str): Text to convert to speech
        emotion (str): Desired emotion (neutral, happy, sad, angry, fear, surprise, etc.)
        voice_id (str): ElevenLabs voice ID
        api_key (str): Your ElevenLabs API key
        output_path (str): Path to save output audio file
    """
    # Map emotions to voice settings
    emotion_settings = {
        "neutral": {"stability": 0.5, "similarity_boost": 0.75, "style": 0.0},
        "happy": {"stability": 0.3, "similarity_boost": 0.65, "style": 0.3},
        "excited": {"stability": 0.2, "similarity_boost": 0.7, "style": 0.5},
        "sad": {"stability": 0.7, "similarity_boost": 0.8, "style": -0.3},
        "angry": {"stability": 0.3, "similarity_boost": 0.6, "style": -0.5},
        "fear": {"stability": 0.4, "similarity_boost": 0.7, "style": -0.4},
        "surprise": {"stability": 0.2, "similarity_boost": 0.6, "style": 0.4},
    }
    
    # Apply SSML markup for additional control (where supported)
    ssml_text = f"""
    <speak>
      <prosody pitch="{'+20%' if emotion in ['happy', 'excited', 'surprise'] else '-10%' if emotion in ['sad', 'fear'] else '+0%'}" 
               rate="{'+15%' if emotion in ['excited', 'angry'] else '-15%' if emotion in ['sad'] else '+0%'}">
        {text}
      </prosody>
    </speak>
    """
    
    # Get settings for the emotion
    settings = emotion_settings.get(emotion.lower(), emotion_settings["neutral"])
    
    # Create API request similar to the basic function, but with emotion settings
    # Implementation depends on API version and capabilities
    
    # Here we would make the actual API call with the SSML markup and settings
    # For simplicity, we'll call our base function and note that a complete implementation
    # would use the SSML formatting and emotion-specific settings
    print(f"Generating speech with emotion: {emotion}")
    return generate_voice_elevenlabs(
        text=text,  # Use SSML text for proper implementation
        voice_id=voice_id,
        api_key=api_key,
        output_path=output_path
    )

# Example usage:
# Set your API key in environment: export ELEVEN_API_KEY="your-api-key"
# generate_voice_elevenlabs("Hello world, this is a test of the ElevenLabs API integration.")
# generate_voice_elevenlabs_with_emotion("I'm so excited to meet you!", emotion="excited")
```

### Using ElevenLabs with AudioCraft

Here's an example of creating a complete audio experience combining ElevenLabs for narration with AudioCraft for background music and effects:

```python
def create_narrated_experience_with_elevenlabs(
    script,
    voice_id="21m00Tcm4TlvDq8ikWAM",
    background_music_prompt="Gentle ambient music with soft piano",
    api_key=None,
    output_path="narrated_experience.wav"
):
    """
    Create a complete audio experience with ElevenLabs narration and AudioCraft background
    
    Args:
        script (list): List of script segments, each a dict with text and optional emotion
        voice_id (str): ElevenLabs voice ID
        background_music_prompt (str): Prompt for AudioCraft music generation
        api_key (str): ElevenLabs API key
        output_path (str): Path to save final audio
    """
    # Generate background music with AudioCraft
    _, music_sr = generate_background_music(
        background_music_prompt,
        duration=30.0,  # Adjust based on expected narration length
        output_path="temp_background.wav"
    )
    
    # Generate all voice segments with ElevenLabs
    voice_segments = []
    for i, segment in enumerate(script):
        # Get text and emotion (default to neutral)
        text = segment["text"]
        emotion = segment.get("emotion", "neutral")
        
        # Generate voice for this segment
        segment_path = f"temp_segment_{i}.wav"
        generate_voice_elevenlabs_with_emotion(
            text=text,
            emotion=emotion,
            voice_id=voice_id,
            api_key=api_key,
            output_path=segment_path
        )
        voice_segments.append(segment_path)
    
    # Combine all voice segments
    # (In a real implementation, you would handle timing and transitions)
    concat_voice_path = "temp_voice_combined.wav"
    concatenate_audio_files(voice_segments, concat_voice_path)
    
    # Mix voice and background music
    mix_voice_and_background(
        voice_path=concat_voice_path,
        background_path="temp_background.wav",
        output_path=output_path,
        voice_level=1.0,
        background_level=0.3
    )
    
    # Clean up temporary files
    for path in voice_segments + ["temp_background.wav", concat_voice_path]:
        if os.path.exists(path):
            os.remove(path)
    
    print(f"Complete narrated experience saved to {output_path}")
    return output_path

# Helper function for concatenating audio files
def concatenate_audio_files(file_paths, output_path):
    """Concatenate multiple audio files into one"""
    import torch
    import torchaudio
    
    # Get info from first file to determine format
    info = torchaudio.info(file_paths[0])
    sample_rate = info.sample_rate
    
    # Load and concatenate all files
    segments = []
    for path in file_paths:
        waveform, sr = torchaudio.load(path)
        if sr != sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
        segments.append(waveform)
    
    # Concatenate along time dimension
    concatenated = torch.cat(segments, dim=1)
    
    # Save the concatenated audio
    torchaudio.save(output_path, concatenated, sample_rate)
    
    return output_path

# Example script
narration_script = [
    {"text": "Welcome to our audio journey.", "emotion": "neutral"},
    {"text": "Listen carefully as we explore the magical forest.", "emotion": "excited"},
    {"text": "Oh no! A storm is approaching.", "emotion": "fear"},
    {"text": "But don't worry, we've found shelter beneath the ancient trees.", "emotion": "happy"}
]

# create_narrated_experience_with_elevenlabs(narration_script)
```

### YourTTS/XTTS Example

```python
from TTS.api import TTS

def generate_voice_xtts(text, voice="en_female", output_path="xtts_output.wav"):
    """Generate voice using XTTS"""
    # Initialize TTS with XTTS model
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
    
    # Generate speech with the selected voice
    tts.tts_to_file(
        text=text,
        file_path=output_path,
        speaker=voice,  # Use different speaker IDs for voice variety
        language="en"
    )
    
    return output_path
```

### Tortoise TTS Example

```python
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio, load_voice

def generate_voice_tortoise(text, voice_samples="random", output_path="tortoise_output.wav"):
    """Generate voice using Tortoise TTS"""
    # Initialize Tortoise TTS
    tts = TextToSpeech()
    
    # Generate with specified voice or random preset
    if voice_samples == "random":
        # Use a random preset voice
        voice_samples = None
        preset = "standard"
    else:
        # Load custom voice samples
        voice_samples = load_voice(voice_samples)
        preset = None
    
    # Generate speech
    gen = tts.tts(
        text,
        voice_samples=voice_samples,
        preset=preset,
        k=1,  # Generate a single sample
        use_deterministic_seed=42  # For reproducibility
    )
    
    # Save audio
    tts.save_wav(gen, output_path)
    
    return output_path
```

## Full Audio Production Example

For a complete example of a narrated audio production that combines voice narration, sound effects, and music, see our [audio narrative production example](https://github.com/example/audio-narrative).

## Next Steps

- Experiment with different TTS systems to find the best voice quality and control
- Create scripts with precise timing for professional audio narratives
- Build a graphical timeline editor for easier audio production
- Explore additional AudioCraft model parameters for better sound quality

## References

- [Bark GitHub Repository](https://github.com/suno-ai/bark)
- [YourTTS/Coqui TTS Documentation](https://github.com/coqui-ai/TTS)
- [Tortoise TTS GitHub](https://github.com/neonbjb/tortoise-tts)
- [AudioCraft Documentation](https://github.com/facebookresearch/audiocraft)
- [Professional Audio Mixing Techniques](https://example.com)
# Chapter 16: Multimedia Integration

> *"We're building an immersive multimedia installation that combines real-time visuals with generated audio. How can we synchronize AudioCraft's output with other media like animations, procedural visuals, and interactive displays to create a cohesive experience?"*  
> — Creative Director, Interactive Art Studio

## Learning Objectives

In this chapter, you'll learn how to:

- Integrate AudioCraft-generated audio with other multimedia elements
- Synchronize audio generation with visual elements
- Create audio-visual integrations for different platforms
- Build systems that connect procedural audio and visuals
- Design reactive multimedia experiences driven by audio analysis

## Introduction

Audio rarely exists in isolation. In most interactive applications, audio works together with visuals, narrative, and user input to create a cohesive experience. This chapter explores techniques for integrating AudioCraft-generated audio with other multimedia elements to create unified, responsive experiences.

We'll explore both technical implementation and design concepts for various multimedia platforms, including games, web applications, installations, and virtual reality. By connecting audio generation to other media systems, we can create truly immersive, synchronized experiences.

## Implementation: Audio-Visual Synchronization System

First, let's create a system that connects audio generation to visual elements, ensuring proper synchronization:

```python
import os
import json
import time
import torch
import numpy as np
import torchaudio
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from audiocraft.models import MusicGen, AudioGen

@dataclass
class VisualizationParameters:
    """Parameters controlling visual aspects of the experience."""
    
    # Color parameters
    color_scheme: str = "default"  # e.g., "warm", "cool", "energetic"
    color_intensity: float = 0.5
    color_variation: float = 0.5
    
    # Motion parameters
    motion_speed: float = 0.5
    motion_complexity: float = 0.5
    motion_smoothness: float = 0.5
    
    # Particle systems
    particle_density: float = 0.5
    particle_size: float = 0.5
    particle_velocity: float = 0.5
    
    # Camera
    camera_movement: float = 0.5  # 0 = static, 1 = highly dynamic
    
    # Custom parameters
    custom_parameters: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {k: v for k, v in self.__dict__.items() if k != 'custom_parameters'} | self.custom_parameters

@dataclass
class AudioVisualSyncPoint:
    """Defines a synchronization point between audio and visuals."""
    
    # Timing
    time_seconds: float
    duration_seconds: float = 1.0
    
    # Audio characteristics at this point
    audio_intensity: float = 0.5
    audio_frequency_focus: str = "mid"  # "low", "mid", "high"
    
    # Visual event to trigger
    visual_event_type: str  # e.g., "color_shift", "particles", "camera"
    visual_event_parameters: Dict[str, float] = field(default_factory=dict)
    
    # Narrative context
    description: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {k: v for k, v in self.__dict__.items()}

class AudioVisualExperience:
    """
    System for creating synchronized audio-visual experiences.
    
    This class manages the generation of audio content and synchronization
    with visual elements for multimedia experiences.
    """
    
    def __init__(
        self,
        output_dir: str = "audio_visual",
        audio_model_type: str = "music",  # "music" or "audio"
        model_size: str = "small",
        device: str = None
    ):
        """
        Initialize the audio-visual experience system.
        
        Args:
            output_dir: Directory for output files
            audio_model_type: Type of audio model to use ("music" or "audio")
            model_size: Size of model to use
            device: Device to run on (cuda, mps, cpu)
        """
        self.output_dir = output_dir
        self.audio_model_type = audio_model_type
        self.model_size = model_size
        
        # Initialize device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize models (lazy loaded)
        self._music_model = None
        self._audio_model = None
        
        # Experience settings
        self.experience_name = "Untitled Experience"
        self.duration_seconds = 30.0
        self.audio_prompt = "Ambient electronic music with evolving texture"
        self.visual_base_parameters = VisualizationParameters()
        self.sync_points = []
    
    def get_model(self):
        """Lazy-load the appropriate audio model."""
        if self.audio_model_type == "music":
            if self._music_model is None:
                print(f"Loading MusicGen model ({self.model_size})...")
                self._music_model = MusicGen.get_pretrained(self.model_size)
                self._music_model.to(self.device)
            return self._music_model
        else:
            if self._audio_model is None:
                print(f"Loading AudioGen model ({self.audio_model_size})...")
                self._audio_model = AudioGen.get_pretrained(self.model_size)
                self._audio_model.to(self.device)
            return self._audio_model
    
    def set_experience_parameters(
        self,
        name: str,
        duration_seconds: float,
        audio_prompt: str,
        visual_parameters: Optional[VisualizationParameters] = None
    ):
        """
        Set basic parameters for the audio-visual experience.
        
        Args:
            name: Name of the experience
            duration_seconds: Total duration in seconds
            audio_prompt: Text prompt for audio generation
            visual_parameters: Base visualization parameters
        """
        self.experience_name = name
        self.duration_seconds = duration_seconds
        self.audio_prompt = audio_prompt
        
        if visual_parameters:
            self.visual_base_parameters = visual_parameters
    
    def add_sync_point(self, sync_point: AudioVisualSyncPoint):
        """
        Add a synchronization point to the experience.
        
        Args:
            sync_point: AudioVisualSyncPoint to add
        """
        self.sync_points.append(sync_point)
        
        # Sort sync points by time
        self.sync_points.sort(key=lambda sp: sp.time_seconds)
    
    def remove_sync_point(self, index: int):
        """
        Remove a synchronization point by index.
        
        Args:
            index: Index of the sync point to remove
        """
        if 0 <= index < len(self.sync_points):
            del self.sync_points[index]
    
    def generate_audio(
        self,
        generation_params: Dict = None
    ) -> Tuple[torch.Tensor, int]:
        """
        Generate audio for the experience.
        
        Args:
            generation_params: Optional generation parameters
            
        Returns:
            Tuple of (audio_tensor, sample_rate)
        """
        # Default generation parameters
        default_params = {
            "duration": self.duration_seconds,
            "temperature": 1.0,
            "cfg_coef": 3.0,
            "top_k": 250
        }
        
        # Merge with provided parameters
        gen_params = dict(default_params)
        if generation_params:
            gen_params.update(generation_params)
        
        # Load model
        model = self.get_model()
        model.set_generation_params(**gen_params)
        
        # Generate audio
        wav = model.generate([self.audio_prompt])
        
        return wav[0], model.sample_rate
    
    def analyze_audio(
        self,
        audio: torch.Tensor,
        sample_rate: int,
        n_segments: int = 10
    ) -> List[Dict]:
        """
        Analyze audio to extract features for visualization.
        
        Args:
            audio: Audio tensor
            sample_rate: Sample rate of audio
            n_segments: Number of segments to analyze
            
        Returns:
            List of dictionaries with audio features per segment
        """
        # Convert to numpy for analysis
        audio_np = audio.cpu().numpy()
        
        # Calculate segment length
        samples_per_segment = len(audio_np) // n_segments
        segment_duration = self.duration_seconds / n_segments
        
        # Analyze each segment
        segments = []
        
        for i in range(n_segments):
            start_idx = i * samples_per_segment
            end_idx = start_idx + samples_per_segment
            segment = audio_np[start_idx:end_idx]
            
            # Calculate RMS (intensity)
            rms = np.sqrt(np.mean(segment ** 2))
            
            # Simple frequency band analysis
            segment_abs = np.abs(segment)
            segment_len = len(segment)
            low_freq = np.mean(segment_abs[:segment_len//3])
            mid_freq = np.mean(segment_abs[segment_len//3:2*segment_len//3])
            high_freq = np.mean(segment_abs[2*segment_len//3:])
            
            # Determine dominant frequency band
            freq_values = [low_freq, mid_freq, high_freq]
            dominant_freq = ["low", "mid", "high"][np.argmax(freq_values)]
            
            # Create segment data
            segment_data = {
                "start_time": i * segment_duration,
                "end_time": (i + 1) * segment_duration,
                "intensity": float(rms / np.max(audio_np)),
                "frequency_profile": {
                    "low": float(low_freq / np.max(audio_np)),
                    "mid": float(mid_freq / np.max(audio_np)),
                    "high": float(high_freq / np.max(audio_np))
                },
                "dominant_frequency": dominant_freq
            }
            
            segments.append(segment_data)
        
        return segments
    
    def suggest_sync_points(
        self,
        audio: torch.Tensor,
        sample_rate: int,
        intensity_threshold: float = 0.7,
        min_spacing_seconds: float = 2.0
    ) -> List[AudioVisualSyncPoint]:
        """
        Analyze audio and suggest synchronization points.
        
        Args:
            audio: Audio tensor
            sample_rate: Sample rate of audio
            intensity_threshold: Threshold for detecting peaks
            min_spacing_seconds: Minimum spacing between sync points
            
        Returns:
            List of suggested AudioVisualSyncPoint objects
        """
        # Convert to numpy for analysis
        audio_np = audio.cpu().numpy()
        
        # Calculate RMS amplitude in windows
        window_size = int(sample_rate * 0.1)  # 100ms windows
        hop_size = window_size // 2
        
        rms_values = []
        time_values = []
        
        for i in range(0, len(audio_np) - window_size, hop_size):
            window = audio_np[i:i+window_size]
            rms = np.sqrt(np.mean(window ** 2))
            time_sec = i / sample_rate
            
            rms_values.append(rms)
            time_values.append(time_sec)
        
        # Normalize RMS values
        rms_values = np.array(rms_values)
        rms_values = rms_values / np.max(rms_values)
        
        # Find peaks
        peaks = []
        last_peak_time = -min_spacing_seconds
        
        for i, (time_sec, rms) in enumerate(zip(time_values, rms_values)):
            if rms > intensity_threshold and time_sec - last_peak_time >= min_spacing_seconds:
                # Simple frequency analysis for this peak
                window_start = int(time_sec * sample_rate)
                window_end = min(window_start + window_size, len(audio_np))
                window = audio_np[window_start:window_end]
                
                window_abs = np.abs(window)
                window_len = len(window)
                low_freq = np.mean(window_abs[:window_len//3])
                mid_freq = np.mean(window_abs[window_len//3:2*window_len//3])
                high_freq = np.mean(window_abs[2*window_len//3:])
                
                freq_values = [low_freq, mid_freq, high_freq]
                dominant_freq = ["low", "mid", "high"][np.argmax(freq_values)]
                
                # Add to peaks
                peaks.append((time_sec, rms, dominant_freq))
                last_peak_time = time_sec
        
        # Create sync points from peaks
        suggested_sync_points = []
        
        visual_event_types = ["color_shift", "particles", "camera_motion", "geometric_change"]
        
        for i, (time_sec, intensity, freq) in enumerate(peaks):
            # Cycle through event types
            event_type = visual_event_types[i % len(visual_event_types)]
            
            # Create parameters based on audio characteristics
            event_params = {}
            
            if event_type == "color_shift":
                event_params = {
                    "intensity": intensity,
                    "speed": intensity * 0.5,
                    "hue_shift": (i % 3) * 0.33  # Cycle through color regions
                }
            elif event_type == "particles":
                event_params = {
                    "emission_rate": intensity,
                    "velocity": intensity * 2.0,
                    "size": 0.5 + intensity * 0.5
                }
            elif event_type == "camera_motion":
                event_params = {
                    "movement_amount": intensity,
                    "shake_intensity": intensity * 0.3,
                    "duration": 1.0 + intensity
                }
            elif event_type == "geometric_change":
                event_params = {
                    "scale": 1.0 + intensity,
                    "rotation_speed": intensity * 2.0,
                    "complexity": 0.3 + intensity * 0.7
                }
            
            # Create sync point
            sync_point = AudioVisualSyncPoint(
                time_seconds=time_sec,
                duration_seconds=1.0 + intensity,
                audio_intensity=intensity,
                audio_frequency_focus=freq,
                visual_event_type=event_type,
                visual_event_parameters=event_params,
                description=f"Auto-generated sync point at {time_sec:.2f}s with {freq} frequency focus"
            )
            
            suggested_sync_points.append(sync_point)
        
        return suggested_sync_points
    
    def generate_and_analyze_experience(
        self,
        output_filename: str = None,
        generation_params: Dict = None,
        auto_suggest_sync_points: bool = True
    ) -> Dict:
        """
        Generate and analyze a complete audio-visual experience.
        
        Args:
            output_filename: Optional filename for saving
            generation_params: Optional generation parameters
            auto_suggest_sync_points: Whether to auto-suggest sync points
            
        Returns:
            Dictionary with experience data
        """
        # Generate audio
        audio, sample_rate = self.generate_audio(generation_params)
        
        # Analyze audio in segments
        segments = self.analyze_audio(audio, sample_rate)
        
        # Auto-suggest sync points if requested
        if auto_suggest_sync_points:
            suggested_points = self.suggest_sync_points(audio, sample_rate)
            
            # Merge with existing sync points
            if not self.sync_points:
                self.sync_points = suggested_points
            else:
                # Only add non-overlapping points
                existing_times = set(sp.time_seconds for sp in self.sync_points)
                for sp in suggested_points:
                    # Check if there's a nearby existing point (within 1 second)
                    has_nearby = any(abs(sp.time_seconds - t) < 1.0 for t in existing_times)
                    if not has_nearby:
                        self.sync_points.append(sp)
                
                # Sort by time
                self.sync_points.sort(key=lambda sp: sp.time_seconds)
        
        # Create experience data
        experience_data = {
            "name": self.experience_name,
            "duration": self.duration_seconds,
            "audio_prompt": self.audio_prompt,
            "visual_parameters": self.visual_base_parameters.to_dict(),
            "segments": segments,
            "sync_points": [sp.to_dict() for sp in self.sync_points]
        }
        
        # Save audio if filename provided
        if output_filename:
            if not output_filename.endswith('.wav'):
                audio_filename = f"{output_filename}.wav"
            else:
                audio_filename = output_filename
                output_filename = output_filename.replace('.wav', '')
            
            audio_path = os.path.join(self.output_dir, audio_filename)
            torchaudio.save(
                audio_path,
                audio.cpu().unsqueeze(0),
                sample_rate
            )
            
            # Save experience data
            data_path = os.path.join(self.output_dir, f"{output_filename}_experience.json")
            with open(data_path, 'w') as f:
                json.dump(experience_data, f, indent=2)
            
            # Generate visualization
            self.visualize_experience(
                audio, 
                sample_rate, 
                os.path.join(self.output_dir, f"{output_filename}_visualization.png")
            )
            
            print(f"Experience saved to {data_path} with audio at {audio_path}")
        
        return experience_data
    
    def visualize_experience(
        self,
        audio: torch.Tensor,
        sample_rate: int,
        output_path: str
    ):
        """
        Create a visualization of the audio and sync points.
        
        Args:
            audio: Audio tensor
            sample_rate: Sample rate of audio
            output_path: Path to save visualization
        """
        audio_np = audio.cpu().numpy()
        
        # Calculate time axis
        time = np.linspace(0, len(audio_np) / sample_rate, len(audio_np))
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot waveform
        plt.subplot(2, 1, 1)
        plt.plot(time, audio_np)
        plt.title(f"Audio Waveform: {self.experience_name}")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        
        # Mark sync points on waveform
        for sp in self.sync_points:
            plt.axvline(x=sp.time_seconds, color='r', linestyle='--', alpha=0.7)
            plt.text(sp.time_seconds, np.max(audio_np) * 0.9, sp.visual_event_type, 
                     rotation=90, verticalalignment='top')
        
        # Plot spectrogram
        plt.subplot(2, 1, 2)
        D = librosa.stft(audio_np)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        plt.colorbar(plt.pcolormesh(time, librosa.fft_frequencies(sr=sample_rate), S_db, shading='gouraud'))
        plt.title("Spectrogram")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.yscale('log')
        plt.ylim([20, sample_rate/2])
        
        # Mark sync points on spectrogram
        for sp in self.sync_points:
            plt.axvline(x=sp.time_seconds, color='r', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        print(f"Visualization saved to {output_path}")
    
    def export_for_unity(self, output_dir: str = None):
        """
        Export the experience in a format suitable for Unity.
        
        Args:
            output_dir: Output directory (defaults to a subdirectory)
            
        Returns:
            Path to exported directory
        """
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, "unity_export")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate if we don't have audio yet
        if not hasattr(self, '_last_audio'):
            print("Generating new audio for export...")
            audio, sample_rate = self.generate_audio()
            self._last_audio = (audio, sample_rate)
        else:
            audio, sample_rate = self._last_audio
        
        # Save audio
        audio_path = os.path.join(output_dir, "experience_audio.wav")
        torchaudio.save(
            audio_path,
            audio.cpu().unsqueeze(0),
            sample_rate
        )
        
        # Create Unity-friendly data format
        unity_data = {
            "experienceName": self.experience_name,
            "audioFile": "experience_audio.wav",
            "duration": self.duration_seconds,
            "sampleRate": sample_rate,
            "baseVisualParameters": self.visual_base_parameters.to_dict(),
            "syncEvents": []
        }
        
        for sp in self.sync_points:
            event_data = {
                "time": sp.time_seconds,
                "duration": sp.duration_seconds,
                "eventType": sp.visual_event_type,
                "parameters": sp.visual_event_parameters,
                "audioIntensity": sp.audio_intensity,
                "frequencyFocus": sp.audio_frequency_focus,
                "description": sp.description
            }
            unity_data["syncEvents"].append(event_data)
        
        # Save Unity data
        unity_data_path = os.path.join(output_dir, "experience_data.json")
        with open(unity_data_path, 'w') as f:
            json.dump(unity_data, f, indent=2)
        
        # Create a README file with instructions
        readme_path = os.path.join(output_dir, "README.md")
        with open(readme_path, 'w') as f:
            f.write(f"# Unity Integration: {self.experience_name}\n\n")
            f.write("This package contains the audio and synchronization data for Unity integration.\n\n")
            f.write("## Files\n\n")
            f.write("- `experience_audio.wav`: Generated audio file\n")
            f.write("- `experience_data.json`: Synchronization data and parameters\n\n")
            f.write("## Unity Integration\n\n")
            f.write("1. Import the `experience_audio.wav` file into your Unity project\n")
            f.write("2. Use the AudioVisualPlayer component to load the `experience_data.json` file\n")
            f.write("3. The player will automatically handle sync events based on timing\n\n")
            f.write("## Sync Events\n\n")
            f.write("| Time (s) | Event Type | Duration (s) | Frequency Focus |\n")
            f.write("|--:|:--|--:|:--|\n")
            
            for sp in self.sync_points:
                f.write(f"| {sp.time_seconds:.2f} | {sp.visual_event_type} | {sp.duration_seconds:.2f} | {sp.audio_frequency_focus} |\n")
        
        print(f"Unity export completed to {output_dir}")
        return output_dir
    
    def export_for_web(self, output_dir: str = None):
        """
        Export the experience in a format suitable for web integration.
        
        Args:
            output_dir: Output directory (defaults to a subdirectory)
            
        Returns:
            Path to exported directory
        """
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, "web_export")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate if we don't have audio yet
        if not hasattr(self, '_last_audio'):
            print("Generating new audio for export...")
            audio, sample_rate = self.generate_audio()
            self._last_audio = (audio, sample_rate)
        else:
            audio, sample_rate = self._last_audio
        
        # Save audio
        audio_path = os.path.join(output_dir, "experience_audio.wav")
        torchaudio.save(
            audio_path,
            audio.cpu().unsqueeze(0),
            sample_rate
        )
        
        # Create web-friendly JSON data
        web_data = {
            "name": self.experience_name,
            "audio": "experience_audio.wav",
            "duration": self.duration_seconds,
            "visualBase": self.visual_base_parameters.to_dict(),
            "syncEvents": []
        }
        
        for sp in self.sync_points:
            event_data = {
                "time": sp.time_seconds,
                "duration": sp.duration_seconds,
                "type": sp.visual_event_type,
                "params": sp.visual_event_parameters,
                "intensity": sp.audio_intensity,
                "frequency": sp.audio_frequency_focus
            }
            web_data["syncEvents"].append(event_data)
        
        # Save data file
        data_path = os.path.join(output_dir, "experience.json")
        with open(data_path, 'w') as f:
            json.dump(web_data, f, indent=2)
        
        # Create basic HTML and JavaScript files
        self._create_web_player_files(output_dir)
        
        print(f"Web export completed to {output_dir}")
        return output_dir
    
    def _create_web_player_files(self, output_dir: str):
        """Create basic HTML and JavaScript files for web playback."""
        # HTML file
        html_path = os.path.join(output_dir, "player.html")
        with open(html_path, 'w') as f:
            f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Audio-Visual Experience Player</title>
    <style>
        body {
            margin: 0;
            font-family: Arial, sans-serif;
            overflow: hidden;
            background-color: #000;
        }
        #canvas {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
        }
        #controls {
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 100;
            background: rgba(0,0,0,0.7);
            padding: 10px;
            border-radius: 5px;
            display: flex;
            align-items: center;
        }
        #play-btn {
            background: #fff;
            color: #000;
            border: none;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            font-size: 16px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 10px;
        }
        #time-display {
            color: #fff;
            margin-left: 10px;
        }
        #loading {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.8);
            display: flex;
            align-items: center;
            justify-content: center;
            color: #fff;
            font-size: 24px;
            z-index: 1000;
        }
    </style>
</head>
<body>
    <div id="loading">Loading experience...</div>
    <canvas id="canvas"></canvas>
    <div id="controls">
        <button id="play-btn">▶</button>
        <input type="range" id="progress" min="0" max="100" value="0" style="width: 300px;">
        <span id="time-display">0:00 / 0:00</span>
    </div>
    <script src="player.js"></script>
</body>
</html>""")
        
        # JavaScript file
        js_path = os.path.join(output_dir, "player.js")
        with open(js_path, 'w') as f:
            f.write("""// Audio-Visual Experience Player

// Experience data
let experienceData = null;
let audio = new Audio();
let isPlaying = false;
let startTime = 0;
let currentTime = 0;

// Visual elements
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

// Controls
const playBtn = document.getElementById('play-btn');
const progress = document.getElementById('progress');
const timeDisplay = document.getElementById('time-display');
const loading = document.getElementById('loading');

// Resize handler
function resizeCanvas() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    draw();
}

// Format time as mm:ss
function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

// Initialize the experience
async function init() {
    // Load experience data
    try {
        const response = await fetch('experience.json');
        experienceData = await response.json();
        
        // Set up audio
        audio.src = experienceData.audio;
        audio.load();
        
        // Wait for audio to be loaded
        await new Promise((resolve) => {
            audio.addEventListener('canplaythrough', resolve, { once: true });
        });
        
        // Update UI
        progress.max = experienceData.duration;
        timeDisplay.textContent = `0:00 / ${formatTime(experienceData.duration)}`;
        
        // Setup event listeners
        playBtn.addEventListener('click', togglePlay);
        progress.addEventListener('input', seek);
        audio.addEventListener('timeupdate', updateProgress);
        window.addEventListener('resize', resizeCanvas);
        
        // Initial resize
        resizeCanvas();
        
        // Hide loading screen
        loading.style.display = 'none';
    } catch (error) {
        console.error('Error loading experience:', error);
        loading.textContent = 'Error loading experience';
    }
}

// Toggle play/pause
function togglePlay() {
    if (isPlaying) {
        audio.pause();
        playBtn.textContent = '▶';
    } else {
        audio.play();
        playBtn.textContent = '⏸';
    }
    isPlaying = !isPlaying;
}

// Seek to position
function seek() {
    audio.currentTime = progress.value;
    currentTime = audio.currentTime;
}

// Update progress bar
function updateProgress() {
    currentTime = audio.currentTime;
    progress.value = currentTime;
    timeDisplay.textContent = `${formatTime(currentTime)} / ${formatTime(experienceData.duration)}`;
}

// Find active sync events at current time
function getActiveSyncEvents() {
    if (!experienceData) return [];
    
    return experienceData.syncEvents.filter(event => {
        return currentTime >= event.time && currentTime < event.time + event.duration;
    });
}

// Visual rendering
function draw() {
    if (!experienceData) return;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Get base parameters
    const visualBase = experienceData.visualBase;
    
    // Get active events
    const activeEvents = getActiveSyncEvents();
    
    // Draw background
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Apply base visualization
    drawBaseVisualization(visualBase);
    
    // Apply active events
    for (const event of activeEvents) {
        // Calculate progress through the event (0-1)
        const progress = (currentTime - event.time) / event.duration;
        
        // Apply different visual effects based on event type
        switch (event.type) {
            case 'color_shift':
                applyColorShift(event, progress);
                break;
            case 'particles':
                drawParticles(event, progress);
                break;
            case 'camera_motion':
                applyCameraMotion(event, progress);
                break;
            case 'geometric_change':
                drawGeometry(event, progress);
                break;
        }
    }
    
    // Request next frame if playing
    if (isPlaying) {
        requestAnimationFrame(draw);
    }
}

// Draw base visualization
function drawBaseVisualization(params) {
    // Simple circular visualization
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const radius = Math.min(canvas.width, canvas.height) * 0.3;
    
    // Base circle
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
    ctx.fillStyle = `hsl(${(Date.now() / 50) % 360}, 70%, 50%)`;
    ctx.globalAlpha = 0.2;
    ctx.fill();
    ctx.globalAlpha = 1.0;
}

// Apply color shift effect
function applyColorShift(event, progress) {
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const size = Math.min(canvas.width, canvas.height) * 0.5;
    
    // Create gradient based on event parameters
    const intensity = event.params.intensity || 0.5;
    const hueShift = event.params.hue_shift || 0;
    const speed = event.params.speed || 0.5;
    
    const hue = ((Date.now() / (1000 / speed)) % 360) + hueShift * 120;
    const gradient = ctx.createRadialGradient(
        centerX, centerY, 0,
        centerX, centerY, size
    );
    
    gradient.addColorStop(0, `hsla(${hue}, 100%, 60%, ${intensity * 0.8})`);
    gradient.addColorStop(0.7, `hsla(${hue + 40}, 100%, 40%, ${intensity * 0.4})`);
    gradient.addColorStop(1, `hsla(${hue + 80}, 100%, 20%, 0)`);
    
    // Draw colored overlay
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

// Draw particle effect
function drawParticles(event, progress) {
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    
    // Extract parameters
    const emissionRate = event.params.emission_rate || 0.5;
    const velocity = event.params.velocity || 1.0;
    const size = event.params.size || 0.5;
    
    // Calculate number of particles based on emission rate
    const particleCount = Math.floor(50 * emissionRate);
    
    // Draw particles
    for (let i = 0; i < particleCount; i++) {
        // Calculate particle position
        const angle = (i / particleCount) * Math.PI * 2;
        const distance = (progress * velocity * 300) + (i % 3) * 50;
        const x = centerX + Math.cos(angle + Date.now() / 1000) * distance;
        const y = centerY + Math.sin(angle + Date.now() / 1000) * distance;
        
        // Draw particle
        ctx.beginPath();
        ctx.arc(x, y, 3 * size, 0, Math.PI * 2);
        ctx.fillStyle = `hsla(${(i * 10) % 360}, 100%, 70%, ${1 - progress})`;
        ctx.fill();
    }
}

// Apply camera motion effect
function applyCameraMotion(event, progress) {
    const amount = event.params.movement_amount || 0.5;
    const shakeIntensity = event.params.shake_intensity || 0.2;
    
    // Apply camera shake
    const shakeX = (Math.random() - 0.5) * shakeIntensity * 30;
    const shakeY = (Math.random() - 0.5) * shakeIntensity * 30;
    
    // Create a zoom effect
    const zoom = 1 + Math.sin(progress * Math.PI) * amount * 0.2;
    
    // Apply transformations
    ctx.save();
    ctx.translate(canvas.width / 2, canvas.height / 2);
    ctx.translate(shakeX, shakeY);
    ctx.scale(zoom, zoom);
    ctx.translate(-canvas.width / 2, -canvas.height / 2);
    
    // Draw a subtle overlay to indicate camera effect
    ctx.fillStyle = `rgba(255, 255, 255, ${progress * 0.1})`;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    ctx.restore();
}

// Draw geometric shapes
function drawGeometry(event, progress) {
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    
    // Extract parameters
    const scale = event.params.scale || 1.0;
    const rotationSpeed = event.params.rotation_speed || 1.0;
    const complexity = event.params.complexity || 0.5;
    
    // Calculate rotation based on progress and speed
    const rotation = progress * Math.PI * 2 * rotationSpeed;
    
    // Calculate size based on scale
    const size = Math.min(canvas.width, canvas.height) * 0.2 * scale;
    
    // Calculate number of shapes based on complexity
    const shapeCount = Math.floor(3 + complexity * 10);
    
    // Draw shapes
    ctx.save();
    ctx.translate(centerX, centerY);
    ctx.rotate(rotation);
    
    for (let i = 0; i < shapeCount; i++) {
        const angle = (i / shapeCount) * Math.PI * 2;
        const x = Math.cos(angle) * size;
        const y = Math.sin(angle) * size;
        const shapeSize = size * 0.2;
        
        ctx.save();
        ctx.translate(x, y);
        ctx.rotate(rotation * (i % 3));
        
        // Alternate between different shapes
        switch (i % 3) {
            case 0: // Square
                ctx.fillStyle = `hsla(${(i * 30) % 360}, 80%, 60%, ${0.7 - progress * 0.3})`;
                ctx.fillRect(-shapeSize / 2, -shapeSize / 2, shapeSize, shapeSize);
                break;
            case 1: // Circle
                ctx.beginPath();
                ctx.arc(0, 0, shapeSize / 2, 0, Math.PI * 2);
                ctx.fillStyle = `hsla(${(i * 30 + 120) % 360}, 80%, 60%, ${0.7 - progress * 0.3})`;
                ctx.fill();
                break;
            case 2: // Triangle
                ctx.beginPath();
                ctx.moveTo(0, -shapeSize / 2);
                ctx.lineTo(shapeSize / 2, shapeSize / 2);
                ctx.lineTo(-shapeSize / 2, shapeSize / 2);
                ctx.closePath();
                ctx.fillStyle = `hsla(${(i * 30 + 240) % 360}, 80%, 60%, ${0.7 - progress * 0.3})`;
                ctx.fill();
                break;
        }
        
        ctx.restore();
    }
    
    ctx.restore();
}

// Initialize the experience
init();
""")
```

## Unity Implementation: Audio-Visual Player

To complement our Python generation systems, let's implement a Unity C# script that handles synchronized audio-visual playback:

```csharp
using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.VFX;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

namespace AudioVisual
{
    [Serializable]
    public class SyncEvent
    {
        public float time;
        public float duration;
        public string eventType;
        public float audioIntensity;
        public string frequencyFocus;
        public Dictionary<string, float> parameters = new Dictionary<string, float>();
    }
    
    [Serializable]
    public class ExperienceData
    {
        public string experienceName;
        public string audioFile;
        public float duration;
        public int sampleRate;
        public Dictionary<string, float> baseVisualParameters = new Dictionary<string, float>();
        public List<SyncEvent> syncEvents = new List<SyncEvent>();
    }
    
    public class AudioVisualPlayer : MonoBehaviour
    {
        [Header("Experience")]
        [SerializeField] private TextAsset experienceDataJson;
        [SerializeField] private AudioClip experienceAudio;
        
        [Header("Visual Components")]
        [SerializeField] private Volume postProcessingVolume;
        [SerializeField] private List<VisualEffectAsset> visualEffects;
        [SerializeField] private Transform visualContainer;
        [SerializeField] private Light mainLight;
        [SerializeField] private Camera mainCamera;
        
        [Header("Settings")]
        [SerializeField] private bool autoPlay = false;
        [SerializeField] private bool loopPlayback = false;
        [SerializeField] private bool showDebugInfo = false;
        
        // Runtime data
        private ExperienceData experienceData;
        private AudioSource audioSource;
        private Dictionary<string, VisualEffect> activeEffects = new Dictionary<string, VisualEffect>();
        private List<SyncEvent> activeEvents = new List<SyncEvent>();
        
        // Post-processing effects
        private ColorAdjustments colorAdjustments;
        private ChromaticAberration chromaticAberration;
        private Bloom bloom;
        private Vignette vignette;
        
        // State variables
        private bool isPlaying = false;
        private bool isInitialized = false;
        
        private void Awake()
        {
            Initialize();
        }
        
        private void Start()
        {
            if (autoPlay)
            {
                PlayExperience();
            }
        }
        
        private void Update()
        {
            if (!isInitialized || !isPlaying)
                return;
            
            // Update active events
            UpdateActiveEvents();
            
            // Update visuals based on current events
            UpdateVisuals();
        }
        
        public void Initialize()
        {
            if (isInitialized)
                return;
            
            // Load experience data
            if (experienceDataJson != null)
            {
                experienceData = JsonUtility.FromJson<ExperienceData>(experienceDataJson.text);
            }
            else
            {
                Debug.LogError("No experience data provided");
                return;
            }
            
            // Verify audio clip
            if (experienceAudio == null)
            {
                Debug.LogError("No audio clip provided");
                return;
            }
            
            // Setup audio source
            audioSource = gameObject.AddComponent<AudioSource>();
            audioSource.clip = experienceAudio;
            audioSource.playOnAwake = false;
            audioSource.loop = loopPlayback;
            
            // Initialize post-processing effects
            if (postProcessingVolume != null)
            {
                postProcessingVolume.profile.TryGet(out colorAdjustments);
                postProcessingVolume.profile.TryGet(out chromaticAberration);
                postProcessingVolume.profile.TryGet(out bloom);
                postProcessingVolume.profile.TryGet(out vignette);
            }
            
            // Create visual effect instances
            InitializeVisualEffects();
            
            isInitialized = true;
            Debug.Log($"Audio-Visual Experience initialized: {experienceData.experienceName}");
        }
        
        private void InitializeVisualEffects()
        {
            // Create a container if not provided
            if (visualContainer == null)
            {
                GameObject container = new GameObject("VisualEffects");
                container.transform.SetParent(transform);
                visualContainer = container.transform;
            }
            
            // Create instances of visual effects
            if (visualEffects != null && visualEffects.Count > 0)
            {
                foreach (var effectAsset in visualEffects)
                {
                    string effectName = effectAsset.name;
                    
                    GameObject effectObj = new GameObject(effectName);
                    effectObj.transform.SetParent(visualContainer);
                    effectObj.transform.localPosition = Vector3.zero;
                    
                    VisualEffect effect = effectObj.AddComponent<VisualEffect>();
                    effect.visualEffectAsset = effectAsset;
                    effect.enabled = false;
                    
                    activeEffects[effectName] = effect;
                }
            }
        }
        
        public void PlayExperience()
        {
            if (!isInitialized)
            {
                Initialize();
            }
            
            audioSource.Play();
            isPlaying = true;
        }
        
        public void PauseExperience()
        {
            if (isPlaying)
            {
                audioSource.Pause();
                isPlaying = false;
            }
        }
        
        public void StopExperience()
        {
            audioSource.Stop();
            isPlaying = false;
            activeEvents.Clear();
            
            // Reset visuals
            ResetVisuals();
        }
        
        public void SeekToTime(float timeInSeconds)
        {
            audioSource.time = Mathf.Clamp(timeInSeconds, 0, experienceData.duration);
            
            // Update active events
            UpdateActiveEvents();
        }
        
        private void UpdateActiveEvents()
        {
            float currentTime = audioSource.time;
            
            // Clear expired events
            activeEvents.RemoveAll(e => currentTime >= e.time + e.duration);
            
            // Add new active events
            foreach (var syncEvent in experienceData.syncEvents)
            {
                if (currentTime >= syncEvent.time && 
                    currentTime < syncEvent.time + syncEvent.duration && 
                    !activeEvents.Contains(syncEvent))
                {
                    activeEvents.Add(syncEvent);
                    
                    // Trigger event start
                    TriggerEventStart(syncEvent);
                    
                    if (showDebugInfo)
                    {
                        Debug.Log($"Event triggered: {syncEvent.eventType} at {currentTime:F2}s");
                    }
                }
            }
        }
        
        private void TriggerEventStart(SyncEvent syncEvent)
        {
            // Activate specific visual effects based on event type
            switch (syncEvent.eventType)
            {
                case "color_shift":
                    if (activeEffects.TryGetValue("ColorShift", out VisualEffect colorEffect))
                    {
                        colorEffect.enabled = true;
                        ApplyEventParameters(colorEffect, syncEvent);
                    }
                    break;
                
                case "particles":
                    if (activeEffects.TryGetValue("Particles", out VisualEffect particleEffect))
                    {
                        particleEffect.enabled = true;
                        ApplyEventParameters(particleEffect, syncEvent);
                    }
                    break;
                
                case "camera_motion":
                    // Camera motion is handled in UpdateVisuals
                    break;
                
                case "geometric_change":
                    if (activeEffects.TryGetValue("Geometry", out VisualEffect geometryEffect))
                    {
                        geometryEffect.enabled = true;
                        ApplyEventParameters(geometryEffect, syncEvent);
                    }
                    break;
            }
        }
        
        private void ApplyEventParameters(VisualEffect effect, SyncEvent syncEvent)
        {
            // Apply parameters to the visual effect
            foreach (var param in syncEvent.parameters)
            {
                if (effect.HasFloat(param.Key))
                {
                    effect.SetFloat(param.Key, param.Value);
                }
            }
            
            // Always set intensity and event progress (0-1)
            if (effect.HasFloat("intensity"))
            {
                effect.SetFloat("intensity", syncEvent.audioIntensity);
            }
            
            if (effect.HasFloat("eventProgress"))
            {
                float progress = (audioSource.time - syncEvent.time) / syncEvent.duration;
                effect.SetFloat("eventProgress", progress);
            }
        }
        
        private void UpdateVisuals()
        {
            // Reset visuals if no active events
            if (activeEvents.Count == 0)
            {
                ResetVisuals();
                return;
            }
            
            // Calculate current values from all active events
            float colorIntensity = 0f;
            float chromaticIntensity = 0f;
            float bloomIntensity = 0f;
            float vignetteIntensity = 0f;
            
            Vector3 cameraShake = Vector3.zero;
            float cameraFOVDelta = 0f;
            Color lightColor = mainLight ? mainLight.color : Color.white;
            float lightIntensity = mainLight ? mainLight.intensity : 1f;
            
            // Process all active events
            foreach (var syncEvent in activeEvents)
            {
                // Calculate event progress (0-1)
                float currentTime = audioSource.time;
                float progress = (currentTime - syncEvent.time) / syncEvent.duration;
                progress = Mathf.Clamp01(progress);
                
                // Apply different effects based on event type
                switch (syncEvent.eventType)
                {
                    case "color_shift":
                        // Update post-processing
                        colorIntensity += syncEvent.audioIntensity * 0.5f;
                        chromaticIntensity += syncEvent.audioIntensity * 0.3f;
                        
                        // Update light color based on parameters
                        if (mainLight && syncEvent.parameters.TryGetValue("hue_shift", out float hueShift))
                        {
                            float h, s, v;
                            Color.RGBToHSV(lightColor, out h, out s, out v);
                            h = (h + hueShift + progress * 0.2f) % 1f;
                            lightColor = Color.HSVToRGB(h, s, v);
                        }
                        
                        // Update visual effect
                        if (activeEffects.TryGetValue("ColorShift", out VisualEffect colorEffect))
                        {
                            ApplyEventParameters(colorEffect, syncEvent);
                        }
                        break;
                    
                    case "particles":
                        // Update bloom
                        bloomIntensity += syncEvent.audioIntensity * 0.7f;
                        
                        // Update light intensity
                        lightIntensity += syncEvent.audioIntensity * 0.5f;
                        
                        // Update visual effect
                        if (activeEffects.TryGetValue("Particles", out VisualEffect particleEffect))
                        {
                            ApplyEventParameters(particleEffect, syncEvent);
                        }
                        break;
                    
                    case "camera_motion":
                        // Calculate camera shake
                        if (syncEvent.parameters.TryGetValue("shake_intensity", out float shakeIntensity))
                        {
                            float shake = shakeIntensity * syncEvent.audioIntensity;
                            cameraShake += new Vector3(
                                (Mathf.PerlinNoise(Time.time * 10, 0) - 0.5f) * shake,
                                (Mathf.PerlinNoise(0, Time.time * 10) - 0.5f) * shake,
                                0
                            );
                        }
                        
                        // Calculate FOV change
                        if (syncEvent.parameters.TryGetValue("movement_amount", out float moveAmount))
                        {
                            cameraFOVDelta += Mathf.Sin(progress * Mathf.PI) * moveAmount * 10f;
                        }
                        
                        // Increase vignette during camera motion
                        vignetteIntensity += syncEvent.audioIntensity * 0.3f;
                        break;
                    
                    case "geometric_change":
                        // Update chromatic aberration
                        chromaticIntensity += syncEvent.audioIntensity * 0.2f;
                        
                        // Update visual effect
                        if (activeEffects.TryGetValue("Geometry", out VisualEffect geometryEffect))
                        {
                            ApplyEventParameters(geometryEffect, syncEvent);
                        }
                        break;
                }
                
                // Update visual effects with current progress
                if (activeEffects.TryGetValue(syncEvent.eventType, out VisualEffect effect))
                {
                    if (effect.HasFloat("eventProgress"))
                    {
                        effect.SetFloat("eventProgress", progress);
                    }
                }
            }
            
            // Apply post-processing changes
            if (colorAdjustments != null)
            {
                colorAdjustments.saturation.value = Mathf.Lerp(0, 30, colorIntensity);
            }
            
            if (chromaticAberration != null)
            {
                chromaticAberration.intensity.value = Mathf.Clamp01(chromaticIntensity);
            }
            
            if (bloom != null)
            {
                bloom.intensity.value = Mathf.Lerp(1, 3, bloomIntensity);
            }
            
            if (vignette != null)
            {
                vignette.intensity.value = Mathf.Clamp01(vignetteIntensity);
            }
            
            // Apply camera effects
            if (mainCamera != null)
            {
                // Apply camera shake
                mainCamera.transform.localPosition = cameraShake;
                
                // Apply FOV change
                mainCamera.fieldOfView = Mathf.Clamp(60 + cameraFOVDelta, 40, 80);
            }
            
            // Apply light changes
            if (mainLight != null)
            {
                mainLight.color = Color.Lerp(Color.white, lightColor, 0.7f);
                mainLight.intensity = Mathf.Clamp(lightIntensity, 0.5f, 2f);
            }
        }
        
        private void ResetVisuals()
        {
            // Reset post-processing
            if (colorAdjustments != null)
            {
                colorAdjustments.saturation.value = 0;
            }
            
            if (chromaticAberration != null)
            {
                chromaticAberration.intensity.value = 0;
            }
            
            if (bloom != null)
            {
                bloom.intensity.value = 1;
            }
            
            if (vignette != null)
            {
                vignette.intensity.value = 0.25f;
            }
            
            // Reset camera
            if (mainCamera != null)
            {
                mainCamera.transform.localPosition = Vector3.zero;
                mainCamera.fieldOfView = 60;
            }
            
            // Reset light
            if (mainLight != null)
            {
                mainLight.color = Color.white;
                mainLight.intensity = 1f;
            }
            
            // Disable all visual effects
            foreach (var effect in activeEffects.Values)
            {
                effect.enabled = false;
            }
        }
        
        private void OnGUI()
        {
            if (showDebugInfo && isPlaying)
            {
                GUI.Box(new Rect(10, 10, 300, 60), "Audio-Visual Experience");
                GUI.Label(new Rect(20, 30, 290, 20), $"Name: {experienceData.experienceName}");
                GUI.Label(new Rect(20, 50, 290, 20), $"Time: {audioSource.time:F2} / {experienceData.duration:F2}");
                
                int yPos = 80;
                GUI.Label(new Rect(20, yPos, 290, 20), $"Active Events: {activeEvents.Count}");
                yPos += 20;
                
                foreach (var evt in activeEvents)
                {
                    float progress = (audioSource.time - evt.time) / evt.duration;
                    GUI.Label(new Rect(30, yPos, 290, 20), $"- {evt.eventType}: {progress:P0}");
                    yPos += 20;
                }
            }
        }
    }
}
```

## Web Integration: React Component for Audio-Visual Playback

For web applications, here's a React component that can integrate with AudioCraft-generated content:

```jsx
import React, { useRef, useState, useEffect } from 'react';
import { Stage, Layer, Circle, Rect } from 'react-konva';
import './AudioVisualPlayer.css';

/**
 * React component for audio-visual playback of AudioCraft generated experiences
 */
const AudioVisualPlayer = ({ experienceUrl }) => {
  // State
  const [experience, setExperience] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const [activeEvents, setActiveEvents] = useState([]);
  const [baseColor, setBaseColor] = useState({ h: 180, s: 70, l: 50 });
  
  // Refs
  const audioRef = useRef(null);
  const canvasRef = useRef(null);
  const animationFrameRef = useRef(null);
  const stageRef = useRef(null);
  const particlesRef = useRef([]);
  
  // Load experience data
  useEffect(() => {
    const loadExperience = async () => {
      try {
        setIsLoading(true);
        const response = await fetch(experienceUrl);
        const data = await response.json();
        setExperience(data);
        
        // Setup audio
        if (audioRef.current) {
          audioRef.current.src = new URL(data.audio, experienceUrl).href;
          audioRef.current.load();
        }
        
        setIsLoading(false);
      } catch (error) {
        console.error('Error loading experience:', error);
        setIsLoading(false);
      }
    };
    
    loadExperience();
    
    // Cleanup animation frame on unmount
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [experienceUrl]);
  
  // Handle audio time updates
  useEffect(() => {
    const handleTimeUpdate = () => {
      if (audioRef.current) {
        setCurrentTime(audioRef.current.currentTime);
      }
    };
    
    // Handle audio loading
    const handleCanPlay = () => {
      if (audioRef.current) {
        setDuration(audioRef.current.duration);
      }
    };
    
    const audioElement = audioRef.current;
    if (audioElement) {
      audioElement.addEventListener('timeupdate', handleTimeUpdate);
      audioElement.addEventListener('canplay', handleCanPlay);
      
      return () => {
        audioElement.removeEventListener('timeupdate', handleTimeUpdate);
        audioElement.removeEventListener('canplay', handleCanPlay);
      };
    }
  }, []);
  
  // Update active events based on current time
  useEffect(() => {
    if (!experience) return;
    
    // Find active events
    const active = experience.syncEvents.filter(
      event => currentTime >= event.time && currentTime < event.time + event.duration
    );
    
    setActiveEvents(active);
  }, [currentTime, experience]);
  
  // Animation loop
  useEffect(() => {
    if (!isPlaying || !experience) return;
    
    // Animation function
    const animate = () => {
      // Slowly rotate base color
      setBaseColor(prev => ({
        h: (prev.h + 0.2) % 360,
        s: prev.s,
        l: prev.l
      }));
      
      // Continue animation loop
      animationFrameRef.current = requestAnimationFrame(animate);
    };
    
    // Start animation loop
    animate();
    
    // Cleanup
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [isPlaying, experience]);
  
  // Play/pause toggle
  const togglePlay = () => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause();
      } else {
        audioRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };
  
  // Seek function
  const handleSeek = (e) => {
    if (audioRef.current) {
      const newTime = parseFloat(e.target.value);
      audioRef.current.currentTime = newTime;
      setCurrentTime(newTime);
    }
  };
  
  // Format time as mm:ss
  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };
  
  // Generate particles for particle events
  useEffect(() => {
    if (!experience) return;
    
    // Find particle events
    const particleEvents = activeEvents.filter(event => event.type === 'particles');
    
    if (particleEvents.length > 0) {
      // Create particles based on the most intense particle event
      const event = particleEvents.reduce((prev, current) => 
        (current.intensity > prev.intensity) ? current : prev, particleEvents[0]);
      
      // Calculate event progress
      const progress = (currentTime - event.time) / event.duration;
      
      // Get parameters
      const emissionRate = event.params?.emission_rate || 0.5;
      const velocity = event.params?.velocity || 1.0;
      const particleSize = event.params?.size || 0.5;
      
      // Generate particles
      const particleCount = Math.floor(50 * emissionRate);
      const newParticles = [];
      
      for (let i = 0; i < particleCount; i++) {
        // Calculate angle and distance based on index
        const angle = (i / particleCount) * Math.PI * 2;
        const distance = progress * velocity * 200 + (i % 3) * 30;
        
        // Calculate position based on stage center
        const stageWidth = stageRef.current?.width() || 800;
        const stageHeight = stageRef.current?.height() || 600;
        
        const x = stageWidth / 2 + Math.cos(angle + Date.now() / 1000) * distance;
        const y = stageHeight / 2 + Math.sin(angle + Date.now() / 1000) * distance;
        
        // Add particle
        newParticles.push({
          id: `particle-${i}`,
          x,
          y, 
          radius: 3 * particleSize,
          hue: (i * 10) % 360,
          opacity: 1 - progress
        });
      }
      
      particlesRef.current = newParticles;
    } else {
      // Clear particles if no particle events are active
      particlesRef.current = [];
    }
  }, [activeEvents, currentTime, experience]);
  
  // Render particles
  const renderParticles = () => {
    return particlesRef.current.map(particle => (
      <Circle
        key={particle.id}
        x={particle.x}
        y={particle.y}
        radius={particle.radius}
        fill={`hsla(${particle.hue}, 100%, 70%, ${particle.opacity})`}
      />
    ));
  };
  
  // Calculate visual effects from active events
  const calculateVisualEffects = () => {
    if (!experience || activeEvents.length === 0) {
      return {
        colorShift: null,
        cameraMotion: null,
        geometryTransform: null
      };
    }
    
    // Initialize effect objects
    let colorShift = null;
    let cameraMotion = null;
    let geometryTransform = null;
    
    // Process active events
    for (const event of activeEvents) {
      // Calculate event progress
      const progress = (currentTime - event.time) / event.duration;
      
      // Handle different event types
      switch (event.type) {
        case 'color_shift':
          // Color shift effect
          const intensity = event.params?.intensity || 0.5;
          const hueShift = event.params?.hue_shift || 0;
          const speed = event.params?.speed || 0.5;
          
          colorShift = {
            hue: ((Date.now() / (1000 / speed)) % 360) + hueShift * 120,
            intensity: intensity * 0.8,
            progress
          };
          break;
          
        case 'camera_motion':
          // Camera motion effect
          const amount = event.params?.movement_amount || 0.5;
          const shakeIntensity = event.params?.shake_intensity || 0.2;
          
          cameraMotion = {
            shakeX: (Math.random() - 0.5) * shakeIntensity * 30,
            shakeY: (Math.random() - 0.5) * shakeIntensity * 30,
            zoom: 1 + Math.sin(progress * Math.PI) * amount * 0.2,
            progress
          };
          break;
          
        case 'geometric_change':
          // Geometric transformation effect
          const scale = event.params?.scale || 1.0;
          const rotationSpeed = event.params?.rotation_speed || 1.0;
          const complexity = event.params?.complexity || 0.5;
          
          geometryTransform = {
            rotation: progress * Math.PI * 2 * rotationSpeed,
            scale: scale,
            complexity: Math.floor(3 + complexity * 10),
            progress
          };
          break;
          
        default:
          break;
      }
    }
    
    return {
      colorShift,
      cameraMotion,
      geometryTransform
    };
  };
  
  // Get effects
  const effects = calculateVisualEffects();
  
  // Render geometric shapes for geometry events
  const renderGeometry = () => {
    if (!effects.geometryTransform) return null;
    
    const geo = effects.geometryTransform;
    const shapes = [];
    
    // Get stage dimensions
    const stageWidth = stageRef.current?.width() || 800;
    const stageHeight = stageRef.current?.height() || 600;
    const centerX = stageWidth / 2;
    const centerY = stageHeight / 2;
    
    // Calculate size
    const size = Math.min(stageWidth, stageHeight) * 0.2 * geo.scale;
    
    for (let i = 0; i < geo.complexity; i++) {
      // Calculate position
      const angle = (i / geo.complexity) * Math.PI * 2;
      const x = Math.cos(angle) * size;
      const y = Math.sin(angle) * size;
      const shapeSize = size * 0.2;
      
      // Create different shapes based on index
      switch (i % 3) {
        case 0: // Square
          shapes.push(
            <Rect
              key={`geo-${i}`}
              x={centerX + x - shapeSize / 2}
              y={centerY + y - shapeSize / 2}
              width={shapeSize}
              height={shapeSize}
              fill={`hsla(${(i * 30) % 360}, 80%, 60%, ${0.7 - geo.progress * 0.3})`}
              rotation={geo.rotation * (i % 3)}
              offsetX={-shapeSize / 2}
              offsetY={-shapeSize / 2}
            />
          );
          break;
          
        case 1: // Circle
          shapes.push(
            <Circle
              key={`geo-${i}`}
              x={centerX + x}
              y={centerY + y}
              radius={shapeSize / 2}
              fill={`hsla(${(i * 30 + 120) % 360}, 80%, 60%, ${0.7 - geo.progress * 0.3})`}
            />
          );
          break;
          
        case 2: // Triangle (simplified as a circle for this example)
          shapes.push(
            <Circle
              key={`geo-${i}`}
              x={centerX + x}
              y={centerY + y}
              radius={shapeSize / 2}
              fill={`hsla(${(i * 30 + 240) % 360}, 80%, 60%, ${0.7 - geo.progress * 0.3})`}
            />
          );
          break;
          
        default:
          break;
      }
    }
    
    return shapes;
  };
  
  return (
    <div className="audio-visual-player">
      {isLoading ? (
        <div className="loading">Loading experience...</div>
      ) : (
        <>
          <div className="canvas-container">
            <Stage 
              width={window.innerWidth} 
              height={window.innerHeight - 80}
              ref={stageRef}
              style={{
                transform: effects.cameraMotion ? 
                  `translate(${effects.cameraMotion.shakeX}px, ${effects.cameraMotion.shakeY}px) scale(${effects.cameraMotion.zoom})` : 
                  'none'
              }}
            >
              <Layer>
                {/* Background */}
                <Rect
                  x={0}
                  y={0}
                  width={window.innerWidth}
                  height={window.innerHeight - 80}
                  fill={effects.colorShift ? 
                    `hsla(${effects.colorShift.hue}, 100%, 50%, ${effects.colorShift.intensity})` : 
                    `hsl(${baseColor.h}, ${baseColor.s}%, ${baseColor.l}%)`
                  }
                />
                
                {/* Base circle */}
                <Circle
                  x={window.innerWidth / 2}
                  y={(window.innerHeight - 80) / 2}
                  radius={Math.min(window.innerWidth, window.innerHeight - 80) * 0.3}
                  fill={`hsla(${(baseColor.h + 180) % 360}, ${baseColor.s}%, ${baseColor.l}%, 0.3)`}
                />
                
                {/* Geometry effects */}
                {renderGeometry()}
                
                {/* Particle effects */}
                {renderParticles()}
              </Layer>
            </Stage>
          </div>
          
          <div className="player-controls">
            <button className="play-button" onClick={togglePlay}>
              {isPlaying ? '⏸' : '▶'}
            </button>
            
            <input
              type="range"
              min="0"
              max={duration}
              value={currentTime}
              onChange={handleSeek}
              className="progress-slider"
            />
            
            <div className="time-display">
              {formatTime(currentTime)} / {formatTime(duration)}
            </div>
          </div>
          
          <audio ref={audioRef} />
        </>
      )}
    </div>
  );
};

export default AudioVisualPlayer;
```

## Multimedia Integration Best Practices

Based on our implementations, here are best practices for integrating AudioCraft with other multimedia elements:

1. **Synchronization Architecture**
   - Design audio-visual experiences with clear synchronization points
   - Use parameter analysis to identify key moments in audio
   - Create metadata that connects audio and visual events
   - Ensure precise timing with frame-accurate playback systems

2. **Cross-Platform Compatibility**
   - Define platform-agnostic experience formats
   - Create specialized exporters for different platforms
   - Use common audio formats compatible across platforms
   - Adapt to different device capabilities and performance levels

3. **Event-Based Integration**
   - Design a clear event system for synchronization
   - Map audio characteristics to visual parameters
   - Use progressive intensity scales for visual effects
   - Support both predefined and dynamically generated events

4. **Audio Analysis and Adaptation**
   - Analyze audio to extract features for visualization
   - Create mappings between frequency bands and visual elements
   - Use amplitude for intensity parameters
   - Support dynamic adaptation to different audio characteristics

5. **Performance Optimization**
   - Create scalable visual systems that adapt to different hardware
   - Optimize memory usage for audio assets
   - Implement level-of-detail systems for visual effects
   - Use pooling for frequently created objects

## Hands-On Challenge: Create a Music Visualization Experience

**Challenge:** Build a complete music visualization system that responds to AudioCraft-generated content.

1. Generate a 1-minute piece of music with distinct sections
2. Create a system that analyzes the audio to extract dynamic features
3. Build visual systems that respond to frequency bands and amplitude
4. Implement transitions between different visual modes
5. Add interactive elements that let users modify the experience

**Steps to complete:**

1. Use MusicGen to create a dynamic musical piece
2. Implement audio analysis to extract frequency and amplitude data
3. Create visualization systems for different musical characteristics
4. Develop transition effects between visualization modes
5. Add user controls for modifying the experience in real-time

## Next Steps

In the next chapter, we'll explore real-time audio generation and processing with AudioCraft. We'll build systems that can generate and manipulate audio on-the-fly, responding to user input and environmental conditions.

Copyright © 2025 Scott Friedman. Licensed under [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).
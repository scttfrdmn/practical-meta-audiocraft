# AudioCraft Glossary

This glossary provides definitions for key terms and concepts used throughout the AudioCraft tutorials. Use this reference to understand technical terminology related to audio generation and machine learning.

## Audio Generation Terms

### AudioCraft
A framework developed by Meta AI Research for audio, music, and speech generation. It includes several models like MusicGen, AudioGen, and EnCodec.

### MusicGen
A text-to-music generation model within AudioCraft that can create musical compositions based on text descriptions.

### AudioGen
A text-to-audio generation model within AudioCraft specifically designed for environmental sounds and sound effects.

### EnCodec
A neural audio codec that compresses and tokenizes audio for processing by language models and then reconstructs the audio from generated tokens.

### Text-to-Audio
The process of generating audio content from text descriptions or prompts.

### Text Conditioning
Using text descriptions to guide or control the attributes of generated audio.

### Melody Conditioning
Using a reference melody or musical pattern to guide the generation of new music while maintaining melodic themes.

### Continuation Generation
Generating additional audio that naturally continues from an existing audio segment.

### Token
A discrete unit of information used in sequence modeling. In AudioCraft, audio is converted into tokens which the language model processes.

### Sample Rate
The number of audio samples per second, typically measured in Hertz (Hz). AudioCraft models generate audio at 32 kHz by default.

## Machine Learning Terms

### Foundation Model
A large AI model trained on extensive data that serves as a base for various applications. MusicGen and AudioGen are foundation models for audio generation.

### Language Model (LM)
In AudioCraft, a neural network architecture that generates sequences of audio tokens based on learned patterns.

### Inference
The process of running a trained model to generate outputs from inputs (e.g., creating audio from text prompts).

### Temperature
A parameter that controls randomness in generation. Higher values (e.g., 1.5) produce more creative but potentially less coherent outputs, while lower values (e.g., 0.5) yield more predictable results.

### Top-k Sampling
A text generation technique that restricts token selection to the k most likely options at each step, controlling diversity and coherence.

### Top-p (Nucleus) Sampling
A dynamic sampling method that selects from the smallest possible set of tokens whose cumulative probability exceeds probability p.

### Classifier-Free Guidance (CFG)
A technique that balances between unconditional generation and text-conditioned generation. Higher CFG values make the output more closely follow the text prompt.

### Batch Processing
Generating multiple outputs simultaneously, which is more efficient than sequential generation.

### CUDA
NVIDIA's parallel computing platform for GPU acceleration, often used to speed up neural network operations.

### MPS (Metal Performance Shaders)
Apple's framework for GPU-accelerated computing on Apple Silicon hardware.

## Deployment Terms

### API (Application Programming Interface)
A set of rules that allows different software applications to communicate with each other. In AudioCraft, APIs enable integration of audio generation capabilities into other applications.

### REST API
A web API that follows Representational State Transfer architectural principles, commonly used for integrating AudioCraft into web services.

### Gradio
A Python library for creating customizable web interfaces for machine learning models, useful for building demos of AudioCraft applications.

### Docker
A platform for developing, shipping, and running applications in isolated environments called containers, which simplifies deployment of AudioCraft applications.

### Containerization
The process of packaging an application along with its dependencies into a standardized unit called a container.

### Inference Server
A server dedicated to running machine learning models for prediction or generation, often optimized for handling multiple requests.

### Streaming Response
A technique where the API sends generated audio incrementally as it's created rather than waiting for the entire generation to complete.

### Asynchronous Processing
Handling tasks like audio generation in the background while allowing the user interface to remain responsive.

## Audio Processing Terms

### Waveform
A visual representation of an audio signal showing how amplitude changes over time.

### Spectrogram
A visual representation of the spectrum of frequencies in a sound as they vary with time.

### Amplitude
The maximum displacement or distance moved by a point on a vibrating body. In audio, relates to the loudness of a sound.

### Mono vs. Stereo
Mono audio has a single channel, while stereo has two channels (left and right) for spatial audio reproduction.

### Normalization
The process of adjusting the amplitude of audio to a standard level, often to prevent clipping or to standardize loudness.

### Resampling
Converting audio from one sample rate to another, which may be necessary when working with different audio systems.

### FFT (Fast Fourier Transform)
An algorithm that converts a signal from its time domain to its frequency domain representation, useful for audio analysis.

### dB (Decibel)
A logarithmic unit used to express the ratio of two values, commonly used to measure sound levels.

## Model Parameters

### Model Size
In AudioCraft, models come in different sizes (small, medium, large) with varying capabilities and resource requirements.

### Duration
The length of the generated audio in seconds. MusicGen typically supports up to 30 seconds per generation.

### Prompt
The text description provided to guide the audio generation process.

### Generation Parameters
Settings that control the behavior of the generation process, including temperature, top-k, top-p, and others.

### Seed
A value that initializes the random number generator to make generations reproducible. The same seed and prompt will produce the same output.

## Technical Environment

### VRAM
Video RAM, the memory on a graphics card. Larger models require more VRAM for generation.

### CPU vs. GPU Generation
Audio can be generated using either CPU (slower) or GPU (faster) processing. GPU acceleration via CUDA or MPS is recommended for practical use.

### Quantization
The process of reducing the precision of a model to decrease memory usage and increase inference speed, sometimes at the cost of quality.

### Batch Size
The number of samples processed simultaneously during generation. Larger batch sizes can be more efficient but require more memory.

## File Formats

### WAV
A standard audio file format that stores audio in an uncompressed format, preserving full quality.

### MP3
A compressed audio file format that reduces file size at the cost of some quality loss.

### OGG
An open container format maintained by the Xiph.Org Foundation, often used for Vorbis audio compression.

### FLAC
Free Lossless Audio Codec, which compresses audio without quality loss.

## Additional Concepts

### Latency
The time delay between initiating audio generation and receiving the output. Lower latency is desirable for interactive applications.

### Throughput
The rate at which audio can be generated, typically measured in seconds of audio generated per unit of time.

### Fine-tuning
The process of adapting a pre-trained model to specific tasks or styles by training it further on specialized data.

### Generation Artifacts
Unwanted anomalies or imperfections in generated audio, such as glitches, clicks, or unnatural transitions.

### Attribution
Properly crediting AI-generated content as being created by artificial intelligence rather than human artists.
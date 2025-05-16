# Getting Started with Meta AudioCraft

This tutorial covers the basics of setting up and using Meta's AudioCraft framework for AI-powered audio generation.

## Installation

To get started with AudioCraft, you'll need to:

1. Set up a Python environment (Python 3.9 recommended)
2. Install PyTorch 2.1.0+
3. Install AudioCraft and dependencies
4. Test your installation

### Platform-Specific Installation Guides

Choose the guide that matches your hardware:

- [Setting Up AudioCraft on Apple Silicon Macs (M1/M2/M3/M4)](mac-m-series.md) - Optimized setup for modern Macs with Metal GPU acceleration
- [Setting Up AudioCraft on Windows and Linux with NVIDIA GPUs](windows-linux-cuda.md) - For systems with CUDA-capable graphics cards
- [CPU-Only Installation Guide](cpu-installation.md) - For systems without compatible GPUs

## First Steps

After installation, start with our beginner tutorials:

- [Basic Usage: Your First Steps with AudioCraft](basic-usage.md) - Create your first generated audio
- [Understanding the AudioCraft Architecture](architecture.md) - Learn how the framework components work together

## Project Structure

AudioCraft consists of several key components:

- **MusicGen**: Text-to-music generation model
- **AudioGen**: Text-to-audio generation model
- **EnCodec**: Neural audio codec for compression
- **LM Models**: Language models specialized for audio generation

## Hardware Requirements

AudioCraft performance varies significantly based on your hardware:

| Hardware | Recommended Model | Generation Speed | Quality |
|----------|------------------|-----------------|---------|
| NVIDIA RTX 3070+ | medium/large | Fast (5-15s) | High |
| Apple M1/M2/M3 | small/medium | Medium (15-45s) | Good |
| CPU only | small | Slow (1-5m) | Basic |

## Common Issues and Solutions

- **Out of Memory Errors**: Try using a smaller model size or generate shorter audio clips
- **Slow Generation**: The first generation is always slower; subsequent generations are faster
- **Installation Problems**: Follow the platform-specific guides for detailed troubleshooting
- **Low Quality Output**: Try more detailed prompts or a larger model size if your hardware supports it

## Next Steps

After completing the getting started tutorials, explore these more advanced topics:

- [MusicGen Tutorials](../musicgen/README.md) - Dive deeper into music generation
- [AudioGen Tutorials](../audiogen/README.md) - Learn to create sound effects and environmental audio

For a comprehensive learning path from beginner to advanced, see our [Tutorial Structure](../../TUTORIAL_STRUCTURE.md) document.
# Chapter 17: Real-Time Audio Generation

> *"Our users want to create music and sound effects on-the-fly as they interact with our application. They don't want to wait for offline generation – they need immediate feedback as they adjust parameters and explore different sounds. How can we use AudioCraft to deliver responsive, real-time audio experiences?"*  
> — Lead Developer, Interactive Music Application

## Learning Objectives

In this chapter, you'll learn how to:

- Implement techniques for low-latency audio generation with AudioCraft
- Create streaming audio pipelines that deliver content in real-time
- Optimize model performance for interactive applications
- Build responsive interfaces for real-time audio creation
- Develop hybrid approaches that combine pre-generation and real-time synthesis

## Introduction

Real-time audio generation presents unique challenges compared to offline generation. Users expect immediate feedback when interacting with audio systems, requiring optimization techniques that balance quality, responsiveness, and computational efficiency.

In this chapter, we'll explore approaches for creating responsive audio generation systems with AudioCraft. While the base models weren't specifically designed for real-time use, we'll develop techniques that make interactive audio experiences possible through clever engineering and user experience design.

## Technical Challenges of Real-Time Generation

Before diving into implementation, let's understand the key challenges of real-time audio generation:

1. **Computational Intensity**: Neural audio models like MusicGen and AudioGen require significant computational resources. A typical 10-second audio generation might take several seconds even on a powerful GPU.

2. **Sequential Generation**: The models generate audio sequentially, making it challenging to produce continuous audio streams without interruption.

3. **Memory Usage**: The models have substantial memory requirements, particularly for longer sequences.

4. **Variable Generation Times**: Generation time can vary based on input prompts, making it difficult to guarantee consistent real-time performance.

5. **Cold Start Latency**: Loading models into memory can take several seconds, creating an initial delay.

Given these challenges, our approach will focus on:

- Chunked generation and streaming
- Aggressive optimization for model inference
- Clever UI design to mask latency
- Pre-generation of common elements
- Hybrid approaches combining neural and traditional synthesis

## Implementation: Low-Latency Audio Generation Server

First, let's create a server that optimizes AudioCraft for faster inference and streaming output:

```python
import os
import torch
import torchaudio
import numpy as np
import time
import threading
import queue
import logging
from typing import Dict, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from audiocraft.models import MusicGen, AudioGen

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("realtime-audiocraft")

class GenerationRequest(BaseModel):
    """Request for audio generation."""
    prompt: str
    duration_seconds: float = 5.0
    model_type: str = "music"  # "music" or "audio"
    model_size: str = "small"
    streaming: bool = True
    generation_params: Dict = {}

@dataclass
class AudioChunk:
    """A chunk of generated audio."""
    audio: torch.Tensor  # Shape: [channels, samples]
    sample_rate: int
    chunk_index: int
    is_final: bool = False
    latency_ms: float = 0.0

class AudioGenerationContext:
    """Context for a specific audio generation request."""
    
    def __init__(
        self,
        request_id: str,
        prompt: str,
        duration_seconds: float,
        model_type: str,
        model_size: str,
        generation_params: Dict
    ):
        self.request_id = request_id
        self.prompt = prompt
        self.duration_seconds = duration_seconds
        self.model_type = model_type
        self.model_size = model_size
        self.generation_params = generation_params
        self.start_time = time.time()
        self.chunks = []
        self.complete = False
        self.error = None
        self.output_queue = queue.Queue()

class OptimizedAudioGenerator:
    """
    Optimized generator for low-latency audio generation.
    
    This class manages model loading, optimization, and generation
    to minimize latency for real-time use cases.
    """
    
    def __init__(
        self,
        cache_dir: str = "optimized_models",
        default_chunk_size: float = 5.0,
        enable_trace: bool = True
    ):
        """
        Initialize the optimized generator.
        
        Args:
            cache_dir: Directory for model caching
            default_chunk_size: Default audio chunk size in seconds
            enable_trace: Whether to enable torch.jit tracing for models
        """
        self.cache_dir = cache_dir
        self.default_chunk_size = default_chunk_size
        self.enable_trace = enable_trace
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize device
        if torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize model cache
        self.model_cache = {}
        
        # Track active generation contexts
        self.contexts = {}
        
        # Worker thread
        self.worker_thread = None
        self.worker_running = False
        self.generation_queue = queue.PriorityQueue()
    
    def start_worker(self):
        """Start the generation worker thread."""
        if self.worker_thread is not None and self.worker_thread.is_alive():
            return
        
        self.worker_running = True
        self.worker_thread = threading.Thread(target=self._generation_worker)
        self.worker_thread.daemon = True
        self.worker_thread.start()
        logger.info("Generation worker started")
    
    def stop_worker(self):
        """Stop the generation worker thread."""
        self.worker_running = False
        if self.worker_thread is not None:
            self.worker_thread.join(timeout=2.0)
            self.worker_thread = None
        logger.info("Generation worker stopped")
    
    def _generation_worker(self):
        """Worker thread for processing generation requests."""
        while self.worker_running:
            try:
                # Get the next request from the queue
                priority, request_id = self.generation_queue.get(timeout=1.0)
                
                # Check if context still exists
                if request_id not in self.contexts:
                    self.generation_queue.task_done()
                    continue
                
                context = self.contexts[request_id]
                
                # Generate audio
                try:
                    self._generate_audio_for_context(context)
                    context.complete = True
                except Exception as e:
                    logger.error(f"Error generating audio: {str(e)}")
                    context.error = str(e)
                    context.complete = True
                
                # Mark task as done
                self.generation_queue.task_done()
                
            except queue.Empty:
                # No tasks in queue
                pass
            except Exception as e:
                logger.error(f"Error in generation worker: {str(e)}")
    
    def get_model(self, model_type: str, model_size: str) -> torch.nn.Module:
        """
        Get a cached and optimized model.
        
        Args:
            model_type: Type of model ("music" or "audio")
            model_size: Size of model ("small", "medium", "large")
            
        Returns:
            Optimized model
        """
        cache_key = f"{model_type}_{model_size}"
        
        if cache_key in self.model_cache:
            return self.model_cache[cache_key]
        
        # Load the model
        if model_type == "music":
            logger.info(f"Loading MusicGen {model_size} model")
            model = MusicGen.get_pretrained(model_size)
        else:
            logger.info(f"Loading AudioGen {model_size} model")
            model = AudioGen.get_pretrained(model_size)
        
        # Move to device
        model.to(self.device)
        
        # Apply optimizations
        if self.enable_trace and self.device == "cuda":
            try:
                logger.info(f"Tracing model for optimization")
                # Apply torch.jit optimizations
                model = self._optimize_model(model)
            except Exception as e:
                logger.warning(f"Failed to trace model: {str(e)}")
        
        # Cache the model
        self.model_cache[cache_key] = model
        return model
    
    def _optimize_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply torch.jit optimizations to a model."""
        # Note: This is a simplified example of optimization
        # Full optimization would require careful handling of model architecture
        # and may not be applicable to all parts of AudioCraft models
        
        # For now, we'll just return the original model
        # In a production system, you would apply torch.jit.script or torch.jit.trace
        # to key components of the model
        return model
    
    def create_generation_context(
        self,
        request: GenerationRequest
    ) -> str:
        """
        Create a context for audio generation.
        
        Args:
            request: Generation request parameters
            
        Returns:
            Context ID for tracking the generation
        """
        # Create a unique ID for this request
        request_id = f"req_{int(time.time() * 1000)}_{np.random.randint(1000, 9999)}"
        
        # Create context
        context = AudioGenerationContext(
            request_id=request_id,
            prompt=request.prompt,
            duration_seconds=request.duration_seconds,
            model_type=request.model_type,
            model_size=request.model_size,
            generation_params=request.generation_params
        )
        
        # Store context
        self.contexts[request_id] = context
        
        # Queue generation with a priority based on duration
        # Shorter durations get higher priority (lower number)
        priority = min(request.duration_seconds, 30.0)
        self.generation_queue.put((priority, request_id))
        
        return request_id
    
    def _generate_audio_for_context(self, context: AudioGenerationContext):
        """
        Generate audio for a given context.
        
        Args:
            context: Generation context
        """
        # Get the model
        model = self.get_model(context.model_type, context.model_size)
        
        # Set generation parameters
        default_params = {
            "duration": context.duration_seconds,
            "temperature": 1.0,
            "cfg_coef": 3.0,
            "top_k": 250
        }
        
        # Merge with provided parameters
        gen_params = dict(default_params)
        if context.generation_params:
            gen_params.update(context.generation_params)
        
        # Apply generation parameters
        model.set_generation_params(**gen_params)
        
        # Generate audio
        generation_start = time.time()
        wav = model.generate([context.prompt])
        generation_time = (time.time() - generation_start) * 1000  # ms
        
        # For simplicity, we'll treat the whole generation as one chunk
        # In a more advanced implementation, you would generate in smaller chunks
        audio_chunk = AudioChunk(
            audio=wav[0],
            sample_rate=model.sample_rate,
            chunk_index=0,
            is_final=True,
            latency_ms=generation_time
        )
        
        # Add to context
        context.chunks.append(audio_chunk)
        
        # Add to output queue
        context.output_queue.put(audio_chunk)
        
        logger.info(f"Generated audio for {context.request_id}: {generation_time:.2f}ms")
    
    def get_generation_status(self, request_id: str) -> Dict:
        """
        Get status of a generation request.
        
        Args:
            request_id: Request ID
            
        Returns:
            Status information
        """
        if request_id not in self.contexts:
            return {"error": "Request not found"}
        
        context = self.contexts[request_id]
        
        return {
            "request_id": request_id,
            "prompt": context.prompt,
            "duration": context.duration_seconds,
            "complete": context.complete,
            "error": context.error,
            "elapsed_time": time.time() - context.start_time,
            "num_chunks": len(context.chunks)
        }
    
    def get_next_audio_chunk(self, request_id: str, timeout: float = 1.0) -> Optional[AudioChunk]:
        """
        Get the next audio chunk for a request, if available.
        
        Args:
            request_id: Request ID
            timeout: Timeout in seconds
            
        Returns:
            Audio chunk or None if no chunk is available
        """
        if request_id not in self.contexts:
            return None
        
        context = self.contexts[request_id]
        
        try:
            return context.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def cleanup_context(self, request_id: str):
        """
        Clean up a generation context.
        
        Args:
            request_id: Request ID to clean up
        """
        if request_id in self.contexts:
            del self.contexts[request_id]
            logger.info(f"Cleaned up context: {request_id}")

# Create FastAPI app
app = FastAPI(title="Real-Time AudioCraft API")

# Create generator instance
generator = OptimizedAudioGenerator()

@app.on_event("startup")
async def startup_event():
    """Start the generator worker on startup."""
    generator.start_worker()

@app.on_event("shutdown")
async def shutdown_event():
    """Stop the generator worker on shutdown."""
    generator.stop_worker()

@app.post("/generate")
async def generate_audio(request: GenerationRequest, background_tasks: BackgroundTasks):
    """
    Generate audio from a text prompt.
    
    Returns a request ID that can be used to check status or stream results.
    """
    # Create generation context
    request_id = generator.create_generation_context(request)
    
    # Add cleanup task
    background_tasks.add_task(generator.cleanup_context, request_id)
    
    return {"request_id": request_id}

@app.get("/status/{request_id}")
async def check_status(request_id: str):
    """Check the status of an audio generation request."""
    return generator.get_generation_status(request_id)

@app.get("/stream/{request_id}")
async def stream_audio(request_id: str):
    """Stream generated audio as it becomes available."""
    if request_id not in generator.contexts:
        return {"error": "Request not found"}
    
    context = generator.contexts[request_id]
    
    # Function to stream audio chunks
    async def audio_stream():
        while not context.complete or not context.output_queue.empty():
            chunk = generator.get_next_audio_chunk(request_id)
            if chunk is not None:
                # Convert to WAV format for streaming
                buffer = io.BytesIO()
                torchaudio.save(
                    buffer,
                    chunk.audio.cpu().unsqueeze(0),
                    chunk.sample_rate,
                    format="wav"
                )
                buffer.seek(0)
                yield buffer.read()
                
                # If this is the final chunk, we're done
                if chunk.is_final:
                    break
            else:
                # No chunk available, wait a bit
                await asyncio.sleep(0.1)
    
    return StreamingResponse(
        audio_stream(),
        media_type="audio/wav"
    )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time audio generation."""
    await websocket.accept()
    
    try:
        while True:
            # Receive request
            data = await websocket.receive_json()
            
            # Convert to request object
            request = GenerationRequest(**data)
            
            # Create generation context
            request_id = generator.create_generation_context(request)
            
            # Send initial response
            await websocket.send_json({"request_id": request_id, "status": "generating"})
            
            # Stream results
            context = generator.contexts[request_id]
            while not context.complete or not context.output_queue.empty():
                chunk = generator.get_next_audio_chunk(request_id)
                if chunk is not None:
                    # Convert to WAV format for streaming
                    buffer = io.BytesIO()
                    torchaudio.save(
                        buffer,
                        chunk.audio.cpu().unsqueeze(0),
                        chunk.sample_rate,
                        format="wav"
                    )
                    buffer.seek(0)
                    
                    # Send audio chunk
                    await websocket.send_bytes(buffer.read())
                    
                    # If this is the final chunk, we're done
                    if chunk.is_final:
                        await websocket.send_json({"status": "complete"})
                        break
                else:
                    # No chunk available, wait a bit
                    await asyncio.sleep(0.1)
            
            # Clean up
            generator.cleanup_context(request_id)
            
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await websocket.send_json({"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Implementation: Chunked Generation and Streaming

Now, let's implement a more advanced approach that generates audio in smaller chunks for real-time streaming:

```python
import os
import torch
import torchaudio
import numpy as np
import threading
import queue
import time
import logging
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from audiocraft.models import MusicGen, AudioGen

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("streaming-audiocraft")

@dataclass
class GenerationProgress:
    """Progress information for a generation job."""
    total_chunks: int
    completed_chunks: int
    current_progress: float
    estimated_time_remaining: float
    latency_ms: float

@dataclass
class GenerationConfig:
    """Configuration for chunked generation."""
    prompt: str
    model_type: str = "music"  # "music" or "audio"
    model_size: str = "small"
    total_duration: float = 10.0
    chunk_duration: float = 2.5
    overlap_duration: float = 0.5
    output_sample_rate: int = 44100
    generation_params: Dict = None
    callback: Optional[Callable] = None

class ChunkedAudioGenerator:
    """
    Generator for streaming audio in small chunks.
    
    This class implements a progressive generation approach that
    produces audio in overlapping chunks to enable real-time streaming.
    """
    
    def __init__(
        self,
        cache_dir: str = "streaming_models",
        default_chunk_duration: float = 2.5,
        default_overlap: float = 0.5,
        chunk_queue_size: int = 3
    ):
        """
        Initialize the chunked generator.
        
        Args:
            cache_dir: Directory for model caching
            default_chunk_duration: Default chunk size in seconds
            default_overlap: Default overlap between chunks in seconds
            chunk_queue_size: Number of chunks to queue up
        """
        self.cache_dir = cache_dir
        self.default_chunk_duration = default_chunk_duration
        self.default_overlap = default_overlap
        self.chunk_queue_size = chunk_queue_size
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize device
        if torch.cuda.is_available():
            self.device = "cuda"
            # For CUDA, we can optimize memory usage
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.9)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize model cache
        self.model_cache = {}
        
        # Generation queue and worker thread
        self.generation_queue = queue.Queue()
        self.worker_thread = None
        self.worker_running = False
        
        # Active generation jobs
        self.active_jobs = {}
    
    def start_worker(self):
        """Start the generation worker thread."""
        if self.worker_thread is not None and self.worker_thread.is_alive():
            return
        
        self.worker_running = True
        self.worker_thread = threading.Thread(target=self._generation_worker)
        self.worker_thread.daemon = True
        self.worker_thread.start()
        logger.info("Chunked generation worker started")
    
    def stop_worker(self):
        """Stop the generation worker thread."""
        self.worker_running = False
        if self.worker_thread is not None:
            self.worker_thread.join(timeout=2.0)
            self.worker_thread = None
        logger.info("Chunked generation worker stopped")
    
    def _generation_worker(self):
        """Worker thread for processing generation requests."""
        while self.worker_running:
            try:
                # Get the next job from the queue
                job_id = self.generation_queue.get(timeout=1.0)
                
                # Check if job still exists
                if job_id not in self.active_jobs:
                    self.generation_queue.task_done()
                    continue
                
                # Process the job
                job = self.active_jobs[job_id]
                
                try:
                    self._process_generation_job(job_id, job)
                except Exception as e:
                    logger.error(f"Error processing job {job_id}: {str(e)}")
                    # Signal error to listener
                    if job["error_queue"] is not None:
                        job["error_queue"].put(str(e))
                
                # Mark job as done
                self.generation_queue.task_done()
                
            except queue.Empty:
                # No jobs in queue
                pass
            except Exception as e:
                logger.error(f"Error in generation worker: {str(e)}")
    
    def get_model(self, model_type: str, model_size: str) -> torch.nn.Module:
        """
        Get or load a model.
        
        Args:
            model_type: Type of model ("music" or "audio")
            model_size: Size of model ("small", "medium", "large")
            
        Returns:
            Model instance
        """
        cache_key = f"{model_type}_{model_size}"
        
        if cache_key in self.model_cache:
            return self.model_cache[cache_key]
        
        # Load model
        if model_type == "music":
            logger.info(f"Loading MusicGen {model_size} model")
            model = MusicGen.get_pretrained(model_size)
        else:
            logger.info(f"Loading AudioGen {model_size} model")
            model = AudioGen.get_pretrained(model_size)
        
        # Move to device
        model.to(self.device)
        
        # Store in cache
        self.model_cache[cache_key] = model
        
        return model
    
    def start_generation(self, config: GenerationConfig) -> str:
        """
        Start generating audio in chunks.
        
        Args:
            config: Generation configuration
            
        Returns:
            Job ID for tracking the generation
        """
        # Create a unique job ID
        job_id = f"job_{int(time.time() * 1000)}_{np.random.randint(1000, 9999)}"
        
        # Calculate number of chunks
        if config.chunk_duration <= 0:
            config.chunk_duration = self.default_chunk_duration
        
        if config.overlap_duration <= 0:
            config.overlap_duration = self.default_overlap
        
        # Ensure overlap is less than chunk duration
        if config.overlap_duration >= config.chunk_duration:
            config.overlap_duration = config.chunk_duration / 2
        
        # Calculate effective chunk duration (excluding overlap)
        effective_duration = config.chunk_duration - config.overlap_duration
        
        # Calculate total chunks needed
        total_chunks = int(np.ceil(config.total_duration / effective_duration))
        
        # Create generation parameters dictionary
        gen_params = config.generation_params or {}
        
        # Create output queue for chunks
        output_queue = queue.Queue(maxsize=self.chunk_queue_size)
        
        # Create error queue for reporting errors
        error_queue = queue.Queue()
        
        # Create job data
        job = {
            "config": config,
            "start_time": time.time(),
            "total_chunks": total_chunks,
            "completed_chunks": 0,
            "chunk_timings": [],
            "output_queue": output_queue,
            "error_queue": error_queue,
            "is_complete": False
        }
        
        # Store job
        self.active_jobs[job_id] = job
        
        # Queue for processing
        self.generation_queue.put(job_id)
        
        logger.info(f"Started chunked generation job {job_id} with {total_chunks} chunks")
        
        return job_id
    
    def _process_generation_job(self, job_id: str, job: Dict):
        """
        Process a generation job.
        
        Args:
            job_id: Job ID
            job: Job data
        """
        config = job["config"]
        
        # Get model
        model = self.get_model(config.model_type, config.model_size)
        sample_rate = model.sample_rate
        
        # Process chunks
        audio_buffer = None
        overlap_samples = int(config.overlap_duration * sample_rate)
        
        for chunk_idx in range(job["total_chunks"]):
            # Skip if job has been cancelled
            if job_id not in self.active_jobs:
                return
            
            # Calculate current duration
            current_position = chunk_idx * (config.chunk_duration - config.overlap_duration)
            current_progress = min(1.0, current_position / config.total_duration)
            
            # Get average chunk generation time
            avg_chunk_time = 0
            if len(job["chunk_timings"]) > 0:
                avg_chunk_time = sum(job["chunk_timings"]) / len(job["chunk_timings"])
            
            # Estimate remaining time
            remaining_chunks = job["total_chunks"] - chunk_idx
            estimated_time_remaining = avg_chunk_time * remaining_chunks
            
            # Update progress
            progress = GenerationProgress(
                total_chunks=job["total_chunks"],
                completed_chunks=chunk_idx,
                current_progress=current_progress,
                estimated_time_remaining=estimated_time_remaining,
                latency_ms=avg_chunk_time * 1000 if avg_chunk_time > 0 else 0
            )
            
            # Call progress callback if provided
            if config.callback is not None:
                try:
                    config.callback(job_id, progress, None)
                except Exception as e:
                    logger.error(f"Error in callback: {str(e)}")
            
            # Generate this chunk
            chunk_start_time = time.time()
            
            try:
                # Set generation params for this chunk
                generation_params = dict(config.generation_params or {})
                generation_params["duration"] = config.chunk_duration
                model.set_generation_params(**generation_params)
                
                # Apply prompt continuation if not the first chunk
                current_prompt = config.prompt
                if chunk_idx > 0 and audio_buffer is not None:
                    # For continuity, we'd use the audio_buffer as conditioning
                    # This is a simplification - actual continuation would require
                    # model-specific handling which varies by AudioCraft release
                    
                    # Here we just use the same prompt for simplicity
                    pass
                
                # Generate audio chunk
                wav = model.generate([current_prompt])
                chunk_audio = wav[0].cpu()
                
                # Record generation time
                chunk_time = time.time() - chunk_start_time
                job["chunk_timings"].append(chunk_time)
                
                # Process chunk
                if audio_buffer is None:
                    # First chunk
                    audio_buffer = chunk_audio
                else:
                    # Subsequent chunk - perform crossfade
                    if overlap_samples > 0:
                        # Create crossfade weights
                        fade_in = torch.linspace(0, 1, overlap_samples)
                        fade_out = torch.linspace(1, 0, overlap_samples)
                        
                        # Apply crossfade
                        buffer_end = audio_buffer[-overlap_samples:]
                        chunk_start = chunk_audio[:overlap_samples]
                        
                        # Perform the crossfade
                        crossfade = (buffer_end * fade_out) + (chunk_start * fade_in)
                        
                        # Combine buffers
                        audio_buffer = torch.cat([
                            audio_buffer[:-overlap_samples],
                            crossfade,
                            chunk_audio[overlap_samples:]
                        ])
                    else:
                        # No overlap, just concatenate
                        audio_buffer = torch.cat([audio_buffer, chunk_audio])
                
                # Extract the chunk to send (excluding future overlap)
                if chunk_idx < job["total_chunks"] - 1:
                    # Not the last chunk, extract excluding next overlap
                    samples_to_extract = int((config.chunk_duration - config.overlap_duration) * sample_rate)
                    
                    # Determine start position
                    if chunk_idx == 0:
                        # First chunk, start from beginning
                        start_pos = 0
                    else:
                        # Subsequent chunks, start from after previous overlap
                        start_pos = int((chunk_idx * (config.chunk_duration - config.overlap_duration)) * sample_rate)
                    
                    # Calculate end position
                    end_pos = start_pos + samples_to_extract
                    
                    # Extract chunk
                    current_chunk = audio_buffer[start_pos:end_pos]
                else:
                    # Last chunk, include everything remaining
                    if chunk_idx == 0:
                        # Special case: only one chunk
                        current_chunk = audio_buffer
                    else:
                        # Extract everything from previous point
                        start_pos = int((chunk_idx * (config.chunk_duration - config.overlap_duration)) * sample_rate)
                        current_chunk = audio_buffer[start_pos:]
                
                # Resample if needed
                if config.output_sample_rate != sample_rate:
                    resampler = torchaudio.transforms.Resample(
                        orig_freq=sample_rate,
                        new_freq=config.output_sample_rate
                    )
                    current_chunk = resampler(current_chunk)
                    output_sr = config.output_sample_rate
                else:
                    output_sr = sample_rate
                
                # Add to output queue
                chunk_data = {
                    "chunk_idx": chunk_idx,
                    "is_final": chunk_idx == job["total_chunks"] - 1,
                    "audio": current_chunk,
                    "sample_rate": output_sr,
                    "latency_ms": chunk_time * 1000
                }
                
                # Put chunk in output queue (block if full)
                job["output_queue"].put(chunk_data, timeout=30.0)
                
                # Update job status
                job["completed_chunks"] = chunk_idx + 1
                
            except Exception as e:
                logger.error(f"Error generating chunk {chunk_idx} for job {job_id}: {str(e)}")
                if job["error_queue"] is not None:
                    job["error_queue"].put(str(e))
                return
        
        # Mark job as complete
        job["is_complete"] = True
        
        logger.info(f"Completed job {job_id} in {time.time() - job['start_time']:.2f}s")
    
    def get_next_chunk(self, job_id: str, timeout: float = 1.0) -> Optional[Dict]:
        """
        Get the next available audio chunk for a job.
        
        Args:
            job_id: Job ID
            timeout: Timeout in seconds
            
        Returns:
            Chunk data or None if no chunk is available
        """
        if job_id not in self.active_jobs:
            return None
        
        job = self.active_jobs[job_id]
        
        try:
            # Check for errors first
            try:
                error = job["error_queue"].get_nowait()
                return {"error": error}
            except queue.Empty:
                pass
            
            # Get next chunk
            return job["output_queue"].get(timeout=timeout)
        except queue.Empty:
            # No chunk available
            return None
    
    def get_job_status(self, job_id: str) -> Dict:
        """
        Get status of a generation job.
        
        Args:
            job_id: Job ID
            
        Returns:
            Job status information
        """
        if job_id not in self.active_jobs:
            return {"error": "Job not found"}
        
        job = self.active_jobs[job_id]
        
        # Calculate progress
        current_position = job["completed_chunks"] * (job["config"].chunk_duration - job["config"].overlap_duration)
        current_progress = min(1.0, current_position / job["config"].total_duration)
        
        # Get average chunk generation time
        avg_chunk_time = 0
        if len(job["chunk_timings"]) > 0:
            avg_chunk_time = sum(job["chunk_timings"]) / len(job["chunk_timings"])
        
        # Estimate remaining time
        remaining_chunks = job["total_chunks"] - job["completed_chunks"]
        estimated_time_remaining = avg_chunk_time * remaining_chunks
        
        return {
            "job_id": job_id,
            "prompt": job["config"].prompt,
            "total_duration": job["config"].total_duration,
            "total_chunks": job["total_chunks"],
            "completed_chunks": job["completed_chunks"],
            "progress": current_progress,
            "is_complete": job["is_complete"],
            "elapsed_time": time.time() - job["start_time"],
            "estimated_time_remaining": estimated_time_remaining,
            "avg_chunk_latency_ms": avg_chunk_time * 1000 if avg_chunk_time > 0 else 0
        }
    
    def cancel_job(self, job_id: str):
        """
        Cancel a generation job.
        
        Args:
            job_id: Job ID to cancel
        """
        if job_id in self.active_jobs:
            logger.info(f"Cancelling job {job_id}")
            job = self.active_jobs[job_id]
            
            # Signal cancellation
            if job["error_queue"] is not None:
                job["error_queue"].put("Job cancelled")
            
            # Remove job
            del self.active_jobs[job_id]
    
    def cleanup_job(self, job_id: str):
        """
        Clean up a completed job.
        
        Args:
            job_id: Job ID to clean up
        """
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            
            # Only clean up if complete
            if job["is_complete"]:
                logger.info(f"Cleaning up completed job {job_id}")
                del self.active_jobs[job_id]

# Example usage
if __name__ == "__main__":
    # Create generator
    generator = ChunkedAudioGenerator()
    generator.start_worker()
    
    # Define callback for progress updates
    def progress_callback(job_id, progress, chunk):
        if progress:
            print(f"Job {job_id}: {progress.completed_chunks}/{progress.total_chunks} chunks, "
                  f"{progress.current_progress*100:.2f}% complete, "
                  f"Latency: {progress.latency_ms:.2f}ms")
        if chunk:
            print(f"Received chunk {chunk['chunk_idx']}, "
                  f"{'final' if chunk['is_final'] else 'not final'}, "
                  f"Latency: {chunk['latency_ms']:.2f}ms")
    
    # Create configuration
    config = GenerationConfig(
        prompt="A gentle ambient piano melody with soft pads",
        model_type="music",
        model_size="small",
        total_duration=15.0,
        chunk_duration=5.0,
        overlap_duration=1.0,
        callback=progress_callback
    )
    
    # Start generation
    job_id = generator.start_generation(config)
    print(f"Started generation job: {job_id}")
    
    # Process chunks as they become available
    while True:
        # Check job status
        status = generator.get_job_status(job_id)
        
        # Get next chunk
        chunk = generator.get_next_chunk(job_id)
        if chunk:
            if "error" in chunk:
                print(f"Error: {chunk['error']}")
                break
            
            # Process chunk
            audio = chunk["audio"]
            sample_rate = chunk["sample_rate"]
            
            # Save to file (in a real app, you'd stream to client)
            output_dir = "chunks"
            os.makedirs(output_dir, exist_ok=True)
            torchaudio.save(
                f"{output_dir}/chunk_{chunk['chunk_idx']}.wav",
                audio.unsqueeze(0),
                sample_rate
            )
            
            # If final chunk, we're done
            if chunk["is_final"]:
                print(f"Final chunk received, generation complete")
                break
        
        # Small delay
        time.sleep(0.1)
    
    # Clean up
    generator.cleanup_job(job_id)
    generator.stop_worker()
```

## Implementation: Real-Time Audio Mixer with Generated Content

This system integrates generated audio with a real-time audio mixer for interactive applications:

```python
import os
import time
import threading
import queue
import pygame
import numpy as np
import torch
import torchaudio
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass

@dataclass
class AudioLayer:
    """A layer in the audio mixer."""
    name: str
    audio: np.ndarray  # [samples]
    sample_rate: int
    volume: float = 1.0
    pan: float = 0.0  # -1 = left, 0 = center, 1 = right
    loop: bool = False
    playing: bool = False
    position: int = 0

class RealtimeAudioMixer:
    """
    Real-time audio mixer for interactive applications.
    
    This class manages multiple audio layers, mixing them in real-time
    and integrating with audio generation services.
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        buffer_size: int = 1024,
        num_channels: int = 2,
        max_layers: int = 32
    ):
        """
        Initialize the audio mixer.
        
        Args:
            sample_rate: Sample rate in Hz
            buffer_size: Audio buffer size in samples
            num_channels: Number of output channels
            max_layers: Maximum number of audio layers
        """
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.num_channels = num_channels
        self.max_layers = max_layers
        
        # Initialize pygame mixer
        pygame.mixer.init(
            frequency=sample_rate,
            size=-16,
            channels=num_channels,
            buffer=buffer_size
        )
        
        # Audio layers
        self.layers = {}
        
        # Generation services
        self.generator = None
        self.generation_queue = queue.Queue()
        self.active_generations = {}
        
        # Thread for handling generations
        self.generation_thread = None
        self.generation_running = False
    
    def start(self):
        """Start the audio mixer."""
        # Start generation thread
        self.generation_running = True
        self.generation_thread = threading.Thread(target=self._generation_worker)
        self.generation_thread.daemon = True
        self.generation_thread.start()
        
        print("Audio mixer started")
    
    def stop(self):
        """Stop the audio mixer."""
        # Stop generation thread
        self.generation_running = False
        if self.generation_thread is not None:
            self.generation_thread.join(timeout=2.0)
            self.generation_thread = None
        
        # Stop pygame mixer
        pygame.mixer.quit()
        
        print("Audio mixer stopped")
    
    def set_generator(self, generator):
        """
        Set the audio generator to use.
        
        Args:
            generator: ChunkedAudioGenerator instance
        """
        self.generator = generator
    
    def add_layer(
        self,
        name: str,
        audio: Union[np.ndarray, torch.Tensor],
        sample_rate: int,
        volume: float = 1.0,
        pan: float = 0.0,
        loop: bool = False,
        autoplay: bool = False
    ) -> bool:
        """
        Add an audio layer to the mixer.
        
        Args:
            name: Layer name
            audio: Audio data (numpy array or torch tensor)
            sample_rate: Sample rate of the audio
            volume: Initial volume (0-1)
            pan: Initial pan (-1 to 1)
            loop: Whether to loop the audio
            autoplay: Whether to start playing immediately
            
        Returns:
            Success status
        """
        if len(self.layers) >= self.max_layers:
            print(f"Cannot add layer: maximum layers ({self.max_layers}) reached")
            return False
        
        if name in self.layers:
            print(f"Layer '{name}' already exists")
            return False
        
        # Convert tensor to numpy if needed
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        
        # Resample if needed
        if sample_rate != self.sample_rate:
            # Simple resample using scipy
            from scipy import signal
            num_samples = int(len(audio) * self.sample_rate / sample_rate)
            audio = signal.resample(audio, num_samples)
        
        # Create layer
        layer = AudioLayer(
            name=name,
            audio=audio,
            sample_rate=self.sample_rate,
            volume=volume,
            pan=pan,
            loop=loop,
            playing=autoplay,
            position=0
        )
        
        # Add to layers
        self.layers[name] = layer
        
        # Create pygame sound
        sound = pygame.mixer.Sound(np.ascontiguousarray(audio * 32767).astype(np.int16))
        sound.set_volume(volume)
        
        # Store sound in layer
        layer.sound = sound
        
        # Start playing if requested
        if autoplay:
            self.play_layer(name)
        
        print(f"Added layer '{name}' ({len(audio) / self.sample_rate:.2f}s)")
        return True
    
    def remove_layer(self, name: str) -> bool:
        """
        Remove an audio layer.
        
        Args:
            name: Layer name
            
        Returns:
            Success status
        """
        if name not in self.layers:
            print(f"Layer '{name}' not found")
            return False
        
        # Stop playing
        self.stop_layer(name)
        
        # Remove layer
        del self.layers[name]
        
        print(f"Removed layer '{name}'")
        return True
    
    def play_layer(self, name: str) -> bool:
        """
        Start playing an audio layer.
        
        Args:
            name: Layer name
            
        Returns:
            Success status
        """
        if name not in self.layers:
            print(f"Layer '{name}' not found")
            return False
        
        layer = self.layers[name]
        
        # Start playing
        if not layer.playing:
            layer.sound.play(-1 if layer.loop else 0)
            layer.playing = True
            print(f"Playing layer '{name}'")
        
        return True
    
    def stop_layer(self, name: str) -> bool:
        """
        Stop playing an audio layer.
        
        Args:
            name: Layer name
            
        Returns:
            Success status
        """
        if name not in self.layers:
            print(f"Layer '{name}' not found")
            return False
        
        layer = self.layers[name]
        
        # Stop playing
        if layer.playing:
            layer.sound.stop()
            layer.playing = False
            print(f"Stopped layer '{name}'")
        
        return True
    
    def set_layer_volume(self, name: str, volume: float) -> bool:
        """
        Set the volume of an audio layer.
        
        Args:
            name: Layer name
            volume: Volume level (0-1)
            
        Returns:
            Success status
        """
        if name not in self.layers:
            print(f"Layer '{name}' not found")
            return False
        
        layer = self.layers[name]
        
        # Set volume
        volume = max(0.0, min(1.0, volume))
        layer.volume = volume
        layer.sound.set_volume(volume)
        
        return True
    
    def set_layer_pan(self, name: str, pan: float) -> bool:
        """
        Set the pan of an audio layer.
        
        Args:
            name: Layer name
            pan: Pan position (-1 to 1)
            
        Returns:
            Success status
        """
        if name not in self.layers:
            print(f"Layer '{name}' not found")
            return False
        
        layer = self.layers[name]
        
        # Set pan
        pan = max(-1.0, min(1.0, pan))
        layer.pan = pan
        
        # Implement panning (simplistic approach - not using pygame's built-in panning)
        if self.num_channels == 2:
            left_vol = min(1.0, 1.0 - pan) * layer.volume
            right_vol = min(1.0, 1.0 + pan) * layer.volume
            
            # This is a simplified approach - for better panning consider using a custom mixer
            layer.sound.set_volume((left_vol + right_vol) / 2)
        
        return True
    
    def generate_layer(
        self,
        prompt: str,
        name: str = None,
        duration: float = 10.0,
        model_type: str = "music",
        model_size: str = "small",
        volume: float = 1.0,
        pan: float = 0.0,
        loop: bool = False,
        autoplay: bool = True,
        callback: Optional[callable] = None
    ) -> str:
        """
        Generate an audio layer using the audio generator.
        
        Args:
            prompt: Text prompt for generation
            name: Layer name (auto-generated if not provided)
            duration: Duration in seconds
            model_type: Model type ("music" or "audio")
            model_size: Model size ("small", "medium", "large")
            volume: Initial volume (0-1)
            pan: Initial pan (-1 to 1)
            loop: Whether to loop the audio
            autoplay: Whether to play once generated
            callback: Optional callback for completion notification
            
        Returns:
            Generation ID
        """
        if self.generator is None:
            print("No audio generator set")
            return None
        
        # Create auto name if not provided
        if name is None:
            name = f"generated_{int(time.time() * 1000)}_{np.random.randint(1000, 9999)}"
        
        # Ensure name is unique
        if name in self.layers:
            base_name = name
            i = 1
            while f"{base_name}_{i}" in self.layers:
                i += 1
            name = f"{base_name}_{i}"
        
        # Create generation config
        from dataclasses import asdict
        config = {
            "prompt": prompt,
            "model_type": model_type,
            "model_size": model_size,
            "total_duration": duration,
            "chunk_duration": min(5.0, duration),
            "overlap_duration": min(1.0, duration / 5),
            "output_sample_rate": self.sample_rate
        }
        
        # Create generation request
        generation_id = f"gen_{int(time.time() * 1000)}_{np.random.randint(1000, 9999)}"
        
        # Store generation info
        self.active_generations[generation_id] = {
            "name": name,
            "config": config,
            "volume": volume,
            "pan": pan,
            "loop": loop,
            "autoplay": autoplay,
            "callback": callback,
            "chunks": [],
            "is_complete": False,
            "is_added": False
        }
        
        # Queue for processing
        self.generation_queue.put(generation_id)
        
        print(f"Queued generation '{generation_id}' for layer '{name}'")
        return generation_id
    
    def _generation_worker(self):
        """Worker thread for handling audio generation."""
        while self.generation_running:
            try:
                # Get next generation
                generation_id = self.generation_queue.get(timeout=1.0)
                
                # Check if still active
                if generation_id not in self.active_generations:
                    self.generation_queue.task_done()
                    continue
                
                gen_info = self.active_generations[generation_id]
                
                try:
                    # Start generation
                    from dataclasses import asdict
                    job_id = self.generator.start_generation(gen_info["config"])
                    
                    # Process chunks
                    all_audio = []
                    
                    while True:
                        # Check status
                        status = self.generator.get_job_status(job_id)
                        
                        # Check for completion
                        if status.get("is_complete", False):
                            break
                        
                        # Get next chunk
                        chunk = self.generator.get_next_chunk(job_id)
                        if chunk:
                            if "error" in chunk:
                                print(f"Error in generation {generation_id}: {chunk['error']}")
                                break
                            
                            # Store chunk
                            gen_info["chunks"].append(chunk)
                            
                            # Extract audio
                            audio = chunk["audio"]
                            if isinstance(audio, torch.Tensor):
                                audio = audio.cpu().numpy()
                            
                            all_audio.append(audio)
                            
                            # If final chunk, we're done
                            if chunk.get("is_final", False):
                                break
                        
                        # Small delay
                        time.sleep(0.1)
                    
                    # Combine all chunks
                    if all_audio:
                        combined_audio = np.concatenate(all_audio)
                        
                        # Add as layer
                        self.add_layer(
                            name=gen_info["name"],
                            audio=combined_audio,
                            sample_rate=self.sample_rate,
                            volume=gen_info["volume"],
                            pan=gen_info["pan"],
                            loop=gen_info["loop"],
                            autoplay=gen_info["autoplay"]
                        )
                        
                        gen_info["is_added"] = True
                        
                        # Call callback if provided
                        if gen_info["callback"] is not None:
                            try:
                                gen_info["callback"](generation_id, gen_info["name"], True)
                            except Exception as e:
                                print(f"Error in generation callback: {str(e)}")
                    
                    # Clean up
                    self.generator.cleanup_job(job_id)
                    
                except Exception as e:
                    print(f"Error processing generation {generation_id}: {str(e)}")
                    
                    # Call callback with error
                    if gen_info["callback"] is not None:
                        try:
                            gen_info["callback"](generation_id, gen_info["name"], False)
                        except Exception as e:
                            print(f"Error in generation callback: {str(e)}")
                
                # Mark as complete
                gen_info["is_complete"] = True
                
                # Mark task as done
                self.generation_queue.task_done()
                
            except queue.Empty:
                # No generations in queue
                pass
            except Exception as e:
                print(f"Error in generation worker: {str(e)}")
    
    def get_generation_status(self, generation_id: str) -> Dict:
        """
        Get status of an audio generation.
        
        Args:
            generation_id: Generation ID
            
        Returns:
            Status information
        """
        if generation_id not in self.active_generations:
            return {"error": "Generation not found"}
        
        gen_info = self.active_generations[generation_id]
        
        return {
            "generation_id": generation_id,
            "name": gen_info["name"],
            "prompt": gen_info["config"]["prompt"],
            "is_complete": gen_info["is_complete"],
            "is_added": gen_info["is_added"],
            "num_chunks": len(gen_info["chunks"])
        }
    
    def cancel_generation(self, generation_id: str) -> bool:
        """
        Cancel an audio generation.
        
        Args:
            generation_id: Generation ID
            
        Returns:
            Success status
        """
        if generation_id not in self.active_generations:
            print(f"Generation '{generation_id}' not found")
            return False
        
        # Remove from active generations
        del self.active_generations[generation_id]
        
        print(f"Cancelled generation '{generation_id}'")
        return True
    
    def cleanup_generation(self, generation_id: str) -> bool:
        """
        Clean up a completed generation.
        
        Args:
            generation_id: Generation ID
            
        Returns:
            Success status
        """
        if generation_id not in self.active_generations:
            print(f"Generation '{generation_id}' not found")
            return False
        
        gen_info = self.active_generations[generation_id]
        
        # Only clean up if complete
        if gen_info["is_complete"]:
            del self.active_generations[generation_id]
            print(f"Cleaned up generation '{generation_id}'")
            return True
        else:
            print(f"Cannot clean up incomplete generation '{generation_id}'")
            return False

# Example usage
if __name__ == "__main__":
    # Create audio mixer
    mixer = RealtimeAudioMixer()
    mixer.start()
    
    # Create chunked audio generator
    from chunked_audio_generator import ChunkedAudioGenerator
    generator = ChunkedAudioGenerator()
    generator.start_worker()
    
    # Set generator
    mixer.set_generator(generator)
    
    # Generate layers
    def generation_callback(generation_id, layer_name, success):
        print(f"Generation {generation_id} for layer {layer_name}: {'succeeded' if success else 'failed'}")
    
    # Generate ambient background
    ambient_id = mixer.generate_layer(
        prompt="Gentle ambient drone with slight movement",
        name="ambient_background",
        duration=20.0,
        model_type="music",
        volume=0.6,
        loop=True,
        callback=generation_callback
    )
    
    # Generate melody layer
    melody_id = mixer.generate_layer(
        prompt="Simple piano melody, reflective and calm",
        name="melody",
        duration=15.0,
        model_type="music",
        volume=0.8,
        pan=0.2,
        loop=True,
        callback=generation_callback
    )
    
    # Run for a while to let audio generate
    try:
        while True:
            time.sleep(1.0)
            
            # Check statuses
            ambient_status = mixer.get_generation_status(ambient_id)
            melody_status = mixer.get_generation_status(melody_id)
            
            print(f"Ambient: {ambient_status}")
            print(f"Melody: {melody_status}")
            
            # If both complete, done
            if ambient_status.get("is_added", False) and melody_status.get("is_added", False):
                print("Both layers generated and added")
                break
    
    except KeyboardInterrupt:
        print("Interrupted")
    
    # Add a sound effect
    sfx_id = mixer.generate_layer(
        prompt="Short chime sound, bright and clear",
        name="notification",
        duration=2.0,
        model_type="audio",
        volume=1.0,
        loop=False,
        autoplay=False,
        callback=generation_callback
    )
    
    # Wait for SFX to be ready
    while True:
        sfx_status = mixer.get_generation_status(sfx_id)
        if sfx_status.get("is_added", False):
            break
        time.sleep(0.5)
    
    # Play SFX a few times
    for i in range(3):
        mixer.play_layer("notification")
        time.sleep(3.0)
    
    # Fade out ambient
    for vol in np.linspace(0.6, 0.0, 10):
        mixer.set_layer_volume("ambient_background", vol)
        time.sleep(0.3)
    
    # Clean up
    mixer.stop()
    generator.stop_worker()
```

## Client-Side Optimization Techniques

While server-side optimizations are critical, client-side techniques also play a crucial role in providing a responsive user experience:

```javascript
// Real-time Audio Generation Client
class AudioCraftClient {
  constructor(serverUrl) {
    this.serverUrl = serverUrl;
    this.socket = null;
    this.audioContext = null;
    this.audioQueue = [];
    this.bufferAhead = 3; // Number of chunks to buffer ahead
    this.isPlaying = false;
    this.isConnected = false;
    this.currentSource = null;
    this.scheduledSources = [];
    this.nextScheduleTime = 0;
    this.callbacks = {
      onConnected: null,
      onDisconnected: null,
      onGenerated: null,
      onError: null,
      onProgress: null
    };
  }

  // Connect to the server
  connect() {
    if (this.socket) {
      this.socket.close();
    }

    // Initialize audio context with user gesture
    this.initAudioContext();

    // Connect to WebSocket server
    this.socket = new WebSocket(this.serverUrl);

    this.socket.onopen = () => {
      console.log('Connected to AudioCraft server');
      this.isConnected = true;
      if (this.callbacks.onConnected) this.callbacks.onConnected();
    };

    this.socket.onclose = () => {
      console.log('Disconnected from AudioCraft server');
      this.isConnected = false;
      if (this.callbacks.onDisconnected) this.callbacks.onDisconnected();
    };

    this.socket.onerror = (error) => {
      console.error('WebSocket error:', error);
      if (this.callbacks.onError) this.callbacks.onError(error);
    };

    this.socket.onmessage = (event) => {
      this.handleMessage(event.data);
    };
  }

  // Initialize audio context (must be called from a user gesture)
  initAudioContext() {
    if (!this.audioContext) {
      // Create audio context
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      this.audioContext = new AudioContext();

      // Create gain node for volume control
      this.gainNode = this.audioContext.createGain();
      this.gainNode.gain.value = 1.0;
      this.gainNode.connect(this.audioContext.destination);
    }
  }

  // Handle incoming WebSocket messages
  async handleMessage(data) {
    // Check if this is a JSON message
    if (typeof data === 'string') {
      try {
        const message = JSON.parse(data);
        
        // Handle different message types
        if (message.type === 'status') {
          // Status update
          if (this.callbacks.onProgress) {
            this.callbacks.onProgress(message.data);
          }
        } else if (message.type === 'complete') {
          // Generation complete
          if (this.callbacks.onGenerated) {
            this.callbacks.onGenerated(message.data);
          }
        } else if (message.type === 'error') {
          // Error message
          console.error('Server error:', message.error);
          if (this.callbacks.onError) {
            this.callbacks.onError(message.error);
          }
        }
        return;
      } catch (e) {
        // Not JSON, continue to handle as binary data
      }
    }
    
    // Handle binary audio data
    if (data instanceof Blob) {
      try {
        // Convert blob to array buffer
        const arrayBuffer = await data.arrayBuffer();
        
        // Decode audio data
        const audioBuffer = await this.audioContext.decodeAudioData(arrayBuffer);
        
        // Add to queue
        this.audioQueue.push(audioBuffer);
        
        // Start playing if not already
        if (!this.isPlaying && this.audioQueue.length >= this.bufferAhead) {
          this.startPlayback();
        } else if (this.isPlaying) {
          // Schedule next chunk if playing
          this.scheduleNextChunk();
        }
      } catch (e) {
        console.error('Error decoding audio data:', e);
        if (this.callbacks.onError) {
          this.callbacks.onError('Error decoding audio: ' + e.message);
        }
      }
    }
  }

  // Generate audio from a prompt
  generate(promptData) {
    if (!this.isConnected) {
      if (this.callbacks.onError) {
        this.callbacks.onError('Not connected to server');
      }
      return false;
    }

    // Reset state
    this.reset();

    // Send generation request
    this.socket.send(JSON.stringify({
      type: 'generate',
      prompt: promptData.prompt,
      duration: promptData.duration || 10.0,
      model_type: promptData.modelType || 'music',
      model_size: promptData.modelSize || 'small',
      streaming: true,
      generation_params: promptData.params || {}
    }));

    return true;
  }

  // Start audio playback
  startPlayback() {
    if (!this.audioContext || this.audioQueue.length === 0) return;

    // Resume audio context if suspended
    if (this.audioContext.state === 'suspended') {
      this.audioContext.resume();
    }

    this.isPlaying = true;
    this.nextScheduleTime = this.audioContext.currentTime;

    // Schedule initial chunks
    this.scheduleNextChunk();
  }

  // Schedule the next audio chunk
  scheduleNextChunk() {
    if (!this.isPlaying || this.audioQueue.length === 0) return;

    const chunk = this.audioQueue.shift();
    
    // Create source
    const source = this.audioContext.createBufferSource();
    source.buffer = chunk;
    source.connect(this.gainNode);
    
    // Schedule source
    source.start(this.nextScheduleTime);
    
    // Store for later cleanup
    this.scheduledSources.push(source);
    
    // Update next schedule time
    this.nextScheduleTime += chunk.duration;
    
    // Set event for when this chunk ends
    source.onended = () => {
      // Remove from scheduled sources
      const index = this.scheduledSources.indexOf(source);
      if (index !== -1) {
        this.scheduledSources.splice(index, 1);
      }
      
      // If no more chunks and queue is empty, we're done
      if (this.scheduledSources.length === 0 && this.audioQueue.length === 0) {
        this.isPlaying = false;
      }
    };
  }

  // Stop playback
  stop() {
    // Stop all scheduled sources
    this.scheduledSources.forEach(source => {
      try {
        source.stop();
      } catch (e) {
        // Source might already be stopped
      }
    });
    
    // Clear scheduled sources
    this.scheduledSources = [];
    
    // Reset state
    this.isPlaying = false;
    this.audioQueue = [];
    this.nextScheduleTime = 0;
  }

  // Reset client state
  reset() {
    // Stop any current playback
    this.stop();
    
    // Clear queue
    this.audioQueue = [];
  }

  // Set buffer ahead amount (how many chunks to buffer before starting playback)
  setBufferAhead(chunks) {
    this.bufferAhead = Math.max(1, chunks);
  }

  // Set volume
  setVolume(volume) {
    if (this.gainNode) {
      // Clamp volume to 0-1
      volume = Math.max(0, Math.min(1, volume));
      
      // Apply volume
      this.gainNode.gain.value = volume;
    }
  }

  // Set callback functions
  setCallbacks(callbacks) {
    for (const key in callbacks) {
      if (typeof callbacks[key] === 'function') {
        this.callbacks[key] = callbacks[key];
      }
    }
  }

  // Disconnect from server
  disconnect() {
    if (this.socket) {
      this.socket.close();
      this.socket = null;
    }
    
    this.isConnected = false;
  }
}

// Example usage:
document.addEventListener('DOMContentLoaded', () => {
  const client = new AudioCraftClient('ws://localhost:8000/ws');
  
  // Set callbacks
  client.setCallbacks({
    onConnected: () => {
      console.log('Connected to server');
      document.getElementById('status').textContent = 'Connected';
      document.getElementById('generate-btn').disabled = false;
    },
    onDisconnected: () => {
      console.log('Disconnected from server');
      document.getElementById('status').textContent = 'Disconnected';
      document.getElementById('generate-btn').disabled = true;
    },
    onProgress: (progress) => {
      console.log('Generation progress:', progress);
      const percent = Math.round(progress.progress * 100);
      document.getElementById('progress').style.width = `${percent}%`;
      document.getElementById('progress').textContent = `${percent}%`;
    },
    onGenerated: (data) => {
      console.log('Generation complete:', data);
      document.getElementById('status').textContent = 'Generation complete';
    },
    onError: (error) => {
      console.error('Error:', error);
      document.getElementById('status').textContent = `Error: ${error}`;
    }
  });
  
  // Connect button
  document.getElementById('connect-btn').addEventListener('click', () => {
    client.connect();
  });
  
  // Generate button
  document.getElementById('generate-btn').addEventListener('click', () => {
    const prompt = document.getElementById('prompt-input').value;
    const duration = parseFloat(document.getElementById('duration-input').value);
    
    if (!prompt) {
      alert('Please enter a prompt');
      return;
    }
    
    document.getElementById('status').textContent = 'Generating...';
    
    client.generate({
      prompt: prompt,
      duration: duration || 10.0,
      modelType: 'music',
      modelSize: 'small'
    });
  });
  
  // Stop button
  document.getElementById('stop-btn').addEventListener('click', () => {
    client.stop();
    document.getElementById('status').textContent = 'Stopped';
  });
  
  // Volume control
  document.getElementById('volume-slider').addEventListener('input', (e) => {
    const volume = parseFloat(e.target.value);
    client.setVolume(volume);
  });
});
```

## Real-Time Audio Generation Best Practices

Based on our implementations, here are best practices for real-time audio generation with AudioCraft:

1. **Chunked Generation and Streaming**
   - Generate audio in small, overlapping chunks
   - Implement proper crossfading between chunks
   - Stream chunks to the client as they're generated
   - Buffer ahead to avoid playback interruptions

2. **Model Optimization**
   - Use smaller model sizes for faster generation
   - Apply quantization for reduced memory usage
   - Implement torch.jit tracing where applicable
   - Use hardware-specific optimizations

3. **Progressive Loading and UI Design**
   - Start playback before the full audio is generated
   - Display progress indicators for generation
   - Provide immediate feedback for user interactions
   - Implement cancelable generations

4. **Hybrid Approaches**
   - Combine pre-generated content with real-time elements
   - Use techniques like layering and crossfading for seamless transitions
   - Implement a library of cached generations for common prompts
   - Create fallbacks for when real-time generation is too slow

5. **Resource Management**
   - Implement proper cleanup for completed generations
   - Monitor and limit memory usage
   - Prioritize shorter generations for quicker feedback
   - Scale compute resources based on demand

## Hands-On Challenge: Create a Real-Time Audio Experience

**Challenge:** Build a complete real-time audio generation experience for an interactive application.

1. Create a system that generates layered audio in response to user interactions
2. Implement a mixer that combines multiple audio streams in real-time
3. Add parameter controls that let users modify the audio as it plays
4. Create smooth transitions between different audio states
5. Optimize the system for responsive performance

**Steps to implement:**

1. Create a chunked audio generation server with WebSocket streaming
2. Implement a client-side audio manager with buffering and mixing
3. Design UI controls for real-time parameter adjustment
4. Build transition logic for smooth audio state changes
5. Implement performance monitoring and optimization

## Next Steps

In the next chapter, we'll explore advanced applications and industry use cases for AudioCraft. We'll look at how these techniques can be applied in games, media production, music creation, and more.

Copyright © 2025 Scott Friedman. Licensed under [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).
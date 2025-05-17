# Building a REST API for AudioCraft

This tutorial will guide you through creating a robust REST API for AudioCraft, enabling you to integrate AI-powered audio generation into your applications. We'll use Flask to create the API and implement best practices for security, performance, and scalability.

## Prerequisites

- AudioCraft successfully installed
- Python 3.9+
- Basic understanding of REST APIs
- Familiarity with Flask or similar web frameworks

## Installation Requirements

```bash
pip install flask flask-cors gunicorn pydub soundfile
```

## Basic API Implementation

Let's start with a simple implementation that provides endpoints for music generation:

```python
# app.py
import os
import uuid
import base64
import io
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
import soundfile as sf
from audiocraft.models import MusicGen
from pydub import AudioSegment
import time

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Configuration
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "generated_audio")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global model references
models = {}

def get_model(model_size="small"):
    """Load and cache model instances."""
    if model_size not in models:
        print(f"Loading {model_size} model...")
        models[model_size] = MusicGen.get_pretrained(model_size)
        
        # Determine device
        if torch.backends.mps.is_available():
            device = "mps"
            print(f"Using MPS (Metal) for {model_size} model")
        elif torch.cuda.is_available():
            device = "cuda"
            print(f"Using CUDA for {model_size} model")
        else:
            device = "cpu"
            print(f"Using CPU for {model_size} model (slow)")
            
        models[model_size].to(device)
    
    return models[model_size]

@app.route("/health", methods=["GET"])
def health_check():
    """API health check endpoint."""
    return jsonify({
        "status": "healthy",
        "version": "1.0.0",
        "models_loaded": list(models.keys())
    })

@app.route("/generate", methods=["POST"])
def generate_audio():
    """Generate audio from text prompt."""
    try:
        # Parse request data
        data = request.json
        prompt = data.get("prompt", "")
        model_size = data.get("model_size", "small")
        duration = float(data.get("duration", 5.0))
        temperature = float(data.get("temperature", 1.0))
        top_k = int(data.get("top_k", 250))
        top_p = float(data.get("top_p", 0.0))
        output_format = data.get("output_format", "wav")
        return_type = data.get("return_type", "url")  # 'url' or 'base64'
        
        # Validate inputs
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400
            
        if model_size not in ["small", "medium", "large"]:
            return jsonify({"error": "Invalid model size"}), 400
            
        if duration < 1.0 or duration > 30.0:
            return jsonify({"error": "Duration must be between 1 and 30 seconds"}), 400
            
        # Get model
        model = get_model(model_size)
        
        # Set generation parameters
        model.set_generation_params(
            duration=duration,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        
        # Start generation
        start_time = time.time()
        print(f"Generating audio for prompt: '{prompt}'")
        
        with torch.no_grad():
            wav = model.generate([prompt])
        
        # Calculate generation time
        generation_time = time.time() - start_time
        print(f"Generation completed in {generation_time:.2f} seconds")
        
        # Create unique filename
        filename = f"{uuid.uuid4()}"
        wav_path = os.path.join(OUTPUT_DIR, f"{filename}.wav")
        
        # Save the audio as WAV first
        sf.write(wav_path, wav[0].cpu().numpy(), model.sample_rate)
        
        # Convert to desired format if not WAV
        if output_format.lower() != "wav":
            output_path = os.path.join(OUTPUT_DIR, f"{filename}.{output_format}")
            audio = AudioSegment.from_wav(wav_path)
            audio.export(output_path, format=output_format)
            os.remove(wav_path)  # Remove WAV file
            final_path = output_path
        else:
            final_path = wav_path
            
        # Generate response based on return type
        if return_type == "base64":
            # Return base64 encoded audio
            with open(final_path, "rb") as audio_file:
                encoded_audio = base64.b64encode(audio_file.read()).decode("utf-8")
                
            return jsonify({
                "success": True,
                "generation_time": generation_time,
                "audio": encoded_audio,
                "format": output_format,
                "sample_rate": model.sample_rate
            })
        else:
            # Return URL to audio file
            # In a production environment, you would use proper URL generation
            # This is a simplified example
            file_url = f"/audio/{os.path.basename(final_path)}"
            
            return jsonify({
                "success": True,
                "generation_time": generation_time,
                "audio_url": file_url,
                "format": output_format,
                "sample_rate": model.sample_rate
            })
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/audio/<filename>", methods=["GET"])
def serve_audio(filename):
    """Serve generated audio files."""
    try:
        file_path = os.path.join(OUTPUT_DIR, filename)
        return send_file(file_path, as_attachment=True)
    except Exception as e:
        return jsonify({"error": str(e)}), 404

if __name__ == "__main__":
    # Pre-load the small model for faster first generation
    get_model("small")
    
    # Run the Flask application
    # Note: In production, use gunicorn or another WSGI server
    app.run(host="0.0.0.0", port=5000, debug=False)
```

## Running the API Server

Save the code above as `app.py` and run it with:

```bash
# Development server
python app.py

# Production server (recommended)
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## API Endpoints

### Health Check

```
GET /health
```

Returns the API status and information about loaded models.

### Generate Audio

```
POST /generate
```

Generate audio from a text prompt.

**Request Body:**

```json
{
  "prompt": "An electronic dance track with a catchy melody",
  "model_size": "small",
  "duration": 10.0,
  "temperature": 1.0,
  "top_k": 250,
  "top_p": 0.0,
  "output_format": "mp3",
  "return_type": "url"
}
```

**Parameters:**

- `prompt` (string, required): Text description of the audio to generate
- `model_size` (string, optional): Model size ("small", "medium", "large"), defaults to "small"
- `duration` (number, optional): Duration in seconds (1-30), defaults to 5.0
- `temperature` (number, optional): Controls randomness (0.1-2.0), defaults to 1.0
- `top_k` (number, optional): Controls diversity, defaults to 250
- `top_p` (number, optional): Nucleus sampling parameter, defaults to 0.0
- `output_format` (string, optional): Audio format ("wav", "mp3", "ogg"), defaults to "wav"
- `return_type` (string, optional): How to return audio ("url" or "base64"), defaults to "url"

**Response (URL return type):**

```json
{
  "success": true,
  "generation_time": 3.45,
  "audio_url": "/audio/a1b2c3d4-e5f6-7890-abcd-ef1234567890.mp3",
  "format": "mp3",
  "sample_rate": 32000
}
```

**Response (base64 return type):**

```json
{
  "success": true,
  "generation_time": 3.45,
  "audio": "base64encodedaudiodata...",
  "format": "mp3",
  "sample_rate": 32000
}
```

### Serve Audio

```
GET /audio/<filename>
```

Returns the generated audio file.

## Adding Melody Conditioning

Let's extend our API to support melody conditioning:

```python
@app.route("/generate-with-melody", methods=["POST"])
def generate_with_melody():
    """Generate audio based on a text prompt and melody."""
    try:
        # Get form data and files
        prompt = request.form.get("prompt", "")
        model_size = request.form.get("model_size", "medium")
        duration = float(request.form.get("duration", 10.0))
        temperature = float(request.form.get("temperature", 1.0))
        output_format = request.form.get("output_format", "wav")
        
        # Check if melody file was uploaded
        if "melody" not in request.files:
            return jsonify({"error": "No melody file provided"}), 400
            
        melody_file = request.files["melody"]
        
        # Save melody file temporarily
        temp_melody_path = os.path.join(OUTPUT_DIR, f"temp_melody_{uuid.uuid4()}.wav")
        melody_file.save(temp_melody_path)
        
        # Load and process the melody
        import torchaudio
        melody, sr = torchaudio.load(temp_melody_path)
        
        # If stereo, convert to mono
        if melody.shape[0] > 1:
            melody = melody.mean(dim=0, keepdim=True)
        
        # Resample if needed
        if sr != 32000:
            resampler = torchaudio.transforms.Resample(sr, 32000)
            melody = resampler(melody)
        
        # Get model
        model = get_model(model_size)
        
        # Set generation parameters
        model.set_generation_params(
            duration=duration,
            temperature=temperature,
        )
        
        # Determine device and move melody tensor
        device = next(model.parameters()).device
        melody = melody.to(device)
        
        # Generate with melody conditioning
        start_time = time.time()
        print(f"Generating audio with melody conditioning for prompt: '{prompt}'")
        
        with torch.no_grad():
            wav = model.generate_with_chroma([prompt], melody.unsqueeze(0))
        
        generation_time = time.time() - start_time
        print(f"Generation completed in {generation_time:.2f} seconds")
        
        # Process and return audio (similar to the generate endpoint)
        filename = f"{uuid.uuid4()}"
        wav_path = os.path.join(OUTPUT_DIR, f"{filename}.wav")
        
        # Save the audio
        sf.write(wav_path, wav[0].cpu().numpy(), model.sample_rate)
        
        # Convert if needed
        if output_format.lower() != "wav":
            output_path = os.path.join(OUTPUT_DIR, f"{filename}.{output_format}")
            audio = AudioSegment.from_wav(wav_path)
            audio.export(output_path, format=output_format)
            os.remove(wav_path)
            final_path = output_path
        else:
            final_path = wav_path
            
        # Clean up temporary melody file
        os.remove(temp_melody_path)
        
        # Return URL to audio file
        file_url = f"/audio/{os.path.basename(final_path)}"
        
        return jsonify({
            "success": True,
            "generation_time": generation_time,
            "audio_url": file_url,
            "format": output_format,
            "sample_rate": model.sample_rate
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500
```

## Implementing Background Processing

For longer generations, it's better to process requests asynchronously. Let's add a task queue system using Celery:

```python
# Add to requirements:
# pip install celery redis

# tasks.py
from celery import Celery
import os
import torch
import soundfile as sf
from audiocraft.models import MusicGen
from pydub import AudioSegment

# Configure Celery
celery_app = Celery('audiocraft_api',
                   broker='redis://localhost:6379/0',
                   backend='redis://localhost:6379/0')

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)

# Global models dict
models = {}

def get_model(model_size="small"):
    """Load and cache model instances."""
    if model_size not in models:
        print(f"Loading {model_size} model...")
        models[model_size] = MusicGen.get_pretrained(model_size)
        
        # Determine device
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
            
        models[model_size].to(device)
    
    return models[model_size]

@celery_app.task
def generate_audio_task(prompt, model_size, duration, temperature, top_k, top_p, output_format):
    """Task to generate audio asynchronously."""
    try:
        # Get model
        model = get_model(model_size)
        
        # Set generation parameters
        model.set_generation_params(
            duration=duration,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        
        # Generate audio
        with torch.no_grad():
            wav = model.generate([prompt])
        
        # Save audio
        output_dir = os.path.join(os.path.dirname(__file__), "generated_audio")
        os.makedirs(output_dir, exist_ok=True)
        
        import uuid
        filename = f"{uuid.uuid4()}"
        wav_path = os.path.join(output_dir, f"{filename}.wav")
        
        # Save as WAV first
        sf.write(wav_path, wav[0].cpu().numpy(), model.sample_rate)
        
        # Convert to desired format if not WAV
        if output_format.lower() != "wav":
            output_path = os.path.join(output_dir, f"{filename}.{output_format}")
            audio = AudioSegment.from_wav(wav_path)
            audio.export(output_path, format=output_format)
            os.remove(wav_path)
            final_path = output_path
        else:
            final_path = wav_path
            
        # Return the filename for later retrieval
        return {
            "success": True,
            "filename": os.path.basename(final_path),
            "format": output_format,
            "sample_rate": model.sample_rate
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}
```

Now update your app.py file to use the Celery tasks:

```python
# app.py (modified to use Celery)
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
from tasks import generate_audio_task

app = Flask(__name__)
CORS(app)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "generated_audio")
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.route("/generate-async", methods=["POST"])
def generate_audio_async():
    """Submit an asynchronous generation task."""
    try:
        data = request.json
        prompt = data.get("prompt", "")
        model_size = data.get("model_size", "small")
        duration = float(data.get("duration", 5.0))
        temperature = float(data.get("temperature", 1.0))
        top_k = int(data.get("top_k", 250))
        top_p = float(data.get("top_p", 0.0))
        output_format = data.get("output_format", "wav")
        
        # Validate inputs
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400
        
        # Submit task to Celery
        task = generate_audio_task.delay(
            prompt, model_size, duration, temperature, top_k, top_p, output_format
        )
        
        return jsonify({
            "task_id": task.id,
            "status": "processing"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/tasks/<task_id>", methods=["GET"])
def check_task(task_id):
    """Check the status of an asynchronous task."""
    try:
        task = generate_audio_task.AsyncResult(task_id)
        
        if task.state == 'PENDING':
            response = {
                'status': 'processing',
                'current': 0,
                'total': 1,
            }
        elif task.state == 'SUCCESS':
            result = task.result
            if result['success']:
                response = {
                    'status': 'completed',
                    'result': result,
                    'audio_url': f"/audio/{result['filename']}"
                }
            else:
                response = {
                    'status': 'failed',
                    'error': result['error']
                }
        else:
            # Failed task
            response = {
                'status': 'failed',
                'error': str(task.info),
            }
            
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
```

## Running the Asynchronous API

To run the asynchronous API, you'll need:

1. A Redis server (for Celery message broker)
2. Celery workers
3. The Flask application

```bash
# Terminal 1: Start Redis (or install as a service)
redis-server

# Terminal 2: Start Celery worker
celery -A tasks worker --loglevel=info

# Terminal 3: Start Flask application
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## Security Considerations

1. **API Rate Limiting**: Implement rate limiting to prevent abuse
2. **Authentication**: Add API key or OAuth authentication
3. **Input Validation**: Validate all user inputs thoroughly
4. **Content Filtering**: Consider filtering inappropriate prompts
5. **Resource Limits**: Set maximum duration and number of concurrent generations

Here's a simple API key implementation:

```python
# Simple API key validation middleware
API_KEYS = {
    "your-api-key-here": {
        "name": "Development Key",
        "rate_limit": 100,  # requests per day
    }
}

def require_api_key(view_function):
    @functools.wraps(view_function)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if api_key and api_key in API_KEYS:
            return view_function(*args, **kwargs)
        else:
            return jsonify({"error": "Invalid or missing API key"}), 401
    return decorated_function

# Apply to routes
@app.route("/generate", methods=["POST"])
@require_api_key
def generate_audio():
    # ...
```

## Client Example

Here's a JavaScript example for calling your API:

```javascript
// Synchronous generation
async function generateAudio(prompt, options = {}) {
  const response = await fetch('https://your-api-url/generate', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': 'your-api-key-here'
    },
    body: JSON.stringify({
      prompt: prompt,
      model_size: options.modelSize || 'small',
      duration: options.duration || 5.0,
      temperature: options.temperature || 1.0,
      top_k: options.topK || 250,
      top_p: options.topP || 0.0,
      output_format: options.format || 'mp3',
      return_type: options.returnType || 'url'
    })
  });
  
  return await response.json();
}

// Asynchronous generation with polling
async function generateAudioAsync(prompt, options = {}) {
  // Submit task
  const submitResponse = await fetch('https://your-api-url/generate-async', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': 'your-api-key-here'
    },
    body: JSON.stringify({
      prompt: prompt,
      model_size: options.modelSize || 'small',
      duration: options.duration || 5.0,
      temperature: options.temperature || 1.0,
      top_k: options.topK || 250,
      top_p: options.topP || 0.0,
      output_format: options.format || 'mp3'
    })
  });
  
  const submitData = await submitResponse.json();
  const taskId = submitData.task_id;
  
  // Poll for result
  const poll = async (resolve, reject) => {
    try {
      const response = await fetch(`https://your-api-url/tasks/${taskId}`, {
        headers: { 'X-API-Key': 'your-api-key-here' }
      });
      
      const data = await response.json();
      
      if (data.status === 'completed') {
        resolve(data);
      } else if (data.status === 'failed') {
        reject(new Error(data.error));
      } else {
        // Still processing, poll again after delay
        setTimeout(() => poll(resolve, reject), 1000);
      }
    } catch (err) {
      reject(err);
    }
  };
  
  return new Promise(poll);
}

// Example usage
generateAudio("A peaceful piano melody with gentle strings")
  .then(result => {
    if (result.success) {
      const audioPlayer = document.getElementById('audioPlayer');
      audioPlayer.src = result.audio_url;
    }
  })
  .catch(error => console.error(error));
```

## Further Improvements

1. **Caching**: Implement Redis caching for commonly requested generations
2. **Logging**: Add comprehensive logging for debugging and analytics
3. **Documentation**: Add Swagger/OpenAPI documentation
4. **Load Balancing**: Distribute workload across multiple servers
5. **Model Versioning**: Support specific model versions

## Conclusion

This tutorial has shown you how to create a REST API for AudioCraft that can handle both synchronous and asynchronous audio generation requests. By following these patterns, you can integrate AudioCraft into your applications while ensuring good performance, scalability, and security.

For production deployments, consider using a robust tech stack:

- **Application**: Flask/FastAPI with Gunicorn
- **Task Queue**: Celery with Redis/RabbitMQ
- **Storage**: AWS S3 or similar for audio files
- **Authentication**: OAuth or API key management system
- **Monitoring**: Prometheus + Grafana
- **Deployment**: Docker + Kubernetes

## Next Steps

- See the [Docker Containerization Guide](docker_containerization.md) for containerizing your API
- Learn about [Scaling Strategies](scaling_strategies.md) for handling high traffic
- Explore [WebSocket Integration](websocket_integration.md) for real-time applications
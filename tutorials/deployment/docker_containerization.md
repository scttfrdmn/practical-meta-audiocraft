# Docker Containerization for AudioCraft

This guide explains how to containerize AudioCraft applications using Docker, allowing for consistent deployment across different environments. We'll cover creating optimized Docker images, implementing multi-container setups, and deploying to various platforms.

## Prerequisites

- Docker installed on your system
- Basic understanding of Docker concepts
- AudioCraft API or application code ready for containerization

## Why Containerize AudioCraft?

Containerizing AudioCraft offers several advantages:

1. **Consistency**: Same environment across development, testing, and production
2. **Isolation**: Dependencies packaged with the application
3. **Portability**: Run anywhere Docker is supported
4. **Scalability**: Easy to scale with orchestration tools like Kubernetes
5. **Version Control**: Track environment changes alongside code

## Basic Dockerfile for AudioCraft

Let's start with a basic Dockerfile for an AudioCraft application:

```dockerfile
# Base image with PyTorch and CUDA support
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install AudioCraft
RUN pip install --no-cache-dir audiocraft

# Copy application code
COPY . .

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
```

Create a `requirements.txt` file with your application dependencies:

```
flask==2.3.3
flask-cors==4.0.0
gunicorn==21.2.0
pydub==0.25.1
soundfile==0.12.1
celery==5.3.4
redis==5.0.1
```

## Building the Docker Image

Build your Docker image:

```bash
docker build -t audiocraft-api .
```

Run the container:

```bash
docker run -p 5000:5000 audiocraft-api
```

## CPU vs. GPU Dockerfiles

### For CPU-only Environments

```dockerfile
# Smaller base image without CUDA
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install dependencies and AudioCraft
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir "torch>=2.1.0" "torchaudio>=2.1.0" --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir audiocraft

# Copy application code
COPY . .

# Set environment variables for CPU optimization
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "app:app"]
```

### For GPU Environments

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    python3-dev \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set Python aliases
RUN ln -sf /usr/bin/python3.9 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Copy requirements
COPY requirements.txt .

# Install dependencies and AudioCraft
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir audiocraft

# Copy application code
COPY . .

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "app:app"]
```

To run the GPU container, you'll need to use NVIDIA's Container Toolkit:

```bash
docker run --gpus all -p 5000:5000 audiocraft-api-gpu
```

## Multi-Container Setup with Docker Compose

For a complete application with Redis for task queuing and a separate worker container:

```yaml
# docker-compose.yml
version: '3.8'

services:
  # API server
  api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "5000:5000"
    environment:
      - REDIS_URL=redis://redis:6379/0
    volumes:
      - ./generated_audio:/app/generated_audio
    depends_on:
      - redis
    restart: unless-stopped

  # Celery worker for background processing
  worker:
    build:
      context: .
      dockerfile: Dockerfile.worker
    environment:
      - REDIS_URL=redis://redis:6379/0
    volumes:
      - ./generated_audio:/app/generated_audio
    depends_on:
      - redis
    command: celery -A tasks worker --loglevel=info
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # Redis for task queue and caching
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    restart: unless-stopped

volumes:
  redis-data:
```

Create a separate Dockerfile for the worker:

```dockerfile
# Dockerfile.worker
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    python3-dev \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set Python aliases
RUN ln -sf /usr/bin/python3.9 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Copy requirements
COPY requirements.txt .

# Install dependencies and AudioCraft
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir audiocraft

# Copy application code
COPY . .

# Default command is set in docker-compose.yml
```

Start the multi-container setup:

```bash
docker-compose up -d
```

## Optimizing Docker Images

### Multi-stage Builds

For smaller images, use multi-stage builds:

```dockerfile
# Build stage
FROM python:3.9 AS builder

WORKDIR /build

COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt
RUN pip install --no-cache-dir --prefix=/install audiocraft

# Runtime stage
FROM python:3.9-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
```

### Layer Caching Optimization

Organize your Dockerfile to take advantage of layer caching:

1. Install dependencies that change less frequently first
2. Copy application code last (since it changes most frequently)
3. Use `.dockerignore` to exclude unnecessary files

Example `.dockerignore`:

```
.git
__pycache__
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.vscode/
.idea/
generated_audio/
*.wav
*.mp3
*.ogg
node_modules/
```

## Model Management Strategies

### Pre-downloading Models

For faster container startup, pre-download models during build:

```dockerfile
# After installing AudioCraft
RUN python -c "from audiocraft.models import MusicGen; MusicGen.get_pretrained('small')"
```

### External Volume for Models

Use a Docker volume to persist models between container restarts:

```yaml
# In docker-compose.yml
services:
  api:
    # ... other config
    volumes:
      - ./generated_audio:/app/generated_audio
      - model-cache:/root/.cache/torch/hub
      
volumes:
  model-cache:
```

## Deployment Examples

### AWS Elastic Container Service (ECS)

1. Build and push your Docker image to Amazon ECR:

```bash
# Create ECR repository
aws ecr create-repository --repository-name audiocraft-api

# Log in to ECR
aws ecr get-login-password | docker login --username AWS --password-stdin your-account-id.dkr.ecr.region.amazonaws.com

# Tag and push the image
docker tag audiocraft-api your-account-id.dkr.ecr.region.amazonaws.com/audiocraft-api:latest
docker push your-account-id.dkr.ecr.region.amazonaws.com/audiocraft-api:latest
```

2. Create an ECS task definition:

```json
{
  "family": "audiocraft-api",
  "networkMode": "awsvpc",
  "executionRoleArn": "arn:aws:iam::your-account-id:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "audiocraft-api",
      "image": "your-account-id.dkr.ecr.region.amazonaws.com/audiocraft-api:latest",
      "essential": true,
      "portMappings": [
        {
          "containerPort": 5000,
          "hostPort": 5000,
          "protocol": "tcp"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/audiocraft-api",
          "awslogs-region": "region",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "memory": 8192,
      "cpu": 2048
    }
  ],
  "requiresCompatibilities": [
    "FARGATE"
  ],
  "cpu": "2048",
  "memory": "8192"
}
```

3. Create a service in your ECS cluster to run the task.

### Google Cloud Run

1. Build and push your Docker image to Google Container Registry:

```bash
# Build the image
docker build -t gcr.io/your-project-id/audiocraft-api .

# Push to GCR
docker push gcr.io/your-project-id/audiocraft-api
```

2. Deploy to Cloud Run:

```bash
gcloud run deploy audiocraft-api \
  --image gcr.io/your-project-id/audiocraft-api \
  --platform managed \
  --region us-central1 \
  --memory 8Gi \
  --cpu 4 \
  --port 5000
```

### Kubernetes Deployment

1. Create a Kubernetes deployment file:

```yaml
# audiocraft-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: audiocraft-api
spec:
  replicas: 2
  selector:
    matchLabels:
      app: audiocraft-api
  template:
    metadata:
      labels:
        app: audiocraft-api
    spec:
      containers:
      - name: audiocraft-api
        image: your-registry/audiocraft-api:latest
        ports:
        - containerPort: 5000
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        volumeMounts:
        - name: model-cache
          mountPath: /root/.cache/torch/hub
      volumes:
      - name: model-cache
        persistentVolumeClaim:
          claimName: model-cache-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: audiocraft-api
spec:
  selector:
    app: audiocraft-api
  ports:
  - port: 80
    targetPort: 5000
  type: LoadBalancer
```

2. Apply the deployment:

```bash
kubectl apply -f audiocraft-deployment.yaml
```

## Production Considerations

### Health Checks

Add a health check endpoint to your API:

```python
@app.route("/health", methods=["GET"])
def health_check():
    """API health check endpoint."""
    return jsonify({
        "status": "healthy",
        "version": "1.0.0",
        "models_loaded": list(models.keys())
    })
```

Configure health checks in your Docker Compose file:

```yaml
services:
  api:
    # ... other config
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

### Resource Allocation

Properly allocate resources based on model size:

| Model Size | Minimum Memory | Recommended CPU | GPU Memory |
|------------|---------------|----------------|------------|
| small      | 4GB           | 2 cores        | 2GB        |
| medium     | 8GB           | 4 cores        | 4GB        |
| large      | 16GB          | 8 cores        | 8GB        |

### Logging and Monitoring

Implement structured logging in your application:

```python
import logging
import json

# Configure structured logging
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        if hasattr(record, 'request_id'):
            log_record['request_id'] = record.request_id
        if record.exc_info:
            log_record['exception'] = self.formatException(record.exc_info)
        return json.dumps(log_record)

# Set up logger
logger = logging.getLogger("audiocraft-api")
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Usage
logger.info("Generated audio", extra={
    "request_id": request_id,
    "prompt": prompt,
    "model_size": model_size,
    "generation_time": generation_time
})
```

Direct logs to stdout/stderr in your container:

```dockerfile
# In your Dockerfile
ENV PYTHONUNBUFFERED=1
```

## Example: Complete Docker Compose Setup with Monitoring

```yaml
version: '3.8'

services:
  # API server
  api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "5000:5000"
    environment:
      - REDIS_URL=redis://redis:6379/0
      - LOG_LEVEL=INFO
    volumes:
      - ./generated_audio:/app/generated_audio
    depends_on:
      - redis
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "3"

  # Celery worker for background processing
  worker:
    build:
      context: .
      dockerfile: Dockerfile.worker
    environment:
      - REDIS_URL=redis://redis:6379/0
      - LOG_LEVEL=INFO
    volumes:
      - ./generated_audio:/app/generated_audio
      - model-cache:/root/.cache/torch/hub
    depends_on:
      - redis
    command: celery -A tasks worker --loglevel=info
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "3"

  # Redis for task queue and caching
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    restart: unless-stopped

  # Prometheus for metrics collection
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    restart: unless-stopped

  # Grafana for visualization
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana-data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=secret
    restart: unless-stopped

volumes:
  redis-data:
  model-cache:
  prometheus-data:
  grafana-data:
```

Create a `prometheus.yml` configuration file:

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'audiocraft-api'
    static_configs:
      - targets: ['api:5000']
```

## Conclusion

This guide has covered the essentials of containerizing AudioCraft applications using Docker. By following these practices, you can create portable, scalable, and maintainable deployments for your audio generation services.

Remember that ML model serving has unique requirements compared to traditional web applications:

1. **Resource Intensive**: Properly allocate memory and CPU/GPU resources
2. **Cold Starts**: Consider model preloading strategies
3. **Scaling**: ML workloads often benefit from vertical scaling (larger machines)
4. **Persistence**: Use volumes for model caching and generated content

For additional optimization techniques, see the [Performance Tuning](performance_tuning.md) guide.

## Next Steps

- Implement [API Security](api_security.md) for your containerized application
- Learn how to set up [Monitoring and Logging](monitoring_logging.md) for production
- Explore [Scaling Strategies](scaling_strategies.md) for high-traffic deployments
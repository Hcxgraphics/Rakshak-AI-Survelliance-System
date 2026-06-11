# Use a slim Python 3.10 base image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Install system packages required by OpenCV and video processing libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to optimize layer caching
COPY requirements.txt /app/requirements.txt

# Install dependencies (Standard pip versions of PyTorch and TensorFlow support CPU/CUDA detection)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source directories and weight assets into container
COPY src/ /app/src/
COPY backend/ /app/backend/
COPY models/ /app/models/
RUN mkdir -p /app/logs /app/evidence

# Set python path environment variable
ENV PYTHONPATH=/app:/app/src:/app/src/deployment
ENV PORT=8000

# Expose FastAPI backend port
EXPOSE 8000

# Command to boot the FastAPI server using the centralized backend entrypoint
CMD ["python", "backend/main.py"]

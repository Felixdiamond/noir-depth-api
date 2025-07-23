# Dockerfile for Project Noir Depth Estimation Service
# DepthAnything V2 with GPU support for Google Cloud Run

FROM python:3.11-slim

# Set environment variables for headless operation
ENV DISPLAY=:99
ENV QT_QPA_PLATFORM=offscreen
ENV OPENCV_IO_ENABLE_OPENEXR=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies for PyTorch and OpenCV
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgoogle-perftools4 \
    curl \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libglib2.0-0 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libatlas-base-dev \
    python3-dev \
    libfontconfig1-dev \
    libcairo2-dev \
    libgdk-pixbuf2.0-dev \
    libpango1.0-dev \
    libgtk2.0-dev \
    git \
    && rm -rf /var/lib/apt/lists/*


# Clone Depth Anything V2 repo (official, not pip-installable)
RUN git clone https://github.com/DepthAnything/Depth-Anything-V2.git /app/Depth-Anything-V2

# Install Depth Anything V2 requirements
RUN pip install --no-cache-dir -r /app/Depth-Anything-V2/requirements.txt

# Copy your service requirements and install 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download model weights (vits example)
RUN mkdir -p /app/checkpoints && \
    curl -L -o /app/checkpoints/depth_anything_v2_vits.pth \
    https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true

# Copy application code
COPY main.py .
COPY test_opencv.py .

# Test OpenCV installation
RUN python test_opencv.py

# Create models directory (if you want to use /app/models for other models)
RUN mkdir -p /app/models

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
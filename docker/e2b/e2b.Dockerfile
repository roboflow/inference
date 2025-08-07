# E2B Sandbox Template for Inference Custom Python Blocks
# This Dockerfile creates a sandboxed environment for executing user-provided
# Python code in Workflows Custom Python Blocks

FROM e2bdev/code-interpreter:latest

# Install system dependencies required by inference
USER root
RUN apt-get update && apt-get install -y \
    libxext6 \
    libopencv-dev \
    libgdal-dev \
    cmake \
    rustc \
    cargo \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files from inference
WORKDIR /tmp/requirements
COPY requirements/_requirements.txt ./
COPY requirements/requirements.cpu.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir \
    -r _requirements.txt \
    -r requirements.cpu.txt \
    && rm -rf ~/.cache/pip

# Install inference package
WORKDIR /app
COPY inference inference
COPY inference_sdk inference_sdk

# Create startup script that pre-imports inference modules
RUN mkdir -p /root/.e2b
COPY docker/e2b/startup.py /root/.e2b/startup.py

# Set environment variables for E2B
ENV PYTHONPATH=/app:$PYTHONPATH
ENV E2B_SANDBOX_TYPE=custom_python_block

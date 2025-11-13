# Jetson 6.2.0 Base Image Comparison

## Purpose
Compare `l4t-jetpack` (full JetPack stack) vs `l4t-cuda` (minimal CUDA runtime) as base images for inference server.

## Base Images

### Current: l4t-jetpack:r36.4.0
- **Includes**: Full JetPack SDK, CUDA, cuDNN, TensorRT, VPI, multimedia APIs, GStreamer
- **Pros**: Everything pre-installed and tested by NVIDIA
- **Cons**:
  - Large base image
  - Pre-installed package conflicts (GDAL 3.4.1, outdated PyTorch)
  - Fight against existing packages
  - Less control over versions

### Prototype: l4t-cuda:12.6.11-runtime
- **Includes**: CUDA 12.6.11 runtime + L4T hardware acceleration libs
- **Pros**:
  - Smaller base image
  - No pre-installed package conflicts
  - Full control over all dependencies
  - Cleaner dependency management
- **Cons**:
  - Need to install/compile more ourselves
  - Potentially more maintenance

## Software Stack

| Component | l4t-jetpack | l4t-cuda (prototype) |
|-----------|-------------|---------------------|
| Base | JetPack r36.4.0 | l4t-cuda:12.6.11-runtime |
| CUDA | 12.2 (from JetPack) | 12.6.11 |
| cuDNN | 8.9 (from JetPack) | Via PyTorch wheels |
| TensorRT | 8.6 (from JetPack) | Via PyTorch wheels |
| PyTorch | 2.8.0 (jetson-ai-lab.io) | 2.8.0 (jetson-ai-lab.io) |
| GDAL | 3.11.5 (compiled) | 3.11.5 (compiled) |

## Build Instructions

### Build l4t-jetpack version (current)
```bash
cd /Users/anorell/roboflow/inference
docker build -f docker/dockerfiles/Dockerfile.onnx.jetson.6.2.0 \
  -t roboflow-inference-jetson-620-jetpack:test \
  --platform linux/arm64 .
```

### Build l4t-cuda version (prototype)
```bash
cd /Users/anorell/roboflow/inference
docker build -f docker/dockerfiles/Dockerfile.onnx.jetson.6.2.0.cuda-base \
  -t roboflow-inference-jetson-620-cuda:test \
  --platform linux/arm64 .
```

## Comparison Script

```bash
#!/bin/bash

echo "========================================="
echo "Jetson 6.2.0 Base Image Comparison"
echo "========================================="
echo ""

# JetPack version
if docker image inspect roboflow-inference-jetson-620-jetpack:test >/dev/null 2>&1; then
    jetpack_size=$(docker image inspect roboflow-inference-jetson-620-jetpack:test --format='{{.Size}}')
    jetpack_size_gb=$(echo "scale=2; $jetpack_size / 1024 / 1024 / 1024" | bc)
    echo "l4t-jetpack version:"
    echo "  Size: ${jetpack_size_gb} GB"
    docker image inspect roboflow-inference-jetson-620-jetpack:test --format='  Layers: {{len .RootFS.Layers}}'
else
    echo "l4t-jetpack version: NOT BUILT"
fi

echo ""

# CUDA version
if docker image inspect roboflow-inference-jetson-620-cuda:test >/dev/null 2>&1; then
    cuda_size=$(docker image inspect roboflow-inference-jetson-620-cuda:test --format='{{.Size}}')
    cuda_size_gb=$(echo "scale=2; $cuda_size / 1024 / 1024 / 1024" | bc)
    echo "l4t-cuda version:"
    echo "  Size: ${cuda_size_gb} GB"
    docker image inspect roboflow-inference-jetson-620-cuda:test --format='  Layers: {{len .RootFS.Layers}}'
else
    echo "l4t-cuda version: NOT BUILT"
fi

echo ""

if [ -n "$jetpack_size" ] && [ -n "$cuda_size" ]; then
    diff_bytes=$((jetpack_size - cuda_size))
    diff_gb=$(echo "scale=2; $diff_bytes / 1024 / 1024 / 1024" | bc)
    percent=$(echo "scale=1; ($diff_bytes * 100) / $jetpack_size" | bc)

    if [ $diff_bytes -gt 0 ]; then
        echo "Difference: l4t-cuda is ${diff_gb} GB smaller (${percent}% reduction)"
    else
        diff_gb=$(echo "scale=2; -$diff_bytes / 1024 / 1024 / 1024" | bc)
        percent=$(echo "scale=1; (-$diff_bytes * 100) / $jetpack_size" | bc)
        echo "Difference: l4t-cuda is ${diff_gb} GB larger (${percent}% increase)"
    fi
fi

echo ""
echo "========================================="
```

## Results

_To be filled after building both images_

### Size Comparison
- **l4t-jetpack**: ? GB
- **l4t-cuda**: ? GB
- **Difference**: ? GB (? % reduction)

### Software Versions
_Extract from running containers:_

```bash
# Check CUDA version
docker run --rm <image> nvcc --version

# Check Python packages
docker run --rm <image> python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
docker run --rm <image> python3 -c "import torchvision; print(f'torchvision: {torchvision.__version__}')"
docker run --rm <image> gdal-config --version

# Check TensorRT (if available)
docker run --rm <image> python3 -c "import tensorrt; print(f'TensorRT: {tensorrt.__version__}')" 2>/dev/null || echo "TensorRT: Not available via Python"
```

## Recommendations

_To be filled after analysis_

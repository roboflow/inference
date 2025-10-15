# Jetson 6.2 Docker Build Investigation

## Problem
Attempted to build Docker image for Jetson AGX Orin (JetPack 6.2.1, L4T 36.4.4, CUDA 12.6) using `roboflow/l4t-ml:r36.4.tegra-aarch64-cu126-22.04` as base image. All builds consistently failed with:

```
ERROR: max depth exceeded
```

## Root Cause
The base image `roboflow/l4t-ml:r36.4.tegra-aarch64-cu126-22.04` has reached Docker's maximum filesystem layer depth limit. This is a hard limit in both legacy Docker builder and BuildKit. Any additional RUN, COPY, or other layer-creating operations cause the "max depth exceeded" error.

## What We Tried

### 1. Package Removal
- Removed source-build packages: `zxing-cpp`, `pyvips`, `paho-mqtt` (these create excessive layers during pip dependency resolution)
- Excluded packages with version conflicts: `bitsandbytes`, incompatible `torch`/`opencv` versions
- **Result:** Still failed - even minimal operations hit the limit

### 2. Split RUN Commands
- Split package installation into multiple RUN commands to reset layer counter
- Used `--no-deps` to avoid dependency resolution layers
- **Result:** Still failed - base image depth leaves no room

### 3. Multi-Stage Build
- Created builder stage to install packages, then copy to fresh final stage
- Used different stage names (`pkgbuilder`, `builder`) and output names to avoid conflicts
- **Result:** Still failed - copying from builder to final stage hit depth limit

### 4. Legacy Builder vs BuildKit
- Tried `DOCKER_BUILDKIT=0` (legacy builder)
- Tried default BuildKit
- **Result:** Both have the same layer depth limit

### 5. Image Flattening
Attempted to flatten the base image to reset layer depth:

#### Method 1: `docker save | docker import`
```bash
docker save roboflow/l4t-ml:r36.4.tegra-aarch64-cu126-22.04 | \
docker import - roboflow/l4t-ml:r36.4-flattened
```
**Result:** Image lost CMD/ENTRYPOINT metadata, file structure broken

#### Method 2: `docker export | docker import` (correct method)
```bash
docker run --name temp-l4t roboflow/l4t-ml:r36.4.tegra-aarch64-cu126-22.04 true
docker export temp-l4t | docker import --change 'CMD ["/bin/bash"]' - roboflow/l4t-ml:r36.4-flat2
```
**Result:** Successfully created flattened image (31.6GB from 37.9GB), BUT CUDA libraries corrupted:
```
ImportError: /usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1: file too short
```

Docker export/import doesn't handle symlinks properly, breaking NVIDIA CUDA shared libraries.

## Final Minimal Dockerfile

Simplified to absolute minimum operations to document the layer depth limitation:

```dockerfile
FROM roboflow/l4t-ml:r36.4.tegra-aarch64-cu126-22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV LANG=en_US.UTF-8

# Install minimal system dependencies
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends lshw git && \
    rm -rf /var/lib/apt/lists/*

# Set up application (minimal layers)
WORKDIR /app
COPY inference/ ./inference/
COPY inference_cli/ ./inference_cli/
COPY inference_sdk/ ./inference_sdk/
COPY docker/config/gpu_http.py ./gpu_http.py
COPY .release .release
COPY requirements requirements
COPY Makefile Makefile

# Let inference_cli wheel pull dependencies
RUN make create_inference_cli_whl PYTHON=python3 && \
    pip3 install --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 dist/inference_cli*.whl

ENV VERSION_CHECK_MODE=continuous \
    PROJECT=roboflow-platform \
    ORT_TENSORRT_FP16_ENABLE=1 \
    ORT_TENSORRT_ENGINE_CACHE_ENABLE=1 \
    # ... other env vars

EXPOSE 9001
ENTRYPOINT uvicorn gpu_http:app --workers $NUM_WORKERS --host $HOST --port $PORT
```

**Status:** This minimal Dockerfile ALSO fails at step 11 with "max depth exceeded"

## Key Findings

1. **Layer depth is cumulative** - The base image's existing layers count toward the total
2. **Both builders affected** - Legacy builder and BuildKit have identical limits
3. **Multi-stage doesn't help** - COPY from builder to final still adds layers
4. **Flattening breaks CUDA** - docker export/import corrupts symlinked NVIDIA libraries
5. **Base image is maxed out** - `roboflow/l4t-ml:r36.4.tegra-aarch64-cu126-22.04` leaves zero room for additional operations

## Solution Required

The `roboflow/l4t-ml:r36.4.tegra-aarch64-cu126-22.04` base image needs to be rebuilt with fewer layers, OR the inference application needs to install dependencies at runtime instead of build time.

The jetson-containers build system (`build.sh`) successfully builds this image, so they must use techniques to manage layer depth (possibly squashing during the build process, or using a different build strategy).

## Package Index Issues Encountered

Along the way, we also encountered:
- **pypi.jetson-ai-lab.dev timeout** - Resolved by using only `--index-url https://pypi.jetson-ai-lab.io/jp6/cu126`
- **Version conflicts** - opencv (4.12.0 vs <=4.10.0.84), torch (2.8 vs <2.7.0), bitsandbytes (only dev versions)
- **Source builds** - Packages without aarch64 wheels create excessive layers during build

## Files Modified

- `docker/dockerfiles/Dockerfile.onnx.jetson.6.2.0` - Simplified to minimal form (still fails due to base image depth)
- Created experimental Dockerfiles (not committed):
  - `Dockerfile.l4t-ml-inference-base` - Attempted enhanced base with common packages
  - `Dockerfile.jetson-inference-test` - Multi-stage build experiments
  - `Dockerfile.jetson-inference-final` - Final attempt with complete package list

## Next Steps

1. Investigate how jetson-containers `build.sh` avoids layer depth issues
2. Consider runtime package installation instead of build-time
3. OR rebuild l4t-ml base with squashing/flattening at intermediate stages
4. OR use lighter base (`nvcr.io/nvidia/l4t-jetpack:r36.4.0` - 9.83GB) and build up from there

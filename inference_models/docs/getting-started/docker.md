# 🐳 Docker Environments

Documentation for using `inference-models` in Docker containers.

!!! info "Recommended for Jetson"
    Docker is the **recommended** installation method for NVIDIA Jetson devices. See [Hardware Compatibility](hardware-compatibility.md) for details.

!!! warning "Work in Progress"
    Docker builds for `inference-models` are currently **in progress**. Some builds are automated, while others are manual. We encourage you to test these images and [raise issues](https://github.com/roboflow/inference/issues) if you encounter any problems.

## 📦 Available Docker Images

Pre-built experimental Docker images are available on Docker Hub under the `roboflow/inference-exp` repository.

### x86_64 / AMD64 Images

| Image Tag | Base | CUDA Version | Status |
|-----------|------|--------------|--------|
| `roboflow/inference-exp:cpu-latest`<br>`roboflow/inference-exp:cpu-<version>` | Ubuntu 22.04 | N/A (CPU only) | 🤖 Automated |
| `roboflow/inference-exp:cu118-latest`<br>`roboflow/inference-exp:cu118-<version>` | Ubuntu 22.04 | CUDA 11.8 | 🤖 Automated |
| `roboflow/inference-exp:cu124-latest`<br>`roboflow/inference-exp:cu124-<version>` | Ubuntu 22.04 | CUDA 12.4 | 🤖 Automated |
| `roboflow/inference-exp:cu126-latest`<br>`roboflow/inference-exp:cu126-<version>` | Ubuntu 22.04 | CUDA 12.6 | 🤖 Automated |
| `roboflow/inference-exp:cu128-latest`<br>`roboflow/inference-exp:cu128-<version>` | Ubuntu 22.04 | CUDA 12.8 | 🤖 Automated |

!!! info "Image Tags"
    - **`-latest`** tags point to the most recent release
    - **`-<version>`** tags (e.g., `cpu-0.17.3`) pin to a specific version for reproducibility

### Jetson Images

| Image Tag | JetPack Version | Status |
|-----------|----------------|--------|
| `roboflow/roboflow-inference-server-jetson-7.2.0:latest` | JetPack 7.2 (L4T 39, Orin & Thor) | ✅ Production |
| `roboflow/roboflow-inference-server-jetson-6.2.0:latest` | JetPack 6.2 (L4T 36.4) — support ends 2027 | ✅ Production |
| `roboflow/inference-exp:jp62-*` | JetPack 6.2 (L4T 36.x) — support ends 2027 | 🚧 In development |
| `roboflow/roboflow-inference-server-jetson-5.1.1:0.62.5-experimental` | JetPack 5.1 (L4T 35.2.1) — **deprecated, removed 2027** | ✋ Manual (experimental) |

!!! note "Image Status"
    - 🤖 **Automated** - Built automatically on releases and main branch pushes
    - ✋ **Manual** - Built manually; not part of automated pipeline
    - 🚧 **In development** - Coming soon

## 🚀 Quick Start

### CPU-only

```bash
docker run -it roboflow/inference-exp:cpu-latest python3
```

### GPU (CUDA 12.8)

```bash
docker run --gpus all -it roboflow/inference-exp:cu128-latest python3
```

### Jetson (JetPack 7.2 — recommended)

```bash
docker run -it \
  --runtime nvidia \
  roboflow/roboflow-inference-server-jetson-7.2.0:latest \
  python3
```

JetPack 7.2 supports both Orin and Thor. For JetPack 6.2 use
`roboflow/roboflow-inference-server-jetson-6.2.0:latest` (support ends 2027).

!!! warning "JetPack 5.1 is deprecated"
    The experimental `roboflow/roboflow-inference-server-jetson-5.1.1:0.62.5-experimental`
    build still exists, but JetPack 5.1.x is deprecated and support will be removed in 2027.
    Migrate to JetPack 7.2.

## 🔨 Building Custom Images

You can build your own Docker images using the Dockerfiles in the `inference_models/dockerfiles/` directory:

## 💡 Usage Tips

### Mounting Volumes

Mount your model cache for cache persistency:

```bash
docker run -it \
  -v ~/.cache/inference:/root/.cache/inference \
  roboflow/inference-exp:cpu-latest \
  bash
```

## 🐛 Reporting Issues

These Docker images are experimental. If you encounter issues:

1. Check the [Hardware Compatibility](hardware-compatibility.md) guide
2. Verify your Docker and NVIDIA runtime setup
3. [Open an issue](https://github.com/roboflow/inference/issues) with:
   - Image tag used
   - Error message
   - Steps to reproduce

## 🚀 Next Steps

- [Installation Guide](installation.md) - Local installation options
- [Hardware Compatibility](hardware-compatibility.md) - Platform-specific requirements
- [Understand Core Concepts](../how-to/understand-core-concepts.md) - Understand the architecture


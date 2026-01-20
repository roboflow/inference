# 🖥️ Hardware Compatibility

Platform-specific compatibility and testing status for `inference-models`.

## 🧪 Testing Coverage & Support Status

The following table shows the current testing and support status for different platforms:

| Platform | OS/Distribution | Installation Method | Support Status |
|----------|----------------|---------------------|----------------|
| **CPU (x86_64)** | Linux (general) | Bare-metal, Docker | ✅ **Stable** |
| **CPU (x86_64)** | macOS | Bare-metal, Docker | ✅ **Stable** |
| **CPU (Apple Silicon)** | macOS | Bare-metal, Docker | ✅ **Stable** |
| **NVIDIA GPU** | Ubuntu 22.04 LTS | Bare-metal, Docker | ✅ **Stable** |
| **NVIDIA GPU** | Ubuntu 24.04 LTS | Bare-metal, Docker | ✅ **Stable** |
| **NVIDIA GPU** | Other Linux distros | Bare-metal, Docker | ⚠️ **Requires verification** |
| **Jetson (JetPack 6.1)** | Ubuntu 22.04 (Jetson) | Docker | ✅ **Stable** |
| **Jetson (JetPack 6.1)** | Ubuntu 22.04 (Jetson) | Bare-metal | ⚠️ **Experimental** |
| **Jetson (JetPack 5.1)** | Ubuntu 20.04 (Jetson) | Docker (custom build) | ✅ **Stable** |
| **Jetson (JetPack 5.1)** | Ubuntu 20.04 (Jetson) | Bare-metal | ❌ **Not possible** |
| **Windows** | Windows 10/11 | Any | ❓ **Not tested** |

!!! warning "Windows Support"
    **Windows is not tested at all.** While the package may install and run on Windows, we have not performed any testing on this platform. Use at your own risk.

## 💻 CPU Support

### x86_64 / AMD64

**Supported platforms:**

- **Linux** - General distribution support
- **macOS** - Intel and Apple Silicon

**Installation methods:**

- ✅ **Bare-metal** - Direct pip/uv installation (see [Installation Guide](installation.md))
- ✅ **Docker** - Containerized deployment (see [Docker Environments](docker.md))

**What works:**

- All CPU-based models
- PyTorch CPU backend
- ONNX Runtime CPU backend

### Apple Silicon (M1/M2/M3/M4)

**Supported platforms:**

- **macOS** with Apple Silicon processors

**Installation methods:**

- ✅ **Bare-metal** - Direct pip/uv installation
- ✅ **Docker** - Containerized deployment

**MPS (Metal Performance Shaders) Support:**

- ⚠️ **Experimental** - MPS GPU acceleration available for select models only
- **Supported models**: RFDetr and other compatible architectures
- **Limitations**: Not all models support MPS; most run on CPU

```bash
# Install on Apple Silicon
pip install inference-models
```

## 🎮 NVIDIA GPU Support

### Tested Distributions

**Stable support:**

- ✅ **Ubuntu 22.04 LTS** - Fully tested and recommended
- ✅ **Ubuntu 24.04 LTS** - Fully tested and recommended

**Other distributions:**

- ⚠️ **Requires verification** - Other Linux distributions (Debian, RHEL, CentOS, etc.) should work but require testing in your specific environment

**Installation methods:**

- ✅ **Bare-metal** - Direct pip/uv installation (see [Installation Guide](installation.md))
- ✅ **Docker** - Containerized deployment (see [Docker Environments](docker.md))

**Requirements:**

- NVIDIA GPU with CUDA 11.8 or 12.x support
- Appropriate NVIDIA drivers installed
- For Docker: NVIDIA Container Toolkit

## 🤖 NVIDIA Jetson Support

### JetPack 6.1 (Recommended)

**Supported devices:**

- Jetson Orin AGX
- Jetson Orin NX
- Jetson Orin Nano

**Installation methods:**

| Method | Status | Description |
|--------|--------|-------------|
| **Docker** | ✅ **Stable** | Recommended for production use |
| **Bare-metal** | ⚠️ **Experimental** | `pip install` works but not extensively tested; use at your own risk |

**Docker installation (recommended):**

See [Docker Environments](docker.md) for pre-built Jetson images.

**Bare-metal installation (experimental):**

```bash
# Use at your own risk - not extensively tested
uv pip install "inference-models[torch-jp6-cu126,onnx-jp6-cu126]"
```

!!! warning "Bare-metal on Jetson"
    Bare-metal installation on Jetson devices is **experimental**. While `pip install` should work, it has not been extensively tested. We recommend using Docker for production deployments.

### JetPack 5.1 (Legacy)

**Supported devices:**

- Jetson Orin AGX
- Jetson Orin NX
- Jetson Orin Nano

**Installation methods:**

| Method | Status | Description |
|--------|--------|-------------|
| **Docker (custom build)** | ✅ **Stable** | Custom Docker build required; see below |
| **Bare-metal** | ❌ **Not possible** | Verified to not work due to dependency conflicts |

**Docker installation:**

JetPack 5.1 requires a custom Docker build. See the [Roboflow inference repository](https://github.com/roboflow/inference/tree/main/docker/dockerfiles) for Jetson 5.1 Dockerfiles.

!!! danger "No Bare-metal Support for JetPack 5.1"
    We have verified that bare-metal installation on JetPack 5.1 is **not possible** due to incompatible system dependencies and library conflicts. You must use the custom Docker build.

### TensorRT on Jetson

!!! warning "Use JetPack TensorRT"

    Jetson devices come with TensorRT pre-installed as part of JetPack. **Do not install the `trt10` extra** on Jetson platforms.

    - Use the system TensorRT: `/usr/lib/aarch64-linux-gnu/libnvinfer.so`
    - Installing PyPI TensorRT packages will cause conflicts

## 🚀 Next Steps

- [Installation Guide](installation.md) - Detailed installation instructions
- [Understand Core Concepts](../how-to/understand-core-concepts.md) - Understand the architecture
- [Supported Models](../models/index.md) - Browse available models


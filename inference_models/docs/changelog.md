# Changelog

## `0.18.5` and earlier versions

### Added
- Initial releases of `inference-models` library
- Support for 50+ computer vision models
- Multi-backend support (ONNX, PyTorch, TensorRT)
- AutoModel API for automatic model loading
- AutoModelPipeline for multi-model workflows
- Comprehensive model package system
- Automatic backend negotiation
- Model caching and optimization
- Support for object detection, instance segmentation, classification, OCR, keypoint detection, and more
- Vision-language models (Florence-2, PaliGemma, Qwen2.5-VL, etc.)
- Interactive segmentation (SAM, SAM2)
- Depth estimation models
- Gaze detection
- Open-vocabulary object detection
- Embeddings models (CLIP, Perception Encoder)

### Documentation
- Complete API reference documentation
- Getting started guides
- Model-specific documentation for all supported models
- How-to guides for common tasks
- Contributors guide
- Error reference documentation

### Backends
- ONNX Runtime support (CPU and GPU)
- PyTorch support (CPU, CUDA, MPS)
- TensorRT support for NVIDIA GPUs
- Automatic backend selection based on hardware

### Features
- Automatic model package negotiation
- Multi-device support (CPU, CUDA, MPS)
- Batch processing support
- Quantization support (FP32, FP16, INT8)
- Model dependency resolution
- Custom weights provider support
- Local model loading
- Docker support with pre-built images

---
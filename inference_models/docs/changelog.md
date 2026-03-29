# Changelog

## `0.24.0`

### Added

- Support for Roboflow License Server proxy in Roboflow weights provider 

---

## `0.23.0`

### Added

- Support for CUDA 13.0 on x86 architecture - as a result of `torch 2.11` release which makes CUDA 13.0 default version

---

## `0.22.1`

### Added

- Ability to restrict maximum input resolution for models

- Restriction of input resolution for RF-DETR - providing ability for caller to avoid OOM when loading models 
with large input resolutions

- New type of error `ModelPackageRestrictedError` - to manifest restrictions of runtime environment with package

---

## `0.22.0`

### Added

- GLM-OCR model added to models zoo 

---

## `0.21.1`

### Fixed

- Lack of model package features denoted in auto-negotiation cache entries was causing errors while re-initialization 
of models which had `required_features` denoted in model registry.

---

## `0.21.0`
### Added

- Support for CUDA Graphs in TRT backend - all TRT models got upgraded - added ability to run with CUDA graphs, at 
the expense of additional VRAM allocation, but with caller control on how many execution contexts for different 
input shapes should be allowed.

---

## `0.20.2`
### Added

- Ability to override certain aspects of model pre-processing (like center-crop, contrast enhancement or grayscale 
which may be performed by caller).  

---


## `0.20.1`
### Fixed

- `AnyModel` typing regarding semantic segmentation model

---

## `0.20.0`
### Added

- Support for `transformers>=5`

- Model registry feature allowing to treat specific model features as required during auto-negotiation  

---

## `0.19.4`
### Fixed

- CUDA stream synchronization issues in TRT models.

---

## `0.19.3`

### Fixed
- Post-processing for RF-DETR segmentation model - missing remapping for class ids regarding masks.

---

## `0.19.2`

### Fixed
- Changed the default ranking for model packages in `AutoLoader` - ONNX to be preferred over Torch. 

---

## `0.19.1`

### Fixed
- Fixed issue with RF-DETR model post-processing causing all results to be empty (TRT implementation) 

---

## `0.19.0`

First **stable release** of `inference-models` library.

### Added
- Locks for thread safety of torch models

### Maintenance
- Established documentation hosting
- Provided documentation links to error messages
- Fixed bugs spotted during tests

---

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
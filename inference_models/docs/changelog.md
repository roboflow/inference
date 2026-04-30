# Changelog

## `0.27.3`

### Fixed

- RFDetr pre- and post-processing aligned with training transforms: pre-processing replaced with a dedicated `PIL → F.resize → F.to_tensor → F.normalize` chain (always stretches to `training_input_size`); post-processing uses topk-flat across (queries × classes) via shared `select_topk_predictions`; fixes a cross-backend divergence at low confidence thresholds.

## `0.27.2`

### Fixed

- Temporarily disabled flash-attention in GLM-OCR for Jetsons, due to incompatibility detected
before release.


## `0.27.1`

### Added

- Improved logging for auto-negotiation of model packages.

## `0.27.0`

### Added

- COCO RLE masks format for all instance segmentation predictions and all models 
supported in the library were patched. `InstanceDetections` mask can now be `InstancesRLEMasks` object
which follows the structure of `pycocotools` masks (providing memory-efficient alternative for dense
representation). Clients who want to use the format, should pass `mask_format="rle"` to `**kwargs` of model
forward pass.

### Changed

- The change with RLE masks format yielded change to base interface of Instance Segmentation models - 
**new abstract property `supported_mask_formats` was added, which is a breaking change for local-code 
instance-segmentation models.** We are not aware of anyone using the library in such mode currently, due 
to the maturity of the library, so we are introducing this change, such that the interface does not 
implicitly enforce supported format.

- Representation of `InstanceDetections` changed - new `InstancesRLEMasks` format is now an alternative for 
`torch.Tensor` used for `dense` mask representation. This change is considered **non-breaking**, as alternative 
representation must be requested by the caller.

---
## `0.26.1`

### Changed

- For Roboflow weights provider, Roboflow License Server proxy transitioned into 
Roboflow Secure Gateway, altering naming conventions of all helper functions which are 
considered private interface of weights provider (hence should not be considered breaking 
for any clients). Along with this change, `LICENSE_SERVER` environmental variable controlling 
the proxy address was replaced to be `SECURE_GATEWAY` - old variable will be deleted in the
release following after the end of Q3 2026.

---

## `0.26.0` 

### Added

- Bringing back changes to filtering proposed in retracted release `0.25.0` 
along with fixes for bugs which caused retraction.

---

## `0.25.2`

### Fixed

- OWLv2 compilation procedure clash with `transformers~=5.5` brought to dependencies along with `0.25.1` release and
Gemma 4.

---

## `0.25.1`

### Added

- Documentation for Gemma 4 multimodal models (`Gemma4HF` / `gemma4_hf.py`): dedicated [model page](models/gemma4.md),
  catalog and site navigation updates, home page pointer, and [environment variables](how-to/environment-variables.md#gemma-4)
  for `INFERENCE_MODELS_GEMMA4_*` defaults.

---

## `0.25.0` **(retracted)**

### Added

- `post_process(...)` on object detection, instance segmentation, keypoint detection, classification, and semantic 
segmentation models now accepts `confidence` as `"best"` (use per-class or global thresholds from 
`RecommendedParameters` when available), `"default"` (model's built-in default), or a float override. Shared NMS 
helpers accept a per-class `torch.Tensor` for single-pass per-class filtering.

---

## `0.24.4`

### Changed

- Behavior of Roboflow weights provider was changed - instead of throwing error each time **any** known model 
package is fetched with manifest not passing validation - it warns about this fact and skips the package.
This change is dictated by potential negative impact on stability which malformed manifests could have, in the face 
of broader change on Roboflow platform making it possible tp externally register packages - sanitization and 
validation is enabled on registry API side, but we introduce defensive change here to prevent potential instability.

### Added

- RF-DETR NAS capabilities for Instance Segmentation

## `0.24.3`

### Changed

- Added `sigmoid` smoothing for instance-segmentation masks in YOLOv8, YOLOv11, YOLOv12 models family.
Smoothing can be enabled / disabled via `masks_smoothing_enabled` parameter of `post_process(...)` method
(which can be passed as `**kwarg` to `forward(...)`) with default set with 
`INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_MASKS_SMOOTHING_ENABLED` (set to `True`). Additionally, the binarization 
threshold for masks can be controlled via `masks_binarization_threshold` parameter - default to be 
controlled with `INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_MASKS_BINARIZATION_THRESHOLD` (set to `0.5` or `0.0` 
depending on `INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_MASKS_SMOOTHING_ENABLED`).

!!! warning "Instance-segmentation masks will change"

    Due to smoothing, there is slight change to segmentation masks expected - mainly regarding edges 
    of predictions which should be smoother now. Change is dictated by alignment to old `inference` versions
    behaviour, effectively drifting from `ultralytics` post-processing.

---

## `0.24.2`

### Fixed

- Issue with `INFERENCE_HOME` derived paths issues when running on Windows (lack `/tmp/cache` dereference to 
Windows path). 

---

## `0.24.1`

### Changed

- Added optional field `alternatives_errors` to `ModelPackageAlternativesExhaustedError`, making it possible 
to report to the caller what types of errors happened during the load - making it possible to deduce if 
problem with loading is recoverable.

---

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

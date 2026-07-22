# Changelog

## Unreleased

Add user-facing changes below using `### Added`, `### Changed`, `### Fixed`, or
`### Removed` subsections as appropriate.

---

## `0.32.2`

### Fixed
- Bump of transitive dependency `gitpython`

---


## `0.32.2`

### Fixed
- Patch `triton-fused-v1` post-processor to use correctly current device alias for comparison.

---

## `0.32.1`

### Fixed
- Patch for security issues 

---

## `0.32.0`

### Changed

- RF-DETR TensorRT object detection now selects `triton-universal-v1`
  preprocessing and `triton-fused-v1` postprocessing by default. Incompatible requests
  use the declared `base` implementation unless strict selection is requested through
  an explicit execution plan. The selected implementations can be controlled with an
  `RFDetrExecutionPlan` or the `INFERENCE_MODELS_RFDETR_PREPROCESSOR` and
  `INFERENCE_MODELS_RFDETR_POSTPROCESSOR` environment variables. No-op preprocessing
  override containers used by the inference server remain on the optimized path;
  active overrides use the declared fallback. Repeated occurrences of the same
  request-level fallback warning are logged only once per model instance.
- Direct RF-DETR TensorRT stage calls remain backward compatible: public
  `pre_process()` synchronizes before returning by default, so its output is ready for
  an independent `forward()` call. Composed `model(...)` and `infer()` calls explicitly
  use the asynchronous exact-tensor readiness handoff to avoid a host synchronization.
  The inference-server object-detection adapter also enables this handoff for models
  that explicitly declare the invocation-level preprocessing parameter.

### Fixed

- SAM3 concept-segmentation postprocessing no longer scales its memory working set with
  detection count × image resolution. `ChunkedPostProcessImage` applies the detection cap
  before mask interpolation and upscales/encodes masks in fixed-size slices
  (`INFERENCE_MODELS_SAM3_MASK_PROCESSING_CHUNK_SIZE`, default 8), eliminating a measured
  +14 GiB host-RAM transient (GPU-OOM CPU fallback) and reducing CUDA peak ~2.8x on
  many-instance images. Outputs are bit-identical to the previous implementation.

### Added

- Composable RF-DETR TensorRT execution plans, implementation contracts and registries,
  compatibility-aware implementation selection, and runtime metadata reporting the
  requested and effective preprocessing and postprocessing implementations.
- NVIDIA Cosmos 3 Edge reasoner (`cosmos-3-edge`, task `vlm`, backend `hugging-face`):
  image/video + text prompting via `prompt(...)` / `prompt_video(...)`, following the
  standard VLM contract. The generative world-model tower ships separately.
- NVIDIA Cosmos 3 Edge generator (`cosmos-3-edge-world`, task `world-model`, backend
  `custom`): image-to-video (`generate_video`), forward dynamics (`start_rollout` +
  `forward_dynamics` with explicit session-state threading), and inverse dynamics
  (`inverse_dynamics`). The step-wise robot policy mode is deferred. The denoising
  runtime ships inside the model package (loaded via `import_class_from_file`), keeping
  NVIDIA's cosmos stack out of `inference_models` dependencies.
- `segment_with_text_prompts` accepts `max_detections` (top-k by score, applied before mask
  interpolation; default `-1` = uncapped) and `mask_format` (`"dense"` default, or `"rle"`
  for COCO RLE at original resolution).

---

## `0.31.0`

### Fixed

- Synchronisation of pre-processing and forward-pass for models running with `onnxruntime` backend.
  Pre-processed input tensors could be consumed by the ONNX session before the CUDA stream that
  produced them finished writing, yielding phantom predictions (in particular under
  `TensorrtExecutionProvider`, where onnxruntime's own input synchronisation is a no-op). Forward
  pass now explicitly synchronises with pre-processing on the torch side. Additionally, CUDA
  streams are shared per `(thread, device, purpose)` instead of being created per model instance,
  which bounds the GPU memory segregated by the torch caching allocator across streams.

### Added

- `align_device_with_onnx_session(...)` exposed in developer tools (public dev API) - makes sure
  the `torch.device` declared for a model is in line with what the `onnxruntime` session can
  actually consume (avoiding runtime errors), with `resolution_mode` (`"fallback"` / `"fail"`)
  and optional `fallback_device` parameters. For now only CUDA primary devices are verified.

---

## `0.30.1`

### Fixed

- PP-OCRv6 pipeline assembles `text` by joining fragments detected on the same
  visual line with spaces; newlines now separate only distinct lines. Previously
  every detected fragment was joined with a newline, splitting single sentences
  the detector returned as multiple boxes.

---

## `0.30.0`

### Added

- Support for [PP-OCRv6](https://github.com/PaddlePaddle/PaddleOCR),
  PaddlePaddle's ultra-lightweight OCR system: text detection
  (`pp-ocrv6-det`) and text recognition (`pp-ocrv6-rec`) models, plus the
  `pp-ocrv6` pipeline chaining both stages into end-to-end OCR. See the
  [model documentation](models/pp-ocrv6.md) for details.

---

## `0.29.7`

### Added

- Enriched `KeyPoints` representation to expose `covariance` and 
`detection_confidence` to streamline changes in `supervision`

- Align changes in RF-DETR model to expose pixel-space `covariance`, 
following up on https://github.com/roboflow/rf-detr/releases/tag/1.8.0.

---

## `0.29.6`

### Added

- Opt-in Triton RF-DETR instance-segmentation RLE post-processing. Set
  `INFERENCE_MODELS_RFDETR_TRITON_POSTPROC_ENABLED=True` to generate COCO RLE
  masks directly from sparse interpolated mask regions on supported CUDA
  inputs.
- Opt-in Triton RF-DETR instance-segmentation preprocessing for the TensorRT
  backend. Set `INFERENCE_MODELS_RFDETR_TRITON_PREPROC_ENABLED=True` to run the
  supported resize and normalize path on CUDA.
- Opt-in Triton RF-DETR instance-segmentation pipelining. Set
  `RFDETR_PIPELINE_DEPTH=2`.

---

## `0.29.4`

### Fixed

- Security issues patch, 19.06.2026 - `bleach>=6.4.0` and `tornado>=6.5.7` in `docs` extras.

---

## `0.29.4`

### Fixed

- Fixed GLM-OCR dtype mismatch on Jetson by casting HuggingFace processor floating-point
inputs to the model dtype resolved for the target device (bfloat16 on supported CUDA hardware,
otherwise float16).

---

## `0.29.3`

### Fixed

- Incompatibility with `supervision==0.29.0` due to init param in `sv.KeyPoints(...)`

---

## `0.29.2`

### Fixed

- Transitive dependency vulnerability patched - `idna>=3.15` required by the package

---
## `0.29.1`

### Fixed

- SAM3 point-prompting feature

---
## `0.29.0`

### Added

- Added RF-DETR preview keypoint support (ONNX backend).
- Added support for fine-tuned YOLO26 semantic segmentation models.

---

## `0.28.7`

### Added
- Added YOLO26 semantic segmentation support (ONNX, TorchScript, and TensorRT backends).

---

## `0.28.6`

### Fixed

- torch.jit.load/script share a process-global which is not thread-safe, introduced lock to prevent race conditions when loading SAM3 and other torchscript models
- `0.28.5` yanked

---

## `0.28.4`

### Added
- Ported SAM3 to inference_models

### Fixed

- There were issues with dependencies while introducing SAM3 hence versions `0.28.2` and `0.28.3`

---

## `0.28.1`

### Fixed

- Detections at image edges are now clipped to the image dimensions.

---

## `0.28.0`

### Removed (BREAKING)

- **MediaPipe is no longer supported.** The `mediapipe` extra and every
  symbol coupled to it have been removed. Consumers comparing against
  `BackendType.MEDIAPIPE` will hit `AttributeError`. Roboflow Universe
  payloads of type `mediapipe-model-package-v1` are now silently filtered
  by `MODEL_PACKAGE_PARSERS.get(...)`. Removed symbols:
  - `inference_models.models.mediapipe_face_detection.MediaPipeFaceDetector`
  - `inference_models.model_pipelines.face_and_gaze_detection.FaceAndGazeDetectionMPAndL2CS`
  - `BackendType.MEDIAPIPE`
  - `mediapipe_package_matches_runtime_environment` and its entry in
    `MODEL_TO_RUNTIME_COMPATIBILITY_MATCHERS`
  - Models registry entry for
    `("mediapipe-face-detector", KEYPOINT_DETECTION_TASK, BackendType.MEDIAPIPE)`
  - `BACKEND_PRIORITY[BackendType.MEDIAPIPE]`
  - Pipelines registry's `face-and-gaze-detection` entry +
    `mediapipe/face-detector` default parameter
  - `MediapipeModelPackageV1`, `parse_mediapipe_model_package`, and the
    `"mediapipe-model-package-v1"` entry in `MODEL_PACKAGE_PARSERS`
  - `RuntimeXRayResult.mediapipe_available` and `is_mediapipe_available()`
  - `INFERENCE_MODELS_MEDIAPIPE_FACE_DETECTOR_DEFAULT_CONFIDENCE`
  - The `[project.optional-dependencies] mediapipe` extra in
    `pyproject.toml`

  The standalone `L2CSNetOnnx` (under `inference_models.models.l2cs`) is
  unaffected and remains supported.

### Fixed

- RFDetr pre- and post-processing aligned with training transforms. Pre-processing replaced with a dedicated `PIL → F.resize → F.to_tensor → F.normalize` chain matching the training pipeline. For model packages with non-stretch `dataset_version_resize_dimensions`, the dataset-version resize (cv2 letterbox / center-crop) runs first, then the PIL stretch to `training_input_size`. Post-processing uses topk-flat across (queries × classes) via shared `select_topk_predictions`. Fixes a cross-backend divergence at low confidence thresholds.
- Fixed a bug where 'best' and 'default' confidence modes were not correctly handled by `RoboflowInstantHF` models.

---

## `0.27.2`

### Fixed

- Temporarily disabled flash-attention in GLM-OCR for Jetsons, due to incompatibility detected
before release.

---

## `0.27.1`

### Added

- Improved logging for auto-negotiation of model packages.

---

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

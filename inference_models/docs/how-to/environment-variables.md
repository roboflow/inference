# Environment Variables Configuration

This guide covers all environment variables available in `inference-models` for configuring model loading, caching, API access, and runtime behavior.

## Quick Start

Set environment variables before importing `inference-models`:

```bash
# Set API key
export ROBOFLOW_API_KEY="your_api_key_here"

# Set model cache directory
export INFERENCE_HOME="/path/to/cache"

# Set device
export DEFAULT_DEVICE="cuda:0"
```

Or use a `.env` file in your project root:

```bash
# .env file
ROBOFLOW_API_KEY=your_api_key_here
MODEL_CACHE_DIR=/path/to/cache
DEFAULT_DEVICE=cuda:0
```

## Core Configuration

### API Authentication

**`ROBOFLOW_API_KEY`** (or `API_KEY`)  
Your Roboflow API key for accessing models.

```bash
export ROBOFLOW_API_KEY="your_api_key_here"
```

Get your API key from: https://docs.roboflow.com/api-reference/authentication

**`ROBOFLOW_ENVIRONMENT`**  
Environment to use: `prod` (default) or `staging`.

```bash
export ROBOFLOW_ENVIRONMENT="prod"
```

**`ROBOFLOW_API_HOST`**  
Override API host URL (auto-set based on environment).

```bash
export ROBOFLOW_API_HOST="https://api.roboflow.com"
```

### Model Cache

**`INFERENCE_HOME`**  
Directory where downloaded models are cached. Default: `/tmp/cache`

```bash
export INFERENCE_HOME="/home/user/.cache/inference-models"
```

### Device Selection

**`DEFAULT_DEVICE`**  
Default device for model inference: `cpu`, `cuda`, `cuda:0`, etc.

```bash
export DEFAULT_DEVICE="cuda:0"  # Use first GPU
export DEFAULT_DEVICE="cpu"     # Use CPU
```

## API Configuration

### Request Settings

**`API_CALLS_TIMEOUT`**  
Timeout for API calls in seconds. Default: `5`

```bash
export API_CALLS_TIMEOUT="10"
```

**`API_CALLS_MAX_TRIES`**  
Maximum retry attempts for API calls. Default: `3`

```bash
export API_CALLS_MAX_TRIES="5"
```

**`IDEMPOTENT_API_REQUEST_CODES_TO_RETRY`**  
HTTP status codes to retry (comma-separated). Default: `408,429,502,503,504`

```bash
export IDEMPOTENT_API_REQUEST_CODES_TO_RETRY="408,429,500,502,503,504"
```

## Backend Configuration

### ONNX Runtime

**`ONNXRUNTIME_EXECUTION_PROVIDERS`**
Override ONNX execution providers, comma separated, no spaces.
Default: `CUDAExecutionProvider,OpenVINOExecutionProvider,CoreMLExecutionProvider,CPUExecutionProvider`

```bash
export ONNX_EXECUTION_PROVIDERS="CPUExecutionProvider"
```

## Prediction Parameter Defaults

These environment variables control the default values for prediction parameters across all models. Individual models may override these defaults. See [Prediction Parameters](prediction-parameters.md) for detailed information about each parameter.

### General Detection Parameters

**`INFERENCE_MODELS_DEFAULT_CONFIDENCE`**
Default confidence threshold for filtering predictions. Default: `0.4`

```bash
export INFERENCE_MODELS_DEFAULT_CONFIDENCE="0.5"
```

**`INFERENCE_MODELS_DEFAULT_IOU_THRESHOLD`**
Default IoU threshold for Non-Maximum Suppression. Default: `0.3`

```bash
export INFERENCE_MODELS_DEFAULT_IOU_THRESHOLD="0.45"
```

**`INFERENCE_MODELS_DEFAULT_MAX_DETECTIONS`**
Default maximum number of detections to return. Default: `300`

```bash
export INFERENCE_MODELS_DEFAULT_MAX_DETECTIONS="100"
```

**`INFERENCE_MODELS_DEFAULT_CLASS_AGNOSTIC_NMS`**
Default for class-agnostic NMS. Default: `false`

```bash
export INFERENCE_MODELS_DEFAULT_CLASS_AGNOSTIC_NMS="true"
```

### General Vision-Language Model Parameters

**`INFERENCE_MODELS_DEFAULT_MAX_NEW_TOKENS`**
Default maximum number of tokens to generate. Default: `4096`

```bash
export INFERENCE_MODELS_DEFAULT_MAX_NEW_TOKENS="1000"
```

**`INFERENCE_MODELS_DEFAULT_NUM_BEAMS`**
Default number of beams for beam search. Default: `3`

```bash
export INFERENCE_MODELS_DEFAULT_NUM_BEAMS="5"
```

**`INFERENCE_MODELS_DEFAULT_DO_SAMPLE`**
Default for sampling during generation. Default: `false`

```bash
export INFERENCE_MODELS_DEFAULT_DO_SAMPLE="true"
```

**`INFERENCE_MODELS_DEFAULT_SKIP_SPECIAL_TOKENS`**
Default for skipping special tokens in output. Default: `false`

```bash
export INFERENCE_MODELS_DEFAULT_SKIP_SPECIAL_TOKENS="true"
```

### Model-Specific Overrides

Individual models can override the general defaults. Below are model-specific environment variables:

#### DeepLabV3+

**`INFERENCE_MODELS_DEEP_LAB_V3_PLUS_DEFAULT_CONFIDENCE`**
Default: `0.5`

```bash
export INFERENCE_MODELS_DEEP_LAB_V3_PLUS_DEFAULT_CONFIDENCE="0.6"
```

#### DINOv3

**`INFERENCE_MODELS_DINOV3_DEFAULT_CONFIDENCE`**
Default: `0.5`

```bash
export INFERENCE_MODELS_DINOV3_DEFAULT_CONFIDENCE="0.6"
```

#### EasyOCR

**`INFERENCE_MODELS_EASYOCR_DEFAULT_CONFIDENCE`**
Default: `0.3`

```bash
export INFERENCE_MODELS_EASYOCR_DEFAULT_CONFIDENCE="0.4"
```

#### Florence-2

**`INFERENCE_MODELS_FLORENCE2_DEFAULT_MAX_NEW_TOKENS`**
Default: Inherits from `INFERENCE_MODELS_DEFAULT_MAX_NEW_TOKENS`

```bash
export INFERENCE_MODELS_FLORENCE2_DEFAULT_MAX_NEW_TOKENS="2048"
```

**`INFERENCE_MODELS_FLORENCE2_DEFAULT_NUM_BEAMS`**
Default: Inherits from `INFERENCE_MODELS_DEFAULT_NUM_BEAMS`

```bash
export INFERENCE_MODELS_FLORENCE2_DEFAULT_NUM_BEAMS="5"
```

**`INFERENCE_MODELS_FLORENCE2_DEFAULT_DO_SAMPLE`**
Default: Inherits from `INFERENCE_MODELS_DEFAULT_DO_SAMPLE`

```bash
export INFERENCE_MODELS_FLORENCE2_DEFAULT_DO_SAMPLE="true"
```

#### Grounding DINO

**`INFERENCE_MODELS_GROUNDING_DINO_DEFAULT_BOX_CONFIDENCE`**
Default: `0.5`

```bash
export INFERENCE_MODELS_GROUNDING_DINO_DEFAULT_BOX_CONFIDENCE="0.6"
```

**`INFERENCE_MODELS_GROUNDING_DINO_DEFAULT_MAX_DETECTIONS`**
Default: Inherits from `INFERENCE_MODELS_DEFAULT_MAX_DETECTIONS`

```bash
export INFERENCE_MODELS_GROUNDING_DINO_DEFAULT_MAX_DETECTIONS="200"
```

**`INFERENCE_MODELS_GROUNDING_DINO_DEFAULT_IOU_THRESHOLD`**
Default: `0.5`

```bash
export INFERENCE_MODELS_GROUNDING_DINO_DEFAULT_IOU_THRESHOLD="0.4"
```

#### MediaPipe Face Detector

**`INFERENCE_MODELS_MEDIAPIPE_FACE_DETECTOR_DEFAULT_CONFIDENCE`**
Default: `0.25`

```bash
export INFERENCE_MODELS_MEDIAPIPE_FACE_DETECTOR_DEFAULT_CONFIDENCE="0.3"
```

#### Moondream2

**`INFERENCE_MODELS_MOONDREAM2_DEFAULT_MAX_NEW_TOKENS`**
Default: `700`

```bash
export INFERENCE_MODELS_MOONDREAM2_DEFAULT_MAX_NEW_TOKENS="1000"
```

#### OWLv2

**`INFERENCE_MODELS_OWLV2_DEFAULT_CONFIDENCE`**
Default: `0.99`

```bash
export INFERENCE_MODELS_OWLV2_DEFAULT_CONFIDENCE="0.95"
```

**`INFERENCE_MODELS_OWLV2_DEFAULT_IOU_THRESHOLD`**
Default: Inherits from `INFERENCE_MODELS_DEFAULT_IOU_THRESHOLD`

```bash
export INFERENCE_MODELS_OWLV2_DEFAULT_IOU_THRESHOLD="0.4"
```

**`INFERENCE_MODELS_OWLV2_DEFAULT_MAX_DETECTIONS`**
Default: Inherits from `INFERENCE_MODELS_DEFAULT_MAX_DETECTIONS`

```bash
export INFERENCE_MODELS_OWLV2_DEFAULT_MAX_DETECTIONS="200"
```

**`INFERENCE_MODELS_OWLV2_DEFAULT_CLASS_AGNOSTIC_NMS`**
Default: Inherits from `INFERENCE_MODELS_DEFAULT_CLASS_AGNOSTIC_NMS`

```bash
export INFERENCE_MODELS_OWLV2_DEFAULT_CLASS_AGNOSTIC_NMS="true"
```

#### PaliGemma

**`INFERENCE_MODELS_PALIGEMMA_DEFAULT_MAX_NEW_TOKENS`**
Default: `400`

```bash
export INFERENCE_MODELS_PALIGEMMA_DEFAULT_MAX_NEW_TOKENS="500"
```

**`INFERENCE_MODELS_PALIGEMMA_DEFAULT_DO_SAMPLE`**
Default: Inherits from `INFERENCE_MODELS_DEFAULT_DO_SAMPLE`

```bash
export INFERENCE_MODELS_PALIGEMMA_DEFAULT_DO_SAMPLE="true"
```

**`INFERENCE_MODELS_PALIGEMMA_DEFAULT_SKIP_SPECIAL_TOKENS`**
Default: `true`

```bash
export INFERENCE_MODELS_PALIGEMMA_DEFAULT_SKIP_SPECIAL_TOKENS="false"
```

#### Qwen2.5-VL

**`INFERENCE_MODELS_QWEN25_VL_DEFAULT_MAX_NEW_TOKENS`**
Default: `512`

```bash
export INFERENCE_MODELS_QWEN25_VL_DEFAULT_MAX_NEW_TOKENS="1024"
```

**`INFERENCE_MODELS_QWEN25_VL_DEFAULT_DO_SAMPLE`**
Default: Inherits from `INFERENCE_MODELS_DEFAULT_DO_SAMPLE`

```bash
export INFERENCE_MODELS_QWEN25_VL_DEFAULT_DO_SAMPLE="true"
```

**`INFERENCE_MODELS_QWEN25_VL_DEFAULT_SKIP_SPECIAL_TOKENS`**
Default: `true`

```bash
export INFERENCE_MODELS_QWEN25_VL_DEFAULT_SKIP_SPECIAL_TOKENS="false"
```

#### Qwen3-VL

**`INFERENCE_MODELS_QWEN3_VL_DEFAULT_MAX_NEW_TOKENS`**
Default: `512`

```bash
export INFERENCE_MODELS_QWEN3_VL_DEFAULT_MAX_NEW_TOKENS="1024"
```

**`INFERENCE_MODELS_QWEN3_VL_DEFAULT_DO_SAMPLE`**
Default: Inherits from `INFERENCE_MODELS_DEFAULT_DO_SAMPLE`

```bash
export INFERENCE_MODELS_QWEN3_VL_DEFAULT_DO_SAMPLE="true"
```

#### ResNet

**`INFERENCE_MODELS_RESNET_DEFAULT_CONFIDENCE`**
Default: Inherits from `INFERENCE_MODELS_DEFAULT_CONFIDENCE`

```bash
export INFERENCE_MODELS_RESNET_DEFAULT_CONFIDENCE="0.5"
```

#### RF-DETR

**`INFERENCE_MODELS_RFDETR_DEFAULT_CONFIDENCE`**
Default: Inherits from `INFERENCE_MODELS_DEFAULT_CONFIDENCE`

```bash
export INFERENCE_MODELS_RFDETR_DEFAULT_CONFIDENCE="0.5"
```

#### Roboflow Instant

**`INFERENCE_MODELS_ROBOFLOW_INSTANT_DEFAULT_CONFIDENCE`**
Default: `0.99`

```bash
export INFERENCE_MODELS_ROBOFLOW_INSTANT_DEFAULT_CONFIDENCE="0.95"
```

**`INFERENCE_MODELS_ROBOFLOW_INSTANT_DEFAULT_IOU_THRESHOLD`**
Default: `0.3`

```bash
export INFERENCE_MODELS_ROBOFLOW_INSTANT_DEFAULT_IOU_THRESHOLD="0.4"
```

**`INFERENCE_MODELS_ROBOFLOW_INSTANT_MAX_DETECTIONS`**
Default: Inherits from `INFERENCE_MODELS_DEFAULT_MAX_DETECTIONS`

```bash
export INFERENCE_MODELS_ROBOFLOW_INSTANT_MAX_DETECTIONS="200"
```

#### SmolVLM

**`INFERENCE_MODELS_SMOL_VLM_DEFAULT_MAX_NEW_TOKENS`**
Default: `400`

```bash
export INFERENCE_MODELS_SMOL_VLM_DEFAULT_MAX_NEW_TOKENS="500"
```

**`INFERENCE_MODELS_SMOL_VLM_DEFAULT_DO_SAMPLE`**
Default: Inherits from `INFERENCE_MODELS_DEFAULT_DO_SAMPLE`

```bash
export INFERENCE_MODELS_SMOL_VLM_DEFAULT_DO_SAMPLE="true"
```

**`INFERENCE_MODELS_SMOL_VLM_DEFAULT_SKIP_SPECIAL_TOKENS`**
Default: `true`

```bash
export INFERENCE_MODELS_SMOL_VLM_DEFAULT_SKIP_SPECIAL_TOKENS="false"
```

#### ViT

**`INFERENCE_MODELS_VIT_CLASSIFIER_DEFAULT_CONFIDENCE`**
Default: Inherits from `INFERENCE_MODELS_DEFAULT_CONFIDENCE`

```bash
export INFERENCE_MODELS_VIT_CLASSIFIER_DEFAULT_CONFIDENCE="0.5"
```

#### YOLACT

**`INFERENCE_MODELS_YOLACT_DEFAULT_CONFIDENCE`**
Default: Inherits from `INFERENCE_MODELS_DEFAULT_CONFIDENCE`

```bash
export INFERENCE_MODELS_YOLACT_DEFAULT_CONFIDENCE="0.5"
```

**`INFERENCE_MODELS_YOLACT_DEFAULT_IOU_THRESHOLD`**
Default: Inherits from `INFERENCE_MODELS_DEFAULT_IOU_THRESHOLD`

```bash
export INFERENCE_MODELS_YOLACT_DEFAULT_IOU_THRESHOLD="0.4"
```

**`INFERENCE_MODELS_YOLACT_DEFAULT_MAX_DETECTIONS`**
Default: Inherits from `INFERENCE_MODELS_DEFAULT_MAX_DETECTIONS`

```bash
export INFERENCE_MODELS_YOLACT_DEFAULT_MAX_DETECTIONS="200"
```

**`INFERENCE_MODELS_YOLACT_DEFAULT_CLASS_AGNOSTIC_NMS`**
Default: Inherits from `INFERENCE_MODELS_DEFAULT_CLASS_AGNOSTIC_NMS`

```bash
export INFERENCE_MODELS_YOLACT_DEFAULT_CLASS_AGNOSTIC_NMS="true"
```

#### YOLO-NAS

**`INFERENCE_MODELS_YOLONAS_DEFAULT_CONFIDENCE`**
Default: Inherits from `INFERENCE_MODELS_DEFAULT_CONFIDENCE`

```bash
export INFERENCE_MODELS_YOLONAS_DEFAULT_CONFIDENCE="0.5"
```

**`INFERENCE_MODELS_YOLONAS_DEFAULT_IOU_THRESHOLD`**
Default: Inherits from `INFERENCE_MODELS_DEFAULT_IOU_THRESHOLD`

```bash
export INFERENCE_MODELS_YOLONAS_DEFAULT_IOU_THRESHOLD="0.4"
```

**`INFERENCE_MODELS_YOLONAS_DEFAULT_MAX_DETECTIONS`**
Default: Inherits from `INFERENCE_MODELS_DEFAULT_MAX_DETECTIONS`

```bash
export INFERENCE_MODELS_YOLONAS_DEFAULT_MAX_DETECTIONS="200"
```

**`INFERENCE_MODELS_YOLONAS_DEFAULT_CLASS_AGNOSTIC_NMS`**
Default: Inherits from `INFERENCE_MODELS_DEFAULT_CLASS_AGNOSTIC_NMS`

```bash
export INFERENCE_MODELS_YOLONAS_DEFAULT_CLASS_AGNOSTIC_NMS="true"
```

#### YOLOv5

**`INFERENCE_MODELS_YOLOV5_DEFAULT_CONFIDENCE`**
Default: Inherits from `INFERENCE_MODELS_DEFAULT_CONFIDENCE`

```bash
export INFERENCE_MODELS_YOLOV5_DEFAULT_CONFIDENCE="0.5"
```

**`INFERENCE_MODELS_YOLOV5_DEFAULT_IOU_THRESHOLD`**
Default: Inherits from `INFERENCE_MODELS_DEFAULT_IOU_THRESHOLD`

```bash
export INFERENCE_MODELS_YOLOV5_DEFAULT_IOU_THRESHOLD="0.4"
```

**`INFERENCE_MODELS_YOLOV5_DEFAULT_MAX_DETECTIONS`**
Default: Inherits from `INFERENCE_MODELS_DEFAULT_MAX_DETECTIONS`

```bash
export INFERENCE_MODELS_YOLOV5_DEFAULT_MAX_DETECTIONS="200"
```

**`INFERENCE_MODELS_YOLOV5_DEFAULT_CLASS_AGNOSTIC_NMS`**
Default: Inherits from `INFERENCE_MODELS_DEFAULT_CLASS_AGNOSTIC_NMS`

```bash
export INFERENCE_MODELS_YOLOV5_DEFAULT_CLASS_AGNOSTIC_NMS="true"
```

#### YOLOv7

**`INFERENCE_MODELS_YOLOV7_DEFAULT_CONFIDENCE`**
Default: Inherits from `INFERENCE_MODELS_DEFAULT_CONFIDENCE`

```bash
export INFERENCE_MODELS_YOLOV7_DEFAULT_CONFIDENCE="0.5"
```

**`INFERENCE_MODELS_YOLOV7_DEFAULT_IOU_THRESHOLD`**
Default: Inherits from `INFERENCE_MODELS_DEFAULT_IOU_THRESHOLD`

```bash
export INFERENCE_MODELS_YOLOV7_DEFAULT_IOU_THRESHOLD="0.4"
```

**`INFERENCE_MODELS_YOLOV7_DEFAULT_MAX_DETECTIONS`**
Default: Inherits from `INFERENCE_MODELS_DEFAULT_MAX_DETECTIONS`

```bash
export INFERENCE_MODELS_YOLOV7_DEFAULT_MAX_DETECTIONS="200"
```

**`INFERENCE_MODELS_YOLOV7_DEFAULT_CLASS_AGNOSTIC_NMS`**
Default: Inherits from `INFERENCE_MODELS_DEFAULT_CLASS_AGNOSTIC_NMS`

```bash
export INFERENCE_MODELS_YOLOV7_DEFAULT_CLASS_AGNOSTIC_NMS="true"
```

#### YOLOv8/v9/v11/v12 (Ultralytics)

**`INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_CONFIDENCE`**
Default: Inherits from `INFERENCE_MODELS_DEFAULT_CONFIDENCE`

```bash
export INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_CONFIDENCE="0.5"
```

**`INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_IOU_THRESHOLD`**
Default: Inherits from `INFERENCE_MODELS_DEFAULT_IOU_THRESHOLD`

```bash
export INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_IOU_THRESHOLD="0.4"
```

**`INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_MAX_DETECTIONS`**
Default: Inherits from `INFERENCE_MODELS_DEFAULT_MAX_DETECTIONS`

```bash
export INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_MAX_DETECTIONS="200"
```

**`INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_CLASS_AGNOSTIC_NMS`**
Default: Inherits from `INFERENCE_MODELS_DEFAULT_CLASS_AGNOSTIC_NMS`

```bash
export INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_CLASS_AGNOSTIC_NMS="true"
```

**`INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_KEY_POINTS_THRESHOLD`**
Default: `0.0`

```bash
export INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_KEY_POINTS_THRESHOLD="0.4"
```

#### YOLOv10

**`INFERENCE_MODELS_YOLOV10_DEFAULT_CONFIDENCE`**
Default: Inherits from `INFERENCE_MODELS_DEFAULT_CONFIDENCE`

```bash
export INFERENCE_MODELS_YOLOV10_DEFAULT_CONFIDENCE="0.5"
```

**`INFERENCE_MODELS_YOLOV10_DEFAULT_MAX_DETECTIONS`**
Default: Inherits from `INFERENCE_MODELS_DEFAULT_MAX_DETECTIONS`

```bash
export INFERENCE_MODELS_YOLOV10_DEFAULT_MAX_DETECTIONS="200"
```

## Logging

**`LOG_LEVEL`**  
Set the log level for the library. Default: `WARNING`

```bash
export LOG_LEVEL="DEBUG"
```

**`VERBOSE_LOG_LEVEL`**  
Set the log level for verbose logging. Default: `INFO`

```bash
export VERBOSE_LOG_LEVEL="DEBUG"
```

**`DISABLE_VERBOSE_LOGGER`**  
Disable verbose logging. Default: `false`

```bash
export DISABLE_VERBOSE_LOGGER="true"
```

**`DISABLE_INTERACTIVE_PROGRESS_BARS`**  
Disable interactive progress bars. Default: `false`

```bash
export DISABLE_INTERACTIVE_PROGRESS_BARS="true"
```


## Advanced Configuration

### Input Validation

**`ALLOW_URL_INPUT`**
Allow URLs as image input. Used by models like OWL-V2, when access to larger 
datasets provided as references is needed. Default: `true`

```bash
export ALLOW_URL_INPUT="true"
```

**`ALLOW_NON_HTTPS_URL_INPUT`**
Allow non-HTTPS URLs. Used by models like OWL-V2, when access to larger 
datasets provided as references is needed. Default: `false`

```bash
export ALLOW_NON_HTTPS_URL_INPUT="true"  # Use with caution
```

**`ALLOW_URL_INPUT_WITHOUT_FQDN`**
Allow URLs without FQDN. Used by models like OWL-V2, when access to larger 
datasets provided as references is needed. Default: `false`

```bash
export ALLOW_URL_INPUT_WITHOUT_FQDN="true"  # Use with caution
```

**`WHITELISTED_DESTINATIONS_FOR_URL_INPUT`**  
Comma-separated list of allowed destinations for URL input. Used by models like OWL-V2, when access to larger 
datasets provided as references is needed. Default: `None`

```bash
export WHITELISTED_DESTINATIONS_FOR_URL_INPUT="google.com,github.com"
```

**`BLACKLISTED_DESTINATIONS_FOR_URL_INPUT`**  
Comma-separated list of allowed destinations for URL input. Used by models like OWL-V2, when access to larger 
datasets provided as references is needed. Default: `None`

```bash
export BLACKLISTED_DESTINATIONS_FOR_URL_INPUT="google.com,github.com"
```

**`ALLOW_LOCAL_STORAGE_ACCESS_FOR_REFERENCE_DATA`**  
Allow local storage access for reference data. Used by models like OWL-V2, when access to larger 
datasets provided as references is needed. Default: `true`

```bash
export ALLOW_LOCAL_STORAGE_ACCESS_FOR_REFERENCE_DATA="true"
```

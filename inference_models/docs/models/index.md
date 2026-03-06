# Models Overview

The `inference-models` library supports a wide range of computer vision models across multiple tasks and backends.

## Supported Tasks

- **Object Detection**: Detect and localize objects in images
- **Instance Segmentation**: Detect objects with pixel-level masks
- **Semantic Segmentation**: Classify every pixel in an image
- **Classification**: Classify entire images or image regions
- **Embeddings**: Generate vector representations for images and text
- **OCR & Document Parsing**: Extract text and structure from documents
- **Interactive Segmentation**: Interactive and automatic segmentation
- **Vision-Language Models**: Multi-modal understanding and generation
- **Depth Estimation**: Predict depth maps from images
- **Specialized**: Gaze detection, face detection, and more

## Model Catalog

**Legend:** ✅ Available | ❌ Not available | 🔑 Requires API key | 📤 Upload only

### Object Detection

| Model                                   | Backends | License    | Commercial License in RF Plan | Pre-trained Weights | Trainable at RF |
|-----------------------------------------|--------|------------|-------------------------------|---------------------|----------------|
| [RF-DETR](rfdetr-object-detection.md)   | `torch`, `onnx`, `trt` | Apache 2.0 | N/A | ✅ | ✅ |
| [YOLOv5](yolov5-object-detection.md)    | `onnx` | AGPL-3.0   | ✅ | ❌ | 📤 |
| [YOLOv8](yolov8-object-detection.md)    | `onnx`, `torch-script`, `trt` | AGPL-3.0   | ✅ | ✅ | ✅ |
| [YOLOv9](yolov9-object-detection.md)    | `onnx`, `torch-script`, `trt` | GPL-3.0    | ❌ | ❌ | 📤 |
| [YOLOv10](yolov10-object-detection.md)  | `onnx`, `trt` | AGPL-3.0   | ✅ | ✅ | 📤 |
| [YOLOv11](yolov11-object-detection.md)  | `onnx`, `torch-script`, `trt` | AGPL-3.0   | ✅ | ✅ | ✅ |
| [YOLOv12](yolov12-object-detection.md)  | `onnx`, `torch-script`, `trt` | AGPL-3.0   | ✅ | ❌ | ✅ |
| [YOLO-NAS](yolonas-object-detection.md) | `onnx`, `trt` | Apache 2.0 | N/A | ✅ | ✅ |
| [Grounding DINO](grounding-dino.md)     | `torch` | Apache 2.0 | N/A | 🔑 | ❌ |
| [OWLv2](owlv2.md)                       | `hugging-face` | Apache 2.0 | N/A | 🔑 | ❌ |
| [Roboflow Instant](rroboflow-instant-object-detection)           | `hugging-face` | Roboflow   | ✅ | ❌ | ✅ |

### Instance Segmentation

| Model | Backends | License | Commercial License in RF Plan | Pre-trained Weights | Trainable at RF |
|-------|--------|---------|-------------------------------|---------------------|-----------------|
| [RF-DETR Seg](rfdetr-instance-segmentation.md) | `torch` | Apache 2.0 | N/A | ✅ | ✅ |
| [YOLOv5 Seg](yolov5-instance-segmentation.md) | `onnx` | AGPL-3.0 | ✅ | ❌ | ✅ |
| [YOLOv7 Seg](yolov7-instance-segmentation.md) | `onnx` | AGPL-3.0 | ❌ | ❌ | ✅ |
| [YOLOv8 Seg](yolov8-instance-segmentation.md) | `onnx`, `torch-script`, `trt` | AGPL-3.0 | ✅ | ✅ | ✅ |
| [YOLOv11 Seg](yolov11-instance-segmentation.md) | `onnx`, `torch-script`, `trt` | AGPL-3.0 | ✅ | ✅ | ✅ |
| [YOLACT](yolact-instance-segmentation.md) | `onnx` | MIT | N/A | ❌ | ✅ |

### Classification

| Model | Backends | License | Commercial License in RF Plan | Pre-trained Weights | Trainable at RF |
|-------|----------|---------|-------------------------------|---------------------|-----------------|
| [ResNet](resnet-classification.md) | `torch`, `onnx`, `trt` | Apache 2.0 | N/A | ✅ | ✅ |
| [ViT](vit-classification.md) | `torch` | Apache 2.0 | N/A | ✅ | ✅ |
| [DINOv3](dinov3-classification.md) | `torch` | Meta DINO | N/A | ❌ | ✅ |
| [YOLOv8 Cls](yolov8-classification.md) | `onnx`, `trt` | AGPL-3.0 | ✅ | ✅ | ✅ |

### Embeddings

| Model | Backends | License | Commercial License in RF Plan | Pre-trained Weights | Trainable at RF |
|-------|----------|---------|-------------------------------|---------------------|-----------------|
| [CLIP](clip.md) | `torch`, `onnx` | MIT | N/A | ✅ | ❌ |
| [Perception Encoder](perception-encoder.md) | `torch` | FAIR Noncommercial | ❌ | ✅ | ❌ |

### Semantic Segmentation

| Model | Backends | License | Commercial License in RF Plan | Pre-trained Weights | Trainable at RF |
|-------|----------|---------|-------------------------------|---------------------|-----------------|
| [DeepLabV3+](deeplabv3plus.md) | `torch`, `onnx`, `trt` | MIT | N/A | ❌ | ✅ |

### OCR & Document Parsing

| Model | Backends | License | Commercial License in RF Plan | Pre-trained Weights | Trainable at RF |
|-------|----------|---------|-------------------------------|---------------------|-----------------|
| [DocTR](doctr.md) | `torch` | Apache 2.0 | N/A | 🔑 | ❌ |
| [EasyOCR](easyocr.md) | `torch` | Apache 2.0 | N/A | 🔑 | ❌ |
| [TrOCR](trocr.md) | `hugging-face` | MIT | N/A | 🔑 | ❌ |

### Interactive Segmentation

| Model | Backends | License | Commercial License in RF Plan | Pre-trained Weights | Trainable at RF |
|-------|----------|---------|-------------------------------|---------------------|-----------------|
| [SAM](sam-interactive-segmentation.md) | `torch` | Apache 2.0 | N/A | 🔑 | ❌ |
| [SAM2](sam2-interactive-segmentation.md) | `torch` | Apache 2.0 | N/A | 🔑 | ❌ |
| [SAM2 RT](sam2-rt-video-tracking.md) | `torch` | Apache 2.0 | N/A | ✅ | ❌ |

### Vision-Language Models

| Model | Backends | License | Commercial License in RF Plan | Pre-trained Weights | Trainable at RF |
|-------|----------|---------|-------------------------------|---------------------|-----------------|
| [Florence-2](florence2.md) | `torch` | MIT | N/A                           | ✅ | ✅ |
| [PaliGemma](paligemma.md) | `torch` | Gemma License | N/A                           | ✅ | ✅ |
| [Qwen2.5-VL](qwen25vl.md) | `torch` | Apache 2.0 | N/A                           | ✅ | ✅ |
| [Qwen3-VL](qwen3vl.md) | `torch` | Apache 2.0 | N/A                           | ✅ | ✅ |
| [Qwen3.5](qwen35.md) | `torch` | Apache 2.0 | N/A                           | ✅ | ✅ |
| [SmolVLM](smolvlm.md) | `torch` | Apache 2.0 | N/A                           | ✅ | ✅ |
| [Moondream2](moondream2.md) | `torch` | Apache 2.0 | N/A                           | ✅ | ❌ |

### Depth Estimation

| Model | Backends | License | Commercial License in RF Plan | Pre-trained Weights | Trainable at RF |
|-------|----------|---------|-------------------------------|---------------------|-----------------|
| [Depth Anything V2](depth-anything-v2.md) | `torch` | Apache 2.0 | N/A | 🔑 | ❌ |
| [Depth Anything V3](depth-anything-v3.md) | `torch` | Apache 2.0 | N/A | 🔑 | ❌ |

### Specialized Models

| Model | Backends | License | Commercial License in RF Plan | Pre-trained Weights | Trainable at RF |
|-------|----------|---------|-------------------------------|---------------------|-----------------|
| [L2CS](l2cs.md) | `torch` | MIT | N/A | ✅ | ❌ |
| [MediaPipe Face](mediapipe-face.md) | `mediapipe` | Apache 2.0 | N/A | ✅ | ❌ |



## Next Steps

- Browse individual model pages for detailed documentation
- See [Quick Overview](../getting-started/overview.md) for usage examples
- Check [Installation Guide](../getting-started/installation.md) for backend setup


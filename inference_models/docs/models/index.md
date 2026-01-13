# Models Overview

The `inference-models` library supports a wide range of computer vision models across multiple tasks and backends.

## Supported Tasks

- **Object Detection**: Detect and localize objects in images
- **Instance Segmentation**: Detect objects with pixel-level masks
- **Classification**: Classify entire images or image regions
- **Embeddings**: Generate vector representations for images and text
- **OCR & Document Parsing**: Extract text and structure from documents
- **Segmentation**: Interactive and automatic segmentation
- **Vision-Language Models**: Multi-modal understanding and generation
- **Depth Estimation**: Predict depth maps from images
- **Specialized**: Gaze detection, face detection, and more

## Model Catalog

### Object Detection

| Model | Model IDs | Backends | License | Access |
|-------|-----------|----------|---------|--------|
| [RFDetr](rfdetr.md) | `rfdetr-base`, `rfdetr-small`, `rfdetr-medium`, `rfdetr-nano`, `rfdetr-large` | Torch, TRT | Apache 2.0 | Public |
| [YOLOv8](yolov8.md) | `yolov8{n,s,m,l,x}-{640,1280}` | ONNX, TRT | AGPL | Public |
| [YOLOv9](yolov9.md) | `yolov9{c,e}-{640,1280}` | ONNX, TRT | AGPL | Public |
| [YOLOv10](yolov10.md) | `yolov10{n,s,m,b,l,x}-640` | ONNX, TRT | AGPL | Public |
| [YOLOv11](yolov11.md) | `yolov11{n,s,m,l,x}-{640,1280}` | ONNX, TRT | AGPL | Public |
| [YOLOv12](yolov12.md) | `yolov12{n,s,m,l,x}-{640,1280}` | ONNX, TRT | AGPL | Public |
| [YOLO-NAS](yolonas.md) | `yolonas{s,m,l}-{640,1280}` | ONNX, TRT | Apache 2.0 | Public |
| [Grounding DINO](grounding-dino.md) | `grounding-dino-{tiny,base}` | Torch | Apache 2.0 | Public |
| [OWLv2](owlv2.md) | `owlv2-{base,large}-patch14` | HF | Apache 2.0 | Public |
| [YOLO-World](yolo-world.md) | `yolo-world-{s,m,l,x}` | Torch | GPL-3.0 | Public |

### Instance Segmentation

| Model | Model IDs | Backends | License | Access |
|-------|-----------|----------|---------|--------|
| [YOLOv8 Seg](yolov8-seg.md) | `yolov8{n,s,m,l,x}-seg-{640,1280}` | ONNX, TRT | AGPL | Public |
| [YOLOv11 Seg](yolov11-seg.md) | `yolov11{n,s,m,l,x}-seg-{640,1280}` | ONNX, TRT | AGPL | Public |
| [YOLACT](yolact.md) | `yolact-{resnet50,resnet101}` | Torch | MIT | Public |

### Classification

| Model | Model IDs | Backends | License | Access |
|-------|-----------|----------|---------|--------|
| [ResNet](resnet.md) | `resnet{18,34,50,101,152}` | Torch | Apache 2.0 | Public |
| [ViT](vit.md) | `vit-{tiny,small,base,large}-patch{16,32}` | Torch | Apache 2.0 | Public |
| [DINOv3](dinov3.md) | `dinov3-{small,base,large,giant}` | Torch | Apache 2.0 | Public |

### Embeddings

| Model | Model IDs | Backends | License | Access |
|-------|-----------|----------|---------|--------|
| [CLIP](clip.md) | `clip/{RN50,RN101,RN50x4,RN50x16,RN50x64,ViT-B-16,ViT-B-32,ViT-L-14,ViT-L-14-336px}` | Torch, ONNX | MIT | Public |
| [Perception Encoder](perception-encoder.md) | `perception-encoder/PE-Core-{B16-224,G14-448,L14-336}` | Torch | FAIR Noncommercial | Public |

### OCR & Document Parsing

| Model | Model IDs | Backends | License | Access |
|-------|-----------|----------|---------|--------|
| [DocTR](doctr.md) | `doctr-{db_resnet50,db_mobilenet_v3}` | Torch | Apache 2.0 | Public |
| [EasyOCR](easyocr.md) | `easyocr-{en,multi}` | Torch | Apache 2.0 | Public |
| [TrOCR](trocr.md) | `trocr-{base,large}-{printed,handwritten}` | HF | Apache 2.0 | Public |

### Segmentation

| Model | Model IDs | Backends | License | Access |
|-------|-----------|----------|---------|--------|
| [SAM](sam.md) | `sam-{vit_b,vit_l,vit_h}` | Torch | Apache 2.0 | Public |
| [SAM2](sam2.md) | `sam2-{tiny,small,base,large}` | Torch | Apache 2.0 | Public |
| [SAM2 RT](sam2-rt.md) | `sam2-rt-{tiny,small,base,large}` | Torch | Apache 2.0 | Public |

### Vision-Language Models

| Model | Model IDs | Backends | License | Access |
|-------|-----------|----------|---------|--------|
| [Florence-2](florence2.md) | `florence-2-{base,large}` | Torch | MIT | Public |
| [PaliGemma](paligemma.md) | `paligemma-{3b,10b}` | Torch | Gemma License | Public |
| [Qwen2.5-VL](qwen25vl.md) | `qwen2.5-vl-{2b,7b,72b}` | Torch | Apache 2.0 | Public |
| [Qwen3-VL](qwen3vl.md) | `qwen3-vl-{2b,8b}` | Torch | Apache 2.0 | Public |
| [SmolVLM](smolvlm.md) | `smolvlm-{256m,2.2b}` | Torch | Apache 2.0 | Public |
| [Moondream2](moondream2.md) | `moondream2` | Torch | Apache 2.0 | Public |

### Depth Estimation

| Model | Model IDs | Backends | License | Access |
|-------|-----------|----------|---------|--------|
| [Depth Anything V2](depth-anything-v2.md) | `depth-anything-v2-{small,base,large}` | Torch | Apache 2.0 | Public |
| [Depth Anything V3](depth-anything-v3.md) | `depth-anything-v3-{small,base,large}` | Torch | Apache 2.0 | Public |

### Specialized Models

| Model | Model IDs | Backends | License | Access |
|-------|-----------|----------|---------|--------|
| [L2CS](l2cs.md) | `l2cs-gaze` | Torch | MIT | Public |
| [MediaPipe Face](mediapipe-face.md) | `mediapipe-face-detection` | MediaPipe | Apache 2.0 | Public |

## Access Levels

- **Public**: Available without API key
- **API Key Gated**: Requires Roboflow API key
- **Private**: Only available to specific workspaces

## Backend Support

### PyTorch (Torch)
- Default backend
- Maximum flexibility
- Good for development
- Supports all model types

### ONNX
- Cross-platform compatibility
- Good CPU and GPU performance
- Required for Roboflow-trained models
- Install: `pip install "inference-models[onnx-cpu]"` or `[onnx-cu12]`

### TensorRT (TRT)
- Maximum GPU performance
- Optimized for NVIDIA GPUs
- Requires exact environment match
- Install: `pip install "inference-models[trt10]"`

### Hugging Face (HF)
- Transformer-based models
- Integrated with HF ecosystem
- Included in base installation

### MediaPipe
- Optimized for mobile/edge
- Efficient face detection
- Install: `pip install "inference-models[mediapipe]"`

## Model Naming Convention

Model IDs follow these patterns:

### Pre-trained Models
```
{architecture}-{variant}-{size}
```
Examples:
- `yolov8n-640` - YOLOv8 nano, 640px input
- `rfdetr-base` - RFDetr base variant
- `clip/ViT-B-32` - CLIP ViT-Base with 32px patches

### Roboflow-Trained Models
```
{workspace}/{project}/{version}
```
Example:
- `my-workspace/my-project/1`

## Common Model Interfaces

All models implement task-specific interfaces:

### Object Detection
```python
model = AutoModel.from_pretrained("yolov8n-640")
predictions: List[Detections] = model(images)
```

### Instance Segmentation
```python
model = AutoModel.from_pretrained("yolov8n-seg-640")
predictions: List[InstanceDetections] = model(images)
```

### Classification
```python
model = AutoModel.from_pretrained("resnet50")
prediction: ClassificationPrediction = model(image)
```

### Embeddings
```python
model = AutoModel.from_pretrained("clip/ViT-B-32")
image_emb = model.embed_image(image)
text_emb = model.embed_text("a photo of a cat")
```

## Next Steps

- Browse individual model pages for detailed documentation
- See [Quick Overview](../getting-started/overview.md) for usage examples
- Check [Installation Guide](../getting-started/installation.md) for backend setup


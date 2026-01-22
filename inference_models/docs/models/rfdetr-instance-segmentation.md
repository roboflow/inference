# RF-DETR - Instance Segmentation

RF-DETR ("Roboflow Detection Transformer") is a groundbreaking real-time transformer-based architecture developed by the Roboflow research team. Designed from the ground up to transfer exceptionally well across diverse domains and dataset sizes, RF-DETR Seg represents a major breakthrough in instance segmentation.

## Overview

**RF-DETR Seg (Preview) is the fastest and most accurate real-time instance segmentation model ever created.** Developed by Roboflow's world-class research team, this model achieves what was previously thought impossible: true real-time transformer-based instance segmentation that outperforms all existing YOLO models.

Key achievements:

- **Unprecedented speed** - Over 150 FPS on T4 GPU, making it the fastest instance segmentation model available
- **Superior accuracy** - 3x faster than YOLO11-x-seg while achieving better accuracy on COCO
- **First of its kind** - The world's first DETR-based segmentation model to achieve true real-time performance
- **Production-ready** - Built by Roboflow for real-world deployment across any domain
- **Apache 2.0 licensed** - Truly open source with no restrictions on commercial use

## License

**Apache 2.0**

!!! info "Open Source License"
    RF-DETR Segmentation is licensed under Apache 2.0, making it free for both commercial and non-commercial use without restrictions.

    Learn more: [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0)

## Pre-trained Model IDs

RF-DETR Segmentation preview model is trained on the COCO dataset (80 classes) and is **open access** (no API key required).

| Model Size | Model ID |
|------------|----------|
| Preview | `rfdetr-seg-preview` |

## Supported Backends

| Backend | Extras Required |
|---------|----------------|
| `torch` | `torch-cpu`, `torch-cu118`, `torch-cu124`, `torch-cu126`, `torch-cu128`, `torch-jp6-cu126` |

## Roboflow Platform Compatibility

| Feature | Supported |
|---------|-----------|
| **Training** | ✅ Train custom models on Roboflow |
| **Upload Weights** | ✅ Upload pre-trained weights |
| **Serverless API (v2)** | ✅ [Deploy via hosted API](https://docs.roboflow.com/deploy/serverless-hosted-api-v2) |
| **Workflows** | ✅ Use in [Workflows](https://docs.roboflow.com/workflows) via Instance Segmentation block |
| **Edge Deployment (Jetson)** | ✅ Deploy on NVIDIA Jetson devices |
| **Self-Hosting** | ✅ Deploy with `inference-models` |

**Custom model ID format:** `project-url/version` (e.g., `my-project-abc123/2`)

## Usage Example

### Using Pre-trained Models

```python
import cv2
import supervision as sv
from inference_models import AutoModel

# Load model and image
model = AutoModel.from_pretrained("rfdetr-seg-preview")
image = cv2.imread("path/to/image.jpg")

# Run inference
predictions = model(image)
detections = predictions[0].to_supervision()

# Annotate image with masks
mask_annotator = sv.MaskAnnotator()
annotated_image = mask_annotator.annotate(image.copy(), detections)

# Save or display
cv2.imwrite("annotated.jpg", annotated_image)
```

## Trained RF-DETR Segmentation Outside Roboflow? Use with `inference-models`

RF-DETR Segmentation offers a **seamless training-to-deployment workflow** that makes it incredibly easy to go from training to production.

### Step 1: Train Your Model

Train RF-DETR Segmentation on your custom dataset using the official [rf-detr repository](https://github.com/roboflow/rf-detr):

```bash
# Clone the RF-DETR training repository
git clone https://github.com/roboflow/rf-detr.git
cd rf-detr

# Install dependencies
pip install -r requirements.txt

# Train on your custom instance segmentation dataset
python train.py --config configs/rfdetr_seg_preview.yaml --data path/to/your/dataset
```

After training completes, you'll have a checkpoint file (e.g., `checkpoint_best.pth`) containing your trained weights.

### Step 2: Deploy Instantly with inference-models

Here's where the magic happens - **no conversion, no export, no hassle**. Simply point `AutoModel` directly at your training checkpoint:

```python
import cv2
import supervision as sv
from inference_models import AutoModel

# Load your freshly trained model directly from the checkpoint
# You MUST specify model_type for checkpoint loading
model = AutoModel.from_pretrained(
    "/path/to/checkpoint_best.pth",
    model_type="rfdetr-seg-preview",  # Required: specify the model architecture
    labels=["class1", "class2", "class3"]  # Optional: your custom class names
)

# That's it! Use it exactly like any other model
image = cv2.imread("path/to/image.jpg")
predictions = model(image)
detections = predictions[0].to_supervision()

# Annotate with masks
mask_annotator = sv.MaskAnnotator()
annotated_image = mask_annotator.annotate(image.copy(), detections)
cv2.imwrite("annotated.jpg", annotated_image)
```

**Important parameters:**

- **`model_type`** (required) - Specifies the RF-DETR Segmentation architecture variant. Currently: `rfdetr-seg-preview`
- **`labels`** (optional) - Class names for your model. Can be:
    - A list of class names: `["person", "car", "dog"]`
    - A registered label set name: `"coco"` (for COCO dataset classes)
    - If not provided, defaults to COCO labels

### Why This Matters

**Frictionless training-to-production workflow:**

- ✅ **No model conversion** - Use training checkpoints directly
- ✅ **No export step** - Skip ONNX/TensorRT export complexity
- ✅ **Instant deployment** - From training to production in seconds
- ✅ **Same API** - Identical interface for pre-trained and custom models
- ✅ **Production-ready** - Leverage all `inference-models` features (multi-backend, caching, optimization)

This seamless workflow eliminates the traditional friction between training and deployment, letting you iterate faster and deploy with confidence.


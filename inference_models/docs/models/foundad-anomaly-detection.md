# FoundAD - Anomaly Detection

FoundAD detects anomalies with a frozen DINOv3 encoder and a small predictor trained to reproduce the features of normal images. Patches the predictor cannot reproduce are anomalous.

## Overview

- **Normal-only training** - the predictor learns the feature manifold of normal images
- **Foundation model features** - frozen DINOv3 ViT-B/16 encoder
- **Calibrated decision** - the threshold is fitted on validation images and saved with the model
- **Heatmaps** - optional per-pixel anomaly evidence
- **GPU recommended** - the encoder is a ViT-B

## License

**MIT (FoundAD) and Meta DINOv3 License (encoder)**

The implementation follows [ymxlzgy/FoundAD](https://github.com/ymxlzgy/FoundAD), distributed under the MIT License. The DINOv3 encoder weights are covered by the Meta DINOv3 License, see [DINOv3 - Classification](dinov3-classification.md#license) for its terms.

**Full License**: [FoundAD and DINOv3 licenses](https://github.com/roboflow/inference/blob/main/inference_models/inference_models/models/foundad/LICENSE.txt)

## Pre-trained Model IDs

FoundAD models are fitted to your own normal images on Roboflow. There are no pre-trained public model IDs.

**Custom model ID format:** `project-url/version` (e.g., `my-project-abc123/2`)

## Supported Backends

| Backend | Extras Required |
|---------|----------------|
| `torch` | `torch-cpu`, `torch-cu118`, `torch-cu124`, `torch-cu126`, `torch-cu128`, `torch-jp6-cu126` |

The package contains the DINOv3 encoder and the predictor. Loading does not download pre-trained weights. The encoder is built with `timm` and needs a `timm` release that ships DINOv3 (`1.0.20` or newer).

## Roboflow Platform Compatibility

| Feature | Supported |
|---------|-----------|
| **Training** | ✅ Train custom models on Roboflow |
| **Upload Weights** | ❌ Not supported |
| **Workflows** | ✅ Use in [Workflows](https://docs.roboflow.com/workflows) via Classification block |
| **Self-Hosting** | ✅ Deploy with `inference-models` |

## Usage Example

```python
import cv2
from inference_models import AutoModel

model = AutoModel.from_pretrained(
    "my-project-abc123/2",
    api_key="your_roboflow_api_key"
)
image = cv2.imread("path/to/image.jpg")

prediction = model(image, include_anomaly_map=True)

result = prediction.images_metadata[0]
print(model.class_names[prediction.class_id[0].item()])  # "normal" or "anomalous"
print(result["anomaly_score"], result["anomaly_threshold"], result["is_anomalous"])
heatmap = result["anomaly_map"]  # float32 array with the input image height and width
```

## Prediction Format

The model returns a `ClassificationPrediction` over the fixed classes `normal` and `anomalous`:

- `class_id` follows the threshold saved with the model: `anomalous` when `anomaly_score >= anomaly_threshold`.
- `confidence` is a monotonic transform of the score margin. The threshold maps to exactly `0.5`, so the
  argmax always agrees with `class_id`. It is not an estimated defect probability.
- `images_metadata[i]` holds `anomaly_score` (raw score, larger means more anomalous), `anomaly_threshold`,
  `is_anomalous` and, when `include_anomaly_map=True` is passed, `anomaly_map`.

The score of an image does not depend on the other images in the batch. The heatmap is local evidence for the
score, not a supervised segmentation mask.

## Input Images

Send the original image. The model package records how the training images were exported from Roboflow
(resize and JPEG encoding) and the model reproduces those steps before scoring, because the saved threshold
is only valid for pixels that look like the training pixels. Images must be 3-channel `uint8` arrays or tensors.

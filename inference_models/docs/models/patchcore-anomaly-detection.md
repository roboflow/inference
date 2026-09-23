# PatchCore - Anomaly Detection

PatchCore detects anomalies by comparing patches of an image with a memory bank of patch features collected from normal images. It needs no anomalous images to train.

## Overview

- **Normal-only training** - the model stores a coreset of WideResNet50 patch features from normal images
- **Nearest-neighbour scoring** - an image is as anomalous as its most unusual patch
- **Calibrated decision** - the threshold is fitted on validation images and saved with the model
- **Heatmaps** - optional per-pixel anomaly evidence
- **CPU friendly** - scoring is a single backbone pass plus a matrix product

## License

**Apache 2.0**

The implementation follows [amazon-science/patchcore-inspection](https://github.com/amazon-science/patchcore-inspection), licensed under Apache 2.0.

**Full License**: [Apache 2.0](https://github.com/roboflow/inference/blob/main/inference_models/inference_models/models/patchcore/LICENSE.txt)

## Pre-trained Model IDs

PatchCore models are fitted to your own normal images on Roboflow. There are no pre-trained public model IDs.

**Custom model ID format:** `project-url/version` (e.g., `my-project-abc123/2`)

## Supported Backends

| Backend | Extras Required |
|---------|----------------|
| `torch` | `torch-cpu`, `torch-cu118`, `torch-cu124`, `torch-cu126`, `torch-cu128`, `torch-jp6-cu126` |

The package contains the full WideResNet50 backbone and the memory bank. Loading does not download pre-trained weights and nearest-neighbour search runs in PyTorch, so no extra dependency is needed.

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

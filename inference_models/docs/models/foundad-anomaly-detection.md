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

print(prediction.anomaly_scores[0].item(), prediction.anomaly_threshold)
print(prediction.is_anomalous[0].item())  # anomaly_score >= anomaly_threshold
heatmap = prediction.anomaly_maps[0].numpy()  # float32 array of shape (image_size, image_size)
overlay = cv2.resize(heatmap, (image.shape[1], image.shape[0]))  # stretch onto the image

# Decide against your own threshold instead of the one saved with the model.
stricter = model(image, anomaly_threshold=2.5)
```

## Prediction Format

The model returns an `AnomalyDetectionPrediction`. There is no class confidence: the verdict is a raw score
compared with a threshold.

- `anomaly_scores` `(bs,)`: the raw image scores, larger means more anomalous. They are not probabilities and
  their scale depends on the trained model.
- `anomaly_threshold`: the decision threshold in effect. It is the threshold fitted on validation images and
  saved with the model, unless the call passes `anomaly_threshold=<float>` to override it.
- `is_anomalous` `(bs,)`: `anomaly_scores >= anomaly_threshold`.
- `anomaly_maps` `(bs, image_size, image_size)`: the local anomaly evidence, only when
  `include_anomaly_map=True` is passed; otherwise `None`.

`class_names` is the fixed pair `["normal", "anomalous"]` (index 1 is the anomalous verdict).

The score of an image does not depend on the other images in the batch. The heatmap is local evidence for the
score, not a supervised segmentation mask. It is returned at the network input resolution (a square of the
package's `image_size`) and covers the whole image as the network saw it, so resizing it to the image's width
and height is a plain stretch. Over HTTP, `inference` serves it as base64 float32 bytes with its shape and the
`inference_sdk` client decodes it back to a numpy array.

## Input Images

Send the original image. The model package records how the training images were exported from Roboflow
(resize and JPEG encoding) and the model reproduces those steps before scoring, because the saved threshold
is only valid for pixels that look like the training pixels. Images must be 3-channel `uint8` arrays or tensors.

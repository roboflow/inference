# L2CS - Gaze Detection

L2CS (Learning to Calibrate and Segment) is a gaze estimation model that predicts where a person is looking based on their face image.

## Overview

L2CS estimates gaze direction from face images by predicting yaw and pitch angles. Key capabilities include:

- **Gaze Estimation** - Predict gaze direction from face images
- **Yaw and Pitch Angles** - Output gaze direction in radians
- **Batch Processing** - Process multiple faces simultaneously
- **Fast Inference** - ONNX-optimized for efficient execution

!!! info "License & Attribution"
    **License**: MIT<br>**Source**: [L2CS-Net](https://github.com/Ahmednull/L2CS-Net)<br>**Paper**: [L2CS-Net: Learning to Calibrate and Segment](https://arxiv.org/abs/2203.03339)

!!! note "Input Requirements"
    L2CS expects **cropped face images** as input. For full face detection + gaze estimation, use the `face-and-gaze-detection` pipeline.

## Pre-trained Model IDs

L2CS pre-trained models are available and **require a Roboflow API key**.

| Model ID | Description |
|----------|-------------|
| `l2cs-net/rn50` | L2CS with ResNet-50 backbone |

## Supported Backends

| Backend | Extras Required |
|---------|----------------|
| `onnx` | `onnx-cpu`, `onnx-gpu` |

## Roboflow Platform Compatibility

| Feature | Supported |
|---------|-----------|
| **Training** | ❌ Not available for training |
| **Upload Weights** | ❌ Not supported |
| **Serverless API (v2)** | ✅ [Deploy via hosted API](https://docs.roboflow.com/deploy/serverless-hosted-api-v2) |
| **Workflows** | ✅ Use in [Workflows](https://inference.roboflow.com/workflows/about/) via Gaze Detection block |
| **Edge Deployment (Jetson)** | ✅ Supported |
| **Self-Hosting** | ✅ Deploy with `inference-models` |

## Usage Examples

### Basic Gaze Estimation (Cropped Faces)

```python
import cv2
import numpy as np
from inference_models import AutoModel

# Load model (requires API key)
model = AutoModel.from_pretrained("l2cs-net/rn50", api_key="your_roboflow_api_key")

# Load cropped face image
face_image = cv2.imread("path/to/face.jpg")

# Predict gaze direction
result = model(face_image)

# Get yaw and pitch in radians
yaw = result.yaw.cpu().numpy()[0]
pitch = result.pitch.cpu().numpy()[0]

# Convert to degrees
yaw_degrees = yaw * 180 / np.pi
pitch_degrees = pitch * 180 / np.pi

print(f"Gaze direction - Yaw: {yaw_degrees:.2f}°, Pitch: {pitch_degrees:.2f}°")
```

### Face Detection + Gaze Estimation Pipeline

```python
import cv2
from inference_models import AutoModelPipeline

# Load the combined pipeline (face detection + gaze estimation)
# Note: Pipeline uses default models and doesn't require API key
pipeline = AutoModelPipeline.from_pretrained("face-and-gaze-detection")

# Load image with faces
image = cv2.imread("path/to/image.jpg")

# Run pipeline
keypoints, detections, gaze_results = pipeline(image)

# Access results
print(f"Detected {len(detections[0])} faces")
print(f"Gaze yaw: {gaze_results[0].yaw}")
print(f"Gaze pitch: {gaze_results[0].pitch}")
```

## Workflows Integration

L2CS can be used in Roboflow Workflows for gaze detection in complex pipelines. The Gaze Detection block can:
- Detect faces and estimate gaze in one step
- Work with pre-cropped face images
- Output yaw and pitch angles for downstream processing

Learn more: [Workflows Documentation](https://inference.roboflow.com/workflows/about/)

## Performance Tips

1. **Use cropped faces** - L2CS works best with properly cropped face images
2. **Use the pipeline** - For full images, use `face-and-gaze-detection` pipeline
3. **Batch processing** - Process multiple faces together for better throughput
4. **ONNX optimization** - The model uses ONNX for efficient inference

## Output Format

The model returns an `L2CSGazeDetection` object with:

- **yaw**: Horizontal gaze angle in radians (torch.Tensor)
- **pitch**: Vertical gaze angle in radians (torch.Tensor)

### Angle Interpretation

- **Yaw**: Horizontal direction
  - Negative: Looking left
  - Positive: Looking right
  - Zero: Looking straight ahead horizontally
  
- **Pitch**: Vertical direction
  - Negative: Looking up
  - Positive: Looking down
  - Zero: Looking straight ahead vertically

## Common Use Cases

- **Attention Tracking** - Monitor where users are looking
- **Driver Monitoring** - Detect driver distraction
- **User Experience Research** - Analyze visual attention patterns
- **Accessibility** - Eye-gaze based interfaces
- **Security** - Detect suspicious behavior patterns


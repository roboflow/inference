# MediaPipe Hands - Hand Landmarks

MediaPipe Hands finds hands and returns 21 landmarks per hand (wrist, and four joints for each finger), plus handedness. This is Google's hand landmarker, with its two models converted to ONNX and the steps between them re-implemented in `inference-models`, so it runs without the `mediapipe` library on every ONNX execution provider, including CUDA and TensorRT on Jetson.

## Overview

The pipeline has two stages, each a registered model of its own:

- **Palm detector** (`mediapipe-palm-detector`) - finds palms and seven palm key points that orient the hand.
- **Hand landmarks** (`mediapipe-hand-landmarks`) - predicts the 21 landmarks, a hand presence score and handedness from a rotated crop of one hand.
- **MediaPipe Hands** (`mediapipe-hands`) - the composite that runs both, the way MediaPipe does.

The composite can take any object detection model in place of the palm detector (the `handDetector` dependency). MediaPipe's palm detector rarely fires on gloved hands; a detector trained on your own gloved hands, followed by the same landmark model, often works where it does not.

!!! info "License & Attribution"
    **License**: Apache 2.0<br>**Source**: [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) (`hand_landmarker.task`, float16, version 1)

!!! note "Handedness"
    Classes are `left_hand` and `right_hand`, for the person in a normal camera view. MediaPipe itself labels hands as if the image were a mirrored selfie view; pass `image_is_mirrored=True` for that convention.

## Supported Backends

| Backend | Extras Required |
|---------|----------------|
| `onnx` | `onnx-cpu`, `onnx-cu118`, `onnx-cu12`, `onnx-jp6-cu126` |

## Output

One keypoint instance per hand, in image pixels:

- `xy` - 21 landmarks (`wrist`, `thumb_cmc` ... `pinky_tip`, MediaPipe's order).
- `confidence` - the hand's score on every landmark; the model predicts no per-landmark visibility.
- `detection_confidence` - palm (or detector) score times hand presence.
- `key_points_metadata` - per hand: `handedness_confidence`, `hand_presence`, `landmarks_z` (relative depth, in the same pixel scale as `x`) and `world_landmarks` (21 x 3, metres, origin at the hand's centre).

The detections returned alongside are boxes around the landmarks.

## Inference Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `confidence` | `0.5` | Minimum palm (or hand detector) score |
| `max_num_hands` | `2` | Most hands returned per image |
| `hand_presence_threshold` | `0.5` | Minimum hand presence from the landmark model |
| `image_is_mirrored` | `False` | Report MediaPipe's mirrored handedness labels |

## Usage Example

```python
import cv2
from inference_models import AutoModel

model = AutoModel.from_pretrained("mediapipe-hands/full", api_key="your_roboflow_api_key")
image = cv2.imread("path/to/image.jpg")

key_points, detections = model(image, max_num_hands=4)
hands = key_points[0].to_supervision()
```

## Converting the Weights

`development/converters/convert_mediapipe_hands_tflite_to_onnx.py` converts the pinned `hand_landmarker.task` into the three model packages. It needs `tensorflow`, `tf2onnx` and `onnx`, which `inference-models` does not depend on at runtime.

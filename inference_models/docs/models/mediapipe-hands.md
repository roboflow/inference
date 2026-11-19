# MediaPipe Hands - Hand Landmarks

MediaPipe Hands finds hands and returns 21 landmarks per hand (wrist, and four joints for each finger), plus handedness. This is Google's hand landmarker, with its two models converted to ONNX and the steps between them re-implemented in `inference-models`, so it runs without the `mediapipe` library. The models are plain ONNX graphs and should run on any ONNX Runtime execution provider, CUDA and TensorRT on Jetson included, but so far only the CPU provider has been tested.

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
- `key_points_metadata` - per hand: `handedness_confidence`, `hand_presence`, `landmarks_z` (relative depth, in the same pixel scale as `x`), `world_landmarks` (21 x 3, metres, origin at the hand's centre), `hand_rect` (the rotated region the landmarks were predicted from) and `next_hand_rect` (the region those landmarks give for the next video frame). Regions are `[center_x, center_y, width, height, rotation]` in unrounded pixels, rotation in radians.

The detections returned alongside are boxes around the landmarks. The palm detector's own detections carry each palm's hand region as `hand_rect`.

## Video

MediaPipe does not run the palm detector on every video frame. The landmarks found on one frame give the hand regions for the next, and the detector only runs while fewer hands are tracked than asked for. `inference_models.models.mediapipe_hands.tracking` holds those rules (`needs_detection`, `merge_tracked_and_detected`, `deduplicate_hands`); the caller keeps each video's regions. The `roboflow_core/mediapipe_hand_landmarks@v1` Workflow block does this for you.

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

# SAM2 RT - Video Object Tracking & Segmentation

SAM2 RT (Real-Time) is an optimized fork of SAM2 designed for efficient video object tracking and instance segmentation.

SAM2 RT provides:

- **Video object tracking** - Track and segment objects across video frames
- **Stateful tracking** - Maintain object identities across frames
- **Prompt-based initialization** - Start tracking with bounding boxes
- **Instance segmentation** - Segment objects in single images
- **Optimized inference** - Faster than standard SAM2 for video use cases

!!! info "License & Attribution"
    **License**: Apache 2.0
    **Source**: [Segment Anything 2 Real Time fork](https://github.com/Gy920/segment-anything-2-real-time) by Gy920
    **Original**: Based on Meta's Segment Anything 2

## Model IDs

SAM2 RT models do **not** require a Roboflow API key.

| Model Size | Model ID |
|------------|----------|
| Tiny | `Gy920/sam2-1-hiera-tiny` |
| Small | `Gy920/sam2-1-hiera-small` |
| Base+ | `Gy920/sam2-1-hiera-base-plus` |
| Large | `Gy920/sam2-1-hiera-large` |

## Installation

SAM2 RT requires special installation from GitHub:

```bash
# First install inference-models with a CUDA backend (GPU required)
pip install "inference-models[torch-cu128]"  # or torch-cu126, torch-cu124, etc.

# Then install SAM2 Real-Time
pip install git+https://github.com/Gy920/segment-anything-2-real-time.git
```

!!! warning "GPU Required"
    SAM2 RT requires a CUDA-capable GPU. CPU-only installation is not supported.

!!! note "PyPI Restriction"
    Due to PyPI restrictions on Git dependencies, SAM2 Real-Time must be installed separately.

## Supported Backends

| Backend | Extras Required |
|---------|----------------|
| `torch` | `torch-cu118`, `torch-cu124`, `torch-cu126`, `torch-cu128` |

## Roboflow Platform Compatibility

| Feature | Supported |
|---------|-----------|
| **Training** | ❌ No custom training |
| **Upload Weights** | ❌ Not applicable |
| **Serverless API (v2)** | ❌ Not available |
| **Workflows** | ❌ Not available |
| **Edge Deployment (Jetson)** | ❌ Not supported |
| **Self-Hosting** | ✅ Deploy with `inference-models` |

## Video Object Tracking

Track objects across video frames with persistent IDs:

```python
import cv2
import supervision as sv
from inference_models import AutoModel

# Load model (no API key needed)
model = AutoModel.from_pretrained("Gy920/sam2-1-hiera-tiny")

mask_annotator = sv.MaskAnnotator(opacity=0.7, color_lookup=sv.ColorLookup.TRACK)
vid = cv2.VideoCapture("video.mp4")

frame_num = 0
while True:
    is_ok, frame = vid.read()
    if not is_ok:
        break
    
    if frame_num == 0:
        # Initialize tracking with bounding boxes on first frame
        object_ids, masks, state = model.prompt(
            frame, 
            bboxes=[(477, 337, 560, 529), (633, 570, 843, 804)]
        )
        frame_num += 1
    else:
        # Track objects in subsequent frames
        object_ids, masks, state = model.track(frame)
    
    # Visualize results
    detections = sv.Detections(
        xyxy=sv.mask_to_xyxy(masks=masks),
        mask=masks,
        tracker_id=object_ids
    )
    
    annotated_frame = mask_annotator.annotate(scene=frame, detections=detections)
    cv2.imshow("Tracking", annotated_frame)
    cv2.waitKey(1)
```

## Instance Segmentation

Segment objects in a single image:

```python
import cv2
import numpy as np
import supervision as sv
from inference_models import AutoModel

# Load model
model = AutoModel.from_pretrained("sam2-rt-hiera-tiny")

mask_annotator = sv.MaskAnnotator()
box_annotator = sv.BoxAnnotator(color=sv.Color.BLACK)

# Load image
image = cv2.imread("image.jpg")

# Segment with bounding box prompt
masks, object_ids, state = model.prompt(image, bboxes=[(117, 303, 670, 650)])

# Create detections
classes = np.array([0 for _ in masks])
detections = sv.Detections(
    xyxy=sv.mask_to_xyxy(masks=masks),
    mask=masks,
    class_id=classes,
)

# Visualize
annotated_frame = mask_annotator.annotate(scene=image, detections=detections)
annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)

cv2.imshow("Segmentation", annotated_frame)
cv2.waitKey(0)
```

## API Reference

### `prompt(image, bboxes, ...)`

Initialize tracking or segment an image with bounding box prompts. Returns `(masks, object_ids, state)`.

### `track(image, ...)`

Track previously initialized objects in a new frame. Returns `(masks, object_ids, state)`.

!!! warning "Must Call `prompt()` First"
    You must call `prompt()` at least once before calling `track()`. The model needs initial prompts to know what to track.

## Use Cases

SAM2 RT is ideal for:

- ✅ **Video object tracking** - Track multiple objects across video frames
- ✅ **Sports analytics** - Track players, balls, and equipment in sports videos
- ✅ **Surveillance** - Monitor and track objects in security footage
- ✅ **Traffic analysis** - Track vehicles and pedestrians
- ✅ **Wildlife monitoring** - Track animals in nature videos
- ✅ **Interactive video annotation** - Quickly annotate video datasets

## Key Differences from SAM2

| Feature | SAM2 | SAM2 RT |
|---------|------|---------|
| **Primary Use Case** | Interactive image segmentation | Video object tracking |
| **API Key** | Required | Not required |
| **Stateful Tracking** | ❌ No | ✅ Yes |
| **Video Optimization** | ❌ No | ✅ Yes |
| **Caching Support** | ✅ Yes | ❌ No |
| **Point Prompts** | ✅ Yes | ❌ No (boxes only) |
| **Multi-mask Output** | ✅ Yes | ❌ No |

## Performance Tips

1. **Use smaller models for speed** - `hiera-tiny` is fastest for tracking
2. **GPU is essential** - Video tracking requires GPU for acceptable performance
3. **Batch processing** - Process video frames sequentially, don't skip frames
4. **State management** - Keep the state dictionary if you need to pause/resume tracking


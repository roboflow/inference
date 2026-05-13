
# `semantic_segmentation_prediction` Kind

Prediction with per-pixel class label and confidence for semantic segmentation

## Data representation


!!! Warning "Data representation"

    This kind has a different internal and external representation. **External** representation is relevant for 
    integration with your workflow, whereas **internal** one is an implementation detail useful for Workflows
    blocks development.



### External

External data representation is relevant for Workflows clients - it dictates what is the input and output format of
data.

Type: `dict`

### Internal

Internal data representation is relevant for Workflows blocks creators - this is the type that will be provided
by Execution Engine in runtime to the block that consumes input of this kind.

Type: `sv.Detections`

## Details


This kind represents a single semantic segmentation prediction as an
[`sv.Detections(...)`](https://supervision.roboflow.com/latest/detection/core/) object
with one detection per predicted class. Each detection carries an RLE-encoded mask
covering all pixels assigned to that class.

**Why RLE and not polygons:**

Semantic segmentation assigns a class label to every pixel in the image. A single class
can appear in multiple spatially disconnected regions (e.g., two separate "person" regions
on opposite sides of the frame). Polygon-based serialization uses `cv2.findContours()`,
which only retains the first contiguous contour and silently discards all others — causing
irreversible data loss for non-contiguous masks. RLE (Run-Length Encoding, COCO standard)
is a pixel-level encoding that represents the complete mask regardless of spatial topology,
making it the only correct serialization format for semantic segmentation masks.

**Internal representation:** `sv.Detections` with:
- `xyxy` — tight bounding box enclosing all pixels of the class
- `class_id` — integer class ID
- `confidence` — mean confidence over all pixels of the class
- `data["class_name"]` — class label string
- `data["rle_mask"]` — numpy object array of COCO RLE dicts `{"size": [H, W], "counts": "..."}`

**Serialised format** (one entry per class in `predictions`):

```json
{
    "image": {"width": 640, "height": 480},
    "predictions": [
        {
            "x": 320.0, "y": 240.0, "width": 200.0, "height": 180.0,
            "confidence": 0.92,
            "class_id": 1,
            "class": "person",
            "detection_id": "a1b2c3d4-...",
            "rle_mask": {"size": [480, 640], "counts": "XYZ..."}
        }
    ]
}
```

**Decoding RLE masks:**

```python
import pycocotools.mask as mask_utils
import numpy as np

rle = prediction["rle_mask"]
binary_mask = mask_utils.decode(rle).astype(bool)  # shape: (H, W)
```


<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>

# Workflow Blocks: Segmentation/Detections Input Patterns

This document catalogs how workflow blocks in `inference/core/workflows/core_steps/` declare input parameters for segmentation data and Detections. All examples use `Selector` with specific `kind` constraints.

## Common Pattern

Blocks accept detections/segmentation via `Selector` fields with `kind` specifying which prediction types are accepted:

```python
from inference.core.workflows.execution_engine.entities.types import (
    OBJECT_DETECTION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
)
from inference.core.workflows.execution_engine.entities.types import Selector

field_name: Selector(kind=[...]) = Field(...)
```

---

## Pattern 1: Simple Detections Input

**Used for**: Blocks that accept object detection OR instance segmentation predictions as a single input.

### File: [analytics/path_deviation/v2.py](inference/core/workflows/core_steps/analytics/path_deviation/v2.py#L129)

```python
class PathDeviationManifest(WorkflowBlockManifest):
    detections: Selector(
        kind=[
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
        ]
    ) = Field(
        description="Tracked object detection or instance segmentation predictions. Must include tracker_id information from a tracking block. The block tracks anchor point positions across frames to build object trajectories and compares them against the reference path. Output detections include path_deviation metadata containing the Fréchet distance from the reference path.",
        examples=["$steps.object_detection_model.predictions"],
    )
```

**Type Annotation**: `Selector(kind=[OBJECT_DETECTION_PREDICTION_KIND, INSTANCE_SEGMENTATION_PREDICTION_KIND])`

### Similar Blocks Using This Pattern:

1. **[Line Counter (v2)](inference/core/workflows/core_steps/analytics/line_counter/v2.py#L141)**
   - Field: `detections`
   - Types: `[OBJECT_DETECTION_PREDICTION_KIND, INSTANCE_SEGMENTATION_PREDICTION_KIND]`
   - Purpose: Count objects crossing a line in video

2. **[Detections Stabilizer (v1)](inference/core/workflows/core_steps/transformations/stabilize_detections/v1.py#L101)**
   - Field: `detections`
   - Types: `[OBJECT_DETECTION_PREDICTION_KIND, INSTANCE_SEGMENTATION_PREDICTION_KIND]`
   - Purpose: Apply smoothing to reduce noise/flickering

3. **[Time in Zone (v3)](inference/core/workflows/core_steps/analytics/time_in_zone/v3.py#L188)**
   - Field: `detections`
   - Types: `[OBJECT_DETECTION_PREDICTION_KIND, INSTANCE_SEGMENTATION_PREDICTION_KIND]`
   - Purpose: Track time objects spend in zones

---

## Pattern 2: Segmentation-Specific Input

**Used for**: Blocks that specifically work with instance segmentation masks.

### File: [classical_cv/mask_edge_snap/v1.py](inference/core/workflows/core_steps/classical_cv/mask_edge_snap/v1.py#L134)

```python
class MaskEdgeSnapManifest(WorkflowBlockManifest):
    segmentation: Selector(kind=[INSTANCE_SEGMENTATION_PREDICTION_KIND]) = Field(
        title="Segmentation",
        description="Instance segmentation predictions with mask field populated. Each mask contour will be snapped to detected edges. If empty, segmentation is passed through unchanged.",
        examples=["$steps.segmentation_model.predictions"],
    )
```

**Type Annotation**: `Selector(kind=[INSTANCE_SEGMENTATION_PREDICTION_KIND])`

### Similar Blocks Using This Pattern:

1. **[Mask Area Measurement (v1)](inference/core/workflows/core_steps/classical_cv/mask_area_measurement/v1.py#L95)**
   - Field: `predictions`
   - Types: `[INSTANCE_SEGMENTATION_PREDICTION_KIND, OBJECT_DETECTION_PREDICTION_KIND]`
   - Purpose: Measure area of masks/bounding boxes

---

## Pattern 3: Multiple Detection Inputs (Fusion)

**Used for**: Blocks that accept multiple detection sets for merging/consensus operations.

### File: [fusion/detections_consensus/v1.py](inference/core/workflows/core_steps/fusion/detections_consensus/v1.py#L120)

```python
class BlockManifest(WorkflowBlockManifest):
    predictions_batches: List[
        Selector(
            kind=[
                OBJECT_DETECTION_PREDICTION_KIND,
                INSTANCE_SEGMENTATION_PREDICTION_KIND,
                KEYPOINT_DETECTION_PREDICTION_KIND,
            ]
        ),
    ] = Field(
        min_items=1,
        description="List of references to detection predictions from multiple models. Each model's predictions must be made against the same input image. Predictions can be from object detection, instance segmentation, or keypoint detection models. The block matches overlapping detections across models...",
        examples=[["$steps.a.predictions", "$steps.b.predictions"]],
        validation_alias=AliasChoices("predictions_batches", "predictions"),
    )
```

**Type Annotation**: `List[Selector(kind=[...PREDICTION_KINDS...])]`
**Key Detail**: Uses `validation_alias` to accept both `predictions_batches` and `predictions`

### Similar Blocks Using This Pattern:

1. **[Detections Combine (v1)](inference/core/workflows/core_steps/transformations/detections_combine/v1.py#L87)**
   - Fields: `prediction_one`, `prediction_two` (two separate Selector fields)
   - Types: `[OBJECT_DETECTION_PREDICTION_KIND, INSTANCE_SEGMENTATION_PREDICTION_KIND]`
   - Purpose: Merge two detection sets

---

## Pattern 4: Optional Detection Input

**Used for**: Blocks where detection input is optional.

### File: [transformations/perspective_correction/v1.py](inference/core/workflows/core_steps/transformations/perspective_correction/v1.py#L124)

```python
class PerspectiveCorrectionManifest(WorkflowBlockManifest):
    predictions: Optional[
        Selector(
            kind=[
                OBJECT_DETECTION_PREDICTION_KIND,
                INSTANCE_SEGMENTATION_PREDICTION_KIND,
            ]
        )
    ] = Field(
        description="Optional object detection or instance segmentation predictions to transform. If provided, bounding boxes, masks, and keypoints are transformed to the top-down coordinate space. If not provided, only image warping is performed (if enabled).",
        default=None,
        examples=[
            "$steps.object_detection_model.predictions",
            "$steps.instance_segmentation_model.predictions",
        ],
    )
```

**Type Annotation**: `Optional[Selector(kind=[...])]`

### Similar Blocks Using This Pattern:

1. **[Dynamic Crop (v1)](inference/core/workflows/core_steps/transformations/dynamic_crop/v1.py#L107)**
   - Field: `predictions`
   - Types: `[OBJECT_DETECTION_PREDICTION_KIND, INSTANCE_SEGMENTATION_PREDICTION_KIND, KEYPOINT_DETECTION_PREDICTION_KIND]`
   - Note: Uses `validation_alias=AliasChoices("predictions", "detections")`

---

## Pattern 5: Single-Class Reference (Overlap/Filter)

**Used for**: Blocks that filter detections based on overlap with a specific class.

### File: [analytics/overlap/v1.py](inference/core/workflows/core_steps/analytics/overlap/v1.py#L82)

```python
class OverlapManifest(WorkflowBlockManifest):
    predictions: Selector(
        kind=[OBJECT_DETECTION_PREDICTION_KIND, INSTANCE_SEGMENTATION_PREDICTION_KIND]
    ) = Field(
        description="Detection predictions (object detection or instance segmentation) containing objects that may overlap with the specified overlap class. The block identifies detections matching the overlap_class_name and finds other detections that spatially overlap with them.",
        examples=["$steps.object_detection_model.predictions"],
    )
```

**Type Annotation**: `Selector(kind=[OBJECT_DETECTION_PREDICTION_KIND, INSTANCE_SEGMENTATION_PREDICTION_KIND])`

---

## Complete Field Declaration Checklist

When declaring a Detections/Segmentation input field, use:

```python
# Basic declaration
field_name: Selector(
    kind=[
        OBJECT_DETECTION_PREDICTION_KIND,
        INSTANCE_SEGMENTATION_PREDICTION_KIND,
    ]
) = Field(
    description="...",
    examples=["$steps.model.predictions"],
)

# Optional field
field_name: Optional[Selector(kind=[...])] = Field(
    default=None,
    description="...",
    examples=[...],
)

# Multiple inputs (list)
field_name: List[Selector(kind=[...])] = Field(
    min_items=1,
    description="...",
    examples=[[...]],
    validation_alias=AliasChoices("field_name", "alternate_name"),
)

# With validation alias for backward compatibility
field_name: Selector(kind=[...]) = Field(
    description="...",
    examples=[...],
    validation_alias=AliasChoices("field_name", "old_field_name"),
)
```

---

## Required Imports

All blocks using these patterns require:

```python
from typing import List, Optional, Union
from pydantic import AliasChoices, Field

from inference.core.workflows.execution_engine.entities.types import (
    OBJECT_DETECTION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    Selector,
)
```

---

## Output Pattern

When returning Detections/Segmentation predictions, blocks use `describe_outputs()`:

```python
@classmethod
def describe_outputs(cls) -> List[OutputDefinition]:
    return [
        OutputDefinition(
            name="predictions",
            kind=[
                OBJECT_DETECTION_PREDICTION_KIND,
                INSTANCE_SEGMENTATION_PREDICTION_KIND,
            ],
        ),
    ]
```

---

## Runtime Type

At runtime, these `Selector` fields receive `sv.Detections` objects (from the `supervision` library):

```python
from supervision import Detections as sv.Detections

def run(
    self,
    detections: sv.Detections,  # Runtime type
) -> BlockResult:
    # Process detections
    return {"predictions": detections}
```

The `sv.Detections` class contains:
- `xyxy`: bounding box coordinates
- `mask`: segmentation masks (if instance segmentation)
- `class_id`: class indices
- `confidence`: detection confidence scores
- `tracker_id`: tracking IDs (if tracking enabled)
- Custom fields via `.data` attribute dictionary

from typing import List, Literal, Optional, Type, Union

import cv2
import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field

from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    CONTOURS_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

SHORT_DESCRIPTION = (
    "Measure the distance from detection centers to the nearest contour."
)

LONG_DESCRIPTION = """
Measure the distance from each detection's center point to the nearest contour boundary, flagging detections that fall within a configurable pixel distance threshold as "close to edge," and returning filtered detections, per-detection distances, and annotated detections with a boolean close_to_edge flag for quality inspection, defect analysis, and proximity detection workflows.

## How This Block Works

This block measures the distance from each detected object's center point to the nearest contour boundary, then flags detections that are within a configurable pixel distance threshold. The block:

1. Receives detection predictions (from an object detection or instance segmentation model) and contours (from the Image Contours block)
2. Handles empty inputs gracefully:
   - If there are no detections or no contours, returns empty results with an empty `close_to_edge` detections set, an empty `distances` list, and all input detections annotated with an empty `close_to_edge` flag array
3. For each detection, calculates the center point of its bounding box:
   - Extracts bounding box coordinates (x_min, y_min, x_max, y_max) from the detection
   - Computes the center as ((x_min + x_max) / 2, (y_min + y_max) / 2)
4. Measures the distance from each detection center to every contour using OpenCV's `cv2.pointPolygonTest`:
   - `cv2.pointPolygonTest` returns a signed distance: positive if the point is inside the contour, negative if outside, and zero if on the contour edge
   - The block takes the absolute value of the distance to treat inside and outside distances equivalently (the goal is to measure proximity to the contour boundary regardless of which side the point is on)
   - Iterates through all contours and keeps the minimum absolute distance to find the nearest contour boundary for each detection
5. Applies the distance threshold to classify each detection:
   - If the minimum distance to the nearest contour is less than or equal to `distance_threshold`, the detection is flagged as "close to edge"
   - Builds a boolean mask (`close_mask`) indicating which detections are close to an edge
6. Produces three outputs:
   - **close_to_edge**: A filtered set of detections containing only those flagged as close to a contour boundary
   - **distances**: A list of float values representing the minimum distance from each detection's center to the nearest contour (one value per input detection, in pixel units)
   - **all_detections_with_flag**: A copy of all input detections with an additional `close_to_edge` boolean field in the detection data, allowing downstream blocks to filter or branch based on proximity

## Common Use Cases

- **Wood Block Defect Inspection**: Detect defects near the edges of wood blocks on a conveyor belt by running a defect detection model and a contour detection pipeline on the same image, then using this block to identify which defects are within a specified pixel distance of the wood block boundary, enabling edge-specific quality control decisions
- **Edge Quality Control**: Flag defects or features that are too close to product boundaries in manufacturing inspection workflows, where parts must have defect-free margins around their edges to meet quality standards, enabling automated pass/fail determination based on defect proximity to product edges
- **Manufacturing Inspection**: Identify items, components, or defects near the edges of containers, trays, or other bounded regions on production lines, enabling automated quality decisions based on whether detected objects are within acceptable margins of container boundaries
- **Proximity Detection**: Determine if detected objects are close to boundaries defined by contours in any image analysis pipeline, enabling spatial reasoning about the relationship between detected objects and shape boundaries for general-purpose proximity workflows

## Connecting to Other Blocks

This block receives detection predictions and contour data, and produces filtered detections and distance measurements:

- **After object detection or instance segmentation model blocks** (e.g., Object Detection Model, Instance Segmentation Model) to receive detection predictions representing the objects whose proximity to contour boundaries should be measured, enabling detection-to-distance measurement pipelines
- **After the Image Contours block** to receive contour data representing the boundaries (edges) of regions of interest in the image, enabling contour-to-distance measurement pipelines (the Image Contours block should be preceded by an Image Threshold block to produce a binary image suitable for contour detection)
- **Before logic blocks** (e.g., Continue If, Expression) to make decisions based on whether detections are close to edges or based on specific distance values, enabling proximity-based conditional workflows
- **Before notification or sink blocks** (e.g., Email Notification, Webhook, Local File Sink) to alert operators or log results when defects are detected near edges, enabling automated alerting and data recording for quality control workflows
- **Before visualization blocks** to display only the close-to-edge detections on the image, enabling visual inspection of edge-proximity results
- **In quality control pipelines** as a key filtering step between detection and decision-making, where the full pipeline typically includes: image input, object detection model, image threshold, image contours, this block (detection to contour distance), and then logic/notification blocks for pass/fail determination

## Usage Examples

**Minimal Configuration** (required fields only, using default distance threshold of 50 pixels):

```json
{
    "version": "1.0",
    "inputs": [
        {"type": "InferenceImage", "name": "image"}
    ],
    "steps": [
        {
            "type": "roboflow_core/object_detection_model@v2",
            "name": "defect_detector",
            "image": "$inputs.image",
            "model_id": "your-defect-model/1"
        },
        {
            "type": "roboflow_core/threshold@v1",
            "name": "image_threshold",
            "image": "$inputs.image",
            "thresh_value": 127,
            "threshold_type": "binary"
        },
        {
            "type": "roboflow_core/contours_detection@v1",
            "name": "contours",
            "image": "$steps.image_threshold.image",
            "line_thickness": 3
        },
        {
            "type": "roboflow_core/detection_to_contour_distance@v1",
            "name": "edge_distance",
            "predictions": "$steps.defect_detector.predictions",
            "contours": "$steps.contours.contours"
        }
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "close_to_edge",
            "selector": "$steps.edge_distance.close_to_edge"
        },
        {
            "type": "JsonField",
            "name": "distances",
            "selector": "$steps.edge_distance.distances"
        }
    ]
}
```

**Full Configuration** (all fields populated, with custom distance threshold):

```json
{
    "version": "1.0",
    "inputs": [
        {"type": "InferenceImage", "name": "image"},
        {"type": "InferenceParameter", "name": "distance_threshold", "default_value": 30}
    ],
    "steps": [
        {
            "type": "roboflow_core/object_detection_model@v2",
            "name": "defect_detector",
            "image": "$inputs.image",
            "model_id": "wood-defect-detection/2"
        },
        {
            "type": "roboflow_core/convert_grayscale@v1",
            "name": "grayscale",
            "image": "$inputs.image"
        },
        {
            "type": "roboflow_core/image_blur@v1",
            "name": "blur",
            "image": "$steps.grayscale.image"
        },
        {
            "type": "roboflow_core/threshold@v1",
            "name": "threshold",
            "image": "$steps.blur.image",
            "thresh_value": 200,
            "threshold_type": "binary_inv"
        },
        {
            "type": "roboflow_core/contours_detection@v1",
            "name": "contours",
            "image": "$steps.threshold.image",
            "line_thickness": 3
        },
        {
            "type": "roboflow_core/detection_to_contour_distance@v1",
            "name": "edge_distance",
            "predictions": "$steps.defect_detector.predictions",
            "contours": "$steps.contours.contours",
            "distance_threshold": "$inputs.distance_threshold"
        }
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "close_to_edge_detections",
            "selector": "$steps.edge_distance.close_to_edge"
        },
        {
            "type": "JsonField",
            "name": "distances",
            "selector": "$steps.edge_distance.distances"
        },
        {
            "type": "JsonField",
            "name": "all_detections_flagged",
            "selector": "$steps.edge_distance.all_detections_with_flag"
        }
    ]
}
```

## Requirements

- **Detection Predictions**: Input detections must come from an object detection or instance segmentation model block. The detections must contain bounding box coordinates (xyxy format) so that center points can be computed.
- **Contour Data**: Input contours must come from the Image Contours block. Each contour should be a numpy array of shape (N, 1, 2) with int32 dtype, as returned by `cv2.findContours`. The input image for contour detection should be processed through an Image Threshold block first to produce a binary image suitable for reliable contour extraction.
- **Distance Threshold**: The `distance_threshold` parameter is specified in pixels. Choose a threshold value appropriate for your image resolution and use case. Larger thresholds will flag more detections as "close to edge," while smaller thresholds will be more selective.
- **Execution Engine Compatibility**: Requires workflow execution engine version >=1.3.0 and <2.0.0.
"""


class DetectionToContourDistanceManifest(WorkflowBlockManifest):
    type: Literal["roboflow_core/detection_to_contour_distance@v1"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Detection to Contour Distance",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "classical_computer_vision",
            "ui_manifest": {
                "section": "classical_cv",
                "icon": "far fa-ruler-combined",
                "blockPriority": 12,
                "opencv": True,
            },
        }
    )

    predictions: Selector(
        kind=[
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
        ]
    ) = Field(
        title="Detections",
        description="Detection predictions (e.g. defects) whose center points will "
        "be measured against the nearest contour.",
        examples=["$steps.model.predictions"],
    )

    contours: Selector(kind=[CONTOURS_KIND]) = Field(
        title="Contours",
        description="Contours produced by the Image Contours block. Each contour is "
        "a numpy array of shape (N, 1, 2) with int32 dtype, as returned by "
        "cv2.findContours.",
        examples=["$steps.contours.contours"],
    )

    distance_threshold: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        title="Distance Threshold (px)",
        description="Maximum pixel distance from a detection center to the nearest "
        "contour for the detection to be considered 'close to edge'.",
        default=50,
        examples=[50, "$inputs.distance_threshold"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="close_to_edge",
                kind=[
                    OBJECT_DETECTION_PREDICTION_KIND,
                    INSTANCE_SEGMENTATION_PREDICTION_KIND,
                ],
            ),
            OutputDefinition(
                name="distances",
                kind=[LIST_OF_VALUES_KIND],
            ),
            OutputDefinition(
                name="all_detections_with_flag",
                kind=[
                    OBJECT_DETECTION_PREDICTION_KIND,
                    INSTANCE_SEGMENTATION_PREDICTION_KIND,
                ],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class DetectionToContourDistanceBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[DetectionToContourDistanceManifest]:
        return DetectionToContourDistanceManifest

    def run(
        self,
        predictions: sv.Detections,
        contours: list,
        distance_threshold: int,
    ) -> BlockResult:
        if len(predictions) == 0 or len(contours) == 0:
            predictions["close_to_edge"] = np.array([], dtype=bool)
            return {
                "close_to_edge": sv.Detections.empty(),
                "distances": [],
                "all_detections_with_flag": predictions,
            }

        distances: List[float] = []
        close_mask: List[bool] = []

        for bbox in predictions.xyxy:
            x_min, y_min, x_max, y_max = bbox
            center = (float((x_min + x_max) / 2), float((y_min + y_max) / 2))

            min_dist = float("inf")
            for contour in contours:
                dist = cv2.pointPolygonTest(contour, center, measureDist=True)
                abs_dist = abs(dist)
                if abs_dist < min_dist:
                    min_dist = abs_dist

            distances.append(min_dist)
            close_mask.append(min_dist <= distance_threshold)

        close_mask_np = np.array(close_mask, dtype=bool)

        all_with_flag = sv.Detections(
            xyxy=predictions.xyxy.copy(),
            mask=predictions.mask.copy() if predictions.mask is not None else None,
            confidence=predictions.confidence.copy()
            if predictions.confidence is not None
            else None,
            class_id=predictions.class_id.copy()
            if predictions.class_id is not None
            else None,
            tracker_id=predictions.tracker_id.copy()
            if predictions.tracker_id is not None
            else None,
            data={k: v.copy() if hasattr(v, "copy") else v for k, v in predictions.data.items()},
        )
        all_with_flag["close_to_edge"] = close_mask_np

        close_detections = predictions[close_mask_np]

        return {
            "close_to_edge": close_detections,
            "distances": distances,
            "all_detections_with_flag": all_with_flag,
        }

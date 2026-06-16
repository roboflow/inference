from typing import List, Literal, Optional, Tuple, Type, Union
from uuid import uuid4

import torch
from pydantic import ConfigDict, Field

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections

from inference.core.env import WORKFLOWS_IMAGE_TENSOR_DEVICE
from inference.core.workflows.execution_engine.constants import (
    CLASS_NAMES_KEY,
    DETECTION_ID_KEY,
    IMAGE_DIMENSIONS_KEY,
    PARENT_COORDINATES_KEY,
    PARENT_DIMENSIONS_KEY,
    PARENT_ID_KEY,
    PREDICTION_TYPE_KEY,
    ROOT_PARENT_COORDINATES_KEY,
    ROOT_PARENT_DIMENSIONS_KEY,
    ROOT_PARENT_ID_KEY,
    SCALING_RELATIVE_TO_PARENT_KEY,
    SCALING_RELATIVE_TO_ROOT_PARENT_KEY,
)
from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.tensor_native_types import (
    TENSOR_NATIVE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    TENSOR_NATIVE_KEYPOINT_DETECTION_PREDICTION_KIND,
    TENSOR_NATIVE_OBJECT_DETECTION_PREDICTION_KIND,
)
from inference.core.workflows.execution_engine.entities.types import Selector
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

OUTPUT_KEY: str = "predictions"

SHORT_DESCRIPTION = "Merge multiple detections into a single bounding box."
LONG_DESCRIPTION = """
Combine multiple detection predictions into a single merged detection with a union bounding box that encompasses all input detections, simplifying multiple detections into one larger detection region for overlapping object consolidation, region creation from multiple objects, and detection simplification workflows.

## How This Block Works

This block merges multiple detections into a single detection by calculating a union bounding box that contains all input detections. The block:

1. Receives detection predictions (object detection, instance segmentation, or keypoint detection) containing multiple detections
2. Validates input (handles empty detections by returning an empty detection result)
3. Calculates the union bounding box from all input detections:
   - Extracts all bounding box coordinates (xyxy format) from input detections
   - Finds the minimum x and y coordinates (leftmost and topmost points) across all boxes
   - Finds the maximum x and y coordinates (rightmost and bottommost points) across all boxes
   - Creates a single bounding box that completely encompasses all input detections
4. Determines the merged detection's confidence:
   - Finds the detection with the lowest confidence score among all input detections
   - Uses this lowest confidence as the merged detection's confidence (conservative approach)
   - Handles cases where confidence scores may not be present
5. Creates a new merged detection with:
   - The calculated union bounding box (encompasses all input detections)
   - A customizable class name (default: "merged_detection", configurable via class_name parameter)
   - The lowest confidence from input detections (conservative confidence assignment)
   - A fixed class_id of 0 for the merged detection
   - A newly generated detection ID (unique identifier for the merged detection)
6. Returns the single merged detection containing all input detections within its bounding box

The block creates a unified bounding box representation of multiple detections, useful for consolidating overlapping or nearby detections into a single region. The union bounding box approach ensures all original detections are completely contained within the merged detection. By using the lowest confidence, the block adopts a conservative approach, ensuring the merged detection's confidence reflects the least certain input detection. The merged detection can be customized with a class name to indicate its merged nature or to represent a specific category.

## Common Use Cases

- **Overlapping Detection Consolidation**: Merge multiple overlapping detections of the same or related objects into a single unified detection (e.g., merge overlapping detections of the same person from multiple frames, consolidate duplicate detections from different models, combine overlapping object parts into one detection), enabling overlapping detection simplification
- **Multi-Object Region Creation**: Create a single bounding box region that encompasses multiple detected objects for area-based analysis (e.g., create a region containing multiple people for crowd analysis, merge detections of objects in a scene into one region, combine multiple detections into a single monitoring zone), enabling multi-object region workflows
- **Nearby Detection Grouping**: Group nearby detections together into a single merged detection (e.g., merge detections of objects close to each other, group nearby detections into clusters, combine adjacent detections for simplified processing), enabling spatial grouping workflows
- **Detection Simplification**: Simplify multiple detections into one larger detection for downstream processing (e.g., reduce multiple detections to one for simpler analysis, consolidate detections for easier visualization, merge detections for streamlined workflows), enabling detection simplification workflows
- **Zone Definition from Detections**: Create zone boundaries from multiple detection locations (e.g., define zones based on detection locations, create regions from detected object positions, establish boundaries from detection clusters), enabling zone creation from detections
- **Redundant Detection Removal**: Merge redundant or duplicate detections into a single representation (e.g., combine duplicate detections from different stages, merge redundant object detections, consolidate repeated detections), enabling redundant detection consolidation workflows

## Connecting to Other Blocks

This block receives multiple detection predictions and produces a single merged detection:

- **After detection blocks** (e.g., Object Detection, Instance Segmentation, Keypoint Detection) to merge multiple detections into one unified detection for simplified processing, enabling detection consolidation workflows
- **After filtering blocks** (e.g., Detections Filter) to merge filtered detections that meet specific criteria into a single detection (e.g., merge filtered detections by class, combine detections after filtering, consolidate filtered results), enabling filtered detection consolidation
- **Before crop blocks** to create a single crop region from multiple detections (e.g., crop a region containing multiple objects, extract area encompassing multiple detections, create unified crop region), enabling multi-detection region extraction
- **Before zone-based blocks** (e.g., Polygon Zone, Dynamic Zone) to define zones based on merged detection regions (e.g., create zones from merged detection areas, establish monitoring zones from merged detections, define regions from consolidated detections), enabling zone creation from merged detections
- **Before visualization blocks** to display simplified merged detections instead of multiple individual detections (e.g., visualize consolidated detection regions, display merged bounding boxes, show simplified detection representation), enabling simplified visualization outputs
- **Before analysis blocks** that benefit from simplified detection representation (e.g., analyze merged detection regions, process consolidated detections, work with simplified detection data), enabling simplified detection analysis workflows
"""

# Native tensor-data input/output shapes. The merge always collapses to a single
# object-detection bbox (no mask, no keypoints) regardless of the input shape, so
# the output kind is object-detection. The keypoint-detection input arrives as a
# Tuple[KeyPoints, Optional[Detections]]; its bounding-box component supplies xyxy.
TensorNativeDetections = Union[Detections, InstanceDetections]
KeyPointPrediction = Tuple[KeyPoints, Optional[Detections]]
TensorNativeMergeInput = Union[Detections, InstanceDetections, KeyPointPrediction]


class DetectionsMergeManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Detections Merge",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
            "ui_manifest": {
                "section": "transformation",
                "icon": "fal fa-object-union",
                "blockPriority": 5,
            },
        }
    )
    type: Literal["roboflow_core/detections_merge@v1"]
    predictions: Selector(
        kind=[
            TENSOR_NATIVE_OBJECT_DETECTION_PREDICTION_KIND,
            TENSOR_NATIVE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
            TENSOR_NATIVE_KEYPOINT_DETECTION_PREDICTION_KIND,
        ]
    ) = Field(
        description="Detection predictions containing multiple detections to merge into a single detection. Supports object detection, instance segmentation, or keypoint detection predictions. All input detections will be combined into one merged detection with a union bounding box that encompasses all input detections. If empty detections are provided, the block returns an empty detection result. The merged detection will contain all input detections within its bounding box boundaries.",
        examples=["$steps.object_detection_model.predictions"],
    )
    class_name: str = Field(
        default="merged_detection",
        description="Class name to assign to the merged detection. The merged detection will use this class name in its data. Default is 'merged_detection' to indicate that this is a merged detection. You can customize this to represent a specific category or to indicate the purpose of the merged detection (e.g., 'crowd', 'group', 'region'). This class name will be stored in the detection's data dictionary.",
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name=OUTPUT_KEY,
                kind=[TENSOR_NATIVE_OBJECT_DETECTION_PREDICTION_KIND],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


def _extract_bbox_detections(
    predictions: TensorNativeMergeInput,
) -> Optional[TensorNativeDetections]:
    """Return the bounding-box ``Detections`` carrier for any supported native
    input. The merge only ever looks at xyxy / confidence, so the keypoint tuple
    is reduced to its ``Detections`` component (which may be ``None`` / empty)."""
    if isinstance(predictions, tuple):
        _, detections = predictions
        return detections
    return predictions


def _build_merged_image_metadata(
    source_metadata: Optional[dict],
    class_id: int,
    class_name: str,
) -> dict:
    """Build the producer ``image_metadata`` for the merged single-row detection.

    The merged row is a fresh object-detection prediction, so ``class_names`` is
    overridden to ``{class_id: class_name}`` and ``prediction_type`` becomes
    ``object-detection``. Parent / root lineage and image dimensions are carried
    over from the source prediction (shared across the same input image), mirroring
    the convention used by the tensor-native detections-consensus block.
    """
    source_metadata = source_metadata or {}
    image_metadata = {
        CLASS_NAMES_KEY: {int(class_id): class_name},
        PARENT_ID_KEY: source_metadata.get(PARENT_ID_KEY),
        PREDICTION_TYPE_KEY: "object-detection",
        PARENT_COORDINATES_KEY: source_metadata.get(PARENT_COORDINATES_KEY),
        PARENT_DIMENSIONS_KEY: source_metadata.get(PARENT_DIMENSIONS_KEY),
        ROOT_PARENT_ID_KEY: source_metadata.get(ROOT_PARENT_ID_KEY),
        ROOT_PARENT_COORDINATES_KEY: source_metadata.get(ROOT_PARENT_COORDINATES_KEY),
        ROOT_PARENT_DIMENSIONS_KEY: source_metadata.get(ROOT_PARENT_DIMENSIONS_KEY),
        IMAGE_DIMENSIONS_KEY: source_metadata.get(IMAGE_DIMENSIONS_KEY),
    }
    image_metadata[SCALING_RELATIVE_TO_PARENT_KEY] = source_metadata.get(
        SCALING_RELATIVE_TO_PARENT_KEY, 1.0
    )
    image_metadata[SCALING_RELATIVE_TO_ROOT_PARENT_KEY] = source_metadata.get(
        SCALING_RELATIVE_TO_ROOT_PARENT_KEY, 1.0
    )
    return image_metadata


def _empty_detections(class_name: str) -> Detections:
    """Empty merged result — built on WORKFLOWS_IMAGE_TENSOR_DEVICE because there
    is no source row whose device we could inherit."""
    return Detections(
        xyxy=torch.zeros(
            (0, 4), dtype=torch.float32, device=WORKFLOWS_IMAGE_TENSOR_DEVICE
        ),
        class_id=torch.zeros(
            (0,), dtype=torch.long, device=WORKFLOWS_IMAGE_TENSOR_DEVICE
        ),
        confidence=torch.zeros(
            (0,), dtype=torch.float32, device=WORKFLOWS_IMAGE_TENSOR_DEVICE
        ),
        image_metadata=_build_merged_image_metadata(
            source_metadata=None, class_id=0, class_name=class_name
        ),
        bboxes_metadata=None,
    )


class DetectionsMergeBlockV1(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return DetectionsMergeManifest

    def run(
        self,
        predictions: TensorNativeMergeInput,
        class_name: str = "merged_detection",
    ) -> BlockResult:
        detections = _extract_bbox_detections(predictions)
        if (
            detections is None
            or detections.xyxy is None
            or int(detections.xyxy.shape[0]) == 0
        ):
            return {OUTPUT_KEY: _empty_detections(class_name=class_name)}

        device = detections.xyxy.device

        # Union bounding box: leftmost/topmost via torch.min over column 0/1, and
        # rightmost/bottommost via torch.max over column 2/3 of the xyxy rows.
        x1 = torch.min(detections.xyxy[:, 0])
        y1 = torch.min(detections.xyxy[:, 1])
        x2 = torch.max(detections.xyxy[:, 2])
        y2 = torch.max(detections.xyxy[:, 3])
        union_bbox = torch.stack([x1, y1, x2, y2]).reshape(1, 4).to(torch.float32)

        # Conservative confidence: the lowest-confidence input row's score.
        if detections.confidence is not None:
            lowest_conf_idx = int(torch.argmin(detections.confidence))
            confidence = detections.confidence[lowest_conf_idx].reshape(1).to(
                device=device, dtype=torch.float32
            )
        else:
            confidence = None

        merged_detection = Detections(
            xyxy=union_bbox.to(device=device),
            class_id=torch.zeros((1,), dtype=torch.long, device=device),
            confidence=confidence,
            image_metadata=_build_merged_image_metadata(
                source_metadata=detections.image_metadata,
                class_id=0,
                class_name=class_name,
            ),
            bboxes_metadata=[{DETECTION_ID_KEY: str(uuid4())}],
        )
        return {OUTPUT_KEY: merged_detection}

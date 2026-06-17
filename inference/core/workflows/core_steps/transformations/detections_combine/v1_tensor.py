from typing import Dict, List, Literal, Optional, Type, Union
from uuid import uuid4

import numpy as np
import torch
from pydantic import ConfigDict, Field

from inference.core.env import WORKFLOWS_IMAGE_TENSOR_DEVICE
from inference.core.workflows.core_steps.common.tensor_native import (
    instance_mask_to_numpy,
)
from inference.core.workflows.execution_engine.constants import (
    CLASS_NAMES_KEY,
    DETECTION_ID_KEY,
)
from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.tensor_native_types import (
    TENSOR_NATIVE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    TENSOR_NATIVE_OBJECT_DETECTION_PREDICTION_KIND,
)
from inference.core.workflows.execution_engine.entities.types import Selector
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks

LONG_DESCRIPTION = """
Combine two sets of detection predictions into a single unified set of detections by merging both detection sets together, preserving all detections from both inputs for multi-source detection aggregation, combining results from multiple models, and consolidating detection sets from different processing stages into one workflow output.

## How This Block Works

This block combines two separate sets of detection predictions into a single unified detection set by merging all detections from both inputs. The block:

1. Receives two separate detection prediction sets (prediction_one and prediction_two), each containing multiple detections from object detection or instance segmentation models
2. Processes both detection sets independently (each set maintains its own detections, properties, masks, and metadata)
3. Merges the two detection sets using supervision's Detections.merge() method:
   - Combines all detections from prediction_one with all detections from prediction_two
   - Preserves all detection properties from both sets (bounding boxes, masks, classes, confidence scores, metadata)
   - Maintains detection order (typically prediction_one detections followed by prediction_two detections)
   - Handles all detection attributes including masks (for instance segmentation), keypoints, class IDs, class names, confidence scores, and custom data fields
4. Returns a single unified detection set containing all detections from both inputs

The block simply concatenates the two detection sets together, preserving all detections and their properties from both sources. Unlike the Detections Merge block (which creates a union bounding box from multiple detections), this block maintains all individual detections from both sets in the output. This is useful for combining detections from different models, different processing stages, or different detection sources into a single workflow stream for unified downstream processing.

## Common Use Cases

- **Multi-Model Detection Aggregation**: Combine detections from multiple detection models into a single unified set (e.g., combine detections from different object detection models, merge results from specialized models, aggregate detections from multiple model outputs), enabling multi-model detection workflows
- **Multi-Stage Detection Combination**: Combine detections from different processing stages or workflow branches (e.g., merge detections from different workflow paths, combine initial detections with refined detections, aggregate detections from multiple processing stages), enabling multi-stage detection aggregation
- **Detection Source Consolidation**: Consolidate detections from different sources or inputs into one set (e.g., combine detections from multiple images or frames, merge detections from different regions, aggregate detections from various sources), enabling detection source unification
- **Classification and Detection Combination**: Combine object detection results with classification results or other detection types (e.g., merge object detections with classification outputs, combine different detection types, aggregate complementary detection sets), enabling multi-type detection workflows
- **Filtered and Unfiltered Detection Combination**: Combine filtered detections with unfiltered detections or combine different filtered subsets (e.g., merge filtered detections by different criteria, combine specific class detections with general detections, aggregate different filtered detection sets), enabling flexible detection combination workflows
- **Workflow Branch Merging**: Merge detection results from different workflow branches back into a single detection stream (e.g., combine parallel processing branch results, merge conditional workflow paths, aggregate branch detection outputs), enabling workflow branch consolidation

## Connecting to Other Blocks

This block receives two detection prediction sets and produces a single combined detection set:

- **After multiple detection blocks** to combine detections from different models into one unified set (e.g., combine detections from multiple object detection models, merge results from different segmentation models, aggregate detections from various model outputs), enabling multi-model detection aggregation workflows
- **After filtering blocks** to combine filtered detection subsets (e.g., merge detections filtered by different criteria, combine class-specific filtered detections, aggregate various filtered detection sets), enabling filtered detection combination workflows
- **At workflow merge points** where different workflow branches need to be combined (e.g., merge parallel processing branch results, combine conditional path outputs, aggregate branch detection streams), enabling workflow branch merging workflows
- **Before downstream processing blocks** that need unified detection sets (e.g., process combined detections together, visualize unified detection sets, analyze aggregated detections), enabling unified detection processing workflows
- **Before crop blocks** to process combined detections together (e.g., crop regions from combined detection sets, extract areas from aggregated detections, process unified detection regions), enabling combined detection region extraction
- **Before visualization blocks** to display unified detection sets (e.g., visualize combined detections from multiple sources, display aggregated detection results, show merged detection outputs), enabling unified detection visualization workflows
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Detections Combine",
            "version": "v1",
            "short_description": "Combines two sets of predictions into a single prediction.",
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
    type: Literal["roboflow_core/detections_combine@v1"]
    prediction_one: Selector(
        kind=[
            TENSOR_NATIVE_OBJECT_DETECTION_PREDICTION_KIND,
            TENSOR_NATIVE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
        ]
    ) = Field(
        description="First set of detection predictions to combine. Supports object detection or instance segmentation predictions. All detections from this set will be included in the output. Detection properties (bounding boxes, masks, classes, confidence scores, metadata) are preserved as-is. This set is combined with prediction_two to create the unified output. Detections from this set typically appear first in the merged output.",
        examples=["$steps.my_object_detection_model.predictions"],
    )
    prediction_two: Selector(
        kind=[
            TENSOR_NATIVE_OBJECT_DETECTION_PREDICTION_KIND,
            TENSOR_NATIVE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
        ]
    ) = Field(
        description="Second set of detection predictions to combine. Supports object detection or instance segmentation predictions. All detections from this set will be included in the output. Detection properties (bounding boxes, masks, classes, confidence scores, metadata) are preserved as-is. This set is combined with prediction_one to create the unified output. Detections from this set are merged with detections from prediction_one to form a single combined detection set.",
        examples=["$steps.my_object_detection_model.predictions"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="predictions",
                kind=[
                    TENSOR_NATIVE_OBJECT_DETECTION_PREDICTION_KIND,
                    TENSOR_NATIVE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
                ],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


TensorNativeDetections = Union[Detections, InstanceDetections]


def _merged_class_names(
    prediction_one: TensorNativeDetections,
    prediction_two: TensorNativeDetections,
) -> Dict[int, str]:
    """Merge the two ``image_metadata[CLASS_NAMES_KEY]`` maps into one combined
    ``{class_id: name}`` map.

    Native predictions carry class names once per image (not per detection), so a
    combine has to reconcile the two maps. ``sv.Detections.merge`` keeps each
    detection's own ``class_name`` (it lives per-row in ``data``) and never re-maps
    ``class_id`` — it just ``np.hstack``-es the two id arrays. We mirror that by
    keeping every id unchanged and unioning the two maps. On an id collision with
    conflicting names the native single-map representation can only hold one name,
    so ``prediction_one`` wins (its detections come first in the output and "own"
    the id), matching the input ordering ``sv.Detections.merge`` preserves.
    """
    names_two = (prediction_two.image_metadata or {}).get(CLASS_NAMES_KEY) or {}
    names_one = (prediction_one.image_metadata or {}).get(CLASS_NAMES_KEY) or {}
    return {**names_two, **names_one}


def _combine_dense_masks(
    prediction_one: InstanceDetections,
    prediction_two: InstanceDetections,
    device: torch.device,
) -> torch.Tensor:
    """Concatenate two dense / RLE mask stacks into one dense ``(n, H, W)`` torch
    tensor on ``device``. Each instance is materialised to numpy (decoding RLE one
    row at a time) and stacked, so dense and RLE inputs combine uniformly. The
    common ``(H, W)`` is taken from whichever input carries instances."""
    mask_rows: List[np.ndarray] = []
    for prediction in (prediction_one, prediction_two):
        for index in range(len(prediction)):
            mask_rows.append(instance_mask_to_numpy(prediction, index))
    if not mask_rows:
        return torch.zeros((0, 0, 0), dtype=torch.bool, device=device)
    return torch.from_numpy(np.stack(mask_rows, axis=0)).to(
        device=device, dtype=torch.bool
    )


def _combine_rle_masks(
    prediction_one: InstanceDetections,
    prediction_two: InstanceDetections,
) -> InstancesRLEMasks:
    """Concatenate two ``InstancesRLEMasks`` lists into one. Both inputs must share
    the same ``image_size`` (they are predictions over the same image); the encoded
    RLE byte strings are simply concatenated, preserving detection order."""
    mask_one: InstancesRLEMasks = prediction_one.mask
    mask_two: InstancesRLEMasks = prediction_two.mask
    if tuple(mask_one.image_size) != tuple(mask_two.image_size):
        raise ValueError(
            "Cannot combine instance-segmentation predictions with RLE masks of "
            f"different image sizes ({mask_one.image_size} vs "
            f"{mask_two.image_size}); both predictions must be made against the "
            "same input image."
        )
    return InstancesRLEMasks(
        image_size=mask_one.image_size,
        masks=list(mask_one.masks) + list(mask_two.masks),
    )


def _combine_masks(
    prediction_one: InstanceDetections,
    prediction_two: InstanceDetections,
    device: torch.device,
) -> Union[torch.Tensor, InstancesRLEMasks]:
    """Combine the two instance masks, keeping the storage shape when it agrees:
    RLE + RLE stays RLE (byte lists concatenated); dense + dense is concatenated
    on-device with ``torch.cat`` (no host round-trip); any remaining RLE/dense mix
    is materialised to a dense ``(n, H, W)`` torch tensor (mirroring bounding_rect's
    RLE-in/RLE-out, dense-in/dense-out convention but for two inputs)."""
    mask_one = prediction_one.mask
    mask_two = prediction_two.mask
    both_rle = isinstance(mask_one, InstancesRLEMasks) and isinstance(
        mask_two, InstancesRLEMasks
    )
    if both_rle:
        return _combine_rle_masks(prediction_one, prediction_two)
    both_dense = isinstance(mask_one, torch.Tensor) and isinstance(
        mask_two, torch.Tensor
    )
    if both_dense:
        # Fast path: keep dense masks on-device, no device->host->device hop.
        return torch.cat([mask_one.to(device), mask_two.to(device)], dim=0)
    # RLE/dense mix: fall back to the per-row numpy materialise path.
    return _combine_dense_masks(prediction_one, prediction_two, device=device)


def _combine_bboxes_metadata(
    prediction_one: TensorNativeDetections,
    prediction_two: TensorNativeDetections,
) -> Optional[List[dict]]:
    """Concatenate the two ``bboxes_metadata`` lists, preserving each box's own
    ``detection_id``. A missing list is padded with empty dicts so the result lines
    up row-for-row with the combined tensors; a per-box ``detection_id`` is filled
    in when absent (the tensor-native serialiser requires one per row)."""
    if (
        prediction_one.bboxes_metadata is None
        and prediction_two.bboxes_metadata is None
    ):
        return None
    combined: List[dict] = []
    for prediction in (prediction_one, prediction_two):
        per_box = prediction.bboxes_metadata
        if per_box is None:
            per_box = [{} for _ in range(len(prediction))]
        for entry in per_box:
            entry = dict(entry or {})
            entry.setdefault(DETECTION_ID_KEY, str(uuid4()))
            combined.append(entry)
    return combined


def _combine_image_metadata(
    prediction_one: TensorNativeDetections,
    prediction_two: TensorNativeDetections,
) -> Optional[dict]:
    """Carry over the per-image lineage (parent/root coordinates, dimensions, ...)
    from the first prediction that has it and overwrite its ``class_names`` with the
    merged ``{class_id: name}`` map so every combined detection's id resolves."""
    base = prediction_one.image_metadata or prediction_two.image_metadata
    if base is None:
        return None
    image_metadata = dict(base)
    image_metadata[CLASS_NAMES_KEY] = _merged_class_names(
        prediction_one=prediction_one,
        prediction_two=prediction_two,
    )
    return image_metadata


class DetectionsCombineBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        prediction_one: TensorNativeDetections,
        prediction_two: TensorNativeDetections,
    ) -> BlockResult:
        # Native counterpart of sv.Detections.merge([p1, p2]): concatenate the two
        # predictions' xyxy / class_id / confidence (torch.cat instead of
        # np.vstack/np.hstack), reconcile the class-name maps, and concatenate the
        # per-box metadata. The device is inherited from the inputs (first non-empty
        # row), falling back to WORKFLOWS_IMAGE_TENSOR_DEVICE for the all-empty case
        # — matching bounding_rect/detections_merge.
        device = WORKFLOWS_IMAGE_TENSOR_DEVICE
        for prediction in (prediction_one, prediction_two):
            if len(prediction) > 0:
                device = prediction.xyxy.device
                break

        xyxy = torch.cat(
            [prediction_one.xyxy.to(device), prediction_two.xyxy.to(device)], dim=0
        )
        class_id = torch.cat(
            [prediction_one.class_id.to(device), prediction_two.class_id.to(device)],
            dim=0,
        )
        confidence = torch.cat(
            [
                prediction_one.confidence.to(device),
                prediction_two.confidence.to(device),
            ],
            dim=0,
        )
        image_metadata = _combine_image_metadata(
            prediction_one=prediction_one,
            prediction_two=prediction_two,
        )
        bboxes_metadata = _combine_bboxes_metadata(
            prediction_one=prediction_one,
            prediction_two=prediction_two,
        )

        one_has_masks = isinstance(prediction_one, InstanceDetections)
        two_has_masks = isinstance(prediction_two, InstanceDetections)
        if one_has_masks != two_has_masks:
            # Match numpy sv.Detections.merge: stack_or_none("mask") raises when
            # some-but-not-all inputs carry masks ("All or none of the 'mask'
            # fields must be None"). Mirror that here instead of silently dropping
            # the instance-segmentation input's masks down the OD output path.
            raise ValueError(
                "Cannot combine an object-detection prediction with an "
                "instance-segmentation prediction: all or none of the combined "
                "predictions must carry masks (matching sv.Detections.merge)."
            )
        has_masks = one_has_masks and two_has_masks
        if has_masks:
            mask = _combine_masks(
                prediction_one=prediction_one,
                prediction_two=prediction_two,
                device=device,
            )
            return {
                "predictions": InstanceDetections(
                    xyxy=xyxy,
                    class_id=class_id,
                    confidence=confidence,
                    mask=mask,
                    image_metadata=image_metadata,
                    bboxes_metadata=bboxes_metadata,
                )
            }
        return {
            "predictions": Detections(
                xyxy=xyxy,
                class_id=class_id,
                confidence=confidence,
                image_metadata=image_metadata,
                bboxes_metadata=bboxes_metadata,
            )
        }

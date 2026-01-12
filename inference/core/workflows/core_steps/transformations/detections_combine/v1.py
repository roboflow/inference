from copy import deepcopy
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field

from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

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
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
        ]
    ) = Field(
        description="First set of detection predictions to combine. Supports object detection or instance segmentation predictions. All detections from this set will be included in the output. Detection properties (bounding boxes, masks, classes, confidence scores, metadata) are preserved as-is. This set is combined with prediction_two to create the unified output. Detections from this set typically appear first in the merged output.",
        examples=["$steps.my_object_detection_model.predictions"],
    )
    prediction_two: Selector(
        kind=[
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
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
                    OBJECT_DETECTION_PREDICTION_KIND,
                    INSTANCE_SEGMENTATION_PREDICTION_KIND,
                ],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class DetectionsCombineBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        prediction_one: sv.Detections,
        prediction_two: sv.Detections,
    ) -> BlockResult:
        return {"predictions": sv.Detections.merge([prediction_one, prediction_two])}

import itertools
from dataclasses import replace
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import supervision as sv
from pydantic import AliasChoices, ConfigDict, Field

from inference.core.workflows.constants import DETECTION_ID_KEY
from inference.core.workflows.entities.base import (
    Batch,
    OriginCoordinatesSystem,
    OutputDefinition,
    ParentImageMetadata,
    WorkflowImageData,
)
from inference.core.workflows.entities.types import (
    BATCH_OF_IMAGES_KIND,
    BATCH_OF_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    BATCH_OF_KEYPOINT_DETECTION_PREDICTION_KIND,
    BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND,
    FlowControl,
    StepOutputImageSelector,
    StepOutputSelector,
    WorkflowImageSelector,
)
from inference.core.workflows.prototypes.block import (
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
Create dynamic crops from an image based on detections from detections-based model.

This is useful when placed after an ObjectDetection block as part of a multi-stage 
workflow. For example, you could use an ObjectDetection block to detect objects, then 
the CropBlock block to crop objects, then an OCR block to run character recognition on 
each of the individual cropped regions.
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "short_description": "Create dynamic crops from a detections model.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
        }
    )
    type: Literal["Crop"]
    images: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
        description="Reference at image to be used as input for step processing",
        examples=["$inputs.image", "$steps.cropping.crops"],
        validation_alias=AliasChoices("images", "image"),
    )
    predictions: StepOutputSelector(
        kind=[
            BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND,
            BATCH_OF_INSTANCE_SEGMENTATION_PREDICTION_KIND,
            BATCH_OF_KEYPOINT_DETECTION_PREDICTION_KIND,
        ]
    ) = Field(
        description="Reference to predictions of detection-like model, that can be based of cropping "
        "(detection must define RoI - eg: bounding box)",
        examples=["$steps.my_object_detection_model.predictions"],
        validation_alias=AliasChoices("predictions", "detections"),
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="crops", kind=[BATCH_OF_IMAGES_KIND]),
        ]


class CropBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    async def run_locally(
        self,
        images: Batch[Optional[WorkflowImageData]],
        predictions: Batch[Optional[sv.Detections]],
    ) -> Tuple[List[Any], FlowControl]:
        result = list(
            itertools.chain.from_iterable(
                crop_image(image=image, detections=detections)
                for image, detections in Batch.zip_nonempty(
                    batches=[images, predictions]
                )
            )
        )
        if len(result) == 0:
            return result, FlowControl(mode="terminate_branch")
        return result, FlowControl(mode="pass")


def crop_image(
    image: WorkflowImageData,
    detections: sv.Detections,
    detection_id_key: str = DETECTION_ID_KEY,
) -> List[Dict[str, WorkflowImageData]]:
    crops = []
    for (x_min, y_min, x_max, y_max), detection_id in zip(
        detections.xyxy.round().astype(dtype=int), detections[detection_id_key]
    ):
        cropped_image = image.numpy_image[y_min:y_max, x_min:x_max]
        parent_metadata = ParentImageMetadata(
            parent_id=detection_id,
            origin_coordinates=OriginCoordinatesSystem(
                left_top_x=x_min,
                left_top_y=y_min,
                origin_width=image.numpy_image.shape[1],
                origin_height=image.numpy_image.shape[0],
            ),
        )
        workflow_root_ancestor_coordinates = replace(
            image.workflow_root_ancestor_metadata.origin_coordinates,
            left_top_x=image.workflow_root_ancestor_metadata.origin_coordinates.left_top_x
            + x_min,
            left_top_y=image.workflow_root_ancestor_metadata.origin_coordinates.left_top_y
            + y_min,
        )
        workflow_root_ancestor_metadata = ParentImageMetadata(
            parent_id=image.workflow_root_ancestor_metadata.parent_id,
            origin_coordinates=workflow_root_ancestor_coordinates,
        )
        crops.append(
            {
                "crops": WorkflowImageData(
                    parent_metadata=parent_metadata,
                    workflow_root_ancestor_metadata=workflow_root_ancestor_metadata,
                    numpy_image=cropped_image,
                )
            }
        )
    return crops

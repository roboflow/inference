import itertools
from typing import Any, Dict, List, Literal, Tuple, Type, Union

import numpy as np
from pydantic import AliasChoices, ConfigDict, Field

from inference.core.utils.image_utils import ImageType, load_image
from inference.core.workflows.constants import (
    DETECTION_ID_KEY,
    HEIGHT_KEY,
    IMAGE_TYPE_KEY,
    IMAGE_VALUE_KEY,
    LEFT_TOP_X_KEY,
    LEFT_TOP_Y_KEY,
    ORIGIN_COORDINATES_KEY,
    ORIGIN_SIZE_KEY,
    PARENT_ID_KEY,
    WIDTH_KEY,
)
from inference.core.workflows.core_steps.common.utils import (
    detection_to_xyxy,
    extract_origin_size_from_images_batch,
)
from inference.core.workflows.entities.base import OutputDefinition
from inference.core.workflows.entities.types import (
    BATCH_OF_IMAGES_KIND,
    BATCH_OF_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    BATCH_OF_KEYPOINT_DETECTION_PREDICTION_KIND,
    BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND,
    BATCH_OF_PARENT_ID_KIND,
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
        description="Reference an image to be used as input for step processing",
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
            OutputDefinition(name="parent_id", kind=[BATCH_OF_PARENT_ID_KIND]),
        ]


class CropBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    async def run_locally(
        self,
        images: List[dict],
        predictions: List[List[dict]],
    ) -> Tuple[List[Any], FlowControl]:
        decoded_images = [load_image(e) for e in images]
        decoded_images = [
            i[0] if i[1] is True else i[0][:, :, ::-1] for i in decoded_images
        ]
        origin_image_shape = extract_origin_size_from_images_batch(
            input_images=images,
            decoded_images=decoded_images,
        )
        result = list(
            itertools.chain.from_iterable(
                crop_image(image=image, detections=detections, origin_size=origin_shape)
                for image, detections, origin_shape in zip(
                    decoded_images, predictions, origin_image_shape
                )
            )
        )
        if len(result) == 0:
            return result, FlowControl(mode="terminate_branch")
        return result, FlowControl(mode="pass")


def crop_image(
    image: np.ndarray,
    detections: List[dict],
    origin_size: dict,
) -> List[Dict[str, Union[dict, str]]]:
    crops = []
    for detection in detections:
        x_min, y_min, x_max, y_max = detection_to_xyxy(detection=detection)
        cropped_image = image[y_min:y_max, x_min:x_max]
        crops.append(
            {
                "crops": {
                    IMAGE_TYPE_KEY: ImageType.NUMPY_OBJECT.value,
                    IMAGE_VALUE_KEY: cropped_image,
                    PARENT_ID_KEY: detection[DETECTION_ID_KEY],
                    ORIGIN_COORDINATES_KEY: {
                        LEFT_TOP_X_KEY: x_min,
                        LEFT_TOP_Y_KEY: y_min,
                        WIDTH_KEY: detection[WIDTH_KEY],
                        HEIGHT_KEY: detection[HEIGHT_KEY],
                        ORIGIN_SIZE_KEY: origin_size,
                    },
                },
                "parent_id": detection[DETECTION_ID_KEY],
            }
        )
    return crops

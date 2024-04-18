import itertools
from typing import Any, Dict, List, Literal, Tuple, Type, Union

import numpy as np
from pydantic import AliasChoices, ConfigDict, Field

from inference.core.utils.image_utils import ImageType, load_image
from inference.enterprise.workflows.complier.steps_executors.constants import (
    CENTER_X_KEY,
    CENTER_Y_KEY,
    DETECTION_ID_KEY,
    HEIGHT_KEY,
    IMAGE_TYPE_KEY,
    IMAGE_VALUE_KEY,
    ORIGIN_COORDINATES_KEY,
    ORIGIN_SIZE_KEY,
    PARENT_ID_KEY,
    WIDTH_KEY,
)
from inference.enterprise.workflows.core_steps.common.utils import (
    detection_to_xyxy,
    extract_origin_size_from_images,
)
from inference.enterprise.workflows.entities.steps import OutputDefinition
from inference.enterprise.workflows.entities.types import (
    IMAGE_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    PARENT_ID_KIND,
    FlowControl,
    InferenceImageSelector,
    OutputStepImageSelector,
    StepOutputSelector,
)
from inference.enterprise.workflows.prototypes.block import (
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
    image: Union[InferenceImageSelector, OutputStepImageSelector] = Field(
        description="Reference at image to be used as input for step processing",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )
    predictions: StepOutputSelector(
        kind=[
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
            KEYPOINT_DETECTION_PREDICTION_KIND,
        ]
    ) = Field(
        description="Reference to predictions of detection-like model, that can be based of cropping "
        "(detection must define RoI - eg: bounding box)",
        examples=["$steps.my_object_detection_model.predictions"],
        validation_alias=AliasChoices("predictions", "detections"),
    )


class CropBlock(WorkflowBlock):

    @classmethod
    def get_input_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="crops", kind=[IMAGE_KIND]),
            OutputDefinition(name="parent_id", kind=[PARENT_ID_KIND]),
        ]

    async def run_locally(
        self,
        image: List[dict],
        predictions: List[List[dict]],
    ) -> Tuple[List[Any], FlowControl]:
        decoded_images = [load_image(e) for e in image]
        decoded_images = [
            i[0] if i[1] is True else i[0][:, :, ::-1] for i in decoded_images
        ]
        origin_image_shape = extract_origin_size_from_images(
            input_images=image,
            decoded_images=decoded_images,
        )
        result = list(
            itertools.chain.from_iterable(
                crop_image(image=i, predictions=d, origin_size=o)
                for i, d, o in zip(decoded_images, predictions, origin_image_shape)
            )
        )
        if len(result) == 0:
            return result, FlowControl(mode="terminate_branch")
        return result, FlowControl(mode="pass")


def crop_image(
    image: np.ndarray,
    predictions: List[dict],
    origin_size: dict,
) -> List[Dict[str, Union[dict, str]]]:
    crops = []
    for detection in predictions:
        x_min, y_min, x_max, y_max = detection_to_xyxy(detection=detection)
        cropped_image = image[y_min:y_max, x_min:x_max]
        crops.append(
            {
                "crops": {
                    IMAGE_TYPE_KEY: ImageType.NUMPY_OBJECT.value,
                    IMAGE_VALUE_KEY: cropped_image,
                    PARENT_ID_KEY: detection[DETECTION_ID_KEY],
                    ORIGIN_COORDINATES_KEY: {
                        CENTER_X_KEY: detection["x"],
                        CENTER_Y_KEY: detection["y"],
                        WIDTH_KEY: detection[WIDTH_KEY],
                        HEIGHT_KEY: detection[HEIGHT_KEY],
                        ORIGIN_SIZE_KEY: origin_size,
                    },
                },
                "parent_id": detection[DETECTION_ID_KEY],
            }
        )
    return crops

import math
from typing import Any, Dict, List, Literal, Type, Union
from uuid import uuid4

import numpy as np
from pydantic import AliasChoices, ConfigDict, Field
import supervision as sv
import zxingcpp

from inference.core.workflows.core_steps.common.utils import (
    anchor_prediction_detections_in_parent_coordinates,
    attach_parent_info,
    attach_prediction_type_info,
    convert_to_sv_detections,
)
from inference.core.utils.image_utils import load_image
from inference.core.workflows.entities.base import OutputDefinition
from inference.core.workflows.entities.types import (
    BATCH_OF_BAR_CODE_DETECTION_KIND,
    BATCH_OF_IMAGE_METADATA_KIND,
    BATCH_OF_PARENT_ID_KIND,
    BATCH_OF_PREDICTION_TYPE_KIND,
    StepOutputImageSelector,
    WorkflowImageSelector,
)
from inference.core.workflows.prototypes.block import (
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
Detect the location of barcodes in an image.

This block is useful for manufacturing and consumer packaged goods projects where you 
need to detect a barcode region in an image. You can then apply Crop block to isolate 
each barcode then apply further processing (i.e. OCR of the characters on a barcode).
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "short_description": "Run Optical Character Recognition on a model.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
        }
    )
    type: Literal["BarcodeDetector", "BarcodeDetection"]
    images: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
        description="Reference at image to be used as input for step processing",
        examples=["$inputs.image", "$steps.cropping.crops"],
        validation_alias=AliasChoices("images", "image"),
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="predictions", kind=[BATCH_OF_BAR_CODE_DETECTION_KIND]
            ),
            OutputDefinition(name="image", kind=[BATCH_OF_IMAGE_METADATA_KIND]),
            OutputDefinition(name="parent_id", kind=[BATCH_OF_PARENT_ID_KIND]),
            OutputDefinition(
                name="prediction_type", kind=[BATCH_OF_PREDICTION_TYPE_KIND]
            ),
        ]


class BarcodeDetectorBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    async def run_locally(
        self,
        images: List[dict],
    ) -> List[Dict[str, Union[sv.Detections, Any]]]:
        decoded_images = [load_image(e)[0] for e in images]
        predictions = [
            {
                "predictions": detect_barcodes(image=image),
                "image": {"width": image.shape[1], "height": image.shape[0]},
            }
            for image in decoded_images
        ]
        return self._post_process_result(image=images, predictions=predictions)

    def _post_process_result(
        self,
        image: List[dict],
        predictions: List[dict],
    ) -> List[Dict[str, Union[sv.Detections, Any]]]:
        predictions = convert_to_sv_detections(predictions)
        predictions = attach_prediction_type_info(
            predictions=predictions,
            prediction_type="barcode-detection",
        )
        predictions = attach_parent_info(images=image, predictions=predictions)
        return anchor_prediction_detections_in_parent_coordinates(
            image=image,
            predictions=predictions,
        )


def detect_barcodes(image: np.ndarray) -> List[dict]:
    barcodes = zxingcpp.read_barcodes(image)
    predictions = []
    for barcode in barcodes:
        width = barcode.position.top_right.x - barcode.position.top_left.x
        height = barcode.position.bottom_left.y - barcode.position.top_left.y
        predictions.append(
            {
                "class": "barcode",
                "class_id": 0,
                "confidence": 1.0,
                "x": int(math.floor(barcode.position.top_left.x + width / 2)),
                "y": int(math.floor(barcode.position.top_left.y + height / 2)),
                "width": width,
                "height": height,
                "detection_id": str(uuid4()),
                "data": barcode.text,
            }
        )
    return predictions

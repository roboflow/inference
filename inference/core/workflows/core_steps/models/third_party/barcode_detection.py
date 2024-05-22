import math
from typing import Any, Dict, List, Literal, Tuple, Type, Union
from uuid import uuid4

import numpy as np
import zxingcpp
from pydantic import AliasChoices, ConfigDict, Field

from inference.core.utils.image_utils import load_image
from inference.core.workflows.entities.base import OutputDefinition
from inference.core.workflows.entities.types import (
    BATCH_OF_BAR_CODE_DETECTION_KIND,
    BATCH_OF_IMAGE_METADATA_KIND,
    BATCH_OF_PARENT_ID_KIND,
    BATCH_OF_PREDICTION_TYPE_KIND,
    FlowControl,
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
        description="Reference an image to be used as input for step processing",
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
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FlowControl]]:
        decoded_images = [load_image(e)[0] for e in images]
        image_parent_ids = [img["parent_id"] for img in images]
        return [
            {
                "predictions": detect_barcodes(image=image, parent_id=parent_id),
                "parent_id": parent_id,
                "image": {"width": image.shape[1], "height": image.shape[0]},
                "prediction_type": "barcode-detection",
            }
            for image, parent_id in zip(decoded_images, image_parent_ids)
        ]


def detect_barcodes(image: np.ndarray, parent_id: str) -> List[dict]:
    barcodes = zxingcpp.read_barcodes(image)
    predictions = []
    for barcode in barcodes:
        width = barcode.position.top_right.x - barcode.position.top_left.x
        height = barcode.position.bottom_left.y - barcode.position.top_left.y
        predictions.append(
            {
                "parent_id": parent_id,
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

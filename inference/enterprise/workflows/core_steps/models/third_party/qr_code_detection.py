from typing import Any, Dict, List, Literal, Tuple, Type, Union
from uuid import uuid4

import cv2
import numpy as np
from pydantic import AliasChoices, ConfigDict, Field

from inference.core.utils.image_utils import load_image
from inference.enterprise.workflows.entities.base import OutputDefinition
from inference.enterprise.workflows.entities.types import (
    BAR_CODE_DETECTION_KIND,
    IMAGE_METADATA_KIND,
    PARENT_ID_KIND,
    PREDICTION_TYPE_KIND,
    FlowControl,
    StepOutputImageSelector,
    WorkflowImageSelector,
)
from inference.enterprise.workflows.prototypes.block import (
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
Detect the location of a QR code.

This block is useful for manufacturing and consumer packaged goods projects where you 
need to detect a QR code region in an image. You can then apply Crop block to isolate 
each QR code then apply further processing (i.e. read a QR code with a custom block).
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "short_description": "Detect the location of QR codes in an image.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
        }
    )
    type: Literal["QRCodeDetector", "QRCodeDetection"]
    images: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
        description="Reference at image to be used as input for step processing",
        examples=["$inputs.image", "$steps.cropping.crops"],
        validation_alias=AliasChoices("images", "image"),
    )


class QRCodeDetectorBlock(WorkflowBlock):

    @classmethod
    def get_input_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="predictions", kind=[BAR_CODE_DETECTION_KIND]),
            OutputDefinition(name="image", kind=[IMAGE_METADATA_KIND]),
            OutputDefinition(name="parent_id", kind=[PARENT_ID_KIND]),
            OutputDefinition(name="prediction_type", kind=[PREDICTION_TYPE_KIND]),
        ]

    async def run_locally(
        self,
        images: List[dict],
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FlowControl]]:
        decoded_images = [load_image(e)[0] for e in images]
        image_parent_ids = [img["parent_id"] for img in images]
        return [
            {
                "predictions": detect_qr_codes(image=image, parent_id=parent_id),
                "parent_id": parent_id,
                "image": {"width": image.shape[1], "height": image.shape[0]},
                "prediction_type": "qrcode-detection",
            }
            for image, parent_id in zip(decoded_images, image_parent_ids)
        ]


def detect_qr_codes(image: np.ndarray, parent_id: str) -> List[dict]:
    detector = cv2.QRCodeDetector()
    retval, detections, points_list, _ = detector.detectAndDecodeMulti(image)
    predictions = []
    for data, points in zip(detections, points_list):
        width = points[2][0] - points[0][0]
        height = points[2][1] - points[0][1]
        predictions.append(
            {
                "parent_id": parent_id,
                "class": "qr_code",
                "class_id": 0,
                "confidence": 1.0,
                "x": points[0][0] + width / 2,
                "y": points[0][1] + height / 2,
                "width": width,
                "height": height,
                "detection_id": str(uuid4()),
                "data": data,
            }
        )
    return predictions

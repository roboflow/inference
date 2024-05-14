from typing import Any, Dict, List, Literal, Type, Union
from uuid import uuid4

import cv2
import numpy as np
from pydantic import AliasChoices, ConfigDict, Field
import supervision as sv

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


class QRCodeDetectorBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    async def run_locally(
        self,
        images: List[dict],
    ) -> List[Dict[str, Union[sv.Detections, Any]]]:
        decoded_images = [load_image(e)[0] for e in images]
        image_parent_ids = [img["parent_id"] for img in images]
        predictions = [
            {
                "predictions": detect_qr_codes(image=image),
                "image": {"width": image.shape[1], "height": image.shape[0]},
                "prediction_type": "qrcode-detection",
            }
            for image, parent_id in zip(decoded_images, image_parent_ids)
        ]
        return self._post_process_result(image=images, predictions=predictions)

    def _post_process_result(
        self,
        image: List[dict],
        predictions: List[dict],
    ) -> List[Dict[str, Union[sv.Detections, Any]]]:
        converted_predictions = convert_to_sv_detections(predictions)
        converted_predictions = attach_prediction_type_info(
            predictions=converted_predictions,
            prediction_type="qrcode-detection",
        )
        converted_predictions = attach_parent_info(
            images=image, predictions=converted_predictions
        )
        for converted_prediction, prediction in zip(converted_predictions, predictions):
            converted_prediction["predictions"]["data"] = [
                d["data"] for d in prediction["predictions"]
            ]
        return anchor_prediction_detections_in_parent_coordinates(
            image=image,
            predictions=converted_predictions,
        )


def detect_qr_codes(image: np.ndarray) -> List[dict]:
    detector = cv2.QRCodeDetector()
    retval, detections, points_list, _ = detector.detectAndDecodeMulti(image)
    predictions = []
    for data, points in zip(detections, points_list):
        width = points[2][0] - points[0][0]
        height = points[2][1] - points[0][1]
        predictions.append(
            {
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

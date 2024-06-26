from typing import Any, Dict, List, Literal, Optional, Type, Union
from uuid import uuid4

import numpy as np
import supervision as sv
import zxingcpp
from pydantic import ConfigDict
from supervision.config import CLASS_NAME_DATA_FIELD

from inference.core.workflows.constants import (
    DETECTED_CODE_KEY,
    DETECTION_ID_KEY,
    IMAGE_DIMENSIONS_KEY,
    PREDICTION_TYPE_KEY,
)
from inference.core.workflows.core_steps.common.utils import (
    attach_parents_coordinates_to_sv_detections,
)
from inference.core.workflows.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.entities.types import (
    BATCH_OF_BAR_CODE_DETECTION_KIND,
    ImageInputField,
    StepOutputImageSelector,
    WorkflowImageSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
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
    images: Union[WorkflowImageSelector, StepOutputImageSelector] = ImageInputField

    @classmethod
    def accepts_batch_input(cls) -> bool:
        return True

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="predictions", kind=[BATCH_OF_BAR_CODE_DETECTION_KIND]
            )
        ]


class BarcodeDetectorBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    async def run(
        self,
        images: Batch[WorkflowImageData],
    ) -> BlockResult:
        results = []
        for image in images:
            qr_code_detections = detect_barcodes(image=image)
            results.append({"predictions": qr_code_detections})
        return results


def detect_barcodes(image: WorkflowImageData) -> sv.Detections:
    barcodes = zxingcpp.read_barcodes(image.numpy_image)
    xyxy = []
    confidence = []
    class_id = []
    class_name = []
    extracted_data = []
    for barcode in barcodes:
        x_min = barcode.position.top_left.x
        y_min = barcode.position.top_left.y
        x_max = barcode.position.bottom_right.x
        y_max = barcode.position.bottom_right.y
        xyxy.append([x_min, y_min, x_max, y_max])
        class_id.append(0)
        class_name.append("barcode")
        confidence.append(1.0)
        extracted_data.append(barcode.text)
    xyxy = np.array(xyxy) if len(xyxy) > 0 else np.empty((0, 4))
    confidence = np.array(confidence) if len(confidence) > 0 else np.empty(0)
    class_id = np.array(class_id).astype(int) if len(class_id) > 0 else np.empty(0)
    class_name = np.array(class_name) if len(class_name) > 0 else np.empty(0)
    detections = sv.Detections(
        xyxy=np.array(xyxy),
        confidence=confidence,
        class_id=class_id,
        data={CLASS_NAME_DATA_FIELD: class_name},
    )
    detections[DETECTION_ID_KEY] = np.array([uuid4() for _ in range(len(detections))])
    detections[PREDICTION_TYPE_KEY] = np.array(["barcode-detection"] * len(detections))
    detections[DETECTED_CODE_KEY] = np.array(extracted_data)
    img_height, img_width = image.numpy_image.shape[:2]
    detections[IMAGE_DIMENSIONS_KEY] = np.array(
        [[img_height, img_width]] * len(detections)
    )
    return attach_parents_coordinates_to_sv_detections(
        detections=detections,
        image=image,
    )

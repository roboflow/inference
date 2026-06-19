"""Tensor-native sibling of `roboflow_core/barcode_detector@v1`.

Classical CV (zxingcpp) on numpy_image — there is no inference_models path — so the
ONLY change is the OUTPUT representation: instead of sv.Detections this builds
`inference_models.Detections` (xyxy / class_id / confidence) with `image_metadata`
(class_names map + prediction_type + lineage, via build_native_image_metadata) and
per-detection `bboxes_metadata` carrying the required detection_id plus the decoded
value under DETECTED_CODE_KEY. The output kind becomes the tensor-native barcode kind.
numpy_image is materialised transparently from the CHW tensor_image; zxingcpp needs
numpy, so the image path is unchanged.
"""

from typing import List, Literal, Optional, Type
from uuid import uuid4

import torch
import zxingcpp
from pydantic import ConfigDict

from inference.core.env import WORKFLOWS_IMAGE_TENSOR_DEVICE
from inference.core.workflows.core_steps.common.tensor_native import (
    build_native_image_metadata,
)
from inference.core.workflows.execution_engine.constants import (
    DETECTED_CODE_KEY,
    DETECTION_ID_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.tensor_native_types import (
    TENSOR_NATIVE_BAR_CODE_DETECTION_KIND,
)
from inference.core.workflows.execution_engine.entities.types import (
    IMAGE_KIND,
    ImageInputField,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)
from inference_models.models.base.object_detection import Detections

PREDICTION_TYPE = "barcode-detection"
CLASS_NAME = "barcode"

LONG_DESCRIPTION = """
Detect the location of barcodes in an image.

This block is useful for manufacturing and consumer packaged goods projects where you
need to detect a barcode region in an image. You can then apply Crop block to isolate
each barcode then apply further processing (i.e. OCR of the characters on a barcode).
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Barcode Detection",
            "version": "v1",
            "short_description": "Detect and read barcodes in an image.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "ui_manifest": {
                "section": "model",
                "icon": "far fa-barcode",
                "blockPriority": 12,
            },
        }
    )
    type: Literal[
        "roboflow_core/barcode_detector@v1", "BarcodeDetector", "BarcodeDetection"
    ]
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="predictions", kind=[TENSOR_NATIVE_BAR_CODE_DETECTION_KIND]
            )
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class BarcodeDetectorBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(self, images: Batch[WorkflowImageData]) -> BlockResult:
        return [{"predictions": detect_barcodes(image=image)} for image in images]


def detect_barcodes(image: WorkflowImageData) -> Detections:
    barcodes = zxingcpp.read_barcodes(image.numpy_image)
    xyxy: List[List[float]] = []
    codes: List[str] = []
    for barcode in barcodes:
        x_min = barcode.position.top_left.x
        y_min = barcode.position.top_left.y
        x_max = barcode.position.bottom_right.x
        y_max = barcode.position.bottom_right.y
        xyxy.append([float(x_min), float(y_min), float(x_max), float(y_max)])
        codes.append(barcode.text)
    return _build_code_detections(
        image=image, xyxy=xyxy, codes=codes, class_name=CLASS_NAME
    )


def _build_code_detections(
    image: WorkflowImageData,
    xyxy: List[List[float]],
    codes: List[str],
    class_name: str,
) -> Detections:
    n = len(xyxy)
    detections = Detections(
        xyxy=(
            torch.tensor(
                xyxy, dtype=torch.float32, device=WORKFLOWS_IMAGE_TENSOR_DEVICE
            )
            if n
            else torch.zeros(
                (0, 4), dtype=torch.float32, device=WORKFLOWS_IMAGE_TENSOR_DEVICE
            )
        ),
        class_id=torch.zeros(
            (n,), dtype=torch.int64, device=WORKFLOWS_IMAGE_TENSOR_DEVICE
        ),
        confidence=torch.ones(
            (n,), dtype=torch.float32, device=WORKFLOWS_IMAGE_TENSOR_DEVICE
        ),
    )
    detections.image_metadata = build_native_image_metadata(
        image=image,
        class_names={0: class_name},
        prediction_type=PREDICTION_TYPE,
        inference_id=str(uuid4()),
    )
    detections.bboxes_metadata = [
        {DETECTION_ID_KEY: str(uuid4()), DETECTED_CODE_KEY: codes[index]}
        for index in range(n)
    ]
    return detections

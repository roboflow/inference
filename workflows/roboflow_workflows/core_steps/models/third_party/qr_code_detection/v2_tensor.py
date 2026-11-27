"""Tensor-native sibling of `roboflow_core/qr_code_detector@v2`.

Classical CV (cv2.QRCodeDetector) on numpy_image — there is no inference_models path
— so the ONLY change is the OUTPUT representation: instead of sv.Detections this
builds `inference_models.Detections` (xyxy / class_id / confidence) with
`image_metadata` (class_names map + prediction_type + lineage, via
build_native_image_metadata) and per-detection `bboxes_metadata` carrying the
required detection_id plus the decoded value under DETECTED_CODE_KEY. The output kind
becomes the tensor-native QR kind. numpy_image is materialised transparently from the
CHW tensor_image; cv2 needs numpy, so the image path is unchanged.
"""

from typing import Annotated, List, Literal, Optional, Tuple, Type, Union
from uuid import uuid4

import cv2
import numpy as np
import torch
from pydantic import ConfigDict, Field
from roboflow_workflows.core_steps.common.tensor_native import (
    build_native_image_metadata,
)
from roboflow_workflows.environment import WORKFLOWS_IMAGE_TENSOR_DEVICE
from roboflow_workflows.execution_engine.constants import (
    DETECTED_CODE_KEY,
    DETECTION_ID_KEY,
)
from roboflow_workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from roboflow_workflows.execution_engine.entities.tensor_native_types import (
    TENSOR_NATIVE_QR_CODE_DETECTION_KIND,
)
from roboflow_workflows.execution_engine.entities.types import (
    FLOAT_KIND,
    IMAGE_KIND,
    ImageInputField,
    Selector,
)
from roboflow_workflows.execution_engine.entities.workload import (
    Discovery,
    RuntimeRestriction,
    WorkOperation,
)
from roboflow_workflows.prototypes.block import (
    BlockResult,
    DependentResource,
    WorkflowBlock,
    WorkflowBlockManifest,
)

from inference_models.models.base.object_detection import Detections

PREDICTION_TYPE = "qrcode-detection"
CLASS_NAME = "qr_code"
MIN_SCALE_FACTOR = 1.0
MAX_SCALE_FACTOR = 4.0

LONG_DESCRIPTION = """
Detect and read the QR codes in an image.

This block is useful for manufacturing, logistics and consumer packaged goods projects
where you need to find QR codes in an image and read them. Each detection's box covers
all four corners of its code, and the decoded text is stored with the detection.

A code that is small in the image, roughly under 3 pixels per module (one black or
white square of the code), often can't be read at the image's own resolution. Set
`scale_factor` to 2.0 or 3.0 to enlarge the image before reading it. Boxes are always
returned in the original image's coordinates.
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "QR Code Detection",
            "version": "v2",
            "short_description": "Detect and read QR codes in an image.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "ui_manifest": {
                "section": "model",
                "icon": "fal fa-qrcode",
                "blockPriority": 13,
                "opencv": True,
            },
        }
    )
    type: Literal["roboflow_core/qr_code_detector@v2"]
    images: Selector(kind=[IMAGE_KIND]) = ImageInputField
    scale_factor: Union[
        Annotated[float, Field(ge=MIN_SCALE_FACTOR, le=MAX_SCALE_FACTOR)],
        Selector(kind=[FLOAT_KIND]),
    ] = Field(  # type: ignore
        default=1.0,
        title="Scale Factor",
        description=(
            "Enlarge the image by this factor before looking for QR codes, from 1.0 "
            "to 4.0. 1.0 reads the image as it is. Codes under about 3 pixels per "
            "module often only decode at 2.0 or 3.0. Boxes are always returned in "
            "the original image's coordinates. Time and memory grow with the square "
            "of the factor."
        ),
        examples=[2.0, "$inputs.qr_scale_factor"],
    )

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="predictions", kind=[TENSOR_NATIVE_QR_CODE_DETECTION_KIND]
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"

    def discover_dependent_resources(self) -> List[DependentResource]:
        # Pure local computer vision: no Roboflow model, no Roboflow project
        # and no third-party model is fetched or called. Declaring [] is a
        # complete answer, not the base "unknown".
        return []

    def discover_work_operations(self) -> List[WorkOperation]:
        return [WorkOperation.IMAGE_ANALYSIS]

    def get_actual_restrictions(
        self, *, ignore_environment_restrictions: bool = False
    ) -> Discovery[RuntimeRestriction]:
        return Discovery[RuntimeRestriction](
            items=[], complete=True, unknown_reasons=[]
        )


class QRCodeDetectorBlockV2(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(self, images: Batch[WorkflowImageData], scale_factor: float) -> BlockResult:
        return [
            {"predictions": detect_qr_codes(image=image, scale_factor=scale_factor)}
            for image in images
        ]


def detect_qr_codes(image: WorkflowImageData, scale_factor: float = 1.0) -> Detections:
    """Detect and decode the QR codes in an image as tensor-native detections.

    Each box is the axis-aligned box around all four corners that OpenCV
    reports, so a rotated or tilted code still gets a box that covers it.

    Args:
        image (WorkflowImageData): Image to search for QR codes.
        scale_factor (float): How much to enlarge the image before looking for
            codes, from 1.0 to 4.0. Boxes are returned in the original image's
            coordinates either way.

    Returns:
        Detections: One detection per QR code, with the decoded text under
            ``DETECTED_CODE_KEY`` in ``bboxes_metadata``. Empty when the image
            has no QR code.

    Raises:
        ValueError: If ``scale_factor`` is outside 1.0 to 4.0.
    """
    original = image.numpy_image
    source = _enlarge(image=original, scale_factor=scale_factor)
    detector = cv2.QRCodeDetector()
    _retval, decoded, points_list, _ = detector.detectAndDecodeMulti(source)
    if points_list is None:
        # OpenCV returns no corner array at all when it finds no QR code.
        points_list = []

    xyxy: List[List[float]] = []
    codes: List[str] = []
    for data, points in zip(decoded, points_list):
        if scale_factor != 1.0:
            points = _to_original_pixels(
                points, original_shape=original.shape, enlarged_shape=source.shape
            )
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        xyxy.append([float(x_min), float(y_min), float(x_max), float(y_max)])
        codes.append(data)
    return _build_code_detections(
        image=image, xyxy=xyxy, codes=codes, class_name=CLASS_NAME
    )


def _enlarge(image: np.ndarray, scale_factor: float) -> np.ndarray:
    """Return the image enlarged by scale_factor, or the image itself at 1.0."""
    if not MIN_SCALE_FACTOR <= scale_factor <= MAX_SCALE_FACTOR:
        raise ValueError(
            f"scale_factor must be between {MIN_SCALE_FACTOR} and "
            f"{MAX_SCALE_FACTOR}, got {scale_factor}."
        )

    if scale_factor == 1.0:
        return image

    height, width = image.shape[:2]
    size = (round(width * scale_factor), round(height * scale_factor))
    enlarged = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)

    return enlarged


def _to_original_pixels(
    points: np.ndarray,
    *,
    original_shape: Tuple[int, ...],
    enlarged_shape: Tuple[int, ...],
) -> np.ndarray:
    """Map corners found in the enlarged image back onto the original image.

    cv2.resize lines up pixel centres, so the mapping goes through them.
    """
    scale = np.array(
        [
            enlarged_shape[1] / original_shape[1],
            enlarged_shape[0] / original_shape[0],
        ]
    )
    original_points = (points + 0.5) / scale - 0.5

    return original_points


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

import time
from typing import List, Literal, Optional, Type, Union

import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field, PositiveInt

try:
    from ultralytics import YOLO
except ImportError:
    pass

from inference.core.logger import logger
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    FLOAT_ZERO_TO_ONE_KIND,
    INTEGER_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    STRING_KIND,
    FloatZeroToOne,
    ImageInputField,
    StepOutputImageSelector,
    WorkflowImageSelector,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
This block performs inference by executing locally stored ultralytics pth file.
This block expects pth file to be available within local filesystem.
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Ultralytics",
            "version": "v1",
            "short_description": "Predict the location of objects with bounding boxes by inferring from locally stored pth file.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
        },
        protected_namespaces=(),
    )
    type: Literal["roboflow_core/ultralytics@v1",]
    images: Union[WorkflowImageSelector, StepOutputImageSelector] = ImageInputField
    model_path: Union[WorkflowParameterSelector(kind=[STRING_KIND]), str] = Field(
        description="Path to locally stored pth file",
        examples=["/path/to/model.pth", "$inputs.class_agnostic_nms"],
    )
    device: Union[WorkflowParameterSelector(kind=[STRING_KIND]), str] = Field(
        default="cpu",
        description="Specifies the device for inference (e.g., cpu, cuda:0, mps or 0)",
        examples=["cuda:0", "$inputs.device"],
    )
    class_agnostic_nms: Union[
        Optional[bool], WorkflowParameterSelector(kind=[BOOLEAN_KIND])
    ] = Field(
        default=False,
        description="Value to decide if NMS is to be used in class-agnostic mode.",
        examples=[True, "$inputs.class_agnostic_nms"],
    )
    confidence: Union[
        FloatZeroToOne,
        WorkflowParameterSelector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
    ] = Field(
        default=0.4,
        description="Confidence threshold for predictions",
        examples=[0.3, "$inputs.confidence_threshold"],
    )
    iou_threshold: Union[
        FloatZeroToOne,
        WorkflowParameterSelector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
    ] = Field(
        default=0.3,
        description="Parameter of NMS, to decide on minimum box intersection over union to merge boxes",
        examples=[0.4, "$inputs.iou_threshold"],
    )
    max_detections: Union[
        PositiveInt, WorkflowParameterSelector(kind=[INTEGER_KIND])
    ] = Field(
        default=300,
        description="Maximum number of detections to return",
        examples=[300, "$inputs.max_detections"],
    )
    half_precision: Union[
        Optional[bool], WorkflowParameterSelector(kind=[BOOLEAN_KIND])
    ] = Field(
        default=False,
        description="Enables half-precision (FP16) inference, which can speed up model inference on supported GPUs with minimal impact on accuracy.",
        examples=[True, "$inputs.half_precision"],
    )
    imgsz: Union[
        int,
        WorkflowParameterSelector(kind=[INTEGER_KIND]),
    ] = Field(
        default=640,
        description="Defines the image size for inference.",
        examples=[1280, "$inputs.imgsz"],
    )

    @classmethod
    def accepts_batch_input(cls) -> bool:
        return True

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="inference_id", kind=[STRING_KIND]),
            OutputDefinition(
                name="predictions", kind=[OBJECT_DETECTION_PREDICTION_KIND]
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


class UltralyticsBlockV1(WorkflowBlock):
    def __init__(self):
        self._model: Optional[YOLO] = None

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        images: Batch[WorkflowImageData],
        model_path: str,
        device: str,
        class_agnostic_nms: Optional[bool],
        confidence: Optional[float],
        iou_threshold: Optional[float],
        max_detections: Optional[int],
        half_precision: bool,
        imgsz: int,
    ) -> BlockResult:
        if "YOLO" not in globals():
            raise RuntimeError(
                "You must install ultralytics in order to use this block."
            )
        if not self._model:
            self._model = YOLO(model_path)

        predictions = []
        for image in images:
            inf = self._model(
                image.numpy_image,
                imgsz=imgsz,
                conf=confidence,
                iou=iou_threshold,
                half=half_precision,
                max_det=max_detections,
                agnostic_nms=class_agnostic_nms,
                device=device,
                verbose=False,
            )[0]
            detections = sv.Detections.from_ultralytics(inf)
            predictions.append(detections)

        return [
            {"inference_id": None, "predictions": prediction}
            for prediction in predictions
        ]

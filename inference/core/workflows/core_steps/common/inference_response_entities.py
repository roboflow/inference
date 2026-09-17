"""Inference response DTOs, owned by Workflows.

`InferenceResponseImage`, `InferenceResponse`, `CvInferenceResponse`,
`WithVisualizationResponse` and `InstanceSegmentationInferenceResponse` were
MOVED here verbatim from `inference/core/entities/responses/inference.py`;
the server module re-exports them, so there is exactly ONE class object per
name. Workflow blocks build these directly (converters return
`InstanceSegmentationInferenceResponse`) and dump them with
`model_dump(by_alias=True, exclude_none=True)`.

`inference/core/workflows/core_steps/common/segmentation_entities.py` owns the
prediction-level classes referenced from `predictions`.
"""

import base64
from typing import Any, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_serializer

from inference.core.workflows.core_steps.common.segmentation_entities import (
    InstanceSegmentationPrediction,
    InstanceSegmentationRLEPrediction,
)


class InferenceResponseImage(BaseModel):
    """Inference response image information.

    Attributes:
        width (int): The original width of the image used in inference.
        height (int): The original height of the image used in inference.
    """

    width: int = Field(description="The original width of the image used in inference")
    height: int = Field(
        description="The original height of the image used in inference"
    )


class ResolvedModel(BaseModel):
    """Identity of the model package used for inference.

    Attributes:
        model_id (str): Canonical model ID returned by the weights provider.
        model_package_id (str): ID of the loaded model package.
        backend (str): Backend of the loaded package.
        quantization (str): Quantization of the loaded package.
    """

    model_config = ConfigDict(protected_namespaces=())
    model_id: str = Field(
        description="Canonical model ID returned by the weights provider. It can differ from the requested alias."
    )
    model_package_id: str = Field(
        description="ID of the package that loaded successfully and produced this result, including after a loading fallback."
    )
    backend: str = Field(
        description="Backend of the loaded package, such as onnx, trt, or torch."
    )
    quantization: str = Field(
        description="Package quantization, such as fp32 or fp16, or unknown when unavailable. This does not specify the input tensor dtype or the precision of every runtime operation."
    )


class InferenceResponse(BaseModel):
    """Base inference response.

    Attributes:
        inference_id (Optional[str]): Unique identifier of inference
        frame_id (Optional[int]): The frame id of the image used in inference if the input was a video.
        time (Optional[float]): The time in seconds it took to produce the predictions including image preprocessing.
        resolved_model (Optional[ResolvedModel]): Identity of the model package used for inference.
    """

    model_config = ConfigDict(protected_namespaces=())
    inference_id: Optional[str] = Field(
        description="Unique identifier of inference", default=None
    )
    frame_id: Optional[int] = Field(
        default=None,
        description="The frame id of the image used in inference if the input was a video",
    )
    time: Optional[float] = Field(
        default=None,
        description="The time in seconds it took to produce the predictions including image preprocessing",
    )
    resolved_model: Optional[ResolvedModel] = Field(
        default=None,
        description="Package identity of the model instance that produced this result.",
    )


class CvInferenceResponse(InferenceResponse):
    """Computer Vision inference response.

    Attributes:
        image (Union[List[inference.core.entities.responses.inference.InferenceResponseImage], inference.core.entities.responses.inference.InferenceResponseImage]): Image(s) used in inference.
    """

    image: Union[List[InferenceResponseImage], InferenceResponseImage]


class WithVisualizationResponse(BaseModel):
    """Response with visualization.

    Attributes:
        visualization (Optional[Any]): Base64 encoded string containing prediction visualization image data.
    """

    visualization: Optional[Any] = Field(
        default=None,
        description="Base64 encoded string containing prediction visualization image data",
    )

    @field_serializer("visualization", when_used="json")
    def serialize_visualisation(self, visualization: Optional[Any]) -> Optional[str]:
        if visualization is None:
            return None
        return base64.b64encode(visualization).decode("utf-8")


class InstanceSegmentationInferenceResponse(
    CvInferenceResponse, WithVisualizationResponse
):
    """Instance Segmentation inference response.

    Attributes:
        predictions (List[Union[
            inference.core.entities.responses.inference.InstanceSegmentationPrediction,
            inference.core.entities.responses.inference.InstanceSegmentationRLEPrediction
        ]]): List of instance segmentation predictions.
    """

    predictions: List[
        Union[InstanceSegmentationPrediction, InstanceSegmentationRLEPrediction]
    ]

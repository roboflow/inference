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
from typing import Any, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_serializer
from roboflow_workflows.core_steps.common.segmentation_entities import (
    InstanceSegmentationPrediction,
    InstanceSegmentationRLEPrediction,
)


class InferenceResponseImage(BaseModel):
    """Inference response image information.

    Attributes:
        width (int): The width of the represented image coordinate frame.
        height (int): The height of the represented image coordinate frame.
    """

    width: int = Field(
        description="The width of the represented image coordinate frame"
    )
    height: int = Field(
        description="The height of the represented image coordinate frame"
    )


class ResolvedModel(BaseModel):
    """Identity of the model and available package details used for inference.

    Attributes:
        model_id (str): ID of the loaded model, using the canonical ID when available.
        model_package_id (Optional[str]): ID of the loaded model package.
        backend (Optional[str]): Backend of the loaded package.
        quantization (Optional[str]): Quantization of the loaded package.
    """

    model_config = ConfigDict(protected_namespaces=())
    model_id: str = Field(
        description="ID of the loaded model, using the canonical ID when available."
    )
    model_package_id: Optional[str] = Field(
        default=None,
        description="ID of the package that loaded successfully and produced this result, including after a loading fallback.",
    )
    backend: Optional[str] = Field(
        default=None,
        description="Backend of the loaded package, such as onnx, trt, or torch.",
    )
    quantization: Optional[str] = Field(
        default=None,
        description="Package quantization, such as fp32 or fp16, or unknown when unavailable. This does not specify the input tensor dtype or the precision of every runtime operation.",
    )


class InferenceResponse(BaseModel):
    """Base inference response.

    Attributes:
        inference_id (Optional[str]): Unique identifier of inference
        frame_id (Optional[int]): The frame id of the image used in inference if the input was a video.
        time (Optional[float]): The time in seconds it took to produce the predictions including image preprocessing.
        resolved_model (Optional[ResolvedModel]): Model identity and available package details for this result.
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
        description="Model identity and available package details for this result.",
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


class MaskCoordinateMetadata(BaseModel):
    """Describe the prediction grid and its mapping to the original image.

    Attributes:
        coordinate_system (str): Coordinate frame shared by boxes, polygons and encoded RLE masks.
        width (int): Width of the encoded mask grid.
        height (int): Height of the encoded mask grid.
        scale_x (float): Multiply mask x coordinates by this to obtain image x.
        scale_y (float): Multiply mask y coordinates by this to obtain image y.
    """

    coordinate_system: Literal["mask_grid"] = Field(
        default="mask_grid",
        description="All prediction geometry uses this grid, matching response image dimensions.",
    )
    width: int = Field(gt=0, description="Mask-grid width in pixels.", examples=[160])
    height: int = Field(gt=0, description="Mask-grid height in pixels.", examples=[107])
    scale_x: float = Field(
        gt=0,
        description="Original image width divided by output-grid width.",
        examples=[4.0],
    )
    scale_y: float = Field(
        gt=0,
        description="Original image height divided by output-grid height.",
        examples=[4.0],
    )


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
    image: InferenceResponseImage = Field(
        description="Dimensions of the coordinate frame shared by bounding boxes, "
        "polygon points and RLE masks. With opt-in this is the selected mask grid; "
        "otherwise it is the original image.",
    )
    original_image: Optional[InferenceResponseImage] = Field(
        default=None,
        description="Original image dimensions retained for opted-in responses, "
        "including when the selected grid has the same dimensions.",
    )
    mask_metadata: Optional[MaskCoordinateMetadata] = Field(
        default=None,
        description="Present for opted-in responses in either format. All prediction "
        "geometry uses this grid. Multiply x coordinates and widths by scale_x, "
        "and y coordinates and heights by scale_y, to map to original_image.",
    )

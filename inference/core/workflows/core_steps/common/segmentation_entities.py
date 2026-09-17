"""Segmentation prediction entities, owned by Workflows.

`Point`, `InstanceSegmentationBasePrediction`, `InstanceSegmentationPrediction`
and `InstanceSegmentationRLEPrediction` (from `inference/core/entities/
responses/inference.py`) and `Sam2SegmentationPrediction` (from
`responses/sam2.py`) were MOVED here verbatim; both server modules re-export
them, so there is exactly ONE class object per name. The blocks build these
from REMOTE responses as well as from their own arithmetic - the SAM 3
interactive parser, the SAM2 remote converter, the SAM3 v1/v2/v3 and
seg-preview remote/proxy paths - so they must keep pydantic validation and
coercion (`"0.9"` -> 0.9, nested `Point` and mask validation, alias `class`).

Pinned by `tests/workflows/unit_tests/core_steps/common/test_segmentation_entities.py`
(identity through the re-export, the remote-parser coercion/rejection matrix
through both remote branches, frozen validation tables).
"""

from typing import Any, Dict, List, Literal, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field


class Point(BaseModel):
    """Point coordinates.

    Attributes:
        x (float): The x-axis pixel coordinate of the point.
        y (float): The y-axis pixel coordinate of the point.
    """

    x: float = Field(description="The x-axis pixel coordinate of the point")
    y: float = Field(description="The y-axis pixel coordinate of the point")


class InstanceSegmentationBasePrediction(BaseModel):
    x: float = Field(description="The center x-axis pixel coordinate of the prediction")
    y: float = Field(description="The center y-axis pixel coordinate of the prediction")
    width: float = Field(
        description="The width of the prediction bounding box in number of pixels"
    )
    height: float = Field(
        description="The height of the prediction bounding box in number of pixels"
    )
    confidence: float = Field(
        description="The detection confidence as a fraction between 0 and 1"
    )
    class_name: str = Field(alias="class", description="The predicted class label")
    class_id: int = Field(description="The class id of the prediction")
    detection_id: str = Field(
        description="Unique identifier of detection",
        default_factory=lambda: str(uuid4()),
    )
    parent_id: Optional[str] = Field(
        description="Identifier of parent image region",
        default=None,
    )


class InstanceSegmentationPrediction(InstanceSegmentationBasePrediction):
    class_confidence: Union[float, None] = Field(
        None, description="The class label confidence as a fraction between 0 and 1"
    )
    points: List[Point] = Field(
        description="The list of points that make up the instance polygon"
    )
    mask_format: Literal["polygon"] = Field(
        default="polygon",
        description="Type of mask format",
    )


class InstanceSegmentationRLEPrediction(InstanceSegmentationBasePrediction):
    rle: dict = Field(
        description="RLE-encoded mask in COCO format: {'size': [H, W], 'counts': '...'}"
    )
    mask_format: Literal["rle"] = Field(
        default="rle",
        description="Type of mask format",
    )


class Sam2SegmentationPrediction(BaseModel):
    """SAM segmentation prediction.

    Attributes:
        masks (Union[List[List[List[int]]], Dict[str, Any], Any]): Mask data - either polygon coordinates or RLE encoding.
        confidence (float): Masks confidences.
        format (Optional[str]): Format of the mask data: 'polygon' or 'rle'.
    """

    masks: Union[List[List[List[int]]], Dict[str, Any]] = Field(
        description="If polygon format, masks is a list of polygons, where each polygon is a list of points, where each point is a tuple containing the x,y pixel coordinates of the point. If rle format, masks is a dictionary with the keys 'size' and 'counts' containing the size and counts of the RLE encoding."
    )
    confidence: float = Field(description="Masks confidences")
    format: Optional[str] = Field(
        default="polygon", description="Format of the mask data: 'polygon' or 'rle'"
    )

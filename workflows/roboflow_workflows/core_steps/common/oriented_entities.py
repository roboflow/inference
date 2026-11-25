"""Native OBB response entities shared by server and Workflow schema introspection."""

from typing import Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field, StrictBool, field_validator
from roboflow_workflows.core_steps.common.inference_response_entities import (
    CvInferenceResponse,
    WithVisualizationResponse,
)


class OrientedObjectDetectionPrediction(BaseModel):
    """Native rectangle/front tuple, expressed in original-image pixels and CW radians."""

    obb: List[float] = Field(
        min_length=5,
        max_length=5,
        description="Ordered cx, cy, width, height, clockwise angle; sides remain ordered.",
    )
    confidence: float = Field(ge=0, le=1, description="Sigmoid class confidence.")
    class_name: str = Field(alias="class", description="Frozen class name.")
    class_id: int = Field(description="Native dataset category identifier.")
    heading: StrictBool = Field(description="Whether this class has a semantic front.")
    detection_id: str = Field(
        description="Stable query/category identity within an input result."
    )
    parent_id: Optional[str] = Field(
        default=None, description="Input image region identity."
    )

    @field_validator("obb")
    @classmethod
    def _check_obb(cls, value):
        import math

        if not all(math.isfinite(v) for v in value) or value[2] <= 0 or value[3] <= 0:
            raise ValueError("OBB requires finite geometry and positive ordered sides")
        return value


class OrientedObjectDetectionInferenceResponse(
    CvInferenceResponse, WithVisualizationResponse
):
    """Native task response; AABB geometry is never substituted for an OBB."""

    predictions: List[OrientedObjectDetectionPrediction] = Field(
        description="Native oriented detections."
    )
    task_type: Literal["oriented-object-detection"] = Field(
        default="oriented-object-detection", description="Native task identity."
    )
    class_heading: Dict[str, bool] = Field(
        description="Frozen class heading capabilities."
    )


class OrientedQuadrilateralPrediction(BaseModel):
    """Ordered native footprint after a general affine/projective transform."""

    geometry_type: Literal["quadrilateral"] = Field(
        description="Explicit nonrectangular footprint."
    )
    corners: List[Tuple[float, float]] = Field(
        min_length=4,
        max_length=4,
        description="Four ordered float pixel corners; first edge follows the native front axis.",
    )
    confidence: float = Field(ge=0, le=1, description="Retained class confidence.")
    class_name: str = Field(alias="class", description="Frozen class name.")
    class_id: int = Field(description="Native category identifier.")
    heading: StrictBool = Field(description="Whether the class has semantic front.")
    detection_id: str = Field(description="Stable source detection identity.")
    parent_id: Optional[str] = Field(
        default=None, description="Parent image/crop identity."
    )


class OrientedWorkflowInferenceResponse(CvInferenceResponse):
    """Wire schema for native rectangle or explicitly warped quadrilateral results."""

    predictions: List[
        Union[OrientedObjectDetectionPrediction, OrientedQuadrilateralPrediction]
    ] = Field(description="Native predictions after supported Workflow operations.")
    task_type: Literal["oriented-object-detection"] = Field(
        default="oriented-object-detection", description="Native task identity."
    )
    class_heading: Dict[str, bool] = Field(
        description="Frozen class heading capabilities."
    )

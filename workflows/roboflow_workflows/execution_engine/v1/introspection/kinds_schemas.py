from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field


class ImageSchema(BaseModel):
    type: Literal["url"]
    value: Any = Field(
        description="Value depends on `type` - for url, one should provide URL to the file, for "
        "`file` - local path, for `base64` - base64 string."
    )


class ImageKeyPoints(BaseModel):
    pt: Tuple[float, float]
    size: float
    angle: float
    response: float
    octave: int
    class_id: int


class ClassificationPrediction(BaseModel):
    class_name: str = Field(alias="class", description="The predicted class label")
    class_id: int = Field(description="Numeric ID associated with the class label")
    confidence: float = Field(
        description="The class label confidence as a fraction between 0 and 1"
    )


class ImageMetadataSchema(BaseModel):
    width: Optional[int] = Field(
        description="The original width of the image used in inference"
    )
    height: Optional[int] = Field(
        description="The original height of the image used in inference"
    )


class MultiClassClassificationSchema(BaseModel):
    image: ImageMetadataSchema
    predictions: List[ClassificationPrediction]
    top: str = Field(description="The top predicted class label")
    confidence: float = Field(
        description="The confidence of the top predicted class label"
    )
    parent_id: Optional[str] = Field(
        description="Identifier of parent image region. Useful when stack of detection-models is in use to refer the RoI being the input to inference",
        default=None,
    )
    prediction_type: Optional[str] = Field(
        description="Type of prediction",
        default=None,
    )
    inference_id: Optional[str] = Field(
        description="Identifier of inference",
        default=None,
    )
    root_parent_id: Optional[str] = Field(
        description="Identifier of root parent image region. Useful when stack of detection-models is in use to refer the RoI being the input to inference",
        default=None,
    )


class MultiLabelClassificationPrediction(BaseModel):
    confidence: float = Field(
        description="The class label confidence as a fraction between 0 and 1"
    )
    class_id: int = Field(description="Numeric ID associated with the class label")


class MultiLabelClassificationSchema(BaseModel):
    image: ImageMetadataSchema
    predictions: Dict[str, MultiLabelClassificationPrediction]
    predicted_classes: List[str] = Field(description="The list of predicted classes")
    parent_id: Optional[str] = Field(
        description="Identifier of parent image region. Useful when stack of detection-models is in use to refer the RoI being the input to inference",
        default=None,
    )
    prediction_type: Optional[str] = Field(
        description="Type of prediction",
        default=None,
    )
    inference_id: Optional[str] = Field(
        description="Identifier of inference",
        default=None,
    )
    root_parent_id: Optional[str] = Field(
        description="Identifier of root parent image region. Useful when stack of detection-models is in use to refer the RoI being the input to inference",
        default=None,
    )


class BoundingBoxSchema(BaseModel):
    width: Union[int, float] = Field(description="Width of bounding box")
    height: Union[int, float] = Field(description="Height of bounding box")
    x: Union[int, float] = Field(description="OX coordinate of bounding box center")
    y: Union[int, float] = Field(description="OY coordinate of bounding box center")
    confidence: float = Field(description="Model confidence for bounding box")
    class_name: str = Field(
        description="Name of the class associated to bounding box",
        title="class",
        alias="class",
    )
    class_id: int = Field(description="Identifier of bounding box class")
    detection_id: str = Field(description="Identifier of detected bounding box")
    parent_id: Optional[str] = Field(
        description="Identifier of parent image region. Useful when stack of detection-models is in use to refer the RoI being the input to inference",
        default=None,
    )


class ObjectDetectionSchema(BaseModel):
    image: ImageMetadataSchema
    predictions: List[BoundingBoxSchema]


class PointSchema(BaseModel):
    x: Union[int, float] = Field(description="OX coordinate of point")
    y: Union[int, float] = Field(description="OY coordinate of point")


class BoundingBoxWithSegmentationSchema(BoundingBoxSchema):
    points: List[PointSchema]


class InstanceSegmentationSchema(BaseModel):
    image: ImageMetadataSchema
    predictions: List[BoundingBoxWithSegmentationSchema]


class KeyPointSchema(BaseModel):
    x: Union[int, float] = Field(description="OX coordinate of point")
    y: Union[int, float] = Field(description="OY coordinate of point")
    confidence: float = Field(description="Model confidence for keypoint")
    class_name: str = Field(
        description="Name of the class associated to keypoint",
        title="class",
        alias="class",
    )
    class_id: int = Field(description="Identifier of keypoint")


class BoundingBoxWithKeyPoints(BoundingBoxSchema):
    keypoints: List[KeyPointSchema]


class KeyPointsDetectionSchema(BaseModel):
    image: ImageMetadataSchema
    predictions: List[BoundingBoxWithKeyPoints]


class BoundingBoxWithCodeSchema(BaseModel):
    data: str = Field(description="Detected code")


class CodeDetectionSchema(BaseModel):
    image: ImageMetadataSchema
    predictions: List[BoundingBoxWithCodeSchema]

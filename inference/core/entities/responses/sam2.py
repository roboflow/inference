from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class Sam2EmbeddingResponse(BaseModel):
    """SAM embedding response.

    Attributes:
        embeddings (Union[List[List[List[List[float]]]], Any]): The SAM embedding.
        time (float): The time in seconds it took to produce the embeddings including preprocessing.
    """

    image_id: str = Field(description="Image id embeddings are cached to")
    time: float = Field(
        description="The time in seconds it took to produce the embeddings including preprocessing"
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
    class_name: Optional[str] = Field(
        default=None, description="Class name from the source detection, if provided."
    )
    class_id: Optional[Union[int, str]] = Field(
        default=None, description="Class id from the source detection, if provided."
    )
    detection_id: Optional[str] = Field(
        default=None, description="Detection id from the source detection, if provided."
    )
    parent_id: Optional[str] = Field(
        default=None, description="Parent id from the source detection, if provided."
    )
    detection_confidence: Optional[float] = Field(
        default=None,
        description="Confidence from the source detection, if provided.",
    )


class Sam2SegmentationResponse(BaseModel):
    predictions: List[Sam2SegmentationPrediction] = Field()
    time: float = Field(
        description="The time in seconds it took to produce the segmentation including preprocessing"
    )

from typing import Any, List, Union

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
        masks (Union[List[List[List[int]]], Any]): The set of output masks.
        low_res_masks (Union[List[List[List[int]]], Any]): The set of output low-resolution masks.
        time (float): The time in seconds it took to produce the segmentation including preprocessing.
    """

    masks: List[List[List[int]]] = Field(
        description="The set of points for output mask as polygon. Each element of list represents single point.",
    )
    confidence: float = Field(description="Masks confidences")


class Sam2SegmentationResponse(BaseModel):
    predictions: List[Sam2SegmentationPrediction] = Field()
    time: float = Field(
        description="The time in seconds it took to produce the segmentation including preprocessing"
    )

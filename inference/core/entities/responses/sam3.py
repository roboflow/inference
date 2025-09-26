from typing import List, Optional

from pydantic import BaseModel, Field


class Sam3EmbeddingResponse(BaseModel):
    image_id: str = Field(description="Image id embeddings are cached to")
    time: float = Field(
        description="The time in seconds it took to produce the embeddings including preprocessing"
    )


class Sam3SegmentationPrediction(BaseModel):
    masks: List[List[List[int]]] = Field(
        description="The set of points for output mask as polygon. Each element of list represents single point."
    )
    confidence: float = Field(description="Masks confidence")
    format: Optional[str] = Field(
        default="polygon",
        description="Format of the mask data: 'polygon' or 'mask'"
    )


class Sam3SegmentationResponse(BaseModel):
    predictions: List[Sam3SegmentationPrediction] = Field()
    time: float = Field(
        description="The time in seconds it took to produce the segmentation including preprocessing"
    )



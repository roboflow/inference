from typing import List, Optional, Union, Dict, Any

from pydantic import BaseModel, Field


class Sam3EmbeddingResponse(BaseModel):
    image_id: str = Field(description="Image id embeddings are cached to")
    time: float = Field(
        description="The time in seconds it took to produce the embeddings including preprocessing"
    )


class Sam3SegmentationPrediction(BaseModel):
    masks: Union[List[List[List[int]]], Dict[str, Any]] = Field(
        description="Mask data - either polygon coordinates or RLE encoding"
    )
    confidence: float = Field(description="Masks confidence")
    format: Optional[str] = Field(
        default="polygon", description="Format of the mask data: 'polygon' or 'rle'"
    )


class Sam3SegmentationResponse(BaseModel):
    predictions: List[Sam3SegmentationPrediction] = Field()
    time: float = Field(
        description="The time in seconds it took to produce the segmentation including preprocessing"
    )


class Sam3PromptEcho(BaseModel):
    prompt_index: int = Field()
    type: Optional[str] = Field(default=None)
    text: Optional[str] = Field(default=None)
    num_boxes: Optional[int] = Field(default=None)


class Sam3PromptResult(BaseModel):
    prompt_index: int = Field()
    echo: Sam3PromptEcho = Field()
    predictions: List[Sam3SegmentationPrediction] = Field()


class Sam3BatchSegmentationResponse(BaseModel):
    prompt_results: List[Sam3PromptResult] = Field()
    time: float = Field(
        description="The time in seconds it took to produce the segmentation including preprocessing"
    )

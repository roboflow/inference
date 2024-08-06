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

    mask: Union[List[List[List[int]]], Any] = Field(
        description="The set of output masks. If request format is json, masks is a list of polygons, where each polygon is a list of points, where each point is a tuple containing the x,y pixel coordinates of the point. If request format is binary, masks is a list of binary numpy arrays. The dimensions of each mask are the same as the dimensions of the input image.",
    )
    low_res_mask: Union[List[List[List[int]]], Any] = Field(
        description="The set of output masks. If request format is json, masks is a list of polygons, where each polygon is a list of points, where each point is a tuple containing the x,y pixel coordinates of the point. If request format is binary, masks is a list of binary numpy arrays. The dimensions of each mask are 256 x 256",
    )

class Sam2SegmentationResponse(BaseModel):
    predictions: List[Sam2SegmentationPrediction] = Field()
    time: float = Field(
        description="The time in seconds it took to produce the segmentation including preprocessing"
    )

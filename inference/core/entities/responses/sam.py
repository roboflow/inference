from typing import Any, List, Union

from pydantic import BaseModel, Field


class SamEmbeddingResponse(BaseModel):
    """SAM embedding response.

    Attributes:
        embeddings (Union[List[List[List[List[float]]]], Any]): The SAM embedding.
        time (float): The time in seconds it took to produce the embeddings including preprocessing.
    """

    embeddings: Union[List[List[List[List[float]]]], Any] = Field(
        examples=["[[[[0.1, 0.2, 0.3, ...] ...] ...]]"],
        description="If request format is json, embeddings is a series of nested lists representing the SAM embedding. If request format is binary, embeddings is a binary numpy array. The dimensions of the embedding are 1 x 256 x 64 x 64.",
    )
    time: float = Field(
        description="The time in seconds it took to produce the embeddings including preprocessing"
    )


class SamSegmentationResponse(BaseModel):
    """SAM segmentation response.

    Attributes:
        masks (Union[List[List[List[int]]], Any]): The set of output masks.
        low_res_masks (Union[List[List[List[int]]], Any]): The set of output low-resolution masks.
        time (float): The time in seconds it took to produce the segmentation including preprocessing.
    """

    masks: Union[List[List[List[int]]], Any] = Field(
        description="The set of output masks. If request format is json, masks is a list of polygons, where each polygon is a list of points, where each point is a tuple containing the x,y pixel coordinates of the point. If request format is binary, masks is a list of binary numpy arrays. The dimensions of each mask are the same as the dimensions of the input image.",
    )
    low_res_masks: Union[List[List[List[int]]], Any] = Field(
        description="The set of output masks. If request format is json, masks is a list of polygons, where each polygon is a list of points, where each point is a tuple containing the x,y pixel coordinates of the point. If request format is binary, masks is a list of binary numpy arrays. The dimensions of each mask are 256 x 256",
    )
    time: float = Field(
        description="The time in seconds it took to produce the segmentation including preprocessing"
    )

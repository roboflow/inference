from typing import Any, List, Optional, Union

from pydantic import BaseModel, Field, root_validator, validator

from inference.core.entities.requests.inference import (
    BaseRequest,
    InferenceRequestImage,
)
from inference.core.env import SAM2_VERSION_ID


class Sam2InferenceRequest(BaseRequest):
    """SAM2 inference request.

    Attributes:
        api_key (Optional[str]): Roboflow API Key.
        sam2_version_id (Optional[str]): The version ID of SAM2 to be used for this request.
    """

    sam2_version_id: Optional[str] = Field(
        default=SAM2_VERSION_ID,
        examples=["hiera_large"],
        description="The version ID of SAM to be used for this request. Must be one of hiera_tiny, hiera_small, hiera_large, hiera_b_plus",
    )

    model_id: Optional[str] = Field(None)

    # TODO[pydantic]: We couldn't refactor the `validator`, please replace it by `field_validator` manually.
    # Check https://docs.pydantic.dev/dev-v2/migration/#changes-to-validators for more information.
    @validator("model_id", always=True)
    def validate_model_id(cls, value, values):
        if value is not None:
            return value
        if values.get("sam_version_id") is None:
            return None
        return f"sam2/{values['sam_version_id']}"


class Sam2EmbeddingRequest(Sam2InferenceRequest):
    """SAM embedding request.

    Attributes:
        image (Optional[inference.core.entities.requests.inference.InferenceRequestImage]): The image to be embedded.
        image_id (Optional[str]): The ID of the image to be embedded used to cache the embedding.
        format (Optional[str]): The format of the response. Must be one of json or binary.
    """

    image: Optional[InferenceRequestImage] = Field(
        default=None,
        description="The image to be embedded",
    )
    image_id: Optional[str] = Field(
        default=None,
        examples=["image_id"],
        description="The ID of the image to be embedded used to cache the embedding.",
    )


class Box(BaseModel):
    x: float = Field()
    y: float = Field()
    width: float = Field()
    height: float = Field()


class Point(BaseModel):
    x: float = Field()
    y: float = Field()
    positive: bool = Field()


class Sam2Prompt(BaseModel):
    box: Optional[Box] = Field(default=None)
    points: Optional[List[Point]] = Field(default=None)


class Sam2PromptSet(BaseModel):
    prompts: Optional[List[Sam2Prompt]] = Field(default=None)

    def add_prompt(self, prompt: Sam2Prompt):
        if self.prompts is None:
            self.prompts = []
        self.prompts.append(prompt)

    def to_sam2_inputs(self):
        if self.prompts is None:
            return {"point_coords": None, "point_labels": None, "box": None}
        return_dict = {"point_coords": [], "point_labels": [], "box": []}
        for prompt in self.prompts:
            if prompt.box is not None:
                x1 = prompt.box.x - prompt.box.width / 2
                y1 = prompt.box.y - prompt.box.height / 2
                x2 = prompt.box.x + prompt.box.width / 2
                y2 = prompt.box.y + prompt.box.height / 2
                return_dict["box"].append([x1, y1, x2, y2])
            if prompt.points is not None:
                return_dict["point_coords"] = [
                    [point.x, point.y] for point in prompt.points
                ]
                return_dict["point_labels"] = [
                    int(point.positive) for point in prompt.points
                ]

        return_dict = {k: v if v else None for k, v in return_dict.items()}
        lengths = set()
        for v in return_dict.values():
            if isinstance(v, list):
                lengths.add(len(v))

        assert len(lengths) in [0, 1], "All prompts must have the same number of points"
        return return_dict


class Sam2SegmentationRequest(Sam2InferenceRequest):
    """SAM segmentation request.

    Attributes:
        format (Optional[str]): The format of the response.
        image (InferenceRequestImage): The image to be segmented.
        image_id (Optional[str]): The ID of the image to be segmented used to retrieve cached embeddings.
        point_coords (Optional[List[List[float]]]): The coordinates of the interactive points used during decoding.
        point_labels (Optional[List[float]]): The labels of the interactive points used during decoding.
    """

    format: Optional[str] = Field(
        default="json",
        examples=["json"],
        description="The format of the response. Must be one of json or binary. If binary, masks are returned as binary numpy arrays. If json, masks are converted to polygons, then returned as json.",
    )
    image: InferenceRequestImage = Field(
        description="The image to be segmented.",
    )
    image_id: Optional[str] = Field(
        default=None,
        examples=["image_id"],
        description="The ID of the image to be segmented used to retrieve cached embeddings. If an embedding is cached, it will be used instead of generating a new embedding. If no embedding is cached, a new embedding will be generated and cached.",
    )
    prompts: Sam2PromptSet = Field(default=Sam2PromptSet(prompts=None))

    mask_input: Optional[Union[List[List[List[float]]], Any]] = Field(
        default=None,
        description="The set of output masks. If request format is json, masks is a list of polygons, where each polygon is a list of points, where each point is a tuple containing the x,y pixel coordinates of the point. If request format is binary, masks is a list of binary numpy arrays. The dimensions of each mask are 256 x 256. This is the same as the output, low resolution mask from the previous inference.",
    )
    mask_input_format: Optional[str] = Field(
        default="json",
        examples=["json"],
        description="The format of the mask input. Must be one of json or binary. If binary, mask input is expected to be a binary numpy array.",
    )
    use_mask_input_cache: Optional[bool] = Field(
        default=False,
        examples=[False],
        description="Whether or not to use the mask input cache. If true, the mask input cache will be used if it exists. If false, the mask input cache will not be used.",
    )
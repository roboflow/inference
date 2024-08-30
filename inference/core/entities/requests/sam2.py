from typing import Any, List, Optional, Tuple, Union

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
    x: float
    y: float
    width: float
    height: float


class Point(BaseModel):
    x: float
    y: float
    positive: bool

    def to_hashable(self) -> Tuple[float, float, bool]:
        return (self.x, self.y, self.positive)


class Sam2Prompt(BaseModel):
    box: Optional[Box] = Field(default=None)
    points: Optional[List[Point]] = Field(default=None)

    def num_points(self) -> int:
        return len(self.points or [])


class Sam2PromptSet(BaseModel):
    prompts: Optional[List[Sam2Prompt]] = Field(
        default=None,
        description="An optional list of prompts for masks to predict. Each prompt can include a bounding box and / or a set of postive or negative points",
    )

    def num_points(self) -> int:
        if not self.prompts:
            return 0
        return sum(prompt.num_points() for prompt in self.prompts)

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
                return_dict["point_coords"].append(
                    list([point.x, point.y] for point in prompt.points)
                )
                return_dict["point_labels"].append(
                    list(int(point.positive) for point in prompt.points)
                )
            else:
                return_dict["point_coords"].append([])
                return_dict["point_labels"].append([])

        if not any(return_dict["point_coords"]):
            return_dict["point_coords"] = None
        if not any(return_dict["point_labels"]):
            return_dict["point_labels"] = None

        return_dict = {k: v if v else None for k, v in return_dict.items()}
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
    prompts: Sam2PromptSet = Field(
        default=Sam2PromptSet(prompts=None),
        example=[{"prompts": [{"points": [{"x": 100, "y": 100, "positive": True}]}]}],
        description="A list of prompts for masks to predict. Each prompt can include a bounding box and / or a set of postive or negative points",
    )
    multimask_output: bool = Field(
        default=True,
        examples=[True],
        description="If true, the model will return three masks. "
        "For ambiguous input prompts (such as a single click), this will often "
        "produce better masks than a single prediction. If only a single "
        "mask is needed, the model's predicted quality score can be used "
        "to select the best mask. For non-ambiguous prompts, such as multiple "
        "input prompts, multimask_output=False can give better results.",
    )

    save_logits_to_cache: bool = Field(
        default=False,
        description="If True, saves the low-resolution logits to the cache for potential future use. "
        "This can speed up subsequent requests with similar prompts on the same image. "
        "This feature is ignored if DISABLE_SAM2_LOGITS_CACHE env variable is set True",
    )
    load_logits_from_cache: bool = Field(
        default=False,
        description="If True, attempts to load previously cached low-resolution logits for the given image and prompt set. "
        "This can significantly speed up inference when making multiple similar requests on the same image. "
        "This feature is ignored if DISABLE_SAM2_LOGITS_CACHE env variable is set True",
    )

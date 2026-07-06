from typing import Any, Dict, List, Optional, Tuple, Union

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
    class_name: Optional[str] = Field(default=None)
    class_id: Optional[Union[int, str]] = Field(default=None)
    detection_id: Optional[str] = Field(default=None)
    parent_id: Optional[str] = Field(default=None)
    confidence: Optional[float] = Field(default=None)

    @root_validator(pre=True)
    def _coerce_class_alias(cls, values):
        if (
            isinstance(values, dict)
            and "class" in values
            and "class_name" not in values
        ):
            values = values.copy()
            values["class_name"] = values["class"]
        return values

    def to_xyxy(self) -> List[float]:
        return [
            self.x - self.width / 2,
            self.y - self.height / 2,
            self.x + self.width / 2,
            self.y + self.height / 2,
        ]

    def prediction_metadata(self) -> Dict[str, Any]:
        return _box_prediction_metadata(self)


class BoxXYXY(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    class_name: Optional[str] = Field(default=None)
    class_id: Optional[Union[int, str]] = Field(default=None)
    detection_id: Optional[str] = Field(default=None)
    parent_id: Optional[str] = Field(default=None)
    confidence: Optional[float] = Field(default=None)

    @root_validator(pre=True)
    def _coerce_box_aliases(cls, values):
        if not isinstance(values, dict):
            return values
        values = values.copy()
        if "class" in values and "class_name" not in values:
            values["class_name"] = values["class"]
        if {"x0", "y0", "x1", "y1"} <= set(values.keys()) and not {
            "x2",
            "y2",
        } <= set(values.keys()):
            values["x2"] = values["x1"]
            values["y2"] = values["y1"]
            values["x1"] = values["x0"]
            values["y1"] = values["y0"]
        return values

    def to_xyxy(self) -> List[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    def prediction_metadata(self) -> Dict[str, Any]:
        return _box_prediction_metadata(self)


BoxPrompt = Union[Box, BoxXYXY]


def _box_prediction_metadata(box: BoxPrompt) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    if box.class_name is not None:
        metadata["class_name"] = box.class_name
    if box.class_id is not None:
        metadata["class_id"] = box.class_id
    if box.detection_id is not None:
        metadata["detection_id"] = box.detection_id
    if box.parent_id is not None:
        metadata["parent_id"] = box.parent_id
    elif box.detection_id is not None:
        metadata["parent_id"] = box.detection_id
    if box.confidence is not None:
        metadata["detection_confidence"] = box.confidence
    return metadata


def _normalize_box_payload(value: Any) -> Any:
    if isinstance(value, (Box, BoxXYXY)):
        return value
    if _is_flat_box(value):
        return BoxXYXY(x1=value[0], y1=value[1], x2=value[2], y2=value[3])
    if not isinstance(value, dict):
        return value

    value = value.copy()
    if "class" in value and "class_name" not in value:
        value["class_name"] = value["class"]

    nested = None
    if "box" in value:
        nested = value["box"]
    elif "bbox" in value:
        nested = value["bbox"]
    if nested is not None:
        nested = _normalize_box_payload(nested)
        if isinstance(nested, (Box, BoxXYXY)):
            nested = nested.dict()
        if isinstance(nested, dict):
            nested = nested.copy()
            for key in (
                "class",
                "class_name",
                "class_id",
                "detection_id",
                "parent_id",
                "confidence",
            ):
                if key in value and nested.get(key) is None:
                    nested[key] = value[key]
        return nested

    return value


def _box_payload_to_prompts(box_payload: Any) -> List[Dict[str, Any]]:
    if box_payload is None:
        return []
    if isinstance(box_payload, dict):
        if "predictions" in box_payload:
            box_payload = box_payload["predictions"]
        elif "detections" in box_payload:
            box_payload = box_payload["detections"]
        elif "boxes" in box_payload:
            box_payload = box_payload["boxes"]
        else:
            box_payload = [box_payload]
    elif _is_flat_box(box_payload):
        box_payload = [box_payload]

    if not isinstance(box_payload, list):
        raise ValueError(
            "boxes/detections must be a list, a single box, or a dict containing predictions/detections"
        )
    return [{"box": _normalize_box_payload(item)} for item in box_payload]


def _append_box_prompts(
    existing_prompts: Any, box_prompts: List[Dict[str, Any]]
) -> Any:
    if existing_prompts is None:
        return {"prompts": box_prompts}
    if isinstance(existing_prompts, Sam2PromptSet):
        prompts = [
            prompt.dict(exclude_none=True) for prompt in existing_prompts.prompts or []
        ]
    elif isinstance(existing_prompts, dict):
        if "prompts" in existing_prompts:
            prompts = existing_prompts.get("prompts") or []
        else:
            prompts = [existing_prompts]
    elif isinstance(existing_prompts, list):
        prompts = existing_prompts
    else:
        return existing_prompts
    return {"prompts": [*prompts, *box_prompts]}


def _is_flat_box(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return False
    try:
        for element in value:
            float(element)
    except (TypeError, ValueError):
        return False
    return True


class Point(BaseModel):
    x: float
    y: float
    positive: bool

    def to_hashable(self) -> Tuple[float, float, bool]:
        return (self.x, self.y, self.positive)


class Sam2Prompt(BaseModel):
    box: Optional[BoxPrompt] = Field(default=None)
    points: Optional[List[Point]] = Field(default=None)

    @validator("box", pre=True)
    def _coerce_box(cls, value):
        if value is None or isinstance(value, (Box, BoxXYXY)):
            return value
        return _normalize_box_payload(value)

    def num_points(self) -> int:
        return len(self.points or [])

    def prediction_metadata(self) -> Dict[str, Any]:
        if self.box is None:
            return {}
        return self.box.prediction_metadata()


class Sam2PromptSet(BaseModel):
    prompts: Optional[List[Sam2Prompt]] = Field(
        default=None,
        description="An optional list of prompts for masks to predict. Each prompt can include a bounding box and / or a set of postive or negative points",
    )

    def num_points(self) -> int:
        if not self.prompts:
            return 0
        return sum(prompt.num_points() for prompt in self.prompts)

    def has_boxes(self) -> bool:
        return any(prompt.box is not None for prompt in self.prompts or [])

    def prediction_metadata(self) -> List[Dict[str, Any]]:
        return [prompt.prediction_metadata() for prompt in self.prompts or []]

    def to_sam2_inputs(self):
        if self.prompts is None:
            return {"point_coords": None, "point_labels": None, "box": None}
        return_dict = {"point_coords": [], "point_labels": [], "box": []}
        for prompt in self.prompts:
            if prompt.box is not None:
                return_dict["box"].append(prompt.box.to_xyxy())
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
        description="The format of the response. Must be one of 'json', 'rle', or 'binary'. If binary, masks are returned as binary numpy arrays. If json, masks are converted to polygons. If rle, masks are converted to RLE format.",
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
        description="A list of prompts for masks to predict. Each prompt can include a bounding box and / or a set of postive or negative points. "
        "Also accepts a flat array of prompts (e.g. 'prompts': [{...}, {...}]) for convenience.",
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
    boxes: Optional[Any] = Field(
        default=None,
        description="Optional list of XYXY boxes or detection dictionaries to segment. "
        "Each item is converted into an interactive box prompt.",
    )
    detections: Optional[Any] = Field(
        default=None,
        description="Optional detection payload to segment. Accepts a list of detections or a dict with a predictions/detections list.",
    )
    box_to_mask: bool = Field(
        default=False,
        description="If true, boxes/detections are treated as interactive box prompts.",
    )
    interactive_box: bool = Field(
        default=False,
        description="Alias for box_to_mask; boxes in this endpoint are interactive prompts.",
    )

    @root_validator(pre=True)
    def _coerce_boxes_to_prompts(cls, values):
        if not isinstance(values, dict):
            return values
        box_payload = values.get("boxes")
        if box_payload is None:
            box_payload = values.get("detections")
        box_prompts = _box_payload_to_prompts(box_payload)
        if not box_prompts:
            return values
        values = values.copy()
        values["prompts"] = _append_box_prompts(values.get("prompts"), box_prompts)
        return values

    # TODO[pydantic]: We couldn't refactor the `validator`, please replace it by `field_validator` manually.
    # Check https://docs.pydantic.dev/dev-v2/migration/#changes-to-validators for more information.
    @validator("prompts", pre=True, always=True)
    def _coerce_prompts(cls, value):
        """
        Accepts any of the following and coerces to Sam2PromptSet:
        - None
        - Sam2PromptSet
        - {"prompts": [...]} (nested)
        - [...] (flat list of prompts)
        - single prompt dict (wrapped to list)
        """
        if value is None:
            return Sam2PromptSet(prompts=None)
        if isinstance(value, Sam2PromptSet):
            return value
        # Nested dict with key 'prompts'
        if isinstance(value, dict):
            if "prompts" in value:
                return Sam2PromptSet(**value)
            # Single prompt dict – wrap and parse
            try:
                return Sam2PromptSet(prompts=[Sam2Prompt(**value)])
            except Exception:
                # Fall-through to attempt generic construction
                return Sam2PromptSet(**value)
        # Flat list of prompts (dicts or Sam2Prompt instances)
        if isinstance(value, list):
            prompts: List[Sam2Prompt] = []
            for item in value:
                if isinstance(item, Sam2Prompt):
                    prompts.append(item)
                elif isinstance(item, dict):
                    prompts.append(Sam2Prompt(**item))
                else:
                    raise ValueError(
                        "Invalid prompt entry; expected dict or Sam2Prompt instance"
                    )
            return Sam2PromptSet(prompts=prompts)
        # Fallback: let Pydantic try
        return value

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

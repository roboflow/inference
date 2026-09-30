"""SAM prompt value objects, owned by Workflows.

`Box`, `Point`, `Sam2Prompt`, `Sam2PromptSet` (from `inference/core/entities/
requests/sam2.py`) and `Sam3Prompt` (from `requests/sam3.py`) were MOVED here
verbatim; both server modules now re-export them, so there is exactly ONE class
object per name. That is what keeps `isinstance(raw_point, Point)` in the two
SAM 3 interactive blocks true for a `Point` built through either import path,
lets the server request classes accept prompt sets the blocks build, and keeps
the `ValidationError` a bad prompt raises the same object either way.

Pinned by `tests/workflows/unit_tests/core_steps/models/foundation/test_sam_prompts.py`
(identity through the re-export, the accepted-input matrix of `_as_sam2_points`,
frozen `to_sam2_inputs()` / payload / validation tables).
"""

from typing import List, Optional, Tuple, Union

from pydantic import BaseModel, Field, validator


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


class Sam3Prompt(BaseModel):
    """Unified prompt that can contain text and/or geometry.

    Absolute pixel coordinates are used for boxes. Labels accept 0/1 or booleans.
    """

    type: Optional[str] = Field(
        default=None,
        description="Optional hint: 'text' or 'visual'. 'visual' requires at least one box.",
    )
    text: Optional[str] = Field(
        default=None,
        description="Concept to segment as a short noun phrase (e.g. 'person'). "
        "All matching instances are returned. Can be combined with exemplar boxes in the same prompt.",
    )

    output_prob_thresh: Optional[float] = Field(
        default=None,
        description="Score threshold for this prompt's outputs. Overrides request-level threshold if set.",
    )

    # Absolute-coordinate boxes (preferred) in pixels.
    # XYWH absolute pixels
    class Box(BaseModel):
        x: float
        y: float
        width: float
        height: float

    # XYXY absolute pixels
    class BoxXYXY(BaseModel):
        x0: float
        y0: float
        x1: float
        y1: float

    # Single unified boxes field; each entry can be XYWH or XYXY
    boxes: Optional[List[Union[Box, BoxXYXY]]] = Field(
        default=None,
        description="Exemplar boxes in absolute pixels, as XYWH entries "
        "({x, y, width, height}, top-left anchored) or XYXY entries ({x0, y0, x1, y1}). "
        "Each box marks an example object; the model segments every instance matching "
        "the exemplars (and text, if provided), not just the boxed objects. "
        "Requires box_labels.",
    )
    box_labels: Optional[List[Union[int, bool]]] = Field(
        default=None,
        description="Per-box exemplar labels, one per entry in boxes: "
        "1/true marks a positive exemplar (segment objects like this), "
        "0/false marks a negative exemplar (exclude objects like this). "
        "Required when boxes is set.",
    )

    @validator("boxes", always=True)
    def _validate_visual_boxes(cls, boxes, values):
        prompt_type = values.get("type")
        if prompt_type == "visual":
            if not boxes or len(boxes) == 0:
                raise ValueError("Visual prompt requires at least one box")
        return boxes

    @validator("box_labels", always=True)
    def _validate_box_labels(cls, labels, values):
        boxes = values.get("boxes")
        if labels is None:
            return labels
        if boxes is None or len(labels) != len(boxes):
            raise ValueError("box_labels must match boxes length when provided")
        return labels

    @validator("output_prob_thresh")
    def _validate_output_prob_thresh(cls, v):
        if v is not None and (v < 0.0 or v > 1.0):
            raise ValueError("output_prob_thresh must be between 0.0 and 1.0")
        return v

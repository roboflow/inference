"""Florence-2 local tasks, decoded answers and detection grounding.

The host supplies models through ModelCache. Each image reaches prompt() as
its original RGB tensor. Standard tasks supply both their task token and full
prompt; custom tasks derive the decoder key from their leading prompt token,
matching the V1 adapter. Decoded boxes, quads and polygons remain unchanged in
parsed_output; image_metadata identifies the input image and its origins.
Text answers are also returned plainly in text. grounding_coordinate_system
reads literal and selected boxes as in V1 (auto) or as explicit pixels or
relative coordinates; native detection boxes keep their own pixels.
"""

import json
import math
import uuid
from typing import Any, Dict, List, Literal, Mapping, NamedTuple, Optional, Union

import torch
from pydantic import (
    Field,
    StrictFloat,
    StrictInt,
    StrictStr,
    ValidationInfo,
    field_validator,
    model_validator,
)
from roboflow_workflows.v2.blocks.models._local_vlm import _image_row, per_row
from roboflow_workflows.v2.blocks.models.cache import ModelCache, ModelRequest
from roboflow_workflows.v2.blocks.models.common import (
    INFERENCE_ID_OUTPUT,
    MODEL_ID_DESCRIPTION,
    MODEL_ID_OUTPUT,
    ModelIdField,
    ModelSelectionParams,
    model_dependency,
    models_cache,
)
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.frames import frame_of
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import (
    DICTIONARY_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
)
from roboflow_workflows.v2.data.predictions import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
)
from roboflow_workflows.v2.engine.declaration import (
    Block,
    BlockParams,
    DependentResource,
    Output,
    Ref,
)

TASK_TOKENS = {
    "open-vocabulary-object-detection": "<OPEN_VOCABULARY_DETECTION>",
    "ocr": "<OCR>",
    "ocr-with-text-detection": "<OCR_WITH_REGION>",
    "caption": "<CAPTION>",
    "detailed-caption": "<DETAILED_CAPTION>",
    "more-detailed-caption": "<MORE_DETAILED_CAPTION>",
    "object-detection-and-caption": "<DENSE_REGION_CAPTION>",
    "object-detection": "<OD>",
    "phrase-grounded-object-detection": "<CAPTION_TO_PHRASE_GROUNDING>",
    "phrase-grounded-instance-segmentation": "<REFERRING_EXPRESSION_SEGMENTATION>",
    "detection-grounded-instance-segmentation": "<REGION_TO_SEGMENTATION>",
    "detection-grounded-classification": "<REGION_TO_CATEGORY>",
    "detection-grounded-caption": "<REGION_TO_DESCRIPTION>",
    "detection-grounded-ocr": "<REGION_TO_OCR>",
    "region-proposal": "<REGION_PROPOSAL>",
    "custom": None,
}
PROMPT_TASKS = {
    "phrase-grounded-object-detection",
    "phrase-grounded-instance-segmentation",
    "custom",
}
GROUNDING_TASKS = {
    task for task in TASK_TOKENS if task.startswith("detection-grounded-")
}
LABEL_TASKS = {
    "<OD>",
    "<DENSE_REGION_CAPTION>",
    "<CAPTION_TO_PHRASE_GROUNDING>",
    "<OCR_WITH_REGION>",
}
TaskType = Literal[tuple(TASK_TOKENS)]
SelectionMode = Literal[
    "first", "last", "biggest", "smallest", "most-confident", "least-confident"
]
SELECTION_MODES = (
    "first",
    "last",
    "biggest",
    "smallest",
    "most-confident",
    "least-confident",
)
ImageParam = Ref(IMAGE_KIND, batch="always")
TaskParam = TaskType | Ref(STRING_KIND, batch="if_varying")
SelectionParam = SelectionMode | Ref(STRING_KIND, batch="if_varying")
CoordinateSystem = Literal["auto", "pixels", "relative"]
COORDINATE_SYSTEMS = ("auto", "pixels", "relative")
TextParam = Optional[StrictStr] | Ref(STRING_KIND, batch="if_varying", allow_none=True)
ClassesParam = Optional[List[StrictStr]] | Ref(
    LIST_OF_VALUES_KIND, batch="if_varying", allow_none=True
)
BoxParam = Union[List[StrictInt], List[StrictFloat]]
GroundingParam = Optional[BoxParam] | Ref(
    LIST_OF_VALUES_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    batch="if_varying",
    allow_none=True,
)


def _is_selector(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(("$inputs.", "$steps."))


def _validate_task(
    task: Any,
    *,
    prompt: Any,
    classes: Any,
    grounding: Any,
    default_task: bool = False,
) -> None:
    if not isinstance(task, str) or task not in TASK_TOKENS:
        raise ValueError(f"Unsupported Florence task_type {task!r}")

    if task in PROMPT_TASKS and (not isinstance(prompt, str) or not prompt.strip()):
        raise ValueError(f"prompt must be non-empty text for task {task!r}")

    if task == "open-vocabulary-object-detection" and not classes:
        hint = (
            " (the default task_type); set classes or choose another task_type"
            if default_task
            else ""
        )
        raise ValueError(f"classes must be a non-empty list for task {task!r}{hint}")

    if task in GROUNDING_TASKS and grounding is None:
        raise ValueError(f"grounding_detection is required for task {task!r}")


def _validate_box(box: Any, *, coordinates: str) -> bool:
    """Check a four-number grounding box and return whether it is relative."""
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError("grounding box must contain exactly four coordinates")

    if coordinates == "auto":
        # V1 typing. JSON writers that drop ".0" (JavaScript) need an
        # explicit system: [0, 0, 1, 1] is pixels here.
        integer = all(type(v) is int for v in box)
        relative = all(type(v) is float for v in box)
        if not (integer or relative) or not all(math.isfinite(v) for v in box):
            raise ValueError(
                "grounding box must contain finite all-integer pixels or all-float "
                "relative coordinates; set grounding_coordinate_system to 'pixels' "
                "or 'relative' to mix integers and floats"
            )
    else:
        if not all(type(v) in (int, float) and math.isfinite(v) for v in box):
            raise ValueError("grounding box must contain four finite numbers")
        relative = coordinates == "relative"

    if relative and not all(0 <= v <= 1 for v in box):
        raise ValueError("relative grounding coordinates must lie in [0, 1]")

    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("grounding box must have positive width and height")

    return relative


class _PreparedImage(NamedTuple):
    image: ImageData
    request: ModelRequest
    task: str
    prompt: str
    decoder_task: str
    classes: List[str]
    skip: bool


class Florence2Block(Block):
    """Run any Florence-2 task using a host-owned local model."""

    type = "v2/florence_2@v2"
    metadata = {"name": "Florence-2", "port_of": "roboflow_core/florence_2@v2"}
    outputs = {
        # Temporary Q3 text convention pending the shared language-kind rebind.
        "raw_output": Output(
            STRING_KIND,
            source="image",
            description="JSON encoding of the decoded task answer, so a text answer is a quoted JSON string; null for a detection-grounded task whose grounding prediction has no box.",
        ),
        "text": Output(
            STRING_KIND,
            source="image",
            description="The decoded answer as plain text for text tasks such as caption and OCR; null for dictionary answers and for detection-grounded rows without a box.",
        ),
        "parsed_output": Output(
            DICTIONARY_KIND,
            source="image",
            description="Decoded Florence dictionary, preserving boxes, polygons and quads; null for text answers. Geometry is in the model package's processed-image pixels: input-image pixels unless the package applies a static crop or resize. image_metadata does not convert it.",
        ),
        "classes": Output(
            LIST_OF_VALUES_KIND,
            source="image",
            description="Requested classes, or decoded labels for V1 label-extracting tasks.",
        ),
        "image_metadata": Output(
            DICTIONARY_KIND,
            source="image",
            description="Input image frame and crop/resize origins; does not convert parsed_output geometry.",
        ),
        "model_id": MODEL_ID_OUTPUT,
        "inference_id": INFERENCE_ID_OUTPUT,
    }

    class Params(ModelSelectionParams):
        """Image, task-dependent prompts and local model selection."""

        image: ImageParam = Field(
            description="RGB input images; one backend call per image."
        )
        model_id: ModelIdField = Field(
            default="florence-2-base",
            description=MODEL_ID_DESCRIPTION,
            examples=["florence-2-base", "$inputs.model_id"],
        )
        task_type: TaskParam = Field(
            default="open-vocabulary-object-detection",
            description="Florence task; determines required prompt, classes or detection grounding.",
        )
        prompt: TextParam = Field(
            default=None,
            description="Text appended to a task token, or the complete custom prompt. Required for phrase grounding and custom tasks.",
        )
        classes: ClassesParam = Field(
            default=None,
            description="Open vocabulary classes, joined with <and>. Overrides prompt as in V1.",
        )
        grounding_coordinate_system: CoordinateSystem = Field(
            default="auto",
            description="How a four-number grounding box is read. auto (V1): all integers are pixels, all floats are relative coordinates in [0, 1], mixed is rejected. pixels or relative: every finite number is read that way, so a JavaScript client's [0, 0, 1, 1] with relative is the whole image. Native detection predictions keep their own pixel boxes.",
        )
        grounding_detection: GroundingParam = Field(
            default=None,
            description="Four coordinates [x0, y0, x1, y1], read as set by grounding_coordinate_system, or a native detection prediction. Overrides prompt/classes as in V1.",
        )
        grounding_selection_mode: SelectionParam = Field(
            default="first",
            description="Which detection box to select; ties use the first row.",
        )

        @field_validator("grounding_detection", mode="before")
        @classmethod
        def _validate_literal_box(cls, value, info: ValidationInfo):
            # Before pydantic converts mixed numbers to floats. The coordinate
            # system is declared first, so it is known here unless invalid.
            coordinates = info.data.get("grounding_coordinate_system")
            if isinstance(value, list) and coordinates is not None:
                _validate_box(value, coordinates=coordinates)
            return value

        @model_validator(mode="after")
        def _validate_task_inputs(self):
            # The engine repeats this check on each resolved row. Selectors
            # count as present until resolution; direct calls check separately.
            if not _is_selector(self.task_type):
                _validate_task(
                    self.task_type,
                    prompt=self.prompt,
                    classes=self.classes,
                    grounding=self.grounding_detection,
                    default_task="task_type" not in self.model_fields_set,
                )
            return self

    def __init__(self, *, models: Union[Mapping[Any, Any], ModelCache]):
        self._models = models_cache(models, block=self.type)

    @classmethod
    def discover_dependent_resources(
        cls, params: BlockParams
    ) -> List[DependentResource]:
        """Describe the selected local model and package.

        Args:
            params: Validated step parameters.

        Returns:
            Model dependency on the host's models resource.
        """
        dependencies = model_dependency(params, task_type="vlm")
        return dependencies

    def run(
        self,
        *,
        image: Batch,
        model_id: Any = "florence-2-base",
        model_package_id: Any = None,
        task_type: Any = "open-vocabulary-object-detection",
        prompt: Any = None,
        classes: Any = None,
        grounding_detection: Any = None,
        grounding_selection_mode: Any = "first",
        grounding_coordinate_system: str = "auto",
    ) -> List[Dict[str, Any]]:
        """Validate every row, resolve models, then decode one answer per image.

        Args:
            image: Batch of RGB ImageData values.
            model_id: Shared model id or one per image in a Batch.
            model_package_id: Shared package id, per-image Batch, or None.
            task_type: Supported task name or per-image Batch.
            prompt: Optional shared text or per-image Batch.
            classes: Optional shared class list or per-image Batch.
            grounding_detection: Shared box/prediction, per-image Batch, or None.
            grounding_selection_mode: Shared selection mode or per-image Batch.
            grounding_coordinate_system: How four-number boxes are read.

        Returns:
            One result per image, including null answers for detection-grounded
            tasks whose grounding prediction has no box.

        Raises:
            ValueError: On invalid task inputs or grounding frames (naming the
                image's logical index), a per-image Batch whose indices differ
                from the image batch's, or answer counts.
        """
        if grounding_coordinate_system not in COORDINATE_SYSTEMS:
            raise ValueError(
                f"grounding_coordinate_system must be one of {list(COORDINATE_SYSTEMS)}, "
                f"got {grounding_coordinate_system!r}"
            )

        columns = [
            per_row(value, image=image, name=name)
            for name, value in (
                ("model_id", model_id),
                ("model_package_id", model_package_id),
                ("task_type", task_type),
                ("prompt", prompt),
                ("classes", classes),
                ("grounding_detection", grounding_detection),
                ("grounding_selection_mode", grounding_selection_mode),
            )
        ]
        prepared = []
        for index, member, values in zip(image.indices, image, zip(*columns)):
            with _image_row(index, block=self.type):
                prepared.append(
                    _prepared_image(
                        member, *values, coordinates=grounding_coordinate_system
                    )
                )

        requests = dict.fromkeys(row.request for row in prepared if not row.skip)
        models = {request: self._models.get(request) for request in requests}
        for request, model in models.items():
            if not callable(getattr(model, "prompt", None)):
                raise ValueError(f"Florence model {request!s} must provide prompt()")

        results = []
        for row in prepared:
            decoder_task = row.decoder_task
            if row.skip:
                result = {
                    "raw_output": None,
                    "text": None,
                    "parsed_output": None,
                    "classes": None,
                }
            else:
                answers = models[row.request].prompt(
                    [row.image.tensor_image],
                    prompt=row.prompt,
                    task=decoder_task,
                    input_color_format="rgb",
                )
                if not isinstance(answers, list) or len(answers) != 1:
                    raise ValueError(
                        f"Florence model must return exactly one answer per image, got {answers!r}"
                    )
                answer = answers[0]
                if not isinstance(answer, dict) or decoder_task not in answer:
                    raise ValueError(
                        f"Florence model answer must contain decoder task {decoder_task!r}"
                    )
                decoded = answer[decoder_task]
                if not isinstance(decoded, (dict, str)):
                    raise ValueError(
                        "Florence decoded answer must be a dictionary or text"
                    )
                extracted = (
                    decoded.get("labels", [])
                    if TASK_TOKENS[row.task] in LABEL_TASKS
                    and isinstance(decoded, dict)
                    else row.classes
                )
                if not isinstance(extracted, list) or not all(
                    isinstance(c, str) for c in extracted
                ):
                    raise ValueError(
                        "Florence decoded labels must be a list of strings"
                    )
                result = {
                    "raw_output": json.dumps(decoded, allow_nan=False),
                    "text": decoded if isinstance(decoded, str) else None,
                    "parsed_output": decoded if isinstance(decoded, dict) else None,
                    "classes": extracted,
                }
            result.update(
                model_id=row.request.model_id,
                inference_id=uuid.uuid4().hex,
                image_metadata=row.image.prediction_metadata(),
            )
            results.append(result)
        return results


def _prepared_image(
    member: Any,
    model_id: Any,
    package_id: Any,
    task: Any,
    text: Any,
    labels: Any,
    grounding: Any,
    mode: Any,
    *,
    coordinates: str,
) -> _PreparedImage:
    if not isinstance(member, ImageData) or member.channels != 3:
        raise ValueError("Florence-2 needs a 3-channel RGB ImageData")
    _validate_task(task, prompt=text, classes=labels, grounding=grounding)
    if text is not None and not isinstance(text, str):
        raise ValueError("prompt must be text or None")
    if labels is not None and (
        not isinstance(labels, list)
        or not labels
        or not all(isinstance(c, str) and c.strip() for c in labels)
    ):
        raise ValueError("classes must be a non-empty list of non-empty strings")
    if mode not in SELECTION_MODES:
        raise ValueError(f"Unknown grounding selection mode: {mode!r}")
    request = ModelRequest(model_id, package_id)
    suffix = text or ""
    if labels is not None:
        suffix = "<and>".join(labels)
    if grounding is not None:
        suffix = _grounding_prompt(
            member, grounding=grounding, mode=mode, coordinates=coordinates
        )
    skip = suffix is None and task in GROUNDING_TASKS
    final_prompt = (TASK_TOKENS[task] or "") + (suffix or "")
    decoder_task = TASK_TOKENS[task] or final_prompt.split(">")[0] + ">"
    prepared = _PreparedImage(
        member, request, task, final_prompt, decoder_task, labels or [], skip
    )

    return prepared


def _grounding_prompt(
    image: ImageData, *, grounding: Any, mode: str, coordinates: str
) -> Optional[str]:
    if isinstance(grounding, list):
        box = grounding
        relative = _validate_box(box, coordinates=coordinates)
    else:
        box = _selected_box(image, prediction=grounding, mode=mode)
        if box is None:
            return None
        relative = False
    height, width = image.size_hw
    normalized = (
        box
        if relative
        else [box[0] / width, box[1] / height, box[2] / width, box[3] / height]
    )
    prompt = "".join(
        f"<loc_{min(999, round(max(0, min(1, v)) * 1000))}>" for v in normalized
    )
    return prompt


def _selected_box(
    image: ImageData, *, prediction: Any, mode: str
) -> Optional[List[float]]:
    # Optional model payload imports stay out of module initialization.
    from inference_models.models.base.instance_segmentation import InstanceDetections
    from inference_models.models.base.object_detection import Detections

    # A keypoint prediction is a (KeyPoints, Detections) pair; its boxes are
    # in the Detections.
    if isinstance(prediction, tuple) and len(prediction) == 2:
        prediction = prediction[1]
    if not isinstance(prediction, (Detections, InstanceDetections)):
        raise ValueError(
            "grounding_detection must be a native detection, instance or keypoint prediction with boxes"
        )
    xyxy = prediction.xyxy
    if not isinstance(xyxy, torch.Tensor) or xyxy.ndim != 2 or xyxy.shape[1] != 4:
        raise ValueError("grounding detections must have shape (N, 4)")
    if not torch.isfinite(xyxy).all() or (xyxy[:, 2:] <= xyxy[:, :2]).any():
        raise ValueError("grounding detections must have finite, positive-area boxes")
    if len(xyxy) == 0:
        return None
    if mode == "first":
        index = 0
    elif mode == "last":
        index = len(xyxy) - 1
    elif mode in ("biggest", "smallest"):
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        index = int(torch.argmax(areas) if mode == "biggest" else torch.argmin(areas))
    else:
        confidence = prediction.confidence
        if (
            not isinstance(confidence, torch.Tensor)
            or confidence.shape != (len(xyxy),)
            or not torch.isfinite(confidence).all()
        ):
            raise ValueError("confidence selection needs one finite confidence per box")
        index = int(
            torch.argmax(confidence)
            if mode == "most-confident"
            else torch.argmin(confidence)
        )
    box = xyxy[index].detach().to("cpu").tolist()
    # Metadata-less native boxes use image-local pixels, as in V1. Framed
    # boxes are mapped through the common root, including parent crops/resizes.
    if prediction.image_metadata is not None:
        source = frame_of(prediction)
        if source.image_id == image.image_id:
            if source.size_hw != image.size_hw or source.root != image.root:
                raise ValueError("grounding frame disagrees with the image")
        else:
            if (
                source.is_composite
                or image.composite_sources is not None
                or source.root.frame_id is None
                or source.root.frame_id != image.root.frame_id
                or source.root.frame_size_hw != image.root.frame_size_hw
            ):
                raise ValueError(
                    "grounding prediction and image must share a known workflow root"
                )
            sx, sy = source.root.scale_xy
            ox, oy = source.root.offset_xy
            tx, ty = image.root.scale_xy
            dx, dy = image.root.offset_xy
            box = [
                (box[0] * sx + ox - dx) / tx,
                (box[1] * sy + oy - dy) / ty,
                (box[2] * sx + ox - dx) / tx,
                (box[3] * sy + oy - dy) / ty,
            ]
    return box


BLOCKS = (Florence2Block,)

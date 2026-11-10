"""SAM3 interactive segmentation: one mask per prompt box or point set.

The block gives the instance-segmentation outputs of decision 018: a
``segmentation`` view to wire to blocks and ``predictions``, its dense
serializable ``InstanceDetections``. Each output is built only when a reader
wants it. Masks and boxes are on the image's device; boxes are half-open
pixel extents of each mask, ``[0, 0, 0, 0]`` for an empty mask (as in V1)::

    roboflow_core/sam3_interactive@v1 (SAM3 promptable visual segmentation)
      boxes wired     one row per prompt row: the prompt's class and row dict
      points given    one more row for the whole point set: "foreground"
      neither         compile error (literal) or run error (selected)
        embed each image that has a prompt (reused while the same pixels
        come back); both prompt groups of an image share its embedding
        segment_with_visual_prompts(boxes)                 -> P box rows
        segment_with_visual_prompts(points, labels)        -> 1 point row
        logits >= 0 are mask pixels; keep rows with score >= confidence

Rows keep the box prompt's row identity when the step has no point prompt,
so a mask row and its detector row stay the same row. A point row is not a
prompt row: with points, every row of the result gets a fresh identity, and
box rows keep their prompt row dicts for lineage. The reserved host mirrors
(``_host_xyxy``, ``_host_class_id``, ``_host_confidence``) are dropped from
those dicts: they describe the prompt's box, not the mask's. The point
row's class id is 0 without boxes, otherwise one past every class id of the
box prompt, so it never takes a detector class's name.

SAM3's image embedding is kept by the step, never by ``ModelCache``, with the
SAM2 block's rule: a later call with the same pixel tensor and the same model
request reuses it; only the latest call's embeddings are kept;
``reset_state()`` and a processing reset drop them.

Deliberate differences from V1 (``segment_anything3_interactive/v1_tensor.py``):

- Mask boxes include the full last occupied pixel: ``max + 1`` on the right
  and bottom, correcting V1 inclusive bounds so crops retain every pixel.
- The model comes from the ``models`` resource by ``model_id`` (default
  ``sam3/sam3_interactive``) and ``model_package_id``; V1's model was fixed.
  ``threshold`` is now ``confidence``, checked to be 0..1.
- The point row's class id is 0 or one past the box prompt's ids; V1 used -1,
  which V2 drawing blocks reject.
- Box rows keep the prompt rows' dicts, less the reserved host mirrors,
  and, without points, their row identity; V1 copied only ``detection_id``
  and the class name. A box row whose class id has no name stays unnamed;
  V1 named it "foreground".
- Points are checked strictly: unknown keys, labels other than true/false/1/0,
  non-finite coordinates and points outside the image are errors. V1 ignored
  unknown keys, read any truthy label as positive and sent outside points to
  the model.
- Literal points and boxes are checked at compile time; a step with neither
  fails to compile. Prompt boxes must be predicted on the image they prompt.
- One embedding per image, kept by the step and shared by the box and point
  prompts; V1 used the model's own embeddings cache.
- Box prompts reach the model as ``xyxy`` unchanged; V1 converted them to
  centre and size and back, which can change the last bit of a float.
- Outputs are a view and dense masks; V1 carried RLE masks.

Hosted execution is deferred, not ported: V1's inference-server mode
(``StepExecutionMode.REMOTE``) and its platform proxy (``SAM3_EXEC_MODE``).
A step that asks for another mode fails validation before any request.
"""

import math
import uuid
from typing import (
    Annotated,
    Any,
    Dict,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections
from pydantic import Field, StrictBool, StrictFloat, StrictStr, model_validator
from roboflow_workflows.v2.blocks.geometry._rows import without_host_mirrors
from roboflow_workflows.v2.blocks.models._shared.segment_anything import (
    FOREGROUND,
    MASK_LOGIT_THRESHOLD,
    PREDICTION_TYPE,
    _Embedding,
    _mask_boxes,
    _metadata,
    _no_instances,
    _prompt_boxes,
    _require_same_frame,
    _segmentation_result,
    _unit_float,
)
from roboflow_workflows.v2.blocks.models.cache import ModelCache, ModelRequest
from roboflow_workflows.v2.blocks.models.common import (
    models_cache,
    require_rgb,
)
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import (
    BOOLEAN_KIND,
    FLOAT_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
)
from roboflow_workflows.v2.data.masks.grid import INSTANCE_SEGMENTATION_VIEW_KIND
from roboflow_workflows.v2.data.prediction_ops import (
    RowSelection,
    class_names_by_id,
    take_rows,
)
from roboflow_workflows.v2.data.predictions import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
)
from roboflow_workflows.v2.data.row_identity import ROW_RECORD_KEY, with_row_identity
from roboflow_workflows.v2.engine.declaration import (
    Block,
    BlockParams,
    DependentResource,
    Output,
    Ref,
)


BLOCK_TYPE = "roboflow_core/sam3_interactive@v1"

SAM3_INTERACTIVE_DEFAULT_MODEL_ID = "sam3/sam3_interactive"

POINT_KEYS = frozenset({"x", "y", "positive"})

class _PointPrompt(NamedTuple):
    """One object's labeled points, shaped as SAM3 reads one prompt."""

    # float64 (1, N, 2) absolute pixel x, y.
    coordinates: torch.Tensor
    # int64 (1, N): 1 marks the object, 0 excludes a region.
    labels: torch.Tensor

class SegmentAnything3InteractiveBlock(Block):
    """Segment the object of each prompt box, and of a point set, with SAM3.

    Each prompt row of ``boxes`` gets one mask, SAM3's best guess for its box,
    with the row's class. ``points`` describe one more object: positive points
    mark it, negative points exclude regions; its row is named "foreground".
    Rows scoring below ``confidence`` are dropped. Without points, kept rows
    keep the box prompt's row identity. SAM3 runs locally from the ``models``
    resource; hosted execution is deferred (see the module docstring).
    """

    type = BLOCK_TYPE
    metadata = {
        "name": "SAM 3 Interactive",
        "port_of": "roboflow_core/sam3_interactive@v1",
        "section": "model",
        "search_keywords": [
            "SAM3",
            "META",
            "point prompt",
            "interactive segmentation",
            "PVS",
        ],
    }
    outputs = {
        "predictions": Output(
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
            source="image",
            description=(
                "Serializable InstanceDetections with a dense (N, H, W) bool mask in "
                "this image's pixel coordinates: segmentation's full-resolution "
                "result, same rows. Use it as a workflow output; wire segmentation "
                "to blocks."
            ),
        ),
        "segmentation": Output(
            INSTANCE_SEGMENTATION_VIEW_KIND,
            source="image",
            description=(
                "SegmentationView of the masks: wire it to blocks. Not "
                "serializable; use predictions as a workflow output."
            ),
        ),
        "model_id": Output(
            STRING_KIND,
            source="image",
            description="Model id that served this image.",
        ),
        "inference_id": Output(
            STRING_KIND,
            source="image",
            description=(
                "Fresh id of this image's result; for results with rows, also the "
                "namespace of their row identity."
            ),
        ),
    }

    class Params(BlockParams):
        # noqa F821: flake8 parses the string "always" in an annotation as a name.
        image: Ref(IMAGE_KIND, batch="always") = Field(  # noqa: F821
            description="RGB images to segment; delivered as one batch."
        )
        points: Optional[List[Any]] | Ref(LIST_OF_VALUES_KIND, allow_none=True) = Field(
            default=None,
            description=(
                "Labeled points of one object, the same for every image, in "
                "absolute pixels: {'x', 'y', 'positive'} objects or [x, y] / "
                "[x, y, positive] lists. positive (default true) marks the "
                "object; false or 0 excludes a region. Points must lie inside "
                "each image."
            ),
            examples=[[{"x": 320, "y": 240, "positive": True}], "$inputs.points"],
        )
        boxes: Optional[
            Ref(
                OBJECT_DETECTION_PREDICTION_KIND,
                INSTANCE_SEGMENTATION_PREDICTION_KIND,
                INSTANCE_SEGMENTATION_VIEW_KIND,
                KEYPOINT_DETECTION_PREDICTION_KIND,
                batch="always",  # noqa: F821
            )
        ] = Field(
            default=None,
            description=(
                "Prompt boxes predicted on the same images, one mask per box. "
                "Keypoint predictions need their boxes. Give points, boxes or both."
            ),
            examples=["$steps.object_detection.predictions"],
        )
        confidence: Annotated[StrictFloat, Field(ge=0, le=1)] | Ref(FLOAT_KIND) = Field(
            default=0.0,
            description=(
                "Minimum SAM3 mask score (its predicted IoU), 0..1. V1 called it "
                "threshold."
            ),
            examples=[0.0, 0.5, "$inputs.sam_confidence"],
        )
        multimask_output: StrictBool | Ref(BOOLEAN_KIND) = Field(
            default=True,
            description=(
                "Let SAM3 propose three masks per prompt and keep the best one; "
                "recommended for ambiguous prompts such as one point. False asks "
                "for one mask."
            ),
            examples=[True, "$inputs.multimask_output"],
        )
        model_id: StrictStr | Ref(STRING_KIND) = Field(
            default=SAM3_INTERACTIVE_DEFAULT_MODEL_ID,
            description=(
                "SAM3 model id, or local model package directory, in the 'models' "
                "resource."
            ),
            examples=[SAM3_INTERACTIVE_DEFAULT_MODEL_ID, "$inputs.sam3_model_id"],
        )
        model_package_id: Optional[StrictStr] | Ref(STRING_KIND, allow_none=True) = (
            Field(
                default=None,
                description=(
                    "Exact model package to load; None lets the loader choose. List "
                    "a model's package ids with "
                    'inference_models.AutoModel.describe_model("<model_id>").'
                ),
                examples=[None],
            )
        )

        @model_validator(mode="after")
        def _check_literals(self) -> "SegmentAnything3InteractiveBlock.Params":
            # Selected points are checked at run time, against each image too.
            if isinstance(self.points, str) and self.points.startswith("$"):
                return self

            has_points = bool(_labeled_points(self.points, block=BLOCK_TYPE))
            if not has_points and self.boxes is None:
                raise ValueError(f"{BLOCK_TYPE} needs points, boxes or both")

            return self

    def __init__(self, *, models: Union[Mapping[Any, Any], ModelCache]):
        self._models = models_cache(models, block=self.type)
        self._embeddings: Dict[Tuple[ModelRequest, str], _Embedding] = {}

    @classmethod
    def discover_dependent_resources(
        cls, params: BlockParams
    ) -> List[DependentResource]:
        """Report the SAM3 model this step needs from the ``models`` resource."""
        resources = [
            DependentResource(
                resource_type="model",
                identifier=params.model_id,
                details={
                    "resource": "models",
                    "task_type": PREDICTION_TYPE,
                    "model_package_id": params.model_package_id,
                },
            )
        ]

        return resources

    def run(
        self,
        *,
        image: Batch,
        points: Optional[Sequence[Any]] = None,
        boxes: Optional[Batch] = None,
        confidence: float = 0.0,
        multimask_output: bool = True,
        model_id: str = SAM3_INTERACTIVE_DEFAULT_MODEL_ID,
        model_package_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Segment every image of the batch.

        Inside an engine call, an output no reader wants is left out.

        Returns:
            One ``{"predictions", "segmentation", "model_id", "inference_id"}``
            per image, in batch order.

        Raises:
            ValueError: On a non-RGB image, an invalid option or point, no
                prompt at all, prompts of another image or a model result that
                does not match its prompts.
        """
        confidence = _unit_float(confidence, name="confidence", block=self.type)
        if not isinstance(multimask_output, bool):
            raise ValueError(
                f"{self.type}: multimask_output must be true or false, got "
                f"{multimask_output!r}"
            )

        parsed_points = _labeled_points(points, block=self.type)
        if boxes is None and not parsed_points:
            raise ValueError(
                f"{self.type} needs points, boxes or both; the selected points "
                "are empty and no boxes are wired"
            )

        members = image.content
        prompts = [None] * len(members) if boxes is None else list(boxes.content)
        if len(prompts) != len(members):
            raise ValueError(
                f"{self.type}: boxes has {len(prompts)} values for {len(members)} "
                "images"
            )

        prompts = [_box_prompt(prompt, block=self.type) for prompt in prompts]
        for member, prompt in zip(members, prompts):
            require_rgb(member, block=self.type)
            _require_same_frame(prompt, member, block=self.type)
            _require_points_inside(parsed_points, member, block=self.type)
        point_prompt = _point_prompt(parsed_points)

        request = ModelRequest(model_id, model_package_id)
        model = self._models.get(request)
        needed = [
            point_prompt is not None or (prompt is not None and len(prompt) > 0)
            for prompt in prompts
        ]
        embeddings = self._embed(model, request=request, images=members, needed=needed)

        results = []
        for member, prompt, embedding in zip(members, prompts, embeddings):
            inference_id = uuid.uuid4().hex
            if embedding is None:
                predictions = _no_instances(
                    member, prompt=prompt, namespace=inference_id
                )
            else:
                predictions = _interactive_instances(
                    model,
                    embedding=embedding,
                    image=member,
                    prompt=prompt,
                    points=point_prompt,
                    confidence=confidence,
                    multimask_output=multimask_output,
                    namespace=inference_id,
                    block=self.type,
                )
            results.append(
                _segmentation_result(
                    predictions,
                    reader=self,
                    model_id=model_id,
                    inference_id=inference_id,
                )
            )

        return results

    def reset_state(self) -> None:
        """Forget the kept image embeddings; the shared model stays loaded."""
        self._embeddings = {}

    def _embed(
        self,
        model: Any,
        *,
        request: ModelRequest,
        images: Sequence[ImageData],
        needed: Sequence[bool],
    ) -> List[Any]:
        """Each needed image's embedding: kept from the last call or computed.

        All missing images are embedded in one ``embed_images`` call, without
        the model's own embeddings cache. The kept set is replaced by this
        call's embeddings.
        """
        previous = self._embeddings
        kept: Dict[Tuple[ModelRequest, str], _Embedding] = {}
        found: List[Any] = [None] * len(images)
        missing = []
        for row, member in enumerate(images):
            if not needed[row]:
                continue

            key = (request, member.image_id)
            entry = kept.get(key) or previous.get(key)
            if entry is not None and entry.holds(member.tensor_image):
                kept[key] = entry
                found[row] = entry.embeddings
            else:
                missing.append(row)

        if missing:
            computed = model.embed_images(
                [images[row].tensor_image for row in missing],
                use_embeddings_cache=False,
            )
            if len(computed) != len(missing):
                raise ValueError(
                    f"{self.type}: embed_images of model {str(request)!r} returned "
                    f"{len(computed)} embeddings for {len(missing)} images"
                )

            for row, embeddings in zip(missing, computed):
                member = images[row]
                kept[(request, member.image_id)] = _Embedding(
                    member.tensor_image, embeddings
                )
                found[row] = embeddings
        self._embeddings = kept

        return found

def _labeled_points(
    points: Optional[Sequence[Any]], *, block: str
) -> List[Tuple[float, float, bool]]:
    """The step's points as ``(x, y, positive)``; empty for ``None`` or ``[]``.

    A point is ``{"x", "y", "positive"}`` (``positive`` optional) or an
    ``[x, y]`` / ``[x, y, positive]`` list. Coordinates are finite numbers;
    ``positive`` is a bool or the integer 1 or 0, and defaults to true.
    """
    if points is None:
        return []

    if not isinstance(points, (list, tuple)):
        raise ValueError(
            f"{block}: points must be a list of points, got {type(points).__name__}"
        )

    parsed = [_labeled_point(point, block=block) for point in points]

    return parsed

def _labeled_point(point: Any, *, block: str) -> Tuple[float, float, bool]:
    if isinstance(point, Mapping):
        if not {"x", "y"} <= set(point) <= POINT_KEYS:
            raise ValueError(
                f"{block}: a point object has the keys x, y and optionally "
                f"positive, got {point!r}"
            )
        x, y, positive = point["x"], point["y"], point.get("positive", True)
    elif isinstance(point, (list, tuple)) and len(point) in (2, 3):
        x, y = point[0], point[1]
        positive = point[2] if len(point) == 3 else True
    else:
        raise ValueError(
            f"{block}: a point is an {{x, y, positive}} object or an [x, y] or "
            f"[x, y, positive] list, got {point!r}"
        )

    for value in (x, y):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{block}: point coordinates are numbers, got {point!r}")
        if not math.isfinite(value):
            raise ValueError(
                f"{block}: point coordinates are finite numbers, got {point!r}"
            )

    # 1 == True in Python, so check the type before the value.
    is_label = isinstance(positive, bool) or (
        type(positive) is int and positive in (0, 1)
    )
    if not is_label:
        raise ValueError(
            f"{block}: a point's positive label is true, false, 1 or 0, got "
            f"{point!r}"
        )

    parsed = (float(x), float(y), bool(positive))

    return parsed

def _point_prompt(
    points: Sequence[Tuple[float, float, bool]],
) -> Optional[_PointPrompt]:
    if not points:
        return None

    prompt = _PointPrompt(
        coordinates=torch.tensor([[[x, y] for x, y, _ in points]], dtype=torch.float64),
        labels=torch.tensor([[int(positive) for _, _, positive in points]]),
    )

    return prompt

def _require_points_inside(
    points: Sequence[Tuple[float, float, bool]], image: ImageData, *, block: str
) -> None:
    height, width = image.size_hw
    for x, y, _ in points:
        if not (0 <= x <= width and 0 <= y <= height):
            raise ValueError(
                f"{block}: point ({x}, {y}) is outside image {image.image_id!r} of "
                f"size {image.size_hw}; points are absolute pixel coordinates"
            )

def _box_prompt(prompt: Any, *, block: str) -> Optional[Detections]:
    """The prompt's boxes as ``Detections``; ``None`` when there is no prompt."""
    if isinstance(prompt, tuple) and prompt[1] is None:
        raise ValueError(
            f"{block}: the keypoint prediction has no boxes; SAM3 needs boxes as "
            "box prompts"
        )

    boxes = _prompt_boxes(prompt, block=block)

    return boxes

def _interactive_instances(
    model: Any,
    *,
    embedding: Any,
    image: ImageData,
    prompt: Optional[Detections],
    points: Optional[_PointPrompt],
    confidence: float,
    multimask_output: bool,
    namespace: str,
    block: str,
) -> InstanceDetections:
    """One image's rows: box prompt rows first, then the point row."""
    device = image.device
    has_boxes = prompt is not None and len(prompt) > 0
    prompt_metadata = {} if prompt is None else prompt.image_metadata or {}
    names = dict(class_names_by_id(prompt_metadata) or {})
    groups = []
    class_ids = []
    row_dicts = None
    if has_boxes:
        groups.append(
            _visual_prompt_masks(
                model,
                embedding=embedding,
                image=image,
                prompt_count=len(prompt),
                multimask_output=multimask_output,
                block=block,
                boxes=[prompt.xyxy],
            )
        )
        class_ids.append(prompt.class_id.to(device=device, dtype=torch.int64))
        if prompt.bboxes_metadata is not None:
            row_dicts = [without_host_mirrors(row) for row in prompt.bboxes_metadata]

    if points is None:
        # Every row is a prompt row: keep the prompt's identity.
        record = prompt_metadata.get(ROW_RECORD_KEY)
    else:
        record = None
        groups.append(
            _visual_prompt_masks(
                model,
                embedding=embedding,
                image=image,
                prompt_count=1,
                multimask_output=multimask_output,
                block=block,
                point_coordinates=[points.coordinates],
                point_labels=[points.labels],
            )
        )
        point_class_id = _point_class_id(prompt)
        names[point_class_id] = FOREGROUND
        class_ids.append(torch.tensor([point_class_id], device=device))
        if row_dicts is not None:
            row_dicts.append({})

    mask = torch.cat([group[0] for group in groups])
    scores = torch.cat([group[1] for group in groups])
    metadata = _metadata(image, class_names=names)
    if record is not None:
        metadata[ROW_RECORD_KEY] = record
    rows = InstanceDetections(
        xyxy=_mask_boxes(mask),
        class_id=torch.cat(class_ids),
        confidence=scores,
        mask=mask,
        image_metadata=metadata,
        bboxes_metadata=row_dicts,
    )

    kept = take_rows(rows, RowSelection.of(len(rows), mask=scores >= confidence))
    if record is None:
        kept = with_row_identity(kept, namespace=namespace)

    return kept

def _visual_prompt_masks(
    model: Any,
    *,
    embedding: Any,
    image: ImageData,
    prompt_count: int,
    multimask_output: bool,
    block: str,
    **prompts: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One prompt group's bool ``(P, H, W)`` masks and ``(P,)`` scores.

    SAM3Torch already keeps the best proposal per prompt, so its masks are
    ``(P, H, W)`` logits at the image's size.
    """
    (prediction,) = model.segment_with_visual_prompts(
        embeddings=[embedding],
        multi_mask_output=multimask_output,
        return_logits=True,
        **prompts,
    )
    logits = torch.as_tensor(prediction.masks)
    scores = torch.as_tensor(prediction.scores)
    expected = (prompt_count, *image.size_hw)
    if tuple(logits.shape) != expected or tuple(scores.shape) != (prompt_count,):
        raise ValueError(
            f"{block}: SAM3 returned masks of shape {tuple(logits.shape)} and "
            f"scores of shape {tuple(scores.shape)} for {prompt_count} prompts on "
            f"a {image.size_hw} image"
        )

    device = image.device
    binary = (logits >= MASK_LOGIT_THRESHOLD).to(device)
    scores = scores.to(device=device, dtype=torch.float32)

    return binary, scores

def _point_class_id(prompt: Optional[Detections]) -> int:
    """0 without boxes; else one past every class id the box prompt uses."""
    if prompt is None:
        return 0

    names = class_names_by_id(prompt.image_metadata) or {}
    used = {int(class_id) for class_id in names}
    used.update(prompt.class_id.tolist())
    point_class_id = max(used, default=-1) + 1

    return point_class_id

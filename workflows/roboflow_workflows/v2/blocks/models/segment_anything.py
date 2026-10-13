"""Segment Anything: SAM2 masks for boxes or a whole image, SAM3 masks for text.

Both blocks give the instance-segmentation outputs of decision 018: a
``segmentation`` view to wire to blocks and ``predictions``, its dense
serializable ``InstanceDetections``. Each output is built only when a reader
wants it. Masks and boxes are on the image's device; boxes are the inclusive
pixel bounds of each mask, ``[0, 0, 0, 0]`` for an empty mask (as in V1)::

    v2/segment_anything@v1 (SAM2)
      boxes not wired   one mask per image: SAM2's best guess, "foreground"
      boxes wired       one mask per prompt row; rows keep the prompt's class,
                        row dict and row identity; no rows -> no masks
        embed each image (reused while the same pixels come back)
        segment_images(embeddings, boxes)   -> best of SAM2's proposals per row
        keep rows with score >= confidence

    v2/sam3@v3 (SAM3, local only)
      one text prompt per class name; no names = one empty prompt "foreground"
        segment_with_text_prompts(mask_format="dense")
        per class: score >= its threshold; drop empty masks
        apply_nms: greedy mask IoU across classes, IoU > threshold drops
        class_id = prompt index, class name = class_mapping applied to it

SAM2's image embedding is kept by the step, never by ``ModelCache``: a later
call with the same pixel tensor (same ``image_id`` and tensor object) and the
same model request reuses it. Only the latest call's embeddings are kept;
``reset_state()`` and a processing reset drop them. The shared model is never
unloaded.

Deliberate differences from V1 (``segment_anything2/v1_tensor.py``,
``segment_anything3/v3_tensor.py``):

- Models come from the ``models`` resource by ``model_id``; SAM2's ``version``
  becomes ``model_id="sam2/<version>"``. ``threshold`` is now ``confidence``.
- SAM2 with wired boxes but no box rows returns no masks; V1 segmented the
  whole image as "foreground".
- SAM2 output rows keep the prompt rows' metadata dicts and row identity; V1
  copied only ``detection_id`` and ``class``.
- SAM3 drops empty masks (V1 v3 raised on them) and validates the
  ``per_class_confidence`` length at run time too. Comma-separated class names
  are stripped of surrounding spaces. Its model's score floor is the lowest
  class threshold; V1 also lowered it to ``confidence``. Each class's own
  threshold follows either way. NMS visits equal scores in row order; V1
  visited them in reverse row order.
- ``output_format`` is gone: outputs are a view and dense predictions, not
  RLE or polygons.

Hosted execution is deferred, not ported: V1's inference-server mode
(``StepExecutionMode.REMOTE``) and SAM3's platform proxy (``SAM3_EXEC_MODE``).
Roboflow's next inference server will define that contract; until then both
blocks run only the local model from the ``models`` resource, and a step that
asks for another mode fails validation before any request is made.
"""

import re
import uuid
import weakref
from typing import (
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

import numpy as np
import torch
from pycocotools import mask as mask_utils
from pydantic import Field, StrictBool, StrictStr, model_validator
from roboflow_workflows.v2.blocks.geometry._rows import without_host_mirrors
from roboflow_workflows.v2.blocks.models.cache import ModelCache, ModelRequest
from roboflow_workflows.v2.blocks.models.common import (
    INFERENCE_ID_OUTPUT,
    MODEL_ID_OUTPUT,
    UnitFloat,
    model_dependency,
    models_cache,
)
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.frames import FRAME_RECORD_KEY, FrameRef
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import (
    BOOLEAN_KIND,
    DICTIONARY_KIND,
    FLOAT_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
)
from roboflow_workflows.v2.data.masks.grid import (
    INSTANCE_SEGMENTATION_VIEW_KIND,
    SegmentationView,
)
from roboflow_workflows.v2.data.prediction_ops import (
    CLASS_NAMES_KEY,
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
    DATA_SELECTOR_PATTERN,
    Block,
    BlockParams,
    DependentResource,
    Output,
    Ref,
)

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections

PREDICTION_TYPE = "instance-segmentation"
FOREGROUND = "foreground"
SAM2_DEFAULT_MODEL_ID = "sam2/hiera_tiny"
SAM3_DEFAULT_MODEL_ID = "sam3/sam3_final"
# SAM2 mask logits at or above this value are foreground, as in V1.
MASK_LOGIT_THRESHOLD = 0.0

SEGMENTATION_OUTPUTS = {
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
    "model_id": MODEL_ID_OUTPUT,
    "inference_id": INFERENCE_ID_OUTPUT,
}

PACKAGE_DESCRIPTION = (
    "Exact model package to load; None lets the loader choose. List a model's "
    'package ids with inference_models.AutoModel.describe_model("<model_id>").'
)


class _Row(NamedTuple):
    """One SAM3 candidate: prompt index, host bool mask, and score."""

    prompt_index: int
    mask: torch.Tensor
    score: float


class SegmentAnything2Block(Block):
    """Segment the object inside each prompt box, or the whole image, with SAM2.

    Without ``boxes`` each image gets one mask, SAM2's most confident guess for
    the whole image, with class id 0 and class name "foreground". With
    ``boxes`` each prompt row gets the most confident of SAM2's proposals for
    its box; rows scoring below ``confidence`` are dropped. Kept rows keep the
    prompt's class id, class names, row dict and row identity, so a mask row
    and its detector row stay the same row.
    """

    type = "v2/segment_anything@v1"
    metadata = {
        "name": "Segment Anything 2 Model",
        "port_of": "roboflow_core/segment_anything@v1",
        "section": "model",
        "search_keywords": ["SAM2", "META"],
    }
    outputs = SEGMENTATION_OUTPUTS

    class Params(BlockParams):
        # noqa F821: flake8 parses the string "always" in an annotation as a name.
        image: Ref(IMAGE_KIND, batch="always") = Field(  # noqa: F821
            description="RGB images to segment; delivered as one batch."
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
                "Omit to segment each whole image. Keypoint predictions need "
                "their boxes."
            ),
            examples=["$steps.object_detection.predictions"],
        )
        model_id: StrictStr | Ref(STRING_KIND) = Field(
            default=SAM2_DEFAULT_MODEL_ID,
            description=(
                "SAM2 model id in the 'models' resource: sam2/hiera_tiny, "
                "sam2/hiera_small, sam2/hiera_b_plus or sam2/hiera_large."
            ),
            examples=["sam2/hiera_large", "$inputs.sam_model_id"],
        )
        model_package_id: Optional[StrictStr] | Ref(STRING_KIND, allow_none=True) = (
            Field(default=None, description=PACKAGE_DESCRIPTION, examples=[None])
        )
        confidence: UnitFloat | Ref(FLOAT_KIND) = Field(
            default=0.0,
            ge=0,
            le=1,
            description=(
                "Minimum SAM2 mask score (its predicted IoU), 0..1. V1 called "
                "it threshold."
            ),
            examples=[0.0, 0.5, "$inputs.sam_confidence"],
        )
        multimask_output: StrictBool | Ref(BOOLEAN_KIND) = Field(
            default=True,
            description=(
                "Let SAM2 propose three masks per prompt and keep the best one; "
                "recommended for ambiguous prompts. False asks for one mask."
            ),
            examples=[True, "$inputs.multimask_output"],
        )

    def __init__(self, *, models: Union[Mapping[Any, Any], ModelCache]):
        self._models = models_cache(models, block=self.type)
        self._embeddings: Dict[Tuple[ModelRequest, str], "_Embedding"] = {}

    @classmethod
    def discover_dependent_resources(
        cls, params: BlockParams
    ) -> List[DependentResource]:
        """Report the SAM2 model this step needs from the ``models`` resource."""
        resources = model_dependency(params, task_type=PREDICTION_TYPE)

        return resources

    def run(
        self,
        *,
        image: Batch,
        boxes: Optional[Batch] = None,
        model_id: str = SAM2_DEFAULT_MODEL_ID,
        model_package_id: Optional[str] = None,
        confidence: float = 0.0,
        multimask_output: bool = True,
    ) -> List[Dict[str, Any]]:
        """Segment every image of the batch.

        Inside an engine call, an output no reader wants is left out.

        Returns:
            One ``{"predictions", "segmentation", "model_id", "inference_id"}``
            per image, in batch order.

        Raises:
            ValueError: On a non-RGB image, an invalid option, prompts of another
                image or a model result that does not match its prompts.
        """
        confidence = _unit_float(confidence, name="confidence", block=self.type)
        if not isinstance(multimask_output, bool):
            raise ValueError(
                f"{self.type}: multimask_output must be true or false, got "
                f"{multimask_output!r}"
            )
        members = image.content
        prompts = [None] * len(members) if boxes is None else list(boxes.content)
        if len(prompts) != len(members):
            raise ValueError(
                f"{self.type}: boxes has {len(prompts)} values for {len(members)} "
                "images"
            )
        prompts = [_prompt_boxes(prompt, block=self.type) for prompt in prompts]
        for member, prompt in zip(members, prompts):
            _require_rgb(member, block=self.type)
            _require_same_frame(prompt, member, block=self.type)

        request = ModelRequest(model_id, model_package_id)
        model = self._models.get(request)
        needed = [prompt is None or len(prompt) > 0 for prompt in prompts]
        embeddings = self._embed(model, request=request, images=members, needed=needed)

        results = []
        for member, prompt, embedding in zip(members, prompts, embeddings):
            inference_id = uuid.uuid4().hex
            if embedding is None:
                predictions = _no_instances(
                    member, prompt=prompt, namespace=inference_id
                )
            else:
                predictions = _sam2_instances(
                    model,
                    embedding=embedding,
                    image=member,
                    prompt=prompt,
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


class SegmentAnything3Block(Block):
    """Segment every instance of each named class with SAM3 text prompts.

    Each class name is one text prompt; its masks get the prompt's index as
    class id. Without class names SAM3 runs one empty prompt named
    "foreground". A mask is kept when its score reaches the class's threshold
    (``per_class_confidence``, else ``confidence``). With ``apply_nms`` a mask
    overlapping a better-scoring mask of any class by more than
    ``nms_iou_threshold`` is dropped. ``class_mapping`` renames classes in
    the result. SAM3 runs locally from the ``models`` resource; the hosted
    proxy is deferred (see the module docstring).
    """

    type = "v2/sam3@v3"
    metadata = {
        "name": "SAM 3",
        "port_of": "roboflow_core/sam3@v3",
        "section": "model",
        "search_keywords": ["SAM3", "META", "text prompt"],
    }
    outputs = SEGMENTATION_OUTPUTS

    class Params(BlockParams):
        # noqa F821: flake8 parses the string "always" in an annotation as a name.
        image: Ref(IMAGE_KIND, batch="always") = Field(  # noqa: F821
            description="RGB images to segment; delivered as one batch."
        )
        class_names: (
            Optional[List[StrictStr]]
            | StrictStr
            | Ref(LIST_OF_VALUES_KIND, STRING_KIND, allow_none=True)
        ) = Field(
            default=None,
            description=(
                "Classes to segment, one text prompt each: a list or a "
                "comma-separated string. Omit to run one empty prompt named "
                "'foreground'."
            ),
            examples=[["car", "person"], "car, person", "$inputs.classes"],
        )
        class_mapping: Optional[Dict[StrictStr, StrictStr]] | Ref(
            DICTIONARY_KIND, allow_none=True
        ) = Field(
            default=None,
            description=(
                "Rename classes in the result, e.g. {'cat': 'gato'}. Prompts "
                "still use the original names."
            ),
            examples=[{"cat": "gato"}],
        )
        confidence: UnitFloat | Ref(FLOAT_KIND) = Field(
            default=0.5,
            ge=0,
            le=1,
            description="Minimum mask score of classes without their own threshold.",
            examples=[0.3, "$inputs.confidence"],
        )
        per_class_confidence: Optional[List[UnitFloat]] | Ref(
            LIST_OF_VALUES_KIND, allow_none=True
        ) = Field(
            default=None,
            description=(
                "Minimum mask score per class, in class_names order; same length "
                "as class_names (1 without class names)."
            ),
            examples=[[0.3, 0.5]],
        )
        apply_nms: StrictBool | Ref(BOOLEAN_KIND) = Field(
            default=True,
            description="Drop masks that overlap a better mask of any class.",
            examples=[True],
        )
        nms_iou_threshold: UnitFloat | Ref(FLOAT_KIND) = Field(
            default=0.9,
            ge=0,
            le=1,
            description="Mask IoU above which apply_nms drops the weaker mask, 0..1.",
            examples=[0.5, 0.9],
        )
        model_id: StrictStr | Ref(STRING_KIND) = Field(
            default=SAM3_DEFAULT_MODEL_ID,
            description=(
                "SAM3 model id in the 'models' resource; change it only for a "
                "fine-tuned SAM3."
            ),
            examples=[SAM3_DEFAULT_MODEL_ID, "$inputs.sam3_model_id"],
        )
        model_package_id: Optional[StrictStr] | Ref(STRING_KIND, allow_none=True) = (
            Field(default=None, description=PACKAGE_DESCRIPTION, examples=[None])
        )

        @model_validator(mode="after")
        def _check_literals(self) -> "SegmentAnything3Block.Params":
            # The engine reruns this validator after resolving selectors.
            names = self.class_names
            if isinstance(names, str) and re.fullmatch(DATA_SELECTOR_PATTERN, names):
                return self

            if isinstance(self.per_class_confidence, list):
                block = SegmentAnything3Block.type
                _check_thresholds(
                    self.per_class_confidence,
                    _prompt_names(names, block=block),
                    block=block,
                )

            return self

    def __init__(self, *, models: Union[Mapping[Any, Any], ModelCache]):
        self._models = models_cache(models, block=self.type)

    @classmethod
    def discover_dependent_resources(
        cls, params: BlockParams
    ) -> List[DependentResource]:
        """Report the SAM3 model this step needs from the ``models`` resource."""
        resources = model_dependency(params, task_type=PREDICTION_TYPE)

        return resources

    def run(
        self,
        *,
        image: Batch,
        class_names: Union[None, str, Sequence[str]] = None,
        class_mapping: Optional[Mapping[str, str]] = None,
        confidence: float = 0.5,
        per_class_confidence: Optional[Sequence[float]] = None,
        apply_nms: bool = True,
        nms_iou_threshold: float = 0.9,
        model_id: str = SAM3_DEFAULT_MODEL_ID,
        model_package_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Segment every image of the batch.

        Inside an engine call, an output no reader wants is left out.

        Returns:
            One ``{"predictions", "segmentation", "model_id", "inference_id"}``
            per image, in batch order.

        Raises:
            ValueError: On a non-RGB image, an invalid option or a model
                result of the wrong shape.
        """
        names = _prompt_names(class_names, block=self.type)
        confidence = _unit_float(confidence, name="confidence", block=self.type)
        thresholds = _thresholds(
            names,
            confidence=confidence,
            per_class_confidence=per_class_confidence,
            block=self.type,
        )
        renamed = _renamed(names, class_mapping=class_mapping, block=self.type)
        nms_iou_threshold = _unit_float(
            nms_iou_threshold, name="nms_iou_threshold", block=self.type
        )
        if not isinstance(apply_nms, bool):
            raise ValueError(
                f"{self.type}: apply_nms must be true or false, got {apply_nms!r}"
            )
        members = image.content
        for member in members:
            _require_rgb(member, block=self.type)

        per_image = self._rows(
            members,
            names=names,
            thresholds=thresholds,
            apply_nms=apply_nms,
            nms_iou_threshold=nms_iou_threshold,
            request=ModelRequest(model_id, model_package_id),
        )

        results = []
        for member, rows in zip(members, per_image):
            inference_id = uuid.uuid4().hex
            predictions = _text_prompt_instances(
                rows, image=member, class_names=renamed, namespace=inference_id
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

    def _rows(
        self,
        images: Sequence[ImageData],
        *,
        names: List[Optional[str]],
        thresholds: List[float],
        apply_nms: bool,
        nms_iou_threshold: float,
        request: ModelRequest,
    ) -> List[List["_Row"]]:
        """Thresholded, non-empty, optionally NMS-filtered rows of each image."""
        model = self._models.get(request)
        # The model applies one floor; each class's own threshold follows.
        outputs = model.segment_with_text_prompts(
            [member.tensor_image for member in images],
            prompts=[{"text": name} for name in names],
            output_prob_thresh=min(thresholds),
            mask_format="dense",
        )
        if len(outputs) != len(images):
            raise ValueError(
                f"{self.type}: model {str(request)!r} returned {len(outputs)} "
                f"results for {len(images)} images"
            )

        per_image = []
        for member, prompt_results in zip(images, outputs):
            rows = _prompt_rows(
                prompt_results,
                prompt_count=len(names),
                size_hw=member.size_hw,
                block=self.type,
            )
            rows = [row for row in rows if row.score >= thresholds[row.prompt_index]]
            if apply_nms:
                rows = _mask_nms(rows, iou_threshold=nms_iou_threshold)
            per_image.append(rows)

        return per_image


class _Embedding:
    """One image's SAM2 embedding and a weak reference to its pixel tensor."""

    def __init__(self, pixels: torch.Tensor, embeddings: Any):
        self._pixels = weakref.ref(pixels)
        self.embeddings = embeddings

    def holds(self, pixels: torch.Tensor) -> bool:
        # A new tensor under a reused image_id (e.g. with_pixels) is a miss.
        held = self._pixels() is pixels

        return held


def _segmentation_result(
    predictions: InstanceDetections,
    *,
    reader: Block,
    model_id: str,
    inference_id: str,
) -> Dict[str, Any]:
    # Keep the reader keyword: the frozen interactive block imports this helper.
    result: Dict[str, Any] = {"model_id": model_id, "inference_id": inference_id}
    if reader.wants("segmentation"):
        result["segmentation"] = SegmentationView.from_dense(predictions)
    if reader.wants("predictions"):
        result["predictions"] = predictions

    return result


def _sam2_instances(
    model: Any,
    *,
    embedding: Any,
    image: ImageData,
    prompt: Optional[Detections],
    confidence: float,
    multimask_output: bool,
    namespace: str,
    block: str,
) -> InstanceDetections:
    boxes = None if prompt is None else [prompt.xyxy]
    (prediction,) = model.segment_images(
        embeddings=[embedding],
        boxes=boxes,
        multi_mask_output=multimask_output,
        return_logits=True,
    )
    logits, scores = _most_confident(prediction.masks, prediction.scores)
    expected = 1 if prompt is None else len(prompt)
    if logits.shape[0] != expected or tuple(logits.shape[1:]) != image.size_hw:
        raise ValueError(
            f"{block}: SAM2 returned masks of shape {tuple(logits.shape)} for "
            f"{expected} prompts on a {image.size_hw} image"
        )

    device = image.device
    binary = (logits >= MASK_LOGIT_THRESHOLD).to(device)
    scores = scores.to(device=device, dtype=torch.float32)
    selection = RowSelection.of(expected, mask=scores >= confidence)
    mask = selection.take(binary)
    if prompt is None:
        predictions = _instances(
            mask,
            class_id=torch.zeros(len(selection), dtype=torch.int64, device=device),
            scores=selection.take(scores),
            metadata=_metadata(image, class_names={0: FOREGROUND}),
        )
        identified = with_row_identity(predictions, namespace=namespace)
        return identified

    kept = take_rows(prompt, selection)
    record = (kept.image_metadata or {}).get(ROW_RECORD_KEY)
    predictions = InstanceDetections(
        xyxy=_mask_boxes(mask),
        class_id=kept.class_id.to(device=device, dtype=torch.int64),
        confidence=selection.take(scores),
        mask=mask,
        image_metadata=_metadata(
            image, class_names=class_names_by_id(prompt.image_metadata)
        ),
        bboxes_metadata=(
            None
            if kept.bboxes_metadata is None
            else [without_host_mirrors(row) for row in kept.bboxes_metadata]
        ),
    )
    identified = _with_record(predictions, record=record, namespace=namespace)

    return identified


def _no_instances(
    image: ImageData, *, prompt: Optional[Detections], namespace: str
) -> InstanceDetections:
    """The empty result of an image whose prompt has no rows."""
    names = None if prompt is None else class_names_by_id(prompt.image_metadata)
    # Shared selection carries the record even for an already empty prompt.
    kept = (
        None
        if prompt is None
        else take_rows(prompt, RowSelection.of(len(prompt), indices=[]))
    )
    record = None if kept is None else (kept.image_metadata or {}).get(ROW_RECORD_KEY)
    predictions = _instances(
        torch.zeros((0, *image.size_hw), dtype=torch.bool, device=image.device),
        scores=torch.zeros(0, dtype=torch.float32, device=image.device),
        class_id=torch.zeros(0, dtype=torch.int64, device=image.device),
        metadata=_metadata(image, class_names=names),
    )
    identified = _with_record(predictions, record=record, namespace=namespace)

    return identified


def _text_prompt_instances(
    rows: Sequence[_Row],
    *,
    image: ImageData,
    class_names: Dict[int, str],
    namespace: str,
) -> InstanceDetections:
    device = image.device
    if rows:
        mask = torch.stack([row.mask for row in rows]).to(device)
    else:
        mask = torch.zeros((0, *image.size_hw), dtype=torch.bool, device=device)
    predictions = _instances(
        mask,
        scores=torch.tensor(
            [row.score for row in rows], dtype=torch.float32, device=device
        ),
        class_id=torch.tensor(
            [row.prompt_index for row in rows], dtype=torch.int64, device=device
        ),
        metadata=_metadata(image, class_names=class_names),
    )
    identified = with_row_identity(predictions, namespace=namespace)

    return identified


def _instances(
    mask: torch.Tensor,
    *,
    scores: torch.Tensor,
    class_id: torch.Tensor,
    metadata: Dict[str, Any],
) -> InstanceDetections:
    predictions = InstanceDetections(
        xyxy=_mask_boxes(mask),
        class_id=class_id,
        confidence=scores,
        mask=mask,
        image_metadata=metadata,
    )

    return predictions


def _with_record(
    predictions: InstanceDetections, *, record: Optional[dict], namespace: str
) -> InstanceDetections:
    """Carry the prompt rows' identity, or start a new one without it."""
    if record is None:
        identified = with_row_identity(predictions, namespace=namespace)
        return identified

    predictions.image_metadata[ROW_RECORD_KEY] = record

    return predictions


def _metadata(
    image: ImageData, *, class_names: Optional[Mapping[int, Any]]
) -> Dict[str, Any]:
    metadata = {
        **FrameRef.from_image(image).prediction_metadata(),
        CLASS_NAMES_KEY: dict(class_names or {}),
        "prediction_type": PREDICTION_TYPE,
    }

    return metadata


def _most_confident(
    masks: torch.Tensor, scores: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The best-scoring proposal per prompt: ``(P, H, W)`` masks, ``(P,)`` scores.

    SAM2 drops the prompt axis for a single prompt, so ``(K, H, W)`` is one
    prompt's ``K`` proposals.
    """
    if masks.ndim == 3:
        masks = masks.unsqueeze(0)
        scores = scores.unsqueeze(0)
    best = torch.argmax(scores, dim=1)
    rows = torch.arange(masks.shape[0], device=masks.device)
    selected = (masks[rows, best], scores[rows, best])

    return selected


def _mask_boxes(mask: torch.Tensor) -> torch.Tensor:
    """Inclusive ``xyxy`` pixel bounds of each mask; zeros for an empty mask."""
    _, height, width = mask.shape
    columns_used = mask.any(dim=1)
    rows_used = mask.any(dim=2)
    x = torch.arange(width, device=mask.device)
    y = torch.arange(height, device=mask.device)
    boxes = torch.stack(
        [
            torch.where(columns_used, x, width).amin(dim=1),
            torch.where(rows_used, y, height).amin(dim=1),
            torch.where(columns_used, x, -1).amax(dim=1),
            torch.where(rows_used, y, -1).amax(dim=1),
        ],
        dim=1,
    ).to(torch.float32)
    empty = ~columns_used.any(dim=1, keepdim=True)
    boxes = boxes.masked_fill(empty, 0.0)

    return boxes


def _prompt_boxes(prompt: Any, *, block: str) -> Optional[Detections]:
    """The prompt's boxes as ``Detections``; ``None`` when there is no prompt."""
    if prompt is None:
        return None

    if isinstance(prompt, SegmentationView):
        return prompt.detections

    if isinstance(prompt, tuple):
        _, detections = prompt
        if detections is None:
            raise ValueError(
                f"{block}: the keypoint prediction has no boxes; SAM2 needs boxes "
                "as prompts"
            )
        prompt = detections
    boxes = Detections(
        xyxy=prompt.xyxy,
        class_id=prompt.class_id,
        confidence=prompt.confidence,
        image_metadata=prompt.image_metadata,
        bboxes_metadata=prompt.bboxes_metadata,
    )

    return boxes


def _require_same_frame(
    prompt: Optional[Detections], image: ImageData, *, block: str
) -> None:
    record = (
        None if prompt is None else (prompt.image_metadata or {}).get(FRAME_RECORD_KEY)
    )
    if record is None:
        return

    frame = FrameRef.from_record(record)
    if frame.image_id != image.image_id or frame.size_hw != image.size_hw:
        raise ValueError(
            f"{block}: boxes were predicted on image {frame.image_id!r} of size "
            f"{frame.size_hw}, not on image {image.image_id!r} of size "
            f"{image.size_hw}; wire boxes predicted on the image you segment"
        )


def _require_rgb(image: ImageData, *, block: str) -> None:
    if image.channels != 3:
        raise ValueError(f"{block} needs a 3-channel RGB image, got {image!r}")


def _unit_float(value: Any, *, name: str, block: str) -> float:
    # Direct callers bypass engine validation; list elements also use this check.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{block}: {name} must be a number 0..1, got {value!r}")

    if not 0 <= value <= 1:
        raise ValueError(f"{block}: {name} must be a number 0..1, got {value!r}")

    number = float(value)

    return number


def _prompt_names(
    class_names: Union[None, str, Sequence[str]], *, block: str = "v2/sam3@v3"
) -> List[Optional[str]]:
    """One text per prompt; ``[None]`` (one empty prompt) without names."""
    if class_names is None:
        return [None]

    if isinstance(class_names, str):
        names = [name.strip() for name in class_names.split(",")]
    elif isinstance(class_names, (list, tuple)):
        names = list(class_names)
    else:
        raise ValueError(
            f"{block}: class_names must be a list of names or a comma-separated "
            f"string, got {type(class_names).__name__}"
        )
    for name in names:
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"{block}: class_names holds non-empty names, got {name!r}"
            )

    prompt_names: List[Optional[str]] = names or [None]

    return prompt_names


def _thresholds(
    names: Sequence[Optional[str]],
    *,
    confidence: float,
    per_class_confidence: Optional[Sequence[float]],
    block: str,
) -> List[float]:
    if per_class_confidence is None:
        return [confidence] * len(names)

    _check_thresholds(per_class_confidence, names, block=block)
    thresholds = [
        _unit_float(value, name="per_class_confidence", block=block)
        for value in per_class_confidence
    ]

    return thresholds


def _check_thresholds(
    per_class_confidence: Any,
    names: Sequence[Optional[str]],
    *,
    block: str = "v2/sam3@v3",
) -> None:
    if not isinstance(per_class_confidence, (list, tuple)):
        raise ValueError(
            f"{block}: per_class_confidence must be a list of numbers, got "
            f"{type(per_class_confidence).__name__}"
        )

    if len(per_class_confidence) != len(names):
        raise ValueError(
            f"{block}: per_class_confidence has {len(per_class_confidence)} "
            f"values for {len(names)} prompts {names}"
        )


def _renamed(
    names: Sequence[Optional[str]],
    *,
    class_mapping: Optional[Mapping[str, str]],
    block: str,
) -> Dict[int, str]:
    if class_mapping is not None and not isinstance(class_mapping, Mapping):
        raise ValueError(
            f"{block}: class_mapping must map class names to new names, got "
            f"{type(class_mapping).__name__}"
        )

    mapping = class_mapping or {}
    renamed = {}
    for index, name in enumerate(names):
        name = name or FOREGROUND
        renamed[index] = mapping.get(name, name)

    return renamed


def _prompt_rows(
    prompt_results: Any, *, prompt_count: int, size_hw: Tuple[int, int], block: str
) -> List[_Row]:
    """Rows of one image's ``segment_with_text_prompts`` result, empty masks dropped."""
    rows = []
    for result in prompt_results:
        index = result.get("prompt_index", 0)
        if not 0 <= index < prompt_count:
            continue
        masks = result.get("masks")
        if masks is None:
            continue
        masks = torch.as_tensor(np.asarray(masks)) != 0
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        scores = [float(score) for score in result.get("scores", [])]
        if masks.ndim != 3 or len(scores) != masks.shape[0]:
            raise ValueError(
                f"{block}: SAM3 returned masks of shape {tuple(masks.shape)} with "
                f"{len(scores)} scores for prompt {index}"
            )
        if masks.shape[0] and tuple(masks.shape[1:]) != size_hw:
            raise ValueError(
                f"{block}: SAM3 returned {tuple(masks.shape[1:])} masks for a "
                f"{size_hw} image"
            )
        for mask, score in zip(masks, scores):
            if mask.any():
                rows.append(_Row(prompt_index=index, mask=mask, score=score))

    return rows


def _mask_nms(rows: List[_Row], *, iou_threshold: float) -> List[_Row]:
    """Greedy cross-class mask NMS, V1's rule; survivors keep their order.

    Masks are visited by descending score (ties in row order). Each kept mask
    drops every later mask with IoU above ``iou_threshold``.
    """
    if len(rows) < 2:
        return rows

    encoded = [
        mask_utils.encode(np.asfortranarray(row.mask.numpy().astype(np.uint8)))
        for row in rows
    ]
    scores = np.array([row.score for row in rows])
    order = np.argsort(-scores, kind="stable")
    ordered = [encoded[index] for index in order]
    ious = mask_utils.iou(ordered, ordered, [0] * len(rows))
    keep = np.ones(len(rows), dtype=bool)
    for position in range(len(rows)):
        if keep[position]:
            later = ious[position, position + 1 :] > iou_threshold
            keep[position + 1 :] &= ~later
    kept_rows = sorted(order[keep].tolist())
    survivors = [rows[index] for index in kept_rows]

    return survivors

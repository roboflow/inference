"""Segment every instance of each named class with SAM3 text prompts.

Canonical type: ``roboflow_core/sam3@v3`` (local only)::

    one text prompt per class name; no names = one empty prompt "foreground"
      segment_with_text_prompts(mask_format="dense")
      per class: score >= its threshold; drop empty masks
      apply_nms: greedy mask IoU across classes, IoU > threshold drops
      class_id = prompt index, class name = class_mapping applied to it

Deliberate differences from V1 (``segment_anything3/v3_tensor.py``):

- Models come from the ``models`` resource by ``model_id``. ``threshold`` is
  now ``confidence``.
- Empty masks are dropped (V1 v3 raised on them) and the
  ``per_class_confidence`` length is validated at run time too.
  Comma-separated class names are stripped of surrounding spaces. The model's
  score floor is the lowest class threshold; V1 also lowered it to
  ``confidence``. Each class's own threshold follows either way. NMS visits
  equal scores in row order; V1 visited them in reverse row order.
- ``output_format`` is gone: outputs are a view and dense predictions, not
  RLE or polygons.
- The hosted proxy (``SAM3_EXEC_MODE``) is deferred, not ported; see
  ``_shared/segment_anything.py``.
"""

import re
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

import numpy as np
import torch
from inference_models.models.base.instance_segmentation import InstanceDetections
from pycocotools import mask as mask_utils
from pydantic import Field, StrictBool, StrictFloat, StrictStr, model_validator
from roboflow_workflows.v2.blocks.models._shared.segment_anything import (
    FOREGROUND,
    PREDICTION_TYPE,
    _instances,
    _metadata,
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
    DICTIONARY_KIND,
    FLOAT_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
)
from roboflow_workflows.v2.data.masks.grid import INSTANCE_SEGMENTATION_VIEW_KIND
from roboflow_workflows.v2.data.predictions import INSTANCE_SEGMENTATION_PREDICTION_KIND
from roboflow_workflows.v2.data.row_identity import with_row_identity
from roboflow_workflows.v2.engine.declaration import (
    DATA_SELECTOR_PATTERN,
    Block,
    BlockParams,
    DependentResource,
    Output,
    Ref,
)


SAM3_DEFAULT_MODEL_ID = "sam3/sam3_final"


class _Row(NamedTuple):
    """One SAM3 candidate: prompt index, host bool mask, and score."""

    prompt_index: int
    mask: torch.Tensor
    score: float


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

    type = "roboflow_core/sam3@v3"
    metadata = {
        "name": "SAM 3",
        "port_of": "roboflow_core/sam3@v3",
        "section": "model",
        "search_keywords": ["SAM3", "META", "text prompt"],
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
        confidence: Annotated[StrictFloat, Field(ge=0, le=1)] | Ref(FLOAT_KIND) = Field(
            default=0.5,
            ge=0,
            le=1,
            description="Minimum mask score of classes without their own threshold.",
            examples=[0.3, "$inputs.confidence"],
        )
        per_class_confidence: Optional[
            List[Annotated[StrictFloat, Field(ge=0, le=1)]]
        ] | Ref(LIST_OF_VALUES_KIND, allow_none=True) = Field(
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
        nms_iou_threshold: Annotated[StrictFloat, Field(ge=0, le=1)] | Ref(
            FLOAT_KIND
        ) = Field(
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
            require_rgb(member, block=self.type)

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
    ) -> List[List[_Row]]:
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


def _prompt_names(
    class_names: Union[None, str, Sequence[str]], *, block: str = "roboflow_core/sam3@v3"
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
    block: str = "roboflow_core/sam3@v3",
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

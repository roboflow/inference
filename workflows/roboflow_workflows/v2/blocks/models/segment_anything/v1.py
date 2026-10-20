"""Segment the object inside each prompt box, or the whole image, with SAM2.

Canonical type: ``roboflow_core/segment_anything@v1``::

    boxes not wired   one mask per image: SAM2's best guess, "foreground"
    boxes wired       one mask per prompt row; rows keep the prompt's class,
                      row dict and row identity; no rows -> no masks
      embed each image (reused while the same pixels come back)
      segment_images(embeddings, boxes)   -> best of SAM2's proposals per row
      keep rows with score >= confidence

The image embedding is kept by the step, never by ``ModelCache``: a later
call with the same pixel tensor (same ``image_id`` and tensor object) and the
same model request reuses it. Only the latest call's embeddings are kept;
``reset_state()`` and a processing reset drop them. The shared model is never
unloaded.

Deliberate differences from V1 (``segment_anything2/v1_tensor.py``):

- Models come from the ``models`` resource by ``model_id``; V1's ``version``
  becomes ``model_id="sam2/<version>"``. ``threshold`` is now ``confidence``.
- With wired boxes but no box rows the block returns no masks; V1 segmented
  the whole image as "foreground".
- Output rows keep the prompt rows' metadata dicts and row identity; V1
  copied only ``detection_id`` and ``class``.
- ``output_format`` is gone: outputs are a view and dense predictions, not
  RLE or polygons.
"""

import uuid
from typing import Annotated, Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections
from pydantic import Field, StrictBool, StrictFloat, StrictStr
from roboflow_workflows.v2.blocks.geometry._rows import without_host_mirrors
from roboflow_workflows.v2.blocks.models._shared.segment_anything import (
    FOREGROUND,
    MASK_LOGIT_THRESHOLD,
    PREDICTION_TYPE,
    _Embedding,
    _instances,
    _mask_boxes,
    _metadata,
    _no_instances,
    _prompt_boxes,
    _require_same_frame,
    _segmentation_result,
    _unit_float,
    _with_record,
)
from roboflow_workflows.v2.blocks.models.cache import ModelCache, ModelRequest
from roboflow_workflows.v2.blocks.models.common import (
    models_cache,
    require_rgb,
)
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import BOOLEAN_KIND, FLOAT_KIND, STRING_KIND
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


SAM2_DEFAULT_MODEL_ID = "sam2/hiera_tiny"


class SegmentAnything2Block(Block):
    """Segment the object inside each prompt box, or the whole image, with SAM2.

    Without ``boxes`` each image gets one mask, SAM2's most confident guess for
    the whole image, with class id 0 and class name "foreground". With
    ``boxes`` each prompt row gets the most confident of SAM2's proposals for
    its box; rows scoring below ``confidence`` are dropped. Kept rows keep the
    prompt's class id, class names, row dict and row identity, so a mask row
    and its detector row stay the same row.
    """

    type = "roboflow_core/segment_anything@v1"
    metadata = {
        "name": "Segment Anything 2 Model",
        "port_of": "roboflow_core/segment_anything@v1",
        "section": "model",
        "search_keywords": ["SAM2", "META"],
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
        confidence: Annotated[StrictFloat, Field(ge=0, le=1)] | Ref(FLOAT_KIND) = Field(
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
        self._embeddings: Dict[Tuple[ModelRequest, str], _Embedding] = {}

    @classmethod
    def discover_dependent_resources(
        cls, params: BlockParams
    ) -> List[DependentResource]:
        """Report the SAM2 model this step needs from the ``models`` resource."""
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
            require_rgb(member, block=self.type)
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
        call's embeddings. Unhit embeddings are released before that call, so
        if it fails only this call's hits stay kept.
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
        # Release unhit embeddings before the model allocates new ones; the
        # loop's ``entry`` may still hold a stale one.
        self._embeddings = kept
        previous = entry = None

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

        return found


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

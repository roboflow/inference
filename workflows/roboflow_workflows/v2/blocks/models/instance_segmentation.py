"""Batched instance segmentation with host-owned ``inference_models`` models.

Model selection, grouping, phases and NMS options follow the object-detection
block (see ``common``). Each image's result is built around one
``SegmentationView``; both mask outputs describe its rows::

    _segmentations(group)   one SegmentationView per image
        class filter (view.select), frame metadata, row identity
        segmentation  the view                 when a reader wants it
        predictions   view.full_res()          when a reader wants it

``segmentation`` is what blocks should wire: box and class consumers read
its boxes without any full-resolution mask, and mask consumers ask its mask
provider for the fidelity they need. ``predictions`` is the serializable,
V1-shaped ``InstanceDetections``; building it may cost one full-image mask
per instance, so the block builds it only when a reader wants it. Both share
one row record, so selections keep them aligned.

The contract does not require dense model output. This reference
implementation asks ``post_process`` for dense masks
(``mask_format="dense"``) and wraps them as a dense-fallback view: the masks
already exist, so leaving out ``predictions`` saves nothing here. A producer
with low-resolution mask grids overrides ``_segmentations`` and then computes
full-resolution masks only for ``predictions`` or an explicit consumer
request. A model whose ``post_process`` cannot return dense masks fails the
step.
"""

import uuid
from dataclasses import replace
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Union

import torch
from pydantic import Field
from roboflow_workflows.v2.blocks.models.cache import ModelCache
from roboflow_workflows.v2.blocks.models.common import (
    CONFIDENCE_MODES_DESCRIPTION,
    INFERENCE_ID_OUTPUT,
    MODEL_ID_OUTPUT,
    ClassFilterField,
    ModelGroup,
    ModelOption,
    NmsParams,
    RequestField,
    UnitFloat,
    checked_confidence,
    checked_count,
    class_filter_tensors,
    forward_groups,
    model_dependency,
    models_cache,
    nms_options,
    prediction_metadata,
    preprocess_groups,
)
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import FLOAT_KIND, STRING_KIND
from roboflow_workflows.v2.data.masks.grid import (
    INSTANCE_SEGMENTATION_VIEW_KIND,
    SegmentationView,
)
from roboflow_workflows.v2.data.prediction_ops import RowSelection
from roboflow_workflows.v2.data.predictions import INSTANCE_SEGMENTATION_PREDICTION_KIND
from roboflow_workflows.v2.data.row_identity import with_row_identity
from roboflow_workflows.v2.engine.declaration import (
    Block,
    BlockParams,
    DependentResource,
    Output,
    Ref,
)
from roboflow_workflows.v2.engine.phases import phase

PREDICTION_TYPE = "instance-segmentation"
CONFIDENCE_MODES = ("best", "default")
MASK_FORMAT = "dense"


class InstanceSegmentationBlock(Block):
    """Segment object instances in a batch of images with host-owned models.

    ``segmentation`` is a ``SegmentationView`` of each image's instances,
    carrying fresh row identity; ``predictions`` is its ``full_res()``: an
    ``InstanceDetections`` with a dense bool ``(N, H, W)`` mask in the image's
    pixel coordinates and the same row record. Each output is built only
    when a reader wants it. ``image_metadata`` keeps what the model wrote and
    adds the image's frame record, ``class_names`` and ``prediction_type``.
    """

    type = "v2/instance_segmentation@v1"
    outputs = {
        "predictions": Output(
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
            source="image",
            description=(
                "Serializable InstanceDetections with a dense (N, H, W) bool "
                "mask in this image's pixel coordinates: segmentation's "
                "full-resolution result, same rows. May cost one full-image "
                "mask per instance; use it as a workflow output, wire "
                "segmentation to blocks."
            ),
        ),
        "segmentation": Output(
            INSTANCE_SEGMENTATION_VIEW_KIND,
            source="image",
            description=(
                "SegmentationView of the instances: wire it to blocks. Box and "
                "class consumers need no full-resolution mask; mask consumers "
                "request the fidelity they need. Not serializable; use "
                "predictions as a workflow output."
            ),
        ),
        "model_id": MODEL_ID_OUTPUT,
        "inference_id": INFERENCE_ID_OUTPUT,
    }

    class Params(NmsParams):
        # noqa F821: flake8 parses the string "always" in an annotation as a name.
        image: Ref(IMAGE_KIND, batch="always") = Field(  # noqa: F821
            description="RGB images to segment; delivered as one batch."
        )
        confidence: (
            UnitFloat | Literal["best", "default"] | Ref(FLOAT_KIND, STRING_KIND)
        ) = Field(
            default="best",
            description=f"Minimum instance confidence. {CONFIDENCE_MODES_DESCRIPTION}",
            examples=["best", 0.4, "$inputs.confidence"],
        )

    def __init__(self, *, models: Union[Mapping[Any, Any], ModelCache]):
        self._models = models_cache(models, block=self.type)

    @classmethod
    def discover_dependent_resources(
        cls, params: BlockParams
    ) -> List[DependentResource]:
        """Report the model this step needs from the ``models`` resource."""
        resources = model_dependency(params, task_type=PREDICTION_TYPE)

        return resources

    @phase
    def network_input(
        self,
        image: Batch,
        model_id: RequestField,
        model_package_id: RequestField,
        confidence: Union[float, str],
        iou_threshold: Optional[float],
        max_detections: Optional[int],
        class_agnostic_nms: Optional[bool],
        class_filter: ClassFilterField,
    ) -> Tuple[ModelGroup, ...]:
        """Check options, resolve each row's model, preprocess each group."""
        confidence = checked_confidence(
            confidence, modes=CONFIDENCE_MODES, block=self.type
        )
        options = nms_options(
            confidence=confidence,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
            class_agnostic_nms=class_agnostic_nms,
        ) + (ModelOption("mask_format", "mask_format", None, MASK_FORMAT),)
        groups = preprocess_groups(
            image,
            cache=self._models,
            model_id=model_id,
            model_package_id=model_package_id,
            options=options,
            block=self.type,
            class_filter=class_filter,
        )

        return groups

    @phase
    def raw_output(self, network_input: Tuple[ModelGroup, ...]) -> Tuple[Any, ...]:
        """Run each group's network once on its rows (``forward``)."""
        raw_output = forward_groups(network_input)

        return raw_output

    @phase
    def result(
        self,
        raw_output: Tuple[Any, ...],
        network_input: Tuple[ModelGroup, ...],
        image: Batch,
    ) -> List[Dict[str, Any]]:
        """Decode instances; one result per image, in batch order.

        Asks ``wants`` for both mask outputs and leaves out the ones nobody
        reads.
        """
        members = image.content
        wants_segmentation = self.wants("segmentation")
        wants_predictions = self.wants("predictions")
        results: List[Any] = [None] * len(members)
        for group, group_output in zip(network_input, raw_output):
            views = self._segmentations(group, group_output)
            checked_count(views, group, block=self.type)

            kept = class_filter_tensors(group, like=views[0].detections.class_id)
            for view, row, kept_ids in zip(views, group.rows, kept):
                view = _keep_view_classes(view, kept_ids=kept_ids)
                image_metadata = prediction_metadata(
                    view.detections.image_metadata,
                    members[row],
                    class_names=group.model.class_names,
                    prediction_type=PREDICTION_TYPE,
                )
                inference_id = uuid.uuid4().hex
                view = _identified_view(
                    view, image_metadata=image_metadata, namespace=inference_id
                )
                result = {
                    "model_id": group.request.model_id,
                    "inference_id": inference_id,
                }
                if wants_segmentation:
                    result["segmentation"] = view
                if wants_predictions:
                    result["predictions"] = view.full_res()
                results[row] = result

        return results

    def _segmentations(
        self, group: ModelGroup, group_output: Any
    ) -> List[SegmentationView]:
        """One view per image of the group, before filtering and identity.

        The reference decodes dense masks with ``post_process`` and wraps
        each image's ``InstanceDetections`` as a dense-fallback view. A
        producer with low-resolution grids overrides this and returns grid
        views (``SegmentationView.from_selected_rows``).
        """
        batch = group.model.post_process(group_output, group.metadata, **group.options)
        views = []
        for predictions in batch:
            _require_dense(predictions, group=group, block=self.type)
            views.append(SegmentationView.from_dense(predictions))

        return views

    def run(
        self,
        *,
        image: Batch,
        model_id: RequestField,
        confidence: Union[float, str] = "best",
        model_package_id: RequestField = None,
        class_filter: ClassFilterField = None,
        iou_threshold: Optional[float] = None,
        max_detections: Optional[int] = None,
        class_agnostic_nms: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """Segment every image of the batch with the same three phases.

        Inside an engine call, an output no reader wants is left out.

        Returns:
            One ``{"predictions", "segmentation", "model_id", "inference_id"}``
            per image, in batch order.
        """
        network_input = self.network_input(
            image,
            model_id,
            model_package_id,
            confidence,
            iou_threshold,
            max_detections,
            class_agnostic_nms,
            class_filter,
        )
        raw_output = self.raw_output(network_input)
        results = self.result(raw_output, network_input, image)

        return results


def _require_dense(predictions: Any, *, group: ModelGroup, block: str) -> None:
    mask = predictions.mask
    if not isinstance(mask, torch.Tensor) or mask.ndim != 3:
        raise ValueError(
            f"{block}: model {str(group.request)!r} returned "
            f"{type(mask).__name__} masks; this block needs dense (N, H, W) "
            f"masks (post_process mask_format={MASK_FORMAT!r})"
        )


def _keep_view_classes(
    view: SegmentationView, *, kept_ids: Optional[torch.Tensor]
) -> SegmentationView:
    if kept_ids is None:
        return view

    mask = torch.isin(view.detections.class_id, kept_ids)
    selected = view.select(RowSelection.of(len(view), mask=mask))

    return selected


def _identified_view(
    view: SegmentationView, *, image_metadata: Dict[str, Any], namespace: str
) -> SegmentationView:
    """The same rows with the final ``image_metadata`` and a fresh row identity.

    The row namespace is the image's ``inference_id``. Masks are not copied
    and nothing is upscaled.
    """
    if view.is_dense_fallback:
        # full_res() of a dense fallback is its wrapped predictions: no cost.
        predictions = replace(view.full_res(), image_metadata=image_metadata)
        predictions = with_row_identity(predictions, namespace=namespace)
        dense_view = SegmentationView.from_dense(predictions)
        return dense_view

    # Grid views come only from a ``_segmentations`` override.
    detections = replace(view.detections, image_metadata=image_metadata)
    grid_view = SegmentationView(
        detections=with_row_identity(detections, namespace=namespace),
        scores=view.scores,
        score_type=view.score_type,
        threshold=view.threshold,
        geometry=view.geometry,
        take_ownership=True,
    )

    return grid_view

"""Batched object detection with host-owned ``inference_models`` models.

One call per workflow run, one model call per phase and distinct request::

    image (Batch of N, CHW RGB) ── network_input: group rows by ModelRequest,
                                                  model.pre_process(group images)
                                   raw_output:    model.forward(group batch)
                                   result:        model.post_process -> Detections,
                                                  class filter, row identity,
                                                  scattered back to row order

A step selects its model with ``model_id`` and optionally ``model_package_id``,
both literal or selected at run time. A constant request makes one group of
all N rows. A per-image request (a batch input) makes one group per distinct
request, so no row ever uses another row's model. A per-image package of
``None`` asks the loader to choose for that image, as a literal ``None`` does
for all of them. A per-image model id of ``None``, or a filtered model id or
package, skips that image: it gets no prediction.

The host passes the ``models`` resource when creating the session, either:

- a mapping of model id (or ``ModelRequest``) to a caller-loaded model,
  e.g. ``plan.create_session({"models": {"rfdetr-nano": model}})``. Nothing is
  loaded; a missing model fails the step;
- a ``ModelCache``, which serves its supplied models and loads the others
  once through its loader, e.g. ``{"models": ModelCache()}`` for
  ``AutoModel``.

A model supplied under a bare model id has no verified package: a request
naming a package fails rather than silently using it. The block copies a
mapping, not its models, when it is constructed; a later change of the host's
dictionary does not change a running block. Models are resolved in
``network_input`` and the later phases use those same objects. Compilation
never loads a model, and nothing here closes one: the host owns their lifetime.

Options: ``confidence`` (a number or the model's ``"best"`` / ``"default"``
mode, ``"best"`` by default), ``iou_threshold``, ``max_detections`` and
``class_agnostic_nms`` reach a model only when its ``post_process`` reads them
(see ``common``). The block applies ``class_filter`` itself, after
``post_process``, per image.

Concurrency: pipelined phases of different runs may overlap
(``phase_overlap``) and sessions may share a cache. Each model must support
the calls it receives and complete its outputs before returning them, as the
default ``inference_models`` backends do; the block adds no lock or copy.
"""

from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Union

from pydantic import Field
from roboflow_workflows.v2.blocks.models.cache import ModelCache
from roboflow_workflows.v2.blocks.models.common import (
    CONFIDENCE_MODES_DESCRIPTION,
    INFERENCE_ID_OUTPUT,
    MODEL_ID_OUTPUT,
    ClassFilterField,
    ModelGroup,
    NmsParams,
    RequestField,
    UnitFloat,
    checked_confidence,
    checked_count,
    class_filter_tensors,
    forward_groups,
    identified_result,
    keep_classes,
    model_dependency,
    models_cache,
    nms_options,
    prediction_metadata,
    preprocess_groups,
)
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import FLOAT_KIND, STRING_KIND
from roboflow_workflows.v2.data.predictions import OBJECT_DETECTION_PREDICTION_KIND
from roboflow_workflows.v2.engine.declaration import (
    Block,
    BlockParams,
    DependentResource,
    Output,
    Ref,
)
from roboflow_workflows.v2.engine.phases import phase

PREDICTION_TYPE = "object-detection"
DEFAULT_CONFIDENCE = "best"
CONFIDENCE_MODES = ("best", "default")


class ObjectDetectionBlock(Block):
    """Detect objects in a batch of images with host-owned models.

    The phases are the model's own ``pre_process``, ``forward`` and
    ``post_process``, each called once per distinct model request in the
    batch; ``run`` composes the same three. ``predictions`` are the model's
    native ``Detections`` in each image's pixel coordinates, carrying fresh
    row identity. Their ``image_metadata`` keeps what the model wrote and adds
    the image's frame record (``FrameRef.prediction_metadata()``: source,
    video time and workflow-root geometry), the model's ``class_names`` and
    ``prediction_type``.
    """

    type = "v2/object_detection@v1"
    outputs = {
        "predictions": Output(
            OBJECT_DETECTION_PREDICTION_KIND,
            source="image",
            description="Native Detections in this image's pixel coordinates.",
        ),
        "model_id": MODEL_ID_OUTPUT,
        "inference_id": INFERENCE_ID_OUTPUT,
    }

    class Params(NmsParams):
        # noqa F821: flake8 parses the string "always" in an annotation as a name.
        image: Ref(IMAGE_KIND, batch="always") = Field(  # noqa: F821
            description="RGB images to detect objects in; delivered as one batch."
        )
        confidence: (
            UnitFloat | Literal["best", "default"] | Ref(FLOAT_KIND, STRING_KIND)
        ) = Field(
            default=DEFAULT_CONFIDENCE,
            description=f"Minimum detection confidence. {CONFIDENCE_MODES_DESCRIPTION}",
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
        )
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
        """Decode boxes (``post_process``); one result per image, in batch order."""
        members = image.content
        results: List[Any] = [None] * len(members)
        for group, group_output in zip(network_input, raw_output):
            batch = group.model.post_process(
                group_output, group.metadata, **group.options
            )
            checked_count(batch, group, block=self.type)

            kept = class_filter_tensors(group, like=batch[0].class_id)
            for predictions, row, kept_ids in zip(batch, group.rows, kept):
                predictions = keep_classes(
                    predictions, class_id=predictions.class_id, kept_ids=kept_ids
                )
                predictions.image_metadata = prediction_metadata(
                    predictions.image_metadata,
                    members[row],
                    class_names=group.model.class_names,
                    prediction_type=PREDICTION_TYPE,
                )
                results[row] = identified_result(
                    predictions, model_id=group.request.model_id
                )

        return results

    def run(
        self,
        *,
        image: Batch,
        model_id: RequestField,
        confidence: Union[float, str] = DEFAULT_CONFIDENCE,
        model_package_id: RequestField = None,
        class_filter: ClassFilterField = None,
        iou_threshold: Optional[float] = None,
        max_detections: Optional[int] = None,
        class_agnostic_nms: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """Detect objects in every image of the batch.

        The same three phases a pipelined run executes as separate stages.

        Args:
            image: The batch of RGB images of one workflow run.
            model_id: Model id for every image, or a ``Batch`` with one per
                image.
            confidence: Minimum detection confidence, or ``"best"`` /
                ``"default"``.
            model_package_id: Package id for every image, a ``Batch`` with one
                per image, or ``None`` to let the loader choose. A ``Batch``
                member may be ``None`` to let the loader choose for that image.
            class_filter: Class names to keep for every image, a ``Batch``
                with one list (or ``None``) per image, or ``None`` to keep
                every class.
            iou_threshold: NMS IoU threshold; ``None`` for the default.
            max_detections: Maximum detections per image; ``None`` for the
                default.
            class_agnostic_nms: Class-agnostic NMS; ``None`` for the default.

        Returns:
            One ``{"predictions", "model_id", "inference_id"}`` per image, in
            batch order.

        Raises:
            PhaseFailure: Phase ``network_input`` when a model cannot be
                resolved (cause ``LookupError`` for a missing or conflicting
                supplied model, or the loader's error), an image is not RGB or
                an option is invalid or not read by the model, or
                ``class_filter`` is empty or names a class the model does not
                have; phase ``result`` when a model returns another number of
                predictions.
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

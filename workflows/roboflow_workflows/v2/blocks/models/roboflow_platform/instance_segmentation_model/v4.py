"""Instance segmentation with separate execution and mask-fidelity controls.

The default ``fast`` implementation asks capable models for native score grids.
It postpones dense masks until ``predictions`` or an accurate consumer needs them.
The ``accurate`` implementation requests the model's existing dense result.
Both return the same two outputs: a view for downstream blocks and dense
predictions for serialization. Models without grid support use dense masks in
either implementation; post-processing failures are never retried silently.

Consumer parameters such as ``mask_quality="accurate"`` or ``"coarse"`` are
independent of this compile-time choice. A fast grid still contains the scores
needed for accurate interpolation. Coarse consumers may threshold its native
cells before resizing. Dense output cannot recover those unthresholded scores.
"""

import uuid
from dataclasses import replace
from typing import Annotated, Any, Dict, List, Literal, Mapping, Optional, Union

import torch
from pydantic import (
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
)
from roboflow_workflows.v2.blocks.models.cache import ModelCache, ModelRequest
from roboflow_workflows.v2.blocks.models.common import (
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_MAX_DETECTIONS,
    EMPTY_CLASS_FILTER_MESSAGE,
    ModelInput,
    ModelOption,
    checked_class_filter,
    checked_confidence,
    checked_count,
    class_filter_ids,
    kept_class_ids,
    model_request,
    models_cache,
    nms_options,
    post_process_options,
    prediction_metadata,
    rgb_tensors,
    scalar_options,
)
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import (
    BOOLEAN_KIND,
    FLOAT_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
)
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
from roboflow_workflows.v2.engine.implementations import Implementation
from roboflow_workflows.v2.engine.phases import phase

from inference_models.models.base.types import InstanceSegmentationGrid

PREDICTION_TYPE = "instance-segmentation"

CONFIDENCE_MODES = ("best", "default")

BLOCK_TYPE = "roboflow_core/roboflow_instance_segmentation_model@v4"


class CompactMasks(Implementation):
    """Keep native mask scores until a consumer requests a raster."""

    name = "fast"
    quality = ("fast",)

    def __init__(self, *, models: Union[Mapping[Any, Any], ModelCache]):
        self._models = models_cache(models, block=BLOCK_TYPE)

    @phase
    def network_input(
        self,
        image: Batch,
        model_id: str,
        model_package_id: Optional[str],
        confidence: Union[float, str],
        iou_threshold: Optional[float],
        max_detections: Optional[int],
        class_agnostic_nms: Optional[bool],
        class_filter: Optional[List[str]],
    ) -> Optional[ModelInput]:
        """Check options, resolve the call's model and preprocess all images."""
        tensors = rgb_tensors(image, block=BLOCK_TYPE)
        confidence = checked_confidence(
            confidence, modes=CONFIDENCE_MODES, block=BLOCK_TYPE
        )
        step_options = scalar_options(
            nms_options(
                confidence=confidence,
                iou_threshold=iou_threshold,
                max_detections=max_detections,
                class_agnostic_nms=class_agnostic_nms,
            ),
            block=BLOCK_TYPE,
        )
        names = checked_class_filter(class_filter, block=BLOCK_TYPE)
        request = model_request(model_id, model_package_id, block=BLOCK_TYPE)
        # An empty batch loads no model and runs nothing.
        if not tensors:
            return None

        model = self._models.get(request)
        mask_format = self._mask_format(model)
        step_options += (
            ModelOption(
                "mask_format",
                "mask_format",
                mask_format if mask_format == "grid" else None,
                "dense",
            ),
        )
        options = post_process_options(
            model.post_process, step_options, block=BLOCK_TYPE, request=request
        )
        class_ids = class_filter_ids(
            names, model.class_names, block=BLOCK_TYPE, request=request
        )

        batch, metadata = model.pre_process(tensors, input_color_format="rgb")
        model_input = ModelInput(
            request=request,
            model=model,
            batch=batch,
            metadata=metadata,
            options=options,
            class_ids=class_ids,
        )

        return model_input

    @phase
    def raw_output(self, network_input: Optional[ModelInput]) -> Any:
        """Run the network once on all images (``forward``)."""
        if network_input is None:
            return None

        raw_output = network_input.model.forward(network_input.batch)

        return raw_output

    @phase
    def result(
        self, raw_output: Any, network_input: Optional[ModelInput], image: Batch
    ) -> List[Dict[str, Any]]:
        """Decode instances; one result per image, in batch order.

        Asks ``wants`` for both mask outputs and leaves out the ones nobody
        reads.
        """
        if network_input is None:
            return []

        wants_segmentation = self.wants("segmentation")
        wants_predictions = self.wants("predictions")
        views = self._segmentations(network_input, raw_output)
        checked_count(
            views,
            image_count=len(image),
            request=network_input.request,
            block=BLOCK_TYPE,
        )

        kept_ids = kept_class_ids(
            network_input.class_ids, like=views[0].detections.class_id
        )
        results = []
        for view, member in zip(views, image):
            view = _keep_view_classes(view, kept_ids=kept_ids)
            image_metadata = prediction_metadata(
                view.detections.image_metadata,
                member,
                class_names=network_input.model.class_names,
                prediction_type=PREDICTION_TYPE,
            )
            inference_id = uuid.uuid4().hex
            view = _identified_view(
                view, image_metadata=image_metadata, namespace=inference_id
            )
            result = {
                "model_id": network_input.request.model_id,
                "inference_id": inference_id,
            }
            if wants_segmentation:
                result["segmentation"] = view
            if wants_predictions:
                result["predictions"] = view.full_res()
            results.append(result)

        return results

    def _mask_format(self, model: Any) -> str:
        """Prefer scores when this model advertises compact output support."""
        formats = getattr(model, "supported_mask_formats", ())
        mask_format = "grid" if "grid" in formats else "dense"

        return mask_format

    def _segmentations(
        self, network_input: ModelInput, raw_output: Any
    ) -> List[SegmentationView]:
        """Wrap the model's selected mask representation without upscaling."""
        batch = network_input.model.post_process(
            raw_output, network_input.metadata, **network_input.options
        )
        mask_format = network_input.options.get("mask_format", "dense")
        views = []
        for predictions in batch:
            _require_mask_format(
                predictions, mask_format=mask_format, request=network_input.request
            )
            views.append(SegmentationView.from_predictions(predictions))

        return views

    def run(
        self,
        *,
        image: Batch,
        model_id: str,
        confidence: Union[float, str] = "best",
        model_package_id: Optional[str] = None,
        class_filter: Optional[List[str]] = None,
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


class DenseMasks(CompactMasks):
    """Use the model's eager dense post-processing as the reference strategy."""

    name = "accurate"
    quality = ("accurate",)

    def _mask_format(self, model: Any) -> str:
        return "dense"


class InstanceSegmentationBlock(Block):
    """Segment object instances in a batch of images with host-owned models.

    ``segmentation`` is a ``SegmentationView`` of each image's instances,
    carrying fresh row identity; ``predictions`` is its ``full_res()``: an
    ``InstanceDetections`` with a dense bool ``(N, H, W)`` mask in the image's
    pixel coordinates and the same row record. Each output is built only
    when a reader wants it. ``image_metadata`` keeps what the model wrote and
    adds the image's frame record, ``class_names`` and ``prediction_type``.
    """

    type = BLOCK_TYPE
    implementations = (CompactMasks, DenseMasks)

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
        "model_id": Output(
            STRING_KIND, source="image", description="Model id that served this image."
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
        model_id: StrictStr | Ref(STRING_KIND) = Field(
            description=(
                "Model id (or local path) of the model in the 'models' resource, "
                "one for every image of the step; a selector may choose it per "
                "run, not per image. To serve images with different models, "
                "route them to one step per model."
            ),
            examples=["rfdetr-nano", "yolov8n-640", "$inputs.model_id"],
        )
        model_package_id: Optional[StrictStr] | Ref(STRING_KIND, allow_none=True) = (
            Field(
                default=None,
                description=(
                    "Exact model package to load, one for every image of the step; "
                    "None lets the loader choose. A selector may choose it per run, "
                    "not per image. A model supplied without a package id rejects a "
                    "package. List a model's package ids with "
                    'inference_models.AutoModel.describe_model("<model_id>").'
                ),
                examples=[None, "$inputs.model_package_id"],
            )
        )
        class_filter: Optional[List[StrictStr]] | Ref(
            LIST_OF_VALUES_KIND, allow_none=True
        ) = Field(
            default=None,
            description=(
                "Class names to keep for every image of the step; omit or null to "
                "keep every class. Every name must be a class of the serving model. "
                "A selector may choose the list per run, not per image. Applied "
                "after the model's own post-processing."
            ),
            examples=[["person", "car"], "$inputs.class_filter"],
        )
        iou_threshold: Optional[StrictFloat] | Ref(FLOAT_KIND) = Field(
            default=None,
            ge=0,
            le=1,
            description=(
                f"NMS IoU threshold, 0..1. Unset: {DEFAULT_IOU_THRESHOLD} for models "
                "with NMS. Setting it for a model without NMS (e.g. RF-DETR) fails "
                "the step."
            ),
            examples=[0.5, "$inputs.iou_threshold"],
        )
        max_detections: Optional[StrictInt] | Ref(INTEGER_KIND) = Field(
            default=None,
            ge=1,
            description=(
                f"Maximum detections per image. Unset: {DEFAULT_MAX_DETECTIONS} for "
                "models that read it. Setting it for a model that does not read it "
                "fails the step."
            ),
            examples=[100, "$inputs.max_detections"],
        )
        class_agnostic_nms: Optional[StrictBool] | Ref(BOOLEAN_KIND) = Field(
            default=None,
            description=(
                "Run NMS across classes. Unset: False for models with NMS. Setting "
                "it for a model without NMS fails the step."
            ),
            examples=[True, "$inputs.class_agnostic_nms"],
        )
        # noqa F821: flake8 parses the string "always" in an annotation as a name.
        image: Ref(IMAGE_KIND, batch="always") = Field(  # noqa: F821
            description="RGB images to segment; delivered as one batch."
        )
        confidence: (
            Annotated[StrictFloat, Field(ge=0, le=1)]
            | Literal["best", "default"]
            | Ref(FLOAT_KIND, STRING_KIND)
        ) = Field(
            default="best",
            description=(
                "Minimum instance confidence. 'best' uses the model's evaluated "
                "per-class thresholds (then its global one, then its default); "
                "'default' uses the model's default threshold; a number 0..1 "
                "applies to every class."
            ),
            examples=["best", 0.4, "$inputs.confidence"],
        )

        @field_validator("model_id", "model_package_id")
        @classmethod
        def _reject_empty(cls, value: Optional[str]) -> Optional[str]:
            if value == "":
                raise ValueError("must be a non-empty string")

            return value

        @field_validator("class_filter")
        @classmethod
        def _reject_empty_filter(cls, value: Any) -> Any:
            if isinstance(value, list) and not value:
                raise ValueError(EMPTY_CLASS_FILTER_MESSAGE)

            return value

    @classmethod
    def discover_dependent_resources(
        cls, params: BlockParams
    ) -> List[DependentResource]:
        """Report the model this step needs from the ``models`` resource."""
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


def _require_mask_format(
    predictions: Any, *, mask_format: str, request: ModelRequest
) -> None:
    mask = predictions.mask
    valid = (
        isinstance(mask, InstanceSegmentationGrid)
        if mask_format == "grid"
        else isinstance(mask, torch.Tensor) and mask.ndim == 3
    )
    if not valid:
        raise ValueError(
            f"{BLOCK_TYPE}: model {str(request)!r} returned "
            f"{type(mask).__name__} masks after requesting mask_format={mask_format!r}"
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
    detections = replace(view.detections, image_metadata=image_metadata)
    identified = with_row_identity(detections, namespace=namespace)
    identified_view = view.with_image_metadata(identified.image_metadata)

    return identified_view

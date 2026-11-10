"""Batched object detection with host-owned ``inference_models`` models.

One model and one set of options serve every image of a call::

    network_input   check options, resolve the model, pre_process all images
    raw_output      model.forward(batch)
    result          post_process -> Detections; class filter, frame metadata,
                    fresh row identity; one result per image, in batch order

The ``models`` resource is a ``ModelCache`` or a mapping of model id (or
``ModelRequest``) to a loaded model (``common.models_cache``). Options reach a
model only through the ``post_process`` keywords it reads
(``common.post_process_options``). Each call carries its own ``ModelInput``;
the block holds only the cache, so overlapping phases share no call state.
"""

from typing import Annotated, Any, Dict, List, Literal, Mapping, Optional, Union

from pydantic import Field, StrictBool, StrictFloat, StrictInt, StrictStr, field_validator
from roboflow_workflows.v2.blocks.models.cache import ModelCache
from roboflow_workflows.v2.blocks.models.common import (
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_MAX_DETECTIONS,
    EMPTY_CLASS_FILTER_MESSAGE,
    ModelInput,
    checked_class_filter,
    checked_confidence,
    checked_count,
    class_filter_ids,
    identified_result,
    keep_classes,
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
    ``post_process``, each called once for the whole batch; ``run`` composes
    the same three. ``predictions`` are the model's
    native ``Detections`` in each image's pixel coordinates, carrying fresh
    row identity. Their ``image_metadata`` keeps what the model wrote and adds
    the image's frame record (``FrameRef.prediction_metadata()``: source,
    video time and workflow-root geometry), the model's ``class_names`` and
    ``prediction_type``.
    """

    type = "roboflow_core/roboflow_object_detection_model@v3"
    outputs = {
        "predictions": Output(
            OBJECT_DETECTION_PREDICTION_KIND,
            source="image",
            description="Native Detections in this image's pixel coordinates.",
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
        model_package_id: Optional[StrictStr] | Ref(
            STRING_KIND, allow_none=True
        ) = Field(
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
            description="RGB images to detect objects in; delivered as one batch."
        )
        confidence: (
            Annotated[StrictFloat, Field(ge=0, le=1)]
            | Literal["best", "default"]
            | Ref(FLOAT_KIND, STRING_KIND)
        ) = Field(
            default=DEFAULT_CONFIDENCE,
            description=(
                "Minimum detection confidence. 'best' uses the model's evaluated "
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

    def __init__(self, *, models: Union[Mapping[Any, Any], ModelCache]):
        self._models = models_cache(models, block=self.type)

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
        """Check options, resolve the call's model and preprocess all images.

        Every argument check runs before the model loads; an empty batch
        returns ``None`` and loads nothing.
        """
        tensors = rgb_tensors(image, block=self.type)
        confidence = checked_confidence(
            confidence, modes=CONFIDENCE_MODES, block=self.type
        )
        step_options = scalar_options(
            nms_options(
                confidence=confidence,
                iou_threshold=iou_threshold,
                max_detections=max_detections,
                class_agnostic_nms=class_agnostic_nms,
            ),
            block=self.type,
        )
        names = checked_class_filter(class_filter, block=self.type)
        request = model_request(model_id, model_package_id, block=self.type)
        if not tensors:
            return None

        model = self._models.get(request)
        options = post_process_options(
            model.post_process, step_options, block=self.type, request=request
        )
        class_ids = class_filter_ids(
            names, model.class_names, block=self.type, request=request
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
        """Decode boxes (``post_process``); one result per image, in batch order."""
        if network_input is None:
            return []

        model = network_input.model
        batch = model.post_process(
            raw_output, network_input.metadata, **network_input.options
        )
        checked_count(
            batch,
            image_count=len(image),
            request=network_input.request,
            block=self.type,
        )

        kept_ids = kept_class_ids(network_input.class_ids, like=batch[0].class_id)
        results = []
        for predictions, member in zip(batch, image):
            predictions = keep_classes(
                predictions, class_id=predictions.class_id, kept_ids=kept_ids
            )
            predictions.image_metadata = prediction_metadata(
                predictions.image_metadata,
                member,
                class_names=model.class_names,
                prediction_type=PREDICTION_TYPE,
            )
            results.append(
                identified_result(
                    predictions, model_id=network_input.request.model_id
                )
            )

        return results

    def run(
        self,
        *,
        image: Batch,
        model_id: str,
        confidence: Union[float, str] = DEFAULT_CONFIDENCE,
        model_package_id: Optional[str] = None,
        class_filter: Optional[List[str]] = None,
        iou_threshold: Optional[float] = None,
        max_detections: Optional[int] = None,
        class_agnostic_nms: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """Detect objects in every image of the batch with the same three phases.

        Args:
            image: The batch of RGB images of one workflow run.
            model_id: Model id for every image.
            confidence: Minimum detection confidence, or ``"best"`` /
                ``"default"``.
            model_package_id: Package id for every image, or ``None`` to let
                the loader choose.
            class_filter: Class names to keep for every image, or ``None`` to
                keep every class.
            iou_threshold: NMS IoU threshold; ``None`` for the default.
            max_detections: Maximum detections per image; ``None`` for the
                default.
            class_agnostic_nms: Class-agnostic NMS; ``None`` for the default.

        Returns:
            One ``{"predictions", "model_id", "inference_id"}`` per image, in
            batch order.

        Raises:
            PhaseFailure: Phase ``network_input`` when a setting is invalid or a
                ``Batch``, an image is not RGB, the model cannot be resolved,
                an option is not read by the model or ``class_filter`` names a
                class the model does not have; phase ``result`` when the model
                returns another number of predictions than images.
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

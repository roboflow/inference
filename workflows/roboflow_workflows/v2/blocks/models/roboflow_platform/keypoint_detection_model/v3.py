"""Batched keypoint detection with host-owned ``inference_models`` models.

Model selection and phases follow the object-detection block (see
``common``): one model and one set of options for every image of a call.
``post_process`` returns skeletons and, for most models, one box per
skeleton::

    (List[KeyPoints], List[Detections] | None)  ->  per image:
        (KeyPoints, Detections | None)            class filter on both,
                                                  frame metadata on both,
                                                  row identity on the boxes
                                                  (on KeyPoints without boxes)

``keypoint_confidence`` reaches the model as ``key_points_threshold``; like
the NMS options it is passed only to models whose ``post_process`` reads it.
A model that applies its thresholds inside ``forward`` (the RF-DETR two-stage
backend) reads neither threshold: ``"default"`` keeps its own, a number fails.
"""

from typing import Annotated, Any, Dict, List, Literal, Mapping, Optional, Union

from pydantic import Field, StrictBool, StrictFloat, StrictInt, StrictStr, field_validator
from roboflow_workflows.v2.blocks.models.cache import ModelCache
from roboflow_workflows.v2.blocks.models.common import (
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_MAX_DETECTIONS,
    EMPTY_CLASS_FILTER_MESSAGE,
    ModelInput,
    ModelOption,
    checked_confidence,
    checked_count,
    checked_class_filter,
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
from roboflow_workflows.v2.data.predictions import KEYPOINT_DETECTION_PREDICTION_KIND
from roboflow_workflows.v2.engine.declaration import (
    Block,
    BlockParams,
    DependentResource,
    Output,
    Ref,
)
from roboflow_workflows.v2.engine.phases import phase


PREDICTION_TYPE = "keypoint-detection"

CONFIDENCE_MODES = ("default",)

DEFAULT_KEYPOINT_CONFIDENCE = 0.0

class KeypointDetectionBlock(Block):
    """Detect skeletons in a batch of images with host-owned models.

    ``predictions`` are ``(KeyPoints, Detections | None)`` in each image's
    pixel coordinates. Both carriers get the same ``image_metadata``: what the
    model wrote on the skeletons, the image's frame record, ``class_names``,
    the model's ``key_points_classes`` and ``prediction_type``.
    """

    type = "roboflow_core/roboflow_keypoint_detection_model@v3"
    outputs = {
        "predictions": Output(
            KEYPOINT_DETECTION_PREDICTION_KIND,
            source="image",
            description=(
                "Native (KeyPoints, Detections | None) of this image. Detections "
                "is None when the model returns no boxes; blocks that need boxes "
                "then fail with an error, and keypoint drawing still works."
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
            description="RGB images to detect skeletons in; delivered as one batch."
        )
        confidence: (
            Annotated[StrictFloat, Field(ge=0, le=1)]
            | Literal["default"]
            | Ref(FLOAT_KIND, STRING_KIND)
        ) = Field(
            default="default",
            description=(
                "Minimum skeleton confidence: a number 0..1, or 'default' "
                "for the model's default threshold."
            ),
            examples=["default", 0.5, "$inputs.confidence"],
        )
        keypoint_confidence: Optional[StrictFloat] | Ref(FLOAT_KIND) = Field(
            default=None,
            ge=0,
            le=1,
            description=(
                "Minimum confidence of a visible keypoint, 0..1. Unset: "
                f"{DEFAULT_KEYPOINT_CONFIDENCE} for models that read it. Setting "
                "it for a model that does not read it fails the step."
            ),
            examples=[0.3, "$inputs.keypoint_confidence"],
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
        keypoint_confidence: Optional[float],
        class_filter: Optional[List[str]],
    ) -> Optional[ModelInput]:
        """Check options, resolve the call's model and preprocess all images."""
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
            )
            + (
                ModelOption(
                    "keypoint_confidence",
                    "key_points_threshold",
                    keypoint_confidence,
                    DEFAULT_KEYPOINT_CONFIDENCE,
                ),
            ),
            block=self.type,
        )
        names = checked_class_filter(class_filter, block=self.type)
        request = model_request(model_id, model_package_id, block=self.type)
        # An empty batch loads no model and runs nothing.
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
        """Decode skeletons (``post_process``); one result per image, in order."""
        if network_input is None:
            return []

        model = network_input.model
        key_points, boxes = model.post_process(
            raw_output, network_input.metadata, **network_input.options
        )
        for carriers in (key_points, boxes):
            if carriers is not None:
                checked_count(
                    carriers,
                    image_count=len(image),
                    request=network_input.request,
                    block=self.type,
                )

        kept_ids = kept_class_ids(
            network_input.class_ids, like=key_points[0].class_id
        )
        results = []
        for position, member in enumerate(image):
            skeletons = key_points[position]
            prediction = keep_classes(
                (skeletons, None if boxes is None else boxes[position]),
                class_id=skeletons.class_id,
                kept_ids=kept_ids,
            )
            metadata = {
                **prediction_metadata(
                    prediction[0].image_metadata,
                    member,
                    class_names=model.class_names,
                    prediction_type=PREDICTION_TYPE,
                ),
                "key_points_classes": model.key_points_classes,
            }
            for carrier in prediction:
                if carrier is not None:
                    carrier.image_metadata = metadata
            results.append(
                identified_result(prediction, model_id=network_input.request.model_id)
            )

        return results

    def run(
        self,
        *,
        image: Batch,
        model_id: str,
        confidence: Union[float, str] = "default",
        model_package_id: Optional[str] = None,
        class_filter: Optional[List[str]] = None,
        iou_threshold: Optional[float] = None,
        max_detections: Optional[int] = None,
        class_agnostic_nms: Optional[bool] = None,
        keypoint_confidence: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Detect skeletons in every image of the batch with the same phases.

        Returns:
            One ``{"predictions", "model_id", "inference_id"}`` per image, in
            batch order.
        """
        network_input = self.network_input(
            image,
            model_id,
            model_package_id,
            confidence,
            iou_threshold,
            max_detections,
            class_agnostic_nms,
            keypoint_confidence,
            class_filter,
        )
        raw_output = self.raw_output(network_input)
        results = self.result(raw_output, network_input, image)

        return results

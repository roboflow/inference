"""Batched keypoint detection with host-owned ``inference_models`` models.

Model selection, grouping and phases follow the object-detection block (see
``common``). ``post_process`` returns skeletons and, for most models, one box
per skeleton::

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

from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Union

from pydantic import Field, StrictFloat
from roboflow_workflows.v2.blocks.models.cache import ModelCache
from roboflow_workflows.v2.blocks.models.common import (
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

    type = "v2/keypoint_detection@v1"
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
        "model_id": MODEL_ID_OUTPUT,
        "inference_id": INFERENCE_ID_OUTPUT,
    }

    class Params(NmsParams):
        # noqa F821: flake8 parses the string "always" in an annotation as a name.
        image: Ref(IMAGE_KIND, batch="always") = Field(  # noqa: F821
            description="RGB images to detect skeletons in; delivered as one batch."
        )
        confidence: UnitFloat | Literal["default"] | Ref(FLOAT_KIND, STRING_KIND) = (
            Field(
                default="default",
                description=(
                    "Minimum skeleton confidence: a number 0..1, or 'default' "
                    "for the model's default threshold."
                ),
                examples=["default", 0.5, "$inputs.confidence"],
            )
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
        keypoint_confidence: Optional[float],
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
        ) + (
            ModelOption(
                "keypoint_confidence",
                "key_points_threshold",
                keypoint_confidence,
                DEFAULT_KEYPOINT_CONFIDENCE,
            ),
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
        """Decode skeletons (``post_process``); one result per image, in order."""
        members = image.content
        results: List[Any] = [None] * len(members)
        for group, group_output in zip(network_input, raw_output):
            key_points, boxes = group.model.post_process(
                group_output, group.metadata, **group.options
            )
            checked_count(key_points, group, block=self.type)
            if boxes is not None:
                checked_count(boxes, group, block=self.type)

            kept = class_filter_tensors(group, like=key_points[0].class_id)
            for position, (row, kept_ids) in enumerate(zip(group.rows, kept)):
                skeletons = key_points[position]
                prediction = keep_classes(
                    (skeletons, None if boxes is None else boxes[position]),
                    class_id=skeletons.class_id,
                    kept_ids=kept_ids,
                )
                metadata = {
                    **prediction_metadata(
                        prediction[0].image_metadata,
                        members[row],
                        class_names=group.model.class_names,
                        prediction_type=PREDICTION_TYPE,
                    ),
                    "key_points_classes": group.model.key_points_classes,
                }
                for carrier in prediction:
                    if carrier is not None:
                        carrier.image_metadata = metadata
                results[row] = identified_result(
                    prediction, model_id=group.request.model_id
                )

        return results

    def run(
        self,
        *,
        image: Batch,
        model_id: RequestField,
        confidence: Union[float, str] = "default",
        model_package_id: RequestField = None,
        class_filter: ClassFilterField = None,
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

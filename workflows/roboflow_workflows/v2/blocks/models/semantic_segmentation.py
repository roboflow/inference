"""Batched semantic segmentation with host-owned ``inference_models`` models.

Model selection, grouping and phases follow the object-detection block (see
``common``). ``predictions`` are the model's ``SemanticSegmentationResult``
per image: an integer ``(H, W)`` class map and the float ``(H, W)``
confidence of each pixel's class, on the model's device. A class grid has
no instances, so there are no boxes, no per-class rows and no row identity.
"""

import uuid
from typing import Any, Dict, List, Literal, Mapping, Tuple, Union

from pydantic import Field
from roboflow_workflows.v2.blocks.models.cache import ModelCache
from roboflow_workflows.v2.blocks.models.common import (
    CONFIDENCE_MODES_DESCRIPTION,
    INFERENCE_ID_OUTPUT,
    MODEL_ID_OUTPUT,
    ModelGroup,
    ModelSelectionParams,
    RequestField,
    UnitFloat,
    checked_confidence,
    checked_count,
    confidence_option,
    forward_groups,
    model_dependency,
    models_cache,
    prediction_metadata,
    preprocess_groups,
)
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import FLOAT_KIND, STRING_KIND
from roboflow_workflows.v2.data.predictions import NATIVE_SEMANTIC_SEGMENTATION_KIND
from roboflow_workflows.v2.engine.declaration import (
    Block,
    BlockParams,
    DependentResource,
    Output,
    Ref,
)
from roboflow_workflows.v2.engine.phases import phase

PREDICTION_TYPE = "semantic-segmentation"
CONFIDENCE_MODES = ("best", "default")


class SemanticSegmentationBlock(Block):
    """Label every pixel of a batch of images with host-owned models.

    Pixels below the confidence threshold get the model's background class,
    as the model's own ``post_process`` decides. ``image_metadata`` keeps what
    the model wrote and adds the image's frame record, ``class_names`` and
    ``prediction_type``.
    """

    type = "v2/semantic_segmentation@v1"
    outputs = {
        "predictions": Output(
            NATIVE_SEMANTIC_SEGMENTATION_KIND,
            source="image",
            description="Native SemanticSegmentationResult in this image's pixels.",
        ),
        "model_id": MODEL_ID_OUTPUT,
        "inference_id": INFERENCE_ID_OUTPUT,
    }

    class Params(ModelSelectionParams):
        # noqa F821: flake8 parses the string "always" in an annotation as a name.
        image: Ref(IMAGE_KIND, batch="always") = Field(  # noqa: F821
            description="RGB images to segment; delivered as one batch."
        )
        confidence: (
            UnitFloat | Literal["best", "default"] | Ref(FLOAT_KIND, STRING_KIND)
        ) = Field(
            default="best",
            description=f"Minimum pixel confidence. {CONFIDENCE_MODES_DESCRIPTION}",
            examples=["best", 0.5, "$inputs.confidence"],
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
    ) -> Tuple[ModelGroup, ...]:
        """Check the confidence, resolve each row's model, preprocess groups."""
        confidence = checked_confidence(
            confidence, modes=CONFIDENCE_MODES, block=self.type
        )
        groups = preprocess_groups(
            image,
            cache=self._models,
            model_id=model_id,
            model_package_id=model_package_id,
            options=(confidence_option(confidence),),
            block=self.type,
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
        """Decode class maps (``post_process``); one result per image, in order."""
        members = image.content
        results: List[Any] = [None] * len(members)
        for group, group_output in zip(network_input, raw_output):
            batch = group.model.post_process(
                group_output, group.metadata, **group.options
            )
            checked_count(batch, group, block=self.type)

            for segmentation, row in zip(batch, group.rows):
                segmentation.image_metadata = prediction_metadata(
                    segmentation.image_metadata,
                    members[row],
                    class_names=group.model.class_names,
                    prediction_type=PREDICTION_TYPE,
                )
                results[row] = {
                    "predictions": segmentation,
                    "model_id": group.request.model_id,
                    "inference_id": uuid.uuid4().hex,
                }

        return results

    def run(
        self,
        *,
        image: Batch,
        model_id: RequestField,
        confidence: Union[float, str] = "best",
        model_package_id: RequestField = None,
    ) -> List[Dict[str, Any]]:
        """Segment every image of the batch with the same three phases.

        Returns:
            One ``{"predictions", "model_id", "inference_id"}`` per image, in
            batch order.
        """
        network_input = self.network_input(
            image, model_id, model_package_id, confidence
        )
        raw_output = self.raw_output(network_input)
        results = self.result(raw_output, network_input, image)

        return results

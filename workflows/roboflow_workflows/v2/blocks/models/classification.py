"""Single- and multi-label image classification with host-owned models.

Both blocks follow the object-detection block's model selection, grouping and
phases (see ``common``)::

    network_input   group rows by ModelRequest, model.pre_process(group images)
    raw_output      model.forward(group batch)
    result          model.post_process -> one prediction per image, in row order

A single-label model returns one batched ``ClassificationPrediction`` for a
group; the block splits it into one-image predictions (``class_id`` ``(1,)``,
``confidence`` ``(1, C)``: the full softmax, sliced, not copied). Top-1 never
depends on a threshold, so this block has no confidence option. A multi-label
model returns one ``MultiLabelClassificationPrediction`` per image; its
``class_ids`` are the classes above the confidence threshold.

A classification has no rows, so it gets no row identity. When the image is a
crop of a detection, the prediction carries that detection's row reference
(``image.parent_row``) for joining labels back to it.
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
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import FLOAT_KIND, STRING_KIND
from roboflow_workflows.v2.data.predictions import CLASSIFICATION_PREDICTION_KIND
from roboflow_workflows.v2.data.row_identity import with_classification_parent
from roboflow_workflows.v2.engine.declaration import (
    Block,
    BlockParams,
    DependentResource,
    Output,
    Ref,
)
from roboflow_workflows.v2.engine.phases import phase

from inference_models.models.base.classification import ClassificationPrediction

PREDICTION_TYPE = "classification"
MULTI_LABEL_CONFIDENCE_MODES = ("best", "default")


class _ClassifierParams(ModelSelectionParams):
    # noqa F821: flake8 parses the string "always" in an annotation as a name.
    image: Ref(IMAGE_KIND, batch="always") = Field(  # noqa: F821
        description="RGB images to classify; delivered as one batch."
    )


class ClassificationBlock(Block):
    """Assign one class to each image of a batch with host-owned models.

    ``predictions`` hold one image each: ``class_id`` ``(1,)`` is the top-1
    class and ``confidence`` ``(1, C)`` the model's full distribution.
    ``images_metadata[0]`` keeps what the model wrote and adds the image's
    frame record, ``class_names`` and ``prediction_type``.
    """

    type = "v2/classification@v1"
    outputs = {
        "predictions": Output(
            CLASSIFICATION_PREDICTION_KIND,
            source="image",
            description="Native one-image ClassificationPrediction.",
        ),
        "model_id": MODEL_ID_OUTPUT,
        "inference_id": INFERENCE_ID_OUTPUT,
    }

    class Params(_ClassifierParams):
        pass

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
        self, image: Batch, model_id: RequestField, model_package_id: RequestField
    ) -> Tuple[ModelGroup, ...]:
        """Resolve each row's model and preprocess each group."""
        groups = preprocess_groups(
            image,
            cache=self._models,
            model_id=model_id,
            model_package_id=model_package_id,
            options=(),
            block=self.type,
            returns_metadata=False,
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
        """Split each group's batched prediction into one per image."""
        members = image.content
        results: List[Any] = [None] * len(members)
        for group, group_output in zip(network_input, raw_output):
            batched = group.model.post_process(group_output, **group.options)
            checked_count(batched.class_id, group, block=self.type)

            for position, row in enumerate(group.rows):
                model_metadata = (
                    batched.images_metadata[position]
                    if batched.images_metadata
                    else None
                )
                prediction = ClassificationPrediction(
                    class_id=batched.class_id[position : position + 1],
                    confidence=batched.confidence[position : position + 1],
                    images_metadata=[
                        prediction_metadata(
                            model_metadata,
                            members[row],
                            class_names=group.model.class_names,
                            prediction_type=PREDICTION_TYPE,
                        )
                    ],
                )
                results[row] = classification_result(
                    prediction, members[row], model_id=group.request.model_id
                )

        return results

    def run(
        self,
        *,
        image: Batch,
        model_id: RequestField,
        model_package_id: RequestField = None,
    ) -> List[Dict[str, Any]]:
        """Classify every image of the batch with the same three phases.

        Returns:
            One ``{"predictions", "model_id", "inference_id"}`` per image, in
            batch order.
        """
        network_input = self.network_input(image, model_id, model_package_id)
        raw_output = self.raw_output(network_input)
        results = self.result(raw_output, network_input, image)

        return results


class MultiLabelClassificationBlock(Block):
    """Assign any number of classes to each image with host-owned models.

    ``predictions`` are the model's ``MultiLabelClassificationPrediction``:
    ``class_ids`` above the confidence threshold and ``confidence`` ``(C,)``
    for every class. ``image_metadata`` keeps what the model wrote and adds the
    image's frame record, ``class_names`` and ``prediction_type``.
    """

    type = "v2/multi_label_classification@v1"
    outputs = {
        "predictions": Output(
            CLASSIFICATION_PREDICTION_KIND,
            source="image",
            description="Native MultiLabelClassificationPrediction of one image.",
        ),
        "model_id": MODEL_ID_OUTPUT,
        "inference_id": INFERENCE_ID_OUTPUT,
    }

    class Params(_ClassifierParams):
        confidence: (
            UnitFloat | Literal["best", "default"] | Ref(FLOAT_KIND, STRING_KIND)
        ) = Field(
            default="best",
            description=f"Threshold of a predicted label. {CONFIDENCE_MODES_DESCRIPTION}",
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
            confidence, modes=MULTI_LABEL_CONFIDENCE_MODES, block=self.type
        )
        groups = preprocess_groups(
            image,
            cache=self._models,
            model_id=model_id,
            model_package_id=model_package_id,
            options=(confidence_option(confidence),),
            block=self.type,
            returns_metadata=False,
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
        """Decode labels (``post_process``); one result per image, in order."""
        members = image.content
        results: List[Any] = [None] * len(members)
        for group, group_output in zip(network_input, raw_output):
            batch = group.model.post_process(group_output, **group.options)
            checked_count(batch, group, block=self.type)

            for prediction, row in zip(batch, group.rows):
                prediction.image_metadata = prediction_metadata(
                    prediction.image_metadata,
                    members[row],
                    class_names=group.model.class_names,
                    prediction_type=PREDICTION_TYPE,
                )
                results[row] = classification_result(
                    prediction, members[row], model_id=group.request.model_id
                )

        return results

    def run(
        self,
        *,
        image: Batch,
        model_id: RequestField,
        confidence: Union[float, str] = "best",
        model_package_id: RequestField = None,
    ) -> List[Dict[str, Any]]:
        """Classify every image of the batch with the same three phases.

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


def classification_result(
    prediction: Any, image: ImageData, *, model_id: str
) -> Dict[str, Any]:
    """One image's outputs; the prediction carries the image's parent row."""
    result = {
        "predictions": with_classification_parent(prediction, image.parent_row),
        "model_id": model_id,
        "inference_id": uuid.uuid4().hex,
    }

    return result

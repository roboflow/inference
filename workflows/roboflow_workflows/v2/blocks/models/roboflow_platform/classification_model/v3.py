"""Assign one class to each image of a batch with one host-owned model per call.

A single-label model returns one batched ``ClassificationPrediction`` for the
call; the block splits it into one-image predictions (``class_id`` ``(1,)``,
``confidence`` ``(1, C)``: the full softmax, sliced, not copied). Top-1 never
depends on a threshold, so this block has no confidence option.
"""

from typing import Any, Dict, List, Mapping, Optional, Union

from inference_models.models.base.classification import ClassificationPrediction
from pydantic import Field, StrictStr, field_validator
from roboflow_workflows.v2.blocks.models._shared.classification import (
    PREDICTION_TYPE,
    classification_result,
)
from roboflow_workflows.v2.blocks.models.cache import ModelCache
from roboflow_workflows.v2.blocks.models.common import (
    ModelInput,
    checked_count,
    model_request,
    models_cache,
    prediction_metadata,
    rgb_tensors,
)
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import STRING_KIND
from roboflow_workflows.v2.data.predictions import CLASSIFICATION_PREDICTION_KIND
from roboflow_workflows.v2.engine.declaration import (
    Block,
    BlockParams,
    DependentResource,
    Output,
    Ref,
)
from roboflow_workflows.v2.engine.phases import phase


class ClassificationBlock(Block):
    """Assign one class to each image of a batch with host-owned models.

    ``predictions`` hold one image each: ``class_id`` ``(1,)`` is the top-1
    class and ``confidence`` ``(1, C)`` the model's full distribution.
    ``images_metadata[0]`` keeps what the model wrote and adds the image's
    frame record, ``class_names`` and ``prediction_type``.
    """

    type = "roboflow_core/roboflow_classification_model@v3"
    outputs = {
        "predictions": Output(
            CLASSIFICATION_PREDICTION_KIND,
            source="image",
            description="Native one-image ClassificationPrediction.",
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
        # noqa F821: flake8 parses the string "always" in an annotation as a name.
        image: Ref(IMAGE_KIND, batch="always") = Field(  # noqa: F821
            description="RGB images to classify; delivered as one batch."
        )

        @field_validator("model_id", "model_package_id")
        @classmethod
        def _reject_empty(cls, value: Optional[str]) -> Optional[str]:
            if value == "":
                raise ValueError("must be a non-empty string")

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
        self, image: Batch, model_id: str, model_package_id: Optional[str]
    ) -> Optional[ModelInput]:
        """Resolve the call's model and preprocess all images.

        An empty batch returns ``None`` and loads nothing.
        """
        tensors = rgb_tensors(image, block=self.type)
        request = model_request(model_id, model_package_id, block=self.type)
        if not tensors:
            return None

        model = self._models.get(request)

        # A single-label classifier has no options; top-1 needs no threshold.
        batch = model.pre_process(tensors, input_color_format="rgb")
        model_input = ModelInput(
            request=request, model=model, batch=batch, metadata=None, options={}
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
        """Split the batched prediction into one per image."""
        if network_input is None:
            return []

        model = network_input.model
        batched = model.post_process(raw_output, **network_input.options)
        checked_count(
            batched.class_id,
            image_count=len(image),
            request=network_input.request,
            block=self.type,
        )

        results = []
        for position, member in enumerate(image):
            model_metadata = (
                batched.images_metadata[position] if batched.images_metadata else None
            )
            prediction = ClassificationPrediction(
                class_id=batched.class_id[position : position + 1],
                confidence=batched.confidence[position : position + 1],
                images_metadata=[
                    prediction_metadata(
                        model_metadata,
                        member,
                        class_names=model.class_names,
                        prediction_type=PREDICTION_TYPE,
                    )
                ],
            )
            results.append(
                classification_result(
                    prediction, member, model_id=network_input.request.model_id
                )
            )

        return results

    def run(
        self,
        *,
        image: Batch,
        model_id: str,
        model_package_id: Optional[str] = None,
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

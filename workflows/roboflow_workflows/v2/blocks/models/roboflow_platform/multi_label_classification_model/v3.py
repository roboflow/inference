"""Assign any number of classes to each image with one host-owned model per call.

The model returns one ``MultiLabelClassificationPrediction`` per image; its
``class_ids`` are the classes above the confidence threshold.
"""

from typing import Annotated, Any, Dict, List, Literal, Mapping, Optional, Union

from pydantic import Field, StrictFloat, StrictStr, field_validator
from roboflow_workflows.v2.blocks.models._shared.classification import (
    PREDICTION_TYPE,
    classification_result,
)
from roboflow_workflows.v2.blocks.models.cache import ModelCache
from roboflow_workflows.v2.blocks.models.common import (
    ModelInput,
    checked_confidence,
    checked_count,
    confidence_option,
    model_request,
    models_cache,
    post_process_options,
    prediction_metadata,
    rgb_tensors,
)
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import FLOAT_KIND, STRING_KIND
from roboflow_workflows.v2.data.predictions import CLASSIFICATION_PREDICTION_KIND
from roboflow_workflows.v2.engine.declaration import (
    Block,
    BlockParams,
    DependentResource,
    Output,
    Ref,
)
from roboflow_workflows.v2.engine.phases import phase

CONFIDENCE_MODES = ("best", "default")


class MultiLabelClassificationBlock(Block):
    """Assign any number of classes to each image with host-owned models.

    ``predictions`` are the model's ``MultiLabelClassificationPrediction``:
    ``class_ids`` above the confidence threshold and ``confidence`` ``(C,)``
    for every class. ``image_metadata`` keeps what the model wrote and adds the
    image's frame record, ``class_names`` and ``prediction_type``.
    """

    type = "roboflow_core/roboflow_multi_label_classification_model@v3"
    outputs = {
        "predictions": Output(
            CLASSIFICATION_PREDICTION_KIND,
            source="image",
            description="Native MultiLabelClassificationPrediction of one image.",
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
        confidence: (
            Annotated[StrictFloat, Field(ge=0, le=1)]
            | Literal["best", "default"]
            | Ref(FLOAT_KIND, STRING_KIND)
        ) = Field(
            default="best",
            description=(
                "Threshold of a predicted label. 'best' uses the model's evaluated "
                "per-class thresholds (then its global one, then its default); "
                "'default' uses the model's default threshold; a number 0..1 "
                "applies to every class."
            ),
            examples=["best", 0.5, "$inputs.confidence"],
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
        self,
        image: Batch,
        model_id: str,
        model_package_id: Optional[str],
        confidence: Union[float, str],
    ) -> Optional[ModelInput]:
        """Check the confidence, resolve the call's model, preprocess images.

        An empty batch returns ``None`` and loads nothing.
        """
        tensors = rgb_tensors(image, block=self.type)
        confidence = checked_confidence(
            confidence, modes=CONFIDENCE_MODES, block=self.type
        )
        request = model_request(model_id, model_package_id, block=self.type)
        if not tensors:
            return None

        model = self._models.get(request)
        options = post_process_options(
            model.post_process,
            (confidence_option(confidence),),
            block=self.type,
            request=request,
        )

        batch = model.pre_process(tensors, input_color_format="rgb")
        model_input = ModelInput(
            request=request, model=model, batch=batch, metadata=None, options=options
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
        """Decode labels (``post_process``); one result per image, in order."""
        if network_input is None:
            return []

        model = network_input.model
        batch = model.post_process(raw_output, **network_input.options)
        checked_count(
            batch,
            image_count=len(image),
            request=network_input.request,
            block=self.type,
        )

        results = []
        for prediction, member in zip(batch, image):
            prediction.image_metadata = prediction_metadata(
                prediction.image_metadata,
                member,
                class_names=model.class_names,
                prediction_type=PREDICTION_TYPE,
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
        confidence: Union[float, str] = "best",
        model_package_id: Optional[str] = None,
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

"""Read text with DocTR models (``StructuredOCRModel``) from ``models``.

Canonical type: ``roboflow_core/ocr_model@v1``. A DocTR row is a ``block``,
``line`` or ``word`` (class ids 0..2), as in V1.
"""

from typing import Any, Dict, List, Mapping, Optional, Union

from pydantic import Field, StrictStr, field_validator
from roboflow_workflows.v2.blocks.models._shared.ocr import (
    PREDICTION_TYPE,
    _structured_results,
)
from roboflow_workflows.v2.blocks.models.cache import ModelCache
from roboflow_workflows.v2.blocks.models.common import (
    ModelInput,
    model_request,
    models_cache,
    rgb_tensors,
)
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import STRING_KIND
from roboflow_workflows.v2.data.predictions import OBJECT_DETECTION_PREDICTION_KIND
from roboflow_workflows.v2.engine.declaration import (
    Block,
    BlockParams,
    DependentResource,
    Output,
    Ref,
)
from roboflow_workflows.v2.engine.phases import phase


DOCTR_MODEL_ID = "doctr/default"


class OCRModelBlock(Block):
    """Read text with DocTR models (``StructuredOCRModel``) from ``models``.

    The phases are the model's ``pre_process``, ``forward`` and
    ``post_process``, each called once for the whole batch; ``run`` composes
    the same three. ``predictions`` hold the model's word, line and
    block boxes with their ``"text"``; the model's ``class_names`` name them.
    """

    type = "roboflow_core/ocr_model@v1"
    metadata = {
        "name": "OCR Model",
        "port_of": "roboflow_core/ocr_model@v1",
        "section": "model",
        "search_keywords": ["ocr", "text", "doctr"],
    }
    outputs = {
        "result": Output(
            STRING_KIND,
            source="image",
            description="Recognised text of the whole image.",
        ),
        "predictions": Output(
            OBJECT_DETECTION_PREDICTION_KIND,
            source="image",
            description=(
                "Native Detections of words, lines and blocks with their 'text'."
            ),
        ),
        "model_id": Output(
            STRING_KIND,
            source="image",
            description="Model id that served this image.",
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
            default=DOCTR_MODEL_ID,
            description=(
                "Model id (or local path) of a DocTR model in the 'models' "
                "resource, one for every image; a selector may choose it per "
                "run, not per image."
            ),
            examples=[DOCTR_MODEL_ID, "$inputs.model_id"],
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
        # noqa F821: flake8 parses the string "always" in an annotation as a name.
        image: Ref(IMAGE_KIND, batch="always") = Field(  # noqa: F821
            description="RGB images to read text from; delivered as one batch."
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
        """Resolve the call's model and preprocess all images."""
        tensors = rgb_tensors(image, block=self.type)
        request = model_request(model_id, model_package_id, block=self.type)
        # An empty batch loads no model and runs nothing.
        if not tensors:
            return None

        model = self._models.get(request)

        # DocTR has no step options.
        batch, metadata = model.pre_process(tensors, input_color_format="rgb")
        model_input = ModelInput(request, model, batch, metadata, options={})

        return model_input

    @phase
    def raw_output(self, network_input: Optional[ModelInput]) -> Any:
        """Run the model once on all images (``forward``)."""
        if network_input is None:
            return None

        raw_output = network_input.model.forward(network_input.batch)

        return raw_output

    @phase
    def result(
        self, raw_output: Any, network_input: Optional[ModelInput], image: Batch
    ) -> List[Dict[str, Any]]:
        """Decode text and boxes (``post_process``); one result per image."""
        if network_input is None:
            return []

        results = _structured_results(raw_output, network_input, image, block=self.type)

        return results

    def run(
        self,
        *,
        image: Batch,
        model_id: str = DOCTR_MODEL_ID,
        model_package_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Read the text of every image of the batch.

        Args:
            image: The batch of RGB images of one workflow run.
            model_id: Model id for every image.
            model_package_id: Package id for every image, or ``None`` to let
                the loader choose.

        Returns:
            One ``{"result", "predictions", "model_id", "inference_id"}`` per
            image, in batch order.

        Raises:
            PhaseFailure: Phase ``network_input`` when a model cannot be
                resolved or an image is not RGB; phase ``result`` when a model
                returns another number of results than images.
        """
        network_input = self.network_input(image, model_id, model_package_id)
        raw_output = self.raw_output(network_input)
        results = self.result(raw_output, network_input, image)

        return results

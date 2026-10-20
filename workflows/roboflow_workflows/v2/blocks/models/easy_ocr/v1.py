"""Read text with the EasyOCR model of one language from ``models``.

Canonical type: ``roboflow_core/easy_ocr@v1``.

Differences from V1, all deliberate:

- No ``quantize`` and no language codes are sent. V1 sent both only to the
  remote HTTP API; its local model ignored them. ``language`` selects the
  model, which carries its own languages.
- The empty result carries float class ids; the block makes them integer so
  that a text-free image is a valid detection prediction.
"""

from typing import Any, Dict, List, Literal, Mapping, Optional, Union

import torch
from inference_models.models.base.object_detection import Detections
from pydantic import Field, StrictStr, field_validator
from roboflow_workflows.v2.blocks.models._shared.ocr import (
    PREDICTION_TYPE,
    TEXT_KEY,
    _structured_results,
)
from roboflow_workflows.v2.blocks.models.cache import ModelCache
from roboflow_workflows.v2.blocks.models.common import (
    ModelInput,
    ModelOption,
    model_request,
    models_cache,
    one_per_call,
    post_process_options,
    rgb_tensors,
)
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import STRING_KIND
from roboflow_workflows.v2.data.prediction_ops import CLASS_NAME_KEY
from roboflow_workflows.v2.data.predictions import OBJECT_DETECTION_PREDICTION_KIND
from roboflow_workflows.v2.engine.declaration import (
    Block,
    BlockParams,
    DependentResource,
    Output,
    Ref,
)
from roboflow_workflows.v2.engine.phases import phase


EASY_OCR_MODELS: Mapping[str, str] = {
    "English": "easy_ocr/english_g2",
    "Japanese": "easy_ocr/japanese_g2",
    "Kannada": "easy_ocr/kannada_g2",
    "Korean": "easy_ocr/korean_g2",
    "Latin": "easy_ocr/latin_g2",
    "Telugu": "easy_ocr/telugu_g2",
    "Simplified Chinese": "easy_ocr/zh_sim_g2",
}


class EasyOCRBlock(Block):
    """Read text with the EasyOCR model of one language from ``models``.

    ``language`` selects the model id (``EASY_OCR_MODELS``). Phases as in
    ``OCRModelBlock``; the model keeps every region it reads (confidence 0,
    as V1). Each row is named by its own recognised text.
    """

    type = "roboflow_core/easy_ocr@v1"
    metadata = {
        "name": "EasyOCR",
        "port_of": "roboflow_core/easy_ocr@v1",
        "section": "model",
        "search_keywords": ["ocr", "text", "easyocr"],
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
                "Native Detections of text regions, each named by its recognised text."
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
        # noqa F821: flake8 parses the string "always" in an annotation as a name.
        image: Ref(IMAGE_KIND, batch="always") = Field(  # noqa: F821
            description="RGB images to read text from; delivered as one batch."
        )
        language: Literal[
            "English",
            "Japanese",
            "Kannada",
            "Korean",
            "Latin",
            "Telugu",
            "Simplified Chinese",
        ] = Field(
            default="English",
            title="Language",
            description=(
                "Language of the EasyOCR model to use; each language has its own "
                "model id in the 'models' resource (e.g. English -> "
                "easy_ocr/english_g2)."
            ),
            examples=["English", "Japanese"],
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

        @field_validator("model_package_id")
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
        """Report the language's model this step needs from ``models``."""
        resources = [
            DependentResource(
                resource_type="model",
                identifier=EASY_OCR_MODELS[params.language],
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
        self, image: Batch, language: str, model_package_id: Optional[str]
    ) -> Optional[ModelInput]:
        """Resolve the language's model and preprocess all images."""
        one_per_call(language, name="language", block=self.type)
        tensors = rgb_tensors(image, block=self.type)
        request = model_request(
            EASY_OCR_MODELS[language], model_package_id, block=self.type
        )
        # An empty batch loads no model and runs nothing.
        if not tensors:
            return None

        model = self._models.get(request)
        # V1 keeps every region the model reads: confidence 0.
        confidence = ModelOption(
            param="confidence", keyword="confidence", value=None, default=0.0
        )
        options = post_process_options(
            model.post_process, (confidence,), block=self.type, request=request
        )

        batch, metadata = model.pre_process(tensors, input_color_format="rgb")
        model_input = ModelInput(request, model, batch, metadata, options)

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

        results = _structured_results(
            raw_output,
            network_input,
            image,
            block=self.type,
            adjust_detections=_easy_ocr_detections,
        )

        return results

    def run(
        self,
        *,
        image: Batch,
        language: str = "English",
        model_package_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Read the text of every image of the batch.

        Args:
            image: The batch of RGB images of one workflow run.
            language: Language whose model reads every image.
            model_package_id: Package id for every image, or ``None`` to let
                the loader choose.

        Returns:
            One ``{"result", "predictions", "model_id", "inference_id"}`` per
            image, in batch order.

        Raises:
            PhaseFailure: Phase ``network_input`` when the model cannot be
                resolved or an image is not RGB; phase ``result`` when the
                model returns another number of results than images.
        """
        network_input = self.network_input(image, language, model_package_id)
        raw_output = self.raw_output(network_input)
        results = self.result(raw_output, network_input, image)

        return results


def _easy_ocr_detections(detections: Detections) -> Detections:
    """Name each row by its recognised text and give empty class ids an int type."""
    # V1 parity: an EasyOCR row's class is its recognised text.
    for row in detections.bboxes_metadata or ():
        row[CLASS_NAME_KEY] = str(row.get(TEXT_KEY, ""))
    # EasyOCR builds class ids with torch.tensor(list): no text gives float32.
    if detections.class_id.is_floating_point() and not detections.class_id.numel():
        detections.class_id = detections.class_id.to(torch.int64)

    return detections

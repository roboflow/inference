"""Local Moondream2 prompted detection, matching the latest V1 tensor block.

Each RGB image reaches detect([tensor], classes=[prompt]) unchanged. Prompts
and exact model/package requests are checked for the entire batch before any
load; per-image values must carry the image batch's indices (see _local_vlm).
The host owns model lifetime; ModelCache reuses instances across frames.
Caption, query, and point are separate backend capabilities, outside this V1
detection port. Roboflow hosted execution awaits the new server contract.
"""

from dataclasses import replace
from typing import Any, Dict, List, Mapping, Union

import torch
from pydantic import Field
from roboflow_workflows.v2.blocks.models._local_vlm import (
    TextField,
    TextParam,
    TokensField,
    TokensParam,
    _check_backend_call,
    _image_row,
    _model_requests,
    _require_rgb,
    checked_text,
    generation_options,
    per_row,
)
from roboflow_workflows.v2.blocks.models.cache import ModelCache
from roboflow_workflows.v2.blocks.models.common import (
    INFERENCE_ID_OUTPUT,
    MODEL_ID_DESCRIPTION,
    MODEL_ID_OUTPUT,
    ModelIdField,
    ModelSelectionParams,
    RequestField,
    identified_result,
    model_dependency,
    models_cache,
    prediction_metadata,
)
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.predictions import OBJECT_DETECTION_PREDICTION_KIND
from roboflow_workflows.v2.engine.declaration import (
    Block,
    BlockParams,
    DependentResource,
    Output,
    Ref,
)

from inference_models import Detections

DEFAULT_MODEL_ID = "moondream2/moondream2_2b_jul24"


class Moondream2Block(Block):
    """Detect the object named by the prompt in each input image."""

    type = "v2/moondream2@v1"
    metadata = {"name": "Moondream2", "port_of": "roboflow_core/moondream2@v1"}
    outputs = {
        "predictions": Output(
            OBJECT_DETECTION_PREDICTION_KIND,
            source="image",
            description="Native detections in image pixel coordinates.",
        ),
        "model_id": MODEL_ID_OUTPUT,
        "inference_id": INFERENCE_ID_OUTPUT,
    }

    class Params(ModelSelectionParams):
        """Select images, the object prompt, and the local model package."""

        image: Ref(IMAGE_KIND, batch="always") = Field(  # noqa: F821
            description="RGB images; one detection call per image."
        )
        model_id: ModelIdField = Field(
            default=DEFAULT_MODEL_ID,
            description=MODEL_ID_DESCRIPTION,
            examples=[DEFAULT_MODEL_ID, "$inputs.model_id"],
        )
        prompt: TextParam = Field(
            default=None,
            description="Object name. Omitted, None or empty becomes an empty class name, as V1.",
            examples=["person", "$inputs.prompt"],
        )
        max_new_tokens: TokensParam = Field(
            default=None,
            ge=1,
            description="Positive generation limit; omitted or None leaves detect()'s backend default.",
            examples=[256, "$inputs.tokens"],
        )

    def __init__(self, *, models: Union[Mapping[Any, Any], ModelCache]):
        """Bind the host-owned models resource.

        Args:
            models: Loaded model mapping or shared ModelCache; nothing loads here.
        """
        self._models = models_cache(models, block=self.type)

    @classmethod
    def discover_dependent_resources(
        cls, params: BlockParams
    ) -> List[DependentResource]:
        """Describe the local model request.

        Args:
            params: Validated step parameters.

        Returns:
            Model resource declaration, including any selected package.
        """
        resources = model_dependency(params, task_type="object-detection")

        return resources

    def run(
        self,
        *,
        image: Batch,
        model_id: RequestField = DEFAULT_MODEL_ID,
        model_package_id: RequestField = None,
        prompt: TextField = None,
        max_new_tokens: TokensField = None,
    ) -> List[Dict[str, Any]]:
        """Detect one prompted class per image, preserving input order.

        Args:
            image: RGB images as a Batch.
            model_id: Model id, or a Batch with one id per image.
            model_package_id: Exact package, per-image packages, or None.
            prompt: Object name, per-image names, or None for an empty name.
            max_new_tokens: Positive limit or None to use the backend default.

        Returns:
            Native predictions, model id and fresh inference id for every image.

        Raises:
            ValueError: On invalid resolved parameters (naming the image's
                logical index), a per-image Batch whose indices differ from the
                image batch's, a backend whose detect() cannot take the call, or
                a bad output type or cardinality.
            LookupError: When the host cannot serve the exact model request.
        """
        _require_rgb(image, block=self.type)
        requests = _model_requests(
            image,
            model_id=model_id,
            model_package_id=model_package_id,
            block=self.type,
        )
        prompts = []
        for index, value in zip(
            image.indices, per_row(prompt, image=image, name="prompt")
        ):
            with _image_row(index, block=self.type):
                prompts.append(checked_text(value, name="prompt") or "")
        options = generation_options(max_new_tokens)
        models = {
            request: self._models.get(request) for request in dict.fromkeys(requests)
        }
        # Bind every backend signature before the first model invocation.
        for request, model in models.items():
            _check_backend_call(
                model,
                "detect",
                block=self.type,
                request=request,
                options=options,
                classes=[""],
                input_color_format="rgb",
            )

        results = []
        for member, request, text in zip(image, requests, prompts):
            predictions = models[request].detect(
                [member.tensor_image],
                classes=[text],
                input_color_format="rgb",
                **options,
            )
            if not isinstance(predictions, list) or len(predictions) != 1:
                raise ValueError(
                    f"{self.type}: model {str(request)!r} must return a list of one detection result"
                )
            prediction = predictions[0]
            # The backend emits shape (0,) when no objects match the prompt.
            if (
                isinstance(prediction, Detections)
                and isinstance(prediction.xyxy, torch.Tensor)
                and prediction.xyxy.shape == (0,)
            ):
                prediction = replace(prediction, xyxy=prediction.xyxy.reshape(0, 4))

            OBJECT_DETECTION_PREDICTION_KIND.check(prediction)
            # A model may reuse its carrier across frames. Do not overwrite it.
            prediction = replace(
                prediction,
                image_metadata=prediction_metadata(
                    prediction.image_metadata,
                    member,
                    class_names=[text],
                    prediction_type="object-detection",
                ),
            )
            results.append(identified_result(prediction, model_id=request.model_id))

        return results


BLOCKS = [Moondream2Block]

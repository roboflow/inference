"""Local Moondream2 prompted detection, matching the latest V1 tensor block.

Each RGB image reaches detect([tensor], classes=[prompt]) unchanged. One exact
model/package request serves every image of a call; prompts may differ per
image and are checked for the entire batch before any load. Caption, query and
point are separate backend capabilities, outside this V1 detection port.
Roboflow hosted execution awaits the new server contract.
"""

from dataclasses import replace
from typing import Any, Dict, List, Mapping, Optional, Union

import torch
from pydantic import Field, StrictInt, StrictStr, field_validator
from roboflow_workflows.v2.blocks.models._local_vlm import (
    check_backend_call,
    checked_text,
    generation_options,
    image_row,
    per_row,
    require_rgb_images,
)
from roboflow_workflows.v2.blocks.models.cache import ModelCache
from roboflow_workflows.v2.blocks.models.common import (
    identified_result,
    model_request,
    models_cache,
    prediction_metadata,
)
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import INTEGER_KIND, STRING_KIND
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

    type = "roboflow_core/moondream2@v1"
    metadata = {"name": "Moondream2", "port_of": "roboflow_core/moondream2@v1"}
    outputs = {
        "predictions": Output(
            OBJECT_DETECTION_PREDICTION_KIND,
            source="image",
            description="Native detections in image pixel coordinates.",
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
        """Select images, the object prompt, and the local model package."""

        model_id: StrictStr | Ref(STRING_KIND) = Field(
            default=DEFAULT_MODEL_ID,
            description=(
                "Model id (or local path) of the model in the 'models' resource, "
                "one for every image of the step; a selector may choose it per "
                "run, not per image. To serve images with different models, "
                "route them to one step per model."
            ),
            examples=[DEFAULT_MODEL_ID, "$inputs.model_id"],
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
            description="RGB images; one detection call per image."
        )
        prompt: Optional[StrictStr] | Ref(
            STRING_KIND, batch="if_varying", allow_none=True
        ) = Field(
            default=None,
            description="Object name. Omitted, None or empty becomes an empty class name, as V1.",
            examples=["person", "$inputs.prompt"],
        )
        max_new_tokens: Optional[StrictInt] | Ref(INTEGER_KIND, allow_none=True) = (
            Field(
                default=None,
                ge=1,
                description="Positive generation limit; omitted or None leaves detect()'s backend default.",
                examples=[256, "$inputs.tokens"],
            )
        )

        @field_validator("model_id", "model_package_id")
        @classmethod
        def _reject_empty(cls, value: Optional[str]) -> Optional[str]:
            if value == "":
                raise ValueError("must be a non-empty string")

            return value

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
        resources = [
            DependentResource(
                resource_type="model",
                identifier=params.model_id,
                details={
                    "resource": "models",
                    "task_type": "object-detection",
                    "model_package_id": params.model_package_id,
                },
            )
        ]

        return resources

    def run(
        self,
        *,
        image: Batch,
        model_id: str = DEFAULT_MODEL_ID,
        model_package_id: Optional[str] = None,
        prompt: Union[None, str, Batch] = None,
        max_new_tokens: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Detect one prompted class per image, preserving input order.

        Args:
            image: RGB images as a Batch.
            model_id: Model id for every image.
            model_package_id: Exact package for every image, or None.
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
        require_rgb_images(image, block=self.type)
        request = model_request(model_id, model_package_id, block=self.type)
        prompts = []
        for index, value in zip(
            image.indices, per_row(prompt, image=image, name="prompt")
        ):
            with image_row(index, block=self.type):
                prompts.append(checked_text(value, name="prompt") or "")
        options = generation_options(max_new_tokens)
        # An empty batch loads no model.
        if not len(image):
            return []

        model = self._models.get(request)
        # Bind the backend signature before the first model invocation.
        check_backend_call(
            model,
            "detect",
            block=self.type,
            request=request,
            options=options,
            classes=[""],
            input_color_format="rgb",
        )

        results = []
        for member, text in zip(image, prompts):
            predictions = model.detect(
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

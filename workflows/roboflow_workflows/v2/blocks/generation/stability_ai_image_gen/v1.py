"""Generate an image from a prompt, or a variation of an input image.

Canonical type: ``roboflow_core/stability_ai_image_gen@v1``.
"""

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import Field, StrictStr
from roboflow_workflows.execution_engine.entities.workload import WorkOperation
from roboflow_workflows.v2.blocks.generation._shared.stability import (
    DEFAULT_CLIENT,
    GENERATION_ENDPOINTS,
    ApiKey,
    StabilityAIClient,
    UnitFloat,
    _check_range,
    _jpeg,
    _new_image_id,
    _request_image,
    _service_dependency,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import FLOAT_KIND, STRING_KIND
from roboflow_workflows.v2.data.secrets import Secret
from roboflow_workflows.v2.engine.declaration import (
    Block,
    BlockParams,
    DependentResource,
    Output,
    Ref,
)


class StabilityAIImageGenerationBlock(Block):
    """Generate an image from a prompt, or a variation of an input image.

    Example: ``prompt="a red bicycle in the snow"`` with ``model="core"``
    returns one new image. Adding ``image`` and ``strength=0.3`` keeps most of
    that image's look.
    """

    type = "roboflow_core/stability_ai_image_gen@v1"
    metadata = {
        "name": "Stability AI Image Generation",
        "port_of": "roboflow_core/stability_ai_image_gen@v1",
        "section": "model",
        "provider": "stability_ai",
    }
    outputs = {
        "image": Output(
            IMAGE_KIND,
            description="Generated RGB image; a new root, unrelated in "
            "geometry to any input.",
        ),
    }

    class Params(BlockParams):
        prompt: StrictStr | Ref(STRING_KIND) = Field(
            description="What the image should show.",
            examples=["a red bicycle in the snow", "$inputs.prompt"],
            json_schema_extra={"multiline": True},
        )
        negative_prompt: Optional[StrictStr] | Ref(STRING_KIND, allow_none=True) = (
            Field(
                default=None,
                description="What the image should not show; not sent when unset.",
                examples=["blurry, text", "$inputs.negative_prompt"],
            )
        )
        model: Literal["ultra", "core", "sd3"] | Ref(STRING_KIND) = Field(
            default="core",
            description="Stability AI model: 'core', 'ultra' or 'sd3'.",
            examples=["core", "$inputs.model"],
        )
        image: Optional[Ref(IMAGE_KIND)] = Field(
            default=None,
            description="Optional starting image for a variation.",
            examples=["$inputs.image"],
        )
        strength: UnitFloat | Ref(FLOAT_KIND) = Field(
            default=0.3,
            description="Influence of image, in [0, 1]: 0 keeps it, 1 ignores "
            "it. Sent only with an image.",
            examples=[0.3, "$inputs.strength"],
        )
        api_key: ApiKey = Field(
            description="Stability AI API key, as a secret or a string.",
            examples=["$steps.secrets.stability_ai_api_key", "$inputs.api_key"],
        )

    def __init__(
        self, *, stability_ai_client: StabilityAIClient = DEFAULT_CLIENT
    ) -> None:
        self._client = stability_ai_client

    @classmethod
    def discover_dependent_resources(
        cls, params: BlockParams
    ) -> List[DependentResource]:
        """Report the Stability AI model the step calls."""
        resources = [_service_dependency(params.model)]

        return resources

    @classmethod
    def discover_work_operations(cls, params: BlockParams) -> List[WorkOperation]:
        """Report the remote inference and image encoding of every call."""
        operations = [
            WorkOperation.MODEL_INFERENCE,
            WorkOperation.EXTERNAL_REQUEST,
            WorkOperation.IMAGE_ENCODING,
        ]

        return operations

    @classmethod
    def discover_restrictions(cls, params: BlockParams) -> List[Any]:
        """Declare that the step has no runtime restrictions."""
        return []

    def run(
        self,
        *,
        prompt: str,
        negative_prompt: Optional[str],
        model: str,
        image: Optional[ImageData],
        strength: float,
        api_key: Union[str, Secret],
    ) -> Dict[str, Any]:
        """Request one generated image.

        Args:
            prompt: What the image should show.
            negative_prompt: What it should not show, or ``None``.
            model: ``core``, ``ultra`` or ``sd3``.
            image: Starting image for a variation, or ``None``.
            strength: Influence of ``image``, in [0, 1].
            api_key: Stability AI key.

        Returns:
            ``image``: the generated image, on the host.

        Raises:
            ValueError: On an unknown model or a strength outside [0, 1].
            StabilityAIError: When the request fails or returns no image.
            SecretUnavailableError: When the key is a replayed secret.
        """
        if model not in GENERATION_ENDPOINTS:
            raise ValueError(
                f"{self.type}: model must be one of {list(GENERATION_ENDPOINTS)}, "
                f"got {model!r}"
            )
        _check_range(strength, name="strength", low=0.0, high=1.0, block=self.type)

        data: Dict[str, Any] = {"prompt": prompt, "output_format": "jpeg"}
        # The service needs multipart even without an image, as V1 sends.
        files: Dict[str, Any] = {"none": ""}
        if image is not None:
            files = {"image": _jpeg(image.tensor_image)}
            data["strength"] = strength
        if negative_prompt is not None:
            data["negative_prompt"] = negative_prompt

        pixels = _request_image(
            self._client,
            GENERATION_ENDPOINTS[model],
            api_key=api_key,
            data=data,
            files=files,
            block=self.type,
        )
        generated = ImageData.from_tensor(pixels, image_id=_new_image_id())

        return {"image": generated}

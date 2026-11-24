"""Extend an image beyond its borders.

Canonical type: ``roboflow_core/stability_ai_outpainting@v1``.
"""

from typing import Annotated, Any, Dict, List, Optional, Union
from pydantic import Field, StrictInt, StrictStr
from roboflow_workflows.execution_engine.entities.workload import WorkOperation
from roboflow_workflows.v2.blocks.generation._shared.stability import (
    DEFAULT_CLIENT,
    MAX_SEED,
    OUTPAINTING_ENDPOINT,
    PRESETS,
    ApiKey,
    Preset,
    Seed,
    StabilityAIClient,
    UnitFloat,
    _check_range,
    _check_seed,
    _jpeg,
    _request_image,
    _service_dependency,
    edited_image,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import FLOAT_KIND, INTEGER_KIND, STRING_KIND
from roboflow_workflows.v2.data.secrets import Secret
from roboflow_workflows.v2.engine.declaration import (
    Block,
    BlockParams,
    DependentResource,
    Output,
    Ref,
)


MAX_SIDE = 2000

Side = Annotated[StrictInt, Field(ge=0, le=MAX_SIDE)]


class StabilityAIOutpaintingBlock(Block):
    """Extend an image beyond its borders.

    Example: ``left=200, right=200`` on a 640x480 image asks for an 1040x480
    image whose middle is the original. Without any side the input is
    returned unchanged and nothing is sent.
    """

    type = "roboflow_core/stability_ai_outpainting@v1"
    metadata = {
        "name": "Stability AI Outpainting",
        "port_of": "roboflow_core/stability_ai_outpainting@v1",
        "section": "model",
        "provider": "stability_ai",
    }
    outputs = {
        "image": Output(
            IMAGE_KIND,
            description="Outpainted RGB image; its parent mapping places the "
            "input at (left, up).",
        ),
    }

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(
            description="Image to outpaint.",
            examples=["$inputs.image", "$steps.cropping.crops"],
        )
        creativity: UnitFloat | Ref(FLOAT_KIND) = Field(
            default=0.5,
            description="How freely new content is invented, in [0, 1].",
            examples=[0.5, "$inputs.creativity"],
        )
        left: Optional[Side] | Ref(INTEGER_KIND, allow_none=True) = Field(
            default=None,
            description=f"Pixels to add on the left, at most {MAX_SIDE}.",
            examples=[200, "$inputs.left"],
        )
        right: Optional[Side] | Ref(INTEGER_KIND, allow_none=True) = Field(
            default=None,
            description=f"Pixels to add on the right, at most {MAX_SIDE}.",
            examples=[200, "$inputs.right"],
        )
        up: Optional[Side] | Ref(INTEGER_KIND, allow_none=True) = Field(
            default=None,
            description=f"Pixels to add at the top, at most {MAX_SIDE}.",
            examples=[200, "$inputs.up"],
        )
        down: Optional[Side] | Ref(INTEGER_KIND, allow_none=True) = Field(
            default=None,
            description=f"Pixels to add at the bottom, at most {MAX_SIDE}.",
            examples=[200, "$inputs.down"],
        )
        prompt: Optional[StrictStr] | Ref(STRING_KIND, allow_none=True) = Field(
            default=None,
            description="Optional description of the new content; not sent when "
            "empty.",
            examples=["a sandy beach", "$inputs.prompt"],
            json_schema_extra={"multiline": True},
        )
        preset: Optional[Preset] = Field(
            default=None,
            description=f"Optional style preset, one of: {', '.join(PRESETS)}.",
            examples=["photographic"],
        )
        seed: Optional[Seed] | Ref(INTEGER_KIND, allow_none=True) = Field(
            default=None,
            description=f"Seed in [0, {MAX_SEED}]; unset or 0 lets the service "
            "choose one.",
            examples=[200, "$inputs.seed"],
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
        """Report the Stability AI outpainting service."""
        resources = [_service_dependency("outpaint")]

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
        image: ImageData,
        creativity: float,
        left: Optional[int],
        right: Optional[int],
        up: Optional[int],
        down: Optional[int],
        prompt: Optional[str],
        preset: Optional[str],
        seed: Optional[int],
        api_key: Union[str, Secret],
    ) -> Dict[str, Any]:
        """Request the outpainted image.

        Args:
            image: Image to extend; not modified.
            creativity: Freedom of the new content, in [0, 1].
            left: Pixels to add on the left, or ``None``.
            right: Pixels to add on the right, or ``None``.
            up: Pixels to add at the top, or ``None``.
            down: Pixels to add at the bottom, or ``None``.
            prompt: Description of the new content, or ``None``.
            preset: Style preset or ``None``.
            seed: Seed or ``None``.
            api_key: Stability AI key.

        Returns:
            ``image``: the outpainted image on ``image``'s device, or ``image``
            itself when no side is set.

        Raises:
            ValueError: On a value out of range.
            StabilityAIError: When the request fails or returns no image.
            SecretUnavailableError: When the key is a replayed secret.
        """
        sides = {"left": left, "right": right, "up": up, "down": down}
        for name, value in sides.items():
            if value is not None:
                _check_range(value, name=name, low=0, high=MAX_SIDE, block=self.type)
        _check_range(creativity, name="creativity", low=0.0, high=1.0, block=self.type)
        _check_seed(seed, block=self.type)
        if not any(sides.values()):
            return {"image": image}

        data: Dict[str, Any] = {"output_format": "jpeg", "creativity": creativity}
        data.update({name: value for name, value in sides.items() if value})
        if preset is not None:
            data["preset"] = preset
        if prompt:
            data["prompt"] = prompt
        if seed:
            data["seed"] = seed

        pixels = _request_image(
            self._client,
            OUTPAINTING_ENDPOINT,
            api_key=api_key,
            data=data,
            files={"image": _jpeg(image.tensor_image)},
            block=self.type,
        )
        padding = (left or 0, up or 0, right or 0, down or 0)
        outpainted = edited_image(image, pixels.to(image.device), padding=padding)

        return {"image": outpainted}

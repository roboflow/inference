"""Repaint the masked objects of an image from a prompt.

Canonical type: ``roboflow_core/stability_ai_inpainting@v1``.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import Field, StrictBool, StrictStr
from roboflow_workflows.execution_engine.entities.workload import WorkOperation
from roboflow_workflows.v2.blocks.generation._shared.stability import (
    DEFAULT_CLIENT,
    INPAINTING_ENDPOINT,
    MAX_SEED,
    PRESETS,
    ApiKey,
    Preset,
    Seed,
    StabilityAIClient,
    _check_seed,
    _jpeg,
    _jpeg_host,
    _request_image,
    _service_dependency,
    edited_image,
    inpainting_mask,
)
from roboflow_workflows.v2.blocks.visualization.annotations.rows import MASK_KINDS
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import BOOLEAN_KIND, INTEGER_KIND, STRING_KIND
from roboflow_workflows.v2.data.secrets import Secret
from roboflow_workflows.v2.engine.declaration import (
    Block,
    BlockParams,
    DependentResource,
    Output,
    Ref,
)


class StabilityAIInpaintingBlock(Block):
    """Repaint the masked objects of an image from a prompt.

    Example: masks of every car from a segmentation model and
    ``prompt="a horse-drawn carriage"`` replace the cars. With
    ``invert_segmentation_mask=True`` the background is repainted instead.
    """

    type = "roboflow_core/stability_ai_inpainting@v1"
    metadata = {
        "name": "Stability AI Inpainting",
        "port_of": "roboflow_core/stability_ai_inpainting@v1",
        "section": "model",
        "provider": "stability_ai",
    }
    outputs = {
        "image": Output(
            IMAGE_KIND,
            description="Inpainted RGB image. At the input's size it keeps the "
            "input's identity and mappings; at another size it is a new child "
            "image whose parent mapping rescales to the input.",
        ),
    }

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(
            description="Image to inpaint.",
            examples=["$inputs.image", "$steps.cropping.crops"],
        )
        segmentation_mask: Ref(*MASK_KINDS) = Field(
            description="Instance segmentation of image (dense, RLE or a mask "
            "view). The union of all masks is repainted.",
            examples=["$steps.model.predictions"],
        )
        prompt: StrictStr | Ref(STRING_KIND) = Field(
            description="What the masked area should show.",
            examples=["a horse-drawn carriage", "$inputs.prompt"],
            json_schema_extra={"multiline": True},
        )
        negative_prompt: Optional[StrictStr] | Ref(STRING_KIND, allow_none=True) = (
            Field(
                default=None,
                description="What the masked area should not show; not sent when "
                "unset.",
                examples=["blurry", "$inputs.negative_prompt"],
            )
        )
        invert_segmentation_mask: StrictBool | Ref(BOOLEAN_KIND) = Field(
            default=False,
            description="Repaint everything outside the masks instead.",
            examples=[False, "$inputs.invert"],
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
        """Report the Stability AI inpainting service."""
        resources = [_service_dependency("inpaint")]

        return resources

    @classmethod
    def discover_work_operations(cls, params: BlockParams) -> List[WorkOperation]:
        """Report the remote inference, mask composition and encoding."""
        operations = [
            WorkOperation.MODEL_INFERENCE,
            WorkOperation.EXTERNAL_REQUEST,
            WorkOperation.IMAGE_ENCODING,
            WorkOperation.IMAGE_COMPOSITION,
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
        segmentation_mask: Any,
        prompt: str,
        negative_prompt: Optional[str],
        invert_segmentation_mask: bool,
        preset: Optional[str],
        seed: Optional[int],
        api_key: Union[str, Secret],
    ) -> Dict[str, Any]:
        """Request the inpainted image.

        Args:
            image: Image to inpaint; not modified.
            segmentation_mask: Instance masks of ``image``.
            prompt: What the masked area should show.
            negative_prompt: What it should not show, or ``None``.
            invert_segmentation_mask: Repaint outside the masks instead.
            preset: Style preset or ``None``.
            seed: Seed or ``None``.
            api_key: Stability AI key.

        Returns:
            ``image``: the inpainted image on ``image``'s device.

        Raises:
            ValueError: On a seed out of range, or masks that do not belong to
                ``image``.
            StabilityAIError: When the request fails or returns no image.
            SecretUnavailableError: When the key is a replayed secret.
        """
        _check_seed(seed, block=self.type)
        mask = inpainting_mask(
            segmentation_mask, image, invert=invert_segmentation_mask, block=self.type
        )

        data: Dict[str, Any] = {"prompt": prompt, "output_format": "jpeg"}
        if negative_prompt is not None:
            data["negative_prompt"] = negative_prompt
        if preset is not None:
            data["preset"] = preset
        if seed:
            data["seed"] = seed
        files = {"image": _jpeg(image.tensor_image), "mask": _jpeg_host(mask)}

        pixels = _request_image(
            self._client,
            INPAINTING_ENDPOINT,
            api_key=api_key,
            data=data,
            files=files,
            block=self.type,
        )
        inpainted = edited_image(image, pixels.to(image.device), padding=(0, 0, 0, 0))

        return {"image": inpainted}

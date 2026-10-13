"""Stability AI image generation, inpainting and outpainting.

Ports of V1 ``roboflow_core/stability_ai_image_gen@v1``,
``roboflow_core/stability_ai_inpainting@v1`` (latest: its tensor variant) and
``roboflow_core/stability_ai_outpainting@v1``. Each call is one multipart POST
through the ``stability_ai_client`` resource::

    block        endpoint                              result image
    image_gen    /v2beta/stable-image/generate/MODEL   new root "stability-..."
    inpainting   /v2beta/stable-image/edit/inpaint     the input's identity at its size;
                                                       another size: child of the input,
                                                       rescaled to it
    outpainting  /v2beta/stable-image/edit/outpaint    child of the input:
                                                       frame_xy = local_xy - (left, up)

Every request asks for JPEG and sends images as JPEG, as V1 does. The
inpainting mask is the union of all instance masks in the image's frame,
white where the service repaints, blurred with V1's 15x15 Gaussian kernel.
Encoding, the mask blur and response decoding run on the host; edited images
return to the input image's device.

``StabilityAIClient`` is the real transport and the resource default. Its
``base_url`` lets a host or a test point it at another server, e.g. a loopback
fake. The API key is a ``secret`` or a plain string; it is revealed only for
the request, and failure messages are scrubbed of it.

Differences from V1:

* Out-of-range ``strength``, ``creativity``, ``seed`` and side lengths fail
  instead of being clamped; an unknown ``model`` fails instead of falling back
  to ``core``.
* ``preset`` is sent. V1 compared the enum member with strings and never sent
  it. Inpainting sends ``negative_prompt``, which V1 accepted but dropped.
* A failed request raises ``StabilityAIError`` with the status and the
  service's message, scrubbed of the key. V1 raised on a non-JSON error body.
* Requests time out after ``StabilityAIClient.timeout`` seconds (default
  120); V1 waited indefinitely.
* An edited image of another size than requested is a new child image whose
  parent mapping rescales to the input, so coordinates stay exact. V1 kept
  the input's metadata unchanged.
* The inpainting mask accepts dense, RLE and view instance segmentations
  placed in the image's frame (``visualization.annotations.rows``).
"""

import uuid
from typing import Annotated, Any, Dict, List, Literal, Mapping, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from pydantic import Field, StrictBool, StrictInt, StrictStr
from roboflow_workflows.execution_engine.entities.workload import WorkOperation
from roboflow_workflows.v2.blocks.visualization.annotations.rows import (
    MASK_KINDS,
    dense_masks_of,
    placed_in,
)
from roboflow_workflows.v2.data.image import FrameMapping, ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import (
    BOOLEAN_KIND,
    FLOAT_KIND,
    INTEGER_KIND,
    STRING_KIND,
)
from roboflow_workflows.v2.data.secrets import (
    SECRET_KIND,
    Secret,
    reveal_secrets,
    to_secret,
)
from roboflow_workflows.v2.engine.declaration import (
    Block,
    BlockParams,
    DependentResource,
    Output,
    Ref,
)

__all__ = [
    "API_HOST",
    "GENERATION_ENDPOINTS",
    "INPAINTING_ENDPOINT",
    "OUTPAINTING_ENDPOINT",
    "PRESETS",
    "StabilityAIClient",
    "StabilityAIError",
    "StabilityAIImageGenerationBlock",
    "StabilityAIInpaintingBlock",
    "StabilityAIOutpaintingBlock",
    "edited_image",
    "inpainting_mask",
]

API_HOST = "https://api.stability.ai"
GENERATION_ENDPOINTS = {
    "ultra": "/v2beta/stable-image/generate/ultra",
    "core": "/v2beta/stable-image/generate/core",
    "sd3": "/v2beta/stable-image/generate/sd3",
}
INPAINTING_ENDPOINT = "/v2beta/stable-image/edit/inpaint"
OUTPAINTING_ENDPOINT = "/v2beta/stable-image/edit/outpaint"
PRESETS = (
    "3d-model",
    "analog-film",
    "anime",
    "cinematic",
    "comic-book",
    "digital-art",
    "enhance",
    "fantasy-art",
    "isometric",
    "line-art",
    "low-poly",
    "modeling-compound",
    "neon-punk",
    "origami",
    "photographic",
    "pixel-art",
    "tile-texture",
)
MAX_SEED = 4294967294
MAX_SIDE = 2000
MASK_BLUR_KERNEL = (15, 15)
DEFAULT_TIMEOUT_SECONDS = 120.0
ERROR_DETAIL_LIMIT = 500
PROVIDER = "stability_ai"
SECTION_METADATA = {"section": "model", "provider": PROVIDER}

Model = Literal["ultra", "core", "sd3"]
Preset = Literal[
    "3d-model",
    "analog-film",
    "anime",
    "cinematic",
    "comic-book",
    "digital-art",
    "enhance",
    "fantasy-art",
    "isometric",
    "line-art",
    "low-poly",
    "modeling-compound",
    "neon-punk",
    "origami",
    "photographic",
    "pixel-art",
    "tile-texture",
]
UnitFloat = Annotated[float, Field(ge=0.0, le=1.0)]
Seed = Annotated[StrictInt, Field(ge=0, le=MAX_SEED)]
Side = Annotated[StrictInt, Field(ge=0, le=MAX_SIDE)]
ApiKey = Union[StrictStr, Ref(SECRET_KIND, STRING_KIND)]


class StabilityAIError(RuntimeError):
    """A Stability AI request failed; the message never holds the API key."""


class StabilityAIClient:
    """Send multipart requests to the Stability AI REST API.

    Stateless and safe to share between steps, sessions and threads; each call
    makes its own ``requests.post``. Environment proxies apply, as in V1.

    Args:
        base_url: Scheme, host and optional port, without a trailing path,
            e.g. ``"http://127.0.0.1:8123"`` for a local fake.
        timeout: Seconds to wait for the connection and for the response.

    Raises:
        ValueError: When ``base_url`` is not an HTTP(S) URL or ``timeout`` is
            not positive.
    """

    def __init__(
        self, *, base_url: str = API_HOST, timeout: float = DEFAULT_TIMEOUT_SECONDS
    ) -> None:
        if not isinstance(base_url, str) or not base_url.startswith(
            ("http://", "https://")
        ):
            raise ValueError(f"base_url must be an HTTP(S) URL, got {base_url!r}")
        if not timeout > 0:
            raise ValueError(f"timeout must be positive, got {timeout!r}")

        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def post(
        self,
        path: str,
        *,
        api_key: str,
        data: Mapping[str, Any],
        files: Mapping[str, Any],
    ) -> bytes:
        """POST one request and return the image bytes of a 200 response.

        Args:
            path: Endpoint path, e.g. ``INPAINTING_ENDPOINT``.
            api_key: Plaintext key for the ``authorization`` header.
            data: Form fields.
            files: Multipart file fields, name to bytes.

        Returns:
            The response body.

        Raises:
            StabilityAIError: On any status other than 200.
            requests.RequestException: On connection failures and timeouts.
        """
        import requests

        response = requests.post(
            f"{self.base_url}{path}",
            headers={"authorization": f"Bearer {api_key}", "accept": "image/*"},
            files=dict(files),
            data=dict(data),
            timeout=self.timeout,
        )
        if response.status_code != 200:
            raise StabilityAIError(
                f"HTTP {response.status_code} from {path}: {_error_detail(response)}"
            )

        return response.content


DEFAULT_CLIENT = StabilityAIClient()


class StabilityAIImageGenerationBlock(Block):
    """Generate an image from a prompt, or a variation of an input image.

    Example: ``prompt="a red bicycle in the snow"`` with ``model="core"``
    returns one new image. Adding ``image`` and ``strength=0.3`` keeps most of
    that image's look.
    """

    type = "v2/stability_ai_image_gen@v1"
    metadata = {
        "name": "Stability AI Image Generation",
        "port_of": "roboflow_core/stability_ai_image_gen@v1",
        **SECTION_METADATA,
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
        model: Model | Ref(STRING_KIND) = Field(
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


class StabilityAIInpaintingBlock(Block):
    """Repaint the masked objects of an image from a prompt.

    Example: masks of every car from a segmentation model and
    ``prompt="a horse-drawn carriage"`` replace the cars. With
    ``invert_segmentation_mask=True`` the background is repainted instead.
    """

    type = "v2/stability_ai_inpainting@v1"
    metadata = {
        "name": "Stability AI Inpainting",
        "port_of": "roboflow_core/stability_ai_inpainting@v1",
        **SECTION_METADATA,
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


class StabilityAIOutpaintingBlock(Block):
    """Extend an image beyond its borders.

    Example: ``left=200, right=200`` on a 640x480 image asks for an 1040x480
    image whose middle is the original. Without any side the input is
    returned unchanged and nothing is sent.
    """

    type = "v2/stability_ai_outpainting@v1"
    metadata = {
        "name": "Stability AI Outpainting",
        "port_of": "roboflow_core/stability_ai_outpainting@v1",
        **SECTION_METADATA,
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


def inpainting_mask(
    predictions: Any, image: ImageData, *, invert: bool, block: str
) -> np.ndarray:
    """Build the inpainting mask V1 sends: blurred white masks on black.

    Args:
        predictions: Instance masks of ``image`` (dense, RLE or a view).
        image: The image to inpaint.
        invert: Swap black and white after blurring.
        block: Block type for error messages.

    Returns:
        ``(height, width, 3)`` ``uint8`` host array in ``image``'s size;
        255 is fully repainted.

    Raises:
        ValueError: When the masks belong to another image or have another
            size than ``image``.
    """
    dense = dense_masks_of(predictions, quality="accurate", block=block)
    dense = placed_in(dense, image, block=block)
    masks = dense.mask
    if len(dense) == 0:
        union = torch.zeros(image.size_hw, dtype=torch.bool)
    elif tuple(masks.shape[1:]) != tuple(image.size_hw):
        raise ValueError(
            f"{block}: masks of (h, w)={tuple(masks.shape[1:])} do not match the "
            f"image of (h, w)={image.size_hw}; segment the image being inpainted"
        )
    else:
        union = masks.bool().any(dim=0)

    white = union.to(device="cpu", dtype=torch.uint8).mul_(255).numpy()
    blurred = cv2.GaussianBlur(
        np.repeat(white[:, :, None], 3, axis=2), MASK_BLUR_KERNEL, 0
    )
    if invert:
        blurred = cv2.bitwise_not(blurred)

    return blurred


def edited_image(
    image: ImageData,
    pixels: torch.Tensor,
    *,
    padding: Tuple[int, int, int, int],
) -> ImageData:
    """Wrap a service result as an edit of ``image``.

    The requested canvas is ``image`` grown by ``padding``; a result of that
    size maps back with scale 1, another size with the ratio between the two.

    Args:
        image: The image that was sent.
        pixels: ``(3, height, width)`` ``uint8`` result.
        padding: Pixels added on the ``(left, up, right, down)`` sides.

    Returns:
        ``image.with_pixels(pixels)`` when nothing was added and the size is
        kept; otherwise a child of ``image`` with a new identity.
    """
    left, up, right, down = padding
    requested_hw = (image.height + up + down, image.width + left + right)
    result_hw = (int(pixels.shape[1]), int(pixels.shape[2]))
    if padding == (0, 0, 0, 0) and result_hw == image.size_hw:
        updated = image.with_pixels(pixels)
        return updated

    to_parent = FrameMapping(
        frame_id=image.image_id,
        frame_size_hw=image.size_hw,
        scale_xy=(requested_hw[1] / result_hw[1], requested_hw[0] / result_hw[0]),
        offset_xy=(-left, -up),
    )
    edited = ImageData(
        tensor_image=pixels,
        image_id=_new_image_id(),
        parent=to_parent,
        root=to_parent.then(image.root),
        video_metadata=image.video_metadata,
        composite_sources=image.composite_sources,
        parent_row=image.parent_row,
    )

    return edited


def _request_image(
    client: Any,
    path: str,
    *,
    api_key: Union[str, Secret],
    data: Dict[str, Any],
    files: Dict[str, Any],
    block: str,
) -> torch.Tensor:
    # The one place the key is plaintext. A failure is scrubbed of it and
    # raised outside the handler, so neither __cause__ nor __context__ keeps
    # the original exception, whose text may hold the key.
    revealed = reveal_secrets(to_secret(api_key, name="api_key"))
    failure = None
    try:
        payload = client.post(path, api_key=revealed.value, data=data, files=files)
    except Exception as error:
        failure = revealed.scrub(f"{type(error).__name__}: {error}")
    if failure is not None:
        raise StabilityAIError(f"{block}: Stability AI request failed: {failure}")

    pixels = _decode_rgb(payload, block=block)

    return pixels


def _decode_rgb(payload: Any, *, block: str) -> torch.Tensor:
    if not isinstance(payload, (bytes, bytearray)) or not payload:
        raise StabilityAIError(f"{block}: Stability AI returned no image bytes")

    bgr = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise StabilityAIError(f"{block}: Stability AI response is not an image")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pixels = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()

    return pixels


def _jpeg(pixels: torch.Tensor) -> bytes:
    # Host boundary: the request body is JPEG bytes, as in V1.
    hwc = pixels.detach().permute(1, 2, 0).cpu().numpy()
    host = hwc[:, :, 0] if hwc.shape[2] == 1 else cv2.cvtColor(hwc, cv2.COLOR_RGB2BGR)
    encoded = _jpeg_host(host)

    return encoded


def _jpeg_host(array: np.ndarray) -> bytes:
    succeeded, buffer = cv2.imencode(".jpg", array)
    if not succeeded:
        raise ValueError(f"cannot encode an array of shape {array.shape} as JPEG")

    encoded = buffer.tobytes()

    return encoded


def _error_detail(response: Any) -> str:
    try:
        detail = str(response.json())
    except ValueError:
        detail = response.text

    shortened = detail[:ERROR_DETAIL_LIMIT]

    return shortened


def _check_range(value: Any, *, name: str, low: float, high: float, block: str) -> None:
    number = isinstance(value, (int, float)) and not isinstance(value, bool)
    if not number or not low <= value <= high:
        raise ValueError(f"{block}: {name} must be in [{low}, {high}], got {value!r}")


def _check_seed(seed: Optional[int], *, block: str) -> None:
    if seed is None:
        return

    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"{block}: seed must be an integer, got {seed!r}")
    _check_range(seed, name="seed", low=0, high=MAX_SEED, block=block)


def _service_dependency(identifier: Any) -> DependentResource:
    resource = DependentResource(
        resource_type="third_party_model",
        identifier=identifier,
        details={"provider": PROVIDER, "resource": "stability_ai_client"},
    )

    return resource


def _new_image_id() -> str:
    identifier = f"stability-{uuid.uuid4().hex}"

    return identifier

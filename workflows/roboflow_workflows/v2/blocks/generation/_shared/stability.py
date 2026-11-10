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
from typing import Annotated, Any, Dict, Literal, Mapping, Optional, Tuple, Union
import cv2
import numpy as np
import torch
from pydantic import Field, StrictInt, StrictStr
from roboflow_workflows.v2.blocks.visualization.annotations.rows import (
    dense_masks_of,
    placed_in,
)
from roboflow_workflows.v2.data.image import FrameMapping, ImageData
from roboflow_workflows.v2.data.kinds import STRING_KIND
from roboflow_workflows.v2.data.secrets import (
    SECRET_KIND,
    Secret,
    reveal_secrets,
    to_secret,
)
from roboflow_workflows.v2.engine.declaration import DependentResource, Ref


__all__ = ['API_HOST', 'GENERATION_ENDPOINTS', 'INPAINTING_ENDPOINT', 'OUTPAINTING_ENDPOINT', 'PRESETS', 'StabilityAIClient', 'StabilityAIError', 'edited_image', 'inpainting_mask']

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

MASK_BLUR_KERNEL = (15, 15)

DEFAULT_TIMEOUT_SECONDS = 120.0

ERROR_DETAIL_LIMIT = 500

PROVIDER = "stability_ai"

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
                f"HTTP {response.status_code} from {path}: "
                f"{_error_detail(response, api_key=api_key)}"
            )

        return response.content

DEFAULT_CLIENT = StabilityAIClient()

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
        scrubbed = revealed.scrub(f"{type(error).__name__}: {error}")
        failure = scrubbed[:ERROR_DETAIL_LIMIT]
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

def _error_detail(response: Any, *, api_key: str) -> str:
    try:
        detail = str(response.json())
    except ValueError:
        detail = response.text

    # Scrub the complete response first: slicing a key at the cutoff would
    # leave a prefix that exact-match scrubbing can no longer recognize.
    revealed = reveal_secrets(to_secret(api_key, name="api_key"))
    shortened = revealed.scrub(detail)[:ERROR_DETAIL_LIMIT]

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

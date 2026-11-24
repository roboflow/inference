"""Google Vision OCR with an explicit host-owned HTTP transport.

This ports V1's text and document OCR to native Detections. The host injects
``google_vision_transport`` (a requests.Session-shaped object) and
``google_vision_base_url`` (e.g. https://vision.googleapis.com). There is no
implicit network client, platform proxy, model load, or prediction cache.
Each image causes one images:annotate request; the transport is reused.
``api_key``, ``ocr_type``, ``language_hints`` and ``timeout`` are one value for
every image of a call; a selector may change them between runs, not images.

Image encoding is an explicit host boundary: RGB/gray tensors become PNG bytes
on the CPU. Returned tensors stay on the input image's device. Predictions
carry the image frame and V2 row identity, including when there is no text.
Errors deliberately omit provider bodies and transport exception text. They
name only the integer HTTP status, the integer provider error code and a
canonical status name such as PERMISSION_DENIED, plus a suggested repair.

Credentials: an ``api_key`` selected from a workflow input of kind ``secret``
arrives as a ``roboflow_workflows.v2.data.secrets.Secret``, which hides the
value in reprs, serialized outputs and recordings. A plaintext ``api_key``
literal is stored unprotected in the saved workflow definition; prefer a
secret selector such as ``$inputs.google_key``.
"""

import base64
import io
import math
from typing import Annotated, Any, List, Literal, Optional, Protocol, Union
from urllib.parse import urlsplit
import torch
from inference_models.models.base.object_detection import Detections
from pydantic import (
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
)
from pydantic_core import PydanticCustomError
from roboflow_workflows.v2.blocks.models.common import one_per_call, prediction_metadata
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import (
    FLOAT_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
)
from roboflow_workflows.v2.data.predictions import OBJECT_DETECTION_PREDICTION_KIND
from roboflow_workflows.v2.data.row_identity import with_row_identity
from roboflow_workflows.v2.data.secrets import SECRET_KIND, to_secret
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref


OCRType = Literal["text_detection", "ocr_text_detection"]

_TimeoutSeconds = Annotated[
    Union[StrictFloat, StrictInt], Field(gt=0, allow_inf_nan=False)
]

_FEATURES = {
    "text_detection": "TEXT_DETECTION",
    "ocr_text_detection": "DOCUMENT_TEXT_DETECTION",
}

_FLOAT32_MAX = torch.finfo(torch.float32).max

_MANAGED_KEY_MESSAGE = (
    "Roboflow-managed keys (rf_key:*, V1's default) are not supported by "
    "roboflow_core/google_vision_ocr@v1; pass your Google Cloud Vision API key, preferably "
    "through a workflow input of kind secret"
)

_STATUS_NAMES = (
    "OK",
    "CANCELLED",
    "UNKNOWN",
    "INVALID_ARGUMENT",
    "DEADLINE_EXCEEDED",
    "NOT_FOUND",
    "ALREADY_EXISTS",
    "PERMISSION_DENIED",
    "RESOURCE_EXHAUSTED",
    "FAILED_PRECONDITION",
    "ABORTED",
    "OUT_OF_RANGE",
    "UNIMPLEMENTED",
    "INTERNAL",
    "UNAVAILABLE",
    "DATA_LOSS",
    "UNAUTHENTICATED",
)

_RETRY_LATER = "provider-side failure; retry later"

_REPAIRS = {
    # Google also reports an invalid API key as HTTP 400 INVALID_ARGUMENT.
    "INVALID_ARGUMENT": (
        "check the API key, the image, and the ocr_type and language_hints settings"
    ),
    "UNAUTHENTICATED": "check the API key",
    "PERMISSION_DENIED": (
        "check the API key, its restrictions, and that the Cloud Vision API "
        "and billing are enabled for its project"
    ),
    "NOT_FOUND": "check google_vision_base_url",
    "RESOURCE_EXHAUSTED": (
        "check the Cloud Vision quota and rate limits of the key's project; "
        "if one is used up, wait or request a higher quota"
    ),
    "DEADLINE_EXCEEDED": "the request deadline expired; retry later",
    "INTERNAL": _RETRY_LATER,
    "UNAVAILABLE": _RETRY_LATER,
}

_HTTP_STATUS_REPAIR_KEYS = {
    400: "INVALID_ARGUMENT",
    401: "UNAUTHENTICATED",
    403: "PERMISSION_DENIED",
    404: "NOT_FOUND",
    429: "RESOURCE_EXHAUSTED",
    500: "INTERNAL",
    503: "UNAVAILABLE",
    504: "DEADLINE_EXCEEDED",
}

_DEFAULT_REPAIR = "see Google Cloud Vision's error documentation for this status"

_TIMEOUT_CLASS_NAMES = frozenset({"Timeout", "TimeoutError"})

class GoogleVisionTransport(Protocol):
    """Host transport, structurally compatible with requests.Session.

    The host owns its lifetime, connection pooling and concurrency safety.
    """

    def post(self, url: str, *, params: dict, json: dict, timeout: float) -> Any:
        """Return an object with status_code and json(), without retries here.

        Args:
            url: The explicitly configured images:annotate endpoint.
            params: Contains the revealed vendor key under ``key``.
            json: One image and its selected OCR feature.
            timeout: Positive request timeout, in seconds.

        Returns:
            A requests.Response-shaped object.
        """

class GoogleVisionOCRError(RuntimeError):
    """A sanitized transport, provider or response-contract failure."""

class GoogleVisionOCRBlock(Block):
    """Read scene or document text through a host-injected vendor transport.

    Args:
        google_vision_transport: requests.Session-shaped transport; required.
        google_vision_base_url: Explicit HTTP(S) origin/base path. No userinfo,
            query or fragment. ``/v1/images:annotate`` is appended.

    Raises:
        ValueError: For a malformed base URL or missing transport.post method.
    """

    type = "roboflow_core/google_vision_ocr@v1"
    metadata = {
        "name": "Google Vision OCR",
        "port_of": "roboflow_core/google_vision_ocr@v1",
        "section": "model",
        "search_keywords": ["google", "vision", "ocr", "text"],
    }
    outputs = {
        "text": Output(
            STRING_KIND, source="image", description="Full recognized text."
        ),
        "language": Output(
            STRING_KIND, source="image", description="Detected locale, or empty."
        ),
        "predictions": Output(
            OBJECT_DETECTION_PREDICTION_KIND,
            source="image",
            description="One native detection per text block, with its recognized text.",
        ),
    }

    class Params(BlockParams):
        """Image batch and literal or selected OCR request settings."""

        # The engine chains pydantic's ValidationError, whose rendering would
        # otherwise echo a rejected literal api_key.
        model_config = ConfigDict(hide_input_in_errors=True)

        image: Ref(IMAGE_KIND, batch="always") = Field(  # noqa: F821
            description="RGB or gray images to read; one vendor request per image."
        )
        api_key: Union[StrictStr, Ref(STRING_KIND, SECRET_KIND)] = (
            Field(
                description=(
                    "Google Cloud Vision API key or secret selector; "
                    "Roboflow-managed rf_key:* keys are unsupported."
                ),
                examples=["$inputs.google_key"],
                repr=False,
            )
        )
        ocr_type: Union[OCRType, Ref(STRING_KIND)] = Field(
            default="text_detection",
            description="Scene text_detection or dense document ocr_text_detection.",
            examples=["ocr_text_detection", "$inputs.ocr_type"],
        )
        language_hints: Union[
            Optional[List[StrictStr]],
            Ref(LIST_OF_VALUES_KIND, allow_none=True),
        ] = Field(
            default=None,
            description="Optional language codes; None omits imageContext, [] sends [].",
            examples=[["en", "pl"], "$inputs.languages"],
        )
        timeout: Union[_TimeoutSeconds, Ref(FLOAT_KIND)] = Field(
            default=30.0,
            description="Finite positive request timeout in seconds; no retries.",
            examples=[30.0, "$inputs.timeout"],
        )

        @field_validator("api_key")
        @classmethod
        def _reject_managed_key(cls, value: Any) -> Any:
            # A migrated V1 definition fails at compile time, not on its first
            # image. Selected values are checked again before any request.
            if isinstance(value, str) and value.startswith("rf_key:"):
                raise PydanticCustomError("managed_api_key", _MANAGED_KEY_MESSAGE)

            return value

    def __init__(
        self,
        *,
        google_vision_transport: GoogleVisionTransport,
        google_vision_base_url: str,
    ) -> None:
        self._endpoint = _endpoint(google_vision_base_url)
        if not callable(getattr(google_vision_transport, "post", None)):
            raise ValueError("google_vision_transport must provide post()")

        self._transport = google_vision_transport

    def run(
        self,
        *,
        image: Batch,
        api_key: Any,
        ocr_type: Any = "text_detection",
        language_hints: Any = None,
        timeout: Any = 30.0,
    ) -> List[dict]:
        """Validate the entire batch, then read each image in batch order.

        Args:
            image: Batch of ImageData objects, including crops/resizes.
            api_key: Vendor str or Secret for every image.
            ocr_type: OCR mode for every image.
            language_hints: None or a list of codes for every image.
            timeout: Positive seconds for every request.

        Returns:
            One mapping of text, language and predictions per input image.

        Raises:
            ValueError: Invalid settings or images, or a ``Batch`` for a
                setting; all fail before any transport call.
            SecretUnavailableError: A replayed secret has no value.
            GoogleVisionOCRError: Sanitized transport/provider/response failure.
        """
        _require_images(image)
        mode, hints, seconds = _checked_settings(
            api_key=api_key,
            ocr_type=ocr_type,
            language_hints=language_hints,
            timeout=timeout,
        )
        # An empty batch sends nothing, so its key is never read.
        if not image.content:
            return []

        key = _plain_api_key(api_key)

        results = []
        for member in image.content:
            request = {
                "image": {"content": _encode_image(member)},
                "features": [{"type": _FEATURES[mode]}],
            }
            if hints is not None:
                request["imageContext"] = {"languageHints": list(hints)}

            result = _execute(
                self._transport,
                endpoint=self._endpoint,
                key=key,
                request={"requests": [request]},
                timeout=seconds,
            )
            results.append(_parse_response(result, image=member))

        return results

def _endpoint(base_url: str) -> str:
    failure = "google_vision_base_url must be an explicit HTTP(S) base URL without credentials, query or fragment"
    if not isinstance(base_url, str) or not base_url or base_url != base_url.strip():
        raise ValueError(failure)

    try:
        parsed = urlsplit(base_url)
        valid = (
            parsed.scheme in ("http", "https")
            and parsed.hostname
            and parsed.username is None
            and parsed.password is None
            and not parsed.query
            and not parsed.fragment
            and parsed.port != 0
            and not any(c.isspace() or ord(c) < 32 for c in base_url)
        )
    except ValueError:
        valid = False
    if not valid:
        raise ValueError(failure)

    endpoint = base_url.rstrip("/") + "/v1/images:annotate"

    return endpoint

def _checked_settings(
    *, api_key: Any, ocr_type: Any, language_hints: Any, timeout: Any
) -> tuple:
    """``(mode, hints, seconds)`` of a call, checked once; the key is not read."""
    for name, value in (
        ("api_key", api_key),
        ("ocr_type", ocr_type),
        ("language_hints", language_hints),
        ("timeout", timeout),
    ):
        one_per_call(value, name=name, block=GoogleVisionOCRBlock.type)
    if not isinstance(ocr_type, str) or ocr_type not in _FEATURES:
        raise ValueError("ocr_type must be text_detection or ocr_text_detection")
    if language_hints is not None and (
        not isinstance(language_hints, list)
        or any(not isinstance(code, str) or not code.strip() for code in language_hints)
    ):
        raise ValueError("language_hints must be None or a list of non-empty strings")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("timeout must be finite positive seconds")

    settings = (ocr_type, language_hints, float(timeout))

    return settings


def _plain_api_key(api_key: Any) -> str:
    """The checked plaintext key; reveals the secret, so call it only to send."""
    secret = to_secret(api_key, name="google_vision_api_key")
    if secret is None:
        raise ValueError("api_key must be a non-empty Google Cloud Vision API key")

    plain = secret.reveal()
    if not plain.strip():
        raise ValueError("api_key must be a non-empty Google Cloud Vision API key")
    if plain.startswith("rf_key:"):
        raise ValueError(_MANAGED_KEY_MESSAGE)

    return plain


def _require_images(image: Batch) -> None:
    if not isinstance(image, Batch):
        raise ValueError("image must be a Batch of ImageData")

    for member in image.content:
        if not isinstance(member, ImageData):
            raise ValueError("image must contain ImageData")


def _encode_image(image: ImageData) -> str:
    # Import only at the explicit host boundary, not at catalogue discovery.
    from PIL import Image

    pixels = image.tensor_image.detach().cpu().permute(1, 2, 0).numpy()
    if image.channels == 1:
        pixels = pixels[:, :, 0]

    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

    return encoded

def _execute(
    transport: GoogleVisionTransport,
    *,
    endpoint: str,
    key: str,
    request: dict,
    timeout: float,
) -> dict:
    # Raise outside exception handlers: even __context__ must not retain a
    # transport exception that may include a URL-encoded credential.
    failure = None
    try:
        response = transport.post(
            endpoint, params={"key": key}, json=request, timeout=timeout
        )
        if response.status_code != 200:
            failure = _http_failure(response)
        else:
            payload = response.json()
    except Exception as error:
        failure = (
            f"Google Vision request timed out after {timeout:g} s; retry later "
            "or raise timeout"
            if _is_timeout(error)
            else "Google Vision transport or JSON decoding failed"
        )
    if failure is not None:
        raise GoogleVisionOCRError(failure)

    if (
        not isinstance(payload, dict)
        or "error" in payload
        or not isinstance(payload.get("responses"), list)
        or len(payload["responses"]) != 1
        or not isinstance(payload["responses"][0], dict)
    ):
        raise GoogleVisionOCRError("Invalid Google Vision response envelope")

    result = payload["responses"][0]
    if "error" in result:
        # Per-image errors are google.rpc.Status: code is a gRPC code.
        code, name = _provider_error(result["error"], grpc_code=True)
        message = _failure_message(
            "Google Vision reported an image error", code=code, name=name
        )
        raise GoogleVisionOCRError(message)

    return result

def _is_timeout(error: Exception) -> bool:
    return any(cls.__name__ in _TIMEOUT_CLASS_NAMES for cls in type(error).__mro__)

def _http_failure(response: Any) -> str:
    status = response.status_code
    try:
        body = response.json()
    except Exception:
        body = None
    # Request-level errors carry the HTTP status as code and a status name.
    error = body.get("error") if isinstance(body, dict) else None
    code, name = _provider_error(error, grpc_code=False)
    if isinstance(status, bool) or not isinstance(status, int):
        status = None
    if status is not None and not 100 <= status <= 599:
        status = None

    prefix = (
        "Google Vision returned an invalid HTTP status"
        if status is None
        else f"Google Vision returned HTTP {status}"
    )
    fallback = _HTTP_STATUS_REPAIR_KEYS.get(status)
    if fallback is None and status is not None and status >= 500:
        fallback = "INTERNAL"
    message = _failure_message(prefix, code=code, name=name, fallback=fallback)

    return message

def _provider_error(error: Any, *, grpc_code: bool) -> tuple:
    """Return only the integer code and a known status name from an error object."""
    if not isinstance(error, dict):
        return None, None

    code = error.get("code")
    if isinstance(code, bool) or not isinstance(code, int) or not 0 <= code <= 999:
        code = None
    name = error.get("status")
    if not isinstance(name, str) or name not in _STATUS_NAMES:
        name = None
    if name is None and grpc_code and code is not None and code < len(_STATUS_NAMES):
        name = _STATUS_NAMES[code]

    return code, name

def _failure_message(
    prefix: str,
    *,
    code: Optional[int],
    name: Optional[str],
    fallback: Optional[str] = None,
) -> str:
    details = [name] if name else []
    if code is not None:
        details.append(f"provider code {code}")
    suffix = f" ({', '.join(details)})" if details else ""
    repair = _REPAIRS.get(name or fallback, _DEFAULT_REPAIR)
    message = f"{prefix}{suffix}; {repair}"

    return message

def _parse_response(result: dict, *, image: ImageData) -> dict:
    failure = False
    try:
        annotations = result.get("textAnnotations", [])
        text = annotations[0]["description"] if annotations else ""
        language = annotations[0].get("locale", "") if annotations else ""
        if not isinstance(text, str) or not isinstance(language, str):
            raise ValueError

        boxes, confidence, box_metadata = [], [], []
        # As in V1, an image without textAnnotations has no text, so its
        # fullTextAnnotation blocks are ignored rather than returned as boxes.
        pages = (
            result.get("fullTextAnnotation", {}).get("pages", []) if annotations else []
        )
        for page in pages:
            for block in page["blocks"]:
                vertices = block["boundingBox"]["vertices"]
                xs = [float(vertex.get("x", 0)) for vertex in vertices]
                ys = [float(vertex.get("y", 0)) for vertex in vertices]
                score = float(block.get("confidence", 1.0))
                # Boxes are float32 tensors; a larger finite value would become inf.
                if not all(
                    math.isfinite(v) and abs(v) <= _FLOAT32_MAX for v in [*xs, *ys]
                ):
                    raise ValueError
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError

                boxes.append([min(xs), min(ys), max(xs), max(ys)])
                confidence.append(score)
                words = [
                    "".join(symbol["text"] for symbol in word["symbols"])
                    for paragraph in block["paragraphs"]
                    for word in paragraph["words"]
                ]
                block_text = " ".join(words)
                # As in V1, each box's "class" is its recognized text. Every box
                # has class_id 0; "text-region" is only the class-id name fallback.
                box_metadata.append({"class": block_text, "text": block_text})
    except (KeyError, IndexError, TypeError, ValueError, AttributeError, OverflowError):
        failure = True
    if failure:
        raise GoogleVisionOCRError("Invalid Google Vision text annotation")

    device = image.tensor_image.device
    predictions = Detections(
        xyxy=torch.tensor(boxes, dtype=torch.float32, device=device).reshape(-1, 4),
        class_id=torch.zeros(len(boxes), dtype=torch.int64, device=device),
        confidence=torch.tensor(confidence, dtype=torch.float32, device=device),
        bboxes_metadata=box_metadata or None,
        image_metadata=prediction_metadata(
            None, image, class_names=("text-region",), prediction_type="ocr"
        ),
    )
    output = {
        "text": text,
        "language": language,
        "predictions": with_row_identity(predictions),
    }

    return output

BLOCKS = (GoogleVisionOCRBlock,)

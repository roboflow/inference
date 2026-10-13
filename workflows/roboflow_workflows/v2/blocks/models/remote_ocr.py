"""Google Vision OCR with an explicit host-owned HTTP transport.

This ports V1's text and document OCR to native Detections. The host injects
``google_vision_transport`` (a requests.Session-shaped object) and
``google_vision_base_url`` (e.g. https://vision.googleapis.com). There is no
implicit network client, platform proxy, model load, or prediction cache.
Each image causes one images:annotate request; the transport is reused.

Image encoding is an explicit host boundary: RGB/gray tensors become PNG bytes
on the CPU. Returned tensors stay on the input image's device. Predictions
carry the image frame and V2 row identity, including when there is no text.
Errors deliberately omit provider bodies and transport exception text.

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
from pydantic import Field, StrictFloat, StrictInt, StrictStr
from roboflow_workflows.v2.blocks.models.common import prediction_metadata
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

    type = "v2/google_vision_ocr@v1"
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

        image: Ref(IMAGE_KIND, batch="always") = Field(  # noqa: F821
            description="RGB or gray images to read; one vendor request per image."
        )
        api_key: Union[StrictStr, Ref(STRING_KIND, SECRET_KIND, batch="if_varying")] = (
            Field(
                description="Your vendor key or secret selector; rf_key:* is unsupported.",
                examples=["$inputs.google_key"],
                repr=False,
            )
        )
        ocr_type: Union[OCRType, Ref(STRING_KIND, batch="if_varying")] = Field(
            default="text_detection",
            description="Scene text_detection or dense document ocr_text_detection.",
            examples=["ocr_text_detection", "$inputs.ocr_type"],
        )
        language_hints: Union[
            Optional[List[StrictStr]],
            Ref(LIST_OF_VALUES_KIND, batch="if_varying", allow_none=True),
        ] = Field(
            default=None,
            description="Optional language codes; None omits imageContext, [] sends [].",
            examples=[["en", "pl"], "$inputs.languages"],
        )
        timeout: Union[_TimeoutSeconds, Ref(FLOAT_KIND, batch="if_varying")] = Field(
            default=30.0,
            description="Finite positive request timeout in seconds; no retries.",
            examples=[30.0, "$inputs.timeout"],
        )

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
            api_key: Vendor str or Secret, shared or a Batch aligned to images.
            ocr_type: OCR mode, shared or a Batch aligned to images.
            language_hints: None or a list of codes, shared or an aligned Batch.
            timeout: Positive seconds, shared or a Batch aligned to images.

        Returns:
            One mapping of text, language and predictions per input image.

        Raises:
            ValueError: Invalid settings, image or parameter batch alignment;
                all settings fail before any transport call.
            SecretUnavailableError: A replayed secret has no value.
            GoogleVisionOCRError: Sanitized transport/provider/response failure.
        """
        rows = _resolved_rows(
            image,
            api_key=api_key,
            ocr_type=ocr_type,
            language_hints=language_hints,
            timeout=timeout,
        )
        results = []
        for member, key, mode, hints, seconds in rows:
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


def _per_row(value: Any, *, image: Batch, name: str) -> tuple:
    if not isinstance(value, Batch):
        return (value,) * len(image)

    if len(value) != len(image) or value.indices != image.indices:
        raise ValueError(f"{name} batch must align with the image batch")

    return value.content


def _resolved_rows(
    image: Batch,
    *,
    api_key: Any,
    ocr_type: Any,
    language_hints: Any,
    timeout: Any,
) -> list:
    if not isinstance(image, Batch):
        raise ValueError("image must be a Batch of ImageData")

    rows = []
    for member, key, mode, hints, seconds in zip(
        image.content,
        _per_row(api_key, image=image, name="api_key"),
        _per_row(ocr_type, image=image, name="ocr_type"),
        _per_row(language_hints, image=image, name="language_hints"),
        _per_row(timeout, image=image, name="timeout"),
    ):
        if not isinstance(member, ImageData):
            raise ValueError("image must contain ImageData")
        if not isinstance(mode, str) or mode not in _FEATURES:
            raise ValueError("ocr_type must be text_detection or ocr_text_detection")
        if hints is not None and (
            not isinstance(hints, list)
            or any(not isinstance(code, str) or not code.strip() for code in hints)
        ):
            raise ValueError(
                "language_hints must be None or a list of non-empty strings"
            )
        if (
            isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds)
            or seconds <= 0
        ):
            raise ValueError("timeout must be finite positive seconds")

        secret = to_secret(key, name="google_vision_api_key")
        if secret is None:
            raise ValueError("api_key must be a non-empty vendor key")

        plain = secret.reveal()
        if not plain.strip() or plain.startswith("rf_key:"):
            raise ValueError(
                "api_key must be a non-empty vendor key; managed proxy keys are unsupported"
            )

        rows.append((member, plain, mode, hints, float(seconds)))

    return rows


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
            failure = "Google Vision returned an unsuccessful HTTP status"
        else:
            payload = response.json()
    except TimeoutError:
        failure = "Google Vision request timed out"
    except Exception:
        failure = "Google Vision transport or JSON decoding failed"
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
        raise GoogleVisionOCRError("Google Vision reported an image error")

    return result


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

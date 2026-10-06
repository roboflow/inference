"""Model-mode helpers for WebRTC streaming.

When ``client.webrtc.stream(model_id=...)`` is used, a minimal single-model
Workflow specification is built under the hood. This module holds that logic:
resolving the model's task type (via explicit parameter or Roboflow API
lookup), mapping the task type to the Workflow block wrapping it, and filling
the model-mode StreamConfig defaults.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

import requests

from inference_sdk.config import RF_API_BASE_URL
from inference_sdk.http.errors import InvalidParameterError
from inference_sdk.http.utils.requests import (
    api_key_safe_raise_for_status,
    deduct_api_key_from_string,
)
from inference_sdk.webrtc.config import StreamConfig

# Map model task type (as reported by the Roboflow model-registry endpoint,
# `GET /models/v1/external/stat` -> modelMetadata.taskType) to the Workflow
# block `type` literal that wraps a Roboflow model of that task type, plus the
# name of the block output holding the primary predictions.
#
# CONTRACT: the serialized predictions dict passes through to user handlers
# verbatim - the SDK does not inspect or normalize it, so its shape is
# whatever the block's output kind serializes to server-side (detection-style
# `{"image", "predictions": [...]}` for detection-family models, `top`/
# `confidence` keys for classification, `rle_mask` entries for semantic
# segmentation, ...). When predictions are unavailable for a frame (pairing
# eviction, pts=None, missing output) the session delivers `data=None` -
# handlers must check for it. Any model wrappable by a generic
# `roboflow_*_model` block can be listed here; VLMs cannot (each VLM family
# has its own dedicated block and manifest, so there is no uniform block to
# generate - stream those via workflow= instead).
TASK_TYPE_TO_BLOCK = {
    "object-detection": {
        "block_type": "roboflow_core/roboflow_object_detection_model@v2",
        "predictions_output": "predictions",
    },
    "instance-segmentation": {
        "block_type": "roboflow_core/roboflow_instance_segmentation_model@v2",
        "predictions_output": "predictions",
    },
    "classification": {
        "block_type": "roboflow_core/roboflow_classification_model@v2",
        "predictions_output": "predictions",
    },
    "multi-label-classification": {
        "block_type": "roboflow_core/roboflow_multi_label_classification_model@v2",
        "predictions_output": "predictions",
    },
    "keypoint-detection": {
        "block_type": "roboflow_core/roboflow_keypoint_detection_model@v2",
        "predictions_output": "predictions",
    },
    "semantic-segmentation": {
        "block_type": "roboflow_core/roboflow_semantic_segmentation_model@v2",
        "predictions_output": "predictions",
    },
}

# Timeout (seconds) for the Roboflow task-type lookup.
_TASK_TYPE_LOOKUP_TIMEOUT = 10


def _raise_unsupported_task_type(task_type: str, model_id: str, origin: str) -> None:
    """Raise a uniform error for task types model_id streaming cannot serve."""
    raise InvalidParameterError(
        f"{origin} task type '{task_type}' for model_id '{model_id}' is not "
        f"supported for model_id streaming. Supported task types: "
        f"{sorted(TASK_TYPE_TO_BLOCK)}. No generic Workflow model block exists "
        "for this task type (VLM families each have their own dedicated "
        "block) - pass a full Workflow wrapping the model via workflow= "
        "instead."
    )


def resolve_task_type(
    model_id: str, task_type: Optional[str], api_key: Optional[str]
) -> str:
    """Resolve a model's task type, either from ``task_type`` or the API.

    When ``task_type`` is provided it is validated against
    ``TASK_TYPE_TO_BLOCK`` and returned as-is (no network call). When None,
    the Roboflow model-registry endpoint (``GET /models/v1/external/stat``)
    is queried for ``modelMetadata.taskType``. The endpoint accepts the
    ``model_id`` verbatim — aliases (e.g. "rfdetr-nano") and non-versioned
    ids are resolved server-side.

    Args:
        model_id: Roboflow model ID or alias (e.g. "rfdetr-nano").
        task_type: Explicit task type, or None to auto-resolve.
        api_key: Roboflow API key used for the lookup (sent as a Bearer
            header; not required for public models).

    Returns:
        A task type string that is a key of ``TASK_TYPE_TO_BLOCK``.

    Raises:
        InvalidParameterError: If ``task_type`` (explicit or resolved) is not
            supported for model_id streaming.
        RuntimeError: If the API lookup fails (network / HTTP / parse error).
    """
    if task_type is not None:
        if task_type not in TASK_TYPE_TO_BLOCK:
            _raise_unsupported_task_type(task_type, model_id, origin="Provided")
        return task_type

    url = f"{RF_API_BASE_URL}/models/v1/external/stat"
    headers = {}
    if api_key:
        # The key travels in a header - never in the URL - so request
        # exceptions (whose text embeds the URL) cannot leak it.
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        response = requests.get(
            url,
            params={"modelId": model_id},
            headers=headers,
            timeout=_TASK_TYPE_LOOKUP_TIMEOUT,
        )
        api_key_safe_raise_for_status(response=response)
        resolved = response.json()["modelMetadata"]["taskType"]
    except Exception as e:
        # Defensive redaction (the key is header-borne, so exception text
        # should never contain it) + `from None` so no unredacted exception
        # is retained as __cause__ in tracebacks.
        safe_error = deduct_api_key_from_string(str(e))
        raise RuntimeError(
            f"Failed to resolve task type for model_id '{model_id}' via the "
            f"Roboflow API: {e.__class__.__name__}: {safe_error}. A 404 means "
            "the model does not exist or the api_key cannot access it. If the "
            "Roboflow API is unreachable (air-gapped / self-hosted), bypass "
            "the lookup by passing task_type= explicitly (one of "
            f"{sorted(TASK_TYPE_TO_BLOCK)})."
        ) from None

    if resolved not in TASK_TYPE_TO_BLOCK:
        _raise_unsupported_task_type(resolved, model_id, origin="Roboflow API reported")
    return resolved


def build_model_workflow(model_id: str, task_type: str) -> dict:
    """Build a minimal single-model workflow spec wrapping a model ID.

    The Workflow model block is chosen from ``TASK_TYPE_TO_BLOCK`` based on
    ``task_type``. The block's primary predictions output is always exposed
    under the "predictions" JsonField name (the selector points at whatever
    the block calls its predictions output) - the name the session reads
    predictions from in model mode. The serialized dict is delivered to user
    handlers verbatim; see the ``TASK_TYPE_TO_BLOCK`` comment for the data
    contract.

    Args:
        model_id: Roboflow model ID (e.g. "rfdetr-nano")
        task_type: Model task type; must be a key of ``TASK_TYPE_TO_BLOCK``.

    Returns:
        Workflow specification dict producing "predictions" and "image" outputs

    Raises:
        InvalidParameterError: If ``task_type`` is not supported.
    """
    block = TASK_TYPE_TO_BLOCK.get(task_type)
    if block is None:
        raise InvalidParameterError(
            f"Unsupported task_type '{task_type}'. Supported task types: "
            f"{sorted(TASK_TYPE_TO_BLOCK)}."
        )
    predictions_output = block["predictions_output"]
    return {
        "version": "1.0",
        "inputs": [{"type": "InferenceImage", "name": "image"}],
        "steps": [
            {
                "type": block["block_type"],
                "name": "model",
                "images": "$inputs.image",
                "model_id": model_id,
            }
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "predictions",
                "coordinates_system": "own",
                "selector": f"$steps.model.{predictions_output}",
            },
            {
                "type": "JsonField",
                "name": "image",
                "selector": "$inputs.image",
            },
        ],
    }


def apply_model_id_defaults(config: Optional[StreamConfig]) -> StreamConfig:
    """Fill default stream/data outputs for model_id mode.

    Defaults route the "image" output through the video path and the
    "predictions" output through the data channel. When the user supplies a
    config, only empty ``stream_output`` / ``data_output`` fields are filled;
    all other settings are preserved. The user's config is never mutated.

    Args:
        config: User-provided stream configuration, or None

    Returns:
        StreamConfig with model_id defaults applied
    """
    if config is None:
        return StreamConfig(stream_output=["image"], data_output=["predictions"])
    return replace(
        config,
        stream_output=config.stream_output or ["image"],
        data_output=config.data_output or ["predictions"],
    )

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
from inference_sdk.http.utils.aliases import resolve_roboflow_model_alias
from inference_sdk.http.utils.requests import (
    api_key_safe_raise_for_status,
    deduct_api_key_from_string,
)
from inference_sdk.webrtc.config import StreamConfig

# Map project task type (as returned by the Roboflow /ort endpoint, and matching
# the model-registry keys in inference/models/utils.py) to the Workflow block
# `type` literal that wraps a Roboflow model of that task type, plus the name of
# the block output holding the primary predictions. Every block below exposes a
# "predictions" output (verified against each block manifest's describe_outputs),
# so the selector is uniform; the mapping keeps the predictions-output name
# explicit so the generated workflow can adapt if a future block renames it.
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

# Timeout (seconds) for the Roboflow /ort task-type lookup.
_TASK_TYPE_LOOKUP_TIMEOUT = 10


def resolve_task_type(
    model_id: str, task_type: Optional[str], api_key: Optional[str]
) -> str:
    """Resolve a model's task type, either from ``task_type`` or the API.

    When ``task_type`` is provided it is validated against
    ``TASK_TYPE_TO_BLOCK`` and returned as-is (no network call). When None,
    the model alias is resolved, the model ID is split into dataset/version,
    and the Roboflow ``/ort`` endpoint is queried for ``ort.type``.

    Args:
        model_id: Roboflow model ID or alias (e.g. "rfdetr-nano").
        task_type: Explicit task type, or None to auto-resolve.
        api_key: Roboflow API key used for the lookup.

    Returns:
        A task type string that is a key of ``TASK_TYPE_TO_BLOCK``.

    Raises:
        InvalidParameterError: If ``task_type`` is unsupported, the model ID
            has no version after de-aliasing, or the resolved task type is
            unsupported.
        RuntimeError: If the API lookup fails (network / parse error).
    """
    if task_type is not None:
        if task_type not in TASK_TYPE_TO_BLOCK:
            raise InvalidParameterError(
                f"Unsupported task_type '{task_type}'. Supported task types: "
                f"{sorted(TASK_TYPE_TO_BLOCK)}."
            )
        return task_type

    resolved_model_id = resolve_roboflow_model_alias(model_id)
    parts = resolved_model_id.split("/")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise InvalidParameterError(
            f"Could not derive a dataset/version pair from model_id "
            f"'{model_id}' (resolved to '{resolved_model_id}'). Model IDs "
            "must be of the form 'dataset/version'. If the task type cannot "
            "be resolved this way, pass task_type= explicitly (one of "
            f"{sorted(TASK_TYPE_TO_BLOCK)})."
        )
    dataset_id, version_id = parts

    url = f"{RF_API_BASE_URL}/ort/{dataset_id}/{version_id}"
    try:
        response = requests.get(
            url,
            params={
                "api_key": api_key,
                "nocache": "true",
                "device": "sdk",
                "dynamic": "true",
            },
            timeout=_TASK_TYPE_LOOKUP_TIMEOUT,
        )
        api_key_safe_raise_for_status(response=response)
        payload = response.json()
        resolved = payload["ort"]["type"]
    except Exception as e:
        # Exception text may embed the request URL (which carries api_key=...);
        # redact before surfacing it to the user.
        safe_error = deduct_api_key_from_string(str(e))
        raise RuntimeError(
            f"Failed to resolve task type for model_id '{model_id}' "
            f"(resolved to '{resolved_model_id}') via the Roboflow API: "
            f"{e.__class__.__name__}: {safe_error}. You can bypass this lookup "
            "by passing task_type= explicitly (one of "
            f"{sorted(TASK_TYPE_TO_BLOCK)})."
        ) from e

    if resolved not in TASK_TYPE_TO_BLOCK:
        raise InvalidParameterError(
            f"Roboflow API reported task type '{resolved}' for model_id "
            f"'{model_id}', which is not supported for model_id streaming. "
            f"Supported task types: {sorted(TASK_TYPE_TO_BLOCK)}."
        )
    return resolved


def build_model_workflow(model_id: str, task_type: str) -> dict:
    """Build a minimal single-model workflow spec wrapping a model ID.

    The Workflow model block is chosen from ``TASK_TYPE_TO_BLOCK`` based on
    ``task_type``. The block's primary predictions output is always exposed
    under the "predictions" JsonField name (the selector points at whatever
    the block calls its predictions output), so the session's raw-dict
    contract is task-agnostic.

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

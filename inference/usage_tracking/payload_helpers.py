import hashlib
import json
import os
from typing import Any, Callable, DefaultDict, Dict, List, Optional, Set, Tuple, Union

import requests

# NOTE: This module is used in isolation, no imports from inference are allowed
# NOTE: Any change made to this file should be matched to changes in redis offloader

_OFFLINE_MODE_PROCESS_LATCH_ENV = "_ROBOFLOW_INFERENCE_OFFLINE_MODE_AT_PROCESS_START"
# This module is also reused by an isolated offloader and must not import
# inference or inference_models. The private marker published by
# inference_models._offline is the process-wide latched decision; fall back
# to the public variable when running fully standalone. Re-publishing the
# marker makes the snapshot survive module re-execution and propagate to
# descendant processes.
_offline_mode_value = os.getenv(
    _OFFLINE_MODE_PROCESS_LATCH_ENV, os.getenv("OFFLINE_MODE", "False")
)
_normalized_offline_mode_value = _offline_mode_value.lower()
if _normalized_offline_mode_value == "true":
    OFFLINE_MODE = True
elif _normalized_offline_mode_value == "false":
    OFFLINE_MODE = False
else:
    raise ValueError(
        "Expected OFFLINE_MODE to be a boolean (true or false), "
        f"got {_offline_mode_value!r}"
    )
os.environ[_OFFLINE_MODE_PROCESS_LATCH_ENV] = str(OFFLINE_MODE)


ResourceID = str
Usage = Union[DefaultDict[str, Any], Dict[str, Any]]
ResourceUsage = Union[DefaultDict[ResourceID, Usage], Dict[ResourceID, Usage]]
APIKey = str
APIKeyHash = str
APIKeyUsage = Union[DefaultDict[APIKey, ResourceUsage], Dict[APIKey, ResourceUsage]]
ResourceCategory = str
ResourceDetails = Dict[str, Any]
SystemDetails = Dict[str, Any]
UsagePayload = Union[APIKeyUsage, ResourceDetails, SystemDetails]


def merge_megapixel_buckets(
    left: Optional[Dict[str, Any]],
    right: Optional[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Sum per-bucket frame and duration counters.

    Kept local to this module so the redis usage offloader can mirror it without
    importing the rest of inference.
    """
    if not left:
        return {key: dict(value) for key, value in (right or {}).items()}
    if not right:
        return {key: dict(value) for key, value in left.items()}

    merged: Dict[str, Dict[str, Any]] = {
        key: {
            "processed_frames": int(value.get("processed_frames", 0) or 0),
            "execution_duration": float(value.get("execution_duration", 0) or 0),
        }
        for key, value in left.items()
    }
    for key, value in right.items():
        frames = int(value.get("processed_frames", 0) or 0)
        duration = float(value.get("execution_duration", 0) or 0)
        existing = merged.get(key)
        if existing is None:
            merged[key] = {
                "processed_frames": frames,
                "execution_duration": duration,
            }
            continue
        existing["processed_frames"] = int(existing["processed_frames"]) + frames
        existing["execution_duration"] = (
            float(existing["execution_duration"]) + duration
        )
    return merged


RESOURCE_DETAILS_KEY = "resource_details"
MODELS_KEY = "models"
CUSTOM_PYTHON_KEY = "custom_python"


def _model_identity(entry: Dict[str, Any]) -> Any:
    return entry.get("model_id")


def _custom_python_identity(entry: Dict[str, Any]) -> Tuple[Any, Any]:
    return entry.get("block_type"), entry.get("step_name")


_BILLABLE_LISTS: Tuple[
    Tuple[str, Callable[[Dict[str, Any]], Any], Tuple[str, ...]], ...
] = (
    (MODELS_KEY, _model_identity, ("frames", "execution_duration")),
    (CUSTOM_PYTHON_KEY, _custom_python_identity, ("execution_duration",)),
)


def _decoded_details(resource_details: Any) -> Optional[Dict[str, Any]]:
    if isinstance(resource_details, str):
        try:
            resource_details = json.loads(resource_details)
        except ValueError:
            return None
    if not isinstance(resource_details, dict):
        return None
    return resource_details


def _is_entry_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(entry, dict) for entry in value)


def _merge_entries(
    entries: List[Dict[str, Any]],
    *,
    identity: Callable[[Dict[str, Any]], Any],
    summed_fields: Tuple[str, ...],
) -> List[Dict[str, Any]]:
    merged: Dict[Any, Dict[str, Any]] = {}
    for entry in entries:
        key = identity(entry)
        existing = merged.get(key)
        if existing is None:
            merged[key] = dict(entry)
            continue
        combined = {**existing, **entry}
        for field in summed_fields:
            if existing.get(field) is None and entry.get(field) is None:
                continue
            combined[field] = (existing.get(field) or 0) + (entry.get(field) or 0)
        merged[key] = combined
    return list(merged.values())


def merge_resource_details(left: Any, right: Any) -> Any:
    left_details = _decoded_details(left)
    right_details = _decoded_details(right)
    if left_details is None or right_details is None:
        return right
    merged = dict(right_details)
    changed = False
    for key, identity, summed_fields in _BILLABLE_LISTS:
        left_entries = left_details.get(key)
        right_entries = right_details.get(key)
        if not _is_entry_list(left_entries):
            continue
        if right_entries is None:
            right_entries = []
        if not _is_entry_list(right_entries):
            continue
        try:
            merged[key] = _merge_entries(
                [*left_entries, *right_entries],
                identity=identity,
                summed_fields=summed_fields,
            )
        except TypeError:
            continue
        changed = True
    if not changed:
        return right
    if isinstance(right, str):
        return json.dumps(merged)
    return merged


def merge_usage_dicts(d1: UsagePayload, d2: UsagePayload):
    merged = {}
    if d1 and d2 and d1.get("resource_id") != d2.get("resource_id"):
        raise ValueError("Cannot merge usage for different resource IDs")
    if "timestamp_start" in d1 and "timestamp_start" in d2:
        merged["timestamp_start"] = min(d1["timestamp_start"], d2["timestamp_start"])
    if "timestamp_stop" in d1 and "timestamp_stop" in d2:
        merged["timestamp_stop"] = max(d1["timestamp_stop"], d2["timestamp_stop"])
    if "processed_frames" in d1 and "processed_frames" in d2:
        merged["processed_frames"] = d1["processed_frames"] + d2["processed_frames"]
    if "source_duration" in d1 and "source_duration" in d2:
        merged["source_duration"] = d1["source_duration"] + d2["source_duration"]
    merged["execution_duration"] = d1.get("execution_duration", 0) + d2.get(
        "execution_duration", 0
    )
    if "megapixel_buckets" in d1 or "megapixel_buckets" in d2:
        merged["megapixel_buckets"] = merge_megapixel_buckets(
            d1.get("megapixel_buckets"),
            d2.get("megapixel_buckets"),
        )
    if RESOURCE_DETAILS_KEY in d1 and RESOURCE_DETAILS_KEY in d2:
        merged[RESOURCE_DETAILS_KEY] = merge_resource_details(
            d1[RESOURCE_DETAILS_KEY], d2[RESOURCE_DETAILS_KEY]
        )
    return {**d1, **d2, **merged}


def get_api_key_usage_containing_resource(
    api_key_hash: APIKey, usage_payloads: List[APIKeyUsage]
) -> Optional[ResourceUsage]:
    for usage_payload in usage_payloads:
        for other_api_key_hash, resource_payloads in usage_payload.items():
            if api_key_hash and other_api_key_hash != api_key_hash:
                continue
            if other_api_key_hash == "":
                continue
            for resource_id, resource_usage in resource_payloads.items():
                if not resource_id:
                    continue
                if not resource_usage or "resource_id" not in resource_usage:
                    continue
                return resource_usage
    return


def zip_usage_payloads(usage_payloads: List[APIKeyUsage]) -> List[APIKeyUsage]:
    system_info_payload = None
    usage_by_exec_session_id: Dict[
        APIKeyHash, Dict[ResourceID, Dict[str, List[ResourceUsage]]]
    ] = {}
    for usage_payload in usage_payloads:
        for api_key_hash, resource_payloads in usage_payload.items():
            if api_key_hash == "":
                continue
            api_key_usage_by_exec_session_id = usage_by_exec_session_id.setdefault(
                api_key_hash, {}
            )
            for (
                resource_usage_key,
                resource_usage_payload,
            ) in resource_payloads.items():
                if resource_usage_key == "":
                    api_key_usage_with_resource = get_api_key_usage_containing_resource(
                        api_key_hash=api_key_hash,
                        usage_payloads=usage_payloads,
                    )
                    if not api_key_usage_with_resource:
                        system_info_payload = {"": resource_usage_payload}
                        continue
                    resource_id = api_key_usage_with_resource["resource_id"]
                    category = api_key_usage_with_resource.get("category")
                    resource_usage_key = f"{category}:{resource_id}"
                    resource_usage_payload["api_key_hash"] = api_key_hash
                    resource_usage_payload["resource_id"] = resource_id
                    resource_usage_payload["category"] = category
                    resource_usage_payload["execution_duration"] = (
                        api_key_usage_with_resource.get("execution_duration", 0)
                    )

                resource_usage_exec_session_id = (
                    api_key_usage_by_exec_session_id.setdefault(resource_usage_key, {})
                )
                exec_session_id = resource_usage_payload.get("exec_session_id", "")
                resource_usage_exec_session_id.setdefault(exec_session_id, []).append(
                    resource_usage_payload
                )

    merged_exec_session_id_streams_usage_payloads: Dict[str, APIKeyUsage] = {}
    merged_exec_session_id_photos_usage_payloads: Dict[str, APIKeyUsage] = {}
    for (
        api_key_hash,
        api_key_usage_by_exec_session_id,
    ) in usage_by_exec_session_id.items():
        for (
            resource_usage_key,
            resource_usage_exec_session_id,
        ) in api_key_usage_by_exec_session_id.items():
            for (
                exec_session_id,
                usage_payloads,
            ) in resource_usage_exec_session_id.items():
                for resource_usage_payload in usage_payloads:
                    if resource_usage_payload.get("fps"):
                        merged_api_key_usage_payloads = (
                            merged_exec_session_id_streams_usage_payloads.setdefault(
                                exec_session_id, {}
                            )
                        )
                    else:
                        merged_api_key_usage_payloads = (
                            merged_exec_session_id_photos_usage_payloads.setdefault(
                                exec_session_id, {}
                            )
                        )
                    merged_api_key_payload = merged_api_key_usage_payloads.setdefault(
                        api_key_hash, {}
                    )
                    merged_resource_payload = merged_api_key_payload.setdefault(
                        resource_usage_key, {}
                    )
                    merged_api_key_payload[resource_usage_key] = merge_usage_dicts(
                        merged_resource_payload,
                        resource_usage_payload,
                    )

    zipped_payloads = list(
        merged_exec_session_id_streams_usage_payloads.values()
    ) + list(merged_exec_session_id_photos_usage_payloads.values())
    if system_info_payload:
        system_info_api_key_hash = next(iter(system_info_payload.values()))[
            "api_key_hash"
        ]
        zipped_payloads.append({system_info_api_key_hash: system_info_payload})
    return zipped_payloads


def send_usage_payload(
    payload: UsagePayload,
    api_usage_endpoint_url: str,
    hashes_to_api_keys: Optional[Dict[APIKeyHash, APIKey]] = None,
    ssl_verify: bool = False,
    extra_headers: Optional[Dict[str, str]] = None,
) -> Set[APIKeyHash]:
    if OFFLINE_MODE:
        # Report every hash as failed so callers that delete on "no failures"
        # (including the Redis usage offloader) retain the payload instead of
        # treating the offline no-op as a successful delivery.
        return set(payload.keys())
    hashes_to_api_keys = hashes_to_api_keys or {}
    api_keys_hashes_failed = set()
    for api_key_hash, workflow_payloads in payload.items():
        if hashes_to_api_keys and api_key_hash not in hashes_to_api_keys:
            api_keys_hashes_failed.add(api_key_hash)
            continue
        api_key = hashes_to_api_keys.get(api_key_hash) or api_key_hash
        if not api_key:
            api_keys_hashes_failed.add(api_key_hash)
            continue
        complete_workflow_payloads = [
            w for w in workflow_payloads.values() if "processed_frames" in w
        ]
        try:
            for workflow_payload in complete_workflow_payloads:
                if "api_key_hash" in workflow_payload:
                    del workflow_payload["api_key_hash"]
                stream_session_id = workflow_payload.pop("stream_session_id", None)
                if stream_session_id:
                    workflow_payload["exec_session_id"] = stream_session_id
                workflow_payload["api_key"] = api_key
            if not extra_headers:
                extra_headers = {}
            response = requests.post(
                api_usage_endpoint_url,
                json=complete_workflow_payloads,
                verify=ssl_verify,
                headers={"Authorization": f"Bearer {api_key}", **extra_headers},
                timeout=1,
            )
        except Exception:
            api_keys_hashes_failed.add(api_key_hash)
            continue
        if response.status_code != 200:
            api_keys_hashes_failed.add(api_key_hash)
            continue
    return api_keys_hashes_failed


def sha256_hash(payload: str, length=5):
    payload_hash = hashlib.sha256(payload.encode())
    return payload_hash.hexdigest()[:length]

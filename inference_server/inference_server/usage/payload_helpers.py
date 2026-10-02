"""Usage rows: merging, grouping into request payloads and sending them."""

import hashlib
import json
import logging
from typing import Any, Callable, DefaultDict, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse

import requests

from inference_server import configuration

logger = logging.getLogger(__name__)

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

MAX_BILLABLE_ENTRIES_PER_ROW = 256
REQUEST_TIMEOUT_S = 1
INFERENCE_VERSION_HEADER = "X-Roboflow-Inference-Version"
ALLOW_CHUNKED_RESPONSE_HEADER = "X-Allow-Chunked"
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
    (MODELS_KEY, _model_identity, ("frames", "execution_duration", "model_latency_ms")),
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
    merged_entries = list(merged.values())

    return merged_entries


def merge_resource_details(left: Any, right: Any) -> Any:
    """Combine the billable lists of two resource details values.

    The later value wins for every key except ``models`` and ``custom_python``.
    Entries of those lists are kept from both sides; entries with the same
    identity (``model_id``, or ``block_type`` with ``step_name``) are combined
    by summing their counters and keeping the other fields of the later entry.

    Args:
        left: Earlier resource details, a dict or its JSON text.
        right: Later resource details, a dict or its JSON text.

    Returns:
        ``right`` itself when neither side carries a billable list or a side
        cannot be decoded, otherwise the merged details in the form of ``right``
        (dict or JSON text).
    """
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
        merged[key] = _merge_entries(
            [*left_entries, *right_entries],
            identity=identity,
            summed_fields=summed_fields,
        )
        changed = True
    if not changed:
        return right
    if isinstance(right, str):
        serialized = json.dumps(merged)

        return serialized

    return merged


def billable_lists_exceed_bound(resource_details: Any) -> bool:
    """Tell whether a billable list is longer than a row may carry.

    Args:
        resource_details: Resource details, a dict or its JSON text.

    Returns:
        True when ``models`` or ``custom_python`` holds more than
        ``MAX_BILLABLE_ENTRIES_PER_ROW`` entries.
    """
    details = _decoded_details(resource_details)
    if details is None:
        return False

    exceeds = any(
        isinstance(details.get(key), list)
        and len(details[key]) > MAX_BILLABLE_ENTRIES_PER_ROW
        for key, _, _ in _BILLABLE_LISTS
    )

    return exceeds


def split_billable_lists(resource_details: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Split resource details whose billable lists are longer than a row may carry.

    Args:
        resource_details: Resource details of one recorded call.

    Returns:
        The details themselves in a list of one when no billable list is longer
        than ``MAX_BILLABLE_ENTRIES_PER_ROW``; otherwise consecutive copies in
        which every oversized list is cut into slices of at most that length.
        A list that is not oversized stays whole in the first copy, and every
        entry appears in exactly one copy.
    """
    bound = MAX_BILLABLE_ENTRIES_PER_ROW
    slices_by_key: Dict[str, List[List[Any]]] = {}
    for key, _, _ in _BILLABLE_LISTS:
        entries = resource_details.get(key)
        if isinstance(entries, list) and len(entries) > bound:
            slices_by_key[key] = [
                entries[start : start + bound]
                for start in range(0, len(entries), bound)
            ]
    if not slices_by_key:
        return [resource_details]

    details_parts: List[Dict[str, Any]] = []
    for index in range(max(len(slices) for slices in slices_by_key.values())):
        part = dict(resource_details)
        for key, _, _ in _BILLABLE_LISTS:
            slices = slices_by_key.get(key)
            if slices is not None and index < len(slices):
                part[key] = slices[index]
            elif index > 0:
                part.pop(key, None)
        details_parts.append(part)

    return details_parts


def _merge_exceeds_bound(d1: UsagePayload, d2: UsagePayload) -> bool:
    if RESOURCE_DETAILS_KEY not in d1 or RESOURCE_DETAILS_KEY not in d2:
        return False

    merged_details = merge_resource_details(
        d1[RESOURCE_DETAILS_KEY], d2[RESOURCE_DETAILS_KEY]
    )
    exceeds = billable_lists_exceed_bound(merged_details)

    return exceeds


def merge_megapixel_buckets(
    left: Optional[Dict[str, Any]],
    right: Optional[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Sum per-bucket frame and duration counters.

    Args:
        left: Buckets accumulated so far.
        right: Buckets to add.

    Returns:
        A new mapping with the counters of both sides summed per bucket.
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


def merge_usage_dicts(d1: UsagePayload, d2: UsagePayload) -> UsagePayload:
    """Merge two usage rows of the same resource.

    Args:
        d1: Earlier row.
        d2: Later row; wins for every field that is not accumulated.

    Returns:
        The merged row.

    Raises:
        ValueError: If the rows belong to different resources.
    """
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
    """Find a row of the given API key that names a resource.

    Args:
        api_key_hash: API key hash the row has to belong to.
        usage_payloads: Payloads to search.

    Returns:
        The first matching row, or None.
    """
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
    """Merge queued payloads into one payload per execution session.

    Rows of the same API key hash, usage key and execution session are merged.
    A row whose billable lists would grow past ``MAX_BILLABLE_ENTRIES_PER_ROW``
    is closed and emitted in a payload of its own.

    Args:
        usage_payloads: Payloads taken from the queue.

    Returns:
        Stream payloads, then image payloads, then rows closed by the list
        bound, then a payload holding only system information, when there is one.
    """
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
    closed_usage_payloads: List[APIKeyUsage] = []
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
                    if _merge_exceeds_bound(
                        merged_resource_payload, resource_usage_payload
                    ):
                        closed_usage_payloads.append(
                            {
                                api_key_hash: {
                                    resource_usage_key: merged_resource_payload
                                }
                            }
                        )
                        merged_resource_payload = {}
                    merged_api_key_payload[resource_usage_key] = merge_usage_dicts(
                        merged_resource_payload,
                        resource_usage_payload,
                    )

    zipped_payloads = (
        list(merged_exec_session_id_streams_usage_payloads.values())
        + list(merged_exec_session_id_photos_usage_payloads.values())
        + closed_usage_payloads
    )
    if system_info_payload:
        system_info_api_key_hash = next(iter(system_info_payload.values()))[
            "api_key_hash"
        ]
        zipped_payloads.append({system_info_api_key_hash: system_info_payload})
    return zipped_payloads


def ssl_verify_for_endpoint(url: str) -> bool:
    """Tell whether TLS certificates are verified for a usage endpoint.

    Args:
        url: URL that will be requested, after any secure gateway wrapping.

    Returns:
        False only when the host is ``localhost`` or ``127.0.0.1``.
    """
    try:
        hostname = urlparse(url).hostname or ""
    except ValueError:
        return True
    return hostname.lower() not in {"localhost", "127.0.0.1"}


def usage_request_headers() -> Dict[str, Any]:
    """Build the headers sent with every usage request besides authorization.

    Returns:
        ``ROBOFLOW_API_EXTRA_HEADERS`` overridden by the server version and
        chunked-response markers.
    """
    headers = {
        INFERENCE_VERSION_HEADER: configuration.SERVER_VERSION,
        ALLOW_CHUNKED_RESPONSE_HEADER: "true",
    }
    if not configuration.ROBOFLOW_API_EXTRA_HEADERS:
        return headers

    try:
        extra_headers: dict = json.loads(configuration.ROBOFLOW_API_EXTRA_HEADERS)
    except ValueError:
        logger.warning("Could not decode ROBOFLOW_API_EXTRA_HEADERS")
        return headers
    extra_headers.update(headers)

    return extra_headers


def _outbound_row(row: Usage, *, api_key: APIKey) -> Dict[str, Any]:
    outbound_row = {key: value for key, value in row.items() if key != "api_key_hash"}
    stream_session_id = outbound_row.pop("stream_session_id", None)
    if stream_session_id:
        outbound_row["exec_session_id"] = stream_session_id
    outbound_row["api_key"] = api_key

    return outbound_row


def send_usage_payload(
    payload: UsagePayload,
    api_usage_endpoint_url: str,
    hashes_to_api_keys: Optional[Dict[APIKeyHash, APIKey]] = None,
    ssl_verify: bool = False,
    extra_headers: Optional[Dict[str, str]] = None,
    may_post: Optional[Callable[[], bool]] = None,
) -> Set[APIKeyHash]:
    """Post the rows of one payload, one request per API key.

    The payload is left untouched; the posted rows are copies carrying the API
    key instead of its hash.

    Args:
        payload: Rows keyed by API key hash, then by usage key.
        api_usage_endpoint_url: URL the rows are posted to.
        hashes_to_api_keys: API keys by their hash; a hash missing from a
            non-empty mapping is not sent.
        ssl_verify: Whether TLS certificates are verified.
        extra_headers: Headers added to the authorization header.
        may_post: Asked before every request; a request is not started once it
            answers False and the key counts as not accepted.

    Returns:
        API key hashes whose rows were not accepted (anything but HTTP 200, or
        no request started).
    """
    if configuration.LEGACY_OFFLINE_MODE:
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
        if may_post is not None and not may_post():
            api_keys_hashes_failed.add(api_key_hash)
            continue
        try:
            complete_workflow_payloads = [
                _outbound_row(w, api_key=api_key)
                for w in workflow_payloads.values()
                if "processed_frames" in w
            ]
            if not extra_headers:
                extra_headers = {}
            response = requests.post(
                api_usage_endpoint_url,
                json=complete_workflow_payloads,
                verify=ssl_verify,
                headers={"Authorization": f"Bearer {api_key}", **extra_headers},
                timeout=REQUEST_TIMEOUT_S,
            )
        except Exception:
            api_keys_hashes_failed.add(api_key_hash)
            continue
        if response.status_code != 200:
            api_keys_hashes_failed.add(api_key_hash)
            continue
    return api_keys_hashes_failed


def sha256_hash(payload: str, length: int = 5) -> str:
    """Hash a text and keep the leading hexadecimal characters.

    Args:
        payload: Text to hash.
        length: Slice end applied to the hexadecimal digest.

    Returns:
        The digest prefix.
    """
    payload_hash = hashlib.sha256(payload.encode())
    return payload_hash.hexdigest()[:length]

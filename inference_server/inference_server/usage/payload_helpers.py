"""Usage rows: merging and grouping into request payloads."""

import hashlib
from typing import Any, DefaultDict, Dict, List, Optional, Tuple, Union

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
        d2: Later row; wins for every field that is not accumulated,
            ``resource_details`` included.

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


def _group_rows_by_key_and_session(
    usage_payloads: List[APIKeyUsage],
) -> Tuple[
    Dict[APIKeyHash, Dict[ResourceID, Dict[str, List[ResourceUsage]]]],
    Optional[APIKeyUsage],
]:
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

    return usage_by_exec_session_id, system_info_payload


def _merge_grouped_rows(
    usage_by_exec_session_id: Dict[
        APIKeyHash, Dict[ResourceID, Dict[str, List[ResourceUsage]]]
    ],
) -> Tuple[Dict[str, APIKeyUsage], Dict[str, APIKeyUsage]]:
    streams_by_exec_session_id: Dict[str, APIKeyUsage] = {}
    images_by_exec_session_id: Dict[str, APIKeyUsage] = {}
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
                grouped_rows,
            ) in resource_usage_exec_session_id.items():
                for resource_usage_payload in grouped_rows:
                    if resource_usage_payload.get("fps"):
                        destination = streams_by_exec_session_id
                    else:
                        destination = images_by_exec_session_id
                    merged_api_key_usage_payloads = destination.setdefault(
                        exec_session_id, {}
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

    return streams_by_exec_session_id, images_by_exec_session_id


def zip_usage_payloads(usage_payloads: List[APIKeyUsage]) -> List[APIKeyUsage]:
    """Merge queued payloads into payloads of one execution session each.

    Rows of the same API key hash, usage key and execution session are merged,
    streams (rows with fps) apart from images.

    Args:
        usage_payloads: Payloads taken from the queue.

    Returns:
        Stream payloads, then image payloads, then a payload holding only
        system information, when there is one.
    """
    usage_by_exec_session_id, system_info_payload = _group_rows_by_key_and_session(
        usage_payloads
    )
    streams, images = _merge_grouped_rows(usage_by_exec_session_id)

    zipped_payloads = list(streams.values()) + list(images.values())
    if system_info_payload:
        system_info_api_key_hash = next(iter(system_info_payload.values()))[
            "api_key_hash"
        ]
        zipped_payloads.append({system_info_api_key_hash: system_info_payload})
    return zipped_payloads


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

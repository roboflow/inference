import hashlib
from typing import Any, DefaultDict, Dict, List, Optional, Set, Union

import requests

from inference.core.roboflow_api import build_roboflow_api_headers

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
) -> Set[APIKeyHash]:
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
                workflow_payload["api_key"] = api_key
            headers = build_roboflow_api_headers(
                explicit_headers={"Authorization": f"Bearer {api_key}"}
            )
            response = requests.post(
                api_usage_endpoint_url,
                json=complete_workflow_payloads,
                verify=ssl_verify,
                headers=headers,
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

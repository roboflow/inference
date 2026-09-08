import hashlib
import os
import sys
from copy import deepcopy
from typing import Any, DefaultDict, Dict, List, Optional, Set, Union
from uuid import uuid4

import requests

# NOTE: This module is used in isolation, no imports from inference are allowed
# NOTE: Any change made to this file should be matched to changes in redis offloader

_OFFLINE_MODE_PROCESS_LATCH_ENV = "_ROBOFLOW_INFERENCE_OFFLINE_MODE_AT_PROCESS_START"
_offline_mode_process_state = sys.modules.get("_roboflow_inference_process_state")
if _offline_mode_process_state is not None and hasattr(
    _offline_mode_process_state, "offline_mode"
):
    OFFLINE_MODE = bool(_offline_mode_process_state.offline_mode)
    # The shared process state is authoritative. Re-publish it in case callers
    # mutated the private marker after package startup.
    os.environ[_OFFLINE_MODE_PROCESS_LATCH_ENV] = str(OFFLINE_MODE)
else:
    # This module is also reused by an isolated offloader. Snapshot its startup
    # environment without importing inference and creating an import cycle.
    # Publishing the private marker makes that snapshot survive module
    # re-execution and propagate to descendant processes.
    _inherited_offline_mode_value = os.getenv(_OFFLINE_MODE_PROCESS_LATCH_ENV)
    _offline_mode_variable_name = (
        _OFFLINE_MODE_PROCESS_LATCH_ENV
        if _inherited_offline_mode_value is not None
        else "OFFLINE_MODE"
    )
    _offline_mode_value = (
        _inherited_offline_mode_value
        if _inherited_offline_mode_value is not None
        else os.getenv("OFFLINE_MODE", "False")
    )
    _normalized_offline_mode_value = _offline_mode_value.lower()
    if _normalized_offline_mode_value == "true":
        OFFLINE_MODE = True
    elif _normalized_offline_mode_value == "false":
        OFFLINE_MODE = False
    else:
        raise ValueError(
            f"Expected {_offline_mode_variable_name} to be a boolean "
            "(true or false), "
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
    if "_usage_report_candidate" in d1 or "_usage_report_candidate" in d2:
        merged["_usage_report_candidate"] = (
            not d1 or d1.get("_usage_report_candidate") is True
        ) and (not d2 or d2.get("_usage_report_candidate") is True)
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
            dict(w) for w in workflow_payloads.values() if "processed_frames" in w
        ]
        try:
            for workflow_payload in complete_workflow_payloads:
                workflow_payload.pop("_legacy_delivery", None)
                workflow_payload.pop("_usage_report_candidate", None)
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


def get_usage_report_capability(
    api_key: str,
    api_plan_endpoint_url: str,
    ssl_verify: bool = True,
    extra_headers: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, Any]]:
    """A failed lookup is unknown, never permission to downgrade a report."""
    if OFFLINE_MODE:
        raise ConnectionError("Offline usage capability lookup")
    response = requests.get(
        api_plan_endpoint_url,
        headers={"Authorization": f"Bearer {api_key}", **(extra_headers or {})},
        verify=ssl_verify,
        timeout=1,
    )
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict):
        raise ValueError("Invalid usage capability response")
    capability = body.get("usage_report_protocol")
    if capability is None:
        return None
    if (
        not isinstance(capability, dict)
        or type(capability.get("version")) is not int
        or capability.get("version") != 1
        or not isinstance(capability.get("workspace_id"), str)
        or not capability["workspace_id"]
        or not isinstance(capability.get("ownership_fingerprint"), str)
        or len(capability["ownership_fingerprint"]) != 64
        or any(c not in "0123456789abcdef" for c in capability["ownership_fingerprint"])
    ):
        raise ValueError("Invalid usage report capability")
    return capability


def prepare_usage_reports(
    resource_payloads: ResourceUsage, capability: Dict[str, Any]
) -> ResourceUsage:
    reports = {}
    for payload in resource_payloads.values():
        if "processed_frames" not in payload:
            continue
        report = deepcopy(payload)
        report.pop("api_key_hash", None)
        report.pop("api_key", None)
        report.pop("_usage_report_candidate", None)
        report.pop("_legacy_delivery", None)
        stream_session = report.pop("stream_session_id", None)
        if stream_session:
            report["exec_session_id"] = stream_session
        report_id = str(uuid4())
        report.update(
            report_version=1,
            report_id=report_id,
            report_workspace_id=capability["workspace_id"],
            report_ownership_fingerprint=capability["ownership_fingerprint"],
        )
        reports[report_id] = report
    return reports


def send_usage_reports(
    reports: ResourceUsage,
    api_key: str,
    api_usage_endpoint_url: str,
    ssl_verify: bool = True,
    extra_headers: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    if OFFLINE_MODE or not reports:
        return {}
    body = [dict(deepcopy(report), api_key=api_key) for report in reports.values()]
    try:
        response = requests.post(
            api_usage_endpoint_url,
            json=body,
            verify=ssl_verify,
            headers={"Authorization": f"Bearer {api_key}", **(extra_headers or {})},
            timeout=1,
        )
        if response.status_code != 200:
            return {}
        return usage_report_outcomes(response.json(), set(reports))
    except Exception:
        return {}


def usage_report_outcomes(body: Any, sent_ids: Set[str]) -> Dict[str, str]:
    """Only explicit terminal outcomes retire a prepared report."""
    if not isinstance(body, dict) or not isinstance(
        body.get("usage_report_results"), list
    ):
        return {}
    outcomes = {}
    seen = set()
    for result in body["usage_report_results"]:
        if not isinstance(result, dict):
            return {}
        report_id = result.get("report_id")
        if (
            not isinstance(report_id, str)
            or report_id not in sent_ids
            or report_id in seen
        ):
            return {}
        seen.add(report_id)
        if result.get("status") == "accepted":
            outcomes[report_id] = "accepted"
        elif (
            result.get("status") == "rejected"
            and isinstance(result.get("reason"), str)
            and result["reason"]
        ):
            outcomes[report_id] = result["reason"]
    return outcomes

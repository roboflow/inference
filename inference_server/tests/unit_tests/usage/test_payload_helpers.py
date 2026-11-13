import json
from typing import Optional
from unittest import mock

import pytest

from inference_server import configuration
from inference_server.usage.delivery import send_usage_payload, ssl_verify_for_endpoint
from inference_server.usage.payload_helpers import (
    get_api_key_usage_containing_resource,
    merge_usage_dicts,
    zip_usage_payloads,
)


def usage_key(
    category: str,
    resource_id: str,
    *,
    billable: bool = True,
    outcome: str = "success",
    error_type: Optional[str] = None,
    error_status_code: Optional[int] = None,
    stream_session_id: Optional[str] = None,
) -> str:
    key = f"{category}:{resource_id}:billable={str(billable).lower()}:outcome={outcome}"
    if outcome == "error":
        key = f"{key}:error_type={error_type or 'unknown'}"
        if error_status_code is not None:
            key = f"{key}:error_status_code={error_status_code}"
    if stream_session_id:
        key = f"{key}:{stream_session_id}"
    return key


def test_merge_usage_dicts_raises_on_mismatched_resource_id():
    usage_payload_1 = {"resource_id": "some"}
    usage_payload_2 = {"resource_id": "other"}

    with pytest.raises(ValueError):
        merge_usage_dicts(d1=usage_payload_1, d2=usage_payload_2)


def test_merge_usage_dicts_merge_with_empty():
    usage_payload_1 = {
        "resource_id": "some",
        "api_key_hash": "some",
        "timestamp_start": 1721032989934855000,
        "timestamp_stop": 1721032989934855001,
        "processed_frames": 1,
        "source_duration": 1,
        "execution_duration": 0,
    }
    usage_payload_2 = {"resource_id": "some", "api_key_hash": "some"}

    assert merge_usage_dicts(d1=usage_payload_1, d2=usage_payload_2) == usage_payload_1
    assert merge_usage_dicts(d1=usage_payload_2, d2=usage_payload_1) == usage_payload_1


def test_merge_usage_dicts():
    usage_payload_1 = {
        "resource_id": "some",
        "api_key_hash": "some",
        "timestamp_start": 1721032989934855000,
        "timestamp_stop": 1721032989934855001,
        "processed_frames": 1,
        "source_duration": 1,
    }
    usage_payload_2 = {
        "resource_id": "some",
        "api_key_hash": "some",
        "timestamp_start": 1721032989934855002,
        "timestamp_stop": 1721032989934855003,
        "processed_frames": 1,
        "source_duration": 1,
    }

    assert merge_usage_dicts(d1=usage_payload_1, d2=usage_payload_2) == {
        "resource_id": "some",
        "api_key_hash": "some",
        "timestamp_start": 1721032989934855000,
        "timestamp_stop": 1721032989934855003,
        "processed_frames": 2,
        "source_duration": 2,
        "execution_duration": 0,
    }


def _row(details, *, frames=1, execution_duration=1.0):
    usage_row = {
        "resource_id": "workflow-1",
        "api_key_hash": "hash",
        "timestamp_start": 1,
        "timestamp_stop": 2,
        "processed_frames": frames,
        "source_duration": 0,
        "execution_duration": execution_duration,
        "resource_details": json.dumps(details),
    }

    return usage_row


def test_merge_usage_dicts_keeps_the_later_resource_details_as_legacy():
    first = _row({"billable": True, "source_info": "first", "only_first": 1})
    second = _row({"billable": True, "source_info": "second"})

    merged = merge_usage_dicts(first, second)

    assert merged["resource_details"] is second["resource_details"]
    assert merged["processed_frames"] == 2
    assert merged["execution_duration"] == 2.0


def test_zip_usage_payloads_sums_the_counters_of_one_resource_and_session():
    payloads = [
        {"hash": {"model:ds/1": _row({"billable": True}, frames=2)}},
        {"hash": {"model:ds/1": _row({"billable": True}, frames=3)}},
        {"hash": {"model:ds/2": _row({"billable": True}, frames=1)}},
    ]

    zipped = zip_usage_payloads(usage_payloads=payloads)

    assert len(zipped) == 1
    rows = zipped[0]["hash"]
    assert rows["model:ds/1"]["processed_frames"] == 5
    assert rows["model:ds/1"]["execution_duration"] == 2.0
    assert rows["model:ds/2"]["processed_frames"] == 1


def test_get_api_key_usage_containing_resource_with_no_payload_containing_api_key():
    usage_payloads = [
        {
            "": {
                "": {
                    "api_key_hash": "",
                    "resource_id": None,
                    "timestamp_start": 1721032989934855000,
                    "timestamp_stop": 1721032989934855001,
                    "processed_frames": 1,
                    "source_duration": 1,
                },
            },
        },
    ]

    api_key_usage_with_resource = get_api_key_usage_containing_resource(
        api_key_hash="fake", usage_payloads=usage_payloads
    )

    assert api_key_usage_with_resource is None


def test_get_api_key_usage_containing_resource_with_no_payload_containing_resource_for_given_api_key():
    usage_payloads = [
        {
            "fake_api1_hash": {
                "resource1": {
                    "api_key_hash": "fake_api1_hash",
                    "resource_id": "resource1",
                    "timestamp_start": 1721032989934855000,
                    "timestamp_stop": 1721032989934855001,
                    "processed_frames": 1,
                    "source_duration": 1,
                },
            },
        },
        {
            "fake_api1_hash": {
                "resource2": {
                    "api_key_hash": "fake_api1_hash",
                    "resource_id": "resource2",
                    "timestamp_start": 1721032989934855002,
                    "timestamp_stop": 1721032989934855003,
                    "processed_frames": 1,
                    "source_duration": 1,
                },
            },
            "": {
                "": {
                    "api_key_hash": "",
                    "resource_id": None,
                    "timestamp_start": 1721032989934855002,
                    "timestamp_stop": 1721032989934855003,
                    "processed_frames": 1,
                    "source_duration": 1,
                },
            },
        },
    ]

    api_key_usage_with_resource = get_api_key_usage_containing_resource(
        api_key_hash="fake_api2_hash", usage_payloads=usage_payloads
    )

    assert api_key_usage_with_resource is None


def test_get_api_key_usage_containing_resource():
    usage_payloads = [
        {
            "fake_api1_hash": {
                "resource1": {
                    "api_key_hash": "fake_api1_hash",
                    "resource_id": "resource1",
                    "timestamp_start": 1721032989934855000,
                    "timestamp_stop": 1721032989934855001,
                    "processed_frames": 1,
                    "source_duration": 1,
                },
            },
        },
        {
            "fake_api2_hash": {
                "resource1": {
                    "api_key_hash": "fake_api2_hash",
                    "resource_id": "resource1",
                    "timestamp_start": 1721032989934855002,
                    "timestamp_stop": 1721032989934855003,
                    "processed_frames": 1,
                    "source_duration": 1,
                },
            },
        },
    ]

    api_key_usage_with_resource = get_api_key_usage_containing_resource(
        api_key_hash="fake_api2_hash", usage_payloads=usage_payloads
    )

    assert api_key_usage_with_resource == {
        "api_key_hash": "fake_api2_hash",
        "resource_id": "resource1",
        "timestamp_start": 1721032989934855002,
        "timestamp_stop": 1721032989934855003,
        "processed_frames": 1,
        "source_duration": 1,
    }


def test_zip_usage_payloads():
    dumped_usage_payloads = [
        {
            "fake_api1_hash": {
                "resource1": {
                    "api_key_hash": "fake_api1_hash",
                    "resource_id": "resource1",
                    "timestamp_start": 1721032989934855000,
                    "timestamp_stop": 1721032989934855001,
                    "processed_frames": 1,
                    "source_duration": 1,
                    "execution_duration": 1,
                },
                "resource2": {
                    "api_key_hash": "fake_api1_hash",
                    "resource_id": "resource2",
                    "timestamp_start": 1721032989934855000,
                    "timestamp_stop": 1721032989934855001,
                    "processed_frames": 1,
                    "source_duration": 1,
                    "execution_duration": 1,
                },
            },
            "fake_api2_hash": {
                "resource1": {
                    "api_key_hash": "fake_api2_hash",
                    "resource_id": "resource1",
                    "timestamp_start": 1721032989934856000,
                    "timestamp_stop": 1721032989934856001,
                    "processed_frames": 1,
                    "source_duration": 1,
                    "execution_duration": 1,
                },
                "resource2": {
                    "api_key_hash": "fake_api2_hash",
                    "resource_id": "resource2",
                    "timestamp_start": 1721032989934856000,
                    "timestamp_stop": 1721032989934856001,
                    "processed_frames": 1,
                    "source_duration": 1,
                    "execution_duration": 1,
                },
            },
        },
        {
            "fake_api1_hash": {
                "resource1": {
                    "api_key_hash": "fake_api1_hash",
                    "resource_id": "resource1",
                    "timestamp_start": 1721032989934855002,
                    "timestamp_stop": 1721032989934855003,
                    "processed_frames": 1,
                    "source_duration": 1,
                    "execution_duration": 2,
                },
                "resource3": {
                    "api_key_hash": "fake_api1_hash",
                    "resource_id": "resource3",
                    "timestamp_start": 1721032989934855000,
                    "timestamp_stop": 1721032989934855001,
                    "processed_frames": 1,
                    "source_duration": 1,
                    "execution_duration": 1,
                },
            },
        },
        {
            "fake_api2_hash": {
                "resource1": {
                    "api_key_hash": "fake_api2_hash",
                    "resource_id": "resource1",
                    "timestamp_start": 1721032989934856002,
                    "timestamp_stop": 1721032989934856003,
                    "processed_frames": 1,
                    "source_duration": 1,
                    "execution_duration": 3,
                },
                "resource3": {
                    "api_key_hash": "fake_api2_hash",
                    "resource_id": "resource3",
                    "timestamp_start": 1721032989934856000,
                    "timestamp_stop": 1721032989934856001,
                    "processed_frames": 1,
                    "source_duration": 1,
                    "execution_duration": 4,
                },
            },
        },
    ]

    zipped_usage_payloads = zip_usage_payloads(usage_payloads=dumped_usage_payloads)

    assert zipped_usage_payloads == [
        {
            "fake_api1_hash": {
                "resource1": {
                    "api_key_hash": "fake_api1_hash",
                    "resource_id": "resource1",
                    "timestamp_start": 1721032989934855000,
                    "timestamp_stop": 1721032989934855003,
                    "processed_frames": 2,
                    "source_duration": 2,
                    "execution_duration": 3,
                },
                "resource2": {
                    "api_key_hash": "fake_api1_hash",
                    "resource_id": "resource2",
                    "timestamp_start": 1721032989934855000,
                    "timestamp_stop": 1721032989934855001,
                    "processed_frames": 1,
                    "source_duration": 1,
                    "execution_duration": 1,
                },
                "resource3": {
                    "api_key_hash": "fake_api1_hash",
                    "resource_id": "resource3",
                    "timestamp_start": 1721032989934855000,
                    "timestamp_stop": 1721032989934855001,
                    "processed_frames": 1,
                    "source_duration": 1,
                    "execution_duration": 1,
                },
            },
            "fake_api2_hash": {
                "resource1": {
                    "api_key_hash": "fake_api2_hash",
                    "resource_id": "resource1",
                    "timestamp_start": 1721032989934856000,
                    "timestamp_stop": 1721032989934856003,
                    "processed_frames": 2,
                    "source_duration": 2,
                    "execution_duration": 4,
                },
                "resource2": {
                    "api_key_hash": "fake_api2_hash",
                    "resource_id": "resource2",
                    "timestamp_start": 1721032989934856000,
                    "timestamp_stop": 1721032989934856001,
                    "processed_frames": 1,
                    "source_duration": 1,
                    "execution_duration": 1,
                },
                "resource3": {
                    "api_key_hash": "fake_api2_hash",
                    "resource_id": "resource3",
                    "timestamp_start": 1721032989934856000,
                    "timestamp_stop": 1721032989934856001,
                    "processed_frames": 1,
                    "source_duration": 1,
                    "execution_duration": 4,
                },
            },
        },
    ]


def test_zip_usage_payloads_keeps_billing_and_outcome_buckets_separate():
    def make_payload(key, billable, error=None):
        resource_details = {"billable": billable}
        if error is not None:
            resource_details["error"] = error
        return {
            "fake_api_hash": {
                key: {
                    "api_key_hash": "fake_api_hash",
                    "resource_id": "workspace/model",
                    "category": "request",
                    "resource_details": json.dumps(resource_details),
                    "exec_session_id": "session-1",
                    "timestamp_start": 1,
                    "timestamp_stop": 2,
                    "processed_frames": 1,
                    "source_duration": 0,
                    "execution_duration": 0.5,
                }
            }
        }

    billable_key = usage_key("request", "workspace/model")
    non_billable_key = usage_key("request", "workspace/model", billable=False)
    error_key = usage_key("request", "workspace/model", billable=False, outcome="error")
    payloads = zip_usage_payloads(
        usage_payloads=[
            make_payload(billable_key, True),
            make_payload(non_billable_key, False),
            make_payload(error_key, False, error="request failed"),
        ]
    )

    assert len(payloads) == 1
    merged = payloads[0]["fake_api_hash"]
    assert set(merged) == {billable_key, non_billable_key, error_key}
    assert all(row["processed_frames"] == 1 for row in merged.values())
    assert all(row["execution_duration"] == 0.5 for row in merged.values())


def test_zip_usage_payloads_with_system_info_missing_resource_id_and_no_resource_id_was_collected():
    dumped_usage_payloads = [
        {
            "api1": {
                "": {
                    "api_key_hash": "api1",
                    "resource_id": "",
                    "timestamp_start": 1721032989934855000,
                    "is_gpu_available": False,
                    "python_version": "3.10.0",
                    "inference_version": "10.10.10",
                    "execution_duration": 0,
                },
            },
        },
        {
            "api2": {
                "resource1": {
                    "api_key_hash": "api2",
                    "resource_id": "resource1",
                    "timestamp_start": 1721032989934856002,
                    "timestamp_stop": 1721032989934856003,
                    "processed_frames": 1,
                    "source_duration": 1,
                    "execution_duration": 0,
                },
            },
        },
    ]

    zipped_usage_payloads = zip_usage_payloads(usage_payloads=dumped_usage_payloads)

    assert zipped_usage_payloads == [
        {
            "api2": {
                "resource1": {
                    "api_key_hash": "api2",
                    "resource_id": "resource1",
                    "timestamp_start": 1721032989934856002,
                    "timestamp_stop": 1721032989934856003,
                    "processed_frames": 1,
                    "source_duration": 1,
                    "execution_duration": 0,
                },
            },
        },
        {
            "api1": {
                "": {
                    "api_key_hash": "api1",
                    "resource_id": "",
                    "timestamp_start": 1721032989934855000,
                    "is_gpu_available": False,
                    "python_version": "3.10.0",
                    "inference_version": "10.10.10",
                    "execution_duration": 0,
                },
            },
        },
    ]


def test_zip_usage_payloads_with_system_info_missing_resource_id():
    dumped_usage_payloads = [
        {
            "api2": {
                "": {
                    "api_key_hash": "api2",
                    "resource_id": "",
                    "timestamp_start": 1721032989934855000,
                    "is_gpu_available": False,
                    "python_version": "3.10.0",
                    "inference_version": "10.10.10",
                },
            },
        },
        {
            "api2": {
                "fake:resource1": {
                    "api_key_hash": "api2",
                    "resource_id": "resource1",
                    "category": "fake",
                    "timestamp_start": 1721032989934856002,
                    "timestamp_stop": 1721032989934856003,
                    "processed_frames": 1,
                    "source_duration": 1,
                },
            },
        },
    ]

    zipped_usage_payloads = zip_usage_payloads(usage_payloads=dumped_usage_payloads)

    assert zipped_usage_payloads == [
        {
            "api2": {
                "fake:resource1": {
                    "api_key_hash": "api2",
                    "resource_id": "resource1",
                    "category": "fake",
                    "timestamp_start": 1721032989934855000,
                    "timestamp_stop": 1721032989934856003,
                    "processed_frames": 1,
                    "source_duration": 1,
                    "is_gpu_available": False,
                    "python_version": "3.10.0",
                    "inference_version": "10.10.10",
                    "execution_duration": 0,
                },
            },
        },
    ]


def test_zip_usage_payloads_with_system_info_missing_resource_id_and_api_key():
    dumped_usage_payloads = [
        {
            "api2": {
                "": {
                    "api_key_hash": "api2",
                    "resource_id": "",
                    "timestamp_start": 1721032989934855000,
                    "is_gpu_available": False,
                    "python_version": "3.10.0",
                    "inference_version": "10.10.10",
                },
            },
        },
        {
            "api2": {
                "fake:resource1": {
                    "api_key_hash": "api2",
                    "resource_id": "resource1",
                    "category": "fake",
                    "timestamp_start": 1721032989934856002,
                    "timestamp_stop": 1721032989934856003,
                    "processed_frames": 1,
                    "source_duration": 1,
                },
            },
        },
    ]

    zipped_usage_payloads = zip_usage_payloads(usage_payloads=dumped_usage_payloads)

    assert zipped_usage_payloads == [
        {
            "api2": {
                "fake:resource1": {
                    "api_key_hash": "api2",
                    "resource_id": "resource1",
                    "category": "fake",
                    "timestamp_start": 1721032989934855000,
                    "timestamp_stop": 1721032989934856003,
                    "processed_frames": 1,
                    "source_duration": 1,
                    "is_gpu_available": False,
                    "python_version": "3.10.0",
                    "inference_version": "10.10.10",
                    "execution_duration": 0,
                },
            },
        },
    ]


def test_zip_usage_payloads_with_different_exec_session_ids():
    dumped_usage_payloads = [
        {
            "fake_api1_hash": {
                "resource1": {
                    "api_key_hash": "fake_api1_hash",
                    "resource_id": "resource1",
                    "timestamp_start": 1721032989934855000,
                    "timestamp_stop": 1721032989934855001,
                    "processed_frames": 1,
                    "source_duration": 1,
                    "fps": 10,
                    "exec_session_id": "session_1",
                    "execution_duration": 0,
                },
                "resource2": {
                    "api_key_hash": "fake_api1_hash",
                    "resource_id": "resource2",
                    "timestamp_start": 1721032989934855000,
                    "timestamp_stop": 1721032989934855001,
                    "processed_frames": 1,
                    "exec_session_id": "session_1",
                    "execution_duration": 0,
                },
            },
            "fake_api2_hash": {
                "resource1": {
                    "api_key_hash": "fake_api2_hash",
                    "resource_id": "resource1",
                    "timestamp_start": 1721032989934856000,
                    "timestamp_stop": 1721032989934856001,
                    "processed_frames": 1,
                    "source_duration": 1,
                    "fps": 10,
                    "exec_session_id": "session_1",
                    "execution_duration": 0,
                },
                "resource2": {
                    "api_key_hash": "fake_api2_hash",
                    "resource_id": "resource2",
                    "timestamp_start": 1721032989934856000,
                    "timestamp_stop": 1721032989934856001,
                    "processed_frames": 1,
                    "exec_session_id": "session_1",
                    "execution_duration": 0,
                },
            },
        },
        {
            "fake_api1_hash": {
                "resource1": {
                    "api_key_hash": "fake_api1_hash",
                    "resource_id": "resource1",
                    "timestamp_start": 1721032989934855002,
                    "timestamp_stop": 1721032989934855003,
                    "processed_frames": 1,
                    "source_duration": 1,
                    "fps": 10,
                    "exec_session_id": "session_1",
                    "execution_duration": 0,
                },
                "resource3": {
                    "api_key_hash": "fake_api1_hash",
                    "resource_id": "resource3",
                    "timestamp_start": 1721032989934855002,
                    "timestamp_stop": 1721032989934855003,
                    "processed_frames": 1,
                    "exec_session_id": "session_1",
                    "execution_duration": 0,
                },
            },
        },
        {
            "fake_api2_hash": {
                "resource1": {
                    "api_key_hash": "fake_api2_hash",
                    "resource_id": "resource1",
                    "timestamp_start": 1721032989934856002,
                    "timestamp_stop": 1721032989934856003,
                    "processed_frames": 1,
                    "source_duration": 1,
                    "fps": 10,
                    "exec_session_id": "session_1",
                    "execution_duration": 0,
                },
                "resource3": {
                    "api_key_hash": "fake_api2_hash",
                    "resource_id": "resource3",
                    "timestamp_start": 1721032989934856002,
                    "timestamp_stop": 1721032989934856003,
                    "processed_frames": 1,
                    "exec_session_id": "session_1",
                    "execution_duration": 0,
                },
            },
        },
        {
            "fake_api1_hash": {
                "resource1": {
                    "api_key_hash": "fake_api1_hash",
                    "resource_id": "resource1",
                    "timestamp_start": 1721032989934855003,
                    "timestamp_stop": 1721032989934855004,
                    "processed_frames": 1,
                    "source_duration": 1,
                    "fps": 10,
                    "exec_session_id": "session_2",
                    "execution_duration": 0,
                },
                "resource2": {
                    "api_key_hash": "fake_api1_hash",
                    "resource_id": "resource2",
                    "timestamp_start": 1721032989934855003,
                    "timestamp_stop": 1721032989934855004,
                    "processed_frames": 1,
                    "exec_session_id": "session_2",
                    "execution_duration": 0,
                },
                "resource3": {
                    "api_key_hash": "fake_api1_hash",
                    "resource_id": "resource3",
                    "timestamp_start": 1721032989934855003,
                    "timestamp_stop": 1721032989934855004,
                    "processed_frames": 1,
                    "exec_session_id": "session_1",
                },
            },
            "fake_api2_hash": {
                "resource1": {
                    "api_key_hash": "fake_api2_hash",
                    "resource_id": "resource1",
                    "timestamp_start": 1721032989934856003,
                    "timestamp_stop": 1721032989934856004,
                    "processed_frames": 1,
                    "source_duration": 1,
                    "fps": 10,
                    "exec_session_id": "session_2",
                    "execution_duration": 0,
                },
                "resource2": {
                    "api_key_hash": "fake_api2_hash",
                    "resource_id": "resource2",
                    "timestamp_start": 1721032989934856003,
                    "timestamp_stop": 1721032989934856004,
                    "processed_frames": 1,
                    "exec_session_id": "session_2",
                    "execution_duration": 0,
                },
                "resource3": {
                    "api_key_hash": "fake_api2_hash",
                    "resource_id": "resource3",
                    "timestamp_start": 1721032989934856003,
                    "timestamp_stop": 1721032989934856004,
                    "processed_frames": 1,
                    "exec_session_id": "session_1",
                    "execution_duration": 0,
                },
            },
            "fake_api3_hash": {
                "resource1": {
                    "api_key_hash": "fake_api3_hash",
                    "resource_id": "resource1",
                    "timestamp_start": 1721032989934857003,
                    "timestamp_stop": 1721032989934857004,
                    "processed_frames": 1,
                    "source_duration": 1,
                    "fps": 10,
                    "exec_session_id": "session_1",
                    "execution_duration": 0,
                },
            },
        },
    ]

    zipped_usage_payloads = zip_usage_payloads(usage_payloads=dumped_usage_payloads)

    assert zipped_usage_payloads == [
        {
            "fake_api1_hash": {
                "resource1": {
                    "api_key_hash": "fake_api1_hash",
                    "resource_id": "resource1",
                    "timestamp_start": 1721032989934855000,
                    "timestamp_stop": 1721032989934855003,
                    "processed_frames": 2,
                    "source_duration": 2,
                    "fps": 10,
                    "exec_session_id": "session_1",
                    "execution_duration": 0,
                },
            },
            "fake_api2_hash": {
                "resource1": {
                    "api_key_hash": "fake_api2_hash",
                    "resource_id": "resource1",
                    "timestamp_start": 1721032989934856000,
                    "timestamp_stop": 1721032989934856003,
                    "processed_frames": 2,
                    "source_duration": 2,
                    "fps": 10,
                    "exec_session_id": "session_1",
                    "execution_duration": 0,
                },
            },
            "fake_api3_hash": {
                "resource1": {
                    "api_key_hash": "fake_api3_hash",
                    "resource_id": "resource1",
                    "timestamp_start": 1721032989934857003,
                    "timestamp_stop": 1721032989934857004,
                    "processed_frames": 1,
                    "source_duration": 1,
                    "fps": 10,
                    "exec_session_id": "session_1",
                    "execution_duration": 0,
                },
            },
        },
        {
            "fake_api1_hash": {
                "resource1": {
                    "api_key_hash": "fake_api1_hash",
                    "resource_id": "resource1",
                    "timestamp_start": 1721032989934855003,
                    "timestamp_stop": 1721032989934855004,
                    "processed_frames": 1,
                    "source_duration": 1,
                    "fps": 10,
                    "exec_session_id": "session_2",
                    "execution_duration": 0,
                },
            },
            "fake_api2_hash": {
                "resource1": {
                    "api_key_hash": "fake_api2_hash",
                    "resource_id": "resource1",
                    "timestamp_start": 1721032989934856003,
                    "timestamp_stop": 1721032989934856004,
                    "processed_frames": 1,
                    "source_duration": 1,
                    "fps": 10,
                    "exec_session_id": "session_2",
                    "execution_duration": 0,
                },
            },
        },
        {
            "fake_api1_hash": {
                "resource2": {
                    "api_key_hash": "fake_api1_hash",
                    "resource_id": "resource2",
                    "timestamp_start": 1721032989934855000,
                    "timestamp_stop": 1721032989934855001,
                    "processed_frames": 1,
                    "exec_session_id": "session_1",
                    "execution_duration": 0,
                },
                "resource3": {
                    "api_key_hash": "fake_api1_hash",
                    "resource_id": "resource3",
                    "timestamp_start": 1721032989934855002,
                    "timestamp_stop": 1721032989934855004,
                    "processed_frames": 2,
                    "exec_session_id": "session_1",
                    "execution_duration": 0,
                },
            },
            "fake_api2_hash": {
                "resource2": {
                    "api_key_hash": "fake_api2_hash",
                    "resource_id": "resource2",
                    "timestamp_start": 1721032989934856000,
                    "timestamp_stop": 1721032989934856001,
                    "processed_frames": 1,
                    "exec_session_id": "session_1",
                    "execution_duration": 0,
                },
                "resource3": {
                    "api_key_hash": "fake_api2_hash",
                    "resource_id": "resource3",
                    "timestamp_start": 1721032989934856002,
                    "timestamp_stop": 1721032989934856004,
                    "processed_frames": 2,
                    "exec_session_id": "session_1",
                    "execution_duration": 0,
                },
            },
        },
        {
            "fake_api1_hash": {
                "resource2": {
                    "api_key_hash": "fake_api1_hash",
                    "resource_id": "resource2",
                    "timestamp_start": 1721032989934855003,
                    "timestamp_stop": 1721032989934855004,
                    "processed_frames": 1,
                    "exec_session_id": "session_2",
                    "execution_duration": 0,
                },
            },
            "fake_api2_hash": {
                "resource2": {
                    "api_key_hash": "fake_api2_hash",
                    "resource_id": "resource2",
                    "timestamp_start": 1721032989934856003,
                    "timestamp_stop": 1721032989934856004,
                    "processed_frames": 1,
                    "exec_session_id": "session_2",
                    "execution_duration": 0,
                },
            },
        },
    ]


def test_zip_usage_payloads_keeps_stream_sessions_separate():
    def make_payload(key, ssid, frames, ts):
        return {
            "fake_api1_hash": {
                key: {
                    "api_key_hash": "fake_api1_hash",
                    "resource_id": "workflow-1",
                    "stream_session_id": ssid,
                    "exec_session_id": "session-1",
                    "timestamp_start": ts,
                    "timestamp_stop": ts + 1,
                    "processed_frames": frames,
                    "fps": 10,
                    "source_duration": frames / 10,
                    "execution_duration": 1,
                }
            }
        }

    dumped_usage_payloads = [
        make_payload(
            "workflows:workflow-1:stream-a", "stream-a", 2, 1721032989934855000
        ),
        make_payload(
            "workflows:workflow-1:stream-b", "stream-b", 3, 1721032989934856000
        ),
        make_payload(
            "workflows:workflow-1:stream-a", "stream-a", 5, 1721032989934857000
        ),
    ]

    zipped_usage_payloads = zip_usage_payloads(usage_payloads=dumped_usage_payloads)

    merged = {}
    for payload in zipped_usage_payloads:
        for resource_payloads in payload.values():
            for key, resource_usage in resource_payloads.items():
                assert key not in merged
                merged[key] = resource_usage
    assert merged["workflows:workflow-1:stream-a"]["processed_frames"] == 7
    assert merged["workflows:workflow-1:stream-a"]["stream_session_id"] == "stream-a"
    assert merged["workflows:workflow-1:stream-b"]["processed_frames"] == 3
    assert merged["workflows:workflow-1:stream-b"]["stream_session_id"] == "stream-b"


@mock.patch("inference_server.usage.delivery.requests.post")
def test_send_usage_payload_serializes_stream_sessions_as_exec_session_ids(
    post_mock,
):
    def make_payload(key, stream_id, frames):
        return {
            "fake_hash": {
                key: {
                    "api_key_hash": "fake_hash",
                    "resource_id": "workflow-1",
                    "stream_session_id": stream_id,
                    "exec_session_id": "process-session",
                    "processed_frames": frames,
                    "fps": 10,
                    "source_duration": frames / 10,
                    "execution_duration": 1,
                }
            }
        }

    payloads = zip_usage_payloads(
        usage_payloads=[
            make_payload("workflows:workflow-1:stream-a", "stream-a", 2),
            make_payload("workflows:workflow-1:stream-b", "stream-b", 3),
        ]
    )
    assert len(payloads) == 1
    post_mock.return_value.status_code = 200

    failed_hashes = send_usage_payload(
        payload=payloads[0],
        api_usage_endpoint_url="https://example.com/usage",
        hashes_to_api_keys={"fake_hash": "fake-api-key"},
    )

    assert failed_hashes == set()
    post_mock.assert_called_once()
    outbound_rows = post_mock.call_args.kwargs["json"]
    assert {row["exec_session_id"] for row in outbound_rows} == {
        "stream-a",
        "stream-b",
    }
    assert all("stream_session_id" not in row for row in outbound_rows)


@mock.patch("inference_server.usage.delivery.requests.post")
@mock.patch.object(configuration, "LEGACY_OFFLINE_MODE", True)
def test_send_usage_payload_posts_in_offline_mode_too(post_mock) -> None:
    payload = {
        "fake_hash": {
            "workflows:workflow-1": {
                "api_key_hash": "fake_hash",
                "resource_id": "workflow-1",
                "processed_frames": 1,
            }
        }
    }
    post_mock.return_value.status_code = 200

    failed_hashes = send_usage_payload(
        payload=payload,
        api_usage_endpoint_url="https://example.com/usage",
        hashes_to_api_keys={"fake_hash": "fake-api-key"},
    )

    assert failed_hashes == set()
    post_mock.assert_called_once()
    assert post_mock.call_args.kwargs["headers"]["Authorization"] == (
        "Bearer fake-api-key"
    )


@pytest.mark.parametrize("hashes_to_api_keys", [None, {}, {"other_hash": "key"}])
@mock.patch("inference_server.usage.delivery.requests.post")
def test_send_usage_payload_never_posts_with_the_hash_as_the_key(
    post_mock, hashes_to_api_keys
) -> None:
    payload = {
        "fake_hash": {
            "workflows:workflow-1": {
                "api_key_hash": "fake_hash",
                "resource_id": "workflow-1",
                "processed_frames": 1,
            }
        }
    }

    failed_hashes = send_usage_payload(
        payload=payload,
        api_usage_endpoint_url="https://example.com/usage",
        hashes_to_api_keys=hashes_to_api_keys,
    )

    assert failed_hashes == {"fake_hash"}
    post_mock.assert_not_called()


@mock.patch("inference_server.usage.delivery.requests.post")
def test_send_usage_payload_leaves_legacy_exec_session_ids(
    post_mock,
):
    payload = {
        "fake_hash": {
            "workflows:legacy": {
                "api_key_hash": "fake_hash",
                "resource_id": "legacy",
                "exec_session_id": "process-session",
                "processed_frames": 3,
                "source_duration": 0.3,
            },
            "workflows:tagged:stream-a": {
                "api_key_hash": "fake_hash",
                "resource_id": "tagged",
                "stream_session_id": "stream-a",
                "exec_session_id": "process-session",
                "processed_frames": 2,
                "source_duration": 0.2,
            },
        }
    }
    post_mock.return_value.status_code = 200

    failed_hashes = send_usage_payload(
        payload=payload,
        api_usage_endpoint_url="https://example.com/usage",
        hashes_to_api_keys={"fake_hash": "fake-api-key"},
    )

    assert failed_hashes == set()
    outbound_rows = post_mock.call_args.kwargs["json"]
    assert {row["resource_id"]: row["exec_session_id"] for row in outbound_rows} == {
        "legacy": "process-session",
        "tagged": "stream-a",
    }
    assert all("stream_session_id" not in row for row in outbound_rows)


@mock.patch("inference_server.usage.delivery.requests.post")
def test_send_usage_payload_retry_sends_identical_rows(post_mock):
    payload = {
        "fake_hash": {
            "workflows:workflow-1:stream-a": {
                "api_key_hash": "fake_hash",
                "resource_id": "workflow-1",
                "stream_session_id": "stream-a",
                "exec_session_id": "process-session",
                "processed_frames": 3,
                "source_duration": 0.3,
            }
        }
    }
    failed_response = mock.MagicMock(status_code=500)
    successful_response = mock.MagicMock(status_code=200)
    post_mock.side_effect = [failed_response, successful_response]

    first_result = send_usage_payload(
        payload=payload,
        api_usage_endpoint_url="https://example.com/usage",
        hashes_to_api_keys={"fake_hash": "fake-api-key"},
    )
    second_result = send_usage_payload(
        payload=payload,
        api_usage_endpoint_url="https://example.com/usage",
        hashes_to_api_keys={"fake_hash": "fake-api-key"},
    )

    assert first_result == {"fake_hash"}
    assert second_result == set()
    assert (
        post_mock.call_args_list[0].kwargs["json"]
        == post_mock.call_args_list[1].kwargs["json"]
    )


def test_ssl_verify_for_endpoint_judges_the_request_host_not_the_embedded_target():
    assert ssl_verify_for_endpoint("http://localhost:8080/usage") is False
    assert ssl_verify_for_endpoint("https://127.0.0.1/usage") is False
    assert ssl_verify_for_endpoint("https://api.roboflow.com/usage") is True
    assert (
        ssl_verify_for_endpoint(
            "https://gateway.local/proxy?url=http%3A%2F%2Flocalhost%3A9000%2Fusage"
        )
        is True
    )

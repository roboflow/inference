import asyncio
import hashlib
import json
import sys
from typing import Optional
from unittest import mock

import pytest

from inference.core.env import LAMBDA
from inference.core.version import __version__ as inference_version
from inference.core.workflows.errors import ClientCausedStepExecutionError
from inference.usage_tracking.payload_helpers import (
    get_api_key_usage_containing_resource,
    merge_usage_dicts,
    send_usage_payload,
    sha256_hash,
    zip_usage_payloads,
)
from inference_sdk.http.errors import HTTPCallErrorError


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


def test_create_empty_usage_dict(usage_collector_with_mocked_threads):
    # given
    usage_default_dict = usage_collector_with_mocked_threads.empty_usage_dict(
        exec_session_id="exec_session_id"
    )

    # when
    fake_api_key_hash = sha256_hash("fake_api_key", length=-1)
    usage_default_dict[fake_api_key_hash]["category:fake_id"]

    # then
    assert json.dumps(usage_default_dict) == json.dumps(
        {
            fake_api_key_hash: {
                "category:fake_id": {
                    "timestamp_start": None,
                    "timestamp_stop": None,
                    "exec_session_id": "exec_session_id",
                    "hostname": "",
                    "ip_address_hash": "",
                    "processed_frames": 0,
                    "fps": 0,
                    "source_duration": 0,
                    "category": "",
                    "resource_id": "",
                    "resource_details": "{}",
                    "hosted": LAMBDA,
                    "api_key_hash": "",
                    "is_gpu_available": False,
                    "api_key_hash": "",
                    "python_version": sys.version.split()[0],
                    "inference_version": inference_version,
                    "enterprise": False,
                    "execution_duration": 0,
                }
            }
        }
    )


def test_merge_usage_dicts_raises_on_mismatched_resource_id():
    # given
    usage_payload_1 = {"resource_id": "some"}
    usage_payload_2 = {"resource_id": "other"}

    with pytest.raises(ValueError):
        merge_usage_dicts(d1=usage_payload_1, d2=usage_payload_2)


def test_merge_usage_dicts_merge_with_empty():
    # given
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
    # given
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


def test_get_api_key_usage_containing_resource_with_no_payload_containing_api_key():
    # given
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

    # when
    api_key_usage_with_resource = get_api_key_usage_containing_resource(
        api_key_hash="fake", usage_payloads=usage_payloads
    )

    # then
    assert api_key_usage_with_resource is None


def test_get_api_key_usage_containing_resource_with_no_payload_containing_resource_for_given_api_key():
    # given
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

    # when
    api_key_usage_with_resource = get_api_key_usage_containing_resource(
        api_key_hash="fake_api2_hash", usage_payloads=usage_payloads
    )

    # then
    assert api_key_usage_with_resource is None


def test_get_api_key_usage_containing_resource():
    # given
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

    # when
    api_key_usage_with_resource = get_api_key_usage_containing_resource(
        api_key_hash="fake_api2_hash", usage_payloads=usage_payloads
    )

    # then
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

    # when
    zipped_usage_payloads = zip_usage_payloads(usage_payloads=dumped_usage_payloads)

    # then
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

    # when
    zipped_usage_payloads = zip_usage_payloads(usage_payloads=dumped_usage_payloads)

    # then
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

    # when
    zipped_usage_payloads = zip_usage_payloads(usage_payloads=dumped_usage_payloads)

    # then
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

    # when
    zipped_usage_payloads = zip_usage_payloads(usage_payloads=dumped_usage_payloads)

    # then
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

    # when
    zipped_usage_payloads = zip_usage_payloads(usage_payloads=dumped_usage_payloads)

    # then
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


def test_system_info_with_dedicated_deployment_id(usage_collector_with_mocked_threads):
    # given
    system_info = usage_collector_with_mocked_threads.system_info(
        ip_address="w.x.y.z",
        hostname="hostname01",
        dedicated_deployment_id="deployment01",
    )

    # then
    expected_system_info = {
        "hostname": f"deployment01:hostname01",
        "ip_address_hash": hashlib.sha256("w.x.y.z".encode()).hexdigest()[:5],
        "is_gpu_available": False,
    }
    for k, v in expected_system_info.items():
        assert system_info[k] == v


def test_system_info_with_no_dedicated_deployment_id(
    usage_collector_with_mocked_threads,
):
    # given
    system_info = usage_collector_with_mocked_threads.system_info(
        ip_address="w.x.y.z", hostname="hostname01"
    )

    # then
    expected_system_info = {
        "hostname": "5aacc",
        "ip_address_hash": hashlib.sha256("w.x.y.z".encode()).hexdigest()[:5],
        "is_gpu_available": False,
    }
    for k, v in expected_system_info.items():
        assert system_info[k] == v


def test_record_malformed_usage(usage_collector_with_mocked_threads):
    # given
    collector = usage_collector_with_mocked_threads

    # when
    collector.record_usage(
        source=None,
        category="model",
        frames=None,
        api_key="fake",
        resource_details=None,
        resource_id=None,
        inference_test_run=None,
        fps=None,
    )

    # then
    api_key = "fake"
    assert api_key in collector._usage
    key = usage_key("model", "None")
    assert key in collector._usage[api_key]
    assert collector._usage[api_key][key]["processed_frames"] == 0
    assert collector._usage[api_key][key]["fps"] == 0
    assert collector._usage[api_key][key]["source_duration"] == 0
    assert collector._usage[api_key][key]["category"] == "model"
    assert collector._usage[api_key][key]["resource_id"] == None
    assert collector._usage[api_key][key]["resource_details"] == "{}"
    assert collector._usage[api_key][key]["api_key_hash"] == api_key


def test_update_usage_payload_preserves_billable_on_cache_miss(
    usage_collector_with_mocked_threads,
):
    """Regression: _update_usage_payload used to overwrite the passed-in
    resource_details with {} when the (api_key, category, resource_id) slot
    was empty, dropping billable=True off outbound payloads. Downstream
    parsers then coerced the missing flag into billable=false.
    """
    # given
    collector = usage_collector_with_mocked_threads
    api_key = "fake-key"
    resource_id = "sam3/sam3_final"
    resource_details = {"billable": True, "source": "workflow-execution"}

    collector.record_system_info()

    # when — _update_usage_payload called without prior record_resource_details
    # (cache miss path)
    collector._update_usage_payload(
        source="test",
        category="model",
        frames=1,
        api_key=api_key,
        resource_details=resource_details,
        resource_id=resource_id,
    )

    # then — billable must still be in the serialized resource_details
    api_key_hash = collector._calculate_api_key_hash(api_key=api_key)
    recorded = collector._usage[api_key_hash][usage_key("model", resource_id)]
    parsed = json.loads(recorded["resource_details"])
    assert parsed.get("billable") is True
    assert parsed.get("source") == "workflow-execution"


@pytest.mark.parametrize(
    "billable_order",
    [
        (True, False),
        (False, True),
    ],
)
def test_record_usage_separates_billable_and_non_billable_buckets(
    usage_collector_with_mocked_threads,
    billable_order,
):
    collector = usage_collector_with_mocked_threads
    api_key = "fake-key"
    resource_id = "workspace/model"

    for billable in billable_order:
        collector.record_usage(
            source="test",
            category="request",
            frames=1,
            api_key=api_key,
            resource_details={"billable": billable},
            resource_id=resource_id,
            execution_duration=0.5,
        )

    usage = collector._usage[api_key]
    billable_key = usage_key("request", resource_id, billable=True)
    non_billable_key = usage_key("request", resource_id, billable=False)

    assert set(usage) == {billable_key, non_billable_key}
    assert usage[billable_key]["processed_frames"] == 1
    assert usage[billable_key]["execution_duration"] == 0.5
    assert json.loads(usage[billable_key]["resource_details"])["billable"] is True
    assert usage[non_billable_key]["processed_frames"] == 1
    assert usage[non_billable_key]["execution_duration"] == 0.5
    assert json.loads(usage[non_billable_key]["resource_details"])["billable"] is False


def test_record_usage_separates_success_and_error_buckets(
    usage_collector_with_mocked_threads,
):
    collector = usage_collector_with_mocked_threads
    api_key = "fake-key"
    resource_id = "workspace/model"

    collector.record_usage(
        source="test",
        category="request",
        api_key=api_key,
        resource_details={"billable": False},
        resource_id=resource_id,
    )
    collector.record_usage(
        source="test",
        category="request",
        api_key=api_key,
        resource_details={"billable": False, "error": "request failed"},
        resource_id=resource_id,
    )

    usage = collector._usage[api_key]
    success_key = usage_key("request", resource_id, billable=False)
    error_key = usage_key("request", resource_id, billable=False, outcome="error")

    assert set(usage) == {success_key, error_key}
    assert "error" not in json.loads(usage[success_key]["resource_details"])
    assert json.loads(usage[error_key]["resource_details"])["error"] == (
        "request failed"
    )


def test_record_usage_bounds_unstructured_errors_to_generic_bucket(
    usage_collector_with_mocked_threads,
):
    collector = usage_collector_with_mocked_threads
    api_key = "fake-key"
    resource_id = "workspace/model"

    for error in ("WorkflowSyntaxError: bad step", "CudaOOMError: out of memory"):
        collector.record_usage(
            source="test",
            category="request",
            api_key=api_key,
            resource_details={"billable": True, "error": error},
            resource_id=resource_id,
        )

    usage = collector._usage[api_key]
    syntax_key = usage_key(
        "request",
        resource_id,
        billable=True,
        outcome="error",
        error_type="unknown",
    )

    assert set(usage) == {syntax_key}
    assert usage[syntax_key]["processed_frames"] == 2
    assert json.loads(usage[syntax_key]["resource_details"])["error"] in {
        "WorkflowSyntaxError: bad step",
        "CudaOOMError: out of memory",
    }


def test_record_usage_separates_structured_error_types_into_own_buckets(
    usage_collector_with_mocked_threads,
):
    collector = usage_collector_with_mocked_threads
    api_key = "fake-key"
    resource_id = "workspace/model"

    for error_type in ("WorkflowSyntaxError", "CudaOOMError"):
        collector.record_usage(
            source="test",
            category="request",
            api_key=api_key,
            resource_details={
                "billable": True,
                "error": f"{error_type}: request failed",
                "error_type": error_type,
            },
            resource_id=resource_id,
        )

    usage = collector._usage[api_key]
    syntax_key = usage_key(
        "request",
        resource_id,
        billable=True,
        outcome="error",
        error_type="WorkflowSyntaxError",
    )
    oom_key = usage_key(
        "request",
        resource_id,
        billable=True,
        outcome="error",
        error_type="CudaOOMError",
    )

    assert set(usage) == {syntax_key, oom_key}
    assert all(row["processed_frames"] == 1 for row in usage.values())


def test_record_usage_bounds_invalid_error_metadata_to_generic_bucket(
    usage_collector_with_mocked_threads,
):
    collector = usage_collector_with_mocked_threads
    api_key = "fake-key"
    resource_id = "workspace/model"
    collector.record_usage(
        source="test",
        category="request",
        api_key=api_key,
        resource_details={
            "billable": True,
            "error": "request failed",
            "error_type": "dynamic error message with spaces",
            "error_status_code": 999,
        },
        resource_id=resource_id,
    )

    key = usage_key(
        "request",
        resource_id,
        billable=True,
        outcome="error",
        error_type="unknown",
    )
    assert set(collector._usage[api_key]) == {key}
    details = json.loads(collector._usage[api_key][key]["resource_details"])
    assert details["error_type"] == "unknown"
    assert "error_status_code" not in details


def test_record_usage_normalizes_error_metadata_before_deriving_resource_id(
    usage_collector_with_mocked_threads,
):
    collector = usage_collector_with_mocked_threads
    api_key = "fake-key"
    resource_details = {
        "billable": True,
        "error": "request failed",
        "error_type": "dynamic error message with spaces",
        "error_status_code": 999,
    }
    normalized_resource_details = {
        "billable": True,
        "error": "request failed",
        "error_type": "unknown",
    }
    expected_resource_id = collector._calculate_resource_hash(
        normalized_resource_details
    )

    collector.record_resource_details(
        category="request",
        resource_details=resource_details,
        api_key=api_key,
    )
    collector.record_usage(
        source="test",
        category="request",
        api_key=api_key,
        resource_details=resource_details,
    )

    key = usage_key(
        "request",
        expected_resource_id,
        billable=True,
        outcome="error",
        error_type="unknown",
    )
    assert set(collector._usage[api_key]) == {key}
    assert len(collector._resource_details[api_key]) == 1
    assert json.loads(collector._usage[api_key][key]["resource_details"]) == (
        normalized_resource_details
    )


def test_resource_details_cache_separates_billing_partitions(
    usage_collector_with_mocked_threads,
):
    collector = usage_collector_with_mocked_threads
    api_key = "fake-key"
    resource_id = "workspace/model"
    collector.record_system_info()

    collector.record_resource_details(
        category="request",
        resource_details={"billable": False, "source_info": "non-billable"},
        resource_id=resource_id,
        api_key=api_key,
    )
    collector.record_resource_details(
        category="request",
        resource_details={"billable": True, "source_info": "billable"},
        resource_id=resource_id,
        api_key=api_key,
    )

    # Simulate the non-billable request reading after the billable request wrote
    # its cache entry. It must still use its own partition and request metadata.
    collector._update_usage_payload(
        source="test",
        category="request",
        api_key=api_key,
        resource_details={"billable": False, "source_info": "non-billable"},
        resource_id=resource_id,
    )

    assert len(collector._resource_details[api_key]) == 2
    key = usage_key("request", resource_id, billable=False)
    details = json.loads(collector._usage[api_key][key]["resource_details"])
    assert details == {"billable": False, "source_info": "non-billable"}


def test_current_request_details_override_cached_values(
    usage_collector_with_mocked_threads,
):
    collector = usage_collector_with_mocked_threads
    api_key = "fake-key"
    resource_id = "workspace/model"
    collector.record_system_info()
    collector.record_resource_details(
        category="request",
        resource_details={"billable": True, "source_info": "cached"},
        resource_id=resource_id,
        api_key=api_key,
    )

    collector._update_usage_payload(
        source="test",
        category="request",
        api_key=api_key,
        resource_details={"billable": True, "source_info": "current"},
        resource_id=resource_id,
    )

    key = usage_key("request", resource_id)
    details = json.loads(collector._usage[api_key][key]["resource_details"])
    assert details["source_info"] == "current"


def test_record_usage_with_exception(usage_collector_with_mocked_threads):
    # given
    usage_collector = usage_collector_with_mocked_threads

    @usage_collector(category="model")
    def test_func(api_key="test_key"):
        raise Exception("test exception")

    # when
    with pytest.raises(Exception, match="test exception"):
        test_func()

    # then
    assert len(usage_collector._usage) == 1
    key = usage_key(
        "model",
        "unknown",
        billable=True,
        outcome="error",
        error_type="Exception",
    )
    details = json.loads(usage_collector._usage["test_key"][key]["resource_details"])
    assert details["billable"] is True
    assert details["error"] == "Exception: test exception"
    assert details["error_type"] == "Exception"
    assert "error_status_code" not in details


@pytest.mark.parametrize("countinference", [True, False])
def test_request_exception_preserves_countinference_intent(
    usage_collector_with_mocked_threads,
    countinference,
):
    usage_collector = usage_collector_with_mocked_threads

    class FakeRequest:
        api_key = "test_key"
        model_id = "workspace/model"

    @usage_collector(category="request")
    def test_func(request, countinference=True, api_key="test_key"):
        raise RuntimeError("request failure")

    with pytest.raises(RuntimeError, match="request failure"):
        test_func(FakeRequest(), countinference=countinference)

    key = usage_key(
        "request",
        "workspace/model",
        billable=countinference,
        outcome="error",
        error_type="RuntimeError",
    )
    details = json.loads(usage_collector._usage["test_key"][key]["resource_details"])
    assert details["billable"] is countinference
    assert details["error"] == "RuntimeError: request failure"
    assert details["error_type"] == "RuntimeError"


def test_returned_http_error_response_records_error_outcome(
    usage_collector_with_mocked_threads,
):
    usage_collector = usage_collector_with_mocked_threads

    class ErrorResponse:
        status_code = 422

    @usage_collector(category="model")
    def test_func(api_key="test_key"):
        return ErrorResponse()

    response = test_func()

    assert response.status_code == 422
    key = usage_key(
        "model",
        "unknown",
        billable=True,
        outcome="error",
        error_type="HTTPResponseError",
        error_status_code=422,
    )
    details = json.loads(usage_collector._usage["test_key"][key]["resource_details"])
    assert details["billable"] is True
    assert details["error"] == "HTTPResponseError422: response returned status 422"
    assert details["error_type"] == "HTTPResponseError"
    assert details["error_status_code"] == 422


def test_wrapped_remote_http_errors_are_partitioned_by_status_code(
    usage_collector_with_mocked_threads,
):
    usage_collector = usage_collector_with_mocked_threads

    @usage_collector(category="request")
    def test_func(status_code, api_key="test_key"):
        inner_error = HTTPCallErrorError(
            description="remote request failed",
            status_code=status_code,
            api_message=None,
        )
        raise ClientCausedStepExecutionError(
            block_id="model",
            status_code=status_code,
            public_message="remote workflow step failed",
            context="workflow_execution | step_execution",
            inner_error=inner_error,
        )

    for status_code in (402, 403):
        with pytest.raises(ClientCausedStepExecutionError):
            test_func(status_code)

    usage = usage_collector._usage["test_key"]
    payment_required_key = usage_key(
        "request",
        "unknown",
        billable=True,
        outcome="error",
        error_type="HTTPCallErrorError",
        error_status_code=402,
    )
    forbidden_key = usage_key(
        "request",
        "unknown",
        billable=True,
        outcome="error",
        error_type="HTTPCallErrorError",
        error_status_code=403,
    )

    assert set(usage) == {payment_required_key, forbidden_key}
    assert (
        json.loads(usage[payment_required_key]["resource_details"])["error_status_code"]
        == 402
    )
    assert (
        json.loads(usage[forbidden_key]["resource_details"])["error_status_code"] == 403
    )


def test_async_exception_records_error_outcome(usage_collector_with_mocked_threads):
    usage_collector = usage_collector_with_mocked_threads
    usage_collector._async_lock = None

    @usage_collector(category="model")
    async def test_func(api_key="test_key"):
        raise RuntimeError("async failure")

    with pytest.raises(RuntimeError, match="async failure"):
        asyncio.run(test_func())

    key = usage_key(
        "model",
        "unknown",
        billable=True,
        outcome="error",
        error_type="RuntimeError",
    )
    details = json.loads(usage_collector._usage["test_key"][key]["resource_details"])
    assert details["billable"] is True
    assert details["error"] == "RuntimeError: async failure"
    assert details["error_type"] == "RuntimeError"


def test_record_usage_with_exception_on_GCP(usage_collector_with_mocked_threads):
    # given
    usage_collector = usage_collector_with_mocked_threads

    @usage_collector(category="model")
    def test_func(api_key="test_key"):
        raise Exception("test exception")

    # when
    with mock.patch("inference.usage_tracking.collector.GCP_SERVERLESS", True):
        with pytest.raises(Exception, match="test exception"):
            test_func()

    # then
    assert len(usage_collector._usage) == 1
    assert "test_key" in usage_collector._usage
    key = usage_key(
        "model",
        "unknown",
        billable=True,
        outcome="error",
        error_type="Exception",
    )
    assert key in usage_collector._usage["test_key"]
    assert (
        json.loads(usage_collector._usage["test_key"][key]["resource_details"]).get(
            "error"
        )
        == "Exception: test exception"
    )


class TestComputeExecutionDuration:
    def test_no_gcp_serverless_returns_raw(self, usage_collector_with_mocked_threads):
        with mock.patch("inference.usage_tracking.collector.GCP_SERVERLESS", False):
            result = usage_collector_with_mocked_threads._compute_execution_duration(
                0.0, 0.05
            )
        assert result == 0.05

    def test_gcp_serverless_applies_floor_by_default(
        self, usage_collector_with_mocked_threads
    ):
        """When GCP_SERVERLESS=True and apply_duration_minimum is None (SDK not installed),
        the 100ms floor should apply for backwards compatibility."""
        with mock.patch("inference.usage_tracking.collector.GCP_SERVERLESS", True):
            with mock.patch(
                "inference.usage_tracking.collector.apply_duration_minimum", None
            ):
                result = (
                    usage_collector_with_mocked_threads._compute_execution_duration(
                        0.0, 0.05
                    )
                )
        assert result == 0.1

    def test_gcp_serverless_with_duration_minimum_true_applies_floor(
        self, usage_collector_with_mocked_threads
    ):
        """When GCP_SERVERLESS=True and apply_duration_minimum=True (direct request),
        the 100ms floor should apply."""
        import contextvars

        cv = contextvars.ContextVar("test_apply_duration_minimum", default=False)
        cv.set(True)
        with mock.patch("inference.usage_tracking.collector.GCP_SERVERLESS", True):
            with mock.patch(
                "inference.usage_tracking.collector.apply_duration_minimum", cv
            ):
                result = (
                    usage_collector_with_mocked_threads._compute_execution_duration(
                        0.0, 0.05
                    )
                )
        assert result == 0.1

    def test_gcp_serverless_with_duration_minimum_false_skips_floor(
        self, usage_collector_with_mocked_threads
    ):
        """When GCP_SERVERLESS=True and apply_duration_minimum=False (remote execution),
        the 100ms floor should NOT apply."""
        import contextvars

        cv = contextvars.ContextVar("test_apply_duration_minimum", default=False)
        cv.set(False)
        with mock.patch("inference.usage_tracking.collector.GCP_SERVERLESS", True):
            with mock.patch(
                "inference.usage_tracking.collector.apply_duration_minimum", cv
            ):
                result = (
                    usage_collector_with_mocked_threads._compute_execution_duration(
                        0.0, 0.05
                    )
                )
        assert result == 0.05

    def test_gcp_serverless_floor_does_not_reduce_large_values(
        self, usage_collector_with_mocked_threads
    ):
        """When execution takes longer than 100ms, the floor has no effect."""
        import contextvars

        cv = contextvars.ContextVar("test_apply_duration_minimum", default=False)
        cv.set(True)
        with mock.patch("inference.usage_tracking.collector.GCP_SERVERLESS", True):
            with mock.patch(
                "inference.usage_tracking.collector.apply_duration_minimum", cv
            ):
                result = (
                    usage_collector_with_mocked_threads._compute_execution_duration(
                        0.0, 0.25
                    )
                )
        assert result == 0.25


def test_source_info_from_request_object_persisted_into_resource_details(
    usage_collector_with_mocked_threads,
):
    # given - a model whose infer_from_request receives the request object, the way
    # core models (e.g. SAM3) are decorated. source_info lives on the request, not
    # as a top-level kwarg.
    usage_collector = usage_collector_with_mocked_threads

    class FakeRequest:
        api_key = "test_key"
        source = "app"
        source_info = "smartpolySegmentImage"

    @usage_collector(category="model")
    def infer_from_request(request, api_key="test_key"):
        return "ok"

    # when
    infer_from_request(FakeRequest())

    # then
    row = usage_collector._usage["test_key"][usage_key("model", "unknown")]
    resource_details = json.loads(row["resource_details"])
    assert resource_details.get("source_info") == "smartpolySegmentImage"
    # source_info on the request object must NOT leak into roboflow_service_name
    assert row.get("roboflow_service_name") != "smartpolySegmentImage"


def test_env_service_name_preserved_alongside_source_info(
    usage_collector_with_mocked_threads,
):
    # given - a serverless deployment that stamps roboflow_service_name via
    # ROBOFLOW_INTERNAL_SERVICE_NAME. The feature tag (source_info) must be captured
    # WITHOUT clobbering the deployment identity (WHERE the inference ran).
    usage_collector = usage_collector_with_mocked_threads

    class FakeRequest:
        api_key = "test_key"
        source = "app"
        source_info = "smartpolySegmentImage"

    @usage_collector(category="model")
    def infer_from_request(request, api_key="test_key"):
        return "ok"

    # when
    with mock.patch(
        "inference.usage_tracking.collector.ROBOFLOW_INTERNAL_SERVICE_NAME",
        "async-serverless-gpu",
    ):
        usage_collector._usage = usage_collector.empty_usage_dict(
            exec_session_id="test"
        )
        infer_from_request(FakeRequest())

    # then
    row = usage_collector._usage["test_key"][usage_key("model", "unknown")]
    assert row["roboflow_service_name"] == "async-serverless-gpu"
    assert (
        json.loads(row["resource_details"]).get("source_info")
        == "smartpolySegmentImage"
    )


def test_source_info_nested_in_kwargs_persisted_into_resource_details(
    usage_collector_with_mocked_threads,
):
    # given - a model whose infer(self, image, **kwargs) collapses request fields
    # into a nested kwargs dict.
    usage_collector = usage_collector_with_mocked_threads

    @usage_collector(category="model")
    def infer(image=None, api_key="test_key", **kwargs):
        return "ok"

    # when
    infer(image="img", api_key="test_key", source_info="autolabelPreview")

    # then
    resource_details = json.loads(
        usage_collector._usage["test_key"][usage_key("model", "unknown")][
            "resource_details"
        ]
    )
    assert resource_details.get("source_info") == "autolabelPreview"


def test_external_source_info_not_persisted_into_resource_details(
    usage_collector_with_mocked_threads,
):
    # given - the default Query value "external" must not pollute resource_details
    usage_collector = usage_collector_with_mocked_threads

    @usage_collector(category="model")
    def infer(image=None, api_key="test_key", **kwargs):
        return "ok"

    # when
    infer(image="img", api_key="test_key", source_info="external")

    # then
    resource_details = json.loads(
        usage_collector._usage["test_key"][usage_key("model", "unknown")][
            "resource_details"
        ]
    )
    assert "source_info" not in resource_details


def test_record_usage_separates_concurrent_streams_by_stream_session_id(
    usage_collector_with_mocked_threads,
):
    """Two pipelines sharing an API key and a workflow must not aggregate into
    one usage entry: stream billing counts concurrent stream session ids, so
    collapsing them under-counts a device's cameras.
    """
    from inference.usage_tracking.stream_session import stream_session_id

    collector = usage_collector_with_mocked_threads
    api_key = "fake"
    resource_id = "workflow-1"

    token = stream_session_id.set("stream-a")
    try:
        collector.record_usage(
            source="rtsp://camera-1",
            category="workflows",
            frames=2,
            api_key=api_key,
            resource_id=resource_id,
            fps=10,
        )
        stream_session_id.set("stream-b")
        collector.record_usage(
            source="rtsp://camera-2",
            category="workflows",
            frames=3,
            api_key=api_key,
            resource_id=resource_id,
            fps=10,
        )
    finally:
        stream_session_id.reset(token)

    usage = collector._usage[api_key]
    key_a = usage_key("workflows", resource_id, stream_session_id="stream-a")
    key_b = usage_key("workflows", resource_id, stream_session_id="stream-b")
    assert key_a in usage
    assert key_b in usage
    assert usage_key("workflows", resource_id) not in usage
    entry_a = usage[key_a]
    entry_b = usage[key_b]
    assert entry_a["stream_session_id"] == "stream-a"
    assert entry_b["stream_session_id"] == "stream-b"
    assert entry_a["processed_frames"] == 2
    assert entry_b["processed_frames"] == 3


def test_record_usage_without_stream_session_id_keeps_legacy_key(
    usage_collector_with_mocked_threads,
):
    from inference.usage_tracking.stream_session import stream_session_id

    collector = usage_collector_with_mocked_threads
    assert stream_session_id.get() is None

    collector.record_usage(
        source="img.jpg",
        category="workflows",
        frames=1,
        api_key="fake",
        resource_id="workflow-1",
    )

    usage = collector._usage["fake"]
    key = usage_key("workflows", "workflow-1")
    assert key in usage
    assert "stream_session_id" not in usage[key]


def test_stream_session_id_does_not_leak_across_threads():
    from threading import Thread

    from inference.usage_tracking.stream_session import stream_session_id

    seen_in_thread = []

    def pipeline_thread():
        stream_session_id.set("thread-local-stream")
        seen_in_thread.append(stream_session_id.get())

    thread = Thread(target=pipeline_thread)
    thread.start()
    thread.join()

    assert seen_in_thread == ["thread-local-stream"]
    assert stream_session_id.get() is None


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


@mock.patch("inference.usage_tracking.payload_helpers.requests.post")
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


@mock.patch("inference.usage_tracking.payload_helpers.requests.post")
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


@mock.patch("inference.usage_tracking.payload_helpers.requests.post")
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

import hashlib
import json

import pytest

from inference.core.env import LAMBDA
from inference.usage_tracking.collector import UsageCollector


def test_create_empty_usage_dict():
    # given
    usage_default_dict = UsageCollector.empty_usage_dict(
        exec_session_id="exec_session_id"
    )

    # when
    fake_api_key_hash = UsageCollector._hash("fake_api_key")
    usage_default_dict[fake_api_key_hash]["category:fake_id"]

    # then
    assert json.dumps(usage_default_dict) == json.dumps(
        {
            fake_api_key_hash: {
                "category:fake_id": {
                    "timestamp_start": None,
                    "timestamp_stop": None,
                    "exec_session_id": "exec_session_id",
                    "processed_frames": 0,
                    "fps": 0,
                    "source_duration": 0,
                    "category": "",
                    "resource_id": "",
                    "hosted": LAMBDA,
                    "api_key_hash": "",
                    "enterprise": False,
                }
            }
        }
    )


def test_merge_usage_dicts_raises_on_mismatched_resource_id():
    # given
    usage_payload_1 = {"resource_id": "some"}
    usage_payload_2 = {"resource_id": "other"}

    with pytest.raises(ValueError):
        UsageCollector._merge_usage_dicts(d1=usage_payload_1, d2=usage_payload_2)


def test_merge_usage_dicts_merge_with_empty():
    # given
    usage_payload_1 = {
        "resource_id": "some",
        "api_key_hash": "some",
        "timestamp_start": 1721032989934855000,
        "timestamp_stop": 1721032989934855001,
        "processed_frames": 1,
        "source_duration": 1,
    }
    usage_payload_2 = {"resource_id": "some", "api_key_hash": "some"}

    assert (
        UsageCollector._merge_usage_dicts(d1=usage_payload_1, d2=usage_payload_2)
        == usage_payload_1
    )
    assert (
        UsageCollector._merge_usage_dicts(d1=usage_payload_2, d2=usage_payload_1)
        == usage_payload_1
    )


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

    assert UsageCollector._merge_usage_dicts(
        d1=usage_payload_1, d2=usage_payload_2
    ) == {
        "resource_id": "some",
        "api_key_hash": "some",
        "timestamp_start": 1721032989934855000,
        "timestamp_stop": 1721032989934855003,
        "processed_frames": 2,
        "source_duration": 2,
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
    api_key_usage_with_resource = UsageCollector._get_api_key_usage_containing_resource(
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
    api_key_usage_with_resource = UsageCollector._get_api_key_usage_containing_resource(
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
    api_key_usage_with_resource = UsageCollector._get_api_key_usage_containing_resource(
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
                },
                "resource2": {
                    "api_key_hash": "fake_api1_hash",
                    "resource_id": "resource2",
                    "timestamp_start": 1721032989934855000,
                    "timestamp_stop": 1721032989934855001,
                    "processed_frames": 1,
                    "source_duration": 1,
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
                },
                "resource2": {
                    "api_key_hash": "fake_api2_hash",
                    "resource_id": "resource2",
                    "timestamp_start": 1721032989934856000,
                    "timestamp_stop": 1721032989934856001,
                    "processed_frames": 1,
                    "source_duration": 1,
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
                },
                "resource3": {
                    "api_key_hash": "fake_api1_hash",
                    "resource_id": "resource3",
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
                    "timestamp_start": 1721032989934856002,
                    "timestamp_stop": 1721032989934856003,
                    "processed_frames": 1,
                    "source_duration": 1,
                },
                "resource3": {
                    "api_key_hash": "fake_api2_hash",
                    "resource_id": "resource3",
                    "timestamp_start": 1721032989934856000,
                    "timestamp_stop": 1721032989934856001,
                    "processed_frames": 1,
                    "source_duration": 1,
                },
            },
        },
    ]

    # when
    zipped_usage_payloads = UsageCollector._zip_usage_payloads(
        usage_payloads=dumped_usage_payloads
    )

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
                },
                "resource2": {
                    "api_key_hash": "fake_api1_hash",
                    "resource_id": "resource2",
                    "timestamp_start": 1721032989934855000,
                    "timestamp_stop": 1721032989934855001,
                    "processed_frames": 1,
                    "source_duration": 1,
                },
                "resource3": {
                    "api_key_hash": "fake_api1_hash",
                    "resource_id": "resource3",
                    "timestamp_start": 1721032989934855000,
                    "timestamp_stop": 1721032989934855001,
                    "processed_frames": 1,
                    "source_duration": 1,
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
                },
                "resource2": {
                    "api_key_hash": "fake_api2_hash",
                    "resource_id": "resource2",
                    "timestamp_start": 1721032989934856000,
                    "timestamp_stop": 1721032989934856001,
                    "processed_frames": 1,
                    "source_duration": 1,
                },
                "resource3": {
                    "api_key_hash": "fake_api2_hash",
                    "resource_id": "resource3",
                    "timestamp_start": 1721032989934856000,
                    "timestamp_stop": 1721032989934856001,
                    "processed_frames": 1,
                    "source_duration": 1,
                },
            },
        },
    ]


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
                },
            },
        },
    ]

    # when
    zipped_usage_payloads = UsageCollector._zip_usage_payloads(
        usage_payloads=dumped_usage_payloads
    )

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
    zipped_usage_payloads = UsageCollector._zip_usage_payloads(
        usage_payloads=dumped_usage_payloads
    )

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
                },
            },
        },
    ]


def test_zip_usage_payloads_with_system_info_missing_resource_id_and_api_key():
    dumped_usage_payloads = [
        {
            "": {
                "": {
                    "api_key_hash": "",
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
    zipped_usage_payloads = UsageCollector._zip_usage_payloads(
        usage_payloads=dumped_usage_payloads
    )

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
                },
            },
        },
    ]


def test_system_info():
    # given
    system_info = UsageCollector.system_info(
        exec_session_id="exec_session_id", time_ns=1, ip_address="w.x.y.z"
    )

    # then
    expected_system_info = {
        "timestamp_start": 1,
        "exec_session_id": "exec_session_id",
        "ip_address_hash": hashlib.sha256("w.x.y.z".encode()).hexdigest()[:5],
        "api_key_hash": "",
        "is_gpu_available": False,
    }
    for k, v in expected_system_info.items():
        assert system_info[k] == v

import hashlib
import json
import sys
from unittest import mock

import pytest

from inference.core.env import LAMBDA
from inference.core.version import __version__ as inference_version
from inference.usage_tracking.payload_helpers import (
    get_api_key_usage_containing_resource,
    merge_usage_dicts,
    sha256_hash,
    zip_usage_payloads,
)


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
    assert "model:None" in collector._usage[api_key]
    assert collector._usage[api_key]["model:None"]["processed_frames"] == 0
    assert collector._usage[api_key]["model:None"]["fps"] == 0
    assert collector._usage[api_key]["model:None"]["source_duration"] == 0
    assert collector._usage[api_key]["model:None"]["category"] == "model"
    assert collector._usage[api_key]["model:None"]["resource_id"] == None
    assert collector._usage[api_key]["model:None"]["resource_details"] == "{}"
    assert collector._usage[api_key]["model:None"]["api_key_hash"] == api_key


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
    assert "model:unknown" in usage_collector._usage["test_key"]
    assert (
        json.loads(
            usage_collector._usage["test_key"]["model:unknown"]["resource_details"]
        ).get("error")
        == "Exception: test exception"
    )

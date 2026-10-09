import contextvars
import hashlib
import json
import logging
import sys
import threading
from queue import Queue
from typing import Optional
from unittest import mock
from urllib.parse import parse_qs, urlparse

import pytest

from inference_sdk.config import execution_id
from inference_server import configuration
from inference_server.usage import collector as collector_module
from inference_server.usage.collector import UsageCollector
from inference_server.usage.payload_helpers import sha256_hash
from inference_server.usage.queues import RedisQueue, SQLiteQueue
from tests.unit_tests.usage.conftest import SYSTEM_INFO
from tests.unit_tests.usage.test_queues import FakeRedis

POST = "inference_server.usage.delivery.requests.post"


def usage_key(
    category: str,
    resource_id: str,
    *,
    billable: bool = True,
    outcome: str = "success",
    preview: bool = False,
    error_type: Optional[str] = None,
    error_status_code: Optional[int] = None,
    stream_session_id: Optional[str] = None,
) -> str:
    key = f"{category}:{resource_id}:billable={str(billable).lower()}:outcome={outcome}"
    if preview:
        key = f"{key}:preview=true"
    if outcome == "error":
        key = f"{key}:error_type={error_type or 'unknown'}"
        if error_status_code is not None:
            key = f"{key}:error_status_code={error_status_code}"
    if stream_session_id:
        key = f"{key}:{stream_session_id}"
    return key


def record(usage_collector, **overrides):
    arguments = {
        "api_key": "fake-key",
        "category": "request",
        "resource_id": "workspace/model",
        "resource_details": {},
        "frames": 1,
        "execution_duration": 0.5,
        "billable": True,
        **overrides,
    }
    usage_collector.record_usage(**arguments)


def queued_payloads(usage_collector):
    payloads = usage_collector._delivery.dump_queue()

    return payloads


def total_frames(payloads):
    frames = sum(
        row["processed_frames"]
        for payload in payloads
        for rows in payload.values()
        for row in rows.values()
    )

    return frames


@pytest.fixture
def stream_session(monkeypatch):
    variable = contextvars.ContextVar("stream_session_id", default=None)
    monkeypatch.setattr(collector_module, "_stream_session_id_var", variable)

    return variable


def test_create_empty_usage_dict(collector, monkeypatch):
    monkeypatch.setattr(
        collector_module,
        "_package_versions",
        lambda: {
            "inference_models_version": "1.2.3",
            "inference_model_manager_version": "0.5.2",
            "roboflow_workflows_version": None,
            "streamvision_version": None,
        },
    )
    usage_default_dict = collector.empty_usage_dict(exec_session_id="exec_session_id")

    fake_api_key_hash = sha256_hash("fake_api_key", length=-1)
    usage_default_dict[fake_api_key_hash]["category:fake_id"]

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
                    "hosted": False,
                    "api_key_hash": "",
                    "is_gpu_available": False,
                    "python_version": sys.version.split()[0],
                    "inference_version": "9.9.9",
                    "inference_models_version": "1.2.3",
                    "inference_model_manager_version": "0.5.2",
                    "roboflow_workflows_version": None,
                    "streamvision_version": None,
                    "enterprise": False,
                    "execution_duration": 0,
                    "megapixel_buckets": {},
                }
            }
        }
    )


def test_package_versions_report_none_for_packages_that_are_not_installed(
    monkeypatch,
):
    def version(name):
        if name == "inference-models":
            return "1.2.3"
        raise collector_module.importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(collector_module.importlib.metadata, "version", version)
    collector_module._package_versions.cache_clear()
    try:
        versions = collector_module._package_versions()
    finally:
        collector_module._package_versions.cache_clear()

    assert versions == {
        "inference_models_version": "1.2.3",
        "inference_model_manager_version": None,
        "roboflow_workflows_version": None,
        "streamvision_version": None,
    }


@pytest.mark.parametrize(
    "setting, value",
    [
        ("LAMBDA", True),
        ("GCP_SERVERLESS", True),
        ("DEDICATED_DEPLOYMENT_ID", "deployment01"),
        ("ROBOFLOW_INTERNAL_SERVICE_SECRET", "internal-secret"),
    ],
)
def test_rows_are_marked_hosted(collector, monkeypatch, setting, value):
    monkeypatch.setattr(configuration, setting, value)

    usage = collector.empty_usage_dict(exec_session_id="session")

    assert usage["hash"]["key"]["hosted"] is True


def test_rows_carry_the_internal_service_identity_of_the_environment(
    collector, monkeypatch
):
    monkeypatch.setattr(configuration, "ROBOFLOW_INTERNAL_SERVICE_SECRET", "secret")
    monkeypatch.setattr(configuration, "ROBOFLOW_INTERNAL_SERVICE_NAME", "service")

    usage = collector.empty_usage_dict(exec_session_id="session")

    assert usage["hash"]["key"]["roboflow_internal_secret"] == "secret"
    assert usage["hash"]["key"]["roboflow_service_name"] == "service"


@pytest.mark.parametrize("queue_size", [1, 2, 10])
def test_full_usage_queue_preserves_batches_across_repeated_saturation(
    collector, monkeypatch, queue_size
):
    class NonBlockingQueue(Queue):
        def put(self, item, block=True, timeout=None):
            super().put(item, block=False)

    monkeypatch.setattr(
        collector._delivery, "queue", NonBlockingQueue(maxsize=queue_size)
    )

    for cycle in range(2):
        expected = {}
        for index in range(4 * queue_size + 1):
            session = f"cycle-{cycle}-session-{index % (queue_size + 1)}"
            api_key = f"test-key-{index % 2}"
            fps = index % 2
            key = usage_key(
                "model", "test-model", outcome="error" if index % 3 else "success"
            )
            row = {
                "resource_id": "test-model",
                "api_key_hash": api_key,
                "exec_session_id": session,
                "fps": fps,
                "processed_frames": 1,
                "source_duration": 1,
                "execution_duration": 2,
            }
            identity = (api_key, key, session, bool(fps))
            expected[identity] = expected.get(identity, 0) + 1
            collector._delivery.enqueue({api_key: {key: row}})

        payloads = collector._delivery.dump_queue()
        assert collector._delivery.queue.empty()
        actual = {}
        for payload in payloads:
            for api_key, rows in payload.items():
                for key, row in rows.items():
                    identity = (api_key, key, row["exec_session_id"], bool(row["fps"]))
                    frames = row["processed_frames"]
                    assert row["source_duration"] == frames
                    assert row["execution_duration"] == 2 * frames
                    actual[identity] = actual.get(identity, 0) + frames
        assert actual == expected


def test_system_info_with_dedicated_deployment_id(collector):
    system_info = collector.system_info(
        ip_address="w.x.y.z",
        hostname="hostname01",
        dedicated_deployment_id="deployment01",
    )

    expected_system_info = {
        "hostname": "deployment01:hostname01",
        "ip_address_hash": hashlib.sha256("w.x.y.z".encode()).hexdigest()[:5],
        "is_gpu_available": False,
    }
    for k, v in expected_system_info.items():
        assert system_info[k] == v


def test_system_info_with_no_dedicated_deployment_id(collector):
    system_info = collector.system_info(ip_address="w.x.y.z", hostname="hostname01")

    expected_system_info = {
        "hostname": "5aacc",
        "ip_address_hash": hashlib.sha256("w.x.y.z".encode()).hexdigest()[:5],
        "is_gpu_available": False,
    }
    for k, v in expected_system_info.items():
        assert system_info[k] == v


def test_system_info_does_not_probe_network_in_offline_mode(collector):
    with mock.patch.object(configuration, "LEGACY_OFFLINE_MODE", True), mock.patch(
        "inference_server.usage.collector.socket.gethostbyname"
    ) as gethostbyname_mock, mock.patch(
        "inference_server.usage.collector.socket.socket"
    ) as socket_mock:
        system_info = collector.system_info(hostname="hostname01")

    assert (
        system_info["ip_address_hash"]
        == hashlib.sha256("127.0.0.1".encode()).hexdigest()[:5]
    )
    gethostbyname_mock.assert_not_called()
    socket_mock.assert_not_called()


def test_system_info_is_computed_once(collector):
    collector._system_info = {}
    with mock.patch.object(
        UsageCollector, "system_info", return_value=dict(SYSTEM_INFO)
    ) as system_info:
        record(collector)
        collector.record_system_info()
        collector._write_current_usage_to_queue()
        record(collector)
        collector.record_system_info()
        collector._write_current_usage_to_queue()

    system_info.assert_called_once()
    key = usage_key("request", "workspace/model")
    rows = [
        payload["fake-key"][key]
        for payload in queued_payloads(collector)
        if "fake-key" in payload
    ]
    assert len(rows) == 2
    for row in rows:
        assert row["hostname"] == "host1"
        assert row["ip_address_hash"] == "ab12c"
        assert row["is_gpu_available"] is False


def test_record_malformed_usage(collector):
    collector.record_usage(
        category="model",
        frames=None,
        api_key="fake",
        resource_details=None,
        resource_id=None,
        fps=None,
        execution_duration=0,
        billable=True,
    )

    api_key = "fake"
    assert api_key in collector._usage
    resource_id = sha256_hash(json.dumps({"billable": True}, sort_keys=True))
    key = usage_key("model", resource_id)
    assert key in collector._usage[api_key]
    assert collector._usage[api_key][key]["processed_frames"] == 0
    assert collector._usage[api_key][key]["fps"] == 0
    assert collector._usage[api_key][key]["source_duration"] == 0
    assert collector._usage[api_key][key]["category"] == "model"
    assert collector._usage[api_key][key]["resource_id"] == resource_id
    assert collector._usage[api_key][key]["resource_details"] == '{"billable": true}'
    assert collector._usage[api_key][key]["api_key_hash"] == api_key


def test_record_usage_records_in_offline_mode(collector):
    with mock.patch.object(configuration, "LEGACY_OFFLINE_MODE", True):
        record(collector, category="model", api_key="fake", frames=3)

    key = usage_key("model", "workspace/model")
    assert collector._usage["fake"][key]["processed_frames"] == 3


def test_offline_mode_collector_is_built_with_the_configured_queue(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(configuration, "LEGACY_OFFLINE_MODE", True)
    monkeypatch.setattr(configuration, "TELEMETRY_USE_PERSISTENT_QUEUE", True)

    usage_collector = UsageCollector(sqlite_db_file_path=tmp_path / "usage.db")

    assert isinstance(usage_collector._delivery.queue, SQLiteQueue)


def test_flush_posts_in_offline_mode(collector):
    record(collector, frames=2)

    with mock.patch.object(configuration, "LEGACY_OFFLINE_MODE", True), mock.patch(
        POST
    ) as post_mock:
        post_mock.return_value.status_code = 200
        collector.flush()

    post_mock.assert_called_once()
    assert post_mock.call_args.kwargs["json"][0]["processed_frames"] == 2
    assert queued_payloads(collector) == []


def usage_log_records(caplog):
    records = [
        record
        for record in caplog.records
        if record.name.startswith("inference_server.usage")
    ]

    return records


def test_an_unreachable_platform_is_logged_at_debug_only_and_keeps_the_rows(
    collector, caplog
):
    record(collector, frames=4)

    with caplog.at_level(logging.DEBUG, logger="inference_server.usage"), mock.patch(
        POST, side_effect=ConnectionError("https://api.example.com?api_key=secret")
    ):
        collector.flush()

    records = usage_log_records(caplog)
    assert records
    assert all(record.levelno == logging.DEBUG for record in records)
    assert any("ConnectionError" in record.getMessage() for record in records)
    assert "api_key=secret" not in caplog.text
    assert total_frames(queued_payloads(collector)) == 4


def test_a_failed_pass_logs_one_usage_line_per_failed_request(collector, caplog):
    record(collector, api_key="key-one", frames=1)
    record(collector, api_key="key-two", frames=1)

    with caplog.at_level(logging.DEBUG, logger="inference_server.usage"), mock.patch(
        POST, side_effect=ConnectionError("down")
    ) as post_mock:
        collector.flush()

    assert post_mock.call_count == 2
    assert len(usage_log_records(caplog)) == 2


@pytest.mark.parametrize("status_code", [401, 500, 503])
def test_a_rejected_usage_request_is_logged_at_debug_only(
    collector, caplog, status_code
):
    record(collector, frames=1)

    with caplog.at_level(logging.DEBUG, logger="inference_server.usage"), mock.patch(
        POST
    ) as post_mock:
        post_mock.return_value.status_code = status_code
        collector.flush()

    records = usage_log_records(caplog)
    assert records
    assert all(record.levelno == logging.DEBUG for record in records)
    assert any(str(status_code) in record.getMessage() for record in records)
    assert total_frames(queued_payloads(collector)) == 1


def test_record_usage_preserves_billable_and_details(collector):
    api_key = "fake-key"
    resource_id = "sam3/sam3_final"

    record(
        collector,
        category="model",
        api_key=api_key,
        resource_details={"source": "workflow-execution"},
        resource_id=resource_id,
    )

    recorded = collector._usage[api_key][usage_key("model", resource_id)]
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
    collector,
    billable_order,
):
    api_key = "fake-key"
    resource_id = "workspace/model"

    for billable in billable_order:
        record(collector, billable=billable, execution_duration=0.5)

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


def test_record_usage_separates_success_and_error_buckets(collector):
    api_key = "fake-key"
    resource_id = "workspace/model"

    record(collector, billable=False)
    record(collector, billable=False, resource_details={"error": "request failed"})

    usage = collector._usage[api_key]
    success_key = usage_key("request", resource_id, billable=False)
    error_key = usage_key("request", resource_id, billable=False, outcome="error")

    assert set(usage) == {success_key, error_key}
    assert "error" not in json.loads(usage[success_key]["resource_details"])
    assert json.loads(usage[error_key]["resource_details"])["error"] == (
        "request failed"
    )


def test_record_usage_bounds_unstructured_errors_to_generic_bucket(collector):
    api_key = "fake-key"
    resource_id = "workspace/model"

    for error in ("WorkflowSyntaxError: bad step", "CudaOOMError: out of memory"):
        record(collector, resource_details={"error": error})

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


def test_record_usage_separates_structured_error_types_into_own_buckets(collector):
    api_key = "fake-key"
    resource_id = "workspace/model"

    for error_type in ("WorkflowSyntaxError", "CudaOOMError"):
        record(
            collector,
            resource_details={"error": f"{error_type}: request failed"},
            error_type=error_type,
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


def test_record_usage_bounds_invalid_error_metadata_to_generic_bucket(collector):
    api_key = "fake-key"
    resource_id = "workspace/model"
    record(
        collector,
        resource_details={"error": "request failed"},
        error_type="dynamic error message with spaces",
        error_status_code=999,
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
    collector,
):
    api_key = "fake-key"
    normalized_resource_details = {
        "billable": True,
        "error": "request failed",
        "error_type": "unknown",
    }
    expected_resource_id = collector._calculate_resource_hash(
        normalized_resource_details
    )

    record(
        collector,
        resource_id="",
        resource_details={"error": "request failed"},
        error_type="dynamic error message with spaces",
        error_status_code=999,
    )

    key = usage_key(
        "request",
        expected_resource_id,
        billable=True,
        outcome="error",
        error_type="unknown",
    )
    assert set(collector._usage[api_key]) == {key}
    assert json.loads(collector._usage[api_key][key]["resource_details"]) == (
        normalized_resource_details
    )


def test_error_type_alone_marks_the_row_as_failed(collector):
    record(collector, error_type="FooError", error_status_code=503)

    key = usage_key(
        "request",
        "workspace/model",
        outcome="error",
        error_type="FooError",
        error_status_code=503,
    )
    details = json.loads(collector._usage["fake-key"][key]["resource_details"])
    assert details == {
        "billable": True,
        "error": "FooError",
        "error_type": "FooError",
        "error_status_code": 503,
    }
    assert collector._usage["fake-key"][key]["processed_frames"] == 1


def test_error_metadata_is_dropped_from_successful_rows(collector):
    record(collector, error_status_code=503)

    key = usage_key("request", "workspace/model")
    details = json.loads(collector._usage["fake-key"][key]["resource_details"])
    assert details == {"billable": True}


@pytest.mark.parametrize(
    "arguments, session, expected",
    [
        ({}, None, "request:r:billable=true:outcome=success"),
        ({"billable": False}, None, "request:r:billable=false:outcome=success"),
        (
            {"is_preview": True},
            None,
            "request:r:billable=true:outcome=success:preview=true",
        ),
        (
            {"billable": False, "is_preview": True},
            None,
            "request:r:billable=false:outcome=success:preview=true",
        ),
        (
            {"error_type": "FooError"},
            None,
            "request:r:billable=true:outcome=error:error_type=FooError",
        ),
        (
            {"error_type": "FooError", "error_status_code": 502},
            None,
            "request:r:billable=true:outcome=error:error_type=FooError"
            ":error_status_code=502",
        ),
        (
            {"error_type": "FooError", "error_status_code": 200},
            None,
            "request:r:billable=true:outcome=error:error_type=FooError",
        ),
        (
            {"error_type": "not a type"},
            None,
            "request:r:billable=true:outcome=error:error_type=unknown",
        ),
        (
            {"is_preview": True, "billable": False, "error_type": "FooError"},
            None,
            "request:r:billable=false:outcome=error:preview=true:error_type=FooError",
        ),
        ({}, "stream-a", "request:r:billable=true:outcome=success:stream-a"),
        (
            {"error_type": "FooError", "error_status_code": 500},
            "stream-a",
            "request:r:billable=true:outcome=error:error_type=FooError"
            ":error_status_code=500:stream-a",
        ),
    ],
)
def test_aggregation_key(collector, stream_session, arguments, session, expected):
    token = stream_session.set(session)
    try:
        record(collector, resource_id="r", **arguments)
    finally:
        stream_session.reset(token)

    assert set(collector._usage["fake-key"]) == {expected}


def test_preview_flag_is_written_when_set_or_already_present(collector):
    record(collector, resource_id="plain")
    record(collector, resource_id="preview", is_preview=True)
    record(collector, resource_id="explicit", resource_details={"is_preview": True})

    usage = collector._usage["fake-key"]
    plain = json.loads(usage[usage_key("request", "plain")]["resource_details"])
    preview = json.loads(
        usage[usage_key("request", "preview", preview=True)]["resource_details"]
    )
    explicit = json.loads(usage[usage_key("request", "explicit")]["resource_details"])
    assert "is_preview" not in plain
    assert preview["is_preview"] is True
    assert explicit["is_preview"] is False


def test_record_usage_separates_concurrent_streams_by_stream_session_id(
    collector, stream_session
):
    api_key = "fake"
    resource_id = "workflow-1"

    token = stream_session.set("stream-a")
    try:
        record(
            collector,
            category="workflows",
            frames=2,
            api_key=api_key,
            resource_id=resource_id,
            fps=10,
        )
        stream_session.set("stream-b")
        record(
            collector,
            category="workflows",
            frames=3,
            api_key=api_key,
            resource_id=resource_id,
            fps=10,
        )
    finally:
        stream_session.reset(token)

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
    assert entry_a["source_duration"] == pytest.approx(0.2)


def test_record_usage_without_stream_session_id_keeps_legacy_key(
    collector, stream_session
):
    assert stream_session.get() is None

    record(
        collector,
        category="workflows",
        frames=1,
        api_key="fake",
        resource_id="workflow-1",
    )

    usage = collector._usage["fake"]
    key = usage_key("workflows", "workflow-1")
    assert key in usage
    assert "stream_session_id" not in usage[key]


def test_record_usage_accumulates_megapixel_buckets(collector):
    details = {
        "task_type": "instance-segmentation",
        "model_architecture": "rfdetr",
        "model_variant": "rfdetr-seg-nano",
    }
    record(
        collector,
        category="model",
        frames=2,
        api_key="test_key",
        resource_id="st-inst-seg/9",
        resource_details=details,
        execution_duration=0.4,
        megapixel_buckets={
            "0.25-0.5": {"processed_frames": 2, "execution_duration": 0.4},
        },
    )
    record(
        collector,
        category="model",
        frames=1,
        api_key="test_key",
        resource_id="st-inst-seg/9",
        resource_details=details,
        execution_duration=0.2,
        megapixel_buckets={
            "0.25-0.5": {"processed_frames": 1, "execution_duration": 0.2},
        },
    )

    key = usage_key("model", "st-inst-seg/9")
    row = collector._usage["test_key"][key]
    assert row["processed_frames"] == 3
    assert row["execution_duration"] == pytest.approx(0.6)
    assert row["megapixel_buckets"]["0.25-0.5"]["processed_frames"] == 3
    assert row["megapixel_buckets"]["0.25-0.5"]["execution_duration"] == pytest.approx(
        0.6
    )
    details = json.loads(row["resource_details"])
    assert details["model_architecture"] == "rfdetr"
    assert details["model_variant"] == "rfdetr-seg-nano"


def test_record_usage_without_api_key_records_nothing(collector):
    record(collector, api_key="")
    record(collector, api_key=None)

    assert not collector._usage
    assert collector._delivery._hashed_api_keys == {}


def test_record_usage_with_zero_frames_keeps_the_row_as_legacy(collector):
    record(collector, frames=0, execution_duration=0.25)

    row = collector._usage["fake-key"][usage_key("request", "workspace/model")]
    assert row["processed_frames"] == 0
    assert row["execution_duration"] == 0.25


def test_record_usage_requires_a_category(collector):
    with pytest.raises(
        ValueError, match="^Category is compulsory when recording resource details.$"
    ):
        record(collector, category="")


def test_record_usage_accumulates_a_row(collector, monkeypatch):
    ticks = iter([100, 101, 200])
    monkeypatch.setattr(collector_module.time, "time_ns", lambda: next(ticks))

    record(collector, frames=2, execution_duration=0.5, fps=4)
    record(collector, frames=3, execution_duration=0.25, fps=5)

    row = collector._usage["fake-key"][usage_key("request", "workspace/model")]
    assert row["timestamp_start"] == 100
    assert row["timestamp_stop"] == 200
    assert row["processed_frames"] == 5
    assert row["execution_duration"] == 0.75
    assert row["source_duration"] == pytest.approx(2 / 4 + 3 / 5)
    assert row["fps"] == 5
    assert row["category"] == "request"
    assert row["resource_id"] == "workspace/model"
    assert row["api_key_hash"] == "fake-key"
    assert row["exec_session_id"] == collector._exec_session_id


def test_explicit_source_duration_replaces_the_derived_one(collector):
    record(collector, frames=10, fps=5, source_duration=7.5)

    row = collector._usage["fake-key"][usage_key("request", "workspace/model")]
    assert row["source_duration"] == 7.5


def test_exec_session_id_argument_wins_over_the_request_execution_id(collector):
    token = execution_id.set("from-context")
    try:
        record(collector, resource_id="context")
        record(collector, resource_id="explicit", exec_session_id="from-argument")
    finally:
        execution_id.reset(token)

    usage = collector._usage["fake-key"]
    assert usage[usage_key("request", "context")]["exec_session_id"] == "from-context"
    assert usage[usage_key("request", "explicit")]["exec_session_id"] == "from-argument"


@pytest.mark.parametrize(
    "name, secret, stored",
    [
        ("internal-service", "secret", True),
        ("external", "secret", False),
        ("internal-service", None, False),
        (None, "secret", False),
    ],
)
def test_internal_service_fields_of_the_request(collector, name, secret, stored):
    record(collector, roboflow_service_name=name, roboflow_internal_secret=secret)

    row = collector._usage["fake-key"][usage_key("request", "workspace/model")]
    assert ("roboflow_service_name" in row) is stored
    assert ("roboflow_internal_secret" in row) is stored
    if stored:
        assert row["roboflow_service_name"] == name
        assert row["roboflow_internal_secret"] == secret


def test_record_usage_merges_model_rows_of_one_model_by_summing_the_counters(
    collector,
):
    details = {"model_architecture": "yolov8", "task_type": "object-detection"}
    record(
        collector,
        category="model",
        resource_id="coco/3",
        resource_details=details,
        frames=2,
        execution_duration=0.5,
        megapixel_buckets={
            "0.25-0.5": {"processed_frames": 2, "execution_duration": 0.5}
        },
    )
    record(
        collector,
        category="model",
        resource_id="coco/3",
        resource_details={**details, "model_variant": "n"},
        frames=3,
        execution_duration=0.25,
        megapixel_buckets={
            "0.25-0.5": {"processed_frames": 3, "execution_duration": 0.25}
        },
    )
    record(
        collector,
        category="model",
        resource_id="other/1",
        resource_details=details,
        frames=1,
        execution_duration=1.0,
    )

    rows = collector._usage["fake-key"]
    assert set(rows) == {
        usage_key("model", "coco/3"),
        usage_key("model", "other/1"),
    }
    merged = rows[usage_key("model", "coco/3")]
    assert merged["processed_frames"] == 5
    assert merged["execution_duration"] == 0.75
    assert merged["megapixel_buckets"] == {
        "0.25-0.5": {"processed_frames": 5, "execution_duration": 0.75}
    }
    assert json.loads(merged["resource_details"]) == {
        **details,
        "model_variant": "n",
        "billable": True,
    }


def test_record_usage_without_lists_keeps_the_later_details_as_legacy(collector):
    record(collector, resource_details={"source_info": "first", "only_first": 1})
    record(collector, resource_details={"source_info": "second"})

    row = collector._usage["fake-key"][usage_key("request", "workspace/model")]
    assert json.loads(row["resource_details"]) == {
        "source_info": "second",
        "billable": True,
    }


def test_row_bound_flushes_the_window_early_and_loses_nothing(collector, monkeypatch):
    monkeypatch.setattr(collector_module, "MAX_AGGREGATED_ROWS", 3)
    collector._delivery.queue = Queue()
    held_rows = []

    for index in range(20):
        record(collector, resource_id=f"resource-{index % 10}", frames=2)
        held_rows.append(sum(len(rows) for rows in collector._usage.values()))

    assert max(held_rows) <= 3
    assert len(collector._delivery.pending) > 0
    assert collector._delivery.queue.qsize() == 0
    collector._write_current_usage_to_queue()
    assert total_frames(queued_payloads(collector)) == 40


def test_row_bound_counts_rows_of_every_api_key(collector, monkeypatch):
    monkeypatch.setattr(collector_module, "MAX_AGGREGATED_ROWS", 2)
    collector._delivery.queue = Queue()

    for index in range(5):
        record(collector, api_key=f"key-{index}")

    assert sum(len(rows) for rows in collector._usage.values()) == 1
    assert len(collector._delivery.pending) == 2
    assert collector._delivery.queue.qsize() == 0
    collector._write_current_usage_to_queue()
    assert total_frames(queued_payloads(collector)) == 5


def test_queue_is_redis_on_serverless_with_a_redis_host(monkeypatch):
    monkeypatch.setattr(configuration, "GCP_SERVERLESS", True)
    monkeypatch.setattr(configuration, "REDIS_HOST", "redis.local")
    monkeypatch.setattr(configuration, "TELEMETRY_USE_PERSISTENT_QUEUE", True)

    usage_collector = UsageCollector(redis_client=FakeRedis())

    assert isinstance(usage_collector._delivery.queue, RedisQueue)
    assert usage_collector._delivery._api_keys_hashing_enabled is False


def test_queue_is_in_memory_when_redis_is_selected_but_not_installed(
    monkeypatch, caplog
):
    monkeypatch.setattr(configuration, "LAMBDA", True)
    monkeypatch.setattr(configuration, "REDIS_HOST", "redis.local")
    monkeypatch.setitem(sys.modules, "redis", None)

    with caplog.at_level(logging.ERROR):
        usage_collector = UsageCollector()

    assert isinstance(usage_collector._delivery.queue, Queue)
    assert usage_collector._delivery.queue.maxsize == 10
    assert usage_collector._delivery._api_keys_hashing_enabled is False
    assert len(caplog.records) == 1


@pytest.mark.parametrize("flag", ["LAMBDA", "GCP_SERVERLESS"])
def test_queue_is_in_memory_on_serverless_without_a_redis_host(monkeypatch, flag):
    monkeypatch.setattr(configuration, flag, True)
    monkeypatch.setattr(configuration, "TELEMETRY_USE_PERSISTENT_QUEUE", True)
    monkeypatch.setattr(configuration, "TELEMETRY_QUEUE_SIZE", 25)

    usage_collector = UsageCollector()

    assert isinstance(usage_collector._delivery.queue, Queue)
    assert usage_collector._delivery.queue.maxsize == 25
    assert usage_collector._delivery._api_keys_hashing_enabled is False


def test_queue_is_in_memory_when_the_persistent_queue_is_switched_off(monkeypatch):
    monkeypatch.setattr(configuration, "REDIS_HOST", "redis.local")

    usage_collector = UsageCollector()

    assert isinstance(usage_collector._delivery.queue, Queue)
    assert usage_collector._delivery._api_keys_hashing_enabled is False


def test_queue_is_sqlite_by_default_and_api_keys_are_hashed(monkeypatch, tmp_path):
    monkeypatch.setattr(configuration, "TELEMETRY_USE_PERSISTENT_QUEUE", True)
    monkeypatch.setattr(configuration, "MODEL_CACHE_DIR", str(tmp_path))

    usage_collector = UsageCollector()

    assert isinstance(usage_collector._delivery.queue, SQLiteQueue)
    assert usage_collector._delivery._api_keys_hashing_enabled is True
    assert (tmp_path / "usage.db").exists()


def test_queue_is_in_memory_when_the_sqlite_file_cannot_be_created(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(configuration, "TELEMETRY_USE_PERSISTENT_QUEUE", True)
    blocker = tmp_path / "blocker"
    blocker.write_text("")

    usage_collector = UsageCollector(sqlite_db_file_path=blocker / "usage.db")

    assert isinstance(usage_collector._delivery.queue, Queue)
    assert usage_collector._delivery._api_keys_hashing_enabled is False


def test_redis_receives_the_window_keyed_by_the_api_key(monkeypatch):
    monkeypatch.setattr(configuration, "GCP_SERVERLESS", True)
    monkeypatch.setattr(configuration, "REDIS_HOST", "redis.local")
    client = FakeRedis()
    usage_collector = UsageCollector(redis_client=client)
    usage_collector._system_info = dict(SYSTEM_INFO)

    record(usage_collector, api_key="api-key-1", exec_session_id="request-1")
    with mock.patch(POST) as post_mock:
        usage_collector.flush()

    post_mock.assert_not_called()
    assert len(client.executed) == 1
    set_call, zadd_call = client.executed[0]
    redis_key = set_call[1]["name"]
    assert redis_key.startswith("{UsageCollector}:")
    assert zadd_call[1]["name"] == "UsageCollector"
    assert list(zadd_call[1]["mapping"]) == [redis_key]
    stored = json.loads(set_call[1]["value"])
    key = usage_key("request", "workspace/model")
    assert set(stored) == {"api-key-1"}
    assert set(stored["api-key-1"]) == {key}
    row = stored["api-key-1"][key]
    assert row["api_key_hash"] == "api-key-1"
    assert row["exec_session_id"] == "request-1"
    assert row["hosted"] is True
    assert row["processed_frames"] == 1
    assert "api_key" not in row


def test_sender_posts_rows_as_legacy(collector, monkeypatch):
    monkeypatch.setattr(configuration, "ROBOFLOW_API_EXTRA_HEADERS", '{"X-Extra": "1"}')
    record(collector, api_key="api-key-1", frames=2, execution_duration=0.5)
    key = usage_key("request", "workspace/model")
    queued_row = dict(collector._usage["api-key-1"][key])

    with mock.patch(POST) as post_mock:
        post_mock.return_value.status_code = 200
        collector.flush()

    post_mock.assert_called_once()
    assert post_mock.call_args.args == ("https://api.example.com/usage/inference",)
    assert post_mock.call_args.kwargs == {
        "json": [
            {
                **{k: v for k, v in queued_row.items() if k != "api_key_hash"},
                "api_key": "api-key-1",
            }
        ],
        "verify": True,
        "headers": {
            "Authorization": "Bearer api-key-1",
            "X-Extra": "1",
            "X-Roboflow-Inference-Version": "9.9.9",
            "X-Allow-Chunked": "true",
        },
        "timeout": 1,
    }
    assert set(post_mock.call_args.kwargs["json"][0]) == {
        "timestamp_start",
        "timestamp_stop",
        "exec_session_id",
        "hostname",
        "ip_address_hash",
        "processed_frames",
        "fps",
        "source_duration",
        "category",
        "resource_id",
        "resource_details",
        "hosted",
        "is_gpu_available",
        "python_version",
        "inference_version",
        "inference_models_version",
        "inference_model_manager_version",
        "roboflow_workflows_version",
        "streamvision_version",
        "enterprise",
        "execution_duration",
        "megapixel_buckets",
        "api_key",
    }
    assert queued_payloads(collector) == []


def test_sender_skips_tls_verification_for_local_endpoints_only(collector, monkeypatch):
    monkeypatch.setattr(
        configuration,
        "TELEMETRY_API_USAGE_ENDPOINT_URL",
        "http://localhost:9000/usage/inference",
    )
    record(collector)

    with mock.patch(POST) as post_mock:
        post_mock.return_value.status_code = 200
        collector.flush()

    assert post_mock.call_args.kwargs["verify"] is False


def test_sender_routes_the_endpoint_through_the_secure_gateway(collector, monkeypatch):
    monkeypatch.setattr(
        "inference_models.weights_providers.roboflow.SECURE_GATEWAY",
        "https://gateway.local",
    )
    monkeypatch.setattr(
        configuration,
        "TELEMETRY_API_USAGE_ENDPOINT_URL",
        "http://localhost:9000/usage/inference",
    )
    record(collector)

    with mock.patch(POST) as post_mock:
        post_mock.return_value.status_code = 200
        collector.flush()

    parsed = urlparse(post_mock.call_args.args[0])
    assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == (
        "https://gateway.local/proxy"
    )
    assert parse_qs(parsed.query)["url"] == ["http://localhost:9000/usage/inference"]
    assert post_mock.call_args.kwargs["verify"] is True


@pytest.mark.parametrize("status_code", [201, 204, 401, 500])
def test_sender_requeues_rows_unless_the_status_is_200(collector, status_code):
    record(collector, api_key="api-key-1", frames=3)

    with mock.patch(POST) as post_mock:
        post_mock.return_value.status_code = status_code
        collector.flush()
        first_body = post_mock.call_args.kwargs["json"]
        post_mock.return_value.status_code = 200
        collector.flush()
        second_body = post_mock.call_args.kwargs["json"]

    assert post_mock.call_count == 2
    assert first_body == second_body
    assert first_body[0]["processed_frames"] == 3
    assert queued_payloads(collector) == []


def test_sender_requeues_rows_when_the_request_raises(collector):
    record(collector, api_key="api-key-1", frames=3)
    record(collector, api_key="api-key-2", frames=4)

    def post(url, **kwargs):
        if kwargs["headers"]["Authorization"] == "Bearer api-key-1":
            raise ConnectionError("unreachable")
        return mock.MagicMock(status_code=200)

    with mock.patch(POST, side_effect=post):
        collector.flush()

    payloads = queued_payloads(collector)
    assert [set(payload) for payload in payloads] == [{"api-key-1"}]
    assert total_frames(payloads) == 3


def test_failed_send_logs_no_key_secret_row_or_exception_text(collector, caplog):
    record(
        collector,
        api_key="planted-api-key",
        resource_id="planted-resource",
        roboflow_service_name="internal-service",
        roboflow_internal_secret="planted-secret",
    )
    error = ConnectionError(
        "https://api.example.com/usage/inference?api_key=planted-api-key "
        "Bearer planted-api-key planted-secret"
    )

    with caplog.at_level(
        logging.DEBUG, logger="inference_server.usage.delivery"
    ), mock.patch(POST, side_effect=error):
        collector.flush()

    assert "Usage request failed: ConnectionError" in caplog.text
    for forbidden in (
        "planted-api-key",
        "planted-secret",
        "planted-resource",
        "Bearer",
        "?",
    ):
        assert forbidden not in caplog.text
    assert total_frames(queued_payloads(collector)) == 1


def test_sqlite_file_never_holds_the_api_key(monkeypatch, tmp_path):
    monkeypatch.setattr(configuration, "TELEMETRY_USE_PERSISTENT_QUEUE", True)
    db_file = tmp_path / "usage.db"
    usage_collector = UsageCollector(sqlite_db_file_path=db_file)
    usage_collector._system_info = dict(SYSTEM_INFO)
    api_key = "planted-api-key"
    api_key_hash = sha256_hash(api_key, length=-1)

    record(usage_collector, api_key=api_key, frames=3)
    usage_collector._write_current_usage_to_queue()
    first_write = db_file.read_bytes()

    with mock.patch(POST) as post_mock:
        post_mock.return_value.status_code = 500
        usage_collector.flush()
        after_failed_send = db_file.read_bytes()
        stored = usage_collector._delivery.queue.get_nowait()
        usage_collector._delivery.queue.put(stored[0])
        post_mock.return_value.status_code = 200
        usage_collector.flush()

    assert api_key.encode() not in first_write
    assert api_key_hash.encode() in first_write
    assert api_key.encode() not in after_failed_send
    assert api_key_hash.encode() in after_failed_send
    assert set(stored[0]) == {api_key_hash}
    row = stored[0][api_key_hash][usage_key("request", "workspace/model")]
    assert row["api_key_hash"] == api_key_hash
    assert "api_key" not in row
    assert post_mock.call_count == 2
    for call in post_mock.call_args_list:
        assert call.kwargs["headers"]["Authorization"] == f"Bearer {api_key}"
        assert call.kwargs["json"][0]["api_key"] == api_key
        assert "api_key_hash" not in call.kwargs["json"][0]
    assert usage_collector._delivery.queue.empty() is True


def test_rows_of_an_unknown_api_key_hash_stay_queued(monkeypatch, tmp_path):
    monkeypatch.setattr(configuration, "TELEMETRY_USE_PERSISTENT_QUEUE", True)
    usage_collector = UsageCollector(sqlite_db_file_path=tmp_path / "usage.db")
    usage_collector._system_info = dict(SYSTEM_INFO)
    record(usage_collector, api_key="known-key")
    usage_collector._delivery.queue.put(
        {
            "hash-of-an-earlier-process": {
                "request:r": {"processed_frames": 7, "execution_duration": 0}
            }
        }
    )

    with mock.patch(POST) as post_mock:
        post_mock.return_value.status_code = 200
        usage_collector.flush()

    post_mock.assert_called_once()
    assert usage_collector._delivery.queue.get_nowait() == [
        {
            "hash-of-an-earlier-process": {
                "request:r": {"processed_frames": 7, "execution_duration": 0}
            }
        }
    ]


def test_start_and_stop_leave_no_thread_alive(collector, monkeypatch):
    monkeypatch.setattr(configuration, "TELEMETRY_FLUSH_INTERVAL", 3600)
    record(collector, frames=2)

    with mock.patch(POST) as post_mock:
        post_mock.return_value.status_code = 200
        collector.start()
        collector.start()
        started = [
            thread.name
            for thread in threading.enumerate()
            if thread.name.startswith("usage-")
        ]
        collector.stop()

    assert sorted(started) == ["usage-collector", "usage-sender"]
    assert len(collector._delivery.threads) == 2
    assert not [thread for thread in collector._delivery.threads if thread.is_alive()]
    assert not [
        thread for thread in threading.enumerate() if thread.name.startswith("usage-")
    ]
    post_mock.assert_called_once()
    assert post_mock.call_args.kwargs["json"][0]["processed_frames"] == 2


def test_stop_before_start_returns_true_does_no_io_and_closes_admission(collector):
    queue = Queue()
    collector._delivery.queue = queue

    with mock.patch(POST) as post_mock, mock.patch.object(
        queue, "put"
    ) as put_mock, mock.patch.object(queue, "get_nowait") as get_mock:
        complete = collector.stop()
        record(collector)

    assert complete is True
    assert collector._delivery.threads == []
    post_mock.assert_not_called()
    put_mock.assert_not_called()
    get_mock.assert_not_called()
    assert collector.ignored_after_stop == 1
    assert not collector._usage


def test_background_loops_survive_a_failing_iteration(collector, monkeypatch, caplog):
    monkeypatch.setattr(configuration, "TELEMETRY_FLUSH_INTERVAL", 0.01)
    calls = threading.Event()

    def failing(*args, **kwargs):
        calls.set()
        raise RuntimeError("planted-secret")

    monkeypatch.setattr(collector._delivery, "drain_pending", failing)
    monkeypatch.setattr(collector._delivery, "send_queued", failing)

    with caplog.at_level(logging.DEBUG):
        collector.start()
        assert calls.wait(timeout=5)
        collector.stop()

    assert not [thread for thread in collector._delivery.threads if thread.is_alive()]
    assert "RuntimeError" in caplog.text
    assert "planted-secret" not in caplog.text


def test_concurrent_recording_loses_no_usage(collector, monkeypatch):
    monkeypatch.setattr(collector_module, "MAX_AGGREGATED_ROWS", 4)
    collector._delivery.queue = Queue()

    def worker(worker_index):
        for index in range(200):
            record(
                collector,
                api_key=f"key-{worker_index % 3}",
                resource_id=f"resource-{index % 7}",
            )

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    collector._write_current_usage_to_queue()
    assert total_frames(queued_payloads(collector)) == 1600

import asyncio
import logging
import threading
import time
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import List, Optional, Tuple

import pytest
from asyncua import ua
from pydantic import ValidationError
from roboflow_workflows.enterprise_blocks import loader as enterprise_loader
from roboflow_workflows.enterprise_blocks.sinks.opc_reader import v1
from roboflow_workflows.enterprise_blocks.sinks.opc_reader.v1 import (
    NOT_CONNECTED_WITHIN_TIMEOUT,
    BlockManifest,
    OPCReaderBlockV1,
    OPCUAReaderConfig,
    OPCUAReaderState,
    _build_config,
    _DataChangeHandler,
    _display_url,
    _SubscriptionWorker,
    _to_output_value,
)
from roboflow_workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    FLOAT_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
)
from roboflow_workflows.execution_engine.entities.workload import (
    WorkOperation,
    restriction_metadata_of,
)
from roboflow_workflows.execution_engine.introspection import blocks_loader
from roboflow_workflows.execution_engine.introspection.workload import (
    describe_workflow_workload,
)

ENTERPRISE_PLUGIN = "roboflow_workflows.enterprise_blocks.loader"
HOSTED_RESTRICTION_CODE = "unavailable_on_hosted_platform"
PER_REQUEST_RESTRICTION_CODE = "connection_and_state_rebuilt_per_request"
OUTPUT_KEYS = {"value", "is_new", "error_status", "error_message"}


def manifest_payload(**overrides) -> dict:
    payload = {
        "type": "roboflow_enterprise/opc_reader@v1",
        "name": "reader",
        "url": "opc.tcp://localhost:4840/opcua/server",
        "namespace": "http://opcua.roboflow.run",
        "object_name": "Line1",
        "variable_name": "LineRunning",
    }
    payload.update(overrides)
    return payload


def run_kwargs(**overrides) -> dict:
    kwargs = {
        "url": "opc.tcp://localhost:4840/opcua/server",
        "namespace": "http://opcua.roboflow.run",
        "object_name": "Line1",
        "variable_name": "LineRunning",
        "timeout": 0.5,
    }
    kwargs.update(overrides)
    return kwargs


def wait_for(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def data_value(value, status: Optional[int] = None) -> ua.DataValue:
    if status is None:
        return ua.DataValue(ua.Variant(value))
    return ua.DataValue(ua.Variant(value), StatusCode_=ua.StatusCode(status))


class FakeWorker:
    """Stands in for the subscription thread; each test scripts what it delivers."""

    instances: List["FakeWorker"] = []
    on_start = staticmethod(lambda state: None)

    def __init__(self, config: OPCUAReaderConfig, state: OPCUAReaderState):
        self.config = config
        self.state = state
        self.started = False
        self.stop_calls = 0
        FakeWorker.instances.append(self)

    def start(self) -> None:
        self.started = True
        FakeWorker.on_start(self.state)

    def stop(self) -> None:
        self.stop_calls += 1


def connected_with(value, status: Optional[int] = None):
    def on_start(state: OPCUAReaderState) -> None:
        state.record_value(data_value(value, status))
        state.record_connected()

    return on_start


@pytest.fixture
def workers(monkeypatch):
    FakeWorker.instances = []
    FakeWorker.on_start = staticmethod(connected_with(True))
    monkeypatch.setattr(v1, "_SubscriptionWorker", FakeWorker)
    monkeypatch.setattr(v1, "GCP_SERVERLESS", False)
    monkeypatch.setattr(v1, "LAMBDA", False)
    yield FakeWorker.instances


@pytest.fixture
def block(workers):
    block = OPCReaderBlockV1()
    yield block
    block.close()


class TestManifest:
    def test_type_identifier_is_namespaced(self):
        manifest = BlockManifest.model_validate(manifest_payload())

        assert manifest.type == "roboflow_enterprise/opc_reader@v1"

    def test_defaults(self):
        payload = manifest_payload()
        del payload["variable_name"]

        manifest = BlockManifest.model_validate(
            payload | {"node_lookup_mode": "direct"}
        )

        assert manifest.variable_name == ""
        assert manifest.publishing_interval_ms == 100
        assert manifest.timeout == 2.0
        assert manifest.user_name is None
        assert manifest.password is None

    @pytest.mark.parametrize("field", ["timeout", "publishing_interval_ms"])
    @pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
    def test_literal_intervals_must_be_positive_and_finite(self, field, value):
        with pytest.raises(ValidationError):
            BlockManifest.model_validate(manifest_payload(**{field: value}))

    def test_default_lookup_mode_is_hierarchical(self):
        manifest = BlockManifest.model_validate(manifest_payload())

        assert manifest.node_lookup_mode == "hierarchical"

    @pytest.mark.parametrize("node_lookup_mode", [None, "hierarchical"])
    def test_hierarchical_lookup_requires_a_variable_name(self, node_lookup_mode):
        payload = manifest_payload()
        del payload["variable_name"]
        if node_lookup_mode is not None:
            payload["node_lookup_mode"] = node_lookup_mode

        with pytest.raises(ValidationError, match="variable_name is required"):
            BlockManifest.model_validate(payload)

    @pytest.mark.parametrize(
        "overrides",
        [
            {"variable_name": "$inputs.variable"},
            {"variable_name": "", "node_lookup_mode": "$inputs.mode"},
            {"variable_name": "", "node_lookup_mode": "direct"},
        ],
    )
    def test_empty_variable_name_check_skips_runtime_values(self, overrides):
        BlockManifest.model_validate(manifest_payload(**overrides))

    @pytest.mark.parametrize(
        "url, expected",
        [
            ("opc.tcp://alice:secret@host:4840", "credentials in the url"),
            ("http://host:4840", "must be opc.tcp://"),
        ],
    )
    def test_invalid_literal_url_is_rejected_at_compile_time(self, url, expected):
        with pytest.raises(ValidationError, match=expected) as error:
            BlockManifest.model_validate(manifest_payload(url=url))

        assert "Invalid url" in str(error.value)

    def test_node_lookup_mode_must_be_known(self):
        with pytest.raises(ValidationError):
            BlockManifest.model_validate(manifest_payload(node_lookup_mode="browse"))

    def test_selectors_accepted_for_every_field(self):
        manifest = BlockManifest.model_validate(
            manifest_payload(
                url="$inputs.url",
                namespace="$inputs.namespace",
                user_name="$inputs.user",
                password="$inputs.password",
                object_name="$inputs.object",
                variable_name="$inputs.variable",
                node_lookup_mode="$inputs.mode",
                publishing_interval_ms="$inputs.interval",
                timeout="$inputs.timeout",
            )
        )

        assert manifest.url == "$inputs.url"
        assert manifest.timeout == "$inputs.timeout"

    def test_password_is_marked_private(self):
        schema = BlockManifest.model_json_schema()

        assert schema["properties"]["password"]["private"] is True

    def test_describe_outputs(self):
        outputs = BlockManifest.describe_outputs()

        assert [output.name for output in outputs] == [
            "value",
            "is_new",
            "error_status",
            "error_message",
        ]
        assert outputs[0].kind == [
            BOOLEAN_KIND,
            INTEGER_KIND,
            FLOAT_KIND,
            STRING_KIND,
            LIST_OF_VALUES_KIND,
        ]

    def test_ui_manifest_marks_enterprise_and_local_only(self):
        schema = BlockManifest.model_json_schema()

        assert schema["block_type"] == "transformation"
        assert schema["ui_manifest"]["section"] == "industrial"
        assert schema["ui_manifest"]["enterprise_only"] is True
        assert schema["ui_manifest"]["local_only"] is True

    def test_registered_in_enterprise_loader(self):
        assert OPCReaderBlockV1 in enterprise_loader.load_enterprise_blocks()


class TestWorkloadDeclarations:
    def test_declares_server_io_and_value_held_between_runs(self):
        manifest = BlockManifest.model_validate(manifest_payload())

        assert manifest.discover_work_operations() == [
            WorkOperation.EXTERNAL_REQUEST,
            WorkOperation.TEMPORAL_BUFFERING,
        ]
        assert manifest.discover_dependent_resources() == []

    @pytest.mark.parametrize("ignore_environment_restrictions", [True, False])
    def test_actual_restrictions_are_complete_and_match_the_editor_ones(
        self, ignore_environment_restrictions
    ):
        manifest = BlockManifest.model_validate(manifest_payload())

        discovery = manifest.get_actual_restrictions(
            ignore_environment_restrictions=ignore_environment_restrictions
        )

        assert discovery.complete is True
        assert sorted(restriction.code for restriction in discovery.items) == [
            PER_REQUEST_RESTRICTION_CODE,
            HOSTED_RESTRICTION_CODE,
        ]

    def test_restriction_axes(self):
        manifest = BlockManifest.model_validate(manifest_payload())
        discovery = manifest.get_actual_restrictions(
            ignore_environment_restrictions=True
        )

        portable = {
            restriction.code: restriction_metadata_of(restriction)
            for restriction in discovery.items
        }

        hosted = portable[HOSTED_RESTRICTION_CODE]
        per_request = portable[PER_REQUEST_RESTRICTION_CODE]
        assert hosted.severity is v1.Severity.HARD
        assert hosted.when.runtimes == [v1.Runtime.HOSTED_SERVERLESS]
        assert hosted.when.configuration_equals == {}
        assert per_request.severity is v1.Severity.SOFT
        assert set(per_request.when.runtimes) == {
            v1.Runtime.SELF_HOSTED_CPU,
            v1.Runtime.SELF_HOSTED_GPU,
            v1.Runtime.DEDICATED_DEPLOYMENT,
        }
        assert per_request.when.input_modes == [v1.RuntimeInputMode.IMAGE]

    def test_public_introspection_describes_the_step_completely(
        self, monkeypatch, workers
    ):
        definition = {
            "version": "1.0",
            "inputs": [{"type": "WorkflowParameter", "name": "opc_url"}],
            "steps": [manifest_payload(url="$inputs.opc_url")],
            "outputs": [
                {
                    "type": "JsonField",
                    "name": "value",
                    "selector": "$steps.reader.value",
                }
            ],
        }
        # the plugin list is read at load time: clear the caches on both sides
        monkeypatch.setenv("WORKFLOWS_PLUGINS", ENTERPRISE_PLUGIN)
        blocks_loader.clear_caches()
        try:
            description = describe_workflow_workload(definition)
        finally:
            blocks_loader.clear_caches()

        [step] = description.steps
        assert step.node_id == "$steps.reader"
        assert step.operations.complete is True
        assert step.restrictions.complete is True
        assert step.resources.items == []
        # compile-time facts only: no subscription was started
        assert workers == []


class TestBuildConfig:
    def test_valid_arguments_build_a_config(self):
        config = _build_config(
            url="opc.tcp://host:4840",
            namespace="2",
            user_name="operator",
            password="secret",
            object_name="Line1",
            variable_name="LineRunning",
            node_lookup_mode="hierarchical",
            publishing_interval_ms="50",
            timeout=1,
        )

        assert config == OPCUAReaderConfig(
            url="opc.tcp://host:4840",
            namespace="2",
            user_name="operator",
            password="secret",
            object_name="Line1",
            variable_name="LineRunning",
            node_lookup_mode="hierarchical",
            timeout=1.0,
            publishing_interval_ms=50.0,
        )

    def test_numeric_namespace_from_a_workflow_input_is_an_index(self):
        config = _build_config(**self._arguments(namespace=2))

        assert config.namespace == "2"

    def test_direct_mode_allows_an_empty_variable_name(self):
        config = _build_config(
            **self._arguments(node_lookup_mode="direct", variable_name=None)
        )

        assert config.variable_name == ""

    def test_empty_credentials_mean_anonymous(self):
        config = _build_config(**self._arguments(user_name="", password=""))

        assert config.user_name is None
        assert config.password is None

    @pytest.mark.parametrize(
        "overrides, expected_fragment",
        [
            ({"url": ""}, "Invalid url"),
            ({"url": None}, "Invalid url"),
            ({"url": "http://host:4840"}, "must be opc.tcp://"),
            ({"url": "opc.tcp://"}, "must be opc.tcp://"),
            ({"url": "opc.tcp://[bad"}, "Invalid IPv6 URL"),
            ({"url": "opc.tcp://host:port"}, "Invalid url opc.tcp://host:port"),
            ({"url": "opc.tcp://host:99999"}, "out of range"),
            ({"namespace": ""}, "Invalid namespace"),
            ({"namespace": True}, "Invalid namespace"),
            ({"object_name": ""}, "Invalid object_name"),
            ({"variable_name": 5}, "Invalid variable_name"),
            ({"node_lookup_mode": "browse"}, "Invalid node_lookup_mode"),
            ({"variable_name": ""}, "variable_name is required"),
            ({"user_name": "operator"}, "Set both user_name and password"),
            ({"password": "secret"}, "Set both user_name and password"),
            ({"timeout": 0}, "Invalid timeout"),
            ({"timeout": -1}, "Invalid timeout"),
            ({"timeout": float("nan")}, "Invalid timeout"),
            ({"timeout": float("inf")}, "Invalid timeout"),
            ({"timeout": "soon"}, "Invalid timeout"),
            ({"timeout": True}, "Invalid timeout"),
            ({"publishing_interval_ms": 0}, "Invalid publishing_interval_ms"),
            ({"publishing_interval_ms": None}, "Invalid publishing_interval_ms"),
        ],
    )
    def test_invalid_arguments_are_rejected(self, overrides, expected_fragment):
        error = _build_config(**self._arguments(**overrides))

        assert isinstance(error, str)
        assert expected_fragment in error

    @staticmethod
    def _arguments(**overrides) -> dict:
        arguments = {
            "url": "opc.tcp://host:4840",
            "namespace": "http://opcua.roboflow.run",
            "user_name": None,
            "password": None,
            "object_name": "Line1",
            "variable_name": "LineRunning",
            "node_lookup_mode": "hierarchical",
            "publishing_interval_ms": 100,
            "timeout": 2.0,
        }
        arguments.update(overrides)
        return arguments


class TestUrlValidation:
    def test_credentials_in_an_invalid_url_are_not_echoed(self):
        error = _build_config(
            **TestBuildConfig._arguments(url="tcp://alice:secret@host:4840")
        )

        assert "secret" not in error
        assert "alice" not in error
        assert "tcp://host:4840" in error

    @pytest.mark.parametrize(
        "url",
        [
            "opc.tcp://alice:secret@host:4840/x",
            "opc.tcp://alice@host:4840",
            # unescaped "/", "?" or "#" in the password hides the "@" from the
            # parser's authority; a numeric prefix would even parse as a port
            "opc.tcp://alice:sec/ret@host:4840/path",
            "opc.tcp://alice:123/secret@host:4840",
            "opc.tcp://alice:se?cret@host:4840",
            "opc.tcp://alice:se#cret@host:4840",
        ],
    )
    def test_url_with_credentials_is_rejected_without_echoing_them(self, url):
        error = _build_config(**TestBuildConfig._arguments(url=url))

        assert "credentials in the url are not accepted" in error
        assert "alice" not in error
        assert "secret" not in error

    @pytest.mark.parametrize(
        "url, expected",
        [
            ("opc.tcp://host:4840/opcua/server", "opc.tcp://host:4840/opcua/server"),
            ("opc.tcp://alice:secret@host:4840/path", "opc.tcp://host:4840/path"),
            ("opc.tcp://alice@host", "opc.tcp://host"),
            ("opc.tcp://a:b@c@host:4840", "opc.tcp://host:4840"),
            ("not a url", "not a url"),
            ("alice:secret@host:4840/path", "host:4840/path"),
        ],
    )
    def test_display_url_drops_userinfo(self, url, expected):
        assert _display_url(url) == expected


class TestCredentialValidation:
    @pytest.mark.parametrize("field", ["user_name", "password"])
    @pytest.mark.parametrize("secret", [123456789, ["hunter2"], {"k": "hunter2"}])
    def test_non_string_credentials_are_rejected_without_echoing_them(
        self, field, secret
    ):
        arguments = TestBuildConfig._arguments(user_name="operator", password="pw")
        arguments[field] = secret

        error = _build_config(**arguments)

        assert error.startswith(f"Invalid {field}: expected a string")
        assert "123456789" not in error
        assert "hunter2" not in error

    @pytest.mark.parametrize(
        "overrides, expected",
        [
            ({"variable_name": False}, "Invalid variable_name"),
            ({"variable_name": 0}, "Invalid variable_name"),
            ({"user_name": False, "password": "pw"}, "Invalid user_name"),
            ({"user_name": "operator", "password": 0}, "Invalid password"),
        ],
    )
    def test_falsy_non_strings_are_rejected_not_treated_as_unset(
        self, overrides, expected
    ):
        arguments = TestBuildConfig._arguments(node_lookup_mode="direct")
        arguments.update(overrides)

        error = _build_config(**arguments)

        assert isinstance(error, str)
        assert error.startswith(expected)

    def test_non_string_url_is_not_echoed(self):
        error = _build_config(**TestBuildConfig._arguments(url={"u": "hunter2"}))

        assert error.startswith("Invalid url")
        assert "hunter2" not in error


class TestOutputValueConversion:
    @pytest.mark.parametrize("value", [None, True, 3, 2.5, "running"])
    def test_plain_values_pass_through(self, value):
        assert _to_output_value(value) == value

    def test_datetime_becomes_iso_string(self):
        moment = datetime(2026, 10, 9, 12, 30, tzinfo=timezone.utc)

        assert _to_output_value(moment) == "2026-10-09T12:30:00+00:00"

    def test_localized_text_becomes_its_text(self):
        assert _to_output_value(ua.LocalizedText("Running", "en")) == "Running"

    def test_arrays_are_converted_element_wise(self):
        moment = datetime(2026, 10, 9, tzinfo=timezone.utc)

        assert _to_output_value([1, (moment, ua.LocalizedText("x"))]) == [
            1,
            ["2026-10-09T00:00:00+00:00", "x"],
        ]

    def test_other_types_become_strings(self):
        assert _to_output_value(ua.NodeId("Line1.Speed", 2)) == str(
            ua.NodeId("Line1.Speed", 2)
        )


class TestReaderState:
    def test_first_value_is_new(self):
        state = OPCUAReaderState()

        state.record_value(data_value(False))

        assert state.has_value is True
        assert state.value is False
        assert state.sequence == 1
        assert state.status_name == "Good"
        assert state.status_is_good is True

    def test_repeated_value_is_not_new(self):
        state = OPCUAReaderState()
        state.record_value(data_value(False))

        state.record_value(data_value(False))

        assert state.sequence == 1

    def test_changed_value_is_new(self):
        state = OPCUAReaderState()
        state.record_value(data_value(False))

        state.record_value(data_value(True))

        assert state.sequence == 2
        assert state.value is True

    @pytest.mark.parametrize(
        "value", [float("nan"), [1.0, float("nan")], [[float("nan")], 2.0]]
    )
    def test_repeated_nan_is_not_new(self, value):
        state = OPCUAReaderState()
        state.record_value(data_value(value))

        state.record_value(data_value(value))

        assert state.sequence == 1

    def test_nan_after_a_number_is_new(self):
        state = OPCUAReaderState()
        state.record_value(data_value(1.5))

        state.record_value(data_value(float("nan")))

        assert state.sequence == 2

    def test_type_change_with_equal_value_is_new(self):
        state = OPCUAReaderState()
        state.record_value(data_value(1))

        state.record_value(data_value(True))

        assert state.sequence == 2

    def test_changed_quality_is_new(self):
        state = OPCUAReaderState()
        state.record_value(data_value(True))

        state.record_value(data_value(True, ua.StatusCodes.BadCommunicationError))

        assert state.sequence == 2
        assert state.status_is_good is False
        assert state.status_name == "BadCommunicationError"

    def test_missing_status_code_counts_as_good(self):
        state = OPCUAReaderState()

        state.record_value(ua.DataValue(ua.Variant(1.5), StatusCode_=None))

        assert state.status_is_good is True

    def test_connection_events_settle_the_first_run(self):
        state = OPCUAReaderState()

        state.record_failure("CONNECTION ERROR: refused")

        assert state.settled.is_set()
        assert state.connected is False
        assert state.error == "CONNECTION ERROR: refused"
        state.record_connected()
        assert state.connected is True
        assert state.error is None


class TestDataChangeHandler:
    def test_data_change_records_the_full_data_value(self):
        state = OPCUAReaderState()
        handler = _DataChangeHandler(state, asyncio.Event())
        notification = SimpleNamespace(
            monitored_item=SimpleNamespace(
                Value=data_value(7, ua.StatusCodes.UncertainLastUsableValue)
            )
        )

        handler.datachange_notification(node=None, val=7, data=notification)

        assert state.value == 7
        assert state.status_name == "UncertainLastUsableValue"

    def test_bad_status_change_marks_the_subscription_lost(self):
        lost = asyncio.Event()
        handler = _DataChangeHandler(OPCUAReaderState(), lost)

        handler.status_change_notification(
            ua.StatusChangeNotification(Status=ua.StatusCode(ua.StatusCodes.BadTimeout))
        )

        assert lost.is_set()
        assert handler.lost_status == "BadTimeout"

    def test_good_status_change_is_ignored(self):
        lost = asyncio.Event()
        handler = _DataChangeHandler(OPCUAReaderState(), lost)

        handler.status_change_notification(
            ua.StatusChangeNotification(Status=ua.StatusCode(ua.StatusCodes.Good))
        )

        assert not lost.is_set()


class TestRun:
    def test_first_run_starts_the_subscription_and_returns_the_value(
        self, block, workers
    ):
        result = block.run(**run_kwargs())

        assert result == {
            "value": True,
            "is_new": True,
            "error_status": False,
            "error_message": None,
        }
        [worker] = workers
        assert worker.started is True
        assert worker.config.object_name == "Line1"

    def test_later_runs_reuse_the_subscription_and_report_no_change(
        self, block, workers
    ):
        first = block.run(**run_kwargs())

        second = block.run(**run_kwargs())

        assert second == {**first, "is_new": False}
        assert len(workers) == 1

    def test_change_between_runs_is_new(self, block, workers):
        block.run(**run_kwargs())

        workers[0].state.record_value(data_value(False))

        assert block.run(**run_kwargs()) == {
            "value": False,
            "is_new": True,
            "error_status": False,
            "error_message": None,
        }

    def test_first_run_gives_up_after_timeout(self, block, workers):
        FakeWorker.on_start = staticmethod(lambda state: None)

        started = time.monotonic()
        result = block.run(**run_kwargs(timeout=0.1))
        elapsed = time.monotonic() - started

        assert result == {
            "value": None,
            "is_new": False,
            "error_status": True,
            "error_message": NOT_CONNECTED_WITHIN_TIMEOUT,
        }
        assert 0.1 <= elapsed < 2.0

    def test_runs_after_the_first_never_wait(self, block, workers):
        FakeWorker.on_start = staticmethod(lambda state: None)
        block.run(**run_kwargs(timeout=0.1))

        class NeverWaited(threading.Event):
            def wait(self, timeout=None):
                raise AssertionError("a run after the first waited for the server")

        workers[0].state.settled = NeverWaited()
        result = block.run(**run_kwargs(timeout=0.1))

        assert result["error_status"] is True

    def test_first_connection_failure_is_reported_without_waiting(self, block, workers):
        FakeWorker.on_start = staticmethod(
            lambda state: state.record_failure("CONNECTION ERROR: refused")
        )

        started = time.monotonic()
        result = block.run(**run_kwargs(timeout=5))

        assert time.monotonic() - started < 0.5
        assert result == {
            "value": None,
            "is_new": False,
            "error_status": True,
            "error_message": "CONNECTION ERROR: refused",
        }

    def test_outage_repeats_the_last_value_with_an_error(self, block, workers):
        block.run(**run_kwargs())

        workers[0].state.record_failure("CONNECTION ERROR: lost connection")
        result = block.run(**run_kwargs())

        assert result["value"] is True
        assert result["is_new"] is False
        assert result["error_status"] is True
        assert result["error_message"].startswith("CONNECTION ERROR: lost connection")
        assert "reconnecting in the background" in result["error_message"]

    def test_returned_arrays_do_not_alias_the_held_value(self, block, workers):
        FakeWorker.on_start = staticmethod(connected_with([1, [2, 3]]))
        first = block.run(**run_kwargs())

        first["value"].append(4)
        first["value"][1].append(5)

        assert block.run(**run_kwargs())["value"] == [1, [2, 3]]

    def test_permanent_failure_is_reported_as_is(self, block, workers):
        block.run(**run_kwargs())

        workers[0].state.record_failure("AUTH ERROR: rejected", permanent=True)

        assert block.run(**run_kwargs())["error_message"] == "AUTH ERROR: rejected"

    def test_value_with_bad_quality_is_an_error(self, block, workers):
        FakeWorker.on_start = staticmethod(
            connected_with(True, ua.StatusCodes.BadCommunicationError)
        )

        result = block.run(**run_kwargs())

        assert result == {
            "value": True,
            "is_new": True,
            "error_status": True,
            "error_message": "Value quality is not Good: BadCommunicationError",
        }

    def test_configuration_change_between_runs_is_rejected(self, block, workers):
        block.run(**run_kwargs())

        result = block.run(**run_kwargs(variable_name="Speed"))

        assert result["error_status"] is True
        assert "changed between runs" in result["error_message"]
        assert len(workers) == 1

    def test_invalid_arguments_start_nothing(self, block, workers):
        result = block.run(**run_kwargs(timeout=0))

        assert result["error_status"] is True
        assert set(result) == OUTPUT_KEYS
        assert workers == []

    @pytest.mark.parametrize("flag", ["GCP_SERVERLESS", "LAMBDA"])
    def test_hosted_platform_rejected_without_connecting(
        self, block, workers, monkeypatch, flag
    ):
        monkeypatch.setattr(v1, flag, True)

        result = block.run(**run_kwargs())

        assert result == {
            "value": None,
            "is_new": False,
            "error_status": True,
            "error_message": "OPC UA Reader is not available on the Roboflow hosted platform",
        }
        assert workers == []

    def test_rejected_run_is_logged_once_per_reason(self, block, workers, caplog):
        with caplog.at_level(logging.ERROR, logger="inference"):
            for _ in range(5):
                block.run(**run_kwargs(timeout=0))
            block.run(**run_kwargs(timeout=-1))

        messages = [record.getMessage() for record in caplog.records]
        assert len(messages) == 2
        assert "Invalid timeout: 0" in messages[0]
        assert "Invalid timeout: -1" in messages[1]

    def test_every_result_carries_every_declared_output(self, block, workers):
        FakeWorker.on_start = staticmethod(lambda state: None)

        results = [
            block.run(**run_kwargs(timeout=0)),
            block.run(**run_kwargs(timeout=0.05)),
        ]
        workers[0].state.record_value(data_value(1))
        workers[0].state.record_connected()
        results.append(block.run(**run_kwargs(timeout=0.05)))

        declared = {output.name for output in BlockManifest.describe_outputs()}
        assert all(set(result) == declared for result in results)


class FakeAsyncClient:
    """Just enough of asyncua's Client for one direct-mode connection attempt."""

    instances: List["FakeAsyncClient"] = []
    read_result: ua.DataValue = ua.DataValue(ua.Variant(True))
    disconnect_hangs = False
    publish_loop_ended = False

    def __init__(self, url, timeout):
        self.url = url
        # asyncua keeps the publish loop task here; done() mirrors whether it ended
        self.uaclient = SimpleNamespace(
            _publish_task=SimpleNamespace(
                done=lambda: FakeAsyncClient.publish_loop_ended
            )
        )
        self.session_timeout = None
        self.socket_closed = False
        self.disconnected = False
        FakeAsyncClient.instances.append(self)

    async def connect(self):
        pass

    def get_node(self, node_id):
        return SimpleNamespace(read_data_value=self._read_data_value)

    async def _read_data_value(self, raise_on_bad_status=True):
        assert raise_on_bad_status is False
        return FakeAsyncClient.read_result

    async def create_subscription(self, period, handler):
        async def subscribe_data_change(node, sampling_interval):
            return 1

        return SimpleNamespace(subscribe_data_change=subscribe_data_change)

    async def check_connection(self):
        pass

    async def disconnect(self):
        if FakeAsyncClient.disconnect_hangs:
            await asyncio.sleep(3600)
        self.disconnected = True

    def disconnect_socket(self):
        self.socket_closed = True


@pytest.fixture
def fake_clients(monkeypatch):
    FakeAsyncClient.instances = []
    FakeAsyncClient.read_result = ua.DataValue(ua.Variant(True))
    FakeAsyncClient.disconnect_hangs = False
    FakeAsyncClient.publish_loop_ended = False
    monkeypatch.setattr(v1, "Client", FakeAsyncClient)
    yield FakeAsyncClient.instances


class TestSubscriptionWorker:
    @staticmethod
    def _config(**overrides) -> OPCUAReaderConfig:
        fields = {
            "url": "opc.tcp://alice:secret@[bad",
            "namespace": "2",
            "user_name": None,
            "password": None,
            "object_name": "Line1",
            "variable_name": "LineRunning",
            "node_lookup_mode": "hierarchical",
            "timeout": 1.0,
            "publishing_interval_ms": 100.0,
        }
        fields.update(overrides)
        return OPCUAReaderConfig(**fields)

    def test_unconstructable_client_is_a_terminal_failure(self):
        state = OPCUAReaderState()
        worker = _SubscriptionWorker(config=self._config(), state=state)

        worker.start()
        try:
            assert state.settled.wait(timeout=5)
            assert worker._finished.wait(timeout=5)
        finally:
            worker.stop()

        assert state.permanent is True
        assert state.error.startswith("Invalid url opc.tcp://[bad:")
        assert "secret" not in state.error

    def _direct_worker(self) -> Tuple[_SubscriptionWorker, OPCUAReaderState]:
        state = OPCUAReaderState()
        worker = _SubscriptionWorker(
            config=self._config(
                url="opc.tcp://host:4840",
                object_name="Line1.LineRunning",
                variable_name="",
                node_lookup_mode="direct",
            ),
            state=state,
        )
        return worker, state

    def test_initially_bad_quality_value_is_recorded_not_retried(self, fake_clients):
        FakeAsyncClient.read_result = data_value(
            False, ua.StatusCodes.BadCommunicationError
        )
        worker, state = self._direct_worker()

        worker.start()
        try:
            assert state.settled.wait(timeout=5)
        finally:
            worker.stop()

        assert state.connected is True
        assert state.has_value is True
        assert state.status_name == "BadCommunicationError"
        assert len(fake_clients) == 1

    def test_unreadable_node_status_is_a_configuration_error(self, fake_clients):
        FakeAsyncClient.read_result = data_value(None, ua.StatusCodes.BadNodeIdUnknown)
        worker, state = self._direct_worker()

        worker.start()
        try:
            assert state.settled.wait(timeout=5)
        finally:
            worker.stop()

        assert state.connected is False
        assert state.has_value is False
        assert state.error.startswith("NODE ERROR: BadNodeIdUnknown")
        assert fake_clients[0].disconnected is True

    def test_publish_loop_ending_normally_counts_as_a_lost_subscription(
        self, fake_clients, monkeypatch
    ):
        # asyncua ends its publish loop without an error on BadNoSubscription, so a
        # healthy session alone does not prove notifications still arrive
        monkeypatch.setattr(v1, "CONNECTION_CHECK_INTERVAL_SECONDS", 0.05)
        monkeypatch.setattr(v1, "RECONNECT_INITIAL_DELAY_SECONDS", 0.05)
        worker, state = self._direct_worker()
        worker.start()
        try:
            assert state.settled.wait(timeout=5)
            assert state.connected is True

            FakeAsyncClient.publish_loop_ended = True
            assert wait_for(lambda: not state.connected)
            assert "stopped delivering notifications" in state.error

            FakeAsyncClient.publish_loop_ended = False
            assert wait_for(lambda: state.connected)
        finally:
            worker.stop()

        assert len(fake_clients) >= 2
        assert fake_clients[0].disconnected is True

    def test_hung_disconnect_is_abandoned_within_the_shutdown_budget(
        self, fake_clients, monkeypatch
    ):
        FakeAsyncClient.disconnect_hangs = True
        monkeypatch.setattr(v1, "DISCONNECT_TIMEOUT_SECONDS", 0.1)
        monkeypatch.setattr(v1, "SHUTDOWN_CALL_TIMEOUT_SECONDS", 2.0)
        worker, state = self._direct_worker()
        worker.start()
        assert state.settled.wait(timeout=5)

        started = time.monotonic()
        worker.stop()

        assert time.monotonic() - started < 1.5
        assert worker._finished.is_set()
        assert fake_clients[0].socket_closed is True
        assert not worker._thread.is_alive()

    def test_stop_closes_the_socket_when_the_task_outlives_the_budget(
        self, fake_clients, monkeypatch
    ):
        FakeAsyncClient.disconnect_hangs = True
        monkeypatch.setattr(v1, "DISCONNECT_TIMEOUT_SECONDS", 3600)
        monkeypatch.setattr(v1, "SHUTDOWN_CALL_TIMEOUT_SECONDS", 0.2)
        monkeypatch.setattr(v1, "LOOP_STOP_JOIN_SECONDS", 2.0)
        worker, state = self._direct_worker()
        worker.start()
        assert state.settled.wait(timeout=5)

        worker.stop()

        assert fake_clients[0].socket_closed is True
        assert not worker._thread.is_alive()

    def test_unexpected_supervisor_failure_settles_the_first_run(self, monkeypatch):
        async def explode(self):
            raise RuntimeError("boom")

        monkeypatch.setattr(_SubscriptionWorker, "_connect_and_watch", explode)
        state = OPCUAReaderState()
        worker = _SubscriptionWorker(
            config=self._config(url="opc.tcp://host:4840"), state=state
        )

        worker.start()
        try:
            assert state.settled.wait(timeout=5)
        finally:
            worker.stop()

        assert state.permanent is True
        assert state.error == "OPC UA Reader stopped unexpectedly: RuntimeError: boom"
        assert not worker._thread.is_alive()


class TestClose:
    def test_close_stops_the_worker_once(self, block, workers):
        block.run(**run_kwargs())

        block.close()
        block.close()

        assert workers[0].stop_calls == 1

    def test_close_without_a_run_is_a_no_op(self, workers):
        OPCReaderBlockV1().close()

        assert workers == []

    def test_close_interrupts_a_first_run_still_waiting(self, block, workers):
        FakeWorker.on_start = staticmethod(lambda state: None)
        results = []
        runner = threading.Thread(
            target=lambda: results.append(block.run(**run_kwargs(timeout=60)))
        )
        runner.start()
        assert wait_for(lambda: workers and workers[0].started)

        started = time.monotonic()
        block.close()
        runner.join(timeout=5)

        assert time.monotonic() - started < 1.0
        assert not runner.is_alive()
        assert results == [
            {
                "value": None,
                "is_new": False,
                "error_status": True,
                "error_message": "OPC UA Reader was closed while connecting.",
            }
        ]
        assert workers[0].stop_calls == 1

    def test_run_after_close_starts_a_new_subscription(self, block, workers):
        block.run(**run_kwargs())
        block.close()

        result = block.run(**run_kwargs())

        assert result["is_new"] is True
        assert len(workers) == 2

    def test_garbage_collection_stops_the_worker(self, workers):
        block = OPCReaderBlockV1()
        block.run(**run_kwargs())
        [worker] = workers

        del block

        assert worker.stop_calls == 1

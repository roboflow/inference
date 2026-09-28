import threading
import time
from types import SimpleNamespace
from typing import List, get_args
from unittest.mock import patch

import pytest
import roboflow_workflows.environment as workflows_environment
from pydantic import ValidationError
from roboflow_workflows.enterprise_blocks.sinks import mqtt_common
from roboflow_workflows.enterprise_blocks.sinks.mqtt_reader import v1
from roboflow_workflows.enterprise_blocks.sinks.mqtt_reader.v1 import (
    LATEST_BUFFER_SIZE,
    NOT_CONNECTED_REPEATING_LAST,
    NOT_CONNECTED_WITHIN_TIMEOUT,
    NOT_RESUBSCRIBED_REPEATING_LAST,
    SEQUENTIAL_BUFFER_SIZE,
    SUBSCRIPTION_REFUSED_QOS,
    BlockManifest,
    MQTTReaderBlockV1,
    MQTTReaderState,
    mqtt_on_connect,
    mqtt_on_connect_fail,
    mqtt_on_disconnect,
    mqtt_on_message,
    mqtt_on_subscribe,
)
from roboflow_workflows.execution_engine.entities.workload import (
    WorkOperation,
    restriction_metadata_of,
)
from roboflow_workflows.execution_engine.introspection import blocks_loader
from roboflow_workflows.execution_engine.introspection.workload import (
    describe_workflow_workload,
)

CLIENT_CLASS_PATH = (
    "roboflow_workflows.enterprise_blocks.sinks.mqtt_reader.v1.mqtt.Client"
)
ENTERPRISE_PLUGIN = "roboflow_workflows.enterprise_blocks.loader"
HOSTED_RESTRICTION_CODE = "unavailable_on_hosted_platform"
PER_REQUEST_RESTRICTION_CODE = "connection_and_state_rebuilt_per_request"
HOSTED_PLATFORM_FLAGS = ("GCP_SERVERLESS", "LAMBDA")


def manifest_payload(**overrides) -> dict:
    payload = {
        "type": "roboflow_enterprise/mqtt_reader@v1",
        "name": "reader",
        "host": "localhost",
        "port": 1883,
        "topic": "plc/state",
    }
    payload.update(overrides)
    return payload


def run_kwargs(**overrides) -> dict:
    kwargs = {
        "host": "localhost",
        "port": 1883,
        "topic": "plc/state",
        "timeout": 0.05,
    }
    kwargs.update(overrides)
    return kwargs


class FakeClient:
    """Stand-in for paho's Client that drives the block's real callbacks.

    connect() succeeds unless `connect_error` is set, loop_start() fires
    on_connect with `connack_reason_code`, subscribe() answers with
    `granted_qos` unless `ack_subscriptions` is False, and deliver()
    pushes a message through on_message.
    """

    connect_error = None
    connack_reason_code = 0
    granted_qos = 0
    ack_subscriptions = True
    fire_on_connect = True
    # CONNACK flags handed to on_connect; a test sets {"session present": 1}
    connect_flags: dict = {}

    def __init__(self, client_id="", clean_session=None, userdata=None):
        # paho 1.6.1's constructor arguments the block chooses
        self.client_id = client_id
        self.clean_session = clean_session
        self.userdata = userdata
        # ordered record of the calls that matter for TLS: must precede connect()
        self.calls = []
        self.on_connect = None
        self.on_connect_fail = None
        self.on_subscribe = None
        self.on_message = None
        self.on_disconnect = None
        self.suppress_exceptions = False
        self.credentials = None
        self.connected_to = None
        self.subscriptions: List[tuple] = []
        self.loop_started = False
        self.loop_stopped = False
        self.disconnected = False
        self.reconnect_delays = None

    def username_pw_set(self, username, password=None):
        self.credentials = (username, password)

    def reconnect_delay_set(self, min_delay, max_delay):
        self.reconnect_delays = (min_delay, max_delay)

    def tls_set(self, ca_certs=None, **kwargs):
        self.calls.append(("tls_set", ca_certs))

    def tls_insecure_set(self, value):
        self.calls.append(("tls_insecure_set", value))

    def connect(self, host, port, keepalive=60):
        self.calls.append(("connect", host, port))
        if self.connect_error is not None:
            raise self.connect_error
        self.connected_to = (host, port)
        self.keepalive = keepalive

    def loop_start(self):
        self.loop_started = True
        if self.fire_on_connect:
            self.on_connect(
                self, self.userdata, dict(self.connect_flags), self.connack_reason_code
            )

    def loop_stop(self):
        self.loop_stopped = True

    def disconnect(self):
        self.disconnected = True

    def subscribe(self, topic, qos=0):
        self.subscriptions.append((topic, qos))
        if self.ack_subscriptions:
            self.on_subscribe(self, self.userdata, 1, [self.granted_qos])

    def deliver(self, payload, topic="plc/state", retain=False):
        message = SimpleNamespace(topic=topic, payload=payload, retain=retain)
        self.on_message(self, self.userdata, message)


@pytest.fixture
def clients() -> List[FakeClient]:
    created: List[FakeClient] = []

    def factory(client_id="", clean_session=None, userdata=None):
        client = FakeClient(
            client_id=client_id, clean_session=clean_session, userdata=userdata
        )
        created.append(client)
        return client

    with patch(CLIENT_CLASS_PATH, side_effect=factory):
        yield created


@pytest.fixture
def block() -> MQTTReaderBlockV1:
    return MQTTReaderBlockV1()


class TestManifest:
    def test_primary_identifier_is_namespaced(self):
        identifiers = get_args(BlockManifest.model_fields["type"].annotation)

        assert identifiers == ("roboflow_enterprise/mqtt_reader@v1",)

    def test_defaults(self):
        manifest = BlockManifest.model_validate(manifest_payload())

        assert manifest.read_mode == "latest"
        assert manifest.qos == 0
        assert manifest.timeout == 0.5
        assert manifest.username is None
        assert manifest.password is None
        assert manifest.client_id is None

    @pytest.mark.parametrize("client_id", ["line1-camera3", "$inputs.pipeline_name"])
    def test_client_id_accepts_literal_and_selector(self, client_id):
        manifest = BlockManifest.model_validate(manifest_payload(client_id=client_id))

        assert manifest.client_id == client_id

    def test_timeout_description_scopes_the_wait_to_the_first_connection(self):
        schema = BlockManifest.model_json_schema()

        description = schema["properties"]["timeout"]["description"]
        assert "first connection" in description
        assert "Later runs never wait" in description

    def test_outputs_unchanged_by_outage_handling(self):
        names = [output.name for output in BlockManifest.describe_outputs()]

        assert "error_status" in names
        assert "error_message" in names
        assert len(names) == 6

    def test_qos_description_states_the_persistent_session_rule(self):
        schema = BlockManifest.model_json_schema()

        assert "client_id" in schema["properties"]["qos"]["description"]
        assert "1 or 2" in schema["properties"]["qos"]["description"]

    @pytest.mark.parametrize("read_mode", ["everything", "next", "", None])
    def test_read_mode_must_be_latest_or_sequential(self, read_mode):
        with pytest.raises(ValidationError):
            BlockManifest.model_validate(manifest_payload(read_mode=read_mode))

    @pytest.mark.parametrize("read_mode", ["latest", "sequential"])
    def test_valid_read_modes_accepted(self, read_mode):
        manifest = BlockManifest.model_validate(manifest_payload(read_mode=read_mode))

        assert manifest.read_mode == read_mode

    @pytest.mark.parametrize("port", [0, -1, 65536])
    def test_literal_port_must_be_within_tcp_range(self, port):
        with pytest.raises(ValidationError):
            BlockManifest.model_validate(manifest_payload(port=port))

    @pytest.mark.parametrize(
        "timeout", [0, -1, float("nan"), float("inf"), float("-inf")]
    )
    def test_literal_timeout_must_be_finite_and_positive(self, timeout):
        with pytest.raises(ValidationError):
            BlockManifest.model_validate(manifest_payload(timeout=timeout))

    def test_selectors_accepted_for_connection_fields(self):
        manifest = BlockManifest.model_validate(
            manifest_payload(
                host="$inputs.host",
                port="$inputs.port",
                topic="$inputs.topic",
                username="$inputs.user",
                password="$inputs.secret",
                timeout="$inputs.timeout",
                qos="$inputs.qos",
            )
        )

        assert manifest.password == "$inputs.secret"

    def test_password_is_marked_private(self):
        schema = BlockManifest.model_json_schema()

        assert schema["properties"]["password"]["private"] is True

    def test_describe_outputs_order(self):
        names = [output.name for output in BlockManifest.describe_outputs()]

        assert names == [
            "value",
            "payload",
            "topic",
            "is_new",
            "error_status",
            "error_message",
        ]

    def test_restrictions_hard_for_hosted_serverless_only(self):
        restrictions = BlockManifest.get_restrictions()

        assert [r.severity.value for r in restrictions] == ["hard", "soft"]
        assert restrictions[0].applies_to_runtimes == [v1.Runtime.HOSTED_SERVERLESS]
        assert restrictions[1].applies_to_runtimes == [
            v1.Runtime.SELF_HOSTED_CPU,
            v1.Runtime.SELF_HOSTED_GPU,
            v1.Runtime.DEDICATED_DEPLOYMENT,
        ]
        assert restrictions[1].applies_to_input_modes == [v1.RuntimeInputMode.IMAGE]
        # the HTTP note must warn about persistent sessions, not only retained messages
        assert "client_id" in restrictions[1].note
        assert "InferencePipeline" in restrictions[1].note

    def test_block_declares_file_system_init_parameter_only(self):
        assert MQTTReaderBlockV1.get_init_parameters() == [
            "allow_access_to_file_system"
        ]


def by_code(restrictions) -> list:
    return sorted(restrictions, key=lambda restriction: restriction.code)


class TestWorkloadDeclarations:
    def test_declares_broker_io_and_buffering_between_runs(self):
        manifest = BlockManifest.model_validate(manifest_payload())

        assert manifest.discover_work_operations() == [
            WorkOperation.EXTERNAL_REQUEST,
            WorkOperation.TEMPORAL_BUFFERING,
        ]

    def test_declares_a_known_empty_resource_set(self):
        manifest = BlockManifest.model_validate(manifest_payload())

        # [] means "pulls no model or project"; None would mean "unknown"
        assert manifest.discover_dependent_resources() == []

    @pytest.mark.parametrize("ignore_environment_restrictions", [True, False])
    def test_actual_restrictions_are_complete_in_both_views(
        self, ignore_environment_restrictions
    ):
        manifest = BlockManifest.model_validate(manifest_payload())

        discovery = manifest.get_actual_restrictions(
            ignore_environment_restrictions=ignore_environment_restrictions
        )

        assert discovery.complete is True
        assert discovery.unknown_reasons == []
        assert [restriction.code for restriction in by_code(discovery.items)] == [
            PER_REQUEST_RESTRICTION_CODE,
            HOSTED_RESTRICTION_CODE,
        ]
        # the editor declaration carries the same entries, codes included
        assert by_code(discovery.items) == by_code(BlockManifest.get_restrictions())

    def test_view_switch_is_keyword_only(self):
        manifest = BlockManifest.model_validate(manifest_payload())

        with pytest.raises(TypeError):
            manifest.get_actual_restrictions(True)

    @pytest.mark.parametrize("ignore_environment_restrictions", [True, False])
    @pytest.mark.parametrize(
        "gcp_serverless, lambda_runtime",
        [(False, False), (True, False), (False, True), (True, True)],
    )
    def test_actual_restrictions_ignore_this_host_hosted_platform_flags(
        self,
        monkeypatch,
        gcp_serverless,
        lambda_runtime,
        ignore_environment_restrictions,
    ):
        # run() reads the flags its module imported; the declaration must not
        manifest = BlockManifest.model_validate(manifest_payload())
        baseline = manifest.get_actual_restrictions(
            ignore_environment_restrictions=ignore_environment_restrictions
        )
        for module in (v1, workflows_environment):
            for flag, value in zip(
                HOSTED_PLATFORM_FLAGS, (gcp_serverless, lambda_runtime)
            ):
                assert hasattr(module, flag), flag
                monkeypatch.setattr(module, flag, value)

        flipped = manifest.get_actual_restrictions(
            ignore_environment_restrictions=ignore_environment_restrictions
        )

        assert flipped == baseline
        assert flipped.complete is True

    def test_codes_keep_the_runtime_and_input_mode_axes(self):
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
        assert hosted.when.input_modes is None
        assert hosted.when.step_execution_modes is None
        # the runtime axis carries the condition; no host flag is named
        assert hosted.when.configuration_equals == {}
        assert per_request.severity is v1.Severity.SOFT
        assert set(per_request.when.runtimes) == {
            v1.Runtime.SELF_HOSTED_CPU,
            v1.Runtime.SELF_HOSTED_GPU,
            v1.Runtime.DEDICATED_DEPLOYMENT,
        }
        assert per_request.when.input_modes == [v1.RuntimeInputMode.IMAGE]
        assert per_request.when.step_execution_modes is None
        assert per_request.when.configuration_equals == {}

    def test_mutating_returned_restrictions_does_not_leak_into_later_calls(self):
        manifest = BlockManifest.model_validate(manifest_payload())
        legacy = BlockManifest.get_restrictions()
        actual = manifest.get_actual_restrictions(ignore_environment_restrictions=True)

        legacy[0].applies_to_runtimes.append(v1.Runtime.SELF_HOSTED_CPU)
        legacy[1].applies_to_input_modes.append(v1.RuntimeInputMode.VIDEO)
        actual.items[0].applies_to_runtimes.clear()
        legacy.pop()

        later_legacy = BlockManifest.get_restrictions()
        later_actual = manifest.get_actual_restrictions(
            ignore_environment_restrictions=True
        )
        assert [restriction.applies_to_runtimes for restriction in later_legacy] == [
            [v1.Runtime.HOSTED_SERVERLESS],
            [
                v1.Runtime.SELF_HOSTED_CPU,
                v1.Runtime.SELF_HOSTED_GPU,
                v1.Runtime.DEDICATED_DEPLOYMENT,
            ],
        ]
        assert later_legacy[1].applies_to_input_modes == [v1.RuntimeInputMode.IMAGE]
        assert later_actual.complete is True
        assert by_code(later_actual.items) == by_code(later_legacy)

    def test_legacy_editor_payload_carries_no_code(self):
        payloads = [
            restriction.to_dict() for restriction in BlockManifest.get_restrictions()
        ]

        notes = [payload.pop("note") for payload in payloads]
        assert all(notes)
        assert payloads == [
            {
                "severity": "hard",
                "applies_to_runtimes": [v1.Runtime.HOSTED_SERVERLESS.value],
            },
            {
                "severity": "soft",
                "applies_to_runtimes": [
                    v1.Runtime.SELF_HOSTED_CPU.value,
                    v1.Runtime.SELF_HOSTED_GPU.value,
                    v1.Runtime.DEDICATED_DEPLOYMENT.value,
                ],
                "applies_to_input_modes": [v1.RuntimeInputMode.IMAGE.value],
            },
        ]

    def test_public_introspection_describes_the_step_completely(
        self, monkeypatch, clients
    ):
        definition = {
            "version": "1.0",
            "inputs": [{"type": "WorkflowParameter", "name": "mqtt_topic"}],
            "steps": [manifest_payload(topic="$inputs.mqtt_topic")],
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
        codes = [restriction.code for restriction in step.restrictions.items]
        assert step.node_id == "$steps.reader"
        assert step.operations.complete is True
        assert step.operations.items == [
            WorkOperation.EXTERNAL_REQUEST,
            WorkOperation.TEMPORAL_BUFFERING,
        ]
        assert step.restrictions.complete is True
        assert sorted(codes) == [PER_REQUEST_RESTRICTION_CODE, HOSTED_RESTRICTION_CODE]
        assert step.resources.complete is True
        assert step.resources.items == []
        assert description.summary.models.complete is True
        assert description.summary.models.items == []
        # compile-time facts only: no MQTT client was built
        assert clients == []


class TestCallbacks:
    def test_state_buffer_sizes_follow_read_mode_constants(self):
        assert LATEST_BUFFER_SIZE == 1
        assert SEQUENTIAL_BUFFER_SIZE == 1000
        assert MQTTReaderState("t", 0, LATEST_BUFFER_SIZE).messages.maxlen == 1

    def test_on_connect_subscribes_and_sets_event_for_accepted_connack(self):
        state = MQTTReaderState(topic="plc/#", qos=1, buffer_size=1)
        client = FakeClient(userdata=state)
        client.on_subscribe = mqtt_on_subscribe

        mqtt_on_connect(client, state, {}, 0)

        assert client.subscriptions == [("plc/#", 1)]
        assert state.connected.is_set()

    @pytest.mark.parametrize(
        "flags, expected",
        [
            ({}, "False"),
            ({"session present": 0}, "False"),
            ({"session present": 1}, "True"),
        ],
    )
    def test_on_connect_logs_session_present_flag(self, caplog, flags, expected):
        state = MQTTReaderState(topic="plc/#", qos=1, buffer_size=1)
        client = FakeClient(userdata=state)
        client.on_subscribe = mqtt_on_subscribe

        with caplog.at_level("INFO", logger="inference"):
            mqtt_on_connect(client, state, flags, 0)

        assert f"session present: {expected}" in caplog.text

    def test_on_connect_resubscribes_on_every_connect(self):
        state = MQTTReaderState(topic="plc/#", qos=0, buffer_size=1)
        client = FakeClient(userdata=state)
        client.on_subscribe = mqtt_on_subscribe

        mqtt_on_connect(client, state, {}, 0)
        mqtt_on_disconnect(client, state, 1)
        mqtt_on_connect(client, state, {}, 0)

        assert client.subscriptions == [("plc/#", 0), ("plc/#", 0)]
        assert state.subscribed.is_set()

    @pytest.mark.parametrize("reason_code", [1, 2, 3, 4, 5])
    def test_on_connect_does_not_subscribe_for_rejected_connack(self, reason_code):
        state = MQTTReaderState(topic="plc/#", qos=0, buffer_size=1)
        client = FakeClient(userdata=state)
        state.connected.set()

        mqtt_on_connect(client, state, {}, reason_code)

        assert client.subscriptions == []
        assert not state.connected.is_set()
        assert state.connack.is_set()
        assert state.refused_code == reason_code

    @pytest.mark.parametrize("reason_code", [1, 2, 4, 5])
    def test_on_connect_stops_the_loop_for_a_permanent_refusal(self, reason_code):
        state = MQTTReaderState(topic="plc/#", qos=0, buffer_size=1)
        client = FakeClient(userdata=state)

        mqtt_on_connect(client, state, {}, reason_code)

        assert client.disconnected is True

    def test_on_connect_keeps_the_loop_for_broker_unavailable(self):
        state = MQTTReaderState(topic="plc/#", qos=0, buffer_size=1)
        client = FakeClient(userdata=state)

        mqtt_on_connect(client, state, {}, 3)

        assert client.disconnected is False
        assert state.refused_code == 3

    @pytest.mark.parametrize("reason_code", [0, 3, 5])
    def test_on_connect_completes_the_state_before_waking_a_waiter(self, reason_code):
        # a run wakes on `connack`; what it then reads must already be final
        state = MQTTReaderState(topic="plc/#", qos=1, buffer_size=1)
        seen = {}

        class RecordingEvent(threading.Event):
            def set(self):
                seen["refused_code"] = state.refused_code
                seen["connected"] = state.connected.is_set()
                super().set()

        state.connack = RecordingEvent()
        client = FakeClient(userdata=state)
        client.on_subscribe = mqtt_on_subscribe

        mqtt_on_connect(client, state, {}, reason_code)

        assert seen == {
            "refused_code": reason_code or None,
            "connected": reason_code == 0,
        }

    def test_on_connect_logs_the_refusal_reason(self, caplog):
        state = MQTTReaderState(topic="plc/#", qos=0, buffer_size=1)

        with caplog.at_level("ERROR", logger="inference"):
            mqtt_on_connect(FakeClient(userdata=state), state, {}, 5)

        assert "not authorised" in caplog.text
        assert "code 5" in caplog.text

    def test_accepted_connack_after_broker_unavailable_resets_refusal(self):
        state = MQTTReaderState(topic="plc/#", qos=0, buffer_size=1)
        client = FakeClient(userdata=state)
        client.on_subscribe = mqtt_on_subscribe

        mqtt_on_connect(client, state, {}, 3)
        mqtt_on_connect(client, state, {}, 0)

        assert state.refused_code is None
        assert state.connected.is_set()
        assert state.connack.is_set()

    def test_on_connect_fail_matches_paho_two_argument_signature(self):
        state = MQTTReaderState(topic="t", qos=0, buffer_size=1)
        state.connected.set()

        mqtt_on_connect_fail(FakeClient(), state)

        assert not state.connected.is_set()

    def test_on_subscribe_flags_refused_subscription(self):
        state = MQTTReaderState(topic="t", qos=0, buffer_size=1)

        mqtt_on_subscribe(FakeClient(), state, 1, [SUBSCRIPTION_REFUSED_QOS])

        assert state.subscribe_failed is True
        assert not state.subscribed.is_set()

    @pytest.mark.parametrize("granted", [0, 1, 2])
    def test_on_subscribe_sets_event_for_granted_qos(self, granted):
        state = MQTTReaderState(topic="t", qos=2, buffer_size=1)

        mqtt_on_subscribe(FakeClient(), state, 1, [granted])

        assert state.subscribed.is_set()
        assert state.subscribe_failed is False

    def test_on_message_appends_raw_tuple_only(self):
        state = MQTTReaderState(topic="t", qos=0, buffer_size=10)
        message = SimpleNamespace(topic="plc/a", payload=b"\xff\xfe", retain=1)

        mqtt_on_message(FakeClient(), state, message)

        assert list(state.messages) == [("plc/a", b"\xff\xfe", True)]

    def test_latest_buffer_keeps_only_newest(self):
        state = MQTTReaderState(topic="t", qos=0, buffer_size=LATEST_BUFFER_SIZE)
        for payload in (b"A", b"B", b"C"):
            mqtt_on_message(
                FakeClient(),
                state,
                SimpleNamespace(topic="t", payload=payload, retain=0),
            )

        assert list(state.messages) == [("t", b"C", False)]

    def test_on_disconnect_clears_both_events(self):
        state = MQTTReaderState(topic="t", qos=0, buffer_size=1)
        state.connected.set()
        state.subscribed.set()

        mqtt_on_disconnect(FakeClient(), state, 1)

        assert not state.connected.is_set()
        assert not state.subscribed.is_set()

    def test_on_disconnect_after_transport_drop_clears_connack(self):
        state = MQTTReaderState(topic="t", qos=0, buffer_size=1)
        state.connack.set()

        mqtt_on_disconnect(FakeClient(), state, 1)

        assert not state.connack.is_set()

    def test_on_disconnect_after_refusal_keeps_connack(self):
        state = MQTTReaderState(topic="t", qos=0, buffer_size=1)
        client = FakeClient(userdata=state)
        mqtt_on_connect(client, state, {}, 5)

        mqtt_on_disconnect(client, state, 0)

        assert state.connack.is_set()
        assert state.refused_code == 5

    def test_on_disconnect_accepts_mqtt5_properties_argument(self):
        state = MQTTReaderState(topic="t", qos=0, buffer_size=1)
        state.connected.set()

        mqtt_on_disconnect(FakeClient(), state, 0, None)

        assert not state.connected.is_set()


class TestInputValidation:
    @pytest.mark.parametrize(
        "timeout", [0, -1, float("nan"), float("inf"), "abc", None]
    )
    def test_invalid_timeout_rejected_before_client_construction(
        self, clients, block, timeout
    ):
        result = block.run(**run_kwargs(timeout=timeout))

        assert result["error_status"] is True
        assert "timeout" in result["error_message"].lower()
        assert clients == []

    @pytest.mark.parametrize("port", [0, 65536, "abc", 1.5, True, None])
    def test_invalid_port_rejected_before_client_construction(
        self, clients, block, port
    ):
        result = block.run(**run_kwargs(port=port))

        assert result["error_status"] is True
        assert "port" in result["error_message"].lower()
        assert clients == []

    def test_selector_supplied_port_coerced_like_manifest(self, clients, block):
        result = block.run(**run_kwargs(port="1883"))

        assert result["error_status"] is False
        assert clients[0].connected_to == ("localhost", 1883)

    @pytest.mark.parametrize("qos", [-1, 3, "x", 1.5, True, None])
    def test_invalid_qos_rejected_before_client_construction(self, clients, block, qos):
        result = block.run(**run_kwargs(qos=qos))

        assert result["error_status"] is True
        assert "qos" in result["error_message"].lower()
        assert clients == []

    def test_selector_supplied_qos_used_for_subscription(self, clients, block):
        block.run(**run_kwargs(qos="2"))

        assert clients[0].subscriptions == [("plc/state", 2)]

    def test_invalid_read_mode_rejected(self, clients, block):
        result = block.run(**run_kwargs(read_mode="everything"))

        assert result["error_status"] is True
        assert "read_mode" in result["error_message"]
        assert clients == []

    def test_password_without_username_rejected(self, clients, block):
        result = block.run(**run_kwargs(password="secret"))

        assert result["error_status"] is True
        assert "username" in result["error_message"].lower()
        assert clients == []

    def test_username_and_password_configure_authentication(self, clients, block):
        block.run(**run_kwargs(username="user", password="secret"))

        assert clients[0].credentials == ("user", "secret")

    @pytest.mark.parametrize("flag", ["GCP_SERVERLESS", "LAMBDA"])
    def test_hosted_platform_rejected_without_client(self, clients, block, flag):
        with patch.object(v1, flag, True):
            result = block.run(**run_kwargs())

        assert result["error_status"] is True
        assert "hosted" in result["error_message"].lower()
        assert clients == []

    @pytest.mark.parametrize("topic", ["", None, 5])
    def test_invalid_topic_rejected_before_client_construction(
        self, clients, block, topic
    ):
        result = block.run(**run_kwargs(topic=topic))

        assert result["error_status"] is True
        assert "topic" in result["error_message"].lower()
        assert clients == []

    @pytest.mark.parametrize("client_id", [None, "", "   "])
    def test_blank_client_id_builds_a_clean_session_client(
        self, clients, block, client_id
    ):
        result = block.run(**run_kwargs(client_id=client_id))

        assert result["error_status"] is False
        assert clients[0].client_id == ""
        assert clients[0].clean_session is None
        assert clients[0].userdata is block._state

    def test_client_id_builds_a_persistent_session_client(self, clients, block):
        result = block.run(**run_kwargs(client_id=" line1 ", qos=1))

        assert result["error_status"] is False
        assert clients[0].client_id == "line1"
        assert clients[0].clean_session is False
        assert clients[0].userdata is block._state
        assert clients[0].subscriptions == [("plc/state", 1)]

    def test_client_id_with_qos_zero_rejected_before_client_construction(
        self, clients, block
    ):
        result = block.run(**run_kwargs(client_id="line1"))

        assert result["error_status"] is True
        assert "client_id" in result["error_message"]
        assert "qos 1 or 2" in result["error_message"]
        assert clients == []

    @pytest.mark.parametrize("client_id", [5, 1.5, True, ["line1"]])
    def test_non_string_client_id_rejected_before_client_construction(
        self, clients, block, client_id
    ):
        result = block.run(**run_kwargs(client_id=client_id, qos=1))

        assert result["error_status"] is True
        assert "client_id" in result["error_message"]
        assert clients == []

    def test_error_outputs_carry_every_declared_key(self, clients, block):
        result = block.run(**run_kwargs(port=0))

        assert result == {
            "value": None,
            "payload": {},
            "topic": None,
            "is_new": False,
            "error_status": True,
            "error_message": result["error_message"],
        }


class TestConnectionLifecycle:
    def test_first_run_connects_subscribes_and_starts_loop(self, clients, block):
        result = block.run(**run_kwargs(username="u", password="p", timeout=0.25))

        client = clients[0]
        assert result["error_status"] is False
        assert client.connected_to == ("localhost", 1883)
        assert client.loop_started is True
        assert client.suppress_exceptions is True
        assert client.subscriptions == [("plc/state", 0)]
        assert client.on_connect is mqtt_on_connect
        assert client.on_connect_fail is mqtt_on_connect_fail
        assert client.on_subscribe is mqtt_on_subscribe
        assert client.on_message is mqtt_on_message
        assert client.on_disconnect is mqtt_on_disconnect
        assert client.reconnect_delays == (0.125, 0.5)
        assert client._connect_timeout == 0.25
        assert client.keepalive == 15

    def test_client_reused_across_runs(self, clients, block):
        block.run(**run_kwargs())
        block.run(**run_kwargs())

        assert len(clients) == 1

    def test_unreachable_broker_on_first_run_leaves_nothing_behind(
        self, clients, block, monkeypatch
    ):
        monkeypatch.setattr(FakeClient, "connect_error", OSError("connection refused"))
        result = block.run(**run_kwargs())

        assert result["error_status"] is True
        assert "not connected" in result["error_message"].lower()
        assert block._client is None
        assert clients[0].loop_started is False

    def test_next_run_after_failed_connect_retries_with_fresh_client(
        self, clients, block, monkeypatch
    ):
        monkeypatch.setattr(FakeClient, "connect_error", OSError("connection refused"))
        block.run(**run_kwargs())

        monkeypatch.setattr(FakeClient, "connect_error", None)
        result = block.run(**run_kwargs())

        assert result["error_status"] is False
        assert len(clients) == 2

    def test_setup_failure_stops_loop_and_resets_client(self, clients, block):
        with patch.object(
            FakeClient, "reconnect_delay_set", side_effect=RuntimeError("boom")
        ):
            result = block.run(**run_kwargs())

        assert result["error_status"] is True
        assert "initialize" in result["error_message"].lower()
        assert block._client is None
        assert clients[0].loop_stopped is True

    def test_connection_not_established_within_timeout_reported(
        self, clients, block, monkeypatch
    ):
        monkeypatch.setattr(FakeClient, "fire_on_connect", False)
        result = block.run(**run_kwargs())

        assert result["error_status"] is True
        assert "not connected" in result["error_message"].lower()
        assert block._client is clients[0]

    def test_rejected_connack_reported_as_a_refusal(self, clients, block, monkeypatch):
        monkeypatch.setattr(FakeClient, "connack_reason_code", 5)
        result = block.run(**run_kwargs())

        assert result["error_status"] is True
        assert "not authorised" in result["error_message"]
        assert "code 5" in result["error_message"]
        assert "client_id" in result["error_message"]
        assert "Raise 'timeout'" not in result["error_message"]
        assert clients[0].subscriptions == []
        assert clients[0].disconnected is True

    @pytest.mark.parametrize(
        "reason_code, reason",
        [
            (1, "unacceptable protocol version"),
            (2, "identifier rejected"),
            (4, "bad user name or password"),
        ],
    )
    def test_other_permanent_refusals_name_their_reason(
        self, clients, block, monkeypatch, reason_code, reason
    ):
        monkeypatch.setattr(FakeClient, "connack_reason_code", reason_code)
        result = block.run(**run_kwargs())

        assert reason in result["error_message"]
        assert clients[0].disconnected is True

    def test_refusal_is_reported_without_waiting_for_timeout(
        self, clients, block, monkeypatch
    ):
        monkeypatch.setattr(FakeClient, "connack_reason_code", 5)

        started = time.monotonic()
        result = block.run(**run_kwargs(timeout=2.0))

        assert result["error_status"] is True
        assert time.monotonic() - started < 0.5

    def test_next_run_after_refusal_repeats_it_without_a_new_connection(
        self, clients, block, monkeypatch
    ):
        monkeypatch.setattr(FakeClient, "connack_reason_code", 5)
        first = block.run(**run_kwargs(timeout=2.0))

        started = time.monotonic()
        second = block.run(**run_kwargs(timeout=2.0))

        assert second == first
        assert time.monotonic() - started < 0.5
        assert len(clients) == 1
        assert [call for call in clients[0].calls if call[0] == "connect"] == [
            ("connect", "localhost", 1883)
        ]

    def test_broker_unavailable_keeps_retrying_and_names_the_reason(
        self, clients, block, monkeypatch
    ):
        monkeypatch.setattr(FakeClient, "connack_reason_code", 3)
        result = block.run(**run_kwargs())

        assert result["error_status"] is True
        assert "broker unavailable" in result["error_message"]
        assert "retrying in the background" in result["error_message"]
        assert clients[0].disconnected is False

    def test_run_succeeds_once_an_unavailable_broker_accepts(
        self, clients, block, monkeypatch
    ):
        monkeypatch.setattr(FakeClient, "connack_reason_code", 3)
        block.run(**run_kwargs())
        client = clients[0]
        client.on_connect(client, client.userdata, {}, 0)

        result = block.run(**run_kwargs())

        assert result["error_status"] is False
        assert block._state.refused_code is None

    def test_no_connack_at_all_reports_not_connected(self, clients, block, monkeypatch):
        monkeypatch.setattr(FakeClient, "fire_on_connect", False)
        result = block.run(**run_kwargs())

        assert result["error_status"] is True
        assert result["error_message"] == NOT_CONNECTED_WITHIN_TIMEOUT

    def test_close_after_refusal_completes(self, clients, block, monkeypatch):
        monkeypatch.setattr(FakeClient, "connack_reason_code", 5)
        block.run(**run_kwargs())

        block.close()

        assert block._client is None
        assert clients[0].loop_stopped is True

    def test_refused_subscription_reported_and_client_kept(
        self, clients, block, monkeypatch
    ):
        monkeypatch.setattr(FakeClient, "granted_qos", SUBSCRIPTION_REFUSED_QOS)
        result = block.run(**run_kwargs())

        assert result["error_status"] is True
        assert "refused" in result["error_message"].lower()
        assert block._client is clients[0]
        assert block._state.connected.is_set()

    def test_refused_subscription_does_not_wait_for_timeout(
        self, clients, block, monkeypatch
    ):
        monkeypatch.setattr(FakeClient, "granted_qos", SUBSCRIPTION_REFUSED_QOS)
        block.run(**run_kwargs(timeout=1.0))

        started = time.monotonic()
        result = block.run(**run_kwargs(timeout=1.0))

        assert result["error_status"] is True
        assert "refused" in result["error_message"].lower()
        assert time.monotonic() - started < 0.5

    def test_subscribe_error_reported_through_refused_path(
        self, clients, block, monkeypatch
    ):
        def failing_subscribe(client, topic, qos=0):
            raise ValueError("Invalid topic.")

        monkeypatch.setattr(FakeClient, "subscribe", failing_subscribe)

        result = block.run(**run_kwargs())

        assert result["error_status"] is True
        assert "refused" in result["error_message"].lower()
        assert block._state.connected.is_set()

    def test_unacknowledged_subscription_reported(self, clients, block, monkeypatch):
        monkeypatch.setattr(FakeClient, "ack_subscriptions", False)
        result = block.run(**run_kwargs())

        assert result["error_status"] is True
        assert "acknowledged" in result["error_message"].lower()

    @pytest.mark.parametrize(
        "change",
        [
            {"host": "other"},
            {"port": 8883},
            {"username": "u"},
            {"username": "u", "password": "p"},
            {"timeout": 0.1},
            {"topic": "plc/other"},
            {"qos": 1},
            {"read_mode": "sequential"},
            {"qos": 1, "client_id": "line1"},
        ],
    )
    def test_changed_parameters_rejected_and_connection_kept(
        self, clients, block, change
    ):
        block.run(**run_kwargs())

        result = block.run(**run_kwargs(**change))

        assert result["error_status"] is True
        assert "changed between runs" in result["error_message"]
        assert len(clients) == 1
        assert clients[0].disconnected is False

    @pytest.mark.parametrize("later_client_id", [None, "", "line2"])
    def test_dropped_or_changed_client_id_rejected_and_connection_kept(
        self, clients, block, later_client_id
    ):
        block.run(**run_kwargs(client_id="line1", qos=1))

        result = block.run(**run_kwargs(client_id=later_client_id, qos=1))

        assert result["error_status"] is True
        assert "client_id" in result["error_message"]
        assert "changed between runs" in result["error_message"]
        assert len(clients) == 1
        assert clients[0].disconnected is False

    def test_same_client_id_with_different_whitespace_is_not_a_change(
        self, clients, block
    ):
        block.run(**run_kwargs(client_id="line1", qos=1))

        result = block.run(**run_kwargs(client_id=" line1 ", qos=1))

        assert result["error_status"] is False
        assert len(clients) == 1

    def test_close_disconnects_stops_loop_and_clears_state(self, clients, block):
        block.run(**run_kwargs())
        state = block._state

        block.close()

        assert clients[0].disconnected is True
        assert clients[0].loop_stopped is True
        assert block._client is None
        assert block._state is None
        assert not state.connected.is_set()
        assert not state.subscribed.is_set()

    def test_close_resets_last_outputs(self, clients, block):
        block.run(**run_kwargs())
        clients[0].deliver(b"x")
        block.run(**run_kwargs())

        block.close()
        result = block.run(**run_kwargs())

        assert len(clients) == 2
        assert result["value"] is None
        assert result["is_new"] is False

    def test_close_without_client_is_noop(self, block):
        block.close()

        assert block._client is None

    def test_close_when_disconnect_raises_still_stops_loop(self, clients, block):
        block.run(**run_kwargs())
        with patch.object(FakeClient, "disconnect", side_effect=RuntimeError("boom")):
            block.close()

        assert clients[0].loop_stopped is True
        assert block._client is None


class TestReading:
    def test_empty_first_run(self, clients, block):
        result = block.run(**run_kwargs())

        assert result == {
            "value": None,
            "payload": {},
            "topic": None,
            "is_new": False,
            "error_status": False,
            "error_message": None,
        }

    def test_json_message_surfaced_with_parsed_payload(self, clients, block):
        block.run(**run_kwargs())
        clients[0].deliver(b'{"state": "RUNNING"}')

        result = block.run(**run_kwargs())

        assert result == {
            "value": '{"state": "RUNNING"}',
            "payload": {"state": "RUNNING"},
            "topic": "plc/state",
            "is_new": True,
            "error_status": False,
            "error_message": None,
        }

    def test_retained_message_delivered_before_first_run_is_surfaced(
        self, clients, block
    ):
        original_loop_start = FakeClient.loop_start

        def loop_start_with_retained(client):
            original_loop_start(client)
            client.deliver(b'{"state": "RUNNING"}', retain=True)

        with patch.object(FakeClient, "loop_start", loop_start_with_retained):
            result = block.run(**run_kwargs())

        assert result["value"] == '{"state": "RUNNING"}'
        assert result["is_new"] is True

    def test_retained_redelivery_of_same_message_is_not_new(self, clients, block):
        block.run(**run_kwargs())
        clients[0].deliver(b'{"state": "RUNNING"}', retain=True)
        first = block.run(**run_kwargs())
        # a reconnect resubscribes and the broker re-sends the retained copy
        clients[0].deliver(b'{"state": "RUNNING"}', retain=True)

        second = block.run(**run_kwargs())

        assert first["is_new"] is True
        assert second == {**first, "is_new": False}

    def test_retained_message_with_new_payload_is_new(self, clients, block):
        block.run(**run_kwargs())
        clients[0].deliver(b'{"state": "RUNNING"}', retain=True)
        block.run(**run_kwargs())
        clients[0].deliver(b'{"state": "PAUSED"}', retain=True)

        result = block.run(**run_kwargs())

        assert result["payload"] == {"state": "PAUSED"}
        assert result["is_new"] is True

    def test_non_retained_duplicate_payload_is_new(self, clients, block):
        block.run(**run_kwargs())
        clients[0].deliver(b"tick")
        block.run(**run_kwargs())
        clients[0].deliver(b"tick")

        result = block.run(**run_kwargs())

        assert result["is_new"] is True

    def test_nothing_new_repeats_previous_outputs_with_is_new_false(
        self, clients, block
    ):
        block.run(**run_kwargs())
        clients[0].deliver(b'{"state": "RUNNING"}', topic="plc/line1")
        first = block.run(**run_kwargs())

        second = block.run(**run_kwargs())

        assert first["is_new"] is True
        assert second == {**first, "is_new": False}

    def test_latest_mode_returns_newest_and_skips_older(self, clients, block):
        block.run(**run_kwargs())
        for payload in (b"A", b"B", b"C"):
            clients[0].deliver(payload)

        result = block.run(**run_kwargs())
        repeat = block.run(**run_kwargs())

        assert result["value"] == "C"
        assert result["is_new"] is True
        assert repeat["value"] == "C"
        assert repeat["is_new"] is False

    def test_sequential_mode_returns_messages_in_order_one_per_run(
        self, clients, block
    ):
        block.run(**run_kwargs(read_mode="sequential"))
        for payload in (b"A", b"B", b"C"):
            clients[0].deliver(payload)

        results = [block.run(**run_kwargs(read_mode="sequential")) for _ in range(4)]

        assert [r["value"] for r in results] == ["A", "B", "C", "C"]
        assert [r["is_new"] for r in results] == [True, True, True, False]

    def test_sequential_mode_drops_oldest_beyond_buffer(self, clients, block):
        block.run(**run_kwargs(read_mode="sequential"))
        for index in range(SEQUENTIAL_BUFFER_SIZE + 1):
            clients[0].deliver(str(index).encode())

        result = block.run(**run_kwargs(read_mode="sequential"))

        assert result["value"] == "1"
        assert len(block._state.messages) == SEQUENTIAL_BUFFER_SIZE - 1

    def test_topic_output_reports_message_topic_for_wildcard_filters(
        self, clients, block
    ):
        block.run(**run_kwargs(topic="plc/#"))
        clients[0].deliver(b"1", topic="plc/line2/state")

        result = block.run(**run_kwargs(topic="plc/#"))

        assert result["topic"] == "plc/line2/state"

    def test_non_json_payload_gives_text_and_empty_payload(self, clients, block):
        block.run(**run_kwargs())
        clients[0].deliver(b"hello")

        result = block.run(**run_kwargs())

        assert result["value"] == "hello"
        assert result["payload"] == {}
        assert result["error_status"] is False

    def test_json_non_object_gives_empty_payload(self, clients, block):
        block.run(**run_kwargs())
        clients[0].deliver(b"[1, 2]")

        result = block.run(**run_kwargs())

        assert result["value"] == "[1, 2]"
        assert result["payload"] == {}

    def test_non_utf8_payload_reported_as_error_without_previous_outputs(
        self, clients, block
    ):
        block.run(**run_kwargs())
        clients[0].deliver(b'{"state": "RUNNING"}')
        block.run(**run_kwargs())
        clients[0].deliver(b"\xff\xfe")

        result = block.run(**run_kwargs())

        assert result["error_status"] is True
        assert "utf-8" in result["error_message"].lower()
        assert result["value"] is None
        assert result["payload"] == {}
        assert result["is_new"] is False

    def test_message_received_on_network_thread_visible_to_run(self, clients, block):
        block.run(**run_kwargs())
        delivered = threading.Event()

        def deliver():
            clients[0].deliver(b"from-thread")
            delivered.set()

        threading.Thread(target=deliver).start()
        assert delivered.wait(timeout=1)

        result = block.run(**run_kwargs())

        assert result["value"] == "from-thread"

    def test_no_raw_prints_in_any_code_path(self, clients, block):
        with patch("builtins.print") as mock_print:
            block.run(**run_kwargs())
            clients[0].deliver(b"x")
            block.run(**run_kwargs())
            block.run(**run_kwargs(port=0))
            block.close()

        mock_print.assert_not_called()


class TestOutageBookkeeping:
    """The two helpers behind the outage outputs, driven directly."""

    def warm_block_with_last(self, clients, block):
        block.run(**run_kwargs())
        clients[0].deliver(b'{"state": "RUNNING"}')
        block.run(**run_kwargs())
        return block

    def test_repeat_with_error_returns_last_message_with_error_fields(
        self, clients, block
    ):
        self.warm_block_with_last(clients, block)

        result = block._repeat_last_with_error("boom")

        assert result == {
            "value": '{"state": "RUNNING"}',
            "payload": {"state": "RUNNING"},
            "topic": "plc/state",
            "is_new": False,
            "error_status": True,
            "error_message": "boom",
        }

    def test_repeat_with_error_without_last_message_is_the_empty_failure(
        self, clients, block
    ):
        block.run(**run_kwargs())

        result = block._repeat_last_with_error("boom")

        assert result == {
            "value": None,
            "payload": {},
            "topic": None,
            "is_new": False,
            "error_status": True,
            "error_message": "boom",
        }

    def test_outage_logs_one_error_line_naming_the_instance(
        self, clients, block, caplog
    ):
        self.warm_block_with_last(clients, block)

        with caplog.at_level("ERROR", logger="inference"):
            block._repeat_last_with_error("boom")
            block._repeat_last_with_error("boom")

        errors = [r for r in caplog.records if r.levelname == "ERROR"]
        assert len(errors) == 1
        assert "MQTT Reader localhost:1883 'plc/state': boom" in errors[0].getMessage()
        assert block._outage_runs == 2

    def test_recovery_logs_one_info_line_with_duration_and_count_and_resets(
        self, clients, block, caplog
    ):
        self.warm_block_with_last(clients, block)
        block._repeat_last_with_error("boom")
        block._repeat_last_with_error("boom")

        with caplog.at_level("INFO", logger="inference"):
            block._log_recovery_if_needed()
            block._log_recovery_if_needed()

        infos = [
            r
            for r in caplog.records
            if r.levelname == "INFO" and "restored after" in r.getMessage()
        ]
        assert len(infos) == 1
        assert "2 runs returned the last message" in infos[0].getMessage()
        assert "localhost:1883 'plc/state'" in infos[0].getMessage()
        assert block._outage_started_at is None
        assert block._outage_runs == 0

    def test_recovery_while_healthy_logs_nothing(self, clients, block, caplog):
        self.warm_block_with_last(clients, block)

        with caplog.at_level("INFO", logger="inference"):
            block._log_recovery_if_needed()

        assert "restored after" not in caplog.text

    def test_first_run_keeps_the_resolved_address_for_the_log(self, clients, block):
        block.run(**run_kwargs())

        assert (block._host, block._port) == ("localhost", 1883)

    def test_close_resets_the_bookkeeping(self, clients, block):
        self.warm_block_with_last(clients, block)
        block._repeat_last_with_error("boom")
        block._subscribed_once = True

        block.close()

        assert block._subscribed_once is False
        assert block._outage_started_at is None
        assert block._outage_runs == 0
        assert (block._host, block._port) == (None, None)


def drop(client: FakeClient) -> None:
    client.on_disconnect(client, client.userdata, 1)


def reconnect(client: FakeClient, reason_code: int = 0) -> None:
    client.on_connect(client, client.userdata, {}, reason_code)


RUNNING = b'{"state": "RUNNING"}'
PAUSED = b'{"state": "PAUSED"}'


class TestOutage:
    """A warm instance (first run reached SUBACK) while the broker is away."""

    def warm(self, clients, block, **kwargs):
        block.run(**run_kwargs(**kwargs))
        clients[0].deliver(RUNNING)
        first = block.run(**run_kwargs(**kwargs))
        assert first["is_new"] is True
        return clients[0], first

    def test_drop_repeats_last_message_with_error_without_waiting(self, clients, block):
        client, first = self.warm(clients, block, timeout=2.0)
        drop(client)

        started = time.monotonic()
        result = block.run(**run_kwargs(timeout=2.0))

        assert time.monotonic() - started < 0.5
        assert result == {
            **first,
            "is_new": False,
            "error_status": True,
            "error_message": NOT_CONNECTED_REPEATING_LAST,
        }
        assert "not connected" in result["error_message"]

    def test_repeated_runs_during_outage_log_one_error_line(
        self, clients, block, caplog
    ):
        client, _ = self.warm(clients, block)
        drop(client)

        with caplog.at_level("ERROR", logger="inference"):
            results = [block.run(**run_kwargs()) for _ in range(4)]

        assert len({str(r) for r in results}) == 1
        errors = [r for r in caplog.records if r.levelname == "ERROR"]
        assert len(errors) == 1
        assert "localhost:1883 'plc/state'" in errors[0].getMessage()

    def test_buffered_messages_drain_before_the_outage_is_reported(
        self, clients, block
    ):
        client, _ = self.warm(clients, block, read_mode="sequential")
        client.deliver(b"A")
        client.deliver(b"B")
        drop(client)

        results = [block.run(**run_kwargs(read_mode="sequential")) for _ in range(3)]

        assert [r["value"] for r in results] == ["A", "B", "B"]
        assert [r["is_new"] for r in results] == [True, True, False]
        assert [r["error_status"] for r in results] == [False, False, True]

    def test_drop_before_any_message_returns_empty_failure_without_waiting(
        self, clients, block
    ):
        block.run(**run_kwargs(timeout=2.0))
        drop(clients[0])

        started = time.monotonic()
        result = block.run(**run_kwargs(timeout=2.0))

        assert time.monotonic() - started < 0.5
        assert result["payload"] == {}
        assert result["value"] is None
        assert result["error_status"] is True
        assert result["error_message"] == NOT_CONNECTED_REPEATING_LAST

    def test_recovery_returns_live_message_and_logs_once(self, clients, block, caplog):
        client, _ = self.warm(clients, block)
        drop(client)
        for _ in range(3):
            block.run(**run_kwargs())

        reconnect(client)
        client.deliver(PAUSED)
        with caplog.at_level("INFO", logger="inference"):
            result = block.run(**run_kwargs())
            block.run(**run_kwargs())

        assert result["payload"] == {"state": "PAUSED"}
        assert result["is_new"] is True
        assert result["error_status"] is False
        infos = [
            r.getMessage() for r in caplog.records if "restored after" in r.getMessage()
        ]
        assert len(infos) == 1
        assert "3 runs returned the last message" in infos[0]

    def test_second_outage_logs_error_again(self, clients, block, caplog):
        client, _ = self.warm(clients, block)
        with caplog.at_level("ERROR", logger="inference"):
            drop(client)
            block.run(**run_kwargs())
            reconnect(client)
            client.deliver(PAUSED)
            block.run(**run_kwargs())
            drop(client)
            block.run(**run_kwargs())

        errors = [r for r in caplog.records if r.levelname == "ERROR"]
        assert len(errors) == 2

    def test_reconnected_with_nothing_new_repeats_without_error_and_no_recovery_line(
        self, clients, block, caplog
    ):
        client, first = self.warm(clients, block)
        drop(client)
        block.run(**run_kwargs())
        reconnect(client)

        with caplog.at_level("INFO", logger="inference"):
            result = block.run(**run_kwargs())

        assert result == {**first, "is_new": False}
        assert "restored after" not in caplog.text

    def test_connected_but_not_resubscribed_repeats_with_resubscribe_message(
        self, clients, block
    ):
        client, first = self.warm(clients, block)
        drop(client)
        client.ack_subscriptions = False
        reconnect(client)

        result = block.run(**run_kwargs())

        assert result == {
            **first,
            "is_new": False,
            "error_status": True,
            "error_message": NOT_RESUBSCRIBED_REPEATING_LAST,
        }

    def test_broker_unavailable_on_reconnect_repeats_with_its_reason(
        self, clients, block
    ):
        client, first = self.warm(clients, block)
        drop(client)
        reconnect(client, 3)

        result = block.run(**run_kwargs())

        assert result["value"] == first["value"]
        assert result["error_status"] is True
        assert "broker unavailable" in result["error_message"]
        assert "retrying in the background" in result["error_message"]
        assert client.disconnected is False

    def test_refused_reconnect_drains_then_repeats_with_refusal(
        self, clients, block, caplog
    ):
        client, _ = self.warm(clients, block, read_mode="sequential")
        client.deliver(b"A")
        drop(client)
        reconnect(client, 5)

        with caplog.at_level("ERROR", logger="inference"):
            results = [
                block.run(**run_kwargs(read_mode="sequential")) for _ in range(3)
            ]

        assert [r["value"] for r in results] == ["A", "A", "A"]
        assert [r["error_status"] for r in results] == [False, True, True]
        assert "not authorised" in results[1]["error_message"]
        assert "does not retry" in results[1]["error_message"]
        assert results[2] == results[1]
        assert client.disconnected is True
        run_path_errors = [
            r
            for r in caplog.records
            if r.levelname == "ERROR" and r.getMessage().startswith("MQTT Reader ")
        ]
        assert len(run_path_errors) == 1
        assert "restored after" not in caplog.text

    def test_refused_resubscription_drains_then_returns_empty_failure(
        self, clients, block
    ):
        client, _ = self.warm(clients, block, read_mode="sequential")
        client.deliver(b"A")
        drop(client)
        client.granted_qos = SUBSCRIPTION_REFUSED_QOS
        reconnect(client)

        results = [block.run(**run_kwargs(read_mode="sequential")) for _ in range(2)]

        assert results[0]["value"] == "A"
        assert results[0]["error_status"] is False
        assert results[1]["payload"] == {}
        assert results[1]["error_status"] is True
        assert "refused subscription" in results[1]["error_message"]

    def test_cold_refused_subscription_drains_the_backlog_first(
        self, clients, block, monkeypatch
    ):
        # a persistent session's backlog lands right after CONNACK, before the
        # SUBACK that refuses the subscription: the messages are still real
        monkeypatch.setattr(FakeClient, "granted_qos", SUBSCRIPTION_REFUSED_QOS)
        original_loop_start = FakeClient.loop_start

        def loop_start_with_backlog(client):
            original_loop_start(client)
            client.deliver(b"A")

        with patch.object(FakeClient, "loop_start", loop_start_with_backlog):
            first = block.run(**run_kwargs(read_mode="sequential"))
        second = block.run(**run_kwargs(read_mode="sequential"))

        assert first["value"] == "A"
        assert first["error_status"] is False
        assert block._subscribed_once is False
        assert second["payload"] == {}
        assert "refused subscription" in second["error_message"]

    def test_cold_instance_still_waits_for_the_broker(
        self, clients, block, monkeypatch
    ):
        monkeypatch.setattr(FakeClient, "fire_on_connect", False)

        started = time.monotonic()
        result = block.run(**run_kwargs(timeout=0.3))

        # a hair of timer slack is allowed on the lower bound
        assert time.monotonic() - started >= 0.25
        assert result["error_message"] == NOT_CONNECTED_WITHIN_TIMEOUT
        assert block._subscribed_once is False


class TestBrokerPolicy:
    def test_allowlisted_broker_is_used(self, clients, block, monkeypatch):
        monkeypatch.setattr(
            mqtt_common, "MQTT_WORKFLOWS_BLOCKS_WHITELISTED_HOSTS", ["LocalHost"]
        )

        result = block.run(**run_kwargs())

        assert result["error_status"] is False
        assert clients[0].connected_to == ("localhost", 1883)

    def test_unlisted_broker_rejected_before_client_construction(
        self, clients, block, monkeypatch
    ):
        monkeypatch.setattr(
            mqtt_common, "MQTT_WORKFLOWS_BLOCKS_WHITELISTED_HOSTS", ["broker.internal"]
        )

        result = block.run(**run_kwargs())

        assert result["error_status"] is True
        assert "not permitted" in result["error_message"]
        assert "broker.internal" not in result["error_message"]
        assert clients == []

    def test_port_mismatch_rejected(self, clients, block, monkeypatch):
        monkeypatch.setattr(
            mqtt_common, "MQTT_WORKFLOWS_BLOCKS_WHITELISTED_HOSTS", ["localhost:8883"]
        )

        result = block.run(**run_kwargs(port=1883))

        assert result["error_status"] is True
        assert clients == []

    def test_operator_broker_replaces_workflow_host_when_user_host_not_allowed(
        self, clients, block, monkeypatch
    ):
        monkeypatch.setattr(
            mqtt_common, "MQTT_WORKFLOWS_BLOCKS_ALLOW_USER_PROVIDED_HOST", False
        )
        monkeypatch.setattr(
            mqtt_common, "MQTT_WORKFLOWS_BLOCKS_WHITELISTED_HOSTS", ["operator:8883"]
        )

        first = block.run(**run_kwargs(host="workflow-a"))
        # identity is the resolved address: a different workflow host is not a change
        second = block.run(**run_kwargs(host="workflow-b"))

        assert first["error_status"] is False
        assert second["error_status"] is False
        assert len(clients) == 1
        assert clients[0].connected_to == ("operator", 8883)

    def test_user_host_not_allowed_without_operator_broker_disables_block(
        self, clients, block, monkeypatch
    ):
        monkeypatch.setattr(
            mqtt_common, "MQTT_WORKFLOWS_BLOCKS_ALLOW_USER_PROVIDED_HOST", False
        )
        monkeypatch.setattr(
            mqtt_common, "MQTT_WORKFLOWS_BLOCKS_WHITELISTED_HOSTS", None
        )

        result = block.run(**run_kwargs())

        assert result["error_status"] is True
        assert "disabled" in result["error_message"]
        assert clients == []


@pytest.fixture
def ca_file(tmp_path):
    # a real regular file: configure_tls() refuses anything else before paho
    # (faked in these tests) would read it
    path = tmp_path / "ca.pem"
    path.write_text("content is irrelevant: paho is faked in these tests")
    return path


class TestTLS:
    def test_manifest_defaults_and_dropdown_values(self):
        manifest = BlockManifest.model_validate(manifest_payload())

        assert manifest.encryption == "none"
        assert manifest.ca_certificate_path is None
        assert (
            BlockManifest.model_validate(manifest_payload(encryption="tls")).encryption
            == "tls"
        )

    @pytest.mark.parametrize("encryption", ["ssl", "TLS", True, None, ""])
    def test_manifest_rejects_other_encryption_values(self, encryption):
        with pytest.raises(ValidationError):
            BlockManifest.model_validate(manifest_payload(encryption=encryption))

    def test_ca_certificate_path_is_shown_only_for_tls(self):
        schema = BlockManifest.model_json_schema()["properties"]

        assert schema["ca_certificate_path"]["relevant_for"] == {
            "encryption": {"values": ["tls"], "required": True}
        }

    @pytest.mark.parametrize("ca_certificate_path", [None, "", "/ca.pem"])
    def test_default_encryption_never_configures_tls(
        self, clients, block, ca_certificate_path
    ):
        result = block.run(**run_kwargs(ca_certificate_path=ca_certificate_path))

        assert result["error_status"] is False
        assert [name for name, *_ in clients[0].calls] == ["connect"]

    def test_tls_without_ca_path_uses_system_store_before_connect(self, clients, block):
        result = block.run(**run_kwargs(encryption="tls"))

        assert result["error_status"] is False
        assert clients[0].calls == [
            ("tls_set", None),
            ("connect", "localhost", 1883),
        ]

    def test_tls_with_ca_path_and_file_system_access(self, clients, ca_file):
        block = MQTTReaderBlockV1(allow_access_to_file_system=True)

        result = block.run(
            **run_kwargs(encryption="tls", ca_certificate_path=str(ca_file))
        )

        assert result["error_status"] is False
        assert clients[0].calls == [
            ("tls_set", str(ca_file)),
            ("connect", "localhost", 1883),
        ]

    def test_tls_with_ca_path_refused_without_file_system_access(self, clients, block):
        result = block.run(
            **run_kwargs(encryption="tls", ca_certificate_path="/etc/passwd")
        )

        assert result["error_status"] is True
        assert "ca_certificate_path" in result["error_message"]
        assert (
            "ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE" in result["error_message"]
        )
        assert block._client is None
        assert clients[0].calls == []
        assert clients[0].loop_started is False

    def test_unloadable_ca_bundle_reported_and_nothing_kept(
        self, clients, monkeypatch, ca_file
    ):
        def failing_tls_set(client, ca_certs=None, **kwargs):
            raise FileNotFoundError("no such file")

        monkeypatch.setattr(FakeClient, "tls_set", failing_tls_set)
        block = MQTTReaderBlockV1(allow_access_to_file_system=True)

        result = block.run(
            **run_kwargs(encryption="tls", ca_certificate_path=str(ca_file))
        )

        assert result["error_status"] is True
        assert "could not load CA bundle" in result["error_message"]
        assert str(ca_file) in result["error_message"]
        assert block._client is None
        assert clients[0].loop_started is False

    def test_next_run_after_tls_failure_retries_with_fresh_client(self, clients, block):
        block.run(**run_kwargs(encryption="tls", ca_certificate_path="/etc/passwd"))

        result = block.run(**run_kwargs(encryption="tls"))

        assert result["error_status"] is False
        assert len(clients) == 2

    @pytest.mark.parametrize(
        "change",
        [{"encryption": "tls"}, {"ca_certificate_path": "other.pem"}],
    )
    def test_changed_tls_parameters_rejected(self, clients, change):
        block = MQTTReaderBlockV1(allow_access_to_file_system=True)
        block.run(**run_kwargs())

        result = block.run(**run_kwargs(**change))

        assert result["error_status"] is True
        assert "changed between runs" in result["error_message"]
        assert len(clients) == 1

    def test_invalid_encryption_at_run_time_rejected(self, clients, block):
        result = block.run(**run_kwargs(encryption="ssl"))

        assert result["error_status"] is True
        assert "encryption" in result["error_message"]
        assert clients == []

    def test_non_string_ca_path_rejected(self, clients, block):
        result = block.run(**run_kwargs(encryption="tls", ca_certificate_path=5))

        assert result["error_status"] is True
        assert "ca_certificate_path" in result["error_message"]
        assert clients == []

    def test_tls_insecure_is_never_called(self, clients, ca_file):
        block = MQTTReaderBlockV1(allow_access_to_file_system=True)

        block.run(**run_kwargs(encryption="tls", ca_certificate_path=str(ca_file)))

        assert all(name != "tls_insecure_set" for name, *_ in clients[0].calls)

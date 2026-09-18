import threading
import time
from types import SimpleNamespace
from typing import List, get_args
from unittest.mock import patch

import pytest
from pydantic import ValidationError
from roboflow_workflows.enterprise_blocks.sinks.mqtt_reader import v1
from roboflow_workflows.enterprise_blocks.sinks.mqtt_reader.v1 import (
    LATEST_BUFFER_SIZE,
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

CLIENT_CLASS_PATH = (
    "roboflow_workflows.enterprise_blocks.sinks.mqtt_reader.v1.mqtt.Client"
)


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

    def __init__(self, userdata=None):
        self.userdata = userdata
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

    def connect(self, host, port):
        if self.connect_error is not None:
            raise self.connect_error
        self.connected_to = (host, port)

    def loop_start(self):
        self.loop_started = True
        if self.fire_on_connect:
            self.on_connect(self, self.userdata, {}, self.connack_reason_code)

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

    def factory(userdata=None):
        client = FakeClient(userdata=userdata)
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

    def test_block_declares_no_init_parameters(self):
        assert MQTTReaderBlockV1.get_init_parameters() == []


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

    def test_rejected_connack_reported_as_not_connected(
        self, clients, block, monkeypatch
    ):
        monkeypatch.setattr(FakeClient, "connack_reason_code", 5)
        result = block.run(**run_kwargs())

        assert result["error_status"] is True
        assert clients[0].subscriptions == []

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

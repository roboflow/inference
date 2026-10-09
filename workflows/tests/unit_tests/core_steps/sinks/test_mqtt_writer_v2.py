import socket
import threading
import time
from types import SimpleNamespace
from typing import get_args
from unittest.mock import MagicMock, patch

import paho.mqtt.client as mqtt
import pytest
from pydantic import ValidationError
from roboflow_workflows.enterprise_blocks.sinks import mqtt_common
from roboflow_workflows.enterprise_blocks.sinks.mqtt_common import (
    MQTT_KEEPALIVE_SECONDS,
)
from roboflow_workflows.enterprise_blocks.sinks.mqtt_writer import v2
from roboflow_workflows.enterprise_blocks.sinks.mqtt_writer.v2 import (
    CLOSE_JOIN_ATTEMPTS,
    FIRE_AND_FORGET_QOS0_DROPPED,
    FIRE_AND_FORGET_QUEUED,
    FIRE_AND_FORGET_QUEUED_UNTIL_RECONNECT,
    MAX_QUEUED_MESSAGES,
    NOT_CONNECTED_WITHIN_TIMEOUT,
    QUEUE_FULL,
    BlockManifest,
    MQTTWriterSinkBlockV2,
    MQTTWriterState,
    mqtt_on_connect,
    mqtt_on_connect_fail,
    mqtt_on_disconnect,
)

CLIENT_CLASS_PATH = (
    "roboflow_workflows.enterprise_blocks.sinks.mqtt_writer.v2.mqtt.Client"
)


def run_kwargs(**overrides) -> dict:
    kwargs = {
        "host": "localhost",
        "port": 1883,
        "topic": "test/topic",
        "message": "Hello, MQTT!",
        "timeout": 0.01,
    }
    kwargs.update(overrides)
    return kwargs


@pytest.fixture
def mock_client_cls():
    with patch(CLIENT_CLASS_PATH) as client_cls:
        publish_result = client_cls.return_value.publish.return_value
        publish_result.rc = mqtt.MQTT_ERR_SUCCESS
        publish_result.is_published.return_value = True
        yield client_cls


@pytest.fixture
def block() -> MQTTWriterSinkBlockV2:
    return MQTTWriterSinkBlockV2()


class TestManifest:
    def test_primary_identifier_is_namespaced(self):
        identifiers = get_args(BlockManifest.model_fields["type"].annotation)

        assert identifiers[0] == "roboflow_enterprise/mqtt_writer_sink@v2"

    @pytest.mark.parametrize(
        "identifier",
        ["roboflow_enterprise/mqtt_writer_sink@v1", "mqtt_writer_sink@v1"],
    )
    def test_v1_identifiers_belong_to_v1(self, identifier):
        with pytest.raises(ValidationError):
            BlockManifest.model_validate(
                {
                    "type": identifier,
                    "name": "mqtt",
                    "host": "localhost",
                    "port": 1883,
                    "topic": "test/topic",
                    "message": "Hello, MQTT!",
                }
            )

    @pytest.mark.parametrize(
        "timeout", [0, -1, float("nan"), float("inf"), float("-inf")]
    )
    def test_literal_timeout_must_be_finite_and_positive(self, timeout):
        with pytest.raises(ValidationError):
            BlockManifest.model_validate(
                {
                    "type": "roboflow_enterprise/mqtt_writer_sink@v2",
                    "name": "mqtt",
                    "host": "localhost",
                    "port": 1883,
                    "topic": "test/topic",
                    "message": "Hello, MQTT!",
                    "timeout": timeout,
                }
            )

    @pytest.mark.parametrize("timeout", [0.5, "$inputs.timeout"])
    def test_valid_timeouts_accepted(self, timeout):
        manifest = BlockManifest.model_validate(
            {
                "type": "roboflow_enterprise/mqtt_writer_sink@v2",
                "name": "mqtt",
                "host": "localhost",
                "port": 1883,
                "topic": "test/topic",
                "message": "Hello, MQTT!",
                "timeout": timeout,
            }
        )

        assert manifest.timeout == timeout

    @pytest.mark.parametrize("port", [0, -1, 65536])
    def test_literal_port_must_be_within_tcp_range(self, port):
        with pytest.raises(ValidationError):
            BlockManifest.model_validate(
                {
                    "type": "roboflow_enterprise/mqtt_writer_sink@v2",
                    "name": "mqtt",
                    "host": "localhost",
                    "port": port,
                    "topic": "test/topic",
                    "message": "Hello, MQTT!",
                }
            )

    @pytest.mark.parametrize("port", [1883, "$inputs.port"])
    def test_valid_ports_accepted(self, port):
        manifest = BlockManifest.model_validate(
            {
                "type": "roboflow_enterprise/mqtt_writer_sink@v2",
                "name": "mqtt",
                "host": "localhost",
                "port": port,
                "topic": "test/topic",
                "message": "Hello, MQTT!",
            }
        )

        assert manifest.port == port


class TestCallbacks:
    def test_on_connect_sets_event_only_for_accepted_connack(self):
        state = MQTTWriterState()

        mqtt_on_connect(MagicMock(), state, {}, 0)

        assert state.connected.is_set()
        assert state.connack.is_set()
        assert state.refused_code is None

    @pytest.mark.parametrize("reason_code", [1, 2, 3, 4, 5])
    def test_on_connect_clears_event_for_rejected_connack(self, reason_code):
        state = MQTTWriterState()
        state.connected.set()

        mqtt_on_connect(MagicMock(), state, {}, reason_code)

        assert not state.connected.is_set()
        assert state.connack.is_set()
        assert state.refused_code == reason_code

    @pytest.mark.parametrize("reason_code", [1, 2, 4, 5])
    def test_on_connect_stops_the_loop_for_a_permanent_refusal(self, reason_code):
        client = MagicMock()

        mqtt_on_connect(client, MQTTWriterState(), {}, reason_code)

        client.disconnect.assert_called_once_with()

    def test_on_connect_keeps_the_loop_for_broker_unavailable(self):
        client = MagicMock()
        state = MQTTWriterState()

        mqtt_on_connect(client, state, {}, 3)

        client.disconnect.assert_not_called()
        assert state.refused_code == 3

    @pytest.mark.parametrize("reason_code", [0, 3, 5])
    def test_on_connect_completes_the_state_before_waking_a_waiter(self, reason_code):
        # a run wakes on `connack`; what it then reads must already be final
        state = MQTTWriterState()
        seen = {}

        class RecordingEvent(threading.Event):
            def set(self):
                seen["refused_code"] = state.refused_code
                seen["connected"] = state.connected.is_set()
                super().set()

        state.connack = RecordingEvent()

        mqtt_on_connect(MagicMock(), state, {}, reason_code)

        assert seen == {
            "refused_code": reason_code or None,
            "connected": reason_code == 0,
        }

    def test_on_connect_logs_the_refusal_reason(self, caplog):
        with caplog.at_level("ERROR", logger="inference"):
            mqtt_on_connect(MagicMock(), MQTTWriterState(), {}, 4)

        assert "bad user name or password" in caplog.text
        assert "code 4" in caplog.text

    def test_accepted_connack_after_broker_unavailable_resets_refusal(self):
        state = MQTTWriterState()

        mqtt_on_connect(MagicMock(), state, {}, 3)
        mqtt_on_connect(MagicMock(), state, {}, 0)

        assert state.refused_code is None
        assert state.connected.is_set()

    def test_on_connect_fail_matches_paho_two_argument_signature(self):
        state = MQTTWriterState()
        state.connected.set()

        # paho 1.6.1 invokes on_connect_fail with exactly (client, userdata)
        mqtt_on_connect_fail(MagicMock(), state)

        assert not state.connected.is_set()

    def test_on_connect_fail_logs_an_outage_once(self):
        # the background loop retries about once per reconnect delay
        state = MQTTWriterState()

        with patch.object(v2, "logger") as mock_logger:
            for _ in range(5):
                mqtt_on_connect_fail(MagicMock(), state)
            assert mock_logger.error.call_count == 1

            mqtt_on_connect(MagicMock(), state, {}, 0)
            mqtt_on_connect_fail(MagicMock(), state)

        assert mock_logger.error.call_count == 2

    def test_repeated_broker_unavailable_logged_once_per_outage(self):
        # code 3 keeps paho reconnecting, so the refusal repeats every retry
        state = MQTTWriterState()

        with patch.object(v2, "logger") as mock_logger:
            for _ in range(5):
                mqtt_on_connect(MagicMock(), state, {}, 3)
            assert mock_logger.error.call_count == 1

            mqtt_on_connect(MagicMock(), state, {}, 0)
            mqtt_on_connect(MagicMock(), state, {}, 3)

        assert mock_logger.error.call_count == 2

    def test_refused_retries_do_not_log_a_disconnect_each_time(self):
        # paho ends each refused CONNACK with on_disconnect before retrying
        state = MQTTWriterState()

        with patch.object(v2, "logger") as mock_logger:
            for _ in range(5):
                mqtt_on_connect(MagicMock(), state, {}, 3)
                mqtt_on_disconnect(MagicMock(), state, 5)

        assert mock_logger.error.call_count == 1
        mock_logger.info.assert_not_called()

    def test_losing_an_established_session_logs_info(self):
        state = MQTTWriterState()
        mqtt_on_connect(MagicMock(), state, {}, 0)

        with patch.object(v2, "logger") as mock_logger:
            mqtt_on_disconnect(MagicMock(), state, 7)

        mock_logger.info.assert_called_once()

    def test_permanent_refusal_logged_after_transient_ones(self):
        state = MQTTWriterState()
        mqtt_on_connect(MagicMock(), state, {}, 3)

        with patch.object(v2, "logger") as mock_logger:
            mqtt_on_connect(MagicMock(), state, {}, 5)

        mock_logger.error.assert_called_once()

    def test_on_disconnect_clears_event(self):
        state = MQTTWriterState()
        state.connected.set()

        mqtt_on_disconnect(MagicMock(), state, 1)

        assert not state.connected.is_set()

    def test_on_disconnect_after_transport_drop_clears_connack(self):
        state = MQTTWriterState()
        state.connack.set()

        mqtt_on_disconnect(MagicMock(), state, 1)

        assert not state.connack.is_set()

    def test_on_disconnect_after_refusal_keeps_connack(self):
        state = MQTTWriterState()
        mqtt_on_connect(MagicMock(), state, {}, 5)

        mqtt_on_disconnect(MagicMock(), state, 0)

        assert state.connack.is_set()
        assert state.refused_code == 5

    def test_on_disconnect_accepts_mqtt5_properties_argument(self):
        state = MQTTWriterState()
        state.connected.set()

        mqtt_on_disconnect(MagicMock(), state, 1, properties=None)

        assert not state.connected.is_set()

    def test_connect_fail_after_transient_refusal_forgets_the_refusal(self):
        state = MQTTWriterState()
        mqtt_on_connect(MagicMock(), state, {}, 3)

        mqtt_on_connect_fail(MagicMock(), state)

        assert state.refused_code is None
        assert not state.connack.is_set()

    def test_connect_fail_after_permanent_refusal_keeps_the_refusal(self):
        state = MQTTWriterState()
        mqtt_on_connect(MagicMock(), state, {}, 5)

        mqtt_on_connect_fail(MagicMock(), state)

        assert state.refused_code == 5
        assert state.connack.is_set()

    def test_reset_clears_everything(self):
        state = MQTTWriterState()
        mqtt_on_connect(MagicMock(), state, {}, 5)
        mqtt_on_connect_fail(MagicMock(), state)
        state.closing.set()

        state.reset()

        assert not state.connected.is_set()
        assert not state.connack.is_set()
        assert state.refused_code is None
        assert state.connect_failing is False
        assert not state.closing.is_set()

    @pytest.mark.parametrize("reason_code", [0, 3, 5])
    def test_on_connect_while_closing_disconnects_without_marking_connected(
        self, reason_code
    ):
        client = MagicMock()
        state = MQTTWriterState()
        state.closing.set()

        mqtt_on_connect(client, state, {}, reason_code)

        assert not state.connected.is_set()
        assert not state.connack.is_set()
        assert state.refused_code is None
        client.disconnect.assert_called_once_with()

    @pytest.mark.timeout(10)
    def test_callbacks_complete_while_the_lifecycle_lock_is_held(self, block):
        client = MagicMock()
        state = block._connection
        state.closing.set()

        with block._lifecycle_lock:
            mqtt_on_connect(client, state, {}, 0)
            mqtt_on_connect_fail(client, state)
            mqtt_on_disconnect(client, state, 0)

        client.disconnect.assert_called_once_with()


def _answer_connect_with(mock_client_cls, reason_code: int) -> None:
    """Make the mocked client's loop_start deliver a CONNACK with `reason_code`
    to the state the block handed paho as userdata."""
    mock_client = mock_client_cls.return_value

    def fire_connack():
        state = mock_client_cls.call_args.kwargs["userdata"]
        mqtt_on_connect(mock_client, state, {}, reason_code)

    mock_client.loop_start.side_effect = fire_connack


class TestRefusedConnection:
    def test_refusal_reported_with_reason_and_inputs(self, mock_client_cls, block):
        _answer_connect_with(mock_client_cls, 5)

        result = block.run(**run_kwargs())

        assert result["error_status"] is True
        assert "not authorised" in result["message"]
        assert "code 5" in result["message"]
        assert "Check username and password" in result["message"]
        assert "Raise 'timeout'" not in result["message"]
        mock_client_cls.return_value.disconnect.assert_called_once_with()
        mock_client_cls.return_value.publish.assert_not_called()

    @pytest.mark.parametrize(
        "reason_code, reason",
        [
            (1, "unacceptable protocol version"),
            (2, "identifier rejected"),
            (4, "bad user name or password"),
        ],
    )
    def test_other_permanent_refusals_name_their_reason(
        self, mock_client_cls, block, reason_code, reason
    ):
        _answer_connect_with(mock_client_cls, reason_code)

        result = block.run(**run_kwargs())

        assert reason in result["message"]
        mock_client_cls.return_value.disconnect.assert_called_once_with()

    def test_refusal_is_reported_without_waiting_for_timeout(
        self, mock_client_cls, block
    ):
        _answer_connect_with(mock_client_cls, 5)

        started = time.monotonic()
        result = block.run(**run_kwargs(timeout=2.0))

        assert result["error_status"] is True
        assert time.monotonic() - started < 0.5

    def test_next_run_after_refusal_repeats_it_without_a_new_connection(
        self, mock_client_cls, block
    ):
        _answer_connect_with(mock_client_cls, 5)
        first = block.run(**run_kwargs(timeout=2.0))

        started = time.monotonic()
        second = block.run(**run_kwargs(timeout=2.0))

        assert second == first
        assert time.monotonic() - started < 0.5
        assert mock_client_cls.call_count == 1
        mock_client_cls.return_value.connect.assert_called_once()

    def test_refusal_with_fail_fast_raises_with_the_reason(
        self, mock_client_cls, block
    ):
        _answer_connect_with(mock_client_cls, 5)

        with pytest.raises(Exception, match="not authorised"):
            block.run(**run_kwargs(fail_fast=True))

    def test_fire_and_forget_logs_a_permanent_refusal_once(
        self, mock_client_cls, block
    ):
        _answer_connect_with(mock_client_cls, 5)

        with patch.object(v2, "logger") as mock_logger:
            results = [block.run(**run_kwargs(fire_and_forget=True)) for _ in range(3)]

        assert all(result["error_status"] is True for result in results)
        assert all("not authorised" in result["message"] for result in results)
        failure_logs = [
            call
            for call in mock_logger.error.call_args_list
            if call.args[0] == "MQTT Writer failure: %s"
        ]
        assert len(failure_logs) == 1

    def test_fire_and_forget_permanent_refusal_still_raises_with_fail_fast(
        self, mock_client_cls, block
    ):
        _answer_connect_with(mock_client_cls, 5)
        block.run(**run_kwargs(fire_and_forget=True))

        with pytest.raises(RuntimeError, match="not authorised"):
            block.run(**run_kwargs(fire_and_forget=True, fail_fast=True))

    def test_broker_unavailable_keeps_retrying_and_names_the_reason(
        self, mock_client_cls, block
    ):
        _answer_connect_with(mock_client_cls, 3)

        result = block.run(**run_kwargs())

        assert result["error_status"] is True
        assert "broker unavailable" in result["message"]
        assert "retrying in the background" in result["message"]
        mock_client_cls.return_value.disconnect.assert_not_called()

    def test_run_succeeds_once_an_unavailable_broker_accepts(
        self, mock_client_cls, block
    ):
        _answer_connect_with(mock_client_cls, 3)
        block.run(**run_kwargs())
        state = mock_client_cls.call_args.kwargs["userdata"]
        mqtt_on_connect(mock_client_cls.return_value, state, {}, 0)

        result = block.run(**run_kwargs())

        assert result["error_status"] is False
        assert state.refused_code is None

    def test_no_connack_at_all_reports_not_connected(self, mock_client_cls, block):
        result = block.run(**run_kwargs())

        assert result["error_status"] is True
        assert result["message"] == NOT_CONNECTED_WITHIN_TIMEOUT

    def test_close_after_refusal_completes_and_resets_state(
        self, mock_client_cls, block
    ):
        _answer_connect_with(mock_client_cls, 5)
        block.run(**run_kwargs())

        block.close()

        assert block.mqtt_client is None
        assert block._connection.refused_code is None
        assert not block._connection.connack.is_set()
        mock_client_cls.return_value.loop_stop.assert_called_once()


class TestRunValidation:
    @pytest.mark.parametrize(
        "timeout",
        [0, -1, float("nan"), float("inf"), float("-inf"), 10**400, 1e308],
    )
    def test_invalid_timeout_rejected_before_client_construction(
        self, mock_client_cls, block, timeout
    ):
        result = block.run(**run_kwargs(timeout=timeout))

        assert result["error_status"] is True
        assert "timeout" in result["message"].lower()
        mock_client_cls.assert_not_called()

    @pytest.mark.parametrize("timeout", [True, False])
    def test_boolean_timeout_rejected_before_client_construction(
        self, mock_client_cls, block, timeout
    ):
        result = block.run(**run_kwargs(timeout=timeout))

        assert result["error_status"] is True
        assert "Invalid timeout" in result["message"]
        mock_client_cls.assert_not_called()

    @pytest.mark.parametrize(
        "port", [0, -1, 65536, "abc", "1883.5", 1883.7, True, None]
    )
    def test_invalid_port_rejected_before_client_construction(
        self, mock_client_cls, block, port
    ):
        result = block.run(**run_kwargs(port=port))

        assert result["error_status"] is True
        assert "port" in result["message"].lower()
        mock_client_cls.assert_not_called()

    @pytest.mark.parametrize("port", ["1883", 1883.0, "1883.0", " 1883 "])
    def test_selector_supplied_port_coerced_like_manifest(
        self, mock_client_cls, block, port
    ):
        # runtime validation coerces on a discarded manifest copy, so run()
        # receives the raw selector value and must coerce it itself
        block._connected.set()

        result = block.run(**run_kwargs(port=port))

        assert result["error_status"] is False
        mock_client_cls.return_value.connect.assert_called_once_with(
            "localhost", 1883, keepalive=MQTT_KEEPALIVE_SECONDS
        )

    @pytest.mark.parametrize("qos", [-1, 3, "abc", 1.5, "1.5", True, False, None])
    def test_invalid_qos_rejected_before_client_construction(
        self, mock_client_cls, block, qos
    ):
        result = block.run(**run_kwargs(qos=qos))

        assert result["error_status"] is True
        assert "qos" in result["message"].lower()
        mock_client_cls.assert_not_called()

    @pytest.mark.parametrize("qos, expected", [("1", 1), (2.0, 2), (0, 0), ("1.0", 1)])
    def test_selector_supplied_qos_coerced_like_other_parameters(
        self, mock_client_cls, block, qos, expected
    ):
        block._connected.set()

        result = block.run(**run_kwargs(qos=qos))

        assert result["error_status"] is False
        mock_client_cls.return_value.publish.assert_called_once_with(
            "test/topic", "Hello, MQTT!", qos=expected, retain=False
        )

    def test_password_without_username_rejected(self, mock_client_cls, block):
        result = block.run(**run_kwargs(password="secret"))

        assert result["error_status"] is True
        assert "username" in result["message"].lower()
        mock_client_cls.assert_not_called()

    def test_username_without_password_configures_authentication(
        self, mock_client_cls, block
    ):
        block._connected.set()

        block.run(**run_kwargs(username="lenny"))

        mock_client_cls.return_value.username_pw_set.assert_called_once_with(
            "lenny", None
        )

    def test_username_with_empty_password_configures_authentication(
        self, mock_client_cls, block
    ):
        block._connected.set()

        block.run(**run_kwargs(username="lenny", password=""))

        mock_client_cls.return_value.username_pw_set.assert_called_once_with(
            "lenny", ""
        )


class TestClientSetup:
    def test_client_connects_synchronously_before_background_loop(
        self, mock_client_cls, block
    ):
        block._connected.set()
        mock_client = mock_client_cls.return_value

        result = block.run(**run_kwargs())

        assert result["error_status"] is False
        mock_client_cls.assert_called_once_with(userdata=block._connection)
        mock_client.connect.assert_called_once_with(
            "localhost", 1883, keepalive=MQTT_KEEPALIVE_SECONDS
        )
        mock_client.connect_async.assert_not_called()
        called_methods = [call[0] for call in mock_client.method_calls]
        assert called_methods.index("connect") < called_methods.index("loop_start")

    def test_client_connects_with_the_short_keepalive(self, mock_client_cls, block):
        block._connected.set()

        block.run(**run_kwargs())

        mock_client_cls.return_value.connect.assert_called_once_with(
            "localhost", 1883, keepalive=MQTT_KEEPALIVE_SECONDS
        )

    def test_timeout_hint_only_for_a_connect_timeout(self, mock_client_cls, block):
        mock_client = mock_client_cls.return_value
        mock_client.connect.side_effect = TimeoutError("timed out")

        result = block.run(**run_kwargs())

        assert "Raise 'timeout'" in result["message"]

    def test_no_timeout_hint_when_the_broker_refuses_the_connection(
        self, mock_client_cls, block
    ):
        mock_client = mock_client_cls.return_value
        mock_client.connect.side_effect = ConnectionRefusedError("refused")

        result = block.run(**run_kwargs())

        assert "not connected" in result["message"]
        assert "Raise 'timeout'" not in result["message"]

    def test_readiness_wait_shares_one_deadline(self, mock_client_cls, block):
        clock = [100.0]
        waits = []

        class FakeEvent:
            def __init__(self, spent, result):
                self.spent = spent
                self.result = result

            def is_set(self):
                return False

            def wait(self, timeout=None):
                waits.append(timeout)
                clock[0] += self.spent
                return self.result

            def clear(self):
                pass

        block._connection.connack = FakeEvent(spent=0.15, result=True)
        block._connection.connected = FakeEvent(spent=0.05, result=False)
        fake_time = SimpleNamespace(monotonic=lambda: clock[0])

        with patch.object(v2, "time", fake_time):
            result = block.run(**run_kwargs(timeout=0.2))
        with patch.object(v2, "logger") as mock_logger:
            block.close()

        assert result["error_status"] is True
        assert result["message"] == NOT_CONNECTED_WITHIN_TIMEOUT
        assert waits == [pytest.approx(0.2), pytest.approx(0.05)]
        mock_logger.error.assert_not_called()

    def test_client_registers_static_callbacks(self, mock_client_cls, block):
        block._connected.set()
        mock_client = mock_client_cls.return_value

        block.run(**run_kwargs())

        assert mock_client.on_connect is mqtt_on_connect
        assert mock_client.on_connect_fail is mqtt_on_connect_fail
        assert mock_client.on_disconnect is mqtt_on_disconnect
        for callback in (
            mock_client.on_connect,
            mock_client.on_connect_fail,
            mock_client.on_disconnect,
        ):
            assert getattr(callback, "__self__", None) is not block

    def test_reconnect_delay_lets_first_retry_finish_within_one_run(
        self, mock_client_cls, block
    ):
        # min_delay must be below the readiness wait (= timeout), otherwise
        # the first reconnect attempt after a connection drop starts exactly
        # as the waiting run's timeout expires and that run always fails
        block._connected.set()

        block.run(**run_kwargs(timeout=0.5))

        mock_client_cls.return_value.reconnect_delay_set.assert_called_once_with(
            min_delay=0.25, max_delay=1.0
        )

    def test_reconnect_delay_follows_a_long_timeout(self, mock_client_cls, block):
        block._connected.set()

        block.run(**run_kwargs(timeout=4.0))

        mock_client_cls.return_value.reconnect_delay_set.assert_called_once_with(
            min_delay=2.0, max_delay=8.0
        )

    def test_tiny_timeout_cannot_make_reconnects_a_busy_loop(
        self, mock_client_cls, block
    ):
        # fire and forget keeps a client retrying after a failed first connect
        mock_client = mock_client_cls.return_value
        mock_client.connect.side_effect = ConnectionRefusedError("refused")

        block.run(**run_kwargs(timeout=1e-6, fire_and_forget=True))

        mock_client.reconnect_delay_set.assert_called_once_with(
            min_delay=0.1, max_delay=1.0
        )

    def test_paho_connect_timeout_bound_to_block_timeout(self, mock_client_cls, block):
        block._connected.set()
        mock_client = mock_client_cls.return_value
        connect_timeout_at_connect_call = []
        mock_client.connect.side_effect = lambda *args, **kwargs: (
            connect_timeout_at_connect_call.append(mock_client._connect_timeout)
        )

        block.run(**run_kwargs(timeout=0.25))

        assert connect_timeout_at_connect_call == [0.25]

    def test_unreachable_broker_on_first_run_leaves_no_background_thread(
        self, mock_client_cls, block
    ):
        mock_client = mock_client_cls.return_value
        mock_client.connect.side_effect = ConnectionRefusedError("refused")

        first_result = block.run(**run_kwargs())

        mock_client.connect.assert_called_once()
        assert first_result["error_status"] is True
        assert "not connected" in first_result["message"].lower()
        # no client kept and no loop started: a failed one-shot run must not
        # leave a reconnecting background thread behind
        assert block.mqtt_client is None
        mock_client.loop_start.assert_not_called()
        mock_client.publish.assert_not_called()

        # broker becomes reachable, the next run retries with a fresh client
        mock_client.connect.side_effect = None
        block._connected.set()

        second_result = block.run(**run_kwargs())

        assert second_result["error_status"] is False
        assert mock_client_cls.call_count == 2
        mock_client.loop_start.assert_called_once()

    def test_unreachable_broker_with_fail_fast_raises(self, mock_client_cls, block):
        mock_client = mock_client_cls.return_value
        mock_client.connect.side_effect = OSError("no route to host")

        with pytest.raises(RuntimeError, match="not connected"):
            block.run(**run_kwargs(fail_fast=True))

        mock_client.connect.assert_called_once()
        mock_client.loop_start.assert_not_called()
        assert block.mqtt_client is None

    def test_setup_failure_resets_client_so_next_run_can_retry(
        self, mock_client_cls, block
    ):
        mock_client = mock_client_cls.return_value
        mock_client.loop_start.side_effect = Exception("boom")

        first_result = block.run(**run_kwargs())

        assert first_result["error_status"] is True
        assert block.mqtt_client is None
        mock_client.loop_stop.assert_called_once()

        mock_client.loop_start.side_effect = None
        block._connected.set()

        second_result = block.run(**run_kwargs())

        assert second_result["error_status"] is False
        assert mock_client_cls.call_count == 2

    def test_first_run_connection_timeout_does_not_poison_client(
        self, mock_client_cls, block
    ):
        mock_client = mock_client_cls.return_value

        first_result = block.run(**run_kwargs())

        assert first_result["error_status"] is True
        assert "not connected" in first_result["message"].lower()
        mock_client.publish.assert_not_called()

        # simulates the background loop establishing the connection later
        block._connected.set()

        second_result = block.run(**run_kwargs())

        assert second_result["error_status"] is False
        mock_client_cls.assert_called_once()
        mock_client.connect.assert_called_once()
        mock_client.publish.assert_called_once()


class TestConnectionOwnership:
    def test_disconnected_run_never_calls_manual_reconnect(
        self, mock_client_cls, block
    ):
        block._connected.set()
        mock_client = mock_client_cls.return_value
        block.run(**run_kwargs())

        block._connected.clear()
        result = block.run(**run_kwargs())

        assert result["error_status"] is True
        assert "not connected" in result["message"].lower()
        mock_client.reconnect.assert_not_called()
        mock_client.publish.assert_called_once()

    def test_changed_connection_parameters_rejected(self, mock_client_cls, block):
        block._connected.set()
        mock_client = mock_client_cls.return_value
        block.run(**run_kwargs(host="broker-a"))

        result = block.run(**run_kwargs(host="broker-b"))

        assert result["error_status"] is True
        assert "parameters" in result["message"].lower()
        mock_client.connect.assert_called_once_with(
            "broker-a", 1883, keepalive=MQTT_KEEPALIVE_SECONDS
        )
        mock_client.publish.assert_called_once()

    def test_changed_credentials_rejected(self, mock_client_cls, block):
        block._connected.set()
        mock_client = mock_client_cls.return_value
        block.run(**run_kwargs(username="lenny", password="old"))

        result = block.run(**run_kwargs(username="lenny", password="new"))

        assert result["error_status"] is True
        mock_client.publish.assert_called_once()

    def test_changed_timeout_rejected(self, mock_client_cls, block):
        # the first timeout permanently configures the reconnect schedule
        block._connected.set()
        mock_client = mock_client_cls.return_value
        block.run(**run_kwargs(timeout=0.5))

        result = block.run(**run_kwargs(timeout=1.0))

        assert result["error_status"] is True
        assert "parameters" in result["message"].lower()
        mock_client.publish.assert_called_once()


class TestPublishing:
    def test_successful_publish(self, mock_client_cls, block):
        block._connected.set()
        mock_client = mock_client_cls.return_value

        result = block.run(**run_kwargs(qos=1, retain=True))

        assert result["error_status"] is False
        assert result["message"] == "Message published successfully"
        mock_client.publish.assert_called_once_with(
            "test/topic", "Hello, MQTT!", qos=1, retain=True
        )

    def test_qos0_publish_confirmation_timeout_reported_as_unconfirmed(
        self, mock_client_cls, block
    ):
        block._connected.set()
        mock_client = mock_client_cls.return_value
        mock_client.publish.return_value.is_published.return_value = False

        with patch.object(v2, "logger") as mock_logger:
            result = block.run(**run_kwargs(qos=0))

        assert result["error_status"] is True
        assert result["message"] == (
            "Publish confirmation timed out; the QoS 0 message was not fully sent "
            "within timeout and may still be sent or may be lost."
        )
        assert "delivery status unknown" not in result["message"].lower()
        assert mock_logger.error.called

    def test_qos1_publish_ack_timeout_reported_as_delivery_unknown(
        self, mock_client_cls, block
    ):
        block._connected.set()
        mock_client = mock_client_cls.return_value
        mock_client.publish.return_value.is_published.return_value = False

        with patch.object(v2, "logger") as mock_logger:
            result = block.run(**run_kwargs(qos=1))

        assert result["error_status"] is True
        assert "delivery status unknown" in result["message"].lower()
        assert mock_logger.error.called

    def test_qos1_publish_with_lost_connection_reported_as_delivery_unknown(
        self, mock_client_cls, block
    ):
        # paho queues QoS 1/2 messages for redelivery when publish() returns
        # MQTT_ERR_NO_CONN - not a final failure
        import paho.mqtt.client as mqtt

        block._connected.set()
        mock_client = mock_client_cls.return_value
        mock_client.publish.return_value.rc = mqtt.MQTT_ERR_NO_CONN

        result = block.run(**run_kwargs(qos=1))

        assert result["error_status"] is True
        assert "delivery status unknown" in result["message"].lower()
        mock_client.publish.return_value.wait_for_publish.assert_not_called()

    def test_qos0_publish_with_lost_connection_reported_as_final_failure(
        self, mock_client_cls, block
    ):
        # paho drops QoS 0 messages on MQTT_ERR_NO_CONN - final failure is honest
        import paho.mqtt.client as mqtt

        block._connected.set()
        mock_client = mock_client_cls.return_value
        publish_result = mock_client.publish.return_value
        publish_result.rc = mqtt.MQTT_ERR_NO_CONN
        publish_result.wait_for_publish.side_effect = RuntimeError(
            "Message publish failed: The client is not currently connected."
        )

        result = block.run(**run_kwargs(qos=0))

        assert result["error_status"] is True
        assert result["message"] == (
            "MQTT broker not connected; QoS 0 messages are not queued, so the "
            "message was dropped."
        )
        publish_result.wait_for_publish.assert_not_called()

    def test_publish_exception_returned_as_error(self, mock_client_cls, block):
        block._connected.set()
        mock_client = mock_client_cls.return_value
        mock_client.publish.side_effect = ValueError("Invalid topic")

        result = block.run(**run_kwargs())

        assert result["error_status"] is True
        assert "Invalid topic" in result["message"]

    def test_no_raw_prints_in_any_code_path(self, mock_client_cls, block):
        with patch("builtins.print") as mock_print:
            block._connected.set()
            block.run(**run_kwargs())
            block._connected.clear()
            block.run(**run_kwargs())

        mock_print.assert_not_called()


class TestFailFast:
    def test_fail_fast_raises_on_connection_failure(self, mock_client_cls, block):
        with pytest.raises(Exception, match="not connected"):
            block.run(**run_kwargs(fail_fast=True))

    def test_fail_fast_raises_on_invalid_timeout(self, mock_client_cls, block):
        with pytest.raises(Exception, match="[Tt]imeout"):
            block.run(**run_kwargs(timeout=-1, fail_fast=True))

    def test_fail_fast_does_not_affect_success(self, mock_client_cls, block):
        block._connected.set()

        result = block.run(**run_kwargs(fail_fast=True))

        assert result["error_status"] is False

    def test_fail_fast_ack_timeout_raises_once_with_clean_message(
        self, mock_client_cls, block
    ):
        block._connected.set()
        mock_client_cls.return_value.publish.return_value.is_published.return_value = (
            False
        )

        with patch.object(v2, "logger") as mock_logger:
            with pytest.raises(RuntimeError, match="^Publish confirmation"):
                block.run(**run_kwargs(fail_fast=True))

        assert mock_logger.error.call_count == 1


class TestQueueLimit:
    def test_client_queue_is_bounded(self, mock_client_cls, block):
        block._connected.set()

        block.run(**run_kwargs())

        mock_client_cls.return_value.max_queued_messages_set.assert_called_once_with(
            MAX_QUEUED_MESSAGES
        )

    @pytest.mark.parametrize("fire_and_forget", [False, True])
    def test_full_queue_reported_as_dropped(
        self, mock_client_cls, block, fire_and_forget
    ):
        block._connected.set()
        publish_result = mock_client_cls.return_value.publish.return_value
        publish_result.rc = mqtt.MQTT_ERR_QUEUE_SIZE

        result = block.run(**run_kwargs(qos=1, fire_and_forget=fire_and_forget))

        assert result == {"error_status": True, "message": QUEUE_FULL}
        publish_result.wait_for_publish.assert_not_called()

    @pytest.mark.parametrize("qos", [0, 1, 2])
    @pytest.mark.parametrize("fire_and_forget", [False, True])
    def test_backlog_bounded_by_unsent_packets(
        self, mock_client_cls, block, fire_and_forget, qos
    ):
        # max_queued_messages_set() bounds QoS 1/2 messages, not the packets
        # paho buffers for a slow socket; a QoS 1/2 run after a QoS 0 backlog
        # must not grow that buffer past the bound either
        block._connected.set()
        mock_client = mock_client_cls.return_value
        block.run(**run_kwargs(qos=0, fire_and_forget=fire_and_forget))
        mock_client._out_packet = [object()] * MAX_QUEUED_MESSAGES
        mock_client.publish.reset_mock()

        result = block.run(**run_kwargs(qos=qos, fire_and_forget=fire_and_forget))

        assert result == {"error_status": True, "message": QUEUE_FULL}
        mock_client.publish.assert_not_called()

    def test_qos0_publishes_below_the_packet_bound(self, mock_client_cls, block):
        block._connected.set()
        mock_client = mock_client_cls.return_value
        block.run(**run_kwargs(qos=0))
        mock_client._out_packet = [object()] * (MAX_QUEUED_MESSAGES - 1)

        result = block.run(**run_kwargs(qos=0, fire_and_forget=True))

        assert result["error_status"] is False
        assert mock_client.publish.call_count == 2


def _connect_then_drop(mock_client_cls, block, **overrides) -> MagicMock:
    """Run once against a connected broker, then simulate a dropped connection
    the background loop has not re-established yet."""
    block._connected.set()
    first = block.run(**run_kwargs(fire_and_forget=True, **overrides))
    assert first["error_status"] is False
    block._connected.clear()
    mock_client = mock_client_cls.return_value
    mock_client.publish.reset_mock()
    return mock_client


class TestFireAndForget:
    def test_manifest_defaults_to_waiting(self):
        manifest = BlockManifest.model_validate(
            {
                "type": "roboflow_enterprise/mqtt_writer_sink@v2",
                "name": "mqtt",
                "host": "localhost",
                "port": 1883,
                "topic": "t",
                "message": "m",
            }
        )

        assert manifest.fire_and_forget is False

    def test_manifest_accepts_selector(self):
        manifest = BlockManifest.model_validate(
            {
                "type": "roboflow_enterprise/mqtt_writer_sink@v2",
                "name": "mqtt",
                "host": "localhost",
                "port": 1883,
                "topic": "t",
                "message": "m",
                "fire_and_forget": "$inputs.fire_and_forget",
            }
        )

        assert manifest.fire_and_forget == "$inputs.fire_and_forget"

    def test_default_run_still_waits_for_confirmation(self, mock_client_cls, block):
        block._connected.set()

        result = block.run(**run_kwargs(qos=1))

        assert result["message"] == "Message published successfully"
        mock_client_cls.return_value.publish.return_value.wait_for_publish.assert_called_once()

    @pytest.mark.parametrize("qos", [0, 1, 2])
    def test_publish_returns_without_waiting_for_confirmation(
        self, mock_client_cls, block, qos
    ):
        block._connected.set()
        mock_client = mock_client_cls.return_value
        # an unconfirmed message must not matter: nothing waits for it
        mock_client.publish.return_value.is_published.return_value = False

        result = block.run(**run_kwargs(qos=qos, retain=True, fire_and_forget=True))

        assert result == {"error_status": False, "message": FIRE_AND_FORGET_QUEUED}
        mock_client.publish.assert_called_once_with(
            "test/topic", "Hello, MQTT!", qos=qos, retain=True
        )
        mock_client.publish.return_value.wait_for_publish.assert_not_called()
        mock_client.publish.return_value.is_published.assert_not_called()

    def test_first_run_still_waits_for_the_broker(self, mock_client_cls, block):
        started = time.monotonic()
        result = block.run(**run_kwargs(timeout=0.2, fire_and_forget=True))

        assert time.monotonic() - started >= 0.2
        assert result["error_status"] is True
        assert result["message"].startswith(NOT_CONNECTED_WITHIN_TIMEOUT)

    def test_first_run_publishes_once_the_broker_accepts(self, mock_client_cls, block):
        _answer_connect_with(mock_client_cls, 0)

        result = block.run(**run_kwargs(fire_and_forget=True))

        assert result == {"error_status": False, "message": FIRE_AND_FORGET_QUEUED}

    def test_unreachable_broker_on_first_run_keeps_client_retrying(
        self, mock_client_cls, block
    ):
        # without the kept client every later frame would pay a fresh
        # synchronous connect while the broker is down
        mock_client = mock_client_cls.return_value
        mock_client.connect.side_effect = ConnectionRefusedError("refused")

        first = block.run(**run_kwargs(timeout=2.0, fire_and_forget=True))

        assert first["error_status"] is True
        assert "not connected" in first["message"].lower()
        assert block.mqtt_client is mock_client
        mock_client.loop_start.assert_called_once()

        started = time.monotonic()
        second = block.run(**run_kwargs(timeout=2.0, fire_and_forget=True))

        assert time.monotonic() - started < 0.5
        assert second == {"error_status": True, "message": FIRE_AND_FORGET_QOS0_DROPPED}
        mock_client_cls.assert_called_once()
        mock_client.connect.assert_called_once()

    @pytest.mark.parametrize("qos", [1, 2])
    def test_unreachable_broker_on_first_run_queues_qos_1_2(
        self, mock_client_cls, block, qos
    ):
        mock_client = mock_client_cls.return_value
        mock_client.connect.side_effect = ConnectionRefusedError("refused")

        result = block.run(**run_kwargs(qos=qos, retain=True, fire_and_forget=True))

        assert result["error_status"] is True
        assert "refused" in result["message"]
        assert "queued by the client" in result["message"]
        mock_client.publish.assert_called_once_with(
            "test/topic", "Hello, MQTT!", qos=qos, retain=True
        )

    def test_unreachable_broker_on_first_run_drops_qos0(self, mock_client_cls, block):
        mock_client = mock_client_cls.return_value
        mock_client.connect.side_effect = ConnectionRefusedError("refused")

        result = block.run(**run_kwargs(qos=0, fire_and_forget=True))

        assert result["error_status"] is True
        assert "refused" in result["message"]
        assert "dropped" in result["message"]
        mock_client.publish.assert_not_called()

    def test_first_run_loop_start_failure_reported_and_nothing_kept(
        self, mock_client_cls, block
    ):
        mock_client = mock_client_cls.return_value
        mock_client.connect.side_effect = ConnectionRefusedError("refused")
        mock_client.loop_start.side_effect = RuntimeError("no thread")

        result = block.run(**run_kwargs(fire_and_forget=True))

        assert result["error_status"] is True
        assert "Failed to initialize MQTT client: no thread" in result["message"]
        assert block.mqtt_client is None
        assert block._connection.connect_failing is False
        mock_client.loop_stop.assert_called_once()

    @pytest.mark.parametrize("qos", [1, 2])
    def test_first_run_without_connack_queues_qos_1_2(
        self, mock_client_cls, block, qos
    ):
        # TCP connected but the broker never answered within timeout
        mock_client = mock_client_cls.return_value

        result = block.run(**run_kwargs(qos=qos, fire_and_forget=True))

        assert result["error_status"] is True
        assert result["message"].startswith(NOT_CONNECTED_WITHIN_TIMEOUT)
        assert "queued by the client" in result["message"]
        mock_client.publish.assert_called_once_with(
            "test/topic", "Hello, MQTT!", qos=qos, retain=False
        )

    def test_first_run_without_connack_drops_qos0(self, mock_client_cls, block):
        result = block.run(**run_kwargs(qos=0, fire_and_forget=True))

        assert result["error_status"] is True
        assert result["message"].startswith(NOT_CONNECTED_WITHIN_TIMEOUT)
        assert "dropped" in result["message"]
        mock_client_cls.return_value.publish.assert_not_called()

    def test_first_run_broker_unavailable_queues_qos1(self, mock_client_cls, block):
        _answer_connect_with(mock_client_cls, 3)

        result = block.run(**run_kwargs(qos=1, fire_and_forget=True))

        assert result["error_status"] is True
        assert "broker unavailable" in result["message"]
        assert "queued by the client" in result["message"]
        mock_client_cls.return_value.publish.assert_called_once()

    def test_first_run_permanent_refusal_does_not_queue(self, mock_client_cls, block):
        _answer_connect_with(mock_client_cls, 5)

        result = block.run(**run_kwargs(qos=1, fire_and_forget=True))

        assert result["error_status"] is True
        assert "not authorised" in result["message"]
        assert "queued" not in result["message"]
        mock_client_cls.return_value.publish.assert_not_called()

    def test_first_run_queue_full_while_not_connected(self, mock_client_cls, block):
        mock_client = mock_client_cls.return_value
        mock_client.publish.return_value.rc = mqtt.MQTT_ERR_QUEUE_SIZE

        result = block.run(**run_kwargs(qos=1, fire_and_forget=True))

        assert result == {"error_status": True, "message": QUEUE_FULL}

    def test_failed_first_connect_is_not_logged_again_by_the_retry_loop(
        self, mock_client_cls, block
    ):
        mock_client_cls.return_value.connect.side_effect = ConnectionRefusedError(
            "refused"
        )
        block.run(**run_kwargs(fire_and_forget=True))

        with patch.object(v2, "logger") as mock_logger:
            mqtt_on_connect_fail(mock_client_cls.return_value, block._connection)

        mock_logger.error.assert_not_called()

    def test_unreachable_broker_on_first_run_with_fail_fast_raises(
        self, mock_client_cls, block
    ):
        mock_client_cls.return_value.connect.side_effect = ConnectionRefusedError(
            "refused"
        )

        with pytest.raises(RuntimeError, match="not connected"):
            block.run(**run_kwargs(fire_and_forget=True, fail_fast=True))

        assert block.mqtt_client is mock_client_cls.return_value

    def test_permanent_refusal_still_reported(self, mock_client_cls, block):
        _answer_connect_with(mock_client_cls, 5)
        block.run(**run_kwargs(fire_and_forget=True))

        result = block.run(**run_kwargs(fire_and_forget=True))

        assert result["error_status"] is True
        assert "not authorised" in result["message"]
        mock_client_cls.return_value.publish.assert_not_called()

    def test_disconnected_qos0_dropped_without_waiting(self, mock_client_cls, block):
        mock_client = _connect_then_drop(mock_client_cls, block, timeout=2.0)

        started = time.monotonic()
        result = block.run(**run_kwargs(qos=0, timeout=2.0, fire_and_forget=True))

        assert time.monotonic() - started < 0.5
        assert result == {"error_status": True, "message": FIRE_AND_FORGET_QOS0_DROPPED}
        mock_client.publish.assert_not_called()
        mock_client.reconnect.assert_not_called()

    @pytest.mark.parametrize(
        "rc",
        [mqtt.MQTT_ERR_NO_CONN, mqtt.MQTT_ERR_SUCCESS],
        ids=["no-socket", "socket"],
    )
    def test_disconnected_qos1_queued_without_waiting(self, mock_client_cls, block, rc):
        # paho returns NO_CONN without a socket and SUCCESS while the socket is
        # still open; either way the message stays queued for the reconnect
        mock_client = _connect_then_drop(mock_client_cls, block, timeout=2.0, qos=1)
        mock_client.publish.return_value.rc = rc

        started = time.monotonic()
        result = block.run(**run_kwargs(qos=1, timeout=2.0, fire_and_forget=True))

        assert time.monotonic() - started < 0.5
        assert result == {
            "error_status": True,
            "message": FIRE_AND_FORGET_QUEUED_UNTIL_RECONNECT,
        }
        mock_client.publish.assert_called_once()
        mock_client.publish.return_value.wait_for_publish.assert_not_called()

    def test_qos0_connection_lost_during_publish_reported_as_dropped(
        self, mock_client_cls, block
    ):
        block._connected.set()
        mock_client_cls.return_value.publish.return_value.rc = mqtt.MQTT_ERR_NO_CONN

        result = block.run(**run_kwargs(qos=0, fire_and_forget=True))

        assert result == {"error_status": True, "message": FIRE_AND_FORGET_QOS0_DROPPED}

    def test_other_publish_error_reported(self, mock_client_cls, block):
        block._connected.set()
        mock_client_cls.return_value.publish.return_value.rc = mqtt.MQTT_ERR_PROTOCOL

        result = block.run(**run_kwargs(fire_and_forget=True))

        assert result["error_status"] is True
        assert result["message"].startswith("Failed to publish message")

    def test_outage_logged_when_it_starts_changes_and_ends(
        self, mock_client_cls, block
    ):
        mock_client = _connect_then_drop(mock_client_cls, block, qos=1)

        with patch.object(v2, "logger") as mock_logger:
            for _ in range(5):
                result = block.run(**run_kwargs(qos=1, fire_and_forget=True))
                assert result["error_status"] is True
            assert mock_logger.error.call_count == 1

            mock_client.publish.return_value.rc = mqtt.MQTT_ERR_QUEUE_SIZE
            for _ in range(3):
                result = block.run(**run_kwargs(qos=1, fire_and_forget=True))
                assert result["message"] == QUEUE_FULL
            assert mock_logger.error.call_count == 2

            block._connected.set()
            mock_client.publish.return_value.rc = mqtt.MQTT_ERR_SUCCESS
            for _ in range(3):
                result = block.run(**run_kwargs(qos=1, fire_and_forget=True))
                assert result["error_status"] is False
            assert mock_logger.error.call_count == 2
            assert mock_logger.info.call_count == 1

    def test_outage_with_fail_fast_raises_on_every_run(self, mock_client_cls, block):
        _connect_then_drop(mock_client_cls, block)

        for _ in range(2):
            with pytest.raises(RuntimeError, match="QoS 0 messages are not queued"):
                block.run(**run_kwargs(fire_and_forget=True, fail_fast=True))

    def test_switching_between_runs_keeps_the_connection(self, mock_client_cls, block):
        block._connected.set()

        first = block.run(**run_kwargs(fire_and_forget=False))
        second = block.run(**run_kwargs(fire_and_forget=True))

        assert first["message"] == "Message published successfully"
        assert second["message"] == FIRE_AND_FORGET_QUEUED
        mock_client_cls.assert_called_once()

    @pytest.mark.parametrize("value", ["maybe", 2, object()])
    def test_non_boolean_rejected_before_client_construction(
        self, mock_client_cls, block, value
    ):
        result = block.run(**run_kwargs(fire_and_forget=value))

        assert result["error_status"] is True
        assert "Invalid fire_and_forget" in result["message"]
        mock_client_cls.assert_not_called()

    @pytest.mark.parametrize("value", [None, "false", "0", 0, False])
    def test_selector_values_meaning_false_keep_waiting(
        self, mock_client_cls, block, value
    ):
        block._connected.set()

        result = block.run(**run_kwargs(fire_and_forget=value))

        assert result["message"] == "Message published successfully"

    @pytest.mark.parametrize("value", ["true", "1", 1, True])
    def test_selector_values_meaning_true_switch_it_on(
        self, mock_client_cls, block, value
    ):
        block._connected.set()

        result = block.run(**run_kwargs(fire_and_forget=value))

        assert result["message"] == FIRE_AND_FORGET_QUEUED


class TestSelectorSwitches:
    """The engine hands run() the raw selector value, not the manifest's
    coerced one, so the block coerces its switches the same way."""

    @pytest.mark.parametrize("value, expected", [("false", False), ("true", True)])
    def test_retain_string_coerced_like_the_manifest(
        self, mock_client_cls, block, value, expected
    ):
        block._connected.set()

        block.run(**run_kwargs(retain=value))

        mock_client_cls.return_value.publish.assert_called_once_with(
            "test/topic", "Hello, MQTT!", qos=0, retain=expected
        )

    def test_invalid_retain_rejected_before_client_construction(
        self, mock_client_cls, block
    ):
        result = block.run(**run_kwargs(retain="maybe"))

        assert result["error_status"] is True
        assert "Invalid retain" in result["message"]
        mock_client_cls.assert_not_called()

    def test_fail_fast_false_string_returns_the_failure(self, mock_client_cls, block):
        result = block.run(**run_kwargs(fail_fast="false"))

        assert result["error_status"] is True
        assert "not connected" in result["message"].lower()

    def test_fail_fast_true_string_raises(self, mock_client_cls, block):
        with pytest.raises(RuntimeError, match="not connected"):
            block.run(**run_kwargs(fail_fast="true"))

    def test_invalid_fail_fast_reported_without_raising(self, mock_client_cls, block):
        result = block.run(**run_kwargs(fail_fast="maybe"))

        assert result["error_status"] is True
        assert "Invalid fail_fast" in result["message"]
        mock_client_cls.assert_not_called()

    def test_unset_switches_mean_their_defaults(self, mock_client_cls, block):
        block._connected.set()

        result = block.run(**run_kwargs(retain=None, fail_fast=None))

        assert result["message"] == "Message published successfully"
        mock_client_cls.return_value.publish.assert_called_once_with(
            "test/topic", "Hello, MQTT!", qos=0, retain=False
        )


class TestCleanup:
    def test_close_disconnects_and_stops_loop_once(self, mock_client_cls, block):
        block._connected.set()
        mock_client = mock_client_cls.return_value
        block.run(**run_kwargs())

        block.close()
        block.close()
        block.__del__()

        mock_client.disconnect.assert_called_once()
        mock_client.loop_stop.assert_called_once()
        assert block.mqtt_client is None
        assert not block._connected.is_set()

    def test_close_stops_loop_even_when_disconnect_raises(self, mock_client_cls, block):
        block._connected.set()
        mock_client = mock_client_cls.return_value
        mock_client.disconnect.side_effect = Exception("socket already dead")
        block.run(**run_kwargs())

        block.close()

        mock_client.loop_stop.assert_called_once()

    def test_close_on_uninitialized_block_is_noop(self, block):
        block.close()

        assert block.mqtt_client is None

    def test_close_clears_readiness_only_after_loop_thread_joined(
        self, mock_client_cls, block
    ):
        # clearing before loop_stop() lets the dying loop's on_connect re-set
        # the event; the event must stay authoritative until the join returns
        block._connected.set()
        mock_client = mock_client_cls.return_value
        event_state_at_loop_stop = []
        mock_client.loop_stop.side_effect = lambda: event_state_at_loop_stop.append(
            block._connected.is_set()
        )
        block.run(**run_kwargs())

        block.close()

        assert event_state_at_loop_stop == [True]
        assert not block._connected.is_set()

    def test_close_marks_the_state_closing_before_disconnecting(
        self, mock_client_cls, block
    ):
        block._connected.set()
        mock_client = mock_client_cls.return_value
        closing_at_disconnect = []
        mock_client.disconnect.side_effect = lambda: closing_at_disconnect.append(
            block._connection.closing.is_set()
        )
        block.run(**run_kwargs())

        block.close()

        assert closing_at_disconnect == [True]
        assert not block._connection.closing.is_set()

    def test_run_after_close_builds_a_fresh_client_that_is_not_closing(
        self, mock_client_cls, block
    ):
        _answer_connect_with(mock_client_cls, 0)
        block.run(**run_kwargs())
        block.close()

        result = block.run(**run_kwargs())

        assert result["error_status"] is False
        assert mock_client_cls.call_count == 2
        assert block._connection.connected.is_set()
        assert not block._connection.closing.is_set()

    def test_close_shuts_the_socket_down_while_the_loop_thread_outlives_the_join(
        self, mock_client_cls, block
    ):
        block._connected.set()
        mock_client = mock_client_cls.return_value
        mock_client._thread.is_alive.return_value = True
        block.run(**run_kwargs())

        block.close()

        assert mock_client._thread.join.call_count == CLOSE_JOIN_ATTEMPTS
        assert mock_client._sock.shutdown.call_count == CLOSE_JOIN_ATTEMPTS
        mock_client._sock.shutdown.assert_called_with(socket.SHUT_RDWR)
        mock_client.loop_stop.assert_called_once()
        mock_client._sock_close.assert_called_once()

    def test_close_does_not_touch_the_socket_once_the_loop_thread_ended(
        self, mock_client_cls, block
    ):
        block._connected.set()
        mock_client = mock_client_cls.return_value
        mock_client._thread.is_alive.return_value = False
        block.run(**run_kwargs())

        block.close()

        assert mock_client._thread.join.call_count == 1
        mock_client._sock.shutdown.assert_not_called()
        mock_client.loop_stop.assert_called_once()
        mock_client._sock_close.assert_called_once()

    def test_close_from_the_network_thread_only_requests_termination(
        self, mock_client_cls, block
    ):
        block._connected.set()
        mock_client = mock_client_cls.return_value
        mock_client._thread = threading.current_thread()
        block.run(**run_kwargs())

        with patch.object(v2, "logger") as mock_logger:
            block.close()

        assert mock_client._thread_terminate is True
        assert block.mqtt_client is None
        assert block._connection.closing.is_set()
        mock_client._sock_close.assert_not_called()
        mock_client._sock.shutdown.assert_not_called()
        mock_logger.error.assert_not_called()

    def test_close_disables_paho_reconnects_before_disconnecting(
        self, mock_client_cls, block
    ):
        block._connected.set()
        mock_client = mock_client_cls.return_value
        reconnect_flag_at_disconnect = []
        mock_client.disconnect.side_effect = (
            lambda: reconnect_flag_at_disconnect.append(
                mock_client._reconnect_on_failure
            )
        )
        block.run(**run_kwargs())

        block.close()

        assert reconnect_flag_at_disconnect == [False]

    @pytest.mark.parametrize("reason_code", [1, 2])
    def test_connack_refusal_after_close_does_not_reconnect(self, block, reason_code):
        client = mqtt.Client(userdata=block._connection)
        client.reconnect = MagicMock()
        block.mqtt_client = client
        block.close()
        client._in_packet["remaining_length"] = 2
        client._in_packet["packet"] = bytes([0, reason_code])

        rc = client._handle_connack()

        client.reconnect.assert_not_called()
        assert rc == mqtt.MQTT_ERR_PROTOCOL
        assert not block._connection.connack.is_set()


class TestBrokerPolicy:
    def test_allowlisted_broker_is_used(self, mock_client_cls, block, monkeypatch):
        monkeypatch.setattr(
            mqtt_common, "MQTT_WORKFLOWS_BLOCKS_WHITELISTED_HOSTS", ["localhost:1883"]
        )
        block._connected.set()

        result = block.run(**run_kwargs())

        assert result["error_status"] is False
        mock_client_cls.return_value.connect.assert_called_once_with(
            "localhost", 1883, keepalive=MQTT_KEEPALIVE_SECONDS
        )

    def test_unlisted_broker_rejected_before_client_construction(
        self, mock_client_cls, block, monkeypatch
    ):
        monkeypatch.setattr(
            mqtt_common, "MQTT_WORKFLOWS_BLOCKS_WHITELISTED_HOSTS", ["broker.internal"]
        )

        result = block.run(**run_kwargs())

        assert result["error_status"] is True
        assert "not permitted" in result["message"]
        assert "broker.internal" not in result["message"]
        mock_client_cls.assert_not_called()

    def test_unlisted_broker_with_fail_fast_raises(
        self, mock_client_cls, block, monkeypatch
    ):
        monkeypatch.setattr(
            mqtt_common, "MQTT_WORKFLOWS_BLOCKS_WHITELISTED_HOSTS", ["broker.internal"]
        )

        with pytest.raises(RuntimeError, match="not permitted"):
            block.run(**run_kwargs(fail_fast=True))

        mock_client_cls.assert_not_called()

    def test_operator_broker_replaces_workflow_host_when_user_host_not_allowed(
        self, mock_client_cls, block, monkeypatch
    ):
        monkeypatch.setattr(
            mqtt_common, "MQTT_WORKFLOWS_BLOCKS_ALLOW_USER_PROVIDED_HOST", False
        )
        monkeypatch.setattr(
            mqtt_common, "MQTT_WORKFLOWS_BLOCKS_WHITELISTED_HOSTS", ["operator:8883"]
        )
        block._connected.set()

        first = block.run(**run_kwargs(host="workflow-a"))
        second = block.run(**run_kwargs(host="workflow-b"))

        assert first["error_status"] is False
        assert second["error_status"] is False
        mock_client_cls.return_value.connect.assert_called_once_with(
            "operator", 8883, keepalive=MQTT_KEEPALIVE_SECONDS
        )

    def test_user_host_not_allowed_without_operator_broker_disables_block(
        self, mock_client_cls, block, monkeypatch
    ):
        monkeypatch.setattr(
            mqtt_common, "MQTT_WORKFLOWS_BLOCKS_ALLOW_USER_PROVIDED_HOST", False
        )
        monkeypatch.setattr(
            mqtt_common, "MQTT_WORKFLOWS_BLOCKS_WHITELISTED_HOSTS", None
        )

        result = block.run(**run_kwargs())

        assert result["error_status"] is True
        assert "disabled" in result["message"]
        mock_client_cls.assert_not_called()


def _call_names(mock_client) -> list:
    return [call[0] for call in mock_client.method_calls]


@pytest.fixture
def ca_file(tmp_path):
    # a real regular file: configure_tls() refuses anything else before paho
    # (faked in these tests) would read it
    path = tmp_path / "ca.pem"
    path.write_text("content is irrelevant: paho is faked in these tests")
    return path


class TestTLS:
    def test_manifest_defaults_and_relevant_for(self):
        manifest = BlockManifest.model_validate(
            {
                "type": "roboflow_enterprise/mqtt_writer_sink@v2",
                "name": "mqtt",
                "host": "localhost",
                "port": 1883,
                "topic": "test/topic",
                "message": "Hello, MQTT!",
            }
        )
        schema = BlockManifest.model_json_schema()["properties"]

        assert manifest.encryption == "none"
        assert manifest.ca_certificate_path is None
        assert schema["ca_certificate_path"]["relevant_for"] == {
            "encryption": {"values": ["tls"], "required": True}
        }

    @pytest.mark.parametrize("encryption", ["ssl", "TLS", True, None, ""])
    def test_manifest_rejects_other_encryption_values(self, encryption):
        with pytest.raises(ValidationError):
            BlockManifest.model_validate(
                {
                    "type": "roboflow_enterprise/mqtt_writer_sink@v2",
                    "name": "mqtt",
                    "host": "localhost",
                    "port": 1883,
                    "topic": "test/topic",
                    "message": "Hello, MQTT!",
                    "encryption": encryption,
                }
            )

    def test_init_parameters_include_file_system_access(self):
        assert MQTTWriterSinkBlockV2.get_init_parameters() == [
            "disable_sinks",
            "allow_access_to_file_system",
        ]

    @pytest.mark.parametrize("ca_certificate_path", [None, "", "/ca.pem"])
    def test_default_encryption_never_configures_tls(
        self, mock_client_cls, block, ca_certificate_path
    ):
        block._connected.set()

        result = block.run(**run_kwargs(ca_certificate_path=ca_certificate_path))

        assert result["error_status"] is False
        mock_client_cls.return_value.tls_set.assert_not_called()
        mock_client_cls.return_value.tls_insecure_set.assert_not_called()

    def test_tls_without_ca_path_uses_system_store_before_connect(
        self, mock_client_cls, block
    ):
        block._connected.set()
        mock_client = mock_client_cls.return_value

        result = block.run(**run_kwargs(encryption="tls"))

        assert result["error_status"] is False
        mock_client.tls_set.assert_called_once_with()
        names = _call_names(mock_client)
        assert names.index("tls_set") < names.index("connect")
        mock_client.tls_insecure_set.assert_not_called()

    def test_tls_with_ca_path_and_file_system_access(self, mock_client_cls, ca_file):
        block = MQTTWriterSinkBlockV2(allow_access_to_file_system=True)
        block._connected.set()
        mock_client = mock_client_cls.return_value

        result = block.run(
            **run_kwargs(encryption="tls", ca_certificate_path=str(ca_file))
        )

        assert result["error_status"] is False
        mock_client.tls_set.assert_called_once_with(ca_certs=str(ca_file))
        names = _call_names(mock_client)
        assert names.index("tls_set") < names.index("connect")

    def test_tls_with_ca_path_refused_without_file_system_access(
        self, mock_client_cls, block
    ):
        mock_client = mock_client_cls.return_value

        result = block.run(
            **run_kwargs(encryption="tls", ca_certificate_path="/etc/passwd")
        )

        assert result["error_status"] is True
        assert "ca_certificate_path" in result["message"]
        assert "ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE" in result["message"]
        assert block.mqtt_client is None
        mock_client.tls_set.assert_not_called()
        mock_client.connect.assert_not_called()
        mock_client.loop_start.assert_not_called()

    def test_tls_refused_with_fail_fast_raises(self, mock_client_cls, block):
        with pytest.raises(RuntimeError, match="ca_certificate_path"):
            block.run(
                **run_kwargs(
                    encryption="tls", ca_certificate_path="/etc/passwd", fail_fast=True
                )
            )

        assert block.mqtt_client is None

    def test_unloadable_ca_bundle_reported_and_nothing_kept(
        self, mock_client_cls, ca_file
    ):
        block = MQTTWriterSinkBlockV2(allow_access_to_file_system=True)
        mock_client = mock_client_cls.return_value
        mock_client.tls_set.side_effect = FileNotFoundError("no such file")

        result = block.run(
            **run_kwargs(encryption="tls", ca_certificate_path=str(ca_file))
        )

        assert result["error_status"] is True
        assert "could not load CA bundle" in result["message"]
        assert str(ca_file) in result["message"]
        assert block.mqtt_client is None
        mock_client.connect.assert_not_called()
        mock_client.loop_start.assert_not_called()

    @pytest.mark.parametrize(
        "change",
        [{"encryption": "tls"}, {"ca_certificate_path": "other.pem"}],
    )
    def test_changed_tls_parameters_rejected(self, mock_client_cls, change):
        block = MQTTWriterSinkBlockV2(allow_access_to_file_system=True)
        block._connected.set()
        block.run(**run_kwargs())

        result = block.run(**run_kwargs(**change))

        assert result["error_status"] is True
        assert "changed between runs" in result["message"]
        mock_client_cls.assert_called_once()

    def test_invalid_encryption_at_run_time_rejected(self, mock_client_cls, block):
        result = block.run(**run_kwargs(encryption="ssl"))

        assert result["error_status"] is True
        assert "encryption" in result["message"]
        mock_client_cls.assert_not_called()

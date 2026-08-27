import threading
from typing import get_args
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from inference.enterprise.workflows.enterprise_blocks.sinks.mqtt_writer import v1
from inference.enterprise.workflows.enterprise_blocks.sinks.mqtt_writer.v1 import (
    BlockManifest,
    MQTTWriterSinkBlockV1,
    mqtt_on_connect,
    mqtt_on_connect_fail,
    mqtt_on_disconnect,
)

CLIENT_CLASS_PATH = (
    "inference.enterprise.workflows.enterprise_blocks.sinks.mqtt_writer.v1.mqtt.Client"
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
        client_cls.return_value.publish.return_value.is_published.return_value = True
        yield client_cls


@pytest.fixture
def block() -> MQTTWriterSinkBlockV1:
    return MQTTWriterSinkBlockV1()


class TestManifest:
    def test_primary_identifier_is_namespaced(self):
        identifiers = get_args(BlockManifest.model_fields["type"].annotation)

        assert identifiers[0] == "roboflow_enterprise/mqtt_writer_sink@v1"

    def test_legacy_identifier_still_accepted(self):
        manifest = BlockManifest.model_validate(
            {
                "type": "mqtt_writer_sink@v1",
                "name": "mqtt",
                "host": "localhost",
                "port": 1883,
                "topic": "test/topic",
                "message": "Hello, MQTT!",
            }
        )

        assert manifest.type == "mqtt_writer_sink@v1"

    @pytest.mark.parametrize(
        "timeout", [0, -1, float("nan"), float("inf"), float("-inf")]
    )
    def test_literal_timeout_must_be_finite_and_positive(self, timeout):
        with pytest.raises(ValidationError):
            BlockManifest.model_validate(
                {
                    "type": "mqtt_writer_sink@v1",
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
                "type": "mqtt_writer_sink@v1",
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
                    "type": "mqtt_writer_sink@v1",
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
                "type": "mqtt_writer_sink@v1",
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
        event = threading.Event()

        mqtt_on_connect(MagicMock(), event, {}, 0)

        assert event.is_set()

    @pytest.mark.parametrize("reason_code", [1, 2, 3, 4, 5])
    def test_on_connect_clears_event_for_rejected_connack(self, reason_code):
        event = threading.Event()
        event.set()

        mqtt_on_connect(MagicMock(), event, {}, reason_code)

        assert not event.is_set()

    def test_on_connect_fail_matches_paho_two_argument_signature(self):
        event = threading.Event()
        event.set()

        # paho 1.6.1 invokes on_connect_fail with exactly (client, userdata)
        mqtt_on_connect_fail(MagicMock(), event)

        assert not event.is_set()

    def test_on_disconnect_clears_event(self):
        event = threading.Event()
        event.set()

        mqtt_on_disconnect(MagicMock(), event, 1)

        assert not event.is_set()

    def test_on_disconnect_accepts_mqtt5_properties_argument(self):
        event = threading.Event()
        event.set()

        mqtt_on_disconnect(MagicMock(), event, 1, properties=None)

        assert not event.is_set()


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

    @pytest.mark.parametrize("port", [0, -1, 65536, "1883"])
    def test_invalid_port_rejected_before_client_construction(
        self, mock_client_cls, block, port
    ):
        result = block.run(**run_kwargs(port=port))

        assert result["error_status"] is True
        assert "port" in result["message"].lower()
        mock_client_cls.assert_not_called()

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
        mock_client_cls.assert_called_once_with(userdata=block._connected)
        mock_client.connect.assert_called_once_with("localhost", 1883)
        mock_client.connect_async.assert_not_called()
        called_methods = [call[0] for call in mock_client.method_calls]
        assert called_methods.index("connect") < called_methods.index("loop_start")

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

    def test_reconnect_delay_derived_from_timeout(self, mock_client_cls, block):
        block._connected.set()

        block.run(**run_kwargs(timeout=0.5))

        mock_client_cls.return_value.reconnect_delay_set.assert_called_once_with(
            min_delay=0.5, max_delay=1.0
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

    def test_unreachable_broker_on_first_run_reports_failure_and_keeps_retrying(
        self, mock_client_cls, block
    ):
        mock_client = mock_client_cls.return_value
        mock_client.connect.side_effect = ConnectionRefusedError("refused")

        first_result = block.run(**run_kwargs())

        mock_client.connect.assert_called_once()
        assert first_result["error_status"] is True
        assert "not connected" in first_result["message"].lower()
        assert block.mqtt_client is not None
        mock_client.loop_start.assert_called_once()
        mock_client.publish.assert_not_called()

        # background loop establishes the connection later
        block._connected.set()

        second_result = block.run(**run_kwargs())

        assert second_result["error_status"] is False
        mock_client_cls.assert_called_once()

    def test_unreachable_broker_with_fail_fast_raises(self, mock_client_cls, block):
        mock_client = mock_client_cls.return_value
        mock_client.connect.side_effect = OSError("no route to host")

        with pytest.raises(RuntimeError, match="not connected"):
            block.run(**run_kwargs(fail_fast=True))

        mock_client.connect.assert_called_once()

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
        mock_client.connect.assert_called_once_with("broker-a", 1883)
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

    def test_publish_ack_timeout_reported_as_delivery_unknown(
        self, mock_client_cls, block
    ):
        block._connected.set()
        mock_client = mock_client_cls.return_value
        mock_client.publish.return_value.is_published.return_value = False

        with patch.object(v1, "logger") as mock_logger:
            result = block.run(**run_kwargs())

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
        assert "failed to publish" in result["message"].lower()

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

        with patch.object(v1, "logger") as mock_logger:
            with pytest.raises(RuntimeError, match="^Publish acknowledgement"):
                block.run(**run_kwargs(fail_fast=True))

        assert mock_logger.error.call_count == 1


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

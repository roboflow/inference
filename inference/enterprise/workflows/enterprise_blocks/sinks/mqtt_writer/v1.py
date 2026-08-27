import math
import threading
from typing import List, Literal, Optional, Tuple, Type, Union

import paho.mqtt.client as mqtt
from pydantic import ConfigDict, Field
from typing_extensions import Annotated

from inference.core.logger import logger
from inference.core.workflows.core_steps.sinks.noop import disabled_sink_response
from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    FLOAT_KIND,
    INTEGER_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
MQTT Writer block for publishing messages to an MQTT broker.

The first run connects synchronously: the TCP connect and the broker's
session acknowledgement are each bounded by `timeout`, so a cold start may
take up to twice `timeout` - raise it for remote brokers. Afterwards a
background network loop maintains the connection and owns reconnects.

Recovery across runs applies only where the same block instance serves many
runs (video pipelines): a run that finds the broker unavailable reports the
failure while the background loop keeps retrying, and a later run can publish
without reconnecting. Over the HTTP API every request builds a fresh block
instance, so each request pays the bounded connect and a failed request is
final for that request. One block instance publishes to a single broker
connection: changing host, port, credentials or timeout between runs is
rejected as a configuration error.

Outputs:
    - error_status (bool): Indicates if an error occurred during the MQTT publishing process.
                          True if there was an error, False if successful.
    - message (str): Status message describing the result of the operation.
                    Contains error details if error_status is True,
                    or success confirmation if error_status is False.
                    A publish acknowledgement timeout is reported as
                    delivery-unknown: the message may still be delivered.

By default failures are returned in the outputs and logged, and the workflow
keeps running. Set fail_fast to True to raise the failure instead, stopping
the workflow run - intended for one-shot requests, not streaming pipelines.
"""


def mqtt_on_connect(client, userdata, flags, reason_code, properties=None):
    # paho invokes on_connect for accepted and rejected CONNACK alike;
    # only reason_code 0 means an established MQTT session
    if reason_code == 0:
        logger.info("MQTT client connected")
        userdata.set()
    else:
        logger.error("MQTT connection refused, result code %s", reason_code)
        userdata.clear()


def mqtt_on_connect_fail(client, userdata):
    # paho 1.6.1 invokes this callback with exactly (client, userdata)
    logger.error("MQTT client failed to establish connection with broker")
    userdata.clear()


def mqtt_on_disconnect(client, userdata, reason_code, properties=None):
    logger.info("MQTT client disconnected, result code %s", reason_code)
    userdata.clear()


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "MQTT Writer",
            "version": "v1",
            "short_description": "Publishes messages to an MQTT broker.",
            "long_description": LONG_DESCRIPTION,
            "license": "Roboflow Enterprise License",
            "block_type": "sink",
            "ui_manifest": {
                "section": "industrial",
                "icon": "fal fa-network-wired",
                "blockPriority": 10,
                "enterprise_only": True,
                "local_only": True,
            },
        }
    )
    type: Literal["roboflow_enterprise/mqtt_writer_sink@v1", "mqtt_writer_sink@v1"]
    host: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="Host of the MQTT broker.",
        examples=["localhost", "$inputs.mqtt_host"],
    )
    port: Union[
        Annotated[int, Field(ge=1, le=65535)],
        Selector(kind=[INTEGER_KIND]),
    ] = Field(
        description="Port of the MQTT broker (1-65535).",
        examples=[1883, "$inputs.mqtt_port"],
    )
    topic: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="MQTT topic to publish the message to.",
        examples=["sensors/temperature", "$inputs.mqtt_topic"],
    )
    message: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="Message to be published.",
        examples=["Hello, MQTT!", "$inputs.mqtt_message"],
    )
    qos: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=0,
        description="Quality of Service level for the message.",
        examples=[0, 1, 2],
    )
    retain: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=False,
        description="Whether the message should be retained by the broker.",
        examples=[True, False],
    )
    timeout: Union[
        Annotated[float, Field(gt=0, allow_inf_nan=False)],
        Selector(kind=[FLOAT_KIND]),
    ] = Field(
        default=0.5,
        description="Timeout for connecting to the MQTT broker and for sending MQTT messages. "
        "Must be a finite number greater than 0.",
        examples=[0.5],
    )
    username: Union[Selector(kind=[STRING_KIND]), str] = Field(
        default=None,
        description="Username for MQTT broker authentication.",
        examples=["$inputs.mqtt_username"],
    )
    password: Union[Selector(kind=[STRING_KIND]), str] = Field(
        default=None,
        description="Password for MQTT broker authentication.",
        examples=["$inputs.mqtt_password"],
    )
    fail_fast: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=False,
        description="If True, MQTT failures raise an error stopping the workflow run "
        "instead of being returned in the block outputs. Intended for one-shot "
        "requests; in streaming pipelines it stops processing.",
        examples=[False],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="error_status", kind=[BOOLEAN_KIND]),
            OutputDefinition(name="message", kind=[STRING_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class MQTTWriterSinkBlockV1(WorkflowBlock):
    def __init__(self, disable_sinks: bool = False):
        self.mqtt_client: Optional[mqtt.Client] = None
        self._connected = threading.Event()
        self._connection_identity: Optional[Tuple] = None
        self._lifecycle_lock = threading.Lock()
        self._disable_sinks = disable_sinks

    def close(self) -> None:
        with self._lifecycle_lock:
            client = self.mqtt_client
            if client is None:
                return
            self.mqtt_client = None
            self._connection_identity = None
            try:
                client.disconnect()
            except Exception as e:
                logger.error("Failed to disconnect MQTT client: %s", e)
            finally:
                try:
                    # loop_stop() joins the network thread without a timeout;
                    # accepted so the thread never outlives the block
                    client.loop_stop()
                finally:
                    # only after the join can no callback re-set the event
                    self._connected.clear()

    def __del__(self):
        try:
            self.close()
        except Exception as e:
            logger.error("Failed to close MQTT client: %s", e)

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["disable_sinks"]

    def run(
        self,
        host: str,
        port: int,
        topic: str,
        message: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        qos: int = 0,
        retain: bool = False,
        timeout: float = 0.5,
        fail_fast: bool = False,
    ) -> BlockResult:
        if self._disable_sinks:
            return disabled_sink_response()
        # selector-resolved values bypass manifest constraints, so validate here
        try:
            timeout_seconds = float(timeout)
        except (TypeError, ValueError, OverflowError):
            timeout_seconds = math.nan
        if (
            not math.isfinite(timeout_seconds)
            or not 0 < timeout_seconds <= threading.TIMEOUT_MAX
        ):
            return self._handle_failure(
                f"Invalid timeout: {timeout!r}. Timeout must be a positive finite "
                "number of seconds within the platform limit.",
                fail_fast=fail_fast,
            )
        timeout = timeout_seconds
        # selector-supplied ports arrive uncoerced (numeric strings, floats
        # from JSON), so coerce like the manifest would before range-checking;
        # paho validates only port <= 0 and above 65535 the socket call raises
        # OverflowError inside the network loop, silently killing it
        try:
            if isinstance(port, bool) or (
                isinstance(port, float) and not port.is_integer()
            ):
                raise ValueError
            port_number = int(port)
        except (TypeError, ValueError):
            port_number = 0
        if not 1 <= port_number <= 65535:
            return self._handle_failure(
                f"Invalid port: {port!r}. Port must be an integer between 1 and 65535.",
                fail_fast=fail_fast,
            )
        port = port_number
        if password is not None and username is None:
            return self._handle_failure(
                "Password provided without username. Set username to enable MQTT authentication.",
                fail_fast=fail_fast,
            )
        with self._lifecycle_lock:
            return self._connect_and_publish(
                host=host,
                port=port,
                topic=topic,
                message=message,
                username=username,
                password=password,
                qos=qos,
                retain=retain,
                timeout=timeout,
                fail_fast=fail_fast,
            )

    def _connect_and_publish(
        self,
        host: str,
        port: int,
        topic: str,
        message: str,
        username: Optional[str],
        password: Optional[str],
        qos: int,
        retain: bool,
        timeout: float,
        fail_fast: bool,
    ) -> BlockResult:
        connection_identity = (host, port, username, password, timeout)
        if self.mqtt_client is None:
            client = None
            connect_error: Optional[OSError] = None
            try:
                client = mqtt.Client(userdata=self._connected)
                if username is not None:
                    client.username_pw_set(username, password)
                client.on_connect = mqtt_on_connect
                client.on_connect_fail = mqtt_on_connect_fail
                client.on_disconnect = mqtt_on_disconnect
                client.reconnect_delay_set(min_delay=timeout, max_delay=2 * timeout)
                # paho 1.6.1 has no public setter for its synchronous connect
                # timeout; without this the DNS + TCP phase runs under paho's
                # 5s default instead of the block's timeout
                client._connect_timeout = timeout
                try:
                    # DNS + TCP happen here, bounded by _connect_timeout, so
                    # the first run gets a real connect budget; the CONNACK
                    # wait below covers the rest of the handshake
                    client.connect(host, port)
                except OSError as e:
                    # broker unreachable right now; keep the client so the
                    # background loop retries and a later run on this
                    # instance can publish
                    connect_error = e
                client.loop_start()
            except Exception as e:
                if client is not None:
                    try:
                        client.loop_stop()
                    except Exception:
                        pass
                return self._handle_failure(
                    f"Failed to initialize MQTT client: {e}", fail_fast=fail_fast
                )
            self.mqtt_client = client
            self._connection_identity = connection_identity
            if connect_error is not None:
                return self._handle_failure(
                    f"MQTT broker not connected ({connect_error}); the client keeps "
                    "retrying in the background. Raise 'timeout' if the broker "
                    "needs longer to connect.",
                    fail_fast=fail_fast,
                )
        elif connection_identity != self._connection_identity:
            return self._handle_failure(
                "MQTT connection parameters (host, port, credentials or timeout) changed "
                "between runs; this block publishes only to the connection configured "
                "on its first run.",
                fail_fast=fail_fast,
            )
        # the background loop owns (re)connecting; runs only wait for readiness
        if not self._connected.wait(timeout=timeout):
            return self._handle_failure(
                "MQTT broker not connected (connection was not established within timeout); "
                "the client keeps retrying in the background. Raise 'timeout' if the "
                "broker needs longer to connect.",
                fail_fast=fail_fast,
            )
        try:
            res: mqtt.MQTTMessageInfo = self.mqtt_client.publish(
                topic, message, qos=qos, retain=retain
            )
        except Exception as e:
            return self._handle_failure(
                f"Failed to publish message: {e}", fail_fast=fail_fast
            )
        if res.rc == mqtt.MQTT_ERR_NO_CONN and qos > 0:
            # paho keeps QoS 1/2 messages queued for delivery after reconnect
            return self._handle_failure(
                "Connection lost before publish; the message is queued by the client, "
                "delivery status unknown and the message may still be delivered.",
                fail_fast=fail_fast,
            )
        try:
            res.wait_for_publish(timeout=timeout)
            published = res.is_published()
        except Exception as e:
            return self._handle_failure(
                f"Failed to publish message: {e}", fail_fast=fail_fast
            )
        if published:
            return {
                "error_status": False,
                "message": "Message published successfully",
            }
        return self._handle_failure(
            "Publish acknowledgement timed out; delivery status unknown "
            "and the message may still be delivered.",
            fail_fast=fail_fast,
        )

    def _handle_failure(self, message: str, fail_fast: bool) -> BlockResult:
        logger.error("MQTT Writer failure: %s", message)
        if fail_fast:
            raise RuntimeError(message)
        return {"error_status": True, "message": message}

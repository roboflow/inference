import logging
import math
import threading
from typing import List, Literal, Optional, Tuple, Type, Union

import paho.mqtt.client as mqtt
from pydantic import ConfigDict, Field
from typing_extensions import Annotated

# Bind by literal name to the shared "inference" logger tree so server
# handlers / filters / propagation still apply when installed, without
# importing the server logger module.
logger = logging.getLogger("inference")
from roboflow_workflows.core_steps.sinks.noop import disabled_sink_response
from roboflow_workflows.enterprise_blocks.sinks.mqtt_common import (
    PERMANENT_CONNACK_CODES,
    TRANSIENT_CONNACK_CODE,
    ConfigurationError,
    configure_tls,
    connection_refused_message,
    resolve_broker_address,
)
from roboflow_workflows.execution_engine.entities.base import OutputDefinition
from roboflow_workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    FLOAT_KIND,
    INTEGER_KIND,
    STRING_KIND,
    Selector,
)
from roboflow_workflows.execution_engine.entities.workload import (
    Discovery,
    RuntimeRestriction,
    WorkOperation,
)
from roboflow_workflows.prototypes.block import (
    BlockResult,
    DependentResource,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
MQTT Writer block for publishing messages to an MQTT broker.

The first run connects synchronously: the TCP connect and the broker's
session acknowledgement are each bounded by `timeout` (DNS resolution is
bounded by the OS resolver instead), so a cold start may take up to twice
`timeout` - raise it for remote brokers. Afterwards a background network
loop maintains the connection and owns reconnects.

A run that cannot connect reports the failure and leaves nothing behind; the
next run on the same block instance (video pipelines) retries from scratch,
while an established connection that later drops is re-established by the
background loop. Over the HTTP API every request builds a fresh block
instance, so each request pays the bounded connect and a failed request is
final for that request. One block instance publishes to a single broker
connection: changing host, port, credentials, timeout or the TLS settings
between runs is rejected as a configuration error.

The server operator may restrict which brokers the MQTT blocks connect to
with `MQTT_WORKFLOWS_BLOCKS_WHITELISTED_HOSTS` (an allowlist of `host[:port]`
entries) and `MQTT_WORKFLOWS_BLOCKS_ALLOW_USER_PROVIDED_HOST` (when False, the
workflow's host and port are ignored and the first allowlist entry is used);
a run the policy forbids is reported like any other failure.

Set `encryption` to `tls` to encrypt the connection and verify the broker's
certificate against the system trust store; the port is not switched
automatically, so set it to the broker's TLS port (usually 8883). A broker
signed by a private CA needs `ca_certificate_path`, a PEM bundle on the
machine running inference, which requires local file system access for
Workflow blocks (`ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE=True`).
Certificate verification cannot be disabled. A peer that accepts the TCP
connection but never completes the TLS handshake is given up on after the
MQTT keepalive interval (60 s), not after `timeout`; real brokers close such
a connection immediately.

A broker that refuses the connection (bad user name or password, not
authorised, an unacceptable protocol version) is reported in the outputs with
the broker's reason, and the block stops reconnecting: retrying with the same
credentials cannot succeed and would only trip the broker's authentication
rate limiting. Fix the configuration and restart the pipeline. A broker
answering "unavailable" is a passing condition, so the client keeps retrying
in the background.

Outputs:
    - error_status (bool): Indicates if an error occurred during the MQTT publishing process.
                          True if there was an error, False if successful.
    - message (str): Status message describing the result of the operation.
                    Contains error details if error_status is True,
                    or success confirmation if error_status is False.
                    A publish acknowledgement timeout on QoS 1/2 is reported
                    as delivery-unknown (the message may still be delivered);
                    an unconfirmed QoS 0 send is reported as lost.

By default failures are returned in the outputs and logged, and the workflow
keeps running. Set fail_fast to True to raise the failure instead, stopping
the workflow run - intended for one-shot requests, not streaming pipelines.
"""


TLS_RELEVANT = {"encryption": {"values": ["tls"], "required": True}}


def _not_connected_message(error: BaseException, encryption: str) -> str:
    if encryption == "tls":
        return (
            f"MQTT broker not connected ({error}). TLS is enabled: check that port is "
            "the broker's TLS port and that ca_certificate_path matches the broker's "
            "certificate authority."
        )
    return (
        f"MQTT broker not connected ({error}). Raise 'timeout' if the broker needs "
        "longer to connect."
    )


WRITER_CONNECTION_INPUTS = "username and password"
NOT_CONNECTED_WITHIN_TIMEOUT = (
    "MQTT broker not connected (connection was not established within timeout); "
    "the client keeps retrying in the background. Raise 'timeout' if the "
    "broker needs longer to connect."
)


class MQTTWriterState:
    """Connection state shared between paho's network thread and the engine thread.

    Passed to paho as ``userdata`` so the callbacks never hold a reference to the
    block itself.

    Attributes:
        connected: Set once the broker accepted the session; cleared on disconnect.
        connack: Set on every CONNACK, accepted or refused, so a run can stop
            waiting as soon as the broker answered; cleared on a transport
            disconnect, kept while a refusal is recorded.
        refused_code: The CONNACK return code of the last refusal, or None after an
            accepted CONNACK. A permanent code (see ``PERMANENT_CONNACK_CODES``) means
            the network loop was stopped and every run reports the refusal.
    """

    def __init__(self):
        self.connected = threading.Event()
        self.connack = threading.Event()
        self.refused_code: Optional[int] = None

    def reset(self) -> None:
        self.connected.clear()
        self.connack.clear()
        self.refused_code = None


def mqtt_on_connect(
    client, state: MQTTWriterState, flags, reason_code, properties=None
):
    # paho invokes on_connect for accepted and rejected CONNACK alike;
    # only reason_code 0 means an established MQTT session. The outcome is
    # recorded before `connack` wakes a waiting run, so the run never reads a
    # half-updated state
    if reason_code == 0:
        state.refused_code = None
        state.connected.set()
        state.connack.set()
        logger.info("MQTT client connected")
        return
    state.refused_code = reason_code
    state.connected.clear()
    state.connack.set()
    logger.error(
        "MQTT connection refused: %s (code %s)",
        mqtt.connack_string(reason_code),
        reason_code,
    )
    if reason_code in PERMANENT_CONNACK_CODES:
        # paho would otherwise reconnect with the same credentials forever;
        # disconnect() puts it in the disconnecting state, so its loop ends
        # after this CONNACK instead of retrying
        client.disconnect()


def mqtt_on_connect_fail(client, state: MQTTWriterState):
    # paho 1.6.1 invokes this callback with exactly (client, userdata)
    logger.error("MQTT client failed to establish connection with broker")
    state.connected.clear()


def mqtt_on_disconnect(client, state: MQTTWriterState, reason_code, properties=None):
    logger.info("MQTT client disconnected, result code %s", reason_code)
    state.connected.clear()
    if state.refused_code is None:
        # a transport drop: the next run waits for the reconnect's CONNACK; after
        # a refusal the answer stays visible so run() reports it without waiting
        state.connack.clear()


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
    encryption: Literal["none", "tls"] = Field(
        default="none",
        description="Transport security for the broker connection. `tls` encrypts the "
        "connection and verifies the broker's certificate; set port to the broker's "
        "TLS port (usually 8883).",
        examples=["none", "tls"],
        json_schema_extra={
            "values_metadata": {
                "none": {"name": "None", "description": "Plain TCP (default)."},
                "tls": {
                    "name": "TLS",
                    "description": "Encrypted, broker certificate verified.",
                },
            },
        },
    )
    ca_certificate_path: Optional[Union[Selector(kind=[STRING_KIND]), str]] = Field(
        default=None,
        description="Path to a PEM CA bundle on the machine running inference, for "
        "brokers whose certificate issuer is not in the system trust store. Requires "
        "local file system access for Workflow blocks "
        "(`ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE=True`); otherwise the run "
        "reports an error. Used only when encryption is `tls`.",
        examples=["/etc/ssl/certs/factory-ca.pem"],
        json_schema_extra={"relevant_for": TLS_RELEVANT},
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

    def discover_work_operations(self) -> List[WorkOperation]:
        return [WorkOperation.EXTERNAL_REQUEST]

    def get_actual_restrictions(
        self, *, ignore_environment_restrictions: bool = False
    ) -> Discovery[RuntimeRestriction]:
        return Discovery[RuntimeRestriction](
            items=[], complete=True, unknown_reasons=[]
        )

    def discover_dependent_resources(self) -> List[DependentResource]:
        return []


class MQTTWriterSinkBlockV1(WorkflowBlock):
    def __init__(
        self, disable_sinks: bool = False, allow_access_to_file_system: bool = False
    ):
        # gates `ca_certificate_path`, a server-side path chosen by the workflow;
        # False by default so a hand-constructed block is safe
        self._allow_access_to_file_system = allow_access_to_file_system
        self.mqtt_client: Optional[mqtt.Client] = None
        self._connection = MQTTWriterState()
        # the same Event object as the state's, kept under its historical name
        self._connected = self._connection.connected
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
                    # only after the join can no callback re-set the events; a
                    # client rebuilt by a later run starts without a stale refusal
                    self._connection.reset()

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
        return ["disable_sinks", "allow_access_to_file_system"]

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
        encryption: str = "none",
        ca_certificate_path: Optional[str] = None,
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
        try:
            if isinstance(qos, bool) or (
                isinstance(qos, float) and not qos.is_integer()
            ):
                raise ValueError
            qos_number = int(qos)
        except (TypeError, ValueError):
            qos_number = -1
        if qos_number not in (0, 1, 2):
            return self._handle_failure(
                f"Invalid qos: {qos!r}. QoS must be 0, 1 or 2.",
                fail_fast=fail_fast,
            )
        qos = qos_number
        if password is not None and username is None:
            return self._handle_failure(
                "Password provided without username. Set username to enable MQTT authentication.",
                fail_fast=fail_fast,
            )
        if encryption not in ("none", "tls"):
            return self._handle_failure(
                f"Invalid encryption: {encryption!r}. Must be 'none' or 'tls'.",
                fail_fast=fail_fast,
            )
        # an empty editor field is "unset"; anything but a string is a wiring error
        if ca_certificate_path is not None and not isinstance(ca_certificate_path, str):
            return self._handle_failure(
                f"Invalid ca_certificate_path: {ca_certificate_path!r}. Must be a path.",
                fail_fast=fail_fast,
            )
        ca_certificate_path = ca_certificate_path or None
        try:
            # the operator's broker policy; the RESOLVED address is what the
            # block connects to and what the connection identity is built from
            host, port = resolve_broker_address(
                host, port, log_override=self.mqtt_client is None
            )
        except ConfigurationError as e:
            return self._handle_failure(str(e), fail_fast=fail_fast)
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
                encryption=encryption,
                ca_certificate_path=ca_certificate_path,
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
        encryption: str = "none",
        ca_certificate_path: Optional[str] = None,
    ) -> BlockResult:
        connection_identity = (
            host,
            port,
            username,
            password,
            timeout,
            encryption,
            ca_certificate_path,
        )
        if self.mqtt_client is None:
            client = None
            try:
                client = mqtt.Client(userdata=self._connection)
                if username is not None:
                    client.username_pw_set(username, password)
                # TLS must be configured before connect(); the CA path is gated
                # by the engine's file-system permission
                configure_tls(
                    client,
                    use_tls=encryption == "tls",
                    ca_certificate_path=ca_certificate_path,
                    allow_access_to_file_system=self._allow_access_to_file_system,
                )
                client.on_connect = mqtt_on_connect
                client.on_connect_fail = mqtt_on_connect_fail
                client.on_disconnect = mqtt_on_disconnect
                # min_delay stays below the readiness wait (= timeout) so the
                # first reconnect attempt after a connection drop can finish
                # within a single run's wait instead of starting as it expires
                client.reconnect_delay_set(min_delay=timeout / 2, max_delay=2 * timeout)
                # paho 1.6.1 has no public setter for its synchronous connect
                # timeout; without this the TCP phase runs under paho's 5s
                # default instead of the block's timeout
                client._connect_timeout = timeout
                # the TCP connect happens here, bounded by _connect_timeout
                # (DNS resolution is not - it runs under the OS resolver
                # timeout); the CONNACK wait below covers the handshake rest
                client.connect(host, port)
                client.loop_start()
            except ConfigurationError as e:
                # TLS setup refused before any socket was opened: nothing kept
                return self._handle_failure(str(e), fail_fast=fail_fast)
            except OSError as e:
                # broker unreachable: nothing is kept and no loop was started,
                # so a failed one-shot run leaves no background thread behind;
                # the next run on this instance retries with a fresh client
                return self._handle_failure(
                    _not_connected_message(e, encryption=encryption),
                    fail_fast=fail_fast,
                )
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
        elif connection_identity != self._connection_identity:
            return self._handle_failure(
                "MQTT connection parameters (host, port, credentials, timeout, encryption "
                "or ca_certificate_path) changed between runs; this block publishes only "
                "to the connection configured on its first run.",
                fail_fast=fail_fast,
            )
        state = self._connection
        # a permanent refusal stopped the network loop: report it on every run
        # without waiting and without touching the broker again
        if state.refused_code in PERMANENT_CONNACK_CODES:
            return self._handle_failure(
                connection_refused_message(
                    state.refused_code, block_inputs=WRITER_CONNECTION_INPUTS
                ),
                fail_fast=fail_fast,
            )
        # the background loop owns (re)connecting; runs only wait for readiness.
        # The wait ends on any CONNACK, so a refusal is reported as soon as the
        # broker answers instead of after the timeout
        if not state.connected.is_set():
            if not state.connack.wait(timeout=timeout):
                return self._handle_failure(
                    NOT_CONNECTED_WITHIN_TIMEOUT, fail_fast=fail_fast
                )
            if state.refused_code in PERMANENT_CONNACK_CODES:
                return self._handle_failure(
                    connection_refused_message(
                        state.refused_code, block_inputs=WRITER_CONNECTION_INPUTS
                    ),
                    fail_fast=fail_fast,
                )
            if not state.connected.wait(timeout=timeout):
                if state.refused_code == TRANSIENT_CONNACK_CODE:
                    return self._handle_failure(
                        connection_refused_message(
                            state.refused_code, block_inputs=WRITER_CONNECTION_INPUTS
                        ),
                        fail_fast=fail_fast,
                    )
                return self._handle_failure(
                    NOT_CONNECTED_WITHIN_TIMEOUT, fail_fast=fail_fast
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
        if qos == 0:
            # an unconfirmed QoS 0 send is a real loss: paho never queues
            # QoS 0 messages and reconnect() clears the pending out packet
            return self._handle_failure(
                "Publish confirmation timed out; the connection dropped before "
                "the message was fully sent and QoS 0 messages are not "
                "retransmitted, so the message is lost.",
                fail_fast=fail_fast,
            )
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

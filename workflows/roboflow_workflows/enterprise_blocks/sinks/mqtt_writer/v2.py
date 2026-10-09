import logging
import math
import threading
import time
from typing import Any, List, Literal, Optional, Tuple, Type, Union

import paho.mqtt.client as mqtt
from pydantic import ConfigDict, Field, TypeAdapter, ValidationError
from typing_extensions import Annotated

# Bind by literal name to the shared "inference" logger tree so server
# handlers / filters / propagation still apply when installed, without
# importing the server logger module.
logger = logging.getLogger("inference")
from roboflow_workflows.core_steps.common.workload_presets import (
    FIRE_AND_FORGET_RESTRICTION,
)
from roboflow_workflows.core_steps.sinks.noop import disabled_sink_response
from roboflow_workflows.enterprise_blocks.sinks.mqtt_common import (
    MQTT_KEEPALIVE_SECONDS,
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
    incomplete_discovery,
    unresolved_selector_problem,
)
from roboflow_workflows.prototypes.block import (
    BlockResult,
    DependentResource,
    Runtime,
    RuntimeInputMode,
    Severity,
    WorkflowBlock,
    WorkflowBlockManifest,
    actual_restrictions_of,
    is_workflow_selector,
)

LONG_DESCRIPTION = """
MQTT Writer block for publishing messages to an MQTT broker.

Compared with v1, this version adds `fire_and_forget`, bounds the client's
outgoing queue, and logs a broker outage once instead of on every retry.

The first run connects synchronously: the TCP connect is bounded by `timeout`
per resolved address and the broker's session acknowledgement by `timeout`.
DNS resolution is bounded by the OS resolver and a TLS handshake by the
15 s MQTT keepalive, not by `timeout`. Raise `timeout` for remote brokers.
Afterwards a background network loop maintains the connection and owns
reconnects with an exponential backoff: the first retry waits half of
`timeout` (at least 0.1 s), each further one doubles that, up to twice
`timeout` (at least 1 s), plus the time each attempt takes. A large `timeout`
therefore also slows recovery after an outage.

Without `fire_and_forget`, a run whose TCP connect fails reports the failure
and leaves nothing behind; the next run on the same block instance (video
pipelines) retries from scratch. Once the TCP connection is open the client
is kept: a broker that does not answer within `timeout` or answers
"unavailable" is retried by the background loop, which also re-establishes an
established connection that later drops. Over the HTTP API every request
builds a fresh block instance, so each request pays the bounded connect and a
failed request is final for that request. One block instance publishes to a
single broker connection: changing host, port, credentials, timeout or the TLS
settings between runs is rejected as a configuration error.

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
MQTT keepalive interval (15 s), not after `timeout`; real brokers close such
a connection immediately.

A broker that refuses the connection (bad user name or password, not
authorised, an unacceptable protocol version) is reported in the outputs with
the broker's reason, and the block stops reconnecting: retrying with the same
credentials cannot succeed and would only trip the broker's authentication
rate limiting. Fix the configuration and restart the pipeline. A broker
answering "unavailable" is a passing condition, so the client keeps retrying
in the background.

Set `fire_and_forget` to True to stop runs from waiting on the broker. A run
hands the message to the client and returns without waiting for the send or
the broker's acknowledgement, so success means "queued for publishing", not
"delivered". The first run still connects synchronously; afterwards a run
never waits for a reconnect. If that first connection attempt fails, the
client is kept and retries in the background, so a broker that is down when
the pipeline starts does not block every frame either. While the broker is
disconnected a QoS 0 message is dropped (paho does not queue QoS 0), and a
QoS 1/2 message is queued by the client and sent once the background loop
reconnects; both are reported with `error_status` set on every run, but logged
only when the failure starts or changes, and the recovery is logged once. In
either mode the block refuses a new publish, and reports it as dropped, once
1000 QoS 1/2 messages await acknowledgement or 1000 packets wait to be sent
(the client's own limit counts only QoS 1/2 messages). After a reconnect the
client re-sends its queued QoS 1/2 messages, so up to about 2000 packets can
be buffered briefly. The limits count messages, not bytes: with large
payloads a long outage can still hold up to 1000 of them in memory, so prefer
QoS 0 for large messages. Fire and forget is meant for InferencePipelines: over
the HTTP API every request builds a fresh block instance, so each request is a
first run, and a message still queued when the request ends is dropped.

Outputs:
    - error_status (bool): Indicates if an error occurred during the MQTT publishing process.
                          True if there was an error, False if successful.
    - message (str): Status message describing the result of the operation.
                    Contains error details if error_status is True,
                    or success confirmation if error_status is False.
                    A publish acknowledgement timeout on QoS 1/2 is reported
                    as delivery-unknown (the message may still be delivered);
                    an unconfirmed QoS 0 send is reported as not fully sent
                    within timeout (it may still be sent or may be lost). With
                    fire_and_forget a successful run only confirms the
                    message was queued.

By default failures are returned in the outputs and logged, and the workflow
keeps running. Set fail_fast to True to raise the failure instead, stopping
the workflow run - intended for one-shot requests, not streaming pipelines.
"""


TLS_RELEVANT = {"encryption": {"values": ["tls"], "required": True}}

# the fire-and-forget client and its queue live in the block instance, which
# the HTTP API rebuilds for every request
FIRE_AND_FORGET_PER_REQUEST_RESTRICTION = RuntimeRestriction(
    code="connection_and_state_rebuilt_per_request",
    severity=Severity.SOFT,
    note=(
        "With fire_and_forget the broker connection and the client's message queue "
        "live in this block instance. Over HTTP every request builds a fresh "
        "instance, so each request still connects before publishing, and a QoS 1/2 "
        "message queued while the broker is down is dropped when the request ends. "
        "Use an InferencePipeline for fire-and-forget publishing."
    ),
    applies_to_runtimes=[
        Runtime.HOSTED_SERVERLESS,
        Runtime.SELF_HOSTED_CPU,
        Runtime.SELF_HOSTED_GPU,
        Runtime.DEDICATED_DEPLOYMENT,
    ],
    applies_to_input_modes=[RuntimeInputMode.IMAGE],
)


BOOLEAN_ADAPTER = TypeAdapter(bool)
INTEGER_ADAPTER = TypeAdapter(int)


def _coerce_integer(value: Any) -> int:
    """Coerce a selector-resolved integer the way the manifest would.

    Accepts what the manifest's validation accepts (``1883``, ``"1883"``,
    ``"1883.0"``, ``1883.0``) except booleans, which are never a port or QoS.

    Args:
        value: The literal or selector-resolved value.

    Returns:
        The integer value.

    Raises:
        ValidationError: When the value is a boolean or not an integral number.
    """
    if isinstance(value, bool):
        return INTEGER_ADAPTER.validate_python(value, strict=True)
    return INTEGER_ADAPTER.validate_python(value)


def _coerce_switch(value: Any) -> bool:
    """Coerce a selector-resolved switch the way the manifest would.

    The engine validates selector values on a copy of the manifest and hands
    ``run()`` the raw value, so a ``"false"`` string would otherwise be truthy.

    Args:
        value: The literal or selector-resolved value; ``None`` (an unset
            input) means the default, False.

    Returns:
        The boolean the manifest's validation accepts the value as.

    Raises:
        ValidationError: When the value is not a boolean or a boolean-like
            string or number.
    """
    if value is None:
        return False
    return BOOLEAN_ADAPTER.validate_python(value)


def _not_connected_message(error: BaseException, encryption: str) -> str:
    if encryption == "tls":
        return (
            f"MQTT broker not connected ({error}). TLS is enabled: check that port is "
            "the broker's TLS port and that ca_certificate_path matches the broker's "
            "certificate authority."
        )
    if isinstance(error, TimeoutError):
        return (
            f"MQTT broker not connected ({error}). Raise 'timeout' if the broker "
            "needs longer to connect."
        )
    return f"MQTT broker not connected ({error})."


WRITER_CONNECTION_INPUTS = "username and password"
NOT_CONNECTED_WITHIN_TIMEOUT = (
    "MQTT broker not connected (connection was not established within timeout); "
    "the client keeps retrying in the background. Raise 'timeout' if the "
    "broker needs longer to connect."
)
# paho keeps QoS 1/2 messages until the broker acknowledges them and QoS 0
# packets until the socket takes them, both by default without limit; a
# fire-and-forget pipeline publishing through a long outage or to a slow broker
# would otherwise grow the buffer for as long as that lasts
MAX_QUEUED_MESSAGES = 1000
# the reconnect schedule follows `timeout` (half of it for the first retry, up
# to twice it between retries), bounded below so a tiny timeout cannot retry a
# dead broker in a busy loop
MIN_RECONNECT_DELAY_SECONDS = 0.1
MIN_MAX_RECONNECT_DELAY_SECONDS = 1.0
QUEUE_FULL = (
    f"MQTT client queue is full ({MAX_QUEUED_MESSAGES} messages awaiting "
    "delivery); the message was dropped."
)
FIRE_AND_FORGET_QUEUED = (
    "Message queued for publishing; delivery is not confirmed (fire_and_forget)."
)
FIRE_AND_FORGET_QUEUED_UNTIL_RECONNECT = (
    "MQTT broker not connected; the message is queued by the client and sent "
    "once the background loop reconnects."
)
FIRE_AND_FORGET_QOS0_DROPPED = (
    "MQTT broker not connected; QoS 0 messages are not queued, so the message "
    "was dropped. The client keeps reconnecting in the background."
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
        connect_failing: Set by a failed connection attempt or a refused CONNACK
            and cleared by an accepted CONNACK, so an outage is logged once
            rather than on every background retry.
    """

    def __init__(self):
        self.connected = threading.Event()
        self.connack = threading.Event()
        self.refused_code: Optional[int] = None
        self.connect_failing = False

    def reset(self) -> None:
        self.connected.clear()
        self.connack.clear()
        self.refused_code = None
        self.connect_failing = False


def mqtt_on_connect(
    client, state: MQTTWriterState, flags, reason_code, properties=None
):
    # paho invokes on_connect for accepted and rejected CONNACK alike;
    # only reason_code 0 means an established MQTT session. The outcome is
    # recorded before `connack` wakes a waiting run, so the run never reads a
    # half-updated state
    if reason_code == 0:
        state.refused_code = None
        state.connect_failing = False
        state.connected.set()
        state.connack.set()
        logger.info("MQTT client connected")
        return
    state.refused_code = reason_code
    state.connected.clear()
    state.connack.set()
    # a transient refusal comes back on every background retry, so only the
    # first one of an outage is an error; a permanent one ends the loop
    repeated = state.connect_failing and reason_code not in PERMANENT_CONNACK_CODES
    state.connect_failing = True
    (logger.debug if repeated else logger.error)(
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
    # paho 1.6.1 invokes this callback with exactly (client, userdata). The
    # background loop retries about once per reconnect delay while the broker
    # is down, so only the first failure of an outage is an error
    if state.connect_failing:
        logger.debug("MQTT client still failing to connect to broker")
    else:
        logger.error("MQTT client failed to establish connection with broker")
    state.connect_failing = True
    state.connected.clear()
    if state.refused_code not in PERMANENT_CONNACK_CODES:
        state.refused_code = None
        state.connack.clear()


def mqtt_on_disconnect(client, state: MQTTWriterState, reason_code, properties=None):
    # paho also ends every refused background retry here; only the loss of an
    # established session is worth an info line
    (logger.info if state.connected.is_set() else logger.debug)(
        "MQTT client disconnected, result code %s", reason_code
    )
    state.connected.clear()
    if state.refused_code is None:
        # a transport drop: the next run waits for the reconnect's CONNACK; after
        # a refusal the answer stays visible so run() reports it without waiting
        state.connack.clear()


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "MQTT Writer",
            "version": "v2",
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
    type: Literal["roboflow_enterprise/mqtt_writer_sink@v2"]
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
    fire_and_forget: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=False,
        description="If True, a run queues the message and returns without waiting "
        "for the broker to receive or acknowledge it, and does not wait for a "
        "reconnect after the first run. Success then means queued, not delivered. "
        "While the broker is disconnected QoS 0 messages are dropped and QoS 1/2 "
        "messages are sent after reconnect.",
        examples=[False, "$inputs.mqtt_fire_and_forget"],
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
        # the delivery caveat exists only when the run does not wait for the broker
        declared: Union[List[RuntimeRestriction], Discovery[RuntimeRestriction]]
        if is_workflow_selector(self.fire_and_forget):
            # a runtime value decides whether delivery is awaited, so the caveat
            # MAY apply: declare nothing complete instead of guessing either way
            declared = incomplete_discovery(
                items=[],
                reasons=[
                    unresolved_selector_problem(
                        node_id=f"$steps.{getattr(self, 'name', '')}",
                        declaration="restrictions",
                        field="fire_and_forget",
                        selector=self.fire_and_forget,
                    )
                ],
            )
        elif self.fire_and_forget:
            declared = [
                FIRE_AND_FORGET_RESTRICTION,
                FIRE_AND_FORGET_PER_REQUEST_RESTRICTION,
            ]
        else:
            declared = []
        return actual_restrictions_of(
            declared=declared,
            node_id=f"$steps.{getattr(self, 'name', '')}",
            ignore_environment_restrictions=ignore_environment_restrictions,
        )

    def discover_dependent_resources(self) -> List[DependentResource]:
        return []


class MQTTWriterSinkBlockV2(WorkflowBlock):
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
        # the fire-and-forget failure last logged; repeats are not logged again
        self._fire_and_forget_failure: Optional[str] = None

    def close(self) -> None:
        with self._lifecycle_lock:
            client = self.mqtt_client
            if client is None:
                return
            self.mqtt_client = None
            self._connection_identity = None
            self._fire_and_forget_failure = None
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
        fire_and_forget: bool = False,
    ) -> BlockResult:
        if self._disable_sinks:
            return disabled_sink_response()
        try:
            fail_fast = _coerce_switch(fail_fast)
        except ValidationError:
            return self._handle_failure(
                f"Invalid fail_fast: {fail_fast!r}. Must be a boolean.",
                fail_fast=False,
            )
        # selector-resolved values bypass manifest constraints, so validate here
        try:
            if isinstance(timeout, bool):
                raise TypeError("a boolean is not a timeout")
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
            port_number = _coerce_integer(port)
        except ValidationError:
            port_number = 0
        if not 1 <= port_number <= 65535:
            return self._handle_failure(
                f"Invalid port: {port!r}. Port must be an integer between 1 and 65535.",
                fail_fast=fail_fast,
            )
        port = port_number
        try:
            qos_number = _coerce_integer(qos)
        except ValidationError:
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
            retain = _coerce_switch(retain)
        except ValidationError:
            return self._handle_failure(
                f"Invalid retain: {retain!r}. Must be a boolean.", fail_fast=fail_fast
            )
        try:
            fire_and_forget = _coerce_switch(fire_and_forget)
        except ValidationError:
            return self._handle_failure(
                f"Invalid fire_and_forget: {fire_and_forget!r}. Must be a boolean.",
                fail_fast=fail_fast,
            )
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
                fire_and_forget=fire_and_forget,
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
        fire_and_forget: bool = False,
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
        created_client = self.mqtt_client is None
        if created_client:
            client = None
            try:
                client = mqtt.Client(userdata=self._connection)
                client.max_queued_messages_set(MAX_QUEUED_MESSAGES)
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
                # within a single run's wait instead of starting as it expires;
                # the floors keep a tiny timeout from turning the background
                # retries against a dead broker into a busy loop
                client.reconnect_delay_set(
                    min_delay=max(timeout / 2, MIN_RECONNECT_DELAY_SECONDS),
                    max_delay=max(2 * timeout, MIN_MAX_RECONNECT_DELAY_SECONDS),
                )
                # paho 1.6.1 has no public setter for its synchronous connect
                # timeout; without this the TCP phase runs under paho's 5s
                # default instead of the block's timeout
                client._connect_timeout = timeout
                # the TCP connect happens here, bounded by _connect_timeout
                # (DNS resolution is not - it runs under the OS resolver
                # timeout); the CONNACK wait below covers the handshake rest
                client.connect(host, port, keepalive=MQTT_KEEPALIVE_SECONDS)
                client.loop_start()
            except ConfigurationError as e:
                # TLS setup refused before any socket was opened: nothing kept
                return self._handle_failure(str(e), fail_fast=fail_fast)
            except OSError as e:
                error = _not_connected_message(e, encryption=encryption)
                if fire_and_forget:
                    return self._keep_retrying_after_failed_connect(
                        client,
                        connection_identity=connection_identity,
                        error=error,
                        topic=topic,
                        message=message,
                        qos=qos,
                        retain=retain,
                        fail_fast=fail_fast,
                    )
                # broker unreachable: nothing is kept and no loop was started,
                # so a failed one-shot run leaves no background thread behind;
                # the next run on this instance retries with a fresh client
                return self._handle_failure(error, fail_fast=fail_fast)
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
            return self._report_failure(
                connection_refused_message(
                    state.refused_code, block_inputs=WRITER_CONNECTION_INPUTS
                ),
                fail_fast=fail_fast,
                fire_and_forget=fire_and_forget,
            )
        # fire and forget waits for the broker only on the run that built the
        # client; afterwards a disconnected run returns at once and leaves the
        # reconnect to the background loop
        waits_for_broker = created_client or not fire_and_forget
        if not waits_for_broker and qos == 0 and not state.connected.is_set():
            return self._fire_and_forget_outage(
                FIRE_AND_FORGET_QOS0_DROPPED, fail_fast=fail_fast
            )
        # the background loop owns (re)connecting; runs only wait for readiness.
        # The wait ends on any CONNACK, so a refusal is reported as soon as the
        # broker answers instead of after the timeout
        if waits_for_broker and not state.connected.is_set():
            not_ready = self._wait_until_connected(timeout)
            if not_ready is not None:
                # a permanent refusal stopped the loop: nothing would be sent
                if not fire_and_forget or state.refused_code in PERMANENT_CONNACK_CODES:
                    return self._report_failure(
                        not_ready,
                        fail_fast=fail_fast,
                        fire_and_forget=fire_and_forget,
                    )
                return self._report_not_connected(
                    not_ready,
                    topic=topic,
                    message=message,
                    qos=qos,
                    retain=retain,
                    fail_fast=fail_fast,
                )
        connected = state.connected.is_set()
        # max_queued_messages_set() bounds the QoS 1/2 messages, not the encoded
        # packets paho buffers while the socket cannot drain them (a broker
        # slower than the stream); 1.6.1 exposes that buffer only privately.
        # Checked for every QoS, so runs alternating QoS cannot exceed it
        if len(self.mqtt_client._out_packet) >= MAX_QUEUED_MESSAGES:
            if fire_and_forget:
                return self._fire_and_forget_outage(QUEUE_FULL, fail_fast=fail_fast)
            return self._handle_failure(QUEUE_FULL, fail_fast=fail_fast)
        try:
            res: mqtt.MQTTMessageInfo = self.mqtt_client.publish(
                topic, message, qos=qos, retain=retain
            )
        except Exception as e:
            return self._handle_failure(
                f"Failed to publish message: {e}", fail_fast=fail_fast
            )
        if fire_and_forget:
            return self._fire_and_forget_result(
                res.rc, qos=qos, connected=connected, fail_fast=fail_fast
            )
        if res.rc == mqtt.MQTT_ERR_QUEUE_SIZE:
            return self._handle_failure(QUEUE_FULL, fail_fast=fail_fast)
        if res.rc == mqtt.MQTT_ERR_NO_CONN and qos > 0:
            # paho keeps QoS 1/2 messages queued for delivery after reconnect
            return self._handle_failure(
                "Connection lost before publish; the message is queued by the client, "
                "delivery status unknown and the message may still be delivered.",
                fail_fast=fail_fast,
            )
        if res.rc == mqtt.MQTT_ERR_NO_CONN and qos == 0:
            return self._handle_failure(
                "MQTT broker not connected; QoS 0 messages are not queued, so the "
                "message was dropped.",
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
            return self._handle_failure(
                "Publish confirmation timed out; the QoS 0 message was not fully "
                "sent within timeout and may still be sent or may be lost.",
                fail_fast=fail_fast,
            )
        return self._handle_failure(
            "Publish acknowledgement timed out; delivery status unknown "
            "and the message may still be delivered.",
            fail_fast=fail_fast,
        )

    def _keep_retrying_after_failed_connect(
        self,
        client: mqtt.Client,
        connection_identity: Tuple,
        error: str,
        topic: str,
        message: str,
        qos: int,
        retain: bool,
        fail_fast: bool,
    ) -> BlockResult:
        # keep the client and let paho's loop retry the connection in the
        # background; otherwise every later run would block on a fresh
        # synchronous connect for as long as the broker is down. This run
        # logs the failure, so the loop's retries do not log it again
        self._connection.connect_failing = True
        try:
            client.loop_start()
        except Exception as e:
            self._connection.connect_failing = False
            try:
                client.loop_stop()
            except Exception:
                pass
            return self._handle_failure(
                f"Failed to initialize MQTT client: {e}", fail_fast=fail_fast
            )
        self.mqtt_client = client
        self._connection_identity = connection_identity
        return self._report_not_connected(
            error,
            topic=topic,
            message=message,
            qos=qos,
            retain=retain,
            fail_fast=fail_fast,
        )

    def _wait_until_connected(self, timeout: float) -> Optional[str]:
        """Wait for the background loop to establish the session.

        Args:
            timeout: Total bound for the CONNACK wait and, after a transient
                refusal, the reconnect.

        Returns:
            None once connected, otherwise the reason the broker is not ready.
        """
        state = self._connection
        deadline = time.monotonic() + timeout
        if not state.connack.wait(timeout=timeout):
            return NOT_CONNECTED_WITHIN_TIMEOUT
        if state.refused_code in PERMANENT_CONNACK_CODES:
            return connection_refused_message(
                state.refused_code, block_inputs=WRITER_CONNECTION_INPUTS
            )
        if not state.connected.wait(timeout=max(0.0, deadline - time.monotonic())):
            if state.refused_code == TRANSIENT_CONNACK_CODE:
                return connection_refused_message(
                    state.refused_code, block_inputs=WRITER_CONNECTION_INPUTS
                )
            return NOT_CONNECTED_WITHIN_TIMEOUT
        return None

    def _report_not_connected(
        self,
        error: str,
        topic: str,
        message: str,
        qos: int,
        retain: bool,
        fail_fast: bool,
    ) -> BlockResult:
        # fire and forget on a run that found no session: QoS 0 has nothing to
        # queue, QoS 1/2 is handed to paho for the background connection
        if qos == 0:
            return self._fire_and_forget_outage(
                f"{error} QoS 0 messages are not queued, so the message was dropped.",
                fail_fast=fail_fast,
            )
        try:
            res = self.mqtt_client.publish(topic, message, qos=qos, retain=retain)
        except Exception as e:
            return self._handle_failure(
                f"Failed to publish message: {e}", fail_fast=fail_fast
            )
        if res.rc == mqtt.MQTT_ERR_QUEUE_SIZE:
            return self._fire_and_forget_outage(QUEUE_FULL, fail_fast=fail_fast)
        return self._fire_and_forget_outage(
            f"{error} The message is queued by the client and sent once it connects; "
            "over the HTTP API it is dropped when the request ends.",
            fail_fast=fail_fast,
        )

    def _fire_and_forget_result(
        self, rc: int, qos: int, connected: bool, fail_fast: bool
    ) -> BlockResult:
        if rc == mqtt.MQTT_ERR_SUCCESS and (connected or qos == 0):
            if self._fire_and_forget_failure is not None:
                logger.info(
                    "MQTT Writer publishing again after: %s",
                    self._fire_and_forget_failure,
                )
                self._fire_and_forget_failure = None
            return {"error_status": False, "message": FIRE_AND_FORGET_QUEUED}
        if rc == mqtt.MQTT_ERR_QUEUE_SIZE:
            return self._fire_and_forget_outage(QUEUE_FULL, fail_fast=fail_fast)
        if qos > 0 and rc in (mqtt.MQTT_ERR_SUCCESS, mqtt.MQTT_ERR_NO_CONN):
            # paho keeps QoS 1/2 messages and sends them after the reconnect
            return self._fire_and_forget_outage(
                FIRE_AND_FORGET_QUEUED_UNTIL_RECONNECT, fail_fast=fail_fast
            )
        if rc == mqtt.MQTT_ERR_NO_CONN:
            # the connection dropped between the readiness check and publish()
            return self._fire_and_forget_outage(
                FIRE_AND_FORGET_QOS0_DROPPED, fail_fast=fail_fast
            )
        return self._handle_failure(
            f"Failed to publish message: {mqtt.error_string(rc)}",
            fail_fast=fail_fast,
        )

    def _report_failure(
        self, message: str, fail_fast: bool, fire_and_forget: bool
    ) -> BlockResult:
        if fire_and_forget:
            return self._fire_and_forget_outage(message, fail_fast=fail_fast)
        return self._handle_failure(message, fail_fast=fail_fast)

    def _fire_and_forget_outage(self, message: str, fail_fast: bool) -> BlockResult:
        # without the readiness wait a pipeline reaches this on every frame of
        # an outage; log when the failure starts or changes, not once per run
        log = message != self._fire_and_forget_failure
        self._fire_and_forget_failure = message
        return self._handle_failure(message, fail_fast=fail_fast, log=log)

    def _handle_failure(
        self, message: str, fail_fast: bool, log: bool = True
    ) -> BlockResult:
        if log:
            logger.error("MQTT Writer failure: %s", message)
        if fail_fast:
            raise RuntimeError(message)
        return {"error_status": True, "message": message}

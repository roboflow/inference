import json
import logging
import math
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Literal, Optional, Tuple, Type, Union

import paho.mqtt.client as mqtt
from pydantic import ConfigDict, Field
from typing_extensions import Annotated

# Bind by literal name to the shared "inference" logger tree so server
# handlers / filters / propagation still apply when installed, without
# importing the server logger module.
logger = logging.getLogger("inference")
from roboflow_workflows.enterprise_blocks.sinks.mqtt_common import (
    MQTT_KEEPALIVE_SECONDS,
    PERMANENT_CONNACK_CODES,
    TRANSIENT_CONNACK_CODE,
    ConfigurationError,
    configure_tls,
    connection_refused_message,
    normalise_client_id,
    resolve_broker_address,
)
from roboflow_workflows.environment import GCP_SERVERLESS, LAMBDA
from roboflow_workflows.execution_engine.entities.base import OutputDefinition
from roboflow_workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    DICTIONARY_KIND,
    FLOAT_KIND,
    INTEGER_KIND,
    SECRET_KIND,
    STRING_KIND,
    Selector,
)
from roboflow_workflows.execution_engine.entities.workload import (
    Discovery,
    WorkOperation,
)
from roboflow_workflows.prototypes.block import (
    BlockResult,
    DependentResource,
    Runtime,
    RuntimeInputMode,
    RuntimeRestriction,
    Severity,
    WorkflowBlock,
    WorkflowBlockManifest,
    actual_restrictions_of,
)

LONG_DESCRIPTION = """
MQTT Reader block for subscribing to an MQTT topic and exposing one received message
per workflow run as a raw string (`value`) and a parsed JSON object (`payload`).

The first run connects synchronously: the TCP connect, the broker's session
acknowledgement and the subscription acknowledgement are each bounded by `timeout`,
and the first run then waits up to `timeout` for a retained message the broker
delivers on subscription, so a cold start may take up to four times `timeout` -
raise it for remote brokers. Afterwards a background network loop maintains the
connection, re-establishes the subscription after every reconnect and buffers
incoming messages between runs.

`read_mode` selects what a run returns from that buffer:

- `latest` (default): the newest message received since the previous run; anything
  older is skipped.
- `sequential`: the next unread message, one per run, in arrival order. A backlog is
  worked through one per run; when more than 1000 messages queue up between two runs
  the oldest are dropped.

When no new message arrived, the outputs of the previous run are repeated with
`is_new` set to False. Retained messages delivered by the broker on subscription are
treated like any other message, except that a retained copy the broker re-sends after
a reconnect is not reported as new when its topic and payload match the message
returned last.

The subscription and the buffer live in the block instance only. An InferencePipeline
keeps them for its lifetime, a restart begins with an empty buffer, and over the HTTP
API every request builds a fresh instance, so only a retained message can be returned
there. By default the session is not persistent either: messages published while the
block was not subscribed are not delivered.

Set `client_id` to a stable name that is unique on the broker (for example the camera
or pipeline name from a workflow input) to keep a persistent session instead. The
presence of the id is the switch: the block then connects with that id and asks the
broker to keep the session, so the broker remembers the subscription and queues QoS 1
and 2 messages while the block is away and delivers the backlog when a block with the
same id reconnects, after a pipeline or process restart included. `qos` must then be 1
or 2 and the publisher must also publish at QoS 1 or higher, because a broker queues
nothing for QoS 0. Use `sequential` to work through the backlog one message per run;
`latest` keeps only the newest of it. QoS 1 may redeliver a message whose
acknowledgement was lost, and `sequential` surfaces that duplicate; QoS 2 avoids it.
Two connections with the same id disconnect each other, so never share an id between
pipelines. Persistent sessions are meant for InferencePipelines: over the HTTP API each
request would resume the session, consume the whole backlog and return one message.

After the first subscription a run never waits for the broker. While the connection is
down the outputs repeat the last message received with `error_status` True and `is_new`
False, so a downstream step can gate on `error_status`; messages buffered before the drop
are still returned first, and with `client_id` the messages the broker queued meanwhile
follow after the reconnect. A connection refused on a reconnect is reported the same way,
with the broker's reason, until the pipeline is restarted. The log carries one ERROR line
when an outage starts and one INFO line with its duration when a live message arrives
again.

One block instance subscribes on a single broker connection: changing host, port,
credentials, timeout, topic, QoS, read mode, client id or the TLS settings between runs
is rejected as a configuration error. While the broker is unreachable every run waits up to `timeout`
for the background reconnect before reporting the failure.

The block is not available on the Roboflow hosted platform (`GCP_SERVERLESS` or
`LAMBDA`): every run fails there without opening a connection. It is intended for
self-hosted inference servers and InferencePipelines that can reach the broker.

The server operator may restrict which brokers the MQTT blocks connect to with
`MQTT_WORKFLOWS_BLOCKS_WHITELISTED_HOSTS` (an allowlist of `host[:port]` entries) and
`MQTT_WORKFLOWS_BLOCKS_ALLOW_USER_PROVIDED_HOST` (when False, the workflow's host and
port are ignored and the first allowlist entry is used); a run the policy forbids is
reported in the outputs. The connection uses a 15 s keepalive, so a broker that
disappears without closing the connection is noticed within about 25 s.

A broker that refuses the connection (bad user name or password, not authorised, a
rejected client id or an unacceptable protocol version) is reported in the outputs with
the broker's reason, and the block stops reconnecting: retrying with the same credentials
cannot succeed and would only trip the broker's authentication rate limiting. Fix the
configuration and restart the pipeline. A broker answering "unavailable" is a passing
condition, so the client keeps retrying in the background.

Set `encryption` to `tls` to encrypt the connection and verify the broker's certificate
against the system trust store; the port is not switched automatically, so set it to the
broker's TLS port (usually 8883). A broker signed by a private CA needs
`ca_certificate_path`, a PEM bundle on the machine running inference, which requires
local file system access for Workflow blocks
(`ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE=True`). Certificate verification cannot be
disabled.

Outputs:
    - value (str): Raw message payload decoded as UTF-8, or None when nothing was
                   received yet.
    - payload (dict): The payload parsed as a JSON object, or {} when the payload is
                      not a JSON object.
    - topic (str): The topic the message arrived on (useful with wildcard filters).
    - is_new (bool): True if this run surfaced a message the previous run did not.
    - error_status (bool): True if the run failed (invalid configuration, refused
                           subscription, undecodable payload) or the broker connection
                           is down or was refused after a healthy start; in the latter
                           cases value, payload and topic repeat the last message
                           received.
    - error_message (str): Details of the failure, or None.

Failures are returned in the outputs and logged; the workflow keeps running.
"""

SUBSCRIPTION_REFUSED_QOS = 0x80
READER_CONNECTION_INPUTS = "username, password and client_id"
NOT_CONNECTED_WITHIN_TIMEOUT = (
    "MQTT broker not connected (connection was not established within timeout); "
    "the client keeps retrying in the background. Raise 'timeout' if the "
    "broker needs longer to connect."
)
NOT_CONNECTED_REPEATING_LAST = (
    "MQTT broker not connected; the client keeps reconnecting in the background and "
    "the outputs repeat the last message received."
)
NOT_RESUBSCRIBED_REPEATING_LAST = (
    "MQTT subscription not re-established yet after a reconnect; the outputs repeat "
    "the last message received."
)
TLS_RELEVANT = {"encryption": {"values": ["tls"], "required": True}}
LATEST_BUFFER_SIZE = 1
SEQUENTIAL_BUFFER_SIZE = 1000

ReceivedMessage = Tuple[str, bytes, bool]


class MQTTReaderState:
    """State shared between paho's network thread and the engine thread.

    Passed to paho as ``userdata`` so the callbacks never hold a reference to the
    block itself (a bound-method callback would form a reference cycle and delay
    the block's cleanup).

    Attributes:
        connected: Set once the broker accepted the session; cleared on disconnect.
        connack: Set on every CONNACK, accepted or refused, so the first run can
            stop waiting as soon as the broker answered; cleared on a transport
            disconnect, kept while a refusal is recorded.
        refused_code: The CONNACK return code of the last refusal, or None after an
            accepted CONNACK. A permanent code (see ``PERMANENT_CONNACK_CODES``) means
            the network loop was stopped and every run reports the refusal.
        subscribed: Set once the broker granted the subscription; cleared on disconnect.
        subscribe_failed: True when the broker refused the subscription.
        topic: Topic filter to subscribe to on every (re)connect.
        qos: Subscription QoS requested from the broker.
        messages: Buffer of ``(topic, payload, retain)`` tuples appended by the
            network thread and consumed by ``run()``; ``maxlen`` drops the oldest.
        message_received: Set once the first message was buffered; lets the first
            run wait for a retained message the broker sends right after SUBACK.
    """

    def __init__(self, topic: str, qos: int, buffer_size: int):
        self.connected = threading.Event()
        self.connack = threading.Event()
        self.refused_code: Optional[int] = None
        self.subscribed = threading.Event()
        self.message_received = threading.Event()
        self.subscribe_failed = False
        self.topic = topic
        self.qos = qos
        self.messages: Deque[ReceivedMessage] = deque(maxlen=buffer_size)


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


def mqtt_on_connect(
    client, state: MQTTReaderState, flags, reason_code, properties=None
):
    # paho invokes on_connect for accepted and rejected CONNACK alike;
    # only reason_code 0 means an established MQTT session. The outcome is
    # recorded before `connack` wakes a waiting run, so the run never reads a
    # half-updated state
    if reason_code != 0:
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
        return

    state.refused_code = None
    logger.info(
        "MQTT client connected (session present: %s)",
        bool(flags.get("session present")),
    )
    state.connected.set()
    state.connack.set()
    # a clean-session broker forgets subscriptions on every reconnect, so the
    # subscription is (re)established here rather than once after connect(); on
    # a persistent session the repeated SUBSCRIBE is idempotent
    state.subscribe_failed = False
    state.subscribed.clear()
    try:
        client.subscribe(state.topic, qos=state.qos)
    except Exception as e:
        # paho rejects e.g. an empty topic locally; report it through the
        # refused path instead of leaving the run waiting for a SUBACK
        logger.error("MQTT subscribe to topic %r failed: %s", state.topic, e)
        state.subscribe_failed = True


def mqtt_on_connect_fail(client, state: MQTTReaderState):
    # paho 1.6.1 invokes this callback with exactly (client, userdata)
    logger.error("MQTT client failed to establish connection with broker")
    state.connected.clear()


def mqtt_on_subscribe(client, state: MQTTReaderState, mid, granted_qos):
    if any(qos == SUBSCRIPTION_REFUSED_QOS for qos in granted_qos):
        logger.error("MQTT broker refused subscription to topic %s", state.topic)
        state.subscribe_failed = True
        state.subscribed.clear()
        return

    logger.info("MQTT client subscribed to topic %s", state.topic)
    state.subscribe_failed = False
    state.subscribed.set()


def mqtt_on_message(client, state: MQTTReaderState, message):
    # the only work done on the network thread: stash the raw message;
    # decoding and parsing happen in run(), where a failure is reported
    # instead of ending the network loop
    state.messages.append((message.topic, message.payload, bool(message.retain)))
    state.message_received.set()


def mqtt_on_disconnect(client, state: MQTTReaderState, reason_code, properties=None):
    logger.info("MQTT client disconnected, result code %s", reason_code)
    state.connected.clear()
    state.subscribed.clear()
    if state.refused_code is None:
        # a transport drop: the next run waits for the reconnect's CONNACK; after
        # a refusal the answer stays visible so run() reports it without waiting
        state.connack.clear()


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "MQTT Reader",
            "version": "v1",
            "short_description": "Reads a message from an MQTT topic on every run: the newest one, or the next unread one.",
            "long_description": LONG_DESCRIPTION,
            "license": "Roboflow Enterprise License",
            "block_type": "transformation",
            "ui_manifest": {
                "section": "industrial",
                "icon": "fal fa-network-wired",
                "blockPriority": 11,
                "enterprise_only": True,
                "local_only": True,
            },
        }
    )
    type: Literal["roboflow_enterprise/mqtt_reader@v1"]
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
        description="MQTT topic filter to subscribe to. Single-level (`+`) and "
        "multi-level (`#`) wildcards are allowed; the `topic` output tells which "
        "topic each message arrived on.",
        examples=["plc/state", "sensors/+/temperature", "$inputs.mqtt_topic"],
    )
    read_mode: Literal["latest", "sequential"] = Field(
        default="latest",
        description="What a run returns from the messages received since the previous "
        "run: `latest` returns the newest and skips older ones; `sequential` returns "
        "the next unread message, one per run, in arrival order.",
        examples=["latest", "sequential"],
        json_schema_extra={
            "values_metadata": {
                "latest": {
                    "name": "Latest",
                    "description": "Return the newest message; skip anything older.",
                },
                "sequential": {
                    "name": "Sequential",
                    "description": "Return the next unread message, one per run, in arrival order.",
                },
            },
        },
    )
    qos: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=0,
        description="Quality of Service level requested for the subscription (0, 1 or 2). "
        "When client_id is set, qos must be 1 or 2: the broker only queues QoS 1 and 2 "
        "messages for a persistent session, and a run with client_id and qos 0 reports "
        "an error. The publisher must also publish at QoS 1 or higher for its messages "
        "to be queued.",
        examples=[0, 1, 2],
    )
    client_id: Optional[Union[Selector(kind=[STRING_KIND]), str]] = Field(
        default=None,
        description="Leave empty for a fresh session on every connection (a "
        "broker-generated id). Set a stable name that is unique on the broker, for "
        "example the camera or pipeline name from a workflow input, to keep a "
        "persistent session: the broker then remembers the subscription and queues "
        "QoS 1/2 messages while this block is away and delivers them when it "
        "reconnects. Two connections with the same id disconnect each other.",
        examples=["$inputs.pipeline_name", "line1-camera3"],
    )
    username: Optional[Union[Selector(kind=[STRING_KIND]), str]] = Field(
        default=None,
        description="Username for MQTT broker authentication.",
        examples=["$inputs.mqtt_username"],
    )
    password: Optional[Union[Selector(kind=[SECRET_KIND]), str]] = Field(
        default=None,
        description="Password for MQTT broker authentication. Requires username.",
        examples=["$inputs.mqtt_password"],
        json_schema_extra={"private": True},
    )
    timeout: Union[
        Annotated[float, Field(gt=0, allow_inf_nan=False)],
        Selector(kind=[FLOAT_KIND]),
    ] = Field(
        default=0.5,
        description="Timeout in seconds for the first connection: the TCP connect, the "
        "broker's connection acknowledgement (accepted or refused) and the subscription "
        "acknowledgement. Later runs never wait: while the connection is down they return "
        "the last message with error_status set, and the client reconnects in the "
        "background. Must be a finite number greater than 0.",
        examples=[0.5],
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
            OutputDefinition(name="value", kind=[STRING_KIND]),
            OutputDefinition(name="payload", kind=[DICTIONARY_KIND]),
            OutputDefinition(name="topic", kind=[STRING_KIND]),
            OutputDefinition(name="is_new", kind=[BOOLEAN_KIND]),
            OutputDefinition(name="error_status", kind=[BOOLEAN_KIND]),
            OutputDefinition(name="error_message", kind=[STRING_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.4.0,<2.0.0"

    @classmethod
    def get_restrictions(cls) -> List[RuntimeRestriction]:
        """Return the legacy editor restrictions of this block.

        Returns:
            Fresh restrictions for the workflow editor on every call. The same
            entries back ``get_actual_restrictions()``, so the codes match.
        """
        return [
            RuntimeRestriction(
                code="unavailable_on_hosted_platform",
                severity=Severity.HARD,
                note=(
                    "On the Roboflow hosted platform every run returns "
                    "`error_status=true` with no message, and no MQTT connection is "
                    "opened from the hosted platform."
                ),
                applies_to_runtimes=[Runtime.HOSTED_SERVERLESS],
            ),
            RuntimeRestriction(
                code="connection_and_state_rebuilt_per_request",
                severity=Severity.SOFT,
                note=(
                    "The subscription and message buffer live in this block "
                    "instance. Over HTTP every request builds a fresh instance, so "
                    "only a retained message can be returned; with client_id set, "
                    "each request resumes the persistent session, consumes its whole "
                    "backlog and returns one message, and concurrent requests with "
                    "the same id disconnect each other. Use an InferencePipeline "
                    "for a live subscription or a persistent session."
                ),
                applies_to_runtimes=[
                    Runtime.SELF_HOSTED_CPU,
                    Runtime.SELF_HOSTED_GPU,
                    Runtime.DEDICATED_DEPLOYMENT,
                ],
                applies_to_input_modes=[RuntimeInputMode.IMAGE],
            ),
        ]

    def discover_work_operations(self) -> List[WorkOperation]:
        """Return the kinds of work a run of this step performs.

        Returns:
            Broker I/O, plus the messages buffered in process memory between
            runs; the last message is returned again (with ``is_new=False``)
            when nothing new arrived.
        """
        return [WorkOperation.EXTERNAL_REQUEST, WorkOperation.TEMPORAL_BUFFERING]

    def get_actual_restrictions(
        self, *, ignore_environment_restrictions: bool = False
    ) -> Discovery[RuntimeRestriction]:
        """Return the restrictions of this step on a target deployment.

        Args:
            ignore_environment_restrictions: If True, return the portable view
                without evaluating configuration predicates against this host.

        Returns:
            Complete discovery of the ``get_restrictions()`` entries. Both
            apply unconditionally: no manifest field switches either on, and
            the hosted-platform condition is the runtime axis, never this
            host's ``GCP_SERVERLESS`` / ``LAMBDA`` flags.
        """
        restrictions = actual_restrictions_of(
            declared=self.get_restrictions(),
            node_id=f"$steps.{getattr(self, 'name', '')}",
            ignore_environment_restrictions=ignore_environment_restrictions,
        )

        return restrictions

    def discover_dependent_resources(self) -> List[DependentResource]:
        """Return the models and projects this step pulls.

        Returns:
            An empty list: the block talks to an MQTT broker only.
        """
        return []


class MQTTReaderBlockV1(WorkflowBlock):
    def __init__(self, allow_access_to_file_system: bool = False):
        # gates `ca_certificate_path`, a server-side path chosen by the workflow;
        # False by default so a hand-constructed block is safe
        self._allow_access_to_file_system = allow_access_to_file_system
        self._client: Optional[mqtt.Client] = None
        self._state: Optional[MQTTReaderState] = None
        self._connection_identity: Optional[Tuple] = None
        # outputs of the last surfaced message, repeated with is_new=False
        self._last: Optional[Dict[str, Any]] = None
        # the first read after subscribing waits for a retained message, which
        # the broker sends right after SUBACK as a separate packet
        self._awaiting_retained = False
        # readiness waits are a cold-start concern: once the first run reached
        # SUBACK, later runs never block on the broker
        self._subscribed_once = False
        # an outage is bracketed in the log: one ERROR line when a run first
        # observes it, one INFO line with duration and run count on recovery
        self._outage_started_at: Optional[float] = None
        self._outage_runs = 0
        # the resolved broker address, kept for the log lines
        self._host: Optional[str] = None
        self._port: Optional[int] = None
        self._lifecycle_lock = threading.Lock()

    def close(self) -> None:
        with self._lifecycle_lock:
            client = self._client
            state = self._state
            if client is None:
                return
            self._client = None
            self._state = None
            self._connection_identity = None
            self._last = None
            self._awaiting_retained = False
            self._subscribed_once = False
            self._outage_started_at = None
            self._outage_runs = 0
            self._host = None
            self._port = None
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
                    # only after the join can no callback re-set the events
                    if state is not None:
                        state.connected.clear()
                        state.subscribed.clear()

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
        return ["allow_access_to_file_system"]

    def run(
        self,
        host: str,
        port: int,
        topic: str,
        read_mode: str = "latest",
        qos: int = 0,
        username: Optional[str] = None,
        password: Optional[str] = None,
        timeout: float = 0.5,
        encryption: str = "none",
        ca_certificate_path: Optional[str] = None,
        client_id: Optional[str] = None,
    ) -> BlockResult:
        if GCP_SERVERLESS or LAMBDA:
            # never open an outbound broker connection from hosted workers
            return self._handle_failure(
                "MQTT Reader is not available on the Roboflow hosted platform"
            )
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
                "number of seconds within the platform limit."
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
                f"Invalid port: {port!r}. Port must be an integer between 1 and 65535."
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
            return self._handle_failure(f"Invalid qos: {qos!r}. QoS must be 0, 1 or 2.")
        qos = qos_number
        if read_mode not in ("latest", "sequential"):
            return self._handle_failure(
                f"Invalid read_mode: {read_mode!r}. Must be 'latest' or 'sequential'."
            )
        if not isinstance(topic, str) or not topic:
            return self._handle_failure(
                f"Invalid topic: {topic!r}. Topic must be a non-empty string."
            )
        if password is not None and username is None:
            return self._handle_failure(
                "Password provided without username. Set username to enable MQTT authentication."
            )
        if encryption not in ("none", "tls"):
            return self._handle_failure(
                f"Invalid encryption: {encryption!r}. Must be 'none' or 'tls'."
            )
        # an empty editor field is "unset"; anything but a string is a wiring error
        if ca_certificate_path is not None and not isinstance(ca_certificate_path, str):
            return self._handle_failure(
                f"Invalid ca_certificate_path: {ca_certificate_path!r}. Must be a path."
            )
        ca_certificate_path = ca_certificate_path or None
        try:
            # blank means "no persistent session"; the presence of the id is the switch
            client_id = normalise_client_id(client_id)
        except ConfigurationError as e:
            return self._handle_failure(str(e))
        if client_id is not None and qos == 0:
            # the broker queues only QoS 1 and 2 for an offline session: a
            # persistent session at QoS 0 would look right and receive nothing
            return self._handle_failure(
                "client_id needs qos 1 or 2: the broker only queues QoS 1 and 2 "
                "messages for a persistent session."
            )
        try:
            # the operator's broker policy; the RESOLVED address is what the
            # block connects to and what the connection identity is built from
            host, port = resolve_broker_address(
                host, port, log_override=self._client is None
            )
        except ConfigurationError as e:
            return self._handle_failure(str(e))
        with self._lifecycle_lock:
            return self._connect_and_read(
                host=host,
                port=port,
                topic=topic,
                read_mode=read_mode,
                qos=qos,
                username=username,
                password=password,
                timeout=timeout,
                encryption=encryption,
                ca_certificate_path=ca_certificate_path,
                client_id=client_id,
            )

    def _connect_and_read(
        self,
        host: str,
        port: int,
        topic: str,
        read_mode: str,
        qos: int,
        username: Optional[str],
        password: Optional[str],
        timeout: float,
        encryption: str,
        ca_certificate_path: Optional[str],
        client_id: Optional[str],
    ) -> BlockResult:
        connection_identity = (
            host,
            port,
            username,
            password,
            timeout,
            topic,
            qos,
            read_mode,
            encryption,
            ca_certificate_path,
            client_id,
        )
        if self._client is None:
            buffer_size = (
                LATEST_BUFFER_SIZE if read_mode == "latest" else SEQUENTIAL_BUFFER_SIZE
            )
            state = MQTTReaderState(topic=topic, qos=qos, buffer_size=buffer_size)
            client = None
            try:
                if client_id is None:
                    # broker-generated id, clean session: nothing outlives the connection
                    client = mqtt.Client(userdata=state)
                else:
                    # a persistent session: the broker keeps the subscription and
                    # queues QoS 1/2 messages while this id is away
                    client = mqtt.Client(
                        client_id=client_id, clean_session=False, userdata=state
                    )
                # a raising callback would end the network thread; log instead
                client.suppress_exceptions = True
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
                client.on_subscribe = mqtt_on_subscribe
                client.on_message = mqtt_on_message
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
                client.connect(host, port, keepalive=MQTT_KEEPALIVE_SECONDS)
                client.loop_start()
            except ConfigurationError as e:
                # TLS setup refused before any socket was opened: nothing kept
                return self._handle_failure(str(e))
            except OSError as e:
                # broker unreachable: nothing is kept and no loop was started,
                # so a failed one-shot run leaves no background thread behind;
                # the next run on this instance retries with a fresh client
                return self._handle_failure(
                    _not_connected_message(e, encryption=encryption)
                )
            except Exception as e:
                if client is not None:
                    try:
                        client.loop_stop()
                    except Exception:
                        pass
                return self._handle_failure(f"Failed to initialize MQTT client: {e}")
            self._client = client
            self._state = state
            self._connection_identity = connection_identity
            self._awaiting_retained = True
            self._host, self._port = host, port
        elif connection_identity != self._connection_identity:
            return self._handle_failure(
                "MQTT connection parameters (host, port, credentials, timeout, topic, "
                "qos, read_mode, encryption, ca_certificate_path or client_id) changed "
                "between runs; this block subscribes only with the configuration of its "
                "first run."
            )
        state = self._state
        if not self._subscribed_once:
            # cold start: bounded waits for the broker's answers, so the first
            # run of a pipeline gets a message or a precise failure
            failure = self._wait_until_subscribed(state, timeout)
            if failure is not None and not state.messages:
                return failure
            if failure is None:
                self._subscribed_once = True
                if self._awaiting_retained:
                    # one-off, bounded: a retained message (or nothing) arrives
                    # right after SUBACK; without this wait the first run would
                    # race it
                    self._awaiting_retained = False
                    state.message_received.wait(timeout=timeout)
        # never wait once subscribed; always drain first, because messages
        # received before a drop or a refusal are real whatever happened to the
        # connection afterwards (a persistent session's backlog even lands
        # before the SUBACK that may refuse the subscription)
        try:
            received_topic, payload, retained = state.messages.popleft()
        except IndexError:
            if state.subscribe_failed:
                # a known refusal is permanent for this configuration
                return self._handle_failure(
                    f"MQTT broker refused subscription to topic {state.topic!r}."
                )
            if state.refused_code in PERMANENT_CONNACK_CODES:
                # a reconnect refused with 1/2/4/5 stopped the loop: no fresh
                # data can arrive until the pipeline is restarted
                return self._repeat_last_with_error(
                    connection_refused_message(
                        state.refused_code, block_inputs=READER_CONNECTION_INPUTS
                    )
                )
            if not state.connected.is_set():
                return self._repeat_last_with_error(
                    connection_refused_message(
                        state.refused_code, block_inputs=READER_CONNECTION_INPUTS
                    )
                    if state.refused_code == TRANSIENT_CONNACK_CODE
                    else NOT_CONNECTED_REPEATING_LAST
                )
            if not state.subscribed.is_set():
                return self._repeat_last_with_error(NOT_RESUBSCRIBED_REPEATING_LAST)
            # nothing new since the previous run
            return self._repeat_last()
        # a live message ends an outage, a retained duplicate included
        self._log_recovery_if_needed()
        try:
            value = payload.decode("utf-8")
        except UnicodeDecodeError:
            return self._handle_failure(
                f"Message received on topic {received_topic!r} is not valid UTF-8."
            )
        parsed_payload: Dict[str, Any] = {}
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                parsed_payload = parsed
        except ValueError:
            pass
        if (
            retained
            and self._last is not None
            and self._last["topic"] == received_topic
            and self._last["value"] == value
        ):
            # the broker re-sends the retained message on every resubscribe
            # (reconnect); an unchanged copy is not a new message
            return self._repeat_last()
        self._last = {
            "value": value,
            "payload": parsed_payload,
            "topic": received_topic,
        }
        return {
            **self._last,
            "is_new": True,
            "error_status": False,
            "error_message": None,
        }

    def _wait_until_subscribed(
        self, state: MQTTReaderState, timeout: float
    ) -> Optional[BlockResult]:
        """The first run's readiness waits; returns the failure result or None."""
        # a permanent refusal stopped the network loop: report it on every run
        # without waiting and without touching the broker again
        if state.refused_code in PERMANENT_CONNACK_CODES:
            return self._handle_failure(
                connection_refused_message(
                    state.refused_code, block_inputs=READER_CONNECTION_INPUTS
                )
            )
        # the background loop owns (re)connecting and (re)subscribing; runs
        # only wait for readiness. The wait ends on any CONNACK, so a refusal is
        # reported as soon as the broker answers instead of after the timeout
        if not state.connack.wait(timeout=timeout):
            return self._handle_failure(NOT_CONNECTED_WITHIN_TIMEOUT)
        if state.refused_code in PERMANENT_CONNACK_CODES:
            return self._handle_failure(
                connection_refused_message(
                    state.refused_code, block_inputs=READER_CONNECTION_INPUTS
                )
            )
        if not state.connected.wait(timeout=timeout):
            if state.refused_code == TRANSIENT_CONNACK_CODE:
                return self._handle_failure(
                    connection_refused_message(
                        state.refused_code, block_inputs=READER_CONNECTION_INPUTS
                    )
                )
            return self._handle_failure(NOT_CONNECTED_WITHIN_TIMEOUT)
        # a known refusal is permanent for this configuration: report it
        # without paying the acknowledgement wait on every run
        if state.subscribe_failed or (
            not state.subscribed.wait(timeout=timeout) and state.subscribe_failed
        ):
            return self._handle_failure(
                f"MQTT broker refused subscription to topic {state.topic!r}."
            )
        if not state.subscribed.is_set():
            return self._handle_failure(
                "MQTT subscription was not acknowledged within timeout. Raise "
                "'timeout' if the broker needs longer to respond."
            )
        return None

    def _repeat_last(self) -> BlockResult:
        if self._last is None:
            return self._empty_result()
        return {
            **self._last,
            "is_new": False,
            "error_status": False,
            "error_message": None,
        }

    def _repeat_last_with_error(self, message: str) -> BlockResult:
        """The outputs while no fresh data can arrive: the last message with the
        error fields set, or the empty failure result when nothing was received yet.

        The first call of an outage logs one ERROR line; later calls log nothing
        (a video pipeline would otherwise write one line per frame).
        """
        if self._outage_started_at is None:
            self._outage_started_at = time.monotonic()
            self._outage_runs = 0
            # host, port and topic identify the block instance in a shared log
            logger.error(
                "MQTT Reader %s:%s %r: %s",
                self._host,
                self._port,
                self._state.topic if self._state is not None else None,
                message,
            )
        self._outage_runs += 1
        if self._last is None:
            return {
                "value": None,
                "payload": {},
                "topic": None,
                "is_new": False,
                "error_status": True,
                "error_message": message,
            }
        return {
            **self._last,
            "is_new": False,
            "error_status": True,
            "error_message": message,
        }

    def _log_recovery_if_needed(self) -> None:
        """Close the outage bracket in the log once a live message arrived again."""
        if self._outage_started_at is None:
            return
        logger.info(
            "MQTT Reader %s:%s %r: connection restored after %.1f s; %d runs returned "
            "the last message received",
            self._host,
            self._port,
            self._state.topic if self._state is not None else None,
            time.monotonic() - self._outage_started_at,
            self._outage_runs,
        )
        self._outage_started_at = None
        self._outage_runs = 0

    @staticmethod
    def _empty_result() -> BlockResult:
        return {
            "value": None,
            "payload": {},
            "topic": None,
            "is_new": False,
            "error_status": False,
            "error_message": None,
        }

    def _handle_failure(self, message: str) -> BlockResult:
        logger.error("MQTT Reader failure: %s", message)
        return {
            "value": None,
            "payload": {},
            "topic": None,
            "is_new": False,
            "error_status": True,
            "error_message": message,
        }

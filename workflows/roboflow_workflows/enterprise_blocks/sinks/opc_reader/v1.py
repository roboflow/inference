import asyncio
import copy
import logging
import math
import random
import threading
import time
import weakref
from dataclasses import dataclass
from datetime import datetime
from typing import Any, List, Literal, Optional, Tuple, Type, Union
from urllib.parse import urlsplit

from asyncua import Client, Node, ua
from asyncua.ua.uaerrors import (
    BadAttributeIdInvalid,
    BadIdentityTokenInvalid,
    BadIdentityTokenRejected,
    BadNodeIdInvalid,
    BadNodeIdUnknown,
    BadNoMatch,
    BadNotReadable,
    BadUserAccessDenied,
)
from pydantic import ConfigDict, Field, model_validator
from typing_extensions import Annotated

# Bind by literal name to the shared "inference" logger tree so server
# handlers / filters / propagation still apply when installed, without
# importing the server logger module.
logger = logging.getLogger("inference")
from roboflow_workflows.enterprise_blocks.sinks.opc_writer.v1 import (
    SESSION_EXHAUSTION_BACKOFF_JITTER_SECONDS,
    SESSION_EXHAUSTION_BACKOFF_SECONDS,
    SESSION_EXHAUSTION_ERRORS,
    SESSION_TIMEOUT_SECONDS,
    SHUTDOWN_CALL_TIMEOUT_SECONDS,
)
from roboflow_workflows.environment import GCP_SERVERLESS, LAMBDA
from roboflow_workflows.execution_engine.entities.base import OutputDefinition
from roboflow_workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    FLOAT_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
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
    is_workflow_selector,
)

BLOCK_TYPE = "roboflow_enterprise/opc_reader@v1"
LONG_DESCRIPTION = """
The **OPC UA Reader** block subscribes to a variable (tag) on an OPC UA server and returns its
current value on every workflow run, using the
[asyncua](https://github.com/FreeOpcUa/opcua-asyncio) library. A typical use is gating a
video workflow on machine state: Ignition (or a PLC) writes a flag such as "line running"
into the OPC UA server on the edge device, and a Continue If step after this block skips
inference while the flag is off.

### How values are read
The block does not poll the server on every frame. The first run connects, reads the
variable and creates an OPC UA subscription for it; from then on the server pushes every
change and the block keeps the latest value in memory. A run returns that value without
any network round trip, so the block costs the same at 60 fps as at 1 fps.

- The first run waits up to `timeout` seconds for the connection and the initial value.
- `publishing_interval_ms` (default 100) is how often the server is asked to sample the
  variable and send changes. A change therefore reaches the workflow within roughly one
  interval. The server may revise the interval upwards.
- `is_new` is True when the value (or its quality) changed since the previous run, and on
  the first run.

### Node Lookup Mode
The variable is located the same way as in the **OPC UA Writer** block:

- **`hierarchical` (default)**: standard browse-path navigation. Each component of
  `object_name` is prefixed with the namespace index.
  - **Example**: `object_name="Roboflow/Crane_11"`, `variable_name="Running"` resolves
    `0:Objects/2:Roboflow/2:Crane_11/2:Running`
- **`direct`**: a string NodeId `ns={namespace};s={object_name}/{variable_name}`, or
  `ns={namespace};s={object_name}` when `variable_name` is empty.
  - **Example**: `object_name="[Sample_Tags]/Ramp"`, `variable_name="Running"` reads
    `ns=2;s=[Sample_Tags]/Ramp/Running`
  - **Best for**: Ignition and other servers with string NodeIds, including the Roboflow
    edge OPC UA server, whose tags are string NodeIds `Folder.Tag` in its own namespace
    (set `object_name` to `Folder.Tag` and leave `variable_name` empty). Pass that
    namespace by URI (`http://opcua.roboflow.run`) rather than by index: the index is
    1 on the edge server but differs between servers

### Connection loss and errors
After the first run, a run never waits for the server. While the connection is down the
block returns the last value received with `error_status` True and `is_new` False, so a
downstream step can gate on `error_status`, and it reconnects and re-subscribes in the
background. The log carries one ERROR line when an outage starts and one INFO line with
its duration when the subscription is restored.

A lost connection is noticed by the OPC UA client's watchdog, which checks the server
once a second and then waits up to `timeout` for the answer, and by the block's own
check of the subscription, also once a second: after a server crash, runs can keep
returning the last value as healthy for up to `timeout` plus about 2 seconds (about 4
seconds with the defaults). To fail closed on a stale value sooner, lower `timeout`.

- A value whose OPC UA quality is not Good (for example when Ignition lost the PLC it
  mirrors) is returned with `error_status` True and the status name in `error_message`.
- A server that rejects the user name or password is reported on every run and is not
  retried, since repeating the same credentials cannot succeed. Fix them and restart the
  pipeline.
- A namespace, object or variable that cannot be found is reported and retried every
  10 seconds.
- A server that refuses a new session because it is at its session limit is retried after
  roughly 30 seconds (jittered, `OPC_SESSION_EXHAUSTION_BACKOFF_SECONDS`).

### Sessions
The connection uses OPC UA security policy `None`, like the OPC UA Writer: messages are not
signed or encrypted, so keep the server on a trusted network. Credentials go in `user_name`
and `password`; a `url` containing `user:password@` is rejected.

Each block instance holds one OPC UA session of its own (separate from the OPC UA Writer's
pooled sessions). Sessions ask the server for a 60 second timeout
(`OPC_SESSION_TIMEOUT_SECONDS`) so one left behind by a crashed process is reclaimed
quickly. The session is closed when the InferencePipeline running the workflow is
joined after it ends (end of stream, `terminate()` or an error), when the block is closed
or garbage collected, and on a normal interpreter exit.

### Where it runs
The subscription and the latest value live in the block instance. An InferencePipeline
keeps them for its lifetime. Over the HTTP API every request builds a fresh instance, so
each request connects, reads the current value and disconnects, which is fine for testing
but not for high-rate use. The block is not available on the Roboflow hosted platform.

Outputs:
    - value: The latest value of the variable (boolean, number, string, or a list for
             array variables; dates are returned as ISO 8601 strings), or None when
             nothing was received yet. NaN and infinite floats are returned as is;
             over the HTTP API they are serialized as null.
    - is_new (bool): True if the value or its quality changed since the previous run.
    - error_status (bool): True when the run failed, the connection is down, or the
                           value's quality is not Good.
    - error_message (str): Details of the failure, or None.
"""

NODE_LOOKUP_MODES = ("hierarchical", "direct")

# Reconnect pacing after a lost or failed connection: doubles from the initial delay up
# to the maximum, and starts over once a subscription is established again.
RECONNECT_INITIAL_DELAY_SECONDS = 0.5
RECONNECT_MAX_DELAY_SECONDS = 10.0
# A namespace, node or permission problem does not fix itself between attempts; retry
# slowly in case the server's address space changes underneath us.
CONFIGURATION_RETRY_DELAY_SECONDS = 10.0
# How often the supervisor checks asyncua's own connection watchdog and publish loop.
CONNECTION_CHECK_INTERVAL_SECONDS = 1.0

# The server refused the user identity: retrying with the same credentials cannot succeed
# and may trip the server's lockout, so the block stops connecting.
AUTHENTICATION_ERRORS = (
    BadUserAccessDenied,
    BadIdentityTokenInvalid,
    BadIdentityTokenRejected,
)
# Raised while resolving, reading or subscribing to the node on an otherwise healthy
# session: the configuration (or the server's address space) is wrong.
NODE_CONFIGURATION_ERRORS = (
    BadNoMatch,
    BadNodeIdUnknown,
    BadNodeIdInvalid,
    BadNotReadable,
    BadAttributeIdInvalid,
    BadUserAccessDenied,
)

# A read answered with one of these statuses means the node cannot be read at all;
# any other Bad or Uncertain status is the quality of a real value.
NODE_CONFIGURATION_STATUS_CODES = frozenset(
    error.code for error in NODE_CONFIGURATION_ERRORS
)
# Below the shutdown budget, so a hung disconnect is abandoned (and its socket closed)
# before stop() gives up on the event loop.
DISCONNECT_TIMEOUT_SECONDS = SHUTDOWN_CALL_TIMEOUT_SECONDS * 0.6
# How long stop() waits for the loop thread to exit once the loop is told to stop.
LOOP_STOP_JOIN_SECONDS = 1.0

UNPARSEABLE_URL = "<unparseable url>"
EXPECTED_URL_FORM = "Expected opc.tcp://host[:port][/path], without credentials."

# Bounds for the manifest and for selector-resolved values. A larger `timeout` would let
# one workflow pin a request thread against an unreachable server; a shorter publishing
# interval than this is below what servers sample at anyway.
MAX_TIMEOUT_SECONDS = 60.0
MIN_PUBLISHING_INTERVAL_MS = 10.0
MAX_PUBLISHING_INTERVAL_MS = 60_000.0

NOT_CONNECTED_WITHIN_TIMEOUT = (
    "OPC UA server not connected within timeout; the block keeps connecting in the "
    "background. Raise 'timeout' if the server needs longer to connect."
)


class NodeConfigurationError(Exception):
    """The namespace, object or variable could not be resolved on the server."""


@dataclass(frozen=True)
class OPCUAReaderConfig:
    """What one block instance subscribes to; fixed by its first run."""

    url: str
    namespace: str
    user_name: Optional[str]
    password: Optional[str]
    object_name: str
    variable_name: str
    node_lookup_mode: str
    timeout: float
    publishing_interval_ms: float


class OPCUAReaderState:
    """Latest value and connection status shared by the subscription thread and `run()`.

    Attributes:
        settled: Set once the first connection attempt finished, successfully or not, so
            the first run can stop waiting as soon as there is an answer.
        connected: True while a subscription is established.
        has_value: True once a value was received.
        value: The latest value, converted to plain Python types.
        status_name: Name of the OPC UA status code of the latest value, e.g. "Good".
        status_is_good: Whether that status code is Good.
        sequence: Incremented whenever the value or its status changes; `run()` compares
            it with the sequence it returned last to compute `is_new`.
        error: The last connection or configuration failure, or None while connected.
        permanent: True when the block stopped connecting (refused credentials).
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.settled = threading.Event()
        self.connected = False
        self.has_value = False
        self.value: Any = None
        self.status_name: Optional[str] = None
        self.status_is_good = False
        self.sequence = 0
        self.error: Optional[str] = None
        self.permanent = False

    def record_value(self, data_value: ua.DataValue) -> None:
        """Store a value delivered by a read or a data change notification.

        Args:
            data_value: The OPC UA DataValue, carrying the value and its status code.
        """
        value = _to_output_value(
            data_value.Value.Value if data_value.Value is not None else None
        )
        status = data_value.StatusCode or ua.StatusCode()
        with self.lock:
            # The server re-sends the current value whenever a subscription is created,
            # including after a reconnect: an unchanged value is not a new one.
            changed = (
                not self.has_value
                or not _same_value(value, self.value)
                or status.name != self.status_name
            )
            if changed:
                self.sequence += 1
            self.has_value = True
            self.value = value
            self.status_name = status.name
            self.status_is_good = status.is_good()

    def record_connected(self) -> None:
        """Mark the subscription as established and clear the last failure."""
        with self.lock:
            self.connected = True
            self.error = None
        self.settled.set()

    def record_failure(self, message: str, *, permanent: bool = False) -> None:
        """Mark the connection as down.

        Args:
            message: What went wrong, returned as `error_message` by later runs.
            permanent: True when the block stopped connecting for good.
        """
        with self.lock:
            self.connected = False
            self.error = message
            self.permanent = permanent
        self.settled.set()


def _display_url(url: str) -> str:
    """Scheme, host and port of `url` for logs and error outputs.

    Credentials and paths are never shown, and a URL that does not parse cleanly is
    replaced by a placeholder: a malformed URL can carry a password in any position.
    """
    try:
        parsed_url = urlsplit(url)
        host = parsed_url.hostname
        port = parsed_url.port
    except ValueError:
        return UNPARSEABLE_URL
    if "@" in url or not parsed_url.scheme or not host:
        return UNPARSEABLE_URL
    if ":" in host:
        host = f"[{host}]"
    if port is None:
        return f"{parsed_url.scheme}://{host}"
    return f"{parsed_url.scheme}://{host}:{port}"


def _url_problem(url: str) -> Optional[str]:
    """Why `url` is not an acceptable OPC UA endpoint, or None.

    The messages are fixed: neither the URL nor a parser error (which quotes the part
    it failed on) is echoed, since a malformed URL can carry a password anywhere.
    """
    if "@" in url:
        # the url field is not private: credentials there would be stored and shown
        # with the workflow definition. An "@" anywhere counts, since a password with
        # an unescaped "/" hides it from the parser's view of the authority.
        return (
            "Invalid url: credentials in the url are not accepted. Set user_name and "
            "password instead."
        )
    try:
        parsed_url = urlsplit(url)
        # raises for a port that is not a number in range
        parsed_url.port
    except ValueError:
        return f"Invalid url: it could not be parsed. {EXPECTED_URL_FORM}"
    if parsed_url.scheme != "opc.tcp" or not parsed_url.hostname:
        return f"Invalid url: {EXPECTED_URL_FORM}"
    return None


def _same_value(left: Any, right: Any) -> bool:
    """Equality that treats NaN as equal to NaN, so an unchanged NaN is not new."""
    if isinstance(left, float) and isinstance(right, float):
        return left == right or (math.isnan(left) and math.isnan(right))
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _same_value(a, b) for a, b in zip(left, right)
        )
    return type(left) is type(right) and left == right


def _to_output_value(value: Any) -> Any:
    """Convert an OPC UA variant value into a plain, serialisable Python value."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, ua.LocalizedText):
        return value.Text
    if isinstance(value, (list, tuple)):
        return [_to_output_value(item) for item in value]
    return str(value)


async def _resolve_node(client: Client, config: OPCUAReaderConfig) -> Tuple[Node, str]:
    """Locate the variable the same way the OPC UA Writer does; returns it and its path."""
    try:
        if config.namespace.isdigit():
            namespace_index = int(config.namespace)
        else:
            namespace_index = await client.get_namespace_index(config.namespace)
    except Exception as exc:
        try:
            namespaces = await client.get_namespace_array()
        except Exception:
            namespaces = ["<unable to fetch namespaces>"]
        raise NodeConfigurationError(
            f"WRONG NAMESPACE ERROR: {exc}. Available namespaces: {namespaces}"
        ) from exc

    if config.node_lookup_mode == "direct":
        if config.variable_name:
            node_id = (
                f"ns={namespace_index};s={config.object_name}/{config.variable_name}"
            )
        else:
            node_id = f"ns={namespace_index};s={config.object_name}"
        try:
            node = client.get_node(node_id)
        except Exception as exc:
            raise NodeConfigurationError(
                f"WRONG OBJECT OR PROPERTY ERROR: Invalid NodeId '{node_id}'. Error: {exc}"
            ) from exc

        return node, node_id

    object_path = "/".join(
        f"{namespace_index}:{component}" for component in config.object_name.split("/")
    )
    node_path = f"0:Objects/{object_path}/{namespace_index}:{config.variable_name}"
    try:
        node = await client.nodes.root.get_child(node_path)
    except BadNoMatch as exc:
        raise NodeConfigurationError(
            f"WRONG OBJECT OR PROPERTY ERROR: Could not find node at hierarchical path "
            f"'{node_path}'. Error: {exc}"
        ) from exc

    return node, node_path


class _DataChangeHandler:
    """Receives the subscription's notifications on the subscription thread."""

    def __init__(self, state: OPCUAReaderState, subscription_lost: asyncio.Event):
        self._state = state
        self._subscription_lost = subscription_lost
        self.lost_status: Optional[str] = None
        # Set by the subscription's first notification, which carries the value the
        # monitored item started from.
        self.initial_value_delivered = asyncio.Event()

    def datachange_notification(self, node: Node, val: Any, data: Any) -> None:
        # The full DataValue carries the quality next to the value.
        self._state.record_value(data.monitored_item.Value)
        self.initial_value_delivered.set()

    def status_change_notification(self, status: ua.StatusChangeNotification) -> None:
        # Sent when the server ends the subscription, for example after its lifetime
        # expired: no further changes will arrive on it, so it has to be re-created.
        if status.Status is not None and not status.Status.is_good():
            self.lost_status = status.Status.name
            self._subscription_lost.set()


class _SubscriptionWorker:
    """Owns one OPC UA session and subscription on a private event loop thread.

    Holds no reference to the block, so the block can be garbage collected (and this
    worker stopped by its finalizer) while the thread is still running.
    """

    def __init__(self, config: OPCUAReaderConfig, state: OPCUAReaderState):
        self._config = config
        self._state = state
        self._url = _display_url(config.url)
        self._loop = asyncio.new_event_loop()
        # Daemon: the loop runs forever, and must not keep the interpreter alive.
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="opc-ua-reader", daemon=True
        )
        self._task: Optional[asyncio.Task] = None
        # Set by stop() next to cancelling the task. Cancellation alone is not enough:
        # asyncua's `close_session()` swallows a CancelledError that lands while it
        # awaits its watchdog, after which the task would carry on into its retry sleep.
        self._stop_requested: Optional[asyncio.Event] = None
        # The client of the current attempt, so stop() can close its socket if the task
        # does not wind down within the shutdown budget.
        self._client: Optional[Client] = None
        self._finished = threading.Event()
        self._reconnect_delay = RECONNECT_INITIAL_DELAY_SECONDS
        # When the current subscription was established; the backoff only starts over
        # once a connection has stayed up this long, so a server that accepts and then
        # drops sessions is not re-approached every half second.
        self._connected_at: Optional[float] = None
        self._outage_started_at: Optional[float] = None
        self._subscribed_once = False

    def start(self) -> None:
        self._thread.start()
        self._loop.call_soon_threadsafe(self._start_task)

    def _start_task(self) -> None:
        self._stop_requested = asyncio.Event()
        self._task = self._loop.create_task(self._supervise())
        # A done callback rather than a `finally` in the coroutine: a task cancelled
        # before its first step never runs its body at all.
        self._task.add_done_callback(self._on_supervisor_done)

    def _on_supervisor_done(self, _task: asyncio.Task) -> None:
        self._finished.set()
        # Nothing else runs on this loop. Ending it here lets the thread exit after the
        # worker gave up for good (refused credentials, an unexpected failure), not only
        # after stop(); a retained block would otherwise keep an idle thread.
        self._loop.stop()

    def stop(self) -> None:
        """Close the session and stop the thread, bounded by the shutdown budget."""
        if not self._thread.is_alive():
            # The supervisor already ended the loop (see `_on_supervisor_done`).
            if not self._loop.is_closed():
                self._loop.close()
            return
        deadline = time.monotonic() + SHUTDOWN_CALL_TIMEOUT_SECONDS
        # FIFO with `_start_task`, so the task always exists by the time this runs.
        self._loop.call_soon_threadsafe(self._cancel_task)
        if self._finished.wait(timeout=max(0.0, deadline - time.monotonic())):
            self._loop.call_soon_threadsafe(self._loop.stop)
        else:
            # The task did not wind down in time: close its socket rather than strand
            # it with a stopped loop.
            self._loop.call_soon_threadsafe(self._abort_connection_and_stop)
        self._thread.join(
            timeout=max(LOOP_STOP_JOIN_SECONDS, deadline - time.monotonic())
        )
        if not self._thread.is_alive():
            self._loop.close()

    def _abort_connection_and_stop(self) -> None:
        if self._client is not None:
            try:
                self._client.disconnect_socket()
            except Exception as exc:
                logger.debug(f"OPC UA Reader socket close error (non-fatal): {exc}")
        # Queued behind the transport's own close callbacks, which actually close the
        # socket, so they run before the loop exits.
        self._loop.call_soon(self._loop.stop)

    def _cancel_task(self) -> None:
        if self._task is not None:
            self._stop_requested.set()
            self._task.cancel()

    async def _supervise(self) -> None:
        try:
            while not self._stop_requested.is_set():
                retry_in = await self._connect_and_watch()
                if retry_in is None:
                    return
                try:
                    await asyncio.wait_for(self._stop_requested.wait(), retry_in)
                except asyncio.TimeoutError:
                    pass
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Never leave runs believing a reconnect is still under way.
            self._report_failure(
                f"OPC UA Reader stopped unexpectedly: {type(exc).__name__}: {exc}",
                permanent=True,
            )

    async def _connect_and_watch(self) -> Optional[float]:
        """One connection attempt; returns seconds until the next, or None to give up."""
        config = self._config
        try:
            client = Client(url=config.url, timeout=config.timeout)
            self._client = client
        except Exception as exc:
            # A URL asyncua cannot parse fails the same way on every attempt.
            self._report_failure(
                f"Invalid url {self._url}: the OPC UA client could not use it "
                f"({type(exc).__name__}).",
                permanent=True,
            )
            return None
        # Ask for a short session timeout instead of the asyncua default of 1 hour, so a
        # session orphaned by a crash stops holding a server slot quickly. The server may
        # revise it upwards.
        client.session_timeout = int(SESSION_TIMEOUT_SECONDS * 1000)
        if config.user_name and config.password:
            client.set_user(config.user_name)
            client.set_password(config.password)
        try:
            await client.connect()
        except asyncio.CancelledError:
            # asyncua cleans up a half-open connection on errors, not on cancellation.
            client.disconnect_socket()
            raise
        except AUTHENTICATION_ERRORS as exc:
            self._report_failure(
                f"AUTH ERROR: the OPC UA server rejected the credentials "
                f"({type(exc).__name__}). Fix user_name and password and restart the "
                f"pipeline; the block does not retry.",
                permanent=True,
            )
            return None
        except SESSION_EXHAUSTION_ERRORS as exc:
            # A slot only frees up when another session times out, and every other
            # client is hitting the same wall: stay away for tens of seconds, jittered.
            retry_in = SESSION_EXHAUSTION_BACKOFF_SECONDS + random.uniform(
                0.0, SESSION_EXHAUSTION_BACKOFF_JITTER_SECONDS
            )
            self._report_failure(
                f"SESSION LIMIT ERROR: server {self._url} refused a new session "
                f"({type(exc).__name__}). Retrying in {retry_in:.0f}s."
            )
            return retry_in
        except Exception as exc:
            return self._report_transient_failure(
                f"CONNECTION ERROR: could not connect to {self._url}: "
                f"{type(exc).__name__}: {exc}"
            )

        subscription_lost = asyncio.Event()
        handler = _DataChangeHandler(self._state, subscription_lost)
        try:
            try:
                node, node_path = await _resolve_node(client, config)
                # Read before subscribing: a wrong NodeId fails here with a precise
                # status, and the first run gets its value without waiting for the
                # first publishing cycle.
                data_value = await node.read_data_value(raise_on_bad_status=False)
                if (
                    data_value.StatusCode is not None
                    and data_value.StatusCode.value in NODE_CONFIGURATION_STATUS_CODES
                ):
                    data_value.StatusCode.check()
                # A Bad value (Ignition lost its PLC, say) is still the current value.
                self._state.record_value(data_value)
                subscription = await client.create_subscription(
                    config.publishing_interval_ms, handler
                )
                await subscription.subscribe_data_change(
                    node, sampling_interval=config.publishing_interval_ms
                )
            except (NodeConfigurationError,) + NODE_CONFIGURATION_ERRORS as exc:
                self._report_failure(
                    exc.args[0]
                    if isinstance(exc, NodeConfigurationError)
                    else f"NODE ERROR: {type(exc).__name__}: {exc}"
                )
                return CONFIGURATION_RETRY_DELAY_SECONDS

            # Settle only on the subscription's own initial value: the read above can be
            # overtaken by a change before the monitored item exists, and that change
            # would never be notified. Servers send the initial value on the first
            # publishing cycle. asyncua stores the negotiated interval on the
            # subscription; the server may have revised our requested interval up.
            publishing_interval_ms = subscription.parameters.RequestedPublishingInterval
            try:
                await asyncio.wait_for(
                    handler.initial_value_delivered.wait(),
                    config.timeout + 2 * publishing_interval_ms / 1000,
                )
            except asyncio.TimeoutError:
                raise ConnectionError(
                    "the server did not deliver the subscription's initial value"
                ) from None
            self._report_connected(node_path)
            while not self._stop_requested.is_set():
                try:
                    await asyncio.wait_for(
                        subscription_lost.wait(), CONNECTION_CHECK_INTERVAL_SECONDS
                    )
                except asyncio.TimeoutError:
                    await _ensure_notifications_flowing(client)
                    continue
                # A dead watchdog also ends the subscription; prefer its error, which
                # names the actual cause.
                await _ensure_notifications_flowing(client)
                raise ConnectionError(
                    f"server ended the subscription ({handler.lost_status})"
                )
            return None
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return self._report_transient_failure(
                f"CONNECTION ERROR: lost connection to {self._url}: "
                f"{type(exc).__name__}: {exc}"
            )
        finally:
            await self._disconnect(client)

    async def _disconnect(self, client: Client) -> None:
        try:
            await asyncio.wait_for(client.disconnect(), DISCONNECT_TIMEOUT_SECONDS)
        except BaseException as exc:
            # Includes CancelledError: a stop() landing here must not leave the socket.
            client.disconnect_socket()
            if isinstance(exc, asyncio.CancelledError):
                raise
            logger.debug(f"OPC UA Reader disconnect error (non-fatal): {exc}")

    def _report_connected(self, node_path: str) -> None:
        self._state.record_connected()
        self._connected_at = time.monotonic()
        if self._outage_started_at is not None:
            logger.info(
                f"OPC UA Reader {self._url} {node_path}: subscription restored "
                f"after {time.monotonic() - self._outage_started_at:.1f}s"
            )
            self._outage_started_at = None
        elif not self._subscribed_once:
            logger.info(f"OPC UA Reader subscribed to {self._url} {node_path}")
        self._subscribed_once = True

    def _report_transient_failure(self, message: str) -> float:
        if (
            self._connected_at is not None
            and time.monotonic() - self._connected_at >= RECONNECT_MAX_DELAY_SECONDS
        ):
            self._reconnect_delay = RECONNECT_INITIAL_DELAY_SECONDS
        retry_in = self._reconnect_delay
        self._reconnect_delay = min(
            self._reconnect_delay * 2, RECONNECT_MAX_DELAY_SECONDS
        )
        self._report_failure(message)
        return retry_in

    def _report_failure(self, message: str, *, permanent: bool = False) -> None:
        self._connected_at = None
        self._state.record_failure(message, permanent=permanent)
        # One ERROR line per outage, not one per reconnect attempt.
        if self._outage_started_at is None:
            self._outage_started_at = time.monotonic()
            logger.error(f"OPC UA Reader {self._url}: {message}")
        else:
            logger.debug(f"OPC UA Reader {self._url}: {message}")


async def _ensure_notifications_flowing(client: Client) -> None:
    """Raise if the session or the subscription's notification delivery has stopped.

    `check_connection()` re-raises the error that ended asyncua's watchdog or publish
    loop, but asyncua ends its publish loop *normally* on `BadNoSubscription` (the server
    dropped the subscription) or when the server reports no subscription at all. The
    session then stays healthy while no notification will ever arrive again, so a
    finished publish loop counts as a lost subscription too.
    """
    await client.check_connection()
    publish_task = getattr(getattr(client, "uaclient", None), "_publish_task", None)
    if publish_task is not None and publish_task.done():
        raise ConnectionError(
            "the server stopped delivering notifications for the subscription"
        )


def _stop_worker(worker: _SubscriptionWorker) -> None:
    try:
        worker.stop()
    except Exception as exc:
        logger.debug(f"OPC UA Reader stop error (non-fatal): {exc}")


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "OPC UA Reader",
            "version": "v1",
            "short_description": "Subscribes to a variable on an OPC UA server and returns its latest value on every run.",
            "long_description": LONG_DESCRIPTION,
            "license": "Roboflow Enterprise License",
            "block_type": "transformation",
            "ui_manifest": {
                "section": "industrial",
                "icon": "fal fa-industry",
                "blockPriority": 11,
                "enterprise_only": True,
                "local_only": True,
            },
        }
    )
    type: Literal[BLOCK_TYPE]
    url: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="URL of the OPC UA server to read from. Put credentials in user_name "
        "and password: a url containing user:password@ is rejected.",
        examples=["opc.tcp://localhost:4840/opcua/server", "$inputs.opc_url"],
    )
    namespace: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="The OPC UA namespace URI or index used to locate objects and variables.",
        examples=["http://opcua.roboflow.run", "2", "$inputs.opc_namespace"],
    )
    user_name: Optional[Union[Selector(kind=[STRING_KIND]), str]] = Field(
        default=None,
        description="Optional username for authentication when connecting to the OPC UA server.",
        examples=["John", "$inputs.opc_user_name"],
    )
    password: Optional[Union[Selector(kind=[SECRET_KIND, STRING_KIND]), str]] = Field(
        default=None,
        description="Optional password for authentication when connecting to the OPC UA server. "
        "Requires user_name. A literal value is stored with the workflow definition; "
        "use a secret or workflow input selector to keep it out.",
        examples=["$inputs.opc_password"],
        json_schema_extra={"private": True},
    )
    object_name: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="The object path containing the variable (hierarchical mode), or the "
        "start of the string NodeId (direct mode).",
        examples=["Line1", "Sensors.LineRunning", "$inputs.opc_object_name"],
    )
    variable_name: Union[Selector(kind=[STRING_KIND]), str] = Field(
        default="",
        description="The name of the variable to read within the object. Required in "
        "hierarchical mode; in direct mode it may be left empty to use object_name as the "
        "full NodeId string.",
        examples=["LineRunning", "", "$inputs.opc_variable_name"],
    )
    node_lookup_mode: Union[
        Selector(kind=[STRING_KIND]),
        Literal["hierarchical", "direct"],
    ] = Field(
        default="hierarchical",
        description="Method to locate the OPC UA node: 'hierarchical' uses path navigation, "
        "'direct' uses NodeId strings (for Ignition-style string-based tags).",
        examples=["hierarchical", "direct"],
    )
    publishing_interval_ms: Union[
        Annotated[
            float,
            Field(
                ge=MIN_PUBLISHING_INTERVAL_MS,
                le=MAX_PUBLISHING_INTERVAL_MS,
                allow_inf_nan=False,
            ),
        ],
        Selector(kind=[FLOAT_KIND, INTEGER_KIND]),
    ] = Field(
        default=100,
        description="How often, in milliseconds, the server samples the variable and sends "
        "changes. Lower values deliver changes sooner; the server may revise it upwards. "
        "Between 10 and 60000.",
        examples=[100, 50, "$inputs.opc_publishing_interval_ms"],
    )
    timeout: Union[
        Annotated[float, Field(gt=0, le=MAX_TIMEOUT_SECONDS, allow_inf_nan=False)],
        Selector(kind=[FLOAT_KIND, INTEGER_KIND]),
    ] = Field(
        default=2.0,
        description="Seconds the first run waits for the connection and the initial value, "
        "and the timeout of each request to the server. Later runs never wait. Greater "
        "than 0 and at most 60.",
        examples=[2.0, "$inputs.opc_timeout"],
    )

    @model_validator(mode="after")
    def _reject_credentials_in_literal_url(self) -> "BlockManifest":
        # Fail at compile time, before the definition is cached; a selector-supplied
        # url is checked at run time.
        if isinstance(self.url, str) and not is_workflow_selector(self.url):
            problem = _url_problem(self.url)
            if problem is not None:
                raise ValueError(problem)
        return self

    @model_validator(mode="after")
    def _require_variable_name_for_hierarchical_lookup(self) -> "BlockManifest":
        # Only literals can be checked here; selector values are validated at run time.
        if self.node_lookup_mode == "hierarchical" and self.variable_name == "":
            raise ValueError(
                "variable_name is required when node_lookup_mode is 'hierarchical'; it "
                "may be empty only in 'direct' mode."
            )
        return self

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="value",
                kind=[
                    BOOLEAN_KIND,
                    INTEGER_KIND,
                    FLOAT_KIND,
                    STRING_KIND,
                    LIST_OF_VALUES_KIND,
                ],
            ),
            OutputDefinition(name="is_new", kind=[BOOLEAN_KIND]),
            OutputDefinition(name="error_status", kind=[BOOLEAN_KIND]),
            OutputDefinition(name="error_message", kind=[STRING_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"

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
                    "`error_status=true` with no value, and no OPC UA connection is "
                    "opened from the hosted platform."
                ),
                applies_to_runtimes=[Runtime.HOSTED_SERVERLESS],
            ),
            RuntimeRestriction(
                code="connection_and_state_rebuilt_per_request",
                severity=Severity.SOFT,
                note=(
                    "The subscription and the latest value live in this block "
                    "instance. Over HTTP every request builds a fresh instance, so "
                    "each request opens a session, reads the current value and closes "
                    "it again. Use an InferencePipeline for a live subscription."
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
            OPC UA server I/O, plus the latest value held in process memory between
            runs; the last value is returned again (with ``is_new=False``) when
            nothing changed.
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
            An empty list: the block talks to an OPC UA server only.
        """
        return []


class OPCReaderBlockV1(WorkflowBlock):

    def __init__(self):
        self._config: Optional[OPCUAReaderConfig] = None
        self._state: Optional[OPCUAReaderState] = None
        self._worker: Optional[_SubscriptionWorker] = None
        self._finalizer: Optional[weakref.finalize] = None
        # The sequence of the value the previous run returned, for `is_new`.
        self._returned_sequence = 0
        # Only the first run waits for the server; later runs read memory only.
        self._first_run_done = False
        # A video pipeline would otherwise log the same rejected input on every frame.
        self._last_logged_failure: Optional[str] = None
        self._lifecycle_lock = threading.Lock()

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def close(self) -> None:
        """Close the session and stop the subscription thread."""
        with self._lifecycle_lock:
            finalizer = self._finalizer
            state = self._state
            self._config = None
            self._state = None
            self._worker = None
            self._finalizer = None
            self._returned_sequence = 0
            self._first_run_done = False
        if state is not None:
            # Wakes a first run still waiting for the server; it then reports the close.
            state.settled.set()
        if finalizer is not None:
            # Runs `_stop_worker` at most once, whoever gets there first.
            finalizer()

    def run(
        self,
        url: str,
        namespace: str,
        object_name: str,
        variable_name: str = "",
        user_name: Optional[str] = None,
        password: Optional[str] = None,
        node_lookup_mode: Literal["hierarchical", "direct"] = "hierarchical",
        publishing_interval_ms: float = 100,
        timeout: float = 2.0,
    ) -> BlockResult:
        """Return the latest value of the subscribed variable.

        The first run starts the subscription and waits up to `timeout` for it; later
        runs read the value held in memory and never wait for the server.

        Args:
            url: The server's opc.tcp:// endpoint, without credentials.
            namespace: Namespace URI or index of the variable.
            object_name: Object path (hierarchical) or NodeId string start (direct).
            variable_name: Variable within the object; may be empty in direct mode.
            user_name: User name, set together with password, or None for anonymous.
            password: Password for user_name.
            node_lookup_mode: `hierarchical` browse path or `direct` string NodeId.
            publishing_interval_ms: Requested sampling and publishing interval.
            timeout: Seconds the first run waits, and the per-request timeout.

        Returns:
            `value`, `is_new`, `error_status` and `error_message`; failures are
            reported in the outputs rather than raised.
        """
        if GCP_SERVERLESS or LAMBDA:
            # never open an outbound OPC UA connection from hosted workers
            return self._rejected_run(
                "OPC UA Reader is not available on the Roboflow hosted platform"
            )
        # selector-resolved values bypass manifest constraints, so validate here
        config_or_error = _build_config(
            url=url,
            namespace=namespace,
            user_name=user_name,
            password=password,
            object_name=object_name,
            variable_name=variable_name,
            node_lookup_mode=node_lookup_mode,
            publishing_interval_ms=publishing_interval_ms,
            timeout=timeout,
        )
        if isinstance(config_or_error, str):
            return self._rejected_run(config_or_error)
        config = config_or_error

        with self._lifecycle_lock:
            if self._worker is None:
                state = OPCUAReaderState()
                worker = _SubscriptionWorker(config=config, state=state)
                worker.start()
                self._config = config
                self._state = state
                self._worker = worker
                # Closes the session when the block is garbage collected (each HTTP
                # request drops its instance) and on a normal interpreter exit.
                self._finalizer = weakref.finalize(self, _stop_worker, worker)
            elif config != self._config:
                return self._rejected_run(
                    "OPC UA Reader configuration (url, namespace, credentials, object, "
                    "variable, lookup mode, publishing interval or timeout) changed "
                    "between runs; this block subscribes only with the configuration of "
                    "its first run."
                )
            state = self._state
            first_run = not self._first_run_done
            self._first_run_done = True
        if first_run:
            # Outside the lock, so close() is never queued behind this wait.
            state.settled.wait(timeout=config.timeout)
        with self._lifecycle_lock:
            if self._state is not state:
                return _failure("OPC UA Reader was closed while connecting.")

            result = self._snapshot(state)

        return result

    def _rejected_run(self, message: str) -> BlockResult:
        """The outputs of a run refused before reading; logged once per distinct reason."""
        if message != self._last_logged_failure:
            self._last_logged_failure = message
            logger.error(f"OPC UA Reader failure: {message}")

        result = _failure(message)
        return result

    def _snapshot(self, state: OPCUAReaderState) -> BlockResult:
        # Connection failures are logged by the worker, once per outage.
        with state.lock:
            if not state.has_value:
                return _failure(state.error or NOT_CONNECTED_WITHIN_TIMEOUT)
            is_new = state.sequence != self._returned_sequence
            self._returned_sequence = state.sequence
            if not state.connected:
                if state.permanent:
                    error = state.error
                else:
                    error = (
                        f"{state.error or 'OPC UA server not connected'}. The block keeps "
                        f"reconnecting in the background and returns the last value "
                        f"received."
                    )
            elif not state.status_is_good:
                error = f"Value quality is not Good: {state.status_name}"
            else:
                error = None
            return {
                # a copy: a downstream step mutating an array must not change the
                # value held for later runs
                "value": copy.deepcopy(state.value),
                "is_new": is_new,
                "error_status": error is not None,
                "error_message": error,
            }


def _build_config(
    url: Any,
    namespace: Any,
    user_name: Any,
    password: Any,
    object_name: Any,
    variable_name: Any,
    node_lookup_mode: Any,
    publishing_interval_ms: Any,
    timeout: Any,
) -> Union[OPCUAReaderConfig, str]:
    """Validate run() arguments; returns the config, or an error message."""
    if not isinstance(url, str) or not url:
        return (
            f"Invalid url: expected a non-empty opc.tcp:// URL, got "
            f"{type(url).__name__}."
        )
    url_problem = _url_problem(url)
    if url_problem is not None:
        return url_problem
    # a namespace index arrives as an int when it comes from a numeric workflow input
    if isinstance(namespace, int) and not isinstance(namespace, bool):
        namespace = str(namespace)
    if not isinstance(namespace, str) or not namespace:
        return f"Invalid namespace: {namespace!r}. Must be a namespace URI or index."
    if not isinstance(object_name, str) or not object_name:
        return f"Invalid object_name: {object_name!r}. Must be a non-empty string."
    if variable_name is None:
        variable_name = ""
    if not isinstance(variable_name, str):
        return f"Invalid variable_name: {variable_name!r}. Must be a string."
    if node_lookup_mode not in NODE_LOOKUP_MODES:
        return (
            f"Invalid node_lookup_mode: {node_lookup_mode!r}. "
            f"Must be 'hierarchical' or 'direct'."
        )
    if node_lookup_mode == "hierarchical" and not variable_name:
        return "variable_name is required in hierarchical node_lookup_mode."
    # never echo these: a wrongly wired selector may carry a secret
    if user_name is not None and not isinstance(user_name, str):
        return f"Invalid user_name: expected a string, got {type(user_name).__name__}."
    if password is not None and not isinstance(password, str):
        return f"Invalid password: expected a string, got {type(password).__name__}."
    # an empty editor field is "unset"
    user_name = user_name or None
    password = password or None
    if (user_name is None) != (password is None):
        return "Set both user_name and password to authenticate, or neither."
    numbers = {}
    for name, raw, is_within_bounds, bounds in (
        (
            "publishing_interval_ms",
            publishing_interval_ms,
            lambda number: MIN_PUBLISHING_INTERVAL_MS
            <= number
            <= MAX_PUBLISHING_INTERVAL_MS,
            "between 10 and 60000",
        ),
        (
            "timeout",
            timeout,
            lambda number: 0 < number <= MAX_TIMEOUT_SECONDS,
            "greater than 0 and at most 60",
        ),
    ):
        try:
            if isinstance(raw, bool):
                raise ValueError
            number = float(raw)
        except (TypeError, ValueError, OverflowError):
            number = math.nan
        if not math.isfinite(number) or not is_within_bounds(number):
            return f"Invalid {name}: {raw!r}. Must be a number {bounds}."
        numbers[name] = number
    return OPCUAReaderConfig(
        url=url,
        namespace=namespace,
        user_name=user_name,
        password=password,
        object_name=object_name,
        variable_name=variable_name,
        node_lookup_mode=node_lookup_mode,
        timeout=numbers["timeout"],
        publishing_interval_ms=numbers["publishing_interval_ms"],
    )


def _failure(message: str) -> BlockResult:
    return {
        "value": None,
        "is_new": False,
        "error_status": True,
        "error_message": message,
    }

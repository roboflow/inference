"""Invocation-driven MQTT blocks with explicit connection ownership.

``MQTTReaderBlock`` borrows a host-created ``MQTTSubscription`` resource named
``mqtt_subscription``. The host starts it before runs, stops admissions and drains
runs before closing it. Processing reset does not reset the subscription: call
``reset_buffer()`` while quiescent if old messages should be discarded. To change
identity, close the old owner, construct another and bind new sessions to it.
Sharing an owner deliberately shares its queue and last result across sessions.

``MQTTWriterBlock`` borrows ``mqtt_writer`` when supplied. Otherwise each publish
owns one client, connection and loop, including cleanup inside BackgroundEffects.
Default writers disable reconnect: an uncertain publish is never replayed by the
block. A cleanup failure is reported after, not instead of, the publish
disposition. A host may explicitly reuse ``MQTTPublisher``; it serializes
operations and must be closed after runs and effects drain. A failed connection
needs a new owner.

Optional paho-mqtt~=1.6.1 is imported only when an owner is constructed. TCP uses
paho's private connect timeout; DNS resolution and loop_stop's join have no hard
deadline. Readiness and publish acknowledgement waits are bounded individually,
not a total-operation deadline. Cleanup also explicitly closes paho 1.6.1's socket
pair after joining, rather than relying on its destructor; a pair that fails
part-way through acquisition is closed before paho receives it. QoS 0 success
means sent; QoS 1/2 success means protocol acknowledgement, not application
processing.

V1 buffering (latest=1, sequential=1000, drop oldest), retained duplicate handling,
JSON-object decoding, TLS and operator broker policy are preserved. Ownership is
a deliberate V2 hosting difference. Optional JSON parsing or payload copying
that exceeds Python's recursion limit yields ``payload={}`` while preserving
the raw ``value`` and topic. This corrects an inherited V1 weakness. No import
of the legacy blocks is needed.
"""

import copy
import importlib
import json
import math
import threading
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Protocol, Union

from pydantic import Field, StrictBool, StrictInt, StrictStr
from roboflow_workflows import environment
from roboflow_workflows.execution_engine.entities.workload import WorkOperation
from roboflow_workflows.v2.blocks.sinks.runtime import (
    BACKGROUND_MESSAGE,
    ERROR_STATUS_OUTPUT,
    MESSAGE_OUTPUT,
    BackgroundEffects,
    deliver,
    disabled_sink_message,
)
from roboflow_workflows.v2.data.kinds import (
    BOOLEAN_KIND,
    DICTIONARY_KIND,
    FLOAT_KIND,
    INTEGER_KIND,
    STRING_KIND,
)
from roboflow_workflows.v2.data.secrets import (
    SECRET_KIND,
    Secret,
    reveal_secrets,
    to_secret,
)
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref

READER_BLOCK_TYPE = "v2/mqtt_reader@v1"
WRITER_BLOCK_TYPE = "v2/mqtt_writer_sink@v1"
_PERMANENT_CODES = frozenset({1, 2, 4, 5})
_REASONS = {
    1: "unacceptable protocol",
    2: "identifier rejected",
    3: "broker unavailable",
    4: "bad username or password",
    5: "not authorised",
}
_UNKNOWN_DELIVERY = "MQTT delivery status unknown; the message may still be delivered. Do not retry automatically."
_PUBLISHED = "Message published successfully"


class MQTTError(RuntimeError):
    """A sanitized configuration, connection, delivery or cleanup failure."""


@dataclass(frozen=True, repr=False)
class MQTTTarget:
    """Immutable connection and subscription identity; credentials never in repr.

    Args:
        host: Broker address, subject to deployment policy.
        port: TCP port; TLS does not change it automatically.
        topic: Subscription filter or publish destination.
        username: Optional authentication user.
        password: Secret credential, never printed in identity errors.
        qos: MQTT quality of service, 0 through 2.
        timeout: Individual TCP/readiness/publish timeout, excluding DNS and join.
        encryption: ``none`` or certificate-verified ``tls``.
        ca_certificate_path: Optional permitted local CA file.
        client_id: Stable id enables persistent reader sessions; requires QoS 1/2.
        read_mode: ``latest`` or ``sequential`` for subscription consumption.
    """

    host: str
    port: int
    topic: str
    username: Optional[str] = None
    password: Optional[Secret] = None
    qos: int = 0
    timeout: float = 0.5
    encryption: str = "none"
    ca_certificate_path: Optional[str] = None
    client_id: Optional[str] = None
    read_mode: str = "latest"

    def __repr__(self) -> str:
        return "MQTTTarget(<private connection identity>)"


def _host_port(entry: str):
    entry = entry.strip().lower()
    if entry.startswith("[") and "]" in entry:
        end = entry.index("]")
        suffix = entry[end + 1 :]
        return entry[1:end], (
            int(suffix[1:]) if suffix.startswith(":") and suffix[1:].isdigit() else None
        )
    if entry.count(":") == 1:
        host, port = entry.split(":")
        return host, int(port) if port.strip().isdigit() else None
    return entry, None


def _policy_target(target: MQTTTarget) -> MQTTTarget:
    entries = environment.MQTT_WORKFLOWS_BLOCKS_WHITELISTED_HOSTS
    entries = (
        None if entries is None else [str(e).strip() for e in entries if str(e).strip()]
    )
    if not environment.MQTT_WORKFLOWS_BLOCKS_ALLOW_USER_PROVIDED_HOST:
        if not entries:
            raise MQTTError("MQTT is disabled: no operator broker is configured.")
        host, port = _host_port(entries[0])
        return replace(target, host=host, port=port or 1883)
    if entries is not None:
        wanted = target.host.strip().lower().strip("[]")
        if not any(
            h == wanted and p in (None, target.port)
            for h, p in map(_host_port, entries)
        ):
            raise MQTTError("MQTT broker is not permitted by deployment policy.")
    return target


def _validated_target(
    target: MQTTTarget, *, reader: bool, allow_files: bool
) -> MQTTTarget:
    if not isinstance(target.host, str) or not target.host.strip():
        raise MQTTError("MQTT host must be a non-empty string.")
    if type(target.port) is not int or not 1 <= target.port <= 65535:
        raise MQTTError("MQTT port must be an integer from 1 to 65535.")
    if type(target.qos) is not int or target.qos not in (0, 1, 2):
        raise MQTTError("MQTT qos must be 0, 1 or 2.")
    if (
        isinstance(target.timeout, bool)
        or not isinstance(target.timeout, (int, float))
        or not math.isfinite(target.timeout)
        or not 0 < target.timeout <= threading.TIMEOUT_MAX
    ):
        raise MQTTError(
            "MQTT timeout must be a positive finite number within the platform limit."
        )
    if not isinstance(target.topic, str) or not target.topic or "\x00" in target.topic:
        raise MQTTError("MQTT topic must be a non-empty UTF-8 string without NUL.")
    try:
        encoded = target.topic.encode("utf-8")
    except UnicodeError:
        raise MQTTError("MQTT topic must be valid UTF-8.") from None
    if len(encoded) > 65535:
        raise MQTTError("MQTT topic exceeds the protocol length limit.")
    if reader:
        parts = target.topic.split("/")
        if any(
            ("+" in p and p != "+") or ("#" in p and (p != "#" or i != len(parts) - 1))
            for i, p in enumerate(parts)
        ):
            raise MQTTError(
                "MQTT subscription wildcard must occupy a whole level; # must be last."
            )
    elif "+" in target.topic or "#" in target.topic:
        raise MQTTError("MQTT publish topic cannot contain wildcards.")
    if target.password is not None and target.username is None:
        raise MQTTError("MQTT password requires username.")
    if target.username is not None and not isinstance(target.username, str):
        raise MQTTError("MQTT username must be a string.")
    if target.encryption not in ("none", "tls"):
        raise MQTTError("MQTT encryption must be none or tls.")
    if target.ca_certificate_path is not None and not isinstance(
        target.ca_certificate_path, str
    ):
        raise MQTTError("MQTT ca_certificate_path must be a string.")
    if target.encryption == "tls" and target.ca_certificate_path:
        if not allow_files:
            raise MQTTError(
                "MQTT ca_certificate_path requires local file-system permission."
            )
        if not Path(target.ca_certificate_path).is_file():
            raise MQTTError("MQTT CA certificate must be a regular file.")
    client_id = target.client_id
    if client_id is not None and not isinstance(client_id, str):
        raise MQTTError("MQTT client_id must be a string.")
    client_id = client_id.strip() or None if client_id is not None else None
    if reader and client_id and target.qos == 0:
        raise MQTTError("MQTT persistent client_id requires qos 1 or 2.")
    if target.read_mode not in ("latest", "sequential"):
        raise MQTTError("MQTT read_mode must be latest or sequential.")
    normalized = replace(
        target,
        client_id=client_id,
        ca_certificate_path=target.ca_certificate_path or None,
    )
    return _policy_target(normalized)


def _same_target(left: MQTTTarget, right: MQTTTarget) -> bool:
    # Secret uses object identity; compare values only here, without logging them.
    left_fields, right_fields = vars(left).copy(), vars(right).copy()
    left_fields["password"] = reveal_secrets(left.password).value
    right_fields["password"] = reveal_secrets(right.password).value
    return left_fields == right_fields


def _import_paho():
    try:
        module = importlib.import_module("paho.mqtt.client")
    except ImportError:
        raise MQTTError(
            'MQTT driver unavailable. Install "paho-mqtt~=1.6.1" in the runtime.'
        ) from None
    return module


class _OpeningTLSContext:
    """Delegate TLS configuration while retaining a socket during handshake."""

    def __init__(self, context, client):
        self._context = context
        self._client = client

    def __getattr__(self, name):
        return getattr(self._context, name)

    def wrap_socket(self, *args, **kwargs):
        sock = self._context.wrap_socket(*args, **kwargs)
        self._client._opening_socket = sock
        return sock


def _new_client(mqtt, **settings):
    # paho 1.6.1 retries CONNACK 1/2 before calling on_connect: it silently
    # downgrades the protocol or generates a new client id. These are permanent
    # failures for our immutable identity, so expose them before SDK fallback.
    class Client(mqtt.Client):
        _opening_socket = None
        _pair_cleanup_failed = False

        def loop_start(self):
            if not hasattr(mqtt, "_socketpair_compat"):
                result = super().loop_start()
                return result
            # paho 1.6.1's loop_start, except that the wake-up pair comes from
            # _open_wakeup_pair: paho's helper drops partially acquired sockets.
            if self._thread is not None:
                return mqtt.MQTT_ERR_INVAL
            self._sockpairR, self._sockpairW = self._open_wakeup_pair()
            self._thread_terminate = False
            self._thread = threading.Thread(target=self._thread_main)
            self._thread.daemon = True
            self._thread.start()

        def _open_wakeup_pair(self):
            # Same TCP pair as paho's _socketpair_compat. Every endpoint acquired
            # before a failure is closed here; close() reports a failed close.
            sdk_socket = mqtt.socket
            acquired = []
            try:
                listener = sdk_socket.socket(
                    sdk_socket.AF_INET, sdk_socket.SOCK_STREAM, sdk_socket.IPPROTO_IP
                )
                acquired.append(listener)
                listener.setsockopt(sdk_socket.SOL_SOCKET, sdk_socket.SO_REUSEADDR, 1)
                listener.bind(("127.0.0.1", 0))
                listener.listen(1)
                port = listener.getsockname()[1]
                reader = sdk_socket.socket(
                    sdk_socket.AF_INET, sdk_socket.SOCK_STREAM, sdk_socket.IPPROTO_IP
                )
                acquired.append(reader)
                reader.setblocking(0)
                try:
                    reader.connect(("127.0.0.1", port))
                except BlockingIOError:
                    pass
                writer = listener.accept()[0]
                acquired.append(writer)
                writer.setblocking(0)
                listener.close()
            except BaseException:
                for sock in reversed(acquired):
                    try:
                        sock.close()
                    except Exception:
                        self._pair_cleanup_failed = True
                raise
            return reader, writer

        def _create_socket_connection(self):
            sock = super()._create_socket_connection()
            self._opening_socket = sock
            return sock

        def reconnect(self):
            # paho stores its socket only after TLS handshake. Track the socket
            # while opening so a failed handshake cannot leave it to GC.
            original_context = self._ssl_context
            if original_context is not None:
                self._ssl_context = _OpeningTLSContext(original_context, self)
            try:
                result = super().reconnect()
                return result
            except BaseException:
                sock = self._opening_socket
                if sock is not None:
                    if sock is self._sock:
                        self._sock_close()
                    else:
                        sock.close()
                raise
            finally:
                self._opening_socket = None
                self._ssl_context = original_context

        def _handle_connack(self):
            packet = self._in_packet["packet"]
            if len(packet) == 2 and packet[1] in (1, 2):
                callback = self.on_connect
                if callback is not None:
                    callback(self, None, {"session present": packet[0] & 1}, packet[1])
                return mqtt.MQTT_ERR_CONN_REFUSED
            result = super()._handle_connack()
            return result

    client = Client(**settings)
    return client


class _Connection:
    """Host-owned lifecycle; operation lock never held by a network callback."""

    def __init__(
        self, target: MQTTTarget, *, reader: bool, allow_access_to_file_system: bool
    ):
        if reader and (environment.GCP_SERVERLESS or environment.LAMBDA):
            raise MQTTError("MQTT Reader is unavailable on the hosted platform.")
        self._target = _validated_target(
            target, reader=reader, allow_files=allow_access_to_file_system
        )
        self._reader = reader
        self._operation = threading.RLock()
        self._condition = threading.Condition(threading.RLock())
        self._closed = False
        self._closing = False
        self._close_error = None
        self._connected = False
        self._subscribed = False
        self._refused = None
        self._subscribe_error = False
        self._subscription_mid = None
        self._messages = deque(maxlen=1 if target.read_mode == "latest" else 1000)
        self._last = {"value": None, "payload": {}, "topic": None}
        self._subscribed_once = False
        self._client = None
        # Fail before acquiring a client, including when restoring a recording.
        password = reveal_secrets(self._target.password).value
        mqtt = _import_paho()
        try:
            self._client = _new_client(
                mqtt,
                client_id=self._target.client_id or "",
                clean_session=not bool(self._target.client_id),
                reconnect_on_failure=reader,
            )
            client = self._client
            client.on_connect = self._on_connect
            client.on_connect_fail = self._on_connect_fail
            client.on_disconnect = self._on_disconnect
            if reader:
                client.on_subscribe = self._on_subscribe
                client.on_message = self._on_message
            if target.username is not None:
                client.username_pw_set(target.username, password)
            if target.encryption == "tls":
                client.tls_set(
                    **(
                        {"ca_certs": target.ca_certificate_path}
                        if target.ca_certificate_path
                        else {}
                    )
                )
            client._connect_timeout = target.timeout
            client.reconnect_delay_set(
                min_delay=target.timeout / 2, max_delay=target.timeout * 2
            )
            if reader:
                result = client.connect_async(
                    self._target.host, self._target.port, keepalive=15
                )
            else:
                result = client.connect(
                    self._target.host, self._target.port, keepalive=15
                )
            if result not in (None, 0):
                raise MQTTError("MQTT connection setup failed.")
            if client.loop_start() not in (None, 0):
                raise MQTTError("MQTT network loop could not start.")
        except BaseException as error:
            try:
                self.close()
            except MQTTError:
                raise MQTTError(
                    "MQTT setup and cleanup failed; host intervention required."
                ) from None
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            raise MQTTError(
                "MQTT setup failed; check driver, credentials, TLS and broker configuration."
            ) from None

    def _on_connect(self, client, userdata, flags, rc):
        with self._condition:
            if self._closed or self._closing:
                return
            self._connected = rc == 0
            self._refused = rc or None
            self._subscribed = False
            self._subscription_mid = None
            self._subscribe_error = False
            if rc in _PERMANENT_CODES:
                # Do not join the callback's own network thread.
                try:
                    client.disconnect()
                except Exception:
                    pass
            elif rc == 0 and self._reader:
                try:
                    result, mid = client.subscribe(
                        self._target.topic, qos=self._target.qos
                    )
                    self._subscription_mid = mid
                    self._subscribe_error = result != 0
                except Exception:
                    self._subscribe_error = True
            self._condition.notify_all()

    def _on_connect_fail(self, client, userdata):
        with self._condition:
            if not self._closed and not self._closing:
                self._connected = self._subscribed = False
                self._condition.notify_all()

    def _on_disconnect(self, client, userdata, rc):
        with self._condition:
            if not self._closed and not self._closing:
                self._connected = self._subscribed = False
                self._subscription_mid = None
                self._condition.notify_all()

    def _on_subscribe(self, client, userdata, mid, granted_qos):
        with self._condition:
            if (
                self._closed
                or self._closing
                or not self._connected
                or mid != self._subscription_mid
            ):
                return
            self._subscribe_error = not granted_qos or any(
                q not in (0, 1, 2) for q in granted_qos
            )
            self._subscribed = not self._subscribe_error
            self._condition.notify_all()

    def _on_message(self, client, userdata, message):
        with self._condition:
            if self._closed or self._closing:
                return
            self._messages.append(
                (message.topic, bytes(message.payload), bool(message.retain))
            )
            self._condition.notify_all()

    def _status_error(self):
        if self._closed or self._closing:
            return "MQTT owner is closed; create and bind a new host owner."
        if self._refused:
            disposition = (
                "host must close and rebind"
                if self._refused in _PERMANENT_CODES
                else (
                    "subscription keeps reconnecting"
                    if self._reader
                    else "operation failed"
                )
            )
            return f"MQTT connection refused: {_REASONS.get(self._refused, 'unknown reason')} (code {self._refused}); {disposition}."
        if self._subscribe_error:
            return "MQTT broker refused subscription or local subscribe failed."
        if not self._connected:
            return "MQTT broker not connected."
        if self._reader and not self._subscribed:
            return "MQTT subscription not established; awaiting SUBACK."
        return None

    def _check_identity(self, target):
        if not _same_target(self._target, target):
            raise MQTTError(
                "MQTT connection identity changed; stop borrowers, close and explicitly rebind the host owner."
            )
        if self._closed or self._closing:
            raise MQTTError("MQTT owner is closed.")

    def close(self) -> None:
        """Reject operations, disconnect, join, then clear callbacks and buffers.

        Call from the host after admissions stop and borrowed runs/effects drain.
        Concurrent closes serialize; repeated calls do not repeat SDK cleanup.
        Cleanup failures remain observable on repeated close calls.

        Raises:
            MQTTError: If SDK cleanup fails; no successful shutdown is claimed.
        """
        with self._condition:
            self._closing = True
            self._condition.notify_all()
        with self._operation:
            with self._condition:
                if self._closed:
                    if self._close_error:
                        raise MQTTError(self._close_error)
                    return
                self._closed = True
                self._condition.notify_all()
            failed = False
            client = self._client
            if client is not None:
                try:
                    client.disconnect()
                except Exception:
                    failed = True
                joined = False
                try:
                    client.loop_stop()
                    joined = True
                except Exception:
                    failed = True
                    thread = getattr(client, "_thread", None)
                    # Thread.start can fail after paho allocated the socket pair.
                    joined = thread is None or not thread.is_alive()
                if joined:
                    # paho 1.6.1 leaves its socket pair until __del__ otherwise.
                    try:
                        client._reset_sockets()
                    except Exception:
                        failed = True
                failed = failed or client._pair_cleanup_failed
                for callback in (
                    "on_connect",
                    "on_connect_fail",
                    "on_disconnect",
                    "on_subscribe",
                    "on_message",
                ):
                    setattr(client, callback, None)
            with self._condition:
                self._connected = self._subscribed = False
                self._messages.clear()
                self._last = {"value": None, "payload": {}, "topic": None}
                if failed:
                    self._close_error = (
                        "MQTT cleanup failed; shutdown is not confirmed."
                    )
            if failed:
                raise MQTTError(self._close_error)


class MQTTSubscription(_Connection):
    """Persistent bounded subscription, explicitly constructed and closed by host.

    Args:
        target: Immutable subscription identity; no live reconfiguration.
        allow_access_to_file_system: Permission for a custom TLS CA path.
    """

    def __init__(
        self, target: MQTTTarget, *, allow_access_to_file_system: bool = False
    ):
        super().__init__(
            target, reader=True, allow_access_to_file_system=allow_access_to_file_system
        )

    def read(self, target: MQTTTarget) -> Dict[str, Any]:
        """Consume one message; repeat the last value when no message is queued.

        Args:
            target: Must match this owner's normalized immutable identity.

        Returns:
            Value, JSON object, actual topic, is_new, error_status and error_message.
            A cold read waits for readiness and once for retained data; subsequent
            reads do not wait. Buffered data is drained even during an outage.

        Raises:
            MQTTError: When the owner is closed or the identity changed.
        """
        with self._operation, self._condition:
            self._check_identity(target)
            if not self._subscribed_once:
                self._condition.wait_for(
                    lambda: self._subscribed
                    or self._refused is not None
                    or self._subscribe_error
                    or bool(self._messages)
                    or self._closed
                    or self._closing,
                    timeout=target.timeout,
                )
                if self._subscribed:
                    self._subscribed_once = True
                    self._condition.wait_for(
                        lambda: bool(self._messages)
                        or not self._connected
                        or self._closed
                        or self._closing,
                        timeout=target.timeout,
                    )
            if not self._messages:
                error = self._status_error()
                return {
                    **self._last_output(),
                    "is_new": False,
                    "error_status": error is not None,
                    "error_message": error,
                }
            topic, payload, retained = self._messages.popleft()
            try:
                value = payload.decode("utf-8")
            except UnicodeDecodeError:
                return {
                    **self._last_output(),
                    "is_new": False,
                    "error_status": True,
                    "error_message": "MQTT message is not valid UTF-8.",
                }
            parsed = {}
            try:
                candidate = json.loads(value)
                if isinstance(candidate, dict):
                    parsed = candidate
            except (ValueError, RecursionError):
                # The optional object view must not destroy the raw message.
                pass
            is_new = not (
                retained
                and self._last["topic"] == topic
                and self._last["value"] == value
            )
            self._last = {"value": value, "payload": parsed, "topic": topic}
            return {
                **self._last_output(),
                "is_new": is_new,
                "error_status": False,
                "error_message": None,
            }

    def _last_output(self) -> Dict[str, Any]:
        payload = {}
        try:
            payload = copy.deepcopy(self._last["payload"])
        except RecursionError:
            # Parsing may succeed at a depth that payload copying cannot handle.
            pass

        outputs = {**self._last, "payload": payload}

        return outputs

    def reset_buffer(self) -> None:
        """Discard buffered and last values while preserving this subscription.

        The host quiesces borrowers first. Callbacks serialize with this operation;
        messages received afterwards belong to the new buffer generation.

        Raises:
            MQTTError: If the owner is closed.
        """
        with self._operation, self._condition:
            if self._closed or self._closing:
                raise MQTTError("MQTT owner is closed.")
            self._messages.clear()
            self._last = {"value": None, "payload": {}, "topic": None}


class MQTTWriter(Protocol):
    """Borrowed writer contract; host serializes and closes any reused resources."""

    def publish(self, target: MQTTTarget, message: str, *, retain: bool) -> None:
        """Publish once or raise; returning means SDK publish evidence was observed.

        Args:
            target: Validated immutable operation settings.
            message: UTF-8 string payload, already detached from workflow inputs.
            retain: Broker retained-message flag.
        """
        ...


class MQTTPublisher(_Connection):
    """Reusable host-owned publisher; publish operations serialize with close.

    Args:
        target: Immutable connection and publish identity.
        allow_access_to_file_system: Permission for a custom TLS CA path.
    """

    def __init__(
        self, target: MQTTTarget, *, allow_access_to_file_system: bool = False
    ):
        super().__init__(
            target,
            reader=False,
            allow_access_to_file_system=allow_access_to_file_system,
        )

    def publish(
        self, target: MQTTTarget, message: str, *, retain: bool = False
    ) -> None:
        """Publish once and wait for evidence, without reconnect or replay.

        Args:
            target: Must match this owner's normalized identity.
            message: Payload passed to paho's UTF-8 encoder.
            retain: Broker retained flag.

        Raises:
            MQTTError: On connection failure, rejected publish or uncertain result.
        """
        with self._operation:
            with self._condition:
                self._check_identity(target)
                self._condition.wait_for(
                    lambda: self._connected
                    or self._refused is not None
                    or self._closing,
                    timeout=target.timeout,
                )
                error = self._status_error()
                if error:
                    raise MQTTError(error)
            try:
                info = self._client.publish(
                    target.topic, message, qos=target.qos, retain=retain
                )
                if info.rc != 0:
                    raise MQTTError(_UNKNOWN_DELIVERY)
                info.wait_for_publish(timeout=target.timeout)
                if not info.is_published():
                    raise MQTTError(_UNKNOWN_DELIVERY)
            except Exception:
                # publish may have sent bytes even when the SDK raises.
                raise MQTTError(_UNKNOWN_DELIVERY) from None


def _publish_with_owned_connection(
    target: MQTTTarget, message: str, *, retain: bool, allow_files: bool
) -> None:
    owner = MQTTPublisher(target, allow_access_to_file_system=allow_files)
    try:
        owner.publish(target, message, retain=retain)
    except Exception as error:
        _close_after_publish(owner, disposition=str(error))
        raise
    except BaseException:
        owner.close()
        raise
    _close_after_publish(owner, disposition=f"{_PUBLISHED}.")


def _close_after_publish(owner: MQTTPublisher, *, disposition: str) -> None:
    # A cleanup error alone would hide whether the single attempt was delivered.
    try:
        owner.close()
    except MQTTError as error:
        raise MQTTError(f"{disposition} {error}") from None


class _MQTTParams(BlockParams):
    host: StrictStr | Ref(STRING_KIND) = Field(
        description="MQTT broker host.", examples=["broker.example"]
    )
    port: StrictInt | Ref(INTEGER_KIND) = Field(
        default=1883,
        ge=1,
        le=65535,
        description="Broker TCP port; set TLS port explicitly.",
    )
    topic: StrictStr | Ref(STRING_KIND) = Field(
        description="MQTT topic or subscription filter.", examples=["factory/state"]
    )
    username: Optional[StrictStr] | Ref(STRING_KIND) = Field(
        default=None, description="Optional broker authentication user."
    )
    password: Optional[StrictStr] | Ref(SECRET_KIND, STRING_KIND) = Field(
        default=None, description="Broker Secret; requires username."
    )
    qos: StrictInt | Ref(INTEGER_KIND) = Field(
        default=0, ge=0, le=2, description="MQTT QoS, 0 through 2."
    )
    timeout: float | Ref(FLOAT_KIND, INTEGER_KIND) = Field(
        default=0.5,
        gt=0,
        allow_inf_nan=False,
        description="Individual readiness/acknowledgement timeout; DNS and cleanup join are not bounded.",
    )
    encryption: Literal["none", "tls"] | Ref(STRING_KIND) = Field(
        default="none", description="TLS verifies broker certificate and hostname."
    )
    ca_certificate_path: Optional[StrictStr] | Ref(STRING_KIND) = Field(
        default=None,
        description="Custom CA file; requires host file-system permission.",
    )


class MQTTReaderBlock(Block):
    """Read one buffered message from a borrowed ``mqtt_subscription`` owner."""

    type = READER_BLOCK_TYPE
    metadata = {
        "name": "MQTT Reader",
        "port_of": "roboflow_enterprise/mqtt_reader@v1",
        "section": "industrial",
    }
    outputs = {
        "value": Output(
            STRING_KIND, description="UTF-8 value, or None before first message."
        ),
        "payload": Output(
            DICTIONARY_KIND, description="Parsed JSON object; otherwise empty."
        ),
        "topic": Output(STRING_KIND, description="Actual received topic, or None."),
        "is_new": Output(BOOLEAN_KIND, description="This run consumed a new message."),
        "error_status": ERROR_STATUS_OUTPUT,
        "error_message": Output(STRING_KIND, description="Failure text, or None."),
    }

    class Params(_MQTTParams):
        read_mode: Literal["latest", "sequential"] = Field(
            default="latest",
            description="Latest message or next unread; overflow drops oldest (1/1000 slots).",
        )
        client_id: Optional[StrictStr] | Ref(STRING_KIND) = Field(
            default=None,
            description="Stable unique id enables persistent session; requires QoS 1/2.",
        )
        disable_reader: StrictBool | Ref(BOOLEAN_KIND) = Field(
            default=False,
            description="Skip before revealing credentials or accessing the host owner.",
        )

    def __init__(
        self,
        *,
        mqtt_subscription: Optional[MQTTSubscription] = None,
        allow_access_to_file_system: bool = False,
    ):
        self._subscription = mqtt_subscription
        self._allow_files = allow_access_to_file_system

    @classmethod
    def discover_work_operations(cls, params: BlockParams):
        """Declare broker access for workload discovery."""
        return [WorkOperation.EXTERNAL_REQUEST]

    def run(
        self,
        host: str,
        topic: str,
        port: int = 1883,
        username: Optional[str] = None,
        password: Optional[Union[str, Secret]] = None,
        qos: int = 0,
        timeout: float = 0.5,
        encryption: str = "none",
        ca_certificate_path: Optional[str] = None,
        read_mode: str = "latest",
        client_id: Optional[str] = None,
        disable_reader: bool = False,
    ):
        """Read from the explicit host owner using the declared step parameters.

        Args:
            host: Broker host.
            topic: Subscription filter.
            port: Broker port.
            username: Optional authentication user.
            password: Secret or literal credential.
            qos: Subscription QoS.
            timeout: Individual readiness timeout.
            encryption: TLS selection.
            ca_certificate_path: Optional permitted CA file.
            read_mode: Latest or sequential consumption.
            client_id: Optional persistent session id.
            disable_reader: Skip before credential access.

        Returns:
            Reader outputs, including observable connection/configuration failures.

        Raises:
            SecretUnavailableError: If the credential was restored without a value.
        """
        empty = {"value": None, "payload": {}, "topic": None, "is_new": False}
        if disable_reader:
            return {**empty, "error_status": False, "error_message": None}
        if environment.GCP_SERVERLESS or environment.LAMBDA:
            return {
                **empty,
                "error_status": True,
                "error_message": "MQTT Reader is unavailable on the hosted platform.",
            }
        secret = to_secret(password, name="password")
        revealed = reveal_secrets(secret)
        try:
            target = _validated_target(
                MQTTTarget(
                    host,
                    port,
                    topic,
                    username,
                    secret,
                    qos,
                    timeout,
                    encryption,
                    ca_certificate_path,
                    client_id,
                    read_mode,
                ),
                reader=True,
                allow_files=self._allow_files,
            )
            if self._subscription is None:
                raise MQTTError(
                    "MQTT Reader requires an explicit host-owned mqtt_subscription resource."
                )
            result = self._subscription.read(target)
        except Exception as error:
            return {
                **empty,
                "error_status": True,
                "error_message": revealed.scrub(str(error)),
            }
        return result


class MQTTWriterBlock(Block):
    """Publish through a borrowed writer or a fully operation-owned default."""

    type = WRITER_BLOCK_TYPE
    metadata = {
        "name": "MQTT Writer",
        "port_of": "roboflow_enterprise/mqtt_writer_sink@v1",
        "section": "industrial",
    }
    outputs = {"error_status": ERROR_STATUS_OUTPUT, "message": MESSAGE_OUTPUT}

    class Params(_MQTTParams):
        message: StrictStr | Ref(STRING_KIND) = Field(
            description="Message encoded as UTF-8 by paho.",
            examples=['{"state":"ready"}'],
        )
        retain: StrictBool | Ref(BOOLEAN_KIND) = Field(
            default=False, description="Retain message at broker."
        )
        fail_fast: StrictBool | Ref(BOOLEAN_KIND) = Field(
            default=False,
            description="Raise on inline failures; background failures remain host effect outcomes.",
        )
        fire_and_forget: StrictBool | Ref(BOOLEAN_KIND) = Field(
            default=False,
            description="Queue on host BackgroundEffects; without a runner execute inline.",
        )
        disable_sink: StrictBool | Ref(BOOLEAN_KIND) = Field(
            default=False, description="Skip before credentials, driver or effects."
        )

    def __init__(
        self,
        *,
        mqtt_writer: Optional[MQTTWriter] = None,
        sink_effects: Optional[BackgroundEffects] = None,
        disable_sinks: bool = False,
        allow_access_to_file_system: bool = False,
    ):
        self._writer = mqtt_writer
        self._effects = sink_effects
        self._disabled = disable_sinks
        self._allow_files = allow_access_to_file_system

    @classmethod
    def discover_work_operations(cls, params: BlockParams):
        """Declare broker access for workload discovery."""
        return [WorkOperation.EXTERNAL_REQUEST]

    def run(
        self,
        host: str,
        topic: str,
        message: str,
        port: int = 1883,
        username: Optional[str] = None,
        password: Optional[Union[str, Secret]] = None,
        qos: int = 0,
        retain: bool = False,
        timeout: float = 0.5,
        encryption: str = "none",
        ca_certificate_path: Optional[str] = None,
        fail_fast: bool = False,
        fire_and_forget: bool = False,
        disable_sink: bool = False,
    ):
        """Publish using the declared immutable operation parameters.

        Args:
            host: Broker host.
            topic: Publish destination.
            message: UTF-8 payload string.
            port: Broker port.
            username: Optional authentication user.
            password: Secret or literal credential.
            qos: Publish QoS.
            retain: Broker retention flag.
            timeout: Individual connection/publish timeout.
            encryption: TLS selection.
            ca_certificate_path: Optional permitted CA file.
            fail_fast: Raise for inline failures.
            fire_and_forget: Admit to host effects when supplied.
            disable_sink: Skip before resource acquisition.

        Returns:
            error_status and message. Scheduled never means delivered.

        Raises:
            SecretUnavailableError: When credentials have no value here.
            MQTTError: On inline failure with fail_fast enabled.
            RuntimeError: If the effects resource is closed.
        """
        if self._disabled or disable_sink:
            return {
                "error_status": False,
                "message": disabled_sink_message(by_policy=self._disabled),
            }
        secret = to_secret(password, name="password")
        revealed = reveal_secrets(secret)
        try:
            target = _validated_target(
                MQTTTarget(
                    host,
                    port,
                    topic,
                    username,
                    secret,
                    qos,
                    timeout,
                    encryption,
                    ca_certificate_path,
                ),
                reader=False,
                allow_files=self._allow_files,
            )
            if not isinstance(message, str) or type(retain) is not bool:
                raise MQTTError("MQTT message must be a string and retain a boolean.")
            message.encode("utf-8")
        except (MQTTError, UnicodeError) as error:
            text = revealed.scrub(str(error))
            if fail_fast:
                raise MQTTError(text) from None
            return {"error_status": True, "message": text}
        # Freeze references before admission; default acquisition occurs in effect.
        writer, allow_files = self._writer, self._allow_files

        def effect():
            try:
                if writer is not None:
                    writer.publish(target, message, retain=retain)
                else:
                    _publish_with_owned_connection(
                        target, message, retain=retain, allow_files=allow_files
                    )
            except Exception as error:
                return True, revealed.scrub(str(error))
            return False, _PUBLISHED

        error_status, text = deliver(
            effect,
            fire_and_forget=fire_and_forget,
            sink_effects=self._effects,
            label=self.type,
        )
        if text == BACKGROUND_MESSAGE and not error_status:
            text = "MQTT publish scheduled; delivery not yet confirmed"
        if fail_fast and error_status:
            raise MQTTError(text) from None
        return {"error_status": error_status, "message": text}

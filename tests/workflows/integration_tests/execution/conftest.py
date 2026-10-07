import os
import os.path
import socket
import ssl
import tempfile
import threading
from collections import deque
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Dict, Generator, Optional, Tuple

import cv2
import numpy as np
import pytest

ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "assets"))
ROCK_PAPER_SCISSORS_ASSETS = os.path.join(ASSETS_DIR, "rock_paper_scissors")

DUMMY_SECRET_ENV_VARIABLE = "DUMMY_SECRET"
os.environ[DUMMY_SECRET_ENV_VARIABLE] = "this-is-not-a-real-secret"


@pytest.fixture(scope="function")
def crowd_image() -> np.ndarray:
    return cv2.imread(os.path.join(ASSETS_DIR, "crowd.jpg"))


@pytest.fixture(scope="function")
def license_plate_image() -> np.ndarray:
    return cv2.imread(os.path.join(ASSETS_DIR, "license_plate.jpg"))


@pytest.fixture(scope="function")
def dogs_image() -> np.ndarray:
    return cv2.imread(os.path.join(ASSETS_DIR, "dogs.jpg"))


@pytest.fixture(scope="function")
def car_image() -> np.ndarray:
    return cv2.imread(os.path.join(ASSETS_DIR, "car.jpg"))


@pytest.fixture(scope="function")
def red_image() -> np.ndarray:
    return cv2.imread(os.path.join(ASSETS_DIR, "red_image.png"))


@pytest.fixture(scope="function")
def fruit_image() -> np.ndarray:
    return cv2.imread(os.path.join(ASSETS_DIR, "multi-fruit.jpg"))


@pytest.fixture(scope="function")
def multi_line_text_image() -> np.ndarray:
    return cv2.imread(os.path.join(ASSETS_DIR, "multi_line_text.jpg"))


@pytest.fixture(scope="function")
def stitch_left_image() -> np.ndarray:
    return cv2.imread(os.path.join(ASSETS_DIR, "stitch", "v_left.jpeg"))


@pytest.fixture(scope="function")
def stitch_right_image() -> np.ndarray:
    return cv2.imread(os.path.join(ASSETS_DIR, "stitch", "v_right.jpeg"))


@pytest.fixture(scope="function")
def left_scissors_right_paper() -> np.ndarray:
    return cv2.imread(
        os.path.join(ROCK_PAPER_SCISSORS_ASSETS, "left_scissors_right_paper.jpg")
    )


@pytest.fixture(scope="function")
def left_rock_right_paper() -> np.ndarray:
    return cv2.imread(
        os.path.join(ROCK_PAPER_SCISSORS_ASSETS, "left_rock_right_paper.jpg")
    )


@pytest.fixture(scope="function")
def left_rock_right_rock() -> np.ndarray:
    return cv2.imread(
        os.path.join(ROCK_PAPER_SCISSORS_ASSETS, "left_rock_right_rock.jpg")
    )


@pytest.fixture(scope="function")
def left_scissors_right_scissors() -> np.ndarray:
    return cv2.imread(
        os.path.join(ROCK_PAPER_SCISSORS_ASSETS, "left_scissors_right_scissors.jpg")
    )


@pytest.fixture(scope="function")
def empty_directory() -> Generator[str, None, None]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


def _numpy_image_as_tensor_input(image: np.ndarray):
    """Convert a BGR HWC numpy test image into the tensor-input form producers
    submit: CHW RGB uint8 torch.Tensor on WORKFLOWS_IMAGE_TENSOR_DEVICE.
    Delegates to the established `tensor_input_utils.numpy_image_as_tensor`
    helper (same conversion + correct device placement); grayscale handled
    here since the helper only covers 3-channel fixtures."""
    import torch

    if image.ndim == 2:
        from inference.core.env import WORKFLOWS_IMAGE_TENSOR_DEVICE

        return (
            torch.from_numpy(np.ascontiguousarray(image).copy())
            .unsqueeze(0)
            .to(WORKFLOWS_IMAGE_TENSOR_DEVICE)
        )
    from tests.workflows.integration_tests.execution.tensor_input_utils import (
        numpy_image_as_tensor,
    )

    return numpy_image_as_tensor(image)


@pytest.fixture(
    scope="function",
    params=["numpy-input", "tensor-input"],
    ids=["numpy-input", "tensor-input"],
)
def image_as_workflow_input(request):
    """_TENSOR_ONLY input hardening: every image runtime parameter must work
    submitted BOTH as np.ndarray (the historical test path, lazy numpy->tensor
    materialisation) AND as torch.Tensor (the producer path, exercising the
    deserializer's tensor arm and tensor-origin lazy numpy materialisation).

    Usage in a tensor-only test: add this fixture and wrap each image input:
    ``runtime_parameters={"image": [image_as_workflow_input(crowd_image)]}``.
    """
    if request.param == "numpy-input":
        return lambda image: image
    return _numpy_image_as_tensor_input


def bool_env(val):
    if isinstance(val, bool):
        return val
    return val.lower() in ["true", "1", "t", "y", "yes"]


@pytest.fixture(scope="function")
def face_image() -> np.ndarray:
    return cv2.imread(os.path.join(ASSETS_DIR, "face.jpeg"))


# Below taken from https://github.com/eclipse-paho/paho.mqtt.python/blob/d45de3737879cfe7a6acc361631fa5cb1ef584bb/tests/testsupport/broker.py
# and extended with SUBSCRIBE/SUBACK, PINGREQ/PINGRESP, retained messages and
# broker-initiated PUBLISH so subscriber blocks can be tested as well.
MQTT_CONNECT = 0x10
MQTT_PUBLISH = 0x30
MQTT_PUBACK = 0x40
MQTT_PUBREC = 0x50
MQTT_PUBREL = 0x60
MQTT_PUBCOMP = 0x70
MQTT_SUBSCRIBE = 0x80
MQTT_SUBACK = 0x90
MQTT_PINGREQ = 0xC0
MQTT_PINGRESP = 0xD0
MQTT_DISCONNECT = 0xE0
MQTT_SUBACK_FAILURE = 0x80


def _encode_remaining_length(length: int) -> bytes:
    encoded = b""
    while True:
        byte, length = length % 128, length // 128
        if length > 0:
            byte |= 0x80
        encoded += bytes([byte])
        if length == 0:
            return encoded


def _split_mqtt_packets(buffer: bytes):
    """Split a byte buffer into complete MQTT control packets; returns (packets, rest)."""
    packets = []
    while buffer:
        length, multiplier, index = 0, 1, 1
        while True:
            if index >= len(buffer):
                return packets, buffer
            byte = buffer[index]
            length += (byte & 0x7F) * multiplier
            multiplier *= 128
            index += 1
            if not byte & 0x80:
                break
        end = index + length
        if end > len(buffer):
            return packets, buffer
        packets.append(buffer[:end])
        buffer = buffer[end:]
    return packets, buffer


def mqtt_topic_matches(topic_filter: str, topic: str) -> bool:
    """MQTT topic-filter matching with `+` (one level) and `#` (rest) wildcards."""
    filter_levels = topic_filter.split("/")
    topic_levels = topic.split("/")
    for index, level in enumerate(filter_levels):
        if level == "#":
            return True
        if index >= len(topic_levels):
            return False
        if level != "+" and level != topic_levels[index]:
            return False
    return len(filter_levels) == len(topic_levels)


class _Session:
    """What a broker keeps for one client id: subscriptions, an offline queue and
    the connection currently bound to it (None while the client is away)."""

    def __init__(self, client_id: str, clean: bool):
        self.client_id = client_id
        self.clean = clean
        self.subscriptions: list = []
        self.queue: deque = deque()
        self.connection: Optional["_Connection"] = None

    def matches(self, topic: str) -> Optional[int]:
        """Highest granted QoS among the subscriptions matching `topic`, or None."""
        granted = [
            qos
            for topic_filter, qos in self.subscriptions
            if mqtt_topic_matches(topic_filter, topic)
        ]
        return max(granted) if granted else None


class _Connection:
    """Per-connection state: the socket, its receive buffer and packet counter."""

    def __init__(self, sock):
        self.sock = sock
        self.buffer = b""
        self.connack_sent = False
        self.stop = False
        self.next_packet_id = 1
        self.session: Optional[_Session] = None
        self.send_lock = threading.Lock()

    def close(self) -> None:
        self.stop = True
        sock, self.sock = self.sock, None
        if sock is None:
            return
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        sock.close()


class FakeMQTTBroker:
    """A minimal MQTT 3.1.1 broker for tests.

    By default it serves one connection per `start()` call, answers CONNECT,
    SUBSCRIBE and PINGREQ, re-sends retained messages on subscription and
    records everything else (including a client's PUBLISH, which is never
    acknowledged) in `messages`.

    `session_aware=True` adds what the persistent-session tests need: CONNECT is
    parsed (client id, clean-session flag), sessions are kept per id in
    `sessions`, a returning id gets CONNACK "session present" and its queued
    messages, a second connection with a live id evicts the first, a client's
    PUBLISH is acknowledged and routed to subscribers (queued for an offline
    persistent session when published at QoS 1 or 2) and `serve()` accepts any
    number of concurrent connections. Delivery is capped at the granted QoS, and
    `suback_reason_code` is that cap (0 by default), so a persistent-session test
    passes `suback_reason_code=2` for the broker to queue anything.
    """

    def __init__(
        self,
        connack_reason_code: int = 0,
        listening: bool = True,
        suback_reason_code: int = 0,
        keep_serving: bool = False,
        tls_context: Optional[ssl.SSLContext] = None,
        session_aware: bool = False,
    ):
        # Bind to "localhost" for maximum performance, as described in:
        # http://docs.python.org/howto/sockets.html#ipc
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.host = "localhost"
        sock.bind((self.host, 0))
        self.port = sock.getsockname()[1]
        self.messages = []
        self.messages_count_to_wait_for = 2
        self.connack_reason_code = connack_reason_code
        # granted QoS returned in SUBACK; 0x80 means the subscription is refused
        self.suback_reason_code = suback_reason_code
        # keep_serving: ignore recv timeouts and the message count, serve until
        # finish() / drop_connection(); needed for subscriber tests
        self.keep_serving = keep_serving
        # tls_context: a server-side SSLContext; the accepted connection is
        # wrapped with it before any MQTT packet is read
        self.tls_context = tls_context
        self.session_aware = session_aware
        self.handshake_failures = 0
        self.subscriptions = []
        self.retained = {}
        self.connections_accepted = 0
        # session_aware only: (client_id, clean_session) per CONNECT, in order
        self.connects = []
        # session_aware only: live and remembered sessions by client id
        self.sessions: Dict[str, _Session] = {}
        self._anonymous_sessions = 0
        self._connections: list = []
        self._closed = False
        self._lock = threading.RLock()

        sock.settimeout(5)

        self._sock = sock
        # bound but not listening: connection attempts are refused until listen()
        if listening:
            self.listen()

    def listen(self):
        self._sock.listen(5)

    # -- accepting -----------------------------------------------------------

    def _accept(self) -> Optional[_Connection]:
        sock = self._sock
        if sock is None:
            raise ValueError("Socket is not open")
        try:
            conn, address = sock.accept()
        except OSError:
            # finish() closed the listening socket while no client had
            # connected (a test whose block fails before connecting), or the
            # accept timed out
            return None
        conn.settimeout(1)
        if self.tls_context is not None:
            try:
                conn = self.tls_context.wrap_socket(conn, server_side=True)
            except (ssl.SSLError, OSError):
                # the client refused our certificate or never spoke TLS:
                # treat it like a closed connection
                self.handshake_failures += 1
                conn.close()
                return None
        connection = _Connection(conn)
        with self._lock:
            self._connections.append(connection)
            self.connections_accepted += 1
        return connection

    def start(self):
        """Accept one connection and serve it in the calling thread."""
        if self._sock is None:
            raise ValueError("Socket is not open")
        if any(c.sock is not None for c in self._connections):
            raise ValueError("Connection is already open")
        connection = self._accept()
        if connection is None:
            return
        self._serve_connection(connection)

    def serve(self):
        """Accept connections until finish(), each served on its own thread."""
        while not self._closed:
            connection = self._accept()
            if connection is None:
                continue
            threading.Thread(
                target=self._serve_connection, args=(connection,), daemon=True
            ).start()

    def _serve_connection(self, connection: _Connection) -> None:
        try:
            while (
                not self._closed
                and not connection.stop
                and (
                    self.keep_serving
                    or len(self.messages) < self.messages_count_to_wait_for
                    or not connection.connack_sent
                )
            ):
                try:
                    chunk = self._receive(connection, 1000)
                except socket.timeout:
                    if self.keep_serving:
                        continue
                    raise
                except (OSError, ValueError):
                    # connection closed, possibly from the test thread
                    break
                print(f"Received {chunk}")
                if not chunk:
                    break
                # exact byte: CONNECT's flags nibble is 0; a TLS ClientHello starts 0x16
                if (
                    not connection.connack_sent
                    and not connection.buffer
                    and chunk[0] != MQTT_CONNECT
                ):
                    # not an MQTT CONNECT (e.g. a TLS ClientHello against this plain
                    # broker): a real broker drops the connection on the protocol
                    # error, which is what lets the client fail fast
                    break
                connection.buffer += chunk
                packets, connection.buffer = _split_mqtt_packets(connection.buffer)
                try:
                    for packet in packets:
                        if self._handle_packet(connection, packet):
                            connection.connack_sent = True
                except (OSError, ValueError):
                    break
        finally:
            self._release(connection)

    def _release(self, connection: _Connection) -> None:
        with self._lock:
            connection.close()
            if connection in self._connections:
                self._connections.remove(connection)
            session = connection.session
            if session is not None and session.connection is connection:
                # a clean session ends with its connection; a persistent one
                # stays, with its subscriptions and queue, until the id returns
                session.connection = None
                if session.clean:
                    self.sessions.pop(session.client_id, None)

    # -- packets -------------------------------------------------------------

    def _handle_packet(self, connection: _Connection, packet: bytes) -> bool:
        """Handle one control packet; returns True when a CONNACK was sent."""
        packet_type = packet[0] & 0xF0
        if packet_type == MQTT_CONNECT:
            self._handle_connect(connection, packet)
            return True
        if packet_type == MQTT_SUBSCRIBE:
            self._handle_subscribe(connection, packet)
            return False
        if packet_type == MQTT_PINGREQ:
            self._send(connection, bytes([MQTT_PINGRESP, 0]))
            return False
        if packet_type in (MQTT_PUBACK, MQTT_PUBREC, MQTT_PUBREL, MQTT_PUBCOMP):
            # QoS 1/2 handshakes, not messages: a broker-initiated QoS 2 PUBLISH
            # is answered with PUBREC and completed with PUBREL; a client's
            # PUBREL (its own QoS 2 publish) is completed with PUBCOMP
            if packet_type == MQTT_PUBREC:
                self._send(connection, bytes([MQTT_PUBREL | 0x02, 2]) + packet[2:4])
            if packet_type == MQTT_PUBREL:
                self._send(connection, bytes([MQTT_PUBCOMP, 2]) + packet[2:4])
            return False
        if packet_type == MQTT_DISCONNECT:
            return False
        self.messages.append(packet)
        if packet_type == MQTT_PUBLISH and self.session_aware:
            self._handle_publish(connection, packet)
        return False

    @staticmethod
    def _body(packet: bytes) -> bytes:
        """Strip the fixed header (type byte + remaining-length varint)."""
        index = 1
        while packet[index] & 0x80:
            index += 1
        return packet[index + 1 :]

    @staticmethod
    def _read_string(body: bytes, index: int) -> Tuple[str, int]:
        length = int.from_bytes(body[index : index + 2], "big")
        index += 2
        return body[index : index + length].decode("utf-8"), index + length

    def _handle_connect(self, connection: _Connection, packet: bytes) -> None:
        print("sending CONNACK")
        if not self.session_aware:
            self._send(connection, b"\x20\x02\x00" + bytes([self.connack_reason_code]))
            return
        body = self._body(packet)
        _, index = self._read_string(body, 0)  # protocol name
        index += 1  # protocol level
        clean_session = bool(body[index] & 0x02)
        index += 3  # connect flags + keepalive
        client_id, _ = self._read_string(body, index)
        self.connects.append((client_id, clean_session))
        if self.connack_reason_code != 0:
            self._send(connection, b"\x20\x02\x00" + bytes([self.connack_reason_code]))
            return
        with self._lock:
            if not client_id:
                # a broker assigns an id to an anonymous (clean) client
                self._anonymous_sessions += 1
                client_id = f"anonymous-{self._anonymous_sessions}"
            session = self.sessions.get(client_id)
            if session is not None and session.connection is not None:
                # a second connection with a live id: the broker drops the first
                session.connection.close()
                session.connection = None
            session_present = session is not None and not clean_session
            if not session_present:
                session = _Session(client_id, clean=clean_session)
                self.sessions[client_id] = session
            session.connection = connection
            connection.session = session
            backlog = list(session.queue)
            session.queue.clear()
        self._send(connection, b"\x20\x02" + bytes([int(session_present), 0]))
        # a real broker delivers the queued messages right after CONNACK, before
        # the client's SUBSCRIBE arrives
        for topic, payload, qos in backlog:
            self._publish_to(connection, topic, payload, qos=qos)

    def _handle_subscribe(self, connection: _Connection, packet: bytes) -> None:
        body = self._body(packet)
        packet_id = body[0:2]
        index = 2
        granted = []
        requested = []
        while index < len(body):
            topic_filter, index = self._read_string(body, index)
            requested_qos = body[index]
            index += 1
            requested.append((topic_filter, requested_qos))
            self.subscriptions.append((topic_filter, requested_qos))
            if self.suback_reason_code == MQTT_SUBACK_FAILURE:
                granted.append(MQTT_SUBACK_FAILURE)
            else:
                granted.append(min(requested_qos, self.suback_reason_code))
        suback = (
            bytes([MQTT_SUBACK])
            + _encode_remaining_length(2 + len(granted))
            + packet_id
            + bytes(granted)
        )
        self._send(connection, suback)
        if self.suback_reason_code == MQTT_SUBACK_FAILURE:
            return
        if connection.session is not None:
            with self._lock:
                for (topic_filter, _), granted_qos in zip(requested, granted):
                    connection.session.subscriptions = [
                        s
                        for s in connection.session.subscriptions
                        if s[0] != topic_filter
                    ] + [(topic_filter, granted_qos)]
        for topic_filter, _ in requested:
            for topic, payload in list(self.retained.items()):
                if mqtt_topic_matches(topic_filter, topic):
                    self._publish_to(connection, topic, payload, retain=True)

    def _handle_publish(self, connection: _Connection, packet: bytes) -> None:
        """Acknowledge a client's PUBLISH and route it to the subscribers."""
        flags = packet[0] & 0x0F
        qos = (flags >> 1) & 0x03
        retain = bool(flags & 0x01)
        body = self._body(packet)
        topic, index = self._read_string(body, 0)
        packet_id = b""
        if qos > 0:
            packet_id = body[index : index + 2]
            index += 2
        payload = body[index:]
        if qos == 1:
            self._send(connection, bytes([MQTT_PUBACK, 2]) + packet_id)
        elif qos == 2:
            # completed with PUBCOMP when the client's PUBREL arrives
            self._send(connection, bytes([MQTT_PUBREC, 2]) + packet_id)
        if retain:
            self.retained[topic] = payload
        self._route(topic, payload, qos)

    # -- publishing ----------------------------------------------------------

    def publish(
        self, topic: str, payload: bytes, retain: bool = False, qos: int = 0
    ) -> None:
        """Publish from the broker side: to the connected client, or, when
        session aware, to every matching subscriber (queued for an offline
        persistent session when `qos` is 1 or 2)."""
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        if self.session_aware:
            if retain:
                self.retained[topic] = payload
            self._route(topic, payload, qos, retain=retain)
            return
        connection = self._current_connection()
        if connection is None:
            raise ValueError("Connection is not open")
        self._publish_to(connection, topic, payload, retain=retain, qos=qos)

    def _route(
        self, topic: str, payload: bytes, qos: int, retain: bool = False
    ) -> None:
        with self._lock:
            targets = []
            for session in list(self.sessions.values()):
                granted = session.matches(topic)
                if granted is None:
                    continue
                effective_qos = min(qos, granted)
                if session.connection is not None:
                    targets.append((session.connection, effective_qos))
                elif not session.clean and effective_qos > 0:
                    # the broker queues only QoS 1 and 2 for an offline session
                    session.queue.append((topic, payload, effective_qos))
        for connection, effective_qos in targets:
            try:
                self._publish_to(
                    connection, topic, payload, retain=retain, qos=effective_qos
                )
            except (OSError, ValueError):
                pass

    def _publish_to(
        self,
        connection: _Connection,
        topic: str,
        payload: bytes,
        retain: bool = False,
        qos: int = 0,
    ) -> None:
        """Send a PUBLISH packet from the broker to one client."""
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        topic_bytes = topic.encode("utf-8")
        variable_header = len(topic_bytes).to_bytes(2, "big") + topic_bytes
        if qos > 0:
            with connection.send_lock:
                packet_id = connection.next_packet_id
                connection.next_packet_id = packet_id % 65535 + 1
            variable_header += packet_id.to_bytes(2, "big")
        body = variable_header + payload
        header = bytes([MQTT_PUBLISH | (qos << 1) | int(retain)])
        self._send(connection, header + _encode_remaining_length(len(body)) + body)

    # -- lifecycle -----------------------------------------------------------

    def _current_connection(self) -> Optional[_Connection]:
        with self._lock:
            live = [c for c in self._connections if c.sock is not None]
        return live[-1] if live else None

    def drop_connection(self) -> None:
        """Close the (most recent) client connection so a later start() or the
        serving loop can accept a new one."""
        connection = self._current_connection()
        if connection is not None:
            connection.close()

    def finish(self):
        self._closed = True
        with self._lock:
            connections = list(self._connections)
        for connection in connections:
            connection.close()

        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def receive_packet(self, num_bytes):
        connection = self._current_connection()
        if connection is None:
            raise ValueError("Connection is not open")
        return self._receive(connection, num_bytes)

    @staticmethod
    def _receive(connection: _Connection, num_bytes: int) -> bytes:
        # capture locally: drop_connection()/finish() clear the attribute
        # from another thread
        sock = connection.sock
        if sock is None:
            raise ValueError("Connection is not open")
        return sock.recv(num_bytes)

    @staticmethod
    def _send(connection: _Connection, data: bytes) -> None:
        with connection.send_lock:
            sock = connection.sock
            if sock is None:
                raise ValueError("Connection is not open")
            sock.sendall(data)


def _write_test_certificate(tmp_path, name, issuer=None, dns_names=()):
    """Self-signed CA (no issuer) or a leaf signed by `issuer`, written as PEM.

    Mirrors `certificate()` in tests/inference/unit_tests/core/interfaces/http/
    test_mtls_enforcement.py; the leaf carries subjectAltName entries so paho's
    hostname verification accepts `localhost`.
    """
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name)])
    now = datetime.now(timezone.utc)
    builder = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer[0].subject if issuer else subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(days=1))
        .add_extension(
            x509.BasicConstraints(ca=issuer is None, path_length=None), critical=True
        )
    )
    if dns_names:
        builder = builder.add_extension(
            x509.SubjectAlternativeName(
                [x509.DNSName(dns_name) for dns_name in dns_names]
            ),
            critical=False,
        )
    cert = builder.sign(issuer[1] if issuer else key, hashes.SHA256())
    cert_path, key_path = tmp_path / (name + ".pem"), tmp_path / (name + ".key")
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return cert, key, cert_path, key_path


@pytest.fixture(scope="function")
def mqtt_test_certificates(tmp_path):
    """A throwaway CA and a `localhost` server certificate for TLS broker tests.

    Yields `ca_path` (PEM the blocks are pointed at through `ca_certificate_path`)
    and `server_context` (what `FakeMQTTBroker(tls_context=...)` wraps with).
    """
    pytest.importorskip("cryptography")
    ca = _write_test_certificate(tmp_path, "mqtt-test-ca")
    server = _write_test_certificate(
        tmp_path, "mqtt-test-server", issuer=ca, dns_names=("localhost",)
    )
    server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_context.load_cert_chain(certfile=str(server[2]), keyfile=str(server[3]))
    return SimpleNamespace(ca_path=str(ca[2]), server_context=server_context)


@pytest.fixture(scope="function")
def fake_mqtt_broker():
    print("Setup broker")
    broker = FakeMQTTBroker()

    yield broker

    print("Teardown broker")
    broker.finish()

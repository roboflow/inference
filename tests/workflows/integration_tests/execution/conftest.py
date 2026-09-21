import os
import os.path
import socket
import ssl
import tempfile
import threading
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Generator, Optional

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


class FakeMQTTBroker:
    def __init__(
        self,
        connack_reason_code: int = 0,
        listening: bool = True,
        suback_reason_code: int = 0,
        keep_serving: bool = False,
        tls_context: Optional[ssl.SSLContext] = None,
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
        self.handshake_failures = 0
        self.subscriptions = []
        self.retained = {}
        self.connections_accepted = 0
        self._stop = False
        self._send_lock = threading.Lock()

        sock.settimeout(5)

        self._sock = sock
        self._conn = None
        # bound but not listening: connection attempts are refused until listen()
        if listening:
            self.listen()

    def listen(self):
        self._sock.listen(1)

    def start(self):
        if self._sock is None:
            raise ValueError("Socket is not open")
        if self._conn is not None:
            raise ValueError("Connection is already open")

        try:
            conn, address = self._sock.accept()
        except OSError:
            # finish() closed the listening socket while no client had
            # connected (a test whose block fails before connecting)
            return
        conn.settimeout(1)
        if self.tls_context is not None:
            try:
                conn = self.tls_context.wrap_socket(conn, server_side=True)
            except (ssl.SSLError, OSError):
                # the client refused our certificate or never spoke TLS:
                # treat it like a closed connection
                self.handshake_failures += 1
                conn.close()
                return
        self._conn = conn
        self.connections_accepted += 1
        self._stop = False
        connack_sent = False
        buffer = b""
        while not self._stop and (
            self.keep_serving
            or len(self.messages) < self.messages_count_to_wait_for
            or not connack_sent
        ):
            try:
                chunk = self.receive_packet(1000)
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
            if not connack_sent and not buffer and chunk[0] != MQTT_CONNECT:
                # not an MQTT CONNECT (e.g. a TLS ClientHello against this plain
                # broker): a real broker drops the connection on the protocol
                # error, which is what lets the client fail fast
                self.drop_connection()
                return
            buffer += chunk
            packets, buffer = _split_mqtt_packets(buffer)
            try:
                for packet in packets:
                    if self._handle_packet(packet):
                        connack_sent = True
            except (OSError, ValueError):
                break

    def _handle_packet(self, packet: bytes) -> bool:
        """Handle one control packet; returns True when a CONNACK was sent."""
        packet_type = packet[0] & 0xF0
        if packet_type == MQTT_CONNECT:
            print("sending CONNACK")
            self._send(b"\x20\x02\x00" + bytes([self.connack_reason_code]))
            return True
        if packet_type == MQTT_SUBSCRIBE:
            self._handle_subscribe(packet)
            return False
        if packet_type == MQTT_PINGREQ:
            self._send(bytes([MQTT_PINGRESP, 0]))
            return False
        if packet_type in (MQTT_PUBACK, MQTT_PUBREC, MQTT_PUBREL, MQTT_PUBCOMP):
            # QoS 1/2 handshakes, not messages: a broker-initiated QoS 2 PUBLISH
            # is answered with PUBREC and completed with PUBREL; a client's
            # PUBREL (its own QoS 2 publish) is completed with PUBCOMP
            if packet_type == MQTT_PUBREC:
                self._send(bytes([MQTT_PUBREL | 0x02, 2]) + packet[2:4])
            if packet_type == MQTT_PUBREL:
                self._send(bytes([MQTT_PUBCOMP, 2]) + packet[2:4])
            return False
        if packet_type == MQTT_DISCONNECT:
            return False
        self.messages.append(packet)
        return False

    def _handle_subscribe(self, packet: bytes) -> None:
        # skip fixed header (type byte + remaining-length varint)
        index = 1
        while packet[index] & 0x80:
            index += 1
        index += 1
        packet_id = packet[index : index + 2]
        index += 2
        granted = []
        while index < len(packet):
            topic_length = int.from_bytes(packet[index : index + 2], "big")
            index += 2
            topic_filter = packet[index : index + topic_length].decode("utf-8")
            index += topic_length
            requested_qos = packet[index]
            index += 1
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
        self._send(suback)
        if self.suback_reason_code == MQTT_SUBACK_FAILURE:
            return
        for topic_filter, _ in self.subscriptions[-len(granted) :]:
            for topic, payload in self.retained.items():
                if mqtt_topic_matches(topic_filter, topic):
                    self.publish(topic, payload, retain=True)

    def publish(
        self, topic: str, payload: bytes, retain: bool = False, qos: int = 0
    ) -> None:
        """Send a PUBLISH packet from the broker to the connected client."""
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        topic_bytes = topic.encode("utf-8")
        variable_header = len(topic_bytes).to_bytes(2, "big") + topic_bytes
        if qos > 0:
            variable_header += (1).to_bytes(2, "big")
        body = variable_header + payload
        header = bytes([MQTT_PUBLISH | (qos << 1) | int(retain)])
        self._send(header + _encode_remaining_length(len(body)) + body)

    def drop_connection(self) -> None:
        """Close the client connection so a later start() can accept a new one."""
        self._stop = True
        if self._conn is not None:
            try:
                self._conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._conn.close()
            self._conn = None

    def finish(self):
        self._stop = True
        if self._conn is not None:
            self._conn.close()
            self._conn = None

        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def receive_packet(self, num_bytes):
        # capture locally: drop_connection()/finish() clear the attribute
        # from another thread
        conn = self._conn
        if conn is None:
            raise ValueError("Connection is not open")

        packet_in = conn.recv(num_bytes)
        return packet_in

    def _send(self, data: bytes) -> None:
        with self._send_lock:
            conn = self._conn
            if conn is None:
                raise ValueError("Connection is not open")
            conn.sendall(data)


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

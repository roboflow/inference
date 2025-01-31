import os
import os.path
import socket
import tempfile
from typing import Generator

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


def bool_env(val):
    if isinstance(val, bool):
        return val
    return val.lower() in ["true", "1", "t", "y", "yes"]


@pytest.fixture(scope="function")
def face_image() -> np.ndarray:
    return cv2.imread(os.path.join(ASSETS_DIR, "face.jpeg"))


# Below taken from https://github.com/eclipse-paho/paho.mqtt.python/blob/d45de3737879cfe7a6acc361631fa5cb1ef584bb/tests/testsupport/broker.py
class FakeMQTTBroker:
    def __init__(self):
        # Bind to "localhost" for maximum performance, as described in:
        # http://docs.python.org/howto/sockets.html#ipc
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.host = "localhost"
        sock.bind((self.host, 0))
        self.port = sock.getsockname()[1]
        self.messages = []
        self.messages_count_to_wait_for = 2

        sock.settimeout(5)
        sock.listen(1)

        self._sock = sock
        self._conn = None

    def start(self):
        if self._sock is None:
            raise ValueError("Socket is not open")
        if self._conn is not None:
            raise ValueError("Connection is already open")

        (conn, address) = self._sock.accept()
        conn.settimeout(1)
        self._conn = conn
        while len(self.messages) < self.messages_count_to_wait_for:
            packet = self.receive_packet(1000)
            print(f"Received {packet}")
            if not packet:
                continue
            if packet.startswith(b"\x10"):
                print("sending CONNACK")
                self._conn.send(b"\x20\x02\x00\x00")
                continue
            self.messages.append(packet)

    def finish(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None

        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def receive_packet(self, num_bytes):
        if self._conn is None:
            raise ValueError("Connection is not open")

        packet_in = self._conn.recv(num_bytes)
        return packet_in


@pytest.fixture(scope="function")
def fake_mqtt_broker():
    print("Setup broker")
    broker = FakeMQTTBroker()

    yield broker

    print("Teardown broker")
    broker.finish()

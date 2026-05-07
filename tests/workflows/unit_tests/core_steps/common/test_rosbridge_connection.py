import sys
import types
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_roslibpy(monkeypatch):
    fake = types.ModuleType("roslibpy")

    class _FakeRos:
        def __init__(self, host, port, is_secure=False):
            self.host = host
            self.port = port
            self.is_secure = is_secure
            self.is_connected = False
            self.run_called = 0
            self.close_called = 0

        def run(self, timeout=None):
            self.is_connected = True
            self.run_called += 1

        def close(self):
            self.is_connected = False
            self.close_called += 1

    class _FakeTopic:
        def __init__(self, *args, **kwargs):
            pass

    fake.Ros = _FakeRos
    fake.Topic = _FakeTopic
    fake.Message = MagicMock()
    monkeypatch.setitem(sys.modules, "roslibpy", fake)
    yield fake


def _fresh_registry():
    from inference.core.workflows.core_steps.common.rosbridge import connection
    reg = connection.RosbridgeConnectionRegistry()
    return reg, connection


def test_normalize_message_type_ros1_to_ros2():
    from inference.core.workflows.core_steps.common.rosbridge.connection import (
        normalize_message_type,
    )
    assert normalize_message_type("std_msgs/String", 2) == "std_msgs/msg/String"
    assert normalize_message_type("std_msgs/msg/String", 2) == "std_msgs/msg/String"
    assert normalize_message_type("std_msgs/msg/String", 1) == "std_msgs/String"
    assert normalize_message_type("std_msgs/String", 1) == "std_msgs/String"


def test_acquire_creates_one_ros_per_endpoint(mock_roslibpy):
    reg, _ = _fresh_registry()
    h1 = reg.acquire("a", 9090, False)
    h2 = reg.acquire("a", 9090, False)
    assert h1.ros is h2.ros, "same endpoint must reuse the Ros instance"


def test_different_endpoints_get_different_ros(mock_roslibpy):
    reg, _ = _fresh_registry()
    h1 = reg.acquire("a", 9090, False)
    h2 = reg.acquire("b", 9090, False)
    assert h1.ros is not h2.ros


def test_release_closes_when_refcount_reaches_zero(mock_roslibpy):
    reg, _ = _fresh_registry()
    h1 = reg.acquire("a", 9090, False)
    h2 = reg.acquire("a", 9090, False)
    ros = h1.ros
    h1.release()
    assert ros.close_called == 0
    h2.release()
    assert ros.close_called == 1


def test_release_is_idempotent(mock_roslibpy):
    reg, _ = _fresh_registry()
    h = reg.acquire("a", 9090, False)
    h.release()
    h.release()  # second call must not error

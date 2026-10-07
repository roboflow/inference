"""Redis backend against a real, owned ``redis-server``: processes and failures."""

import os
import select
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest
from roboflow_workflows.execution_engine.v2.state import (
    MISSING,
    ManagedState,
    StateBackendError,
    StateOutcomeUnknownError,
)

# Keep the optional-dependency skip ahead of Redis-dependent imports below.
pytest.importorskip("redis")

from roboflow_workflows.execution_engine.v2.state.codec import (  # noqa: E402
    global_storage_key,
)
from roboflow_workflows.execution_engine.v2.state.redis import (  # noqa: E402
    RedisStateBackend,
)

from ._redis_server import OwnedRedisServer  # noqa: E402

_WORKER = Path(__file__).with_name("_redis_worker.py")


def _spawn_workers(redis_server, namespace, *, mode, processes, count):
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    workers = [
        subprocess.Popen(
            [
                sys.executable,
                "-B",
                str(_WORKER),
                redis_server.url,
                namespace,
                mode,
                str(count),
                f"worker-{number}",
            ],
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for number in range(processes)
    ]
    control = ManagedState(RedisStateBackend(redis_server.url), namespace=namespace)
    deadline = time.monotonic() + 60
    while control.global_.get("ready", 0) < processes:
        assert time.monotonic() < deadline, "workers did not start"
        time.sleep(0.01)
    control.global_.set("go", True)

    outputs = []
    for worker in workers:
        stdout, stderr = worker.communicate(timeout=120)
        assert worker.returncode == 0, stderr
        outputs.append(stdout.strip())
    control.backend.close()

    return outputs


def test_independent_processes_increment_exactly(redis_server, namespace):
    processes, count = 4, 500

    _spawn_workers(
        redis_server, namespace, mode="incr", processes=processes, count=count
    )

    state = ManagedState(RedisStateBackend(redis_server.url), namespace=namespace)
    assert state.global_.get("total") == processes * count
    assert state.for_source("cam_a").get("total") == processes * count


def test_independent_processes_compare_and_set_has_one_winner(redis_server, namespace):
    state = ManagedState(RedisStateBackend(redis_server.url), namespace=namespace)
    state.for_source("cam_a").set("machine", "idle")

    outputs = _spawn_workers(redis_server, namespace, mode="cas", processes=6, count=1)

    winners = [f"worker-{n}" for n, output in enumerate(outputs) if output == "won"]
    assert len(winners) == 1
    assert outputs.count("lost") == 5
    assert state.for_source("cam_a").get("machine") == winners[0]


def test_importing_state_does_not_import_redis():
    code = (
        "import sys\n"
        "import roboflow_workflows.execution_engine.v2.state as state\n"
        "state.ManagedState().global_.incr('k')\n"
        "assert 'redis' not in sys.modules, 'redis imported'\n"
    )
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))

    completed = subprocess.run(
        [sys.executable, "-B", "-c", code],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr


def test_unreachable_server_is_not_applied_and_hides_credentials():
    backend = RedisStateBackend("redis://user:hunter2@127.0.0.1:1/0", timeout=0.2)
    state = ManagedState(backend, namespace="n")

    with pytest.raises(StateBackendError) as raised:
        state.global_.incr("k")

    assert not isinstance(raised.value, StateOutcomeUnknownError)
    assert "not applied" in str(raised.value)
    assert "hunter2" not in str(raised.value)
    assert "hunter2" not in repr(backend)
    assert backend.address == "redis://127.0.0.1:1/0"


def test_lost_reply_is_unknown_and_never_retried(redis_server, namespace):
    proxy = _ReplyDroppingProxy(redis_server.socket_path)
    try:
        state = ManagedState(
            RedisStateBackend(f"unix://{proxy.path}", timeout=2.0), namespace=namespace
        )
        assert state.global_.incr("k") == 1

        proxy.drop_next_reply()
        with pytest.raises(StateOutcomeUnknownError, match="not retried"):
            state.global_.incr("k")
        assert state.global_.incr("k") == 3
    finally:
        proxy.close()

    direct = ManagedState(RedisStateBackend(redis_server.url), namespace=namespace)
    assert direct.global_.get("k") == 3


def test_stalled_server_times_out(redis_server, namespace):
    state = ManagedState(
        RedisStateBackend(redis_server.url, timeout=0.2), namespace=namespace
    )
    state.global_.set("k", 1)
    admin = redis_server.admin()

    admin.client_pause(600, all=False)
    started = time.monotonic()
    with pytest.raises(StateOutcomeUnknownError):
        state.global_.incr("k")
    elapsed = time.monotonic() - started
    admin.client_pause(600)
    with pytest.raises(StateBackendError) as raised:
        state.global_.get("k")
    time.sleep(0.7)

    assert elapsed < 0.5
    assert not isinstance(raised.value, StateOutcomeUnknownError)
    assert state.global_.get("k") in (1, 2)


def test_server_restart_reconnects_without_replaying(namespace):
    directory = Path(tempfile.mkdtemp(prefix="wf2s-", dir="/tmp"))
    server = OwnedRedisServer(directory)
    try:
        server.wait_ready()
        state = ManagedState(
            RedisStateBackend(server.url, timeout=0.5), namespace=namespace
        )
        assert state.global_.incr("k") == 1
        server.stop()

        with pytest.raises(StateBackendError) as raised:
            state.global_.incr("k")
        assert not isinstance(raised.value, StateOutcomeUnknownError)

        server = OwnedRedisServer(directory)
        server.wait_ready()
        assert state.global_.get("k", MISSING) is MISSING
        assert state.global_.incr("k") == 1
    finally:
        server.stop()
        shutil.rmtree(directory, ignore_errors=True)


def test_foreign_value_type_is_a_backend_error(redis_server, namespace):
    state = ManagedState(RedisStateBackend(redis_server.url), namespace=namespace)
    redis_server.admin().lpush(global_storage_key(namespace, "k"), "x")

    with pytest.raises(StateBackendError, match="WRONGTYPE"):
        state.global_.get("k")
    with pytest.raises(StateBackendError, match="WRONGTYPE"):
        state.global_.compare_and_set("k", MISSING, 1)


def test_closed_backend_rejects_operations(redis_server, namespace):
    backend = RedisStateBackend(redis_server.url)
    state = ManagedState(backend, namespace=namespace)
    state.global_.set("k", 1)

    backend.close()

    with pytest.raises(StateBackendError, match="closed"):
        state.global_.get("k")


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError):
        RedisStateBackend("127.0.0.1:6379")
    with pytest.raises(ValueError):
        RedisStateBackend("redis://127.0.0.1:6379/0", timeout=0)


class _ReplyDroppingProxy:
    """Unix-socket proxy that can swallow one Redis reply and cut the client.

    The command reaches Redis and runs; only its reply is lost, as when a
    network drops after the server applied a write.
    """

    def __init__(self, upstream: Path) -> None:
        self._directory = Path(tempfile.mkdtemp(prefix="wf2p-", dir="/tmp"))
        self.path = self._directory / "proxy.sock"
        self._upstream = upstream
        self._drop = threading.Event()
        self._closed = threading.Event()
        self._listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._listener.bind(str(self.path))
        self._listener.listen()
        self._listener.settimeout(0.05)
        self._thread = threading.Thread(target=self._accept, daemon=True)
        self._thread.start()

    def drop_next_reply(self) -> None:
        self._drop.set()

    def close(self) -> None:
        self._closed.set()
        self._thread.join(timeout=5)
        self._listener.close()
        shutil.rmtree(self._directory, ignore_errors=True)

    def _accept(self) -> None:
        while not self._closed.is_set():
            try:
                client, _ = self._listener.accept()
            except socket.timeout:
                continue
            upstream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            upstream.connect(str(self._upstream))
            threading.Thread(
                target=self._pump, args=(client, upstream), daemon=True
            ).start()

    def _pump(self, client: socket.socket, upstream: socket.socket) -> None:
        try:
            while not self._closed.is_set():
                readable, _, _ = select.select([client, upstream], [], [], 0.05)
                for ready in readable:
                    data = ready.recv(65536)
                    if not data:
                        return
                    if ready is client:
                        upstream.sendall(data)
                    elif self._drop.is_set():
                        self._drop.clear()
                        return
                    else:
                        client.sendall(data)
        finally:
            client.close()
            upstream.close()

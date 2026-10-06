import json
import socket
import socketserver
import threading
from http.server import BaseHTTPRequestHandler

import pytest

from inference_server.ops import docker_stats
from inference_server.ops import router as ops_router
from tests.unit_tests.ops.conftest import running_on_event_loop

NOT_CONFIGURED_BODY = (
    b'{"error":"Device statistics endpoint is not enabled.",'
    b'"hint":"Mount the Docker socket and point its location when running the docker '
    b"container to collect device stats "
    b"(i.e. `docker run ... -v /var/run/docker.sock:/var/run/docker.sock "
    b'-e DOCKER_SOCKET_PATH=/var/run/docker.sock ...`)."}'
)
FAILURE_BODY = b'{"error":"An error occurred while fetching container stats."}'
SOCKET_NAME = "docker.sock"


class _DockerHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.server.seen_paths.append(self.path)
        body = self.server.response_body
        self.send_response(self.server.response_status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:
        return None


class _DockerServer(socketserver.UnixStreamServer):
    response_status = 200
    response_body = b'{"cpu_stats": {"online_cpus": 4}}'

    def get_request(self):
        request, _ = super().get_request()

        return request, ("docker", 0)


@pytest.fixture
def docker_engine(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    server = _DockerServer(SOCKET_NAME, _DockerHandler)
    server.seen_paths = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def test_route_answers_legacy_hint_with_404_when_socket_path_is_unset(ops_client):
    response = ops_client().get("/device/stats")

    assert response.status_code == 404
    assert response.content == NOT_CONFIGURED_BODY


def test_route_answers_legacy_hint_with_500_when_socket_is_not_mounted(
    ops_client, tmp_path
):
    not_a_socket = tmp_path / "plain-file"
    not_a_socket.write_text("")

    missing = ops_client(DOCKER_SOCKET_PATH=str(tmp_path / "missing")).get(
        "/device/stats"
    )
    plain = ops_client(DOCKER_SOCKET_PATH=str(not_a_socket)).get("/device/stats")

    assert missing.status_code == 500
    assert missing.content == NOT_CONFIGURED_BODY
    assert plain.status_code == 500
    assert plain.content == NOT_CONFIGURED_BODY


def test_route_returns_docker_stats_for_this_container(ops_client, docker_engine):
    response = ops_client(DOCKER_SOCKET_PATH=SOCKET_NAME).get("/device/stats")

    assert response.status_code == 200
    assert response.content == b'{"stats":{"cpu_stats":{"online_cpus":4}}}'
    assert docker_engine.seen_paths == [
        f"/containers/{socket.gethostname()}/stats?stream=false"
    ]


def test_route_answers_fixed_text_when_docker_call_fails(ops_client, docker_engine):
    docker_engine.response_status = 500
    docker_engine.response_body = b"engine exploded: secret-detail"

    response = ops_client(DOCKER_SOCKET_PATH=SOCKET_NAME).get("/device/stats")

    assert response.status_code == 500
    assert response.content == FAILURE_BODY


def test_failed_docker_call_leaves_no_engine_text_in_the_log_buffer(
    ops_client, docker_engine, log_buffer
):
    docker_engine.response_status = 500
    docker_engine.response_body = b"engine exploded: secret-detail"

    ops_client(DOCKER_SOCKET_PATH=SOCKET_NAME).get("/device/stats")

    buffered = log_buffer()
    assert "Could not fetch container stats" in buffered
    assert "RuntimeError" in buffered
    assert "500" in buffered
    assert "secret-detail" not in buffered
    assert "Traceback" not in buffered


def _json_body_of_size(size: int) -> bytes:
    prefix, suffix = b'{"a":"', b'"}'

    return prefix + b"x" * (size - len(prefix) - len(suffix)) + suffix


def test_docker_answer_above_the_size_limit_is_a_failed_call(ops_client, docker_engine):
    docker_engine.response_body = _json_body_of_size(
        docker_stats.DOCKER_STATS_MAX_RESPONSE_BYTES + 1
    )

    response = ops_client(DOCKER_SOCKET_PATH=SOCKET_NAME).get("/device/stats")

    assert response.status_code == 500
    assert response.content == FAILURE_BODY


def test_docker_answer_at_the_size_limit_is_accepted(ops_client, docker_engine):
    docker_engine.response_body = _json_body_of_size(
        docker_stats.DOCKER_STATS_MAX_RESPONSE_BYTES
    )

    response = ops_client(DOCKER_SOCKET_PATH=SOCKET_NAME).get("/device/stats")

    assert response.status_code == 200


def test_route_answers_fixed_text_when_docker_answer_is_not_json(
    ops_client, docker_engine
):
    docker_engine.response_body = b"not json"

    response = ops_client(DOCKER_SOCKET_PATH=SOCKET_NAME).get("/device/stats")

    assert response.status_code == 500
    assert response.content == FAILURE_BODY


def test_route_runs_docker_call_off_the_event_loop(
    ops_client, docker_engine, monkeypatch
):
    seen = []

    def _fake_stats(docker_socket_path: str) -> dict:
        seen.append((docker_socket_path, running_on_event_loop()))

        return {"stats": {}}

    monkeypatch.setattr(ops_router, "get_container_stats", _fake_stats)

    response = ops_client(DOCKER_SOCKET_PATH=SOCKET_NAME).get("/device/stats")

    assert response.status_code == 200
    assert seen == [(SOCKET_NAME, False)]


def test_docker_call_uses_a_finite_timeout(monkeypatch):
    timeouts = []

    class _Socket:
        def __init__(self, *args) -> None:
            pass

        def settimeout(self, value: float) -> None:
            timeouts.append(value)

        def connect(self, path: str) -> None:
            raise socket.timeout("timed out")

        def close(self) -> None:
            pass

    monkeypatch.setattr(docker_stats.socket, "socket", _Socket)

    with pytest.raises(docker_stats.ContainerStatsError) as error:
        docker_stats.get_container_stats("/var/run/docker.sock")

    assert timeouts == [docker_stats.DOCKER_STATS_TIMEOUT_S]
    assert docker_stats.DOCKER_STATS_TIMEOUT_S == 10.0
    assert str(error.value) == "An error occurred while fetching container stats."


def test_socket_check_accepts_only_unix_sockets(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    plain_file = tmp_path / "plain-file"
    plain_file.write_text(json.dumps({}))
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(SOCKET_NAME)

    try:
        assert docker_stats.is_docker_socket_mounted(SOCKET_NAME) is True
        assert docker_stats.is_docker_socket_mounted(str(plain_file)) is False
        assert docker_stats.is_docker_socket_mounted("missing") is False
    finally:
        listener.close()

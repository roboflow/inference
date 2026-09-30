import asyncio
import io
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import Mock

import av
import pytest
from aiortc.mediastreams import MediaStreamError
from urllib3.connection import HTTPConnection
from urllib3.response import HTTPResponse

from inference.core.utils import mjpeg


@pytest.fixture
def mjpeg_server():
    frame = av.VideoFrame(16, 16, "yuvj420p")
    for plane in frame.planes:
        plane.update(bytes([128]) * plane.buffer_size)
    buffer = io.BytesIO()
    with av.open(buffer, "w", format="mjpeg") as output:
        stream = output.add_stream("mjpeg", rate=25)
        stream.width = stream.height = 16
        stream.pix_fmt = "yuvj420p"
        for packet in stream.encode(frame):
            output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    jpeg = buffer.getvalue()
    stopped = threading.Event()
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            calls.append((self.path, self.headers["Host"]))
            redirects = {
                "/redirect": "/stream",
                "/private": "http://127.0.0.1:9/private",
                "/file": "file:///tmp/secret.jpg",
                "/loop": "/loop",
            }
            if self.path in redirects:
                self.send_response(302)
                self.send_header("Location", redirects[self.path])
                self.end_headers()
                # The redirect body must never be drained before validation.
                stopped.wait(3)
                return
            if self.path == "/stall-headers":
                stopped.wait(3)
                return
            multipart = self.path == "/multipart"
            self.send_response(200)
            self.send_header(
                "Content-Type",
                (
                    "multipart/x-mixed-replace; boundary=frame"
                    if multipart
                    else "image/jpeg"
                ),
            )
            self.end_headers()
            if self.path == "/stall-body":
                stopped.wait(3)
                return
            if self.path == "/trickle":
                try:
                    while not stopped.wait(0.01):
                        self.wfile.write(b"x")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            if self.path == "/playlist":
                self.wfile.write(b"#EXTM3U\n#EXTINF:1,\nhttp://127.0.0.1:9/secret\n")
                return
            if multipart:
                part = (
                    b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(jpeg)).encode()
                    + b"\r\n\r\n"
                    + jpeg
                    + b"\r\n"
                )
                self.wfile.write(part * 30 + b"--frame--\r\n")
            else:
                self.wfile.write(jpeg * 30)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port, calls
    finally:
        stopped.set()
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize(
    "url",
    [
        "/tmp/secret.jpg",
        "file:///tmp/secret.jpg",
        "tcp://127.0.0.1:9",
        "concat:http://example.com/a|file:/tmp/secret.jpg",
        "http:///stream",
        "http://user:pass@camera.example:bad/stream",
        "http://camera.example\\@127.0.0.1/stream",
        "http://[::1",
    ],
)
def test_mjpeg_rejects_invalid_urls_before_connecting(url, monkeypatch):
    connect = Mock(side_effect=AssertionError("Must not connect"))
    monkeypatch.setattr(HTTPConnection, "connect", connect)
    with pytest.raises(RuntimeError, match="Failed to open MJPEG"):
        mjpeg.open_mjpeg_player(url)
    connect.assert_not_called()


@pytest.mark.parametrize(
    "host",
    [
        "127.0.0.1",
        "10.0.0.1",
        "169.254.169.254",
        "100.64.0.1",
        "[::1]",
        "[::ffff:127.0.0.1]",
        "224.0.0.1",
        "[ff02::1]",
        "localhost",
        "metadata.internal",
        "2130706433",
        "0x7f000001",
        "017700000001",
    ],
)
def test_mjpeg_blocks_non_public_destinations(host, monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **k: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))
        ],
    )
    connect = Mock(side_effect=AssertionError("Must not connect"))
    monkeypatch.setattr(HTTPConnection, "connect", connect)
    with pytest.raises(RuntimeError, match="URLAddressNotAllowedError"):
        mjpeg.open_mjpeg_player(f"http://{host}/stream")
    connect.assert_not_called()


def _route_public_camera_to_test_server(monkeypatch, port):
    original_resolve = socket.getaddrinfo
    resolutions = []

    def resolve(host, *args, **kwargs):
        if host == "camera.example":
            resolutions.append(host)
            ip = "8.8.8.8" if len(resolutions) == 1 else "127.0.0.1"
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port))]
        return original_resolve(host, *args, **kwargs)

    def connect_to_test_server(connection):
        # A hostname here would let a second DNS lookup change the target.
        assert connection.host == "8.8.8.8"
        return socket.create_connection(("127.0.0.1", port))

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    monkeypatch.setattr(HTTPConnection, "_new_conn", connect_to_test_server)
    return resolutions


@pytest.mark.parametrize("legacy_reader", [False, True])
@pytest.mark.parametrize("path", ["/stream", "/multipart"])
def test_mjpeg_decodes_public_stream_pins_dns_and_closes_resources(
    path, legacy_reader, mjpeg_server, monkeypatch
):
    if legacy_reader:
        monkeypatch.setattr(HTTPResponse, "read1", None)
    port, calls = mjpeg_server
    resolutions = _route_public_camera_to_test_server(monkeypatch, port)
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    closed = Mock()
    original_close = mjpeg.SSRFProtectedHTTPAdapter.close

    def close(adapter):
        closed()
        original_close(adapter)

    monkeypatch.setattr(mjpeg.SSRFProtectedHTTPAdapter, "close", close)
    player = mjpeg.open_mjpeg_player(f"http://camera.example{path}")

    async def receive():
        try:
            frame = await asyncio.wait_for(player.video.recv(), timeout=2)
            assert (frame.width, frame.height) == (16, 16)
        finally:
            player.video.stop()

    asyncio.run(receive())
    assert resolutions == ["camera.example"]
    assert calls == [(path, "camera.example")]
    closed.assert_called_once()


@pytest.mark.parametrize(
    "path, message",
    [("/private", "URLAddressNotAllowedError"), ("/file", "ValueError")],
)
def test_mjpeg_revalidates_redirect_destinations(
    path, message, mjpeg_server, monkeypatch
):
    port, calls = mjpeg_server
    _route_public_camera_to_test_server(monkeypatch, port)
    started = time.monotonic()
    with pytest.raises(RuntimeError, match=message):
        mjpeg.open_mjpeg_player(f"http://camera.example{path}")
    assert time.monotonic() - started < 1
    assert calls == [(path, "camera.example")]


def test_mjpeg_allows_explicit_private_camera_opt_in_and_relative_redirect(
    mjpeg_server,
):
    port, calls = mjpeg_server
    player = mjpeg.open_mjpeg_player(
        f"http://127.0.0.1:{port}/redirect", allow_non_global_addresses=True
    )
    player.video.stop()
    assert [path for path, _ in calls] == ["/redirect", "/stream"]


def test_mjpeg_caps_redirects(mjpeg_server):
    port, calls = mjpeg_server
    with pytest.raises(RuntimeError, match="ValueError"):
        mjpeg.open_mjpeg_player(
            f"http://127.0.0.1:{port}/loop", allow_non_global_addresses=True
        )
    assert len(calls) == mjpeg.MJPEG_MAX_REDIRECTS + 1


@pytest.mark.parametrize("path", ["/stall-headers", "/stall-body", "/trickle"])
def test_mjpeg_times_out_stalled_and_trickling_streams(path, mjpeg_server, monkeypatch):
    port, _ = mjpeg_server
    monkeypatch.setattr(mjpeg, "MJPEG_TIMEOUT_SECONDS", 0.15)
    started = time.monotonic()
    with pytest.raises(RuntimeError):
        mjpeg.open_mjpeg_player(
            f"http://127.0.0.1:{port}{path}", allow_non_global_addresses=True
        )
    assert time.monotonic() - started < 1


def test_mjpeg_does_not_autodetect_network_playlists(mjpeg_server, monkeypatch):
    port, calls = mjpeg_server
    connections = []
    original_connect = HTTPConnection._new_conn

    def connect(connection):
        connections.append(connection.port)
        assert connection.port == port
        return original_connect(connection)

    monkeypatch.setattr(HTTPConnection, "_new_conn", connect)
    player = mjpeg.open_mjpeg_player(
        f"http://127.0.0.1:{port}/playlist", allow_non_global_addresses=True
    )

    async def receive():
        try:
            with pytest.raises(MediaStreamError):
                await asyncio.wait_for(player.video.recv(), timeout=2)
        finally:
            player.video.stop()

    asyncio.run(receive())
    assert len(connections) == 1
    assert len(calls) == 1


def test_mjpeg_errors_do_not_expose_relative_url_credentials(monkeypatch):
    monkeypatch.setattr(
        mjpeg.SSRFProtectedHTTPAdapter,
        "send",
        Mock(
            side_effect=mjpeg.requests.ConnectionError(
                "Max retries: /camera?token=secret"
            )
        ),
    )
    with pytest.raises(RuntimeError) as error:
        mjpeg.open_mjpeg_player(
            "http://user:password@camera.example/camera?token=secret"
        )
    assert "secret" not in str(error.value)
    assert "password" not in str(error.value)
    assert "ConnectionError" in str(error.value)

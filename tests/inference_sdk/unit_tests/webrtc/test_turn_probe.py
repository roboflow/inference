"""Tests for picking a reachable TURN transport before aiortc sees the config."""

import asyncio
import time

import pytest
from aioice import stun
from aiortc import RTCConfiguration, RTCIceServer
from aiortc.rtcicetransport import connection_kwargs

from inference_sdk.webrtc import turn_probe

UDP = "turn:turn1.example.com:3478"
TCP = "turn:turn1.example.com:3478?transport=tcp"
TLS = "turns:turns1.example.com:443?transport=tcp"


def _config(*urls: str) -> RTCConfiguration:
    server = RTCIceServer(urls=list(urls), username="user", credential="secret")
    return RTCConfiguration(iceServers=[server])


def _aiortc_turn(config: RTCConfiguration) -> tuple:
    """What aiortc will actually use: (port, ssl, transport, username)."""
    kwargs = connection_kwargs(config.iceServers)
    return (
        kwargs["turn_server"][1],
        kwargs["turn_ssl"],
        kwargs["turn_transport"],
        kwargs["turn_username"],
    )


def _prefer(monkeypatch, config, reachable) -> RTCConfiguration:
    async def fake_probe(url: str) -> bool:
        if url in reachable:
            return True
        raise OSError("unreachable")

    monkeypatch.setattr(turn_probe, "probe_turn_url", fake_probe)
    return asyncio.run(turn_probe.prefer_reachable_turn(config))


@pytest.mark.parametrize(
    ("reachable", "expected"),
    [
        ({UDP, TCP, TLS}, (3478, False, "udp", "user")),
        ({TCP, TLS}, (3478, False, "tcp", "user")),
        ({TLS}, (443, True, "tcp", "user")),
        (set(), (3478, False, "udp", "user")),
    ],
    ids=["udp-ok", "udp-blocked", "only-tls", "none-reachable"],
)
def test_aiortc_uses_first_reachable_turn_url(monkeypatch, reachable, expected):
    config = _config(UDP, TCP, TLS)

    result = _prefer(monkeypatch, config, reachable)

    assert _aiortc_turn(result) == expected
    assert config.iceServers[0].urls == [UDP, TCP, TLS]


@pytest.mark.parametrize(
    "config", [None, _config(TLS), _config("stun:stun.example.com:3478", TLS)]
)
def test_fewer_than_two_turn_urls_are_returned_unprobed(monkeypatch, config):
    assert _prefer(monkeypatch, config, reachable=set()) is config


def test_none_reachable_warns(monkeypatch, caplog):
    _prefer(monkeypatch, _config(UDP, TLS), reachable=set())

    assert "No TURN transport reachable" in caplog.text


@pytest.mark.parametrize(("udp_answers", "max_seconds"), [(True, 0.5), (False, 2.5)])
def test_hanging_probes_are_bounded(monkeypatch, udp_answers, max_seconds):
    async def fake_probe(url: str) -> bool:
        if udp_answers and url == UDP:
            return True
        await asyncio.Event().wait()

    monkeypatch.setattr(turn_probe, "probe_turn_url", fake_probe)
    started = time.monotonic()

    asyncio.run(turn_probe.prefer_reachable_turn(_config(UDP, TCP, TLS)))

    assert time.monotonic() - started < max_seconds


def _binding_response(request: bytes) -> bytes:
    transaction_id = stun.parse_message(request).transaction_id
    response = stun.Message(
        message_method=stun.Method.BINDING,
        message_class=stun.Class.RESPONSE,
        transaction_id=transaction_id,
    )
    return bytes(response)


class _UdpResponder(asyncio.DatagramProtocol):
    def connection_made(self, transport) -> None:
        self.transport = transport

    def datagram_received(self, data: bytes, addr) -> None:
        self.transport.sendto(_binding_response(data), addr)


async def _handle_tcp(reader, writer) -> None:
    header = await reader.readexactly(20)
    body = await reader.readexactly(int.from_bytes(header[2:4], "big"))
    writer.write(_binding_response(header + body))
    await writer.drain()
    writer.close()


def test_probe_turn_url_over_udp_and_tcp():
    async def scenario() -> list:
        loop = asyncio.get_running_loop()
        udp, _ = await loop.create_datagram_endpoint(
            _UdpResponder, local_addr=("127.0.0.1", 0)
        )
        tcp = await asyncio.start_server(_handle_tcp, "127.0.0.1", 0)
        udp_port = udp.get_extra_info("sockname")[1]
        tcp_port = tcp.sockets[0].getsockname()[1]
        try:
            return await asyncio.gather(
                turn_probe.probe_turn_url(f"turn:127.0.0.1:{udp_port}"),
                turn_probe.probe_turn_url(f"turn:127.0.0.1:{tcp_port}?transport=tcp"),
            )
        finally:
            udp.close()
            tcp.close()

    assert asyncio.run(scenario()) == [True, True]


def test_probe_turn_url_raises_on_closed_port():
    async def scenario() -> None:
        server = await asyncio.start_server(_handle_tcp, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        server.close()
        await server.wait_closed()
        await turn_probe.probe_turn_url(f"turn:127.0.0.1:{port}?transport=tcp")

    with pytest.raises(OSError):
        asyncio.run(scenario())

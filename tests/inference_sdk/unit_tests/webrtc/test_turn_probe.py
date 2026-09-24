"""Tests for picking a reachable TURN transport before aiortc sees the config."""

import asyncio
import logging
import time
from typing import Dict, List

import pytest
from aioice import stun
from aiortc import RTCConfiguration, RTCIceServer

from inference_sdk.webrtc import turn_probe

UDP_URL = "turn:turn1.example.com:3478"
TCP_URL = "turn:turn1.example.com:3478?transport=tcp"
TLS_URL = "turns:turns1.example.com:443?transport=tcp"


def _platform_config() -> RTCConfiguration:
    return RTCConfiguration(
        iceServers=[
            RTCIceServer(
                urls=[UDP_URL, TCP_URL, TLS_URL],
                username="user",
                credential="secret",
            )
        ]
    )


def _stub_probes(monkeypatch, reachable: Dict[str, bool]) -> List[str]:
    probed = []

    async def fake_probe(url: str) -> bool:
        probed.append(url)
        if reachable[url]:
            return True
        raise OSError("unreachable")

    monkeypatch.setattr(turn_probe, "probe_turn_url", fake_probe)
    return probed


def _first_url(config: RTCConfiguration) -> str:
    return config.iceServers[0].urls[0]


def test_udp_reachable_keeps_udp_first(monkeypatch) -> None:
    # given
    _stub_probes(monkeypatch, {UDP_URL: True, TCP_URL: True, TLS_URL: True})
    config = _platform_config()

    # when
    result = asyncio.run(turn_probe.prefer_reachable_turn(config))

    # then
    assert result.iceServers[0].urls == [UDP_URL, TCP_URL, TLS_URL]


def test_udp_blocked_tcp_reachable_moves_tcp_first(monkeypatch) -> None:
    # given
    _stub_probes(monkeypatch, {UDP_URL: False, TCP_URL: True, TLS_URL: True})
    config = _platform_config()

    # when
    result = asyncio.run(turn_probe.prefer_reachable_turn(config))

    # then
    assert _first_url(result) == TCP_URL
    assert sorted(result.iceServers[0].urls) == sorted([UDP_URL, TCP_URL, TLS_URL])


def test_only_tls_reachable_moves_turns_first(monkeypatch) -> None:
    # given
    _stub_probes(monkeypatch, {UDP_URL: False, TCP_URL: False, TLS_URL: True})
    config = _platform_config()

    # when
    result = asyncio.run(turn_probe.prefer_reachable_turn(config))

    # then
    assert _first_url(result) == TLS_URL
    assert result.iceServers[0].username == "user"
    assert result.iceServers[0].credential == "secret"


def test_reordering_does_not_mutate_the_input_config(monkeypatch) -> None:
    # given
    _stub_probes(monkeypatch, {UDP_URL: False, TCP_URL: False, TLS_URL: True})
    config = _platform_config()

    # when
    asyncio.run(turn_probe.prefer_reachable_turn(config))

    # then
    assert config.iceServers[0].urls == [UDP_URL, TCP_URL, TLS_URL]


def test_reachable_url_in_a_later_server_moves_that_server_first(
    monkeypatch,
) -> None:
    # given
    _stub_probes(monkeypatch, {UDP_URL: False, TLS_URL: True})
    udp_server = RTCIceServer(urls=[UDP_URL], username="a", credential="x")
    tls_server = RTCIceServer(urls=[TLS_URL], username="b", credential="y")
    config = RTCConfiguration(iceServers=[udp_server, tls_server])

    # when
    result = asyncio.run(turn_probe.prefer_reachable_turn(config))

    # then
    assert [server.urls for server in result.iceServers] == [[TLS_URL], [UDP_URL]]
    assert result.iceServers[0].username == "b"


def test_nothing_reachable_returns_original_config_and_warns(
    monkeypatch, caplog
) -> None:
    # given
    _stub_probes(monkeypatch, {UDP_URL: False, TCP_URL: False, TLS_URL: False})
    config = _platform_config()

    # when
    with caplog.at_level(logging.WARNING):
        result = asyncio.run(turn_probe.prefer_reachable_turn(config))

    # then
    assert result is config
    assert "No TURN transport reachable" in caplog.text


@pytest.mark.parametrize(
    "config",
    [
        None,
        RTCConfiguration(
            iceServers=[RTCIceServer(urls=[TLS_URL], username="u", credential="c")]
        ),
        RTCConfiguration(
            iceServers=[
                RTCIceServer(urls=["stun:stun.example.com:3478"]),
                RTCIceServer(urls=TLS_URL, username="u", credential="c"),
            ]
        ),
    ],
)
def test_single_turn_url_or_no_config_is_not_probed(monkeypatch, config) -> None:
    # given
    probed = _stub_probes(monkeypatch, {})

    # when
    result = asyncio.run(turn_probe.prefer_reachable_turn(config))

    # then
    assert result is config
    assert probed == []


def test_udp_success_does_not_wait_for_slower_probes(monkeypatch) -> None:
    # given
    async def fake_probe(url: str) -> bool:
        if url == UDP_URL:
            return True
        await asyncio.Event().wait()

    monkeypatch.setattr(turn_probe, "probe_turn_url", fake_probe)
    config = _platform_config()

    # when
    started = time.monotonic()
    result = asyncio.run(turn_probe.prefer_reachable_turn(config))
    elapsed = time.monotonic() - started

    # then
    assert _first_url(result) == UDP_URL
    assert elapsed < 0.5


def test_hanging_probes_are_bounded_by_the_default_timeout(monkeypatch) -> None:
    # given
    async def fake_probe(url: str) -> bool:
        await asyncio.Event().wait()

    monkeypatch.setattr(turn_probe, "probe_turn_url", fake_probe)
    config = _platform_config()

    # when
    started = time.monotonic()
    result = asyncio.run(turn_probe.prefer_reachable_turn(config))
    elapsed = time.monotonic() - started

    # then
    assert result is config
    assert elapsed < 2.5


def _binding_response(data: bytes) -> bytes:
    request = stun.parse_message(data)
    response = stun.Message(
        message_method=stun.Method.BINDING,
        message_class=stun.Class.RESPONSE,
        transaction_id=request.transaction_id,
    )
    return bytes(response)


class _UdpStunResponder(asyncio.DatagramProtocol):
    def connection_made(self, transport) -> None:
        self.transport = transport

    def datagram_received(self, data: bytes, addr) -> None:
        self.transport.sendto(_binding_response(data), addr)


def test_probe_turn_url_gets_an_answer_over_udp() -> None:
    async def scenario() -> bool:
        loop = asyncio.get_running_loop()
        transport, _ = await loop.create_datagram_endpoint(
            _UdpStunResponder, local_addr=("127.0.0.1", 0)
        )
        port = transport.get_extra_info("sockname")[1]
        try:
            return await turn_probe.probe_turn_url(f"turn:127.0.0.1:{port}")
        finally:
            transport.close()

    assert asyncio.run(scenario()) is True


def test_probe_turn_url_gets_an_answer_over_tcp() -> None:
    async def handle(reader, writer) -> None:
        header = await reader.readexactly(20)
        body = await reader.readexactly(int.from_bytes(header[2:4], "big"))
        writer.write(_binding_response(header + body))
        await writer.drain()
        writer.close()

    async def scenario() -> bool:
        server = await asyncio.start_server(handle, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            return await turn_probe.probe_turn_url(
                f"turn:127.0.0.1:{port}?transport=tcp"
            )
        finally:
            server.close()
            await server.wait_closed()

    assert asyncio.run(scenario()) is True


def test_probe_turn_url_raises_on_closed_tcp_port() -> None:
    async def scenario() -> None:
        server = await asyncio.start_server(lambda r, w: None, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        server.close()
        await server.wait_closed()
        await turn_probe.probe_turn_url(f"turn:127.0.0.1:{port}?transport=tcp")

    with pytest.raises(OSError):
        asyncio.run(scenario())

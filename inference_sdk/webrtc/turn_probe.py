"""Pick a reachable TURN transport before handing the config to aiortc.

aiortc only uses the first valid TURN URL in an ``RTCConfiguration``. The
Roboflow platform lists UDP first, so on networks that block outbound UDP the
TCP and TLS relays are never tried. Probing each URL with an unauthenticated
STUN Binding Request and moving the first reachable one to the front fixes that
without changing behavior when UDP works.
"""

import asyncio
import ssl
from typing import List, Optional, Tuple

from aioice import stun
from aiortc import RTCConfiguration, RTCIceServer
from aiortc.rtcicetransport import parse_stun_turn_uri

from inference_sdk.utils.logging import get_logger

logger = get_logger("webrtc.turn_probe")

TURN_PROBE_TIMEOUT = 2.0
STUN_HEADER_LENGTH = 20


class _StunUdpProtocol(asyncio.DatagramProtocol):
    def __init__(self, response: asyncio.Future) -> None:
        self._response = response

    def datagram_received(self, data: bytes, addr) -> None:
        if not self._response.done():
            self._response.set_result(data)

    def error_received(self, exc: Exception) -> None:
        if not self._response.done():
            self._response.set_exception(exc)


async def _probe_udp(host: str, port: int, request: bytes) -> bytes:
    loop = asyncio.get_running_loop()
    response = loop.create_future()
    transport, _ = await loop.create_datagram_endpoint(
        lambda: _StunUdpProtocol(response), remote_addr=(host, port)
    )
    try:
        transport.sendto(request)
        return await response
    finally:
        transport.close()


async def _probe_stream(
    host: str, port: int, request: bytes, ssl_context: Optional[ssl.SSLContext]
) -> bytes:
    server_hostname = host if ssl_context else None
    reader, writer = await asyncio.open_connection(
        host, port, ssl=ssl_context, server_hostname=server_hostname
    )
    try:
        writer.write(request)
        await writer.drain()
        header = await reader.readexactly(STUN_HEADER_LENGTH)
        body = await reader.readexactly(int.from_bytes(header[2:4], "big"))
        return header + body
    finally:
        writer.close()


async def probe_turn_url(url: str) -> bool:
    """Check that a TURN server answers over the transport its URL names.

    Sends one STUN Binding Request, which TURN servers answer without
    credentials, over UDP, TCP or TLS as the URL specifies.

    Args:
        url: A ``turn:`` or ``turns:`` URL as accepted by aiortc.

    Returns:
        True when the server sent back a STUN message for the request.

    Raises:
        ValueError: If the URL is not a TURN URL aiortc can parse.
        OSError: If the server cannot be reached over the URL's transport.
    """
    parsed = parse_stun_turn_uri(url)
    request = stun.Message(
        message_method=stun.Method.BINDING, message_class=stun.Class.REQUEST
    )

    if parsed["transport"] == "udp":
        data = await _probe_udp(parsed["host"], parsed["port"], bytes(request))
    else:
        ssl_context = (
            ssl.create_default_context() if parsed["scheme"] == "turns" else None
        )
        data = await _probe_stream(
            parsed["host"], parsed["port"], bytes(request), ssl_context
        )

    response = stun.parse_message(data)
    reachable = response.transaction_id == request.transaction_id

    return reachable


def _turn_urls(config: Optional[RTCConfiguration]) -> List[Tuple[RTCIceServer, str]]:
    """List (server, url) for every TURN URL, in config order."""
    servers = config.iceServers if config and config.iceServers else []
    candidates = [
        (server, url)
        for server in servers
        for url in (server.urls if isinstance(server.urls, list) else [server.urls])
        if url.startswith("turn")
    ]

    return candidates


async def prefer_reachable_turn(
    config: Optional[RTCConfiguration],
    *,
    timeout: float = TURN_PROBE_TIMEOUT,
) -> Optional[RTCConfiguration]:
    """Move the first reachable TURN URL to the front of the configuration.

    aiortc only uses the first TURN URL it finds. All URLs are probed in
    parallel, and the first one *in the original order* that answers is moved
    to the front, so UDP stays preferred when it works. The call returns as
    soon as that choice is certain, and never takes longer than ``timeout``.

    Args:
        config: ICE configuration for the local peer connection, or None.
        timeout: Overall time budget for all probes, in seconds.

    Returns:
        A copy of ``config`` with the reachable TURN URL in front when it is not
        already the first one, otherwise ``config`` itself. The input is never
        mutated. With fewer than two TURN URLs nothing is probed.
    """
    candidates = _turn_urls(config)
    if len(candidates) < 2:
        return config

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    tasks = [asyncio.ensure_future(probe_turn_url(url)) for _, url in candidates]
    try:
        for (server, url), task in zip(candidates, tasks):
            await asyncio.wait([task], timeout=max(deadline - loop.time(), 0))
            done = task.done() and not task.cancelled()
            if not done or task.exception() or task.result() is not True:
                continue

            logger.debug("Using TURN %s (first reachable)", url)
            if task is tasks[0]:
                return config

            # aiortc uses only the first TURN URL, so a copy of the reachable
            # one in front wins; the rest of the config stays as it was.
            first = RTCIceServer(
                urls=url,
                username=server.username,
                credential=server.credential,
                credentialType=server.credentialType,
            )
            reordered = RTCConfiguration(
                iceServers=[first, *config.iceServers],
                bundlePolicy=config.bundlePolicy,
            )
            return reordered
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    logger.warning(
        "No TURN transport reachable (%s); keeping default order",
        ", ".join(url for _, url in candidates),
    )
    return config

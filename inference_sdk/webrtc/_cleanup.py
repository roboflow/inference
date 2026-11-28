"""Bound asynchronous teardown and abort only a peer's owned ICE transports."""

import asyncio
from typing import Any, Awaitable, Callable, Optional

from aioice.turn import TurnClientTcpProtocol, TurnClientUdpProtocol, TurnTransport

from inference_sdk.utils.logging import get_logger

logger = get_logger("webrtc.cleanup")


def _abort_peer_transports(peer: Any) -> None:
    """Abort unique transports reachable through this peer's ICE connections.

    aiortc exposes the SCTP/transceiver transports publicly, but aioice exposes
    protocols only through its connection. A TURN wrapper owns a UDP or TCP
    carrier rather than exposing abort itself. Keep those private compatibility
    boundaries here; never search a loop or the process for unrelated sockets.
    """
    ice_transports = []
    sctp = getattr(peer, "sctp", None)
    if sctp is not None:
        ice_transports.append(sctp.transport.transport)

    get_transceivers = getattr(peer, "getTransceivers", None)
    if callable(get_transceivers):
        for transceiver in get_transceivers():
            ice_transports.append(transceiver.receiver.transport.transport)

    seen = set()
    errors = []
    for ice in ice_transports:
        connection = getattr(ice, "_connection", None)
        for protocol in tuple(getattr(connection, "_protocols", ())):
            transport = getattr(protocol, "transport", None)
            if transport is None:
                continue
            try:
                if isinstance(transport, TurnTransport):
                    inner = getattr(transport, "_TurnTransport__inner_protocol", None)
                    if not isinstance(
                        inner, (TurnClientUdpProtocol, TurnClientTcpProtocol)
                    ):
                        raise RuntimeError("Unsupported WebRTC TURN transport layout")
                    transport = inner.transport
                if id(transport) in seen:
                    continue
                seen.add(id(transport))
                transport.abort()
            except Exception as error:
                # One failing socket must not strand the rest of this peer.
                errors.append(error)

    if errors:
        raise RuntimeError("Failed to abort a WebRTC ICE transport") from errors[0]


def _remember_cleanup_error(
    first: Optional[BaseException], error: BaseException, *, phase: str
) -> BaseException:
    """Retain the first failure and log later failures without replacing it."""
    if first is None:
        return error
    logger.error(
        "Additional WebRTC %s cleanup failure",
        phase,
        exc_info=(type(error), error, error.__traceback__),
    )
    return first


async def _await_cleanup(
    operation: Awaitable[None],
    *,
    timeout: float,
    cancel_timeout: float,
    phase: str,
    on_timeout: Optional[Callable[[], None]] = None,
) -> None:
    """Give an operation finite grace without prematurely cancelling its future."""
    task = asyncio.ensure_future(operation)
    try:
        done, _ = await asyncio.wait({task}, timeout=timeout)
        if not done and on_timeout is not None:
            logger.warning(
                "WebRTC %s exceeded cleanup grace; aborting owned ICE", phase
            )
            on_timeout()
            # Let real connection_lost callbacks finish the existing close task.
            # Cancelling it first can poison aiortc's shared close future.
            done, _ = await asyncio.wait({task}, timeout=cancel_timeout)

        if not done:
            raise TimeoutError(f"WebRTC {phase} cleanup timed out")

        task.result()
    finally:
        if not task.done():
            task.cancel()
            done, _ = await asyncio.wait({task}, timeout=cancel_timeout)
            if not done:
                logger.error("WebRTC %s task did not respond to cancellation", phase)
            elif not task.cancelled():
                # Retrieve any secondary error without masking the primary one.
                try:
                    task.result()
                except Exception:
                    logger.exception("WebRTC %s task failed during cancellation", phase)

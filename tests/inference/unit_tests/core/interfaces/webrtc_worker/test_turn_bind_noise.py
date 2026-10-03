import asyncio
import logging
from typing import Callable
from unittest.mock import patch

import pytest
from aioice.stun import Class, Message, Method, TransactionFailed

from inference.core.interfaces.webrtc_worker import webrtc as worker_webrtc


def _context(exception, **extra) -> dict:
    return {
        "message": "Task exception was never retrieved",
        "exception": exception,
        **extra,
    }


def _failed(method: Method, code: int) -> TransactionFailed:
    response = Message(
        message_method=method,
        message_class=Class.ERROR,
        transaction_id=b"x" * 12,
        attributes={"ERROR-CODE": (code, "")},
    )

    return TransactionFailed(response)


def test_refused_channel_binds_are_logged_at_debug_only(caplog):
    loop = asyncio.new_event_loop()
    try:
        worker_webrtc._quiet_turn_bind_failures(loop)
        context = _context(
            _failed(Method.CHANNEL_BIND, 403),
            future=loop.create_future(),
        )

        with (
            patch.object(worker_webrtc.logger, "debug") as debug,
            caplog.at_level(logging.DEBUG),
        ):
            loop.call_exception_handler(context)

        assert [r for r in caplog.records if r.name == "asyncio"] == []
        debug.assert_called_once_with(
            "TURN channel bind refused: %s", context["exception"]
        )
    finally:
        loop.close()


@pytest.mark.parametrize(
    "build_context",
    [
        lambda loop: _context(
            _failed(Method.REFRESH, 403), future=loop.create_future()
        ),
        lambda loop: _context(
            _failed(Method.CHANNEL_BIND, 401), future=loop.create_future()
        ),
        lambda loop: {
            "message": "Exception in callback",
            "exception": _failed(Method.CHANNEL_BIND, 403),
            "handle": object(),
        },
        lambda loop: _context(RuntimeError("boom"), future=loop.create_future()),
    ],
    ids=["refresh_403", "channel_bind_401", "callback_not_task", "runtime_error"],
)
def test_other_loop_exceptions_still_reach_the_default_handler(
    build_context: Callable[[asyncio.AbstractEventLoop], dict], caplog
):
    loop = asyncio.new_event_loop()
    try:
        worker_webrtc._quiet_turn_bind_failures(loop)
        context = build_context(loop)

        with (
            patch.object(worker_webrtc.logger, "debug") as debug,
            caplog.at_level(logging.DEBUG),
        ):
            loop.call_exception_handler(context)

        asyncio_records = [
            r
            for r in caplog.records
            if r.name == "asyncio" and r.levelno == logging.ERROR
        ]
        assert len(asyncio_records) == 1
        assert context["message"] in asyncio_records[0].message
        debug.assert_not_called()
    finally:
        loop.close()

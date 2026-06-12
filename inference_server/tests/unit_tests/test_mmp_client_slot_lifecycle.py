"""Slot lifecycle on infer() failure paths.

Regression: when _submit_and_wait raised (timeout / client disconnect), the
caller's `submit_req_id` was never assigned, so the finally block took the
"never submitted" branch and sent T_FREE for a slot the worker still held —
the slot got reused mid-batch and the worker decoded foreign bytes.
The correct behaviour after a successful submit is T_CANCEL (MMP frees the
slot once the worker drains the ticket).
"""

from __future__ import annotations

import asyncio

import pytest

from inference_server.proxies.mmp_client import T_CANCEL, T_FREE, T_SUBMIT, MMPClient


class _RecordingSock:
    def __init__(self) -> None:
        self.sent: list[list[bytes]] = []

    async def send_multipart(self, frames: list[bytes]) -> None:
        self.sent.append(frames)


def _make_client() -> tuple[MMPClient, _RecordingSock]:
    client = MMPClient(
        mmp_addr="inproc://test",
        shm_name="test",
        shm_data_size=1024,
        infer_timeout_s=0.05,
        alloc_timeout_s=0.05,
        pipeline_csv_path="",
    )
    sock = _RecordingSock()
    client._sock = sock
    return client, sock


def _msg_types(sock: _RecordingSock) -> list[bytes]:
    return [frames[0] for frames in sock.sent]


def test_submit_timeout_sends_cancel_not_free():
    async def _run() -> list[bytes]:
        client, sock = _make_client()

        async def _fake_alloc(model_id, instance=""):
            return 5

        client._alloc_slot = _fake_alloc
        client._write_input = lambda slot_id, chunk, offset: None

        with pytest.raises(asyncio.TimeoutError):
            await client.infer(model_id="m", image=b"\xff\xd8data")

        # Drain the fire-and-forget send tasks
        for _ in range(5):
            await asyncio.sleep(0)
        return _msg_types(sock)

    types = asyncio.run(_run())
    assert T_SUBMIT in types
    assert T_FREE not in types, "slot freed while worker may still hold its ticket"
    assert T_CANCEL in types


def test_alloc_ok_but_submit_send_fails_frees_slot():
    """No worker ticket exists if T_SUBMIT was never sent — free is correct."""

    async def _run() -> list[bytes]:
        client, sock = _make_client()

        async def _fake_alloc(model_id, instance=""):
            return 5

        def _boom(slot_id, chunk, offset):
            raise RuntimeError("write failed")

        client._alloc_slot = _fake_alloc
        client._write_input = _boom

        with pytest.raises(RuntimeError, match="write failed"):
            await client.infer(model_id="m", image=b"\xff\xd8data")

        for _ in range(5):
            await asyncio.sleep(0)
        return _msg_types(sock)

    types = asyncio.run(_run())
    assert T_SUBMIT not in types
    assert T_CANCEL not in types
    assert T_FREE in types

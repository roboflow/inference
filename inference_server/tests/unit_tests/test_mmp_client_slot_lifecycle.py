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
import pickle
import struct

import pytest

from inference_server.errors import UploadTooSlowError
from inference_server.proxies.mmp_client import (
    T_ALLOC,
    T_ALLOC_OK,
    T_CANCEL,
    T_ERROR,
    T_FREE,
    T_MODEL_READY,
    T_RESULT_READY,
    T_SUBMIT,
    MMPClient,
)


async def _chunks(*parts: bytes):
    for part in parts:
        yield part


async def _slow_chunks():
    yield b"\xff\xd8"
    await asyncio.sleep(0.05)
    yield b"x"


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
    )
    sock = _RecordingSock()
    client._sock = sock
    return client, sock


def _msg_types(sock: _RecordingSock) -> list[bytes]:
    return [frames[0] for frames in sock.sent]


def test_submit_timeout_sends_cancel_not_free():
    async def _run() -> list[bytes]:
        client, sock = _make_client()

        async def _fake_alloc(req_id, model_id, instance=""):
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

        async def _fake_alloc(req_id, model_id, instance=""):
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


# ---------------------------------------------------------------------------
# Slot ownership: one req_id per inference, T_FREE carries it
# ---------------------------------------------------------------------------


def test_single_req_id_for_alloc_submit_free():
    async def _run():
        client, sock = _make_client()
        client._write_input = lambda slot_id, chunk, offset: None
        client._read_slot_header = lambda slot_id: None
        client._read_result = lambda slot_id, sz: pickle.dumps({"ok": 1})

        task = asyncio.create_task(client.infer(model_id="m", image=b"\xff\xd8x"))
        for _ in range(5):
            await asyncio.sleep(0)
        alloc_frame = [f for f in sock.sent if f[0] == T_ALLOC][0]
        alloc_req = struct.unpack_from(">Q", alloc_frame[1])[0]

        client._dispatch(T_ALLOC_OK, [struct.pack(">QI", alloc_req, 5)])
        for _ in range(5):
            await asyncio.sleep(0)
        submit_frame = [f for f in sock.sent if f[0] == T_SUBMIT][0]
        submit_req = struct.unpack_from(">Q", submit_frame[1])[0]
        assert submit_req == alloc_req

        client._dispatch(T_RESULT_READY, [struct.pack(">QII", submit_req, 5, 10)])
        result = await task
        assert result == {"ok": 1}
        for _ in range(5):
            await asyncio.sleep(0)
        free_frame = [f for f in sock.sent if f[0] == T_FREE][0]
        free_req, free_slot = struct.unpack(">QI", free_frame[1])
        assert free_req == alloc_req
        assert free_slot == 5

    asyncio.run(_run())


def test_infer_stream_writes_chunks_and_submits_total_size():
    async def _run():
        client, sock = _make_client()
        writes = []
        client._write_input = lambda slot_id, chunk, offset: writes.append(
            (slot_id, bytes(chunk), offset)
        )
        client._read_slot_header = lambda slot_id: None
        client._read_result = lambda slot_id, sz: pickle.dumps({"ok": 1})

        task = asyncio.create_task(
            client.infer_stream(
                model_id="m",
                image_chunks=_chunks(b"\xff\xd8", b"abc"),
                content_length=5,
            )
        )
        await asyncio.sleep(0)
        alloc_frame = [f for f in sock.sent if f[0] == T_ALLOC][0]
        req_id = struct.unpack_from(">Q", alloc_frame[1])[0]
        client._dispatch(T_ALLOC_OK, [struct.pack(">QI", req_id, 5)])
        for _ in range(5):
            await asyncio.sleep(0)
        submit_frame = [f for f in sock.sent if f[0] == T_SUBMIT][0]
        submit_req, slot_id, input_sz = struct.unpack_from(">QII", submit_frame[1])
        assert submit_req == req_id
        assert slot_id == 5
        assert input_sz == 5
        assert writes == [(5, b"\xff\xd8", 0), (5, b"abc", 2)]

        client._dispatch(T_RESULT_READY, [struct.pack(">QII", req_id, 5, 10)])
        assert await task == {"ok": 1}

    asyncio.run(_run())


def test_infer_stream_upload_timeout_frees_unsubmitted_slot():
    async def _run() -> list[bytes]:
        client, sock = _make_client()

        async def _fake_alloc(req_id, model_id, instance=""):
            return 5

        client._alloc_slot = _fake_alloc
        client._write_input = lambda slot_id, chunk, offset: None

        with pytest.raises(UploadTooSlowError):
            await client.infer_stream(
                model_id="m",
                image_chunks=_slow_chunks(),
                content_length=3,
                upload_timeout_s=0.01,
            )

        for _ in range(5):
            await asyncio.sleep(0)
        return _msg_types(sock)

    types = asyncio.run(_run())
    assert T_SUBMIT not in types
    assert T_CANCEL not in types
    assert T_FREE in types


def test_disconnect_race_prefers_completed_result():
    async def _run():
        client, _ = _make_client()
        fut = asyncio.get_running_loop().create_future()
        fut.set_result(("result", 5, 10))

        class _DisconnectedRequest:
            async def is_disconnected(self):
                return True

        return await client._wait_with_disconnect(fut, _DisconnectedRequest())

    assert asyncio.run(_run()) == ("result", 5, 10)


def test_alloc_cancellation_drops_pending_future():
    async def _run():
        client, sock = _make_client()
        task = asyncio.create_task(client._alloc_slot(12345, "m"))
        await asyncio.sleep(0)
        assert 12345 in client._pending
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert 12345 not in client._pending

    asyncio.run(_run())


def test_free_and_cancel_tasks_are_strongly_referenced():
    async def _run():
        client, sock = _make_client()
        client._free_slot(5, 111)
        client._cancel_req(222)
        assert len(client._bg_tasks) == 2
        for _ in range(5):
            await asyncio.sleep(0)
        assert len(client._bg_tasks) == 0
        return _msg_types(sock)

    types = asyncio.run(_run())
    assert T_FREE in types and T_CANCEL in types


def test_start_rejects_shm_geometry_mismatch():
    """Slot-size disagreement between client and MMP must fail at startup,
    not silently corrupt neighboring slots at runtime."""
    from multiprocessing.shared_memory import SharedMemory

    shm = SharedMemory(create=True, size=(64 + 1024) * 4)  # 4 slots of 1024
    try:
        client = MMPClient(
            mmp_addr="inproc://test",
            shm_name=shm.name,
            shm_data_size=2048,  # WRONG: MMP created 1024-byte slots
        )

        async def _run():
            client._ctx = None

            class _Sock:
                def setsockopt(self, *a):
                    pass

                def connect(self, *a):
                    pass

            import inference_server.proxies.mmp_client as mod

            # start() builds ctx/socket first — patch them out, keep SHM attach
            orig_ctx = mod.zmq.asyncio.Context
            with pytest.raises(RuntimeError, match="geometry mismatch"):
                try:
                    mod.zmq.asyncio.Context = lambda: type(
                        "C",
                        (),
                        {"socket": lambda s, t: _Sock(), "term": lambda s: None},
                    )()
                    await client.start()
                finally:
                    mod.zmq.asyncio.Context = orig_ctx
                    if client._recv_task is not None:
                        client._recv_task.cancel()

        asyncio.run(_run())
    finally:
        shm.close()
        shm.unlink()


# ---------------------------------------------------------------------------
# Typed error classification (theme 7)
# ---------------------------------------------------------------------------


def test_oversized_image_raises_payload_too_large():
    from inference_server.errors import PayloadTooLargeError

    async def _run():
        client, _ = _make_client()
        with pytest.raises(PayloadTooLargeError):
            await client.infer(model_id="m", image=b"x" * 2048)  # slot is 1024

    asyncio.run(_run())


def test_alloc_timeout_raises_server_busy():
    from inference_server.errors import ServerBusyError

    async def _run():
        client, _ = _make_client()  # alloc_timeout_s=0.05, no reply ever
        with pytest.raises(ServerBusyError):
            await client._alloc_slot(1, "m")

    asyncio.run(_run())


def test_alloc_error_reply_raises_server_busy():
    from inference_server.errors import ServerBusyError

    async def _run():
        client, sock = _make_client()
        task = asyncio.create_task(client._alloc_slot(77, "m"))
        await asyncio.sleep(0)
        client._dispatch(T_ERROR, [struct.pack(">QB", 77, 1)])  # _ERR_POOL_FULL
        with pytest.raises(ServerBusyError):
            await task

    asyncio.run(_run())


def test_stats_error_reply_raises_instead_of_empty_dict():
    async def _run():
        client, sock = _make_client()
        task = asyncio.create_task(client.stats())
        await asyncio.sleep(0)
        req_id = struct.unpack_from(">Q", sock.sent[0][1])[0]
        client._dispatch(T_ERROR, [struct.pack(">QB", req_id, 4)])
        with pytest.raises(RuntimeError, match="stats"):
            await task

    asyncio.run(_run())


def test_ensure_loaded_caches_and_skips_second_send():
    async def _run():
        client, sock = _make_client()
        client.ensure_cache_ttl_s = 60.0
        task = asyncio.create_task(client.ensure_loaded("m"))
        await asyncio.sleep(0)
        req_id = struct.unpack_from(">Q", sock.sent[0][1])[0]
        client._dispatch(T_MODEL_READY, [struct.pack(">Q", req_id)])
        assert (await task)[0] == "model_ready"
        assert len(sock.sent) == 1

        assert (await client.ensure_loaded("m"))[0] == "model_ready"
        assert (
            len(sock.sent) == 1
        ), "warm cache must skip the T_ENSURE_LOADED round-trip"

    asyncio.run(_run())


def test_shm_admission_skips_alloc_when_pool_full():
    from inference_model_manager.backends.utils.shm_pool import SHMPool
    from inference_server.errors import ServerBusyError

    async def _run():
        pool = SHMPool.create(n_slots=4, input_mb=0.001)
        try:
            client, sock = _make_client()
            client.shm_admission = True
            client._shm = pool._shm
            client.n_slots = 4
            client.slot_total = pool.slot_bytes

            # Free slots present → admission allows the send.
            task = asyncio.create_task(client._alloc_slot(1, "m"))
            await asyncio.sleep(0)
            assert any(f[0] == T_ALLOC for f in sock.sent), "should send when free>0"
            client._dispatch(T_ALLOC_OK, [struct.pack(">QI", 1, 0)])
            assert await task == 0

            # Drain the pool → free_count published as 0 in the meta block.
            for _ in range(4):
                pool.alloc_slot(timeout=0)
            sock.sent.clear()
            with pytest.raises(ServerBusyError):
                await client._alloc_slot(2, "m")
            assert not sock.sent, "must NOT send T_ALLOC when pool full"
        finally:
            pool.close()

    asyncio.run(_run())

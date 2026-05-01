"""Integration tests for ModelManagerProcess hot path.

Runs MMP in an asyncio event loop inside a background thread.
Client uses a synchronous ZMQ DEALER (no model inference — pure protocol).
A mock backend implements signal_slot() to simulate SubprocessBackend.
"""

from __future__ import annotations

import asyncio
import struct
import threading
import time
import uuid

import pytest
import zmq

from inference_model_manager.backends.utils.shm_pool import SHMPool, SlotStatus
from inference_model_manager.model_manager_process import (
    T_ALLOC,
    T_ALLOC_OK,
    T_ENSURE_LOADED,
    T_ERROR,
    T_FREE,
    T_LOAD_TIMEOUT,
    T_MODEL_READY,
    T_RESULT_READY,
    T_SUBMIT,
    ModelManagerProcess,
)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

_TEST_INPUT_MB = 0.1  # 100 KB — tiny for tests
_TIMEOUT_S = 5.0  # max seconds any single recv should take


def _rand_addr() -> str:
    """tcp://127.0.0.1:0 — OS assigns a free port; MMP reads actual via LAST_ENDPOINT."""
    return "tcp://127.0.0.1:0"


def _req_id() -> int:
    return uuid.uuid4().int & 0xFFFF_FFFF_FFFF_FFFF


class _MMPHarness:
    """Starts MMP in a background asyncio thread; provides sync ZMQ client."""

    def __init__(self, n_slots: int = 8):
        self._bind_addr = _rand_addr()
        self._ready = threading.Event()
        self._mmp = ModelManagerProcess(
            n_slots=n_slots,
            input_mb=_TEST_INPUT_MB,
            stale_reap_interval_s=1.0,
            stale_slot_max_age_s=2.0,
        )
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="mmp-test-loop"
        )
        self._thread.start()
        assert self._ready.wait(timeout=_TIMEOUT_S), "MMP did not start in time"

        # After MMP starts, read actual bound address (port 0 → real port)
        self.addr = self._mmp.bound_addr or self._bind_addr

        self._ctx = zmq.Context()
        self._dealer = self._ctx.socket(zmq.DEALER)
        self._dealer.setsockopt(zmq.RCVTIMEO, int(_TIMEOUT_S * 1000))
        self._dealer.setsockopt(zmq.LINGER, 0)
        self._dealer.setsockopt(
            zmq.IDENTITY,
            f"uv_{threading.get_ident()}_{uuid.uuid4().hex[:8]}".encode(),
        )
        self._dealer.connect(self.addr)

    def _run_loop(self) -> None:
        asyncio.run(self._mmp.run(addr=self._bind_addr, ready_event=self._ready))

    @property
    def mmp(self) -> ModelManagerProcess:
        return self._mmp

    @property
    def shm_name(self) -> str:
        return self._mmp.shm_name

    # --- sync ZMQ helpers ---

    def send(self, msg_type: bytes, payload: bytes) -> None:
        self._dealer.send_multipart([msg_type, payload])

    def recv(self) -> tuple[bytes, bytes]:
        """Returns (msg_type, payload)."""
        parts = self._dealer.recv_multipart()
        return parts[0], parts[1]

    # --- protocol helpers ---

    def ensure_loaded(self, flavor: str, wait_ms: int = 5000) -> bytes:
        rid = _req_id()
        flavor_b = flavor.encode()
        payload = struct.pack(">QIH", rid, wait_ms, len(flavor_b)) + flavor_b
        self.send(T_ENSURE_LOADED, payload)
        msg_type, frame = self.recv()
        return msg_type

    def alloc(self, flavor: str) -> tuple[int, int]:
        """Returns (req_id, slot_id)."""
        rid = _req_id()
        flavor_b = flavor.encode()
        payload = struct.pack(">QH", rid, len(flavor_b)) + flavor_b
        self.send(T_ALLOC, payload)
        msg_type, frame = self.recv()
        assert msg_type == T_ALLOC_OK, f"expected T_ALLOC_OK, got {msg_type!r}"
        _, slot_id = struct.unpack(">QI", frame)
        return rid, slot_id

    def submit(self, req_id: int, slot_id: int, flavor: str, data: bytes) -> None:
        """Write data into SHM slot then send T_SUBMIT."""
        pool = SHMPool.attach(
            self.shm_name,
            n_slots=self._mmp._n_slots,
            input_mb=_TEST_INPUT_MB,
        )
        try:
            pool.data_memoryview(slot_id)[: len(data)] = data
        finally:
            pool.close()

        flavor_b = flavor.encode()
        header = (
            struct.pack(">QIIH", req_id, slot_id, len(data), len(flavor_b)) + flavor_b
        )
        self.send(T_SUBMIT, header)

    def free(self, slot_id: int) -> None:
        self.send(T_FREE, struct.pack(">I", slot_id))

    def teardown(self) -> None:
        self._mmp.stop()
        self._dealer.close()
        self._ctx.term()
        self._thread.join(timeout=3.0)


class _MockBackend:
    """Simulates SubprocessBackend: calls mmp.on_result() from a thread."""

    is_healthy = True

    def __init__(self, mmp: ModelManagerProcess, result_bytes: bytes = b"ok"):
        self._mmp = mmp
        self._result_bytes = result_bytes

    def signal_slot(
        self, slot_id: int, req_id: int, params_bytes: bytes = b"{}"
    ) -> None:
        """Write fake result to SHM slot, call on_result from a thread."""

        def _do() -> None:
            # Write result into the SHM pool's result area
            pool = SHMPool.attach(
                self._mmp.shm_name,
                n_slots=self._mmp._n_slots,
                input_mb=_TEST_INPUT_MB,
            )
            try:
                data = self._result_bytes
                pool.data_memoryview(slot_id)[: len(data)] = data
            finally:
                pool.close()
            self._mmp.on_result(req_id, slot_id, len(self._result_bytes))

        threading.Thread(target=_do, daemon=True).start()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def harness():
    h = _MMPHarness()
    yield h
    h.teardown()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEnsureLoaded:
    def test_model_ready_when_preregistered(self):
        """Pre-registered flavor → T_MODEL_READY immediately."""
        h = _MMPHarness()
        h.mmp.register_backend("yolov8n", _MockBackend(h.mmp))
        msg = h.ensure_loaded("yolov8n")
        h.teardown()
        assert msg == T_MODEL_READY

    def test_multiple_waiters_all_notified(self):
        """Two concurrent T_ENSURE_LOADED for same pre-registered model → both get T_MODEL_READY."""
        import queue

        h = _MMPHarness()
        h.mmp.register_backend("shared-model", _MockBackend(h.mmp))

        results: queue.Queue = queue.Queue()

        def _client() -> None:
            ctx = zmq.Context()
            sock = ctx.socket(zmq.DEALER)
            sock.setsockopt(zmq.RCVTIMEO, int(_TIMEOUT_S * 1000))
            sock.setsockopt(zmq.LINGER, 0)
            sock.setsockopt(zmq.IDENTITY, uuid.uuid4().bytes)
            sock.connect(h.addr)
            rid = _req_id()
            flavor_b = b"shared-model"
            payload = struct.pack(">QIH", rid, 5000, len(flavor_b)) + flavor_b
            sock.send_multipart([T_ENSURE_LOADED, payload])
            try:
                parts = sock.recv_multipart()
                results.put(parts[0])
            except zmq.Again:
                results.put(b"TIMEOUT")
            sock.close()
            ctx.term()

        t1 = threading.Thread(target=_client, daemon=True)
        t2 = threading.Thread(target=_client, daemon=True)
        t1.start()
        time.sleep(0.05)
        t2.start()
        t1.join(timeout=_TIMEOUT_S + 1)
        t2.join(timeout=_TIMEOUT_S + 1)
        h.teardown()

        assert results.get_nowait() == T_MODEL_READY
        assert results.get_nowait() == T_MODEL_READY


class TestAlloc:
    def test_alloc_returns_valid_slot(self, harness):
        harness.mmp.register_backend("m", _MockBackend(harness.mmp))
        harness.ensure_loaded("m")
        req_id, slot_id = harness.alloc("m")
        assert 0 <= slot_id < harness.mmp._n_slots
        harness.free(slot_id)

    def test_alloc_all_slots(self, harness):
        harness.mmp.register_backend("m", _MockBackend(harness.mmp))
        harness.ensure_loaded("m")
        slots = []
        for _ in range(harness.mmp._n_slots):
            _, slot_id = harness.alloc("m")
            slots.append(slot_id)
        assert sorted(slots) == list(range(harness.mmp._n_slots))
        for s in slots:
            harness.free(s)

    def test_alloc_pool_exhausted_returns_error(self, harness):
        harness.mmp.register_backend("m", _MockBackend(harness.mmp))
        harness.ensure_loaded("m")
        # Drain all slots without freeing
        for _ in range(harness.mmp._n_slots):
            harness.alloc("m")
        # Next alloc should get T_ERROR
        rid = _req_id()
        flavor_b = b"m"
        payload = struct.pack(">QH", rid, len(flavor_b)) + flavor_b
        harness.send(T_ALLOC, payload)
        msg_type, _ = harness.recv()
        assert msg_type == T_ERROR


class TestFullLifecycle:
    def test_single_request_end_to_end(self, harness):
        """T_ENSURE_LOADED→T_ALLOC→SHM write→T_SUBMIT→T_RESULT_READY→T_FREE."""
        result_payload = b"inference result data"
        mock = _MockBackend(harness.mmp, result_bytes=result_payload)
        harness.mmp.register_backend("yolov8n", mock)

        assert harness.ensure_loaded("yolov8n") == T_MODEL_READY

        req_id, slot_id = harness.alloc("yolov8n")

        image_bytes = b"\xFF\xD8\xFF" * 10  # fake JPEG header
        harness.submit(req_id, slot_id, "yolov8n", image_bytes)

        msg_type, frame = harness.recv()
        assert msg_type == T_RESULT_READY, f"got {msg_type!r}"
        resp_req_id, resp_slot_id, result_sz = struct.unpack(">QII", frame)
        assert resp_req_id == req_id
        assert resp_slot_id == slot_id
        assert result_sz == len(result_payload)

        # Read result from SHM
        pool = SHMPool.attach(
            harness.shm_name,
            n_slots=harness.mmp._n_slots,
            input_mb=_TEST_INPUT_MB,
        )
        result = bytes(pool.data_memoryview(slot_id)[:result_sz])
        pool.close()
        assert result == result_payload

        harness.free(slot_id)
        # Give MMP time to free slot
        time.sleep(0.05)
        assert harness.mmp._pool.free_count == harness.mmp._n_slots

    def test_multiple_sequential_requests(self, harness):
        """N sequential requests all succeed and slots are returned."""
        mock = _MockBackend(harness.mmp, result_bytes=b"result")
        harness.mmp.register_backend("m", mock)
        harness.ensure_loaded("m")

        for _ in range(harness.mmp._n_slots):
            req_id, slot_id = harness.alloc("m")
            harness.submit(req_id, slot_id, "m", b"img")
            msg_type, _ = harness.recv()
            assert msg_type == T_RESULT_READY
            harness.free(slot_id)

        time.sleep(0.05)
        assert harness.mmp._pool.free_count == harness.mmp._n_slots

    def test_slot_status_transitions(self, harness):
        """Verify header status transitions throughout the lifecycle."""
        result_ready = threading.Event()

        class _SlowBackend:
            is_healthy = True

            def signal_slot(self, slot_id, req_id, params_bytes=b"{}"):
                def _do():
                    pool = SHMPool.attach(
                        harness.shm_name,
                        n_slots=harness.mmp._n_slots,
                        input_mb=_TEST_INPUT_MB,
                    )
                    try:
                        pool.data_memoryview(slot_id)[:2] = b"ok"
                    finally:
                        pool.close()
                    harness.mmp.on_result(req_id, slot_id, 2)

                threading.Thread(target=_do, daemon=True).start()

        harness.mmp.register_backend("m", _SlowBackend())
        harness.ensure_loaded("m")

        req_id, slot_id = harness.alloc("m")
        # After alloc: ALLOCATED
        pool = SHMPool.attach(
            harness.shm_name,
            n_slots=harness.mmp._n_slots,
            input_mb=_TEST_INPUT_MB,
        )
        assert pool.read_header(slot_id).status == SlotStatus.ALLOCATED

        harness.submit(req_id, slot_id, "m", b"img")
        msg_type, frame = harness.recv()
        assert msg_type == T_RESULT_READY

        # After result: DONE
        assert pool.read_header(slot_id).status == SlotStatus.DONE
        pool.close()

        harness.free(slot_id)


class TestNoBackend:
    def test_submit_with_no_backend_returns_error(self, harness):
        """T_SUBMIT for unregistered flavor → T_ERROR (no backend)."""
        # Use stub load (marks loaded but no backend registered)
        harness.ensure_loaded("ghost-model")
        req_id, slot_id = harness.alloc("ghost-model")
        harness.submit(req_id, slot_id, "ghost-model", b"data")
        msg_type, _ = harness.recv()
        assert msg_type == T_ERROR


class TestStaleReaper:
    def test_stale_slot_reaped_and_error_sent(self):
        """Slot held > max_age_s with no T_FREE → reaper sends T_ERROR."""
        h = _MMPHarness(n_slots=4)

        class _StuckBackend:
            """signal_slot does nothing — simulates a crashed worker."""

            def signal_slot(self, slot_id, req_id, params_bytes=b"{}"):
                pass  # never calls on_result

        h.mmp.register_backend("stuck", _StuckBackend())
        h.ensure_loaded("stuck")
        req_id, slot_id = h.alloc("stuck")
        h.submit(req_id, slot_id, "stuck", b"data")

        # Wait longer than stale_slot_max_age_s (2.0s) + reap interval (1.0s)
        msg_type, _ = h.recv()  # reaper sends T_ERROR
        h.teardown()

        assert msg_type == T_ERROR

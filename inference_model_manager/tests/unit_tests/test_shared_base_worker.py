from inference_model_manager.backends.shared_base_protocol import HeadIndexRegistry
from inference_model_manager.backends.shared_base_worker import (
    group_pending_by_head,
    handle_drop_head,
    handle_load_head,
)


def test_group_pending_routes_known_heads():
    reg = HeadIndexRegistry()
    reg.add("a", object())  # index 0
    reg.add("b", object())  # index 1
    pending = [
        (10, 100, 0, b"{}"),
        (11, 101, 1, b"{}"),
        (12, 102, 0, b"{}"),
    ]

    groups, unknown = group_pending_by_head(pending, reg)

    assert groups == {0: [(10, 100, b"{}"), (12, 102, b"{}")], 1: [(11, 101, b"{}")]}
    assert unknown == []


def test_group_pending_collects_unknown_head_index():
    reg = HeadIndexRegistry()
    reg.add("a", object())  # index 0
    reg.remove("a")  # index 0 retired
    pending = [(10, 100, 0, b"{}"), (11, 101, 7, b"{}")]

    groups, unknown = group_pending_by_head(pending, reg)

    assert groups == {}
    # both the retired index and the never-seen index are rejected, not misrouted.
    assert sorted(unknown) == [(10, 100), (11, 101)]


def test_handle_load_head_registers_and_acks():
    reg = HeadIndexRegistry()
    model = object()

    def load_fn(payload):
        return model, {"model_mro_names": ["H"], "max_batch_size": 8, "class_names": None}

    ack = handle_load_head(
        {"req_id": 1, "head_id": "h1", "api_key": "k"}, reg, load_fn
    )

    assert ack == {
        "req_id": 1,
        "ok": True,
        "head_id": "h1",
        "head_index": 0,
        "model_mro_names": ["H"],
        "max_batch_size": 8,
        "class_names": None,
    }
    assert reg.get(0) is model


def test_handle_load_head_failure_is_isolated():
    reg = HeadIndexRegistry()
    reg.add("existing", object())

    def load_fn(payload):
        raise RuntimeError("CUDA out of memory")

    ack = handle_load_head({"req_id": 2, "head_id": "h2"}, reg, load_fn)

    assert ack["ok"] is False
    assert "CUDA out of memory" in ack["error"]
    # base/other heads untouched: failed head not registered, existing head intact.
    assert "h2" not in reg
    assert "existing" in reg


def test_handle_load_head_duplicate_is_idempotent():
    # Retry after a parent-side control timeout must re-ack the existing head
    # (same index) instead of erroring — the error made the head permanently
    # unloadable after one timed-out load.
    reg = HeadIndexRegistry()
    model = object()
    reg.add("h1", model)

    def load_fn(payload):
        raise AssertionError("must not reload an already-loaded head")

    ack = handle_load_head({"req_id": 3, "head_id": "h1"}, reg, load_fn)

    assert ack["ok"] is True
    assert ack["head_id"] == "h1"
    assert ack["head_index"] == 0
    assert reg.get(0) is model


def test_handle_drop_head_removes_then_acks():
    reg = HeadIndexRegistry()
    reg.add("h1", object())
    removed = []

    ack = handle_drop_head(
        {"req_id": 4, "head_id": "h1"}, reg, on_removed=lambda idx, hid: removed.append((idx, hid))
    )

    assert ack == {"req_id": 4, "ok": True, "head_id": "h1"}
    assert "h1" not in reg
    assert removed == [(0, "h1")]


def test_handle_drop_unknown_head_acks_ok():
    reg = HeadIndexRegistry()
    ack = handle_drop_head({"req_id": 5, "head_id": "ghost"}, reg)
    assert ack == {"req_id": 5, "ok": True, "head_id": "ghost"}


# ----------------------------- _error_slots gating -----------------------------


class _FakeSock:
    def __init__(self):
        self.sent = []

    def send_multipart(self, frames):
        self.sent.append(frames)


class _Log:
    def warning(self, *a, **k):
        pass

    def exception(self, *a, **k):
        pass


def _result_frames(sock):
    import struct

    from inference_model_manager.backends.subproc import _MSG_RESULT

    return [struct.unpack(">QII", f[1]) for f in sock.sent if f[0] == _MSG_RESULT]


def test_error_slots_skips_completed_slot():
    from inference_model_manager.backends.shared_base_worker import _error_slots
    from inference_model_manager.backends.utils.shm_pool import SHMPool, SlotStatus

    pool = SHMPool.create(n_slots=2, input_mb=0.1)
    try:
        s_live = pool.alloc_slot()
        pool.mark_allocated(s_live, request_id=11)
        pool.mark_written(s_live, 4)
        s_done = pool.alloc_slot()
        pool.mark_allocated(s_done, request_id=22)
        pool.mark_written(s_done, 4)
        pool.mark_done(s_done, 8, request_id=22)  # already resulted pre-crash

        sock = _FakeSock()
        _error_slots(pool, sock, [(s_live, 11), (s_done, 22)], "batch crashed", _Log())

        assert pool.read_header(s_live).status == SlotStatus.ERROR
        # Completed slot untouched: no error stomp, no duplicate _MSG_RESULT.
        hdr = pool.read_header(s_done)
        assert hdr.status == SlotStatus.DONE
        assert hdr.result_size == 8
        assert _result_frames(sock) == [(11, s_live, 0)]
    finally:
        pool.close()


def test_error_slots_skips_rebound_slot():
    from inference_model_manager.backends.shared_base_worker import _error_slots
    from inference_model_manager.backends.utils.shm_pool import SHMPool, SlotStatus

    pool = SHMPool.create(n_slots=1, input_mb=0.1)
    try:
        s = pool.alloc_slot()
        pool.mark_allocated(s, request_id=11)
        pool.mark_written(s, 4)
        # Reaper reclaimed and rebound the slot to a new request mid-crash.
        pool.free_slot(s, request_id=11)
        s2 = pool.alloc_slot()
        assert s2 == s
        pool.mark_allocated(s2, request_id=99)

        sock = _FakeSock()
        _error_slots(pool, sock, [(s, 11)], "batch crashed", _Log())

        hdr = pool.read_header(s)
        assert hdr.status == SlotStatus.ALLOCATED  # new owner untouched
        assert hdr.request_id == 99
        assert _result_frames(sock) == []
    finally:
        pool.close()

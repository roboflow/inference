import queue

import pytest

from inference_model_manager.backends.shared_base import (
    SharedBaseSubprocessBackend,
    SharedHeadControlPlane,
    _HeadSlotTracker,
)
from inference_model_manager.backends.shared_base_protocol import (
    MSG_HEAD_SLOT_READY,
    decode_control,
    pack_head_slot,
)


# ----------------------------- slot tracker -----------------------------


def test_tracker_per_head_counts():
    t = _HeadSlotTracker()
    t.signal("a", 1)
    t.signal("a", 2)
    t.signal("b", 3)

    assert t.outstanding("a") == 2
    assert t.outstanding("b") == 1
    assert t.total() == 3


def test_tracker_complete_decrements_right_head():
    t = _HeadSlotTracker()
    t.signal("a", 1)
    t.signal("b", 2)

    assert t.complete(1) == "a"
    assert t.outstanding("a") == 0
    assert t.outstanding("b") == 1


def test_tracker_complete_unknown_slot_is_noop():
    t = _HeadSlotTracker()
    assert t.complete(99) is None


def test_tracker_forget_drops_head_and_its_slots():
    t = _HeadSlotTracker()
    t.signal("a", 1)
    t.signal("a", 2)
    t.forget("a")

    assert t.outstanding("a") == 0
    assert t.complete(1) is None  # slot mapping gone too


# ----------------------------- owner logic (no spawn) -----------------------------


def _auto_ack_send(cp, **ack_fields):
    def send(tag, payload):
        cp.on_ack({"req_id": decode_control(payload)["req_id"], **ack_fields})

    return send


def _owner_without_worker(on_death=None, on_empty=None, on_result=None):
    import threading

    owner = SharedBaseSubprocessBackend.__new__(SharedBaseSubprocessBackend)
    owner._base_key = "bk"
    owner._recv_dead = False
    owner._death_handled = False
    owner._retired = False
    owner._state_value = "loaded"
    owner._outbound = queue.Queue()
    owner._tracker = _HeadSlotTracker()
    owner._slot_lock = threading.Lock()
    owner._slot_meta = {}
    owner._load_lock = threading.Lock()
    owner._inflight_loads = 0
    owner._worker_stats = {}
    owner._worker_stats_event = None
    owner._on_shared_worker_death = on_death
    owner._on_empty = on_empty
    owner._on_result_callback = on_result
    owner._pool = None
    cp = SharedHeadControlPlane(send=lambda *a: None)
    cp._send = _auto_ack_send(cp, ok=True, head_index=0)
    owner._control = cp
    return owner


def test_signal_slot_enqueues_head_frame_and_tracks():
    owner = _owner_without_worker()
    owner._control.load_head("h1", {})

    owner.signal_slot("h1", slot_id=7, req_id=42, params_bytes=b"{}")

    kind, head_index, slot_id, req_id, params = owner._outbound.get_nowait()
    assert (kind, head_index, slot_id, req_id) == ("slot", 0, 7, 42)
    # frame packs the head_index the worker routes on.
    assert pack_head_slot(slot_id, req_id, head_index)[:14]
    assert owner._tracker.outstanding("h1") == 1


def test_signal_slot_rejects_unknown_head():
    owner = _owner_without_worker()
    with pytest.raises(RuntimeError, match="not loaded"):
        owner.signal_slot("ghost", 1, 1)


def test_signal_slot_rejects_when_worker_dead():
    owner = _owner_without_worker()
    owner._control.load_head("h1", {})
    owner._recv_dead = True
    with pytest.raises(RuntimeError, match="worker dead"):
        owner.signal_slot("h1", 1, 1)


def test_worker_death_fails_control_and_fires_callback():
    fired = []
    owner = _owner_without_worker(on_death=fired.append)

    owner._handle_worker_death("boom")

    assert owner._state_value == "unhealthy"
    assert owner._recv_dead is True
    assert fired == ["bk"]
    # second call is idempotent
    owner._handle_worker_death("again")
    assert fired == ["bk"]


def test_drop_last_head_tears_down_worker():
    owner = _owner_without_worker()
    owner._control.load_head("h1", {})
    unloaded = []
    owner.unload = lambda: unloaded.append(True)
    owner._control._send = _auto_ack_send(owner._control, ok=True)

    owner.drop_head("h1")

    assert owner._control.head_count() == 0
    assert unloaded == [True]


def test_drop_one_of_two_heads_keeps_worker():
    owner = _owner_without_worker()
    owner._control.load_head("h1", {})
    owner._control._send = _auto_ack_send(owner._control, ok=True, head_index=1)
    owner._control.load_head("h2", {})
    unloaded = []
    owner.unload = lambda: unloaded.append(True)
    owner._control._send = _auto_ack_send(owner._control, ok=True)

    owner.drop_head("h1")

    assert owner._control.head_count() == 1
    assert unloaded == []


# ---- result / death slot completion + retire gating ----


class _FakePool:
    def __init__(self, payload):
        self._payload = payload

    def data_memoryview(self, slot_id):
        return memoryview(self._payload)


def test_submit_slot_registers_future_and_frame():
    owner = _owner_without_worker()
    owner._control.load_head("h1", {})
    fut = object()

    owner.submit_slot("h1", slot_id=5, req_id=9, future=fut, params_bytes=b"{}")

    assert owner._slot_meta[5] == (9, fut, "h1")
    assert owner._tracker.outstanding("h1") == 1
    kind, head_index, slot_id, req_id, _ = owner._outbound.get_nowait()
    assert (kind, head_index, slot_id, req_id) == ("slot", 0, 5, 9)


def test_handle_result_resolves_future_and_calls_callback():
    import pickle
    from concurrent.futures import Future

    results = []
    payload = pickle.dumps({"ok": 1})
    owner = _owner_without_worker(on_result=lambda r, s, sz: results.append((r, s, sz)))
    owner._pool = _FakePool(payload)
    owner._control.load_head("h1", {})
    fut = Future()
    owner.submit_slot("h1", 5, 9, future=fut)
    owner._outbound.get_nowait()

    owner._handle_result(9, 5, result_sz=len(payload))

    assert fut.result() == {"ok": 1}
    assert results == [(9, 5, len(payload))]
    assert owner._tracker.outstanding("h1") == 0
    assert 5 not in owner._slot_meta


def test_handle_result_error_size_fails_future():
    from concurrent.futures import Future

    owner = _owner_without_worker()
    owner._control.load_head("h1", {})
    fut = Future()
    owner.submit_slot("h1", 5, 9, future=fut)
    owner._outbound.get_nowait()

    owner._handle_result(9, 5, result_sz=0)

    with pytest.raises(RuntimeError, match="inference error"):
        fut.result()


def test_worker_death_completes_inflight_slots():
    from concurrent.futures import Future

    errored = []
    owner = _owner_without_worker(on_result=lambda r, s, sz: errored.append((r, s, sz)))
    owner._control.load_head("h1", {})
    fut = Future()
    owner.submit_slot("h1", 5, 9, future=fut)
    owner._outbound.get_nowait()

    owner._handle_worker_death("boom")

    assert errored == [(9, 5, 0)]  # callback fired with result_sz=0 (error)
    with pytest.raises(RuntimeError, match="worker died"):
        fut.result()
    assert owner._slot_meta == {}
    assert owner._tracker.outstanding("h1") == 0  # drain/queue_depth not left stuck


def test_maybe_retire_skips_when_load_in_flight():
    retired = []
    owner = _owner_without_worker(on_empty=lambda bk, o: retired.append(bk))
    owner.unload = lambda: None
    owner._inflight_loads = 1  # a concurrent first head is still loading

    owner._maybe_retire()  # head_count==0 but a load is pending

    assert owner._retired is False
    assert retired == []


def test_maybe_retire_reaps_when_empty_and_idle():
    retired = []
    owner = _owner_without_worker(on_empty=lambda bk, o: retired.append(bk))
    unloaded = []
    owner.unload = lambda: unloaded.append(True)

    owner._maybe_retire()  # no heads, no in-flight load

    assert owner._retired is True
    assert unloaded == [True]
    assert retired == ["bk"]


def test_begin_load_reservation_blocks_retire_until_released():
    retired = []
    owner = _owner_without_worker(on_empty=lambda bk, o: retired.append(bk))
    owner.unload = lambda: None

    assert owner.begin_load() is True
    assert owner.begin_load() is True  # two concurrent first-head loads
    owner.end_load()  # one fails: head_count==0 but a reservation remains
    assert owner._retired is False
    assert retired == []
    owner.end_load()  # last reservation released → reap
    assert owner._retired is True
    assert retired == ["bk"]


def test_begin_load_returns_false_when_retired():
    owner = _owner_without_worker()
    owner.unload = lambda: None
    owner._retired = True

    assert owner.begin_load() is False


def test_begin_load_returns_false_when_worker_dead():
    # Death sets _recv_dead before the cache entry is popped — a reservation must
    # still be refused so the manager creates a fresh owner instead.
    owner = _owner_without_worker()
    owner._recv_dead = True

    assert owner.begin_load() is False


class _Closeable:
    def __init__(self):
        self.closed = False

    def close(self, *a, **k):
        self.closed = True

    def term(self):
        self.closed = True


def test_close_transport_detaches_resources():
    owner = SharedBaseSubprocessBackend.__new__(SharedBaseSubprocessBackend)
    owner._pool = _Closeable()
    owner._zmq_sock = _Closeable()
    owner._zmq_ctx = _Closeable()
    owner._zmq_addr = "tcp://127.0.0.1:5555"

    owner._close_transport()

    assert owner._pool.closed
    assert owner._zmq_sock.closed
    assert owner._zmq_ctx.closed


def test_refresh_worker_stats_returns_cached_when_dead():
    owner = _owner_without_worker()
    owner._recv_dead = True
    owner._worker_stats = {"throughput_fps": 7.0}

    assert owner.refresh_worker_stats() == {"throughput_fps": 7.0}

import threading

import pytest

from inference_model_manager.backends.shared_base import (
    HeadMetadata,
    SharedHeadBackend,
    SharedHeadControlPlane,
)
from inference_model_manager.backends.shared_base_protocol import (
    MSG_DROP_HEAD,
    MSG_LOAD_HEAD,
    decode_control,
)


# ----------------------------- control plane -----------------------------


def _auto_ack(cp, ack_fields):
    """Return a send() that immediately acks every request — simulates the worker."""

    def send(tag, payload):
        req_id = decode_control(payload)["req_id"]
        cp.on_ack({"req_id": req_id, **ack_fields})

    return send


def test_load_head_resolves_on_ack():
    cp = SharedHeadControlPlane(send=lambda *a: None)
    cp._send = _auto_ack(
        cp,
        {"ok": True, "head_index": 5, "model_mro_names": ["Foo"], "max_batch_size": 8},
    )

    meta = cp.load_head("head/1", {"api_key": "k", "device": "cpu"})

    assert meta == HeadMetadata(
        head_index=5, model_mro_names=["Foo"], max_batch_size=8, class_names=None
    )
    assert cp.has_head("head/1")
    assert cp.head_count() == 1


def test_load_head_sends_load_tag_and_fields():
    sent = {}

    def send(tag, payload):
        sent["tag"] = tag
        sent["payload"] = decode_control(payload)
        cp.on_ack({"req_id": sent["payload"]["req_id"], "ok": True, "head_index": 0})

    cp = SharedHeadControlPlane(send=send)
    cp.load_head("head/1", {"api_key": "k", "device": "cuda:0"})

    assert sent["tag"] == MSG_LOAD_HEAD
    assert sent["payload"]["head_id"] == "head/1"
    assert sent["payload"]["device"] == "cuda:0"


def test_load_head_raises_on_negative_ack():
    cp = SharedHeadControlPlane(send=lambda *a: None)
    cp._send = _auto_ack(cp, {"ok": False, "error": "CUDA OOM"})

    with pytest.raises(RuntimeError, match="CUDA OOM"):
        cp.load_head("head/1", {})
    assert not cp.has_head("head/1")


def test_load_head_times_out_when_no_ack():
    cp = SharedHeadControlPlane(send=lambda *a: None)  # never acks

    with pytest.raises(TimeoutError):
        cp.load_head("head/1", {}, timeout_s=0.05)
    # pending cleaned up, not leaked.
    assert cp._pending == {}


def test_load_head_raises_when_send_fails():
    def send(tag, payload):
        raise ConnectionError("socket dead")

    cp = SharedHeadControlPlane(send=send)
    with pytest.raises(ConnectionError):
        cp.load_head("head/1", {})
    assert cp._pending == {}


def test_fail_all_unblocks_pending_load():
    cp = SharedHeadControlPlane(send=lambda *a: None)  # never acks
    result = {}

    def worker():
        try:
            cp.load_head("head/1", {}, timeout_s=5)
        except Exception as exc:  # noqa: BLE001
            result["exc"] = exc

    t = threading.Thread(target=worker)
    t.start()
    # let the loader register its pending future, then kill the worker.
    while not cp._pending:
        pass
    cp.fail_all(RuntimeError("worker died"))
    t.join(timeout=2)

    assert isinstance(result.get("exc"), RuntimeError)


def test_load_head_after_death_raises_immediately():
    cp = SharedHeadControlPlane(send=lambda *a: None)
    cp.fail_all(RuntimeError("dead"))

    with pytest.raises(RuntimeError, match="dead"):
        cp.load_head("head/1", {})


def test_drop_head_forgets_only_after_ack():
    cp = SharedHeadControlPlane(send=lambda *a: None)
    cp._send = _auto_ack(cp, {"ok": True, "head_index": 0})
    cp.load_head("head/1", {})

    sent = {}

    def send(tag, payload):
        sent["tag"] = tag
        cp.on_ack({"req_id": decode_control(payload)["req_id"], "ok": True})

    cp._send = send
    cp.drop_head("head/1")

    assert sent["tag"] == MSG_DROP_HEAD
    assert not cp.has_head("head/1")
    assert cp.head_count() == 0


def test_drop_head_keeps_head_on_negative_ack():
    cp = SharedHeadControlPlane(send=lambda *a: None)
    cp._send = _auto_ack(cp, {"ok": True, "head_index": 0})
    cp.load_head("head/1", {})

    cp._send = _auto_ack(cp, {"ok": False, "error": "boom"})
    with pytest.raises(RuntimeError, match="boom"):
        cp.drop_head("head/1")
    assert cp.has_head("head/1")


def test_drop_unknown_head_is_noop():
    cp = SharedHeadControlPlane(send=lambda *a: pytest.fail("should not send"))
    cp.drop_head("ghost")


# ----------------------------- head view -----------------------------


class _FakeOwner:
    def __init__(self):
        self.device = "cuda:0"
        self.state = "loaded"
        self.is_healthy = True
        self.is_accepting = True
        self._heads = {"head/1"}
        self.dropped = []
        self.drained = []
        self.signals = []
        self._depth = 3

    def has_head(self, head_id):
        return head_id in self._heads

    def head_queue_depth(self, head_id):
        return self._depth

    def drop_head(self, head_id):
        self.dropped.append(head_id)
        self._heads.discard(head_id)

    def drain_head(self, head_id, timeout_s):
        self.drained.append((head_id, timeout_s))

    def signal_slot(self, head_id, slot_id, req_id, params_bytes):
        self.signals.append((head_id, slot_id, req_id, params_bytes))

    def submit_slot(self, head_id, slot_id, req_id, future, params_bytes):
        self.submits = getattr(self, "submits", [])
        self.submits.append((head_id, slot_id, req_id, future, params_bytes))

    def set_on_result_callback(self, callback):
        self.result_cb = callback

    def worker_stats(self):
        return {"throughput_fps": 12.0, "inference_count": 5}

    def refresh_worker_stats(self, timeout_s=1.0):
        return self.worker_stats()


def _view(owner):
    meta = HeadMetadata(
        head_index=2, model_mro_names=["H"], max_batch_size=4, class_names=["c"]
    )
    return SharedHeadBackend(owner, "head/1", meta)


def test_view_delegates_liveness_and_serves_metadata():
    owner = _FakeOwner()
    view = _view(owner)

    assert view.device == "cuda:0"
    assert view.state == "loaded"
    assert view.is_healthy is True
    assert view.is_accepting is True
    assert view.max_batch_size == 4
    assert view.class_names == ["c"]
    assert view.queue_depth == 3
    assert view._model_mro_names == ["H"]


def test_view_worker_pid_is_none():
    assert _view(_FakeOwner()).worker_pid is None
    assert _view(_FakeOwner()).stats()["worker_pid"] is None


def test_view_unhealthy_when_head_gone():
    owner = _FakeOwner()
    owner._heads.discard("head/1")
    view = _view(owner)

    assert view.is_healthy is False
    assert view.is_accepting is False


def test_view_signal_routes_to_owner_with_head_id():
    owner = _FakeOwner()
    _view(owner).signal_slot(7, 99, b"{}")

    assert owner.signals == [("head/1", 7, 99, b"{}")]


def test_view_unload_drops_head_only():
    owner = _FakeOwner()
    _view(owner).unload()

    assert owner.dropped == ["head/1"]


def test_view_drain_then_drop():
    owner = _FakeOwner()
    _view(owner).drain_and_unload(timeout_s=12)

    assert owner.drained == [("head/1", 12)]
    assert owner.dropped == ["head/1"]


def test_view_submit_slot_routes_to_owner_with_head_id():
    owner = _FakeOwner()
    sentinel = object()
    _view(owner).submit_slot(7, 99, sentinel, b"{}")

    assert owner.submits == [("head/1", 7, 99, sentinel, b"{}")]


def test_view_set_on_result_callback_routes_to_owner():
    owner = _FakeOwner()
    cb = lambda r, s, sz: None
    _view(owner).set_on_result_callback(cb)

    assert owner.result_cb is cb


# ----------------- control-timeout split-brain reconciliation -----------------


def test_late_load_ack_reconciles_head():
    cp = SharedHeadControlPlane(send=lambda *a: None)  # never acks in time
    with pytest.raises(TimeoutError):
        cp.load_head("head/1", {}, timeout_s=0.05)

    # Worker finished the load after the timeout — late ack must register the
    # head parent-side instead of being dropped (orphaned head = VRAM leak).
    cp.on_ack(
        {
            "req_id": 1,
            "ok": True,
            "head_id": "head/1",
            "head_index": 7,
            "model_mro_names": ["Foo"],
        }
    )
    assert cp.has_head("head/1")
    assert cp.metadata("head/1").head_index == 7


def test_load_head_returns_existing_metadata_for_loaded_head():
    cp = SharedHeadControlPlane(send=lambda *a: None)
    cp._send = _auto_ack(cp, {"ok": True, "head_index": 0, "model_mro_names": ["Foo"]})
    first = cp.load_head("head/1", {})
    # Retry (e.g. after a reconciled timeout) must be idempotent, not an error.
    second = cp.load_head("head/1", {})
    assert second == first


def test_late_drop_ack_removes_head():
    cp = SharedHeadControlPlane(send=lambda *a: None)
    cp._send = _auto_ack(cp, {"ok": True, "head_index": 0})
    cp.load_head("head/1", {})

    cp.on_ack({"req_id": 99, "ok": True, "head_id": "head/1"})
    assert not cp.has_head("head/1")


def test_drop_head_timeout_forgets_head_and_returns():
    cp = SharedHeadControlPlane(send=lambda *a: None)
    cp._send = _auto_ack(cp, {"ok": True, "head_index": 0})
    cp.load_head("head/1", {})

    cp._send = lambda *a: None  # drop never acked
    cp.drop_head("head/1", timeout_s=0.05)  # must not raise
    # Forgotten parent-side so head_count reaches 0 and the owner can retire —
    # retirement kills the worker, reclaiming the head even if the drop never landed.
    assert not cp.has_head("head/1")
    assert cp.head_count() == 0
    assert cp._pending == {}

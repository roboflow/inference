from dataclasses import dataclass, field

from inference_model_manager.model_manager import ModelManager
from inference_model_manager.model_manager_process import ModelManagerProcess, ModelState


# ----- ModelManager death hook plumbing -----


def test_manager_death_pops_cache_and_calls_hook():
    mm = ModelManager.__new__(ModelManager)
    import threading

    mm._lifecycle_lock = threading.Lock()
    mm._shared_workers = {"bk-1": object()}
    called = []
    mm._shared_death_hook = called.append

    mm._on_shared_worker_death("bk-1")

    assert "bk-1" not in mm._shared_workers
    assert called == ["bk-1"]


def test_manager_death_without_hook_just_pops():
    mm = ModelManager.__new__(ModelManager)
    import threading

    mm._lifecycle_lock = threading.Lock()
    mm._shared_workers = {"bk-1": object()}
    mm._shared_death_hook = None

    mm._on_shared_worker_death("bk-1")  # no hook → no error

    assert mm._shared_workers == {}


# ----- MMP shared-head bookkeeping + death cleanup -----


def _mmp():
    p = ModelManagerProcess.__new__(ModelManagerProcess)
    p._shared_heads = {}
    p._head_base_key = {}
    p._models = {}
    p._inflight = {}
    p._backends = {}

    class _Mgr:
        def __init__(self):
            self._backends = {}

    p._manager = _Mgr()
    return p


def _track(p, base_key, head_id):
    p._shared_heads.setdefault(base_key, set()).add(head_id)
    p._head_base_key[head_id] = base_key
    p._models[head_id] = ModelState()
    p._models[head_id].loaded = True
    p._backends[head_id] = object()
    p._manager._backends[head_id] = object()


def test_untrack_removes_head_and_empty_base():
    p = _mmp()
    _track(p, "bk-1", "head/1")
    _track(p, "bk-1", "head/2")

    p._untrack_shared_head("head/1")
    assert p._shared_heads["bk-1"] == {"head/2"}
    assert "head/1" not in p._head_base_key

    p._untrack_shared_head("head/2")
    assert "bk-1" not in p._shared_heads  # last head gone → base entry dropped


def test_untrack_unknown_head_is_noop():
    p = _mmp()
    p._untrack_shared_head("ghost")  # no error


def test_cleanup_dead_shared_base_marks_all_heads_not_loaded():
    p = _mmp()
    _track(p, "bk-1", "head/1")
    _track(p, "bk-1", "head/2")
    p._inflight = {10: "head/1", 11: "head/2", 12: "other"}

    p._cleanup_dead_shared_base("bk-1")

    for hid in ("head/1", "head/2"):
        assert p._models[hid].loaded is False
        assert p._models[hid].loading is False
        assert hid not in p._backends
        assert hid not in p._manager._backends
        assert hid not in p._head_base_key
    assert "bk-1" not in p._shared_heads
    # only the dead base's inflight slots cleared; unrelated slot survives.
    assert p._inflight == {12: "other"}


def test_cleanup_unknown_base_is_noop():
    p = _mmp()
    p._cleanup_dead_shared_base("ghost")  # no error

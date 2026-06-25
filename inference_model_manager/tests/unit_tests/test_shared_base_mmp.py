from dataclasses import dataclass, field

from inference_model_manager.model_manager import ModelManager
from inference_model_manager.model_manager_process import (
    ModelManagerProcess,
    ModelState,
)

# ----- ModelManager death hook plumbing -----


def test_manager_death_pops_cache_and_calls_hook():
    mm = ModelManager.__new__(ModelManager)
    import threading

    mm._lifecycle_lock = threading.Lock()
    mm._shared_workers = {"bk-1": object()}
    mm._shared_base_preloads = {}
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
    mm._shared_base_preloads = {}
    mm._shared_death_hook = None

    mm._on_shared_worker_death("bk-1")  # no hook → no error

    assert mm._shared_workers == {}


# ----- MMP shared-head bookkeeping + death cleanup -----


def _mmp():
    p = ModelManagerProcess.__new__(ModelManagerProcess)
    p._shared_heads = {}
    p._head_base_key = {}
    p._preloaded_shared_bases = {}
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


def test_cleanup_dead_shared_base_marks_preloaded_base_not_loaded():
    p = _mmp()
    p._preloaded_shared_bases["base"] = "bk-1"
    p._models["base"] = ModelState(loaded=True, pinned=True)

    p._cleanup_dead_shared_base("bk-1")

    assert p._preloaded_shared_bases == {}
    assert p._models["base"].loaded is False
    assert p._models["base"].loading is False


def test_cleanup_unknown_base_is_noop():
    p = _mmp()
    p._cleanup_dead_shared_base("ghost")  # no error


# ----- VRAM accounting: shared heads vs base-key eviction units -----

import asyncio
import time


def _vmmp():
    p = ModelManagerProcess.__new__(ModelManagerProcess)
    p._shared_heads = {}
    p._head_base_key = {}
    p._preloaded_shared_bases = {}
    p._models = {}
    p._backends = {}
    p._pending = {}
    p._unloading = set()
    p._model_access = {}
    p._model_request_times = {}
    p._vram_window = {}
    p._vram_meta_cache = {}
    p._vram_idle_cutoff_s = 300.0
    p._idle_timeout_s = 300.0
    p._vram_recent_window_s = 60.0
    return p


def _load_head(p, base_key, head_id, *, access=0.0):
    p._shared_heads.setdefault(base_key, set()).add(head_id)
    p._head_base_key[head_id] = base_key
    p._models[head_id] = ModelState(loaded=True)
    p._model_access[head_id] = access


def test_shared_head_footprint_is_zero():
    p = _vmmp()
    _load_head(p, "bk-1", "head/1")
    p._vram_meta_cache["head/1"] = 9999  # would be huge as a normal model
    assert p._footprint_mb("head/1") == 0.0


def test_base_key_footprint_from_window():
    p = _vmmp()
    _load_head(p, "bk-1", "head/1")
    from collections import deque

    p._vram_window["bk-1"] = deque([1200.0, 1500.0])
    assert p._footprint_mb("bk-1") == 1500.0  # whole worker VRAM


def test_evictable_units_excludes_heads_includes_base_key():
    p = _vmmp()
    _load_head(p, "bk-1", "head/1")
    _load_head(p, "bk-1", "head/2")
    p._models["normal"] = ModelState(loaded=True)

    units = set(p._evictable_units())
    assert units == {"normal", "bk-1"}  # heads are not units; base_key is


def test_base_key_evictable_only_when_all_heads_cold():
    p = _vmmp()
    old = time.monotonic() - 10_000  # cold
    _load_head(p, "bk-1", "head/1", access=old)
    _load_head(p, "bk-1", "head/2", access=time.monotonic())  # hot

    now = time.monotonic()
    assert p._unit_evictable("bk-1", now, set(), p._vram_idle_cutoff_s) is False

    p._model_access["head/2"] = old  # now both cold
    assert p._unit_evictable("bk-1", now, set(), p._vram_idle_cutoff_s) is True


def test_preloaded_base_key_not_evictable_even_when_heads_cold():
    p = _vmmp()
    old = time.monotonic() - 10_000
    _load_head(p, "bk-1", "head/1", access=old)
    _load_head(p, "bk-1", "head/2", access=old)
    p._preloaded_shared_bases["base"] = "bk-1"

    now = time.monotonic()
    assert p._unit_evictable("bk-1", now, set(), p._vram_idle_cutoff_s) is False
    assert p._plan_evictions(1.0) is None


def test_plan_evictions_uses_base_key_footprint():
    p = _vmmp()
    old = time.monotonic() - 10_000
    _load_head(p, "bk-1", "head/1", access=old)
    _load_head(p, "bk-1", "head/2", access=old)
    from collections import deque

    p._vram_window["bk-1"] = deque([2000.0])

    plan = p._plan_evictions(1500.0)
    assert plan == ["bk-1"]  # one base unit covers the deficit; heads not listed


def test_evict_target_base_key_drops_all_heads():
    p = _vmmp()
    _load_head(p, "bk-1", "head/1")
    _load_head(p, "bk-1", "head/2")
    evicted = []

    async def fake_evict(model_id):
        evicted.append(model_id)
        return True

    p._evict_model = fake_evict
    ok = asyncio.run(p._evict_target("bk-1"))

    assert ok is True
    assert sorted(evicted) == ["head/1", "head/2"]


def test_collect_stats_attributes_owner_pid_to_base_key(monkeypatch):
    import inference_model_manager.model_manager_process as mod

    captured = {}

    def fake_gpu(pid_to_flavor=None):
        captured["map"] = dict(pid_to_flavor or {})
        return {"gpus": [], "per_model_gpu_mb": {}}

    monkeypatch.setattr(mod, "_collect_gpu_stats", fake_gpu)

    p = _vmmp()

    class _Owner:
        worker_pid = 4242

    class _Mgr:
        def stats(self):
            return {}

        def shared_owners(self):
            return {"bk-1": _Owner()}

    p._manager = _Mgr()
    p._collect_stats()

    assert captured["map"].get(4242) == "bk-1"  # base worker VRAM → base_key


def test_shared_head_marginal_mb_subtracts_base():
    p = _vmmp()
    p._vram_meta_cache["head/1"] = 5000  # whole model (base + head)
    p._vram_meta_cache["owlv2"] = 4200  # base alone
    res = type("R", (), {"dep_model_id": "owlv2"})()

    assert p._shared_head_marginal_mb(res, "head/1", "k") == 800


def test_shared_head_marginal_clamped_at_zero():
    p = _vmmp()
    p._vram_meta_cache["head/1"] = 4000
    p._vram_meta_cache["owlv2"] = 4200  # noisy: base > whole
    res = type("R", (), {"dep_model_id": "owlv2"})()

    assert p._shared_head_marginal_mb(res, "head/1", "k") == 0


def test_admission_plan_uses_need_override():
    p = _vmmp()
    p._vram_headroom_mb = 0.0
    p._gpu_free_mb = lambda: 1000.0
    p._vram_meta_cache["head/1"] = 5000  # full would force evict
    # marginal need fits in free → admit without eviction
    decision, victims, deficit = asyncio.run(
        p._vram_admission_plan("head/1", "k", need_mb=300)
    )
    assert decision == "admit"
    # ...whereas the full footprint would not fit
    decision2, _, _ = asyncio.run(p._vram_admission_plan("head/1", "k"))
    assert decision2 != "admit"


def test_pick_candidate_excludes_target_base():
    p = _vmmp()
    old = time.monotonic() - 10_000
    _load_head(p, "bk-1", "head/1", access=old)  # only cold unit is its own base
    from collections import deque

    p._vram_window["bk-1"] = deque([2000.0])

    # without exclusion the base is the candidate; excluded → nothing to evict.
    assert p._pick_eviction_candidate() == "bk-1"
    assert p._pick_eviction_candidate(exclude={"bk-1"}) is None


def test_admission_plan_excludes_target_base():
    p = _vmmp()
    p._vram_headroom_mb = 0.0
    p._gpu_free_mb = lambda: 100.0  # tight: needs eviction
    old = time.monotonic() - 10_000
    _load_head(p, "bk-1", "head/1", access=old)  # only cold unit is its own base
    from collections import deque

    p._vram_window["bk-1"] = deque([2000.0])

    # Without exclusion the base would be picked to satisfy the deficit...
    d_no, victims, _ = asyncio.run(p._vram_admission_plan("head/2", "k", need_mb=500))
    assert d_no == "evict" and victims == ["bk-1"]

    # ...with the base excluded there's nothing else to evict → no_capacity.
    d_ex, _, _ = asyncio.run(
        p._vram_admission_plan("head/2", "k", need_mb=500, exclude={"bk-1"})
    )
    assert d_ex == "no_capacity"


def test_check_and_evict_uses_evict_target_for_base_key(monkeypatch):
    import inference_model_manager.model_manager_process as mod

    p = _vmmp()
    p._manager = object()  # non-None so _check_and_evict proceeds
    p._evict_threshold = 0.5
    old = time.monotonic() - 10_000
    _load_head(p, "bk-1", "head/1", access=old)
    _load_head(p, "bk-1", "head/2", access=old)
    monkeypatch.setattr(mod, "_gpu_used_fraction", lambda: 0.9)

    targeted = []

    async def fake_target(key):
        targeted.append(key)
        return True

    p._evict_target = fake_target
    asyncio.run(p._check_and_evict())

    assert targeted == [
        "bk-1"
    ]  # base key routed through _evict_target, not _evict_model

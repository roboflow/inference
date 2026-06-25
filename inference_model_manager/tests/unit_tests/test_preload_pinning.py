import asyncio
import time
from collections import deque
from types import SimpleNamespace

from inference_model_manager.model_manager_process import (
    T_OK,
    ModelManagerProcess,
    ModelState,
)


def _pmmp():
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


def _load_head(p, base_key, head_id, *, access, pinned=False):
    p._shared_heads.setdefault(base_key, set()).add(head_id)
    p._head_base_key[head_id] = base_key
    p._models[head_id] = ModelState(loaded=True, pinned=pinned)
    p._model_access[head_id] = access


# ----- pinned models are never evictable -----


def test_pinned_model_not_idle():
    p = _pmmp()
    old = time.monotonic() - 10_000  # cold
    p._models["m"] = ModelState(loaded=True, pinned=True)
    p._model_access["m"] = old
    now = time.monotonic()
    assert p._flavor_loaded_idle("m", now, set(), p._idle_timeout_s) is False


def test_pinned_model_excluded_from_pick():
    p = _pmmp()
    old = time.monotonic() - 10_000
    p._models["pinned"] = ModelState(loaded=True, pinned=True)
    p._model_access["pinned"] = old
    assert (
        p._pick_eviction_candidate() is None
    )  # only unit is pinned → nothing to evict


def test_pinned_model_not_in_eviction_plan():
    p = _pmmp()
    old = time.monotonic() - 10_000
    p._models["pinned"] = ModelState(loaded=True, pinned=True)
    p._model_access["pinned"] = old
    p._vram_meta_cache["pinned"] = 5000
    assert p._plan_evictions(1000.0) is None  # cannot cover deficit; pinned won't go


def test_unpinned_sibling_still_evictable():
    p = _pmmp()
    old = time.monotonic() - 10_000
    p._models["pinned"] = ModelState(loaded=True, pinned=True)
    p._model_access["pinned"] = old
    p._models["free"] = ModelState(loaded=True)
    p._model_access["free"] = old
    assert p._pick_eviction_candidate() == "free"


# ----- a pinned head keeps its shared base resident -----


def test_pinned_head_blocks_base_eviction():
    p = _pmmp()
    old = time.monotonic() - 10_000
    _load_head(p, "bk-1", "head/1", access=old, pinned=True)
    _load_head(p, "bk-1", "head/2", access=old)  # cold but sibling is pinned
    p._vram_window["bk-1"] = deque([2000.0])
    now = time.monotonic()
    assert p._unit_evictable("bk-1", now, set(), p._idle_timeout_s) is False
    assert p._pick_eviction_candidate() is None


def test_base_evictable_when_no_head_pinned():
    p = _pmmp()
    old = time.monotonic() - 10_000
    _load_head(p, "bk-1", "head/1", access=old)
    _load_head(p, "bk-1", "head/2", access=old)
    now = time.monotonic()
    assert p._unit_evictable("bk-1", now, set(), p._idle_timeout_s) is True


# ----- T_LOAD (preload/admin) pins; nothing else does -----


def _load_mmp():
    p = _pmmp()
    sent = []

    async def fake_send(identity, tag, payload):
        sent.append((tag, payload))

    p._send = fake_send
    p._spawn = lambda coro, name=None: coro.close()  # don't actually run the load

    async def fake_load(*a, **k):
        return None

    p._load_model = fake_load
    return p, sent


def test_t_load_pins_new_model():
    import struct

    p, _ = _load_mmp()
    frame = struct.pack(">QH", 1, len(b"m")) + b"m" + struct.pack(">H", 0)
    asyncio.run(p._handle_load(b"id", [frame]))
    assert p._models["m"].pinned is True


def test_t_load_pins_already_loaded_model():
    import struct

    p, sent = _load_mmp()
    p._models["m"] = ModelState(loaded=True)  # loaded via a prior request, not pinned
    frame = struct.pack(">QH", 1, len(b"m")) + b"m" + struct.pack(">H", 0)
    asyncio.run(p._handle_load(b"id", [frame]))
    assert p._models["m"].pinned is True  # admin load pins it retroactively
    assert sent and sent[0][0] == T_OK  # already loaded → immediate T_OK


def test_model_state_default_unpinned():
    assert ModelState().pinned is False


def test_pinned_top_level_base_uses_shared_base_preload(monkeypatch):
    import inference_model_manager.shared_base_resolution as resolver

    resolution = SimpleNamespace(base_key="bk-1")
    monkeypatch.setattr(resolver, "resolve_shared_base", lambda *a, **k: None)
    monkeypatch.setattr(
        resolver, "resolve_shared_base_model", lambda *a, **k: resolution
    )

    p = _pmmp()
    p._shared_metadata_cache = {}
    p._vram_admission = False
    p._load_oom_max_evictions = 1
    p._batch_max_size = 0
    p._batch_max_wait_ms = 5.0
    p._decoder = "imagecodecs"
    p._models["google/owlv2-large-patch14-ensemble"] = ModelState(
        loading=True, pinned=True
    )
    p._flush_load_waiters = lambda model_id: None

    calls = []

    class _Manager:
        def __contains__(self, model_id):
            return False

        def has_shared_base(self, base_key):
            return False

        def load_shared_base(self, *args, **kwargs):
            calls.append((args, kwargs))

    p._manager = _Manager()

    asyncio.run(
        p._load_model_inner(
            "google/owlv2-large-patch14-ensemble", api_key="k", device="cpu"
        )
    )

    assert calls
    assert calls[0][0][:3] == (
        "google/owlv2-large-patch14-ensemble",
        "k",
        resolution,
    )
    assert p._preloaded_shared_bases == {"google/owlv2-large-patch14-ensemble": "bk-1"}
    assert p._models["google/owlv2-large-patch14-ensemble"].loaded is True

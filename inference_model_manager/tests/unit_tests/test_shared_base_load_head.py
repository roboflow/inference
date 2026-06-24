from types import SimpleNamespace

import pytest

from inference_model_manager.backends.shared_base import HeadMetadata, SharedHeadBackend
from inference_model_manager.model_manager import ModelManager


def _resolution(base_key="bk-1"):
    return SimpleNamespace(
        dep_name="feature_extractor",
        dep_model_id="owlv2",
        dep_metadata_package_id=None,
        resolved_package_id="owlv2-pkg-a",
        base_key=base_key,
    )


class _FakeOwner:
    def __init__(self, base_key):
        self.base_key = base_key
        self._heads = {}
        self._next = 0
        self.dropped = []
        self._inflight = 0
        self._retired = False
        self._on_empty = None
        self.unloaded = False

    @property
    def retired(self):
        return self._retired

    def begin_load(self):
        if self._retired:
            return False
        self._inflight += 1
        return True

    def end_load(self):
        self._inflight -= 1
        if self.head_count() == 0 and self._inflight == 0:
            self._retired = True
            self.unloaded = True
            if self._on_empty is not None:
                self._on_empty(self.base_key, self)

    def load_head(self, head_id, api_key):
        idx = self._next
        self._next += 1
        meta = HeadMetadata(head_index=idx, model_mro_names=["H"], max_batch_size=4)
        self._heads[head_id] = meta
        return meta

    def head_count(self):
        return len(self._heads)

    def has_head(self, head_id):
        return head_id in self._heads


def _patch_factory(mm, owners, *, fail_on=None):
    created = []

    def factory(base_key, resolution, api_key, **kwargs):
        if fail_on is not None and base_key in fail_on:
            raise RuntimeError("base load failed")
        owner = owners.setdefault(base_key, _FakeOwner(base_key))
        # Mirror the real owner's reap hook so cache cleanup is exercised end-to-end.
        owner._on_empty = mm._retire_shared_owner
        created.append(base_key)
        return owner

    mm._make_shared_owner = factory
    return created


def _mm():
    mm = ModelManager.__new__(ModelManager)
    import threading

    mm._backends = {}
    mm._loading_ids = set()
    mm._lifecycle_lock = threading.Lock()
    mm._shared_workers = {}
    mm._shared_loading_keys = set()
    mm._shared_cv = threading.Condition(mm._lifecycle_lock)
    return mm


def test_load_shared_head_registers_view(monkeypatch):
    monkeypatch.setattr(
        "inference_model_manager.registry_defaults.lazy_register_by_names",
        lambda names: None,
    )
    mm = _mm()
    owners = {}
    _patch_factory(mm, owners)

    mm.load_shared_head("head/1", "k", _resolution())

    view = mm._backends["head/1"]
    assert isinstance(view, SharedHeadBackend)
    assert "bk-1" in mm._shared_workers
    assert owners["bk-1"].has_head("head/1")


def test_two_heads_same_base_reuse_one_owner(monkeypatch):
    monkeypatch.setattr(
        "inference_model_manager.registry_defaults.lazy_register_by_names",
        lambda names: None,
    )
    mm = _mm()
    owners = {}
    created = _patch_factory(mm, owners)

    mm.load_shared_head("head/1", "k", _resolution())
    mm.load_shared_head("head/2", "k", _resolution())

    assert created == ["bk-1"]  # owner created once, reused for second head
    assert len(mm._shared_workers) == 1
    assert owners["bk-1"].head_count() == 2


def test_different_base_keys_create_separate_owners(monkeypatch):
    monkeypatch.setattr(
        "inference_model_manager.registry_defaults.lazy_register_by_names",
        lambda names: None,
    )
    mm = _mm()
    owners = {}
    created = _patch_factory(mm, owners)

    mm.load_shared_head("head/1", "k", _resolution("bk-1"))
    mm.load_shared_head("head/2", "k", _resolution("bk-2"))

    assert sorted(created) == ["bk-1", "bk-2"]
    assert set(mm._shared_workers) == {"bk-1", "bk-2"}


def test_duplicate_head_id_rejected(monkeypatch):
    monkeypatch.setattr(
        "inference_model_manager.registry_defaults.lazy_register_by_names",
        lambda names: None,
    )
    mm = _mm()
    _patch_factory(mm, {})
    mm.load_shared_head("head/1", "k", _resolution())

    with pytest.raises(ValueError, match="already loaded"):
        mm.load_shared_head("head/1", "k", _resolution())


def test_base_creation_failure_clears_sentinel_and_raises():
    mm = _mm()
    _patch_factory(mm, {}, fail_on={"bk-1"})

    with pytest.raises(RuntimeError, match="base load failed"):
        mm.load_shared_head("head/1", "k", _resolution())

    # sentinel released + head reservation cleared, so a retry can proceed.
    assert mm._shared_loading_keys == set()
    assert mm._loading_ids == set()
    assert "head/1" not in mm._backends


def test_first_head_load_failure_reaps_owner(monkeypatch):
    mm = _mm()

    class _FailingHeadOwner(_FakeOwner):
        def load_head(self, head_id, api_key):
            raise RuntimeError("CUDA OOM")

    owner = _FailingHeadOwner("bk-1")

    def factory(base_key, resolution, api_key, **kwargs):
        owner._on_empty = mm._retire_shared_owner
        return owner

    mm._make_shared_owner = factory

    with pytest.raises(RuntimeError, match="CUDA OOM"):
        mm.load_shared_head("head/1", "k", _resolution())

    # end_load reaps the empty/idle worker; cache cleared, not orphaned.
    assert "bk-1" not in mm._shared_workers
    assert owner.unloaded is True


def test_dead_cached_owner_is_replaced(monkeypatch):
    monkeypatch.setattr(
        "inference_model_manager.registry_defaults.lazy_register_by_names",
        lambda names: None,
    )
    mm = _mm()
    owners = {}
    created = _patch_factory(mm, owners)

    # Seed a dead owner in the cache (worker died, not yet popped).
    dead = _FakeOwner("bk-1")
    dead._retired = False
    dead.begin_load = lambda: False  # _recv_dead → reservation refused
    mm._shared_workers["bk-1"] = dead

    mm.load_shared_head("head/1", "k", _resolution("bk-1"))

    # A fresh owner was created and serves the head; the dead one was bypassed.
    assert created == ["bk-1"]
    assert mm._shared_workers["bk-1"] is owners["bk-1"]
    assert owners["bk-1"].has_head("head/1")


def test_fresh_owner_dead_after_init_is_not_published(monkeypatch):
    mm = _mm()

    class _BornDeadOwner(_FakeOwner):
        def begin_load(self):
            return False  # worker died between __init__ and reservation

        def unload(self):
            self.unloaded = True

    owner = _BornDeadOwner("bk-1")
    mm._make_shared_owner = lambda *a, **k: owner

    with pytest.raises(RuntimeError, match="died during startup"):
        mm.load_shared_head("head/1", "k", _resolution("bk-1"))

    assert "bk-1" not in mm._shared_workers   # never published
    assert mm._shared_loading_keys == set()   # sentinel released
    assert mm._loading_ids == set()
    assert owner.unloaded is True             # dead worker torn down

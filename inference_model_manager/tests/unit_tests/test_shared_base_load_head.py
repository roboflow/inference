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
        self._dead = False
        self._on_empty = None
        self.unloaded = False
        self.loaded = []

    @property
    def retired(self):
        return self._retired

    @property
    def alive(self):
        return not self._retired and not self._dead

    def begin_load(self):
        if not self.alive:
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

    def load_head(self, head_id, api_key, model_id_or_path=None):
        self.loaded.append((head_id, model_id_or_path))
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
    mm._shared_base_preloads = {}
    mm._shared_death_hook = None
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


def test_load_shared_head_uses_stripped_model_id_for_owner(monkeypatch):
    monkeypatch.setattr(
        "inference_model_manager.registry_defaults.lazy_register_by_names",
        lambda names: None,
    )
    mm = _mm()
    owners = {}
    _patch_factory(mm, owners)

    mm.load_shared_head("head/1:3", "k", _resolution(), model_id_or_path="head/1")

    assert "head/1:3" in mm._backends
    assert owners["bk-1"].loaded == [("head/1:3", "head/1")]


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
        def load_head(self, head_id, api_key, model_id_or_path=None):
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

    assert "bk-1" not in mm._shared_workers  # never published
    assert mm._shared_loading_keys == set()  # sentinel released
    assert mm._loading_ids == set()
    assert owner.unloaded is True  # dead worker torn down


def test_has_shared_base():
    mm = _mm()
    assert mm.has_shared_base("bk-1") is False

    owner = _FakeOwner("bk-1")
    mm._shared_workers["bk-1"] = owner
    assert mm.has_shared_base("bk-1") is True

    owner._retired = True
    assert mm.has_shared_base("bk-1") is False  # retired owner is not resident

    owner._retired = False
    owner._dead = True
    assert mm.has_shared_base("bk-1") is False  # dead-but-cached owner is not resident


def test_load_shared_base_retains_owner_until_unload():
    mm = _mm()
    owners = {}
    _patch_factory(mm, owners)

    mm.load_shared_base("google/owlv2-large-patch14-ensemble", "k", _resolution())

    owner = owners["bk-1"]
    assert mm.shared_base_preloads() == {"google/owlv2-large-patch14-ensemble": "bk-1"}
    assert "bk-1" in mm._shared_workers
    assert owner.unloaded is False

    mm.unload_shared_base("google/owlv2-large-patch14-ensemble")

    assert mm.shared_base_preloads() == {}
    assert owner.unloaded is True
    assert "bk-1" not in mm._shared_workers


def test_shared_worker_death_drops_base_preload_without_release():
    mm = _mm()
    owner = _FakeOwner("bk-1")
    mm._shared_workers["bk-1"] = owner
    mm._shared_base_preloads["base"] = ("bk-1", owner)

    mm._on_shared_worker_death("bk-1")

    assert mm._shared_workers == {}
    assert mm._shared_base_preloads == {}
    assert owner.unloaded is False  # worker is already dead; do not call end_load()

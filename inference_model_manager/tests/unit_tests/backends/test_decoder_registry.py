import importlib.metadata

import pytest

from inference_model_manager.backends.decode import (
    DECODER_FACTORIES,
    make_decoder,
    register_decoder,
)
from inference_model_manager.backends.decode import (
    _reset_entry_point_decoders_for_tests,
)


def test_imagecodecs_registered_by_default():
    assert "imagecodecs" in DECODER_FACTORIES
    assert callable(make_decoder("imagecodecs", device="cpu"))


def test_register_custom_decoder():
    register_decoder("fake", lambda device: (lambda data: b"decoded"))
    try:
        assert make_decoder("fake", device="cpu")(b"x") == b"decoded"
    finally:
        DECODER_FACTORIES.pop("fake", None)


def test_unknown_decoder_lists_names():
    with pytest.raises(ValueError, match="imagecodecs"):
        make_decoder("nope", device="cpu")


def test_imagecodecs_decoder_returns_model_ready_bgr():
    import imagecodecs
    import numpy as np

    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    data = imagecodecs.jpeg_encode(rgb, level=100)
    out = make_decoder("imagecodecs", device="cpu")(data)
    assert out[..., 2].mean() > 200
    assert out[..., 0].mean() < 55


def test_unknown_name_triggers_entry_point_discovery(monkeypatch):
    def ep_factory(device):
        return lambda data: b"ep-decoded"

    class _FakeEntryPoint:
        name = "ep-decoder"

        def load(self):
            return ep_factory

    def fake_entry_points(*, group):
        assert group == "inference_model_manager.decoders"
        return [_FakeEntryPoint()]

    _reset_entry_point_decoders_for_tests()
    monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)
    try:
        assert make_decoder("ep-decoder", device="cpu")(b"x") == b"ep-decoded"
    finally:
        DECODER_FACTORIES.pop("ep-decoder", None)
        _reset_entry_point_decoders_for_tests()


def test_entry_point_discovery_runs_once(monkeypatch):
    calls = {"n": 0}

    def fake_entry_points(*, group):
        calls["n"] += 1
        return []

    _reset_entry_point_decoders_for_tests()
    monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)
    try:
        with pytest.raises(ValueError):
            make_decoder("still-nope-1", device="cpu")
        with pytest.raises(ValueError):
            make_decoder("still-nope-2", device="cpu")
        assert calls["n"] == 1
    finally:
        _reset_entry_point_decoders_for_tests()


def test_entry_point_does_not_overwrite_existing_factory(monkeypatch):
    calls = []

    def existing_factory(device):
        calls.append("existing")
        return lambda data: b"existing"

    def colliding_ep_factory(device):
        calls.append("ep-collide")
        return lambda data: b"ep-collide"

    def new_ep_factory(device):
        calls.append("ep-new")
        return lambda data: b"ep-new"

    class _CollidingEntryPoint:
        name = "collide"

        def load(self):
            return colliding_ep_factory

    class _NewEntryPoint:
        name = "brand-new"

        def load(self):
            return new_ep_factory

    def fake_entry_points(*, group):
        return [_CollidingEntryPoint(), _NewEntryPoint()]

    _reset_entry_point_decoders_for_tests()
    monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)
    register_decoder("collide", existing_factory)
    try:
        assert make_decoder("brand-new", device="cpu")(b"x") == b"ep-new"
        assert DECODER_FACTORIES["collide"] is existing_factory
        assert "ep-collide" not in calls
    finally:
        DECODER_FACTORIES.pop("collide", None)
        DECODER_FACTORIES.pop("brand-new", None)
        _reset_entry_point_decoders_for_tests()


def test_failing_entry_point_load_does_not_poison_discovery(monkeypatch):
    state = {"calls": 0}

    def working_factory(device):
        return lambda data: b"fixed"

    class _BoomEntryPoint:
        name = "boom"

        def load(self):
            raise RuntimeError("boom")

    class _WorkingEntryPoint:
        name = "fixed"

        def load(self):
            return working_factory

    def fake_entry_points(*, group):
        state["calls"] += 1
        if state["calls"] == 1:
            return [_BoomEntryPoint()]
        return [_WorkingEntryPoint()]

    _reset_entry_point_decoders_for_tests()
    monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)
    try:
        with pytest.raises(RuntimeError, match="boom"):
            make_decoder("boom", device="cpu")
        assert make_decoder("fixed", device="cpu")(b"x") == b"fixed"
        assert state["calls"] == 2
    finally:
        DECODER_FACTORIES.pop("fixed", None)
        _reset_entry_point_decoders_for_tests()

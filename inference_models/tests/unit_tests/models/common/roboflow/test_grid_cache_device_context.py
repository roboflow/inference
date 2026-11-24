"""Geometry cache capture detection follows the tensor's device."""

from contextlib import contextmanager
from types import SimpleNamespace

import torch

from inference_models.models.common.roboflow import _grid_geometry_cache as module


def test_capture_on_nonambient_target_does_not_publish(monkeypatch):
    state = {"current": 0, "queried": []}
    sentinel = object()
    module.clear()

    @contextmanager
    def device(target):
        previous = state["current"]
        state["current"] = target.index
        try:
            yield
        finally:
            state["current"] = previous

    def capturing():
        state["queried"].append(state["current"])
        return state["current"] == 1

    monkeypatch.setattr(torch.cuda, "device", device)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", capturing)
    monkeypatch.setattr(module, "_build", lambda *args: sentinel)
    result = module.geometry_constants(
        SimpleNamespace(device=torch.device("cuda:1")), [0] * 4, [1] * 4, [0] * 4
    )
    assert result is sentinel
    assert module.cached_entries() == 0
    assert state == {"current": 0, "queried": [1]}

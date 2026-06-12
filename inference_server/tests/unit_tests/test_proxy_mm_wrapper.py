from __future__ import annotations

import pickle
from unittest.mock import AsyncMock, MagicMock

import pytest

from inference_server.proxies.mm_wrapper import MMWrapper


def _fake_manager(process_return=None):
    mgr = MagicMock()
    mgr.process_async = AsyncMock(return_value=process_return)
    mgr.load = MagicMock()
    mgr.unload = MagicMock()
    mgr.stats = MagicMock(return_value={"mmp_models": {}})
    return mgr


@pytest.mark.asyncio
async def test_infer_forwards_task_image_and_params_to_manager():
    mgr = _fake_manager(process_return={"detections": []})
    wrapper = MMWrapper(mgr)
    image = b"\xff\xd8\xff"
    result = await wrapper.infer(
        model_id="acme/1",
        image=image,
        task="prompt",
        instance="",
        params={"confidence": 0.5, "prompt": "hi"},
    )
    assert result == {"detections": []}
    mgr.process_async.assert_awaited_once()
    args, kwargs = mgr.process_async.await_args
    assert args == ("acme/1",)
    assert kwargs["task"] == "prompt"
    assert kwargs["images"] == image
    assert kwargs["confidence"] == 0.5
    assert kwargs["prompt"] == "hi"


@pytest.mark.asyncio
async def test_infer_with_no_params_still_includes_images_kwarg():
    mgr = _fake_manager(process_return="ok")
    wrapper = MMWrapper(mgr)
    await wrapper.infer(model_id="m", image=b"x", task=None, params=None)
    kwargs = mgr.process_async.await_args.kwargs
    assert kwargs["images"] == b"x"
    assert kwargs["task"] is None


@pytest.mark.asyncio
async def test_infer_raw_pickle_returns_bytes():
    fake_pred = {"foo": "bar"}
    mgr = _fake_manager(process_return=fake_pred)
    wrapper = MMWrapper(mgr)
    out = await wrapper.infer(model_id="m", image=b"x", raw_pickle=True)
    assert isinstance(out, bytes)
    assert pickle.loads(out) == fake_pred


@pytest.mark.asyncio
async def test_ensure_loaded_returns_model_ready_when_loaded():
    mgr = _fake_manager()
    mgr.stats = MagicMock(return_value={"mmp_models": {"acme/1": {}}})
    wrapper = MMWrapper(mgr)
    status = await wrapper.ensure_loaded("acme/1")
    assert status[0] == "model_ready"


@pytest.mark.asyncio
async def test_stats_unwraps_manager_stats():
    mgr = _fake_manager()
    mgr.stats = MagicMock(return_value={"a": 1})
    wrapper = MMWrapper(mgr)
    out = await wrapper.stats()
    assert out == {"a": 1}


@pytest.mark.asyncio
async def test_interface_raises_runtime_error_when_model_not_loaded():
    mgr = _fake_manager()
    mgr.stats = MagicMock(return_value={"mmp_models": {}})
    wrapper = MMWrapper(mgr)
    with pytest.raises(RuntimeError, match="not loaded"):
        await wrapper.interface("ghost")


@pytest.mark.asyncio
async def test_interface_returns_tasks_for_loaded_model():
    mgr = _fake_manager()
    mgr.stats = MagicMock(
        return_value={"mmp_models": {"acme/1": {"tasks": {"infer": {}}}}}
    )
    wrapper = MMWrapper(mgr)
    info = await wrapper.interface("acme/1")
    assert info["model_id"] == "acme/1"
    assert info["tasks"] == {"infer": {}}


@pytest.mark.asyncio
async def test_infer_requests_raw_prediction_from_manager():
    """Bundled mode must hand L1 the RAW prediction (serialize=False) — the
    registry-typed dict broke L1 serializers that expect .xyxy etc."""
    mgr = _fake_manager(process_return={"raw": 1})
    wrapper = MMWrapper(mgr)
    await wrapper.infer(model_id="m", image=b"x")
    kwargs = mgr.process_async.await_args.kwargs
    assert kwargs["serialize"] is False


@pytest.mark.asyncio
async def test_concurrent_ensure_loaded_loads_once():
    import asyncio
    import threading
    import time

    class _SlowManager:
        def __init__(self):
            self.loaded = set()
            self.load_calls = 0
            self._lock = threading.Lock()

        def __contains__(self, model_id):
            return model_id in self.loaded

        def load(self, model_id, api_key, device=None):
            with self._lock:
                self.load_calls += 1
            time.sleep(0.05)
            self.loaded.add(model_id)

    mgr = _SlowManager()
    wrapper = MMWrapper(mgr)
    results = await asyncio.gather(
        wrapper.ensure_loaded("m"), wrapper.ensure_loaded("m")
    )
    assert [r[0] for r in results] == ["model_ready", "model_ready"]
    assert mgr.load_calls == 1

import importlib
import os
from unittest import mock

from inference.core import env as env_module
from inference.core.cache import model_monitoring as model_monitoring_cache_module
from inference.core.cache.memory import MemoryCache
from inference.core.managers import metrics


def test_get_system_info_returns_info() -> None:
    info = metrics.get_system_info()
    assert isinstance(info, dict)
    assert "platform" in info


@mock.patch.object(metrics.platform, "system", side_effect=RuntimeError("fail"))
def test_get_system_info_returns_info_even_on_exception(
    system_mock: mock.MagicMock,
) -> None:
    info = metrics.get_system_info()
    assert isinstance(info, dict)
    assert info == {}
    system_mock.assert_called_once_with()


def test_get_inference_results_for_model_reads_from_model_monitoring_cache(
    monkeypatch,
) -> None:
    cache = MemoryCache()
    monkeypatch.setattr(
        metrics.model_monitoring_cache_module, "model_monitoring_cache", cache
    )
    cache.zadd(
        key="inference:server-1:model/1",
        value={
            "request": {"image": "payload", "model_id": "model/1"},
            "response": {"image": "response-image", "time": 0.25},
        },
        score=123.0,
    )

    results = metrics.get_inference_results_for_model(
        inference_server_id="server-1",
        model_id="model/1",
    )

    assert results == [
        {
            "request_time": 123.0,
            "inference": {
                "request": {"model_id": "model/1"},
                "response": {"time": 0.25},
            },
        }
    ]


def test_model_monitoring_cache_uses_memory_cache_when_forced() -> None:
    original_backend = os.environ.get("MODEL_MONITORING_CACHE_BACKEND")
    try:
        os.environ["MODEL_MONITORING_CACHE_BACKEND"] = "memory"
        importlib.reload(env_module)
        importlib.reload(model_monitoring_cache_module)

        assert isinstance(
            model_monitoring_cache_module.model_monitoring_cache, MemoryCache
        )
    finally:
        if original_backend is None:
            os.environ.pop("MODEL_MONITORING_CACHE_BACKEND", None)
        else:
            os.environ["MODEL_MONITORING_CACHE_BACKEND"] = original_backend
        importlib.reload(env_module)
        importlib.reload(model_monitoring_cache_module)

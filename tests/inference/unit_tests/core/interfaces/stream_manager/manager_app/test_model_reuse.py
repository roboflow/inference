from queue import Queue
from threading import Lock
from unittest.mock import MagicMock

import pytest

from inference.core.interfaces.stream_manager.manager_app import app
from inference.core.interfaces.stream_manager.manager_app import (
    inference_pipeline_manager as worker_module,
)
from inference.core.interfaces.stream_manager.manager_app.entities import (
    OperationStatus,
)


def worker(key="clip", cache_key="tenant-a", idle=False, idle_since=None):
    process = MagicMock()
    process.is_alive.return_value = True
    return app.ManagedInferencePipeline(
        pipeline_id=key,
        pipeline_manager=process,
        command_queue=Queue(),
        responses_queue=Queue(),
        operation_lock=Lock(),
        is_idle=idle,
        model_cache_key=cache_key,
        idle_since=idle_since,
    )


@pytest.fixture(autouse=True)
def cache_settings(monkeypatch):
    monkeypatch.setattr(app, "STREAM_MANAGER_MODEL_CACHE_SIZE", 1)
    monkeypatch.setattr(app, "STREAM_MANAGER_MODEL_CACHE_TTL", 300)
    monkeypatch.setattr(app.time, "monotonic", lambda: 1000)
    monkeypatch.setattr(app, "_get_current_process_ram_usage_mb", lambda: 0)


def test_retiring_a_clip_keeps_worker_but_invalidates_its_public_id():
    cached = worker()
    table = {"clip": cached}
    app.cache_idle_worker(table, "clip")
    assert "clip" not in table
    assert cached.is_idle
    assert cached.idle_since == 1000
    cached.pipeline_manager.join.assert_not_called()
    assert app.handle_command(table, "request", "clip", {})["error_type"] == "not_found"
    assert (
        app.handle_command(table, "request", cached.pipeline_id, {})["error_type"]
        == "not_found"
    )
    second = app.get_or_spawn_pipeline_process(table, model_cache_key="tenant-a")
    assert second is cached
    assert second.pipeline_id != "clip"
    assert not second.is_idle
    assert second.idle_since is None


def test_matching_loaded_worker_is_preferred_over_empty_preload():
    preload = worker("preload", cache_key=None, idle=True)
    cached = worker("cached", idle=True, idle_since=900)
    table = {"preload": preload, "cached": cached}
    assert app.get_or_spawn_pipeline_process(table, "tenant-a") is cached
    assert preload.is_idle


@pytest.mark.parametrize("new_key", ["tenant-b", None])
def test_different_credential_and_webrtc_never_get_cached_models(monkeypatch, new_key):
    cached = worker(idle=True, idle_since=900)
    table = {"clip": cached}

    def spawn(processes_table, mark_as_idle):
        processes_table["new"] = worker("new", cache_key=None)
        return "new"

    monkeypatch.setattr(app, "spawn_managed_pipeline_process", spawn)
    fresh = app.get_or_spawn_pipeline_process(table, new_key)
    assert fresh is not cached
    assert fresh.model_cache_key == new_key
    cached.pipeline_manager.terminate.assert_called_once()
    assert "clip" not in table


@pytest.mark.parametrize("dead,expired", [(True, False), (False, True)])
def test_checkout_discards_dead_or_expired_workers(monkeypatch, dead, expired):
    cached = worker(idle=True, idle_since=600 if expired else 900)
    cached.pipeline_manager.is_alive.return_value = not dead
    table = {"clip": cached}

    def spawn(processes_table, mark_as_idle):
        processes_table["new"] = worker("new")
        return "new"

    monkeypatch.setattr(app, "spawn_managed_pipeline_process", spawn)
    assert app.get_or_spawn_pipeline_process(table, "tenant-a") is not cached
    cached.pipeline_manager.terminate.assert_called_once()


def test_idle_pool_is_bounded_and_evicts_oldest():
    old = worker("old", idle=True, idle_since=900)
    fresh = worker("fresh")
    active = worker("active", cache_key="tenant-b")
    table = {"old": old, "fresh": fresh, "active": active}
    app.cache_idle_worker(table, "fresh")
    assert "old" not in table
    assert table["active"] is active
    assert sum(w.is_idle for w in table.values()) == 1
    old.pipeline_manager.terminate.assert_called_once()
    active.pipeline_manager.terminate.assert_not_called()


def test_expired_idle_worker_is_reaped_without_another_clip(monkeypatch):
    cached = worker(idle=True, idle_since=600)
    table = {"clip": cached}
    monkeypatch.setattr(app, "PROCESSES_TABLE", table)
    monkeypatch.setattr(app, "_get_process_memory_usage_mb", lambda process: 0)

    class EndSweep(Exception):
        pass

    monkeypatch.setattr(app.time, "sleep", lambda _: (_ for _ in ()).throw(EndSweep()))
    with pytest.raises(EndSweep):
        app.check_process_health()
    assert table == {}
    cached.pipeline_manager.terminate.assert_called_once()


def test_termination_returns_worker_only_after_pipeline_cleanup(monkeypatch):
    cached = worker()
    table = {"clip": cached}
    cached.responses_queue.put(
        ("request", {"status": OperationStatus.SUCCESS, "model_cache_retained": True})
    )
    handler = object.__new__(app.InferencePipelinesManagerHandler)
    handler._processes_table = table
    handler.request = MagicMock()
    monkeypatch.setattr(app, "send_data_trough_socket", MagicMock())
    handler._terminate_pipeline("request", "clip", {"type": "terminate"})
    assert cached.command_queue.get()[1]["_keep_model_cache"] is True
    assert cached.is_idle
    cached.pipeline_manager.join.assert_not_called()


def test_reuse_disabled_still_exits_and_joins_worker(monkeypatch):
    monkeypatch.setattr(app, "STREAM_MANAGER_MODEL_CACHE_SIZE", 0)
    cached = worker()
    cached.responses_queue.put(("request", {"status": OperationStatus.SUCCESS}))
    table = {"clip": cached}
    handler = object.__new__(app.InferencePipelinesManagerHandler)
    handler._processes_table = table
    handler.request = MagicMock()
    monkeypatch.setattr(app, "send_data_trough_socket", MagicMock())
    handler._terminate_pipeline(
        "request", "clip", {"type": "terminate", "_keep_model_cache": True}
    )
    assert cached.command_queue.get()[1]["_keep_model_cache"] is False
    assert table == {}
    cached.pipeline_manager.join.assert_called_once()


def test_worker_injects_same_manager_into_fresh_workflows(monkeypatch):
    process = worker_module.InferencePipelineManager("clip", Queue(), Queue())
    created = []

    def init(**kwargs):
        created.append(kwargs)
        return MagicMock()

    monkeypatch.setattr(worker_module.InferencePipeline, "init_with_workflow", init)
    payload = {
        "_reuse_model_manager": True,
        "video_configuration": {
            "type": "VideoConfiguration",
            "video_reference": "/clip.mp4",
        },
        "processing_configuration": {
            "type": "WorkflowConfiguration",
            "workflow_specification": {},
        },
    }
    process._initialise_pipeline("first", payload)
    manager = process._model_manager
    first_pipeline = process._inference_pipeline
    process._execute_termination(keep_model_cache=True)
    first_pipeline.terminate.assert_called_once()
    first_pipeline.join.assert_called_once()
    assert process._inference_pipeline is None
    assert process._watchdog is None
    assert process._buffer_sink is None
    assert process._consumption_timeout is None
    assert not process._stop
    process._initialise_pipeline("second", payload)
    assert created[0]["model_manager"] is manager is created[1]["model_manager"]
    assert created[0]["watchdog"] is not created[1]["watchdog"]
    assert (
        created[0]["on_prediction"].__self__ is not created[1]["on_prediction"].__self__
    )
    assert process._inference_pipeline is not first_pipeline
    process._execute_termination()
    assert process._stop


def test_failed_initialization_discards_cached_worker(monkeypatch):
    cached = worker(idle=False)
    cached.responses_queue.put(("request", {"status": OperationStatus.FAILURE}))
    table = {"clip": cached}
    monkeypatch.setattr(app, "get_or_spawn_pipeline_process", lambda **kwargs: cached)
    handler = object.__new__(app.InferencePipelinesManagerHandler)
    handler._processes_table = table
    handler.request = MagicMock()
    monkeypatch.setattr(app, "send_data_trough_socket", MagicMock())
    handler._initialise_pipeline("request", {"retain_results_on_eof": True})
    assert table == {}
    cached.pipeline_manager.terminate.assert_called_once()


def test_worker_with_inference_errors_is_not_recycled():
    process = worker_module.InferencePipelineManager("clip", Queue(), Queue())
    process._inference_pipeline = MagicMock()
    process._watchdog = MagicMock()
    update = MagicMock(severity=worker_module.UpdateSeverity.ERROR)
    process._watchdog.get_report.return_value.video_source_status_updates = [update]
    process._terminate_pipeline("request", keep_model_cache=True)
    assert process._stop
    response = process._responses_queue.get(timeout=1)[1]
    assert response["status"] == OperationStatus.SUCCESS
    assert response["model_cache_retained"] is False

from queue import Queue
from threading import Lock
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from inference.core.interfaces.stream_manager.manager_app import app
from inference.core.interfaces.stream_manager.manager_app.entities import (
    InitialisePipelinePayload,
    OperationStatus,
)


def payload(**kwargs):
    return InitialisePipelinePayload(
        video_configuration={
            "type": "VideoConfiguration",
            "video_reference": "/clip.mp4",
        },
        processing_configuration={
            "type": "WorkflowConfiguration",
            "workflow_specification": {},
        },
        **kwargs,
    )


@pytest.mark.parametrize("timeout", [None, 0, -1, float("nan"), float("inf")])
def test_retention_requires_a_bounded_idle_timeout(timeout):
    with pytest.raises(ValidationError, match="finite, positive"):
        payload(retain_results_on_eof=True, consumption_timeout=timeout)


def test_retention_is_opt_in_and_accepts_a_bounded_idle_timeout():
    assert not payload().retain_results_on_eof
    assert payload(
        retain_results_on_eof=True, consumption_timeout=60
    ).retain_results_on_eof


def test_dead_worker_cannot_block_the_manager_waiting_for_a_response():
    process = MagicMock()
    process.is_alive.return_value = False
    response = app.get_response_ignoring_thrash(Queue(), "request", process)
    assert response["status"] == OperationStatus.FAILURE
    assert response["error_type"] == "not_found"


def test_queued_response_is_preserved_even_if_worker_has_exited():
    responses = Queue()
    responses.put(("request", {"status": OperationStatus.SUCCESS}))
    process = MagicMock()
    process.is_alive.return_value = False
    assert (
        app.get_response_ignoring_thrash(responses, "request", process)["status"]
        == "success"
    )


class StopSweep(Exception):
    pass


def test_health_sweep_preserves_retained_results_but_reaps_exited_worker(monkeypatch):
    process = MagicMock()
    process.is_alive.return_value = True
    pipeline = app.ManagedInferencePipeline(
        pipeline_id="clip",
        pipeline_manager=process,
        command_queue=Queue(),
        responses_queue=Queue(),
        operation_lock=Lock(),
        is_idle=False,
        retain_results_on_eof=True,
    )
    table = {"clip": pipeline}
    command = MagicMock()
    monkeypatch.setattr(app, "PROCESSES_TABLE", table)
    monkeypatch.setattr(app, "_get_current_process_ram_usage_mb", lambda: 0)
    monkeypatch.setattr(app, "_get_process_memory_usage_mb", lambda process: 0)
    monkeypatch.setattr(app, "handle_command", command)
    monkeypatch.setattr(app.time, "sleep", lambda _: (_ for _ in ()).throw(StopSweep()))
    with pytest.raises(StopSweep):
        app.check_process_health()
    assert "clip" in table
    command.assert_not_called()
    process.terminate.assert_not_called()
    process.is_alive.return_value = False
    with pytest.raises(StopSweep):
        app.check_process_health()
    assert table == {}
    process.join.assert_called_once()


def test_failed_initialization_reaps_the_unusable_worker(monkeypatch):
    process = MagicMock()
    process.is_alive.return_value = False
    pipeline = app.ManagedInferencePipeline(
        pipeline_id="clip",
        pipeline_manager=process,
        command_queue=Queue(),
        responses_queue=Queue(),
        operation_lock=Lock(),
        is_idle=False,
    )
    pipeline.responses_queue.put(("request", {"status": OperationStatus.FAILURE}))
    table = {"clip": pipeline}
    handler = object.__new__(app.InferencePipelinesManagerHandler)
    handler._processes_table = table
    handler.request = MagicMock()
    monkeypatch.setattr(
        app, "get_or_spawn_pipeline_process", lambda processes_table: pipeline
    )
    monkeypatch.setattr(app, "send_data_trough_socket", MagicMock())
    handler._initialise_pipeline("request", {"retain_results_on_eof": True})
    assert pipeline.retain_results_on_eof
    assert table == {}
    process.terminate.assert_called_once()
    process.join.assert_called_once_with(timeout=5)

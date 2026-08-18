import copy
import multiprocessing as mp
import os
import signal

import pytest
from job_process import (
    JOB_EXECUTION_PROCESS,
    MAX_MESSAGE_BYTES,
    PROTOCOL_VERSION,
    RemoteStats,
    bounded_child_event,
    bounded_parent_command,
    drop_supervisor_credentials,
    remove_supervisor_credentials,
    resolve_job_execution_mode,
    restore_supervisor_credentials,
    send_parent_command,
    wait_for_process_exit,
)


def _lifecycle_probe(connection, crash=False):
    connection.send(
        bounded_child_event(
            {
                "version": PROTOCOL_VERSION,
                "type": "started",
                "state": "running",
                "runtime": {"processId": os.getpid()},
            }
        )
    )
    if crash:
        os._exit(91)
    command = bounded_parent_command(connection.recv())
    assert command["type"] == "stop"
    connection.send(
        bounded_child_event(
            {
                "version": PROTOCOL_VERSION,
                "type": "stopped",
                "state": "stopped",
                "runtime": {"processId": os.getpid()},
            }
        )
    )
    connection.close()


def _spawn_probe(context, crash=False):
    parent, child = context.Pipe(duplex=True)
    process = context.Process(target=_lifecycle_probe, args=(child, crash))
    process.start()
    child.close()
    started = bounded_child_event(parent.recv())
    return process, parent, started


def test_execution_mode_defaults_to_thread_and_accepts_process(monkeypatch):
    monkeypatch.delenv("PROCESSOR_JOB_EXECUTION_MODE", raising=False)
    assert resolve_job_execution_mode() == "thread"
    assert resolve_job_execution_mode("process") == JOB_EXECUTION_PROCESS
    with pytest.raises(ValueError, match="must be one of"):
        resolve_job_execution_mode("fork")


def test_child_protocol_does_not_accept_payload_or_credentials():
    event = {
        "version": PROTOCOL_VERSION,
        "type": "status",
        "state": "running",
        "stats": {"frames": 7},
        "runtime": {"processId": 42},
        "imageOutputs": ["visualization"],
        "defaultImageOutput": "visualization",
        "error": None,
    }
    assert bounded_child_event(event) == event
    for forbidden in ("job", "apiKey", "sourceUrl", "workflowSpecification", "frame"):
        with pytest.raises(ValueError, match="field"):
            bounded_child_event({**event, forbidden: "secret"})
    with pytest.raises(ValueError, match="field"):
        bounded_child_event({**event, "stats": {"apiKey": "secret"}})
    with pytest.raises(ValueError, match="field"):
        bounded_child_event({**event, "runtime": {"sourceUrl": "secret"}})
    with pytest.raises(ValueError, match="forbidden field"):
        bounded_child_event(
            {**event, "stats": {"counters": {"processorAccessToken": 1}}}
        )


def test_child_protocol_accepts_bounded_stream_runtime_configuration():
    event = {
        "version": PROTOCOL_VERSION,
        "type": "started",
        "state": "running",
        "stats": {"frames": 0},
        "runtime": {
            "sourceFpsLimiterAtProducer": True,
            "streamDecodingBufferSize": 2,
            "streamBufferConsumption": "lazy",
        },
    }

    assert bounded_child_event(event) == event


def test_child_protocol_rejects_unbounded_event():
    with pytest.raises(ValueError, match="error is invalid"):
        bounded_child_event(
            {
                "version": PROTOCOL_VERSION,
                "type": "failure",
                "error": "x" * MAX_MESSAGE_BYTES,
            }
        )


def test_child_protocol_accepts_only_bounded_image_redacted_results():
    event = {
        "version": PROTOCOL_VERSION,
        "type": "result",
        "result": {
            "frameId": 41,
            "timestamp": "2026-08-14T12:00:00Z",
            "latencyMs": 12.5,
            "outputs": {
                "predictions": [{"class": "car", "confidence": 0.9}],
                "visualization": {
                    "type": "image_ref",
                    "output": "visualization",
                },
            },
        },
    }
    assert bounded_child_event(event) == event
    with pytest.raises(ValueError, match="forbidden field"):
        bounded_child_event(
            {
                **event,
                "result": {**event["result"], "apiKey": "secret"},
            }
        )
    with pytest.raises(ValueError, match="size limit"):
        bounded_child_event(
            {
                **event,
                "result": {
                    **event["result"],
                    "outputs": {"predictions": "x" * MAX_MESSAGE_BYTES},
                },
            }
        )
    with pytest.raises(ValueError, match="only JSON values"):
        bounded_child_event(
            {
                **event,
                "result": {
                    **event["result"],
                    "outputs": {"encoded": b"not-json"},
                },
            }
        )


def test_parent_protocol_is_control_only_and_bounded():
    command = {
        "version": PROTOCOL_VERSION,
        "type": "watch",
        "watch": {"requested": True, "output": "visualization"},
    }
    assert bounded_parent_command(command) == command
    with pytest.raises(ValueError, match="unexpected fields"):
        bounded_parent_command({**command, "apiKey": "secret"})
    with pytest.raises(ValueError, match="unexpected fields"):
        bounded_parent_command(
            {
                "version": PROTOCOL_VERSION,
                "type": "watch",
                "watch": {"requested": True, "frame": b"pixels"},
            }
        )


def test_remote_stats_are_copied_and_runtime_is_parent_selected():
    stats = RemoteStats()
    child = {"frames": 3, "counters": {"inferred": 3}}
    stats.replace(child)
    child["counters"]["inferred"] = 999
    first = stats.snapshot(runtime={"processId": 55})
    assert first == {
        "frames": 3,
        "counters": {"inferred": 3},
        "runtime": {"processId": 55},
    }
    mutated = copy.deepcopy(first)
    mutated["frames"] = 0
    assert stats.snapshot()["frames"] == 3


def test_child_drops_supervisor_only_credentials():
    env = {
        "VIDEO_PROC_SERVICE_SECRET": "fleet",
        "PROCESSOR_PUBSUB_SUBSCRIPTION": "subscription",
        "PROCESSOR_PUBLIC_URL": "https://processor.example",
        "GATEWAY_PUBLIC_BASE": "https://gateway.example",
        "ROBOFLOW_API_KEY": "job-key",
        "CUDA_VISIBLE_DEVICES": "0",
    }
    drop_supervisor_credentials(env)
    assert "VIDEO_PROC_SERVICE_SECRET" not in env
    assert "PROCESSOR_PUBSUB_SUBSCRIPTION" not in env
    assert "PROCESSOR_PUBLIC_URL" not in env
    assert "GATEWAY_PUBLIC_BASE" not in env
    assert "ROBOFLOW_API_KEY" not in env
    assert env["CUDA_VISIBLE_DEVICES"] == "0"


def test_supervisor_credentials_can_be_restored_after_sanitized_spawn_window():
    env = {
        "VIDEO_PROC_SERVICE_SECRET": "fleet",
        "ROBOFLOW_API_KEY": "fallback",
        "CUDA_VISIBLE_DEVICES": "0",
    }
    removed = remove_supervisor_credentials(env)
    assert env == {"CUDA_VISIBLE_DEVICES": "0"}
    restore_supervisor_credentials(removed, env)
    assert env == {
        "VIDEO_PROC_SERVICE_SECRET": "fleet",
        "ROBOFLOW_API_KEY": "fallback",
        "CUDA_VISIBLE_DEVICES": "0",
    }


def test_spawned_jobs_have_distinct_pids_and_cancel_gracefully():
    context = mp.get_context("spawn")
    first, first_pipe, first_started = _spawn_probe(context)
    second, second_pipe, second_started = _spawn_probe(context)
    try:
        child_pids = {
            first_started["runtime"]["processId"],
            second_started["runtime"]["processId"],
        }
        assert os.getpid() not in child_pids
        assert len(child_pids) == 2
        stop = {"version": PROTOCOL_VERSION, "type": "stop"}
        first_pipe.send(bounded_parent_command(stop))
        first_stopped = bounded_child_event(first_pipe.recv())
        assert first_stopped["type"] == "stopped"
        first.join(timeout=5)
        assert first.exitcode == 0
        assert second.is_alive()
        second_pipe.send(bounded_parent_command(stop))
        assert bounded_child_event(second_pipe.recv())["type"] == "stopped"
        second.join(timeout=5)
        assert second.exitcode == 0
    finally:
        for process in (first, second):
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
        first_pipe.close()
        second_pipe.close()


def test_hard_child_crash_is_observable_without_killing_sibling():
    context = mp.get_context("spawn")
    crashed, crashed_pipe, _ = _spawn_probe(context, crash=True)
    sibling, sibling_pipe, _ = _spawn_probe(context)
    try:
        crashed.join(timeout=5)
        assert crashed.exitcode == 91
        assert sibling.is_alive()
        sibling_pipe.send(
            bounded_parent_command({"version": PROTOCOL_VERSION, "type": "stop"})
        )
        assert bounded_child_event(sibling_pipe.recv())["state"] == "stopped"
        sibling.join(timeout=5)
        assert sibling.exitcode == 0
    finally:
        if sibling.is_alive():
            sibling.terminate()
            sibling.join(timeout=2)
        crashed_pipe.close()
        sibling_pipe.close()


def test_eof_before_reap_preserves_exit_code_and_broken_pipe_is_expected():
    context = mp.get_context("spawn")
    crashed, crashed_pipe, _ = _spawn_probe(context)
    sibling, sibling_pipe, _ = _spawn_probe(context)
    stop = {"version": PROTOCOL_VERSION, "type": "stop"}
    try:
        os.kill(crashed.pid, signal.SIGKILL)
        # This is the live failure ordering: IPC EOF is observed before the
        # supervisor has joined the child and finalized Process.exitcode.
        with pytest.raises(EOFError):
            crashed_pipe.recv()
        assert wait_for_process_exit(crashed, timeout_s=5) == -signal.SIGKILL

        # Cleanup of an already-dead child must not escalate to a whole-worker
        # containment restart, and the sibling must remain independently live.
        assert send_parent_command(crashed_pipe, stop) is False
        assert sibling.is_alive()
        assert send_parent_command(sibling_pipe, stop) is True
        assert bounded_child_event(sibling_pipe.recv())["state"] == "stopped"
        assert wait_for_process_exit(sibling, timeout_s=5) == 0
    finally:
        for process in (crashed, sibling):
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
        crashed_pipe.close()
        sibling_pipe.close()

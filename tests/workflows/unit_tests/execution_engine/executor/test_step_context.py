from concurrent.futures import ThreadPoolExecutor
from unittest import mock

from inference.core.workflows.execution_engine.v1.executor import core
from inference.usage_tracking.stream_session import stream_session_id


@mock.patch.object(core, "run_step")
def test_safe_execute_step_rebinds_stream_session_id_in_worker_thread(run_step_mock):
    # given - blocks (e.g. vision events, dataset upload) read the pipeline's
    # stream session id from the contextvar, which does not propagate into
    # ThreadPoolExecutor workers on its own
    seen = {}

    def capture(**kwargs):
        seen["stream_session_id"] = stream_session_id.get()

    run_step_mock.side_effect = capture

    # when
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(
            core.safe_execute_step,
            step_selector="$steps.some_step",
            workflow=mock.MagicMock(),
            execution_data_manager=mock.MagicMock(),
            pipeline_stream_session_id="stream-a",
        ).result()

    # then
    assert seen["stream_session_id"] == "stream-a"


@mock.patch.object(core, "run_step")
def test_safe_execute_step_clears_stale_stream_session_id(run_step_mock):
    # given - pool threads are reused; a previous pipeline's id must not leak
    seen = {}

    def capture(**kwargs):
        seen["stream_session_id"] = stream_session_id.get()

    run_step_mock.side_effect = capture

    def stale_then_execute():
        stream_session_id.set("stale-stream")
        core.safe_execute_step(
            step_selector="$steps.some_step",
            workflow=mock.MagicMock(),
            execution_data_manager=mock.MagicMock(),
        )

    # when
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(stale_then_execute).result()

    # then
    assert seen["stream_session_id"] is None

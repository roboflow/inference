"""The recorder records only for a host that declares it closes the engine.

These tests go through a real `ExecutionEngine`, because the declaration is an
engine init parameter: the block's own tests cannot show that a workflow input
is unable to set it.
"""

import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Set

import numpy as np
from roboflow_workflows.execution_engine.core import ExecutionEngine

RECORDER_WORKFLOW = {
    "version": "1.0",
    "inputs": [{"type": "WorkflowImage", "name": "image"}],
    "steps": [
        {
            "type": "roboflow_core/video_clip_recorder@v1",
            "name": "recorder",
            "image": "$inputs.image",
        }
    ],
    "outputs": [],
}
ENCODER_THREAD_PREFIX = "video-clip-x264-encoder-"


def _frame_with_caller_supplied_video_metadata() -> Dict[str, Any]:
    # What an HTTP client can send as an image input: the video metadata is
    # part of the request, so the recorder cannot trust it to identify a host.
    return {
        "type": "numpy_object",
        "value": np.zeros((48, 64, 3), dtype=np.uint8),
        "video_metadata": {
            "video_identifier": "camera-1",
            "frame_number": 1,
            "frame_timestamp": datetime.now().isoformat(),
            "fps": 30,
            "comes_from_video_file": True,
        },
    }


def _thread_ids() -> Set[int]:
    return {thread.ident for thread in threading.enumerate()}


def _encoder_threads_started_since(known_thread_ids: Set[int]) -> List[str]:
    # Only threads this test started: threads other tests left winding down in the
    # same process may exit at any moment and must not change the result.
    return [
        thread.name
        for thread in threading.enumerate()
        if thread.ident not in known_thread_ids
        and thread.name.startswith(ENCODER_THREAD_PREFIX)
    ]


def _wait_until_no_encoder_thread_since(
    known_thread_ids: Set[int], timeout: float = 10.0
) -> List[str]:
    deadline = time.monotonic() + timeout
    while _encoder_threads_started_since(known_thread_ids):
        if time.monotonic() > deadline:
            break
        time.sleep(0.01)

    return _encoder_threads_started_since(known_thread_ids)


def _recorder_block(execution_engine: ExecutionEngine) -> Any:
    return execution_engine._engine._compiled_workflow.steps["recorder"].step


def test_request_host_never_starts_a_recorder_thread() -> None:
    # given: a host that builds an engine per request and passes no declaration
    known_thread_ids = _thread_ids()
    execution_engine = ExecutionEngine.init(
        workflow_definition=RECORDER_WORKFLOW,
        init_parameters={"workflows_core.api_key": None},
    )

    # when: the engine is deliberately not closed, as on a host that drops it
    execution_engine.run(
        runtime_parameters={"image": [_frame_with_caller_supplied_video_metadata()]}
    )

    # then: nothing is left running that only `close()` could stop
    assert _recorder_block(execution_engine)._streams == {}
    assert _encoder_threads_started_since(known_thread_ids) == []


def test_host_that_manages_block_lifecycle_records_until_close() -> None:
    # given: what `InferencePipeline.init_with_workflow` sets
    known_thread_ids = _thread_ids()
    execution_engine = ExecutionEngine.init(
        workflow_definition=RECORDER_WORKFLOW,
        init_parameters={
            "workflows_core.api_key": None,
            "workflows_core.host_manages_block_lifecycle": True,
        },
    )

    try:
        # when
        execution_engine.run(
            runtime_parameters={"image": [_frame_with_caller_supplied_video_metadata()]}
        )

        # then
        assert list(_recorder_block(execution_engine)._streams) == ["camera-1"]
        assert _encoder_threads_started_since(known_thread_ids) == [
            "video-clip-x264-encoder-camera-1"
        ]
    finally:
        execution_engine.close()

    # then: the close the host promised stops the encoder thread
    assert _wait_until_no_encoder_thread_since(known_thread_ids) == []

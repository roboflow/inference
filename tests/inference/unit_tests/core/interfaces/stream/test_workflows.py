from datetime import datetime
from types import SimpleNamespace

import numpy as np

from inference.core.interfaces.camera.entities import VideoFrame
from inference.core.interfaces.stream.model_handlers.workflows import WorkflowRunner
from inference.core.interfaces.stream.model_handlers.workflows_context import (
    is_workflow_stream_flush_active,
)


class _FakeExecutionEngine:
    def __init__(self, stream_buffer_depth: int) -> None:
        self._stream_buffer_depth = stream_buffer_depth
        step = SimpleNamespace(
            is_stream_pipelined=lambda: stream_buffer_depth > 0,
            stream_pipeline_depth=lambda: stream_buffer_depth,
        )
        self._engine = SimpleNamespace(
            _compiled_workflow=SimpleNamespace(
                steps={"segmentation": SimpleNamespace(step=step)}
            )
        )
        self.calls = []

    def run(
        self,
        runtime_parameters,
        fps,
        serialize_results,
        _is_preview,
    ):
        frame_number = runtime_parameters["image"][0]["video_metadata"].frame_number
        flush_active = is_workflow_stream_flush_active()
        self.calls.append(
            {
                "frame_number": frame_number,
                "flush_active": flush_active,
                "fps": fps,
                "serialize_results": serialize_results,
                "is_preview": _is_preview,
            }
        )
        if flush_active:
            prediction_frame = frame_number
        else:
            prediction_frame = frame_number - self._stream_buffer_depth
        return [{"predictions": f"frame-{prediction_frame}"}]


def _make_frame(frame_id: int) -> VideoFrame:
    return VideoFrame(
        image=np.zeros((8, 8, 3), dtype=np.uint8),
        frame_id=frame_id,
        frame_timestamp=datetime.fromtimestamp(frame_id),
        fps=30.0,
        measured_fps=None,
        source_id=0,
        comes_from_video_file=True,
    )


def test_workflow_runner_without_stream_buffering_returns_current_frame() -> None:
    engine = _FakeExecutionEngine(stream_buffer_depth=0)
    runner = WorkflowRunner(
        workflows_parameters=None,
        execution_engine=engine,
        image_input_name="image",
        video_metadata_input_name="video_metadata",
        serialize_results=True,
        _is_preview=True,
    )
    frame = _make_frame(1)

    result = runner([frame])

    assert result is not None
    assert result.predictions == [{"predictions": "frame-1"}]
    assert result.video_frames == [frame]
    assert engine.calls == [
        {
            "frame_number": 1,
            "flush_active": False,
            "fps": 30.0,
            "serialize_results": True,
            "is_preview": True,
        }
    ]


def test_workflow_runner_buffers_frames_until_delayed_prediction_arrives() -> None:
    engine = _FakeExecutionEngine(stream_buffer_depth=1)
    runner = WorkflowRunner(
        workflows_parameters=None,
        execution_engine=engine,
        image_input_name="image",
        video_metadata_input_name="video_metadata",
    )
    frame_1 = _make_frame(1)
    frame_2 = _make_frame(2)

    first_result = runner([frame_1])
    second_result = runner([frame_2])
    flushed_results = runner.flush()

    assert first_result is None
    assert second_result is not None
    assert second_result.predictions == [{"predictions": "frame-1"}]
    assert second_result.video_frames == [frame_1]
    assert flushed_results is not None
    assert len(flushed_results) == 1
    assert flushed_results[0].predictions == [{"predictions": "frame-2"}]
    assert flushed_results[0].video_frames == [frame_2]
    assert engine.calls == [
        {
            "frame_number": 1,
            "flush_active": False,
            "fps": 30.0,
            "serialize_results": False,
            "is_preview": False,
        },
        {
            "frame_number": 2,
            "flush_active": False,
            "fps": 30.0,
            "serialize_results": False,
            "is_preview": False,
        },
        {
            "frame_number": 2,
            "flush_active": True,
            "fps": 30.0,
            "serialize_results": False,
            "is_preview": False,
        },
    ]

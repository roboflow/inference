import datetime

import numpy as np

import inference.core.workflows.core_steps.fusion.time_travel.v1 as time_travel_v1
from inference.core.workflows.core_steps.fusion.time_travel.v1 import (
    BUFFER_MARGIN,
    TimeTravelBlockV1,
)
from inference.core.workflows.execution_engine.entities.base import (
    VideoMetadata,
    WorkflowImageData,
)


def _image(frame_number: int, video_id: str = "vid_1") -> WorkflowImageData:
    metadata = VideoMetadata(
        video_identifier=video_id,
        frame_number=frame_number,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570800).astimezone(
            tz=datetime.timezone.utc
        ),
        fps=30,
        comes_from_video_file=True,
    )
    return WorkflowImageData(
        parent_metadata=None,
        numpy_image=np.zeros((2, 2, 3), dtype=np.uint8),
        video_metadata=metadata,
    )


def test_past_offset_returns_previous_value() -> None:
    # given
    block = TimeTravelBlockV1()

    # when - feed a monotonic frame sequence, offset -2 (two frames ago)
    results = [block.run(image=_image(n), data=f"det-{n}", offset=-2) for n in range(5)]

    # then
    assert (
        results[0]["output"] is None
    ), "frame 0 has no frame -2 buffered yet -> default"
    assert results[0]["is_available"] is False
    assert results[2]["output"] == "det-0", "frame 2 references frame 0"
    assert results[2]["is_available"] is True
    assert results[2]["reference_frame_number"] == 2
    assert results[4]["output"] == "det-2", "frame 4 references frame 2"
    assert results[4]["is_available"] is True


def test_past_offset_missing_default_value() -> None:
    # given
    block = TimeTravelBlockV1()

    # when
    result = block.run(image=_image(0), data="det-0", offset=-1, default_value="NA")

    # then
    assert result["output"] == "NA"
    assert result["is_available"] is False


def test_future_offset_returns_current_value(monkeypatch) -> None:
    # given - value behavior is independent of the (opt-in) stream-pipeline flag.
    monkeypatch.setattr(time_travel_v1, "STREAM_LOOKAHEAD_ENABLED", False)
    block = TimeTravelBlockV1()

    # when
    result = block.run(image=_image(7), data="det-7", offset=10)

    # then - from the emitted frame (7 - 10) perspective, "+10 ahead" is the current
    # frame, whose value is already known.
    assert result["output"] == "det-7"
    assert result["is_available"] is True
    assert result["reference_frame_number"] == 7 - 10


def test_future_lookahead_disabled_by_default() -> None:
    # given - default (flag off): the runner must not be turned into a pipelined
    # runner, so synchronous consumers (WebRTC/webexec, HTTP) keep working.
    block = TimeTravelBlockV1()

    # when
    block.run(image=_image(7), data="det-7", offset=10)

    # then
    assert time_travel_v1.STREAM_LOOKAHEAD_ENABLED is False
    assert block.can_activate_stream_pipeline() is False
    assert block.is_stream_pipelined() is False
    assert block.stream_pipeline_depth() == 0


def test_future_lookahead_declares_depth_when_enabled(monkeypatch) -> None:
    # given - opt-in flag on (InferencePipeline video path only).
    monkeypatch.setattr(time_travel_v1, "STREAM_LOOKAHEAD_ENABLED", True)
    block = TimeTravelBlockV1()

    # when
    block.run(image=_image(7), data="det-7", offset=10)

    # then
    assert block.can_activate_stream_pipeline() is True
    assert block.is_stream_pipelined() is True
    assert block.stream_pipeline_depth() == 10


def test_zero_offset_returns_current_value_without_delay() -> None:
    # given
    block = TimeTravelBlockV1()

    # when
    result = block.run(image=_image(3), data="det-3", offset=0)

    # then
    assert result["output"] == "det-3"
    assert result["is_available"] is True
    assert result["reference_frame_number"] == 3
    assert block.is_stream_pipelined() is False
    assert block.stream_pipeline_depth() == 0


def test_state_is_isolated_per_video_identifier() -> None:
    # given
    block = TimeTravelBlockV1()

    # when - interleave two streams with the same frame numbers
    block.run(image=_image(0, video_id="a"), data="a-0", offset=-1)
    block.run(image=_image(0, video_id="b"), data="b-0", offset=-1)
    result_a = block.run(image=_image(1, video_id="a"), data="a-1", offset=-1)
    result_b = block.run(image=_image(1, video_id="b"), data="b-1", offset=-1)

    # then
    assert result_a["output"] == "a-0"
    assert result_b["output"] == "b-0"


def test_buffer_is_bounded() -> None:
    # given
    block = TimeTravelBlockV1()
    offset = -3

    # when - process many frames
    for n in range(500):
        block.run(image=_image(n), data=f"det-{n}", offset=offset)

    # then - only the retention window is kept in memory
    buffer = block._buffers["vid_1"]
    assert len(buffer) <= abs(offset) + BUFFER_MARGIN + 1


def test_flush_and_close_contract() -> None:
    # given
    block = TimeTravelBlockV1()
    block.run(image=_image(0), data="det-0", offset=5)

    # then - drain emits nothing (trailing frames have no future)
    assert block.flush_stream_pipeline_outputs() == []
    # close must NOT wipe persistent state: it is called after every non-deferred
    # run_workflow, and buffers have to survive across frames.
    block.close_stream_pipeline()
    assert block._buffers != {}


def test_state_persists_across_runs_after_close() -> None:
    # given - simulates the standalone ExecutionEngine.run() path, where
    # close_stream_pipeline() is invoked after each run.
    block = TimeTravelBlockV1()

    # when
    block.run(image=_image(0), data="det-0", offset=-1)
    block.close_stream_pipeline()
    result = block.run(image=_image(1), data="det-1", offset=-1)
    block.close_stream_pipeline()

    # then - frame 0 is still available to frame 1 despite the close between runs
    assert result["output"] == "det-0"
    assert result["is_available"] is True

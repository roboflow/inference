"""Real compiled JPEG lists, selected limits and downstream byte consumers."""

from datetime import datetime

import cv2
import numpy as np
import pytest
import torch
from roboflow_workflows.execution_engine.entities.base import VideoMetadata
from roboflow_workflows.v2.blocks.control.blocks import ContinueIfBlock
from roboflow_workflows.v2.blocks.temporal.image_stack import ImageStackBlock
from roboflow_workflows.v2.data.batch import (
    Batch,
    EntryMetadata,
    InputValue,
    SampleContext,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import (
    BOOLEAN_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
)
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref
from roboflow_workflows.v2.engine.errors import StepExecutionError, WorkflowInputError
from roboflow_workflows.v2.engine.plan import CompileOptions


class DecodeMarkers(Block):
    """Consume a single JPEG list as an ordinary Ref payload."""

    type = "test/decode_jpegs@v1"
    outputs = {"markers": Output(LIST_OF_VALUES_KIND, source="frames")}

    class Params(BlockParams):
        frames: Ref(LIST_OF_VALUES_KIND)

    def run(self, *, frames):
        decoded = [
            cv2.imdecode(np.frombuffer(blob, dtype=np.uint8), cv2.IMREAD_COLOR)
            for blob in frames
        ]
        assert all(item is not None for item in decoded)
        return {"markers": [round(float(item.mean())) for item in decoded]}


def frame(marker, *, number, video="a", size=(80, 160)):
    metadata = VideoMetadata(
        video_identifier=video,
        frame_number=number,
        frame_timestamp=datetime(2026, 1, 1),
    )
    return ImageData.from_tensor(
        torch.full((3, *size), marker, dtype=torch.uint8), video_metadata=metadata
    )


def plan(mode, *, gated=False, only_image=False):
    steps = [
        {
            "type": ImageStackBlock.type,
            "name": "stack",
            "image": "$inputs.image",
            "stack_size": "$inputs.size",
            "clear": "$inputs.clear",
            "resolution_width": "$inputs.width",
            "resolution_height": "$inputs.height",
        },
        {"type": DecodeMarkers.type, "name": "decode", "frames": "$steps.stack.frames"},
    ]
    if gated:
        steps.insert(
            0,
            {
                "type": ContinueIfBlock.type,
                "name": "gate",
                "condition": "$inputs.keep",
                "next_steps": ["$steps.stack"],
            },
        )
    selectors = {
        "frames": "$steps.stack.frames",
        "count": "$steps.stack.frames_count",
        "markers": "$steps.decode.markers",
    }
    if only_image:
        selectors = {"image": "$inputs.image"}
    definition = {
        "version": "2.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
            *[
                {"type": "WorkflowParameter", "name": name}
                for name in ("size", "clear", "width", "height", "keep")
            ],
        ],
        "steps": steps,
        "outputs": [
            {"type": "JsonField", "name": name, "selector": selector}
            for name, selector in selectors.items()
        ],
    }
    return compile_workflow(
        definition,
        catalogue=Catalogue(
            [ImageStackBlock, DecodeMarkers, ContinueIfBlock],
            kinds=[IMAGE_KIND, BOOLEAN_KIND, INTEGER_KIND, LIST_OF_VALUES_KIND],
        ),
        options=CompileOptions(block_execution=mode),
    )


def run(session, image, **kwargs):
    inputs = {
        "image": image,
        "size": 4,
        "clear": False,
        "width": 1920,
        "height": 1080,
        "keep": True,
        **kwargs,
    }
    result = session.run(inputs)
    values = {}
    for name, value in result.outputs.data.items():
        if isinstance(value, Batch):
            members = list(value)
            value = members[0] if members else None
        values[name] = value
    return result, values


@pytest.mark.parametrize("mode", ["run", "phases"])
def test_selectors_resize_clear_and_downstream_lists(mode):
    session = plan(mode).create_session()
    try:
        for n in range(1, 5):
            result, values = run(session, frame(n * 20, number=n))
        assert values["markers"] == [80, 60, 40, 20]
        result, values = run(session, frame(100, number=5), size=2, width=64)
        assert values["markers"] == [100, 80]
        assert values["count"] == 2
        assert [
            cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR).shape[:2]
            for b in values["frames"]
        ] == [(32, 64), (80, 160)]
        result, values = run(session, frame(120, number=6), size=4)
        assert values["markers"] == [120, 100, 80]
        result, values = run(session, frame(140, number=7), clear=True)
        assert values["markers"] == [140]
        assert values["count"] == 1
    finally:
        session.close()


@pytest.mark.parametrize("mode", ["run", "phases"])
def test_interleaved_sources_reset_and_current_context(mode):
    session = plan(mode).create_session()
    try:
        for video, number, marker, expected in [
            ("a", 2, 20, [20]),
            ("b", 1, 60, [60]),
            ("a", 3, 40, [40, 20]),
            ("a", 1, 80, [80]),
            ("b", 2, 100, [100, 60]),
        ]:
            result, values = run(session, frame(marker, number=number, video=video))
            assert values["markers"] == expected
            assert values["count"] == len(expected)
            assert all(isinstance(blob, bytes) for blob in values["frames"])
            output = result.outputs.data["frames"]
            if isinstance(output, Batch):
                # History is one row, not an expanded collection of frames.
                assert len(output) == 1
    finally:
        session.close()


@pytest.mark.parametrize("mode", ["run", "phases"])
def test_image_context_source_fallback(mode):
    session = plan(mode).create_session()
    try:
        for source, marker, expected in [
            ("left", 20, [20]),
            ("right", 60, [60]),
            ("left", 40, [40, 20]),
        ]:
            image = frame(marker, number=marker, video="")
            value = InputValue(
                data=image,
                metadata=EntryMetadata(sample={(0,): SampleContext(source_id=source)}),
            )
            result, values = run(session, value)
            assert values["markers"] == expected
            metadata = result.outputs.metadata["frames"]
            assert metadata.sample_at((0,)).source_id == source
    finally:
        session.close()


@pytest.mark.parametrize("mode", ["run", "phases"])
def test_filtered_image_does_not_clear_or_append(mode):
    session = plan(mode, gated=True).create_session()
    try:
        run(session, frame(20, number=1))
        result, values = run(session, frame(40, number=2), keep=False, clear=True)
        assert "frames" not in values
        result, values = run(session, frame(60, number=3))
        assert values["markers"] == [60, 20]
    finally:
        session.close()


@pytest.mark.parametrize("mode", ["run", "phases"])
@pytest.mark.parametrize("field,value", [("size", 0), ("width", 63), ("height", 1081)])
def test_invalid_selected_limits_raise(mode, field, value):
    session = plan(mode).create_session()
    try:
        with pytest.raises(StepExecutionError, match="must be an integer"):
            run(session, frame(20, number=1), **{field: value})
    finally:
        session.close()


@pytest.mark.parametrize("mode", ["run", "phases"])
def test_undemanded_history_still_encodes_stateful_arrivals(mode, monkeypatch):
    import roboflow_workflows.v2.blocks.temporal.image_stack as module

    original = module.encode_image_to_jpeg_bytes
    seen = []

    def encode(*args, **kwargs):
        seen.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "encode_image_to_jpeg_bytes", encode)
    session = plan(mode, only_image=True).create_session()
    try:
        result, values = run(session, frame(20, number=1))
        assert isinstance(values["image"], ImageData)
        assert seen == [True]
    finally:
        session.close()


@pytest.mark.parametrize("mode", ["run", "phases"])
def test_empty_image_batch_skips_without_changing_history(mode):
    session = plan(mode).create_session()
    try:
        run(session, frame(20, number=1))
        result, values = run(session, [], clear=True)
        assert result.rows() == []
        result, values = run(session, frame(60, number=3))
        assert values["markers"] == [60, 20]
    finally:
        session.close()


@pytest.mark.parametrize("mode", ["run", "phases"])
def test_none_or_missing_required_image_rejected_before_history_change(mode):
    session = plan(mode).create_session()
    try:
        run(session, frame(20, number=1))
        with pytest.raises(WorkflowInputError):
            run(session, None, clear=True)
        inputs = {"size": 4, "clear": True, "width": 1920, "height": 1080, "keep": True}
        with pytest.raises(WorkflowInputError, match="Missing"):
            session.run(inputs)
        result, values = run(session, frame(60, number=3))
        assert values["markers"] == [60, 20]
    finally:
        session.close()


@pytest.mark.parametrize("mode", ["run", "phases"])
def test_session_state_reset_forgets_encoded_history(mode):
    session = plan(mode).create_session()
    try:
        run(session, frame(20, number=1))
        session.instances[("stack",)].reset_state()
        result, values = run(session, frame(60, number=3))
        assert values["markers"] == [60]
    finally:
        session.close()


@pytest.mark.parametrize("mode", ["run", "phases"])
def test_mixed_video_identity_and_sample_identity_survive_sparse_reordering(mode):
    """History follows identity, while output context follows the current row."""
    session = plan(mode).create_session()

    def feed(rows, sources, *, indices=None, known=None):
        images = Batch(rows, indices=indices, known_indices=known)
        context = EntryMetadata(
            sample={
                index: SampleContext(source_id=source)
                for index, source in zip(images.indices, sources)
            }
        )
        result, _ = run(session, InputValue(data=images, metadata=context))
        return result

    try:
        # Different sample sources with the same video identifier share history.
        # An image without a video identifier uses its own sample source.
        result = feed(
            [
                frame(20, number=1, video="video"),
                frame(60, number=1, video=""),
                frame(40, number=2, video="video"),
            ],
            ["first", "fallback", "alternate"],
        )
        assert list(result.outputs.data["markers"]) == [[20], [60], [40, 20]]
        assert [
            result.outputs.metadata["frames"].sample_at((i,)).source_id
            for i in range(3)
        ] == ["first", "fallback", "alternate"]

        # Missing rows do not affect the video history. The remaining fallback
        # source moves from index1 to index2 but still sees its earlier frame.
        result = feed(
            [frame(80, number=2, video="")],
            ["fallback"],
            indices=[(2,)],
            known=[(0,), (1,), (2,)],
        )
        assert result.outputs.data["frames"].indices == ((2,),)
        assert list(result.outputs.data["markers"]) == [[80, 60]]
        assert result.outputs.metadata["frames"].sample_at((2,)).source_id == "fallback"

        result = feed(
            [frame(100, number=3, video="video"), frame(120, number=3, video="")],
            ["third", "fallback"],
        )
        assert list(result.outputs.data["markers"]) == [[100, 40, 20], [120, 80, 60]]
    finally:
        session.close()

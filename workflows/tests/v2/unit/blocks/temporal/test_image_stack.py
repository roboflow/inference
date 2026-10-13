"""JPEG semantics, ownership, source capacity and resolved configuration."""

from datetime import datetime
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from pydantic import ValidationError
from roboflow_workflows.execution_engine.entities.base import VideoMetadata
from roboflow_workflows.v2.blocks.temporal.image_stack import (
    ImageStackBlock,
    ImageStackCapacityError,
)
from roboflow_workflows.v2.data.batch import SampleContext
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.engine.context import ExecutionContext, use_execution_context
from roboflow_workflows.v2.engine.recording.codecs import CodecRegistry


def image(marker=20, *, number=1, video="a", size=(80, 160), channels=3):
    pixels = torch.full((channels, *size), marker, dtype=torch.uint8)
    metadata = (
        None
        if video is None
        else VideoMetadata(
            video_identifier=video,
            frame_number=number,
            frame_timestamp=datetime(2026, 1, 1),
        )
    )
    return ImageData.from_tensor(pixels, video_metadata=metadata)


def call(block, frame, *, source=None, **kwargs):
    sample = SampleContext(source_id=source) if source is not None else None
    scope = SimpleNamespace(sample_at=lambda index, field=None: sample)
    context = ExecutionContext(
        step_path=("stack",),
        block_type=block.type,
        session_id="test",
        run_id="test",
        indices=((0,),),
        call_scope=scope,
    )
    with use_execution_context(context):
        result = block.run(image=frame, **kwargs)

    return result


def decode(blob):
    result = cv2.imdecode(np.frombuffer(blob, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert result is not None
    return result


def markers(result):
    return [round(float(decode(blob).mean())) for blob in result["frames"]]


def test_newest_order_resize_retention_and_clear():
    block = ImageStackBlock()
    for n in range(1, 5):
        result = call(block, image(n * 20, number=n), stack_size=4)
    assert markers(result) == [80, 60, 40, 20]
    result = call(block, image(100, number=5), stack_size=2, resolution_width=64)
    assert markers(result) == [100, 80]
    assert [decode(b).shape[:2] for b in result["frames"]] == [(32, 64), (80, 160)]
    result = call(block, image(120, number=6), stack_size=4)
    assert markers(result) == [120, 100, 80]
    result = call(block, image(140, number=7), clear=True)
    assert markers(result) == [140]
    assert result["frames_count"] == 1


def test_tensor_and_returned_list_mutation_cannot_change_history():
    block = ImageStackBlock()
    frame = image(40)
    result = call(block, frame)
    original = result["frames"][0]
    frame.tensor_image.fill_(200)
    result["frames"].clear()
    result = call(block, image(80, number=2))
    assert result["frames"][1] == original
    assert markers(result) == [80, 40]


@pytest.mark.parametrize("channels", [1, 3])
def test_downsampling_preserves_ratio_and_never_upsamples(channels):
    block = ImageStackBlock()
    for size, expected in [
        ((300, 100), (64, 21)),
        ((2, 5000), (1, 64)),
        ((20, 30), (20, 30)),
    ]:
        result = call(
            block,
            image(size=size, channels=channels),
            clear=True,
            resolution_width=64,
            resolution_height=64,
        )
        assert decode(result["frames"][0]).shape[:2] == expected


def test_rgb_to_bgr_and_noncontiguous_pixels():
    frame = image(size=(80, 160))
    frame.tensor_image[0].fill_(220)
    frame.tensor_image[1].fill_(80)
    frame.tensor_image[2].fill_(20)
    frame = ImageData.from_tensor(
        frame.tensor_image.transpose(1, 2), video_metadata=frame.video_metadata
    )
    before = frame.tensor_image.clone()
    result = call(ImageStackBlock(), frame)
    np.testing.assert_allclose(
        decode(result["frames"][0]).mean((0, 1)), [20, 80, 220], atol=3
    )
    assert torch.equal(before, frame.tensor_image)


def test_source_isolation_regression_equal_number_and_reset():
    block = ImageStackBlock()
    call(block, image(20, number=2, video="a"))
    call(block, image(60, number=1, video="b"))
    assert markers(call(block, image(40, number=2, video="a"))) == [40, 20]
    assert markers(call(block, image(80, number=1, video="a"))) == [80]
    assert markers(call(block, image(100, number=2, video="b"))) == [100, 60]
    block.reset_state()
    assert markers(call(block, image(120, number=3, video="b"))) == [120]


def test_source_policy_reads_image_context_and_supports_still_images():
    block = ImageStackBlock()
    call(block, image(20, video=None), source="left")
    call(block, image(80, video=""), source="right")
    assert markers(call(block, image(40, video=None), source="left")) == [40, 20]
    assert markers(call(block, image(100, video=""), source="right")) == [100, 80]
    assert markers(call(block, image(60, video=None))) == [60]
    assert markers(call(block, image(120, video="a"), source="left")) == [120]


def test_capacity_rejects_before_encoding_without_eviction(monkeypatch):
    block = ImageStackBlock()
    call(block, image(20), max_sources=1)
    import roboflow_workflows.v2.blocks.temporal.image_stack as module

    with monkeypatch.context() as patch:
        patch.setattr(
            module,
            "encode_image_to_jpeg_bytes",
            lambda *args, **kwargs: pytest.fail("must reject before encode"),
        )
        with pytest.raises(ImageStackCapacityError, match="max_sources=1"):
            call(block, image(40, video="b"), max_sources=1)
    assert markers(call(block, image(60, number=2), max_sources=1)) == [60, 20]
    block.reset_state()
    assert markers(call(block, image(80, video="b"), max_sources=1)) == [80]


@pytest.mark.parametrize(
    "field,value",
    [
        ("stack_size", 0),
        ("stack_size", 65),
        ("stack_size", True),
        ("resolution_width", 63),
        ("resolution_width", 1921),
        ("resolution_height", 63),
        ("resolution_height", 1081),
        ("stack_size", 2.5),
        ("clear", 1),
        ("max_sources", 0),
    ],
)
def test_invalid_literals_and_runtime_values_rejected_without_state(field, value):
    with pytest.raises(ValidationError):
        ImageStackBlock.Params(image="$inputs.image", **{field: value})
    block = ImageStackBlock()
    expected_error = {
        "clear": "clear must be boolean",
        "max_sources": "max_sources must be a positive integer",
    }.get(field, f"{field} must be an integer")
    with pytest.raises(ValueError, match=expected_error):
        call(block, image(), **{field: value})
    assert not block._buffers


def test_encoding_failure_preserves_history(monkeypatch):
    block = ImageStackBlock()
    call(block, image(20))
    import roboflow_workflows.v2.blocks.temporal.image_stack as module

    def fail(*args, **kwargs):
        raise RuntimeError("encoder failure")

    with monkeypatch.context() as patch:
        patch.setattr(module, "encode_image_to_jpeg_bytes", fail)
        with pytest.raises(RuntimeError, match="encoder failure"):
            call(block, image(40, number=2), clear=True, stack_size=1)
    assert markers(call(block, image(60, number=3))) == [60, 20]


def test_generic_recording_codec_round_trips_exact_jpeg_bytes():
    class Blobs:
        def __init__(self):
            self.items = []

        def put(self, data):
            self.items.append(bytes(data))
            return {"index": len(self.items) - 1}

        def get(self, ref):
            return self.items[ref["index"]]

    value = call(ImageStackBlock(), image())["frames"]
    blobs = Blobs()
    codecs = CodecRegistry()
    encoded = codecs.encoder(blobs).encode(value)
    assert codecs.decoder(blobs).decode(encoded) == value

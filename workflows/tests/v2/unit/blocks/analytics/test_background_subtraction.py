"""``v2/background_subtraction@v1`` called directly on generated scenes.

The oracle is a static gradient with a 32 px white square moving 12 px per
frame. Once the background is learned, the strip the square newly covers is
foreground and the untouched rows stay background. Interleaved sources must
give exactly the masks of the same sources run alone.
"""

from datetime import datetime, timedelta
from typing import Any, List, Optional

import pytest
import torch
from pydantic import ValidationError
from roboflow_workflows.execution_engine.entities.base import VideoMetadata
from roboflow_workflows.v2.blocks.analytics.background_subtraction import (
    DEFAULT_MAX_SOURCES,
    BackgroundModelCapacityError,
    BackgroundSubtractionBlock,
)
from roboflow_workflows.v2.data.batch import SampleContext
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.engine.context import ExecutionContext, use_execution_context

T0 = datetime(2026, 10, 13)
HEIGHT, WIDTH, SIDE, STEP, TOP = 96, 192, 32, 12, 32


def scene(t: int, *, moving: bool = True, channels: int = 3) -> torch.Tensor:
    pixels = (torch.arange(WIDTH) // 3).to(torch.uint8).expand(channels, HEIGHT, WIDTH)
    pixels = pixels.clone()
    if moving:
        pixels[:, TOP : TOP + SIDE, STEP * t : STEP * t + SIDE] = 255

    return pixels


def frame(
    t: int,
    *,
    video: Optional[str] = "cam-a",
    number: Optional[int] = None,
    pixels: Optional[torch.Tensor] = None,
) -> ImageData:
    metadata = None
    if video is not None:
        metadata = VideoMetadata(
            video_identifier=video,
            frame_number=t if number is None else number,
            frame_timestamp=T0 + timedelta(seconds=t),
        )
    pixels = scene(t) if pixels is None else pixels
    image = ImageData.from_tensor(pixels, video_metadata=metadata)

    return image


def masks(block: BackgroundSubtractionBlock, images: List[ImageData], **params) -> list:
    outputs = [block.run(image=image, **params)["image"] for image in images]

    return [output.tensor_image[0].clone() for output in outputs]


def test_mask_keeps_geometry_identity_and_metadata():
    image = frame(0)
    output = BackgroundSubtractionBlock().run(image=image)["image"]

    assert output.size_hw == image.size_hw and output.channels == 3
    assert output.image_id == image.image_id
    assert output.parent == image.parent and output.root == image.root
    assert output.video_metadata is image.video_metadata
    assert output.tensor_image.dtype == torch.uint8
    assert torch.equal(output.tensor_image[0], output.tensor_image[1])
    assert torch.equal(output.tensor_image[0], output.tensor_image[2])
    assert set(output.tensor_image.unique().tolist()) <= {0, 127, 255}
    assert torch.equal(image.tensor_image, scene(0)), "input pixels changed"


def test_moving_square_strip_is_foreground_and_far_rows_are_background():
    block = BackgroundSubtractionBlock()
    outputs = masks(block, [frame(t) for t in range(10)])

    for t in range(5, 10):
        leading = outputs[t][TOP : TOP + SIDE, STEP * t + 20 : STEP * t + SIDE]
        assert bool((leading == 255).all()), t
        assert int(outputs[t][: TOP - 4].count_nonzero()) == 0, t


def test_first_frame_is_foreground_and_a_static_scene_settles_to_background():
    static = [frame(t, pixels=scene(t, moving=False)) for t in range(6)]
    outputs = masks(BackgroundSubtractionBlock(), static)

    assert int(outputs[0].count_nonzero()) == HEIGHT * WIDTH
    assert all(int(mask.count_nonzero()) == 0 for mask in outputs[2:])


def test_interleaved_sources_match_the_same_sources_run_alone():
    moving = [frame(t, video="moving") for t in range(9)]
    static = [frame(t, video="static", pixels=scene(t, moving=False)) for t in range(9)]
    alone = masks(BackgroundSubtractionBlock(), moving, history=5)
    alone += masks(BackgroundSubtractionBlock(), static, history=5)

    block = BackgroundSubtractionBlock()
    together = [masks(block, [a, b], history=5) for a, b in zip(moving, static)]
    together = [pair[0] for pair in together] + [pair[1] for pair in together]

    assert all(torch.equal(a, b) for a, b in zip(alone, together))


def test_one_block_shares_no_model_with_another_instance():
    # Two steps (or sessions) each own their models.
    first, second = BackgroundSubtractionBlock(), BackgroundSubtractionBlock()
    masks(first, [frame(t) for t in range(6)])
    fresh = masks(second, [frame(0)])[0]

    assert torch.equal(fresh, masks(BackgroundSubtractionBlock(), [frame(0)])[0])


@pytest.mark.parametrize(
    "restart",
    [
        pytest.param(frame(3, number=1), id="lower-frame-number"),
        pytest.param(frame(3, pixels=scene(3)[:, :, :96].clone()), id="new-size"),
        pytest.param(frame(3, pixels=scene(3, channels=1)), id="new-channels"),
    ],
)
def test_source_restarts_like_a_fresh_model(restart):
    block = BackgroundSubtractionBlock()
    masks(block, [frame(t) for t in range(3)])

    restarted = masks(block, [restart])[0]

    assert torch.equal(restarted, masks(BackgroundSubtractionBlock(), [restart])[0])


def test_equal_and_skipped_frame_numbers_do_not_restart():
    block = BackgroundSubtractionBlock()
    outputs = masks(block, [frame(0), frame(1), frame(1, number=1), frame(9)])

    fresh = masks(BackgroundSubtractionBlock(), [frame(9)])[0]
    assert not torch.equal(outputs[-1], fresh)


def test_capacity_error_changes_no_model_and_reset_frees_sources():
    block = BackgroundSubtractionBlock()
    for camera in ("a", "b"):
        block.run(image=frame(0, video=camera), max_sources=2)

    with pytest.raises(BackgroundModelCapacityError) as failed:
        block.run(image=frame(0, video="c"), max_sources=2)
    message = str(failed.value)
    assert "'c'" in message and "max_sources=2" in message
    assert "restart the pipeline" in message
    assert len(block._sources) == 2
    block.run(image=frame(1, video="a"), max_sources=2)

    block.reset_state()
    block.run(image=frame(0, video="c"), max_sources=2)
    assert len(block._sources) == 1


def test_default_capacity_is_bounded():
    assert (
        BackgroundSubtractionBlock.Params(image="$inputs.image").max_sources
        == DEFAULT_MAX_SOURCES
        == 16
    )


def _run_without_video(block, image: ImageData, source: Optional[str]):
    sample = SampleContext(source_id=source) if source is not None else None

    class Scope:
        def sample_at(self, index, *, field=None):
            return sample if field == "image" else None

    context = ExecutionContext(
        step_path=("subtract",),
        block_type=block.type,
        session_id="session",
        run_id="run",
        indices=((0,),),
        call_scope=Scope(),
    )
    with use_execution_context(context):
        result = block.run(image=image, max_sources=2)

    return result


def test_images_without_video_metadata_key_by_sample_source_then_default():
    block = BackgroundSubtractionBlock()
    _run_without_video(block, frame(0, video=None), source="left")
    _run_without_video(block, frame(0, video=None), source=None)

    assert sorted(block._sources._states) == ["default", "left"]


def test_plain_run_without_video_metadata_uses_the_default_source():
    # No engine context: a direct call keys the image by "default".
    block = BackgroundSubtractionBlock()
    outputs = masks(block, [frame(t, video=None) for t in range(6)])

    assert sorted(block._sources._states) == ["default"]
    expected = masks(BackgroundSubtractionBlock(), [frame(t) for t in range(6)])
    assert all(torch.equal(a, b) for a, b in zip(outputs, expected))


def test_noncontiguous_and_grayscale_inputs_match_contiguous_color_ones():
    images = [frame(t) for t in range(6)]
    views = [frame(t, pixels=_channels_last_view(scene(t))) for t in range(6)]
    assert not views[0].tensor_image.is_contiguous()

    assert all(
        torch.equal(a, b)
        for a, b in zip(
            masks(BackgroundSubtractionBlock(), images),
            masks(BackgroundSubtractionBlock(), views),
        )
    )
    gray = BackgroundSubtractionBlock().run(image=frame(0, pixels=scene(0, channels=1)))
    assert gray["image"].channels == 3


def _channels_last_view(pixels: torch.Tensor) -> torch.Tensor:
    view = pixels.permute(1, 2, 0).contiguous().permute(2, 0, 1)

    return view


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_mask_stays_on_the_input_device():
    image = frame(0, pixels=scene(0).to("cuda"))
    output = BackgroundSubtractionBlock().run(image=image)["image"]

    assert output.device == image.device


@pytest.mark.parametrize(
    "params, name",
    [
        ({"threshold": 0}, "threshold"),
        ({"history": "5"}, "history"),
        ({"history": True}, "history"),
        ({"max_sources": 0}, "max_sources"),
    ],
)
def test_selected_values_that_are_not_positive_integers_are_rejected(params, name):
    with pytest.raises(ValueError, match=f"{name} must be a positive integer"):
        BackgroundSubtractionBlock().run(image=frame(0), **params)


def test_literal_parameters_are_validated_and_selectors_accepted():
    params: Any = BackgroundSubtractionBlock.Params
    with pytest.raises(ValidationError):
        params(image="$inputs.image", threshold=0)
    with pytest.raises(ValidationError):
        params(image="$inputs.image", history=2.5)
    with pytest.raises(ValidationError):
        params(image="$inputs.image", max_sources="$inputs.limit")

    params(image="$inputs.image", threshold="$inputs.t", history="$inputs.h")

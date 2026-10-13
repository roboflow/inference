"""``v2/frame_delay@v1`` called directly, with a hand-built call context.

The oracle is latest V1 ``roboflow_core/frame_delay@v1``: any data, literal
or selected offset, ``default_value``, history sized by the
deepest offset seen, restart only on a strictly lower frame number. V2 uses
an explicit source capacity instead of V1's silent 16-source eviction.

Two aliases are told apart throughout:

* incoming alias: the caller's image. The block must not keep it.
* returned retained alias: the delayed value. It is the block's own kept
  object, so it lives until a later frame evicts it.
"""

import gc
import weakref
from datetime import datetime, timedelta
from typing import Any, List, Optional
from unittest import mock

import pytest
import torch
from pydantic import ValidationError
from roboflow_workflows.execution_engine.entities.base import VideoMetadata
from roboflow_workflows.v2.blocks.temporal import FrameDelayBlock
from roboflow_workflows.v2.blocks.temporal.frame_delay import (
    MAX_OFFSET,
    MAX_TRACKED_SOURCES,
    FrameHistoryCapacityError,
)
from roboflow_workflows.v2.data.batch import SampleContext
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.engine.context import (
    ExecutionContext,
    use_execution_context,
)

from inference_models.models.base.object_detection import Detections

DEVICES = [
    pytest.param("cpu", id="cpu"),
    pytest.param(
        "mps",
        id="mps",
        marks=pytest.mark.skipif(
            not torch.backends.mps.is_available(), reason="needs MPS"
        ),
    ),
    pytest.param(
        "cuda",
        id="cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    ),
]
T0 = datetime(2026, 10, 13)
UNSET = object()


class _Scope:
    """Call scope where only the image carries a sample context.

    As with a ``data`` without a context: the whole call has no common
    context, and only ``field="image"`` finds the image's source.
    """

    def __init__(self, image_sample: Optional[SampleContext]) -> None:
        self._image_sample = image_sample

    def sample_at(self, index, *, field=None):
        return self._image_sample if field == "image" else None


def _frame(number: int, *, video: str = "cam-a", device: str = "cpu") -> ImageData:
    pixels = torch.full((3, 4, 4), number % 256, dtype=torch.uint8, device=device)
    metadata = VideoMetadata(
        video_identifier=video,
        frame_number=number,
        frame_timestamp=T0 + timedelta(seconds=number),
    )
    image = ImageData.from_tensor(
        pixels, image_id=f"{video}-{number}", video_metadata=metadata
    )

    return image


def _call(
    block: FrameDelayBlock,
    image: ImageData,
    *,
    data: Any = UNSET,
    offset: int = -1,
    default_value: Any = None,
    image_device: str = "same",
    source: Optional[str] = None,
    max_sources: Any = UNSET,
) -> dict:
    """One call; ``data`` defaults to the current frame number."""
    sample = SampleContext(source_id=source) if source is not None else None
    context = ExecutionContext(
        step_path=("delay",),
        block_type=FrameDelayBlock.type,
        session_id="session",
        run_id="run",
        indices=((0,),),
        call_scope=_Scope(sample),
    )
    if data is UNSET:
        data = image.video_metadata.frame_number
    capacity = {} if max_sources is UNSET else {"max_sources": max_sources}
    with use_execution_context(context):
        result = block.run(
            image=image,
            data=data,
            offset=offset,
            default_value=default_value,
            image_device=image_device,
            **capacity,
        )

    return result


def _outputs(numbers: List[int], *, offsets: Any = -1, **kwargs: Any) -> List[Any]:
    """``output`` per frame number; ``data`` is the frame number itself."""
    block = FrameDelayBlock()
    if isinstance(offsets, int):
        offsets = [offsets] * len(numbers)
    outputs = [
        _call(block, _frame(number), offset=offset, **kwargs)["output"]
        for number, offset in zip(numbers, offsets)
    ]

    return outputs


def _params(**overrides: Any) -> FrameDelayBlock.Params:
    fields = {"image": "$inputs.image", "data": "$inputs.data", "offset": -1}
    params = FrameDelayBlock.Params(**{**fields, **overrides})

    return params


# -------------------------------------------------------------- declaration
def test_declaration_matches_v1_outputs_and_declares_retention():
    spec = FrameDelayBlock.__block_spec__

    assert spec.type == "v2/frame_delay@v1"
    assert spec.prunable is False
    assert set(FrameDelayBlock.outputs) == {
        "output",
        "is_available",
        "reference_frame_number",
    }
    assert spec.retains_inputs == ("data",)
    assert spec.retains_outputs == ("output",)
    assert MAX_TRACKED_SOURCES == 256 and MAX_OFFSET == 256
    assert _params().max_sources == 256


def test_output_takes_the_common_context_while_flags_follow_the_image():
    # source=None merges the context of every binding at run time. It does
    # not make the mutation checks trace data or image through the output;
    # retains_inputs and retains_outputs (declaration test above) cover the
    # kept and returned values.
    outputs = FrameDelayBlock.outputs

    assert outputs["output"].source is None
    assert outputs["is_available"].source == "image"
    assert outputs["reference_frame_number"].source == "image"


@pytest.mark.parametrize("offset", [0, -1, -MAX_OFFSET, "$inputs.offset"])
def test_literal_offsets_in_range_and_selected_offsets_are_accepted(offset):
    assert _params(offset=offset).offset == offset


@pytest.mark.parametrize("offset", [1, -MAX_OFFSET - 1])
def test_future_and_too_deep_literal_offsets_are_rejected(offset):
    with pytest.raises(ValidationError):
        _params(offset=offset)


@pytest.mark.parametrize("offset", [1, -MAX_OFFSET - 1])
def test_a_selected_offset_out_of_range_fails_the_call(offset):
    with pytest.raises(ValueError, match="offset must be from"):
        _call(FrameDelayBlock(), _frame(1), offset=offset)


@pytest.mark.parametrize("value", [None, False, True, 0, 7, 0.5, "", "none"])
def test_literal_default_values_keep_their_type(value):
    default = _params(default_value=value).default_value

    assert default == value and type(default) is type(value)


@pytest.mark.parametrize("value", [[1], {"a": 1}])
def test_non_scalar_default_values_are_rejected(value):
    with pytest.raises(ValidationError):
        _params(default_value=value)


def test_image_device_is_same_or_cpu():
    assert _params().image_device == "same"
    assert _params(image_device="cpu").image_device == "cpu"
    with pytest.raises(ValidationError):
        _params(image_device="cuda")


# ---------------------------------------------------------------- semantics
def test_outputs_are_value_availability_and_the_current_frame_number():
    block = FrameDelayBlock()

    results = [
        _call(block, _frame(n), data=f"d{n}", offset=-2, default_value="none")
        for n in (1, 2, 3)
    ]

    assert [r["output"] for r in results] == ["none", "none", "d1"]
    assert [r["is_available"] for r in results] == [False, False, True]
    assert [r["reference_frame_number"] for r in results] == [1, 2, 3]


def test_offset_zero_returns_this_frames_value():
    assert _outputs([1, 2], offsets=0) == [1, 2]


def test_a_frame_number_gap_returns_the_default_not_the_previous_arrival():
    assert _outputs([1, 2, 4, 5], default_value=-1) == [-1, 1, -1, 4]


@pytest.mark.parametrize("restart", [1, 3])
def test_a_strictly_lower_frame_number_restarts_the_history(restart):
    assert _outputs([4, 5, restart, restart + 1]) == [None, 4, None, restart]


def test_a_repeated_frame_number_replaces_its_value_without_restarting():
    block = FrameDelayBlock()
    _call(block, _frame(1), data="first")
    _call(block, _frame(2), data="old")

    repeated = _call(block, _frame(2), data="new")
    later = _call(block, _frame(3), data="x")

    assert repeated["output"] == "first"
    assert later["output"] == "new"


def test_duplicate_crops_of_one_frame_overwrite_each_other():
    # Crops share source and frame number, so the last crop's value wins.
    block = FrameDelayBlock()
    for crop in ("crop-0", "crop-1"):
        _call(block, _frame(1), data=crop)

    seen = [_call(block, _frame(2), data=crop)["output"] for crop in ("c0", "c1")]
    same_frame = _call(block, _frame(2), data="c2", offset=0)["output"]

    assert seen == ["crop-1", "crop-1"]
    assert same_frame == "c2"


def test_a_restart_clears_only_its_own_source():
    block = FrameDelayBlock()
    for number in (1, 2):
        _call(block, _frame(number, video="cam-a"))
        _call(block, _frame(number, video="cam-b"))

    _call(block, _frame(1, video="cam-a"))
    cam_b = _call(block, _frame(3, video="cam-b"))
    cam_a = _call(block, _frame(2, video="cam-a"))

    assert cam_b["output"] == 2
    assert cam_a["output"] == 1


# ------------------------------------------------------ variable offsets
def test_a_deep_offset_keeps_history_for_later_shallow_then_deep_lookups():
    offsets = [-3, -1, -1, -1, -3, -1, -3]

    assert _outputs([1, 2, 3, 4, 5, 6, 7], offsets=offsets) == [
        None,
        1,
        2,
        3,
        2,
        5,
        4,
    ]


def test_a_deeper_offset_mid_stream_finds_only_frames_still_kept():
    # Depth 1 kept frames 4-5 only: frame 3 is gone when -3 is first asked.
    offsets = [-1, -1, -1, -1, -1, -3, -3, -3]

    outputs = _outputs([1, 2, 3, 4, 5, 6, 7, 8], offsets=offsets, default_value=0)

    assert outputs == [0, 1, 2, 3, 4, 0, 4, 5]


def test_depth_is_per_source_and_forgotten_on_restart():
    snapshots = []

    def kept_numbers(block: FrameDelayBlock, key: str) -> List[int]:
        return list(block._histories[key].frames)

    block = FrameDelayBlock()
    for number in range(1, 6):
        _call(block, _frame(number, video="deep"), offset=-3)
        _call(block, _frame(number, video="shallow"), offset=-1)
    snapshots.append(kept_numbers(block, "deep"))
    snapshots.append(kept_numbers(block, "shallow"))
    for number in (1, 2, 3):
        _call(block, _frame(number, video="deep"), offset=-1)

    assert snapshots == [[2, 3, 4, 5], [4, 5]]
    assert kept_numbers(block, "deep") == [2, 3]


# ------------------------------------------------------- present None
def test_a_present_none_is_kept_and_differs_from_the_default():
    block = FrameDelayBlock()
    _call(block, _frame(1), data=None, default_value=0)

    kept_none = _call(block, _frame(2), data="x", default_value=0)
    not_kept = _call(block, _frame(4), data="y", default_value=0)

    assert (kept_none["output"], kept_none["is_available"]) == (None, True)
    assert (not_kept["output"], not_kept["is_available"]) == (0, False)


def test_default_none_is_unavailable_while_a_kept_none_is_available():
    block = FrameDelayBlock()

    first = _call(block, _frame(1), data=None)
    second = _call(block, _frame(2), data="x")

    assert (first["output"], first["is_available"]) == (None, False)
    assert (second["output"], second["is_available"]) == (None, True)


# ------------------------------------------------------------ source keys
def test_the_video_identifier_is_the_key_even_when_sources_differ():
    block = FrameDelayBlock()
    _call(block, _frame(1, video="cam-a"), source="one")

    result = _call(block, _frame(2, video="cam-a"), source="two")

    assert result["output"] == 1


def test_without_a_video_identifier_the_image_source_is_the_key():
    block = FrameDelayBlock()
    _call(block, _frame(1, video=""), data="a1", source="cam-a")
    _call(block, _frame(1, video=""), data="b1", source="cam-b")

    result = _call(block, _frame(2, video=""), source="cam-a")

    assert result["output"] == "a1"


def test_with_neither_every_frame_shares_the_default_source():
    block = FrameDelayBlock()
    _call(block, _frame(1, video=""))

    result = _call(block, _frame(2, video=""))

    assert result["output"] == 1


def test_an_image_without_video_metadata_is_rejected():
    image = ImageData.from_tensor(torch.zeros((3, 2, 2), dtype=torch.uint8))

    with pytest.raises(ValueError, match="no video_metadata"):
        _call(FrameDelayBlock(), image, data=1)


@pytest.mark.parametrize("count, limit", [(17, None), (20, None), (20, 20), (257, 257)])
def test_interleaved_sources_keep_history_without_eviction(count, limit):
    block = FrameDelayBlock()
    capacity = {} if limit is None else {"max_sources": limit}
    for camera in range(count):
        _call(block, _frame(1, video=f"cam-{camera}"), data=camera, **capacity)

    for camera in range(count):
        result = _call(block, _frame(2, video=f"cam-{camera}"), **capacity)
        assert (result["output"], result["is_available"]) == (camera, True)


@pytest.mark.parametrize("limit", [2, MAX_TRACKED_SOURCES])
def test_capacity_failure_copies_nothing_and_preserves_every_existing_history(limit):
    block = FrameDelayBlock()
    for camera in range(limit):
        _call(block, _frame(1, video=f"cam-{camera}"), data=camera, max_sources=limit)
    before = {
        key: (history, list(history.frames.items()), history.depth)
        for key, history in block._histories.items()
    }
    rejected = _frame(1, video="overflow")
    with mock.patch.object(ImageData, "copy", side_effect=AssertionError("copied")):
        with pytest.raises(FrameHistoryCapacityError, match=f"max_sources={limit}"):
            _call(block, rejected, data=rejected, offset=-3, max_sources=limit)

    assert list(block._histories) == list(before)
    for key, (history, frames, depth) in before.items():
        assert block._histories[key] is history
        assert list(history.frames.items()) == frames and history.depth == depth
    for camera in range(limit):
        result = _call(block, _frame(2, video=f"cam-{camera}"), max_sources=limit)
        assert (result["output"], result["is_available"]) == (camera, True)

    block.reset_state()
    assert not block._histories
    assert not _call(block, rejected, max_sources=limit)["is_available"]
    assert _call(block, _frame(2, video="overflow"), max_sources=limit)["output"] == 1


@pytest.mark.parametrize("limit", [0, -1, True, False, 1.5, "2", None, "$inputs.limit"])
def test_source_capacity_is_a_positive_literal_integer(limit):
    with pytest.raises(ValidationError):
        _params(max_sources=limit)
    block = FrameDelayBlock()
    with pytest.raises(ValueError, match="max_sources must be a positive integer"):
        _call(block, _frame(1), max_sources=limit)
    assert not block._histories


def test_capacity_allocates_only_for_seen_sources_and_accepts_one():
    block = FrameDelayBlock()
    assert not block._histories
    assert _params(max_sources=1).max_sources == 1
    _call(block, _frame(1), max_sources=1)
    assert len(block._histories) == 1


def test_reset_state_forgets_history_of_this_block_only():
    block, other = FrameDelayBlock(), FrameDelayBlock()
    for instance in (block, other):
        _call(instance, _frame(1))

    block.reset_state()

    assert _call(block, _frame(2))["is_available"] is False
    assert _call(other, _frame(2))["is_available"] is True
    assert _call(block, _frame(3))["is_available"] is True


# -------------------------------------------------------------- ownership
def test_non_image_values_are_kept_and_returned_by_reference():
    detections = Detections(
        xyxy=torch.tensor([[1.0, 1.0, 3.0, 3.0]]),
        class_id=torch.tensor([0]),
        confidence=torch.tensor([0.5]),
    )
    image_in_dict = {"image": _frame(9)}
    for value in (detections, image_in_dict, ["a", 1], ("a", 1)):
        block = FrameDelayBlock()
        _call(block, _frame(1), data=value)

        assert _call(block, _frame(2))["output"] is value


@pytest.mark.parametrize("device", DEVICES)
def test_a_kept_image_is_a_same_device_copy_with_its_own_metadata(device):
    block = FrameDelayBlock()
    incoming = _frame(1, device=device)
    _call(block, _frame(1, device=device), data=incoming)

    delayed = _call(block, _frame(2, device=device))["output"]

    assert delayed is not incoming and delayed.device == incoming.device
    assert delayed.tensor_image.data_ptr() != incoming.tensor_image.data_ptr()
    assert delayed.image_id == "cam-a-1"
    assert delayed.video_metadata is incoming.video_metadata


@pytest.mark.parametrize("device", DEVICES)
def test_image_device_cpu_keeps_images_in_host_memory(device):
    block = FrameDelayBlock()
    _call(block, _frame(1), data=_frame(1, device=device), image_device="cpu")

    delayed = _call(block, _frame(2), image_device="cpu")["output"]

    assert delayed.device.type == "cpu"
    assert int(delayed.tensor_image.max()) == 1


def test_images_in_nested_lists_and_tuples_are_copied_and_the_rest_shared():
    label = {"name": "x"}
    crops = [_frame(1), (_frame(2), label), "text"]
    block = FrameDelayBlock()
    _call(block, _frame(1), data=crops)

    kept = _call(block, _frame(2))["output"]

    assert type(kept) is list and type(kept[1]) is tuple
    assert kept is not crops and kept[1] is not crops[1]
    assert kept[0] is not crops[0] and kept[1][0] is not crops[1][0]
    assert kept[0].image_id == crops[0].image_id
    assert kept[1][1] is label and kept[2] == "text"
    for copied, original in ((kept[0], crops[0]), (kept[1][0], crops[1][0])):
        pointer = original.tensor_image.data_ptr()
        assert copied.tensor_image.data_ptr() != pointer


@pytest.mark.parametrize("device", DEVICES)
def test_incoming_image_storage_is_released_after_the_call(device):
    block = FrameDelayBlock()
    incoming = _frame(1, device=device)
    storages = [weakref.ref(incoming.tensor_image.untyped_storage())]
    nested = [_frame(1, device=device)]
    storages.append(weakref.ref(nested[0].tensor_image.untyped_storage()))

    _call(block, incoming, data=incoming)
    _call(block, _frame(2, device=device), data=nested)
    del incoming, nested
    gc.collect()

    assert [storage() for storage in storages] == [None, None]


@pytest.mark.parametrize("device", DEVICES)
def test_incoming_mutation_after_the_call_does_not_reach_the_history(device):
    block = FrameDelayBlock()
    incoming = _frame(1, device=device)
    _call(block, incoming, data=incoming)

    incoming.tensor_image.fill_(0)
    delayed = _call(block, _frame(2, device=device))["output"]

    assert int(delayed.tensor_image.min()) == 1


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("offset", [0, -1, -3])
def test_at_most_abs_offset_plus_one_images_are_kept(device, offset):
    copies = []
    real_copy = ImageData.copy

    def recording_copy(image, **kwargs):
        copied = real_copy(image, **kwargs)
        copies.append(weakref.ref(copied.tensor_image.untyped_storage()))

        return copied

    block = FrameDelayBlock()
    with mock.patch.object(ImageData, "copy", recording_copy):
        for number in range(1, 8):
            image = _frame(number, device=device)
            _call(block, image, data=image, offset=offset)
    gc.collect()

    alive = [n + 1 for n, ref in enumerate(copies) if ref() is not None]
    assert len(copies) == 7
    assert alive == list(range(7 - abs(offset), 8))


def test_offset_zero_returns_the_kept_copy_until_a_later_frame_evicts_it():
    block = FrameDelayBlock()
    incoming = _frame(1)

    returned = _call(block, incoming, data=incoming, offset=0)["output"]
    storage = weakref.ref(returned.tensor_image.untyped_storage())
    assert returned is not incoming
    del returned
    gc.collect()
    retained_by_block = storage() is not None
    _call(block, _frame(2), data=_frame(2), offset=0)
    gc.collect()

    assert retained_by_block
    assert storage() is None


def test_mutating_a_returned_value_changes_the_history_it_shares():
    # The retained alias the block declares: a deep offset returns it again.
    block = FrameDelayBlock()
    _call(block, _frame(1), data=[1])
    _call(block, _frame(2), data=[2], offset=-1)["output"].append("changed")

    again = _call(block, _frame(3), data=[3], offset=-2)["output"]

    assert again == [1, "changed"]

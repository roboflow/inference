"""Tests for the tensor-data-representation sibling of the frame_delay block.

The sibling reuses the numpy block's buffering machinery verbatim; the only
behavioral change is that image payloads are spilled to host memory at
buffer-insertion time so that CUDA tensors (e.g. Jetson bridge-pool buffers)
are released instead of being pinned for the delay window. The numpy test
module (`test_frame_delay.py`) is flag-agnostic and keeps covering the shared
delay semantics in both flag directions; the tests below cover the tensor
sibling only, hence the `_TENSOR_ONLY` marker.
"""

import datetime
import gc
import weakref

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from inference.core.env import (
    ENABLE_TENSOR_DATA_REPRESENTATION,
    WORKFLOWS_IMAGE_TENSOR_DEVICE,
)
from inference.core.workflows.core_steps.fusion.frame_delay.v1 import (
    MAX_OFFSET,
    MAX_TRACKED_VIDEOS,
)
from inference.core.workflows.core_steps.fusion.frame_delay.v1 import (
    FrameDelayBlockV1 as NumpyFrameDelayBlockV1,
)
from inference.core.workflows.core_steps.fusion.frame_delay.v1_tensor import (
    BlockManifest,
    FrameDelayBlockV1,
    _spill_images_to_host,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    VideoMetadata,
    WorkflowImageData,
)

_TENSOR_ONLY = pytest.mark.skipif(
    not ENABLE_TENSOR_DATA_REPRESENTATION,
    reason="tensor-native variant; runs only with ENABLE_TENSOR_DATA_REPRESENTATION=True",
)


def _video_metadata(frame_number: int, video_id: str = "vid_1") -> VideoMetadata:
    return VideoMetadata(
        video_identifier=video_id,
        frame_number=frame_number,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570800).astimezone(
            tz=datetime.timezone.utc
        ),
        fps=30,
        comes_from_video_file=True,
    )


def _metadata_image(frame_number: int, video_id: str = "vid_1") -> WorkflowImageData:
    """The `image` input of the block: only provides video metadata."""
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="metadata_provider"),
        numpy_image=np.zeros((2, 2, 3), dtype=np.uint8),
        video_metadata=_video_metadata(frame_number=frame_number, video_id=video_id),
    )


def _tensor_born_image(frame_number: int = 0) -> WorkflowImageData:
    """A frame that exists only as a CHW RGB uint8 tensor (R=10, G=20, B=30),
    like the frames handed over by the Jetson tensor bridge."""
    chw = torch.zeros((3, 4, 6), dtype=torch.uint8)
    chw[0].fill_(10)
    chw[1].fill_(20)
    chw[2].fill_(30)
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="tensor_frame"),
        tensor_image=chw,
        video_metadata=_video_metadata(frame_number=frame_number),
    )


@_TENSOR_ONLY
def test_delay_semantics_match_numpy_sibling_for_non_image_payloads() -> None:
    # given
    numpy_block = NumpyFrameDelayBlockV1()
    tensor_block = FrameDelayBlockV1()

    # when - identical monotonic frame sequence on both siblings
    for n in range(6):
        expected = numpy_block.run(image=_metadata_image(n), data=f"det-{n}", offset=-2)
        actual = tensor_block.run(image=_metadata_image(n), data=f"det-{n}", offset=-2)

        # then - result dicts are identical frame by frame
        assert actual == expected


@_TENSOR_ONLY
def test_non_image_payloads_are_buffered_untouched() -> None:
    # given - a payload holding GPU-representable data that is NOT an image;
    # per the block's documented policy it must be stored as-is (delaying
    # mask-carrying predictions still retains their memory)
    block = FrameDelayBlockV1()
    payload = {"mask": torch.ones((2, 3), dtype=torch.bool), "value": 42}

    # when
    result = block.run(image=_metadata_image(0), data=payload, offset=0)

    # then - the very same object is buffered and emitted
    assert result["output"] is payload


@_TENSOR_ONLY
def test_image_payload_is_spilled_to_host_when_buffered() -> None:
    # given
    block = FrameDelayBlockV1()
    image = _tensor_born_image(frame_number=0)

    # when
    result = block.run(image=_metadata_image(0), data=image, offset=0)
    emitted = result["output"]

    # then - the emitted object is a host-resident copy, not the input
    assert isinstance(emitted, WorkflowImageData)
    assert emitted is not image
    assert emitted.is_tensor_materialised() is False
    assert emitted._tensor_image is None
    assert emitted._numpy_image is not None
    # CHW RGB (10, 20, 30) -> HWC BGR (30, 20, 10)
    assert emitted.numpy_image.shape == (4, 6, 3)
    assert np.all(emitted.numpy_image == np.array([30, 20, 10], dtype=np.uint8))
    # lineage and video metadata are carried over exactly
    assert emitted._parent_metadata is image._parent_metadata
    assert emitted._workflow_root_ancestor_metadata is (
        image._workflow_root_ancestor_metadata
    )
    assert emitted._video_metadata is image._video_metadata
    # the buffer holds the spilled copy, and the input image is left untouched
    assert block._buffers["vid_1"][0] is emitted
    assert image.is_tensor_materialised() is True


@_TENSOR_ONLY
def test_spill_releases_the_buffered_tensor_storage() -> None:
    # given - the block buffers the frame, then every external reference to
    # the tensor is dropped (as happens when the bridge frame goes out of
    # scope after the workflow step)
    block = FrameDelayBlockV1()
    tensor = torch.full((3, 4, 6), 7, dtype=torch.uint8)
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="tensor_frame"),
        tensor_image=tensor,
        video_metadata=_video_metadata(frame_number=0),
    )
    stored_tensor = image._tensor_image
    tensor_ref = weakref.ref(stored_tensor)

    # when
    result = block.run(image=_metadata_image(0), data=image, offset=0)
    del image, tensor, stored_tensor
    gc.collect()

    # then - nothing in the block retains the tensor (best-effort refcount
    # check via weakref), and the emitted numpy buffer owns its memory rather
    # than aliasing the (now released) tensor storage
    assert tensor_ref() is None
    assert result["output"].numpy_image.flags["OWNDATA"] is True


@_TENSOR_ONLY
def test_emitted_image_re_materialises_tensor_lazily() -> None:
    # given
    block = FrameDelayBlockV1()
    image = _tensor_born_image(frame_number=0)
    expected_pixels = image._tensor_image.detach().to("cpu").clone()

    # when
    emitted = block.run(image=_metadata_image(0), data=image, offset=0)["output"]

    # then - the tensor is rebuilt lazily, on the configured device, with the
    # exact original pixels (CHW RGB -> HWC BGR -> CHW RGB round trip)
    assert emitted.is_tensor_materialised() is False
    re_uploaded = emitted.tensor_image
    assert emitted.is_tensor_materialised() is True
    assert re_uploaded.device.type == WORKFLOWS_IMAGE_TENSOR_DEVICE.type
    assert torch.equal(re_uploaded.detach().to("cpu"), expected_pixels)


@_TENSOR_ONLY
def test_images_inside_list_payloads_are_spilled() -> None:
    # given - LIST_OF_VALUES payloads (e.g. a collapsed list of crops) may mix
    # images with other values
    block = FrameDelayBlockV1()
    tensor_image = _tensor_born_image(frame_number=0)
    host_image = _metadata_image(0)
    payload = [tensor_image, "det-0", host_image]

    # when
    emitted = block.run(image=_metadata_image(0), data=payload, offset=0)["output"]

    # then - only the tensor-materialised image is replaced by a spilled copy
    assert emitted is not payload
    assert emitted[0] is not tensor_image
    assert emitted[0].is_tensor_materialised() is False
    assert emitted[0]._numpy_image is not None
    assert emitted[1] == "det-0"
    assert emitted[2] is host_image


@_TENSOR_ONLY
def test_containers_without_tensor_images_keep_identity() -> None:
    # given
    host_image = _metadata_image(0)
    payload = [host_image, "det-0", 42]

    # when
    spilled = _spill_images_to_host(data=payload)

    # then - nothing to spill, the original container object is kept
    assert spilled is payload


@_TENSOR_ONLY
def test_images_nested_in_dicts_are_not_spilled() -> None:
    # given - documented policy: only top-level images and images inside
    # list/tuple containers are spilled; dict payloads are opaque
    tensor_image = _tensor_born_image(frame_number=0)
    payload = {"frame": tensor_image}

    # when
    spilled = _spill_images_to_host(data=payload)

    # then
    assert spilled is payload
    assert tensor_image.is_tensor_materialised() is True


@_TENSOR_ONLY
def test_buffer_is_bounded_on_tensor_sibling() -> None:
    # given
    block = FrameDelayBlockV1()
    offset = -3

    # when
    for n in range(500):
        block.run(image=_metadata_image(n), data=f"det-{n}", offset=offset)

    # then - eviction machinery inherited from the numpy sibling is intact
    buffer = block._buffers["vid_1"]
    assert len(buffer) == abs(offset) + 1
    assert sorted(buffer) == [496, 497, 498, 499]


@_TENSOR_ONLY
def test_buffer_is_cleared_when_frame_numbers_restart_on_tensor_sibling() -> None:
    # given
    block = FrameDelayBlockV1()
    for n in range(1000, 1010):
        block.run(image=_metadata_image(n), data=f"old-{n}", offset=-1)

    # when
    result = block.run(image=_metadata_image(0), data="new-0", offset=-1)

    # then
    assert result["is_available"] is False
    assert list(block._buffers["vid_1"]) == [0]


@_TENSOR_ONLY
def test_inactive_video_buffers_are_evicted_on_tensor_sibling() -> None:
    # given
    block = FrameDelayBlockV1()

    # when
    for stream in range(MAX_TRACKED_VIDEOS + 5):
        block.run(
            image=_metadata_image(0, video_id=f"vid_{stream}"), data="det-0", offset=-1
        )

    # then
    assert len(block._buffers) == MAX_TRACKED_VIDEOS
    assert "vid_0" not in block._buffers
    assert f"vid_{MAX_TRACKED_VIDEOS + 4}" in block._buffers


@_TENSOR_ONLY
def test_positive_offset_rejected_at_runtime_on_tensor_sibling() -> None:
    # given
    block = FrameDelayBlockV1()

    # when / then
    with pytest.raises(ValueError):
        block.run(image=_metadata_image(7), data="det-7", offset=10)


@_TENSOR_ONLY
def test_offset_validation_enforced_by_tensor_sibling_manifest() -> None:
    # when / then - the inherited manifest validators are active
    with pytest.raises(ValidationError):
        BlockManifest(
            type="roboflow_core/frame_delay@v1",
            image="$inputs.image",
            data="$steps.model.predictions",
            offset=10,
        )
    with pytest.raises(ValidationError):
        BlockManifest(
            type="roboflow_core/frame_delay@v1",
            image="$inputs.image",
            data="$steps.model.predictions",
            offset=-(MAX_OFFSET + 1),
        )
    manifest = BlockManifest(
        type="roboflow_core/frame_delay@v1",
        name="frame_delay",
        image="$inputs.image",
        data="$steps.model.predictions",
        offset=-1,
    )
    assert manifest.offset == -1


@_TENSOR_ONLY
def test_tensor_sibling_manifest_masquerades_as_the_same_block() -> None:
    # then - same block identity, extended documentation
    schema_extra = BlockManifest.model_config["json_schema_extra"]
    assert schema_extra["name"] == "Frame Delay"
    assert schema_extra["version"] == "v1"
    assert "Tensor Data Representation" in schema_extra["long_description"]

"""Grid geometry constants: uncached CPU, value-keyed CUDA reuse, builder parity."""

from contextlib import nullcontext

import pytest
import torch

from inference_models.entities import ImageDimensions
from inference_models.models.common.roboflow import _grid_geometry_cache as cache
from inference_models.models.common.roboflow.instance_segmentation_grid import (
    build_instance_segmentation_grid,
)
from inference_models.models.common.roboflow.model_packages import (
    PreProcessingMetadata,
    StaticCropOffset,
)

VALUES = dict(padding=[4, 8] * 2, scale=[0.5, 0.5] * 2, offset=[20, 30] * 2)
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(autouse=True)
def empty_cache():
    cache.clear()
    yield
    cache.clear()


def _metadata(**changes):
    metadata = PreProcessingMetadata(
        pad_left=4,
        pad_top=8,
        pad_right=4,
        pad_bottom=8,
        scale_width=0.5,
        scale_height=0.5,
        original_size=ImageDimensions(height=200, width=300),
        size_after_pre_processing=ImageDimensions(height=100, width=160),
        inference_size=ImageDimensions(height=66, width=88),
        static_crop_offset=StaticCropOffset(20, 30, 160, 100),
    )
    return metadata._replace(**changes)


def _expected_boxes(image_bboxes, metadata):
    # The builder's expression before the cache.
    boxes = image_bboxes.detach().to(dtype=torch.float32).clone()
    crop = metadata.static_crop_offset
    boxes.sub_(boxes.new_tensor([metadata.pad_left, metadata.pad_top] * 2))
    boxes.div_(boxes.new_tensor([metadata.scale_width, metadata.scale_height] * 2))
    boxes.add_(boxes.new_tensor([crop.offset_x, crop.offset_y] * 2))
    return boxes


def _build(device, dtype=torch.float32):
    image_bboxes = torch.tensor(
        [[10.25, 12.5, 70.0, 60.75], [-1.0, 0.0, 88.0, 66.0]],
        dtype=dtype,
        device=device,
    )
    masks = torch.rand(2, 33, 44, device=device)
    boxes, grid = build_instance_segmentation_grid(
        image_bboxes=image_bboxes,
        masks=masks,
        metadata=_metadata(),
        score_type="logits",
    )
    return image_bboxes, masks, boxes, grid


def test_cpu_constants_are_new_tensors_and_never_cached():
    boxes = torch.zeros(3, 4, dtype=torch.float64)
    first = cache.geometry_constants(boxes, **VALUES)
    second = cache.geometry_constants(boxes, **VALUES)

    assert first[0] is not second[0]
    assert cache.cached_entries() == 0
    for actual, values in zip(first, VALUES.values()):
        assert actual.dtype == torch.float64
        assert torch.equal(actual, boxes.new_tensor(values))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int64])
def test_cpu_builder_maps_boxes_as_before_and_clones_inputs(dtype):
    image_bboxes, masks, boxes, grid = _build("cpu", dtype)
    original = image_bboxes.clone()

    assert boxes.dtype == torch.float32
    assert torch.equal(boxes, _expected_boxes(image_bboxes, _metadata()))
    assert torch.equal(image_bboxes, original)
    assert boxes.data_ptr() != image_bboxes.data_ptr()
    assert torch.equal(grid.scores, masks)
    assert grid.scores.data_ptr() != masks.data_ptr()
    assert grid.grid_padding_ltrb == (2, 4, 2, 4)
    assert cache.cached_entries() == 0


@pytest.mark.parametrize(
    "changes",
    [
        dict(scale_width=0.0),
        dict(scale_height=float("nan")),
        dict(inference_size=ImageDimensions(height=0, width=88)),
        dict(static_crop_offset=StaticCropOffset(200, 30, 160, 100)),
    ],
)
def test_invalid_geometry_raises_before_constants(changes):
    with pytest.raises(ValueError):
        build_instance_segmentation_grid(
            image_bboxes=torch.zeros(1, 4),
            masks=torch.zeros(1, 33, 44),
            metadata=_metadata(**changes),
            score_type="logits",
        )
    assert cache.cached_entries() == 0


def test_capture_bypasses_the_cache(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    class CudaLike:
        device = torch.device("cuda", 0)

        def new_tensor(self, values):
            return torch.tensor(values)

    first = cache.geometry_constants(CudaLike(), **VALUES)

    assert torch.equal(first[0], torch.tensor(VALUES["padding"]))
    assert cache.cached_entries() == 0


def test_values_key_separates_signed_zero_and_types():
    assert cache._values_key([0.0]) != cache._values_key([-0.0])
    assert cache._values_key([0]) != cache._values_key([0.0])
    assert cache._values_key([0.5, 1]) == cache._values_key([0.5, 1])


@requires_cuda
def test_cuda_hit_returns_the_same_exact_tensors():
    boxes = torch.zeros(3, 4, device="cuda")
    first = cache.geometry_constants(boxes, **VALUES)
    second = cache.geometry_constants(boxes, **VALUES)
    half = cache.geometry_constants(boxes.half(), **VALUES)

    assert second is first
    assert half is not first and half[0].dtype == torch.float16
    assert cache.cached_entries() == 2
    for actual, values in zip(first, VALUES.values()):
        assert torch.equal(actual, boxes.new_tensor(values))


@requires_cuda
def test_cuda_entries_are_bounded_and_rebuilt_exactly(monkeypatch):
    monkeypatch.setattr(cache, "MAX_ENTRIES", 2)
    boxes = torch.zeros(1, 4, device="cuda")
    first = cache.geometry_constants(boxes, [1.0] * 4, [1.0] * 4, [0.0] * 4)
    cache.geometry_constants(boxes, [2.0] * 4, [1.0] * 4, [0.0] * 4)
    cache.geometry_constants(boxes, [3.0] * 4, [1.0] * 4, [0.0] * 4)

    assert cache.cached_entries() == 2
    rebuilt = cache.geometry_constants(boxes, [1.0] * 4, [1.0] * 4, [0.0] * 4)
    assert rebuilt is not first
    assert torch.equal(rebuilt[0], boxes.new_tensor([1.0] * 4))


@requires_cuda
def test_cuda_entry_from_side_stream_is_ordered_for_the_caller():
    boxes = torch.zeros(1, 4, device="cuda")
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        torch.cuda._sleep(50_000_000)
        produced = cache.geometry_constants(boxes, **VALUES)
    consumed = cache.geometry_constants(boxes, **VALUES)
    copy = consumed[2].clone()
    torch.cuda.current_stream().synchronize()

    assert consumed is produced
    assert torch.equal(copy, boxes.new_tensor(VALUES["offset"]))


@requires_cuda
def test_cuda_builder_matches_uncached_expression_on_hits():
    for _ in range(2):
        image_bboxes, _, boxes, _ = _build("cuda")
        assert torch.equal(boxes, _expected_boxes(image_bboxes, _metadata()))
    assert cache.cached_entries() == 1


@requires_cuda
def test_cuda_hit_survives_pending_consumer_eviction(monkeypatch):
    monkeypatch.setattr(cache, "MAX_ENTRIES", 1)
    boxes = torch.zeros(1, 4, device="cuda")
    cache.geometry_constants(boxes, **VALUES)
    consumer = torch.cuda.Stream()
    with torch.cuda.stream(consumer):
        torch.cuda._sleep(50_000_000)
        hit = cache.geometry_constants(boxes, **VALUES)
        copied = hit[0].clone()
        del hit
    cache.geometry_constants(boxes, padding=[9] * 4, scale=[1] * 4, offset=[0] * 4)
    pressure = [torch.full((4,), 77.0, device="cuda") for _ in range(256)]
    consumer.synchronize()
    assert torch.equal(copied, boxes.new_tensor(VALUES["padding"]))
    assert cache.cached_entries() == 1
    del pressure


@requires_cuda
def test_cuda_real_capture_bypasses_existing_entry(monkeypatch):
    boxes = torch.ones(1, 4, device="cuda")
    original = cache.geometry_constants(boxes, **VALUES)
    before = dict(cache._entries)
    # Isolate cache bypass using a capture-safe builder. The unchanged public
    # new_tensor(host-values) expression may itself be unsupported in capture.
    monkeypatch.setattr(
        cache,
        "_build",
        lambda boxes, *values: (boxes.clone(), boxes.clone(), boxes.clone()),
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = cache.geometry_constants(boxes, **VALUES)
    graph.replay()
    torch.cuda.synchronize()
    assert all(t.shape == boxes.shape and torch.equal(t, boxes) for t in captured)
    assert list(cache._entries) == list(before)
    assert all(cache._entries[k] is v for k, v in before.items())
    assert cache.geometry_constants(boxes, **VALUES) is original

"""CUDA parity of the mask and region-effect annotation blocks (optional).

Skipped without CUDA. On CUDA inputs every block must::

    keep its output on the input's device and leave the input unchanged
    equal its CPU output: exact, except crop and pixelate cells within 1 level
        (float32 resampling weights may round differently on the device)
    never wait for the whole device (torch.cuda.synchronize raises here)
    queue its work on the caller's current stream

A CUDA mask view materializes its accurate masks once for all readers; coarse
reads none.
"""

from typing import Any, Callable, Dict, Tuple

import numpy as np
import pytest
import torch
from roboflow_workflows.v2.blocks.visualization.annotations import (
    BlurVisualizationBlock,
    CropVisualizationBlock,
    MaskVisualizationBlock,
    PixelateVisualizationBlock,
    PolygonVisualizationBlock,
    SemanticSegmentationVisualizationBlock,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.masks import SegmentationView

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.semantic_segmentation import (
    SemanticSegmentationResult,
)

from .support import (
    detections,
    grid_view,
    image_of,
    instance_detections,
    pixels_of,
    rle_detections,
    run_v2,
    scene,
    semantic,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

CUDA = "cuda"


def image_on(image: ImageData, device: str) -> ImageData:
    moved = ImageData(
        tensor_image=image.tensor_image.to(device),
        image_id=image.image_id,
        parent=image.parent,
        root=image.root,
    )

    return moved


def view_on(view: SegmentationView, device: str) -> SegmentationView:
    boxes = view.detections
    moved = SegmentationView(
        detections=Detections(
            xyxy=boxes.xyxy.to(device),
            class_id=boxes.class_id.to(device),
            confidence=boxes.confidence.to(device),
            image_metadata=boxes.image_metadata,
        ),
        scores=view.scores.to(device),
        score_type=view.score_type,
        threshold=view.threshold,
        geometry=view.geometry,
        take_ownership=True,
    )

    return moved


def rle_on(image: ImageData, device: str) -> InstanceDetections:
    # RLE masks stay host data; the boxes live on the device.
    encoded = rle_detections(image)
    moved = InstanceDetections(
        xyxy=encoded.xyxy.to(device),
        class_id=encoded.class_id.to(device),
        confidence=encoded.confidence.to(device),
        mask=encoded.mask,
        image_metadata=encoded.image_metadata,
    )

    return moved


def semantic_on(image: ImageData, device: str) -> SemanticSegmentationResult:
    result = semantic(image)
    moved = SemanticSegmentationResult(
        segmentation_map=result.segmentation_map.to(device),
        confidence=result.confidence.to(device),
        image_metadata=result.image_metadata,
    )

    return moved


Factory = Callable[[ImageData, str], Any]
# name: (block, predictions on a device, parameters, allowed level difference)
CASES: Dict[str, Tuple[type, Factory, Dict[str, Any], int]] = {
    "mask dense": (
        MaskVisualizationBlock,
        lambda image, device: instance_detections(image, device=device),
        {"opacity": 0.4},
        0,
    ),
    "mask rle": (MaskVisualizationBlock, rle_on, {}, 0),
    "mask view accurate": (
        MaskVisualizationBlock,
        lambda image, device: view_on(grid_view(image), device),
        {"mask_quality": "accurate"},
        0,
    ),
    "mask view coarse": (
        MaskVisualizationBlock,
        lambda image, device: view_on(grid_view(image), device),
        {"mask_quality": "coarse"},
        0,
    ),
    "polygon dense": (
        PolygonVisualizationBlock,
        lambda image, device: instance_detections(image, device=device),
        {"thickness": 2},
        0,
    ),
    "polygon view coarse": (
        PolygonVisualizationBlock,
        lambda image, device: view_on(grid_view(image), device),
        {"mask_quality": "coarse"},
        0,
    ),
    "blur boxes odd": (
        BlurVisualizationBlock,
        lambda image, device: detections(image, device=device),
        {"kernel_size": 15},
        0,
    ),
    "blur boxes even": (
        BlurVisualizationBlock,
        lambda image, device: detections(image, device=device),
        {"kernel_size": 4, "padding": 2},
        0,
    ),
    "blur masks": (
        BlurVisualizationBlock,
        lambda image, device: instance_detections(image, device=device),
        {"kernel_size": 5, "padding": 3},
        0,
    ),
    "pixelate cells": (
        PixelateVisualizationBlock,
        lambda image, device: detections(image, device=device),
        {"pixel_size": 6},
        1,
    ),
    "pixelate mean fill": (
        PixelateVisualizationBlock,
        lambda image, device: detections(image, device=device),
        {"pixel_size": 40},
        0,
    ),
    "crop scaled": (
        CropVisualizationBlock,
        lambda image, device: detections(image, device=device),
        {"scale_factor": 1.3, "position": "CENTER"},
        1,
    ),
    "crop center of mass": (
        CropVisualizationBlock,
        lambda image, device: instance_detections(image, device=device),
        {"position": "CENTER_OF_MASS"},
        1,
    ),
    "semantic": (
        SemanticSegmentationVisualizationBlock,
        semantic_on,
        {"skip_classes": ["road"]},
        0,
    ),
}


@pytest.fixture
def cpu_image() -> ImageData:
    return image_of(scene())


def cpu_output(name: str, image: ImageData) -> np.ndarray:
    block_class, predictions_on, parameters, _ = CASES[name]
    output = run_v2(
        block_class, image=image, predictions=predictions_on(image, "cpu"), **parameters
    )
    pixels = pixels_of(output)

    return pixels


def assert_within(output: np.ndarray, expected: np.ndarray, *, levels: int) -> None:
    difference = np.abs(output.astype(np.int16) - expected.astype(np.int16))
    assert int(difference.max()) <= levels


@pytest.mark.parametrize("name", list(CASES))
def test_cuda_output_equals_cpu_output(cpu_image: ImageData, name: str) -> None:
    block_class, predictions_on, parameters, levels = CASES[name]
    image = image_on(cpu_image, CUDA)
    before = image.tensor_image.clone()

    output = run_v2(
        block_class,
        image=image,
        predictions=predictions_on(cpu_image, CUDA),
        **parameters,
    )

    assert output.tensor_image.device == image.tensor_image.device
    assert torch.equal(image.tensor_image, before)
    assert_within(pixels_of(output), cpu_output(name, cpu_image), levels=levels)


@pytest.mark.parametrize("name", list(CASES))
def test_cuda_blocks_never_synchronize_the_device(
    cpu_image: ImageData, name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    block_class, predictions_on, parameters, _ = CASES[name]
    image = image_on(cpu_image, CUDA)
    predictions = predictions_on(cpu_image, CUDA)
    torch.cuda.synchronize()

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("an annotation block waited for the whole device")

    monkeypatch.setattr(torch.cuda, "synchronize", refuse)
    output = run_v2(block_class, image=image, predictions=predictions, **parameters)

    assert output.tensor_image.is_cuda


@pytest.mark.parametrize("name", list(CASES))
def test_cuda_blocks_run_on_the_callers_stream(cpu_image: ImageData, name: str) -> None:
    block_class, predictions_on, parameters, levels = CASES[name]
    source = image_on(cpu_image, CUDA)
    predictions = predictions_on(cpu_image, CUDA)
    late_pixels = torch.zeros_like(source.tensor_image)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(side):
        # The pixels arrive late on the caller's stream (PyTorch's own test
        # helper spins it); work queued on another stream would read zeros.
        torch.cuda._sleep(50_000_000)
        late_pixels.copy_(source.tensor_image)
        image = source.with_pixels(late_pixels)
        output = run_v2(block_class, image=image, predictions=predictions, **parameters)
        result = output.tensor_image.clone()
    side.synchronize()

    expected = cpu_output(name, cpu_image)
    assert_within(result.permute(1, 2, 0).cpu().numpy(), expected, levels=levels)


def test_cuda_view_shares_one_accurate_materialization(cpu_image: ImageData) -> None:
    image = image_on(cpu_image, CUDA)
    view = view_on(grid_view(cpu_image), CUDA)
    readers = (
        (MaskVisualizationBlock, {}),
        (PolygonVisualizationBlock, {}),
        (BlurVisualizationBlock, {}),
        (CropVisualizationBlock, {"position": "CENTER_OF_MASS"}),
    )

    for block_class, parameters in readers:
        run_v2(
            block_class,
            image=image,
            predictions=view,
            mask_quality="coarse",
            **parameters,
        )
    assert view.materialization_count == 0
    outputs = [
        run_v2(
            block_class,
            image=image,
            predictions=view,
            mask_quality="accurate",
            **parameters,
        )
        for block_class, parameters in readers
    ]

    assert view.materialization_count == 1
    cpu_view = grid_view(cpu_image)
    for (block_class, parameters), output in zip(readers, outputs):
        expected = run_v2(
            block_class,
            image=cpu_image,
            predictions=cpu_view,
            mask_quality="accurate",
            **parameters,
        )
        levels = 1 if block_class is CropVisualizationBlock else 0
        assert_within(pixels_of(output), pixels_of(expected), levels=levels)


@pytest.mark.parametrize(
    "block_class, parameters",
    [
        (PixelateVisualizationBlock, {}),
        (CropVisualizationBlock, {"position": "TOP_CENTER"}),
    ],
)
def test_cuda_box_only_paths_never_materialize_a_view(
    cpu_image: ImageData, block_class: type, parameters: Dict[str, Any]
) -> None:
    image = image_on(cpu_image, CUDA)
    view = view_on(grid_view(cpu_image), CUDA)

    output = run_v2(block_class, image=image, predictions=view, **parameters)

    assert view.materialization_count == 0
    assert output.tensor_image.is_cuda

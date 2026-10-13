"""Compiled image + segmentation -> ``v2/mask_edge_snap@v1`` -> outputs.

The block is not in ``create_catalogue()`` yet (root registers it), so the
workflow compiles against an explicit ``Catalogue``. Segmentations are workflow
inputs: no model runs here. Each batch element pairs one image with the
segmentation made on it; outputs are compared with a direct block call and with
the latest V1 implementation.
"""

import numpy as np
import pytest
import torch
from roboflow_workflows.core_steps.classical_cv.mask_edge_snap.v1_tensor import (
    refine_masks,
)
from roboflow_workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)
from roboflow_workflows.v2.blocks.image.mask_edge_snap import MaskEdgeSnapBlock
from roboflow_workflows.v2.data.frames import FrameRef, frame_of
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.predictions import INSTANCE_SEGMENTATION_PREDICTION_KIND
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.errors import (
    ParamsValidationError,
    ResolvedParameterError,
    StepExecutionError,
)
from roboflow_workflows.v2.engine.pipelining.options import PipelineOptions
from roboflow_workflows.v2.engine.plan import CompileOptions

from inference_models.models.base.instance_segmentation import InstanceDetections

CATALOGUE = Catalogue(
    [MaskEdgeSnapBlock], kinds=[IMAGE_KIND, INSTANCE_SEGMENTATION_PREDICTION_KIND]
)
SIZE_HW = (96, 128)
TOLERANCE = 10
DEFINITION = {
    "version": "2.0",
    "inputs": [
        {"type": "WorkflowBatchInput", "name": "image", "kind": ["image"]},
        {
            "type": "WorkflowBatchInput",
            "name": "segmentation",
            "kind": ["instance_segmentation_prediction"],
        },
        {"type": "WorkflowParameter", "name": "tolerance", "kind": ["integer"]},
    ],
    "steps": [
        {
            "type": MaskEdgeSnapBlock.type,
            "name": "snap",
            "image": "$inputs.image",
            "segmentation": "$inputs.segmentation",
            "pixel_tolerance": "$inputs.tolerance",
            "min_contour_area": 0,
        }
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "refined",
            "selector": "$steps.snap.refined_segmentation",
        },
        {"type": "JsonField", "name": "edges", "selector": "$steps.snap.edges"},
    ],
}


def _scene(seed: int) -> tuple:
    # A bright square at x 30-79, y 20-59 and a mask four pixels too large.
    # Sobel peaks on both sides of a step; approaching from outside, the
    # nearer dark-side pixel wins, so left and top snap to 29 and 19.
    gray = np.full(SIZE_HW, 25, dtype=np.uint8)
    gray[20:60, 30:80] = 210
    noise = np.random.default_rng(seed).integers(-5, 6, size=SIZE_HW)
    gray = np.clip(gray.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    rgb = np.stack([gray, gray, 255 - gray], axis=-1)
    mask = np.zeros(SIZE_HW, dtype=bool)
    mask[16:64, 26:84] = True

    return rgb, mask, [29.0, 19.0, 80.0, 60.0]


def _image(rgb: np.ndarray, image_id: str) -> ImageData:
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
    image = ImageData.from_tensor(tensor, image_id=image_id)

    return image


def _segmentation(masks, *, image: ImageData) -> InstanceDetections:
    count = len(masks)
    dense = (
        torch.from_numpy(np.stack(masks))
        if count
        else torch.zeros((0, *SIZE_HW), dtype=torch.bool)
    )
    boxes = []
    for mask in masks:
        ys, xs = np.nonzero(mask)
        boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
    predictions = InstanceDetections(
        xyxy=torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        class_id=torch.zeros(count, dtype=torch.long),
        confidence=torch.full((count,), 0.8),
        mask=dense,
        image_metadata={**image.prediction_metadata(), "class_names": {0: "part"}},
        bboxes_metadata=[
            {"detection_id": f"{image.image_id}-{i}"} for i in range(count)
        ],
    )

    return predictions


def _inputs() -> tuple:
    rgb, mask, snapped_box = _scene(seed=0)
    with_rows = _image(rgb, "with-rows")
    without_rows = _image(_scene(seed=1)[0], "without-rows")
    inputs = {
        "image": [with_rows, without_rows],
        "segmentation": [
            _segmentation([mask], image=with_rows),
            _segmentation([], image=without_rows),
        ],
        "tolerance": TOLERANCE,
    }

    return inputs, rgb, snapped_box


def _v1(rgb: np.ndarray, segmentation: InstanceDetections) -> tuple:
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="v1"),
        numpy_image=np.ascontiguousarray(rgb[:, :, ::-1]),
    )
    refined, edges = refine_masks(
        image=image,
        segmentation=segmentation,
        pixel_tolerance=TOLERANCE,
        sigma=1.0,
        min_contour_area=0.0,
        dilation_iterations=2,
        boundary_band_width=15,
        adaptive_window_size=41,
    )

    return refined, edges


def _check(rows, inputs, rgb, snapped_box) -> None:
    assert len(rows) == 2
    first, second = rows
    images = inputs["image"]
    segmentation = inputs["segmentation"][0]
    v1_refined, v1_edges = _v1(rgb, segmentation)

    refined = first["refined"]
    assert refined.xyxy.tolist() == [snapped_box]
    assert refined.mask.dtype == torch.bool
    assert tuple(refined.mask.shape) == (1, *SIZE_HW)
    assert torch.equal(refined.mask, v1_refined.mask)
    assert refined.bboxes_metadata == [{"detection_id": "with-rows-0"}]
    assert frame_of(refined) == FrameRef.from_image(images[0])
    assert torch.equal(first["edges"].mask, v1_edges.mask)
    assert torch.equal(first["edges"].xyxy, v1_edges.xyxy)
    assert frame_of(first["edges"]) == FrameRef.from_image(images[0])

    assert len(second["refined"]) == 0
    assert frame_of(second["refined"]) == FrameRef.from_image(images[1])
    assert frame_of(second["edges"]) == FrameRef.from_image(images[1])


@pytest.mark.parametrize("mode", ["run", "phases"])
def test_compiled_workflow_snaps_masks_per_batch_element(mode):
    plan = compile_workflow(
        DEFINITION, catalogue=CATALOGUE, options=CompileOptions(block_execution=mode)
    )
    session = plan.create_session()
    inputs, rgb, snapped_box = _inputs()
    try:
        result = session.run(inputs)
    finally:
        session.close()

    _check(result.rows(), inputs, rgb, snapped_box)


def test_pipelined_runs_match_and_leave_inputs_unchanged():
    plan = compile_workflow(DEFINITION, catalogue=CATALOGUE)
    session = plan.create_session()
    inputs, rgb, snapped_box = _inputs()
    mask_before = inputs["segmentation"][0].mask.clone()
    try:
        with session.pipeline(options=PipelineOptions(max_in_flight=2)) as pipe:
            futures = [pipe.submit(inputs) for _ in range(2)]
        results = [future.result() for future in futures]
    finally:
        session.close()

    for result in results:
        _check(result.rows(), inputs, rgb, snapped_box)
    assert torch.equal(inputs["segmentation"][0].mask, mask_before)
    assert inputs["segmentation"][0].xyxy.tolist() == [[26.0, 16.0, 84.0, 64.0]]


def test_out_of_range_literal_fails_compilation():
    definition = {
        **DEFINITION,
        "steps": [{**DEFINITION["steps"][0], "boundary_band_width": 0}],
    }

    with pytest.raises(ParamsValidationError, match="boundary_band_width"):
        compile_workflow(definition, catalogue=CATALOGUE)


@pytest.mark.parametrize("tolerance", [0, 101])
@pytest.mark.parametrize("mode", ["run", "phases"])
def test_selected_out_of_range_tolerance_has_resolved_parameter_path(tolerance, mode):
    plan = compile_workflow(
        DEFINITION, catalogue=CATALOGUE, options=CompileOptions(block_execution=mode)
    )
    session = plan.create_session()
    inputs, _, _ = _inputs()
    inputs["tolerance"] = tolerance
    try:
        with pytest.raises(StepExecutionError) as error:
            session.run(inputs)
    finally:
        session.close()
    assert isinstance(error.value.__cause__, ResolvedParameterError)
    assert error.value.__cause__.field_path == ("pixel_tolerance",)


def test_selected_nan_sigma_has_resolved_parameter_path():
    definition = {
        **DEFINITION,
        "inputs": [
            *DEFINITION["inputs"],
            {"type": "WorkflowParameter", "name": "sigma", "kind": ["float"]},
        ],
        "steps": [{**DEFINITION["steps"][0], "sigma": "$inputs.sigma"}],
    }
    session = compile_workflow(definition, catalogue=CATALOGUE).create_session()
    inputs, _, _ = _inputs()
    inputs["sigma"] = float("nan")
    try:
        with pytest.raises(StepExecutionError) as error:
            session.run(inputs)
    finally:
        session.close()
    assert isinstance(error.value.__cause__, ResolvedParameterError)
    assert error.value.__cause__.field_path == ("sigma",)

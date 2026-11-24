"""Real RF-DETR post-processing through the V2 model and mask painter.

Network outputs are synthetic; no model weights or accelerator are needed.
The post-processors, adapter, compiler, phases and painter are production code.
"""

from typing import Any

import pytest
import torch
from roboflow_workflows.v2.blocks.models.roboflow_platform.instance_segmentation_model.v4 import (
    InstanceSegmentationBlock,
)
from roboflow_workflows.v2.blocks.visualization.annotations.mask_visualization.v1 import (
    MaskVisualizationBlock,
)
from roboflow_workflows.v2.data.frames import frame_of
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.masks.grid import INSTANCE_SEGMENTATION_VIEW_KIND
from roboflow_workflows.v2.data.predictions import NATIVE_KINDS
from roboflow_workflows.v2.data.row_identity import row_record
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.errors import StepExecutionError
from roboflow_workflows.v2.engine.plan import CompileOptions

from inference_models.entities import ImageDimensions
from inference_models.models.common.roboflow.model_packages import (
    PreProcessingMetadata,
    StaticCropOffset,
)
from inference_models.models.rfdetr.common import (
    post_process_instance_segmentation_results,
    post_process_instance_segmentation_results_to_grid,
)


class _RFDETRPostprocessor:
    class_names = ("shape",)
    supported_mask_formats = {"grid", "dense"}

    def __init__(self):
        self.formats = []

    def pre_process(self, images, *, input_color_format):
        assert input_color_format == "rgb"
        metadata = []
        for image in images:
            height, width = image.shape[1:]
            metadata.append(
                PreProcessingMetadata(
                    original_size=ImageDimensions(height, width),
                    size_after_pre_processing=ImageDimensions(height, width),
                    inference_size=ImageDimensions(32, 48),
                    pad_left=0,
                    pad_top=0,
                    pad_right=0,
                    pad_bottom=0,
                    scale_width=48 / width,
                    scale_height=32 / height,
                    static_crop_offset=StaticCropOffset(0, 0, width, height),
                )
            )

        return len(images), metadata

    def forward(self, count):
        y, x = torch.meshgrid(torch.arange(8), torch.arange(12), indexing="ij")
        scores = 5.3 - (x - 5.2).square() * 0.3 - (y - 3.3).square() * 0.9
        raw = (
            torch.tensor([[[0.5, 0.5, 0.8, 0.8]]]).repeat(count, 1, 1),
            torch.full((count, 1, 1), 4.0),
            scores[None, None].repeat(count, 1, 1, 1),
        )

        return raw

    def post_process(self, raw, metadata, *, mask_format="dense", confidence=0.3):
        self.formats.append(mask_format)
        postprocess = (
            post_process_instance_segmentation_results_to_grid
            if mask_format == "grid"
            else post_process_instance_segmentation_results
        )
        results = postprocess(
            bboxes=raw[0],
            logits=raw[1],
            masks=raw[2],
            pre_processing_meta=metadata,
            threshold=confidence,
            num_classes=1,
            classes_re_mapping=None,
        )

        return results


def _image():
    image = ImageData.from_tensor(
        torch.full((3, 96, 144), 60, dtype=torch.uint8), image_id="camera-frame"
    )

    return image


def _plan(*, implementation=None, mode="run", dense_output=False):
    definition: dict[str, Any] = {
        "version": "2.0",
        "inputs": [
            {"type": "WorkflowBatchInput", "name": "image", "kind": ["image"]},
            {"type": "WorkflowParameter", "name": "fidelity", "kind": ["string"]},
        ],
        "steps": [
            {
                "type": InstanceSegmentationBlock.type,
                "name": "model",
                "image": "$inputs.image",
                "model_id": "synthetic",
                "confidence": 0.3,
            },
            {
                "type": MaskVisualizationBlock.type,
                "name": "paint",
                "image": "$inputs.image",
                "predictions": "$steps.model.segmentation",
                "mask_quality": "$inputs.fidelity",
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "view",
                "selector": "$steps.model.segmentation",
            },
            {"type": "JsonField", "name": "image", "selector": "$steps.paint.image"},
        ],
    }
    if implementation is not None:
        definition["execution"] = {"step_quality": {"$steps.model": implementation}}
    if dense_output:
        definition["outputs"].append(
            {
                "type": "JsonField",
                "name": "predictions",
                "selector": "$steps.model.predictions",
            }
        )
    plan = compile_workflow(
        definition,
        catalogue=Catalogue(
            [InstanceSegmentationBlock, MaskVisualizationBlock],
            kinds=[*NATIVE_KINDS, INSTANCE_SEGMENTATION_VIEW_KIND],
        ),
        options=CompileOptions(block_execution=mode),
    )

    return plan


@pytest.mark.parametrize("mode", ["run", "phases"])
def test_fast_grid_keeps_fidelity_runtime_selectable(mode):
    model = _RFDETRPostprocessor()
    plan = _plan(mode=mode)
    assert plan.step(("model",)).implementation.name == "fast"
    session = plan.create_session({"models": {"synthetic": model}})

    (coarse,) = session.run({"image": [_image()], "fidelity": "coarse"}).rows()
    assert coarse["view"].scores.shape == (1, 8, 12)
    assert coarse["view"].materialization_count == 0
    assert frame_of(coarse["view"].detections).image_id == "camera-frame"
    assert row_record(coarse["view"].detections).rows == 1

    (accurate,) = session.run({"image": [_image()], "fidelity": "accurate"}).rows()
    assert accurate["view"].materialization_count == 1
    assert model.formats == ["grid", "grid"]
    assert not torch.equal(coarse["image"].tensor_image, accurate["image"].tensor_image)


@pytest.mark.parametrize("mode", ["run", "phases"])
@pytest.mark.parametrize("implementation", ["fast", "accurate"])
def test_both_strategies_honor_accurate_output_and_shared_rows(mode, implementation):
    model = _RFDETRPostprocessor()
    plan = _plan(implementation=implementation, mode=mode, dense_output=True)
    assert plan.step(("model",)).implementation.name == implementation
    (result,) = (
        plan.create_session({"models": {"synthetic": model}})
        .run({"image": [_image()], "fidelity": "accurate"})
        .rows()
    )

    oracle = _RFDETRPostprocessor()
    count, metadata = oracle.pre_process(
        [_image().tensor_image], input_color_format="rgb"
    )
    (expected,) = oracle.post_process(
        oracle.forward(count), metadata, mask_format="dense"
    )
    assert model.formats == ["grid" if implementation == "fast" else "dense"]
    assert torch.equal(result["predictions"].mask, expected.mask)
    assert torch.equal(result["predictions"].xyxy, expected.xyxy)
    assert result["predictions"] is result["view"].full_res()
    assert row_record(result["predictions"]) == row_record(result["view"].detections)


def test_fast_strategy_supports_a_dense_only_model():
    model = _RFDETRPostprocessor()
    model.supported_mask_formats = {"dense"}
    session = _plan(implementation="fast").create_session(
        {"models": {"synthetic": model}}
    )
    (result,) = session.run({"image": [_image()], "fidelity": "coarse"}).rows()

    assert model.formats == ["dense"]
    assert result["view"].is_dense_fallback


def test_grid_failure_is_reported_without_retrying_dense():
    class BrokenGrid(_RFDETRPostprocessor):
        def post_process(self, raw, metadata, *, mask_format="dense", confidence=0.3):
            self.formats.append(mask_format)
            raise ValueError("grid producer failed")

    model = BrokenGrid()
    session = _plan().create_session({"models": {"synthetic": model}})
    with pytest.raises(StepExecutionError, match="grid producer failed"):
        session.run({"image": [_image()], "fidelity": "accurate"})

    assert model.formats == ["grid"]


@pytest.mark.parametrize("mode", ["run", "phases"])
def test_dense_only_model_without_format_parameter_remains_supported(mode):
    class DenseOnly(_RFDETRPostprocessor):
        supported_mask_formats = {"dense"}

        def post_process(self, raw, metadata, *, confidence=0.3):
            return super().post_process(
                raw, metadata, confidence=confidence, mask_format="dense"
            )

    model = DenseOnly()
    session = _plan(mode=mode).create_session({"models": {"synthetic": model}})
    (result,) = session.run({"image": [_image()], "fidelity": "accurate"}).rows()

    assert model.formats == ["dense"]
    assert result["view"].is_dense_fallback

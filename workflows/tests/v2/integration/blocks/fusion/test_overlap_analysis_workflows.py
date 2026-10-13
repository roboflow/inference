"""Overlap inspection through real compiled selector, batch and nested workflows."""

import pytest
import torch
from inference_models.entities import ImageDimensions
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections
from inference_models.models.common.roboflow.model_packages import (
    PreProcessingMetadata,
    StaticCropOffset,
)
from roboflow_workflows.v2.blocks.fusion.overlap_analysis import OverlapAnalysisBlock
from roboflow_workflows.v2.data.batch import EntryMetadata, InputValue, SampleContext
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.masks.grid import MaskGridGeometry, SegmentationView
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.errors import StepExecutionError
from roboflow_workflows.v2.engine.plan import CompileOptions

CATALOGUE = Catalogue([OverlapAnalysisBlock])


def _prediction(boxes, *, segmentation=False):
    image = ImageData.from_tensor(
        torch.zeros((3, 16, 16), dtype=torch.uint8), image_id="reference"
    )
    fields = dict(
        xyxy=torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        class_id=torch.zeros(len(boxes), dtype=torch.long),
        confidence=torch.ones(len(boxes)),
        image_metadata={**image.prediction_metadata(), "class_names": {0: "crate"}},
        bboxes_metadata=[
            {"detection_id": f"row-{i}", "parent_id": "source"}
            for i in range(len(boxes))
        ],
    )
    if segmentation == "grid":
        metadata = PreProcessingMetadata(
            pad_left=0,
            pad_top=0,
            pad_right=0,
            pad_bottom=0,
            original_size=ImageDimensions(16, 16),
            size_after_pre_processing=ImageDimensions(16, 16),
            inference_size=ImageDimensions(16, 16),
            scale_width=1,
            scale_height=1,
            static_crop_offset=StaticCropOffset(0, 0, 16, 16),
        )
        scores = torch.zeros((len(boxes), 8, 8))
        scores[:, :4, :4] = 1
        result = SegmentationView(
            detections=Detections(**fields),
            scores=scores,
            score_type="probabilities",
            threshold=0.5,
            geometry=MaskGridGeometry.from_pre_processing(
                metadata, grid_size_hw=(8, 8)
            ),
        )
    elif segmentation:
        masks = torch.zeros((len(boxes), 16, 16), dtype=torch.bool)
        masks[:, :8, :8] = True
        result = InstanceDetections(**fields, mask=masks)
    else:
        result = Detections(**fields)
    return result


def _definition(*, batch=False, segmentation=False):
    kind = (
        "instance_segmentation_view"
        if segmentation == "grid"
        else (
            "instance_segmentation_prediction"
            if segmentation
            else "object_detection_prediction"
        )
    )
    definition = {
        "version": "2.0",
        "inputs": [
            {
                "type": "WorkflowBatchInput" if batch else "WorkflowParameter",
                "name": name,
                "kind": [kind],
            }
            for name in ("reference", "candidate")
        ]
        + [{"type": "WorkflowParameter", "name": "threshold", "kind": ["float"]}],
        "steps": [
            {
                "type": OverlapAnalysisBlock.type,
                "name": "inspect",
                "reference_predictions": "$inputs.reference",
                "candidate_predictions": "$inputs.candidate",
                "min_overlap": "$inputs.threshold",
            }
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "pairs",
                "selector": "$steps.inspect.overlaps",
            }
        ],
    }
    return definition


@pytest.mark.parametrize("mode", ["run", "phases"])
@pytest.mark.parametrize("segmentation", [False, True, "grid"])
def test_selected_threshold_and_reference_entry_metadata(mode, segmentation):
    plan = compile_workflow(
        _definition(segmentation=segmentation),
        catalogue=CATALOGUE,
        options=CompileOptions(block_execution=mode),
    )
    session = plan.create_session()
    reference = _prediction([[0, 0, 8, 8]], segmentation=segmentation)
    candidate = _prediction([[0, 0, 4, 8]], segmentation=segmentation)
    # For IS both masks are equal despite the different boxes.
    expected = 1.0 if segmentation else 0.5
    inputs = {
        "reference": InputValue(
            reference, EntryMetadata(sample={(): SampleContext("camera-A")})
        ),
        "candidate": candidate,
        "threshold": expected,
    }
    try:
        result = session.run(inputs)
        record = result.rows()[0]["pairs"][0]
        assert record["overlap_ratio"] == expected
        assert record["reference_detection_id"] == "row-0"
        assert record["reference_class"] == "crate"
        assert result.outputs.metadata["pairs"].sample_at(()).source_id == "camera-A"
        native_reference = reference.detections if segmentation == "grid" else reference
        assert native_reference.bboxes_metadata[0]["parent_id"] == "source"
        if segmentation == "grid":
            assert reference.materialization_count == 0
        if not segmentation:
            assert session.run({**inputs, "threshold": 0.5001}).rows()[0]["pairs"] == []
        with pytest.raises(StepExecutionError, match="min_overlap"):
            session.run({**inputs, "threshold": 1.1})
    finally:
        session.close()


@pytest.mark.parametrize("mode", ["run", "phases"])
def test_nested_batched_overlap_preserves_empty_rows_and_dimensionality(mode):
    child = _definition(batch=True)
    outer = {
        "version": "2.0",
        "inputs": child["inputs"],
        "steps": [
            {
                "type": "inner_workflow",
                "name": "nested",
                "workflow_definition": child,
                "parameter_bindings": {
                    item["name"]: f"$inputs.{item['name']}" for item in child["inputs"]
                },
            }
        ],
        "outputs": [
            {"type": "JsonField", "name": "pairs", "selector": "$steps.nested.pairs"}
        ],
    }
    session = compile_workflow(
        outer, catalogue=CATALOGUE, options=CompileOptions(block_execution=mode)
    ).create_session()
    try:
        result = session.run(
            {
                "reference": [_prediction([[0, 0, 8, 8]]), _prediction([])],
                "candidate": [_prediction([[0, 0, 4, 8]]), _prediction([[0, 0, 8, 8]])],
                "threshold": 0.5,
            }
        )
        rows = result.rows()
        assert rows[0]["pairs"][0]["overlap_ratio"] == 0.5
        assert rows[1]["pairs"] == []
        assert len(rows) == 2
    finally:
        session.close()

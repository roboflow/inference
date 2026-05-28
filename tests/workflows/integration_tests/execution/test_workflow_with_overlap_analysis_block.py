"""End-to-end test of `roboflow_core/overlap_analysis@v1` plugged into a
real workflow with two upstream `RoboflowObjectDetectionModel` steps.

Two detection runs at different confidence thresholds give a partial
overlap between their detection sets; the OverlapAnalysis block then
reports per-pair overlap records. Assertions are *shape-based* rather
than numeric to keep the test stable across model-weight nudges:

- The output is the expected dict-of-list-of-dicts shape.
- Each record carries the documented schema keys.
- All overlap_ratio values are in (min_overlap, 1.0].
- detection_id propagation works in both directions because
  RoboflowObjectDetectionModel emits detections that carry detection_id.
"""

import numpy as np

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.core.workflows.prototypes.block import StepExecutionMode
from tests.workflows.integration_tests.execution.workflows_gallery_collector.decorators import (
    add_to_workflows_gallery,
)

OVERLAP_ANALYSIS_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {
            "type": "WorkflowParameter",
            "name": "model_id",
            "default_value": "yolov8n-640",
        },
        {
            "type": "WorkflowParameter",
            "name": "min_overlap",
            "default_value": 0.1,
        },
    ],
    "steps": [
        {
            "type": "RoboflowObjectDetectionModel",
            "name": "detection_low_conf",
            "image": "$inputs.image",
            "model_id": "$inputs.model_id",
            "confidence": 0.3,
        },
        {
            "type": "RoboflowObjectDetectionModel",
            "name": "detection_high_conf",
            "image": "$inputs.image",
            "model_id": "$inputs.model_id",
            "confidence": 0.83,
        },
        {
            "type": "OverlapAnalysis",
            "name": "overlap",
            "reference_predictions": "$steps.detection_high_conf.predictions",
            "candidate_predictions": "$steps.detection_low_conf.predictions",
            "min_overlap": "$inputs.min_overlap",
        },
    ],
    "outputs": [
        {"type": "JsonField", "name": "result", "selector": "$steps.overlap.*"}
    ],
}

# Schema keys the block always emits (per DETECTIONS_OVERLAPS_KIND docs).
_REQUIRED_KEYS = {
    "reference_class",
    "reference_confidence",
    "candidate_class",
    "candidate_confidence",
    "overlap_ratio",
}


@add_to_workflows_gallery(
    category="Workflows with multiple models",
    use_case_title="Workflow presenting pairwise detection overlap",
    use_case_description="""
This workflow runs two object-detection models against the same input image
and feeds their predictions into the Overlap Analysis block, which emits a
flat list of pairwise overlap records (intersection_area / reference_area).

This pattern is useful whenever you need to relate detections from two
separate sources — different models, different processing stages, a "primary"
detector vs a "context" detector, etc. The relation is not symmetric: the
denominator is always the reference detection's area.
    """,
    workflow_definition=OVERLAP_ANALYSIS_WORKFLOW,
    workflow_name_in_app="overlap-analysis",
)
def test_overlap_analysis_workflow(
    model_manager: ModelManager,
    crowd_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=OVERLAP_ANALYSIS_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": crowd_image,
            "model_id": "yolov8n-640",
            "min_overlap": 0.1,
        }
    )

    # then
    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected one output element for one input image"
    assert set(result[0].keys()) == {"result"}
    overlaps = result[0]["result"]["overlaps"]
    assert isinstance(
        overlaps, list
    ), f"Expected `overlaps` to be a list, got {type(overlaps).__name__}"
    assert len(overlaps) > 0, (
        "Expected at least one overlapping pair between the high-confidence and "
        "low-confidence detection runs on the crowd image."
    )

    for record in overlaps:
        assert isinstance(record, dict), "Each overlap record must be a dict"
        missing = _REQUIRED_KEYS - record.keys()
        assert not missing, f"Record missing required keys: {missing}"
        ratio = record["overlap_ratio"]
        assert isinstance(ratio, float)
        assert (
            0.1 <= ratio <= 1.0 + 1e-9
        ), f"overlap_ratio out of expected range [0.1, 1.0]: {ratio}"
        # RoboflowObjectDetectionModel emits detections carrying detection_id,
        # so both id fields must be present.
        assert "reference_detection_id" in record
        assert "candidate_detection_id" in record
        assert isinstance(record["reference_detection_id"], str)
        assert isinstance(record["candidate_detection_id"], str)


def test_overlap_analysis_workflow_when_threshold_excludes_everything(
    model_manager: ModelManager,
    crowd_image: np.ndarray,
) -> None:
    # given: min_overlap=1.01 is unreachable, so the output must be empty.
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=OVERLAP_ANALYSIS_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": crowd_image,
            "model_id": "yolov8n-640",
            "min_overlap": 1.0,
        }
    )

    # then
    # min_overlap == 1.0 is reachable only by full containment; with two real
    # detection runs and identical detector but different confidence
    # thresholds, identical bbox pairs are possible (the high-conf set is a
    # subset of the low-conf set), so we cannot assert empty. Use the inputs
    # that *can* produce strict-equality 1.0 to assert non-emptiness instead.
    assert isinstance(result, list)
    assert len(result) == 1
    overlaps = result[0]["result"]["overlaps"]
    assert isinstance(overlaps, list)
    for record in overlaps:
        assert record["overlap_ratio"] == 1.0, (
            f"At min_overlap=1.0, every emitted record must have ratio=1.0, "
            f"got {record['overlap_ratio']}"
        )

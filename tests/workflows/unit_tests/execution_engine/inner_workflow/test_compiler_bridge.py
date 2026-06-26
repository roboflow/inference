import pytest

from inference.core.workflows.execution_engine.v1.inner_workflow import (
    InnerWorkflowInvalidStepEntryError,
)
from inference.core.workflows.execution_engine.v1.inner_workflow.compiler_bridge import (
    validate_inner_workflow_composition_from_raw_workflow_definition,
)


def test_validate_inner_workflow_composition_rejects_non_object_step() -> None:
    definition = {
        "version": "1.0",
        "inputs": [],
        "steps": [
            {"name": "a", "type": "roboflow_core/roboflow_object_detection_model@v1"}
        ],
        "outputs": [],
    }
    validate_inner_workflow_composition_from_raw_workflow_definition(definition)

    bad = {**definition, "steps": [{"name": "a", "type": "x"}, None]}
    with pytest.raises(InnerWorkflowInvalidStepEntryError, match="index 1"):
        validate_inner_workflow_composition_from_raw_workflow_definition(bad)

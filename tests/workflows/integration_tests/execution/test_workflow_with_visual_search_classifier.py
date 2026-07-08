from unittest import mock

import numpy as np

from inference.core.env import (
    ENABLE_TENSOR_DATA_REPRESENTATION,
    WORKFLOWS_MAX_CONCURRENT_STEPS,
)
from inference.core.workflows.execution_engine.core import ExecutionEngine

# Under ENABLE_TENSOR_DATA_REPRESENTATION the loader swaps in the v1_tensor
# sibling, which binds its own copies of the roboflow_api helpers - patch the
# module that actually runs.
_VISUAL_SEARCH_CLASSIFIER_MODULE = (
    "inference.core.workflows.core_steps.integrations.roboflow."
    "visual_search_classifier.v1_tensor"
    if ENABLE_TENSOR_DATA_REPRESENTATION
    else "inference.core.workflows.core_steps.integrations.roboflow."
    "visual_search_classifier.v1"
)

WORKFLOW_WITH_VISUAL_SEARCH_CLASSIFIER = {
    "version": "1.0",
    "inputs": [{"type": "WorkflowImage", "name": "image"}],
    "steps": [
        {
            "type": "roboflow_core/visual_search_classifier@v1",
            "name": "visual_search_classifier",
            "image": "$inputs.image",
            "target_project": "classification-reference",
        },
        {
            "type": "PropertyDefinition",
            "name": "top_class",
            "data": "$steps.visual_search_classifier.predictions",
            "operations": [
                {"type": "ClassificationPropertyExtract", "property_name": "top_class"}
            ],
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "visual_search_output",
            "selector": "$steps.visual_search_classifier.*",
        },
        {
            "type": "JsonField",
            "name": "top_class",
            "selector": "$steps.top_class.output",
        },
    ],
}


def test_workflow_with_visual_search_classifier_and_property_definition() -> None:
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_VISUAL_SEARCH_CLASSIFIER,
        init_parameters={"workflows_core.api_key": "api-key"},
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    with mock.patch(
        f"{_VISUAL_SEARCH_CLASSIFIER_MODULE}.get_roboflow_workspace",
        return_value="my-workspace",
    ) as workspace_mock, mock.patch(
        f"{_VISUAL_SEARCH_CLASSIFIER_MODULE}.search_project_images_at_roboflow"
    ) as search_mock:
        search_mock.return_value = {
            "results": [
                {
                    "id": "img-1",
                    "url": "https://example.com/reference.jpg",
                    "score": 1.64,
                    "labels": [
                        {"class": "pass", "class_id": 2},
                        {"class": "review", "class_id": 5},
                    ],
                }
            ]
        }

        result = execution_engine.run(
            runtime_parameters={
                "image": np.zeros((8, 12, 3), dtype=np.uint8),
            }
        )

    workspace_mock.assert_called_once_with(api_key="api-key")
    assert result[0]["top_class"] == ["pass", "review"]
    visual_search_output = result[0]["visual_search_output"]
    assert "classification_predictions" not in visual_search_output
    predictions_payload = visual_search_output["predictions"]
    if ENABLE_TENSOR_DATA_REPRESENTATION:
        # In-process the flag ships a native MultiLabelClassificationPrediction;
        # the classification-kind serializer restores the numpy dict shape the
        # assertions below pin (the byte-parity contract).
        from inference.core.workflows.core_steps.common.serializers_tensor import (
            serialise_native_classification,
        )

        predictions_payload = serialise_native_classification(
            prediction=predictions_payload
        )
    assert predictions_payload["predicted_classes"] == [
        "pass",
        "review",
    ]
    assert predictions_payload["predictions"] == {
        "pass": {"class_id": 2, "confidence": 0.82},
        "review": {"class_id": 5, "confidence": 0.82},
    }
    assert predictions_payload["image"] == {"width": 12, "height": 8}

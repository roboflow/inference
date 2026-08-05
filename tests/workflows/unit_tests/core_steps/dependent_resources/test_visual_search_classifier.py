"""
Dependent-resources discovery tests for the Roboflow Visual Search
Classifier block (``roboflow_core/visual_search_classifier@v1``).

Same contract as ``visual_search@v1``: only the ``target_project`` platform
project is declared. The optional ``workspace`` field is NOT emitted as a
dependent resource, whether left unset or explicitly provided.
"""

from inference.core.workflows.core_steps.integrations.roboflow.visual_search_classifier.v1 import (
    BlockManifest as VisualSearchClassifierV1Manifest,
)
from inference.core.workflows.prototypes.block import roboflow_platform_project


def test_visual_search_classifier_v1_declares_only_target_project() -> None:
    manifest = VisualSearchClassifierV1Manifest.model_validate(
        {
            "type": "roboflow_core/visual_search_classifier@v1",
            "name": "classifier",
            "image": "$inputs.image",
            "target_project": "reference-images",
        }
    )

    result = manifest.discover_dependent_resources()

    assert len(result) == 1
    assert result == [roboflow_platform_project(project_url="reference-images")]


def test_visual_search_classifier_v1_ignores_explicit_workspace() -> None:
    manifest = VisualSearchClassifierV1Manifest.model_validate(
        {
            "type": "roboflow_core/visual_search_classifier@v1",
            "name": "classifier",
            "image": "$inputs.image",
            "workspace": "my-workspace",
            "target_project": "reference-images",
        }
    )

    result = manifest.discover_dependent_resources()

    # Single-element list: workspace is not emitted as a dependency.
    assert len(result) == 1
    assert result == [roboflow_platform_project(project_url="reference-images")]


def test_visual_search_classifier_v1_returns_selector_fed_target_project_verbatim() -> (
    None
):
    manifest = VisualSearchClassifierV1Manifest.model_validate(
        {
            "type": "roboflow_core/visual_search_classifier@v1",
            "name": "classifier",
            "image": "$inputs.image",
            "target_project": "$inputs.target_project",
        }
    )

    result = manifest.discover_dependent_resources()

    assert len(result) == 1
    assert result == [roboflow_platform_project(project_url="$inputs.target_project")]

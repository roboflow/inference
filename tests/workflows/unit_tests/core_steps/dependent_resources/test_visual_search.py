"""
Dependent-resources discovery tests for the Roboflow Visual Search block
(``roboflow_core/visual_search@v1``).

The block declares only the ``target_project`` platform project. The
required ``workspace`` field is deliberately NOT emitted as a dependent
resource — it scopes the API call, but the dependency is the project.
"""

from inference.core.workflows.core_steps.integrations.roboflow.visual_search.v1 import (
    BlockManifest as VisualSearchV1Manifest,
)
from inference.core.workflows.prototypes.block import roboflow_platform_project


def test_visual_search_v1_declares_only_target_project() -> None:
    manifest = VisualSearchV1Manifest.model_validate(
        {
            "type": "roboflow_core/visual_search@v1",
            "name": "search",
            "image": "$inputs.image",
            "workspace": "my-workspace",
            "target_project": "reference-images",
        }
    )

    result = manifest.discover_dependent_resources()

    # Single-element list: workspace is not emitted as a dependency.
    assert len(result) == 1
    assert result == [roboflow_platform_project(project_url="reference-images")]


def test_visual_search_v1_returns_selector_fed_target_project_verbatim() -> None:
    manifest = VisualSearchV1Manifest.model_validate(
        {
            "type": "roboflow_core/visual_search@v1",
            "name": "search",
            "image": "$inputs.image",
            "workspace": "$inputs.workspace",
            "target_project": "$inputs.target_project",
        }
    )

    result = manifest.discover_dependent_resources()

    assert len(result) == 1
    assert result == [roboflow_platform_project(project_url="$inputs.target_project")]

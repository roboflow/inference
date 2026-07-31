"""
Dependent-resources discovery tests for the Roboflow Dataset Upload sinks
(``roboflow_core/roboflow_dataset_upload@v1`` and ``@v2``).

Both versions declare exactly one dependency: the ``target_project``
Roboflow platform project — literal values and selectors are returned
verbatim.
"""

from inference.core.workflows.core_steps.sinks.roboflow.dataset_upload.v1 import (
    BlockManifest as DatasetUploadV1Manifest,
)
from inference.core.workflows.core_steps.sinks.roboflow.dataset_upload.v2 import (
    BlockManifest as DatasetUploadV2Manifest,
)
from inference.core.workflows.prototypes.block import roboflow_platform_project


def test_dataset_upload_v1_declares_target_project() -> None:
    manifest = DatasetUploadV1Manifest.model_validate(
        {
            "type": "roboflow_core/roboflow_dataset_upload@v1",
            "name": "sink",
            "images": "$inputs.image",
            "target_project": "my_project",
            "usage_quota_name": "quota-1",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_project(project_url="my_project"),
    ]


def test_dataset_upload_v1_returns_selector_fed_target_project_verbatim() -> None:
    manifest = DatasetUploadV1Manifest.model_validate(
        {
            "type": "roboflow_core/roboflow_dataset_upload@v1",
            "name": "sink",
            "images": "$inputs.image",
            "target_project": "$inputs.target_project",
            "usage_quota_name": "quota-1",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_project(project_url="$inputs.target_project"),
    ]


def test_dataset_upload_v2_declares_target_project() -> None:
    manifest = DatasetUploadV2Manifest.model_validate(
        {
            "type": "roboflow_core/roboflow_dataset_upload@v2",
            "name": "sink",
            "images": "$inputs.image",
            "target_project": "my_dataset",
            "usage_quota_name": "quota-1",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_project(project_url="my_dataset"),
    ]


def test_dataset_upload_v2_returns_selector_fed_target_project_verbatim() -> None:
    manifest = DatasetUploadV2Manifest.model_validate(
        {
            "type": "roboflow_core/roboflow_dataset_upload@v2",
            "name": "sink",
            "images": "$inputs.image",
            "target_project": "$inputs.target_al_dataset",
            "usage_quota_name": "quota-1",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_project(project_url="$inputs.target_al_dataset"),
    ]

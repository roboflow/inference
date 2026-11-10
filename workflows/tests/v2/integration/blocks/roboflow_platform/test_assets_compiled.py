"""Compiler/session checks with explicit local catalogue and fake ports."""

import pytest
import torch
from roboflow_workflows.v2.blocks import create_catalogue
from roboflow_workflows.v2.blocks.roboflow_platform.asset_library_attributes.v1 import AssetLibraryAttributesBlock
from roboflow_workflows.v2.blocks.roboflow_platform.roboflow_dataset_upload.v2 import DatasetUploadBlock
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.secrets import Secret
from roboflow_workflows.v2.engine.compilation import compile_workflow


class Platform:
    def __init__(self):
        self.updates = []
        self.uploads = []

    def get_workspace(self, *, api_key):
        assert isinstance(api_key, Secret)
        return "workspace"

    def update_attributes(self, **arguments):
        self.updates.append(arguments["updates"])

    def upload(self, request, *, api_key):
        assert isinstance(api_key, Secret)
        self.uploads.append(request)
        return False, "Successfully registered image"


def plan():
    catalogue = create_catalogue().with_blocks(
        [AssetLibraryAttributesBlock, DatasetUploadBlock]
    )
    definition = {
        "version": "2.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
            {"type": "WorkflowParameter", "name": "source_id"},
            {"type": "WorkflowParameter", "name": "label"},
        ],
        "steps": [
            {
                "type": "roboflow_core/asset_library_attributes@v1",
                "name": "attributes",
                "source_id": "$inputs.source_id",
                "metadata": {"label": "$inputs.label"},
                "tags": ["collected", "$inputs.label"],
            },
            {
                "type": "roboflow_core/roboflow_dataset_upload@v2",
                "name": "collect",
                "images": "$inputs.image",
                "target_project": "project",
                "usage_quota_name": "shared",
                "image_name": "$inputs.source_id",
                "registration_tags": ["collected", "$inputs.label"],
                "fire_and_forget": False,
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "attributes",
                "selector": "$steps.attributes.message",
            },
            {
                "type": "JsonField",
                "name": "upload",
                "selector": "$steps.collect.message",
            },
        ],
    }
    return compile_workflow(definition, catalogue=catalogue)


def test_compiled_scalar_broadcast_and_batched_images():
    platform = Platform()
    with plan().create_session(
        resources={
            "api_key": "invented",
            "asset_library_port": platform,
            "dataset_upload_port": platform,
        }
    ) as session:
        images = [
            ImageData.from_tensor(
                torch.zeros((3, 7, 11), dtype=torch.uint8), image_id=str(index)
            )
            for index in range(2)
        ]
        rows = session.run(
            {"image": images, "source_id": "source", "label": "red"}
        ).rows()
    assert len(rows) == 2
    assert all(
        row["attributes"] == "Image attributes update submitted"
        and row["upload"] == "Successfully registered image"
        for row in rows
    )
    assert platform.updates == [
        [
            {
                "imageId": "source",
                "metadata": {"label": "red"},
                "addTags": ["collected", "red"],
            }
        ]
    ]
    assert [request.image_name for request in platform.uploads] == ["source", "source"]


def test_compiled_policy_disables_every_port_call_without_credential():
    platform = Platform()
    image = ImageData.from_tensor(torch.zeros((3, 7, 11), dtype=torch.uint8))
    with plan().create_session(
        resources={
            "disable_sinks": True,
            "asset_library_port": platform,
            "dataset_upload_port": platform,
        }
    ) as session:
        rows = session.run(
            {"image": [image], "source_id": "source", "label": "red"}
        ).rows()
    assert rows == [
        {
            "attributes": "Sink was disabled by workflow execution policy",
            "upload": "Sink was disabled by workflow execution policy",
        }
    ]
    assert not platform.updates and not platform.uploads

from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pytest

from inference.core.cache import MemoryCache
from inference.core.workflows.core_steps.sinks.roboflow.edit_image_metadata import v1
from inference.core.workflows.core_steps.sinks.roboflow.edit_image_metadata.v1 import (
    SINGLE_UPDATE_SUCCESS_MESSAGE,
    SKIPPED_EMPTY_UPDATE_MESSAGE,
    BlockManifest,
    EditImageMetadataBlockV1,
    build_effective_updates,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    ImageParentMetadata,
    WorkflowImageData,
)


def _workflow_image() -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        numpy_image=np.zeros((32, 32, 3), dtype=np.uint8),
    )


def test_manifest_parsing_valid() -> None:
    raw_manifest = {
        "type": "roboflow_core/edit_image_metadata@v1",
        "name": "edit_metadata",
        "images": "$inputs.image",
        "source_id": "$inputs.source_id",
        "metadata": "$inputs.metadata",
        "tags": "$inputs.tags",
        "disable_sink": "$inputs.disable_sink",
    }

    manifest = BlockManifest.model_validate(raw_manifest)

    assert manifest.images == "$inputs.image"
    assert manifest.source_id == "$inputs.source_id"
    assert manifest.metadata == "$inputs.metadata"
    assert manifest.tags == "$inputs.tags"
    assert manifest.disable_sink == "$inputs.disable_sink"
    assert BlockManifest.get_parameters_accepting_batches() == [
        "images",
        "source_id",
        "metadata",
        "tags",
    ]


def test_build_effective_updates_skips_empty_rows_and_merges_duplicate_source_ids() -> (
    None
):
    updates, source_id_to_result_indices, results = build_effective_updates(
        images_count=4,
        source_ids=Batch(
            content=["img-1", "img-2", "img-1", "img-3"],
            indices=[(0,), (1,), (2,), (3,)],
        ),
        metadata=Batch(
            content=[
                {"color": "red", "score": 0.1},
                None,
                {"color": "blue"},
                {},
            ],
            indices=[(0,), (1,), (2,), (3,)],
        ),
        tags=Batch(
            content=[
                ["shared", "first"],
                None,
                ["shared", "last"],
                [],
            ],
            indices=[(0,), (1,), (2,), (3,)],
        ),
    )

    assert updates == [
        {
            "imageId": "img-1",
            "metadata": {"color": "blue", "score": 0.1},
            "addTags": ["first", "shared", "last"],
        }
    ]
    assert source_id_to_result_indices == {"img-1": [0, 2]}
    assert results == [
        None,
        {"error_status": False, "message": SKIPPED_EMPTY_UPDATE_MESSAGE},
        None,
        {"error_status": False, "message": SKIPPED_EMPTY_UPDATE_MESSAGE},
    ]


def test_run_when_api_key_is_not_specified() -> None:
    block = EditImageMetadataBlockV1(cache=MemoryCache(), api_key=None)

    with pytest.raises(ValueError):
        block.run(
            images=Batch(content=[_workflow_image()], indices=[(0,)]),
            source_id=Batch(content=["img-1"], indices=[(0,)]),
            metadata=Batch(content=[{"color": "red"}], indices=[(0,)]),
        )


def test_run_when_disabled() -> None:
    block = EditImageMetadataBlockV1(cache=MemoryCache(), api_key="my_api_key")
    image = _workflow_image()

    result = block.run(
        images=Batch(content=[image, image], indices=[(0,), (1,)]),
        source_id=Batch(content=["img-1", "img-2"], indices=[(0,), (1,)]),
        disable_sink=True,
    )

    assert result == [
        {
            "error_status": False,
            "message": "Sink was disabled by parameter `disable_sink`",
        },
        {
            "error_status": False,
            "message": "Sink was disabled by parameter `disable_sink`",
        },
    ]


@mock.patch.object(v1, "get_workspace_name")
@mock.patch.object(v1, "update_image_metadata_at_roboflow")
def test_run_single_effective_update_calls_single_endpoint(
    update_image_metadata_at_roboflow_mock: MagicMock,
    get_workspace_name_mock: MagicMock,
) -> None:
    get_workspace_name_mock.return_value = "my-workspace"
    update_image_metadata_at_roboflow_mock.return_value = {"success": True}
    block = EditImageMetadataBlockV1(cache=MemoryCache(), api_key="my_api_key")

    result = block.run(
        images=Batch(content=[_workflow_image()], indices=[(0,)]),
        source_id=Batch(content=["img-1"], indices=[(0,)]),
        metadata=Batch(content=[{"color": "red"}], indices=[(0,)]),
        tags=Batch(content=[["auto"]], indices=[(0,)]),
    )

    assert result == [{"error_status": False, "message": SINGLE_UPDATE_SUCCESS_MESSAGE}]
    update_image_metadata_at_roboflow_mock.assert_called_once_with(
        api_key="my_api_key",
        workspace_id="my-workspace",
        image_id="img-1",
        metadata={"color": "red"},
        add_tags=["auto"],
    )


@mock.patch.object(v1, "get_workspace_name")
@mock.patch.object(v1, "update_image_metadata_at_roboflow")
def test_run_single_endpoint_error_returns_per_image_error(
    update_image_metadata_at_roboflow_mock: MagicMock,
    get_workspace_name_mock: MagicMock,
) -> None:
    get_workspace_name_mock.return_value = "my-workspace"
    update_image_metadata_at_roboflow_mock.side_effect = Exception("API error")
    block = EditImageMetadataBlockV1(cache=MemoryCache(), api_key="my_api_key")

    result = block.run(
        images=Batch(content=[_workflow_image()], indices=[(0,)]),
        source_id=Batch(content=["img-1"], indices=[(0,)]),
        metadata=Batch(content=[{"color": "red"}], indices=[(0,)]),
    )

    assert result[0]["error_status"] is True
    assert "API error" in result[0]["message"]


@mock.patch.object(v1, "get_workspace_name")
@mock.patch.object(v1, "batch_update_image_metadata_at_roboflow")
def test_run_batch_endpoint_uses_merged_updates_and_preserves_input_order(
    batch_update_image_metadata_at_roboflow_mock: MagicMock,
    get_workspace_name_mock: MagicMock,
) -> None:
    get_workspace_name_mock.return_value = "my-workspace"
    batch_update_image_metadata_at_roboflow_mock.return_value = {"taskId": "task-123"}
    block = EditImageMetadataBlockV1(cache=MemoryCache(), api_key="my_api_key")
    image = _workflow_image()

    result = block.run(
        images=Batch(
            content=[image, image, image, image], indices=[(0,), (1,), (2,), (3,)]
        ),
        source_id=Batch(
            content=["img-1", "img-2", "img-1", "img-3"],
            indices=[(0,), (1,), (2,), (3,)],
        ),
        metadata=Batch(
            content=[{"color": "red"}, {"score": 0.9}, {"color": "blue"}, None],
            indices=[(0,), (1,), (2,), (3,)],
        ),
        tags=Batch(
            content=[["a", "x"], ["b"], ["a", "c"], None],
            indices=[(0,), (1,), (2,), (3,)],
        ),
    )

    submitted = {"error_status": False, "message": "Submitted as async task task-123"}
    assert result == [
        submitted,
        submitted,
        submitted,
        {"error_status": False, "message": SKIPPED_EMPTY_UPDATE_MESSAGE},
    ]
    batch_update_image_metadata_at_roboflow_mock.assert_called_once_with(
        api_key="my_api_key",
        workspace_id="my-workspace",
        updates=[
            {
                "imageId": "img-1",
                "metadata": {"color": "blue"},
                "addTags": ["x", "a", "c"],
            },
            {"imageId": "img-2", "metadata": {"score": 0.9}, "addTags": ["b"]},
        ],
    )


@mock.patch.object(v1, "get_workspace_name")
@mock.patch.object(v1, "batch_update_image_metadata_at_roboflow")
def test_run_batch_endpoint_error_raises(
    batch_update_image_metadata_at_roboflow_mock: MagicMock,
    get_workspace_name_mock: MagicMock,
) -> None:
    get_workspace_name_mock.return_value = "my-workspace"
    batch_update_image_metadata_at_roboflow_mock.side_effect = Exception(
        "preflight error"
    )
    block = EditImageMetadataBlockV1(cache=MemoryCache(), api_key="my_api_key")
    image = _workflow_image()

    with pytest.raises(Exception, match="preflight error"):
        block.run(
            images=Batch(content=[image, image], indices=[(0,), (1,)]),
            source_id=Batch(content=["img-1", "img-2"], indices=[(0,), (1,)]),
            metadata=Batch(content=[{"a": 1}, {"b": 2}], indices=[(0,), (1,)]),
        )


@mock.patch.object(v1, "get_workspace_name")
def test_run_raises_when_effective_update_count_exceeds_batch_limit(
    get_workspace_name_mock: MagicMock,
) -> None:
    get_workspace_name_mock.return_value = "my-workspace"
    block = EditImageMetadataBlockV1(cache=MemoryCache(), api_key="my_api_key")
    image = _workflow_image()
    count = v1.MAX_BATCH_UPDATES + 1
    indices = [(i,) for i in range(count)]

    with pytest.raises(ValueError, match="at most 1000 updates"):
        block.run(
            images=Batch(content=[image] * count, indices=indices),
            source_id=Batch(
                content=[f"img-{i}" for i in range(count)], indices=indices
            ),
            metadata=Batch(
                content=[{"value": i} for i in range(count)], indices=indices
            ),
        )

from types import SimpleNamespace
from typing import Any, List
from unittest import mock

import pytest

from inference.core.cache import MemoryCache
from inference.core.workflows.core_steps.sinks.roboflow.asset_library_attributes import (
    v1,
)
from inference.core.workflows.core_steps.sinks.roboflow.asset_library_attributes.v1 import (
    SKIPPED_EMPTY_UPDATE_MESSAGE,
    UPDATE_SUCCESS_MESSAGE,
    BlockManifest,
    RoboflowAssetLibraryAttributesBlockV1,
    _extract_response_body,
    _format_api_error,
    _normalize_to_per_row,
    build_effective_updates,
)
from inference.core.workflows.execution_engine.entities.base import Batch


def make_batch(content: List[Any]) -> Batch:
    return Batch(content=content, indices=[(i,) for i in range(len(content))])


@pytest.fixture
def block() -> RoboflowAssetLibraryAttributesBlockV1:
    return RoboflowAssetLibraryAttributesBlockV1(
        cache=MemoryCache(),
        api_key="my_api_key",
        update_attributes_offloader=None,
    )


@pytest.fixture
def mocked_v1():
    with (
        mock.patch.object(v1, "get_workspace_name") as workspace_mock,
        mock.patch.object(v1, "batch_update_image_metadata_at_roboflow") as batch_mock,
        mock.patch.object(
            v1, "update_image_metadata_at_roboflow"
        ) as single_mock,
    ):
        workspace_mock.return_value = "my-workspace"
        batch_mock.return_value = {"taskId": "task-123"}
        single_mock.return_value = {"status": "ok"}
        yield SimpleNamespace(
            workspace=workspace_mock,
            update_batch=batch_mock,
            update_single=single_mock,
        )


def test_manifest_parsing_valid() -> None:
    raw_manifest = {
        "type": "roboflow_core/asset_library_attributes@v1",
        "name": "edit_metadata",
        "source_id": "$inputs.source_id",
        "metadata": {"location": "$inputs.location", "static": "abc"},
        "tags": ["static-tag", "$inputs.dynamic_tag"],
        "disable_sink": "$inputs.disable_sink",
    }

    manifest = BlockManifest.model_validate(raw_manifest)

    assert manifest.source_id == "$inputs.source_id"
    assert manifest.metadata == {"location": "$inputs.location", "static": "abc"}
    assert manifest.tags == ["static-tag", "$inputs.dynamic_tag"]
    assert manifest.disable_sink == "$inputs.disable_sink"
    assert BlockManifest.get_parameters_accepting_batches() == ["source_id"]


def test_build_effective_updates_broadcasts_scalar_metadata_and_tags_and_merges_duplicates() -> (
    None
):
    effective = build_effective_updates(
        source_ids=make_batch(["img-1", "img-2", "img-1"]),
        metadata={"color": "red", "score": 0.1},
        tags=["shared", "first"],
    )

    assert effective.updates == [
        {
            "imageId": "img-1",
            "metadata": {"color": "red", "score": 0.1},
            "addTags": ["shared", "first"],
        },
        {
            "imageId": "img-2",
            "metadata": {"color": "red", "score": 0.1},
            "addTags": ["shared", "first"],
        },
    ]
    assert effective.result_indices_by_id == {"img-1": [0, 2], "img-2": [1]}
    assert effective.results == [None, None, None]


def test_build_effective_updates_skips_rows_when_metadata_and_tags_are_empty() -> None:
    effective = build_effective_updates(
        source_ids=make_batch(["img-1", "img-2"]),
        metadata=None,
        tags=None,
    )

    assert effective.updates == []
    assert effective.result_indices_by_id == {}
    assert effective.results == [
        {"error_status": False, "message": SKIPPED_EMPTY_UPDATE_MESSAGE},
        {"error_status": False, "message": SKIPPED_EMPTY_UPDATE_MESSAGE},
    ]


def test_run_when_api_key_is_not_specified() -> None:
    block_no_key = RoboflowAssetLibraryAttributesBlockV1(
        cache=MemoryCache(),
        api_key=None,
        update_attributes_offloader=None,
    )

    with pytest.raises(ValueError):
        block_no_key.run(
            source_id=make_batch(["img-1"]),
            metadata={"color": "red"},
        )


def test_run_when_disabled(block: RoboflowAssetLibraryAttributesBlockV1) -> None:
    result = block.run(source_id=make_batch(["img-1", "img-2"]), disable_sink=True)

    disabled = {
        "error_status": False,
        "message": "Sink was disabled by parameter `disable_sink`",
    }
    assert result == [disabled, disabled]


def test_run_single_effective_update_calls_single_image_endpoint(
    block: RoboflowAssetLibraryAttributesBlockV1, mocked_v1
) -> None:
    result = block.run(
        source_id=make_batch(["img-1"]),
        metadata={"color": "red"},
        tags=["auto"],
    )

    assert result == [{"error_status": False, "message": UPDATE_SUCCESS_MESSAGE}]
    mocked_v1.update_single.assert_called_once_with(
        api_key="my_api_key",
        workspace_id="my-workspace",
        image_id="img-1",
        metadata={"color": "red"},
        add_tags=["auto"],
    )
    mocked_v1.update_batch.assert_not_called()


def test_run_single_effective_update_error_returns_per_image_error(
    block: RoboflowAssetLibraryAttributesBlockV1, mocked_v1
) -> None:
    mocked_v1.update_single.side_effect = Exception("API error")

    result = block.run(
        source_id=make_batch(["img-1"]),
        metadata={"color": "red"},
    )

    assert result[0]["error_status"] is True
    assert "API error" in result[0]["message"]


def test_run_batch_endpoint_broadcasts_shared_metadata_and_dedupes_source_ids(
    block: RoboflowAssetLibraryAttributesBlockV1, mocked_v1
) -> None:
    result = block.run(
        source_id=make_batch(["img-1", "img-2", "img-1", "img-3"]),
        metadata={"color": "red"},
        tags=["a", "x"],
    )

    submitted = {"error_status": False, "message": UPDATE_SUCCESS_MESSAGE}
    assert result == [submitted, submitted, submitted, submitted]
    mocked_v1.update_batch.assert_called_once_with(
        api_key="my_api_key",
        workspace_id="my-workspace",
        updates=[
            {"imageId": "img-1", "metadata": {"color": "red"}, "addTags": ["a", "x"]},
            {"imageId": "img-2", "metadata": {"color": "red"}, "addTags": ["a", "x"]},
            {"imageId": "img-3", "metadata": {"color": "red"}, "addTags": ["a", "x"]},
        ],
    )


def test_run_batch_endpoint_error_returns_per_image_error(
    block: RoboflowAssetLibraryAttributesBlockV1, mocked_v1
) -> None:
    mocked_v1.update_batch.side_effect = Exception("preflight error")

    result = block.run(
        source_id=make_batch(["img-1", "img-2"]),
        metadata={"a": 1},
    )

    assert result[0]["error_status"] is True
    assert result[1]["error_status"] is True
    assert "preflight error" in result[0]["message"]
    assert "preflight error" in result[1]["message"]


def test_error_message_includes_response_body_from_http_error(
    block: RoboflowAssetLibraryAttributesBlockV1, mocked_v1
) -> None:
    """When the API returns a 400 with a JSON body, the error message should include it."""
    fake_response = mock.MagicMock()
    fake_response.json.return_value = {"error": "tags must be a list of strings"}
    http_error = Exception("400 Bad Request")
    http_error.response = fake_response
    mocked_v1.update_single.side_effect = http_error

    result = block.run(
        source_id=make_batch(["img-1"]),
        metadata={"color": "red"},
    )

    assert result[0]["error_status"] is True
    assert "tags must be a list of strings" in result[0]["message"]


def test_extract_response_body_from_chained_cause() -> None:
    fake_response = mock.MagicMock()
    fake_response.json.return_value = {"detail": "invalid field"}
    inner = Exception("HTTP 400")
    inner.response = fake_response
    outer = RuntimeError("wrapped")
    outer.__cause__ = inner

    body = _extract_response_body(outer)
    assert "invalid field" in body


def test_format_api_error_without_response_body() -> None:
    msg = _format_api_error("Something failed", ValueError("bad input"))
    assert "ValueError" in msg
    assert "bad input" in msg
    assert "Response body" not in msg


def test_run_raises_when_effective_update_count_exceeds_batch_limit(
    block: RoboflowAssetLibraryAttributesBlockV1, mocked_v1
) -> None:
    count = v1.MAX_BATCH_UPDATES + 1

    with pytest.raises(ValueError, match="at most 1000 updates"):
        block.run(
            source_id=make_batch([f"img-{i}" for i in range(count)]),
            metadata={"value": "x"},
        )


def test_normalize_to_per_row_none() -> None:
    assert _normalize_to_per_row(None, 3) == [None, None, None]


def test_normalize_to_per_row_batch() -> None:
    assert _normalize_to_per_row(make_batch(["a", "b"]), 2) == ["a", "b"]


def test_normalize_to_per_row_plain_scalar_broadcasts() -> None:
    d = {"color": "red"}
    result = _normalize_to_per_row(d, 2)
    assert result == [d, d]


def test_normalize_to_per_row_dict_with_batch_values() -> None:
    result = _normalize_to_per_row(
        {"color": make_batch(["red", "blue"]), "static": "abc"}, 2
    )
    assert result == [
        {"color": "red", "static": "abc"},
        {"color": "blue", "static": "abc"},
    ]


def test_normalize_to_per_row_list_with_batch_values() -> None:
    result = _normalize_to_per_row(
        ["static-tag", make_batch(["label-a", "label-b"])], 2
    )
    assert result == [
        ["static-tag", "label-a"],
        ["static-tag", "label-b"],
    ]


def test_build_effective_updates_resolves_batch_values_in_metadata_and_tags() -> None:
    effective = build_effective_updates(
        source_ids=make_batch(["img-1", "img-2"]),
        metadata={"score": make_batch([0.9, 0.3])},
        tags=[make_batch(["cat", "dog"])],
    )

    assert effective.updates == [
        {"imageId": "img-1", "metadata": {"score": 0.9}, "addTags": ["cat"]},
        {"imageId": "img-2", "metadata": {"score": 0.3}, "addTags": ["dog"]},
    ]


def test_run_uses_injected_offloader_instead_of_calling_api(mocked_v1) -> None:
    offloader = mock.MagicMock(
        return_value={"error_status": False, "message": "queued"}
    )
    block = RoboflowAssetLibraryAttributesBlockV1(
        cache=MemoryCache(),
        api_key="my_api_key",
        update_attributes_offloader=offloader,
    )

    result = block.run(
        source_id=make_batch(["img-1", "img-2"]),
        metadata={"color": "red"},
        tags=["a"],
    )

    offloader.assert_called_once_with(
        workspace_id="my-workspace",
        updates=[
            {"imageId": "img-1", "metadata": {"color": "red"}, "addTags": ["a"]},
            {"imageId": "img-2", "metadata": {"color": "red"}, "addTags": ["a"]},
        ],
        api_key="my_api_key",
    )
    mocked_v1.update_batch.assert_not_called()
    assert result == [
        {"error_status": False, "message": "queued"},
        {"error_status": False, "message": "queued"},
    ]

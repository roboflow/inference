import hashlib
from unittest import mock
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi import BackgroundTasks

from inference.core.cache import MemoryCache
from inference.core.workflows.core_steps.sinks.roboflow.roboflow_custom_metadata import (
    RoboflowCustomMetadataBlock,
    get_workspace_name,
    add_custom_metadata_request,
)


def test_get_workspace_name_when_cache_contains_workspace_name() -> None:
    # given
    api_key = "my_api_key"
    api_key_hash = hashlib.md5(api_key.encode("utf-8")).hexdigest()
    expected_cache_key = f"workflows:api_key_to_workspace:{api_key_hash}"
    cache = MemoryCache()
    cache.set(key=expected_cache_key, value="my_workspace")

    # when
    result = get_workspace_name(api_key=api_key, cache=cache)

    # then
    assert (
        result == "my_workspace"
    ), "Expected return value from the cache to be returned"


@patch(
    "inference.core.workflows.core_steps.sinks.roboflow.roboflow_custom_metadata.get_roboflow_workspace"
)
def test_get_workspace_name_when_cache_does_not_contain_workspace_name(
    get_roboflow_workspace_mock: MagicMock,
) -> None:
    # given
    api_key = "my_api_key"
    cache = MemoryCache()
    api_key_hash = hashlib.md5(api_key.encode("utf-8")).hexdigest()
    expected_cache_key = f"workflows:api_key_to_workspace:{api_key_hash}"
    get_roboflow_workspace_mock.return_value = "workspace_from_api"

    # when
    result = get_workspace_name(api_key=api_key, cache=cache)

    # then
    assert (
        result == "workspace_from_api"
    ), "Expected return value from the API to be returned"
    assert (
        cache.get(expected_cache_key) == "workspace_from_api"
    ), "Expected retrieved workspace to be saved in cache"


@patch(
    "inference.core.workflows.core_steps.sinks.roboflow.roboflow_custom_metadata.add_custom_metadata"
)
def test_add_custom_metadata_request_success(
    add_custom_metadata_mock: MagicMock,
) -> None:
    # given
    add_custom_metadata_mock.return_value = True
    cache = MemoryCache()
    api_key = "my_api_key"
    inference_ids = np.array(["id1", "id2"])
    field_name = "location"
    field_value = "toronto"

    # when
    result = add_custom_metadata_request(
        cache=cache,
        api_key=api_key,
        inference_ids=inference_ids,
        field_name=field_name,
        field_value=field_value,
    )

    # then
    assert result is True, "Expected metadata to be added successfully"


@patch(
    "inference.core.workflows.core_steps.sinks.roboflow.roboflow_custom_metadata.add_custom_metadata"
)
def test_add_custom_metadata_request_failure(
    add_custom_metadata_mock: MagicMock,
) -> None:
    # given
    add_custom_metadata_mock.side_effect = Exception("API error")
    cache = MemoryCache()
    api_key = "my_api_key"
    inference_ids = np.array(["id1", "id2"])
    field_name = "location"
    field_value = "toronto"

    # when
    result = add_custom_metadata_request(
        cache=cache,
        api_key=api_key,
        inference_ids=inference_ids,
        field_name=field_name,
        field_value=field_value,
    )

    # then
    assert result is False, "Expected metadata addition to fail"


@pytest.mark.asyncio
async def test_run_when_api_key_is_not_specified() -> None:
    # given
    block = RoboflowCustomMetadataBlock(
        cache=MemoryCache(),
        background_tasks=None,
        api_key=None,
    )

    # when
    with pytest.raises(ValueError):
        _ = await block.run(
            fire_and_forget=True,
            field_name="location",
            field_value=["toronto"],
            predictions=[{"inference_id": np.array(["id1"])}],
        )


@pytest.mark.asyncio
async def test_run_when_no_inference_ids() -> None:
    # given
    block = RoboflowCustomMetadataBlock(
        cache=MemoryCache(),
        background_tasks=None,
        api_key="my_api_key",
    )

    # when
    result = await block.run(
        fire_and_forget=True,
        field_name="location",
        field_value=["toronto"],
        predictions=[],
    )

    # then
    assert result == [
        {
            "error_status": True,
            "predictions": [],
            "message": "Custom metadata upload failed because no inference_ids were received",
        }
    ], "Expected failure due to no inference_ids"


@pytest.mark.asyncio
async def test_run_when_no_field_name() -> None:
    # given
    block = RoboflowCustomMetadataBlock(
        cache=MemoryCache(),
        background_tasks=None,
        api_key="my_api_key",
    )

    # when
    result = await block.run(
        fire_and_forget=True,
        field_name=None,
        field_value=["toronto"],
        predictions=[{"inference_id": np.array(["id1"])}],
    )

    # then
    assert result == [
        {
            "error_status": True,
            "predictions": [{"inference_id": np.array(["id1"])}],
            "message": "Custom metadata upload failed because no field_name was inputted",
        }
    ], "Expected failure due to no field_name"


@pytest.mark.asyncio
async def test_run_when_no_field_value() -> None:
    # given
    block = RoboflowCustomMetadataBlock(
        cache=MemoryCache(),
        background_tasks=None,
        api_key="my_api_key",
    )

    # when
    result = await block.run(
        fire_and_forget=True,
        field_name="location",
        field_value=None,
        predictions=[{"inference_id": np.array(["id1"])}],
    )

    # then
    assert result == [
        {
            "error_status": True,
            "predictions": [{"inference_id": np.array(["id1"])}],
            "message": "Custom metadata upload failed because no field_value was received",
        }
    ], "Expected failure due to no field_value"


@pytest.mark.asyncio
@patch(
    "inference.core.workflows.core_steps.sinks.roboflow.roboflow_custom_metadata.add_custom_metadata_request"
)
async def test_run_when_fire_and_forget(
    add_custom_metadata_request_mock: MagicMock,
) -> None:
    # given
    background_tasks = BackgroundTasks()
    block = RoboflowCustomMetadataBlock(
        cache=MemoryCache(),
        background_tasks=background_tasks,
        api_key="my_api_key",
    )
    add_custom_metadata_request_mock.return_value = True

    # when
    result = await block.run(
        fire_and_forget=True,
        field_name="location",
        field_value=["toronto"],
        predictions=[{"inference_id": np.array(["id1"])}],
    )

    # then
    assert result == [
        {
            "error_status": False,
            "predictions": [{"inference_id": np.array(["id1"])}],
            "message": "Custom metadata upload was successful",
        }
    ], "Expected success message"
    assert len(background_tasks.tasks) == 1, "Expected background task to be added"


@pytest.mark.asyncio
@patch(
    "inference.core.workflows.core_steps.sinks.roboflow.roboflow_custom_metadata.add_custom_metadata_request"
)
async def test_run_when_not_fire_and_forget(
    add_custom_metadata_request_mock: MagicMock,
) -> None:
    # given
    block = RoboflowCustomMetadataBlock(
        cache=MemoryCache(),
        background_tasks=None,
        api_key="my_api_key",
    )
    add_custom_metadata_request_mock.return_value = True

    # when
    result = await block.run(
        fire_and_forget=False,
        field_name="location",
        field_value=["toronto"],
        predictions=[{"inference_id": np.array(["id1"])}],
    )

    # then
    assert result == [
        {
            "error_status": False,
            "predictions": [{"inference_id": np.array(["id1"])}],
            "message": "Custom metadata upload was successful",
        }
    ], "Expected success message"
    add_custom_metadata_request_mock.assert_called_once()

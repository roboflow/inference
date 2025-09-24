import hashlib
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import supervision as sv
from fastapi import BackgroundTasks

from inference.core.cache import MemoryCache
from inference.core.workflows.core_steps.sinks.roboflow.custom_metadata.v1 import (
    RoboflowCustomMetadataBlockV1,
    add_custom_metadata_request,
    get_workspace_name,
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
    "inference.core.workflows.core_steps.sinks.roboflow.custom_metadata.v1.get_roboflow_workspace"
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
    "inference.core.workflows.core_steps.sinks.roboflow.custom_metadata.v1.add_custom_metadata"
)
def test_add_custom_metadata_request_success(
    add_custom_metadata_mock: MagicMock,
) -> None:
    # given
    add_custom_metadata_mock.return_value = True
    cache = MemoryCache()
    api_key = "my_api_key"
    api_key_hash = hashlib.md5(api_key.encode("utf-8")).hexdigest()
    expected_cache_key = f"workflows:api_key_to_workspace:{api_key_hash}"
    cache.set(key=expected_cache_key, value="my_workspace")
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
    assert result == (
        False,
        "Custom metadata upload was successful",
    ), "Expected metadata to be added successfully"
    assert (
        cache.get(expected_cache_key) == "my_workspace"
    ), "Expected workspace name to be in cache"


@patch(
    "inference.core.workflows.core_steps.sinks.roboflow.custom_metadata.v1.add_custom_metadata"
)
def test_add_custom_metadata_request_failure(
    add_custom_metadata_mock: MagicMock,
) -> None:
    # given
    add_custom_metadata_mock.side_effect = Exception("API error")
    cache = MemoryCache()
    api_key = "my_api_key"
    api_key_hash = hashlib.md5(api_key.encode("utf-8")).hexdigest()
    expected_cache_key = f"workflows:api_key_to_workspace:{api_key_hash}"
    cache.set(key=expected_cache_key, value="my_workspace")
    inference_ids = ["id1", "id2"]
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
    assert result[0] is True, "Expected metadata addition to fail"
    assert (
        cache.get(expected_cache_key) == "my_workspace"
    ), "Expected workspace name to be in cache"


def test_run_when_api_key_is_not_specified() -> None:
    # given
    block = RoboflowCustomMetadataBlockV1(
        cache=MemoryCache(),
        api_key=None,
        background_tasks=None,
        thread_pool_executor=None,
    )
    predictions = sv.Detections(
        xyxy=np.array([[1, 2, 3, 4]]),
        confidence=np.array([0.1]),
        class_id=np.array([1]),
    )
    predictions.data["inference_id"] = np.array(["id1"])

    # when
    with pytest.raises(ValueError):
        _ = block.run(
            fire_and_forget=True,
            field_name="location",
            field_value="toronto",
            predictions=predictions,
        )


def test_run_when_no_inference_ids() -> None:
    # given
    block = RoboflowCustomMetadataBlockV1(
        cache=MemoryCache(),
        api_key="my_api_key",
        background_tasks=None,
        thread_pool_executor=None,
    )

    # when
    result = block.run(
        fire_and_forget=True,
        field_name="location",
        field_value="toronto",
        predictions=sv.Detections.empty(),
    )

    # then
    assert result == {
        "error_status": True,
        "message": "Custom metadata upload failed because no inference_ids were received. This is known bug "
        "(https://github.com/roboflow/inference/issues/567). Please provide a report for the "
        "problem under mentioned issue.",
    }, "Expected failure due to no inference_ids"


@patch(
    "inference.core.workflows.core_steps.sinks.roboflow.custom_metadata.v1.add_custom_metadata_request"
)
def test_run_when_fire_and_forget_with_background_tasks(
    add_custom_metadata_request_mock: MagicMock,
) -> None:
    # given
    background_tasks = BackgroundTasks()
    block = RoboflowCustomMetadataBlockV1(
        cache=MemoryCache(),
        api_key="my_api_key",
        background_tasks=background_tasks,
        thread_pool_executor=None,
    )
    add_custom_metadata_request_mock.return_value = (
        False,
        "Custom metadata upload was successful",
    )
    predictions = sv.Detections(
        xyxy=np.array([[1, 2, 3, 4]]),
        confidence=np.array([0.1]),
        class_id=np.array([1]),
    )
    predictions.data["inference_id"] = np.array(["id1"])
    # when
    result = block.run(
        fire_and_forget=True,
        field_name="location",
        field_value="toronto",
        predictions=predictions,
    )

    # then
    assert result == {
        "error_status": False,
        "message": "Registration happens in the background task",
    }, "Expected success message"
    assert len(background_tasks.tasks) == 1, "Expected background task to be added"


@patch(
    "inference.core.workflows.core_steps.sinks.roboflow.custom_metadata.v1.add_custom_metadata_request"
)
def test_run_with_classification_results(
    add_custom_metadata_request_mock: MagicMock,
) -> None:
    # given
    background_tasks = BackgroundTasks()
    block = RoboflowCustomMetadataBlockV1(
        cache=MemoryCache(),
        api_key="my_api_key",
        background_tasks=background_tasks,
        thread_pool_executor=None,
    )
    add_custom_metadata_request_mock.return_value = (
        False,
        "Custom metadata upload was successful",
    )
    predictions = {"inference_id": "some-id"}

    # when
    result = block.run(
        fire_and_forget=True,
        field_name="location",
        field_value="toronto",
        predictions=predictions,
    )

    # then
    assert result == {
        "error_status": False,
        "message": "Registration happens in the background task",
    }, "Expected success message"
    assert len(background_tasks.tasks) == 1, "Expected background task to be added"


@patch(
    "inference.core.workflows.core_steps.sinks.roboflow.custom_metadata.v1.add_custom_metadata_request"
)
def test_run_with_classification_results_when_inference_id_is_not_given(
    add_custom_metadata_request_mock: MagicMock,
) -> None:
    # given
    background_tasks = BackgroundTasks()
    block = RoboflowCustomMetadataBlockV1(
        cache=MemoryCache(),
        api_key="my_api_key",
        background_tasks=background_tasks,
        thread_pool_executor=None,
    )
    add_custom_metadata_request_mock.return_value = (
        False,
        "Custom metadata upload was successful",
    )
    predictions = {"predictions": ["a", "b", "c"]}

    # when
    result = block.run(
        fire_and_forget=True,
        field_name="location",
        field_value="toronto",
        predictions=predictions,
    )

    # then
    assert result == {
        "error_status": True,
        "message": "Custom metadata upload failed because no inference_ids were received. This is known bug "
        "(https://github.com/roboflow/inference/issues/567). Please provide a report for the "
        "problem under mentioned issue.",
    }, "Expected failure due to no inference_ids"


@patch(
    "inference.core.workflows.core_steps.sinks.roboflow.custom_metadata.v1.add_custom_metadata_request"
)
def test_run_when_fire_and_forget_with_thread_pool(
    add_custom_metadata_request_mock: MagicMock,
) -> None:
    # given
    with ThreadPoolExecutor() as thread_pool_executor:
        block = RoboflowCustomMetadataBlockV1(
            cache=MemoryCache(),
            api_key="my_api_key",
            background_tasks=None,
            thread_pool_executor=thread_pool_executor,
        )
        add_custom_metadata_request_mock.return_value = (
            False,
            "Custom metadata upload was successful",
        )
        predictions = sv.Detections(
            xyxy=np.array([[1, 2, 3, 4]]),
            confidence=np.array([0.1]),
            class_id=np.array([1]),
        )
        predictions.data["inference_id"] = np.array(["id1"])

        # when
        result = block.run(
            fire_and_forget=True,
            field_name="location",
            field_value="toronto",
            predictions=predictions,
        )

        # then
        assert result == {
            "error_status": False,
            "message": "Registration happens in the background task",
        }, "Expected success message"


@patch(
    "inference.core.workflows.core_steps.sinks.roboflow.custom_metadata.v1.add_custom_metadata_request"
)
def test_run_when_not_fire_and_forget(
    add_custom_metadata_request_mock: MagicMock,
) -> None:
    # given
    block = RoboflowCustomMetadataBlockV1(
        cache=MemoryCache(),
        api_key="my_api_key",
        background_tasks=None,
        thread_pool_executor=None,
    )
    add_custom_metadata_request_mock.return_value = (
        False,
        "Custom metadata upload was successful",
    )
    predictions = sv.Detections(
        xyxy=np.array([[1, 2, 3, 4]]),
        confidence=np.array([0.1]),
        class_id=np.array([1]),
    )
    predictions.data["inference_id"] = np.array(["id1"])

    # when
    result = block.run(
        fire_and_forget=False,
        field_name="location",
        field_value="toronto",
        predictions=predictions,
    )

    # then
    assert result == {
        "error_status": False,
        "message": "Custom metadata upload was successful",
    }, "Expected success message"
    add_custom_metadata_request_mock.assert_called_once()


@patch(
    "inference.core.workflows.core_steps.sinks.roboflow.custom_metadata.v1.add_custom_metadata_request"
)
def test_run_with_field_value(
    add_custom_metadata_request_mock: MagicMock,
) -> None:
    # given
    block = RoboflowCustomMetadataBlockV1(
        cache=MemoryCache(),
        api_key="my_api_key",
        background_tasks=None,
        thread_pool_executor=None,
    )
    add_custom_metadata_request_mock.return_value = (
        False,
        "Custom metadata upload was successful",
    )
    predictions = sv.Detections(
        xyxy=np.array([[1, 2, 3, 4]]),
        confidence=np.array([0.1]),
        class_id=np.array([1]),
    )
    predictions.data["inference_id"] = np.array(["id1"])
    field_value = "new_york"

    # when
    result = block.run(
        fire_and_forget=False,
        field_name="location",
        field_value=field_value,
        predictions=predictions,
    )

    # then
    assert result == {
        "error_status": False,
        "message": "Custom metadata upload was successful",
    }, "Expected success message"
    add_custom_metadata_request_mock.assert_called_once_with(
        cache=block._cache,
        api_key=block._api_key,
        inference_ids=["id1"],
        field_name="location",
        field_value=field_value,
    )

from typing import Optional, Union
from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi import BackgroundTasks
from pydantic import ValidationError

from inference.core.cache import MemoryCache
from inference.core.workflows.core_steps.sinks.roboflow.dataset_upload import v2
from inference.core.workflows.core_steps.sinks.roboflow.dataset_upload.v2 import (
    BlockManifest,
    RoboflowDatasetUploadBlockV2,
    maybe_register_datapoint_at_roboflow,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    ImageParentMetadata,
    WorkflowImageData,
)


@mock.patch.object(v2, "register_datapoint_at_roboflow")
@mock.patch.object(v2, "random")
def test_maybe_register_datapoint_at_roboflow_when_data_sampled_off(
    random_mock: MagicMock,
    register_datapoint_at_roboflow_mock: MagicMock,
) -> None:
    # given
    random_mock.random.return_value = 0.38

    # when
    result = maybe_register_datapoint_at_roboflow(
        image=MagicMock(),
        prediction=MagicMock(),
        target_project="some",
        usage_quota_name="some",
        data_percentage=36.0,
        persist_predictions=True,
        minutely_usage_limit=10,
        hourly_usage_limit=100,
        daily_usage_limit=1000,
        max_image_size=(128, 128),
        compression_level=75,
        registration_tags=[],
        fire_and_forget=False,
        labeling_batch_prefix="some",
        new_labeling_batch_frequency="never",
        cache=MagicMock(),
        background_tasks=MagicMock(),
        thread_pool_executor=MagicMock(),
        api_key="XXX",
    )

    # then
    register_datapoint_at_roboflow_mock.assert_not_called()
    assert result == (False, "Registration skipped due to sampling settings")


@mock.patch.object(v2, "register_datapoint_at_roboflow")
@mock.patch.object(v2, "random")
def test_maybe_register_datapoint_at_roboflow_when_data_sample_accepted(
    random_mock: MagicMock,
    register_datapoint_at_roboflow_mock: MagicMock,
) -> None:
    # given
    random_mock.random.return_value = 0.38
    register_datapoint_at_roboflow_mock.return_value = (False, "ok")

    # when
    result = maybe_register_datapoint_at_roboflow(
        image=MagicMock(),
        prediction=MagicMock(),
        target_project="some",
        usage_quota_name="some",
        data_percentage=40.0,
        persist_predictions=True,
        minutely_usage_limit=10,
        hourly_usage_limit=100,
        daily_usage_limit=1000,
        max_image_size=(128, 128),
        compression_level=75,
        registration_tags=[],
        fire_and_forget=False,
        labeling_batch_prefix="some",
        new_labeling_batch_frequency="never",
        cache=MagicMock(),
        background_tasks=MagicMock(),
        thread_pool_executor=MagicMock(),
        api_key="XXX",
    )

    # then
    register_datapoint_at_roboflow_mock.assert_called_once(), "Expected to be called when data is sampled"
    assert result == (False, "ok"), "Expected to see mock return value"


@pytest.mark.parametrize("image_field_name", ["image", "images"])
@pytest.mark.parametrize("image_selector", ["$inputs.image", "$steps.some.image"])
@pytest.mark.parametrize("predictions", ["$steps.some.predictions", None])
def test_manifest_parsing_when_valid_input_provided_and_fields_not_linked(
    image_field_name: str,
    image_selector: str,
    predictions: Optional[str],
) -> None:
    # given
    raw_manifest = {
        "type": "roboflow_core/roboflow_dataset_upload@v2",
        "name": "some",
        image_field_name: image_selector,
        "predictions": predictions,
        "target_project": "some1",
        "usage_quota_name": "my_quota",
        "data_percentage": 37.5,
        "persist_predictions": False,
        "minutely_usage_limit": 10,
        "hourly_usage_limit": 100,
        "daily_usage_limit": 1000,
        "max_image_size": (100, 200),
        "compression_level": 100,
        "registration_tags": ["a", "b"],
        "disable_sink": True,
        "fire_and_forget": False,
        "labeling_batch_prefix": "my_batch",
        "labeling_batches_recreation_frequency": "never",
    }

    # when
    result = BlockManifest.model_validate(raw_manifest)

    # then
    assert result == BlockManifest(
        type="roboflow_core/roboflow_dataset_upload@v2",
        name="some",
        images=image_selector,
        predictions=predictions,
        target_project="some1",
        usage_quota_name="my_quota",
        data_percentage=37.5,
        persist_predictions=False,
        minutely_usage_limit=10,
        hourly_usage_limit=100,
        daily_usage_limit=1000,
        max_image_size=(100, 200),
        compression_level=100,
        registration_tags=["a", "b"],
        disable_sink=True,
        fire_and_forget=False,
        labeling_batch_prefix="my_batch",
        labeling_batches_recreation_frequency="never",
    )


@pytest.mark.parametrize("image_field_name", ["image", "images"])
@pytest.mark.parametrize("image_selector", ["$inputs.image", "$steps.some.image"])
@pytest.mark.parametrize("predictions", ["$steps.some.predictions", None])
def test_manifest_parsing_when_registration_tags_given_as_selector(
    image_field_name: str,
    image_selector: str,
    predictions: Optional[str],
) -> None:
    raw_manifest = {
        "type": "roboflow_core/roboflow_dataset_upload@v2",
        "name": "some",
        image_field_name: image_selector,
        "predictions": predictions,
        "target_project": "some1",
        "usage_quota_name": "my_quota",
        "data_percentage": "$inputs.data_percentage",
        "persist_predictions": "$inputs.persist_predictions",
        "minutely_usage_limit": 10,
        "hourly_usage_limit": 100,
        "daily_usage_limit": 1000,
        "max_image_size": (100, 200),
        "compression_level": 100,
        "registration_tags": "$inputs.tags",
        "disable_sink": "$inputs.disable_sink",
        "fire_and_forget": "$inputs.fire_and_forget",
        "labeling_batch_prefix": "$inputs.labeling_batch_prefix",
        "labeling_batches_recreation_frequency": "never",
    }

    result = BlockManifest.model_validate(raw_manifest)

    assert result == BlockManifest(
        type="roboflow_core/roboflow_dataset_upload@v2",
        name="some",
        images=image_selector,
        predictions=predictions,
        target_project="some1",
        usage_quota_name="my_quota",
        data_percentage="$inputs.data_percentage",
        persist_predictions="$inputs.persist_predictions",
        minutely_usage_limit=10,
        hourly_usage_limit=100,
        daily_usage_limit=1000,
        max_image_size=(100, 200),
        compression_level=100,
        registration_tags="$inputs.tags",
        disable_sink="$inputs.disable_sink",
        fire_and_forget="$inputs.fire_and_forget",
        labeling_batch_prefix="$inputs.labeling_batch_prefix",
        labeling_batches_recreation_frequency="never",
    )


@pytest.mark.parametrize("image_field_name", ["image", "images"])
@pytest.mark.parametrize("image_selector", ["$inputs.image", "$steps.some.image"])
@pytest.mark.parametrize("predictions", ["$steps.some.predictions", None])
def test_manifest_parsing_when_valid_input_provided_and_fields_linked(
    image_field_name: str,
    image_selector: str,
    predictions: Optional[str],
) -> None:
    # given
    raw_manifest = {
        "type": "roboflow_core/roboflow_dataset_upload@v2",
        "name": "some",
        image_field_name: image_selector,
        "predictions": predictions,
        "target_project": "some1",
        "usage_quota_name": "my_quota",
        "data_percentage": "$inputs.data_percentage",
        "persist_predictions": "$inputs.persist_predictions",
        "minutely_usage_limit": 10,
        "hourly_usage_limit": 100,
        "daily_usage_limit": 1000,
        "max_image_size": (100, 200),
        "compression_level": 100,
        "registration_tags": ["a", "b", "$inputs.tag"],
        "disable_sink": "$inputs.disable_sink",
        "fire_and_forget": "$inputs.fire_and_forget",
        "labeling_batch_prefix": "$inputs.labeling_batch_prefix",
        "labeling_batches_recreation_frequency": "never",
    }

    # when
    result = BlockManifest.model_validate(raw_manifest)

    # then
    assert result == BlockManifest(
        type="roboflow_core/roboflow_dataset_upload@v2",
        name="some",
        images=image_selector,
        predictions=predictions,
        target_project="some1",
        usage_quota_name="my_quota",
        data_percentage="$inputs.data_percentage",
        persist_predictions="$inputs.persist_predictions",
        minutely_usage_limit=10,
        hourly_usage_limit=100,
        daily_usage_limit=1000,
        max_image_size=(100, 200),
        compression_level=100,
        registration_tags=["a", "b", "$inputs.tag"],
        disable_sink="$inputs.disable_sink",
        fire_and_forget="$inputs.fire_and_forget",
        labeling_batch_prefix="$inputs.labeling_batch_prefix",
        labeling_batches_recreation_frequency="never",
    )


@pytest.mark.parametrize("compression_level", [0, -1, 101])
def test_manifest_parsing_when_compression_level_invalid(
    compression_level: float,
) -> None:
    raw_manifest = {
        "type": "roboflow_core/roboflow_dataset_upload@v2",
        "name": "some",
        "images": "$inputs.image",
        "predictions": None,
        "target_project": "some1",
        "usage_quota_name": "my_quota",
        "data_percentage": "$inputs.data_percentage",
        "persist_predictions": "$inputs.persist_predictions",
        "minutely_usage_limit": 10,
        "hourly_usage_limit": 100,
        "daily_usage_limit": 1000,
        "max_image_size": (100, 200),
        "compression_level": compression_level,
        "registration_tags": ["a", "b", "$inputs.tag"],
        "disable_sink": "$inputs.disable_sink",
        "fire_and_forget": "$inputs.fire_and_forget",
        "labeling_batch_prefix": "$inputs.labeling_batch_prefix",
        "labeling_batches_recreation_frequency": "never",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(raw_manifest)


@pytest.mark.parametrize("data_percentage", [-1.0, 101.0])
def test_manifest_parsing_when_data_percentage_invalid(data_percentage: float) -> None:
    raw_manifest = {
        "type": "roboflow_core/roboflow_dataset_upload@v2",
        "name": "some",
        "images": "$inputs.image",
        "predictions": None,
        "target_project": "some1",
        "usage_quota_name": "my_quota",
        "data_percentage": data_percentage,
        "persist_predictions": "$inputs.persist_predictions",
        "minutely_usage_limit": 10,
        "hourly_usage_limit": 100,
        "daily_usage_limit": 1000,
        "max_image_size": (100, 200),
        "compression_level": 85,
        "registration_tags": ["a", "b", "$inputs.tag"],
        "disable_sink": "$inputs.disable_sink",
        "fire_and_forget": "$inputs.fire_and_forget",
        "labeling_batch_prefix": "$inputs.labeling_batch_prefix",
        "labeling_batches_recreation_frequency": "never",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(raw_manifest)


@mock.patch.object(v2, "register_datapoint_at_roboflow")
def test_run_sink_when_data_sampled_off(
    register_datapoint_at_roboflow_mock: MagicMock,
) -> None:
    # given
    background_tasks = BackgroundTasks()
    cache = MemoryCache()
    data_collector_block = RoboflowDatasetUploadBlockV2(
        cache=cache,
        api_key="my_api_key",
        background_tasks=background_tasks,
        thread_pool_executor=None,
    )
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        numpy_image=np.zeros((512, 256, 3), dtype=np.uint8),
    )
    register_datapoint_at_roboflow_mock.return_value = False, "OK"
    indices = [(0,), (1,), (2,)]

    # when
    result = data_collector_block.run(
        images=Batch(content=[image, image, image], indices=indices),
        predictions=None,
        target_project="my_project",
        usage_quota_name="my_quota",
        data_percentage=0.0,
        persist_predictions=True,
        minutely_usage_limit=10,
        hourly_usage_limit=100,
        daily_usage_limit=1000,
        max_image_size=(128, 128),
        compression_level=75,
        registration_tags=["some"],
        disable_sink=False,
        fire_and_forget=False,
        labeling_batch_prefix="my_batch",
        labeling_batches_recreation_frequency="never",
    )

    # then
    assert (
        result
        == [
            {
                "error_status": False,
                "message": "Registration skipped due to sampling settings",
            }
        ]
        * 3
    ), "Expected nothing registered"
    register_datapoint_at_roboflow_mock.assert_not_called()


@mock.patch.object(v2, "register_datapoint_at_roboflow")
def test_run_sink_when_data_sampled(
    register_datapoint_at_roboflow_mock: MagicMock,
) -> None:
    # given
    background_tasks = BackgroundTasks()
    cache = MemoryCache()
    data_collector_block = RoboflowDatasetUploadBlockV2(
        cache=cache,
        api_key="my_api_key",
        background_tasks=background_tasks,
        thread_pool_executor=None,
    )
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        numpy_image=np.zeros((512, 256, 3), dtype=np.uint8),
    )
    register_datapoint_at_roboflow_mock.return_value = False, "OK"
    indices = [(0,), (1,), (2,)]

    # when
    result = data_collector_block.run(
        images=Batch(content=[image, image, image], indices=indices),
        predictions=None,
        target_project="my_project",
        usage_quota_name="my_quota",
        data_percentage=100.1,  # more than 1.0 to ensure always True on sampling
        persist_predictions=True,
        minutely_usage_limit=10,
        hourly_usage_limit=100,
        daily_usage_limit=1000,
        max_image_size=(128, 128),
        compression_level=75,
        registration_tags=["some"],
        disable_sink=False,
        fire_and_forget=False,
        labeling_batch_prefix="my_batch",
        labeling_batches_recreation_frequency="never",
    )

    # then
    assert (
        result
        == [
            {
                "error_status": False,
                "message": "OK",
            }
        ]
        * 3
    ), "Expected data registered"
    assert register_datapoint_at_roboflow_mock.call_count == 3


@mock.patch.object(v2, "register_datapoint_at_roboflow")
def test_run_sink_with_image_name_parameter(
    register_datapoint_at_roboflow_mock: MagicMock,
) -> None:
    # given
    background_tasks = BackgroundTasks()
    cache = MemoryCache()
    data_collector_block = RoboflowDatasetUploadBlockV2(
        cache=cache,
        api_key="my_api_key",
        background_tasks=background_tasks,
        thread_pool_executor=None,
    )
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        numpy_image=np.zeros((512, 256, 3), dtype=np.uint8),
    )
    register_datapoint_at_roboflow_mock.return_value = False, "OK"
    indices = [(0,), (1,)]

    # when
    result = data_collector_block.run(
        images=Batch(content=[image, image], indices=indices),
        predictions=None,
        target_project="my_project",
        usage_quota_name="my_quota",
        data_percentage=100.0,
        persist_predictions=True,
        minutely_usage_limit=10,
        hourly_usage_limit=100,
        daily_usage_limit=1000,
        max_image_size=(128, 128),
        compression_level=75,
        registration_tags=["some"],
        disable_sink=False,
        fire_and_forget=False,
        labeling_batch_prefix="my_batch",
        labeling_batches_recreation_frequency="never",
        image_name=Batch(content=["serial_001", "serial_002"], indices=indices),
    )

    # then
    assert result == [
        {"error_status": False, "message": "OK"},
        {"error_status": False, "message": "OK"},
    ], "Expected data registered"
    assert register_datapoint_at_roboflow_mock.call_count == 2

    # Verify image_name was passed correctly
    calls = register_datapoint_at_roboflow_mock.call_args_list
    assert calls[0].kwargs["image_name"] == "serial_001"
    assert calls[1].kwargs["image_name"] == "serial_002"


@mock.patch.object(v2, "register_datapoint_at_roboflow")
def test_run_sink_with_image_name_from_workflow_image_data(
    register_datapoint_at_roboflow_mock: MagicMock,
) -> None:
    # given
    background_tasks = BackgroundTasks()
    cache = MemoryCache()
    data_collector_block = RoboflowDatasetUploadBlockV2(
        cache=cache,
        api_key="my_api_key",
        background_tasks=background_tasks,
        thread_pool_executor=None,
    )
    # Create images with image_name set in the WorkflowImageData
    image1 = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        numpy_image=np.zeros((512, 256, 3), dtype=np.uint8),
        image_name="from_image_data_001",
    )
    image2 = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        numpy_image=np.zeros((512, 256, 3), dtype=np.uint8),
        image_name="from_image_data_002",
    )
    register_datapoint_at_roboflow_mock.return_value = False, "OK"
    indices = [(0,), (1,)]

    # when - no explicit image_name parameter, should use from WorkflowImageData
    result = data_collector_block.run(
        images=Batch(content=[image1, image2], indices=indices),
        predictions=None,
        target_project="my_project",
        usage_quota_name="my_quota",
        data_percentage=100.0,
        persist_predictions=True,
        minutely_usage_limit=10,
        hourly_usage_limit=100,
        daily_usage_limit=1000,
        max_image_size=(128, 128),
        compression_level=75,
        registration_tags=["some"],
        disable_sink=False,
        fire_and_forget=False,
        labeling_batch_prefix="my_batch",
        labeling_batches_recreation_frequency="never",
    )

    # then
    assert result == [
        {"error_status": False, "message": "OK"},
        {"error_status": False, "message": "OK"},
    ], "Expected data registered"
    assert register_datapoint_at_roboflow_mock.call_count == 2

    # Verify image_name from WorkflowImageData was passed correctly
    calls = register_datapoint_at_roboflow_mock.call_args_list
    assert calls[0].kwargs["image_name"] == "from_image_data_001"
    assert calls[1].kwargs["image_name"] == "from_image_data_002"


def test_manifest_parsing_with_image_name_field() -> None:
    # given
    raw_manifest = {
        "type": "roboflow_core/roboflow_dataset_upload@v2",
        "name": "some",
        "images": "$inputs.image",
        "predictions": None,
        "target_project": "some1",
        "usage_quota_name": "my_quota",
        "data_percentage": 100.0,
        "persist_predictions": True,
        "minutely_usage_limit": 10,
        "hourly_usage_limit": 100,
        "daily_usage_limit": 1000,
        "max_image_size": (100, 200),
        "compression_level": 95,
        "registration_tags": [],
        "disable_sink": False,
        "fire_and_forget": False,
        "labeling_batch_prefix": "my_batch",
        "labeling_batches_recreation_frequency": "never",
        "image_name": "$inputs.filename",
    }

    # when
    result = BlockManifest.model_validate(raw_manifest)

    # then
    assert result.image_name == "$inputs.filename"


def test_manifest_parsing_with_static_image_name() -> None:
    # given
    raw_manifest = {
        "type": "roboflow_core/roboflow_dataset_upload@v2",
        "name": "some",
        "images": "$inputs.image",
        "predictions": None,
        "target_project": "some1",
        "usage_quota_name": "my_quota",
        "data_percentage": 100.0,
        "persist_predictions": True,
        "minutely_usage_limit": 10,
        "hourly_usage_limit": 100,
        "daily_usage_limit": 1000,
        "max_image_size": (100, 200),
        "compression_level": 95,
        "registration_tags": [],
        "disable_sink": False,
        "fire_and_forget": False,
        "labeling_batch_prefix": "my_batch",
        "labeling_batches_recreation_frequency": "never",
        "image_name": "my_static_image_name",
    }

    # when
    result = BlockManifest.model_validate(raw_manifest)

    # then
    assert result.image_name == "my_static_image_name"

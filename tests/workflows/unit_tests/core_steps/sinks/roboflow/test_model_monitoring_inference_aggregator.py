from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from unittest.mock import ANY, MagicMock, patch

import numpy as np
import supervision as sv
from fastapi import BackgroundTasks

from inference.core.cache import MemoryCache
from inference.core.workflows.core_steps.sinks.roboflow.model_monitoring_inference_aggregator.v1 import (
    ModelMonitoringInferenceAggregatorBlockV1,
)


@patch("inference.core.roboflow_api.send_inference_results_to_model_monitoring")
@patch(
    "inference.core.workflows.core_steps.sinks.roboflow.model_monitoring_inference_aggregator.v1.get_roboflow_workspace"
)
def test_run_not_in_reporting_range_success(
    get_roboflow_workspace_mock: MagicMock,
    send_inference_results_to_model_monitoring_mock: MagicMock,
) -> None:
    # given
    get_roboflow_workspace_mock.return_value = "my_workspace"
    send_inference_results_to_model_monitoring_mock.return_value = 200
    unique_aggregator_key = "session-test_run_not_in_reporting_range_success"
    cache = MemoryCache()
    cache_key = f"workflows:steps_cache:roboflow_core/model_monitoring_inference_aggregator@v1:{unique_aggregator_key}:last_report_time"
    cache.set(cache_key, (datetime.now() + timedelta(days=1)).isoformat())
    predictions = sv.Detections(
        xyxy=np.array(
            [
                [100, 100, 200, 200],
                [150, 150, 250, 250],
            ]
        ),
        data={
            "class_name": np.array(["Hardhat", "Person"]),
            "detection_id": np.array(
                [
                    "422e8419-1386-41b4-bba5-d5c507895bda",
                    "d97e9579-80e4-4ac3-8fd8-26485606170a",
                ]
            ),
            "inference_id": np.array(["id1", "id2"]),
            "prediction_type": np.array(["object-detection", "object-detection"]),
        },
        confidence=np.array([0.9, 0.8]),
    )

    # when
    block = ModelMonitoringInferenceAggregatorBlockV1(
        cache=cache,
        api_key="my_api_key",
        background_tasks=None,
        thread_pool_executor=None,
    )
    result = block.run(
        fire_and_forget=True,
        frequency=10,
        predictions=predictions,
        unique_aggregator_key=unique_aggregator_key,
        model_id="my_model_id",
    )

    # then
    assert result == {
        "error_status": False,
        "message": "Not in reporting range, skipping report. (Ok)",
    }, "Expected successful upload"
    assert cache.get(cache_key) is not None


@patch(
    "inference.core.workflows.core_steps.sinks.roboflow.model_monitoring_inference_aggregator.v1.send_inference_results_to_model_monitoring"
)
@patch(
    "inference.core.workflows.core_steps.sinks.roboflow.model_monitoring_inference_aggregator.v1.get_roboflow_workspace"
)
def test_run_in_reporting_range_success_with_object_detection(
    get_roboflow_workspace_mock: MagicMock,
    send_inference_results_to_model_monitoring_mock: MagicMock,
) -> None:
    # given
    unique_aggregator_key = (
        "session-test_run_in_reporting_range_success_with_object_detection"
    )
    send_inference_results_to_model_monitoring_mock.return_value = (
        False,
        "Data sent successfully",
    )
    get_roboflow_workspace_mock.return_value = "workspace-name"
    cache_key = f"workflows:steps_cache:roboflow_core/model_monitoring_inference_aggregator@v1:{unique_aggregator_key}:last_report_time"
    cache = MemoryCache()
    cache.set(cache_key, datetime(2024, 11, 10, 12, 0, 0).isoformat())
    api_key = "my_api_key"
    predictions = sv.Detections(
        xyxy=np.array(
            [
                [100, 100, 200, 200],
                [150, 150, 250, 250],
            ]
        ),
        data={
            "class_name": np.array(["Hardhat", "Person"]),
            "detection_id": np.array(
                [
                    "422e8419-1386-41b4-bba5-d5c507895bda",
                    "d97e9579-80e4-4ac3-8fd8-26485606170a",
                ]
            ),
            "inference_id": np.array(["id1", "id2"]),
            "prediction_type": np.array(["object-detection", "object-detection"]),
        },
        confidence=np.array([0.9, 0.8]),
    )

    # when
    block = ModelMonitoringInferenceAggregatorBlockV1(
        cache=cache,
        api_key=api_key,
        background_tasks=None,
        thread_pool_executor=None,
    )
    result = block.run(
        fire_and_forget=False,
        frequency=10,
        predictions=predictions,
        unique_aggregator_key=unique_aggregator_key,
        model_id="construction-safety/10",
    )

    # then
    assert result == {
        "error_status": False,
        "message": "Data sent successfully",
    }, "Expected successful upload"
    send_inference_results_to_model_monitoring_mock.assert_called_once_with(
        "my_api_key",
        "workspace-name",
        {
            "timestamp": ANY,
            "source": "workflow",
            "source_info": "ModelMonitoringInferenceAggregatorBlockV1",
            "inference_results": sorted(
                [
                    {
                        "class_name": "Hardhat",
                        "confidence": 0.9,
                        "inference_id": "id1",
                        "model_type": "object-detection",
                        "model_id": "construction-safety/10",
                    },
                    {
                        "class_name": "Person",
                        "confidence": 0.8,
                        "inference_id": "id2",
                        "model_type": "object-detection",
                        "model_id": "construction-safety/10",
                    },
                ],
                key=lambda x: x["inference_id"],
            ),
            "device_id": ANY,
            "platform": ANY,
            "platform_release": ANY,
            "platform_version": ANY,
            "architecture": ANY,
            "hostname": ANY,
            "ip_address": ANY,
            "mac_address": ANY,
            "processor": ANY,
            "inference_server_version": ANY,
        },
    )
    assert cache.get(cache_key) != datetime(2024, 11, 10, 12, 0, 0).isoformat()


@patch(
    "inference.core.workflows.core_steps.sinks.roboflow.model_monitoring_inference_aggregator.v1.send_inference_results_to_model_monitoring"
)
@patch(
    "inference.core.workflows.core_steps.sinks.roboflow.model_monitoring_inference_aggregator.v1.get_roboflow_workspace"
)
def test_run_in_reporting_range_success_with_single_label_classification(
    get_roboflow_workspace_mock: MagicMock,
    send_inference_results_to_model_monitoring_mock: MagicMock,
) -> None:
    # given
    unique_aggregator_key = (
        "session-test_run_in_reporting_range_success_with_single_label_classification"
    )
    send_inference_results_to_model_monitoring_mock.return_value = (
        False,
        "Data sent successfully",
    )
    get_roboflow_workspace_mock.return_value = "workspace-name"
    cache_key = f"workflows:steps_cache:roboflow_core/model_monitoring_inference_aggregator@v1:{unique_aggregator_key}:last_report_time"
    cache = MemoryCache()
    cache.set(cache_key, datetime(2024, 11, 10, 12, 0, 0).isoformat())
    api_key = "my_api_key"
    predictions = {
        "inference_id": "491d086d-4c6d-41b1-8915-a36ee2af5f6f",
        "time": 0.19997841701842844,
        "image": {"width": 416, "height": 416},
        "predictions": [{"class": "pill", "class_id": 0, "confidence": 1.0}],
        "top": "pill",
        "confidence": 1.0,
        "prediction_type": "classification",
        "parent_id": "image2",
        "root_parent_id": "image2",
    }

    # when
    block = ModelMonitoringInferenceAggregatorBlockV1(
        cache=cache,
        api_key=api_key,
        background_tasks=None,
        thread_pool_executor=None,
    )
    result = block.run(
        fire_and_forget=False,
        frequency=10,
        predictions=predictions,
        unique_aggregator_key=unique_aggregator_key,
        model_id="pills-classification/1",
    )

    # then
    assert result == {
        "error_status": False,
        "message": "Data sent successfully",
    }, "Expected successful upload"
    send_inference_results_to_model_monitoring_mock.assert_called_once_with(
        "my_api_key",
        "workspace-name",
        {
            "timestamp": ANY,
            "source": "workflow",
            "source_info": "ModelMonitoringInferenceAggregatorBlockV1",
            "inference_results": [
                {
                    "class_name": "pill",
                    "confidence": 1.0,
                    "inference_id": "491d086d-4c6d-41b1-8915-a36ee2af5f6f",
                    "model_type": "classification",
                    "model_id": "pills-classification/1",
                }
            ],
            "device_id": ANY,
            "platform": ANY,
            "platform_release": ANY,
            "platform_version": ANY,
            "architecture": ANY,
            "hostname": ANY,
            "ip_address": ANY,
            "mac_address": ANY,
            "processor": ANY,
            "inference_server_version": ANY,
        },
    )
    assert cache.get(cache_key) != datetime(2024, 11, 10, 12, 0, 0).isoformat()


@patch(
    "inference.core.workflows.core_steps.sinks.roboflow.model_monitoring_inference_aggregator.v1.send_inference_results_to_model_monitoring"
)
@patch(
    "inference.core.workflows.core_steps.sinks.roboflow.model_monitoring_inference_aggregator.v1.get_roboflow_workspace"
)
def test_run_in_reporting_range_success_with_multi_label_classification(
    get_roboflow_workspace_mock: MagicMock,
    send_inference_results_to_model_monitoring_mock: MagicMock,
) -> None:
    # given
    send_inference_results_to_model_monitoring_mock.return_value = (
        False,
        "Data sent successfully",
    )
    get_roboflow_workspace_mock.return_value = "workspace-name"
    unique_aggregator_key = (
        "session-test_run_in_reporting_range_success_with_multi_label_classification"
    )
    cache_key = f"workflows:steps_cache:roboflow_core/model_monitoring_inference_aggregator@v1:{unique_aggregator_key}:last_report_time"
    cache = MemoryCache()
    cache.set(cache_key, datetime(2024, 11, 10, 12, 0, 0).isoformat())
    api_key = "my_api_key"
    predictions = {
        "inference_id": "5a1fc086-c2eb-43b4-9f75-e71ec67c91e8",
        "time": 0.29568295809440315,
        "image": {"width": 355, "height": 474},
        "predictions": {
            "cat": {
                "confidence": 0.5594449043273926,
                "class_id": 0,
                "model_id": "animals/3",
            },
            "dog": {
                "confidence": 0.4901779294013977,
                "class_id": 1,
                "model_id": "animals/3",
            },
        },
        "predicted_classes": ["cat"],
        "prediction_type": "classification",
        "parent_id": "image",
        "root_parent_id": "image",
    }

    # when
    block = ModelMonitoringInferenceAggregatorBlockV1(
        cache=cache,
        api_key=api_key,
        background_tasks=None,
        thread_pool_executor=None,
    )
    result = block.run(
        fire_and_forget=False,
        frequency=10,
        predictions=predictions,
        unique_aggregator_key=unique_aggregator_key,
        model_id="animals/32",
    )

    # then
    assert result == {
        "error_status": False,
        "message": "Data sent successfully",
    }, "Expected successful upload"
    send_inference_results_to_model_monitoring_mock.assert_called_once_with(
        "my_api_key",
        "workspace-name",
        {
            "timestamp": ANY,
            "source": "workflow",
            "source_info": "ModelMonitoringInferenceAggregatorBlockV1",
            "inference_results": sorted(
                [
                    {
                        "model_id": "animals/32",
                        "class_name": "cat",
                        "confidence": 0.5594449043273926,
                        "inference_id": "5a1fc086-c2eb-43b4-9f75-e71ec67c91e8",
                        "model_type": "classification",
                    },
                    {
                        "model_id": "animals/32",
                        "class_name": "dog",
                        "confidence": 0.4901779294013977,
                        "inference_id": "5a1fc086-c2eb-43b4-9f75-e71ec67c91e8",
                        "model_type": "classification",
                    },
                ],
                key=lambda x: x["inference_id"],
            ),
            "device_id": ANY,
            "platform": ANY,
            "platform_release": ANY,
            "platform_version": ANY,
            "architecture": ANY,
            "hostname": ANY,
            "ip_address": ANY,
            "mac_address": ANY,
            "processor": ANY,
            "inference_server_version": ANY,
        },
    )
    assert cache.get(cache_key) != datetime(2024, 11, 10, 12, 0, 0).isoformat()


@patch(
    "inference.core.workflows.core_steps.sinks.roboflow.model_monitoring_inference_aggregator.v1.send_inference_results_to_model_monitoring"
)
@patch(
    "inference.core.workflows.core_steps.sinks.roboflow.model_monitoring_inference_aggregator.v1.get_roboflow_workspace"
)
def test_send_inference_results_to_model_monitoring_failure(
    get_roboflow_workspace_mock: MagicMock,
    send_inference_results_to_model_monitoring_mock: MagicMock,
) -> None:
    # given
    send_inference_results_to_model_monitoring_mock.side_effect = Exception("API error")
    get_roboflow_workspace_mock.return_value = "workspace-name"
    unique_aggregator_key = (
        "session-test_send_inference_results_to_model_monitoring_failure"
    )
    cache_key = f"workflows:steps_cache:roboflow_core/model_monitoring_inference_aggregator@v1:{unique_aggregator_key}:last_report_time"
    cache = MemoryCache()
    cache.set(cache_key, datetime(2024, 11, 10, 12, 0, 0).isoformat())
    api_key = "my_api_key"
    predictions = sv.Detections(
        xyxy=np.array(
            [
                [100, 100, 200, 200],
                [150, 150, 250, 250],
            ]
        ),
        data={
            "class_name": np.array(["Hardhat", "Person"]),
            "detection_id": np.array(
                [
                    "422e8419-1386-41b4-bba5-d5c507895bda",
                    "d97e9579-80e4-4ac3-8fd8-26485606170a",
                ]
            ),
            "inference_id": np.array(["id1", "id2"]),
            "prediction_type": np.array(["object-detection", "object-detection"]),
        },
        confidence=np.array([0.9, 0.8]),
    )

    # when
    block = ModelMonitoringInferenceAggregatorBlockV1(
        cache=cache,
        api_key=api_key,
        background_tasks=None,
        thread_pool_executor=None,
    )
    result = block.run(
        fire_and_forget=False,
        frequency=1,
        predictions=predictions,
        unique_aggregator_key=unique_aggregator_key,
        model_id="my_model_id",
    )

    # then
    assert type(result) == dict, "Expected result to be a dict"
    assert result["error_status"] is True, "Expected upload to fail"
    assert (
        "Error while uploading inference data" in result["message"]
    ), "Expected error message in result"
    assert cache.get(cache_key) is not None


@patch(
    "inference.core.workflows.core_steps.sinks.roboflow.model_monitoring_inference_aggregator.v1.get_roboflow_workspace"
)
@patch("inference.core.roboflow_api.send_inference_results_to_model_monitoring")
def test_run_when_not_in_reporting_range(
    send_inference_results_to_model_monitoring_mock: MagicMock,
    get_roboflow_workspace_mock: MagicMock,
) -> None:
    # given
    send_inference_results_to_model_monitoring_mock.return_value = (
        False,
        "Not in reporting range, skipping report. (Ok)",
    )
    get_roboflow_workspace_mock.return_value = "workspace-name"
    unique_aggregator_key = "session-test_run_not_in_reporting_range_success"
    cache_key = f"workflows:steps_cache:roboflow_core/model_monitoring_inference_aggregator@v1:{unique_aggregator_key}:last_report_time"
    cache = MemoryCache()
    cache.set(cache_key, datetime(2034, 11, 10, 12, 0, 0).isoformat())
    api_key = "my_api_key"
    predictions = sv.Detections(
        xyxy=np.array(
            [
                [100, 100, 200, 200],
                [150, 150, 250, 250],
            ]
        ),
        data={
            "class_name": np.array(["Hardhat", "Person"]),
            "detection_id": np.array(
                [
                    "422e8419-1386-41b4-bba5-d5c507895bda",
                    "d97e9579-80e4-4ac3-8fd8-26485606170a",
                ]
            ),
            "inference_id": np.array(["id1", "id2"]),
            "prediction_type": np.array(["object-detection", "object-detection"]),
        },
        confidence=np.array([0.9, 0.8]),
    )

    # when
    block = ModelMonitoringInferenceAggregatorBlockV1(
        cache=cache,
        api_key=api_key,
        background_tasks=None,
        thread_pool_executor=None,
    )
    result = block.run(
        fire_and_forget=False,
        frequency=10,
        predictions=predictions,
        unique_aggregator_key=unique_aggregator_key,
        model_id="my_model_id",
    )

    # then
    assert result == {
        "error_status": False,
        "message": "Not in reporting range, skipping report. (Ok)",
    }, "Expected skipping report due to frequency"
    assert cache.get(cache_key) is not None


@patch(
    "inference.core.workflows.core_steps.sinks.roboflow.model_monitoring_inference_aggregator.v1.get_roboflow_workspace"
)
@patch("inference.core.roboflow_api.send_inference_results_to_model_monitoring")
def test_run_when_fire_and_forget_with_background_tasks(
    send_inference_results_to_model_monitoring_mock: MagicMock,
    get_roboflow_workspace_mock: MagicMock,
) -> None:
    # given
    background_tasks = BackgroundTasks()
    unique_aggregator_key = (
        "session-test_run_when_fire_and_forget_with_background_tasks"
    )
    cache_key = f"workflows:steps_cache:roboflow_core/model_monitoring_inference_aggregator@v1:{unique_aggregator_key}:last_report_time"
    send_inference_results_to_model_monitoring_mock.return_value = (
        False,
        "Inference data upload was successful",
    )
    get_roboflow_workspace_mock.return_value = "workspace-name"
    cache = MemoryCache()
    cache.set(cache_key, datetime(2024, 11, 10, 12, 0, 0).isoformat())
    api_key = "my_api_key"
    predictions = sv.Detections(
        xyxy=np.array(
            [
                [100, 100, 200, 200],
                [150, 150, 250, 250],
            ]
        ),
        data={
            "class_name": np.array(["Hardhat", "Person"]),
            "detection_id": np.array(
                [
                    "422e8419-1386-41b4-bba5-d5c507895bda",
                    "d97e9579-80e4-4ac3-8fd8-26485606170a",
                ]
            ),
            "inference_id": np.array(["id1", "id2"]),
            "prediction_type": np.array(["object-detection", "object-detection"]),
        },
        confidence=np.array([0.9, 0.8]),
    )

    # when
    block = ModelMonitoringInferenceAggregatorBlockV1(
        cache=cache,
        api_key=api_key,
        background_tasks=background_tasks,
        thread_pool_executor=None,
    )
    result = block.run(
        fire_and_forget=True,
        frequency=10,
        predictions=predictions,
        unique_aggregator_key=unique_aggregator_key,
        model_id="my_model_id",
    )

    # then
    assert result == {
        "error_status": False,
        "message": "Reporting happens in the background task",
    }, "Expected background task to be added"
    assert len(background_tasks.tasks) == 1, "Expected one background task"


@patch(
    "inference.core.workflows.core_steps.sinks.roboflow.model_monitoring_inference_aggregator.v1.get_roboflow_workspace"
)
@patch("inference.core.roboflow_api.send_inference_results_to_model_monitoring")
def test_run_when_fire_and_forget_with_thread_pool(
    send_inference_results_to_model_monitoring_mock: MagicMock,
    get_roboflow_workspace_mock: MagicMock,
) -> None:
    # given
    with ThreadPoolExecutor() as thread_pool_executor:
        send_inference_results_to_model_monitoring_mock.return_value = (
            False,
            "Inference data upload was successful",
        )
        get_roboflow_workspace_mock.return_value = "workspace-name"
        cache = MemoryCache()
        unique_aggregator_key = "session-test_run_when_fire_and_forget_with_thread_pool"
        cache_key = f"workflows:steps_cache:roboflow_core/model_monitoring_inference_aggregator@v1:{unique_aggregator_key}:last_report_time"
        cache.set(cache_key, datetime(2024, 11, 10, 12, 0, 0).isoformat())
        api_key = "my_api_key"
        predictions = sv.Detections(
            xyxy=np.array(
                [
                    [100, 100, 200, 200],
                    [150, 150, 250, 250],
                ]
            ),
            data={
                "class_name": np.array(["Hardhat", "Person"]),
                "detection_id": np.array(
                    [
                        "422e8419-1386-41b4-bba5-d5c507895bda",
                        "d97e9579-80e4-4ac3-8fd8-26485606170a",
                    ]
                ),
                "inference_id": np.array(["id1", "id2"]),
                "prediction_type": np.array(["object-detection", "object-detection"]),
            },
            confidence=np.array([0.9, 0.8]),
        )

        # when
        block = ModelMonitoringInferenceAggregatorBlockV1(
            cache=cache,
            api_key=api_key,
            background_tasks=None,
            thread_pool_executor=thread_pool_executor,
        )
        result = block.run(
            fire_and_forget=True,
            frequency=10,
            predictions=predictions,
            unique_aggregator_key=unique_aggregator_key,
            model_id="my_model_id",
        )

        # then
        assert result == {
            "error_status": False,
            "message": "Reporting happens in the background task",
        }, "Expected background task to be added"

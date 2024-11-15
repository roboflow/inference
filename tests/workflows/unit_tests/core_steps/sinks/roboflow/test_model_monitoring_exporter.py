from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
import supervision as sv
from fastapi import BackgroundTasks
import numpy as np

from inference.core.cache import MemoryCache
from inference.core.workflows.core_steps.sinks.roboflow.model_monitoring_exporter.v1 import (
    RoboflowModelMonitoringExporterBlockV1,
    export_to_model_monitoring_request,
)


@patch(
    "inference.core.workflows.core_steps.sinks.roboflow.model_monitoring_exporter.v1.export_to_model_monitoring_request"
)
def test_run_not_in_reporting_range_success(
    export_to_model_monitoring_request_mock: MagicMock,
) -> None:
    # given
    export_to_model_monitoring_request_mock.return_value = (
        False,
        "Not in reporting range, skipping report. (Ok)",
    )
    cache = MemoryCache()
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
    block = RoboflowModelMonitoringExporterBlockV1(
        cache=cache,
        api_key=api_key,
        background_tasks=None,
        thread_pool_executor=None,
    )
    result = block.run(
        fire_and_forget=True,
        frequency=1,
        predictions=predictions,
    )

    # then
    assert result == {
        "error_status": False,
        "message": "Not in reporting range, skipping report. (Ok)",
    }, "Expected successful upload"
    assert cache.get("roboflow_model_monitoring_last_report_time") is not None


@patch(
    "inference.core.workflows.core_steps.sinks.roboflow.model_monitoring_exporter.v1.export_to_model_monitoring_request"
)
@patch(
    "inference.core.workflows.core_steps.sinks.roboflow.model_monitoring_exporter.v1.get_roboflow_workspace"
)
def test_run_in_reporting_range_success(
    get_roboflow_workspace_mock: MagicMock,
    export_to_model_monitoring_request_mock: MagicMock,
) -> None:
    # given
    export_to_model_monitoring_request_mock.return_value = (
        False,
        "Data exported successfully",
    )
    get_roboflow_workspace_mock.return_value = "workspace-name"
    cache = MemoryCache()
    cache.set(
        "roboflow_model_monitoring_last_report_time",
        datetime(2024, 11, 10, 12, 0, 0).isoformat(),
    )
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
    block = RoboflowModelMonitoringExporterBlockV1(
        cache=cache,
        api_key=api_key,
        background_tasks=None,
        thread_pool_executor=None,
    )
    result = block.run(
        fire_and_forget=False,
        frequency=10,
        predictions=predictions,
    )

    # then
    assert result == {
        "error_status": False,
        "message": "Data exported successfully",
    }, "Expected successful upload"
    export_to_model_monitoring_request_mock.assert_called_once_with(
        cache=cache,
        api_key=api_key,
        predictions=predictions,
    )
    assert (
        cache.get("roboflow_model_monitoring_last_report_time")
        != datetime(2024, 11, 10, 12, 0, 0).isoformat()
    )


@patch(
    "inference.core.workflows.core_steps.sinks.roboflow.model_monitoring_exporter.v1.export_inference_to_model_monitoring"
)
@patch(
    "inference.core.workflows.core_steps.sinks.roboflow.model_monitoring_exporter.v1.get_roboflow_workspace"
)
def test_export_to_model_monitoring_request_failure(
    get_roboflow_workspace_mock: MagicMock,
    export_to_model_monitoring_request_mock: MagicMock,
) -> None:
    # given
    export_to_model_monitoring_request_mock.side_effect = Exception("API error")
    get_roboflow_workspace_mock.return_value = "workspace-name"
    cache = MemoryCache()
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
    result = export_to_model_monitoring_request(
        cache=cache,
        api_key=api_key,
        predictions=predictions,
    )

    # then
    assert result[0] is True, "Expected upload to fail"
    assert (
        "Error while uploading inference data" in result[1]
    ), "Expected error message in result"
    assert cache.get("roboflow_model_monitoring_last_report_time") is None


@patch(
    "inference.core.workflows.core_steps.sinks.roboflow.model_monitoring_exporter.v1.get_roboflow_workspace"
)
@patch(
    "inference.core.workflows.core_steps.sinks.roboflow.model_monitoring_exporter.v1.export_to_model_monitoring_request"
)
def test_run_when_not_in_reporting_range(
    export_to_model_monitoring_request_mock: MagicMock,
    get_roboflow_workspace_mock: MagicMock,
) -> None:
    # given
    export_to_model_monitoring_request_mock.return_value = (
        False,
        "Not in reporting range, skipping report. (Ok)",
    )
    get_roboflow_workspace_mock.return_value = "workspace-name"
    cache = MemoryCache()
    cache.set(
        "roboflow_model_monitoring_last_report_time",
        datetime(2024, 11, 10, 12, 0, 0).isoformat(),
    )
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
    block = RoboflowModelMonitoringExporterBlockV1(
        cache=cache,
        api_key=api_key,
        background_tasks=None,
        thread_pool_executor=None,
    )
    result = block.run(
        fire_and_forget=False,
        frequency=10,
        predictions=predictions,
    )

    # then
    assert result == {
        "error_status": False,
        "message": "Not in reporting range, skipping report. (Ok)",
    }, "Expected skipping report due to frequency"
    assert cache.get("roboflow_model_monitoring_last_report_time") is not None


@patch(
    "inference.core.workflows.core_steps.sinks.roboflow.model_monitoring_exporter.v1.get_roboflow_workspace"
)
@patch(
    "inference.core.workflows.core_steps.sinks.roboflow.model_monitoring_exporter.v1.export_to_model_monitoring_request"
)
def test_run_when_fire_and_forget_with_background_tasks(
    export_to_model_monitoring_request_mock: MagicMock,
    get_roboflow_workspace_mock: MagicMock,
) -> None:
    # given
    background_tasks = BackgroundTasks()
    export_to_model_monitoring_request_mock.return_value = (
        False,
        "Inference data upload was successful",
    )
    get_roboflow_workspace_mock.return_value = "workspace-name"
    cache = MemoryCache()
    cache.set(
        "roboflow_model_monitoring_last_report_time",
        datetime(2024, 11, 10, 12, 0, 0).isoformat(),
    )
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
    block = RoboflowModelMonitoringExporterBlockV1(
        cache=cache,
        api_key=api_key,
        background_tasks=background_tasks,
        thread_pool_executor=None,
    )
    result = block.run(
        fire_and_forget=True,
        frequency=10,
        predictions=predictions,
    )

    # then
    assert result == {
        "error_status": False,
        "message": "Reporting happens in the background task",
    }, "Expected background task to be added"
    assert len(background_tasks.tasks) == 1, "Expected one background task"


@patch(
    "inference.core.workflows.core_steps.sinks.roboflow.model_monitoring_exporter.v1.get_roboflow_workspace"
)
@patch(
    "inference.core.workflows.core_steps.sinks.roboflow.model_monitoring_exporter.v1.export_to_model_monitoring_request"
)
def test_run_when_fire_and_forget_with_thread_pool(
    export_to_model_monitoring_request_mock: MagicMock,
    get_roboflow_workspace_mock: MagicMock,
) -> None:
    # given
    with ThreadPoolExecutor() as thread_pool_executor:
        export_to_model_monitoring_request_mock.return_value = (
            False,
            "Inference data upload was successful",
        )
        get_roboflow_workspace_mock.return_value = "workspace-name"
        cache = MemoryCache()
        cache.set(
            "roboflow_model_monitoring_last_report_time",
            datetime(2024, 11, 10, 12, 0, 0).isoformat(),
        )
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
        block = RoboflowModelMonitoringExporterBlockV1(
            cache=cache,
            api_key=api_key,
            background_tasks=None,
            thread_pool_executor=thread_pool_executor,
        )
        result = block.run(
            fire_and_forget=True,
            frequency=10,
            predictions=predictions,
        )

        # then
        assert result == {
            "error_status": False,
            "message": "Reporting happens in the background task",
        }, "Expected background task to be added"

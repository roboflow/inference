from unittest import mock

import pytest
import supervision as sv

from inference.core.workflows.core_steps.loader import load_blocks
from inference.core.workflows.core_steps.sinks.local_file.v1 import (
    BlockManifest as LocalFileManifestV1,
)
from inference.core.workflows.core_steps.sinks.local_file.v2 import (
    BlockManifest as LocalFileManifestV2,
)
from inference.core.workflows.core_steps.sinks.local_file.v2 import LocalFileSinkBlockV2
from inference.core.workflows.core_steps.sinks.noop import disabled_sink_response
from inference.core.workflows.core_steps.sinks.onvif_movement.v1 import (
    BlockManifest as ONVIFManifestV1,
)
from inference.core.workflows.core_steps.sinks.onvif_movement.v2 import (
    BlockManifest as ONVIFManifestV2,
)
from inference.core.workflows.core_steps.sinks.onvif_movement.v2 import ONVIFSinkBlockV2
from inference.core.workflows.core_steps.sinks.roboflow.custom_metadata.v1 import (
    BlockManifest as CustomMetadataManifestV1,
)
from inference.core.workflows.core_steps.sinks.roboflow.custom_metadata.v2 import (
    BlockManifest as CustomMetadataManifestV2,
)
from inference.core.workflows.core_steps.sinks.roboflow.custom_metadata.v2 import (
    RoboflowCustomMetadataBlockV2,
)
from inference.core.workflows.core_steps.sinks.roboflow.model_monitoring_inference_aggregator.v1 import (
    BlockManifest as ModelMonitoringManifestV1,
)
from inference.core.workflows.core_steps.sinks.roboflow.model_monitoring_inference_aggregator.v2 import (
    BlockManifest as ModelMonitoringManifestV2,
)
from inference.core.workflows.core_steps.sinks.roboflow.model_monitoring_inference_aggregator.v2 import (
    ModelMonitoringInferenceAggregatorBlockV2,
)
from inference.core.workflows.core_steps.sinks.s3.v1 import (
    BlockManifest as S3ManifestV1,
)
from inference.core.workflows.core_steps.sinks.s3.v2 import (
    BlockManifest as S3ManifestV2,
)
from inference.core.workflows.core_steps.sinks.s3.v2 import S3SinkBlockV2

CORE_MANIFESTS = [
    (LocalFileManifestV1, LocalFileManifestV2, "roboflow_core/local_file_sink@v2"),
    (ONVIFManifestV1, ONVIFManifestV2, "roboflow_core/onvif_sink@v2"),
    (
        CustomMetadataManifestV1,
        CustomMetadataManifestV2,
        "roboflow_core/roboflow_custom_metadata@v2",
    ),
    (
        ModelMonitoringManifestV1,
        ModelMonitoringManifestV2,
        "roboflow_core/model_monitoring_inference_aggregator@v2",
    ),
    (S3ManifestV1, S3ManifestV2, "roboflow_core/s3_sink@v2"),
]


@pytest.mark.parametrize("v1_manifest,v2_manifest,expected_type", CORE_MANIFESTS)
def test_v2_manifest_adds_noop_without_changing_v1(
    v1_manifest,
    v2_manifest,
    expected_type: str,
) -> None:
    assert "disable_sink" not in v1_manifest.model_fields
    assert v2_manifest.model_fields["disable_sink"].default is False
    assert v2_manifest.model_fields["type"].annotation.__args__ == (expected_type,)
    assert v2_manifest.model_config["json_schema_extra"]["version"] == "v2"


def test_all_core_v2_sinks_are_registered() -> None:
    registered_blocks = set(load_blocks())

    assert {
        LocalFileSinkBlockV2,
        ONVIFSinkBlockV2,
        RoboflowCustomMetadataBlockV2,
        ModelMonitoringInferenceAggregatorBlockV2,
        S3SinkBlockV2,
    } <= registered_blocks


def test_local_file_noop_precedes_environment_and_file_access() -> None:
    block = LocalFileSinkBlockV2(
        allow_access_to_file_system=False,
        allowed_write_directory=None,
    )
    block._verify_write_access_to_directory = mock.MagicMock()

    result = block.run(
        content="content",
        file_type="txt",
        output_mode="separate_files",
        target_directory="/not/allowed",
        file_name_prefix="result",
        max_entries_per_file=1,
        disable_sink=True,
    )

    assert result == disabled_sink_response()
    block._verify_write_access_to_directory.assert_not_called()


def test_onvif_noop_preserves_predictions_without_camera_access() -> None:
    block = object.__new__(ONVIFSinkBlockV2)
    block.event_loop = mock.MagicMock()
    block.get_camera = mock.MagicMock()
    predictions = sv.Detections.empty()

    result = block.run(
        predictions=predictions,
        camera_ip="invalid",
        camera_port=80,
        camera_username="user",
        camera_password="password",
        movement_type="Follow",
        default_position_preset=None,
        zoom_if_able=True,
        follow_tracker=True,
        dead_zone=10,
        camera_update_rate_limit=10,
        flip_y_movement=False,
        flip_x_movement=False,
        move_to_position_after_idle_seconds=0,
        pid_kp=1,
        pid_ki=0,
        pid_kd=0,
        minimum_camera_speed=0,
        simulate_variable_speed=False,
        disable_sink=True,
    )

    assert result == {"predictions": predictions, "seeking": False}
    block.get_camera.assert_not_called()


def test_custom_metadata_noop_precedes_api_key_and_task_access() -> None:
    background_tasks = mock.MagicMock()
    executor = mock.MagicMock()
    cache = mock.MagicMock()
    block = RoboflowCustomMetadataBlockV2(
        cache=cache,
        api_key=None,
        background_tasks=background_tasks,
        thread_pool_executor=executor,
    )

    result = block.run(
        fire_and_forget=True,
        field_name="field",
        field_value="value",
        predictions={},
        disable_sink=True,
    )

    assert result == disabled_sink_response()
    background_tasks.add_task.assert_not_called()
    executor.submit.assert_not_called()
    cache.assert_not_called()


def test_model_monitoring_noop_precedes_api_key_cache_and_aggregation() -> None:
    cache = mock.MagicMock()
    block = ModelMonitoringInferenceAggregatorBlockV2(
        cache=cache,
        api_key=None,
        background_tasks=mock.MagicMock(),
        thread_pool_executor=mock.MagicMock(),
    )
    block._predictions_aggregator = mock.MagicMock()

    result = block.run(
        fire_and_forget=True,
        predictions={"top": "class"},
        frequency=1,
        unique_aggregator_key="key",
        model_id="model/1",
        disable_sink=True,
    )

    assert result == disabled_sink_response()
    cache.assert_not_called()
    block._predictions_aggregator.collect.assert_not_called()


@mock.patch("inference.core.workflows.core_steps.sinks.s3.v1.create_s3_client")
def test_s3_noop_precedes_client_and_buffer_access(
    create_s3_client: mock.MagicMock,
) -> None:
    block = S3SinkBlockV2()

    result = block.run(
        content="content",
        file_type="txt",
        output_mode="append_log",
        bucket_name="bucket",
        s3_prefix="prefix",
        file_name_prefix="result",
        max_entries_per_file=10,
        disable_sink=True,
    )

    assert result == disabled_sink_response()
    assert block._buffer == []
    assert block._entries_in_buffer == 0
    create_s3_client.assert_not_called()

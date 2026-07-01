from unittest.mock import MagicMock, patch

import pytest

from inference.enterprise.workflows.enterprise_blocks.streams.set_stream_camera_parameters.v1 import (
    SetStreamCameraParametersBlockManifest,
    SetStreamCameraParametersBlockV1,
)
from inference.enterprise.workflows.stream_camera_parameters.entities import (
    ApplyCameraParametersResult,
)
from inference.enterprise.workflows.stream_camera_parameters.service import (
    StreamCameraParametersError,
    apply_camera_register_parameters,
    resolve_pipeline_id,
)


class TestSetStreamCameraParametersBlockManifest:

    def test_manifest_validates(self):
        raw = {
            "type": "roboflow_enterprise/set_stream_camera_parameters@v1",
            "name": "camera_params",
            "parameters": {"v4l2_camera_properties": {"exposure_absolute": 200}},
            "depends_on": "$steps.plc_read.output",
        }
        manifest = SetStreamCameraParametersBlockManifest.model_validate(raw)
        assert manifest.only_if_changed is True
        assert manifest.persist is False


class TestResolvePipelineId:

    @patch(
        "inference.enterprise.workflows.stream_camera_parameters.service.edge_client.list_pipeline_ids"
    )
    def test_uses_single_active_pipeline(self, mock_list):
        mock_list.return_value = ["encoded-stream"]
        assert resolve_pipeline_id(None) == "encoded-stream"

    @patch(
        "inference.enterprise.workflows.stream_camera_parameters.service.edge_client.list_pipeline_ids"
    )
    def test_requires_stream_name_for_multiple_pipelines(self, mock_list):
        mock_list.return_value = ["a", "b"]
        with pytest.raises(StreamCameraParametersError):
            resolve_pipeline_id(None)

    def test_encodes_stream_name(self):
        assert resolve_pipeline_id("line scan") == "line%20scan"


class TestApplyCameraRegisterParameters:

    @patch(
        "inference.enterprise.workflows.stream_camera_parameters.service.edge_client.post_camera_parameters"
    )
    @patch(
        "inference.enterprise.workflows.stream_camera_parameters.service.resolve_pipeline_id"
    )
    def test_applies_parameters(self, mock_resolve, mock_post):
        mock_resolve.return_value = "stream-1"
        mock_post.return_value = ApplyCameraParametersResult(
            success=True,
            applied=["exposure_absolute"],
        )

        result = apply_camera_register_parameters(
            {"v4l2_camera_properties": {"exposure_absolute": 200}}
        )

        assert result.success is True
        mock_post.assert_called_once_with(
            "stream-1",
            {"v4l2_camera_properties": {"exposure_absolute": 200}},
            persist=False,
            only_if_changed=True,
        )

    def test_rejects_empty_parameters(self):
        result = apply_camera_register_parameters({})
        assert result.success is False


class TestSetStreamCameraParametersBlockV1:

    @patch(
        "inference.enterprise.workflows.enterprise_blocks.streams.set_stream_camera_parameters.v1.apply_camera_register_parameters"
    )
    def test_run_returns_service_outputs(self, mock_apply):
        mock_apply.return_value = ApplyCameraParametersResult(
            success=True,
            applied=["AcquisitionLineRate"],
            skipped=False,
            message="",
        )
        block = SetStreamCameraParametersBlockV1()

        output = block.run(
            parameters={"AcquisitionLineRate": 128000},
            stream_name="",
            persist=False,
            only_if_changed=True,
            depends_on=MagicMock(),
        )

        assert output["success"] is True
        assert output["applied"] == ["AcquisitionLineRate"]
        mock_apply.assert_called_once_with(
            {"AcquisitionLineRate": 128000},
            stream_name=None,
            persist=False,
            only_if_changed=True,
        )

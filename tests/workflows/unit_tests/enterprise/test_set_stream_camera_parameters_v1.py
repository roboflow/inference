from unittest.mock import MagicMock, patch

import pytest

from inference.enterprise.workflows.enterprise_blocks.streams.set_stream_camera_parameters.v1 import (
    SetStreamCameraParametersBlockManifest,
    SetStreamCameraParametersBlockV1,
)
from inference.enterprise.workflows.stream_camera_parameters.entities import (
    ApplyCameraParametersResult,
)
from inference.enterprise.workflows.stream_camera_parameters.register_catalog import (
    build_parameter_delta,
    registers_for_camera_family,
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
            "value": "$inputs.focus_value",
            "register": "focus",
            "camera_family": "ai1",
            "stream_name": "aione",
            "depends_on": "$inputs.image",
        }
        manifest = SetStreamCameraParametersBlockManifest.model_validate(raw)
        assert manifest.only_if_changed is True
        assert manifest.persist is False


class TestRegisterCatalog:

    def test_focus_delta_for_ai1(self):
        delta = build_parameter_delta("focus", 42, camera_family="ai1")
        assert delta == {"v4l2_camera_properties": {"lens_position": 42}}

    def test_line_rate_for_lucid_line_scan(self):
        delta = build_parameter_delta("line_rate", 128000, camera_family="lucid_line_scan")
        assert delta == {"AcquisitionLineRate": 128000}

    def test_registers_for_ai1_include_focus(self):
        assert "focus" in registers_for_camera_family("ai1")


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
            applied=["lens_position"],
        )

        result = apply_camera_register_parameters(
            {"v4l2_camera_properties": {"lens_position": 25}}
        )

        assert result.success is True
        mock_post.assert_called_once_with(
            "stream-1",
            {"v4l2_camera_properties": {"lens_position": 25}},
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
    def test_run_builds_register_delta_from_value(self, mock_apply):
        mock_apply.return_value = ApplyCameraParametersResult(
            success=True,
            applied=["lens_position"],
            skipped=False,
            message="",
        )
        block = SetStreamCameraParametersBlockV1()

        output = block.run(
            value=50,
            register_key="focus",
            camera_family="ai1",
            stream_name="aione",
            device_id="",
            manual_register_key="",
            parameters={},
            persist=False,
            only_if_changed=True,
            depends_on=MagicMock(),
        )

        assert output["success"] is True
        mock_apply.assert_called_once_with(
            {"v4l2_camera_properties": {"lens_position": 50}},
            stream_name="aione",
            persist=False,
            only_if_changed=True,
        )

    @patch(
        "inference.enterprise.workflows.enterprise_blocks.streams.set_stream_camera_parameters.v1.apply_camera_register_parameters"
    )
    def test_run_uses_raw_parameters_when_provided(self, mock_apply):
        mock_apply.return_value = ApplyCameraParametersResult(success=True, applied=["custom"])
        block = SetStreamCameraParametersBlockV1()

        block.run(
            value=1,
            register_key="focus",
            camera_family="ai1",
            stream_name="",
            device_id="",
            manual_register_key="",
            parameters={"AcquisitionLineRate": 128000},
            persist=False,
            only_if_changed=False,
            depends_on=MagicMock(),
        )

        mock_apply.assert_called_once_with(
            {"AcquisitionLineRate": 128000},
            stream_name=None,
            persist=False,
            only_if_changed=False,
        )

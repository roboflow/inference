from unittest.mock import MagicMock, patch

import pytest

from inference.enterprise.workflows.enterprise_blocks.streams.set_stream_camera_parameters.v1 import (
    SetStreamCameraParametersBlockManifest,
    SetStreamCameraParametersBlockV1,
)
from inference.enterprise.workflows.edge_camera_parameters_client.entities import (
    ApplyCameraParametersResult,
)
from inference.enterprise.workflows.edge_camera_parameters_client.edge_client import (
    list_pipeline_ids,
)
from inference.enterprise.workflows.edge_camera_parameters_client.register_catalog import (
    build_parameter_delta,
    get_register_binding,
    get_register_labels_map,
    registers_for_camera_family,
)
from inference.enterprise.workflows.edge_camera_parameters_client.service import (
    EdgeCameraParametersError,
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

    def test_line_rate_for_basler_line_scan(self):
        delta = build_parameter_delta("line_rate", 21000, camera_family="basler_line_scan")
        assert delta == {"AcquisitionLineRate": 21000}

    def test_lines_per_frame_for_basler_line_scan(self):
        delta = build_parameter_delta("lines_per_frame", 4096, camera_family="basler_line_scan")
        assert delta == {"LinesPerFrame": 4096}

    def test_registers_for_basler_line_scan_include_line_rate(self):
        registers = registers_for_camera_family("basler_line_scan")
        assert "line_rate" in registers
        assert "lines_per_frame" in registers
        assert "exposure_time" in registers

    def test_registers_for_ai1_include_focus(self):
        assert "focus" in registers_for_camera_family("ai1")

    def test_exposure_time_for_ai1_includes_exposure_mode(self):
        delta = build_parameter_delta("exposure_time", 175, camera_family="ai1")
        assert delta == {
            "v4l2_camera_properties": {"exposure_mode": 0, "exposure_time": 175}
        }

    def test_register_labels_loaded_from_json(self):
        labels = get_register_labels_map()
        assert labels["exposure_time"] == "Exposure time"

    def test_ai1_exposure_schema_from_json(self):
        binding = get_register_binding("exposure_time", "ai1")
        assert binding["valueSchema"]["max"] == 300


class TestEdgeClient:

    @patch("inference.enterprise.workflows.edge_camera_parameters_client.edge_client.requests.get")
    def test_list_pipeline_ids_parses_roboflow_edge_response(self, mock_get):
        mock_get.return_value.json.return_value = {
            "success": True,
            "data": {"pipelines": [{"pipeline_id": "aione", "status": "RUNNING"}]},
        }
        mock_get.return_value.raise_for_status = MagicMock()

        assert list_pipeline_ids("http://127.0.0.1:8000") == ["aione"]


class TestResolvePipelineId:

    @patch(
        "inference.enterprise.workflows.edge_camera_parameters_client.service.edge_client.list_pipeline_ids"
    )
    def test_uses_single_active_pipeline(self, mock_list):
        mock_list.return_value = ["encoded-stream"]
        assert resolve_pipeline_id(None) == "encoded-stream"

    @patch(
        "inference.enterprise.workflows.edge_camera_parameters_client.service.edge_client.list_pipeline_ids"
    )
    def test_requires_stream_name_for_multiple_pipelines(self, mock_list):
        mock_list.return_value = ["a", "b"]
        with pytest.raises(EdgeCameraParametersError):
            resolve_pipeline_id(None)

    def test_encodes_stream_name(self):
        assert resolve_pipeline_id("line scan") == "line%20scan"


class TestApplyCameraRegisterParameters:

    @patch(
        "inference.enterprise.workflows.edge_camera_parameters_client.service.edge_client.post_camera_parameters"
    )
    @patch(
        "inference.enterprise.workflows.edge_camera_parameters_client.service.resolve_pipeline_id"
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
            register="focus",
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
            register="focus",
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

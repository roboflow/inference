from unittest.mock import MagicMock, patch

from inference.enterprise.workflows.enterprise_blocks.sinks.PLCethernetIP.v1 import (
    PLCBlockManifest,
    PLCBlockV1,
)


COMMON_MANIFEST_KWARGS = dict(
    type="roboflow_core/sinks@v1",
    name="plc_step",
    plc_ip="192.168.1.10",
    mode="write",
    depends_on="$steps.some_step.predictions",
)


def test_manifest_accepts_static_dict_for_tags_to_write():
    manifest = PLCBlockManifest(
        **COMMON_MANIFEST_KWARGS,
        tags_to_write={"camera_fault": True, "defect_count": 5},
    )
    assert manifest.tags_to_write == {"camera_fault": True, "defect_count": 5}


def test_manifest_accepts_step_selector_for_tags_to_write():
    manifest = PLCBlockManifest(
        **COMMON_MANIFEST_KWARGS,
        tags_to_write="$steps.some_step.output",
    )
    assert manifest.tags_to_write == "$steps.some_step.output"


def test_manifest_accepts_input_selector_for_tags_to_write():
    manifest = PLCBlockManifest(
        **COMMON_MANIFEST_KWARGS,
        tags_to_write="$inputs.my_dict",
    )
    assert manifest.tags_to_write == "$inputs.my_dict"


def test_manifest_describe_outputs():
    outputs = PLCBlockManifest.describe_outputs()
    assert len(outputs) == 1
    assert outputs[0].name == "plc_results"


def test_manifest_get_execution_engine_compatibility():
    assert PLCBlockManifest.get_execution_engine_compatibility() == ">=1.0.0,<2.0.0"


@patch(
    "inference.enterprise.workflows.enterprise_blocks.sinks.PLCethernetIP.v1.pylogix"
)
def test_successful_read_mode(mock_pylogix):
    mock_comm = MagicMock()
    mock_pylogix.PLC.return_value.__enter__ = MagicMock(return_value=mock_comm)
    mock_pylogix.PLC.return_value.__exit__ = MagicMock(return_value=False)

    def fake_read(tag):
        resp = MagicMock()
        resp.Status = "Success"
        resp.Value = 42
        return resp

    mock_comm.Read.side_effect = fake_read

    block = PLCBlockV1()
    result = block.run(
        plc_ip="192.168.1.10",
        mode="read",
        tags_to_read=["sensor_1", "sensor_2"],
        tags_to_write={},
        depends_on=None,
    )

    assert "plc_results" in result
    plc_output = result["plc_results"][0]
    assert plc_output["read"]["sensor_1"] == 42
    assert plc_output["read"]["sensor_2"] == 42


@patch(
    "inference.enterprise.workflows.enterprise_blocks.sinks.PLCethernetIP.v1.pylogix"
)
def test_successful_write_mode(mock_pylogix):
    mock_comm = MagicMock()
    mock_pylogix.PLC.return_value.__enter__ = MagicMock(return_value=mock_comm)
    mock_pylogix.PLC.return_value.__exit__ = MagicMock(return_value=False)

    def fake_write(tag, value):
        resp = MagicMock()
        resp.Status = "Success"
        return resp

    mock_comm.Write.side_effect = fake_write

    block = PLCBlockV1()
    result = block.run(
        plc_ip="192.168.1.10",
        mode="write",
        tags_to_read=[],
        tags_to_write={"camera_fault": 1, "defect_count": 5},
        depends_on=None,
    )

    assert "plc_results" in result
    plc_output = result["plc_results"][0]
    assert plc_output["write"]["camera_fault"] == "WriteSuccess"
    assert plc_output["write"]["defect_count"] == "WriteSuccess"


@patch(
    "inference.enterprise.workflows.enterprise_blocks.sinks.PLCethernetIP.v1.pylogix"
)
def test_read_and_write_mode(mock_pylogix):
    mock_comm = MagicMock()
    mock_pylogix.PLC.return_value.__enter__ = MagicMock(return_value=mock_comm)
    mock_pylogix.PLC.return_value.__exit__ = MagicMock(return_value=False)

    def fake_read(tag):
        resp = MagicMock()
        resp.Status = "Success"
        resp.Value = 99
        return resp

    def fake_write(tag, value):
        resp = MagicMock()
        resp.Status = "Success"
        return resp

    mock_comm.Read.side_effect = fake_read
    mock_comm.Write.side_effect = fake_write

    block = PLCBlockV1()
    result = block.run(
        plc_ip="192.168.1.10",
        mode="read_and_write",
        tags_to_read=["sensor_1"],
        tags_to_write={"output_1": 10},
        depends_on=None,
    )

    plc_output = result["plc_results"][0]
    assert "read" in plc_output
    assert "write" in plc_output
    assert plc_output["read"]["sensor_1"] == 99
    assert plc_output["write"]["output_1"] == "WriteSuccess"


@patch(
    "inference.enterprise.workflows.enterprise_blocks.sinks.PLCethernetIP.v1.pylogix"
)
def test_read_failure(mock_pylogix):
    mock_comm = MagicMock()
    mock_pylogix.PLC.return_value.__enter__ = MagicMock(return_value=mock_comm)
    mock_pylogix.PLC.return_value.__exit__ = MagicMock(return_value=False)

    def fake_read(tag):
        resp = MagicMock()
        resp.Status = "Error"
        return resp

    mock_comm.Read.side_effect = fake_read

    block = PLCBlockV1()
    result = block.run(
        plc_ip="192.168.1.10",
        mode="read",
        tags_to_read=["bad_tag"],
        tags_to_write={},
        depends_on=None,
    )

    assert result["plc_results"][0]["read"]["bad_tag"] == "ReadFailure"


@patch(
    "inference.enterprise.workflows.enterprise_blocks.sinks.PLCethernetIP.v1.pylogix"
)
def test_write_failure(mock_pylogix):
    mock_comm = MagicMock()
    mock_pylogix.PLC.return_value.__enter__ = MagicMock(return_value=mock_comm)
    mock_pylogix.PLC.return_value.__exit__ = MagicMock(return_value=False)

    def fake_write(tag, value):
        resp = MagicMock()
        resp.Status = "Error"
        return resp

    mock_comm.Write.side_effect = fake_write

    block = PLCBlockV1()
    result = block.run(
        plc_ip="192.168.1.10",
        mode="write",
        tags_to_read=[],
        tags_to_write={"bad_tag": 1},
        depends_on=None,
    )

    assert result["plc_results"][0]["write"]["bad_tag"] == "WriteFailure"


@patch(
    "inference.enterprise.workflows.enterprise_blocks.sinks.PLCethernetIP.v1.pylogix"
)
def test_read_exception(mock_pylogix):
    mock_comm = MagicMock()
    mock_pylogix.PLC.return_value.__enter__ = MagicMock(return_value=mock_comm)
    mock_pylogix.PLC.return_value.__exit__ = MagicMock(return_value=False)

    mock_comm.Read.side_effect = RuntimeError("connection lost")

    block = PLCBlockV1()
    result = block.run(
        plc_ip="192.168.1.10",
        mode="read",
        tags_to_read=["flaky_tag"],
        tags_to_write={},
        depends_on=None,
    )

    assert result["plc_results"][0]["read"]["flaky_tag"] == "ReadFailure"


@patch(
    "inference.enterprise.workflows.enterprise_blocks.sinks.PLCethernetIP.v1.pylogix"
)
def test_write_exception(mock_pylogix):
    mock_comm = MagicMock()
    mock_pylogix.PLC.return_value.__enter__ = MagicMock(return_value=mock_comm)
    mock_pylogix.PLC.return_value.__exit__ = MagicMock(return_value=False)

    mock_comm.Write.side_effect = RuntimeError("connection lost")

    block = PLCBlockV1()
    result = block.run(
        plc_ip="192.168.1.10",
        mode="write",
        tags_to_read=[],
        tags_to_write={"flaky_tag": 1},
        depends_on=None,
    )

    assert result["plc_results"][0]["write"]["flaky_tag"] == "WriteFailure"

import inspect
from unittest import mock

import pytest
import supervision as sv

from inference.core.workflows.core_steps.loader import (
    REGISTERED_INITIALIZERS,
    load_blocks,
)
from inference.enterprise.workflows.enterprise_blocks.loader import (
    load_enterprise_blocks,
)
from inference.enterprise.workflows.enterprise_blocks.sinks.PLC_modbus.v1 import (
    ModbusTCPBlockV1,
)
from inference.roboflow_workflows_plugin.loader import (
    load_blocks as load_roboflow_platform_blocks,
)


def _load_sink_blocks():
    result = []
    for block in (
        load_blocks() + load_enterprise_blocks() + load_roboflow_platform_blocks()
    ):
        block_type = block.get_manifest().model_json_schema().get("block_type")
        if block_type in {"sink", "sinks"}:
            result.append(block)
    # This deprecated read/write block predates consistent sink classification.
    result.append(ModbusTCPBlockV1)
    return list(dict.fromkeys(result))


SINK_BLOCKS = _load_sink_blocks()

RELOCATED_SINK_TYPES = {
    "roboflow_core/roboflow_dataset_upload@v1",
    "roboflow_core/roboflow_dataset_upload@v2",
    "roboflow_core/roboflow_custom_metadata@v1",
    "roboflow_core/model_monitoring_inference_aggregator@v1",
    "roboflow_core/asset_library_attributes@v1",
    "roboflow_core/roboflow_vision_events@v1",
    "roboflow_core/vision_event_bundle@v1",
}


def test_relocated_roboflow_sinks_are_still_in_the_policy_inventory() -> None:
    covered = {
        block.get_manifest().model_fields["type"].annotation.__args__[0]
        for block in SINK_BLOCKS
    }
    assert RELOCATED_SINK_TYPES <= covered, RELOCATED_SINK_TYPES - covered


def test_sink_disabling_defaults_to_false() -> None:
    assert REGISTERED_INITIALIZERS["disable_sinks"] is False


@pytest.mark.parametrize("block", SINK_BLOCKS)
def test_sink_accepts_disable_sinks_init_parameter(block) -> None:
    assert "disable_sinks" in block.get_init_parameters()
    assert "disable_sinks" in inspect.signature(block).parameters


@pytest.mark.parametrize("block", SINK_BLOCKS)
def test_injected_policy_noops_sink_with_declared_output_shape(block) -> None:
    init_arguments = {}
    for name, parameter in inspect.signature(block).parameters.items():
        if name == "disable_sinks":
            init_arguments[name] = True
        elif parameter.default is inspect.Parameter.empty:
            init_arguments[name] = mock.MagicMock()
    instance = block(**init_arguments)
    run_arguments = {}
    for name, parameter in inspect.signature(instance.run).parameters.items():
        if parameter.default is not inspect.Parameter.empty:
            continue
        if name == "disable_sink":
            run_arguments[name] = False
        elif name == "mode":
            run_arguments[name] = "write"
        elif name in {"images", "source_id"}:
            run_arguments[name] = [mock.MagicMock()]
        elif name == "predictions":
            run_arguments[name] = sv.Detections.empty()
        else:
            run_arguments[name] = mock.MagicMock()

    result = instance.run(**run_arguments)

    if isinstance(result, list) and result:
        result = result[0]
    assert result is not None
    if isinstance(result, dict):
        declared_outputs = {
            output.name for output in block.get_manifest().describe_outputs()
        }
        assert declared_outputs <= set(result)

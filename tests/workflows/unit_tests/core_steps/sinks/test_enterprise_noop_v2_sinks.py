from unittest import mock

import pytest

from inference.core.workflows.core_steps.sinks.noop import disabled_sink_response
from inference.enterprise.workflows.enterprise_blocks.loader import (
    load_enterprise_blocks,
)
from inference.enterprise.workflows.enterprise_blocks.sinks.microsoft_sql_server.v1 import (
    BlockManifest as SQLManifestV1,
)
from inference.enterprise.workflows.enterprise_blocks.sinks.microsoft_sql_server.v2 import (
    BlockManifest as SQLManifestV2,
)
from inference.enterprise.workflows.enterprise_blocks.sinks.microsoft_sql_server.v2 import (
    MicrosoftSQLServerSinkBlockV2,
)
from inference.enterprise.workflows.enterprise_blocks.sinks.mqtt_writer.v1 import (
    BlockManifest as MQTTManifestV1,
)
from inference.enterprise.workflows.enterprise_blocks.sinks.mqtt_writer.v2 import (
    BlockManifest as MQTTManifestV2,
)
from inference.enterprise.workflows.enterprise_blocks.sinks.mqtt_writer.v2 import (
    MQTTWriterSinkBlockV2,
)
from inference.enterprise.workflows.enterprise_blocks.sinks.PLC_modbus.v1 import (
    ModbusTCPBlockManifest as ModbusManifestV1,
)
from inference.enterprise.workflows.enterprise_blocks.sinks.PLC_modbus.v2 import (
    ModbusTCPBlockManifest as ModbusManifestV2,
)
from inference.enterprise.workflows.enterprise_blocks.sinks.PLC_modbus.v2 import (
    ModbusTCPBlockV2,
)
from inference.enterprise.workflows.enterprise_blocks.sinks.PLCethernetIP.v1 import (
    PLCBlockManifest as PLCManifestV1,
)
from inference.enterprise.workflows.enterprise_blocks.sinks.PLCethernetIP.v2 import (
    BlockManifest as PLCManifestV2,
)
from inference.enterprise.workflows.enterprise_blocks.sinks.PLCethernetIP.v2 import (
    PLCBlockV2,
)

ENTERPRISE_MANIFESTS = [
    (ModbusManifestV1, ModbusManifestV2, "roboflow_core/modbus_tcp@v2"),
    (PLCManifestV1, PLCManifestV2, "roboflow_core/sinks@v2"),
    (
        SQLManifestV1,
        SQLManifestV2,
        "roboflow_core/microsoft_sql_server_sink@v2",
    ),
    (MQTTManifestV1, MQTTManifestV2, "mqtt_writer_sink@v2"),
]


@pytest.mark.parametrize("v1_manifest,v2_manifest,expected_type", ENTERPRISE_MANIFESTS)
def test_v2_manifest_adds_noop_without_changing_v1(
    v1_manifest,
    v2_manifest,
    expected_type: str,
) -> None:
    assert "disable_sink" not in v1_manifest.model_fields
    assert v2_manifest.model_fields["disable_sink"].default is False
    assert v2_manifest.model_fields["type"].annotation.__args__ == (expected_type,)
    assert v2_manifest.model_config["json_schema_extra"]["version"] == "v2"


def test_all_enterprise_v2_sinks_are_registered() -> None:
    registered_blocks = set(load_enterprise_blocks())

    assert {
        ModbusTCPBlockV2,
        PLCBlockV2,
        MicrosoftSQLServerSinkBlockV2,
        MQTTWriterSinkBlockV2,
    } <= registered_blocks


@mock.patch(
    "inference.enterprise.workflows.enterprise_blocks.sinks.PLC_modbus.v1.ModbusClient"
)
def test_modbus_noop_precedes_client_access(modbus_client: mock.MagicMock) -> None:
    block = ModbusTCPBlockV2()

    result = block.run(
        plc_ip="invalid",
        plc_port=502,
        mode="read_and_write",
        registers_to_read=[1],
        registers_to_write={1: 2},
        depends_on=None,
        disable_sink=True,
    )

    assert result == {"modbus_results": []}
    modbus_client.assert_not_called()


@mock.patch(
    "inference.enterprise.workflows.enterprise_blocks.sinks.PLCethernetIP.v1.pylogix.PLC"
)
def test_ethernet_ip_noop_precedes_client_access(plc_client: mock.MagicMock) -> None:
    block = PLCBlockV2()

    result = block.run(
        plc_ip="invalid",
        mode="read_and_write",
        tags_to_read=["tag"],
        tags_to_write={"tag": 1},
        depends_on=None,
        disable_sink=True,
    )

    assert result == {"plc_results": []}
    plc_client.assert_not_called()


def test_sql_noop_precedes_task_and_connection_access() -> None:
    background_tasks = mock.MagicMock()
    executor = mock.MagicMock()
    block = MicrosoftSQLServerSinkBlockV2(
        background_tasks=background_tasks,
        thread_pool_executor=executor,
    )
    block._process_data = mock.MagicMock()

    result = block.run(
        host="invalid",
        port=1433,
        database="database",
        table_name="table",
        data={"value": 1},
        disable_sink=True,
    )

    assert result == disabled_sink_response()
    block._process_data.assert_not_called()
    background_tasks.add_task.assert_not_called()
    executor.submit.assert_not_called()


@mock.patch(
    "inference.enterprise.workflows.enterprise_blocks.sinks.mqtt_writer.v1.mqtt.Client"
)
def test_mqtt_noop_precedes_client_access(mqtt_client: mock.MagicMock) -> None:
    block = MQTTWriterSinkBlockV2()

    result = block.run(
        host="invalid",
        port=1883,
        topic="topic",
        message="message",
        disable_sink=True,
    )

    assert result == disabled_sink_response()
    assert block.mqtt_client is None
    mqtt_client.assert_not_called()

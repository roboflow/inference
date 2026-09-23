"""Actual restrictions of the industrial sinks (OPC UA, PLC).

These blocks used to return a known-empty ``get_actual_restrictions()``. The
actual declarations now correct those omissions, while the legacy editor
``get_restrictions()`` stays unchanged. All manifests are real and validated;
nothing here opens a connection to a PLC or an OPC UA server.
"""

import json
from typing import Any, Dict

import pytest
from pydantic import ValidationError
from roboflow_workflows.core_steps.common.workload_presets import (
    COOLDOWN_ACTUAL_RESTRICTION,
    PLC_LAN_ACCESS_ACTUAL_RESTRICTION,
)
from roboflow_workflows.core_steps.sinks.onvif_movement.v1 import (
    NO_LAN_FROM_HOSTED_RESTRICTION,
)
from roboflow_workflows.enterprise_blocks.sinks.opc_writer.v1 import (
    BlockManifest as OPCWriterSinkBlockManifest,
)
from roboflow_workflows.enterprise_blocks.sinks.plc.v1 import (
    PLCReaderBlockManifest,
    PLCWriterBlockManifest,
)
from roboflow_workflows.enterprise_blocks.sinks.PLC_modbus.v1 import (
    ModbusTCPBlockManifest,
)
from roboflow_workflows.enterprise_blocks.sinks.PLCethernetIP.v1 import PLCBlockManifest
from roboflow_workflows.execution_engine.entities.workload import (
    RestrictionMetadata,
    Runtime,
    Severity,
    restriction_metadata_of,
)
from roboflow_workflows.prototypes.block import COOLDOWN_HTTP_SOFT_RESTRICTION

from tests.unit_tests.workload_declaration_helpers import (
    declared_restrictions,
    portable_restrictions,
    portable_restrictions_discovery,
)

LAN_CODE = "requires_lan_access_to_device"
COOLDOWN_CODE = "cooldown_timer_resets_on_stateless_http"
HOSTED_RUNTIMES = {Runtime.HOSTED_SERVERLESS, Runtime.DEDICATED_DEPLOYMENT}


def _opc_writer() -> OPCWriterSinkBlockManifest:
    return OPCWriterSinkBlockManifest.model_validate(
        {
            "type": "roboflow_enterprise/opc_writer_sink@v1",
            "name": "opc_writer",
            "url": "opc.tcp://10.0.0.5:4840",
            "namespace": "http://example.com/ns",
            "object_name": "Line1",
            "variable_name": "Status",
            "value": "ok",
        }
    )


def _modbus() -> ModbusTCPBlockManifest:
    return ModbusTCPBlockManifest.model_validate(
        {
            "type": "roboflow_core/modbus_tcp@v1",
            "name": "modbus",
            "plc_ip": "10.0.1.31",
            "mode": "read",
            "registers_to_read": [1000],
            "depends_on": "$steps.model.predictions",
        }
    )


def _ethernet_ip() -> PLCBlockManifest:
    return PLCBlockManifest.model_validate(
        {
            "type": "roboflow_core/sinks@v1",
            "name": "ethernet_ip",
            "plc_ip": "192.168.1.10",
            "mode": "write",
            "tags_to_write": {"camera_fault": 1},
            "depends_on": "$steps.model.predictions",
        }
    )


def _reader(**overrides: Any) -> PLCReaderBlockManifest:
    fields: Dict[str, Any] = {
        "type": "roboflow_core/plc_reader@v1",
        "name": "plc_reader",
        "tags_to_read": ["camera_msg"],
    }
    fields.update(overrides)
    return PLCReaderBlockManifest.model_validate(fields)


def _writer(**overrides: Any) -> PLCWriterBlockManifest:
    fields: Dict[str, Any] = {
        "type": "roboflow_core/plc_writer@v1",
        "name": "plc_writer",
        "tag": "camera_fault",
        "value": 1,
    }
    fields.update(overrides)
    return PLCWriterBlockManifest.model_validate(fields)


def _assert_lan_restriction(restriction: RestrictionMetadata) -> None:
    assert restriction.code == LAN_CODE
    assert restriction.severity is Severity.HARD
    assert set(restriction.when.runtimes) == HOSTED_RUNTIMES
    # the PLC is contacted from this process wherever models run
    assert restriction.when.step_execution_modes is None
    assert restriction.when.input_modes is None
    assert restriction.when.configuration_equals == {}


def _assert_portable_json(restriction: RestrictionMetadata) -> None:
    payload = json.loads(json.dumps(restriction.model_dump(mode="json")))
    assert RestrictionMetadata.model_validate(payload) == restriction
    assert payload["code"] == restriction.code
    assert payload["severity"] == restriction.severity.value
    # portable declarations carry no free-text note
    assert "note" not in payload


def test_opc_writer_declares_the_actual_cooldown_caveat() -> None:
    manifest = _opc_writer()

    assert declared_restrictions(manifest) == [COOLDOWN_ACTUAL_RESTRICTION]
    discovery = portable_restrictions_discovery(manifest)
    assert discovery.complete is True
    assert discovery.unknown_reasons == []
    [restriction] = discovery.items
    assert restriction.code == COOLDOWN_CODE
    assert restriction.severity is Severity.SOFT
    assert set(restriction.when.runtimes) == HOSTED_RUNTIMES
    # state is lost wherever the model runs, so no LOCAL / REMOTE filter
    assert restriction.when.step_execution_modes is None
    assert restriction.when.input_modes is None
    _assert_portable_json(restriction)


def test_opc_writer_does_not_use_the_legacy_cooldown_preset() -> None:
    declared = declared_restrictions(_opc_writer())

    assert COOLDOWN_HTTP_SOFT_RESTRICTION not in declared
    assert restriction_metadata_of(declared[0]) != restriction_metadata_of(
        COOLDOWN_HTTP_SOFT_RESTRICTION
    )


def test_opc_writer_host_view_keeps_the_unconditional_cooldown_caveat() -> None:
    discovery = _opc_writer().get_actual_restrictions(
        ignore_environment_restrictions=False
    )

    assert discovery.complete is True
    assert discovery.items == [COOLDOWN_ACTUAL_RESTRICTION]


@pytest.mark.parametrize(
    "manifest_class",
    [
        OPCWriterSinkBlockManifest,
        ModbusTCPBlockManifest,
        PLCBlockManifest,
        PLCReaderBlockManifest,
        PLCWriterBlockManifest,
    ],
)
def test_legacy_editor_restrictions_stay_unchanged(manifest_class: type) -> None:
    assert manifest_class.get_restrictions() == []


@pytest.mark.parametrize("build_manifest", [_modbus, _ethernet_ip])
def test_legacy_plc_blocks_declare_plc_reachability(build_manifest) -> None:
    manifest = build_manifest()

    assert declared_restrictions(manifest) == [PLC_LAN_ACCESS_ACTUAL_RESTRICTION]
    discovery = portable_restrictions_discovery(manifest)
    assert discovery.complete is True
    assert discovery.unknown_reasons == []
    [restriction] = discovery.items
    _assert_lan_restriction(restriction)
    _assert_portable_json(restriction)


def test_plc_restriction_shares_the_onvif_code_and_axes_but_not_the_note() -> None:
    assert restriction_metadata_of(
        PLC_LAN_ACCESS_ACTUAL_RESTRICTION
    ) == restriction_metadata_of(NO_LAN_FROM_HOSTED_RESTRICTION)
    assert "PLC" in PLC_LAN_ACCESS_ACTUAL_RESTRICTION.note
    assert "camera" not in PLC_LAN_ACCESS_ACTUAL_RESTRICTION.note.lower()


@pytest.mark.parametrize("build_manifest", [_reader, _writer])
@pytest.mark.parametrize("connection_mode", ["ethernet_ip", "modbus"])
def test_direct_plc_modes_declare_plc_reachability(
    build_manifest, connection_mode: str
) -> None:
    manifest = build_manifest(connection_mode=connection_mode)

    assert declared_restrictions(manifest) == [PLC_LAN_ACCESS_ACTUAL_RESTRICTION]
    discovery = portable_restrictions_discovery(manifest)
    assert discovery.complete is True
    [restriction] = discovery.items
    _assert_lan_restriction(restriction)
    _assert_portable_json(restriction)


@pytest.mark.parametrize("build_manifest", [_reader, _writer])
def test_relay_mode_is_not_labelled_as_requiring_customer_lan(
    build_manifest,
) -> None:
    # the relay is reached over a configurable HTTP address, so nothing proves
    # it sits on a customer LAN
    for manifest in (build_manifest(), build_manifest(connection_mode="relay")):
        discovery = portable_restrictions_discovery(manifest)
        assert discovery.complete is True
        assert discovery.items == []
        assert portable_restrictions(manifest) == []


@pytest.mark.parametrize("build_manifest", [_reader, _writer])
def test_connection_mode_cannot_be_a_selector(build_manifest) -> None:
    # connection_mode is a literal-only field, so no unresolved_selector case
    # can reach get_actual_restrictions()
    with pytest.raises(ValidationError):
        build_manifest(connection_mode="$inputs.connection_mode")

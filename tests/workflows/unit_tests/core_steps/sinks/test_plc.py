from unittest.mock import MagicMock, patch

import pytest
import requests
from pydantic import ValidationError

from inference.enterprise.workflows.enterprise_blocks.sinks.plc.client import (
    relay_base_url,
)
from inference.enterprise.workflows.enterprise_blocks.sinks.plc.direct import (
    _parse_modbus_tag,
)
from inference.enterprise.workflows.enterprise_blocks.sinks.plc.v1 import (
    PLCReaderBlockManifest,
    PLCReaderBlockV1,
    PLCWriterBlockManifest,
    PLCWriterBlockV1,
)

V1 = "inference.enterprise.workflows.enterprise_blocks.sinks.plc.v1"
CLIENT = "inference.enterprise.workflows.enterprise_blocks.sinks.plc.client"
DIRECT = "inference.enterprise.workflows.enterprise_blocks.sinks.plc.direct"


# ================================ helpers ================================


def _http_response(status_code=200, json_body=None, text=""):
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.json.return_value = json_body if json_body is not None else {}
    resp.text = text
    return resp


def _read_batch_body(values):
    return {
        "tags": [{"name": k, "value": v} for k, v in values.items()],
        "count": len(values),
    }


def _write_batch_body(results):
    out = [{"name": k, "success": v} for k, v in results.items()]
    return {"results": out, "success_count": sum(out_i["success"] for out_i in out)}


def _relay_session(v1_requests, client_requests):
    client_requests.exceptions = requests.exceptions
    session = MagicMock()
    v1_requests.Session.return_value = session
    return session


def _plc_response(status="Success", value=None):
    resp = MagicMock()
    resp.Status = status
    resp.Value = value
    return resp


def _setup_pylogix(mock_pylogix):
    # The block holds the pylogix.PLC() instance directly (reused across frames) rather
    # than using it as a context manager, so PLC() returns the comm object itself.
    comm = MagicMock()
    mock_pylogix.PLC.return_value = comm
    return comm


def _modbus_register_response(value, error=False):
    r = MagicMock()
    r.isError.return_value = error
    r.registers = [value]
    return r


def _modbus_bit_response(value, error=False):
    r = MagicMock()
    r.isError.return_value = error
    r.bits = [value]
    return r


def _modbus_write_response(error=False):
    r = MagicMock()
    r.isError.return_value = error
    return r


def _setup_modbus(mock_cls, connect=True):
    # A freshly-created ModbusTcpClient reports connected == False until connect() is
    # called, so the transport's lazy-connect path runs; connect() returns `connect`.
    client = MagicMock()
    client.connected = False
    client.connect.return_value = connect
    mock_cls.return_value = client
    return client


def _malformed_http_response(status_code=200):
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.json.side_effect = ValueError("Expecting value")
    resp.text = "not json"
    return resp


def _writer_manifest_kwargs(**overrides):
    kwargs = {
        "type": "roboflow_core/plc_writer@v1",
        "name": "plc_writer",
        "depends_on": "$steps.prev.output",
        "tag": "camera_fault",
        "value": 1,
    }
    kwargs.update(overrides)
    return kwargs


# ============================ small unit helpers ============================


@pytest.mark.timeout(10)
def test_relay_base_url():
    assert relay_base_url("127.0.0.1", 8007) == "http://127.0.0.1:8007"
    assert relay_base_url("192.168.1.10", 9000) == "http://192.168.1.10:9000"
    # A full URL is used as-is (trailing slash stripped).
    assert relay_base_url("http://host:9001/", 8007) == "http://host:9001"


@pytest.mark.timeout(10)
def test_parse_modbus_tag():
    assert _parse_modbus_tag("holding:100") == ("holding", 100)
    assert _parse_modbus_tag("100") == ("holding", 100)  # bare -> holding
    assert _parse_modbus_tag("coil:0") == ("coil", 0)
    assert _parse_modbus_tag(" Input : 5 ") == ("input", 5)
    with pytest.raises(ValueError):
        _parse_modbus_tag("bogus:1")
    with pytest.raises(ValueError):
        _parse_modbus_tag("holding:xyz")


# ================================ relay mode ================================


@pytest.mark.timeout(10)
@patch(f"{CLIENT}.requests")
@patch(f"{V1}.requests")
def test_relay_read_defaults_to_localhost(v1_requests, client_requests):
    session = _relay_session(v1_requests, client_requests)
    session.post.return_value = _http_response(
        json_body=_read_batch_body({"camera_ready": True, "sku_number": 42})
    )

    result = PLCReaderBlockV1().run(tags_to_read=["camera_ready", "sku_number"])

    assert result["tag_values"] == {"camera_ready": True, "sku_number": 42}
    assert result["error_status"] is False
    assert session.post.call_args.args[0] == "http://127.0.0.1:8007/read_batch"


@pytest.mark.timeout(10)
@patch(f"{CLIENT}.requests")
@patch(f"{V1}.requests")
def test_relay_write_custom_host_port(v1_requests, client_requests):
    session = _relay_session(v1_requests, client_requests)
    session.post.return_value = _http_response(
        json_body=_write_batch_body({"camera_fault": True})
    )

    result = PLCWriterBlockV1().run(
        tag="camera_fault",
        value=True,
        depends_on=None,
        ip_address="192.168.1.50",
        relay_port=9000,
    )

    assert result["write_result"] == "WriteSuccess"
    assert result["error_status"] is False
    assert session.post.call_args.args[0] == "http://192.168.1.50:9000/write_batch"


@pytest.mark.timeout(10)
@patch(f"{CLIENT}.requests")
@patch(f"{V1}.requests")
def test_relay_write_disabled(v1_requests, client_requests):
    session = _relay_session(v1_requests, client_requests)
    result = PLCWriterBlockV1().run(
        tag="x", value=1, depends_on=None, disable_sink=True
    )
    assert result == {"write_result": "", "error_status": False}
    session.post.assert_not_called()


@pytest.mark.timeout(10)
@patch(f"{CLIENT}.requests")
@patch(f"{V1}.requests")
def test_relay_read_unknown_tag_fails_whole_batch(v1_requests, client_requests):
    session = _relay_session(v1_requests, client_requests)
    session.post.return_value = _http_response(
        status_code=404, json_body={"detail": "Unknown tag: nope"}
    )
    result = PLCReaderBlockV1().run(tags_to_read=["ok", "nope"])
    assert result["tag_values"] == {"ok": "ReadFailure", "nope": "ReadFailure"}
    assert result["error_status"] is True


@pytest.mark.timeout(10)
@patch(f"{CLIENT}.requests")
@patch(f"{V1}.requests")
def test_relay_read_per_tag_error_in_200(v1_requests, client_requests):
    session = _relay_session(v1_requests, client_requests)
    session.post.return_value = _http_response(
        json_body={
            "tags": [
                {"name": "good", "value": 5},
                {"name": "bad", "value": None, "error": "Not connected"},
            ],
            "count": 2,
        }
    )
    result = PLCReaderBlockV1().run(tags_to_read=["good", "bad"])
    assert result["tag_values"]["good"] == 5
    assert result["tag_values"]["bad"] == "ReadFailure"
    assert result["error_status"] is True


@pytest.mark.timeout(10)
@patch(f"{CLIENT}.requests")
@patch(f"{V1}.requests")
def test_relay_read_malformed_body(v1_requests, client_requests):
    session = _relay_session(v1_requests, client_requests)
    session.post.return_value = _malformed_http_response()
    result = PLCReaderBlockV1().run(tags_to_read=["t"])
    assert result["tag_values"]["t"] == "ReadFailure"
    assert result["error_status"] is True


@pytest.mark.timeout(10)
@patch(f"{CLIENT}.requests")
@patch(f"{V1}.requests")
def test_relay_read_missing_tag_in_response(v1_requests, client_requests):
    session = _relay_session(v1_requests, client_requests)
    session.post.return_value = _http_response(
        json_body=_read_batch_body({"present": 1})
    )
    result = PLCReaderBlockV1().run(tags_to_read=["present", "absent"])
    assert result["tag_values"]["present"] == 1
    assert result["tag_values"]["absent"] == "ReadFailure"
    assert result["error_status"] is True


@pytest.mark.timeout(10)
@patch(f"{CLIENT}.requests")
@patch(f"{V1}.requests")
def test_relay_read_connection_error(v1_requests, client_requests):
    session = _relay_session(v1_requests, client_requests)
    session.post.side_effect = requests.exceptions.ConnectionError("refused")
    result = PLCReaderBlockV1().run(tags_to_read=["t"])
    assert result["tag_values"]["t"] == "ReadFailure"
    assert result["error_status"] is True


@pytest.mark.timeout(10)
@patch(f"{CLIENT}.requests")
@patch(f"{V1}.requests")
def test_relay_read_value_error_timeout_marks_failure(v1_requests, client_requests):
    # A non-positive timeout (e.g. a selector resolving to 0) makes requests raise a plain
    # ValueError before sending; it must be reported as a failure, not crash the step.
    session = _relay_session(v1_requests, client_requests)
    session.post.side_effect = ValueError(
        "Timeout value connect was 0, but it must be > 0"
    )
    result = PLCReaderBlockV1().run(tags_to_read=["t"], request_timeout=0)
    assert result["tag_values"]["t"] == "ReadFailure"
    assert result["error_status"] is True


@pytest.mark.timeout(10)
@patch(f"{CLIENT}.requests")
@patch(f"{V1}.requests")
def test_relay_read_empty_makes_no_call(v1_requests, client_requests):
    session = _relay_session(v1_requests, client_requests)
    result = PLCReaderBlockV1().run(tags_to_read=[])
    assert result == {"tag_values": {}, "error_status": False}
    session.post.assert_not_called()


@pytest.mark.timeout(10)
@patch(f"{CLIENT}.requests")
@patch(f"{V1}.requests")
def test_relay_write_rejected(v1_requests, client_requests):
    session = _relay_session(v1_requests, client_requests)
    session.post.return_value = _http_response(
        status_code=403, json_body={"detail": "Tag 'ro' is not writable"}
    )
    result = PLCWriterBlockV1().run(tag="ro", value=2, depends_on=None)
    assert result["write_result"] == "WriteFailure"
    assert result["error_status"] is True


@pytest.mark.timeout(10)
@patch(f"{CLIENT}.requests")
@patch(f"{V1}.requests")
def test_relay_write_value_rejected(v1_requests, client_requests):
    session = _relay_session(v1_requests, client_requests)
    session.post.return_value = _http_response(
        json_body={
            "results": [
                {"name": "bad", "success": False, "error": "out of range"},
            ],
            "success_count": 0,
        }
    )
    result = PLCWriterBlockV1().run(tag="bad", value=99999, depends_on=None)
    assert result["write_result"] == "WriteFailure"
    assert result["error_status"] is True


@pytest.mark.timeout(10)
@patch(f"{CLIENT}.requests")
@patch(f"{V1}.requests")
def test_relay_uses_split_connect_read_timeout(v1_requests, client_requests):
    # The relay's synchronous PLC batch can run for seconds, so the caller-supplied value
    # is the *read* budget (default 10s); the connect phase keeps a short fixed timeout so a
    # down relay still fails fast. requests receives them as a (connect, read) tuple.
    session = _relay_session(v1_requests, client_requests)
    session.post.return_value = _http_response(json_body=_read_batch_body({"t": 1}))

    PLCReaderBlockV1().run(tags_to_read=["t"])
    assert session.post.call_args.kwargs["timeout"] == (3, 10)

    PLCReaderBlockV1().run(tags_to_read=["t"], request_timeout=30)
    assert session.post.call_args.kwargs["timeout"] == (3, 30)


@pytest.mark.timeout(10)
@patch(f"{CLIENT}.requests")
@patch(f"{V1}.requests")
def test_relay_session_reused_across_runs(v1_requests, client_requests):
    session = _relay_session(v1_requests, client_requests)
    session.post.return_value = _http_response(json_body=_read_batch_body({"t": 1}))
    block = PLCReaderBlockV1()
    block.run(tags_to_read=["t"])
    block.run(tags_to_read=["t"])
    assert v1_requests.Session.call_count == 1
    assert session.post.call_count == 2


# ============================== EtherNet/IP mode =============================


@pytest.mark.timeout(10)
@patch(f"{DIRECT}.pylogix")
def test_ethernet_read(mock_pylogix):
    comm = _setup_pylogix(mock_pylogix)
    comm.Read.side_effect = lambda tag: _plc_response(value={"t": 7}[tag])

    result = PLCReaderBlockV1().run(
        tags_to_read=["t"],
        connection_mode="ethernet_ip",
        ip_address="10.0.0.5",
        processor_slot=2,
    )

    assert result["tag_values"] == {"t": 7}
    assert result["error_status"] is False
    assert comm.IPAddress == "10.0.0.5"
    assert comm.ProcessorSlot == 2


@pytest.mark.timeout(10)
@patch(f"{DIRECT}.pylogix")
def test_ethernet_write_with_failure(mock_pylogix):
    comm = _setup_pylogix(mock_pylogix)
    comm.Write.return_value = _plc_response(status="Connection lost")

    result = PLCWriterBlockV1().run(
        tag="bad",
        value=2,
        depends_on=None,
        connection_mode="ethernet_ip",
        ip_address="10.0.0.5",
    )

    assert result["write_result"] == "WriteFailure"
    assert result["error_status"] is True


@pytest.mark.timeout(10)
@patch(f"{DIRECT}.pylogix")
def test_ethernet_write_string_value(mock_pylogix):
    # EtherNet/IP supports string writes (Logix STRING tags).
    comm = _setup_pylogix(mock_pylogix)
    comm.Write.return_value = _plc_response(status="Success")
    result = PLCWriterBlockV1().run(
        tag="Message",
        value="hello",
        depends_on=None,
        connection_mode="ethernet_ip",
        ip_address="10.0.0.5",
    )
    assert result["write_result"] == "WriteSuccess"
    comm.Write.assert_called_once_with("Message", "hello")


# ================================ Modbus mode ================================


@pytest.mark.timeout(10)
@patch(f"{DIRECT}.ModbusTcpClient")
def test_modbus_read_register_and_coil(mock_cls):
    client = _setup_modbus(mock_cls)
    client.read_holding_registers.return_value = _modbus_register_response(123)
    client.read_coils.return_value = _modbus_bit_response(True)

    result = PLCReaderBlockV1().run(
        tags_to_read=["holding:100", "coil:0"],
        connection_mode="modbus",
        ip_address="10.0.0.9",
        modbus_port=5020,
        modbus_unit_id=3,
    )

    assert result["tag_values"] == {"holding:100": 123, "coil:0": True}
    assert result["error_status"] is False
    mock_cls.assert_called_once_with("10.0.0.9", port=5020)
    client.read_holding_registers.assert_called_once_with(100, count=1, slave=3)
    client.read_coils.assert_called_once_with(0, count=1, slave=3)


@pytest.mark.timeout(10)
@patch(f"{DIRECT}.ModbusTcpClient")
def test_modbus_read_error_and_bad_tag(mock_cls):
    client = _setup_modbus(mock_cls)
    client.read_holding_registers.return_value = _modbus_register_response(
        0, error=True
    )

    result = PLCReaderBlockV1().run(
        tags_to_read=["holding:1", "bogus:2"],
        connection_mode="modbus",
    )

    assert result["tag_values"]["holding:1"] == "ReadFailure"  # PLC error
    assert result["tag_values"]["bogus:2"] == "ReadFailure"  # parse error
    assert result["error_status"] is True


@pytest.mark.timeout(10)
@patch(f"{DIRECT}.ModbusTcpClient")
def test_modbus_write_register_coil_and_readonly(mock_cls):
    client = _setup_modbus(mock_cls)
    client.write_register.return_value = _modbus_write_response()
    client.write_coil.return_value = _modbus_write_response()

    block = PLCWriterBlockV1()
    common = dict(depends_on=None, connection_mode="modbus", modbus_unit_id=2)
    register = block.run(tag="holding:100", value=25, **common)
    coil = block.run(tag="coil:0", value=True, **common)
    readonly = block.run(tag="input:5", value=1, **common)

    assert register["write_result"] == "WriteSuccess"
    assert coil["write_result"] == "WriteSuccess"
    # input registers are read-only -> rejected before any client call
    assert readonly["write_result"] == "WriteFailure"
    assert readonly["error_status"] is True
    client.write_register.assert_called_once_with(100, 25, slave=2)
    client.write_coil.assert_called_once_with(0, True, slave=2)


@pytest.mark.timeout(10)
@patch(f"{DIRECT}.ModbusTcpClient")
def test_modbus_write_rejects_invalid_values_without_coercing(mock_cls):
    client = _setup_modbus(mock_cls)
    client.write_register.return_value = _modbus_write_response()
    client.write_coil.return_value = _modbus_write_response()

    # A non-integral register, a non-boolean coil value, and an out-of-range register must
    # all fail rather than silently writing 2 / True / wrapped. (A literal string is blocked
    # at manifest validation; a selector could still resolve to one of these at runtime.)
    block = PLCWriterBlockV1()
    common = dict(depends_on=None, connection_mode="modbus")
    for tag, value in [
        ("holding:100", 2.9),
        ("coil:0", "False"),
        ("holding:200", 70000),
    ]:
        result = block.run(tag=tag, value=value, **common)
        assert result["write_result"] == "WriteFailure"
        assert result["error_status"] is True

    # No bad value should ever reach the transport.
    client.write_register.assert_not_called()
    client.write_coil.assert_not_called()


@pytest.mark.timeout(10)
@patch(f"{DIRECT}.ModbusTcpClient")
def test_modbus_write_accepts_integral_float_and_binary_coil(mock_cls):
    client = _setup_modbus(mock_cls)
    client.write_register.return_value = _modbus_write_response()
    client.write_coil.return_value = _modbus_write_response()

    # 25.0 is integral -> 25; 1 is a valid coil bit -> True.
    block = PLCWriterBlockV1()
    common = dict(depends_on=None, connection_mode="modbus")
    register = block.run(tag="holding:100", value=25.0, **common)
    coil = block.run(tag="coil:0", value=1, **common)

    assert register["write_result"] == "WriteSuccess"
    assert coil["write_result"] == "WriteSuccess"
    assert register["error_status"] is False and coil["error_status"] is False
    client.write_register.assert_called_once_with(100, 25, slave=1)
    client.write_coil.assert_called_once_with(0, True, slave=1)


@pytest.mark.timeout(10)
@patch(f"{DIRECT}.ModbusTcpClient")
def test_modbus_connect_failure_marks_all_failed(mock_cls):
    _setup_modbus(mock_cls, connect=False)

    result = PLCReaderBlockV1().run(
        tags_to_read=["holding:1", "holding:2"],
        connection_mode="modbus",
    )

    assert result["tag_values"] == {
        "holding:1": "ReadFailure",
        "holding:2": "ReadFailure",
    }
    assert result["error_status"] is True


# ====================== persistent direct connections =======================


@pytest.mark.timeout(10)
@patch(f"{DIRECT}.ModbusTcpClient")
def test_modbus_client_reused_across_frames(mock_cls):
    # A frame-by-frame workflow must not open/close a TCP connection every frame: the
    # client is built once, connected once, reused, and never closed between frames.
    client = _setup_modbus(mock_cls)
    client.read_holding_registers.return_value = _modbus_register_response(1)

    def _connect():
        client.connected = True
        return True

    client.connect.side_effect = _connect

    block = PLCReaderBlockV1()
    for _ in range(3):
        block.run(
            tags_to_read=["holding:1"],
            connection_mode="modbus",
            ip_address="10.0.0.9",
            modbus_port=502,
        )

    mock_cls.assert_called_once_with("10.0.0.9", port=502)
    client.connect.assert_called_once()
    client.close.assert_not_called()


@pytest.mark.timeout(10)
@patch(f"{DIRECT}.ModbusTcpClient")
def test_modbus_client_reconnects_after_drop(mock_cls):
    # If the connection drops (connected == False again), the next frame reconnects on the
    # same cached client rather than failing outright.
    client = _setup_modbus(mock_cls)
    client.read_holding_registers.return_value = _modbus_register_response(1)

    def _connect():
        client.connected = True
        return True

    client.connect.side_effect = _connect

    block = PLCReaderBlockV1()
    block.run(tags_to_read=["holding:1"], connection_mode="modbus")
    client.connected = False  # simulate a dropped socket between frames
    block.run(tags_to_read=["holding:1"], connection_mode="modbus")

    mock_cls.assert_called_once()  # same client object
    assert client.connect.call_count == 2  # reconnected


@pytest.mark.timeout(10)
@patch(f"{DIRECT}.ModbusTcpClient")
def test_modbus_client_rebuilt_when_target_changes(mock_cls):
    # Changing the address/port closes the old client and opens a new one.
    first, second = MagicMock(), MagicMock()
    for c in (first, second):
        c.connected = False
        c.connect.return_value = True
        c.read_holding_registers.return_value = _modbus_register_response(1)
    mock_cls.side_effect = [first, second]

    block = PLCReaderBlockV1()
    block.run(
        tags_to_read=["holding:1"], connection_mode="modbus", ip_address="10.0.0.1"
    )
    block.run(
        tags_to_read=["holding:1"], connection_mode="modbus", ip_address="10.0.0.2"
    )

    assert mock_cls.call_count == 2
    first.close.assert_called_once()  # old client torn down on target change


@pytest.mark.timeout(10)
@patch(f"{DIRECT}.pylogix")
def test_ethernet_client_reused_across_frames(mock_pylogix):
    comm = _setup_pylogix(mock_pylogix)
    comm.Read.return_value = _plc_response(value=7)

    block = PLCReaderBlockV1()
    for _ in range(3):
        block.run(
            tags_to_read=["t"], connection_mode="ethernet_ip", ip_address="10.0.0.5"
        )

    # One pylogix.PLC() for all three frames; not closed between them.
    mock_pylogix.PLC.assert_called_once()
    comm.Close.assert_not_called()


# ===================== write value-type validation (manifest) ================


@pytest.mark.timeout(10)
def test_manifest_allows_string_write_for_ethernet_ip():
    m = PLCWriterBlockManifest.model_validate(
        _writer_manifest_kwargs(
            connection_mode="ethernet_ip", tag="Message", value="hello"
        )
    )
    assert m.value == "hello"


@pytest.mark.timeout(10)
def test_manifest_rejects_string_write_for_relay_and_modbus():
    for mode in ("relay", "modbus"):
        with pytest.raises(ValidationError):
            PLCWriterBlockManifest.model_validate(
                _writer_manifest_kwargs(connection_mode=mode, value="hello")
            )


@pytest.mark.timeout(10)
def test_manifest_allows_numeric_write_for_all_modes():
    for mode in ("relay", "ethernet_ip", "modbus"):
        for value in (1, 2.5, True):
            m = PLCWriterBlockManifest.model_validate(
                _writer_manifest_kwargs(connection_mode=mode, value=value)
            )
            assert m.connection_mode == mode


@pytest.mark.timeout(10)
def test_manifest_allows_selector_value_for_all_modes():
    # A selector value (resolved at runtime) is accepted in every mode, including relay and
    # Modbus, so a single static tag can be mapped to a dynamic workflow output without a
    # whole-dict selector or a formatter step.
    for mode in ("relay", "ethernet_ip", "modbus"):
        m = PLCWriterBlockManifest.model_validate(
            _writer_manifest_kwargs(connection_mode=mode, value="$steps.counter.count")
        )
        assert m.value == "$steps.counter.count"


@pytest.mark.timeout(10)
def test_manifest_rejects_non_positive_request_timeout():
    # A literal 0 / negative relay timeout would make requests raise before sending, so both
    # blocks reject it at validation time; a selector is still allowed (guarded at runtime).
    for bad in (0, -1):
        with pytest.raises(ValidationError):
            PLCReaderBlockManifest.model_validate(
                {
                    "type": "roboflow_core/plc_reader@v1",
                    "name": "r",
                    "request_timeout": bad,
                }
            )
        with pytest.raises(ValidationError):
            PLCWriterBlockManifest.model_validate(
                _writer_manifest_kwargs(request_timeout=bad)
            )
    # A selector and a positive literal are both accepted.
    assert (
        PLCReaderBlockManifest.model_validate(
            {
                "type": "roboflow_core/plc_reader@v1",
                "name": "r",
                "request_timeout": "$inputs.timeout",
            }
        ).request_timeout
        == "$inputs.timeout"
    )


# ============================ dispatch / registration ========================


@pytest.mark.timeout(10)
@patch(f"{DIRECT}.ModbusTcpClient")
@patch(f"{DIRECT}.pylogix")
@patch(f"{CLIENT}.requests")
@patch(f"{V1}.requests")
def test_relay_mode_never_touches_direct(
    v1_requests, client_requests, mock_pylogix, mock_modbus
):
    session = _relay_session(v1_requests, client_requests)
    session.post.return_value = _http_response(json_body=_read_batch_body({"t": 1}))

    PLCReaderBlockV1().run(tags_to_read=["t"])  # default connection_mode == relay

    mock_pylogix.PLC.assert_not_called()
    mock_modbus.assert_not_called()


@pytest.mark.timeout(10)
def test_blocks_registered_and_old_deprecated():
    from inference.enterprise.workflows.enterprise_blocks.loader import (
        load_enterprise_blocks,
    )
    from inference.enterprise.workflows.enterprise_blocks.sinks.PLC_modbus.v1 import (
        ModbusTCPBlockManifest,
    )
    from inference.enterprise.workflows.enterprise_blocks.sinks.PLCethernetIP.v1 import (
        PLCBlockManifest,
    )

    names = {b.__name__ for b in load_enterprise_blocks()}
    assert {"PLCReaderBlockV1", "PLCWriterBlockV1"}.issubset(names)
    assert "PLCRelayReaderBlockV1" not in names
    # The direct blocks stay registered (existing workflows keep running) but deprecated.
    assert "PLCBlockV1" in names and "ModbusTCPBlockV1" in names
    assert PLCBlockManifest.model_config["json_schema_extra"]["deprecated"] is True
    assert (
        ModbusTCPBlockManifest.model_config["json_schema_extra"]["deprecated"] is True
    )

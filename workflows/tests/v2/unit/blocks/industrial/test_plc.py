"""PLC reader/writer: relay transport over loopback HTTP, direct stand-ins, lifetime.

Nothing leaves this process or ``127.0.0.1``. ``LoopbackRelay`` is a real HTTP
server speaking the relay's batch API, so relay tests exercise the actual
``requests`` transport. EtherNet/IP and Modbus use in-process stand-ins for
``pylogix.PLC`` and ``ModbusTcpClient``; installed drivers and real PLCs are
not exercised here.
"""

import base64
import json
import logging
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest
from roboflow_workflows.execution_engine.entities.workload import WorkOperation
from roboflow_workflows.v2.blocks.industrial import plc
from roboflow_workflows.v2.blocks.industrial.plc import (
    READ_FAILURE,
    WRITE_FAILURE,
    WRITE_SUCCESS,
    PlcConnections,
    PlcDriverUnavailableError,
    PlcReaderBlock,
    PlcTarget,
    PlcWriterBlock,
    coerce_coil_value,
    coerce_register_value,
    parse_modbus_tag,
    relay_base_url,
)
from roboflow_workflows.v2.data.secrets import Secret, SecretUnavailableError
from roboflow_workflows.v2.engine.declaration import spec_of
from roboflow_workflows.v2.engine.errors import ParamsValidationError

PASSWORD = "pw-SENTINEL-41c9"
LOGGER_NAME = "roboflow_workflows.v2.blocks.industrial.plc"
WAIT = 10


# --------------------------------- fixtures ----------------------------------


Responder = Callable[[str, Any], Tuple[int, bytes]]


class LoopbackRelay:
    """PLC Relay double on ``127.0.0.1:<free port>``; records every request.

    ``respond(path, body)`` returns ``(status, raw body)``; the default
    answers each read tag with ``values[tag]`` and each write with success.
    A set ``hold`` makes the handler wait for it first (timeout tests).
    """

    def __init__(self, values: Optional[Dict[str, Any]] = None) -> None:
        self.values = dict(values or {})
        self.respond: Responder = self._default
        self.received: List[Dict[str, Any]] = []
        self.hold: Optional[threading.Event] = None
        relay = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"null")
                relay.received.append(
                    {
                        "path": self.path,
                        "headers": {k.lower(): v for k, v in self.headers.items()},
                        "body": body,
                    }
                )
                if relay.hold is not None:
                    relay.hold.wait(WAIT)
                status, raw = relay.respond(self.path, body)
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *args: Any) -> None:
                return None

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    @property
    def host(self) -> str:
        return self._httpd.server_address[0]

    @property
    def port(self) -> int:
        return self._httpd.server_address[1]

    def _default(self, path: str, body: Any) -> Tuple[int, bytes]:
        if path == "/read_batch":
            tags = [{"name": t, "value": self.values[t]} for t in body["tags"]]
            return 200, json.dumps({"tags": tags}).encode()

        results = [{"name": w["name"], "success": True} for w in body["writes"]]
        return 200, json.dumps({"results": results}).encode()

    def __enter__(self) -> "LoopbackRelay":
        self._thread.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        if self.hold is not None:
            self.hold.set()
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)


def json_response(status: int, payload: Any) -> Tuple[int, bytes]:
    return status, json.dumps(payload).encode()


class FakePlcClient:
    """Borrowed ``plc_client``: records calls, answers with fixed results."""

    def __init__(self, *, values: Optional[Dict[str, Any]] = None) -> None:
        self.values = values or {}
        self.calls: List[Tuple[str, PlcTarget, Any]] = []
        self.closed = False

    def read_tags(self, target: PlcTarget, tags: List[Any]):
        self.calls.append(("read", target, list(tags)))
        return {t: self.values.get(t, READ_FAILURE) for t in tags}, False

    def write_tags(self, target: PlcTarget, tags_to_write):
        self.calls.append(("write", target, dict(tags_to_write)))
        return {t: WRITE_SUCCESS for t in tags_to_write}, False

    def close(self) -> None:
        self.closed = True


class EipResponse:
    def __init__(self, status: str, value: Any = None) -> None:
        self.Status = status
        self.Value = value


class FakeEipPlc:
    """``pylogix.PLC`` stand-in with a tag table."""

    def __init__(self, address: str, slot: int) -> None:
        self.IPAddress, self.ProcessorSlot = address, slot
        self.tags: Dict[str, Any] = {"Program:Main.Count": 7, "Name": "abc"}
        self.closed = 0

    def Read(self, tag: str) -> EipResponse:  # noqa: N802
        if tag == "Explodes":
            raise OSError("socket reset")
        if tag not in self.tags:
            return EipResponse("Path segment error")
        return EipResponse("Success", self.tags[tag])

    def Write(self, tag: str, value: Any) -> EipResponse:  # noqa: N802
        if tag not in self.tags:
            return EipResponse("Path segment error")
        self.tags[tag] = value
        return EipResponse("Success")

    def Close(self) -> None:  # noqa: N802
        self.closed += 1


class ModbusResponse:
    def __init__(self, *, registers=None, bits=None, error: bool = False) -> None:
        self.registers, self.bits, self._error = registers, bits, error

    def isError(self) -> bool:  # noqa: N802
        return self._error


class FakeModbus:
    """``ModbusTcpClient`` stand-in; records (operation, address, value, slave)."""

    def __init__(self, address: str, port: int, *, connects: bool = True) -> None:
        self.address, self.port = address, port
        self.connected = False
        self._connects = connects
        self.calls: List[Tuple[str, int, Any, int]] = []
        self.closed = 0

    def connect(self) -> bool:
        self.connected = self._connects
        return self._connects

    def _read(self, name: str, address: int, slave: int, **payload: Any):
        self.calls.append((name, address, None, slave))
        if address == 999:
            return ModbusResponse(error=True)
        return ModbusResponse(**payload)

    def read_holding_registers(self, address: int, *, count: int, slave: int):
        return self._read("holding", address, slave, registers=[address + 1])

    def read_input_registers(self, address: int, *, count: int, slave: int):
        return self._read("input", address, slave, registers=[address + 2])

    def read_coils(self, address: int, *, count: int, slave: int):
        return self._read("coil", address, slave, bits=[1])

    def read_discrete_inputs(self, address: int, *, count: int, slave: int):
        return self._read("discrete", address, slave, bits=[0])

    def write_register(self, address: int, value: int, *, slave: int):
        self.calls.append(("write_register", address, value, slave))
        return ModbusResponse(error=address == 999)

    def write_coil(self, address: int, value: bool, *, slave: int):
        self.calls.append(("write_coil", address, value, slave))
        return ModbusResponse()

    def close(self) -> None:
        self.closed += 1


class RecordingSession:
    """Wraps a real ``requests.Session``; counts sessions and closes."""

    made: List["RecordingSession"] = []

    def __init__(self) -> None:
        import requests

        self._session = requests.Session()
        self.closed = False
        RecordingSession.made.append(self)

    def post(self, *args: Any, **kwargs: Any):
        return self._session.post(*args, **kwargs)

    def close(self) -> None:
        self.closed = True
        self._session.close()


@pytest.fixture()
def relay():
    with LoopbackRelay({"camera_msg": 1, "sku_number": "A-17"}) as server:
        yield server


@pytest.fixture()
def recorded_sessions(monkeypatch):
    RecordingSession.made = []
    monkeypatch.setattr(plc, "_new_session", RecordingSession)
    return RecordingSession.made


def read(**params: Any) -> Dict[str, Any]:
    """Run the reader through the default per-call client."""
    block = PlcReaderBlock(plc_client=params.pop("plc_client", None))
    result = block.run(**{**reader_defaults(), **params})

    return result


def write(**params: Any) -> Dict[str, Any]:
    """Run the writer through the default per-call client."""
    block = PlcWriterBlock(
        plc_client=params.pop("plc_client", None),
        disable_sinks=params.pop("disable_sinks", False),
    )
    result = block.run(**{**writer_defaults(), **params})

    return result


def reader_defaults() -> Dict[str, Any]:
    params = spec_of(PlcReaderBlock).validate_params({}).model_dump()

    return params


def writer_defaults() -> Dict[str, Any]:
    params = spec_of(PlcWriterBlock).validate_params({"tag": "t", "value": 1})

    return params.model_dump()


def target(**overrides: Any) -> PlcTarget:
    fields = {"connection_mode": "relay", "address": "127.0.0.1", **overrides}

    return PlcTarget(**fields)


# ------------------------------ relay transport ------------------------------


def test_relay_read_posts_one_batch_and_converts_values(relay) -> None:
    # when
    result = read(
        tags_to_read=["camera_msg", "sku_number"],
        ip_address=relay.host,
        relay_port=relay.port,
    )

    # then
    assert result == {
        "tag_values": {"camera_msg": 1, "sku_number": "A-17"},
        "error_status": False,
    }
    [request] = relay.received
    assert request["path"] == "/read_batch"
    assert request["body"] == {"tags": ["camera_msg", "sku_number"]}
    assert request["headers"]["content-type"] == "application/json"
    assert "authorization" not in request["headers"]


def test_relay_read_marks_per_tag_errors_and_missing_tags(relay, caplog) -> None:
    # given
    relay.respond = lambda path, body: json_response(
        200,
        {
            "tags": [
                {"name": "camera_msg", "value": 3},
                {"name": "sku_number", "error": "type mismatch"},
                "not an entry",
            ]
        },
    )

    # when
    with caplog.at_level(logging.ERROR, logger=LOGGER_NAME):
        result = read(
            tags_to_read=["camera_msg", "sku_number", "unlisted"],
            ip_address=f"http://{relay.host}:{relay.port}/",
        )

    # then
    assert result == {
        "tag_values": {
            "camera_msg": 3,
            "sku_number": READ_FAILURE,
            "unlisted": READ_FAILURE,
        },
        "error_status": True,
    }
    assert "type mismatch" in caplog.text
    assert "did not return a value for tag 'unlisted'" in caplog.text


@pytest.mark.parametrize(
    "response",
    [
        json_response(404, {"detail": "unknown tag 'camera_msg'"}),
        (200, b"not json"),
        json_response(200, ["a", "list"]),
    ],
    ids=["whole-batch-404", "non-json-200", "non-object-200"],
)
def test_relay_batch_rejection_or_malformed_body_fails_every_tag(
    relay, response
) -> None:
    # given
    relay.respond = lambda path, body: response

    # when
    reading = read(
        tags_to_read=["camera_msg", "sku_number"],
        ip_address=relay.host,
        relay_port=relay.port,
    )
    writing = write(
        tag="camera_msg", value=1, ip_address=relay.host, relay_port=relay.port
    )

    # then
    assert reading == {
        "tag_values": {"camera_msg": READ_FAILURE, "sku_number": READ_FAILURE},
        "error_status": True,
    }
    assert writing == {"write_result": WRITE_FAILURE, "error_status": True}


def test_relay_write_posts_name_and_value_and_reports_success(relay) -> None:
    # when
    result = write(
        tag="camera_fault", value=2.5, ip_address=relay.host, relay_port=relay.port
    )

    # then
    assert result == {"write_result": WRITE_SUCCESS, "error_status": False}
    [request] = relay.received
    assert request["path"] == "/write_batch"
    assert request["body"] == {"writes": [{"name": "camera_fault", "value": 2.5}]}


@pytest.mark.parametrize(
    "results",
    [
        [{"name": "camera_fault", "success": False, "error": "out of range"}],
        [{"name": "other", "success": True}],
    ],
    ids=["rejected-value", "missing-result"],
)
def test_relay_write_failure_per_tag_is_write_failure(relay, results) -> None:
    # given
    relay.respond = lambda path, body: json_response(200, {"results": results})

    # when
    result = write(
        tag="camera_fault", value=70000, ip_address=relay.host, relay_port=relay.port
    )

    # then
    assert result == {"write_result": WRITE_FAILURE, "error_status": True}
    assert len(relay.received) == 1


def test_relay_read_timeout_fails_every_tag_without_retry(relay) -> None:
    # given: the relay holds the response past the 1 s read timeout
    relay.hold = threading.Event()

    # when
    result = read(
        tags_to_read=["camera_msg"],
        ip_address=relay.host,
        relay_port=relay.port,
        request_timeout=1,
    )
    relay.hold.set()

    # then
    assert result == {"tag_values": {"camera_msg": READ_FAILURE}, "error_status": True}
    assert len(relay.received) == 1


def test_relay_write_timeout_is_one_unretried_write_failure(relay) -> None:
    # given: the relay received the write but answers too late
    relay.hold = threading.Event()

    # when
    result = write(
        tag="camera_fault",
        value=1,
        ip_address=relay.host,
        relay_port=relay.port,
        request_timeout=1,
    )
    relay.hold.set()

    # then
    assert result == {"write_result": WRITE_FAILURE, "error_status": True}
    assert len(relay.received) == 1


def test_unreachable_relay_fails_every_tag() -> None:
    # given: a port that was free a moment ago
    with LoopbackRelay() as closed:
        host, port = closed.host, closed.port

    # when
    result = read(tags_to_read=["a", "b"], ip_address=host, relay_port=port)

    # then
    assert result == {
        "tag_values": {"a": READ_FAILURE, "b": READ_FAILURE},
        "error_status": True,
    }


def test_a_resolved_non_positive_timeout_fails_tags_without_sending(relay) -> None:
    # when: a selected timeout skips literal validation, as in V1
    result = read(
        tags_to_read=["camera_msg"],
        ip_address=relay.host,
        relay_port=relay.port,
        request_timeout=0,
    )

    # then
    assert result == {"tag_values": {"camera_msg": READ_FAILURE}, "error_status": True}
    assert relay.received == []


def test_a_non_json_relay_value_raises_as_in_v1(relay) -> None:
    # when / then
    with pytest.raises(TypeError):
        write(tag="t", value=object(), ip_address=relay.host, relay_port=relay.port)
    assert relay.received == []


def test_empty_tag_list_reads_nothing_and_sends_nothing(
    relay, recorded_sessions
) -> None:
    # when
    result = read(tags_to_read=[], ip_address=relay.host, relay_port=relay.port)

    # then
    assert result == {"tag_values": {}, "error_status": False}
    assert relay.received == []


@pytest.mark.parametrize(
    "address, port, expected",
    [
        ("192.168.1.10", 8007, "http://192.168.1.10:8007"),
        (" 10.0.0.2 ", 9000, "http://10.0.0.2:9000"),
        ("http://relay:8007/", 1, "http://relay:8007"),
        ("https://relay", 1, "https://relay"),
    ],
)
def test_relay_base_url(address: str, port: int, expected: str) -> None:
    assert relay_base_url(address, port) == expected


# ----------------------------- secrets and redaction -------------------------


def test_a_secret_relay_url_sends_basic_auth_and_outputs_no_plaintext(relay) -> None:
    # given
    address = Secret(f"http://plc:{PASSWORD}@{relay.host}:{relay.port}", name="RELAY")

    # when
    result = read(tags_to_read=["camera_msg"], ip_address=address)

    # then
    assert result == {"tag_values": {"camera_msg": 1}, "error_status": False}
    expected = base64.b64encode(f"plc:{PASSWORD}".encode()).decode()
    assert relay.received[0]["headers"]["authorization"] == f"Basic {expected}"
    assert PASSWORD not in repr(result)


def test_relay_errors_echoing_the_secret_url_are_scrubbed(relay, caplog) -> None:
    # given: a relay error message that repeats the full secret URL
    url = f"http://plc:{PASSWORD}@{relay.host}:{relay.port}"
    relay.respond = lambda path, body: json_response(
        200, {"tags": [{"name": "camera_msg", "error": f"bad request to {url}"}]}
    )

    # when
    with caplog.at_level(logging.ERROR, logger=LOGGER_NAME):
        result = read(tags_to_read=["camera_msg"], ip_address=Secret(url, name="RELAY"))

    # then
    assert result["tag_values"] == {"camera_msg": READ_FAILURE}
    assert "bad request to <secret RELAY>" in caplog.text
    assert PASSWORD not in caplog.text


def test_an_unreachable_secret_relay_logs_no_password(caplog) -> None:
    # given
    with LoopbackRelay() as closed:
        url = f"http://plc:{PASSWORD}@{closed.host}:{closed.port}"

    # when
    with caplog.at_level(logging.ERROR, logger=LOGGER_NAME):
        result = read(tags_to_read=["a"], ip_address=Secret(url, name="RELAY"))

    # then
    assert result["error_status"] is True
    assert "Failed to reach PLC Relay" in caplog.text
    assert PASSWORD not in caplog.text


def test_an_unavailable_secret_fails_before_any_client_is_used(
    recorded_sessions,
) -> None:
    # given
    client = FakePlcClient()

    # when / then
    with pytest.raises(SecretUnavailableError):
        read(
            tags_to_read=["a"],
            ip_address=Secret.unavailable("RELAY"),
            plc_client=client,
        )
    with pytest.raises(SecretUnavailableError):
        write(ip_address=Secret.unavailable("RELAY"))
    assert client.calls == []
    assert recorded_sessions == []


def test_a_target_repr_shows_the_secret_name_only() -> None:
    # given
    address = Secret(f"http://plc:{PASSWORD}@relay:8007", name="RELAY")

    # when
    shown = repr(target(address=address))

    # then
    assert "Secret('RELAY')" in shown
    assert PASSWORD not in shown


# ------------------------------ disabled writer ------------------------------


@pytest.mark.parametrize(
    "disable_sinks, disable_sink",
    [(True, False), (False, True)],
    ids=["by-policy", "by-parameter"],
)
def test_a_disabled_writer_touches_no_client_and_reveals_nothing(
    disable_sinks: bool, disable_sink: bool, recorded_sessions
) -> None:
    # given
    client = FakePlcClient()

    # when
    borrowed = write(
        plc_client=client,
        disable_sinks=disable_sinks,
        disable_sink=disable_sink,
        ip_address=Secret.unavailable("RELAY"),
    )
    default = write(disable_sinks=disable_sinks, disable_sink=disable_sink)

    # then
    assert borrowed == default == {"write_result": "", "error_status": False}
    assert client.calls == []
    assert recorded_sessions == []


# ----------------------------- client lifetime -------------------------------


def test_a_borrowed_client_is_used_as_given_and_never_closed() -> None:
    # given
    client = FakePlcClient(values={"a": 5})
    reader = PlcReaderBlock(plc_client=client)
    writer = PlcWriterBlock(plc_client=client)

    # when
    first = reader.run(**{**reader_defaults(), "tags_to_read": ["a"]})
    second = reader.run(**{**reader_defaults(), "tags_to_read": ["a"]})
    written = writer.run(**{**writer_defaults(), "tag": "a", "value": 6})

    # then
    assert first == second == {"tag_values": {"a": 5}, "error_status": False}
    assert written == {"write_result": WRITE_SUCCESS, "error_status": False}
    assert [call[0] for call in client.calls] == ["read", "read", "write"]
    assert client.calls[2][2] == {"a": 6}
    assert client.closed is False


def test_borrowed_plc_connections_reuse_one_session_until_the_host_closes(
    relay, recorded_sessions
) -> None:
    # given
    connections = PlcConnections(make_session=RecordingSession)
    block = PlcReaderBlock(plc_client=connections)
    params = {
        **reader_defaults(),
        "tags_to_read": ["camera_msg"],
        "ip_address": relay.host,
        "relay_port": relay.port,
    }

    # when
    results = [block.run(**params) for _ in range(3)]
    open_before_close = [session.closed for session in recorded_sessions]
    connections.close()

    # then
    assert all(r["tag_values"] == {"camera_msg": 1} for r in results)
    assert len(recorded_sessions) == 1
    assert open_before_close == [False]
    assert recorded_sessions[0].closed is True
    with pytest.raises(RuntimeError, match="PlcConnections is closed"):
        block.run(**params)


def test_the_default_path_opens_and_closes_one_session_per_call(
    relay, recorded_sessions
) -> None:
    # when
    for _ in range(2):
        read(tags_to_read=["camera_msg"], ip_address=relay.host, relay_port=relay.port)

    # then
    assert len(recorded_sessions) == 2
    assert all(session.closed for session in recorded_sessions)


def test_the_default_path_closes_its_session_when_the_call_raises(
    relay, recorded_sessions
) -> None:
    # when
    with pytest.raises(TypeError):
        write(tag="t", value=object(), ip_address=relay.host, relay_port=relay.port)

    # then
    assert [session.closed for session in recorded_sessions] == [True]


def test_close_waits_for_a_running_call_on_shared_connections() -> None:
    # given: a held relay call, and a lock that reports when close waits on it
    import requests

    entered, release, close_waiting = (threading.Event() for _ in range(3))
    order: List[str] = []

    class HeldSession:
        def post(self, *args: Any, **kwargs: Any):
            entered.set()
            release.wait(WAIT)
            order.append("post done")
            raise requests.ConnectionError("no relay")

        def close(self) -> None:
            order.append("session closed")

    class ReportingLock:
        def __init__(self, lock: Any) -> None:
            self._lock = lock

        def __enter__(self) -> None:
            if threading.current_thread() is closing:
                close_waiting.set()
            self._lock.acquire()

        def __exit__(self, *exc_info: Any) -> None:
            self._lock.release()

    connections = PlcConnections(make_session=HeldSession)
    connections._lock = ReportingLock(connections._lock)
    reading = threading.Thread(
        target=connections.read_tags, args=(target(), ["a"]), daemon=True
    )
    closing = threading.Thread(target=connections.close, daemon=True)
    reading.start()
    assert entered.wait(WAIT)

    # when: close is waiting before the held call may finish
    closing.start()
    assert close_waiting.wait(WAIT)
    release.set()
    reading.join(WAIT)
    closing.join(WAIT)

    # then
    assert order == ["post done", "session closed"]


# --------------------------- direct EtherNet/IP ------------------------------


def eip_connections() -> Tuple[PlcConnections, List[FakeEipPlc]]:
    made: List[FakeEipPlc] = []

    def make(address: str, slot: int) -> FakeEipPlc:
        made.append(FakeEipPlc(address, slot))
        return made[-1]

    return PlcConnections(make_eip_client=make), made


def test_ethernet_ip_reads_and_writes_per_tag() -> None:
    # given
    connections, made = eip_connections()
    reader = PlcReaderBlock(plc_client=connections)
    writer = PlcWriterBlock(plc_client=connections)
    mode = {
        "connection_mode": "ethernet_ip",
        "ip_address": "10.0.0.5",
        "processor_slot": 2,
    }

    # when
    read_result = reader.run(
        **{
            **reader_defaults(),
            **mode,
            "tags_to_read": ["Program:Main.Count", "Missing", "Explodes"],
        }
    )
    write_result = writer.run(
        **{**writer_defaults(), **mode, "tag": "Name", "value": "xyz"}
    )
    missing = writer.run(**{**writer_defaults(), **mode, "tag": "Missing", "value": 1})

    # then
    assert read_result == {
        "tag_values": {
            "Program:Main.Count": 7,
            "Missing": READ_FAILURE,
            "Explodes": READ_FAILURE,
        },
        "error_status": True,
    }
    assert write_result == {"write_result": WRITE_SUCCESS, "error_status": False}
    assert missing == {"write_result": WRITE_FAILURE, "error_status": True}
    assert len(made) == 1
    assert (made[0].IPAddress, made[0].ProcessorSlot) == ("10.0.0.5", 2)
    assert made[0].tags["Name"] == "xyz"


def test_ethernet_ip_client_is_replaced_on_target_change_and_closed_on_close() -> None:
    # given
    connections, made = eip_connections()

    # when
    connections.read_tags(
        target(connection_mode="ethernet_ip", processor_slot=0), ["Name"]
    )
    connections.read_tags(
        target(connection_mode="ethernet_ip", processor_slot=1), ["Name"]
    )
    connections.close()

    # then
    assert [plc_.ProcessorSlot for plc_ in made] == [0, 1]
    assert [plc_.closed for plc_ in made] == [1, 1]


# ------------------------------ direct Modbus --------------------------------


def modbus_connections(**kwargs: Any) -> Tuple[PlcConnections, List[FakeModbus]]:
    made: List[FakeModbus] = []

    def make(address: str, port: int) -> FakeModbus:
        made.append(FakeModbus(address, port, **kwargs))
        return made[-1]

    return PlcConnections(make_modbus_client=make), made


def test_modbus_reads_each_area_with_the_unit_id() -> None:
    # given
    connections, made = modbus_connections()
    tags = [
        "holding:100",
        "100",
        "input:5",
        "coil:0",
        "discrete:2",
        "holding:999",
        "bogus:1",
    ]

    # when
    values, had_failure = connections.read_tags(
        target(connection_mode="modbus", modbus_port=1502, modbus_unit_id=7), tags
    )

    # then
    assert values == {
        "holding:100": 101,
        "100": 101,
        "input:5": 7,
        "coil:0": True,
        "discrete:2": False,
        "holding:999": READ_FAILURE,
        "bogus:1": READ_FAILURE,
    }
    assert had_failure is True
    assert (made[0].address, made[0].port) == ("127.0.0.1", 1502)
    assert {call[3] for call in made[0].calls} == {7}


@pytest.mark.parametrize(
    "tag, value, expected, sent",
    [
        ("holding:10", 5, WRITE_SUCCESS, ("write_register", 10, 5, 1)),
        ("holding:10", 7.0, WRITE_SUCCESS, ("write_register", 10, 7, 1)),
        ("coil:3", True, WRITE_SUCCESS, ("write_coil", 3, True, 1)),
        ("coil:3", 0, WRITE_SUCCESS, ("write_coil", 3, False, 1)),
        ("holding:10", 2.9, WRITE_FAILURE, None),
        ("holding:10", 70000, WRITE_FAILURE, None),
        ("holding:10", True, WRITE_FAILURE, None),
        ("coil:3", "False", WRITE_FAILURE, None),
        ("coil:3", 2, WRITE_FAILURE, None),
        ("input:1", 1, WRITE_FAILURE, None),
        ("discrete:1", True, WRITE_FAILURE, None),
        ("holding:x", 1, WRITE_FAILURE, None),
        ("holding:999", 1, WRITE_FAILURE, ("write_register", 999, 1, 1)),
    ],
)
def test_modbus_write_validates_area_and_value_before_sending(
    tag: str, value: Any, expected: str, sent: Optional[tuple]
) -> None:
    # given
    connections, made = modbus_connections()

    # when
    result = PlcWriterBlock(plc_client=connections).run(
        **{**writer_defaults(), "connection_mode": "modbus", "tag": tag, "value": value}
    )

    # then
    assert result == {
        "write_result": expected,
        "error_status": expected == WRITE_FAILURE,
    }
    assert made[0].calls == ([sent] if sent else [])


def test_modbus_connect_failure_fails_every_tag() -> None:
    # given
    connections, made = modbus_connections(connects=False)

    # when
    values, had_failure = connections.read_tags(
        target(connection_mode="modbus"), ["holding:1", "coil:2"]
    )

    # then
    assert values == {"holding:1": READ_FAILURE, "coil:2": READ_FAILURE}
    assert had_failure is True
    assert made[0].calls == []


def test_modbus_helpers() -> None:
    assert parse_modbus_tag(" Coil : 4 ") == ("coil", 4)
    assert parse_modbus_tag(12) == ("holding", 12)
    assert coerce_coil_value(1.0) is True
    assert coerce_register_value(65535.0) == 65535
    with pytest.raises(ValueError, match="unknown Modbus area"):
        parse_modbus_tag("bits:1")


# ------------------------------ optional drivers ------------------------------


@pytest.mark.parametrize(
    "mode, module, pin",
    [
        ("ethernet_ip", "pylogix", "pylogix==1.0.5"),
        ("modbus", "pymodbus.client", "pymodbus>=3.6.9,<=3.8.3"),
    ],
)
def test_a_missing_driver_fails_the_call_naming_the_exact_pin(
    monkeypatch, mode: str, module: str, pin: str
) -> None:
    # given: importing the driver fails, installed or not
    monkeypatch.setitem(sys.modules, module, None)

    # when / then
    with pytest.raises(PlcDriverUnavailableError) as raised:
        read(tags_to_read=["holding:1"], connection_mode=mode)
    assert pin in str(raised.value)
    assert "roboflow-workflows[enterprise]" in str(raised.value)


def test_importing_the_blocks_loads_no_plc_driver() -> None:
    # given
    code = (
        "import sys; import roboflow_workflows.v2.blocks.industrial.plc; "
        "print([m for m in ('pylogix', 'pymodbus') if m in sys.modules])"
    )

    # when
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )

    # then
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "[]"


# --------------------------------- validation --------------------------------


def writer_params(**raw: Any):
    params = spec_of(PlcWriterBlock).validate_params({"tag": "t", **raw})

    return params


@pytest.mark.parametrize(
    "raw",
    [
        {"value": "on"},
        {"value": "on", "connection_mode": "modbus"},
        {"value": 1, "request_timeout": 0},
        {"value": 1, "request_timeout": -3},
        {"value": 1, "relay_port": "8007"},
        {"value": 1, "connection_mode": "s7"},
    ],
)
def test_writer_rejects_invalid_literals(raw: Dict[str, Any]) -> None:
    with pytest.raises(ParamsValidationError):
        writer_params(**raw)


def test_literal_string_rule_names_the_mode() -> None:
    with pytest.raises(ParamsValidationError, match="only supported in Direct"):
        writer_params(value="on", connection_mode="relay")


@pytest.mark.parametrize(
    "raw",
    [
        {"value": "on", "connection_mode": "ethernet_ip"},
        {"value": "$inputs.value"},
        {"value": True, "request_timeout": "$inputs.timeout"},
        {"value": 1.5, "depends_on": "$steps.reader.tag_values"},
    ],
)
def test_writer_accepts_valid_literals_and_selectors(raw: Dict[str, Any]) -> None:
    writer_params(**raw)


def test_reader_rejects_non_string_tags_and_zero_timeout() -> None:
    spec = spec_of(PlcReaderBlock)
    with pytest.raises(ParamsValidationError):
        spec.validate_params({"tags_to_read": [100]})
    with pytest.raises(ParamsValidationError, match="positive number of seconds"):
        spec.validate_params({"request_timeout": 0})


def test_writer_requires_tag_and_value() -> None:
    with pytest.raises(ParamsValidationError):
        spec_of(PlcWriterBlock).validate_params({"tag": "t"})


@pytest.mark.parametrize("block", [PlcReaderBlock, PlcWriterBlock])
def test_discovery_declares_external_request_and_direct_lan_access(block) -> None:
    # given
    base = {} if block is PlcReaderBlock else {"tag": "t", "value": 1}
    spec = spec_of(block)

    # when
    relay_params = spec.validate_params(base)
    direct_params = spec.validate_params({**base, "connection_mode": "modbus"})

    # then
    assert block.discover_work_operations(relay_params) == [
        WorkOperation.EXTERNAL_REQUEST
    ]
    assert block.discover_restrictions(relay_params) == []
    assert [r.code for r in block.discover_restrictions(direct_params)] == [
        "requires_lan_access_to_device"
    ]
    assert block.discover_dependent_resources(relay_params) == []

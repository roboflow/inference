"""PLC reader and writer through the real compiler, session, recording and replay.

The relay is a loopback HTTP double on ``127.0.0.1:<free port>``; credentials
are invented sentinels. The catalogue is test-local until root registers the
blocks::

    $inputs ─▶ v2/plc_reader@v1 ─▶ test/next_count@v1 ─▶ v2/plc_writer@v1
                    │  POST /read_batch                     │  POST /write_batch
                    └────────────── LoopbackRelay ◀─────────┘
"""

import base64
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from pydantic import Field, StrictStr
from roboflow_workflows import environment
from roboflow_workflows.v2.blocks.industrial.plc import (
    READ_FAILURE,
    WRITE_FAILURE,
    WRITE_SUCCESS,
    PlcConnections,
    PlcReaderBlock,
    PlcTarget,
    PlcWriterBlock,
)
from roboflow_workflows.v2.blocks.secrets import (
    SECRET_RECORDING_CODEC,
    EnvironmentSecretsStoreBlock,
    StaticSecrets,
)
from roboflow_workflows.v2.data.kinds import DICTIONARY_KIND, INTEGER_KIND
from roboflow_workflows.v2.data.secrets import SECRET_KIND
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref
from roboflow_workflows.v2.engine.errors import ActiveRunError, StepExecutionError

from tests.v2.support.recording.capture import WAIT, Feed, feed_values, field, group

PASSWORD = "pw-SENTINEL-93be"
FLAG = "ALLOW_WORKFLOW_BLOCKS_ACCESSING_ENVIRONMENTAL_VARIABLES"


class LoopbackRelay:
    """PLC Relay double: answers reads from ``values``, applies writes to them."""

    def __init__(self, values: Dict[str, Any]) -> None:
        self.values = dict(values)
        self.rejected_values: List[Any] = []
        self.received: List[Dict[str, Any]] = []
        relay = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length))
                relay.received.append(
                    {
                        "path": self.path,
                        "authorization": self.headers.get("Authorization"),
                        "body": body,
                    }
                )
                raw = json.dumps(relay.answer(self.path, body)).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *args: Any) -> None:
                return None

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def answer(self, path: str, body: Any) -> Dict[str, Any]:
        if path == "/read_batch":
            tags = [{"name": t, "value": self.values[t]} for t in body["tags"]]
            return {"tags": tags}

        results = []
        for write in body["writes"]:
            accepted = isinstance(write["value"], (bool, int, float))
            if accepted:
                self.values[write["name"]] = write["value"]
            else:
                self.rejected_values.append(write["value"])
            results.append({"name": write["name"], "success": accepted})
        return {"results": results}

    @property
    def address(self) -> str:
        return self._httpd.server_address[0]

    @property
    def port(self) -> int:
        return self._httpd.server_address[1]

    def paths(self) -> List[str]:
        return [request["path"] for request in self.received]

    def __enter__(self) -> "LoopbackRelay":
        self._thread.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)


class NextCount(Block):
    """Test-local transform: ``values[key] + 1``."""

    type = "test/next_count@v1"
    outputs = {"output": Output(INTEGER_KIND, description="The next count.")}

    class Params(BlockParams):
        values: Ref(DICTIONARY_KIND) = Field(description="Tag values.")
        key: StrictStr = Field(description="Tag to increment.")

    def run(self, *, values: Dict[str, Any], key: str) -> Dict[str, Any]:
        return {"output": values[key] + 1}


CATALOGUE = Catalogue(
    [PlcReaderBlock, PlcWriterBlock, NextCount, EnvironmentSecretsStoreBlock],
    sources=[Feed],
    kinds=[SECRET_KIND],
    codecs=[SECRET_RECORDING_CODEC],
)


@pytest.fixture()
def relay():
    with LoopbackRelay({"count": 41, "fault": False}) as server:
        yield server


def output(name: str, selector: str) -> Dict[str, str]:
    return {"type": "JsonField", "name": name, "selector": selector}


def parameter(name: str, **extra: Any) -> Dict[str, Any]:
    return {"type": "WorkflowParameter", "name": name, **extra}


def read_increment_write(**writer: Any) -> Dict[str, Any]:
    address = {"ip_address": "$inputs.relay", "relay_port": "$inputs.port"}
    definition = {
        "version": "2.0",
        "inputs": [parameter("relay"), parameter("port")],
        "steps": [
            {
                "type": "v2/plc_reader@v1",
                "name": "reader",
                "tags_to_read": ["count", "fault"],
                **address,
            },
            {
                "type": "test/next_count@v1",
                "name": "next",
                "values": "$steps.reader.tag_values",
                "key": "count",
            },
            {
                "type": "v2/plc_writer@v1",
                "name": "writer",
                "tag": "count",
                "value": "$steps.next.output",
                **address,
                **writer,
            },
        ],
        "outputs": [
            output("tag_values", "$steps.reader.tag_values"),
            output("read_error", "$steps.reader.error_status"),
            output("write_result", "$steps.writer.write_result"),
            output("write_error", "$steps.writer.error_status"),
        ],
    }

    return definition


def run_once(plan: Any, relay: LoopbackRelay, **resources: Any) -> Dict[str, Any]:
    session = plan.create_session(resources=resources)
    row = session.run({"relay": relay.address, "port": relay.port}).rows()[0]

    return row


def _target(relay: LoopbackRelay) -> PlcTarget:
    return PlcTarget(
        connection_mode="relay", address=relay.address, relay_port=relay.port
    )


def chain_text(error: Optional[BaseException]) -> str:
    texts = []
    while error is not None:
        texts.append(f"{type(error).__name__}: {error}")
        error = error.__cause__ or error.__context__

    return "\n".join(texts)


def test_read_transform_write_over_the_relay(relay) -> None:
    # given
    plan = compile_workflow(read_increment_write(), catalogue=CATALOGUE)

    # when
    row = run_once(plan, relay)

    # then
    assert row == {
        "tag_values": {"count": 41, "fault": False},
        "read_error": False,
        "write_result": WRITE_SUCCESS,
        "write_error": False,
    }
    assert relay.paths() == ["/read_batch", "/write_batch"]
    assert relay.received[1]["body"] == {"writes": [{"name": "count", "value": 42}]}
    assert relay.values["count"] == 42


def test_a_host_owned_client_is_shared_across_steps_and_runs(relay) -> None:
    # given
    sessions_made: List[Any] = []

    def make_session() -> Any:
        import requests

        sessions_made.append(requests.Session())
        return sessions_made[-1]

    plan = compile_workflow(read_increment_write(), catalogue=CATALOGUE)
    connections = PlcConnections(make_session=make_session)
    session = plan.create_session(resources={"plc_client": connections})

    # when
    rows = [
        session.run({"relay": relay.address, "port": relay.port}).rows()[0]
        for _ in range(3)
    ]
    # the engine never closed the borrowed client: it still works
    after_runs = connections.read_tags(_target(relay), ["count"])
    connections.close()

    # then
    assert [r["tag_values"]["count"] for r in rows] == [41, 42, 43]
    assert len(sessions_made) == 1
    assert after_runs == ({"count": 44}, False)
    assert relay.paths() == ["/read_batch", "/write_batch"] * 3 + ["/read_batch"]


@pytest.mark.parametrize(
    "resources, writer",
    [
        ({"disable_sinks": True}, {}),
        ({}, {"disable_sink": True}),
    ],
    ids=["host-policy", "step-parameter"],
)
def test_a_disabled_writer_sends_nothing(relay, resources, writer) -> None:
    # given
    plan = compile_workflow(read_increment_write(**writer), catalogue=CATALOGUE)

    # when
    row = run_once(plan, relay, **resources)

    # then
    assert row["write_result"] == ""
    assert row["write_error"] is False
    assert relay.paths() == ["/read_batch"]
    assert relay.values["count"] == 41


def test_a_literal_string_value_is_rejected_at_compile_time() -> None:
    definition = read_increment_write()
    definition["steps"][2]["value"] = "on"

    with pytest.raises(Exception, match="only supported in Direct"):
        compile_workflow(definition, catalogue=CATALOGUE)


def test_a_selected_string_reaches_the_relay_and_fails_that_tag(relay) -> None:
    # given: V1 checks literals only; the relay rejects the resolved string
    definition = read_increment_write()
    definition["inputs"].append(parameter("value"))
    definition["steps"][2]["value"] = "$inputs.value"
    plan = compile_workflow(definition, catalogue=CATALOGUE)

    # when
    row = (
        plan.create_session()
        .run({"relay": relay.address, "port": relay.port, "value": "on"})
        .rows()[0]
    )

    # then
    assert row["write_result"] == WRITE_FAILURE
    assert row["write_error"] is True
    assert relay.rejected_values == ["on"]


def test_depends_on_orders_a_literal_write_after_the_read(relay) -> None:
    # given
    definition = read_increment_write()
    writer = definition["steps"][2]
    writer.update(tag="fault", value=True, depends_on="$steps.reader.tag_values")
    del definition["steps"][1]

    plan = compile_workflow(definition, catalogue=CATALOGUE)

    # when
    row = run_once(plan, relay)

    # then
    assert row["write_result"] == WRITE_SUCCESS
    assert relay.paths() == ["/read_batch", "/write_batch"]


def test_an_unreachable_relay_gives_failure_outputs_not_a_step_error() -> None:
    # given
    definition = read_increment_write()
    definition["steps"] = definition["steps"][:1]
    definition["outputs"] = definition["outputs"][:2]
    plan = compile_workflow(definition, catalogue=CATALOGUE)
    with LoopbackRelay({}) as closed:
        address, port = closed.address, closed.port

    # when
    row = plan.create_session().run({"relay": address, "port": port}).rows()[0]

    # then
    assert row == {
        "tag_values": {"count": READ_FAILURE, "fault": READ_FAILURE},
        "read_error": True,
    }


def test_a_missing_direct_driver_fails_the_step_with_the_pin(monkeypatch) -> None:
    # given
    monkeypatch.setitem(sys.modules, "pymodbus.client", None)
    plan = compile_workflow(
        {
            "version": "2.0",
            "steps": [
                {
                    "type": "v2/plc_reader@v1",
                    "name": "reader",
                    "connection_mode": "modbus",
                    "tags_to_read": ["holding:1"],
                }
            ],
            "outputs": [output("tag_values", "$steps.reader.tag_values")],
        },
        catalogue=CATALOGUE,
    )

    # when
    with pytest.raises(StepExecutionError) as raised:
        plan.create_session().run({})

    # then
    assert "pymodbus>=3.6.9,<=3.8.3" in chain_text(raised.value)


# --------------------------------- secrets -----------------------------------

STORE_STEP = {
    "type": "v2/environment_secrets_store@v1",
    "name": "secrets",
    "variables_storing_secrets": ["PLC_RELAY_URL"],
}


def secret_reader_definition() -> Dict[str, Any]:
    definition = {
        "version": "2.0",
        "steps": [
            STORE_STEP,
            {
                "type": "v2/plc_reader@v1",
                "name": "reader",
                "tags_to_read": ["count"],
                "ip_address": "$steps.secrets.plc_relay_url",
            },
        ],
        "outputs": [
            output("relay_url", "$steps.secrets.plc_relay_url"),
            output("tag_values", "$steps.reader.tag_values"),
        ],
    }

    return definition


def secret_url(relay: LoopbackRelay) -> str:
    return f"http://plc:{PASSWORD}@{relay.address}:{relay.port}"


def test_a_secret_relay_url_authenticates_and_serializes_as_a_marker(relay) -> None:
    # given
    plan = compile_workflow(secret_reader_definition(), catalogue=CATALOGUE)
    secrets = StaticSecrets({"PLC_RELAY_URL": secret_url(relay)})

    # when
    result = plan.create_session(resources={"secrets": secrets}).run({})
    row = result.rows(serialize=True)[0]

    # then
    assert row == {"relay_url": "<secret PLC_RELAY_URL>", "tag_values": {"count": 41}}
    expected = base64.b64encode(f"plc:{PASSWORD}".encode()).decode()
    assert relay.received[0]["authorization"] == f"Basic {expected}"
    assert PASSWORD not in repr(result.rows()) + repr(row)


def recorded_definition() -> Dict[str, Any]:
    writer = {
        "type": "v2/plc_writer@v1",
        "tag": "count",
        "ip_address": "$steps.secrets.plc_relay_url",
    }
    definition = {
        "version": "2.0",
        "inputs": [parameter("capture_dir")],
        "sources": [{"type": "test/recorded_feed@v1", "name": "cam", "feed": "cam"}],
        "steps": [
            STORE_STEP,
            {**writer, "name": "writer", "value": "$sources.cam.value"},
        ],
        "outputs": [
            group(
                "frames",
                "$sources.cam.value",
                field("value", "$sources.cam.value"),
                field("relay_url", "$steps.secrets.plc_relay_url"),
                field("write_result", "$steps.writer.write_result"),
            )
        ],
        "recording": {
            "type": "file",
            "directory": "$inputs.capture_dir",
            "groups": ["frames"],
        },
        "retrospective": {
            "type": "workflow",
            "input_group": "frames",
            "workflow": {
                "steps": [
                    {
                        **writer,
                        "name": "rewrite",
                        "value": "$sources.frames.value",
                        "ip_address": "$sources.frames.relay_url",
                    }
                ],
                "outputs": [
                    group(
                        "reviewed",
                        "$sources.frames.value",
                        field("write_result", "$steps.rewrite.write_result"),
                    )
                ],
            },
        },
    }

    return definition


def test_a_replayed_secret_relay_url_fails_and_writes_nothing(
    relay, tmp_path: Path, monkeypatch
) -> None:
    # given: two recorded pulses, each written through the secret URL
    plan = compile_workflow(recorded_definition(), catalogue=CATALOGUE)
    session = plan.create_session(
        {
            "feeds": {"cam": feed_values(1.0, 2.0)},
            "secrets": StaticSecrets({"PLC_RELAY_URL": secret_url(relay)}),
        }
    )
    delivered: List[Any] = []
    session.start(
        {"capture_dir": str(tmp_path / "rec")}, handlers={"frames": delivered.append}
    ).wait(WAIT)
    recorded_writes = [r["body"] for r in relay.received]
    stored = b"".join(
        path.read_bytes() for path in (tmp_path / "rec").rglob("*") if path.is_file()
    )
    # an open environment holding the value must never be reread
    monkeypatch.setattr(environment, FLAG, True)
    monkeypatch.setenv("PLC_RELAY_URL", secret_url(relay))
    reviewed: List[Any] = []

    # when
    run = plan.retrospective.start(
        recording=tmp_path / "rec", handlers={"reviewed": reviewed.append}
    )
    with pytest.raises(ActiveRunError) as raised:
        run.wait(WAIT)

    # then
    assert recorded_writes == [
        {"writes": [{"name": "count", "value": 1.0}]},
        {"writes": [{"name": "count", "value": 2.0}]},
    ]
    assert [r.outputs.data["write_result"] for r in delivered] == [WRITE_SUCCESS] * 2
    assert b"PLC_RELAY_URL" in stored
    assert PASSWORD.encode() not in stored
    text = chain_text(raised.value)
    assert "SecretUnavailableError" in text
    assert PASSWORD not in text
    assert len(relay.received) == 2
    assert reviewed == []

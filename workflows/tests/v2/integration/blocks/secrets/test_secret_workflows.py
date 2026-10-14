"""Secrets through the real compiler, session, recording and replay.

Every value here is an invented sentinel; requests go only to an in-process
fake client and recordings only to pytest's temporary directory::

    secrets resource ─▶ roboflow_core/environment_secrets_store@v1 ─▶ test/secret_sink@v1 ─▶ client
                                     │                         (reveals, scrubs)
                                     └─▶ outputs / recording (name only)
    recording ─▶ replay ─▶ test/secret_sink@v1 ─▶ SecretUnavailableError, nothing sent
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from pydantic import Field
from roboflow_workflows import environment
from roboflow_workflows.v2.blocks.secrets import (
    SECRET_RECORDING_CODEC,
    EnvironmentSecretsStoreBlock,
    StaticSecrets,
)
from roboflow_workflows.v2.blocks.state.state_set.v1 import StateSetBlock
from roboflow_workflows.v2.data.kinds import BOOLEAN_KIND, STRING_KIND
from roboflow_workflows.v2.data.secrets import (
    SECRET_KIND,
    Secret,
    SecretUnavailableError,
    reveal_secrets,
    to_secret,
)
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref
from roboflow_workflows.v2.engine.errors import (
    ActiveRunError,
    StepExecutionError,
    WorkflowInputError,
)
from roboflow_workflows.v2.engine.recording.codecs import CodecRegistry
from roboflow_workflows.v2.engine.recording.store import open_recording

from tests.v2.support.recording.capture import WAIT, Feed, feed_values, field, group

SENTINEL = "sk-test-SENTINEL-7f3a"
HEADER_SENTINEL = "sk-test-HEADER-5d10"
FLAG = "ALLOW_WORKFLOW_BLOCKS_ACCESSING_ENVIRONMENTAL_VARIABLES"


class FakeClient:
    """Keeps every plaintext request; can fail with the token in its message."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.sent: List[Dict[str, Any]] = []

    def send(self, *, token: str, headers: Dict[str, str]) -> None:
        if self.fail:
            raise ConnectionError(
                f"GET https://hooks.test/?key={token} failed with headers {headers}"
            )
        self.sent.append({"token": token, "headers": headers})


class SecretSink(Block):
    """A credential consumer written the way sink blocks should be."""

    type = "test/secret_sink@v1"
    outputs = {
        "error_status": Output(BOOLEAN_KIND, description="Whether sending failed."),
        "message": Output(STRING_KIND, description="Scrubbed outcome."),
    }

    class Params(BlockParams):
        token: str | Ref(SECRET_KIND, STRING_KIND) = Field(description="Credential.")
        headers: Dict[str, str | Ref(SECRET_KIND, STRING_KIND)] = Field(
            default_factory=dict, description="Extra headers; values may be secrets."
        )
        value: Optional[Any] | Ref() = Field(default=None, description="Pulse value.")

    def __init__(self, *, client: FakeClient) -> None:
        self._client = client

    def run(self, *, token: Any, headers: Dict[str, Any], value: Any) -> Dict[str, Any]:
        token = to_secret(token, name="token")
        revealed = reveal_secrets({"token": token, "headers": headers})
        try:
            self._client.send(**revealed.value)
        except ConnectionError as error:
            return {"error_status": True, "message": revealed.scrub(str(error))}

        return {"error_status": False, "message": "sent"}


CATALOGUE = Catalogue(
    [EnvironmentSecretsStoreBlock, SecretSink, StateSetBlock],
    sources=[Feed],
    kinds=[SECRET_KIND],
    codecs=[SECRET_RECORDING_CODEC],
)
STORE_STEP = {
    "type": "roboflow_core/environment_secrets_store@v1",
    "name": "secrets",
    "variables_storing_secrets": ["SLACK_TOKEN", "HEADER_TOKEN"],
}


def _sink_step(**params: Any) -> Dict[str, Any]:
    step = {
        "type": "test/secret_sink@v1",
        "name": "sink",
        "token": "$steps.secrets.slack_token",
        "headers": {"X-Api-Key": "$steps.secrets.header_token", "Accept": "json"},
        **params,
    }

    return step


def _output(name: str, selector: str) -> Dict[str, str]:
    return {"type": "JsonField", "name": name, "selector": selector}


def _compile(steps: List[Dict[str, Any]], outputs: List[Dict[str, str]], **extra: Any):
    definition = {"version": "2.0", "steps": steps, "outputs": outputs, **extra}
    plan = compile_workflow(definition, catalogue=CATALOGUE)

    return plan


def _host_secrets() -> StaticSecrets:
    return StaticSecrets({"SLACK_TOKEN": SENTINEL, "HEADER_TOKEN": HEADER_SENTINEL})


def _chain_text(error: BaseException) -> str:
    texts = []
    while error is not None:
        texts.append(f"{type(error).__name__}: {error}")
        error = error.__cause__ or error.__context__

    return "\n".join(texts)


def test_a_sink_gets_plaintext_while_outputs_show_only_names() -> None:
    # given
    plan = _compile(
        [STORE_STEP, _sink_step()],
        [
            _output("token", "$steps.secrets.slack_token"),
            _output("message", "$steps.sink.message"),
        ],
    )
    client = FakeClient()
    session = plan.create_session(
        resources={"secrets": _host_secrets(), "client": client}
    )

    # when
    result = session.run({})
    python_row = result.rows()[0]
    serialized_row = result.rows(serialize=True)[0]

    # then
    assert client.sent == [
        {"token": SENTINEL, "headers": {"X-Api-Key": HEADER_SENTINEL, "Accept": "json"}}
    ]
    assert isinstance(python_row["token"], Secret)
    assert python_row["token"].name == "SLACK_TOKEN"
    assert serialized_row == {"token": "<secret SLACK_TOKEN>", "message": "sent"}
    assert SENTINEL not in repr(python_row) + repr(serialized_row)


def test_a_failing_request_message_is_scrubbed_of_nested_secrets() -> None:
    # given
    plan = _compile(
        [STORE_STEP, _sink_step()], [_output("message", "$steps.sink.message")]
    )
    session = plan.create_session(
        resources={"secrets": _host_secrets(), "client": FakeClient(fail=True)}
    )

    # when
    message = session.run({}).rows(serialize=True)[0]["message"]

    # then
    assert "key=<secret SLACK_TOKEN>" in message
    assert "<secret HEADER_TOKEN>" in message
    assert SENTINEL not in message
    assert HEADER_SENTINEL not in message


def test_a_secret_workflow_input_reaches_the_sink_and_serializes_as_a_marker() -> None:
    # given
    plan = _compile(
        [_sink_step(token="$inputs.token", headers={})],
        [_output("token", "$inputs.token"), _output("message", "$steps.sink.message")],
        inputs=[{"type": "WorkflowParameter", "name": "token", "kind": ["secret"]}],
    )
    client = FakeClient()

    # when
    row = (
        plan.create_session(resources={"client": client})
        .run({"token": SENTINEL})
        .rows(serialize=True)[0]
    )

    # then
    assert client.sent == [{"token": SENTINEL, "headers": {}}]
    assert row == {"token": "<secret input>", "message": "sent"}


def test_a_non_string_secret_input_fails_without_showing_the_value() -> None:
    # given
    plan = _compile(
        [_sink_step(token="$inputs.token", headers={})],
        [_output("message", "$steps.sink.message")],
        inputs=[{"type": "WorkflowParameter", "name": "token", "kind": ["secret"]}],
    )

    # when
    with pytest.raises(WorkflowInputError) as raised:
        plan.create_session(resources={"client": FakeClient()}).run({"token": 734921})

    # then
    assert "a secret must be a str, got int" in str(raised.value)
    assert "734921" not in str(raised.value)


def test_a_closed_environment_fails_clearly_and_reads_nothing(monkeypatch) -> None:
    # given
    monkeypatch.setattr(environment, FLAG, False)
    monkeypatch.setenv("SLACK_TOKEN", SENTINEL)
    plan = _compile(
        [{**STORE_STEP, "variables_storing_secrets": ["SLACK_TOKEN"]}],
        [_output("token", "$steps.secrets.slack_token")],
    )

    # when
    with pytest.raises(StepExecutionError) as raised:
        plan.create_session().run({})

    # then
    text = _chain_text(raised.value)
    assert FLAG in text
    assert SENTINEL not in text


def test_an_open_environment_yields_secrets_and_none_for_unset(monkeypatch) -> None:
    # given
    monkeypatch.setattr(environment, FLAG, True)
    monkeypatch.setenv("V2_TEST_SECRET_TOKEN", SENTINEL)
    monkeypatch.delenv("V2_TEST_SECRET_UNSET", raising=False)
    plan = _compile(
        [
            {
                **STORE_STEP,
                "variables_storing_secrets": [
                    "V2_TEST_SECRET_TOKEN",
                    "V2_TEST_SECRET_UNSET",
                ],
            }
        ],
        [
            _output("token", "$steps.secrets.v2_test_secret_token"),
            _output("unset", "$steps.secrets.v2_test_secret_unset"),
        ],
    )

    # when
    row = plan.create_session().run({}).rows()[0]

    # then
    assert row["token"].reveal() == SENTINEL
    assert row["unset"] is None


def test_host_static_secrets_work_with_a_closed_environment(monkeypatch) -> None:
    # given
    monkeypatch.setattr(environment, FLAG, False)
    plan = _compile([STORE_STEP], [_output("token", "$steps.secrets.slack_token")])

    # when
    row = plan.create_session(resources={"secrets": _host_secrets()}).run({}).rows()[0]

    # then
    assert row["token"].reveal() == SENTINEL


def test_writing_a_secret_to_managed_state_fails_without_the_value() -> None:
    # given
    plan = _compile(
        [
            STORE_STEP,
            {
                "type": "roboflow_core/state_set@v1",
                "name": "keep",
                "key": "token",
                "value": "$steps.secrets.slack_token",
            },
        ],
        [_output("applied", "$steps.keep.applied")],
    )

    # when
    with pytest.raises(StepExecutionError) as raised:
        plan.create_session(resources={"secrets": _host_secrets()}).run({})

    # then
    assert "Secret is not a portable JSON value" in _chain_text(raised.value)
    assert SENTINEL not in _chain_text(raised.value)


def _recorded_definition() -> Dict[str, Any]:
    return {
        "version": "2.0",
        "inputs": [{"type": "WorkflowParameter", "name": "capture_dir"}],
        "sources": [{"type": "test/recorded_feed@v1", "name": "cam", "feed": "cam"}],
        "steps": [STORE_STEP, _sink_step(value="$sources.cam.value")],
        "outputs": [
            group(
                "frames",
                "$sources.cam.value",
                field("value", "$sources.cam.value"),
                field("token", "$steps.secrets.slack_token"),
                field("message", "$steps.sink.message"),
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
                        "type": "test/secret_sink@v1",
                        "name": "resend",
                        "token": "$sources.frames.token",
                        "value": "$sources.frames.value",
                    }
                ],
                "outputs": [
                    group(
                        "reviewed",
                        "$sources.frames.value",
                        field("message", "$steps.resend.message"),
                    )
                ],
            },
        },
    }


@pytest.fixture()
def recorded(tmp_path: Path):
    """Two recorded pulses whose sink calls received the plaintext."""
    plan = compile_workflow(_recorded_definition(), catalogue=CATALOGUE)
    client = FakeClient()
    delivered: List[Any] = []
    session = plan.create_session(
        {
            "feeds": {"cam": feed_values(1.0, 2.0)},
            "secrets": _host_secrets(),
            "client": client,
        }
    )
    run = session.start(
        {"capture_dir": str(tmp_path / "rec")}, handlers={"frames": delivered.append}
    )
    run.wait(WAIT)

    return plan, tmp_path / "rec", client, delivered


def test_a_recording_stores_secret_names_but_never_values(recorded) -> None:
    # given
    _, directory, client, delivered = recorded

    # when
    stored = b"".join(
        path.read_bytes() for path in directory.rglob("*") if path.is_file()
    )

    # then
    assert [request["token"] for request in client.sent] == [SENTINEL, SENTINEL]
    assert [result.outputs.data["token"].name for result in delivered] == [
        "SLACK_TOKEN",
        "SLACK_TOKEN",
    ]
    assert b"SLACK_TOKEN" in stored
    assert SENTINEL.encode() not in stored
    assert HEADER_SENTINEL.encode() not in stored


def test_reading_a_recording_restores_unavailable_secrets(recorded) -> None:
    # given
    _, directory, _, _ = recorded
    recording = open_recording(
        directory, codecs=CodecRegistry(CATALOGUE.codecs.values())
    )

    # when
    tokens = [
        chunk.values["token"] for chunk in recording.group("frames").iter_chunks()
    ]

    # then
    assert [token.name for token in tokens] == ["SLACK_TOKEN", "SLACK_TOKEN"]
    assert not any(token.available for token in tokens)
    with pytest.raises(SecretUnavailableError):
        tokens[0].reveal()


def test_a_replayed_secret_makes_the_sink_fail_and_sends_nothing(
    recorded, monkeypatch
) -> None:
    # given: even an open environment holding the value is never reread
    plan, directory, _, _ = recorded
    monkeypatch.setattr(environment, FLAG, True)
    monkeypatch.setenv("SLACK_TOKEN", SENTINEL)
    replay_client = FakeClient()
    reviewed: List[Any] = []

    # when
    run = plan.retrospective.start(
        recording=directory,
        resources={"client": replay_client},
        handlers={"reviewed": reviewed.append},
    )
    with pytest.raises(ActiveRunError) as raised:
        run.wait(WAIT)

    # then
    text = _chain_text(raised.value)
    assert "SecretUnavailableError: secret 'SLACK_TOKEN' has no value here" in text
    assert SENTINEL not in text
    assert replay_client.sent == []
    assert reviewed == []

"""OPC author tests through the real compiler; recording ports, no network."""

from dataclasses import FrozenInstanceError
from threading import Event, Thread

import pytest
from roboflow_workflows.v2.blocks.industrial.opc import (
    BACKGROUND_WRITE_MESSAGE,
    BLOCK_TYPE,
    OPCSinkError,
    OPCWriterSinkBlock,
)
from roboflow_workflows.v2.blocks.secrets import SECRET_RECORDING_CODEC
from roboflow_workflows.v2.blocks.sinks.runtime import BackgroundEffects
from roboflow_workflows.v2.data.secrets import SECRET_KIND, Secret
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.errors import (
    ParamsValidationError,
    StepExecutionError,
)
from roboflow_workflows.v2.engine.introspection.workload import discover_workload

CATALOGUE = Catalogue(
    [OPCWriterSinkBlock], kinds=[SECRET_KIND], codecs=[SECRET_RECORDING_CODEC]
)
WAIT = 5


class RecordingPort:
    """Caller-owned fake operation port; close is observable."""

    def __init__(self, *, entered=None, gate=None, error=None):
        self.calls = []
        self.entered = entered
        self.gate = gate
        self.error = error
        self.closed = False

    def write(self, target, request):
        assert not self.closed
        self.calls.append((target, request))
        if self.entered is not None:
            self.entered.set()
        if self.gate is not None:
            assert self.gate.wait(WAIT)
        if self.error is not None:
            raise self.error

    def close(self):
        self.closed = True


def compile_opc(**overrides):
    step = dict(
        type=BLOCK_TYPE,
        name="sink",
        url="$inputs.endpoint",
        namespace="2",
        user_name="writer",
        password="$inputs.password",
        object_name="Line/Station",
        variable_name="Count",
        value="$inputs.value",
        value_type="Integer",
        cooldown_seconds=0,
        fire_and_forget="$inputs.background",
    )
    step.update(overrides)
    definition = {
        "version": "2.0",
        "inputs": [
            {"type": "WorkflowParameter", "name": "value", "default_value": 42},
            {"type": "WorkflowParameter", "name": "endpoint", "kind": ["secret"]},
            {"type": "WorkflowParameter", "name": "password", "kind": ["secret"]},
            {"type": "WorkflowParameter", "name": "background", "default_value": False},
        ],
        "steps": [step],
        "outputs": [
            {"type": "JsonField", "name": name, "selector": f"$steps.sink.{name}"}
            for name in ("disabled", "throttling_status", "error_status", "message")
        ],
    }
    plan = compile_workflow(definition, catalogue=CATALOGUE)

    return plan


def inputs(**overrides):
    values = dict(
        value=42,
        endpoint="opc.tcp://writer:PW-SENTINEL@private.invalid:4840/secret-endpoint",
        password="PW-SENTINEL",
        background=False,
    )
    values.update(overrides)

    return values


def row(session, **overrides):
    result = session.run(inputs(**overrides)).rows(serialize=True)[0]

    return result


def test_compiled_retained_session_converts_values_and_borrows_client():
    plan = compile_opc()
    port = RecordingPort()
    with plan.create_session(resources={"opc_client": port}) as session:
        for value in [str(2**63 - 1), "-42", 0]:
            result = row(session, value=value)
            assert result == {
                "disabled": False,
                "throttling_status": False,
                "error_status": False,
                "message": "Value set successfully",
            }
        assert [request.value for _, request in port.calls] == [2**63 - 1, -42, 0]
        target, request = port.calls[0]
        assert target.url.name == "input" and target.password.name == "input"
        with pytest.raises(FrozenInstanceError):
            request.value = 12
    assert not port.closed


def test_borrowed_client_survives_two_sessions_and_processing_reset():
    plan = compile_opc()
    port = RecordingPort()
    first = plan.create_session(resources={"opc_client": port})
    second = plan.create_session(resources={"opc_client": port})
    try:
        assert not row(first)["error_status"]
        receipt = first.apply_update(first.prepare_update(plan, reset=True))
        assert receipt is not None
        assert not port.closed
        first.close()
        assert not row(second, value=7)["error_status"]
    finally:
        first.close()
        second.close()
    assert not port.closed and len(port.calls) == 2
    port.close()
    assert port.closed


def test_reset_with_old_effect_outstanding_preserves_its_immutable_request_and_host_port():
    gate, entered = Event(), Event()
    old_port = RecordingPort(entered=entered, gate=gate)
    new_port = RecordingPort()
    plan = compile_opc()
    effects = BackgroundEffects(max_workers=1, max_pending=3)
    session = plan.create_session(
        resources={"opc_client": old_port, "sink_effects": effects}
    )
    try:
        assert (
            row(session, value=41, background=True)["message"]
            == BACKGROUND_WRITE_MESSAGE
        )
        assert entered.wait(WAIT)
        update = session.prepare_update(
            plan,
            resources={"opc_client": new_port, "sink_effects": effects},
            reset=True,
        )
        receipt = session.apply_update(update)
        assert receipt is not None
        assert not old_port.closed and not new_port.closed
        assert not row(session, value=99, background=True)["error_status"]
        session.close()
        assert not old_port.closed and not new_port.closed
        assert not effects.drain(timeout=0)
    finally:
        gate.set()
        session.close()
        effects.close()
    assert old_port.calls[0][1].value == 41
    assert new_port.calls[0][1].value == 99
    assert effects.stats().completed == 2
    assert not old_port.closed and not new_port.closed
    old_port.close()
    new_port.close()


def test_queued_effect_detaches_selector_input_dictionary_before_host_mutation():
    entered, gate = Event(), Event()

    def blocker():
        entered.set()
        assert gate.wait(WAIT)
        return False, "done"

    port = RecordingPort()
    plan = compile_opc()
    with BackgroundEffects(max_pending=2) as effects:
        effects.deliver(blocker, label="test blocker")
        assert entered.wait(WAIT)
        with plan.create_session(
            resources={"opc_client": port, "sink_effects": effects}
        ) as session:
            values = inputs(value="42", background=True)
            try:
                assert not session.run(values).rows()[0]["error_status"]
                values["value"] = "99"
                values["endpoint"] = "opc.tcp://mutated.invalid:1"
                values["password"] = "changed"
                assert port.calls == []
            finally:
                gate.set()
    target, request = port.calls[0]
    assert request.value == 42
    assert "private.invalid" in target.url.reveal()
    assert target.password.reveal() == "PW-SENTINEL"


def test_effect_close_drains_before_host_closes_borrowed_port_and_rejects_new_dispatch():
    entered, gate, closed = Event(), Event(), Event()
    port = RecordingPort(entered=entered, gate=gate)
    plan = compile_opc()
    effects = BackgroundEffects()
    session = plan.create_session(
        resources={"opc_client": port, "sink_effects": effects}
    )

    def close_effects():
        effects.close()
        closed.set()

    closer = Thread(target=close_effects)
    try:
        assert row(session, background=True)["message"] == BACKGROUND_WRITE_MESSAGE
        assert entered.wait(WAIT)
        session.close()
        closer.start()
        assert not closed.is_set() and not port.closed
    finally:
        gate.set()
        if closer.ident is not None:
            closer.join(WAIT)
        else:
            effects.close()
        session.close()
    assert closed.is_set() and not closer.is_alive()
    assert not port.closed
    with plan.create_session(
        resources={"opc_client": port, "sink_effects": effects}
    ) as after:
        result = row(after, background=True)
        assert result["error_status"] and "dispatch failed" in result["message"]
    assert len(port.calls) == 1
    port.close()


def test_background_failure_scrubs_credentials_in_retained_failure_and_logs(caplog):
    port = RecordingPort(
        error=OPCSinkError(
            "endpoint opc.tcp://writer:PW-SENTINEL@private.invalid:4840/secret-endpoint password PW-SENTINEL"
        )
    )
    with BackgroundEffects() as effects:
        with compile_opc().create_session(
            resources={"opc_client": port, "sink_effects": effects}
        ) as session:
            assert not row(session, background=True)["error_status"]
    text = repr(effects.failures()) + caplog.text
    assert "PW-SENTINEL" not in text and "private.invalid" not in text
    assert effects.stats().failed == 1
    assert not port.closed


def test_compiled_disabled_sink_does_not_reveal_replayed_secrets():
    port = RecordingPort()
    with compile_opc(disable_sink=True).create_session(
        resources={"opc_client": port}
    ) as session:
        result = row(
            session,
            endpoint=Secret.unavailable("endpoint"),
            password=Secret.unavailable("password"),
        )
    assert result["disabled"] and not result["error_status"]
    assert port.calls == []


def test_compiled_unavailable_secret_stops_before_dispatch():
    port = RecordingPort()
    with compile_opc().create_session(resources={"opc_client": port}) as session:
        with pytest.raises(StepExecutionError):
            row(session, password=Secret.unavailable("password"))
    assert port.calls == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("timeout", 0),
        ("max_retries", 0),
        ("cooldown_seconds", -1),
        ("retry_backoff_seconds", float("inf")),
        ("value_type", "Byte"),
        ("node_lookup_mode", "missing"),
    ],
)
def test_invalid_literal_parameters_fail_compilation(field, value):
    with pytest.raises(ParamsValidationError):
        compile_opc(**{field: value})


def test_selected_container_value_fails_before_port_use():
    port = RecordingPort()
    with compile_opc().create_session(resources={"opc_client": port}) as session:
        with pytest.raises(StepExecutionError):
            row(session, value={"mutable": [1]})
    assert port.calls == []


def test_workload_declares_external_effect_and_incomplete_background_restrictions():
    workload = discover_workload(compile_opc()).step("$steps.sink").describe()
    assert "external_request" in repr(workload["operations"]).lower()
    restrictions = workload["restrictions"]
    assert restrictions["complete"] is False
    assert len(restrictions["items"]) == 1
    assert restrictions["unknown_reasons"][0]["details"]["field"] == "fire_and_forget"


def test_literal_background_restrictions_are_complete_and_preserve_cooldown_caveat():
    description = (
        discover_workload(compile_opc(fire_and_forget=True))
        .step("$steps.sink")
        .describe()
    )
    assert description["restrictions"]["complete"] is True
    assert len(description["restrictions"]["items"]) == 2

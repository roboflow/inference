"""Real compiler/session/effects with in-process MQTT boundaries; no broker."""

import socket
import threading
from types import SimpleNamespace

import pytest
from roboflow_workflows.v2.blocks.industrial import mqtt
from roboflow_workflows.v2.blocks.secrets import SECRET_RECORDING_CODEC
from roboflow_workflows.v2.blocks.sinks.runtime import BackgroundEffects
from roboflow_workflows.v2.data.secrets import SECRET_KIND, Secret
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.errors import StepExecutionError

CATALOGUE = Catalogue(
    [mqtt.MQTTWriterBlock, mqtt.MQTTReaderBlock],
    kinds=[SECRET_KIND],
    codecs=[SECRET_RECORDING_CODEC],
)


@pytest.fixture(autouse=True)
def no_network_and_policy(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No network permission")

    for name in ["connect", "connect_ex", "bind"]:
        monkeypatch.setattr(socket.socket, name, forbidden)
    monkeypatch.setattr(
        mqtt.environment, "MQTT_WORKFLOWS_BLOCKS_ALLOW_USER_PROVIDED_HOST", True
    )
    monkeypatch.setattr(
        mqtt.environment, "MQTT_WORKFLOWS_BLOCKS_WHITELISTED_HOSTS", None
    )
    monkeypatch.setattr(mqtt.environment, "GCP_SERVERLESS", False)
    monkeypatch.setattr(mqtt.environment, "LAMBDA", False)


def plan(*, reader=False, **settings):
    step = {
        "type": mqtt.READER_BLOCK_TYPE if reader else mqtt.WRITER_BLOCK_TYPE,
        "name": "mqtt",
        "host": "broker.example",
        "port": 1883,
        "topic": "factory/state",
        "timeout": 0.001,
    }
    if not reader:
        step["message"] = "$inputs.message"
    step.update(settings)
    names = (
        ["value", "payload", "is_new", "error_status", "error_message"]
        if reader
        else ["error_status", "message"]
    )
    definition = {
        "version": "2.0",
        "inputs": [
            {"type": "WorkflowParameter", "name": "message", "default_value": "hello"},
            {"type": "WorkflowParameter", "name": "password", "kind": ["secret"]},
        ],
        "steps": [step],
        "outputs": [
            {"type": "JsonField", "name": n, "selector": f"$steps.mqtt.{n}"}
            for n in names
        ],
    }
    return compile_workflow(definition, catalogue=CATALOGUE)


class RecordingWriter:
    def __init__(self, *, error=False, gate=None):
        self.calls = []
        self.closes = 0
        self.error = error
        self.gate = gate

    def publish(self, target, message, *, retain):
        assert self.closes == 0
        self.calls.append((target, message, retain))
        if self.gate:
            entered, release = self.gate
            entered.set()
            assert release.wait(5)
        assert self.closes == 0
        if self.error:
            raise RuntimeError("rejected PASSWORD-SENTINEL")

    def close(self):
        self.closes += 1


class CallbackClient:
    """Fake paho boundary for the actual host subscription class."""

    def __init__(self, **kwargs):
        self.stops = 0
        self.disconnects = 0
        self.socket_resets = 0

    def reconnect_delay_set(self, **kwargs):
        pass

    def connect_async(self, *args, **kwargs):
        pass

    def loop_start(self):
        self.on_connect(self, None, {}, 0)
        self.on_subscribe(self, None, 1, [0])

    def subscribe(self, *args, **kwargs):
        return 0, 1

    def disconnect(self):
        self.disconnects += 1

    def loop_stop(self):
        self.stops += 1

    def _reset_sockets(self):
        self.socket_resets += 1

    def emit(self, value):
        self.on_message(
            self,
            None,
            SimpleNamespace(topic="factory/state", payload=value, retain=False),
        )


CLEANUP_FAILED = "MQTT cleanup failed; shutdown is not confirmed."


def failing_cleanup_writer_sdk(monkeypatch, *, confirmed):
    """Fake paho writer boundary whose disconnect fails after one publish."""
    publishes = []

    class WriterClient(CallbackClient):
        def username_pw_set(self, *args):
            pass

        def connect(self, *args, **kwargs):
            return 0

        def loop_start(self):
            self.on_connect(self, None, {}, 0)

        def publish(self, *args, **kwargs):
            publishes.append(args)
            return SimpleNamespace(
                rc=0,
                wait_for_publish=lambda **kwargs: None,
                is_published=lambda: confirmed,
            )

        def disconnect(self):
            raise RuntimeError("disconnect echoed PASSWORD-SENTINEL")

    monkeypatch.setattr(
        mqtt, "_import_paho", lambda: SimpleNamespace(Client=WriterClient)
    )
    return publishes


@pytest.mark.parametrize(
    "confirmed,disposition",
    [(False, mqtt._UNKNOWN_DELIVERY), (True, "Message published successfully.")],
)
def test_background_cleanup_failure_keeps_publish_disposition(
    monkeypatch, confirmed, disposition
):
    publishes = failing_cleanup_writer_sdk(monkeypatch, confirmed=confirmed)
    effects = BackgroundEffects(max_workers=1)
    compiled = plan(
        fire_and_forget=True, qos=1, username="u", password="$inputs.password"
    )
    try:
        with compiled.create_session(resources={"sink_effects": effects}) as session:
            row = session.run({"message": "one", "password": "PASSWORD-SENTINEL"}).rows(
                serialize=True
            )[0]
        assert row == {
            "error_status": False,
            "message": "MQTT publish scheduled; delivery not yet confirmed",
        }
    finally:
        effects.close()
    assert [f.message for f in effects.failures()] == [
        f"{disposition} {CLEANUP_FAILED}"
    ]
    assert effects.stats().failed == 1 and effects.stats().completed == 0
    assert len(publishes) == 1
    assert "PASSWORD-SENTINEL" not in str(effects.failures())


def test_compiled_fail_fast_cleanup_failure_keeps_unknown_delivery(monkeypatch):
    publishes = failing_cleanup_writer_sdk(monkeypatch, confirmed=False)
    compiled = plan(fail_fast=True, qos=1, username="u", password="$inputs.password")
    with compiled.create_session() as session:
        with pytest.raises(StepExecutionError) as error:
            session.run({"password": "PASSWORD-SENTINEL"})
    assert f"{mqtt._UNKNOWN_DELIVERY} {CLEANUP_FAILED}" in str(error.value)
    assert "PASSWORD-SENTINEL" not in str(error.value)
    assert len(publishes) == 1


def test_retained_writer_reuses_borrowed_host_and_never_closes_it():
    writer = RecordingWriter()
    compiled = plan()
    first = compiled.create_session(resources={"mqtt_writer": writer})
    second = compiled.create_session(resources={"mqtt_writer": writer})
    try:
        assert not first.run({"message": "first"}).rows(serialize=True)[0][
            "error_status"
        ]
        first.close()
        assert not second.run({"message": "second"}).rows(serialize=True)[0][
            "error_status"
        ]
        writer.error = True
        row = second.run({"message": "third"}).rows(serialize=True)[0]
        assert row["error_status"]
        assert writer.closes == 0
    finally:
        first.close()
        second.close()
        writer.close()
    assert writer.closes == 1
    assert [c[1] for c in writer.calls] == ["first", "second", "third"]


@pytest.mark.parametrize("error", [False, True])
def test_processing_reset_retains_outstanding_old_effect_and_host_lifetime(error):
    entered, release = threading.Event(), threading.Event()
    writer = RecordingWriter(error=error, gate=(entered, release))
    effects = BackgroundEffects(max_workers=1)
    compiled = plan(fire_and_forget=True, username="u", password="$inputs.password")
    session = compiled.create_session(
        resources={"mqtt_writer": writer, "sink_effects": effects}
    )
    try:
        row = session.run({"message": "old", "password": "PASSWORD-SENTINEL"}).rows(
            serialize=True
        )[0]
        assert "scheduled" in row["message"]
        assert entered.wait(3)
        old = session.instances[("mqtt",)]
        session.apply_update(session.prepare_update(compiled, reset=True))
        assert session.instances[("mqtt",)] is not old
        assert effects.stats().pending == 1 and writer.closes == 0
        session.run({"message": "new", "password": "PASSWORD-SENTINEL"})
        session.close()
        assert writer.closes == 0 and effects.stats().pending == 2
    finally:
        release.set()
        session.close()
        effects.close()
    assert [c[1] for c in writer.calls] == ["old", "new"]
    assert effects.stats().failed == 2 * int(error)
    assert effects.stats().completed == 2 * int(not error)
    assert effects.stats().abandoned == 0
    assert "PASSWORD-SENTINEL" not in str(effects.failures())
    writer.close()


def test_default_acquisition_waits_for_admitted_effect_start(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    effects = BackgroundEffects(max_workers=1)
    owners = []

    class Owner(RecordingWriter):
        def __init__(self, target, **kwargs):
            super().__init__()
            owners.append(self)

    monkeypatch.setattr(mqtt, "MQTTPublisher", Owner)

    def hold():
        entered.set()
        assert release.wait(5)
        return False, "released"

    effects.deliver(hold, label="barrier")
    assert entered.wait(3)
    compiled = plan(fire_and_forget=True)
    session = compiled.create_session(resources={"sink_effects": effects})
    inputs = {"message": "frozen"}
    try:
        session.run(inputs)
        inputs["message"] = "changed"
        assert not owners
        session.close()
        assert not owners
    finally:
        release.set()
        effects.close()
        session.close()
    assert len(owners) == 1 and owners[0].closes == 1
    assert owners[0].calls[0][1] == "frozen"


def test_compiled_disabled_paths_do_not_reveal_secret_or_acquire(monkeypatch):
    monkeypatch.setattr(mqtt, "_import_paho", lambda: pytest.fail("driver acquired"))
    for reader in [False, True]:
        compiled = plan(
            reader=reader,
            password="$inputs.password",
            **({"disable_reader": True} if reader else {"disable_sink": True}),
        )
        with compiled.create_session() as session:
            assert not session.run({"password": Secret.unavailable("pw")}).rows(
                serialize=True
            )[0]["error_status"]


def test_compiled_secret_failure_and_fail_fast_are_observable():
    compiled = plan(username="u", password="$inputs.password", fail_fast=True)
    writer = RecordingWriter(error=True)
    with compiled.create_session(resources={"mqtt_writer": writer}) as session:
        with pytest.raises(StepExecutionError) as error:
            session.run({"password": "PASSWORD-SENTINEL"})
        assert "PASSWORD-SENTINEL" not in str(error.value)
    assert len(writer.calls) == 1


def test_compiled_reader_persists_between_runs_and_session_closes_do_not_close_owner(
    monkeypatch,
):
    monkeypatch.setattr(
        mqtt, "_import_paho", lambda: SimpleNamespace(Client=CallbackClient)
    )
    target = mqtt.MQTTTarget(
        "broker.example", 1883, "factory/state", timeout=0.001, read_mode="sequential"
    )
    owner = mqtt.MQTTSubscription(target)
    other = mqtt.MQTTSubscription(target)
    compiled = plan(reader=True, read_mode="sequential")
    first = compiled.create_session(resources={"mqtt_subscription": owner})
    shared = compiled.create_session(resources={"mqtt_subscription": owner})
    separate = compiled.create_session(resources={"mqtt_subscription": other})
    try:
        owner._client.emit(b"one")
        owner._client.emit(b"two")
        other._client.emit(b"independent")
        assert first.run({}).rows(serialize=True)[0]["value"] == "one"
        first.close()
        assert shared.run({}).rows(serialize=True)[0]["value"] == "two"
        assert separate.run({}).rows(serialize=True)[0]["value"] == "independent"
        owner._client.emit(b"before-reset")
        shared.apply_update(shared.prepare_update(compiled, reset=True))
        assert shared.run({}).rows(serialize=True)[0]["value"] == "before-reset"
        owner.reset_buffer()
        assert shared.run({}).rows(serialize=True)[0]["value"] is None
        assert owner._client.stops == other._client.stops == 0
    finally:
        first.close()
        shared.close()
        separate.close()
        owner.close()
        other.close()
    assert owner._client.stops == other._client.stops == 1
    assert owner._client.socket_resets == other._client.socket_resets == 1


def test_compiled_validation_prevents_invalid_qos_and_nonstring_message():
    from roboflow_workflows.v2.engine.errors import ParamsValidationError

    with pytest.raises(ParamsValidationError):
        plan(qos=3)
    writer = RecordingWriter()
    with plan().create_session(resources={"mqtt_writer": writer}) as session:
        with pytest.raises(StepExecutionError):
            session.run({"message": 42})
    assert not writer.calls

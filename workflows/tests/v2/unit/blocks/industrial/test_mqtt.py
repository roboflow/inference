"""In-process fake-paho boundary tests. No installed-SDK or broker claim."""

import copy
import json
import socket
import sys
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest
from roboflow_workflows.v2.blocks.industrial import mqtt
from roboflow_workflows.v2.data.secrets import Secret, SecretUnavailableError


class FakeClient:
    """Deterministic callback boundary; no actual MQTT implementation."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.events = []
        self.rc = 0
        self.confirmed = True
        self.connack = 0
        self.fail = None
        self.mid = 0
        self.publish_gate = None
        self.close_gate = None
        self.on_connect = self.on_disconnect = self.on_subscribe = None
        self.on_message = self.on_connect_fail = None

    def _record(self, name, *args):
        self.events.append((name, args))
        if self.fail == name:
            raise RuntimeError("driver echoed SENTINEL-PASSWORD")

    def username_pw_set(self, *args):
        self._record("auth", *args)

    def tls_set(self, **kwargs):
        self._record("tls", kwargs)

    def reconnect_delay_set(self, **kwargs):
        self._record("reconnect_delay", kwargs)

    def connect(self, *args, **kwargs):
        self._record("connect", args, kwargs)
        return 0

    def connect_async(self, *args, **kwargs):
        return self.connect(*args, **kwargs)

    def loop_start(self):
        self._record("start")
        self.on_connect(self, None, {}, self.connack)
        if self.on_subscribe and self.connack == 0:
            self.on_subscribe(self, None, self.mid, [0])

    def subscribe(self, *args, **kwargs):
        self._record("subscribe", args, kwargs)
        self.mid += 1
        return 0, self.mid

    def publish(self, *args, **kwargs):
        self._record("publish", args, kwargs)
        if self.publish_gate:
            entered, release = self.publish_gate
            entered.set()
            assert release.wait(5)
        return SimpleNamespace(
            rc=self.rc, wait_for_publish=self.wait, is_published=lambda: self.confirmed
        )

    def wait(self, **kwargs):
        self._record("wait", kwargs)

    def disconnect(self):
        self._record("disconnect")
        if self.close_gate:
            entered, release = self.close_gate
            entered.set()
            assert release.wait(5)
        if self.on_disconnect:
            self.on_disconnect(self, None, 0)

    def loop_stop(self):
        self._record("stop")

    def _reset_sockets(self):
        self._record("reset_sockets")

    def emit(self, payload, *, topic="factory/state", retained=False):
        if self.on_message:
            self.on_message(
                self,
                None,
                SimpleNamespace(topic=topic, payload=payload, retain=retained),
            )


@pytest.fixture(autouse=True)
def policy_and_no_network(monkeypatch):
    monkeypatch.setattr(
        mqtt.environment, "MQTT_WORKFLOWS_BLOCKS_ALLOW_USER_PROVIDED_HOST", True
    )
    monkeypatch.setattr(
        mqtt.environment, "MQTT_WORKFLOWS_BLOCKS_WHITELISTED_HOSTS", None
    )
    monkeypatch.setattr(mqtt.environment, "GCP_SERVERLESS", False)
    monkeypatch.setattr(mqtt.environment, "LAMBDA", False)

    def forbidden(*args, **kwargs):
        raise AssertionError("No sockets or listeners permitted")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket.socket, "bind", forbidden)


@pytest.fixture
def factory(monkeypatch):
    clients = []
    options = {}

    class ConfiguredClient(FakeClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            for key, value in options.items():
                setattr(self, key, value)
            clients.append(self)

    monkeypatch.setattr(
        mqtt, "_import_paho", lambda: SimpleNamespace(Client=ConfiguredClient)
    )
    return clients, options


def target(**kwargs):
    return replace(
        mqtt.MQTTTarget("broker.example", 1883, "factory/state", timeout=0.001),
        **kwargs,
    )


def names(client):
    return [event[0] for event in client.events]


def test_disabled_precedes_secret_driver_and_effects(monkeypatch):
    monkeypatch.setattr(mqtt, "_import_paho", lambda: pytest.fail("SDK acquired"))
    secret = Secret.unavailable("BROKER")
    assert not mqtt.MQTTWriterBlock(disable_sinks=True).run(
        "b", "t", "m", password=secret
    )["error_status"]
    assert not mqtt.MQTTReaderBlock().run(
        "b", "t", password=secret, disable_reader=True
    )["error_status"]


def test_unavailable_secret_is_observable_before_acquisition(factory):
    with pytest.raises(SecretUnavailableError):
        mqtt.MQTTWriterBlock().run(
            "b", "t", "m", username="u", password=Secret.unavailable("BROKER")
        )
    assert factory[0] == []


@pytest.mark.parametrize("point", ["auth", "tls", "connect", "start"])
def test_partial_setup_unwinds_once_and_never_echoes_driver_credentials(factory, point):
    clients, options = factory
    options["fail"] = point
    result = mqtt.MQTTWriterBlock().run(
        "b", "t", "m", username="u", password="SENTINEL-PASSWORD", encryption="tls"
    )
    assert result["error_status"] and "SENTINEL" not in result["message"]
    assert names(clients[0])[-3:] == ["disconnect", "stop", "reset_sockets"]
    assert names(clients[0]).count("publish") == 0


@pytest.mark.parametrize("confirmed,rc", [(True, 0), (False, 0), (False, 4)])
@pytest.mark.parametrize("qos", [0, 1, 2])
def test_default_writer_one_attempt_and_exact_cleanup(factory, confirmed, rc, qos):
    clients, options = factory
    options.update(confirmed=confirmed, rc=rc)
    result = mqtt.MQTTWriterBlock().run(
        "b", "t", "é", qos=qos, retain=True, timeout=0.001
    )
    assert result["error_status"] == (not confirmed or rc != 0)
    if result["error_status"]:
        assert "unknown" in result["message"]
    client = clients[0]
    assert names(client).count("publish") == 1
    assert names(client)[-3:] == ["disconnect", "stop", "reset_sockets"]
    assert client.kwargs["reconnect_on_failure"] is False
    publish = next(e for e in client.events if e[0] == "publish")
    assert publish[1] == (("t", "é"), {"qos": qos, "retain": True})


CLEANUP_FAILED = "MQTT cleanup failed; shutdown is not confirmed."


@pytest.mark.parametrize(
    "options,disposition,publishes",
    [
        ({"confirmed": True}, "Message published successfully.", 1),
        ({"confirmed": False}, mqtt._UNKNOWN_DELIVERY, 1),
        ({"rc": 4}, mqtt._UNKNOWN_DELIVERY, 1),
        (
            {"connack": 5},
            "MQTT connection refused: not authorised (code 5); host must close and rebind.",
            0,
        ),
    ],
)
@pytest.mark.parametrize("point", ["disconnect", "stop", "reset_sockets"])
def test_cleanup_failure_keeps_primary_publish_disposition(
    factory, point, options, disposition, publishes
):
    factory[1].update(options, fail=point)
    result = mqtt.MQTTWriterBlock().run(
        "b", "t", "m", username="u", password="SENTINEL-PASSWORD", qos=1
    )
    assert result == {
        "error_status": True,
        "message": f"{disposition} {CLEANUP_FAILED}",
    }
    client = factory[0][0]
    assert names(client).count("publish") == publishes
    assert names(client).count("stop") == 1
    assert len(factory[0]) == 1


def test_fail_fast_cleanup_failure_keeps_unknown_delivery_without_replay(factory):
    factory[1].update(confirmed=False, fail="reset_sockets")
    with pytest.raises(mqtt.MQTTError) as raised:
        mqtt.MQTTWriterBlock().run(
            "b",
            "t",
            "m",
            username="u",
            password="SENTINEL-PASSWORD",
            qos=2,
            fail_fast=True,
        )
    assert str(raised.value) == f"{mqtt._UNKNOWN_DELIVERY} {CLEANUP_FAILED}"
    assert names(factory[0][0]).count("publish") == 1
    assert len(factory[0]) == 1


class PairSocket:
    """Recorded wake-up pair endpoint; never an operating-system socket."""

    def __init__(self, sockets, fail, close_error):
        if fail == "allocate" and sockets:
            raise OSError("second allocation failed")
        self.sockets, self.fail, self.close_error = sockets, fail, close_error
        self.closes = 0
        sockets.append(self)

    def _step(self, name):
        if name == self.fail:
            raise OSError(f"{name} failed SENTINEL-PASSWORD")

    def setsockopt(self, *args):
        self._step("setsockopt")

    def bind(self, address):
        self._step("bind")

    def listen(self, backlog):
        self._step("listen")

    def getsockname(self):
        return "127.0.0.1", 40000

    def setblocking(self, flag):
        pass

    def connect(self, address):
        self._step("connect")
        raise BlockingIOError()

    def accept(self):
        self._step("accept")
        accepted = PairSocket(self.sockets, None, self.close_error)
        return accepted, ("127.0.0.1", 40001)

    def close(self):
        self.closes += 1
        if self.close_error:
            raise OSError("close failed")


class PairClient(FakeClient):
    """Fake with paho 1.6.1's threaded-loop fields, so the adapter owns the pair."""

    _thread = None
    _sockpairR = _sockpairW = None

    def _thread_main(self):
        self.on_connect(self, None, {}, 0)

    def loop_stop(self):
        self._record("stop")
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def _reset_sockets(self):
        self._record("reset_sockets")
        for name in ("_sockpairR", "_sockpairW"):
            sock = getattr(self, name)
            if sock is not None:
                setattr(self, name, None)
                sock.close()


@pytest.fixture
def pair_sdk(monkeypatch):
    sockets, settings, clients = [], {"fail": None, "close_error": False}, []

    class Client(PairClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            clients.append(self)

    def new_socket(*args):
        return PairSocket(sockets, settings["fail"], settings["close_error"])

    sdk = SimpleNamespace(
        Client=Client,
        MQTT_ERR_INVAL=3,
        _socketpair_compat=lambda: pytest.fail("leaking paho helper used"),
        socket=SimpleNamespace(
            socket=new_socket,
            AF_INET=2,
            SOCK_STREAM=1,
            IPPROTO_IP=0,
            SOL_SOCKET=1,
            SO_REUSEADDR=4,
        ),
    )
    monkeypatch.setattr(mqtt, "_import_paho", lambda: sdk)
    return SimpleNamespace(sockets=sockets, settings=settings, clients=clients)


def test_wakeup_pair_is_transferred_and_closed_once(pair_sdk):
    # The fake network thread reports CONNACK; allow it time to run.
    owner = mqtt.MQTTPublisher(target(timeout=5))
    client = pair_sdk.clients[0]
    listener, reader, writer = pair_sdk.sockets
    assert (client._sockpairR, client._sockpairW) == (reader, writer)
    assert listener.closes == 1 and reader.closes == writer.closes == 0
    owner.publish(target(timeout=5), "one")
    owner.close()
    assert [s.closes for s in pair_sdk.sockets] == [1, 1, 1]
    assert client._thread is None and client._sockpairR is client._sockpairW is None


@pytest.mark.parametrize(
    "fail,acquired",
    [
        ("setsockopt", 1),
        ("bind", 1),
        ("listen", 1),
        ("allocate", 1),
        ("connect", 2),
        ("accept", 2),
    ],
)
def test_partial_wakeup_pair_closes_each_acquired_endpoint(pair_sdk, fail, acquired):
    pair_sdk.settings["fail"] = fail
    result = mqtt.MQTTWriterBlock().run(
        "b", "t", "m", username="u", password="SENTINEL-PASSWORD"
    )
    assert result == {
        "error_status": True,
        "message": "MQTT setup failed; check driver, credentials, TLS and broker configuration.",
    }
    assert [s.closes for s in pair_sdk.sockets] == [1] * acquired
    client = pair_sdk.clients[0]
    assert client._sockpairR is client._sockpairW is client._thread is None
    assert names(client)[-3:] == ["disconnect", "stop", "reset_sockets"]


def test_failed_endpoint_close_is_reported_after_closing_the_rest(pair_sdk):
    pair_sdk.settings.update(fail="accept", close_error=True)
    result = mqtt.MQTTWriterBlock().run("b", "t", "m")
    assert result == {
        "error_status": True,
        "message": "MQTT setup and cleanup failed; host intervention required.",
    }
    assert [s.closes for s in pair_sdk.sockets] == [1, 1]


def test_thread_start_failure_closes_transferred_pair(pair_sdk, monkeypatch):
    def failed_start(thread):
        raise RuntimeError("cannot start thread")

    monkeypatch.setattr(threading.Thread, "start", failed_start)
    result = mqtt.MQTTWriterBlock().run("b", "t", "m")
    # Joining an unstarted thread fails, so shutdown stays unconfirmed (c1).
    assert result == {
        "error_status": True,
        "message": "MQTT setup and cleanup failed; host intervention required.",
    }
    assert [s.closes for s in pair_sdk.sockets] == [1, 1, 1]
    assert pair_sdk.clients[0]._sockpairR is None


def test_close_idempotent_and_closed_owner_rejects_publish(factory):
    owner = mqtt.MQTTPublisher(target())
    owner.publish(target(), "one")
    owner.close()
    owner.close()
    with pytest.raises(mqtt.MQTTError, match="closed"):
        owner.publish(target(), "two")
    assert names(factory[0][0]).count("stop") == 1


def test_tls_permission_precedes_driver_acquisition(factory):
    result = mqtt.MQTTWriterBlock().run(
        "b", "t", "m", encryption="tls", ca_certificate_path="/not/opened"
    )
    assert result["error_status"] and "permission" in result["message"]
    assert not factory[0]


def test_ca_must_be_regular_file_before_acquisition(factory, tmp_path):
    with pytest.raises(mqtt.MQTTError, match="regular file"):
        mqtt.MQTTSubscription(
            target(encryption="tls", ca_certificate_path=str(tmp_path)),
            allow_access_to_file_system=True,
        )
    assert not factory[0]


def test_tls_configuration_has_no_insecure_override(factory, tmp_path):
    ca = tmp_path / "ca.pem"
    ca.write_text("mock boundary only")
    owner = mqtt.MQTTSubscription(
        target(encryption="tls", ca_certificate_path=str(ca)),
        allow_access_to_file_system=True,
    )
    assert ("tls", ({"ca_certs": str(ca)},)) in factory[0][0].events
    owner.close()


@pytest.mark.parametrize(
    "mode,count,first", [("latest", 3, "2"), ("sequential", 1002, "2")]
)
def test_queue_overflow_drops_oldest_and_repeat_is_detached(
    factory, mode, count, first
):
    t = target(read_mode=mode)
    owner = mqtt.MQTTSubscription(t)
    try:
        for i in range(count):
            factory[0][0].emit(str(i).encode())
        result = owner.read(t)
        assert result["value"] == first and result["is_new"]
        if mode == "latest":
            assert owner.read(t)["is_new"] is False
        else:
            assert owner.read(t)["value"] == "3"
        owner.reset_buffer()
        factory[0][0].emit(b'{"nested":{"a":1}}')
        result = owner.read(t)
        result["payload"]["nested"]["a"] = 99
        assert owner.read(t)["payload"] == {"nested": {"a": 1}}
    finally:
        owner.close()


def test_retained_duplicate_and_live_duplicate_semantics(factory):
    t = target(read_mode="sequential")
    owner = mqtt.MQTTSubscription(t)
    try:
        for retained, expected in [(False, True), (True, False), (False, True)]:
            factory[0][0].emit(b"hello", retained=retained)
            assert owner.read(t)["is_new"] is expected
        factory[0][0].emit(b"\xff")
        assert owner.read(t)["error_status"]
        factory[0][0].emit(b"[1,2]")
        assert owner.read(t)["payload"] == {}
    finally:
        owner.close()


def test_disconnect_drains_then_repeats_failure_and_reconnect_requires_matching_suback(
    factory,
):
    t = target(read_mode="sequential")
    owner = mqtt.MQTTSubscription(t)
    client = factory[0][0]
    try:
        client.emit(b"one")
        assert owner.read(t)["value"] == "one"
        client.emit(b"two")
        old_mid = client.mid
        client.on_disconnect(client, None, 7)
        assert owner.read(t)["value"] == "two"
        for _ in range(2):
            assert owner.read(t)["error_status"]
        client.on_connect(client, None, {}, 0)
        client.on_subscribe(client, None, old_mid, [0])
        assert owner.read(t)["error_status"]
        client.on_subscribe(client, None, client.mid, [1])
        assert not owner.read(t)["error_status"]
        assert names(client).count("subscribe") == 2
    finally:
        owner.close()


@pytest.mark.parametrize("code", [1, 2, 3, 4, 5])
def test_connack_refusal_repeated_and_only_permanent_refusals_disconnect(factory, code):
    factory[1]["connack"] = code
    owner = mqtt.MQTTSubscription(target())
    client = factory[0][0]
    try:
        for _ in range(2):
            result = owner.read(target())
            assert result["error_status"] and f"code {code}" in result["error_message"]
        assert names(client).count("disconnect") == int(code != 3)
    finally:
        owner.close()


def test_failed_suback_repeated_and_message_buffer_still_drains(factory):
    owner = mqtt.MQTTSubscription(target())
    client = factory[0][0]
    try:
        client.on_subscribe(client, None, client.mid, [128])
        for _ in range(2):
            assert "refused subscription" in owner.read(target())["error_message"]
        client.emit(b"buffered")
        assert owner.read(target())["value"] == "buffered"
    finally:
        owner.close()


def test_callback_during_close_cannot_repopulate_and_rebind_is_new_owner(factory):
    owner = mqtt.MQTTSubscription(target())
    client = factory[0][0]
    callback = client.on_message
    entered, release = threading.Event(), threading.Event()
    client.close_gate = entered, release
    thread = threading.Thread(target=owner.close)
    thread.start()
    try:
        assert entered.wait(3)
        callback(
            client, None, SimpleNamespace(topic="t", payload=b"late", retain=False)
        )
        assert not owner._messages
    finally:
        release.set()
        thread.join(3)
    assert not thread.is_alive()
    callback(client, None, SimpleNamespace(topic="t", payload=b"later", retain=False))
    assert not owner._messages
    new_target = target(topic="new/topic")
    new_owner = mqtt.MQTTSubscription(new_target)
    try:
        factory[0][1].emit(b"new", topic="new/topic")
        assert new_owner.read(new_target)["value"] == "new"
        with pytest.raises(mqtt.MQTTError, match="closed"):
            owner.reset_buffer()
    finally:
        new_owner.close()


def test_close_serializes_after_publish_and_rejects_later_admission(factory):
    owner = mqtt.MQTTPublisher(target())
    entered, release, closing = threading.Event(), threading.Event(), threading.Event()
    factory[0][0].publish_gate = entered, release
    publish = threading.Thread(target=lambda: owner.publish(target(), "one"))

    def close():
        closing.set()
        owner.close()

    closer = threading.Thread(target=close)
    publish.start()
    try:
        assert entered.wait(3)
        closer.start()
        assert closing.wait(3)
        assert "disconnect" not in names(factory[0][0])
    finally:
        release.set()
        publish.join(3)
        closer.join(3)
    assert not publish.is_alive() and not closer.is_alive()
    with pytest.raises(mqtt.MQTTError, match="closed"):
        owner.publish(target(), "two")
    assert names(factory[0][0]).count("publish") == 1


@pytest.mark.parametrize(
    "change",
    [
        {"host": "other"},
        {"topic": "other"},
        {"qos": 1},
        {"timeout": 1},
        {"read_mode": "sequential"},
        {"encryption": "tls"},
        {"username": "u"},
        {"password": Secret("SENTINEL", name="p")},
    ],
)
def test_identity_change_rejected_without_reconfigure(factory, change):
    owner = mqtt.MQTTSubscription(target())
    try:
        with pytest.raises(mqtt.MQTTError, match="identity changed") as raised:
            owner.read(replace(target(), **change))
        assert "SENTINEL" not in str(raised.value)
        assert names(factory[0][0]).count("connect") == 1
    finally:
        owner.close()


def test_equivalent_secret_values_match_without_identity_leak(factory):
    owner = mqtt.MQTTSubscription(
        target(username="u", password=Secret("SENTINEL", name="a"))
    )
    try:
        result = owner.read(target(username="u", password=Secret("SENTINEL", name="b")))
        assert not result["error_status"]
        assert "SENTINEL" not in repr(owner._target)
    finally:
        owner.close()


def test_operator_policy_override_and_allowlist_are_applied_before_acquisition(
    factory, monkeypatch
):
    monkeypatch.setattr(
        mqtt.environment, "MQTT_WORKFLOWS_BLOCKS_ALLOW_USER_PROVIDED_HOST", False
    )
    monkeypatch.setattr(
        mqtt.environment, "MQTT_WORKFLOWS_BLOCKS_WHITELISTED_HOSTS", ["[::1]:8883"]
    )
    result = mqtt.MQTTWriterBlock().run("ignored", "t", "m")
    assert not result["error_status"]
    assert next(e for e in factory[0][0].events if e[0] == "connect")[1][0] == (
        "::1",
        8883,
    )
    monkeypatch.setattr(
        mqtt.environment, "MQTT_WORKFLOWS_BLOCKS_ALLOW_USER_PROVIDED_HOST", True
    )
    assert mqtt.MQTTWriterBlock().run("denied", "t", "m")["error_status"]
    assert len(factory[0]) == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"port": 0},
        {"qos": 3},
        {"timeout": float("inf")},
        {"timeout": 0},
        {"topic": "a/#/b"},
        {"client_id": "fixed", "qos": 0},
    ],
)
def test_invalid_host_owner_settings_do_not_acquire(factory, changes):
    with pytest.raises(mqtt.MQTTError):
        mqtt.MQTTSubscription(target(**changes))
    assert factory[0] == []


def test_missing_reader_owner_and_missing_optional_driver_are_actionable(monkeypatch):
    assert "host-owned" in mqtt.MQTTReaderBlock().run("b", "t")["error_message"]

    def absent(name):
        raise ImportError(name)

    monkeypatch.setattr(mqtt.importlib, "import_module", absent)
    result = mqtt.MQTTWriterBlock().run("b", "t", "m")
    assert "paho-mqtt~=1.6.1" in result["message"]


def test_reset_buffer_serializes_with_waiting_message_callback(factory):
    owner = mqtt.MQTTSubscription(target())
    client = factory[0][0]
    entered = threading.Event()

    def callback():
        entered.set()
        client.emit(b"after-reset")

    client.emit(b"before-reset")
    thread = threading.Thread(target=callback)
    try:
        with owner._condition:
            thread.start()
            assert entered.wait(3)
            owner.reset_buffer()
            assert not owner._messages
        thread.join(3)
        assert not thread.is_alive()
        assert owner.read(target())["value"] == "after-reset"
    finally:
        thread.join(3)
        owner.close()


@pytest.mark.parametrize("boundary", ["parser", "copy"])
def test_json_recursion_limit_preserves_raw_record_and_next_read(
    factory, boundary
):
    if boundary == "parser":
        raw = "[" * 10000 + "0" + "]" * 10000
        with pytest.raises(RecursionError):
            json.loads(raw)
    else:
        # JSON accepts this object; deepcopy needs more stack frames per level.
        depth = sys.getrecursionlimit() * 2 // 3
        raw = '{"nested":' + "[" * depth + "0" + "]" * depth + "}"
        parsed = json.loads(raw)
        with pytest.raises(RecursionError):
            copy.deepcopy(parsed)

    t = target(read_mode="sequential")
    owner = mqtt.MQTTSubscription(t)
    try:
        client = factory[0][0]
        client.emit(b"prior")
        assert owner.read(t)["value"] == "prior"
        client.emit(raw.encode(), topic="factory/deep")
        first = owner.read(t)
        assert first == {
            "value": raw,
            "payload": {},
            "topic": "factory/deep",
            "is_new": True,
            "error_status": False,
            "error_message": None,
        }
        assert len(owner._messages) == 0
        assert owner.read(t) == {**first, "is_new": False}
        client.emit(raw.encode(), topic="factory/deep", retained=True)
        assert owner.read(t) == {**first, "is_new": False}
        client.emit('{"next":"łódź/東京","null":null}'.encode())
        client.emit(b"last")
        next_row = owner.read(t)
        assert next_row["payload"] == {"next": "łódź/東京", "null": None}
        assert next_row["is_new"] and not next_row["error_status"]
        assert len(owner._messages) == 1
        assert owner.read(t)["value"] == "last"
    finally:
        owner.close()


@pytest.mark.parametrize(
    "raw, payload",
    [
        (
            '{"nested":[{"unicode":"łódź/東京","null":null}]}',
            {"nested": [{"unicode": "łódź/東京", "null": None}]},
        ),
        ("[1,[2]]", {}),
        ("null", {}),
        ('"hello"', {}),
        ("0", {}),
        ("", {}),
        ("{broken", {}),
    ],
)
def test_optional_json_view_keeps_shallow_semantics(factory, raw, payload):
    t = target()
    owner = mqtt.MQTTSubscription(t)
    try:
        factory[0][0].emit(raw.encode())
        result = owner.read(t)
        assert result["value"] == raw and result["payload"] == payload
        assert result["is_new"] and not result["error_status"]
        result["payload"].clear()
        assert owner.read(t)["payload"] == payload
    finally:
        owner.close()


@pytest.mark.parametrize("operation", ["loads", "deepcopy"])
def test_optional_json_view_does_not_hide_unexpected_errors(
    factory, monkeypatch, operation
):
    def fail(*args, **kwargs):
        raise RuntimeError("unexpected host failure")

    t = target()
    owner = mqtt.MQTTSubscription(t)
    try:
        factory[0][0].emit(b'{"ok":true}')
        module = mqtt.json if operation == "loads" else mqtt.copy
        monkeypatch.setattr(module, operation, fail)
        with pytest.raises(RuntimeError, match="unexpected host failure"):
            owner.read(t)
    finally:
        owner.close()

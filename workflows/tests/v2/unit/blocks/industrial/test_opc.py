"""Network-free author tests. Driver objects below are explicit stand-ins."""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import pytest
from roboflow_workflows.v2.blocks.industrial import opc
from roboflow_workflows.v2.blocks.sinks.runtime import BackgroundEffects
from roboflow_workflows.v2.data.secrets import Secret, SecretUnavailableError


class RecordingPort:
    def __init__(self, error=None):
        self.calls = []
        self.error = error
        self.closed = False

    def write(self, target, request):
        self.calls.append((target, request))
        if self.error:
            raise self.error

    def close(self):
        self.closed = True


def call(block, **overrides):
    values = dict(
        url="opc.tcp://user:pw@private.invalid:4840/endpoint",
        namespace="2",
        user_name="user",
        password="pw",
        object_name="Line/Station",
        variable_name="Count",
        value="42",
        value_type="Integer",
        fire_and_forget=False,
        cooldown_seconds=0,
    )
    values.update(overrides)

    return block.run(**values)


@pytest.fixture
def fake_sdk(monkeypatch):
    """Fake SDK API with real asyncio task and loop lifetime accounting."""
    state = SimpleNamespace(
        clients=[],
        loops=[],
        connections=0,
        writes=[],
        disconnects=0,
        socket_closes=0,
        tasks_cancelled=0,
        connect_errors=[],
        write_error=None,
        close_error=None,
        namespace_error=False,
        node_error=False,
        constructor_error=None,
    )

    class BadUserAccessDenied(Exception):
        pass

    class BadTooManySessions(Exception):
        pass

    class BadMaxConnectionsReached(Exception):
        pass

    class BadTypeMismatch(Exception):
        pass

    errors = SimpleNamespace(
        BadUserAccessDenied=BadUserAccessDenied,
        BadTooManySessions=BadTooManySessions,
        BadMaxConnectionsReached=BadMaxConnectionsReached,
        BadTypeMismatch=BadTypeMismatch,
    )

    class Node:
        async def write_value(self, value, variant):
            state.writes.append((value, variant))
            logging.getLogger("asyncua.common.node").warning(
                "secret endpoint %s password pw",
                "opc.tcp://user:pw@private.invalid:4840/endpoint",
            )
            if state.write_error:
                raise state.write_error

        async def read_data_type_as_variant_type(self):
            return SimpleNamespace(name="Int32")

    class Client:
        def __init__(self, url, timeout):
            if state.constructor_error:
                raise state.constructor_error
            self.url = url
            self.timeout = timeout
            self.paths = []
            self.namespaces = []
            self.user = self.password = None
            self.nodes = SimpleNamespace(root=self)
            state.clients.append(self)

        def set_user(self, user):
            self.user = user

        def set_password(self, password):
            self.password = password

        async def _monitor(self):
            try:
                await asyncio.Event().wait()
            finally:
                state.tasks_cancelled += 1

        async def connect(self):
            state.connections += 1
            asyncio.create_task(self._monitor())
            await asyncio.sleep(0)
            if state.connect_errors:
                raise state.connect_errors.pop(0)

        async def disconnect(self):
            state.disconnects += 1
            if state.close_error:
                raise state.close_error

        def disconnect_socket(self):
            state.socket_closes += 1

        async def get_namespace_index(self, uri):
            self.namespaces.append(uri)
            if state.namespace_error:
                raise ValueError("private URI secret")

            return 4

        async def get_namespace_array(self):
            return ["standard", "urn:factory"]

        def get_node(self, path):
            self.paths.append(path)
            if state.node_error:
                raise ValueError("private node secret")

            return Node()

        async def get_child(self, path):
            return self.get_node(path)

    types = SimpleNamespace(**{name: name for name in opc._VALUE_TYPES})
    sdk = SimpleNamespace(Client=Client, ua=SimpleNamespace(VariantType=types))
    original_import = opc.importlib.import_module

    def load(name, *args, **kwargs):
        if name == "asyncua":
            return sdk
        if name == "asyncua.ua.uaerrors":
            return errors

        return original_import(name, *args, **kwargs)

    # The SDK's node logger is created at module import, as in real asyncua.
    logging.getLogger("asyncua.common.node")
    monkeypatch.setattr(opc.importlib, "import_module", load)
    original_loop = asyncio.events.new_event_loop

    def loop():
        created = original_loop()
        state.loops.append(created)

        return created

    monkeypatch.setattr(asyncio.events, "new_event_loop", loop)
    state.errors = errors
    state.client_class = Client
    return state


@pytest.mark.parametrize(
    "value_type,value,expected,variant",
    [
        ("Boolean", " True ", True, "Boolean"),
        ("Boolean", "1", True, "Boolean"),
        ("Boolean", "false", False, "Boolean"),
        ("Boolean", "not-json", False, "Boolean"),
        ("Double", "1.25", 1.25, "Double"),
        ("Float", "-1.25", -1.25, "Float"),
        ("Integer", str(2**63 - 1), 2**63 - 1, "Int64"),
        ("Int64", -(2**63), -(2**63), "Int64"),
        ("Int32", 2**31 - 1, 2**31 - 1, "Int32"),
        ("Int16", -32768, -32768, "Int16"),
        ("SByte", 127, 127, "SByte"),
        ("UInt16", 65535, 65535, "UInt16"),
        ("UInt32", 2**32 - 1, 2**32 - 1, "UInt32"),
        ("UInt64", str(2**64 - 1), 2**64 - 1, "UInt64"),
        ("String", True, "True", "String"),
        ("String", '{"x": 1}', '{"x": 1}', "String"),
        ("Integer", 2.9, 2, "Int64"),
        ("Int16", True, 1, "Int16"),
    ],
)
def test_v1_scalar_conversion_and_integer_alias(value_type, value, expected, variant):
    port = RecordingPort()
    row = call(
        opc.OPCWriterSinkBlock(opc_client=port), value_type=value_type, value=value
    )
    assert row["error_status"] is False
    request = port.calls[0][1]
    assert (request.value, request.variant_type) == (expected, variant)
    assert type(request.value) is type(expected)


@pytest.mark.parametrize("variant", list(opc._INTEGER_RANGES))
def test_each_integer_range_accepts_both_edges_and_rejects_beyond_them(variant):
    minimum, maximum = opc._INTEGER_RANGES[variant]
    port = RecordingPort()
    block = opc.OPCWriterSinkBlock(opc_client=port)
    for value in [minimum, maximum]:
        assert call(block, value_type=variant, value=value)["error_status"] is False
    for value in [minimum - 1, maximum + 1]:
        row = call(block, value_type=variant, value=value)
        assert row["error_status"] is True
        assert "out of range" in row["message"]
    assert len(port.calls) == 2


@pytest.mark.parametrize(
    "value_type,value",
    [
        ("Float", 3.5e38),
        ("Double", float("nan")),
        ("Double", "inf"),
        ("Int64", "12.5"),
        ("Byte", 12),
        ("String", [12]),
        ("String", {"x": 12}),
    ],
)
def test_invalid_conversion_never_reaches_port(value_type, value):
    port = RecordingPort()
    row = call(
        opc.OPCWriterSinkBlock(opc_client=port), value_type=value_type, value=value
    )
    assert row["error_status"]
    assert port.calls == []


@pytest.mark.parametrize("policy,parameter", [(True, False), (False, True)])
def test_disable_precedes_secret_reveal_and_sdk_import(monkeypatch, policy, parameter):
    def forbidden(*args, **kwargs):
        raise AssertionError("SDK touched")

    monkeypatch.setattr(opc.importlib, "import_module", forbidden)
    row = call(
        opc.OPCWriterSinkBlock(disable_sinks=policy),
        disable_sink=parameter,
        url=Secret.unavailable("endpoint"),
        password=Secret.unavailable("password"),
    )
    assert row["disabled"] and not row["error_status"]
    assert "policy" in row["message"] if policy else "disable_sink" in row["message"]


def test_unavailable_secret_fails_before_any_port_use():
    port = RecordingPort()
    with pytest.raises(SecretUnavailableError):
        call(
            opc.OPCWriterSinkBlock(opc_client=port),
            password=Secret.unavailable("password"),
        )
    assert port.calls == []


def test_missing_driver_is_lazy_actionable_and_not_retried(monkeypatch):
    names = []

    def missing(name):
        names.append(name)
        raise ModuleNotFoundError("missing")

    monkeypatch.setattr(opc.importlib, "import_module", missing)
    block = opc.OPCWriterSinkBlock()
    assert names == []
    row = call(block)
    assert row["error_status"]
    assert "asyncua~=1.1.5" in row["message"] and "opc_client" in row["message"]
    assert names == ["asyncua"]


@pytest.mark.parametrize(
    "namespace,mode,variable,path",
    [
        ("2", "direct", "Count", "ns=2;s=Line/Station/Count"),
        ("2", "direct", "", "ns=2;s=Line/Station"),
        (
            "urn:factory",
            "hierarchical",
            "Count",
            ["0:Objects", "4:Line", "4:Station", "4:Count"],
        ),
    ],
)
def test_default_lookup_paths_and_success_cleanup(
    fake_sdk, namespace, mode, variable, path
):
    row = call(
        opc.OPCWriterSinkBlock(),
        namespace=namespace,
        node_lookup_mode=mode,
        variable_name=variable,
    )
    assert row["message"] == "Value set successfully"
    assert fake_sdk.clients[0].paths == [path]
    assert fake_sdk.clients[0].namespaces == ([] if namespace == "2" else [namespace])
    assert fake_sdk.writes == [(42, "Int64")]
    assert fake_sdk.disconnects == fake_sdk.tasks_cancelled == 1
    assert all(loop.is_closed() for loop in fake_sdk.loops)


def test_connection_retries_close_each_partial_client_and_backoff_is_exponential(
    fake_sdk, monkeypatch
):
    fake_sdk.connect_errors = [OSError("pw"), OSError("private endpoint")]
    delays = []
    original_sleep = asyncio.sleep

    async def sleep(delay):
        if delay:
            delays.append(delay)
        await original_sleep(0)

    monkeypatch.setattr(opc.asyncio, "sleep", sleep)
    row = call(opc.OPCWriterSinkBlock(), max_retries=3, retry_backoff_seconds=0.015)
    assert not row["error_status"]
    assert delays == [0.015, 0.030]
    assert fake_sdk.connections == fake_sdk.disconnects == fake_sdk.tasks_cancelled == 3
    assert len(fake_sdk.writes) == 1
    assert all(loop.is_closed() for loop in fake_sdk.loops)


def test_exhausted_connect_attempts_close_and_do_not_write(fake_sdk):
    fake_sdk.connect_errors = [OSError("pw endpoint")] * 3
    row = call(opc.OPCWriterSinkBlock(), retry_backoff_seconds=0)
    assert row["error_status"] and "3 attempt" in row["message"]
    assert fake_sdk.connections == fake_sdk.disconnects == fake_sdk.tasks_cancelled == 3
    assert fake_sdk.writes == []
    assert all(loop.is_closed() for loop in fake_sdk.loops)


@pytest.mark.parametrize(
    "error_name",
    ["BadUserAccessDenied", "BadTooManySessions", "BadMaxConnectionsReached"],
)
def test_auth_and_capacity_errors_fail_without_connection_retry(fake_sdk, error_name):
    fake_sdk.connect_errors = [
        getattr(fake_sdk.errors, error_name)("private endpoint pw")
    ]
    row = call(opc.OPCWriterSinkBlock())
    assert row["error_status"]
    assert fake_sdk.connections == fake_sdk.disconnects == 1
    assert fake_sdk.writes == []


def test_write_ack_failure_is_not_replayed_and_original_error_survives_close_failure(
    fake_sdk,
):
    fake_sdk.write_error = TimeoutError("pw private endpoint")
    fake_sdk.close_error = RuntimeError("CLOSE SECRET")
    row = call(opc.OPCWriterSinkBlock(), max_retries=5)
    assert row["error_status"]
    assert (
        "write failed" in row["message"] and "outcome may be unknown" in row["message"]
    )
    assert "cleanup failed" in row["message"]
    assert len(fake_sdk.writes) == fake_sdk.connections == fake_sdk.disconnects == 1
    assert fake_sdk.socket_closes == fake_sdk.tasks_cancelled == 1
    assert all(loop.is_closed() for loop in fake_sdk.loops)
    assert "CLOSE SECRET" not in row["message"]


def test_close_failure_after_success_is_not_reported_as_success(fake_sdk):
    fake_sdk.close_error = OSError("pw")
    row = call(opc.OPCWriterSinkBlock())
    assert row["error_status"] and "cleanup failed" in row["message"]
    assert len(fake_sdk.writes) == 1
    assert all(loop.is_closed() for loop in fake_sdk.loops)


def test_partial_constructor_failure_closes_owned_event_loop(fake_sdk):
    fake_sdk.constructor_error = RuntimeError("private endpoint")
    row = call(opc.OPCWriterSinkBlock(), retry_backoff_seconds=0)
    assert row["error_status"]
    assert fake_sdk.clients == []
    assert all(loop.is_closed() for loop in fake_sdk.loops)


@pytest.mark.parametrize(
    "phase,message",
    [("namespace_error", "namespace lookup"), ("node_error", "node lookup")],
)
def test_lookup_failures_are_actionable_and_always_close(fake_sdk, phase, message):
    setattr(fake_sdk, phase, True)
    row = call(opc.OPCWriterSinkBlock(), namespace="urn:factory")
    assert row["error_status"] and message in row["message"]
    assert fake_sdk.disconnects == 1 and fake_sdk.writes == []


def test_type_mismatch_reports_concrete_server_type_and_never_retries(fake_sdk):
    fake_sdk.write_error = fake_sdk.errors.BadTypeMismatch("private endpoint")
    row = call(opc.OPCWriterSinkBlock())
    assert row["error_status"] and "node expects Int32" in row["message"]
    assert len(fake_sdk.writes) == fake_sdk.disconnects == 1


def test_driver_logs_and_failure_messages_do_not_expose_credentials(fake_sdk, caplog):
    caplog.set_level(logging.DEBUG)
    fake_sdk.write_error = OSError("opc.tcp://user:pw@private.invalid:4840/endpoint pw")
    row = call(opc.OPCWriterSinkBlock())
    assert "private.invalid" not in row["message"] + caplog.text
    assert "password pw" not in caplog.text
    assert logging.getLogger("asyncua.common.node").filters == []


def test_borrowed_error_is_scrubbed_and_client_is_never_closed():
    port = RecordingPort(
        opc.OPCSinkError(
            "endpoint opc.tcp://user:pw@private.invalid:4840/endpoint password pw user user"
        )
    )
    row = call(opc.OPCWriterSinkBlock(opc_client=port))
    assert row["error_status"]
    assert (
        "private.invalid" not in row["message"] and "password pw" not in row["message"]
    )
    assert not port.closed and len(port.calls) == 1


def test_cooldown_is_monotonic_per_instance_and_admission_is_atomic():
    port = RecordingPort()
    block = opc.OPCWriterSinkBlock(opc_client=port, clock=lambda: 10)
    other = opc.OPCWriterSinkBlock(opc_client=port, clock=lambda: 10)
    with ThreadPoolExecutor(max_workers=8) as workers:
        rows = list(workers.map(lambda _: call(block, cooldown_seconds=5), range(16)))
    assert sum(row["throttling_status"] for row in rows) == 15
    assert len(port.calls) == 1
    assert not call(other, cooldown_seconds=5)["throttling_status"]
    assert len(port.calls) == 2
    # A throttled call need not reveal replayed credentials.
    assert call(block, cooldown_seconds=5, password=Secret.unavailable("PW"))[
        "throttling_status"
    ]


def test_connection_circuit_backoff_expires_and_is_independent_per_block(fake_sdk):
    now = [0.0]
    block = opc.OPCWriterSinkBlock(clock=lambda: now[0])
    fake_sdk.connect_errors = [OSError("secret")] * 3
    assert call(block, retry_backoff_seconds=0)["error_status"]
    assert "backoff applies" in call(block)["message"]
    assert fake_sdk.connections == 3
    assert not call(opc.OPCWriterSinkBlock())["error_status"]
    now[0] = 2.0
    assert not call(block)["error_status"]


def test_full_dispatch_is_a_failure_and_does_not_consume_cooldown():
    gate, entered = Event(), Event()

    def blocker():
        entered.set()
        assert gate.wait(5)
        return False, "done"

    port = RecordingPort()
    with BackgroundEffects(max_pending=1, when_full="reject") as effects:
        effects.deliver(blocker, label="test blocker")
        assert entered.wait(5)
        block = opc.OPCWriterSinkBlock(
            opc_client=port, sink_effects=effects, clock=lambda: 0
        )
        try:
            row = call(block, fire_and_forget=True, cooldown_seconds=5)
            assert row["error_status"] and not row["throttling_status"]
            assert port.calls == []
        finally:
            gate.set()
        assert effects.drain(timeout=5)
        assert not call(block, fire_and_forget=True, cooldown_seconds=5)["error_status"]
    assert len(port.calls) == 1


def test_closed_dispatch_is_a_failure_and_acquires_nothing(fake_sdk):
    effects = BackgroundEffects()
    effects.close()
    row = call(opc.OPCWriterSinkBlock(sink_effects=effects), fire_and_forget=True)
    assert row["error_status"] and "dispatch failed" in row["message"]
    assert fake_sdk.clients == []


def test_queued_default_acquires_only_when_run_and_releases_afterward(fake_sdk):
    gate, entered = Event(), Event()

    def blocker():
        entered.set()
        assert gate.wait(5)
        return False, "done"

    with BackgroundEffects(max_pending=2) as effects:
        effects.deliver(blocker, label="test blocker")
        assert entered.wait(5)
        try:
            row = call(
                opc.OPCWriterSinkBlock(sink_effects=effects), fire_and_forget=True
            )
            assert not row["error_status"] and "background task" in row["message"]
            assert fake_sdk.clients == [] and fake_sdk.loops == []
        finally:
            gate.set()
    assert fake_sdk.disconnects == fake_sdk.tasks_cancelled == 1
    assert all(loop.is_closed() for loop in fake_sdk.loops)


def test_fire_and_forget_without_effects_runs_synchronously(fake_sdk):
    row = call(opc.OPCWriterSinkBlock(), fire_and_forget=True)
    assert row["message"] == "Value set successfully"
    assert fake_sdk.disconnects == 1


def test_connect_failure_with_close_failure_preserves_connect_phase_and_stops_retries(
    fake_sdk,
):
    fake_sdk.connect_errors = [OSError("private endpoint pw")] * 3
    fake_sdk.close_error = RuntimeError("close pw")
    row = call(opc.OPCWriterSinkBlock(), retry_backoff_seconds=0)
    assert row["error_status"]
    assert "connection failed after 1 attempt" in row["message"]
    assert "cleanup failed" in row["message"]
    assert fake_sdk.connections == fake_sdk.disconnects == fake_sdk.socket_closes == 1
    assert fake_sdk.writes == []
    assert fake_sdk.tasks_cancelled == 1
    assert all(loop.is_closed() for loop in fake_sdk.loops)


def test_default_adapter_failure_does_not_export_a_raw_driver_exception_chain(fake_sdk):
    fake_sdk.write_error = OSError("private endpoint raw password")
    target = opc.OPCTarget(
        Secret("opc.tcp://private.invalid", name="url"), None, None, 2, 3, 0
    )
    request = opc.OPCWrite("2", "Line", "Count", 42, "Int64", "direct")
    with pytest.raises(opc.OPCSinkError) as raised:
        opc.AsyncuaOPCClient().write(target, request)
    assert raised.value.__context__ is None
    assert raised.value.__cause__ is None
    assert "raw password" not in str(raised.value)


@pytest.mark.parametrize(
    "setting,expected", [("8", 8000), ("nan", 60000), ("-1", 60000), ("invalid", 60000)]
)
def test_session_timeout_environment_setting_preserves_positive_finite_v1_policy(
    fake_sdk, monkeypatch, setting, expected
):
    monkeypatch.setenv("OPC_SESSION_TIMEOUT_SECONDS", setting)
    assert not call(opc.OPCWriterSinkBlock())["error_status"]
    assert fake_sdk.clients[0].session_timeout == expected


def test_session_capacity_backoff_uses_environment_base_plus_bounded_jitter(
    fake_sdk, monkeypatch
):
    monkeypatch.setenv("OPC_SESSION_EXHAUSTION_BACKOFF_SECONDS", "45")
    monkeypatch.setattr(opc.random, "uniform", lambda low, high: 10)
    fake_sdk.connect_errors = [fake_sdk.errors.BadTooManySessions("private endpoint")]
    now = [0.0]
    block = opc.OPCWriterSinkBlock(clock=lambda: now[0])
    assert call(block)["error_status"]
    now[0] = 54.9
    assert "backoff applies" in call(block)["message"]
    now[0] = 55.0
    assert not call(block)["error_status"]
    assert fake_sdk.connections == 2


def test_cooldown_reopens_at_exact_monotonic_boundary():
    now = [10.0]
    port = RecordingPort()
    block = opc.OPCWriterSinkBlock(opc_client=port, clock=lambda: now[0])
    assert not call(block, cooldown_seconds=5)["throttling_status"]
    now[0] = 14.9
    assert call(block, cooldown_seconds=5)["throttling_status"]
    now[0] = 15.0
    assert not call(block, cooldown_seconds=5)["throttling_status"]
    assert len(port.calls) == 2


def test_cancelled_queued_effect_never_acquires_driver_resources(fake_sdk):
    gate, entered = Event(), Event()

    def blocker():
        entered.set()
        assert gate.wait(5)
        return False, "done"

    effects = BackgroundEffects(max_pending=2)
    effects.deliver(blocker, label="test blocker")
    assert entered.wait(5)
    try:
        row = call(opc.OPCWriterSinkBlock(sink_effects=effects), fire_and_forget=True)
        assert row["message"] == opc.BACKGROUND_WRITE_MESSAGE
        effects.close(timeout=0)
        assert effects.stats().abandoned == 1
        assert effects.stats().completed == 0
        assert fake_sdk.clients == [] and fake_sdk.loops == []
    finally:
        gate.set()
        assert effects.drain(timeout=5)
        effects.close()


def test_running_async_host_loop_fails_actionably_without_driver_acquisition(fake_sdk):
    async def host():
        result = call(opc.OPCWriterSinkBlock())
        assert result["error_status"]
        assert "synchronous worker thread" in result["message"]
        assert "opc_client" in result["message"]
        assert fake_sdk.clients == []

    asyncio.run(host())
    assert len(fake_sdk.loops) == 1
    assert fake_sdk.loops[0].is_closed()


@pytest.mark.parametrize("phase", ["connect", "write", "disconnect"])
def test_cancellation_unwinds_default_lifetime_without_replaying_writes(
    fake_sdk, monkeypatch, phase
):
    if phase == "connect":
        original_connect = fake_sdk.client_class.connect

        async def cancelled_connect(client):
            await original_connect(client)
            raise asyncio.CancelledError("private endpoint pw")

        monkeypatch.setattr(fake_sdk.client_class, "connect", cancelled_connect)
    elif phase == "write":
        fake_sdk.write_error = asyncio.CancelledError("private endpoint pw")
    else:
        fake_sdk.close_error = asyncio.CancelledError("private endpoint pw")

    result = call(opc.OPCWriterSinkBlock())
    assert result["error_status"]
    assert fake_sdk.connections == fake_sdk.disconnects == 1
    assert len(fake_sdk.writes) == (0 if phase == "connect" else 1)
    assert fake_sdk.tasks_cancelled == 1
    assert all(loop.is_closed() for loop in fake_sdk.loops)
    assert "private endpoint" not in result["message"]

"""Admission decisions, concurrent leases and the pre-network host boundary."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from roboflow_workflows.core_steps.integrations.databricks import admission
from roboflow_workflows.core_steps.integrations.databricks.sql_image_source import (
    isolation,
    v1,
)

HOST = "dbc-test.cloud.databricks.com"
PATH = "/sql/1.0/warehouses/abc"


def controller(**limits):
    """Build a process-local admission controller with selected quota overrides.

    Args:
        **limits (int): Operator limits replacing AdmissionLimits defaults.

    Returns:
        DatabricksAdmission: Isolated controller for this test.
    """
    control = admission.DatabricksAdmission(
        admission.LocalAdmissionBackend(), limits=admission.AdmissionLimits(**limits)
    )

    return control


def enter(control, *, tenant="tenant", hostname=HOST, path=PATH):
    """Prepare an admission context using the fixture tenant and destination.

    Args:
        control (DatabricksAdmission): Controller whose quotas are under test.
        tenant (str): Trusted tenant identity for the simulated invocation.
        hostname (str): Validated destination hostname.
        path (str): SQL warehouse endpoint path.

    Returns:
        AbstractContextManager: Admission context with a ten-second execution deadline.
    """
    context = control.admit(
        tenant=tenant, hostname=hostname, http_path=path, deadline=time.monotonic() + 10
    )

    return context


@pytest.mark.parametrize("scope", ["tenant", "workspace", "warehouse", "fleet"])
def test_each_concurrency_limit_and_release(scope):
    """Verify each concurrency limit and release.

    Args:
        scope (str): Tenant, workspace, warehouse or fleet quota selected by the case.
    """
    control = controller(**{scope + "_concurrency": 1})
    with enter(control):
        with pytest.raises(admission.AdmissionRejected):
            with enter(control):
                pytest.fail("admitted twice")
    with enter(control):
        pass


def test_destination_limits_cross_tenants_and_endpoint_aliases():
    """Verify destination limits cross tenants and endpoint aliases."""
    control = controller(warehouse_concurrency=1)
    with enter(control, tenant="alice"):
        with pytest.raises(admission.AdmissionRejected):
            with enter(control, tenant="bob", path="/sql/1.0/endpoints/abc"):
                pytest.fail("alias bypassed warehouse quota")
        with enter(control, tenant="bob", path="/sql/1.0/warehouses/other"):
            pass
    control = controller(workspace_concurrency=1)
    with enter(control, tenant="alice"):
        with pytest.raises(admission.AdmissionRejected):
            with enter(control, tenant="bob", path="/sql/1.0/warehouses/other"):
                pytest.fail("warehouse rotation bypassed workspace quota")


@pytest.mark.parametrize("scope", ["tenant", "workspace", "warehouse", "fleet"])
def test_rates_are_not_refunded_after_success_or_failure(scope):
    """Verify rates are not refunded after success or failure.

    Args:
        scope (str): Tenant, workspace, warehouse or fleet quota selected by the case.
    """
    control = controller(**{scope + "_rate": 1})
    with pytest.raises(ValueError):
        with enter(control):
            raise ValueError("worker failed")
    with pytest.raises(admission.AdmissionRejected):
        with enter(control):
            pytest.fail("released rate budget")


def test_local_lease_and_rate_expiration(monkeypatch):
    """Verify local lease and rate expiration.

    Args:
        monkeypatch (pytest.MonkeyPatch): Scoped replacements restored after the test.
    """
    control = controller(tenant_concurrency=1, tenant_rate=1)
    clock = Mock(return_value=100.0)
    monkeypatch.setattr(admission.time, "monotonic", clock)
    keys = admission._keys(tenant="tenant", hostname=HOST, http_path=PATH)
    assert control.backend.acquire(
        keys=keys, token="orphan", lease_seconds=70, limits=control.limits.ordered()
    )
    clock.return_value = 171.0
    with enter(control):
        pass


def test_backend_errors_fail_closed_without_revealing_details():
    """Verify backend errors fail closed without revealing details."""
    backend = Mock(distributed=True)
    backend.acquire.side_effect = RuntimeError("redis://private-credential@host")
    control = admission.DatabricksAdmission(backend, limits=admission.AdmissionLimits())
    with pytest.raises(admission.AdmissionRejected, match="unavailable") as error:
        with enter(control):
            pytest.fail("backend failure admitted")
    assert "private-credential" not in str(error.value)


def _compete(controls):
    start = threading.Barrier(11)
    attempted = threading.Event()
    release = threading.Event()
    lock = threading.Lock()
    results = []

    def _record(value):
        with lock:
            results.append(value)
            if len(results) == 10:
                attempted.set()

    def _run(index):
        start.wait(timeout=5)
        try:
            with enter(controls[index % len(controls)]):
                _record(True)
                assert release.wait(timeout=5)
        except admission.AdmissionRejected:
            _record(False)

    with ThreadPoolExecutor(10) as pool:
        futures = [pool.submit(_run, index) for index in range(10)]
        start.wait(timeout=5)
        try:
            assert attempted.wait(timeout=5)
            assert sum(results) == 2
        finally:
            release.set()
        for future in futures:
            future.result()


def test_local_admission_is_atomic_across_threads():
    """Verify local admission is atomic across threads."""
    _compete([controller()])


def test_redis_lua_is_atomic_across_independent_clients_and_releases_only_active_leases():
    """Verify redis lua is atomic across independent clients and releases only active leases."""
    fakeredis = pytest.importorskip(
        "fakeredis", reason="install fakeredis[lua] for offline Lua integration tests"
    )
    server = fakeredis.FakeServer()
    clients = [fakeredis.FakeRedis(server=server) for _ in range(2)]
    controls = [
        admission.DatabricksAdmission(
            admission.RedisAdmissionBackend(client), limits=admission.AdmissionLimits()
        )
        for client in clients
    ]
    _compete(controls)
    keys = admission._keys(tenant="tenant", hostname=HOST, http_path=PATH)
    assert all(clients[0].zcard(key) == 0 for key in keys[:4])
    assert all(clients[0].zcard(key) == 2 for key in keys[4:])
    assert all(clients[0].pttl(key) > 0 for key in keys[4:])
    # Simulate an expired orphan without waiting for a real request timeout.
    clients[0].zadd(keys[1], {"orphan": 1})
    with enter(controls[1]):
        assert clients[0].zscore(keys[1], "orphan") is None
    server.connected = False
    with pytest.raises(admission.AdmissionRejected, match="unavailable"):
        with enter(controls[0]):
            pytest.fail("Redis outage allowed local fallback")


@pytest.fixture
def request_inputs():
    """Supply valid image-reader inputs for host-boundary tests.

    Returns:
        dict: Block inputs containing a synthetic warehouse token.
    """
    inputs = dict(
        server_hostname=HOST,
        http_path=PATH,
        access_token="databricks-token",
        catalog="main",
        schema_name="vision",
        table="images",
        image_column="bytes",
        image_name_column=None,
        metadata_columns=[],
        max_rows=1,
        timeout_seconds=5,
        skip_undecodable=True,
    )

    return inputs


@pytest.mark.parametrize(
    "failure", ["local_backend", "missing_key", "unknown_workspace", "quota"]
)
def test_hosted_rejection_precedes_dns_and_process_launch(
    monkeypatch, request_inputs, failure
):
    """Verify hosted rejection precedes dns and process launch.

    Args:
        monkeypatch (pytest.MonkeyPatch): Scoped replacements restored after the test.
        request_inputs (dict): Valid image-reader inputs without caller-controlled tenant identity.
        failure (str): Missing identity, backend or capacity condition to exercise.
    """
    monkeypatch.setattr(v1, "ensure_block_enabled", lambda: None)
    monkeypatch.setattr(v1, "running_on_hosted_platform", lambda: True)
    lookup = Mock(side_effect=AssertionError("destination DNS"))
    launch = Mock(side_effect=AssertionError("process launch"))
    monkeypatch.setattr(v1, "resolve_and_validate_ips", lookup)
    monkeypatch.setattr(isolation, "run_isolated", launch)
    backend = Mock(distributed=failure != "local_backend")
    backend.acquire.return_value = False
    platform = Mock()
    platform.get_roboflow_workspace.return_value = (
        None if failure == "unknown_workspace" else "authenticated-tenant"
    )
    block = v1.DatabricksSQLImageSourceBlockV1(
        api_key=None if failure == "missing_key" else "roboflow-key",
        platform_client=platform,
        databricks_admission=admission.DatabricksAdmission(
            backend, limits=admission.AdmissionLimits()
        ),
    )
    with pytest.raises(admission.AdmissionRejected):
        block.run(**request_inputs)
    lookup.assert_not_called()
    launch.assert_not_called()


def test_trusted_workspace_and_normalized_destination_are_used_and_lease_covers_worker(
    monkeypatch, request_inputs
):
    """Verify trusted workspace and normalized destination are used and lease covers worker.

    Args:
        monkeypatch (pytest.MonkeyPatch): Scoped replacements restored after the test.
        request_inputs (dict): Valid image-reader inputs without caller-controlled tenant identity.
    """
    monkeypatch.setattr(v1, "ensure_block_enabled", lambda: None)
    monkeypatch.setattr(v1, "running_on_hosted_platform", lambda: True)
    backend = Mock(distributed=True)
    backend.acquire.return_value = True
    platform = Mock()
    platform.get_roboflow_workspace.return_value = "authenticated-tenant"

    def _run(*args, **kwargs):
        backend.release.assert_not_called()

        return []

    monkeypatch.setattr(isolation, "run_isolated", _run)
    block = v1.DatabricksSQLImageSourceBlockV1(
        api_key="trusted-host-key",
        platform_client=platform,
        databricks_admission=admission.DatabricksAdmission(
            backend, limits=admission.AdmissionLimits()
        ),
    )
    block.run(**dict(request_inputs, server_hostname=HOST.upper() + "."))
    platform.get_roboflow_workspace.assert_called_once_with("trusted-host-key")
    assert backend.acquire.call_args.kwargs["keys"] == admission._keys(
        tenant="authenticated-tenant", hostname=HOST, http_path=PATH
    )
    backend.release.assert_called_once()


def test_host_cache_binding_reuses_redis_and_local_fallback_cannot_enable_hosted():
    """Verify host cache binding reuses redis and local fallback cannot enable hosted."""
    client = Mock()
    shared = admission.admission_from_cache(SimpleNamespace(client=client))
    assert shared.backend._client is client
    local = admission.admission_from_cache(SimpleNamespace())
    with pytest.raises(admission.AdmissionRejected):
        local.require_shared_backend(hosted=True)


def test_real_redis_ceiling_across_processes():
    """Verify real redis ceiling across processes."""
    import json
    import os
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    binary = os.environ.get("DATABRICKS_ADMISSION_TEST_REDIS_BINARY")
    if not binary:
        pytest.skip(
            "set DATABRICKS_ADMISSION_TEST_REDIS_BINARY for real Redis integration"
        )
    import redis

    with tempfile.TemporaryDirectory(prefix="db-redis-", dir="/tmp") as temporary:
        socket_path = str(Path(temporary) / "redis.sock")
        server = subprocess.Popen(
            [
                binary,
                "--port",
                "0",
                "--unixsocket",
                socket_path,
                "--unixsocketperm",
                "700",
                "--save",
                "",
                "--appendonly",
                "no",
                "--dir",
                temporary,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        clients = []
        try:
            client = redis.Redis(unix_socket_path=socket_path, socket_timeout=1)
            until = time.monotonic() + 5
            while True:
                try:
                    client.ping()
                    break
                except redis.ConnectionError:
                    if time.monotonic() >= until:
                        raise

                    time.sleep(0.01)
            keys = admission._keys(tenant="tenant", hostname=HOST, http_path=PATH)
            script = """
import json, sys, redis
from roboflow_workflows.core_steps.integrations.databricks.admission import RedisAdmissionBackend, AdmissionLimits
client = redis.Redis(unix_socket_path=sys.argv[1], socket_timeout=2)
accepted = RedisAdmissionBackend(client).acquire(keys=tuple(json.loads(sys.argv[2])), token=sys.argv[3], lease_seconds=30, limits=AdmissionLimits().ordered())
print(int(accepted))
"""
            for index in range(8):
                clients.append(
                    subprocess.Popen(
                        [
                            sys.executable,
                            "-c",
                            script,
                            socket_path,
                            json.dumps(keys),
                            str(index),
                        ],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                )
            results = []
            for process in clients:
                stdout, stderr = process.communicate(timeout=15)
                assert process.returncode == 0, stderr
                results.append(int(stdout.strip()))
            assert sum(results) == 2
            assert client.zcard(keys[1]) == 2
            assert client.zcard(keys[5]) == 2
            # Failed admissions must not consume another tenant's budget.
            other = admission._keys(tenant="other", hostname=HOST, http_path=PATH)
            backend = admission.RedisAdmissionBackend(client)
            restrictive = admission.AdmissionLimits(workspace_concurrency=2)
            assert not backend.acquire(
                keys=other,
                token="blocked",
                lease_seconds=30,
                limits=restrictive.ordered(),
            )
            assert client.zcard(other[1]) == client.zcard(other[5]) == 0
            for token, accepted in enumerate(results):
                if accepted:
                    backend.release(keys=keys, token=str(token))
            assert client.zcard(keys[1]) == 0
            assert client.zcard(keys[5]) == 2
            rates = admission.AdmissionLimits(tenant_rate=2)
            assert not backend.acquire(
                keys=keys,
                token="rate-blocked",
                lease_seconds=30,
                limits=rates.ordered(),
            )
            # Expired orphan records are removed atomically on the next attempt.
            for key in keys:
                for token in client.zrange(key, 0, -1):
                    client.zadd(key, {token: 1})
            assert backend.acquire(
                keys=keys,
                token="after-expiry",
                lease_seconds=30,
                limits=rates.ordered(),
            )
            server.terminate()
            server.wait(timeout=5)
            control = admission.DatabricksAdmission(backend, limits=rates)
            with pytest.raises(admission.AdmissionRejected, match="unavailable"):
                with enter(control):
                    pytest.fail("Redis outage admitted work")
        finally:
            for process in clients + [server]:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)

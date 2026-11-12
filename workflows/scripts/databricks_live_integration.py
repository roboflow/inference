"""Live integration suite for the Databricks SQL Image Source block.

Runs the block against a real Databricks SQL warehouse, either in-process
(``--target local``) or through an inference server / hosted async-serverless
producer URL (``--target server --base-url ...``, which submits each case to
``POST /workflows/run`` with the Workflow specification inline), and verifies
failure and recovery behaviour, not only the happy path. Not part of the unit
suite: it needs a warehouse and credentials, supplied through the environment
(never on the command line):

    DATABRICKS_SERVER_HOSTNAME   workspace host (e.g. dbc-xxxx.cloud.databricks.com)
    DATABRICKS_HTTP_PATH         /sql/1.0/warehouses/<id>
    DATABRICKS_TOKEN             read-only identity used by the block under test
    DATABRICKS_SETUP_TOKEN       separate identity allowed to create the fixture table
    DATABRICKS_TABLE             catalog.schema.table for the fixtures
    ROBOFLOW_API_KEY             only for --target server

Cases (see the PR discussion for the acceptance table):

    import        20 small JPEGs + metadata, fetched across several batches
    bad-image     malformed row is skipped, run succeeds, no restart needed
    big-metadata  oversized metadata row is skipped (documented behaviour)
    bad-token     invalid credential fails without exposing the token
    budget        only with --expect-budget-failure: a deployment configured
                  with a reduced DATABRICKS_SQL_IMAGE_SOURCE_MAX_RUN_BYTES fails
                  the run with the budget error instead of retaining the rows
    disabled      only with --expect-disabled: the target rejects the Workflow
                  at compile time and no query reaches the warehouse
    slow          a query held busy past ``--slow-deadline`` is cancelled: the
                  job fails within deadline + cleanup grace, the warehouse's
                  query history shows the statement CANCELED, nothing RUNNING
    recovery      three slow failures in a row, then a successful import; the
                  history shows no accumulated RUNNING queries
    concurrent    an unrelated trivial Workflow keeps answering while the slow
                  query runs (server target only); p95 latency under a bound

Exit code 0 means every selected case passed. Nothing secret is printed.
Example:

    python scripts/databricks_live_integration.py --target local --setup
    python scripts/databricks_live_integration.py --target server \\
        --base-url https://serverless.roboflow.com --cases import,slow,recovery
"""

import argparse
import base64
import io
import json
import os
import statistics
import sys
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

BLOCK_TYPE = "roboflow_core/databricks_sql_image_source@v1"
FIXTURE_ROWS = 20
CLEANUP_GRACE_SECONDS = 35  # SOCKET_TIMEOUT_SECONDS + RETRY_DELAY_MAX_SECONDS
RUN_ID = uuid.uuid4().hex[:12]


def env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        print(f"missing environment variable {name}")
        sys.exit(2)
    return value


# ----------------------------------------------------------------- fixtures


def tiny_jpeg(seed: int) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (48, 32), ((seed * 37) % 255, (seed * 59) % 255, 90)).save(
        buffer, format="JPEG"
    )
    return buffer.getvalue()


def sql_connect(token: str):
    from databricks import sql

    return sql.connect(
        server_hostname=env("DATABRICKS_SERVER_HOSTNAME"),
        http_path=env("DATABRICKS_HTTP_PATH"),
        access_token=token,
        use_cloud_fetch=False,
        enable_telemetry=False,
    )


def setup_fixtures(table: str) -> None:
    """Create the fixture table with the setup identity (idempotent)."""
    rows: List[Tuple[int, str, bytes, str]] = [
        (i, "ok", tiny_jpeg(i), f"note-{i}") for i in range(FIXTURE_ROWS)
    ]
    rows.append((900, "bad", b"this is not an image", "malformed"))
    rows.append((901, "bigmeta", tiny_jpeg(901), "m" * 70_000))
    with sql_connect(env("DATABRICKS_SETUP_TOKEN")) as connection:
        with connection.cursor() as cursor:
            cursor.execute(f"DROP TABLE IF EXISTS {table}")
            cursor.execute(
                f"CREATE TABLE {table} (id BIGINT, kind STRING, bytes BINARY, note STRING)"
            )
            for row_id, kind, payload, note in rows:
                encoded = base64.b64encode(payload).decode()
                cursor.execute(
                    f"INSERT INTO {table} VALUES ({row_id}, '{kind}', "
                    f"unbase64('{encoded}'), '{note}')"
                )
    print(f"fixtures ready in {table}: {len(rows)} rows")


# ------------------------------------------------------------ query history


def query_history(
    statuses: List[str], since_ms: int, token: str
) -> List[Dict[str, Any]]:
    host = env("DATABRICKS_SERVER_HOSTNAME")
    params = {
        "filter_by.query_start_time_range.start_time_ms": since_ms,
        "max_results": 100,
    }
    for status in statuses:
        params.setdefault("filter_by.statuses", []).append(status)
    response = requests.get(
        f"https://{host}/api/2.0/sql/history/queries",
        params=params,
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    response.raise_for_status()
    return response.json().get("res", [])


def wait_for_history(
    marker: str, statuses: List[str], since_ms: int, token: str, timeout: float
) -> Optional[Dict[str, Any]]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for entry in query_history(statuses, since_ms, token):
            if marker in entry.get("query_text", ""):
                return entry
        time.sleep(3)
    return None


# ------------------------------------------------------------------ targets


def workflow_specification(
    sql_statement: str, *, max_rows: int, timeout_seconds: int, skip: bool = True
) -> Dict[str, Any]:
    return {
        "version": "1.0",
        "inputs": [{"type": "WorkflowParameter", "name": "token"}],
        "steps": [
            {
                "type": BLOCK_TYPE,
                "name": "db",
                "server_hostname": env("DATABRICKS_SERVER_HOSTNAME"),
                "http_path": env("DATABRICKS_HTTP_PATH"),
                "access_token": "$inputs.token",
                "sql_statement": sql_statement,
                "image_column": "bytes",
                "image_name_column": "id",
                "max_rows": max_rows,
                "timeout_seconds": timeout_seconds,
                "skip_undecodable": skip,
            }
        ],
        "outputs": [
            {"type": "JsonField", "name": "images", "selector": "$steps.db.images"},
            {"type": "JsonField", "name": "names", "selector": "$steps.db.image_names"},
            {"type": "JsonField", "name": "metadata", "selector": "$steps.db.metadata"},
        ],
    }


TRIVIAL_WORKFLOW = {
    "version": "1.0",
    "inputs": [{"type": "WorkflowParameter", "name": "value"}],
    "steps": [],
    "outputs": [{"type": "JsonField", "name": "echo", "selector": "$inputs.value"}],
}


class RunError(Exception):
    def __init__(self, message: str, elapsed: float):
        super().__init__(message)
        self.elapsed = elapsed


class LocalTarget:
    def run(
        self, specification: Dict[str, Any], inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        from roboflow_workflows.core_steps.common.entities import StepExecutionMode
        from roboflow_workflows.execution_engine.core import ExecutionEngine

        started = time.monotonic()
        try:
            engine = ExecutionEngine.init(
                workflow_definition=specification,
                init_parameters={
                    "workflows_core.step_execution_mode": StepExecutionMode.LOCAL
                },
                max_concurrent_steps=1,
            )
            result = engine.run(runtime_parameters=inputs)[0]
        except Exception as error:
            raise RunError(
                f"{type(error).__name__}: {error}", time.monotonic() - started
            )
        images = result.get("images") or []
        shapes = [tuple(image.numpy_image.shape) for image in images]
        return {
            "names": result.get("names", []),
            "metadata": result.get("metadata", []),
            "shapes": shapes,
            "elapsed": time.monotonic() - started,
        }


class ServerTarget:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = env("ROBOFLOW_API_KEY")

    def run(
        self, specification: Dict[str, Any], inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        started = time.monotonic()
        response = requests.post(
            f"{self.base_url}/workflows/run",
            json={
                "api_key": self.api_key,
                "specification": specification,
                "inputs": inputs,
            },
            timeout=900,
        )
        elapsed = time.monotonic() - started
        if response.status_code != 200:
            raise RunError(
                f"HTTP {response.status_code}: {response.text[:500]}", elapsed
            )
        output = response.json()["outputs"][0]
        images = output.get("images") or []
        shapes = []
        for image in images:
            value = image.get("value") if isinstance(image, dict) else None
            if value:
                from PIL import Image

                with Image.open(io.BytesIO(base64.b64decode(value))) as decoded:
                    shapes.append((decoded.height, decoded.width, 3))
        return {
            "names": output.get("names", []),
            "metadata": output.get("metadata", []),
            "shapes": shapes,
            "elapsed": elapsed,
        }


# -------------------------------------------------------------------- cases


def case_import(target, table: str, token: str) -> None:
    sql = f"SELECT id, bytes, note FROM {table} WHERE kind = 'ok' ORDER BY id /* rf-live {RUN_ID} */"
    result = target.run(
        workflow_specification(sql, max_rows=FIXTURE_ROWS, timeout_seconds=120),
        {"token": token},
    )
    assert result["names"] == [str(i) for i in range(FIXTURE_ROWS)], result["names"]
    assert all(shape == (32, 48, 3) for shape in result["shapes"]), result["shapes"]
    assert [m["note"] for m in result["metadata"]][:2] == ["note-0", "note-1"]
    assert FIXTURE_ROWS > 8, "fixture must span more than one fetch batch of 8"
    print(f"  import: {len(result['names'])} rows in {result['elapsed']:.1f}s")


def case_bad_image(target, table: str, token: str) -> None:
    sql = f"SELECT id, bytes FROM {table} WHERE kind IN ('ok', 'bad') ORDER BY id /* rf-live {RUN_ID} */"
    result = target.run(
        workflow_specification(sql, max_rows=100, timeout_seconds=120), {"token": token}
    )
    assert "900" not in result["names"] and len(result["names"]) == FIXTURE_ROWS
    print("  bad-image: malformed row skipped, run succeeded")


def case_big_metadata(target, table: str, token: str) -> None:
    sql = f"SELECT id, bytes, note FROM {table} WHERE kind IN ('ok', 'bigmeta') ORDER BY id /* rf-live {RUN_ID} */"
    result = target.run(
        workflow_specification(sql, max_rows=100, timeout_seconds=120), {"token": token}
    )
    assert "901" not in result["names"], "70 KB metadata row must be skipped"
    print("  big-metadata: oversized metadata row skipped")


def case_bad_token(target, table: str, token: str) -> None:
    sql = f"SELECT id, bytes FROM {table} WHERE kind = 'ok' LIMIT 1 /* rf-live {RUN_ID} */"
    bogus = "dapi" + uuid.uuid4().hex
    try:
        target.run(
            workflow_specification(sql, max_rows=1, timeout_seconds=60),
            {"token": bogus},
        )
    except RunError as error:
        assert bogus not in str(error), "credential leaked into the error"
        print(
            f"  bad-token: controlled failure in {error.elapsed:.1f}s, token not exposed"
        )
        return
    raise AssertionError("invalid credential was accepted")


def case_budget(target, table: str, token: str) -> None:
    sql = f"SELECT id, bytes FROM {table} WHERE kind = 'ok' ORDER BY id /* rf-live {RUN_ID} */"
    try:
        target.run(
            workflow_specification(sql, max_rows=100, timeout_seconds=120),
            {"token": token},
        )
    except RunError as error:
        assert "budget" in str(error), str(error)[:300]
        print("  budget: reduced run budget failed the run as documented")
        return
    raise AssertionError("expected the reduced budget to fail the run")


def case_disabled(target, table: str, token: str, since_ms: int) -> None:
    sql = f"SELECT id, bytes FROM {table} WHERE kind = 'ok' LIMIT 1 /* rf-live-disabled {RUN_ID} */"
    try:
        target.run(
            workflow_specification(sql, max_rows=1, timeout_seconds=60),
            {"token": token},
        )
    except RunError as error:
        assert "DATABRICKS_SQL_IMAGE_SOURCE_ENABLED" in str(error), str(error)[:300]
    else:
        raise AssertionError("disabled target executed the block")
    time.sleep(5)
    seen = [
        e
        for e in query_history(
            ["QUEUED", "RUNNING", "FINISHED", "FAILED", "CANCELED"], since_ms, token
        )
        if f"rf-live-disabled {RUN_ID}" in e.get("query_text", "")
    ]
    assert not seen, "a query reached the warehouse while disabled"
    print("  disabled: rejected at compile time, no warehouse query")


def slow_statement(table: str, marker: str, extra: str) -> str:
    busy = (
        extra
        or "(SELECT count(*) FROM range(2000000000) a CROSS JOIN range(2000000000) b) > 0"
    )
    return f"SELECT id, bytes FROM {table} WHERE kind = 'ok' AND {busy} LIMIT 1 /* {marker} */"


def case_slow(
    target, table: str, token: str, deadline: int, extra_sql: str
) -> Dict[str, Any]:
    marker = f"rf-live-slow {RUN_ID}-{uuid.uuid4().hex[:6]}"
    since_ms = int(time.time() * 1000) - 60_000
    started = time.monotonic()
    try:
        target.run(
            workflow_specification(
                slow_statement(table, marker, extra_sql),
                max_rows=1,
                timeout_seconds=deadline,
            ),
            {"token": token},
        )
    except RunError as error:
        elapsed = time.monotonic() - started
        assert (
            elapsed < deadline + CLEANUP_GRACE_SECONDS
        ), f"returned after {elapsed:.1f}s"
        assert "timeout" in str(error).lower() or "cancel" in str(error).lower(), str(
            error
        )[:300]
    else:
        raise AssertionError(
            "slow query finished before the deadline; raise --slow-sql cost"
        )
    active = wait_for_history(
        marker, ["RUNNING", "QUEUED"], since_ms, token, timeout=20
    )
    cancelled = wait_for_history(
        marker, ["CANCELED", "FAILED"], since_ms, token, timeout=CLEANUP_GRACE_SECONDS
    )
    assert cancelled is not None, "warehouse never reported the statement CANCELED"
    assert cancelled.get("status") in ("CANCELED", "FAILED"), cancelled.get("status")
    still_running = [
        e
        for e in query_history(["RUNNING", "QUEUED"], since_ms, token)
        if marker in e.get("query_text", "")
    ]
    assert not still_running, "the statement is still running after cancellation"
    print(
        f"  slow: returned in {elapsed:.1f}s (deadline {deadline}s + grace {CLEANUP_GRACE_SECONDS}s), "
        f"warehouse state {cancelled.get('status')}"
        + ("" if active is None else ", was observed active before cancellation")
    )
    return cancelled


def case_recovery(
    target, table: str, token: str, deadline: int, extra_sql: str
) -> None:
    since_ms = int(time.time() * 1000) - 60_000
    for _ in range(3):
        case_slow(target, table, token, deadline, extra_sql)
    case_import(target, table, token)
    running = [
        e
        for e in query_history(["RUNNING", "QUEUED"], since_ms, token)
        if "rf-live" in e.get("query_text", "")
    ]
    assert not running, f"{len(running)} test queries still running after recovery"
    print("  recovery: subsequent import succeeded, no accumulated running queries")


def case_concurrent(
    target, table: str, token: str, deadline: int, extra_sql: str, bound: float
) -> None:
    if not isinstance(target, ServerTarget):
        print("  concurrent: skipped (server target only)")
        return
    latencies: List[float] = []
    stop = threading.Event()

    def probe() -> None:
        while not stop.is_set():
            started = time.monotonic()
            try:
                target.run(TRIVIAL_WORKFLOW, {"value": "ping"})
            except RunError:
                latencies.append(float("inf"))
            else:
                latencies.append(time.monotonic() - started)
            time.sleep(1)

    thread = threading.Thread(target=probe, daemon=True)
    thread.start()
    try:
        case_slow(target, table, token, deadline, extra_sql)
    finally:
        stop.set()
        thread.join(timeout=30)
    assert latencies, "no probe completed"
    p95 = sorted(latencies)[int(0.95 * (len(latencies) - 1))]
    assert p95 < bound, f"p95 latency of unrelated jobs {p95:.2f}s exceeds {bound}s"
    print(
        f"  concurrent: {len(latencies)} probes, p95 {p95:.2f}s, median {statistics.median(latencies):.2f}s"
    )


# --------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--target", choices=["local", "server"], default="local")
    parser.add_argument(
        "--base-url",
        default=None,
        help="inference server or producer URL for --target server",
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="create the fixture table first (setup identity)",
    )
    parser.add_argument(
        "--cases",
        default="import,bad-image,big-metadata,bad-token,slow,recovery,concurrent",
    )
    parser.add_argument("--expect-budget-failure", action="store_true")
    parser.add_argument("--expect-disabled", action="store_true")
    parser.add_argument(
        "--slow-deadline",
        type=int,
        default=10,
        help="timeout_seconds for the slow cases",
    )
    parser.add_argument(
        "--slow-sql",
        default="",
        help="override the busy predicate of the slow statement",
    )
    parser.add_argument("--latency-bound-seconds", type=float, default=5.0)
    parser.add_argument("--report", default=None, help="write a JSON report here")
    args = parser.parse_args()

    table = env("DATABRICKS_TABLE")
    token = env("DATABRICKS_TOKEN")
    target: Any = (
        ServerTarget(args.base_url) if args.target == "server" else LocalTarget()
    )
    if args.target == "server" and not args.base_url:
        parser.error("--base-url is required with --target server")
    if args.setup:
        setup_fixtures(table)

    print(
        f"run {RUN_ID} against {args.target}"
        + (f" {args.base_url}" if args.base_url else "")
    )
    if args.target == "local":
        import databricks.sql as connector

        print(
            f"connector {connector.__version__}, pyarrow installed: {_installed('pyarrow')}"
        )

    since_ms = int(time.time() * 1000) - 60_000
    cases: Dict[str, Callable[[], Any]] = {
        "import": lambda: case_import(target, table, token),
        "bad-image": lambda: case_bad_image(target, table, token),
        "big-metadata": lambda: case_big_metadata(target, table, token),
        "bad-token": lambda: case_bad_token(target, table, token),
        "slow": lambda: case_slow(
            target, table, token, args.slow_deadline, args.slow_sql
        ),
        "recovery": lambda: case_recovery(
            target, table, token, args.slow_deadline, args.slow_sql
        ),
        "concurrent": lambda: case_concurrent(
            target,
            table,
            token,
            args.slow_deadline,
            args.slow_sql,
            args.latency_bound_seconds,
        ),
    }
    if args.expect_budget_failure:
        cases = {"budget": lambda: case_budget(target, table, token)}
    if args.expect_disabled:
        cases = {"disabled": lambda: case_disabled(target, table, token, since_ms)}

    selected = (
        [name for name in args.cases.split(",") if name in cases]
        if not (args.expect_budget_failure or args.expect_disabled)
        else list(cases)
    )
    report: Dict[str, Any] = {"run_id": RUN_ID, "target": args.target, "cases": {}}
    failures = 0
    for name in selected:
        print(f"[{name}]")
        try:
            cases[name]()
            report["cases"][name] = "passed"
        except Exception as error:  # report, keep going, cleanup below
            failures += 1
            report["cases"][
                name
            ] = f"FAILED: {type(error).__name__}: {str(error)[:300]}"
            print(f"  FAILED: {type(error).__name__}: {str(error)[:300]}")
    # Cleanup: no test statement may remain running. STATEMENT_TIMEOUT bounds
    # each one server-side; report any that is still visible as RUNNING.
    leftovers = [
        e
        for e in query_history(["RUNNING", "QUEUED"], since_ms, token)
        if "rf-live" in e.get("query_text", "")
    ]
    report["running_after_suite"] = len(leftovers)
    if leftovers:
        failures += 1
        print(f"CLEANUP: {len(leftovers)} test statements still running")
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
    print("result:", "PASS" if failures == 0 else f"FAIL ({failures})")
    return 0 if failures == 0 else 1


def _installed(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


if __name__ == "__main__":
    sys.exit(main())

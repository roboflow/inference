"""Live integration suite for the Databricks SQL Image Source block.

Runs the block against a real Databricks SQL warehouse, either locally
(``--target local``) or through an approved inference server
(``--target server``, which submits each case to
``POST /workflows/run`` with the Workflow specification inline), and verifies
failure and recovery behaviour, not only the happy path. Not part of the unit
suite: it needs a warehouse and credentials, supplied through the environment
(never on the command line):

    DATABRICKS_SERVER_HOSTNAME   workspace host (e.g. dbc-xxxx.cloud.databricks.com)
    DATABRICKS_HTTP_PATH         /sql/1.0/warehouses/<id>
    DATABRICKS_TOKEN             read-only identity used by the block under test
    DATABRICKS_SETUP_TOKEN       separate identity allowed to create fixture tables/views (required for every run)
    DATABRICKS_TABLE             catalog.schema.table for the fixtures
    ROBOFLOW_API_KEY             only for --target server
    DATABRICKS_LIVE_BASE_URL      approved HTTPS server origin (server target)

Cases (see the PR discussion for the acceptance table):

    import        20 small JPEGs + metadata, fetched across several batches
    bad-image     malformed row is skipped, run succeeds, no restart needed
    big-metadata  oversized metadata rows are reported, including all-rejected runs
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
        --cases import,slow,recovery

The server target exercises the synchronous /workflows/run API. It does not
submit or poll async-serverless jobs; that deployment needs separate validation.
"""

import argparse
import atexit
import base64
import io
import json
import os
import re
import statistics
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

import requests

BLOCK_TYPE = "roboflow_core/databricks_sql_image_source@v1"
FIXTURE_ROWS = 20
CLEANUP_GRACE_SECONDS = 35  # SOCKET_TIMEOUT_SECONDS + RETRY_DELAY_MAX_SECONDS
RUN_ID = uuid.uuid4().hex[:12]
_CREATED_VIEWS: List[str] = []
CASE_NAMES = (
    "import",
    "bad-image",
    "big-metadata",
    "bad-token",
    "slow",
    "recovery",
    "concurrent",
)
DEFAULT_CASES = ",".join(CASE_NAMES)


def env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        print(f"missing environment variable {name}")
        sys.exit(2)
    return value


def _safe_error(error: Exception) -> str:
    text = f"{type(error).__name__}: {error}"
    for name in ("DATABRICKS_TOKEN", "DATABRICKS_SETUP_TOKEN", "ROBOFLOW_API_KEY"):
        secret = os.environ.get(name)
        if secret:
            text = text.replace(secret, "***")

    result = text[:300]

    return result


def _server_origin(value: str) -> str:
    """Validate the operator-configured destination before reading credentials."""
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
        or "\\" in value
        or any(character.isspace() for character in value)
    ):
        raise ValueError(
            "DATABRICKS_LIVE_BASE_URL must be an HTTPS origin without credentials, path, query or fragment"
        )

    # Accessing port also rejects nonnumeric or out-of-range values.
    if parsed.port not in (None, 443):
        raise ValueError("DATABRICKS_LIVE_BASE_URL must use HTTPS port 443")

    origin = value.rstrip("/")

    return origin


def _selected_cases(value: str) -> List[str]:
    names = value.split(",")
    if not names or any(name not in CASE_NAMES for name in names):
        raise ValueError("--cases must contain only: " + DEFAULT_CASES)

    selected_cases = list(dict.fromkeys(names))

    return selected_cases


def _fixture_table(value: str) -> str:
    # Setup-only fixture DDL uses a restricted ordinary identifier. The block
    # itself uses native IDENTIFIER parameters for all names.
    if not re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*", value
    ):
        raise ValueError(
            "DATABRICKS_TABLE must be catalog.schema.table using ordinary SQL identifiers"
        )

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
    statuses: List[str], *, since_ms: int, token: str
) -> List[Dict[str, Any]]:
    """Read warehouse query history without following credentialed redirects.

    Args:
        statuses (List[str]): Warehouse states to include.
        since_ms (int): Earliest start time in Unix milliseconds.
        token (str): Identity allowed to read the test queries.

    Returns:
        Query-history entries returned by the warehouse.

    Raises:
        RuntimeError: If the warehouse returns a non-success status.
        requests.RequestException: If the history request fails.
    """
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
        allow_redirects=False,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Query history returned HTTP {response.status_code}")

    entries = response.json().get("res", [])

    return entries


def wait_for_history(
    marker: str, *, statuses: List[str], since_ms: int, token: str, timeout: float
) -> Optional[Dict[str, Any]]:
    """Wait for a marked query to appear in one of the requested states.

    Args:
        marker (str): Unique substring identifying this test query.
        statuses (List[str]): Warehouse states accepted by the caller.
        since_ms (int): Earliest query start time in Unix milliseconds.
        token (str): Credential with query-history access.
        timeout (float): Maximum polling duration in seconds.

    Returns:
        Optional[Dict[str, Any]]: First matching entry, or None when polling expires.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for entry in query_history(statuses, since_ms=since_ms, token=token):
            if marker in entry.get("query_text", ""):
                return entry
        time.sleep(3)

    return None


# ------------------------------------------------------------------ targets


def create_fixture_view(statement: str) -> Tuple[str, List[str]]:
    """Create a test-only view with the setup credential, separate from block SQL.

    Args:
        statement (str): Runner-owned SELECT defining the fixture selection.

    Returns:
        Qualified view name and its non-image metadata column names.
    """
    table = _fixture_table(env("DATABRICKS_TABLE"))
    marker = re.search(r"/\* (rf_live_slow_[a-z0-9_]+) \*/", statement)
    name = marker.group(1) if marker else f"rf_live_{RUN_ID}_{uuid.uuid4().hex[:6]}"
    view = ".".join(table.split(".")[:2] + [name])
    with sql_connect(env("DATABRICKS_SETUP_TOKEN")) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "CREATE VIEW IDENTIFIER(:view) AS " + statement,
                parameters={"view": view},
            )
            _CREATED_VIEWS.append(view)
            cursor.execute(
                "DESCRIBE TABLE IDENTIFIER(:view)", parameters={"view": view}
            )
            columns = [
                row[0]
                for row in cursor.fetchall()
                if row[0] and not row[0].startswith("#")
            ]
    fixture = view, [column for column in columns if column != "bytes"]

    return fixture


def cleanup_fixture_views() -> None:
    """Drop only the temporary views created by this test process."""
    if not _CREATED_VIEWS:
        return

    try:
        with sql_connect(env("DATABRICKS_SETUP_TOKEN")) as connection:
            with connection.cursor() as cursor:
                for view in _CREATED_VIEWS:
                    cursor.execute(
                        "DROP VIEW IF EXISTS IDENTIFIER(:view)",
                        parameters={"view": view},
                    )
        _CREATED_VIEWS.clear()
    except Exception as error:
        print("fixture view cleanup failed: " + _safe_error(error), file=sys.stderr)
        raise


def workflow_specification(
    sql_statement: str, *, max_rows: int, timeout_seconds: int, skip: bool = True
) -> Dict[str, Any]:
    """Create a fixture view and a workflow that reads it through the fixed template.

    Args:
        sql_statement (str): Runner-owned SELECT used only to create the fixture view.
        max_rows (int): Maximum rows the block should fetch from the view.
        timeout_seconds (int): Deadline passed to the block under test.
        skip (bool): Whether the block should skip undecodable rows.

    Returns:
        Dict[str, Any]: Workflow definition using a runtime token input.
    """
    view, metadata_columns = create_fixture_view(sql_statement)
    catalog, schema_name, table = view.split(".")
    specification = {
        "version": "1.0",
        "inputs": [{"type": "WorkflowParameter", "name": "token"}],
        "steps": [
            {
                "type": BLOCK_TYPE,
                "name": "db",
                "server_hostname": env("DATABRICKS_SERVER_HOSTNAME"),
                "http_path": env("DATABRICKS_HTTP_PATH"),
                "access_token": "$inputs.token",
                "catalog": catalog,
                "schema_name": schema_name,
                "table": table,
                "metadata_columns": metadata_columns,
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
            {"type": "JsonField", "name": "errors", "selector": "$steps.db.errors"},
        ],
    }

    return specification


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
    """Execute live cases in the local Workflows engine."""

    def run(
        self, specification: Dict[str, Any], *, inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run a workflow locally and normalize its image outputs.

        Args:
            specification (Dict[str, Any]): Inline workflow definition to execute.
            inputs (Dict[str, Any]): Runtime inputs, including the warehouse credential.

        Returns:
            Dict[str, Any]: Names, metadata, image shapes and elapsed seconds.

        Raises:
            RunError: Local workflow compilation or execution fails.
        """
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
        shapes = [
            tuple(image.numpy_image.shape) for image in images if image is not None
        ]
        outputs = {
            "names": result.get("names", []),
            "metadata": result.get("metadata", []),
            "errors": result.get("errors", []),
            "shapes": shapes,
            "elapsed": time.monotonic() - started,
        }

        return outputs


class ServerTarget:
    """Execute cases through the operator-configured synchronous inference API."""

    def __init__(self):
        """Validate the configured server origin before loading its API key."""
        self.base_url = _server_origin(env("DATABRICKS_LIVE_BASE_URL"))
        self.api_key = env("ROBOFLOW_API_KEY")

    def run(
        self, specification: Dict[str, Any], *, inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run a workflow on the approved server and normalize image outputs.

        Args:
            specification (Dict[str, Any]): Inline workflow definition.
            inputs (Dict[str, Any]): Runtime inputs, including the warehouse credential.

        Returns:
            Names, metadata, image shapes and elapsed seconds.

        Raises:
            RunError: If the server returns an unsuccessful response.
            requests.RequestException: If the server request fails.
        """
        started = time.monotonic()
        response = requests.post(
            f"{self.base_url}/workflows/run",
            json={
                "api_key": self.api_key,
                "specification": specification,
                "inputs": inputs,
            },
            timeout=900,
            allow_redirects=False,
        )
        elapsed = time.monotonic() - started
        if response.status_code != 200:
            raise RunError(f"HTTP {response.status_code}: {response.text}", elapsed)

        output = response.json()["outputs"][0]
        images = output.get("images") or []
        shapes = []
        for image in images:
            value = image.get("value") if isinstance(image, dict) else None
            if value:
                from PIL import Image

                with Image.open(io.BytesIO(base64.b64decode(value))) as decoded:
                    shapes.append((decoded.height, decoded.width, 3))
        result = {
            "names": output.get("names", []),
            "metadata": output.get("metadata", []),
            "errors": output.get("errors", []),
            "shapes": shapes,
            "elapsed": elapsed,
        }

        return result


# -------------------------------------------------------------------- cases


def case_import(target, *, table: str, token: str) -> None:
    """Verify a multi-batch image import and its associated names and metadata.

    Args:
        target (LocalTarget | ServerTarget): Runner used to execute the fixture workflow.
        table (str): Fully qualified fixture table.
        token (str): Read-only warehouse credential.

    Raises:
        AssertionError: The returned images, names or metadata differ from the fixture.
    """
    sql = f"SELECT id, bytes, note FROM {table} WHERE kind = 'ok' ORDER BY id /* rf-live {RUN_ID} */"
    result = target.run(
        workflow_specification(sql, max_rows=FIXTURE_ROWS, timeout_seconds=120),
        inputs={"token": token},
    )
    assert sorted(result["names"], key=int) == [
        str(i) for i in range(FIXTURE_ROWS)
    ], result["names"]
    assert len(result["shapes"]) == FIXTURE_ROWS
    assert all(shape == (32, 48, 3) for shape in result["shapes"]), result["shapes"]
    assert {
        name: metadata["note"]
        for name, metadata in zip(result["names"], result["metadata"])
    } == {str(i): f"note-{i}" for i in range(FIXTURE_ROWS)}
    assert FIXTURE_ROWS > 8, "fixture must span more than one fetch batch of 8"
    print(f"  import: {len(result['names'])} rows in {result['elapsed']:.1f}s")


def case_bad_image(target, *, table: str, token: str) -> None:
    """Verify that a malformed image is skipped while valid fixture rows survive.

    Args:
        target (LocalTarget | ServerTarget): Runner used to execute the fixture workflow.
        table (str): Fully qualified fixture table.
        token (str): Read-only warehouse credential.

    Raises:
        AssertionError: The malformed row is emitted or valid rows are lost.
    """
    sql = f"SELECT id, bytes FROM {table} WHERE kind IN ('ok', 'bad') ORDER BY id /* rf-live {RUN_ID} */"
    result = target.run(
        workflow_specification(sql, max_rows=100, timeout_seconds=120),
        inputs={"token": token},
    )
    assert "900" not in result["names"] and len(result["names"]) == FIXTURE_ROWS
    print("  bad-image: malformed row skipped, run succeeded")


def case_big_metadata(target, *, table: str, token: str) -> None:
    """Verify value and aggregate limits, including an all-rejected import.

    Args:
        target (LocalTarget | ServerTarget): Local or server workflow runner.
        table (str): Fully qualified fixture table.
        token (str): Read-only warehouse credential.

    Raises:
        AssertionError: If rejected metadata is not reported separately from images.
    """
    sql = f"SELECT id, bytes, note FROM {table} WHERE kind IN ('ok', 'bigmeta') ORDER BY id /* rf-live {RUN_ID} */"
    result = target.run(
        workflow_specification(sql, max_rows=100, timeout_seconds=120),
        inputs={"token": token},
    )
    assert "901" not in result["names"]
    errors = [error for error in result["errors"] if error is not None]
    assert len(errors) == 1
    assert errors[0]["image_name"] == "901"
    assert errors[0]["code"] == "METADATA_VALUE_TOO_LONG"
    assert sum(name is not None for name in result["names"]) == FIXTURE_ROWS

    # Seventeen independently bounded values exceed the 64 KiB row limit.
    columns = ", ".join(
        f"CASE WHEN id = 901 THEN repeat('m', 4096) ELSE '' END AS note_{i}"
        for i in range(17)
    )
    sql = (
        f"SELECT id, bytes, {columns} FROM {table} "
        f"WHERE kind IN ('ok', 'bigmeta') ORDER BY id /* rf-live {RUN_ID} */"
    )
    result = target.run(
        workflow_specification(sql, max_rows=100, timeout_seconds=120),
        inputs={"token": token},
    )
    assert "901" not in result["names"], "metadata totaling over 64 KiB must be skipped"
    assert sum(name is not None for name in result["names"]) == FIXTURE_ROWS
    errors = [error for error in result["errors"] if error is not None]
    assert len(errors) == 1 and errors[0]["code"] == "METADATA_TOO_LARGE"

    result = target.run(
        workflow_specification(
            f"SELECT id, bytes, note FROM {table} WHERE id = 901 /* rf-live {RUN_ID} */",
            max_rows=100,
            timeout_seconds=120,
        ),
        inputs={"token": token},
    )
    assert result["names"] == [None] and result["metadata"] == [None]
    assert result["shapes"] == []
    assert result["errors"][0]["code"] == "METADATA_VALUE_TOO_LONG"
    print("  big-metadata: rejected rows reported, including all-rejected import")


def case_bad_token(target, *, table: str, token: str) -> None:
    """Verify invalid credentials fail without appearing in the error report.

    Args:
        target (LocalTarget | ServerTarget): Runner used to execute the fixture workflow.
        table (str): Fully qualified fixture table.
        token (str): Unused valid credential retained for the common case interface.

    Raises:
        AssertionError: Invalid credentials are accepted or exposed in an error.
    """
    sql = f"SELECT id, bytes FROM {table} WHERE kind = 'ok' LIMIT 1 /* rf-live {RUN_ID} */"
    bogus = "dapi" + uuid.uuid4().hex
    try:
        target.run(
            workflow_specification(sql, max_rows=1, timeout_seconds=60),
            inputs={"token": bogus},
        )
    except RunError as error:
        assert bogus not in str(error), "credential leaked into the error"
        print(
            f"  bad-token: controlled failure in {error.elapsed:.1f}s, token not exposed"
        )

        return
    raise AssertionError("invalid credential was accepted")


def case_budget(target, *, table: str, token: str) -> None:
    """Verify that an undersized invocation budget fails the entire import.

    Args:
        target (LocalTarget | ServerTarget): Runner configured with a reduced run budget.
        table (str): Fully qualified fixture table.
        token (str): Read-only warehouse credential.

    Raises:
        AssertionError: The workflow succeeds or fails for an unrelated reason.
    """
    sql = f"SELECT id, bytes FROM {table} WHERE kind = 'ok' ORDER BY id /* rf-live {RUN_ID} */"
    try:
        target.run(
            workflow_specification(sql, max_rows=100, timeout_seconds=120),
            inputs={"token": token},
        )
    except RunError as error:
        assert "budget" in str(error), str(error)
        print("  budget: reduced run budget failed the run as documented")

        return
    raise AssertionError("expected the reduced budget to fail the run")


def case_disabled(target, *, table: str, token: str, since_ms: int) -> None:
    """Verify that a disabled block never reaches the warehouse.

    Args:
        target (LocalTarget | ServerTarget): Runner configured with the block disabled.
        table (str): Fully qualified fixture table.
        token (str): Read-only warehouse credential with query-history access.
        since_ms (int): Earliest query-history timestamp to inspect, in Unix milliseconds.

    Raises:
        AssertionError: Execution succeeds, fails unexpectedly, or appears in history.
    """
    sql = f"SELECT id, bytes FROM {table} WHERE kind = 'ok' LIMIT 1 /* rf-live-disabled {RUN_ID} */"
    try:
        target.run(
            workflow_specification(sql, max_rows=1, timeout_seconds=60),
            inputs={"token": token},
        )
    except RunError as error:
        assert "DATABRICKS_SQL_IMAGE_SOURCE_ENABLED" in str(error), str(error)
    else:
        raise AssertionError("disabled target executed the block")
    time.sleep(5)
    seen = [
        e
        for e in query_history(
            ["QUEUED", "RUNNING", "FINISHED", "FAILED", "CANCELED"],
            since_ms=since_ms,
            token=token,
        )
        if f"rf-live-disabled {RUN_ID}" in e.get("query_text", "")
    ]
    assert not seen, "a query reached the warehouse while disabled"
    print("  disabled: rejected at compile time, no warehouse query")


def slow_statement(table: str, *, marker: str, extra: str) -> str:
    """Build a setup-only fixture SELECT with an expensive predicate.

    Args:
        table (str): Fully qualified fixture table validated by the runner.
        marker (str): Unique query-history marker for this invocation.
        extra (str): Operator-provided expensive predicate, or empty for the default.

    Returns:
        str: SELECT used to define a temporary view, never passed as block input.
    """
    busy = (
        extra
        or "(SELECT count(*) FROM range(2000000000) a CROSS JOIN range(2000000000) b) > 0"
    )
    result = f"SELECT id, bytes FROM {table} WHERE kind = 'ok' AND {busy} LIMIT 1 /* {marker} */"

    return result


def case_slow(
    target, *, table: str, token: str, deadline: int, extra_sql: str
) -> Dict[str, Any]:
    """Observe an active query and verify its cancellation after a timeout.

    Args:
        target (LocalTarget | ServerTarget): Local or server workflow runner.
        table (str): Fully qualified fixture table.
        token (str): Read-only warehouse credential with query-history access.
        deadline (int): Invocation timeout in seconds.
        extra_sql (str): Optional expensive SQL predicate for the fixture query.

    Returns:
        The terminal query-history entry confirming cancellation.

    Raises:
        AssertionError: If execution, active observation or cancellation fails.
    """
    marker = f"rf_live_slow_{RUN_ID}_{uuid.uuid4().hex[:6]}"
    since_ms = int(time.time() * 1000) - 60_000
    started = time.monotonic()
    finished = threading.Event()
    outcome: Dict[str, Any] = {}

    def _run_query() -> None:
        try:
            target.run(
                workflow_specification(
                    slow_statement(table, marker=marker, extra=extra_sql),
                    max_rows=1,
                    timeout_seconds=deadline,
                ),
                inputs={"token": token},
            )
        except Exception as error:
            outcome["error"] = error
        finally:
            outcome["elapsed"] = time.monotonic() - started
            finished.set()

    thread = threading.Thread(target=_run_query, daemon=True)
    thread.start()
    observed_active = False
    try:
        while not finished.is_set() and time.monotonic() < started + deadline:
            entries = query_history(["RUNNING"], since_ms=since_ms, token=token)
            # A historical RUNNING row received after completion is not evidence
            # that we observed the query while this invocation was active.
            if not finished.is_set() and time.monotonic() < started + deadline:
                observed_active |= any(
                    marker in entry.get("query_text", "") for entry in entries
                )
            finished.wait(timeout=0.5)
    finally:
        thread.join(
            timeout=max(
                0, started + deadline + CLEANUP_GRACE_SECONDS - time.monotonic()
            )
        )

    assert (
        not thread.is_alive()
    ), "target did not finish within deadline plus cleanup allowance"
    elapsed = outcome["elapsed"]
    assert elapsed < deadline + CLEANUP_GRACE_SECONDS, f"returned after {elapsed:.1f}s"
    assert (
        observed_active
    ), "query was not observed RUNNING before the invocation deadline"
    error = outcome.get("error")
    assert isinstance(
        error, RunError
    ), "slow query did not fail with a controlled timeout"
    assert "timeout" in str(error).lower() or "cancel" in str(error).lower(), str(error)

    terminal = wait_for_history(
        marker,
        statuses=["CANCELED", "FAILED", "FINISHED"],
        since_ms=since_ms,
        token=token,
        timeout=CLEANUP_GRACE_SECONDS,
    )
    assert terminal is not None, "warehouse never reported a terminal statement state"
    # FAILED also includes permissions, malformed SQL and unrelated failures.
    # Until a specific statement-timeout error is verified, require CANCELED.
    assert (
        terminal.get("status") == "CANCELED"
    ), "warehouse did not confirm cancellation"
    still_running = [
        e
        for e in query_history(["RUNNING", "QUEUED"], since_ms=since_ms, token=token)
        if marker in e.get("query_text", "")
    ]
    assert not still_running, "the statement is still running after cancellation"
    print(
        f"  slow: observed RUNNING before deadline; returned in {elapsed:.1f}s; "
        "warehouse confirmed CANCELED"
    )

    return terminal


def case_recovery(
    target, *, table: str, token: str, deadline: int, extra_sql: str
) -> None:
    """Verify repeated timeouts leave the runner able to import images.

    Args:
        target (LocalTarget | ServerTarget): Runner used for timeout and recovery cases.
        table (str): Fully qualified fixture table.
        token (str): Read-only warehouse credential with query-history access.
        deadline (int): Timeout in seconds for each slow invocation.
        extra_sql (str): Optional expensive predicate in the setup-only fixture query.

    Raises:
        AssertionError: A timeout, recovery import or outstanding-query check fails.
    """
    since_ms = int(time.time() * 1000) - 60_000
    for _ in range(3):
        case_slow(
            target, table=table, token=token, deadline=deadline, extra_sql=extra_sql
        )
    case_import(target, table=table, token=token)
    running = [
        e
        for e in query_history(["RUNNING", "QUEUED"], since_ms=since_ms, token=token)
        if RUN_ID in e.get("query_text", "")
    ]
    assert not running, f"{len(running)} test queries still running after recovery"
    print("  recovery: subsequent import succeeded, no accumulated running queries")


def case_concurrent(
    target, *, table: str, token: str, deadline: int, extra_sql: str, bound: float
) -> None:
    """Measure unrelated request latency while a server invocation times out.

    Args:
        target (LocalTarget | ServerTarget): Server runner; local targets skip this case.
        table (str): Fully qualified fixture table.
        token (str): Read-only warehouse credential with query-history access.
        deadline (int): Timeout in seconds for the slow invocation.
        extra_sql (str): Optional expensive predicate in the fixture query.
        bound (float): Maximum acceptable p95 latency in seconds for unrelated requests.

    Raises:
        AssertionError: No probe completes, the latency bound is exceeded, or cancellation fails.
    """
    if not isinstance(target, ServerTarget):
        print("  concurrent: skipped (server target only)")

        return

    latencies: List[float] = []
    stop = threading.Event()

    def _probe() -> None:
        while not stop.is_set():
            started = time.monotonic()
            try:
                target.run(TRIVIAL_WORKFLOW, inputs={"value": "ping"})
            except RunError:
                latencies.append(float("inf"))
            else:
                latencies.append(time.monotonic() - started)
            time.sleep(1)

    thread = threading.Thread(target=_probe, daemon=True)
    thread.start()
    try:
        case_slow(
            target, table=table, token=token, deadline=deadline, extra_sql=extra_sql
        )
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
    """Run explicitly selected live cases using environment credentials.

    Returns:
        Zero when all selected cases and cleanup checks pass, otherwise one.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--target", choices=["local", "server"], default="local")
    parser.add_argument(
        "--setup",
        action="store_true",
        help="create the fixture table first (setup identity)",
    )
    parser.add_argument(
        "--cases",
        default=DEFAULT_CASES,
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

    try:
        selected = _selected_cases(args.cases)
        if not 5 <= args.slow_deadline <= 600:
            raise ValueError("--slow-deadline must be between 5 and 600 seconds")

        if args.expect_budget_failure and args.expect_disabled:
            raise ValueError("select only one special expectation")

        table = _fixture_table(env("DATABRICKS_TABLE"))
        target: Any = ServerTarget() if args.target == "server" else LocalTarget()
    except ValueError as error:
        parser.error(str(error))
    token = env("DATABRICKS_TOKEN")
    if args.setup:
        setup_fixtures(table)

    print(f"run {RUN_ID} against {args.target}")
    if args.target == "local":
        import databricks.sql as connector

        print(
            f"connector {connector.__version__}, pyarrow installed: {_installed('pyarrow')}"
        )

    since_ms = int(time.time() * 1000) - 60_000
    cases: Dict[str, Callable[[], Any]] = {
        "import": lambda: case_import(target, table=table, token=token),
        "bad-image": lambda: case_bad_image(target, table=table, token=token),
        "big-metadata": lambda: case_big_metadata(target, table=table, token=token),
        "bad-token": lambda: case_bad_token(target, table=table, token=token),
        "slow": lambda: case_slow(
            target,
            table=table,
            token=token,
            deadline=args.slow_deadline,
            extra_sql=args.slow_sql,
        ),
        "recovery": lambda: case_recovery(
            target,
            table=table,
            token=token,
            deadline=args.slow_deadline,
            extra_sql=args.slow_sql,
        ),
        "concurrent": lambda: case_concurrent(
            target,
            table=table,
            token=token,
            deadline=args.slow_deadline,
            extra_sql=args.slow_sql,
            bound=args.latency_bound_seconds,
        ),
    }
    if args.expect_budget_failure:
        cases = {"budget": lambda: case_budget(target, table=table, token=token)}
    if args.expect_disabled:
        cases = {
            "disabled": lambda: case_disabled(
                target, table=table, token=token, since_ms=since_ms
            )
        }

    if args.expect_budget_failure or args.expect_disabled:
        selected = list(cases)
    report: Dict[str, Any] = {"run_id": RUN_ID, "target": args.target, "cases": {}}
    failures = 0
    for name in selected:
        print(f"[{name}]")
        try:
            cases[name]()
            report["cases"][name] = "passed"
        except Exception as error:  # report, keep going, cleanup below
            failures += 1
            detail = _safe_error(error)
            report["cases"][name] = f"FAILED: {detail}"
            print(f"  FAILED: {detail}")
    # Cleanup: no test statement may remain running. STATEMENT_TIMEOUT bounds
    # each one server-side; report any that is still visible as RUNNING.
    leftovers = [
        e
        for e in query_history(["RUNNING", "QUEUED"], since_ms=since_ms, token=token)
        if RUN_ID in e.get("query_text", "")
    ]
    report["running_after_suite"] = len(leftovers)
    if leftovers:
        failures += 1
        print(f"CLEANUP: {len(leftovers)} test statements still running")
    try:
        cleanup_fixture_views()
    except Exception:
        failures += 1
        report["fixture_cleanup"] = "FAILED"
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("result:", "PASS" if failures == 0 else f"FAIL ({failures})")
    result = 0 if failures == 0 else 1

    return result


def _installed(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


if __name__ == "__main__":
    atexit.register(cleanup_fixture_views)
    try:
        exit_code = main()
    except Exception as error:
        print(_safe_error(error), file=sys.stderr)
        exit_code = 1
    sys.exit(exit_code)

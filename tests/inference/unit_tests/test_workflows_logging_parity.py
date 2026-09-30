import json
import os
import subprocess
import sys

# Runs in a child interpreter: logger configuration is decided at import time
# from API_LOGGING_ENABLED, and the structured branch installs a StreamHandler
# on stderr with propagate=False.
PROBE = r"""
import logging

from asgi_correlation_id import correlation_id
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

from inference_sdk.config import execution_id

# A real provider, so spans carry trace/span ids instead of the no-op proxy's zeros.
trace.set_tracer_provider(TracerProvider())

from inference.core.logger import logger as server_logger  # noqa: E402
from inference.core.telemetry import start_span  # noqa: E402

correlation_id.set("corr-123")
execution_id.set("exec-456")
with start_span("parity"):
    server_logger.info("from-server")
    logging.getLogger("inference.core.workflows.smoke").info("from-workflows")
"""


def _run(api_logging_enabled: str, stream: str) -> list:
    env = {
        **os.environ,
        "API_LOGGING_ENABLED": api_logging_enabled,
        "LOG_LEVEL": "INFO",  # env.py:576 defaults to WARNING - INFO records would vanish
        "CORRELATION_ID_LOG_KEY": "request_id",
        "GCP_SERVERLESS": "False",  # the GCP branch renames `event` to `message`
        "DISABLE_VERSION_CHECK": "True",
        "OFFLINE_MODE": "False",  # telemetry helpers are no-ops in offline mode
    }
    env.pop("DEDICATED_DEPLOYMENT_ID", None)  # also selects the GCP branch
    proc = subprocess.run(
        [sys.executable, "-c", PROBE], env=env, capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
    return [
        line
        for line in getattr(proc, stream).splitlines()
        if "from-server" in line or "from-workflows" in line
    ]


def test_structured_mode_keeps_context_for_workflow_loggers() -> None:
    # The structured branch installs logging.StreamHandler() -> stderr (logger.py:164).
    lines = _run("True", stream="stderr")
    assert len(lines) == 2, f"expected exactly one record each, got: {lines}"
    server, workflows = (json.loads(line) for line in lines)
    assert workflows["event"] == "from-workflows"
    # Context established in the probe must survive the stdlib path.
    assert workflows["request_id"] == "corr-123" == server["request_id"]
    assert workflows["execution_id"] == "exec-456" == server["execution_id"]
    assert workflows["trace_id"] == server["trace_id"]
    assert workflows["span_id"] == server["span_id"]
    # Same envelope: everything except callsite-specific fields.
    volatile = {"event", "timestamp", "filename", "func_name", "lineno"}
    assert set(server) - volatile == set(workflows) - volatile


def test_plain_mode_still_reaches_the_rich_handler_exactly_once() -> None:
    # RichHandler's console writes to stdout (logger.py:171).
    lines = _run("False", stream="stdout")
    assert len(lines) == 2, f"expected exactly one record each, got: {lines}"
    assert "from-workflows" in lines[1] and "INFO" in lines[1]

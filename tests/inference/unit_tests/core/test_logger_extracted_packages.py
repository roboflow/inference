"""`EXTRACTED_PACKAGE_LOGGERS` must share the "inference" logger's handler.

Each case runs in a fresh interpreter since `inference.core.logger` binds its
handler and level once per process, at import time.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]

# Stripped from the child env; each test sets back only what its case needs.
_CONTROLLED_VARIABLES = (
    "ROBOFLOW_API_KEY",
    "API_KEY",
    "GCP_SERVERLESS",
    "DEDICATED_DEPLOYMENT_ID",
    "ENABLE_IN_MEMORY_LOGS",
    "API_LOGGING_ENABLED",
    "LOG_LEVEL",
)

_STRUCTURAL_CHECKS_SCRIPT = """
import importlib
import io
import json
import logging

# inference.core re-exports "logger", shadowing the submodule attribute path.
logger_module = importlib.import_module("inference.core.logger")

inference_logger = logging.getLogger("inference")
extracted_loggers = {
    name: logging.getLogger(name)
    for name in logger_module.EXTRACTED_PACKAGE_LOGGERS
}

loggers = {
    name: {
        "handlers_match": extracted_logger.handlers == inference_logger.handlers,
        "level_match": extracted_logger.level == inference_logger.level,
        "propagate_false": extracted_logger.propagate is False,
    }
    for name, extracted_logger in extracted_loggers.items()
}

handler = inference_logger.handlers[0]
sink = io.StringIO()
if hasattr(handler, "console"):
    handler.console.file = sink
else:
    handler.stream = sink

logging.getLogger("roboflow_workflows.some.module").info("probe-message")

probe_count = sink.getvalue().count("probe-message")
print(json.dumps({"loggers": loggers, "probe_count": probe_count}))
"""

_MEMORY_LOGGING_SCRIPT = """
import json
import logging

import inference.core.logger
import inference.core.logging.memory_handler

inference.core.logging.memory_handler.setup_memory_logging()

logging.getLogger("roboflow_workflows.probe").info("memory-probe-message")
logging.getLogger("inference.probe").info("memory-probe-message")

logs = inference.core.logging.memory_handler.get_recent_logs(limit=1000)
workflows_count = sum(
    1
    for log in logs
    if log["logger"] == "roboflow_workflows.probe"
    and "memory-probe-message" in log["message"]
)
inference_count = sum(
    1
    for log in logs
    if log["logger"] == "inference.probe" and "memory-probe-message" in log["message"]
)
counts = {"workflows_count": workflows_count, "inference_count": inference_count}
print(json.dumps(counts))
"""


def _run_child(
    overrides: Optional[Dict[str, str]] = None,
    script: str = _STRUCTURAL_CHECKS_SCRIPT,
) -> dict:
    environment = {
        name: value
        for name, value in os.environ.items()
        if name not in _CONTROLLED_VARIABLES
    }
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "DISABLE_VERSION_CHECK": "True",
            "PYTHONPATH": os.pathsep.join(
                [str(REPO_ROOT / "workflows"), str(REPO_ROOT / "inference_models")]
            ),
        }
    )
    environment.update(overrides or {})
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=environment,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    result = json.loads(completed.stdout.strip().splitlines()[-1])

    return result


def _assert_structural_checks_pass(result: dict) -> None:
    assert result["loggers"]
    for checks in result["loggers"].values():
        assert checks["handlers_match"] is True
        assert checks["level_match"] is True
        assert checks["propagate_false"] is True


@pytest.mark.parametrize(
    "log_level, probe_count",
    [("INFO", 1), ("ERROR", 0)],
)
def test_extracted_logger_handler_matches_the_rich_handler_branch(
    log_level: str, probe_count: int
) -> None:
    result = _run_child({"LOG_LEVEL": log_level, "API_LOGGING_ENABLED": "False"})

    _assert_structural_checks_pass(result)
    assert result["probe_count"] == probe_count


def test_extracted_logger_handler_matches_the_api_logging_branch() -> None:
    # ProcessorFormatter formats a plain StreamHandler; sink still sees the message.
    result = _run_child({"LOG_LEVEL": "INFO", "API_LOGGING_ENABLED": "True"})

    _assert_structural_checks_pass(result)
    assert result["probe_count"] == 1


def test_memory_logging_captures_extracted_package_records() -> None:
    result = _run_child(
        {"ENABLE_IN_MEMORY_LOGS": "True", "LOG_LEVEL": "INFO"},
        script=_MEMORY_LOGGING_SCRIPT,
    )

    assert result["workflows_count"] == 1
    assert result["inference_count"] == 1

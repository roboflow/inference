import logging
import secrets
import subprocess
from pathlib import Path
from threading import Lock
from typing import List, Optional

import requests

from inference_server import configuration

logger = logging.getLogger(__name__)

NOTEBOOK_DIR = Path("/notebooks")
NOTEBOOK_PROBE_TIMEOUT_S = 2.0
NOTEBOOK_START_ERROR_MESSAGE = (
    "Notebook server could not be started. "
    "Check that jupyter-lab is installed in this environment."
)

_GENERATED_TOKEN = secrets.token_urlsafe(32)
_start_lock = Lock()
_process: Optional[subprocess.Popen] = None


class NotebookStartError(Exception):
    pass


def notebook_token() -> str:
    token = configuration.NOTEBOOK_PASSWORD or _GENERATED_TOKEN

    return token


def notebook_command() -> List[str]:
    token = notebook_token()
    command = [
        "jupyter-lab",
        "--allow-root",
        f"--port={configuration.NOTEBOOK_PORT}",
        "--ip=0.0.0.0",
        f"--notebook-dir={NOTEBOOK_DIR}",
        f"--NotebookApp.token={token}",
        f"--NotebookApp.password={token}",
    ]

    return command


def check_notebook_is_running() -> bool:
    try:
        response = requests.get(
            f"http://localhost:{configuration.NOTEBOOK_PORT}/",
            timeout=NOTEBOOK_PROBE_TIMEOUT_S,
        )
    except Exception:
        return False

    return response.status_code == 200


def start_notebook() -> None:
    global _process
    with _start_lock:
        if _process is not None and _process.poll() is None:
            return
        if check_notebook_is_running():
            return

        try:
            NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
            _process = subprocess.Popen(notebook_command())
        except OSError as error:
            logger.exception("Could not start the notebook server")
            raise NotebookStartError(NOTEBOOK_START_ERROR_MESSAGE) from error

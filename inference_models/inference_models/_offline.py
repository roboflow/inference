"""Dependency-light process-wide OFFLINE_MODE startup latch."""

import os
import sys
from typing import Any, Optional

from inference_models.errors import InvalidEnvVariable

_OFFLINE_MODE_PROCESS_LATCH_ENV = (
    "_ROBOFLOW_INFERENCE_OFFLINE_MODE_AT_PROCESS_START"
)
_OFFLINE_MODE_PROCESS_STATE_MODULE = "_roboflow_inference_process_state"
OFFLINE_MODE_CONTRACT_VERSION = 2
_existing_offline_mode_process_state = sys.modules.get(
    _OFFLINE_MODE_PROCESS_STATE_MODULE
)
OFFLINE_MODE = bool(
    getattr(_existing_offline_mode_process_state, "offline_mode", False)
)
if (
    _existing_offline_mode_process_state is not None
    and hasattr(_existing_offline_mode_process_state, "offline_mode")
):
    # A direct module reload must consume and re-publish the process latch, not
    # trust either environment variable after startup.
    os.environ[_OFFLINE_MODE_PROCESS_LATCH_ENV] = str(OFFLINE_MODE)
    if OFFLINE_MODE:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _parse_offline_mode(value: Any, variable_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized_value = value.lower()
        if normalized_value == "true":
            return True
        if normalized_value == "false":
            return False
    raise InvalidEnvVariable(
        message=(
            f"Expected a environment variable `{variable_name}` to be "
            f"(true or false) but got '{value}'"
        ),
        help_url=(
            "https://inference-models.roboflow.com/errors/"
            "runtime-environment/#invalidenvvariable"
        ),
    )


def publish_offline_mode(
    offline_mode_process_state: Any,
    requested_offline_mode: Optional[str],
    inherited_offline_mode: Optional[str],
) -> bool:
    """Publish a snapshot while the package initializer owns the shared lock."""

    global OFFLINE_MODE

    if not hasattr(offline_mode_process_state, "offline_mode"):
        initial_offline_mode = (
            requested_offline_mode
            if inherited_offline_mode is None
            else inherited_offline_mode
        )
        if initial_offline_mode is None:
            initial_offline_mode = "False"
        requested_variable_name = (
            "OFFLINE_MODE"
            if inherited_offline_mode is None
            else _OFFLINE_MODE_PROCESS_LATCH_ENV
        )
        offline_mode_process_state.offline_mode = _parse_offline_mode(
            value=initial_offline_mode,
            variable_name=requested_variable_name,
        )

    OFFLINE_MODE = bool(offline_mode_process_state.offline_mode)
    os.environ[_OFFLINE_MODE_PROCESS_LATCH_ENV] = str(OFFLINE_MODE)
    if OFFLINE_MODE:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return OFFLINE_MODE

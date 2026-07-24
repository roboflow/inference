import _thread
import os
import sys

_OFFLINE_MODE_PROCESS_LATCH_ENV = (
    "_ROBOFLOW_INFERENCE_OFFLINE_MODE_AT_PROCESS_START"
)
_OFFLINE_MODE_PROCESS_STATE_MODULE = "_roboflow_inference_process_state"

_offline_mode_process_state = sys.modules.get(_OFFLINE_MODE_PROCESS_STATE_MODULE)
_offline_mode_initialization_lock_owned = False
if _offline_mode_process_state is None:
    _candidate_process_state = type(sys)(_OFFLINE_MODE_PROCESS_STATE_MODULE)
    _candidate_process_state.lock = _thread.RLock()
    # Insert an already-owned lock. This closes the gap where a competing
    # package could observe a newly reserved state and publish before its owner.
    _candidate_process_state.lock.acquire()
    _offline_mode_process_state = sys.modules.setdefault(
        _OFFLINE_MODE_PROCESS_STATE_MODULE,
        _candidate_process_state,
    )
    if _offline_mode_process_state is _candidate_process_state:
        _offline_mode_initialization_lock_owned = True
    else:
        _candidate_process_state.lock.release()

if not _offline_mode_initialization_lock_owned:
    # Supports a fail-closed mixed-version startup where an older companion
    # package created the shared state before this package was imported.
    _offline_mode_initialization_lock = (
        _offline_mode_process_state.__dict__.setdefault(
            "lock", _thread.RLock()
        )
    )
    _offline_mode_initialization_lock.acquire()
else:
    _offline_mode_initialization_lock = _offline_mode_process_state.lock

try:
    if (
        not hasattr(_offline_mode_process_state, "offline_mode")
        and hasattr(
            _offline_mode_process_state, "offline_mode_initialization_error"
        )
    ):
        raise _offline_mode_process_state.offline_mode_initialization_error

    if (
        hasattr(os, "register_at_fork")
        and not getattr(_offline_mode_process_state, "at_fork_registered", False)
    ):

        def _reset_offline_mode_lock_after_fork() -> None:
            # A child created while another thread owns the lock would otherwise
            # inherit a permanently locked mutex.
            _offline_mode_process_state.lock = _thread.RLock()

        os.register_at_fork(after_in_child=_reset_offline_mode_lock_after_fork)
        _offline_mode_process_state.at_fork_registered = True

    if not hasattr(_offline_mode_process_state, "offline_mode_startup_snapshot"):
        # The owner snapshots both variables while already holding the shared
        # lock. A competing package cannot publish a later environment value.
        _offline_mode_process_state.offline_mode_startup_snapshot = (
            os.environ.get("OFFLINE_MODE"),
            os.environ.get(_OFFLINE_MODE_PROCESS_LATCH_ENV),
        )
    (
        _offline_mode_from_process_environment,
        _inherited_offline_mode_at_import,
    ) = _offline_mode_process_state.offline_mode_startup_snapshot

    def _parse_offline_mode(value: object, variable_name: str) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized_value = value.lower()
            if normalized_value == "true":
                return True
            if normalized_value == "false":
                return False
        raise ValueError(
            f"Expected {variable_name} to be a boolean (true or false), got {value!r}"
        )

    try:
        from dotenv import dotenv_values, load_dotenv
    except ImportError:  # pragma: no cover - only for dependency-light imports
        dotenv_values = None
        load_dotenv = None

    _dotenv_path = os.path.join(os.getcwd(), ".env")
    _offline_mode_from_dotenv = None
    _offline_mode_declared_in_dotenv = False
    if (
        not hasattr(_offline_mode_process_state, "offline_mode")
        and _offline_mode_from_process_environment is None
        and _inherited_offline_mode_at_import is None
        and dotenv_values is not None
    ):
        # Read the startup value independently from os.environ. load_dotenv does
        # not override a value another thread adds while this import is in flight.
        _dotenv_values = dotenv_values(_dotenv_path)
        _offline_mode_declared_in_dotenv = "OFFLINE_MODE" in _dotenv_values
        _offline_mode_from_dotenv = _dotenv_values.get("OFFLINE_MODE")

    # Preserve the historical behavior of loading every other project-local value.
    if load_dotenv is not None:
        load_dotenv(_dotenv_path)

    if not hasattr(_offline_mode_process_state, "offline_mode"):
        if _inherited_offline_mode_at_import is not None:
            _latched_offline_mode = _parse_offline_mode(
                _inherited_offline_mode_at_import,
                _OFFLINE_MODE_PROCESS_LATCH_ENV,
            )
        else:
            _initial_offline_mode = _offline_mode_from_process_environment
            if (
                _initial_offline_mode is None
                and _offline_mode_declared_in_dotenv
            ):
                _initial_offline_mode = _offline_mode_from_dotenv
            elif _initial_offline_mode is None:
                _initial_offline_mode = "False"
            _latched_offline_mode = _parse_offline_mode(
                _initial_offline_mode,
                "OFFLINE_MODE",
            )
        _offline_mode_process_state.offline_mode = _latched_offline_mode
except Exception as error:
    if not hasattr(_offline_mode_process_state, "offline_mode"):
        # A competing package must not recover from a malformed first owner's
        # startup configuration by re-reading a later environment value.
        _offline_mode_process_state.offline_mode_initialization_error = error
    raise
finally:
    _offline_mode_initialization_lock.release()

_LATCHED_OFFLINE_MODE = bool(_offline_mode_process_state.offline_mode)
os.environ[_OFFLINE_MODE_PROCESS_LATCH_ENV] = str(_LATCHED_OFFLINE_MODE)
if _LATCHED_OFFLINE_MODE:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict

try:
    _requested_offline_mode = _parse_offline_mode(
        os.getenv("OFFLINE_MODE", "False"),
        "OFFLINE_MODE",
    )
except ValueError:
    _requested_offline_mode = None
if (
    _requested_offline_mode is None
    or _requested_offline_mode != _LATCHED_OFFLINE_MODE
):
    warnings.warn(
        "Changing OFFLINE_MODE at runtime is not supported. The new value is "
        "being ignored; restart the process to change offline mode.",
        RuntimeWarning,
        stacklevel=1,
    )

if TYPE_CHECKING:
    from inference.core.interfaces.stream.inference_pipeline import InferencePipeline
    from inference.core.interfaces.stream.stream import Stream
    from inference.core.models.base import Model
    from inference.models.utils import get_model, get_roboflow_model

_LAZY_ATTRIBUTES: Dict[str, Callable[[], Any]] = {
    "Stream": lambda: _import_from("inference.core.interfaces.stream.stream", "Stream"),
    "InferencePipeline": lambda: _import_from(
        "inference.core.interfaces.stream.inference_pipeline", "InferencePipeline"
    ),
    "get_model": lambda: _import_from("inference.models.utils", "get_model"),
    "get_roboflow_model": lambda: _import_from(
        "inference.models.utils", "get_roboflow_model"
    ),
}


def _import_from(module_path: str, attribute_name: str) -> Any:
    """Import and return an attribute from the specified module."""
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, attribute_name)


def __getattr__(name: str) -> Any:
    """Implement lazy loading for module attributes."""
    if name in _LAZY_ATTRIBUTES:
        return _LAZY_ATTRIBUTES[name]()
    raise AttributeError(f"module 'inference' has no attribute '{name}'")


__all__ = [
    "InferencePipeline",
    "Stream",
    "get_model",
    "get_roboflow_model",
    "Model",
]

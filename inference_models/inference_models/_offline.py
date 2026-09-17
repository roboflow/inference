"""Process-wide OFFLINE_MODE startup latch — the single source of truth.

``inference`` depends on ``inference_models`` (unconditionally, in every
supported install), so this module is the one place the offline decision is
made. ``inference/__init__.py`` imports ``inference_models`` as its first
statement for exactly this reason: whichever package the user touches first,
this module runs before anything heavy.

Responsibilities, in order:

1. Decide ``OFFLINE_MODE`` once per process: an inherited parent-process
   marker wins, then the ``OFFLINE_MODE`` environment variable, then the
   project-local ``.env`` file, else ``False``. The decided value is
   re-published as the marker so spawned children latch the same decision even
   if ``os.environ`` is mutated later.
2. Pin the Hugging Face and Ultralytics cache locations (always, online
   included) so artefacts fetched during a warm-up run land on the cache
   volume a later offline process will mount. Explicit user configuration
   always wins — everything is ``setdefault``.
3. When offline: publish the Hugging Face / Transformers / Ultralytics
   offline switches. Those libraries snapshot the environment at import time,
   which is why this must run first and why an ``ultralytics`` import that
   beat us here is a hard error.

Changing ``OFFLINE_MODE`` at runtime is not supported;
``inference_models.configuration`` warns when the environment no longer
matches the latched value.
"""

import os
import sys
import tempfile

from dotenv import dotenv_values, load_dotenv

from inference_models.errors import InvalidEnvVariable

OFFLINE_MODE_PROCESS_LATCH_ENV = "_ROBOFLOW_INFERENCE_OFFLINE_MODE_AT_PROCESS_START"
# Backwards-compatible alias — external tooling greps for the private name.
_OFFLINE_MODE_PROCESS_LATCH_ENV = OFFLINE_MODE_PROCESS_LATCH_ENV
# A failed import evicts this module, so a malformed startup value is recorded
# here to keep later import attempts fail-closed: retrying with a mutated
# environment must not silently select a different mode after the process
# already refused to start.
_OFFLINE_MODE_STARTUP_ERROR_ENV = "_ROBOFLOW_INFERENCE_OFFLINE_MODE_STARTUP_ERROR"


def _parse_offline_mode(value: object, variable_name: str) -> bool:
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


def _decide_offline_mode() -> bool:
    inherited_value = os.environ.get(OFFLINE_MODE_PROCESS_LATCH_ENV)
    if inherited_value is not None:
        return _parse_offline_mode(inherited_value, OFFLINE_MODE_PROCESS_LATCH_ENV)
    requested_value = os.environ.get("OFFLINE_MODE")
    if requested_value is not None:
        return _parse_offline_mode(requested_value, "OFFLINE_MODE")
    # `inference` has historically loaded the project-local .env file. The
    # security-sensitive startup flag must resolve identically no matter which
    # package imports first, so the owner reads .env itself. A bare
    # `OFFLINE_MODE` line (present but valueless) fails closed instead of
    # silently selecting online mode.
    dotenv_startup_values = dotenv_values(_DOTENV_PATH)
    if "OFFLINE_MODE" in dotenv_startup_values:
        return _parse_offline_mode(
            dotenv_startup_values["OFFLINE_MODE"], "OFFLINE_MODE"
        )
    return False


def _pin_dependency_cache_locations() -> None:
    # Keep implicit Hugging Face downloads inside the cache volume that a
    # fresh offline process will mount. The Hub and Transformers snapshot
    # these variables at import time, so publish every relevant path before
    # importing either library. Explicit user configuration always wins.
    dependency_cache_root = (
        os.environ.get("INFERENCE_HOME")
        or os.environ.get("MODEL_CACHE_DIR")
        or "/tmp/cache"
    )
    os.environ.setdefault(
        "HF_HOME",
        os.path.join(dependency_cache_root, "hf_home"),
    )
    os.environ.setdefault(
        "HF_HUB_CACHE",
        os.environ.get("HUGGINGFACE_HUB_CACHE")
        or os.environ.get("TRANSFORMERS_CACHE")
        or os.path.join(os.environ["HF_HOME"], "hub"),
    )
    os.environ.setdefault(
        "HF_MODULES_CACHE",
        os.path.join(os.environ["HF_HOME"], "modules"),
    )
    # Ultralytics persists settings even while merely loading them. Keep this
    # mutable runtime state out of the model artifact volume, which is expected
    # to be mounted read-only during an offline restart.
    os.environ.setdefault(
        "YOLO_CONFIG_DIR",
        os.path.join(
            tempfile.gettempdir(),
            "roboflow-inference",
            "ultralytics",
        ),
    )


def _publish_dependency_offline_switches(enforce_import_order: bool) -> None:
    # The import-order check protects the FIRST establishment of the latch.
    # Re-executions with an inherited marker (module reloads, spawned workers)
    # legitimately happen after heavy libraries were imported under the same,
    # already-enforced decision.
    if enforce_import_order and any(
        module_name == "ultralytics" or module_name.startswith("ultralytics.")
        for module_name in sys.modules
    ):
        raise RuntimeError(
            "Ultralytics was imported before inference-models could establish "
            "OFFLINE_MODE. Restart the process and import inference_models "
            "(or inference) first."
        )
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["YOLO_OFFLINE"] = "True"


_DOTENV_PATH = os.path.join(os.getcwd(), ".env")

_recorded_startup_error = os.environ.get(_OFFLINE_MODE_STARTUP_ERROR_ENV)
if _recorded_startup_error is not None:
    raise InvalidEnvVariable(
        message=_recorded_startup_error,
        help_url=(
            "https://inference-models.roboflow.com/errors/"
            "runtime-environment/#invalidenvvariable"
        ),
    )
_first_establishment = os.environ.get(OFFLINE_MODE_PROCESS_LATCH_ENV) is None
try:
    OFFLINE_MODE = _decide_offline_mode()
except InvalidEnvVariable as _startup_error:
    os.environ[_OFFLINE_MODE_STARTUP_ERROR_ENV] = str(_startup_error.args[0])
    raise
# Preserve the historical behavior of loading every other project-local value.
# The decision above is already made, so a concurrent mutation cannot change it.
load_dotenv(_DOTENV_PATH)
_pin_dependency_cache_locations()
os.environ[OFFLINE_MODE_PROCESS_LATCH_ENV] = str(OFFLINE_MODE)
if OFFLINE_MODE:
    _publish_dependency_offline_switches(enforce_import_order=_first_establishment)

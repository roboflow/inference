"""Legacy environment compatibility for the inference server.

Maps the environment variable names and defaults of the legacy inference
server onto the names the new packages read. Every entry is justified in
``inference_server/docs/legacy_env.md``.
"""

import logging
import os

logger = logging.getLogger(__name__)

LEGACY_ENV_ALIASES = {
    "MAX_ACTIVE_MODELS": "INFERENCE_MAX_ACTIVE_MODELS",
    "PRELOAD_MODELS": "INFERENCE_PRELOAD_MODELS",
    "MEMORY_FREE_THRESHOLD": "INFERENCE_MEMORY_FREE_THRESHOLD",
    "API_KEY": "ROBOFLOW_API_KEY",
    "LEGACY_MMP_LOAD_WAIT_S": "INFERENCE_LEGACY_LOAD_TIMEOUT_S",
    "LEGACY_MMP_INFER_TIMEOUT_S": "INFERENCE_INFER_TIMEOUT_S",
}

LEGACY_FLAG_ALIASES = {
    "RUNS_ON_JETSON": "RUNNING_ON_JETSON",
}

LEGACY_DEFAULTS = {
    "ALLOW_URL_TO_NON_GLOBAL_ADDRESSES": "True",
    "MAX_IMAGE_URL_REDIRECTS": "30",
    "ALLOW_LOADING_IMAGES_FROM_LOCAL_FILESYSTEM": "True",
}

ROBOFLOW_SERVICE_URLS = {
    ("us", "prod"): {
        "api": "https://api.roboflow.com",
        "app": "https://app.roboflow.com",
    },
    ("us", "staging"): {
        "api": "https://api.roboflow.one",
        "app": "https://app.roboflow.one",
    },
    ("eu", "prod"): {
        "api": "https://api.roboflow.eu",
        "app": "https://app.roboflow.eu",
    },
    ("eu", "staging"): {
        "api": "https://api.roboflow-eu.one",
        "app": "https://app.roboflow-eu.one",
    },
}

DERIVED_URL_DEFAULTS = {
    "API_BASE_URL": "api",
    "BUILDER_ORIGIN": "app",
}

_DEFAULT_REGION = "us"
_US_PROD_PROJECT_NAME = "roboflow-platform"
_TRUE_FLAG_VALUES = frozenset({"true", "1", "yes", "y", "t"})


def _resolve_region() -> str:
    region = os.environ.get("ROBOFLOW_REGION", _DEFAULT_REGION).strip().lower()
    if region not in {known_region for known_region, _ in ROBOFLOW_SERVICE_URLS}:
        return _DEFAULT_REGION

    return region


def _resolve_environment() -> str:
    environment = os.environ.get("ROBOFLOW_ENVIRONMENT")
    if environment is not None:
        return "prod" if environment.strip().lower() == "prod" else "staging"

    project = os.environ.get("PROJECT", _US_PROD_PROJECT_NAME)
    if project != _US_PROD_PROJECT_NAME:
        return "staging"

    return "prod"


def _apply_derived_defaults() -> None:
    environment = _resolve_environment()
    if "ROBOFLOW_ENVIRONMENT" not in os.environ and "PROJECT" in os.environ:
        os.environ["ROBOFLOW_ENVIRONMENT"] = environment

    service_urls = ROBOFLOW_SERVICE_URLS[(_resolve_region(), environment)]
    for name, service in DERIVED_URL_DEFAULTS.items():
        os.environ.setdefault(name, service_urls[service])

    os.environ.setdefault("ROBOFLOW_API_HOST", os.environ["API_BASE_URL"])


def _apply_aliases() -> None:
    for legacy_name, new_name in LEGACY_ENV_ALIASES.items():
        if legacy_name in os.environ and new_name not in os.environ:
            os.environ[new_name] = os.environ[legacy_name]
            logger.warning("Legacy env %s mapped to %s", legacy_name, new_name)


def _apply_flag_aliases() -> None:
    for legacy_name, new_name in LEGACY_FLAG_ALIASES.items():
        if legacy_name not in os.environ or new_name in os.environ:
            continue

        if os.environ[legacy_name].strip().lower() in _TRUE_FLAG_VALUES:
            os.environ[new_name] = "True"
            logger.warning("Legacy env %s mapped to %s", legacy_name, new_name)


def apply_legacy_env() -> None:
    """Write legacy environment aliases and defaults into ``os.environ``.

    First the region-dependent legacy defaults are derived: ``API_BASE_URL``
    and ``BUILDER_ORIGIN`` from ``ROBOFLOW_REGION`` and ``PROJECT`` /
    ``ROBOFLOW_ENVIRONMENT`` when unset, ``ROBOFLOW_ENVIRONMENT`` from
    ``PROJECT`` when only the latter is set, and ``ROBOFLOW_API_HOST`` from
    the resulting ``API_BASE_URL`` when unset. Then every legacy name that is
    set while its new name is unset is copied to the new name with a warning;
    a flag alias propagates only a value that parses as true. Finally every
    legacy default is applied to a new name that is still unset. Explicit
    values always win. Runs before any configuration module is imported and
    is safe to call more than once.
    """
    _apply_derived_defaults()
    _apply_aliases()
    _apply_flag_aliases()

    for name, value in LEGACY_DEFAULTS.items():
        os.environ.setdefault(name, value)

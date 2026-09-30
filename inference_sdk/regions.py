"""Canonical Roboflow region / environment selection.

Single source of truth for resolving default Roboflow service URLs from the
two public switches:

* ``ROBOFLOW_REGION`` - ``us`` (default) or ``eu``
* ``ROBOFLOW_ENVIRONMENT`` - ``prod`` (default) or ``staging``

Explicit URL env variables (``API_BASE_URL``, ``ROBOFLOW_API_HOST``,
``RF_API_BASE_URL``, ...) always take precedence over values resolved here -
callers apply this module's output only as the default. The legacy
``PROJECT`` variable (``roboflow-platform`` vs anything else) is honoured as
an environment signal wherever it was honoured before, with
``ROBOFLOW_ENVIRONMENT`` taking precedence when both are set.
"""

import os
import warnings
from typing import Optional

PROD_ENVIRONMENT_NAME = "prod"
STAGING_ENVIRONMENT_NAME = "staging"
US_PROD_PROJECT_NAME = "roboflow-platform"

DEFAULT_REGION = "us"
DEFAULT_ENVIRONMENT = PROD_ENVIRONMENT_NAME

ROBOFLOW_SERVICE_URLS = {
    ("us", "prod"): {
        "api": "https://api.roboflow.com",
        "app": "https://app.roboflow.com",
        "serverless": "https://serverless.roboflow.com",
    },
    ("us", "staging"): {
        "api": "https://api.roboflow.one",
        "app": "https://app.roboflow.one",
        "serverless": "https://serverless.roboflow.one",
    },
    ("eu", "prod"): {
        "api": "https://api.roboflow.eu",
        "app": "https://app.roboflow.eu",
        "serverless": "https://serverless.roboflow.eu",
    },
    ("eu", "staging"): {
        "api": "https://api.roboflow-eu.one",
        "app": "https://app.roboflow-eu.one",
        "serverless": "https://serverless.roboflow-eu.one",
    },
}

SUPPORTED_REGIONS = sorted({region for region, _ in ROBOFLOW_SERVICE_URLS})


def get_roboflow_region() -> str:
    """Return the selected Roboflow region (``us`` or ``eu``).

    Unknown values warn on stderr and fall back to ``us`` - never raise at
    import time.
    """
    region = os.getenv("ROBOFLOW_REGION", DEFAULT_REGION).strip().lower()
    if region not in SUPPORTED_REGIONS:
        warnings.warn(
            f"Unknown ROBOFLOW_REGION {region!r} - falling back to "
            f"{DEFAULT_REGION!r}. Supported regions: {', '.join(SUPPORTED_REGIONS)}.",
        )
        return DEFAULT_REGION
    return region


def get_roboflow_environment(project: Optional[str] = None) -> str:
    """Return the selected Roboflow environment (``prod`` or ``staging``).

    ``ROBOFLOW_ENVIRONMENT`` wins when set (any value other than ``prod``
    selects staging, matching historical behaviour). Otherwise the legacy
    ``project`` signal decides (``roboflow-platform`` means prod), and with
    neither present the environment defaults to prod.
    """
    environment = os.getenv("ROBOFLOW_ENVIRONMENT")
    if environment is not None:
        if environment.strip().lower() == PROD_ENVIRONMENT_NAME:
            return PROD_ENVIRONMENT_NAME
        return STAGING_ENVIRONMENT_NAME
    if project is not None and project != US_PROD_PROJECT_NAME:
        return STAGING_ENVIRONMENT_NAME
    return DEFAULT_ENVIRONMENT


def resolve_roboflow_service_url(
    service: str,
    region: Optional[str] = None,
    environment: Optional[str] = None,
    project: Optional[str] = None,
) -> str:
    """Resolve the default URL for a Roboflow service in the selected region / environment."""
    if region is None:
        region = get_roboflow_region()
    if environment is None:
        environment = get_roboflow_environment(project=project)
    return ROBOFLOW_SERVICE_URLS[(region, environment)][service]

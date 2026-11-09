"""Availability of the Databricks blocks on the current server.

Two independent conditions decide whether a Databricks block is offered:

* the runtime policy allows it: ``DATABRICKS_SQL_IMAGE_SOURCE_ENABLED`` is the
  explicit rollout switch. Unset, the host resolves it to ``False`` on the
  Roboflow hosted platform (``GCP_SERVERLESS`` / ``LAMBDA``) and ``True``
  elsewhere; hosted workers set it explicitly. The value is read when the
  process starts, so changing it is a redeployment rather than a live switch.
* the ``databricks-sql-connector`` package is importable (the
  ``roboflow-workflows[databricks]`` extra). Installing the extra and enabling
  the block are separate decisions.

The loader consults this module so that ``/workflows/blocks/describe`` and
execution agree: an unavailable block is neither advertised nor runnable, and a
Workflow that references it - saved or submitted directly - is rejected at
compile time with the reason, before any connection is attempted.
"""

import importlib.util
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Optional

from roboflow_workflows.environment import (
    DATABRICKS_SQL_IMAGE_SOURCE_ENABLED,
    GCP_SERVERLESS,
    LAMBDA,
)

DATABRICKS_SQL_IMAGE_SOURCE_TYPE = "roboflow_core/databricks_sql_image_source@v1"
SUPPORTED_CONNECTOR_VERSION = "4.8.0"
CONNECTOR_INSTALL_HINT = (
    "install the `roboflow-workflows[databricks]` extra (pip install "
    '"roboflow-workflows[databricks]") or `databricks-sql-connector` next to the '
    "Workflows runtime"
)
DISABLED_HINT = (
    "the block is disabled on this server (DATABRICKS_SQL_IMAGE_SOURCE_ENABLED is "
    "false, which is the default on the Roboflow hosted platform); an operator "
    "enables it with DATABRICKS_SQL_IMAGE_SOURCE_ENABLED=true and a restart"
)


def connector_available() -> bool:
    """Return True when ``databricks.sql`` is imported or importable."""
    if sys.modules.get("databricks.sql") is not None:
        return True
    try:
        return importlib.util.find_spec("databricks.sql") is not None
    except (ImportError, ValueError):
        return False


def running_on_hosted_platform() -> bool:
    """Return True on the Roboflow hosted platform (serverless / lambda)."""
    return bool(GCP_SERVERLESS or LAMBDA)


def block_enabled() -> bool:
    """Return the resolved value of ``DATABRICKS_SQL_IMAGE_SOURCE_ENABLED``."""
    return bool(DATABRICKS_SQL_IMAGE_SOURCE_ENABLED)


def databricks_unavailability_reason() -> Optional[str]:
    """Explain why the Databricks block is not offered here, or ``None``.

    Returns:
        A human-readable reason when the block must not be registered, or
        ``None`` when both the runtime policy and the connector allow it.
    """
    if not block_enabled():
        return DISABLED_HINT

    if not connector_available():
        result = (
            "the `databricks-sql-connector` package is not installed; "
            + CONNECTOR_INSTALL_HINT
        )

        return result

    try:
        installed_version = version("databricks-sql-connector")
    except PackageNotFoundError:
        result = (
            "the Databricks connector distribution is missing; "
            + CONNECTOR_INSTALL_HINT
        )

        return result
    if installed_version != SUPPORTED_CONNECTOR_VERSION:
        result = (
            f"Databricks connector {installed_version} is unsupported; "
            f"this transport requires {SUPPORTED_CONNECTOR_VERSION}. "
            + CONNECTOR_INSTALL_HINT
        )

        return result

    return None


def databricks_block_available() -> bool:
    """Return True when the Databricks block may be registered and run."""
    return databricks_unavailability_reason() is None

"""Secure defaults for the self-hosted inference server.

The library defaults in `inference.core.env` also serve in-process use
(`get_model(...).infer("image.jpg")`, `InferencePipeline`), where the caller
already runs code on the host. A server exposes the same features to every
client that can reach its port, including any web page open in the browser of
the machine that runs it. The server entry points call
`apply_self_hosted_server_defaults()` before anything imports
`inference.core.env`, so these defaults reach only the server.

Each default is applied with `os.environ.setdefault`: an explicit environment
variable always wins. Roboflow-hosted runtimes (serverless, Lambda and
dedicated deployments) bring their own authorization policy and keep their
current behavior.

A server without workspace API key checks should only take connections from
this machine. On any other address it refuses to start while a setting that
lets requests run code on the host, or read its files or secrets, is on.

This module must not import `inference.core`, which reads the environment at
import time.
"""

import ipaddress
import multiprocessing
import os
import signal
import sys
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

from dotenv import dotenv_values, load_dotenv

HOSTED_RUNTIME_ENV_NAMES = ("LAMBDA", "GCP_SERVERLESS")

SELF_HOSTED_SERVER_DEFAULTS = {
    # Only API keys of the workspace that the server belongs to may call it.
    "REQUIRE_API_KEY_FOR_LOCAL_DEPLOYMENT": "True",
    # Custom Python blocks run arbitrary code in the server process.
    "ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS": "False",
    # Request payloads may not read files or environment variables of the host,
    # or write files to it.
    "ALLOW_LOADING_IMAGES_FROM_LOCAL_FILESYSTEM": "False",
    "ALLOW_WORKFLOW_BLOCKS_ACCESSING_ENVIRONMENTAL_VARIABLES": "False",
    "ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE": "False",
    # Image URLs may not reach loopback, link-local or private addresses,
    # directly or through a redirect.
    "ALLOW_URL_TO_NON_GLOBAL_ADDRESSES": "False",
    "VALIDATE_IMAGE_URL_REDIRECTS": "True",
    # Webhook sinks may not reach loopback, link-local or private addresses.
    "ALLOW_WEBHOOK_WORKFLOWS_SINK_TO_NON_GLOBAL_ADDRESSES": "False",
}


# Settings that let a request run code on the host or read its files or
# secrets. Without workspace API key checks, only a loopback server may use them.
HOST_ACCESS_SETTINGS = (
    "ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS",
    "ALLOW_LOADING_IMAGES_FROM_LOCAL_FILESYSTEM",
    "ALLOW_WORKFLOW_BLOCKS_ACCESSING_ENVIRONMENTAL_VARIABLES",
    "ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE",
    "ALLOW_UNSAFE_GSTREAMER_PIPELINES",
    # Unpickles request payloads.
    "ALLOW_NUMPY_INPUT",
    # Load model packages that may run code, or straight from local paths.
    "ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES",
    "ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES",
)

# Set by apply_self_hosted_server_defaults(): True when only this machine can
# connect to the server. The HTTP API serves callers without workspace API key
# checks only then.
LOOPBACK_ONLY = False


def _is_true(name: str) -> bool:
    # inference.core.env accepts only "true" (any case) as true; stripping
    # makes this check slightly looser, never stricter.
    return os.getenv(name, "").strip().lower() == "true"


def _has_list(name: str) -> bool:
    return any(item.strip() for item in os.getenv(name, "").split(","))


def is_loopback_name(host: Optional[str]) -> bool:
    """Tell whether a host name or address refers to this machine only.

    Args:
        host: Host name or IP address, optionally in brackets (IPv6).

    Returns:
        True for "localhost" and loopback addresses, False otherwise.
    """
    if not host:
        return False
    host = host.strip().strip("[]").lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def url_hostname(value: str) -> Optional[str]:
    """Extract the hostname of a URL or of a Host header value.

    Args:
        value: A URL, or a Host header value such as "example.com:9001".

    Returns:
        The lowercased hostname, or None if the value has none.
    """
    try:
        return urlsplit(value if "//" in value else "//" + value).hostname
    except ValueError:
        return None


def _is_loopback_only() -> bool:
    # PUBLISHED_HOST is the host address that a container port is published on
    # (`inference server start` sets it). Without it, HOST is the bind address.
    # Launchers such as run_uvicorn.sh pick the bind address before Python reads
    # ./.env, but importing inference (inference_models) has already loaded
    # ./.env into the environment, so a value set there cannot be trusted.
    from_dotenv = dotenv_values(Path.cwd() / ".env")
    if "HOST" in from_dotenv or "PUBLISHED_HOST" in from_dotenv:
        return False
    return is_loopback_name(
        os.getenv("PUBLISHED_HOST") or os.getenv("HOST") or "0.0.0.0"
    )


def apply_self_hosted_server_defaults() -> None:
    """Apply the self-hosted server defaults to the process environment.

    Call it before anything imports `inference.core.env`. Explicit environment
    variables and ./.env entries win over the defaults. Sets `LOOPBACK_ONLY`.

    Returns:
        None.

    Raises:
        SystemExit: The server takes connections from other machines without
            workspace API key checks while a setting in `HOST_ACCESS_SETTINGS`
            is on.
    """
    global LOOPBACK_ONLY
    loopback_only = _is_loopback_only()
    # inference.core.env loads ./.env without overriding the environment. Load it
    # first so .env settings are checked here and win over these defaults.
    load_dotenv(Path.cwd() / ".env")
    if any(_is_true(name) for name in HOSTED_RUNTIME_ENV_NAMES) or os.getenv(
        "DEDICATED_DEPLOYMENT_WORKSPACE_URL"
    ):
        return None
    LOOPBACK_ONLY = loopback_only
    has_whitelist = _has_list("WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT")
    has_workspace = bool(
        os.getenv("ROBOFLOW_API_KEY") or os.getenv("API_KEY") or has_whitelist
    )
    for name, value in SELF_HOSTED_SERVER_DEFAULTS.items():
        if name == "REQUIRE_API_KEY_FOR_LOCAL_DEPLOYMENT" and (
            # API keys cannot be mapped to workspaces without API connectivity.
            _is_true("OFFLINE_MODE")
            # Nothing to check keys against; only local callers reach the server.
            or (loopback_only and not has_workspace)
        ):
            continue
        os.environ.setdefault(name, value)
    if (
        loopback_only
        or _is_true("REQUIRE_API_KEY_FOR_LOCAL_DEPLOYMENT")
        or has_whitelist
    ):
        return None
    enabled = [name for name in HOST_ACCESS_SETTINGS if _is_true(name)]
    if enabled:
        # A uvicorn (0.30+) or gunicorn worker that exits is started again, so
        # stop the supervisor too; otherwise the workers would restart forever.
        if multiprocessing.parent_process() is not None:
            os.kill(os.getppid(), signal.SIGTERM)
        raise SystemExit(
            f"Refusing to start: {', '.join(enabled)} is on, but the server takes "
            "connections from other machines without workspace API key checks, so "
            "anyone who can reach it could use it. Either make it reachable from "
            "this machine only (`inference server start` does so by default; with "
            "docker run, publish the port as -p 127.0.0.1:9001:9001 and set "
            "PUBLISHED_HOST=127.0.0.1), or set ROBOFLOW_API_KEY or "
            "WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT and keep "
            "REQUIRE_API_KEY_FOR_LOCAL_DEPLOYMENT on."
        )
    # The logger is not configured yet, so write straight to stderr.
    print(
        "SECURITY: the server takes connections from other machines without "
        "workspace API key checks, so anyone who can reach its port can use "
        "every route. Make it reachable from this machine only (publish the "
        "port as -p 127.0.0.1:9001:9001 and set PUBLISHED_HOST=127.0.0.1), or "
        "set ROBOFLOW_API_KEY or WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT and "
        "keep REQUIRE_API_KEY_FOR_LOCAL_DEPLOYMENT on.",
        file=sys.stderr,
    )

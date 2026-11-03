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

This module must not import `inference.core`, which reads the environment at
import time.
"""

import os

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
}


def _is_true(name: str) -> bool:
    return os.getenv(name, "").strip().lower() == "true"


def apply_self_hosted_server_defaults() -> None:
    if any(_is_true(name) for name in HOSTED_RUNTIME_ENV_NAMES) or os.getenv(
        "DEDICATED_DEPLOYMENT_WORKSPACE_URL"
    ):
        return None
    for name, value in SELF_HOSTED_SERVER_DEFAULTS.items():
        if name == "REQUIRE_API_KEY_FOR_LOCAL_DEPLOYMENT" and _is_true("OFFLINE_MODE"):
            # API keys cannot be mapped to workspaces without API connectivity.
            continue
        os.environ.setdefault(name, value)

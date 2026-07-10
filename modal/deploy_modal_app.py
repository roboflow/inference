#!/usr/bin/env python3
"""
Deploy script for the Modal web endpoint.

This script deploys the Modal web endpoint for running custom Python blocks.
It requires the 'modal' package to be installed.

Usage:
    # Install modal first
    pip install modal

    # Set credentials (choose one method)
    # Option 1: Set environment variables
    export MODAL_TOKEN_ID="your_token_id"
    export MODAL_TOKEN_SECRET="your_token_secret"

    # Option 2: Use credentials from ~/.modal.toml (automatic fallback)
    # You can set this up by running `modal setup`.

    # Then deploy
    python modal/deploy_modal_app.py
"""

import os
import sys
from utils import (
    initialize_modal,
    prepare_deployment,
    validate_deployment_prerequisites,
)

from inference.core.env import WEBEXEC_INFERENCE_VERSION, WEBEXEC_MODAL_APP_NAME
from inference.core.version import __version__


prepare_deployment()

# Initialize Modal and get credentials
token_id, token_secret = initialize_modal()

import modal

# Import the app from the modal directory
try:
    from modal_app import app

    MODAL_INSTALLED = True
except ImportError as e:
    print(f"Error: Could not import modal_app: {e}")
    print("\nPlease make sure you're running this script from the correct directory.")
    sys.exit(1)

inference_version = WEBEXEC_INFERENCE_VERSION
if not inference_version:
    try:
        from inference.core.version import __version__

        inference_version = __version__
    except ImportError:
        inference_version = "latest"

validate_deployment_prerequisites(app, MODAL_INSTALLED)

# Deploy the app
print("=" * 60)
print("Deploying Modal Web Endpoint for Custom Python Blocks")
print("=" * 60)
print(f"App name: {app.name}")
print(f"Inference version: {inference_version}")
print("\nDeploying...")

try:
    with modal.enable_output():
        deployed_app = app.deploy(tag=inference_version)

    print("\n✅ Deployment successful!")
    print(f"\nDeployed app details:")
    print(f"  Name: {deployed_app.name}")
    print(f"  App ID: {deployed_app.app_id}")

    print(f"  Version: {inference_version}")
    # Try to get the actual URLs from the deployed app
    print("\n📡 Web Endpoint URLs:")
    try:
        cls = modal.Cls.from_name(
            app_name=WEBEXEC_MODAL_APP_NAME,
            name="Executor",
        )
        # Create an instance to get the method
        instance = cls(workspace_id="test")
        # Get the execute_block method's web URL
        http_base_url = None
        ws_base_url = None
        if hasattr(instance, "execute_block") and hasattr(
            instance.execute_block, "get_web_url"
        ):
            actual_url = instance.execute_block.get_web_url()
            if actual_url:
                # Remove query params to get base URL
                http_base_url = actual_url.split("?")[0]
                print(f"  HTTP:      {http_base_url}")
                print(
                    "             Set MODAL_WEB_ENDPOINT_URL for HTTP transport "
                    "and Modal validation."
                )
        if hasattr(instance, "wsapp") and hasattr(instance.wsapp, "get_web_url"):
            actual_ws_url = instance.wsapp.get_web_url()
            if actual_ws_url:
                ws_base_url = actual_ws_url.split("?")[0]
                print(f"  WebSocket: {ws_base_url}")
                print("             Set MODAL_WS_ENDPOINT_URL for websocket execution.")

        if not http_base_url and not ws_base_url:
            raise Exception("Could not get web URL from methods")

        print(
            "\nNote: keep both Modal methods deployed. The websocket transport uses "
            "wsapp for execution, and code validation still uses the HTTP "
            "execute-block endpoint."
        )
        print(
            "      Custom endpoint deployments should set both "
            "MODAL_WEB_ENDPOINT_URL and MODAL_WS_ENDPOINT_URL."
        )

        if http_base_url:
            print("\n📝 Example test command:")
            print(f"""
curl -X POST "{http_base_url}?workspace_id=test" \\
  -H "Content-Type: application/json" \\
  -H "Modal-Key: {token_id}" \\
  -H "Modal-Secret: {token_secret}" \\
  -d '{{"code_str": "def run(): return {{\\"test\\": \\"ok\\"}}", "run_function_name": "run", "inputs_json": "{{}}"}}'
""")
    except Exception as e:
        print(f"  Could not retrieve actual URL dynamically: {e}")
        print(
            "  The URL should be visible in the Modal dashboard at https://modal.com/apps"
        )
        print(
            f"  Expected format: https://roboflow--{app.name}-executor-{{truncated}}.modal.run"
        )
        print("\n  Keep both execute-block (HTTP) and wsapp (websocket) deployed.")
        print(
            "  Set MODAL_WEB_ENDPOINT_URL for validation/HTTP and "
            "MODAL_WS_ENDPOINT_URL for websocket execution."
        )

    print("\n✅ Ready for production use!")

except Exception as e:
    print(f"\n❌ Deployment failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

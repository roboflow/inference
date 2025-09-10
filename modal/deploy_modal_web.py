#!/usr/bin/env python3
"""
Deploy script for the Modal web endpoint - requires inference package installed locally.

Usage:
    # Install inference first
    pip install inference
    
    # Then deploy
    python deploy_modal_web.py
    
Environment variables required:
    MODAL_TOKEN_ID - Modal authentication token ID
    MODAL_TOKEN_SECRET - Modal authentication token secret
    
If you can't install inference locally, use deploy_modal_web_standalone.py instead.
"""

import os
import sys

# Add the inference package to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Check Modal credentials
if not os.environ.get("MODAL_TOKEN_ID"):
    print("Error: MODAL_TOKEN_ID environment variable not set")
    print("Please set: export MODAL_TOKEN_ID='your-token-id'")
    sys.exit(1)
    
if not os.environ.get("MODAL_TOKEN_SECRET"):
    print("Error: MODAL_TOKEN_SECRET environment variable not set")
    print("Please set: export MODAL_TOKEN_SECRET='your-token-secret'")
    sys.exit(1)

try:
    import modal
except ImportError:
    print("Error: Modal is not installed. Run: pip install modal")
    sys.exit(1)

# Try to import the app from inference
try:
    from inference.core.workflows.execution_engine.v1.dynamic_blocks.modal_executor_web import (
        app,
        MODAL_INSTALLED,
    )
except ImportError as e:
    print(f"Error: Could not import inference package: {e}")
    print("\nYou have two options:")
    print("1. Install inference: pip install inference")
    print("2. Use the standalone script: python deploy_modal_web_standalone.py")
    sys.exit(1)

if not MODAL_INSTALLED:
    print("Error: Modal is not installed properly")
    sys.exit(1)

if not app:
    print("Error: Modal app not initialized properly")
    sys.exit(1)

# Deploy the app
print("=" * 60)
print("Deploying Modal Web Endpoint for Custom Python Blocks")
print("=" * 60)
print(f"App name: {app.name}")
print("\nDeploying...")

try:
    with modal.enable_output():
        deployed_app = app.deploy()
    
    print("\n✅ Deployment successful!")
    print(f"\nDeployed app details:")
    print(f"  Name: {deployed_app.name}")
    print(f"  App ID: {deployed_app.app_id}")
    
    print("\n📡 Web Endpoint URL:")
    print("  The URL will be in the format:")
    print("  https://roboflow--inference-custom-blocks-web--customblockexecutor-execute-block.modal.run")
    print("  (Note: Modal may truncate long URLs)")
    
    print("\n📝 Example test command:")
    print(f"""
curl -X POST "https://roboflow--inference-custom-blocks-web-customblockexecuto-4874a9.modal.run?workspace_id=test" \\
  -H "Content-Type: application/json" \\
  -d '{{"code_str": "def run(): return {{\\"test\\": \\"ok\\"}}", "run_function_name": "run", "inputs_json": "{{}}"}}'
""")
    
    print("\n✅ Ready for production use!")
    
except Exception as e:
    print(f"\n❌ Deployment failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

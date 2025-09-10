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
    
    # Try to get the actual URL from the deployed app
    print("\n📡 Web Endpoint URL:")
    try:
        # Get the CustomBlockExecutor class
        cls = modal.Cls.from_name("inference-custom-blocks-web", "CustomBlockExecutor")
        # Create an instance to get the method
        instance = cls(workspace_id="test")
        # Get the execute_block method's web URL
        if hasattr(instance, 'execute_block') and hasattr(instance.execute_block, 'get_web_url'):
            actual_url = instance.execute_block.get_web_url()
            if actual_url:
                # Remove query params to get base URL
                base_url = actual_url.split('?')[0]
                print(f"  {base_url}")
                
                print("\n📝 Example test command:")
                print(f"""
curl -X POST "{base_url}?workspace_id=test" \\
  -H "Content-Type: application/json" \\
  -d '{{"code_str": "def run(): return {{\\"test\\": \\"ok\\"}}", "run_function_name": "run", "inputs_json": "{{}}"}}'
""")
            else:
                raise Exception("Could not get web URL from method")
        else:
            raise Exception("Method does not have get_web_url")
    except Exception as e:
        print(f"  Could not retrieve actual URL dynamically: {e}")
        print("  The URL should be visible in the Modal dashboard at https://modal.com/apps")
        print("  Expected format: https://roboflow--inference-custom-blocks-web-{truncated}.modal.run")
        print("\n  Set the MODAL_WEB_ENDPOINT_URL environment variable with the actual URL.")
    
    print("\n✅ Ready for production use!")
    
except Exception as e:
    print(f"\n❌ Deployment failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

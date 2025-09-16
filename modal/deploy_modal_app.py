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

import sys
from utils import initialize_modal, prepare_deployment, validate_deployment_prerequisites

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

validate_deployment_prerequisites(app, MODAL_INSTALLED)

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
        # Get the Executor class
        cls = modal.Cls.from_name("webexec", "Executor")
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
  -H "Modal-Key: {token_id}" \\
  -H "Modal-Secret: {token_secret}" \\
  -d '{{"code_str": "def run(): return {{\\"test\\": \\"ok\\"}}", "run_function_name": "run", "inputs_json": "{{}}"}}'
""")
            else:
                raise Exception("Could not get web URL from method")
        else:
            raise Exception("Method does not have get_web_url")
    except Exception as e:
        print(f"  Could not retrieve actual URL dynamically: {e}")
        print("  The URL should be visible in the Modal dashboard at https://modal.com/apps")
        print("  Expected format: https://roboflow--webexec-executor-{truncated}.modal.run")
        print("\n  Set the MODAL_WEB_ENDPOINT_URL environment variable with the actual URL.")
    
    print("\n✅ Ready for production use!")
    
except Exception as e:
    print(f"\n❌ Deployment failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

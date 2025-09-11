#!/usr/bin/env python3
"""
Deploy the Modal App for Custom Python Blocks execution.

This script deploys the Modal App with parameterized executors
for running custom Python blocks in sandboxed environments.

Usage:
    # Option 1: Set environment variables
    export MODAL_TOKEN_ID="your_token_id"
    export MODAL_TOKEN_SECRET="your_token_secret"
    
    # Option 2: Use credentials from ~/.modal.toml (automatic fallback)
    
    # Deploy the app
    python modal/deploy_modal_app.py
"""

import sys
from utils import initialize_modal, prepare_deployment, validate_deployment_prerequisites

prepare_deployment()


def main():
    """Deploy the Modal App."""
    print("=" * 60)
    print("Deploying Modal App for Custom Python Blocks")
    print("=" * 60)
    
    # Initialize Modal and get credentials
    token_id, token_secret = initialize_modal()
    
    # Import after setting credentials
    from inference.core.workflows.execution_engine.v1.dynamic_blocks.modal_executor import (
        app,
        MODAL_INSTALLED,
    )
    
    validate_deployment_prerequisites(app, MODAL_INSTALLED)
    
    print("\nDeploying app: inference-custom-blocks")
    
    try:
        # Deploy the app
        app.deploy(
            name="inference-custom-blocks",
            tag="v1-with-optimizations"
        )
        
        print("\n✅ Successfully deployed Modal App with optimizations!")
        print("\nYou can now:")
        print("1. Test with: python modal/test_modal_blocks.py")
        print("2. Use in workflows with WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE=modal")
        print("3. View deployment at: https://modal.com/apps")
        
    except Exception as e:
        print(f"\n❌ Deployment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

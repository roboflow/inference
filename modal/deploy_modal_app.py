#!/usr/bin/env python3
"""
Deploy the Modal App for Custom Python Blocks execution.

This script deploys the Modal App with parameterized executors
for running custom Python blocks in sandboxed environments.

Usage:
    # Set environment variables first
    export MODAL_TOKEN_ID="your_token_id"
    export MODAL_TOKEN_SECRET="your_token_secret"
    
    # Deploy the app
    python modal/deploy_modal_app.py
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to import inference modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import modal
from inference.core.workflows.execution_engine.v1.dynamic_blocks.modal_executor import (
    app, 
    CustomBlockExecutor
)


def main():
    """Deploy the Modal App."""
    print("=" * 60)
    print("Deploying Modal App for Custom Python Blocks")
    print("=" * 60)
    
    # Check environment
    if not os.environ.get("MODAL_TOKEN_ID"):
        print("ERROR: MODAL_TOKEN_ID environment variable not set")
        print("Please set Modal credentials before deploying")
        sys.exit(1)
    
    print("\nDeploying app: inference-custom-blocks")
    print("This will create parameterized executors for workspace isolation")
    
    try:
        # Deploy the app
        app.deploy(
            name="inference-custom-blocks",
            tag="v1"
        )
        
        print("\n✅ Successfully deployed Modal App!")
        print("\nYou can now:")
        print("1. Test with: python modal/test_modal_blocks.py")
        print("2. Use in workflows with WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE=modal")
        print("3. View deployment at: https://modal.com/apps")
        
    except Exception as e:
        print(f"\n❌ Deployment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

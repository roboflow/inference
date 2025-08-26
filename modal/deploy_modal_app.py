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

import os
import sys
from pathlib import Path
import configparser

# Add parent directory to path to import inference modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Check if modal is installed
try:
    import modal
except ImportError:
    print("ERROR: Modal is not installed")
    print("Please install with: pip install modal")
    sys.exit(1)


def load_modal_credentials():
    """Load Modal credentials from environment or ~/.modal.toml file.
    
    Returns:
        tuple: (token_id, token_secret) or (None, None) if not found
    """
    # First check environment variables
    token_id = os.environ.get("MODAL_TOKEN_ID")
    token_secret = os.environ.get("MODAL_TOKEN_SECRET")
    
    if token_id and token_secret:
        print("✓ Using Modal credentials from environment variables")
        return token_id, token_secret
    
    # Try to read from ~/.modal.toml
    modal_toml_path = Path.home() / ".modal.toml"
    if modal_toml_path.exists():
        try:
            config = configparser.ConfigParser()
            config.read(modal_toml_path)
            
            # Modal uses the [default] section in .modal.toml
            if "default" in config:
                token_id = config.get("default", "token_id", fallback=None)
                token_secret = config.get("default", "token_secret", fallback=None)
                
                if token_id and token_secret:
                    print(f"✓ Using Modal credentials from {modal_toml_path}")
                    # Set environment variables for the Modal client
                    os.environ["MODAL_TOKEN_ID"] = token_id
                    os.environ["MODAL_TOKEN_SECRET"] = token_secret
                    return token_id, token_secret
        except Exception as e:
            print(f"Warning: Could not parse {modal_toml_path}: {e}")
    
    return None, None


def main():
    """Deploy the Modal App."""
    print("=" * 60)
    print("Deploying Modal App for Custom Python Blocks")
    print("=" * 60)
    
    # Load credentials
    token_id, token_secret = load_modal_credentials()
    
    if not token_id or not token_secret:
        print("\nERROR: Modal credentials not found")
        print("\nPlease provide credentials using one of these methods:")
        print("1. Set environment variables:")
        print("   export MODAL_TOKEN_ID='your_token_id'")
        print("   export MODAL_TOKEN_SECRET='your_token_secret'")
        print("\n2. Run 'modal setup' to create ~/.modal.toml")
        print("\n3. Create ~/.modal.toml manually with:")
        print("   [default]")
        print("   token_id = 'your_token_id'")
        print("   token_secret = 'your_token_secret'")
        sys.exit(1)
    
    # Import after setting credentials
    from inference.core.workflows.execution_engine.v1.dynamic_blocks.modal_executor import (
        app, 
        CustomBlockExecutor,
        MODAL_INSTALLED,
        MODAL_AVAILABLE
    )
    
    if not MODAL_INSTALLED:
        print("ERROR: Modal is not installed")
        print("Please install with: pip install modal")
        sys.exit(1)
    
    if not app:
        print("ERROR: Modal app could not be created")
        print("Please check your Modal installation and credentials")
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

import os
import sys
from pathlib import Path
import importlib.util


def prepare_deployment():
    """Prepare the environment for deployment by modifying sys.path."""
    # Add parent directory to path to import inference modules
    sys.path.insert(0, str(Path(__file__).parent.parent))


def validate_deployment_prerequisites(app, modal_installed):
    """Validate that the Modal app and installation are ready for deployment."""
    if not modal_installed:
        print("ERROR: Modal is not installed properly.")
        print("Please install with: pip install modal")
        sys.exit(1)

    if not app:
        print("ERROR: Modal app could not be created or initialized.")
        print("Please check your Modal installation and credentials.")
        sys.exit(1)


def initialize_modal():
    """Check for Modal installation and credentials, then return credentials.

    This function will exit the script with an error message if Modal is not
    installed or if credentials are not found.

    Returns:
        tuple: (token_id, token_secret)
    """
    # Check if modal is installed
    if not importlib.util.find_spec("modal"):
        print("ERROR: Modal is not installed")
        print("Please install with: pip install modal")
        sys.exit(1)

    # Load credentials
    token_id, token_secret = load_modal_credentials()

    if not token_id or not token_secret:
        print("\nERROR: Modal credentials not found")
        print("\nPlease provide credentials using one of these methods:")
        print("1. Set environment variables:")
        print("   export MODAL_TOKEN_ID='your_token_id'")
        print("   export MODAL_TOKEN_SECRET='your_token_secret'")
        print("\n2. Run 'modal setup' to create ~/.modal.toml")
        print("\n3. Create ~/.modal.toml manually with a profile like:")
        print("   [default]  # or [your-profile-name]")
        print("   token_id = \"your_token_id\"")
        print("   token_secret = \"your_token_secret\"")
        print("   active = true  # optional, marks this as the active profile")
        sys.exit(1)
    
    return token_id, token_secret

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
            # Try using toml if available, otherwise parse manually
            try:
                import toml
                config = toml.load(modal_toml_path)
                
                # Find the active profile or use the first available profile
                active_profile = None
                for section, values in config.items():
                    if isinstance(values, dict) and values.get("active", False):
                        active_profile = section
                        break
                
                # Use active profile, or fall back to first available
                if not active_profile and config:
                    active_profile = list(config.keys())[0]
                
                if active_profile and active_profile in config:
                    profile = config[active_profile]
                    token_id = profile.get("token_id")
                    token_secret = profile.get("token_secret")
                    
                    if token_id and token_secret:
                        print(f"✓ Using Modal credentials from {modal_toml_path} (profile: {active_profile})")
                        os.environ["MODAL_TOKEN_ID"] = token_id
                        os.environ["MODAL_TOKEN_SECRET"] = token_secret
                        return token_id, token_secret
                        
            except ImportError:
                # Fallback: Parse TOML file manually for basic key-value pairs
                import re
                content = modal_toml_path.read_text()
                
                # Find sections and their content
                sections = re.findall(r'\[([^\]]+)\](.*?)(?=\[|$)', content, re.DOTALL)
                
                active_profile = None
                profiles = {}
                
                for section_name, section_content in sections:
                    # Parse key-value pairs in section
                    profile_data = {}
                    
                    # Match lines like: key = "value" or key = 'value'
                    for match in re.finditer(r'^(\w+)\s*=\s*["\']([^"\']+)["\']', section_content, re.MULTILINE):
                        key, value = match.groups()
                        profile_data[key] = value
                    
                    # Also match boolean values
                    for match in re.finditer(r'^(\w+)\s*=\s*(true|false)', section_content, re.MULTILINE):
                        key, value = match.groups()
                        profile_data[key] = value == 'true'
                    
                    profiles[section_name] = profile_data
                    
                    # Check if this is the active profile
                    if profile_data.get('active', False):
                        active_profile = section_name
                
                # Use active profile or first available
                if not active_profile and profiles:
                    active_profile = list(profiles.keys())[0]
                
                if active_profile and active_profile in profiles:
                    profile = profiles[active_profile]
                    token_id = profile.get("token_id")
                    token_secret = profile.get("token_secret")
                    
                    if token_id and token_secret:
                        print(f"✓ Using Modal credentials from {modal_toml_path} (profile: {active_profile})")
                        os.environ["MODAL_TOKEN_ID"] = token_id
                        os.environ["MODAL_TOKEN_SECRET"] = token_secret
                        return token_id, token_secret
                    
        except Exception as e:
            print(f"Warning: Could not parse {modal_toml_path}: {e}")
    
    return None, None

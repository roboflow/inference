import os
import sys

def define_env(env):
    """Hook function to define macros for MkDocs."""
    
    @env.macro
    def get_version():
        """Read version from inference/core/version.py"""
        # Get the path to the root of the repository
        current_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.join(current_dir, '..', '..')
        version_file_path = os.path.join(repo_root, 'inference', 'core', 'version.py')
        
        try:
            # Read the version file
            with open(version_file_path, 'r') as f:
                content = f.read()
            
            # Extract version using simple string parsing
            for line in content.split('\n'):
                if line.strip().startswith('__version__'):
                    # Extract version from: __version__ = "0.51.10"
                    version = line.split('=')[1].strip().strip('"').strip("'")
                    return version
            
            return "unknown"
        except Exception as e:
            print(f"Warning: Could not read version from {version_file_path}: {e}")
            return "unknown"
    
    # Make VERSION available globally to all templates
    env.variables['VERSION'] = get_version()
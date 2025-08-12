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
            # Execute the version.py file and extract __version__
            namespace = {}
            with open(version_file_path, 'r') as f:
                exec(f.read(), namespace)   
            return namespace['__version__']
        except Exception as e:
            print(f"Warning: Could not read version from {version_file_path}: {e}")
            return "unknown"
    
    # Make VERSION available globally to all templates
    env.variables['VERSION'] = get_version()
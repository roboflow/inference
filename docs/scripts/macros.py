import sys
from pathlib import Path

def define_env(env):
    """Hook function to define macros for MkDocs."""
    
    @env.macro
    def get_version():
        """Read version from inference/core/version.py"""
        # Find the root of the repository by iterating up parent directories
        current_path = Path(__file__).resolve()
        for parent in current_path.parents:
            # Check if this directory contains the 'inference' subdirectory
            if (parent / 'inference').is_dir():
                repo_root = parent
                break
        else:
            raise FileNotFoundError("Could not find repository root with 'inference' directory")
        
        version_file_path = repo_root.joinpath('inference', 'core', 'version.py')
        
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
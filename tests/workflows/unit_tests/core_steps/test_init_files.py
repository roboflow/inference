import os
from pathlib import Path
import pytest


def test_init_files_present():
    """Test that all directories under core_steps have __init__.py files."""
    # Get project root by finding the directory containing setup.py
    current_path = Path(__file__).resolve()
    project_root = next(p for p in current_path.parents if (p / 'setup.py').exists())
    
    missing_init_dirs = []
    
    for root, dirs, files in os.walk(str(project_root / "inference" / "core" / "workflows" / "core_steps")):
        # Skip __pycache__ directories
        if "__pycache__" in root:
            continue
            
        if os.path.basename(root) != "core_steps":  # Skip the root core_steps directory
            if "__init__.py" not in files:
                # Get the path relative to project root for better readability
                rel_path = os.path.relpath(root, project_root)
                missing_init_dirs.append(rel_path)
    
    assert not missing_init_dirs, f"The following directories are missing __init__.py files:\n{chr(10).join(missing_init_dirs)}"

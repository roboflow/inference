import os
from pathlib import Path

import pytest


def test_init_files_present_in_inference_core_workflows_core_steps():
    current_path = Path(__file__).resolve()
    project_root = next(p for p in current_path.parents if (p / "setup.py").exists())

    missing_init_dirs = []

    for root, dirs, files in os.walk(
        str(project_root / "inference" / "core" / "workflows" / "core_steps")
    ):
        if "__pycache__" in root:
            continue

        if os.path.basename(root) != "core_steps":
            if "__init__.py" not in files:
                rel_path = os.path.relpath(root, project_root)
                missing_init_dirs.append(rel_path)

    assert (
        not missing_init_dirs
    ), f"The following directories are missing __init__.py files:\n{chr(10).join(missing_init_dirs)}"


def test_init_files_present_in_inference_enterprise_workflows_core_steps():
    current_path = Path(__file__).resolve()
    project_root = next(p for p in current_path.parents if (p / "setup.py").exists())

    missing_init_dirs = []

    for root, dirs, files in os.walk(
        str(project_root / "inference" / "enterprise" / "workflows" / "core_steps")
    ):
        if "__pycache__" in root:
            continue

        if os.path.basename(root) != "core_steps":
            if "__init__.py" not in files:
                rel_path = os.path.relpath(root, project_root)
                missing_init_dirs.append(rel_path)

    assert (
        not missing_init_dirs
    ), f"The following directories are missing __init__.py files:\n{chr(10).join(missing_init_dirs)}"

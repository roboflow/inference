import os
from pathlib import Path

import roboflow_workflows
import roboflow_workflows.enterprise_blocks as enterprise_blocks


def _walk_missing_init(root_dir: Path, ignore_relative=()):
    missing = []
    walked_any = False
    for root, _dirs, files in os.walk(str(root_dir)):
        if "__pycache__" in root:
            continue
        rel = os.path.relpath(root, root_dir)
        if any(ignored in rel for ignored in ignore_relative):
            continue
        # Ignore leftover cache-only directories from previous checkouts,
        # but still require package markers on every ancestor of real source.
        if not any(Path(root).rglob("*.py")):
            continue
        walked_any = True
        if Path(root) == root_dir:
            continue
        if "__init__.py" not in files:
            missing.append(rel)
    return walked_any, missing


def test_init_files_present_in_roboflow_workflows_core_steps():
    core_steps_root = Path(roboflow_workflows.__file__).parent / "core_steps"
    walked, missing = _walk_missing_init(
        core_steps_root,
        ignore_relative=(os.path.join("visualizations", "common", "fonts", "assets"),),
    )
    assert walked, f"walker found no directories under {core_steps_root}"
    assert (
        not missing
    ), "The following directories are missing __init__.py files:\n" + "\n".join(missing)


def test_init_files_present_in_roboflow_workflows_enterprise_blocks():
    root = Path(enterprise_blocks.__file__).parent
    walked, missing = _walk_missing_init(root)
    assert walked, f"walker found no directories under {root}"
    assert (
        not missing
    ), "The following directories are missing __init__.py files:\n" + "\n".join(missing)

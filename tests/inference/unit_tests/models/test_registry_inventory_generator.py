"""Keep registry snapshots regenerable as model registrations evolve."""

import subprocess
from pathlib import Path

from tests.inference.unit_tests.models import generate_registry_inventory as generator


def test_inventory_regenerates_added_removed_models_and_new_flag(monkeypatch, tmp_path):
    """Generate a changed registry without requiring an eager source revision.

    Args:
        monkeypatch: Pytest patch fixture.
        tmp_path: Directory for the modified registration source.
    """
    source = Path(__file__).resolve().parents[4] / "inference/models/utils.py"
    changed = tmp_path / "utils.py"
    changed.write_text(
        source.read_text()
        + "\nfrom inference.core.env import EXAMPLE_MODEL_ENABLED\n"
        + 'ROBOFLOW_MODEL_TYPES.pop(("object-detection", "yolov8n"), None)\n'
        + "if EXAMPLE_MODEL_ENABLED:\n"
        + '    ROBOFLOW_MODEL_TYPES[("object-detection", "example-new")] = '
        + '_LazyModelClass("example.models:NewModel")\n'
    )

    def reject_history(*args, **kwargs):
        raise AssertionError("Inventory regeneration must not read git history")

    monkeypatch.setattr(subprocess, "check_output", reject_history)
    inventory = generator._generate_inventory(source_path=changed)
    assert inventory["flags"]["EXAMPLE_MODEL_ENABLED"]
    for paths in inventory["cases"].values():
        assert ["object-detection", "example-new"] in paths["example.models:NewModel"]
        assert all(
            ["object-detection", "yolov8n"] not in keys for keys in paths.values()
        )

    registry = generator._registry_paths(
        flags={**inventory["flags"], "EXAMPLE_MODEL_ENABLED": False},
        use_adapters=True,
        use_proxy=False,
        source_path=changed,
    )
    assert ("object-detection", "example-new") not in registry

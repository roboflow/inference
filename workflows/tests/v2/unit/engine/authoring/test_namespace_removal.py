"""E5 keeps canonical V2 imports and removes the temporary alpha namespace."""

import importlib
import subprocess
import sys
import textwrap

import pytest


def test_temporary_namespace_is_not_importable() -> None:
    script = textwrap.dedent("""
        import importlib
        import importlib.util
        from pathlib import Path
        import roboflow_workflows.execution_engine as legacy

        old = "roboflow_workflows.execution_engine.v2"
        assert not Path(legacy.__file__).with_name("v2").exists()
        assert importlib.util.find_spec(old) is None
        for suffix in (
            "", ".blocks", ".data", ".kinds", ".errors", ".updates",
            ".reactions", ".reactions.testing", ".blocks.image_data",
            ".blocks.kinds", ".blocks.predictions", ".blocks.control",
            ".blocks.state_machine", ".blocks.visualization",
            ".blocks.visualization.blocks", ".blocks.visualization.drawing",
        ):
            try:
                importlib.import_module(old + suffix)
            except ModuleNotFoundError as error:
                assert error.name == old, (suffix, error.name)
            else:
                raise AssertionError("Temporary namespace still imports: " + old + suffix)
        assert importlib.util.find_spec("roboflow_workflows.execution_engine.v1.core")
        """)
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr


def test_canonical_module_discovery_keeps_packaged_catalogue() -> None:
    from roboflow_workflows.v2.blocks import create_catalogue
    from roboflow_workflows.v2.engine import Catalogue

    discovered = Catalogue.from_modules(["roboflow_workflows.v2.blocks"])
    direct = create_catalogue()

    assert len(direct.block_types) > 0
    assert discovered.block_types == direct.block_types
    assert discovered.operator_types == direct.operator_types
    assert sorted(discovered.kinds) == sorted(direct.kinds)
    for block_type in direct.block_types:
        assert discovered.entry(block_type).spec == direct.entry(block_type).spec


@pytest.mark.parametrize(
    "name", ["ManagedState", "PassivePipeline", "UpdateReceipt", "open_recording"]
)
def test_canonical_engine_keeps_public_lazy_names(name: str) -> None:
    engine = importlib.import_module("roboflow_workflows.v2.engine")

    assert name in engine.__all__
    assert getattr(engine, name).__module__.startswith("roboflow_workflows.v2.")

"""Deprecated ``execution_engine.v2`` import paths re-export the canonical objects.

Fifteen old modules keep working for existing V2 alpha callers. Each one is a
separate module that re-exports named objects; none is an alias of another
module and none holds a second implementation.
"""

import importlib
import sys
from pathlib import Path

import pytest
from roboflow_workflows.v2.engine import Catalogue

OLD = "roboflow_workflows.execution_engine.v2"
FORWARDED = {
    OLD: "roboflow_workflows.v2.engine",
    f"{OLD}.blocks": "roboflow_workflows.v2.blocks",
    f"{OLD}.data": "roboflow_workflows.v2.data",
    f"{OLD}.kinds": "roboflow_workflows.v2.data.kinds",
    f"{OLD}.errors": "roboflow_workflows.v2.engine.errors",
    f"{OLD}.blocks.image_data": "roboflow_workflows.v2.data.image",
    f"{OLD}.blocks.kinds": "roboflow_workflows.v2.data.image_kinds",
    f"{OLD}.blocks.predictions": "roboflow_workflows.v2.data.predictions",
    f"{OLD}.blocks.control": "roboflow_workflows.v2.blocks.control",
    f"{OLD}.blocks.state_machine": "roboflow_workflows.v2.blocks.state.machines",
    f"{OLD}.reactions.testing": "roboflow_workflows.v2.engine.reactions.testing",
    f"{OLD}.updates": "roboflow_workflows.v2.engine.updates",
    f"{OLD}.blocks.visualization": "roboflow_workflows.v2.blocks.visualization",
    f"{OLD}.blocks.visualization.blocks": (
        "roboflow_workflows.v2.blocks.visualization.blocks"
    ),
    f"{OLD}.blocks.visualization.drawing": (
        "roboflow_workflows.v2.blocks.visualization.drawing"
    ),
}
# Engine names that the canonical engine loads on first use.
LAZY_ENGINE_NAMES = (
    "ManagedState",
    "PassivePipeline",
    "UpdateReceipt",
    "open_recording",
)


@pytest.mark.parametrize("old_name", sorted(FORWARDED))
def test_every_forwarded_name_is_the_canonical_object(old_name: str) -> None:
    old = importlib.import_module(old_name)
    canonical = importlib.import_module(FORWARDED[old_name])

    assert old is not canonical
    assert old.__name__ == old_name
    assert "execution_engine" in Path(old.__file__).parts
    assert len(old.__all__) > 0
    for name in old.__all__:
        assert getattr(old, name) is getattr(canonical, name), name


def test_old_engine_entry_point_keeps_lazy_names_lazy() -> None:
    old = importlib.import_module(OLD)
    canonical = importlib.import_module(FORWARDED[OLD])

    assert set(LAZY_ENGINE_NAMES) <= set(old.__all__)
    for name in LAZY_ENGINE_NAMES:
        assert getattr(old, name) is getattr(canonical, name)
    with pytest.raises(AttributeError):
        getattr(old, "NotAnEngineName")


def test_old_modules_are_not_aliases_in_sys_modules() -> None:
    for old_name, canonical_name in FORWARDED.items():
        importlib.import_module(old_name)
        importlib.import_module(canonical_name)

        assert sys.modules[old_name] is not sys.modules[canonical_name]


def test_reactions_carrier_forwards_nothing() -> None:
    importlib.import_module(f"{OLD}.reactions.testing")
    carrier = importlib.import_module(f"{OLD}.reactions")

    assert not hasattr(carrier, "__all__")
    assert [name for name in vars(carrier) if not name.startswith("_")] == ["testing"]


def test_old_catalogue_module_string_gives_the_canonical_catalogue() -> None:
    old = Catalogue.from_modules([f"{OLD}.blocks"])
    canonical = Catalogue.from_modules(["roboflow_workflows.v2.blocks"])

    assert len(canonical.block_types) > 0
    assert old.block_types == canonical.block_types
    assert old.operator_types == canonical.operator_types
    assert sorted(old.kinds) == sorted(canonical.kinds)
    for block_type in canonical.block_types:
        assert old.entry(block_type).spec == canonical.entry(block_type).spec

import sys
import types
from pathlib import Path

from inference.core.workflows.core_steps.transformations.dynamic_crop.v1 import (
    DynamicCropBlockV1,
)
from inference.core.workflows.execution_engine.introspection import blocks_loader


def _install_fake_plugin(monkeypatch, name: str, **attributes) -> None:
    module = types.ModuleType(name)
    module.load_blocks = lambda: [DynamicCropBlockV1]
    for key, value in attributes.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)


def test_plugin_blocks_are_tagged_with_the_module_name_by_default(monkeypatch) -> None:
    _install_fake_plugin(monkeypatch, "fake_plugin_default")
    blocks = blocks_loader._load_blocks_from_plugin("fake_plugin_default")
    assert [b.block_source for b in blocks] == ["fake_plugin_default"]


def test_plugin_may_declare_its_block_source(monkeypatch) -> None:
    # Enterprise blocks use this to stay under `workflows_core`, so the
    # server's `workflows_core.disable_sinks` override keeps reaching them.
    _install_fake_plugin(
        monkeypatch, "fake_plugin_core", BLOCKS_SOURCE="workflows_core"
    )
    blocks = blocks_loader._load_blocks_from_plugin("fake_plugin_core")
    assert [b.block_source for b in blocks] == ["workflows_core"]


def test_plugin_initializers_follow_the_declared_source(monkeypatch) -> None:
    _install_fake_plugin(
        monkeypatch,
        "fake_plugin_init",
        BLOCKS_SOURCE="workflows_core",
        REGISTERED_INITIALIZERS={"thing": 1},
    )
    assert blocks_loader._load_initializers_from_plugin("fake_plugin_init") == {
        "workflows_core.thing": 1
    }


def test_blocks_loader_does_not_import_the_server_package() -> None:
    source = Path(blocks_loader.__file__).read_text(encoding="utf-8")
    assert "inference.enterprise" not in source
    assert "from inference.core.env import" not in source

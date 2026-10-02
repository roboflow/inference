"""Tests of the explicit V2 block catalogue."""

import subprocess
import sys
import textwrap
from pathlib import Path
from types import ModuleType

import pytest
from roboflow_workflows.execution_engine.v2.catalogue import (
    CATALOGUE_ATTRIBUTE,
    V2_ENGINE_VERSION,
    Catalogue,
)
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Output,
    Ref,
)
from roboflow_workflows.execution_engine.v2.errors import CatalogueError
from roboflow_workflows.execution_engine.v2.kinds import FLOAT_KIND, WILDCARD_KIND, Kind
from roboflow_workflows.execution_engine.v2.resources import Factory

CONSTRUCTED = []

IMAGE_KIND = Kind(name="image", description="Test image kind.", validate=lambda _: True)


class Scale(Block):
    """Scale a number."""

    type = "demo/scale@v1"
    aliases = ("Scale",)
    outputs = {"scaled": Output(FLOAT_KIND)}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)

    def __init__(self, *, audit=None):
        CONSTRUCTED.append(type(self).__name__)

    def run(self, *, value) -> dict:
        return {"scaled": value}


class Blur(Block):
    """Blur an image."""

    type = "demo/blur@v1"
    outputs = {"image": Output(IMAGE_KIND)}

    class Params(BlockParams):
        image: Ref(IMAGE_KIND)

    def __init__(self):
        CONSTRUCTED.append(type(self).__name__)

    def run(self, *, image) -> dict:
        return {"image": image}


class ScaleClone(Block):
    """Claims an alias already used by Scale."""

    type = "demo/scale_clone@v1"
    aliases = ("Scale",)

    def run(self) -> dict:
        return {}


class FutureOnly(Block):
    """Requires a future engine version."""

    type = "demo/future@v1"
    engine_compatibility = ">=3.0"

    def run(self) -> dict:
        return {}


class AbstractHelper(Block):
    """No type: not registrable."""


def test_registers_classes_and_finds_them_by_type_or_alias() -> None:
    catalogue = Catalogue([Scale, Blur], namespace="demo")

    assert catalogue.block_types == ("demo/scale@v1", "demo/blur@v1")
    assert catalogue.entry("Scale").spec.block_class is Scale
    assert catalogue.entry("demo/blur@v1").namespace == "demo"
    assert "Scale" in catalogue and "unknown" not in catalogue
    assert catalogue.find("unknown") is None
    assert len(catalogue) == 2


def test_kinds_are_collected_from_block_declarations() -> None:
    catalogue = Catalogue([Blur])

    assert catalogue.kind("image") is IMAGE_KIND
    assert set(catalogue.kinds) == {"*", "image"}
    with pytest.raises(CatalogueError, match="Unknown kind"):
        catalogue.kind("float")


def test_unknown_type_lists_known_types() -> None:
    with pytest.raises(CatalogueError, match="demo/scale@v1"):
        Catalogue([Scale]).entry("demo/missing@v1")


@pytest.mark.parametrize(
    "blocks, fragment",
    [
        ([Scale, ScaleClone], "already registered"),
        ([AbstractHelper], "abstract"),
        ([object], "Block subclass"),
        ([FutureOnly], "requires engine"),
    ],
)
def test_invalid_registrations_fail(blocks, fragment) -> None:
    with pytest.raises(CatalogueError, match=fragment):
        Catalogue(blocks)


def test_conflicting_kind_objects_are_rejected() -> None:
    with pytest.raises(CatalogueError, match="Two different kinds"):
        Catalogue([Blur], kinds=[Kind(name="image")])


def test_merge_keeps_one_copy_of_the_same_class_and_detects_conflicts() -> None:
    first = Catalogue([Scale], namespace="demo")
    second = Catalogue([Scale, Blur], namespace="demo")

    merged = Catalogue.merge(first, second)

    assert merged.block_types == ("demo/scale@v1", "demo/blur@v1")
    with pytest.raises(CatalogueError, match="already registered"):
        Catalogue.merge(first, Catalogue([Scale], namespace="other"))


def test_providers_are_scoped_by_namespace_and_must_not_conflict() -> None:
    audit = []
    catalogue = Catalogue([Scale], namespace="demo", providers={"audit": audit})

    assert catalogue.providers["demo"]["audit"] is audit
    with pytest.raises(CatalogueError, match="two different providers"):
        Catalogue.merge(
            catalogue, Catalogue(namespace="demo", providers={"audit": Factory(list)})
        )


def test_with_blocks_returns_a_new_catalogue() -> None:
    base = Catalogue([Scale])

    extended = base.with_blocks([Blur], namespace="dynamic")

    assert base.block_types == ("demo/scale@v1",)
    assert extended.entry("demo/blur@v1").namespace == "dynamic"


def test_describe_is_pure_and_complete() -> None:
    CONSTRUCTED.clear()
    catalogue = Catalogue([Scale, Blur], namespace="demo")

    description = catalogue.describe()

    assert CONSTRUCTED == []
    assert description["engine_version"] == V2_ENGINE_VERSION
    assert [block["type"] for block in description["blocks"]] == [
        "demo/scale@v1",
        "demo/blur@v1",
    ]
    scale = description["blocks"][0]
    assert scale["namespace"] == "demo"
    assert scale["aliases"] == ["Scale"]
    assert scale["resources"][0]["name"] == "audit"
    assert scale["params_schema"]["properties"]["value"]["selector"]["kinds"] == [
        "float"
    ]
    assert {
        "name": "image",
        "description": "Test image kind.",
        "validates": True,
        "deserializes": False,
        "serializes": False,
        "converts_output": False,
    } in (description["kinds"])


def test_from_modules_imports_requested_plugins_only(
    tmp_path: Path, monkeypatch
) -> None:
    plugin = tmp_path / "v2_test_plugin_module.py"
    plugin.write_text(textwrap.dedent(f"""
            from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
            from roboflow_workflows.execution_engine.v2.declaration import Block


            class PluginBlock(Block):
                type = "plugin/echo@v1"

                def run(self) -> dict:
                    return {{}}


            def {CATALOGUE_ATTRIBUTE}():
                return Catalogue([PluginBlock], namespace="plugin")
            """))
    empty = tmp_path / "v2_test_empty_module.py"
    empty.write_text("VALUE = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    catalogue = Catalogue.from_modules(["v2_test_plugin_module"])

    assert catalogue.entry("plugin/echo@v1").namespace == "plugin"
    with pytest.raises(CatalogueError, match=CATALOGUE_ATTRIBUTE):
        Catalogue.from_modules(["v2_test_empty_module"])
    with pytest.raises(CatalogueError, match="Cannot import"):
        Catalogue.from_modules(["v2_test_missing_module"])
    for name in ("v2_test_plugin_module", "v2_test_empty_module"):
        sys.modules.pop(name, None)


class WildcardEcho(Block):
    """A block using the neutral wildcard in its annotations."""

    type = "demo/wildcard_echo@v1"
    outputs = {"value": Output(WILDCARD_KIND)}

    class Params(BlockParams):
        value: Ref(WILDCARD_KIND)

    def run(self, *, value) -> dict:
        return {"value": value}


@pytest.mark.parametrize("explicit_first", [False, True])
def test_explicit_wildcard_survives_catalogue_merge_order(explicit_first) -> None:
    policy = Kind(name="*", serialize=str)
    neutral = Catalogue([WildcardEcho])
    explicit = Catalogue(kinds=[policy])
    catalogues = (explicit, neutral) if explicit_first else (neutral, explicit)

    merged = Catalogue.merge(*catalogues)

    assert merged.kind("*") is policy
    assert merged.entry(WildcardEcho.type).spec.block_class is WildcardEcho
    assert neutral.kind("*") is WILDCARD_KIND
    assert explicit.kind("*") is policy


def test_neutral_wildcard_annotations_do_not_override_explicit_policy() -> None:
    policy = Kind(name="*", serialize=str)

    catalogue = Catalogue([WildcardEcho], kinds=[policy, WILDCARD_KIND])

    assert catalogue.kind("*") is policy
    assert catalogue.entry(WildcardEcho.type).spec.kinds == (WILDCARD_KIND,)
    assert Catalogue().kind("*") is WILDCARD_KIND


def test_with_blocks_keeps_explicit_wildcard_policy() -> None:
    policy = Kind(name="*", serialize=str)
    catalogue = Catalogue(kinds=[policy])

    extended = catalogue.with_blocks([WildcardEcho])

    assert extended.kind("*") is policy
    assert extended.block_types == (WildcardEcho.type,)
    assert catalogue.block_types == ()


@pytest.mark.parametrize("explicit_first", [False, True])
def test_from_modules_keeps_explicit_wildcard_policy(
    monkeypatch, explicit_first
) -> None:
    policy = Kind(name="*", serialize=str)
    neutral = ModuleType("v2_neutral_wildcard_plugin")
    explicit = ModuleType("v2_explicit_wildcard_plugin")
    setattr(neutral, CATALOGUE_ATTRIBUTE, Catalogue([WildcardEcho]))
    setattr(explicit, CATALOGUE_ATTRIBUTE, lambda: Catalogue(kinds=[policy]))
    for module in (neutral, explicit):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    modules = (explicit, neutral) if explicit_first else (neutral, explicit)

    catalogue = Catalogue.from_modules([module.__name__ for module in modules])

    assert catalogue.kind("*") is policy
    assert catalogue.block_types == (WildcardEcho.type,)


def test_conflicting_explicit_wildcard_policies_are_rejected() -> None:
    first = Kind(name="*", serialize=str)
    second = Kind(name="*", serialize=repr)

    with pytest.raises(CatalogueError, match="Two different kinds"):
        Catalogue(kinds=[first, second])
    with pytest.raises(CatalogueError, match="Two different kinds"):
        Catalogue.merge(Catalogue(kinds=[first]), Catalogue(kinds=[second]))


def test_ordinary_kind_conflicts_remain_strict_when_merging() -> None:
    with pytest.raises(CatalogueError, match="Two different kinds"):
        Catalogue.merge(Catalogue([Blur]), Catalogue(kinds=[Kind(name="image")]))


def test_generic_catalogue_never_imports_media_modules() -> None:
    script = textwrap.dedent("""
        import importlib.abc
        import sys

        blocked = (
            "torch", "numpy", "cv2", "supervision", "inference_models",
            "inference", "roboflow_workflows.execution_engine.v2.blocks",
        )

        class NoMediaImports(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
                    raise AssertionError("Unexpected media import: " + fullname)
                return None

        sys.meta_path.insert(0, NoMediaImports())
        from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
        from roboflow_workflows.execution_engine.v2.kinds import Kind, WILDCARD_KIND
        policy = Kind(name="*", serialize=str)
        catalogue = Catalogue.merge(Catalogue(), Catalogue(kinds=[policy]))
        assert catalogue.kind("*") is policy
        assert Catalogue().kind("*") is WILDCARD_KIND
        assert not any(
            name == prefix or name.startswith(prefix + ".")
            for name in sys.modules for prefix in blocked
        )
        """)

    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )

    assert completed.returncode == 0, completed.stderr

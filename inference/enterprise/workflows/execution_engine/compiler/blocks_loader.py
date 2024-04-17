import importlib
import logging
import os
from typing import Any, Callable, Dict, List, Union

from inference.enterprise.workflows.core_steps.loader import load_blocks_classes
from inference.enterprise.workflows.entities.blocks_descriptions import (
    BlockDescription,
    BlocksDescription,
)
from inference.enterprise.workflows.entities.types import Kind
from inference.enterprise.workflows.errors import (
    PluginInterfaceError,
    PluginLoadingError,
)
from inference.enterprise.workflows.execution_engine.compiler.entities import (
    BlockSpecification,
)

WORKFLOWS_PLUGINS_ENV = "WORKFLOWS_PLUGINS"


def describe_available_blocks() -> BlocksDescription:
    blocks = load_workflow_blocks()
    declared_kinds = []
    result = []
    for block in blocks:
        block_manifest = block.manifest_class.schema()
        outputs_manifest = block.block_class.describe_outputs()
        declared_kinds.extend(
            get_kinds_declared_for_block(block_manifest=block_manifest)
        )
        result.append(
            BlockDescription(
                block_manifest=block_manifest,
                outputs_manifest=outputs_manifest,
                fully_qualified_class_name=block.identifier,
            )
        )
    declared_kinds = list(set(declared_kinds))
    return BlocksDescription(blocks=result, declared_kinds=declared_kinds)


def get_kinds_declared_for_block(block_manifest: dict) -> List[Kind]:
    result = []
    for property_name, property_definition in block_manifest["properties"].items():
        union_elements = property_definition.get(
            "anyOf", property_definition.get("oneOf", [property_definition])
        )
        for element in union_elements:
            for raw_kind in element.get("kind", []):
                parsed_kind = Kind.model_validate(raw_kind)
                result.append(parsed_kind)
    return result


def load_workflow_blocks() -> List[BlockSpecification]:
    core_blocks = load_core_workflow_blocks()
    plugins_blocks = load_plugins_blocks()
    return core_blocks + plugins_blocks


def load_core_workflow_blocks() -> List[BlockSpecification]:
    core_blocks = load_blocks_classes()
    return [
        BlockSpecification(
            block_source="workflows_core",
            identifier=get_full_type_name(t=block),
            block_class=block,
            manifest_class=block.get_input_manifest(),
        )
        for block in core_blocks
    ]


def load_plugins_blocks() -> List[BlockSpecification]:
    plugins_to_load = os.getenv(WORKFLOWS_PLUGINS_ENV)
    if plugins_to_load is None:
        return []
    custom_blocks = []
    for plugin_name in plugins_to_load.split(","):
        custom_blocks.extend(load_blocks_from_plugin(plugin_name=plugin_name))
    return custom_blocks


def load_blocks_from_plugin(plugin_name: str) -> List[BlockSpecification]:
    try:
        return _load_blocks_from_plugin(plugin_name=plugin_name)
    except ImportError as e:
        raise PluginLoadingError(
            public_message=f"It is not possible to load workflow plugin `{plugin_name}`. "
            f"Make sure the library providing custom step is correctly installed in Python environment.",
            context="workflow_compilation | blocks_loading",
            inner_error=e,
        ) from e
    except AttributeError as e:
        raise PluginInterfaceError(
            public_message=f"Provided workflow plugin `{plugin_name}` do not implement blocks loading "
            f"interface correctly and cannot be loaded.",
            context="workflow_compilation | blocks_loading",
            inner_error=e,
        ) from e


def _load_blocks_from_plugin(plugin_name: str) -> List[BlockSpecification]:
    module = importlib.import_module(plugin_name)
    blocks = module.load_blocks()
    return [
        BlockSpecification(
            block_source=plugin_name,
            identifier=get_full_type_name(t=block),
            block_class=block,
            manifest_class=block.get_input_manifest(),
        )
        for block in blocks
    ]


def load_initializers() -> Dict[str, Union[Any, Callable[[None], Any]]]:
    plugins_to_load = os.getenv(WORKFLOWS_PLUGINS_ENV)
    if plugins_to_load is None:
        return {}
    result = {}
    for plugin_name in plugins_to_load.split(","):
        result.update(load_initializers_from_plugin(plugin_name=plugin_name))
    return result


def load_initializers_from_plugin(plugin_name: str) -> Dict[str, Callable[[None], Any]]:
    try:
        logging.info(f"Loading workflows initializers from plugin {plugin_name}")
        return _load_initializers_from_plugin(plugin_name=plugin_name)
    except ImportError as e:
        raise PluginLoadingError(
            public_message=f"It is not possible to load workflow plugin `{plugin_name}`. "
            f"Make sure the library providing custom step is correctly installed in Python environment.",
            context="workflow_compilation | blocks_loading",
            inner_error=e,
        ) from e


def _load_initializers_from_plugin(
    plugin_name: str,
) -> Dict[str, Callable[[None], Any]]:
    module = importlib.import_module(plugin_name)
    registered_initializers = getattr(module, "REGISTERED_INITIALIZERS", {})
    return {
        f"{plugin_name}.{parameter_name}": initializer
        for parameter_name, initializer in registered_initializers.items()
    }


def get_full_type_name(t: type) -> str:
    t_class = t.__name__
    t_module = t.__module__
    if t_module == "builtins":
        return t_class.__qualname__
    return t_module + "." + t.__qualname__

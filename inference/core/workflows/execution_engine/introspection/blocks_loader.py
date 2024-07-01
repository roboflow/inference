import importlib
import logging
import os
from collections import Counter
from typing import Any, Callable, Dict, List, Union

from inference.core.workflows.core_steps.loader import load_blocks
from inference.core.workflows.entities.types import Kind
from inference.core.workflows.errors import PluginInterfaceError, PluginLoadingError
from inference.core.workflows.execution_engine.compiler.entities import (
    BlockSpecification,
)
from inference.core.workflows.execution_engine.introspection.entities import (
    BlockDescription,
    BlocksDescription,
)
from inference.core.workflows.execution_engine.introspection.schema_parser import (
    retrieve_selectors_from_schema,
)
from inference.core.workflows.execution_engine.introspection.utils import (
    build_human_friendly_block_name,
    get_full_type_name,
)

WORKFLOWS_PLUGINS_ENV = "WORKFLOWS_PLUGINS"


def describe_available_blocks() -> BlocksDescription:
    blocks = load_workflow_blocks()
    declared_kinds = []
    result = []
    for block in blocks:
        block_schema = block.manifest_class.model_json_schema()
        outputs_manifest = block.manifest_class.describe_outputs()
        schema_selectors = retrieve_selectors_from_schema(
            schema=block_schema,
            inputs_dimensionality_offsets=block.manifest_class.get_input_dimensionality_offsets(),
            dimensionality_reference_property=block.manifest_class.get_dimensionality_reference_property(),
        )
        block_kinds = [
            k
            for s in schema_selectors.values()
            for r in s.allowed_references
            for k in r.kind
        ]
        declared_kinds.extend(block_kinds)
        for output in outputs_manifest:
            declared_kinds.extend(output.kind)
        manifest_type_identifiers = get_manifest_type_identifiers(
            block_schema=block_schema,
            block_source=block.block_source,
            block_identifier=block.identifier,
        )
        result.append(
            BlockDescription(
                manifest_class=block.manifest_class,
                block_class=block.block_class,
                block_schema=block_schema,
                outputs_manifest=outputs_manifest,
                block_source=block.block_source,
                fully_qualified_block_class_name=block.identifier,
                human_friendly_block_name=build_human_friendly_block_name(
                    fully_qualified_name=block.identifier, block_schema=block_schema
                ),
                manifest_type_identifier=manifest_type_identifiers[0],
                manifest_type_identifier_aliases=manifest_type_identifiers[1:],
            )
        )
    _validate_loaded_blocks_names_uniqueness(blocks=result)
    _validate_loaded_blocks_manifest_type_identifiers(blocks=result)
    declared_kinds = list(set(declared_kinds))
    _validate_used_kinds_uniqueness(declared_kinds=declared_kinds)
    return BlocksDescription(blocks=result, declared_kinds=declared_kinds)


def get_manifest_type_identifiers(
    block_schema: dict,
    block_source: str,
    block_identifier: str,
) -> List[str]:
    if "type" not in block_schema["properties"]:
        raise PluginInterfaceError(
            public_message="Required `type` property not defined for block "
            f"`{block_identifier}` loaded from `{block_source}",
            context="workflow_compilation | blocks_loading",
        )
    constant_literal = block_schema["properties"]["type"].get("const")
    if constant_literal is not None:
        return [constant_literal]
    valid_aliases = block_schema["properties"]["type"].get("enum", [])
    if len(valid_aliases) > 0:
        return valid_aliases
    raise PluginInterfaceError(
        public_message="`type` property for block is required to be `Literal` "
        "defining at least one unique value to identify block in JSON "
        f"definitions. Block `{block_identifier}` loaded from `{block_source} "
        f"does not fit that requirement.",
        context="workflow_compilation | blocks_loading",
    )


def load_workflow_blocks() -> List[BlockSpecification]:
    core_blocks = load_core_workflow_blocks()
    plugins_blocks = load_plugins_blocks()
    return core_blocks + plugins_blocks


def load_core_workflow_blocks() -> List[BlockSpecification]:
    core_blocks = load_blocks()
    already_spotted_blocks = set()
    result = []
    for block in core_blocks:
        if block in already_spotted_blocks:
            continue
        result.append(
            BlockSpecification(
                block_source="workflows_core",
                identifier=get_full_type_name(selected_type=block),
                block_class=block,
                manifest_class=block.get_manifest(),
            )
        )
        already_spotted_blocks.add(block)
    return result


def load_plugins_blocks() -> List[BlockSpecification]:
    plugins_to_load = get_plugin_modules()
    custom_blocks = []
    for plugin_name in plugins_to_load:
        custom_blocks.extend(load_blocks_from_plugin(plugin_name=plugin_name))
    return custom_blocks


def get_plugin_modules() -> List[str]:
    plugins_to_load = os.environ.get(WORKFLOWS_PLUGINS_ENV)
    if plugins_to_load is None:
        return []
    return plugins_to_load.split(",")


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
    already_spotted_blocks = set()
    result = []
    for block in blocks:
        if block in already_spotted_blocks:
            continue
        result.append(
            BlockSpecification(
                block_source=plugin_name,
                identifier=get_full_type_name(selected_type=block),
                block_class=block,
                manifest_class=block.get_manifest(),
            )
        )
        already_spotted_blocks.add(block)
    return result


def load_initializers() -> Dict[str, Union[Any, Callable[[None], Any]]]:
    plugins_to_load = os.environ.get(WORKFLOWS_PLUGINS_ENV)
    if plugins_to_load is None:
        return {}
    result = {}
    for plugin_name in plugins_to_load.split(","):
        result.update(load_initializers_from_plugin(plugin_name=plugin_name))
    return result


def load_initializers_from_plugin(
    plugin_name: str,
) -> Dict[str, Union[Any, Callable[[None], Any]]]:
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


def _validate_loaded_blocks_names_uniqueness(blocks: List[BlockDescription]) -> None:
    block_names_lookup = {}
    for block in blocks:
        if block.human_friendly_block_name in block_names_lookup:
            clashing_block = block_names_lookup[block.human_friendly_block_name]
            raise PluginLoadingError(
                public_message=f"Block defined in {block.block_source} plugin with fully qualified class "
                f"name {block.fully_qualified_block_class_name} clashes in terms of "
                f"the human friendly name (value={block.human_friendly_block_name}) with other "
                f"block - defined in {clashing_block.block_source} with fully qualified class name: "
                f"{clashing_block.fully_qualified_block_class_name}.",
                context="workflow_compilation | blocks_loading",
            )
        block_names_lookup[block.human_friendly_block_name] = block
    return None


def _validate_loaded_blocks_manifest_type_identifiers(
    blocks: List[BlockDescription],
) -> None:
    types_already_defined = {}
    for block in blocks:
        all_types = [
            block.manifest_type_identifier
        ] + block.manifest_type_identifier_aliases
        for type_name in all_types:
            if type_name in types_already_defined:
                clashing_block = types_already_defined[type_name]
                raise PluginLoadingError(
                    public_message=f"Block defined in {block.block_source} plugin with fully qualified class "
                    f"name {block.fully_qualified_block_class_name} clashes in terms of "
                    f"the manifest type identifier (or its alias): {type_name} - defined in "
                    f"{clashing_block.block_source} with fully qualified class name: "
                    f"{clashing_block.fully_qualified_block_class_name}.",
                    context="workflow_compilation | blocks_loading",
                )
            types_already_defined[type_name] = block
    return None


def _validate_used_kinds_uniqueness(declared_kinds: List[Kind]) -> None:
    kinds_names_counter = Counter(k.name for k in declared_kinds)
    non_unique_kinds = [k for k, v in kinds_names_counter.items() if v > 1]
    if non_unique_kinds:
        raise PluginLoadingError(
            public_message=f"Loaded plugins blocks define kinds causing names clash "
            f"(problematic kinds: {non_unique_kinds}). This is most likely caused "
            f"by loading plugins that defines custom kinds which accidentally hold "
            f"the same name.",
            context="workflow_compilation | blocks_loading",
        )

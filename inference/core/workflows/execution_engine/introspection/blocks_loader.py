import importlib
import logging
import os
from collections import Counter
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Union

from packaging.specifiers import SpecifierSet
from packaging.version import Version

from inference.core.workflows.core_steps.loader import (
    REGISTERED_INITIALIZERS,
    load_blocks,
    load_kinds,
)
from inference.core.workflows.errors import (
    PluginInterfaceError,
    PluginLoadingError,
    WorkflowExecutionEngineVersionError,
)
from inference.core.workflows.execution_engine.entities.types import Kind
from inference.core.workflows.execution_engine.introspection.entities import (
    BlockDescription,
    BlocksDescription,
)
from inference.core.workflows.execution_engine.introspection.utils import (
    build_human_friendly_block_name,
    get_full_type_name,
)
from inference.core.workflows.execution_engine.profiling.core import (
    WorkflowsProfiler,
    execution_phase,
)
from inference.core.workflows.execution_engine.v1.compiler.entities import (
    BlockSpecification,
)
from inference.core.workflows.execution_engine.v1.dynamic_blocks.entities import (
    BLOCK_SOURCE,
)
from inference.core.workflows.prototypes.block import WorkflowBlock

WORKFLOWS_PLUGINS_ENV = "WORKFLOWS_PLUGINS"
WORKFLOWS_CORE_PLUGIN_NAME = "workflows_core"


def describe_available_blocks(
    dynamic_blocks: List[BlockSpecification],
    execution_engine_version: Optional[Union[str, Version]] = None,
) -> BlocksDescription:
    blocks = (
        load_workflow_blocks(execution_engine_version=execution_engine_version)
        + dynamic_blocks
    )
    result = []
    for block in blocks:
        block_schema = block.manifest_class.model_json_schema()
        outputs_manifest = block.manifest_class.describe_outputs()
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
                execution_engine_compatibility=block.manifest_class.get_execution_engine_compatibility(),
                input_dimensionality_offsets=block.manifest_class.get_input_dimensionality_offsets(),
                dimensionality_reference_property=block.manifest_class.get_dimensionality_reference_property(),
                output_dimensionality_offset=block.manifest_class.get_output_dimensionality_offset(),
            )
        )
    _validate_loaded_blocks_manifest_type_identifiers(blocks=result)
    declared_kinds = load_all_defined_kinds()
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
            context="blocks_loading",
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
        f"definitions. Block `{block_identifier}` loaded from `{block_source}` "
        f"does not fit that requirement.",
        context="blocks_loading",
    )


@execution_phase(
    name="blocks_loading",
    categories=["execution_engine_operation"],
)
def load_workflow_blocks(
    execution_engine_version: Optional[Union[str, Version]] = None,
    profiler: Optional[WorkflowsProfiler] = None,
) -> List[BlockSpecification]:
    if isinstance(execution_engine_version, str):
        try:
            execution_engine_version = Version(execution_engine_version)
        except ValueError as error:
            raise WorkflowExecutionEngineVersionError(
                public_message=f"Could not parse execution engine version `{execution_engine_version}` while "
                f"workflow blocks loading",
                inner_error=error,
                context="blocks_loading",
            )
    core_blocks = load_core_workflow_blocks()
    plugins_blocks = load_plugins_blocks()
    all_blocks = core_blocks + plugins_blocks
    filtered_blocks = []
    for block in all_blocks:
        if not is_block_compatible_with_execution_engine(
            execution_engine_version=execution_engine_version,
            block_execution_engine_compatibility=block.manifest_class.get_execution_engine_compatibility(),
            block_source=block.block_source,
            block_identifier=block.identifier,
        ):
            continue
        filtered_blocks.append(block)
    return filtered_blocks


@lru_cache()
def load_core_workflow_blocks() -> List[BlockSpecification]:
    core_blocks = load_blocks()
    already_spotted_blocks = set()
    result = []
    for block in core_blocks:
        manifest_class = block.get_manifest()
        identifier = get_full_type_name(selected_type=block)
        if block in already_spotted_blocks:
            continue
        result.append(
            BlockSpecification(
                block_source=WORKFLOWS_CORE_PLUGIN_NAME,
                identifier=identifier,
                block_class=block,
                manifest_class=manifest_class,
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


def load_blocks_from_plugin(plugin_name: str) -> List[BlockSpecification]:
    try:
        return _load_blocks_from_plugin(plugin_name=plugin_name)
    except ImportError as e:
        raise PluginLoadingError(
            public_message=f"It is not possible to load workflow plugin `{plugin_name}`. "
            f"Make sure the library providing custom step is correctly installed in Python environment.",
            context="blocks_loading",
            inner_error=e,
        ) from e
    except AttributeError as e:
        raise PluginInterfaceError(
            public_message=f"Provided workflow plugin `{plugin_name}` do not implement blocks loading "
            f"interface correctly and cannot be loaded.",
            context="blocks_loading",
            inner_error=e,
        ) from e


def _load_blocks_from_plugin(plugin_name: str) -> List[BlockSpecification]:
    module = importlib.import_module(plugin_name)
    blocks = module.load_blocks()
    already_spotted_blocks = set()
    result = []
    if not isinstance(blocks, list):
        raise PluginInterfaceError(
            public_message=f"Provided workflow plugin `{plugin_name}` implement `load_blocks()` function "
            f"incorrectly. Expected to return list of entries being subclass of `WorkflowBlock`, "
            f"but entry of different characteristics found: {type(blocks)}.",
            context="blocks_loading",
        )
    for i, block in enumerate(blocks):
        if not isinstance(block, type) or not issubclass(block, WorkflowBlock):
            raise PluginInterfaceError(
                public_message=f"Provided workflow plugin `{plugin_name}` implement `load_blocks()` function "
                f"incorrectly. Expected to return list of entries being subclass of `WorkflowBlock`, "
                f"but entry of different characteristics found: {block} at position: {i}.",
                context="blocks_loading",
            )
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


def is_block_compatible_with_execution_engine(
    execution_engine_version: Optional[Version],
    block_execution_engine_compatibility: Optional[str],
    block_source: str,
    block_identifier: str,
) -> bool:
    if block_execution_engine_compatibility is None or execution_engine_version is None:
        return True
    try:
        return SpecifierSet(block_execution_engine_compatibility).contains(
            execution_engine_version
        )
    except ValueError as error:
        raise PluginInterfaceError(
            public_message=f"Could not parse either version of Execution Engine ({execution_engine_version}) or "
            f"EE version requirements ({block_execution_engine_compatibility}) for "
            f"block `{block_identifier}` loaded from `{block_source}`.",
            inner_error=error,
            context="blocks_loading",
        )


@execution_phase(
    name="blocks_initializers_loading",
    categories=["execution_engine_operation"],
)
def load_initializers(
    profiler: Optional[WorkflowsProfiler] = None,
) -> Dict[str, Union[Any, Callable[[None], Any]]]:
    plugins_to_load = get_plugin_modules()
    result = load_core_blocks_initializers()
    for plugin_name in plugins_to_load:
        result.update(load_initializers_from_plugin(plugin_name=plugin_name))
    return result


def load_core_blocks_initializers() -> Dict[str, Union[Any, Callable[[None], Any]]]:
    return {
        f"{WORKFLOWS_CORE_PLUGIN_NAME}.{parameter_name}": initializer
        for parameter_name, initializer in REGISTERED_INITIALIZERS.items()
    }


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
            context="blocks_loading",
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
                block_identifier = _produce_readable_block_identifier(block=block)
                clashing_block_identifier = _produce_readable_block_identifier(
                    block=clashing_block
                )
                raise PluginLoadingError(
                    public_message=f"Block `{block_identifier}`, defined in `{block.block_source}` plugin,"
                    f"clashes in terms of the manifest type identifier (or its alias): "
                    f"`{type_name}` with `{clashing_block_identifier}` defined in "
                    f"`{clashing_block.block_source}` plugin.",
                    context="blocks_loading",
                )
            types_already_defined[type_name] = block
    return None


def _produce_readable_block_identifier(block: BlockDescription) -> str:
    if block.block_source == BLOCK_SOURCE:
        return block.human_friendly_block_name
    return block.fully_qualified_block_class_name


def _validate_used_kinds_uniqueness(declared_kinds: List[Kind]) -> None:
    kinds_names_counter = Counter(k.name for k in declared_kinds)
    non_unique_kinds = [k for k, v in kinds_names_counter.items() if v > 1]
    if non_unique_kinds:
        raise PluginLoadingError(
            public_message=f"Loaded plugins blocks define kinds causing names clash "
            f"(problematic kinds: {non_unique_kinds}). This is most likely caused "
            f"by loading plugins that defines custom kinds which accidentally hold "
            f"the same name.",
            context="blocks_loading",
        )


def load_all_defined_kinds() -> List[Kind]:
    core_blocks_kinds = load_kinds()
    plugins_kinds = load_plugins_kinds()
    declared_kinds = core_blocks_kinds + plugins_kinds
    declared_kinds = list(set(declared_kinds))
    _validate_used_kinds_uniqueness(declared_kinds=declared_kinds)
    return declared_kinds


def load_plugins_kinds() -> List[Kind]:
    plugins_to_load = get_plugin_modules()
    result = []
    for plugin_name in plugins_to_load:
        result.extend(load_plugin_kinds(plugin_name=plugin_name))
    return result


def load_plugin_kinds(plugin_name: str) -> List[Kind]:
    try:
        return _load_plugin_kinds(plugin_name=plugin_name)
    except ImportError as e:
        raise PluginLoadingError(
            public_message=f"It is not possible to load kinds from workflow plugin `{plugin_name}`. "
            f"Make sure the library providing custom step is correctly installed in Python environment.",
            context="blocks_loading",
            inner_error=e,
        ) from e
    except AttributeError as e:
        raise PluginInterfaceError(
            public_message=f"Provided workflow plugin `{plugin_name}` do not implement blocks loading "
            f"interface correctly and cannot be loaded.",
            context="blocks_loading",
            inner_error=e,
        ) from e


def _load_plugin_kinds(plugin_name: str) -> List[Kind]:
    module = importlib.import_module(plugin_name)
    if not hasattr(module, "load_kinds"):
        return []
    kinds_extractor = getattr(module, "load_kinds")
    if not callable(kinds_extractor):
        logging.warning(
            f"Found `load_kinds` symbol in plugin `{plugin_name}` module init, but it is not callable. "
            f"Not importing kinds from that plugin."
        )
        return []
    kinds = kinds_extractor()
    if not isinstance(kinds, list) or not all(isinstance(e, Kind) for e in kinds):
        raise PluginInterfaceError(
            public_message=f"Provided workflow plugin `{plugin_name}` do not implement blocks loading "
            f"interface correctly and cannot be loaded. Return value of `load_kinds()` "
            f"is not list of objects `Kind`.",
            context="blocks_loading",
        )
    return kinds


def get_plugin_modules() -> List[str]:
    plugins_to_load = os.environ.get(WORKFLOWS_PLUGINS_ENV)
    if plugins_to_load is None:
        return []
    return plugins_to_load.split(",")

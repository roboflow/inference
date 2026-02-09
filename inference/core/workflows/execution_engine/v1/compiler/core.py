import json
import time
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Union

import networkx as nx
from packaging.version import Version

from inference.core import logger

_compiler_module_start = time.perf_counter()
logger.warning("[COLD_START] compiler/core.py: Module loading started...")

_entities_base_start = time.perf_counter()
from inference.core.workflows.execution_engine.entities.base import WorkflowParameter
logger.warning("[COLD_START] compiler/core.py: entities.base in %.3fs", time.perf_counter() - _entities_base_start)

_blocks_loader_start = time.perf_counter()
logger.warning("[COLD_START] compiler/core.py: Starting blocks_loader import (THIS IS THE HEAVY ONE)...")
from inference.core.workflows.execution_engine.introspection.blocks_loader import (
    load_initializers,
    load_kinds_deserializers,
    load_kinds_serializers,
    load_workflow_blocks,
)
logger.warning("[COLD_START] compiler/core.py: blocks_loader imported in %.3fs", time.perf_counter() - _blocks_loader_start)

_profiling_start = time.perf_counter()
from inference.core.workflows.execution_engine.profiling.core import (
    WorkflowsProfiler,
    execution_phase,
)
logger.warning("[COLD_START] compiler/core.py: profiling.core in %.3fs", time.perf_counter() - _profiling_start)

_cache_start = time.perf_counter()
from inference.core.workflows.execution_engine.v1.compiler.cache import (
    BasicWorkflowsCache,
)
logger.warning("[COLD_START] compiler/core.py: cache in %.3fs", time.perf_counter() - _cache_start)

_entities_start = time.perf_counter()
from inference.core.workflows.execution_engine.v1.compiler.entities import (
    BlockSpecification,
    CompiledWorkflow,
    InputSubstitution,
    ParsedWorkflowDefinition,
)
logger.warning("[COLD_START] compiler/core.py: entities in %.3fs", time.perf_counter() - _entities_start)

_graph_start = time.perf_counter()
from inference.core.workflows.execution_engine.v1.compiler.graph_constructor import (
    prepare_execution_graph,
)
logger.warning("[COLD_START] compiler/core.py: graph_constructor in %.3fs", time.perf_counter() - _graph_start)

_steps_start = time.perf_counter()
from inference.core.workflows.execution_engine.v1.compiler.steps_initialiser import (
    initialise_steps,
)
logger.warning("[COLD_START] compiler/core.py: steps_initialiser in %.3fs", time.perf_counter() - _steps_start)

_parser_start = time.perf_counter()
from inference.core.workflows.execution_engine.v1.compiler.syntactic_parser import (
    parse_workflow_definition,
)
logger.warning("[COLD_START] compiler/core.py: syntactic_parser in %.3fs", time.perf_counter() - _parser_start)

_utils_start = time.perf_counter()
from inference.core.workflows.execution_engine.v1.compiler.utils import (
    construct_input_selector,
)
logger.warning("[COLD_START] compiler/core.py: utils in %.3fs", time.perf_counter() - _utils_start)

_validator_start = time.perf_counter()
from inference.core.workflows.execution_engine.v1.compiler.validator import (
    validate_workflow_specification,
)
logger.warning("[COLD_START] compiler/core.py: validator in %.3fs", time.perf_counter() - _validator_start)

_debugger_start = time.perf_counter()
from inference.core.workflows.execution_engine.v1.debugger.core import (
    dump_execution_graph,
)
logger.warning("[COLD_START] compiler/core.py: debugger in %.3fs", time.perf_counter() - _debugger_start)

_dynamic_blocks_start = time.perf_counter()
from inference.core.workflows.execution_engine.v1.dynamic_blocks.block_assembler import (
    compile_dynamic_blocks,
    ensure_dynamic_blocks_allowed,
)
logger.warning("[COLD_START] compiler/core.py: dynamic_blocks in %.3fs", time.perf_counter() - _dynamic_blocks_start)

_prototypes_start = time.perf_counter()
from inference.core.workflows.prototypes.block import WorkflowBlockManifest
logger.warning("[COLD_START] compiler/core.py: prototypes.block in %.3fs", time.perf_counter() - _prototypes_start)

logger.warning("[COLD_START] compiler/core.py: Module loading completed in %.3fs", time.perf_counter() - _compiler_module_start)


@dataclass(frozen=True)
class GraphCompilationResult:
    execution_graph: nx.DiGraph
    parsed_workflow_definition: ParsedWorkflowDefinition
    available_blocks: List[BlockSpecification]
    initializers: Dict[str, Union[Any, Callable[[None], Any]]]
    kinds_serializers: Dict[str, Callable[[Any], Any]]
    kinds_deserializers: Dict[str, Callable[[str, Any], Any]]


COMPILATION_CACHE = BasicWorkflowsCache[GraphCompilationResult](
    cache_size=256,
    hash_functions=[
        (
            "workflow_definition",
            partial(json.dumps, sort_keys=True),
        ),
        ("execution_engine_version", lambda version: str(version)),
    ],
)


@execution_phase(
    name="workflow_compilation",
    categories=["execution_engine_operation"],
)
def compile_workflow(
    workflow_definition: dict,
    init_parameters: Dict[str, Union[Any, Callable[[None], Any]]],
    execution_engine_version: Optional[Version] = None,
    profiler: Optional[WorkflowsProfiler] = None,
) -> CompiledWorkflow:
    graph_compilation_results = compile_workflow_graph(
        workflow_definition=workflow_definition,
        execution_engine_version=execution_engine_version,
        profiler=profiler,
        init_parameters=init_parameters,
    )
    steps = initialise_steps(
        steps_manifest=graph_compilation_results.parsed_workflow_definition.steps,
        available_blocks=graph_compilation_results.available_blocks,
        explicit_init_parameters=init_parameters,
        initializers=graph_compilation_results.initializers,
        profiler=profiler,
    )
    input_substitutions = collect_input_substitutions(
        workflow_definition=graph_compilation_results.parsed_workflow_definition,
    )
    steps_by_name = {step.manifest.name: step for step in steps}
    dump_execution_graph(execution_graph=graph_compilation_results.execution_graph)
    return CompiledWorkflow(
        workflow_definition=graph_compilation_results.parsed_workflow_definition,
        workflow_json=workflow_definition,
        init_parameters=init_parameters,
        execution_graph=graph_compilation_results.execution_graph,
        steps=steps_by_name,
        input_substitutions=input_substitutions,
        kinds_serializers=graph_compilation_results.kinds_serializers,
        kinds_deserializers=graph_compilation_results.kinds_deserializers,
    )


def compile_workflow_graph(
    workflow_definition: dict,
    execution_engine_version: Optional[Version] = None,
    profiler: Optional[WorkflowsProfiler] = None,
    init_parameters: Dict[str, Union[Any, Callable[[None], Any]]] = {},
) -> GraphCompilationResult:
    key = COMPILATION_CACHE.get_hash_key(
        workflow_definition=workflow_definition,
        execution_engine_version=execution_engine_version,
    )
    cached_value = COMPILATION_CACHE.get(key=key)
    if cached_value is not None:
        dynamic_blocks_definitions = workflow_definition.get(
            "dynamic_blocks_definitions", []
        )
        ensure_dynamic_blocks_allowed(
            dynamic_blocks_definitions=dynamic_blocks_definitions
        )
        return cached_value
    statically_defined_blocks = load_workflow_blocks(
        execution_engine_version=execution_engine_version,
        profiler=profiler,
    )
    initializers = load_initializers(profiler=profiler)
    kinds_serializers = load_kinds_serializers(profiler=profiler)
    kinds_deserializers = load_kinds_deserializers(profiler=profiler)
    dynamic_blocks = compile_dynamic_blocks(
        dynamic_blocks_definitions=workflow_definition.get(
            "dynamic_blocks_definitions", []
        ),
        profiler=profiler,
        api_key=init_parameters.get("workflows_core.api_key", None),
    )
    available_blocks = statically_defined_blocks + dynamic_blocks
    parsed_workflow_definition = parse_workflow_definition(
        raw_workflow_definition=workflow_definition,
        available_blocks=available_blocks,
        profiler=profiler,
    )
    validate_workflow_specification(
        workflow_definition=parsed_workflow_definition,
        profiler=profiler,
    )
    execution_graph = prepare_execution_graph(
        workflow_definition=parsed_workflow_definition,
        profiler=profiler,
    )
    result = GraphCompilationResult(
        execution_graph=execution_graph,
        parsed_workflow_definition=parsed_workflow_definition,
        available_blocks=available_blocks,
        initializers=initializers,
        kinds_serializers=kinds_serializers,
        kinds_deserializers=kinds_deserializers,
    )
    COMPILATION_CACHE.cache(key=key, value=result)
    return result


def collect_input_substitutions(
    workflow_definition: ParsedWorkflowDefinition,
) -> List[InputSubstitution]:
    result = []
    for declared_input in workflow_definition.inputs:
        if not isinstance(declared_input, WorkflowParameter):
            continue
        input_substitutions = collect_substitutions_for_selected_input(
            input_name=declared_input.name,
            steps=workflow_definition.steps,
        )
        result.extend(input_substitutions)
    return result


def collect_substitutions_for_selected_input(
    input_name: str,
    steps: List[WorkflowBlockManifest],
) -> List[InputSubstitution]:
    input_selector = construct_input_selector(input_name=input_name)
    substitutions = []
    for step in steps:
        for field in step.model_fields:
            if getattr(step, field) != input_selector:
                continue
            substitution = InputSubstitution(
                input_parameter_name=input_name,
                step_manifest=step,
                manifest_property=field,
            )
            substitutions.append(substitution)
    return substitutions

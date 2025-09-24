import json
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Union

import networkx as nx
from packaging.version import Version

from inference.core.workflows.execution_engine.entities.base import WorkflowParameter
from inference.core.workflows.execution_engine.introspection.blocks_loader import (
    load_initializers,
    load_kinds_deserializers,
    load_kinds_serializers,
    load_workflow_blocks,
)
from inference.core.workflows.execution_engine.profiling.core import (
    WorkflowsProfiler,
    execution_phase,
)
from inference.core.workflows.execution_engine.v1.compiler.cache import (
    BasicWorkflowsCache,
)
from inference.core.workflows.execution_engine.v1.compiler.entities import (
    BlockSpecification,
    CompiledWorkflow,
    InputSubstitution,
    ParsedWorkflowDefinition,
)
from inference.core.workflows.execution_engine.v1.compiler.graph_constructor import (
    prepare_execution_graph,
)
from inference.core.workflows.execution_engine.v1.compiler.steps_initialiser import (
    initialise_steps,
)
from inference.core.workflows.execution_engine.v1.compiler.syntactic_parser import (
    parse_workflow_definition,
)
from inference.core.workflows.execution_engine.v1.compiler.utils import (
    construct_input_selector,
)
from inference.core.workflows.execution_engine.v1.compiler.validator import (
    validate_workflow_specification,
)
from inference.core.workflows.execution_engine.v1.debugger.core import (
    dump_execution_graph,
)
from inference.core.workflows.execution_engine.v1.dynamic_blocks.block_assembler import (
    compile_dynamic_blocks,
    ensure_dynamic_blocks_allowed,
)
from inference.core.workflows.prototypes.block import WorkflowBlockManifest


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

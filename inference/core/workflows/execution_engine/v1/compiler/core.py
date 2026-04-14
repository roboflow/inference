import json
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Union

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
    GraphCompilationResult,
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
from inference.core.workflows.execution_engine.v1.inner_workflow.compiler_bridge import (
    resolve_inner_workflow_steps_in_parsed_definition,
    validate_inner_workflow_composition_from_workflow_dict,
)
from inference.core.workflows.execution_engine.v1.inner_workflow.constants import (
    USE_INNER_WORKFLOW_BLOCK_TYPE,
)
from inference.core.workflows.execution_engine.v1.inner_workflow.reference_resolution import (
    normalize_inner_workflow_references_in_definition,
)
from inference.core.workflows.prototypes.block import WorkflowBlockManifest

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
    inner_workflows = _compile_inner_workflows(
        parsed=graph_compilation_results.parsed_workflow_definition,
        init_parameters=init_parameters,
        execution_engine_version=execution_engine_version,
        profiler=profiler,
    )
    return CompiledWorkflow(
        workflow_definition=graph_compilation_results.parsed_workflow_definition,
        workflow_json=workflow_definition,
        init_parameters=init_parameters,
        execution_graph=graph_compilation_results.execution_graph,
        steps=steps_by_name,
        input_substitutions=input_substitutions,
        kinds_serializers=graph_compilation_results.kinds_serializers,
        kinds_deserializers=graph_compilation_results.kinds_deserializers,
        inner_workflows=inner_workflows,
    )


def _compile_inner_workflows(
    parsed: ParsedWorkflowDefinition,
    init_parameters: Dict[str, Union[Any, Callable[[None], Any]]],
    execution_engine_version: Optional[Version],
    profiler: Optional[WorkflowsProfiler],
) -> Dict[str, CompiledWorkflow]:
    inner: Dict[str, CompiledWorkflow] = {}
    for sm in parsed.steps:
        if getattr(sm, "type", None) != USE_INNER_WORKFLOW_BLOCK_TYPE:
            continue
        inner[sm.name] = compile_workflow(
            workflow_definition=sm.workflow,
            init_parameters=init_parameters,
            execution_engine_version=execution_engine_version,
            profiler=profiler,
        )
    return inner


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
    workflow_for_compile = normalize_inner_workflow_references_in_definition(
        workflow_definition=workflow_definition,
        init_parameters=init_parameters,
    )
    statically_defined_blocks = load_workflow_blocks(
        execution_engine_version=execution_engine_version,
        profiler=profiler,
    )
    initializers = load_initializers(profiler=profiler)
    kinds_serializers = load_kinds_serializers(profiler=profiler)
    kinds_deserializers = load_kinds_deserializers(profiler=profiler)
    dynamic_blocks = compile_dynamic_blocks(
        dynamic_blocks_definitions=workflow_for_compile.get(
            "dynamic_blocks_definitions", []
        ),
        profiler=profiler,
        api_key=init_parameters.get("workflows_core.api_key", None),
    )
    available_blocks = statically_defined_blocks + dynamic_blocks
    parsed_workflow_definition = parse_workflow_definition(
        raw_workflow_definition=workflow_for_compile,
        available_blocks=available_blocks,
        profiler=profiler,
    )
    validate_workflow_specification(
        workflow_definition=parsed_workflow_definition,
        profiler=profiler,
    )
    validate_inner_workflow_composition_from_workflow_dict(workflow_for_compile)
    parsed_workflow_definition = resolve_inner_workflow_steps_in_parsed_definition(
        parsed=parsed_workflow_definition,
        raw_workflow_definition=workflow_for_compile,
        compile_workflow_graph_fn=compile_workflow_graph,
        available_blocks=available_blocks,
        execution_engine_version=execution_engine_version,
        init_parameters=init_parameters,
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

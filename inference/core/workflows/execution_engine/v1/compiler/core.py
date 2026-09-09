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
    validate_inner_workflow_composition_from_raw_workflow_definition,
)
from inference.core.workflows.execution_engine.v1.inner_workflow.dynamic_blocks_collection import (
    apply_collected_dynamic_blocks_definitions_to_workflow_root,
    collect_dynamic_blocks_definitions_from_workflow_definition,
)
from inference.core.workflows.execution_engine.v1.inner_workflow.inline import (
    inline_inner_workflow_steps,
)
from inference.core.workflows.execution_engine.v1.inner_workflow.reference_resolution import (
    normalize_inner_workflow_references_in_definition,
    workflow_definition_contains_unresolved_inner_workflow_reference,
)
from inference.core.workflows.prototypes.block import WorkflowBlockManifest
from inference.core.workflows.prototypes.workspace_resolver import (
    NULL_WORKSPACE_RESOLVER,
)

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


def _effective_workspace_resolver(
    init_parameters: Dict[str, Union[Any, Callable[[None], Any]]],
):
    """The resolver the GENERATED BLOCK will use, so compilation-time Modal
    validation and runtime execution cannot disagree.

    `ExecutionEngineV1.init` mirrors the effective value into
    `dynamic_workflows_blocks.workspace_resolver`, preserving an explicit
    override in that namespace; reading `workflows_core.*` here would consult a
    different object (round-2 defect 5).
    """
    return init_parameters.get(
        "dynamic_workflows_blocks.workspace_resolver",
        init_parameters.get(
            "workflows_core.workspace_resolver", NULL_WORKSPACE_RESOLVER
        ),
    )


def _effective_dynamic_api_key(
    init_parameters: Dict[str, Union[Any, Callable[[None], Any]]],
) -> Optional[str]:
    """The api key the GENERATED BLOCK will hold (`steps_initialiser` prefers
    `dynamic_workflows_blocks.api_key`, which the engine mirrors from
    `workflows_core.api_key` unless a caller set it explicitly). Compile-time
    Modal validation resolves the workspace with this key, so it names the
    same sandbox the block later executes in (round-5 defect 2)."""
    return init_parameters.get(
        "dynamic_workflows_blocks.api_key",
        init_parameters.get("workflows_core.api_key"),
    )


def _is_resolver_dependent(
    workflow_definition: dict, dynamic_blocks_definitions: List[dict]
) -> bool:
    """True when compiling this definition consults an injected resolver.

    Compilation resolves inner-workflow references and compiles dynamic blocks;
    both consult resolvers, and `COMPILATION_CACHE` keys on neither them nor the
    api key. Rather than invent a lifetime-safe context, such definitions simply
    are not cached - they are the rare case, and the cache exists for the plain
    ones. This also removes the authentication-context hazard for exactly the
    definitions where it mattered.
    """
    if dynamic_blocks_definitions:
        return True
    return workflow_definition_contains_unresolved_inner_workflow_reference(
        workflow_definition=workflow_definition
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
    init_parameters: Optional[Dict[str, Union[Any, Callable[[None], Any]]]] = None,
) -> GraphCompilationResult:
    if init_parameters is None:
        init_parameters = {}
    pre_resolution_dynamic_blocks_definitions = (
        collect_dynamic_blocks_definitions_from_workflow_definition(
            workflow_definition=workflow_definition
        )
    )
    cacheable = not _is_resolver_dependent(
        workflow_definition=workflow_definition,
        dynamic_blocks_definitions=pre_resolution_dynamic_blocks_definitions,
    )
    key = COMPILATION_CACHE.get_hash_key(
        workflow_definition=workflow_definition,
        execution_engine_version=execution_engine_version,
    )
    cached_value = COMPILATION_CACHE.get(key=key) if cacheable else None
    if cached_value is not None:
        ensure_dynamic_blocks_allowed(
            dynamic_blocks_definitions=pre_resolution_dynamic_blocks_definitions
        )
        return cached_value

    raw_workflow_definition: Dict[str, Any] = (
        normalize_inner_workflow_references_in_definition(
            workflow_definition=workflow_definition,
            init_parameters=init_parameters,
        )
    )
    dynamic_blocks_definitions = (
        apply_collected_dynamic_blocks_definitions_to_workflow_root(
            workflow_definition=raw_workflow_definition,
        )
    )
    statically_defined_blocks = load_workflow_blocks(
        execution_engine_version=execution_engine_version,
        profiler=profiler,
    )
    initializers = load_initializers(profiler=profiler)
    kinds_serializers = load_kinds_serializers(profiler=profiler)
    kinds_deserializers = load_kinds_deserializers(profiler=profiler)
    dynamic_blocks = compile_dynamic_blocks(
        dynamic_blocks_definitions=dynamic_blocks_definitions,
        profiler=profiler,
        api_key=_effective_dynamic_api_key(init_parameters),
        workspace_resolver=_effective_workspace_resolver(init_parameters),
    )
    available_blocks = statically_defined_blocks + dynamic_blocks
    validate_inner_workflow_composition_from_raw_workflow_definition(
        raw_workflow_definition
    )
    inlined_raw_workflow_definition: Dict[str, Any] = inline_inner_workflow_steps(
        raw_workflow_definition,
        available_blocks=available_blocks,
        profiler=profiler,
    )
    parsed_workflow_definition = parse_workflow_definition(
        raw_workflow_definition=inlined_raw_workflow_definition,
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
    if cacheable:
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

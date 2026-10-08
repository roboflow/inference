"""Compiler of the V2 execution engine.

``compile_workflow`` turns a ``version: "2.0"`` definition into a validated,
inspectable ``CompiledWorkflow``. It never constructs blocks or sources, calls
resource providers or executes submitted Python::

    definition ──▶ definition.parse_workflow     structure, inputs → layouts,
                                                 sources, output groups
               ──▶ composition.compose_workflow  nested scopes, saved references,
                                                 limits, dynamic definitions
               ──▶ dynamic_blocks.build_dynamic_catalogue   (only when defined)
               ──▶ sources.plan_sources          source params, static bindings,
                                                 scoped port layouts
               ──▶ compiler.compile_composition  bindings, order, layouts, gates,
                                                 causal domains, groups,
                                                 mutation analysis, quality
               ──▶ demand.apply_demand           requested outputs, recorded
                                                 groups, prunable steps
               ──▶ controls.compile_controls     root controls: closures, state
                                                 policies (only when declared)
               ──▶ recording.compilation         root recording/retrospective
                                                 declarations (only when set)
               ──▶ CompiledWorkflow

Example::

    plan = compile_workflow(definition, catalogue=catalogue)
    session = plan.create_session()
    result = session.run({"image": frame})
"""

import dataclasses
import importlib
from typing import Any, Mapping, Optional

from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.compilation.compiler import (
    compile_composition,
)
from roboflow_workflows.execution_engine.v2.compilation.demand import (
    DemandPlan,
    apply_demand,
)
from roboflow_workflows.execution_engine.v2.compilation.composition import (
    Composition,
    ReferenceResolver,
    compose_workflow,
)
from roboflow_workflows.execution_engine.v2.compilation.controls import (
    compile_controls,
    control_members,
)
from roboflow_workflows.execution_engine.v2.compilation.definition import (
    NESTED_WORKFLOW_TYPES,
    ROOT_AXIS,
    SUPPORTED_VERSION,
    WorkflowReference,
)
from roboflow_workflows.execution_engine.v2.errors import (
    ContractError,
    WorkflowCompileError,
)
from roboflow_workflows.execution_engine.v2.plan import CompiledWorkflow, CompileOptions

DYNAMIC_BLOCKS_MODULE = "roboflow_workflows.execution_engine.v2.dynamic_blocks"
RECORDING_COMPILATION_MODULE = (
    "roboflow_workflows.execution_engine.v2.recording.compilation"
)

__all__ = [
    "NESTED_WORKFLOW_TYPES",
    "ROOT_AXIS",
    "SUPPORTED_VERSION",
    "DemandPlan",
    "ReferenceResolver",
    "WorkflowReference",
    "apply_demand",
    "compile_workflow",
]


def compile_workflow(
    definition: Mapping[str, Any],
    *,
    catalogue: Catalogue,
    options: CompileOptions = CompileOptions(),
    reference_resolver: Optional[ReferenceResolver] = None,
) -> CompiledWorkflow:
    """Compile a V2 workflow definition against an explicit catalogue.

    Args:
        definition: Workflow definition with ``"version": "2.0"``. It is
            copied, never modified.
        catalogue: Blocks and kinds the definition may use.
        options: Nesting limits, mutation conflict policy and whether dynamic
            blocks may later execute submitted code.
        reference_resolver: Called with a ``WorkflowReference`` for each
            distinct saved child workflow; returns its definition. Each
            reference is fetched at most once per compilation.

    Returns:
        The validated plan. Its warnings list mutation conflicts (under the
        default ``warn`` policy) and dropped dynamic duplicates. A root
        ``recording`` declaration sets ``plan.recording``; a root
        ``retrospective`` declaration is compiled beside the plan into
        ``plan.retrospective``. ``plan.demand`` records which steps the
        requested outputs (``options.requested_outputs``) kept and dropped.

    Raises:
        WorkflowCompileError: A subclass naming the step path, field path and
            reason: ``SelectorError``, ``UnknownBlockError``,
            ``ParamsValidationError``, ``KindMismatchError``, ``LineageError``,
            ``CycleError``, ``NestedWorkflowError``,
            ``MutationConflictError``, ``DemandError``,
            ``UnsupportedQualityError`` or ``RecordingDefinitionError``.
    """
    plan = compile_definition(
        definition,
        catalogue=catalogue,
        options=options,
        reference_resolver=reference_resolver,
    )

    return plan


def compile_definition(
    definition: Mapping[str, Any],
    *,
    catalogue: Catalogue,
    options: CompileOptions,
    reference_resolver: Optional[ReferenceResolver],
    location: str = "",
) -> CompiledWorkflow:
    """``compile_workflow`` with a definition path prefix for messages.

    Args:
        definition: Workflow definition with ``"version": "2.0"``.
        catalogue: Blocks and kinds the definition may use.
        options: Compile options.
        reference_resolver: Resolver of saved child workflows.
        location: Path prefix of the definition, ``""`` for a root; a
            prefixed definition cannot declare root-only sections.

    Returns:
        The validated plan.
    """
    if not isinstance(catalogue, Catalogue):
        raise WorkflowCompileError(
            f"catalogue must be a Catalogue, got {type(catalogue).__name__}"
        )
    if not isinstance(options, CompileOptions):
        raise WorkflowCompileError(
            f"options must be CompileOptions, got {type(options).__name__}"
        )

    composition = compose_workflow(
        definition,
        options=options,
        reference_resolver=reference_resolver,
        location=location,
    )
    full_catalogue = _with_dynamic_blocks(
        catalogue, composition=composition, options=options
    )
    plan = compile_composition(composition, catalogue=full_catalogue, options=options)
    root = composition.root.workflow
    recording_compilation = None
    recorded_groups = ()
    if root.recording is not None or root.retrospective is not None:
        # Imported only for definitions that record or analyse a recording, so
        # ordinary compilation skips the stage compilers (retrospective, replay,
        # results). The recording package itself is already loaded: the
        # catalogue imports its codecs.
        recording_compilation = importlib.import_module(RECORDING_COMPILATION_MODULE)
    if root.recording is not None:
        recorded_groups = recording_compilation.recorded_group_names(
            root.recording, plan=plan
        )
    # Control members survive demand narrowing even when initially disabled,
    # so re-enabling them needs no new plan.
    members = control_members(plan, root.controls)
    plan = apply_demand(
        plan,
        requested=options.requested_outputs,
        recorded_groups=recorded_groups,
        retained=[
            (path, f"member of control {name!r}")
            for name, paths in members.items()
            for path in paths
        ],
    )
    plan = compile_controls(
        plan,
        root.controls,
        members=members,
        recorded_groups=recorded_groups,
        recording=root.recording,
    )
    if recording_compilation is None:
        return plan

    # The retrospective workflow has its own outputs; the request applies to
    # the primary plan only.
    staged = recording_compilation.compile_stages(
        plan,
        definition=definition,
        recording=root.recording,
        retrospective=root.retrospective,
        catalogue=catalogue,
        options=dataclasses.replace(options, requested_outputs=None),
        reference_resolver=reference_resolver,
    )

    return staged


def _with_dynamic_blocks(
    catalogue: Catalogue, *, composition: Composition, options: CompileOptions
) -> Catalogue:
    if not composition.dynamic_definitions:
        return catalogue

    # Imported only when a definition declares dynamic blocks, so static
    # compilation does not depend on the dynamic-block machinery.
    # Each definition is built on its own, so an error names its structural
    # location in the (possibly nested) definition.
    dynamic_blocks = importlib.import_module(DYNAMIC_BLOCKS_MODULE)
    dynamic_catalogues = []
    for item in composition.dynamic_definitions:
        try:
            dynamic_catalogues.append(
                dynamic_blocks.build_dynamic_catalogue(
                    [item.definition],
                    catalogue=catalogue,
                    allow_local_code=options.allow_local_code,
                )
            )
        except dynamic_blocks.DynamicBlockError as error:
            raise dynamic_blocks.DynamicBlockError(
                str(error), location=item.location
            ) from error
    try:
        merged = Catalogue.merge(catalogue, *dynamic_catalogues)
    except ContractError as error:
        raise WorkflowCompileError(
            f"Dynamic block definitions conflict with the catalogue: {error}"
        ) from error

    return merged

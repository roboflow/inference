"""Compiler of the V2 execution engine.

``compile_workflow`` turns a ``version: "2.0"`` definition into a validated,
inspectable ``CompiledWorkflow``. It never constructs blocks, calls resource
providers or executes submitted Python::

    definition ──▶ definition.parse_workflow     structure, inputs → layouts
               ──▶ composition.compose_workflow  nested scopes, saved references,
                                                 limits, dynamic definitions
               ──▶ dynamic_blocks.build_dynamic_catalogue   (only when defined)
               ──▶ compiler.compile_composition  bindings, order, layouts, gates,
                                                 mutation analysis
               ──▶ CompiledWorkflow

Example::

    plan = compile_workflow(definition, catalogue=catalogue)
    session = plan.create_session()
    result = session.run({"image": frame})
"""

import importlib
from typing import Any, Mapping, Optional

from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.compilation.compiler import (
    compile_composition,
)
from roboflow_workflows.execution_engine.v2.compilation.composition import (
    Composition,
    ReferenceResolver,
    compose_workflow,
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
from roboflow_workflows.execution_engine.v2.plan import (
    CompiledWorkflow,
    CompileOptions,
)

DYNAMIC_BLOCKS_MODULE = "roboflow_workflows.execution_engine.v2.dynamic_blocks"

__all__ = [
    "NESTED_WORKFLOW_TYPES",
    "ROOT_AXIS",
    "SUPPORTED_VERSION",
    "ReferenceResolver",
    "WorkflowReference",
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
        default ``warn`` policy) and dropped dynamic duplicates.

    Raises:
        WorkflowCompileError: A subclass naming the step path, field path and
            reason: ``SelectorError``, ``UnknownBlockError``,
            ``ParamsValidationError``, ``KindMismatchError``, ``LineageError``,
            ``CycleError``, ``NestedWorkflowError`` or
            ``MutationConflictError``.
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
    )
    full_catalogue = _with_dynamic_blocks(
        catalogue, composition=composition, options=options
    )
    plan = compile_composition(composition, catalogue=full_catalogue, options=options)

    return plan


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

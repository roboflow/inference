"""Build a local inner-workflow declaration using the existing compiler contract.

The historical file name is retained for the scoped candidate. This is a
definition helper, with no runtime block, dispatcher, registration or transport.
The compiler embeds inline children or calls the supplied saved-workflow resolver
at compilation; the parent session owns execution and resource lifetime.
"""

from copy import deepcopy
from typing import Any, Mapping, Optional

from roboflow_workflows.v2.engine.compilation import WorkflowReference
from roboflow_workflows.v2.engine.compilation.definition import parse_workflow


def inner_workflow_step(
    name: str,
    *,
    workflow_definition: Optional[Mapping[str, Any]] = None,
    reference: Optional[WorkflowReference] = None,
    parameter_bindings: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Build a detached embedded child step for a parent V2 definition.

    Args:
        name: Child name in the parent scope.
        workflow_definition: Inline child definition, exclusive with reference.
        reference: Saved child identity; the caller supplies its resolver when
            compiling the parent. This helper never calls the resolver.
        parameter_bindings: Child input names to parent selectors or literals.

    Returns:
        A V1-shaped step accepted by the existing local V2 compiler. Ordinary
        mutable data is copied; Secret carriers retain their existing semantics.

    Raises:
        WorkflowCompileError: Invalid name, bindings or child source. The parent
            compiler later validates child inputs, outputs, kinds and limits.
    """
    step = {
        "type": "roboflow_core/inner_workflow@v1",
        "name": name,
        "execution_mode": "embedded",
        "parameter_bindings": deepcopy(
            {} if parameter_bindings is None else parameter_bindings
        ),
    }
    if workflow_definition is not None:
        step["workflow_definition"] = deepcopy(workflow_definition)

    if reference is not None:
        step.update(
            workflow_workspace_id=reference.workspace_id,
            workflow_id=reference.workflow_id,
            workflow_version_id=reference.version_id,
        )

    # Use the same structural validation as hand-written workflow definitions.
    parse_workflow({"version": "2.0", "inputs": [], "steps": [step], "outputs": []})

    return step

"""Naming and resolving plan sources, shared by the introspection modules.

A source is a workflow input, a step output, a compile-time constant, a
nested workflow's input port (decision 021) or a gated child output port
(decision 026). A port forwards another source, possibly another port for
deeper nesting::

    $steps.outer/inner: $inputs.x --source--> $steps.outer: $inputs.x --source--> $inputs.x
    $steps.outer.out --source--> $steps.outer: $inputs.x
"""

from typing import Any, Optional

from roboflow_workflows.execution_engine.v2.errors import format_step_path
from roboflow_workflows.execution_engine.v2.plan import (
    ChildInputPort,
    ChildOutputPort,
    CompiledWorkflow,
    Constant,
    InputPort,
    StepPort,
)


def origin_of(plan: CompiledWorkflow, source: Any) -> Any:
    """Follow child ports to the input, step output or constant behind them.

    Args:
        plan: Compiled plan holding the child inputs and outputs.
        source: Any plan source.

    Returns:
        The first source that is not a child input or output port.
    """
    origin = plan.origin(source)

    return origin


def node_of(source: Any) -> Optional[str]:
    """Node id of a source; ``None`` for a constant, which has no node.

    Args:
        source: Any plan source.

    Returns:
        ``$inputs.<name>``, the producing step's node id, or a child input
        port such as ``$steps.child: $inputs.image``.
    """
    if isinstance(source, Constant):
        return None
    if isinstance(source, StepPort):
        return format_step_path(source.step)

    node = source.describe()

    return node


def output_of(source: Any) -> Optional[str]:
    """Step output name for step sources, else ``None``."""
    if isinstance(source, StepPort):
        return source.output

    return None


def selector_of(source: Any) -> Optional[str]:
    """Selector text a parent definition uses to bind ``source``; ``None`` for constants."""
    if isinstance(source, (InputPort, StepPort)):
        return source.describe()
    if isinstance(source, ChildInputPort):
        return f"$inputs.{source.name}"
    if isinstance(source, ChildOutputPort):
        return f"$steps.{source.scope[-1]}.{source.name}"

    return None

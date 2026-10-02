"""Naming and resolving plan sources, shared by the introspection modules.

A source is a workflow input, a step output, a declared source's port, a
compile-time constant, a nested workflow's input port (decision 021) or a
gated child output port (decision 026). A port forwards another source,
possibly another port for deeper nesting::

    $steps.outer/inner: $inputs.x --source--> $steps.outer: $inputs.x --source--> $inputs.x
    $steps.outer.out --source--> $steps.outer: $inputs.x
    $steps.detect.image --source--> $sources.camera (output "image")
    $steps.best.image --source--> $operators.clip (output "frames")
"""

from typing import Any, Optional

from roboflow_workflows.execution_engine.v2.errors import format_step_path
from roboflow_workflows.execution_engine.v2.plan import (
    ChildInputPort,
    ChildOutputPort,
    CompiledWorkflow,
    Constant,
    InputPort,
    SourcePort,
    StepPort,
)


def source_node(name: str) -> str:
    """Node id of a declared source, ``$sources.<name>``.

    Args:
        name: Declared source name.

    Returns:
        The node id.
    """
    return f"$sources.{name}"


def operator_node(name: str) -> str:
    """Node id of a declared operator, ``$operators.<name>``.

    Args:
        name: Declared operator name.

    Returns:
        The node id.
    """
    return f"$operators.{name}"


def domain_node(port: SourcePort) -> str:
    """Node id of the source or operator emitting ``port``.

    Args:
        port: Source or operator port.

    Returns:
        ``$sources.<name>`` or ``$operators.<name>``.
    """
    if port.origin == "operator":
        return operator_node(port.source)

    return source_node(port.source)


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
        ``$inputs.<name>``, the producing step's node id, the declared
        source's or operator's node id (``$sources.<name>``,
        ``$operators.<name>``) or a child input port such as
        ``$steps.child: $inputs.image``.
    """
    if isinstance(source, Constant):
        return None
    if isinstance(source, StepPort):
        return format_step_path(source.step)
    if isinstance(source, SourcePort):
        return domain_node(source)

    node = source.describe()

    return node


def output_of(source: Any) -> Optional[str]:
    """Step output or source/operator port name for those sources, else ``None``."""
    if isinstance(source, (StepPort, SourcePort)):
        return source.output

    return None


def selector_of(source: Any) -> Optional[str]:
    """Selector text a parent definition uses to bind ``source``; ``None`` for constants."""
    if isinstance(source, (InputPort, StepPort, SourcePort)):
        return source.describe()
    if isinstance(source, ChildInputPort):
        return f"$inputs.{source.name}"
    if isinstance(source, ChildOutputPort):
        return f"$steps.{source.scope[-1]}.{source.name}"

    return None

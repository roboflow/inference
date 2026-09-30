"""Error types shared by the V2 execution engine.

The V2 runtime distinguishes three failure families:

* ``ContractError``: a data, block, catalogue or resource contract was
  violated. Raised by the data API (malformed batches, layouts or metadata),
  by block declarations, by the catalogue and when an execution session
  cannot provide or construct a block.
* ``WorkflowCompileError``: a workflow definition cannot be compiled.
* ``WorkflowExecutionError``: a compiled workflow failed while running. An
  active run reports its one terminal failure as ``ActiveRunError``.

Subclasses carry structured location data (step path, field path, index) in
addition to their message. The engine preserves an underlying exception as
``__cause__``. The three base classes accept a plain message, so older
callers that raise them with one string keep working.
"""

from typing import Any, Literal, Optional, Tuple

StepPath = Tuple[str, ...]
"""Scope path of a step, e.g. ``("child", "scale")`` for a nested step."""

FieldPath = Tuple[Any, ...]
"""Field name followed by list positions or mapping keys inside that field."""


def format_step_path(step_path: StepPath) -> str:
    """Render a step path the way error messages and descriptions show it.

    Args:
        step_path: Scope path of a step.

    Returns:
        ``"$steps.child/scale"`` style text; ``"<workflow>"`` for an empty
        path; ``"$sources.camera"`` for a source's reserved path
        ``("$sources", "camera")``.
    """
    if not step_path:
        return "<workflow>"
    if len(step_path) == 2 and step_path[0] == "$sources":
        return f"$sources.{step_path[1]}"

    rendered = "$steps." + "/".join(step_path)

    return rendered


class ContractError(ValueError):
    """A V2 data, block or registry contract was violated."""


class DeclarationError(ContractError):
    """A block class declares an invalid or inconsistent contract.

    Raised while the class body is being created, so invalid blocks fail at
    import time rather than during compilation or execution.

    Args:
        message: Human-readable explanation.
        block_class: Name of the offending class, when known.
    """

    def __init__(self, message: str, *, block_class: Optional[str] = None):
        prefix = f"Block class {block_class}: " if block_class else ""
        super().__init__(prefix + message)
        self.block_class = block_class


class CatalogueError(ContractError):
    """A catalogue cannot be built or queried (duplicates, unknown types)."""


class ResourceError(ContractError):
    """An execution session cannot provide a resource or construct a block.

    Args:
        message: Human-readable explanation.
        step_path: Step whose construction failed.
        block_type: Block type of that step.
        parameter: Constructor parameter involved, when relevant.
    """

    def __init__(
        self,
        message: str,
        *,
        step_path: StepPath = (),
        block_type: Optional[str] = None,
        parameter: Optional[str] = None,
    ):
        super().__init__(f"{format_step_path(step_path)}: {message}")
        self.step_path = step_path
        self.block_type = block_type
        self.parameter = parameter


class ResolvedParameterError(ContractError):
    """A resolved parameter value violates the block's ``Params`` declaration.

    Raised by ``BlockSpec.validate_resolved_arguments``; the executor wraps it
    into ``StepExecutionError`` with this error as the cause.

    Args:
        message: Human-readable explanation.
        field_path: Parameter name followed by list positions or dict keys;
            empty for a model-level validator failure.
    """

    def __init__(self, message: str, *, field_path: FieldPath = ()):
        super().__init__(message)
        self.field_path = field_path


class WorkflowCompileError(ValueError):
    """A V2 workflow definition could not be compiled.

    Args:
        message: Human-readable explanation.
        step_path: Step the problem belongs to; empty for workflow-level issues.
        field_path: Parameter name and position inside it, when relevant.
    """

    def __init__(
        self,
        message: str,
        *,
        step_path: StepPath = (),
        field_path: FieldPath = (),
    ):
        super().__init__(message)
        self.step_path = step_path
        self.field_path = field_path


class ParamsValidationError(WorkflowCompileError):
    """Step parameters do not satisfy the block's ``Params`` model."""


class SelectorError(WorkflowCompileError):
    """A selector is malformed, unknown or used where it is not allowed."""


class UnknownBlockError(WorkflowCompileError):
    """A step names a block type the catalogue does not contain."""


class KindMismatchError(WorkflowCompileError):
    """A bound value's kinds are incompatible with the consuming field."""


class LineageError(WorkflowCompileError):
    """Bound values or control decisions do not share a compatible lineage."""


class CycleError(WorkflowCompileError):
    """Step dependencies or nested workflow references form a cycle."""


class NestedWorkflowError(WorkflowCompileError):
    """A nested workflow cannot be composed (bindings, outputs, limits)."""


class MutationConflictError(WorkflowCompileError):
    """Declared in-place mutations conflict under strict mutation handling."""


class WorkflowExecutionError(RuntimeError):
    """A compiled V2 workflow failed during execution."""


class StepExecutionError(WorkflowExecutionError):
    """A block raised, or returned a result that violates its declaration.

    Args:
        message: Human-readable explanation.
        step_path: Failing step.
        block_type: Block type of that step.
        index: Logical invocation index; ``None`` for a vectorized call.
    """

    def __init__(
        self,
        message: str,
        *,
        step_path: StepPath,
        block_type: str,
        index: Optional[Tuple[int, ...]] = None,
    ):
        location = format_step_path(step_path)
        if index is not None:
            location = f"{location} at index {list(index)}"
        super().__init__(f"{location} ({block_type}): {message}")
        self.step_path = step_path
        self.block_type = block_type
        self.index = index


class WorkflowInputError(WorkflowExecutionError):
    """Workflow inputs failed preparation, validation or deserialization."""


ActiveRunStage = Literal[
    "start", "open", "read", "emission", "step", "handler", "observer", "close"
]


class ActiveRunError(WorkflowExecutionError):
    """An active run failed; the one terminal error of that run.

    The first failure of a run becomes its ``ActiveRunError`` and keeps the
    original exception as ``__cause__``. Errors raised while the run cleans
    up after that failure (for example a source ``close`` that also raises)
    are appended to ``suppressed`` so they never disappear.

    Args:
        message: Human-readable explanation.
        stage: Where the run failed: ``start`` (validation or construction
            before acquisition), ``open``, ``read``, ``emission`` (an
            emission violating the source's declaration), ``step``,
            ``handler``, ``observer`` (a session observer callback raised)
            or ``close``.
        source: Declared name of the source involved, when any.
        pulse: Sequence number of the pulse involved, when any.
        group: Output group whose handler failed, when any.
        step_path: Failing step, when a step failed.
    """

    def __init__(
        self,
        message: str,
        *,
        stage: ActiveRunStage,
        source: Optional[str] = None,
        pulse: Optional[int] = None,
        group: Optional[str] = None,
        step_path: Optional[StepPath] = None,
    ):
        where = [f"stage {stage}"]
        if source is not None:
            where.append(f"source {source!r}")
        if pulse is not None:
            where.append(f"pulse {pulse}")
        if group is not None:
            where.append(f"group {group!r}")
        if step_path is not None:
            where.append(format_step_path(step_path))
        super().__init__(f"Active run failed ({', '.join(where)}): {message}")
        self.stage = stage
        self.source = source
        self.pulse = pulse
        self.group = group
        self.step_path = step_path
        self.suppressed: Tuple[BaseException, ...] = ()

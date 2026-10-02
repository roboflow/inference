"""Compile targets and the selection of a block's implementation.

A logical block may list alternative implementations (``Block.implementations``);
an ordinary block is its own single ``default`` implementation. Compilation picks
one per step for the caller's ``Target``::

    plan = compile_workflow(
        definition,
        catalogue=catalogue,
        options=CompileOptions(target=Target(frozenset({"cpu", "torch"}))),
    )
    plan.step(("classify",)).selected.name        # e.g. "torch-cpu"

Selection is a subset check in declared order: the first implementation whose
``requires`` are all among the target's capabilities wins. Nothing is imported,
probed or loaded, and step parameters never take part: a target is the caller's
claim about the intended environment, not hardware attestation. Every
implementation is reported with the requirements the target misses, so the plan
explains why the winner won and why the others did not.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, Iterable, List, Tuple

from roboflow_workflows.execution_engine.v2.errors import (
    ContractError,
    StepPath,
    WorkflowCompileError,
    format_step_path,
)

if TYPE_CHECKING:
    from roboflow_workflows.execution_engine.v2.declaration import BlockSpec
    from roboflow_workflows.execution_engine.v2.implementations import (
        ImplementationSpec,
    )

__all__ = [
    "ImplementationChoice",
    "Target",
    "UnsupportedTargetError",
    "check_choice",
    "consider",
    "default_implementation",
    "select_implementation",
]

Considered = Tuple[Tuple[str, Tuple[str, ...]], ...]
"""Implementation name and the requirements a target misses, in declared order."""


@dataclass(frozen=True)
class Target:
    """Capabilities the caller declares for the environment a plan will run in.

    Args:
        capabilities: Capability names such as ``"cpu"``, ``"torch"`` or
            ``"mps"``. Any iterable of non-empty strings; stored as a frozenset.

    Raises:
        ContractError: When ``capabilities`` is a single string or holds
            anything other than non-empty strings.
    """

    capabilities: FrozenSet[str]

    def __post_init__(self) -> None:
        capabilities = self.capabilities
        if isinstance(capabilities, str) or not isinstance(capabilities, Iterable):
            raise ContractError(
                "Target capabilities must be a collection of names, got "
                f"{capabilities!r}"
            )

        names = tuple(capabilities)
        invalid = [name for name in names if not isinstance(name, str) or not name]
        if invalid:
            raise ContractError(
                f"Target capabilities must be non-empty strings, got {invalid!r}"
            )

        object.__setattr__(self, "capabilities", frozenset(names))

    @classmethod
    def cpu(cls) -> "Target":
        """Return the default target, ``{"cpu"}``."""
        target = cls(frozenset({"cpu"}))

        return target

    def missing(self, requires: Iterable[str]) -> Tuple[str, ...]:
        """Return the requirements this target lacks, sorted.

        Args:
            requires: Requirements of one implementation.

        Returns:
            Missing capability names; empty when the target satisfies all.
        """
        missing = tuple(sorted(set(requires) - self.capabilities))

        return missing

    def describe(self) -> List[str]:
        """Return the capabilities as a sorted list."""
        return sorted(self.capabilities)


@dataclass(frozen=True)
class ImplementationChoice:
    """The implementation compilation selected for one step, and why.

    Args:
        spec: The selected implementation, one of ``BlockSpec.implementations``.
        target: Target the selection was made for.
        considered: Every implementation of the block in declared order with
            the requirements ``target`` misses; the selected one is the first
            with none missing.
    """

    spec: "ImplementationSpec"
    target: Target
    considered: Considered

    def __post_init__(self) -> None:
        considered = tuple((name, tuple(missing)) for name, missing in self.considered)
        object.__setattr__(self, "considered", considered)

    @property
    def name(self) -> str:
        """Name of the selected implementation."""
        return self.spec.name

    def describe(self) -> Dict[str, Any]:
        """Return a JSON-friendly description of the selection."""
        description = {
            "name": self.name,
            "target": self.target.describe(),
            "considered": [
                {"name": name, "missing": list(missing), "selected": name == self.name}
                for name, missing in self.considered
            ],
        }

        return description


class UnsupportedTargetError(WorkflowCompileError):
    """No implementation of a step's block fits the compile target.

    Args:
        message: Human-readable explanation.
        target: The compile target.
        considered: Every implementation with the requirements it misses.
        step_path: The step whose block cannot run on ``target``.
    """

    def __init__(
        self,
        message: str,
        *,
        target: Target,
        considered: Considered,
        step_path: StepPath = (),
    ):
        super().__init__(message, step_path=step_path)
        self.target = target
        self.considered = considered


def consider(block: "BlockSpec", *, target: Target) -> Considered:
    """Return every implementation of ``block`` with what ``target`` misses.

    Args:
        block: Declaration of the logical block.
        target: Compile target.

    Returns:
        ``(name, missing requirements)`` per implementation, in declared order.
    """
    considered = tuple(
        (implementation.name, target.missing(implementation.requires))
        for implementation in block.implementations
    )

    return considered


def select_implementation(
    block: "BlockSpec", *, target: Target, step_path: StepPath
) -> ImplementationChoice:
    """Select the first declared implementation of ``block`` that fits ``target``.

    Args:
        block: Declaration of the logical block.
        target: Compile target.
        step_path: Step being compiled, for error messages.

    Returns:
        The selection and the reasons for every alternative.

    Raises:
        UnsupportedTargetError: When every implementation misses a requirement.
    """
    considered = consider(block, target=target)
    for implementation, (_, missing) in zip(block.implementations, considered):
        if not missing:
            choice = ImplementationChoice(
                spec=implementation, target=target, considered=considered
            )
            return choice

    reasons = "; ".join(
        f"{name!r} needs {list(missing)}" for name, missing in considered
    )
    raise UnsupportedTargetError(
        f"{format_step_path(step_path)} ({block.type}) has no implementation for "
        f"target {target.describe()}: {reasons}",
        target=target,
        considered=considered,
        step_path=step_path,
    )


def default_implementation(block: "BlockSpec") -> "ImplementationSpec":
    """Return the implementation of an ordinary block: the block class itself.

    Args:
        block: Declaration of the block.

    Returns:
        The block's single default implementation.

    Raises:
        ContractError: When the block lists alternative implementations, so
            only an ``ImplementationChoice`` can say which one runs.
    """
    implementations = block.implementations
    if len(implementations) != 1 or (
        implementations[0].implementation_class is not block.block_class
    ):
        names = [implementation.name for implementation in implementations]
        raise ContractError(
            f"{block.type} lists implementations {names}; a planned step needs an "
            "ImplementationChoice from select_implementation"
        )

    return implementations[0]


def check_choice(block: "BlockSpec", choice: Any) -> None:
    """Check that ``choice`` is exactly what selection yields for its target.

    A hand-built or altered choice is rejected, so a plan cannot construct a
    class the block does not list or one its target does not admit.

    Args:
        block: Declaration of the logical block.
        choice: ``ImplementationChoice`` of a planned step, or ``None`` for an
            ordinary block.

    Raises:
        ContractError: When ``choice`` is missing for a block with listed
            implementations, is not an ``ImplementationChoice``, or differs
            from the deterministic selection for its target.
    """
    if choice is None:
        default_implementation(block)
        return
    if not isinstance(choice, ImplementationChoice):
        raise ContractError(
            f"implementation must be an ImplementationChoice, got "
            f"{type(choice).__name__}"
        )

    try:
        expected = select_implementation(block, target=choice.target, step_path=())
    except UnsupportedTargetError as error:
        raise ContractError(
            f"implementation {choice.name!r} cannot run on target "
            f"{choice.target.describe()}: {error}"
        ) from error
    if expected.spec is not choice.spec or expected.considered != choice.considered:
        raise ContractError(
            f"implementation {choice.name!r} with considered {list(choice.considered)} "
            f"is not the selection for target {choice.target.describe()}, which is "
            f"{expected.name!r} with considered {list(expected.considered)}"
        )

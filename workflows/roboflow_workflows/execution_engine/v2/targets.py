"""Compile targets, quality precedence and the selection of a block's implementation.

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

Quality narrows that choice. A ``QualityRequest`` carries up to three labels,
in precedence order ``step > workflow > deployment``::

    step        root ``execution.step_quality["$steps.seg"]`` of the definition
    workflow    root ``execution.quality`` of the definition
    deployment  ``CompileOptions.quality`` of the caller

The first level that names a label is tried: among the implementations the
target admits, the first (declared order) serving that label wins, and the
choice records the label and its level. A step-level label nothing serves is
an ``UnsupportedQualityError``: the author asked for it explicitly. A
workflow or deployment label nothing serves is recorded as ignored and the
next level is tried; without any served label the plain target selection
applies. Labels are literals: a quality value selects a compiled
implementation, so it is never a runtime selector. ``check_quality_label`` is
the one grammar every entry point uses (definition, ``CompileOptions``,
``QualityRequest``, ``Implementation.quality``).

A hint nothing honours is visible on the plan, not only per step:
``unused_quality_hints`` lists every workflow or deployment label no step
honoured, with the reason and the labels the plan's steps do serve, so a typo
or a hint every step overrides shows in ``plan.describe()["quality"]``.
"""

import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
)

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
    "QUALITY_LEVELS",
    "IgnoredQuality",
    "ImplementationChoice",
    "QualityChoice",
    "QualityRequest",
    "QualitySettings",
    "Target",
    "UnsupportedQualityError",
    "UnsupportedTargetError",
    "UnusedQualityHint",
    "check_choice",
    "check_quality_label",
    "consider",
    "default_implementation",
    "select_implementation",
    "unused_quality_hints",
]

Considered = Tuple[Tuple[str, Tuple[str, ...]], ...]
"""Implementation name and the requirements a target misses, in declared order."""

QUALITY_LEVELS: Tuple[str, ...] = ("step", "workflow", "deployment")
"""Quality precedence, highest first."""

_QUALITY_LABEL = re.compile(r"[A-Za-z0-9_\-]+")


def check_quality_label(
    value: Any,
    *,
    location: str,
    fail: Callable[[str], Exception] = ContractError,
) -> str:
    """Check that ``value`` is a literal quality label and return it.

    Args:
        value: The candidate label.
        location: Where the value was written, for the message.
        fail: Builds the exception to raise from a message.

    Returns:
        ``value`` unchanged.

    Raises:
        Exception: ``fail(message)`` for a selector such as ``$inputs.quality``
            or a value that is not letters, digits, ``_`` and ``-``.
    """
    if isinstance(value, str) and value.startswith("$"):
        raise fail(
            f"{location} is the selector {value!r}; a quality label selects a "
            "compiled implementation, so it must be a literal, never a runtime "
            "value"
        )
    if not isinstance(value, str) or not _QUALITY_LABEL.fullmatch(value):
        raise fail(
            f"{location} must be a quality label of letters, digits, _ or -, got "
            f"{value!r}"
        )

    return value


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
class QualityRequest:
    """The quality labels asked for at every level, for one step.

    Args:
        step: Label of the root ``execution.step_quality`` entry naming the
            step; ``None`` when the definition names none.
        workflow: Label of the root ``execution.quality``; ``None`` when the
            definition sets none.
        deployment: ``CompileOptions.quality``; ``None`` when unset.

    Raises:
        ContractError: When a label is not a literal quality label
            (``check_quality_label``).
    """

    step: Optional[str] = None
    workflow: Optional[str] = None
    deployment: Optional[str] = None

    def __post_init__(self) -> None:
        for level in QUALITY_LEVELS:
            label = getattr(self, level)
            if label is not None:
                check_quality_label(label, location=f"{level} quality")

    @property
    def is_empty(self) -> bool:
        """Whether no level asks for a label."""
        return all(getattr(self, level) is None for level in QUALITY_LEVELS)

    def levels(self) -> Tuple[Tuple[str, str], ...]:
        """Return ``(level, label)`` pairs in precedence order, unset levels skipped."""
        pairs = tuple(
            (level, getattr(self, level))
            for level in QUALITY_LEVELS
            if getattr(self, level) is not None
        )

        return pairs

    def describe(self) -> Dict[str, Optional[str]]:
        """Return a JSON-friendly description."""
        description = {level: getattr(self, level) for level in QUALITY_LEVELS}

        return description


@dataclass(frozen=True)
class QualityChoice:
    """The quality label a selection honoured, and the level that asked for it.

    Args:
        label: The served label.
        level: ``step``, ``workflow`` or ``deployment``.
    """

    label: str
    level: str

    def describe(self) -> Dict[str, str]:
        """Return a JSON-friendly description."""
        return {"label": self.label, "level": self.level}


@dataclass(frozen=True)
class IgnoredQuality:
    """A workflow or deployment label the step's block could not serve.

    Args:
        label: The requested label.
        level: ``workflow`` or ``deployment``.
        reason: Why nothing served it.
    """

    label: str
    level: str
    reason: str

    def describe(self) -> Dict[str, str]:
        """Return a JSON-friendly description."""
        return {"label": self.label, "level": self.level, "reason": self.reason}


@dataclass(frozen=True)
class QualitySettings:
    """The root ``execution`` quality settings of a compiled definition.

    Args:
        workflow: Workflow-level label, or ``None``.
        steps: Step-level labels by step path.
    """

    workflow: Optional[str] = None
    steps: Mapping[StepPath, str] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "steps",
            MappingProxyType(
                {tuple(path): label for path, label in self.steps.items()}
            ),
        )

    @property
    def is_empty(self) -> bool:
        """Whether the definition sets no quality at all."""
        return self.workflow is None and not self.steps

    def request_for(
        self, path: StepPath, *, deployment: Optional[str]
    ) -> QualityRequest:
        """Build the request of one step from these settings and the deployment label.

        Args:
            path: Step path.
            deployment: ``CompileOptions.quality``.

        Returns:
            The step's quality request.
        """
        request = QualityRequest(
            step=self.steps.get(tuple(path)),
            workflow=self.workflow,
            deployment=deployment,
        )

        return request

    def describe(self) -> Dict[str, Any]:
        """Return a JSON-friendly description."""
        description = {
            "workflow": self.workflow,
            "steps": {
                format_step_path(path): label for path, label in self.steps.items()
            },
        }

        return description


@dataclass(frozen=True)
class ImplementationChoice:
    """The implementation compilation selected for one step, and why.

    Args:
        spec: The selected implementation, one of ``BlockSpec.implementations``.
        target: Target the selection was made for.
        considered: Every implementation of the block in declared order with
            the requirements ``target`` misses; the selected one is the first
            with none missing, or the first fitting one serving the honoured
            quality label.
        request: The quality labels asked for at every level.
        selected_quality: The honoured label and its level; ``None`` when no
            level asked for a label the block serves.
        ignored: Workflow or deployment labels nothing served, in precedence
            order.
    """

    spec: "ImplementationSpec"
    target: Target
    considered: Considered
    request: QualityRequest = field(default_factory=QualityRequest)
    selected_quality: Optional[QualityChoice] = None
    ignored: Tuple[IgnoredQuality, ...] = ()

    def __post_init__(self) -> None:
        considered = tuple((name, tuple(missing)) for name, missing in self.considered)
        object.__setattr__(self, "considered", considered)
        object.__setattr__(self, "ignored", tuple(self.ignored))

    @property
    def name(self) -> str:
        """Name of the selected implementation."""
        return self.spec.name

    @property
    def quality_label(self) -> Optional[str]:
        """The honoured quality label, or ``None``."""
        selected = self.selected_quality
        label = selected.label if selected is not None else None

        return label

    def describe(self) -> Dict[str, Any]:
        """Return a JSON-friendly description of the selection.

        The ``quality`` entry appears only when some level asked for a label,
        so plans without quality settings describe exactly as before.
        """
        description = {
            "name": self.name,
            "target": self.target.describe(),
            "considered": [
                {"name": name, "missing": list(missing), "selected": name == self.name}
                for name, missing in self.considered
            ],
        }
        if not self.request.is_empty:
            description["quality"] = {
                "requested": self.request.describe(),
                "selected": (
                    self.selected_quality.describe()
                    if self.selected_quality is not None
                    else None
                ),
                "ignored": [item.describe() for item in self.ignored],
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


class UnsupportedQualityError(WorkflowCompileError):
    """A step-level quality label that no fitting implementation serves.

    Args:
        message: Human-readable explanation.
        label: The requested label.
        served: Labels served by every implementation of the block, by name.
        step_path: The step the label was set for.
    """

    def __init__(
        self,
        message: str,
        *,
        label: str,
        served: Mapping[str, Tuple[str, ...]],
        step_path: StepPath = (),
    ):
        super().__init__(message, step_path=step_path)
        self.label = label
        self.served = dict(served)


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
    block: "BlockSpec",
    *,
    target: Target,
    step_path: StepPath,
    quality_request: Optional[QualityRequest] = None,
) -> ImplementationChoice:
    """Select the implementation of ``block`` for ``target`` and ``quality_request``.

    Without a quality request, or when no level names a label the block
    serves, the first declared implementation that fits ``target`` wins.

    Args:
        block: Declaration of the logical block.
        target: Compile target.
        step_path: Step being compiled, for error messages.
        quality_request: Labels asked for at each level; ``None`` asks for
            none.

    Returns:
        The selection and the reasons for every alternative.

    Raises:
        UnsupportedTargetError: When every implementation misses a requirement.
        UnsupportedQualityError: When the step-level label is served by no
            implementation that fits the target.
    """
    request = quality_request if quality_request is not None else QualityRequest()
    considered = consider(block, target=target)
    fitting = [
        implementation
        for implementation, (_, missing) in zip(block.implementations, considered)
        if not missing
    ]
    if not fitting:
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

    ignored: List[IgnoredQuality] = []
    for level, label in request.levels():
        serving = [item for item in fitting if item.serves(label)]
        if serving:
            choice = ImplementationChoice(
                spec=serving[0],
                target=target,
                considered=considered,
                request=request,
                selected_quality=QualityChoice(label=label, level=level),
                ignored=tuple(ignored),
            )
            return choice

        served = {
            implementation.name: tuple(sorted(implementation.quality))
            for implementation in block.implementations
        }
        reason = _unserved_reason(label, served=served, fitting=fitting)
        if level == "step":
            raise UnsupportedQualityError(
                f"{format_step_path(step_path)} ({block.type}) sets quality "
                f"{label!r}, but {reason}",
                label=label,
                served=served,
                step_path=step_path,
            )
        ignored.append(IgnoredQuality(label=label, level=level, reason=reason))

    choice = ImplementationChoice(
        spec=fitting[0],
        target=target,
        considered=considered,
        request=request,
        selected_quality=None,
        ignored=tuple(ignored),
    )

    return choice


def _unserved_reason(
    label: str,
    *,
    served: Mapping[str, Tuple[str, ...]],
    fitting: List["ImplementationSpec"],
) -> str:
    """Explain why no fitting implementation serves ``label``."""
    if not any(served.values()):
        return "no implementation of the block declares quality labels"

    unfit = sorted(
        name
        for name, labels in served.items()
        if label in labels and all(item.name != name for item in fitting)
    )
    if unfit:
        return (
            f"the implementation(s) {unfit} serving {label!r} do not fit the compile "
            "target"
        )

    listed = ", ".join(
        f"{name!r} serves {list(labels)}" for name, labels in served.items()
    )

    return f"no implementation serves {label!r} ({listed})"


@dataclass(frozen=True)
class UnusedQualityHint:
    """A workflow or deployment label that no step of a plan honours.

    Args:
        label: The requested label.
        level: ``workflow`` or ``deployment``.
        reason: Why no step honours it.
        available: Labels the plan's steps can serve on the compile target,
            sorted.
    """

    label: str
    level: str
    reason: str
    available: Tuple[str, ...] = ()

    def describe(self) -> Dict[str, Any]:
        """Return a JSON-friendly description."""
        return {
            "label": self.label,
            "level": self.level,
            "reason": self.reason,
            "available": list(self.available),
        }


def unused_quality_hints(
    hints: QualityRequest,
    *,
    steps: Iterable[Tuple[StepPath, "BlockSpec", Optional[ImplementationChoice]]],
) -> Tuple[UnusedQualityHint, ...]:
    """Return the workflow and deployment hints no step honours.

    A hint stays a hint: this reports, it never rejects. A label is unused
    when no step's choice honours it at its level, either because no step
    serves it (a typo, a model without that variant) or because every step
    serving it honours a higher-precedence label.

    Args:
        hints: The workflow and deployment labels of the plan.
        steps: Path, block declaration and implementation choice per step.

    Returns:
        One entry per unused hint, in precedence order.
    """
    servable: Dict[StepPath, FrozenSet[str]] = {}
    honoured = set()
    for path, block, choice in steps:
        fitting = (
            {name for name, missing in choice.considered if not missing}
            if choice is not None
            else {implementation.name for implementation in block.implementations}
        )
        servable[path] = frozenset(
            label
            for implementation in block.implementations
            if implementation.name in fitting
            for label in implementation.quality
        )
        if choice is not None and choice.selected_quality is not None:
            honoured.add(choice.selected_quality)
    available = tuple(sorted(set().union(*servable.values())))

    unused = []
    for level, label in hints.levels():
        if level == "step" or QualityChoice(label=label, level=level) in honoured:
            continue
        serving = [path for path, labels in servable.items() if label in labels]
        if serving:
            reason = (
                f"every step serving {label!r} "
                f"({[format_step_path(path) for path in serving]}) honours a "
                "higher-precedence label"
            )
        else:
            reason = f"no step serves {label!r} on the compile target"
        unused.append(
            UnusedQualityHint(
                label=label, level=level, reason=reason, available=available
            )
        )

    return tuple(unused)


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
    """Check that ``choice`` is exactly what selection yields for its inputs.

    A hand-built or altered choice is rejected, so a plan cannot construct a
    class the block does not list, one its target does not admit, or one
    its quality request did not select.

    Args:
        block: Declaration of the logical block.
        choice: ``ImplementationChoice`` of a planned step, or ``None`` for an
            ordinary block.

    Raises:
        ContractError: When ``choice`` is missing for a block with listed
            implementations, is not an ``ImplementationChoice``, or differs
            from the deterministic selection for its target and request.
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
        expected = select_implementation(
            block,
            target=choice.target,
            step_path=(),
            quality_request=choice.request,
        )
    except UnsupportedTargetError as error:
        raise ContractError(
            f"implementation {choice.name!r} cannot run on target "
            f"{choice.target.describe()}: {error}"
        ) from error
    except UnsupportedQualityError as error:
        raise ContractError(
            f"implementation {choice.name!r} cannot be selected for quality request "
            f"{choice.request.describe()}: {error}"
        ) from error
    if (
        expected.spec is not choice.spec
        or expected.considered != choice.considered
        or expected.selected_quality != choice.selected_quality
        or expected.ignored != choice.ignored
    ):
        raise ContractError(
            f"implementation {choice.name!r} with considered {list(choice.considered)} "
            f"and quality {choice.describe().get('quality')} is not the selection "
            f"for target {choice.target.describe()} and request "
            f"{choice.request.describe()}, which is {expected.name!r} with "
            f"considered {list(expected.considered)} and quality "
            f"{expected.describe().get('quality')}"
        )

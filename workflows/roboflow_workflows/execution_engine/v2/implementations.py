"""Alternative implementations of one logical block.

The logical block owns the contract exactly once: ``type``, ``Params``,
``outputs``, mutation, batching, aliases and documentation. It lists small
runnable classes in preference order; compilation selects one per step for
the caller's target (``targets.select_implementation``)::

    class TorchCpu(Implementation):
        name = "torch-cpu"
        requires = ("cpu", "torch")

        def __init__(self, *, weights_path): ...     # resources of this choice only
        @phase
        def tensor(self, *, image): ...              # optional phases
        def run(self, *, image): ...                 # explicit composition

    class TorchMps(TorchCpu):                        # shares the phases
        name = "torch-mps"
        requires = ("mps", "torch")

    class Classify(Block):
        type = "demo/classify@v1"
        class Params(BlockParams): ...
        outputs = {...}
        implementations = (TorchMps, TorchCpu)       # preference order

An implementation restates none of the contract, may have its own phase
graph or none, and may share phases through a common base class. Its
``phase_overlap`` says whether a pipelined run may execute different phases
of its one instance for different pulses at the same time. A block
without ``implementations`` is its own single ``default`` implementation.
Reading the declarations constructs nothing and loads no model.

Quality: an implementation may list the quality labels it serves
(``quality = ("fast",)``). A definition's root ``execution`` section, or
``CompileOptions.quality``, then asks for a label at compile time and the
selection prefers an implementation serving it (``targets``). Implementations
without labels never take part in quality selection, so legacy blocks behave
exactly as before.
"""

import inspect
import re
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Dict, FrozenSet, Iterable, Optional, Tuple

from roboflow_workflows.execution_engine.v2.context import ExecutionContextReader
from roboflow_workflows.execution_engine.v2.errors import (
    ContractError,
    DeclarationError,
)
from roboflow_workflows.execution_engine.v2.phases import (
    PhaseGraph,
    declare_phases,
    read_phase_graph,
)
from roboflow_workflows.execution_engine.v2.resources import (
    ResourceSpec,
    read_resource_specs,
)

__all__ = [
    "CONTRACT_ATTRIBUTES",
    "DEFAULT_IMPLEMENTATION",
    "PHASE_OVERLAP_ATTRIBUTE",
    "QUALITY_ATTRIBUTE",
    "RESERVED_PHASE_NAMES",
    "Implementation",
    "ImplementationSpec",
    "check_keyword_signature",
    "read_implementation_specs",
    "read_phase_overlap",
    "read_quality_labels",
]

DEFAULT_IMPLEMENTATION = "default"
"""Name of the single implementation of a block without ``implementations``."""

CONTRACT_ATTRIBUTES: Tuple[str, ...] = (
    "type",
    "aliases",
    "Params",
    "outputs",
    "output_fields",
    "describe_outputs",
    "accepts_empty",
    "mutates",
    "prunable",
    "engine_compatibility",
    "metadata",
    "implementations",
)
"""Block attributes forming the logical contract; implementations may not set them."""

PHASE_OVERLAP_ATTRIBUTE = "phase_overlap"
"""Class attribute allowing concurrent phases of one instance (see ``Implementation``)."""

QUALITY_ATTRIBUTE = "quality"
"""Class attribute listing the quality labels an implementation serves."""

RESERVED_PHASE_NAMES: Tuple[str, ...] = CONTRACT_ATTRIBUTES + (
    "name",
    "requires",
    PHASE_OVERLAP_ATTRIBUTE,
    QUALITY_ATTRIBUTE,
    "run",
    "wants",
    "execution_context",
    "discover_dependent_resources",
    "discover_work_operations",
    "discover_restrictions",
)
"""Attributes of ``Block`` and ``Implementation`` that a phase cannot shadow."""

_NAME = re.compile(r"[A-Za-z0-9_\-.]+")
_QUALITY_LABEL = re.compile(r"[A-Za-z0-9_\-]+")


class Implementation(ExecutionContextReader):
    """Base class of a runnable alternative of a logical block.

    Class attributes:
        name: Identity, unique among the block's implementations; letters,
            digits, ``_``, ``-`` and ``.``.
        requires: Capabilities the compile target must have, e.g.
            ``("cpu", "torch")``.
        quality: Quality labels this implementation serves, e.g.
            ``("fast",)`` or ``("balanced", "accurate")``; letters, digits,
            ``_`` and ``-``. Empty (default) opts out of quality selection.
            A step-level request for a label no fitting implementation of
            the block serves is a compile error; a workflow or deployment
            level request that nothing serves is recorded and ignored.
        phase_overlap: ``True`` (default) lets a pipelined run execute
            different phases of this one instance at the same time, for
            different pulses: pulse 1 may run ``tensor`` while pulse 0 runs
            ``logits``. One phase never runs twice at once. Set ``False``
            when phases share mutable state on ``self`` across a call; a
            pipelined run then holds the whole call, all phases and their
            futures, before the next pulse enters. Serial runs and
            ``run``-mode steps always execute one whole call at a time.
            Phase signatures describe data flow, not the safety of ``self``;
            a resource shared with other steps needs its own lock.

    Resources are the keyword parameters of ``__init__``, resolved only for
    the selected implementation. ``run`` takes the block's ``Params`` fields
    and returns what the block's ``run`` would. Phases follow ``@phase``.

    An implementation may define the classmethods
    ``discover_dependent_resources``, ``discover_work_operations`` and
    ``discover_restrictions`` (see ``Block``). Hooks it does not define fall
    back to the logical block's.
    """

    name: ClassVar[str]
    requires: ClassVar[Tuple[str, ...]] = ()
    quality: ClassVar[Tuple[str, ...]] = ()
    phase_overlap: ClassVar[bool] = True

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        declare_phases(
            cls,
            reserved=RESERVED_PHASE_NAMES,
            fail=lambda message: DeclarationError(
                f"Implementation class {cls.__name__}: {message}"
            ),
        )

    def run(self, **kwargs: Any) -> Any:
        """Execute one invocation; concrete implementations must override it."""
        raise NotImplementedError(f"{type(self).__name__} does not implement run()")


@dataclass(frozen=True)
class ImplementationSpec:
    """Validated declaration of one implementation of a logical block.

    Args:
        name: Implementation name; ``"default"`` for an ordinary block.
        implementation_class: Class the session constructs: the
            ``Implementation`` subclass, or the block class itself.
        requires: Capabilities a target needs to select it.
        resources: Constructor resources of ``implementation_class``.
        phases: Phase graph, or ``None`` without phases.
        phase_overlap: Whether a pipelined run may overlap different phases
            of one instance; meaningful only with ``phases``.
        quality: Quality labels this implementation serves; empty when it
            takes no part in quality selection.
    """

    name: str
    implementation_class: type
    requires: FrozenSet[str]
    resources: Tuple[ResourceSpec, ...]
    phases: Optional[PhaseGraph]
    phase_overlap: bool = True
    quality: FrozenSet[str] = frozenset()

    def serves(self, label: str) -> bool:
        """Whether this implementation declares the quality ``label``."""
        return label in self.quality

    def describe(self) -> Dict[str, Any]:
        """Return a JSON-friendly description without constructing anything."""
        owner = self.implementation_class
        description = {
            "name": self.name,
            "class": f"{owner.__module__}.{owner.__qualname__}",
            "requires": sorted(self.requires),
            "resources": [resource.describe() for resource in self.resources],
            "phases": self.phases.describe() if self.phases is not None else None,
            "phase_overlap": self.phase_overlap,
        }
        if self.quality:
            description["quality"] = sorted(self.quality)

        return description


def read_quality_labels(
    owner: type, *, fail: Callable[[str], Exception]
) -> FrozenSet[str]:
    """Read and validate the ``quality`` class attribute.

    Args:
        owner: ``Implementation`` subclass or block class.
        fail: Builds the exception to raise from a message.

    Returns:
        The declared labels; empty when the owner declares none.

    Raises:
        Exception: ``fail(message)`` when the value is not a tuple of labels
            made of letters, digits, ``_`` and ``-``, or repeats a label.
    """
    declared = getattr(owner, QUALITY_ATTRIBUTE, ())
    if isinstance(declared, str) or not isinstance(declared, (tuple, list)):
        raise fail(
            f"quality must be a tuple of labels such as ('fast',), got {declared!r}"
        )

    labels = tuple(declared)
    invalid = [
        label
        for label in labels
        if not isinstance(label, str) or not _QUALITY_LABEL.fullmatch(label)
    ]
    if invalid:
        raise fail(f"quality labels must be letters, digits, _ or -, got {invalid!r}")
    if len(set(labels)) != len(labels):
        raise fail(f"quality repeats a label: {list(labels)}")

    return frozenset(labels)


def read_phase_overlap(owner: type, *, fail: Callable[[str], Exception]) -> bool:
    """Read and validate the ``phase_overlap`` class attribute.

    Args:
        owner: ``Implementation`` subclass or block class.
        fail: Builds the exception to raise from a message.

    Returns:
        The declared value.

    Raises:
        Exception: ``fail(message)`` when the value is not a bool.
    """
    value = getattr(owner, PHASE_OVERLAP_ATTRIBUTE, True)
    if not isinstance(value, bool):
        raise fail(
            f"phase_overlap must be True or False, got {value!r}; False keeps "
            "one whole call (every phase) of an instance at a time when pipelined"
        )

    return value


def read_implementation_specs(
    declared: Any,
    *,
    fields: Iterable[str],
    fail: Callable[[str], Exception],
) -> Tuple[ImplementationSpec, ...]:
    """Validate the implementations a logical block lists.

    Args:
        declared: The block's non-empty ``implementations`` value.
        fields: The block's ``Params`` field names.
        fail: Builds the exception to raise from a message.

    Returns:
        One spec per implementation, in declared (preference) order.

    Raises:
        Exception: ``fail(message)`` for a malformed list, a duplicate name,
            or an implementation that restates the contract, lacks a valid
            ``name``/``requires``/``run``, or declares invalid resources or
            phases.
    """
    if isinstance(declared, str) or not isinstance(declared, (tuple, list)):
        raise fail(
            f"implementations must be a tuple of Implementation classes, got "
            f"{declared!r}"
        )

    field_names = tuple(fields)
    specs = tuple(
        _read_implementation(item, fields=field_names, fail=fail) for item in declared
    )
    names = [spec.name for spec in specs]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise fail(f"implementations repeat name(s) {repeated}")

    return specs


def _read_implementation(
    implementation: Any,
    *,
    fields: Tuple[str, ...],
    fail: Callable[[str], Exception],
) -> ImplementationSpec:
    if not isinstance(implementation, type) or not issubclass(
        implementation, Implementation
    ):
        raise fail(f"implementations lists {implementation!r}, not an Implementation")

    label = implementation.__name__

    def fail_here(message: str) -> Exception:
        return fail(f"implementation {label}: {message}")

    restated = [
        attribute
        for attribute in CONTRACT_ATTRIBUTES
        if any(attribute in vars(klass) for klass in implementation.__mro__[:-1])
    ]
    if restated:
        raise fail_here(
            f"restates contract attribute(s) {restated}; the logical block declares "
            "them once for every implementation"
        )

    name = getattr(implementation, "name", None)
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise fail_here(f"name must be letters, digits, _, - or ., got {name!r}")

    requires = implementation.requires
    if isinstance(requires, str) or not isinstance(requires, (tuple, list)):
        raise fail_here(
            f"requires must be a tuple of capability names, got {requires!r}"
        )
    if not all(isinstance(item, str) and item for item in requires):
        raise fail_here(f"requires holds a non-name: {list(requires)!r}")

    if implementation.run is Implementation.run:
        raise fail_here("does not implement run()")
    check_keyword_signature(
        implementation.run, name="run", fields=fields, fail=fail_here
    )

    try:
        resources = read_resource_specs(implementation)
    except ContractError as error:
        raise fail_here(str(error)) from error

    spec = ImplementationSpec(
        name=name,
        implementation_class=implementation,
        requires=frozenset(requires),
        resources=resources,
        phases=read_phase_graph(implementation, external=fields, fail=fail_here),
        phase_overlap=read_phase_overlap(implementation, fail=fail_here),
        quality=read_quality_labels(implementation, fail=fail_here),
    )

    return spec


def check_keyword_signature(
    method: Callable[..., Any],
    *,
    name: str,
    fields: Iterable[str],
    fail: Callable[[str], Exception],
) -> None:
    """Check that a method takes every ``Params`` field as a keyword argument.

    Args:
        method: Unbound method, e.g. ``run`` of a block or implementation.
        name: Method name for messages.
        fields: The ``Params`` field names.
        fail: Builds the exception to raise from a message.

    Raises:
        Exception: ``fail(message)`` for a parameter not passable by keyword,
            a required parameter that is not a field, or a field the method
            does not accept.
    """
    field_names = tuple(fields)
    parameters = list(inspect.signature(method).parameters.values())[1:]
    accepts_any_keyword = False
    accepted_names = set()
    for parameter in parameters:
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            accepts_any_keyword = True
            continue
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.POSITIONAL_ONLY,
        ):
            raise fail(
                f"{name}() parameter {parameter.name!r} must be passable by keyword"
            )
        accepted_names.add(parameter.name)
        if (
            parameter.name not in field_names
            and parameter.default is inspect.Parameter.empty
        ):
            raise fail(
                f"{name}() requires {parameter.name!r}, which is not a Params field"
            )

    missing = [field for field in field_names if field not in accepted_names]
    if missing and not accepts_any_keyword:
        raise fail(f"{name}() does not accept Params field(s) {missing}")

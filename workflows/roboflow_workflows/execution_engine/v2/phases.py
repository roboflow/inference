"""Phases: named steps of one block call, wired by their signatures.

A block (or an ``Implementation``) may split its work into ``@phase``
methods. A phase parameter named like a ``Params`` field receives that call
argument; a parameter named like another phase receives that phase's return
value. Exactly one phase feeds no other phase: the result phase, whose return
value is the complete block result, the same as ``run`` returns::

    class Classify(Block):
        ...
        @phase
        def tensor(self, *, image): ...               # field input
        @phase
        def logits(self, *, tensor): ...              # branch 1
        @phase
        def flipped(self, *, tensor): ...             # branch 2
        @phase
        def prediction(self, *, logits, flipped, image): ...   # join = result

        def run(self, *, image):                      # explicit composition
            tensor = self.tensor(image=image)
            return self.prediction(
                logits=self.logits(tensor=tensor),
                flipped=self.flipped(tensor=tensor),
                image=image,
            )

``run`` stays the explicit, directly callable composition. ``run_phases``
executes the same graph from its declaration: serially, each phase once per
call, in a deterministic topological order. A pipelined run wraps each phase
(``around_phase``) in that phase's stage, so another pulse may run an earlier
phase of the same instance meanwhile; ``phase_overlap = False`` on the class
forbids that. Phase results are private values of that call; they are never
workflow outputs or selectors, and each is released after its last consumer,
or when a phase fails.

A phase behaves the same when ``run`` calls it and when ``run_phases`` does:

    returns a Future (also inside lists, dicts, tuples, Batch)
                         -> resolved before the caller or a consumer sees it
                            (``readiness.resolve_futures``)
    returns a coroutine  -> rejected, also nested, never awaited: phases are
                            synchronous; an unstarted one is closed, a started
                            one is left to its owner
    raises               -> PhaseFailure(phase=name), original as __cause__

Waiting on a Future does not cancel anything: work a phase or its caller
submitted stays owned by whoever submitted it. A resolved host Future is not
evidence that device work behind it completed.
"""

import functools
import inspect
from collections import Counter
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    ContextManager,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
)

from roboflow_workflows.execution_engine.v2.errors import ContractError
from roboflow_workflows.execution_engine.v2.readiness import resolve_futures

__all__ = [
    "PhaseFailure",
    "PhaseGraph",
    "PhaseSpec",
    "declare_phases",
    "phase",
    "read_phase_graph",
    "run_phases",
]

_MARKER = "__workflows_phase__"
"""Attribute of a phase method: the phase name."""


class PhaseFailure(Exception):
    """A phase raised, or one of its futures failed.

    The original exception is the ``__cause__``. A failure inside a phase
    that another phase called keeps the innermost phase's name.

    Args:
        phase: Name of the failing phase.
        error: The original exception.
    """

    def __init__(self, phase: str, error: BaseException):
        super().__init__(f"phase {phase!r} failed: {type(error).__name__}: {error}")
        self.phase = phase


def phase(function: Callable[..., Any]) -> Callable[..., Any]:
    """Mark an instance method as a phase of its class.

    The method stays directly callable with its own signature. Each call
    returns a ready value: futures are resolved, coroutines rejected, and
    failures become ``PhaseFailure`` naming the phase (see module doc).

    Args:
        function: Method defined with ``def`` taking keyword-passable
            parameters without defaults.

    Returns:
        The wrapped method.

    Raises:
        TypeError: When ``function`` is not a plain synchronous function.
    """
    if not inspect.isfunction(function):
        raise TypeError(f"@phase decorates a method defined with def, got {function!r}")
    if inspect.iscoroutinefunction(function) or inspect.isasyncgenfunction(function):
        raise TypeError(
            f"@phase {function.__name__!r} is async; phases are synchronous and may "
            "return a concurrent.futures.Future instead"
        )

    name = function.__name__

    @functools.wraps(function)
    def call_phase(*args: Any, **kwargs: Any) -> Any:
        try:
            result = resolve_futures(function(*args, **kwargs), reject_awaitables=True)
        except Exception as error:
            # The failure's traceback keeps this frame; it must not keep the
            # phase inputs (other phases' results) too.
            args = kwargs = None
            if isinstance(error, PhaseFailure):
                raise
            raise PhaseFailure(name, error) from error

        return result

    setattr(call_phase, _MARKER, name)

    return call_phase


@dataclass(frozen=True)
class PhaseSpec:
    """One phase of a graph.

    Args:
        name: Method name.
        parameters: Parameter names in signature order.
        upstream: The parameters that receive other phases' results.
    """

    name: str
    parameters: Tuple[str, ...]
    upstream: Tuple[str, ...]

    @property
    def external(self) -> Tuple[str, ...]:
        """Parameters that receive call arguments."""
        return tuple(name for name in self.parameters if name not in self.upstream)

    def describe(self) -> Dict[str, Any]:
        """Return a JSON-friendly description."""
        description = {
            "name": self.name,
            "parameters": list(self.parameters),
            "upstream": list(self.upstream),
        }

        return description


@dataclass(frozen=True)
class PhaseGraph:
    """Validated, acyclic phase graph of one class.

    Args:
        phases: Every phase, in the order ``run_phases`` executes them.
        result: Name of the result phase, the only one no phase consumes.
    """

    phases: Tuple[PhaseSpec, ...]
    result: str

    def describe(self) -> Dict[str, Any]:
        """Return a JSON-friendly description."""
        description = {
            "result": self.result,
            "phases": [spec.describe() for spec in self.phases],
        }

        return description


def declare_phases(
    owner: type, *, reserved: Iterable[str], fail: Callable[[str], Exception]
) -> None:
    """Prepare the phases of a class while it is being created.

    ``Block`` and ``Implementation`` call this for every subclass, before
    any other declaration check:

    * a phase named like a ``reserved`` attribute (``metadata``, ``run``,
      ``execution_context``, ...) is rejected early, with a clear message;
    * an undecorated override of an inherited phase becomes a phase, so it
      behaves like the phase it replaces when ``run`` calls it directly.

    Other owners decorate their overrides themselves.

    Args:
        owner: Class being created.
        reserved: Attribute names a phase cannot take.
        fail: Builds the exception to raise from a message.

    Raises:
        Exception: ``fail(message)`` for a phase with a reserved name.
    """
    shadowing = sorted(set(_phase_names(owner)) & set(reserved))
    if shadowing:
        raise fail(
            f"phase(s) {shadowing} shadow attributes the engine reads from the "
            "class; rename the phase(s)"
        )

    inherited = {
        name
        for base in owner.__mro__[1:]
        for name, value in vars(base).items()
        if _is_phase(value)
    }
    for name in inherited:
        override = vars(owner).get(name)
        if inspect.isfunction(override) and not _is_phase(override):
            setattr(owner, name, phase(override))


def read_phase_graph(
    owner: type,
    *,
    external: Iterable[str],
    fail: Callable[[str], Exception],
) -> Optional[PhaseGraph]:
    """Read and validate the phase graph declared by a class.

    Phases are collected through the class hierarchy; each is validated in
    its effective (most derived) form.

    Args:
        owner: Class declaring ``@phase`` methods. It is not instantiated.
        external: Names a phase parameter may take from the call arguments,
            e.g. the ``Params`` field names.
        fail: Builds the exception to raise from a message.

    Returns:
        The graph, or ``None`` when ``owner`` declares no phases.

    Raises:
        Exception: ``fail(message)`` when a phase is not an instance method,
            a parameter is unknown, has a default or is not passable by
            keyword, a phase is named like an external name, the
            graph has a cycle, or not exactly one phase is the result.
    """
    names = _phase_names(owner)
    if not names:
        return None

    external_names = set(external)
    parameters = {
        name: _phase_parameters(
            owner, name, external=external_names, phases=names, fail=fail
        )
        for name in names
    }
    specs = {
        name: PhaseSpec(
            name=name,
            parameters=parameters[name],
            upstream=tuple(item for item in parameters[name] if item in parameters),
        )
        for name in names
    }
    ordered = _topological_order(specs, fail=fail)
    consumed = {item for spec in specs.values() for item in spec.upstream}
    sinks = [name for name in names if name not in consumed]
    if len(sinks) != 1:
        raise fail(
            f"phases {sinks} feed no other phase; exactly one phase is the result, "
            "the others must feed it (directly or through other phases)"
        )

    graph = PhaseGraph(phases=tuple(specs[name] for name in ordered), result=sinks[0])

    return graph


def _is_phase(value: Any) -> bool:
    return inspect.isfunction(value) and hasattr(value, _MARKER)


def _phase_names(owner: type) -> List[str]:
    """Phase names in declaration order, base classes first."""
    names: Dict[str, None] = {}
    for klass in reversed(owner.__mro__):
        for name, value in vars(klass).items():
            if _is_phase(value):
                names[name] = None

    return list(names)


def _phase_parameters(
    owner: type,
    name: str,
    *,
    external: set,
    phases: List[str],
    fail: Callable[[str], Exception],
) -> Tuple[str, ...]:
    method = inspect.getattr_static(owner, name)
    if not inspect.isfunction(method):
        raise fail(
            f"phase {name!r} is overridden by {type(method).__name__}; a phase is "
            "an instance method"
        )
    if not _is_phase(method):
        raise fail(f"phase {name!r} is overridden without @phase")
    if getattr(method, _MARKER) != name:
        raise fail(
            f"phase {name!r} is function {getattr(method, _MARKER)!r}; define phases "
            "with def under their own name"
        )
    if name in external:
        raise fail(f"phase {name!r} has the name of a Params field; rename the phase")

    signature = list(inspect.signature(method).parameters.values())
    if not signature or signature[0].kind not in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ):
        raise fail(f"phase {name!r} must be an instance method taking self")

    parameters = []
    for parameter in signature[1:]:
        where = f"phase {name!r} parameter {parameter.name!r}"
        if parameter.kind not in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            raise fail(
                f"{where} must be a named keyword parameter; phase inputs are "
                "bound by name"
            )
        if parameter.default is not inspect.Parameter.empty:
            raise fail(
                f"{where} has a default; every phase input is bound to a Params "
                "field or a phase, so remove the default"
            )
        if parameter.name not in external and parameter.name not in phases:
            raise fail(
                f"{where} names neither a Params field nor a phase; "
                f"fields: {sorted(external)}, phases: {phases}"
            )
        parameters.append(parameter.name)

    return tuple(parameters)


def _topological_order(
    specs: Mapping[str, PhaseSpec], *, fail: Callable[[str], Exception]
) -> List[str]:
    """Kahn's order, ties broken by declaration order; a cycle is reported."""
    waiting = {name: set(spec.upstream) for name, spec in specs.items()}
    ordered: List[str] = []
    while True:
        ready = [name for name, upstream in waiting.items() if not upstream]
        if not ready:
            break
        name = ready[0]
        ordered.append(name)
        del waiting[name]
        for upstream in waiting.values():
            upstream.discard(name)

    if waiting:
        cycle = _find_cycle(waiting)
        raise fail(f"phases form a cycle: {' -> '.join(cycle)}")

    return ordered


def _find_cycle(waiting: Mapping[str, set]) -> List[str]:
    """A cycle among phases that never became ready, as a closed path."""
    path = [next(iter(waiting))]
    while path.count(path[-1]) < 2:
        upstream = sorted(waiting[path[-1]])
        path.append(upstream[0])
    start = path.index(path[-1])
    cycle = list(reversed(path[start:]))

    return cycle


def run_phases(
    instance: Any,
    graph: PhaseGraph,
    arguments: Mapping[str, Any],
    *,
    on_phase: Optional[Callable[[str], None]] = None,
    around_phase: Optional[Callable[[str], ContextManager[None]]] = None,
) -> Any:
    """Execute a phase graph once on ``instance`` and return the result phase's value.

    Phases run serially in ``graph`` order; each runs exactly once and gets
    its upstream results and its call arguments by name. Results live only
    in this call: each is released after its last consumer, and all of them
    when a phase fails. Nothing is cached on ``instance`` or ``graph``.

    Args:
        instance: Object with the graph's phase methods, e.g. a block.
        graph: Graph read from the instance's class.
        arguments: Call arguments by name; extra names are ignored.
        on_phase: Called with each phase name before the phase runs.
        around_phase: Context manager factory entered with each phase name
            around that phase's call, which includes resolving its futures;
            a pipelined run holds the phase's stage with it.

    Returns:
        The result phase's ready value.

    Raises:
        ContractError: When ``arguments`` lack a name a phase needs.
        PhaseFailure: When a phase raises or returns a failing future.
    """
    missing = sorted(
        {name for spec in graph.phases for name in spec.external} - set(arguments)
    )
    if missing:
        raise ContractError(f"run_phases misses argument(s) {missing}")

    consumers = Counter(name for spec in graph.phases for name in spec.upstream)
    produced: Dict[str, Any] = {}
    try:
        for spec in graph.phases:
            if on_phase is not None:
                on_phase(spec.name)
            inputs = {
                name: produced[name] if name in spec.upstream else arguments[name]
                for name in spec.parameters
            }
            method = getattr(instance, spec.name)
            if around_phase is None:
                produced[spec.name] = method(**inputs)
            else:
                with around_phase(spec.name):
                    produced[spec.name] = method(**inputs)
            for name in spec.upstream:
                consumers[name] -= 1
                if not consumers[name]:
                    del produced[name]
        result = produced.pop(graph.result)
    finally:
        # A failure's traceback keeps this frame alive; it must not keep
        # the intermediates too.
        produced.clear()
        inputs = None

    return result

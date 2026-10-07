"""Plan the reactions of a composed workflow: handlers, signals, state, groups.

Runs after the block steps are planned, so every emitter's declared events
are known::

    root signals    ──▶ PlannedSignal          kind names → catalogue kinds
    every scope:
      state         ──▶ one StateDefaults      a key declared twice needs equal values
      state_machines──▶ PlannedMachine         see compilation.machines
      handlers      ──▶ resolve `on` in the declaring scope (same EventResolver)
                        compile the handler workflow (same catalogue and options)
                        PlannedHandler checks bindings: input, axes, field, kinds
    root handler groups ──▶ PlannedHandlerGroup   anchor output exists
    ReactionPlan(...)       setter steps authorized, group handlers and outputs
                            exist, cascade cycles rejected

The plan types own every rule they can check. Their ``ReactionContractError``
carries the compile error type and location, so a definition gets the same
typed diagnostic as before and the rule is written once.

Paths keep scope identity: a child used twice (a diamond) yields two
handlers, ``("a", "notify")`` and ``("b", "notify")``, subscribed to
``("a", "zone")`` and ``("b", "zone")``.

State keys form one namespace for the whole workflow, whatever scope
declares them, so the scopes' initial values merge into one
``StateDefaults``. Equal values (a diamond repeats them) are kept once;
different values for one key are rejected.
"""

import json
from typing import Any, Callable, Dict, List, Mapping, Tuple

from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.compilation.composition import (
    ComposedHandler,
    Scope,
)
from roboflow_workflows.execution_engine.v2.compilation.machines import (
    EventResolver,
    plan_machines,
)
from roboflow_workflows.execution_engine.v2.errors import (
    ContractError,
    SelectorError,
    StepPath,
    WorkflowCompileError,
)
from roboflow_workflows.execution_engine.v2.events import Event
from roboflow_workflows.execution_engine.v2.plan import CompiledWorkflow, PlannedStep
from roboflow_workflows.execution_engine.v2.reactions.plan import (
    PlannedHandler,
    PlannedHandlerGroup,
    PlannedSignal,
    ReactionPlan,
    StateDefaults,
    as_compile_error,
    scoped_name,
)

CompileHandler = Callable[[Scope], CompiledWorkflow]
"""Compiles a handler's workflow scope into its passive plan."""


def collect_handlers(root: Scope) -> Tuple[ComposedHandler, ...]:
    """Return the handlers of every scope, root first, in declaration order.

    Args:
        root: Root scope of the composition.

    Returns:
        Composed handlers.
    """
    handlers = tuple(
        handler for scope in root.scopes() for handler in scope.handlers.values()
    )

    return handlers


def plan_reactions(
    root: Scope,
    *,
    steps: Mapping[StepPath, PlannedStep],
    catalogue: Catalogue,
    compile_handler: CompileHandler,
    active: bool,
) -> ReactionPlan:
    """Plan handlers, signals, state and handler groups of a workflow.

    Args:
        root: Root scope of the composition.
        steps: Planned block steps by path.
        catalogue: The plan's catalogue; handler plans use the same one.
        compile_handler: Compiles one handler workflow scope.
        active: Whether the workflow declares sources.

    Returns:
        The reaction plan; ``ReactionPlan.EMPTY`` when nothing is declared.

    Raises:
        WorkflowCompileError: A subclass naming the handler, field and reason.
    """
    scopes = list(root.scopes())
    declares = any(
        scope.handlers or scope.workflow.machines or scope.workflow.state is not None
        for scope in scopes
    )
    if not declares and not root.workflow.signals:
        return ReactionPlan.EMPTY

    signals = _plan_signals(root, catalogue=catalogue)
    state = _merge_state(scopes)

    events = EventResolver(scopes, steps=steps, signals=signals, active=active)
    machines = plan_machines(events)
    handlers = tuple(
        _plan_handler(
            scope,
            composed,
            events=events,
            compile_handler=compile_handler,
            active=active,
        )
        for scope in scopes
        for composed in scope.handlers.values()
    )
    groups = _plan_groups(root, handlers=handlers, active=active)
    try:
        reactions = ReactionPlan(
            handlers=handlers,
            signals=signals,
            groups=groups,
            state=state,
            machines=machines,
        )
    except ContractError as error:
        raise as_compile_error(error, where="Reactions are inconsistent") from error

    return reactions


def _plan_signals(root: Scope, *, catalogue: Catalogue) -> Dict[str, PlannedSignal]:
    signals: Dict[str, PlannedSignal] = {}
    for declaration in root.workflow.signals:
        fields = {}
        for field_name, kind_names in declaration.fields.items():
            unknown = [name for name in kind_names if name not in catalogue.kinds]
            if unknown:
                raise WorkflowCompileError(
                    f"{declaration.location}.fields.{field_name} ($signals."
                    f"{declaration.name}) declares unknown kinds {unknown}; the "
                    f"catalogue knows {sorted(catalogue.kinds)}"
                )
            fields[field_name] = tuple(catalogue.kinds[name] for name in kind_names)
        try:
            event = Event(fields, description=declaration.description)
        except ContractError as error:
            raise WorkflowCompileError(
                f"{declaration.location} ($signals.{declaration.name}) is invalid: "
                f"{error}"
            ) from error
        signals[declaration.name] = PlannedSignal(name=declaration.name, event=event)

    return signals


def _merge_state(scopes: List[Scope]) -> StateDefaults:
    merged: Dict[str, Dict[str, Any]] = {"global": {}, "source": {}}
    declared_at: Dict[Tuple[str, str], str] = {}
    for scope in scopes:
        declaration = scope.workflow.state
        if declaration is None:
            continue
        defaults = declaration.defaults
        for kind, values in (("global", defaults.global_), ("source", defaults.source)):
            for key, value in values.items():
                where = f"{declaration.location}.{kind}.{key}"
                if (kind, key) not in declared_at:
                    merged[kind][key] = value
                    declared_at[(kind, key)] = where
                    continue
                known = merged[kind][key]
                if _canonical(known) != _canonical(value):
                    raise WorkflowCompileError(
                        f"{where} starts at {value!r}, but {declared_at[(kind, key)]} "
                        f"starts at {known!r}; every scope of a workflow shares one "
                        f"{kind} state, so a key has one initial value. Rename one "
                        "key or declare the same value",
                        step_path=scope.path,
                    )

    state = StateDefaults(global_=merged["global"], source=merged["source"])

    return state


def _canonical(value: Any) -> str:
    # Type-sensitive like managed state equality: 1 != 1.0 != True.
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)

    return text


def _plan_handler(
    scope: Scope,
    composed: ComposedHandler,
    *,
    events: EventResolver,
    compile_handler: CompileHandler,
    active: bool,
) -> PlannedHandler:
    declaration = composed.declaration
    path = composed.path
    where = f"{declaration.location} ($handlers.{scoped_name(path)})"
    if scope.workflow.step(declaration.name) is not None:
        raise WorkflowCompileError(
            f"{where} has the name of $steps.{scoped_name(path)}; a handler and a "
            "step of one workflow need different names",
            step_path=path,
        )
    if declaration.mode == "async" and not active:
        raise WorkflowCompileError(
            f"{where} is async, but the workflow declares no sources; async "
            "handlers need an active run that owns their workers. Use "
            "execution mode 'sync' in a passive workflow",
            step_path=path,
            field_path=("execution", "mode"),
        )

    origin, event = events.resolve(scope, declaration.on, where=where, path=path)
    plan = _compile_handler_workflow(composed, compile_handler=compile_handler)
    try:
        handler = PlannedHandler(
            path=path,
            origin=origin,
            event=event,
            plan=plan,
            bindings=declaration.bindings,
            constants=declaration.constants,
            mode=declaration.mode,
            queue=declaration.queue,
        )
    except ContractError as error:
        raise as_compile_error(
            error, where=declaration.location, step_path=path
        ) from error

    return handler


def _plan_groups(
    root: Scope, *, handlers: Tuple[PlannedHandler, ...], active: bool
) -> Tuple[PlannedHandlerGroup, ...]:
    by_path = {handler.path: handler for handler in handlers}
    groups: List[PlannedHandlerGroup] = []
    for declaration in root.workflow.handler_groups:
        where = f"{declaration.location} ({declaration.name})"
        selector = f"$handlers.{scoped_name(declaration.handler)}"
        if not active:
            raise WorkflowCompileError(
                f"{where} is anchored at {selector}, but the workflow declares no "
                "sources; a passive workflow returns flat outputs. Read handler "
                "results through a reaction observer instead"
            )
        # ReactionPlan checks the handler and the field outputs; the anchor
        # output is not part of the planned group.
        handler = by_path.get(declaration.handler)
        anchor = declaration.anchor_output
        if handler is not None and anchor is not None and anchor not in handler.outputs:
            raise SelectorError(
                f"{where} selects outputs {[anchor]} that {selector} does not "
                f"declare; its workflow outputs are {list(handler.outputs)}",
                field_path=("anchor",),
            )
        groups.append(
            PlannedHandlerGroup(
                name=declaration.name,
                handler=declaration.handler,
                fields=declaration.fields,
            )
        )

    return tuple(groups)


def _compile_handler_workflow(
    composed: ComposedHandler, *, compile_handler: CompileHandler
) -> CompiledWorkflow:
    try:
        plan = compile_handler(composed.workflow)
    except WorkflowCompileError as error:
        raise in_handler(error, path=composed.path) from error

    return plan


def in_handler(error: WorkflowCompileError, *, path: StepPath) -> WorkflowCompileError:
    """Re-attribute an error of a handler workflow to the handler.

    The error type is kept when its constructor is the plain
    ``WorkflowCompileError`` one.

    Args:
        error: Error raised while compiling the handler workflow.
        path: Handler path.

    Returns:
        An error whose message names the handler and whose step path is the
        handler path followed by the step path inside the handler workflow.
    """
    error_type = type(error)
    if error_type.__init__ is not WorkflowCompileError.__init__:
        error_type = WorkflowCompileError
    wrapped = error_type(
        f"$handlers.{scoped_name(path)} workflow: {error}",
        step_path=path + tuple(error.step_path),
        field_path=error.field_path,
    )

    return wrapped

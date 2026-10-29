"""Passive history: operators that keep their state across passive runs.

A plan with operators and no sources is a passive history plan. Each
submission, a direct ``session.run`` or a pipeline submission, is one pulse
of the ``$passive`` domain, and it runs its operators inline::

    submission: one $passive pulse, one ticket, one team, one RunState
      ├─ $passive route: steps fed by inputs, and static steps (once)
      ├─ operator A, fed by $passive: push -> zero or one pulse
      │    ├─ one: A's route runs in the same RunState
      │    └─ none: A's ports and route are absent, A is a silent operator
      │       of the run, and its stage turns retire
      └─ operator B, fed by A or $passive: the same, in plan order
    one RunResult holds every domain the submission reached

No pool, queue, Future or thread is added: an emitted route runs on the
submission's workers, inside its ``max_in_flight`` slot. Operators admitted
here declare ``passive_max_emissions = 1`` and read one upstream domain
(``plan``). The session keeps their instances, a ``PassiveHistory``, from
the first submission until its processing resets or it closes; both close
them and drop partial tails. A failed submission poisons the history: later
submissions fail until a processing reset replaces it. Admitted submissions
of a cancelled or failed pipeline still stop as aborted.
"""

import functools
from typing import Callable, Dict, List, Optional

from roboflow_workflows.v2.engine.active.execution import (
    absent_entry,
    continue_operator_pulse,
    operator_arrivals,
)
from roboflow_workflows.v2.engine.active.pulses import OperatorSlot, close_operator
from roboflow_workflows.v2.engine.errors import ContractError, WorkflowExecutionError
from roboflow_workflows.v2.engine.execution.steps import RunState, execute_steps
from roboflow_workflows.v2.engine.operators import Arrival, OperatorPulse
from roboflow_workflows.v2.engine.pipelining.stages import PASSIVE_DOMAIN, Coordination
from roboflow_workflows.v2.engine.plan import (
    CompiledWorkflow,
    PlannedOperator,
    PulseKey,
)


class PassiveHistory:
    """The operator instances one session keeps between passive runs.

    Args:
        plan: A passive history plan.

    Raises:
        WorkflowExecutionError: When an operator constructor raised. Its
            exception is the ``__cause__``; the operators built before it
            are closed, and their close failures are named in the message.
    """

    def __init__(self, plan: CompiledWorkflow) -> None:
        self.slots: Dict[str, OperatorSlot] = _construct_slots(plan)
        # The first failure of a submission; set once, cleared by a reset.
        self.failure: Optional[BaseException] = None

    def raise_if_failed(self, coordination: Coordination) -> None:
        """Refuse a submission after a failed one, until a processing reset.

        Args:
            coordination: The submission's coordination. A cancel or a
                failure aborts it before the history records its failure, so
                an admitted submission of an aborted pipeline stops as
                aborted, not as refused.

        Raises:
            RunAborted: When a submission failed before and the coordination
                aborted.
            ContractError: When a submission failed before.
        """
        if self.failure is not None:
            coordination.checkpoint()
            raise ContractError(
                "Passive history failed in an earlier run; its operators may "
                "hold partial state. Apply a processing reset "
                "(session.update(plan, reset=True)) before running again"
            ) from self.failure

    def close(self) -> None:
        """Close every operator once; raise after all of them were tried.

        Partial tails are dropped: a passive run has no finish delivery.

        Raises:
            Exception: What the only failing ``close`` raised.
            WorkflowExecutionError: When several closes raised; the first
                failure is the ``__cause__``.
        """
        failures = [close_operator(slot) for slot in self.slots.values()]
        errors = [failure.__cause__ for failure in failures if failure is not None]
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise WorkflowExecutionError(
                "Closing passive history operators failed: "
                + "; ".join(str(failure) for failure in failures if failure)
            ) from errors[0]

    def retire(self) -> Dict[str, Callable[[], None]]:
        """Closers of every operator, for a reset's ``updates.Retirement``.

        Returns:
            One closer per operator name; each raises what ``close`` raised.
        """
        closers = {
            name: functools.partial(_close_retired, slot)
            for name, slot in self.slots.items()
        }

        return closers


def execute_history(run: RunState) -> None:
    """Execute one passive history submission and the routes its operators emit.

    Args:
        run: The submission's run state, before any step ran.

    Raises:
        RunAborted: When the run's pipeline aborted.
        ContractError: When an earlier submission failed, or an operator
            returned more than its declared one pulse.
        Exception: What a step or an operator raised; the history is then
            poisoned until a processing reset.
    """
    history = run.session._passive_history_state(
        functools.partial(PassiveHistory, run.session.plan)
    )
    history.raise_if_failed(run.coordination)
    try:
        _execute(run, history=history)
    except BaseException as error:
        # Kept open: another admitted submission may still be inside a push.
        # Session close or a processing reset closes it after they drained.
        if history.failure is None:
            history.failure = error
        raise


def _execute(run: RunState, *, history: PassiveHistory) -> None:
    ordinal = run.ticket.ordinal if run.ticket is not None else 0
    root = PulseKey(active_run_id=run.run_id, source=PASSIVE_DOMAIN, sequence=ordinal)
    run.pulse = root
    execute_steps(run, run.plan.schedule(PASSIVE_DOMAIN))

    emitted: Dict[str, PulseKey] = {PASSIVE_DOMAIN: root}
    for name, slot in history.slots.items():  # plan order: upstreams first
        feeding = emitted.get(slot.planned.upstream_domains[0])
        arrivals = None
        if feeding is not None:
            run.pulse = feeding  # operator_arrivals reads the feeding domain
            arrivals = operator_arrivals(run, slot.planned)
        emission = _push(run, slot, arrivals=arrivals, history=history)
        if emission is None:
            _skip_route(run, slot.planned)
            continue

        pulse = PulseKey(active_run_id=run.run_id, source=name, sequence=ordinal)
        continue_operator_pulse(run, pulse=pulse, emission=emission)
        emitted[name] = pulse
        execute_steps(run, run.plan.schedule(name))
        slot.counters.processed += 1

    # The result keeps every entry's own sample and temporal contexts; it
    # claims no common clock for the domains it holds.
    run.pulse = root
    run.causes = ()


def _push(
    run: RunState,
    slot: OperatorSlot,
    *,
    arrivals: Optional[List[Arrival]],
    history: PassiveHistory,
) -> Optional[OperatorPulse]:
    """Push the feeding pulse's arrivals; the emitted pulse, or ``None``.

    The submission takes its turn at the operator's stage even when its
    upstream emitted nothing (``arrivals`` is ``None``): a zero-call turn
    retires at once, so a later submission never waits for it.
    """
    stages = run.coordination.stages(
        run, {"push": slot.stage}, calls=0 if arrivals is None else 1
    )
    pulses: List[OperatorPulse] = []
    if arrivals is not None:
        with stages.call("push"):
            # A submission that failed while this one waited for its turn.
            history.raise_if_failed(run.coordination)
            slot.counters.arrivals += len(arrivals)
            pulses = slot.instance.push(arrivals)
            _check_pulses(slot, pulses)
            slot.counters.emitted += len(pulses)
    stages.finish()
    emission = pulses[0] if pulses else None

    return emission


def _check_pulses(slot: OperatorSlot, pulses: object) -> None:
    if not isinstance(pulses, (list, tuple)) or not all(
        isinstance(pulse, OperatorPulse) for pulse in pulses
    ):
        raise ContractError(
            f"$operators.{slot.name}.push() must return a list of OperatorPulse, "
            f"got {pulses!r}"
        )
    if len(pulses) > 1:
        raise ContractError(
            f"$operators.{slot.name}.push() returned {len(pulses)} pulses, but "
            "the operator declares passive_max_emissions = 1; none of them ran"
        )


def _skip_route(run: RunState, operator: PlannedOperator) -> None:
    """Mark an operator's ports and route absent for a pulse it did not emit."""
    for port_name, port in operator.outputs.items():
        run.ports[operator.port(port_name)] = absent_entry(port.layout)
    for step in run.plan.schedule(operator.name).steps:
        run.coordination.step_stages(run, step, calls=0)
        for output_name, output in step.outputs.items():
            run.outputs[(step.path, output_name)] = absent_entry(output.layout)
        run.decisions[step.path] = absent_entry(step.invocation_layout)
    run.silent_operators = run.silent_operators | {operator.name}
    run.record("passive_operator_no_emission", operator=operator.name)


def _construct_slots(plan: CompiledWorkflow) -> Dict[str, OperatorSlot]:
    slots: Dict[str, OperatorSlot] = {}
    for name, planned in plan.operators.items():
        operator_class = planned.spec.operator_class
        try:
            instance = operator_class(
                name=name, params=planned.params, inputs=planned.inputs
            )
        except Exception as error:
            rollback = [close_operator(slot) for slot in slots.values()]
            closing = [str(failure) for failure in rollback if failure is not None]
            also = f"; closing the operators built before it failed: {closing}"
            raise WorkflowExecutionError(
                f"$operators.{name}: constructor of {operator_class.__qualname__} "
                f"failed: {type(error).__name__}: {error}" + (also if closing else "")
            ) from error
        slots[name] = OperatorSlot(planned=planned, instance=instance)

    return slots


def _close_retired(slot: OperatorSlot) -> None:
    """Close an operator a reset replaced; raise what its ``close`` raised."""
    failure = close_operator(slot)
    if failure is not None:
        raise failure.__cause__

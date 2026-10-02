"""Named examples. Each checks its own observations and returns JSON evidence."""

from dataclasses import dataclass
from typing import Any, Callable, Dict

from active_case import active
from contract_cases import (
    authoring_errors,
    mutation,
    phase_error,
    private_phase,
    selection,
)
from host import DemoContext
from model_cases import classify, direct, mps


@dataclass(frozen=True)
class Case:
    """One runnable example.

    Attributes:
        description: One line shown by ``--list``.
        run: Runs the example and returns its evidence.
    """

    description: str
    run: Callable[[DemoContext], Dict[str, Any]]


CASES: Dict[str, Case] = {
    "classify": Case(
        "Two photos and their crops; nested gate; run and phase mode give identical predictions",
        classify,
    ),
    "direct": Case(
        "The implementation called without the engine: run() and run_phases agree",
        direct,
    ),
    "selection": Case(
        "Implementation chosen per target at compile time; unphased fallback; cuda rejected",
        selection,
    ),
    "mps": Case(
        "The MPS implementation on this host; skipped when MPS is unavailable",
        mps,
    ),
    "active": Case(
        "Timed frames, gated crops in a child workflow and a window of predictions",
        active,
    ),
    "mutation": Case(
        "In-place redaction inside a phase: warning, strict rejection, ordered effect",
        mutation,
    ),
    "phase-error": Case(
        "A failing phase is named with its step in both modes",
        phase_error,
    ),
    "private-phase": Case(
        "Selecting a phase result as a workflow output is rejected",
        private_phase,
    ),
    "authoring-errors": Case(
        "Phase cycles, unknown phase parameters and restated contracts fail at class definition",
        authoring_errors,
    ),
}

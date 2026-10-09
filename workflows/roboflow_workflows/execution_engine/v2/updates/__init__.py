"""Live graph updates: change a session's plan while keeping compatible blocks.

Compile the new definition, inspect the comparison, then update the session::

    diff = compare_plans(session.plan, new_plan)      # pure, constructs nothing
    receipt = session.update(new_plan)                # prepare + apply
    receipt.graph_version                             # 1 after the first update

or in two steps, to inspect the constructed candidate before the commit::

    update = session.prepare_update(new_plan)
    session.apply_update(update)                      # or update.discard()

An update keeps the session: retained steps keep their instances and
block-local state, managed state and session resources stay, and only added
steps are constructed. Results carry the ``graph_version`` that produced them,
separate from the control version. A rejected update raises a
``GraphUpdateError`` and the session keeps its current graph.

Updates are additive by default. A changed step, including a changed
parameter, is rejected, and so is removing a step, also by removing its last
consumer. ``updates.diff`` explains each reason, and says which ones a reset
can apply.

A reset is an explicit second kind of update. It replaces everything
downstream of the sources: every step, operator, handler session and the
engine-owned managed state start fresh; sources, their resources and the
caller's resources and managed state stay::

    assessment = session.assess_update(new_plan)     # pure, constructs nothing
    assessment.kind        # "preserve", "reset" or "unsupported"
    assessment.describe()  # reasons and reset consequences, JSON-friendly
    update = session.prepare_update(new_plan, reset=True)
    receipt = session.apply_update(update)
    receipt.reset, receipt.processing_version          # True, 1

A preserving update never resets: ``prepare_update(plan)`` keeps raising
``IncompatibleUpdateError`` for a plan only a reset applies. Results carry
the ``processing_version`` next to the ``graph_version``. See
``updates.reset`` for what is kept and released.

``session.apply_update`` updates idle sessions only: a direct run, an open
passive pipeline or an unfinished active run makes it raise
``SessionBusyError``. A closed session raises ``SessionClosedError``.

A running active run switches graphs through its own handle::

    update = session.prepare_update(new_plan)         # while the run runs
    receipt = run.apply_update(update, handlers={"masks": on_masks})

The run pauses admission, lets every admitted pulse, delivery, reaction and
accepted signal settle, publishes the new graph and resumes: sources stay
open, operators keep their buffers, retained steps keep their state. See
``ActiveRun.apply_update`` for the handler rule, the timeout and what is
rejected.
"""

from roboflow_workflows.execution_engine.v2.updates.assessment import (
    ResetConsequences,
    StateConsequence,
    UpdateAssessment,
    assess_update,
)
from roboflow_workflows.execution_engine.v2.updates.diff import (
    PRESERVE,
    RESET,
    UNSUPPORTED,
    PlanChange,
    PlanDiff,
    compare_plans,
)
from roboflow_workflows.execution_engine.v2.updates.prepared import (
    APPLIED,
    DISCARDED,
    PREPARED,
    ActiveUpdateReceipt,
    PreparedUpdate,
    UpdateReceipt,
    prepare_update,
)
from roboflow_workflows.execution_engine.v2.updates.reset import (
    Cleanup,
    Retirement,
    prepare_reset,
)

__all__ = [
    "APPLIED",
    "DISCARDED",
    "PRESERVE",
    "PREPARED",
    "RESET",
    "UNSUPPORTED",
    "ActiveUpdateReceipt",
    "Cleanup",
    "PlanChange",
    "PlanDiff",
    "PreparedUpdate",
    "ResetConsequences",
    "Retirement",
    "StateConsequence",
    "UpdateAssessment",
    "UpdateReceipt",
    "assess_update",
    "compare_plans",
    "prepare_reset",
    "prepare_update",
]

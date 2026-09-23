"""The registered observer default, and the protocol the server's observer meets.

The engine's default is the workflows-local `NullExecutionObserver`, which
records nothing - so a root that forgets the binding keeps working, keeps its
tests green, and silently stops billing. What each root actually binds is
asserted against a real engine in `test_image_codec_binding.py` (the four
server/CLI roots) and `test_direct_caller_bindings.py` (the two maintained
scripts); the inventory of roots lives in
`test_workflows_composition_roots.py`.
"""

from inference.core.interfaces.workflows_execution_observer import (
    UsageTrackingExecutionObserver,
)
from inference.core.workflows.core_steps.loader import REGISTERED_INITIALIZERS
from inference.core.workflows.prototypes.observer import (
    ExecutionObserver,
    NullExecutionObserver,
)


def test_the_registered_default_is_the_null_observer() -> None:
    # Blocks that declare the parameter and get no host binding resolve to a
    # no-op, so a workflows-only process runs without a billing stack.
    assert isinstance(
        REGISTERED_INITIALIZERS["execution_observer"], NullExecutionObserver
    )


def test_the_bound_observer_and_the_default_share_the_protocol() -> None:
    assert isinstance(UsageTrackingExecutionObserver(), ExecutionObserver)
    assert isinstance(REGISTERED_INITIALIZERS["execution_observer"], ExecutionObserver)

"""Request-scoped billing intent for usage tracking.

An authenticated ``countinference=false`` is decided once, at the HTTP boundary,
but it has to reach usage rows recorded far from that decision. Stamping
``usage_billable`` onto the incoming request carries it as far as that request
object travels, which is enough for the ``/infer/*`` routes. It is not enough
for Workflows: a block builds its own ``InferenceRequest`` from manifest fields
and holds no reference to the caller's query parameters, so there is nothing to
stamp.

Publishing the decision as a context variable closes that gap without threading
a parameter through every block that takes a model manager. Anything the
collector records inside the scope is attributed to the caller's intent.

ContextVars do not propagate into ``ThreadPoolExecutor`` workers, so the
Execution Engine re-binds this in every step thread - see ``safe_execute_step``.

Nor do they cross a process boundary. A block running in
``StepExecutionMode.REMOTE`` records no usage of its own - the model runs on the
server it calls, and that server bills it - so the scope has to travel as
request parameters instead. ``remote_billing_parameters`` renders it into the
fields the SDK already serializes for both API versions.

The sam3 and seg_preview blocks post to the platform inference proxy by hand
rather than through the SDK, and are deliberately not covered: that endpoint
does not record usage, so there is no bill to suppress.

This module must stay import-light: it is imported by the usage collector, by
the Execution Engine, and by every Workflow block with a remote execution path.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Iterator

from inference.core.env import ROBOFLOW_SERVICE_SECRET

usage_billing_suppressed: ContextVar[bool] = ContextVar(
    "usage_billing_suppressed", default=False
)


@contextmanager
def billing_suppressed(suppressed: bool) -> Iterator[None]:
    """Scope a billing decision to the work performed inside the block.

    Args:
        suppressed: True to record usage inside the scope as non-billable.

    Yields:
        None, for the duration of the scope.
    """
    token = usage_billing_suppressed.set(suppressed)
    try:
        yield
    finally:
        usage_billing_suppressed.reset(token)


def remote_billing_parameters() -> Dict[str, Any]:
    """Render the active scope as ``InferenceConfiguration`` fields.

    Spread into the config a Workflow block builds for remote step execution, so
    the server that actually runs the model - and therefore records the usage
    row - learns the caller's intent.

    The scope carries only a boolean, but the receiving end re-validates, so the
    secret has to travel too. It need not be the caller's own: validation is an
    equality check against ``ROBOFLOW_SERVICE_SECRET``, so a secret that passed
    the gate at the HTTP boundary is by definition the one configured here.
    That keeps a secret out of the context variable itself.

    Only ever sent to an operator-configured destination - ``HOSTED_*_URL`` or
    ``LOCAL_INFERENCE_API_URL`` - never to a host the caller chooses.

    Returns:
        Fields to spread into ``InferenceConfiguration``; empty unless the
        caller opted out of billing and a secret is configured to prove it.
    """
    if not usage_billing_suppressed.get():
        return {}
    if not ROBOFLOW_SERVICE_SECRET:
        return {}

    return {
        "count_inference": False,
        "service_secret": ROBOFLOW_SERVICE_SECRET,
    }

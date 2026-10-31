"""Step error handlers owned by the Workflows engine.

The engine ships only the mapping-free default below. Raw-manager callers
(``ExecutionEngine.init`` with a ``ModelManager``) receive the server handler
transparently via the ``__workflows_bind__`` host hook; no explicit wiring is
needed. Standalone or provider-only callers may pass an explicit handler via
``step_error_handler``; the server handler lives in
``inference.core.interfaces.workflows_step_error_handlers``.
"""


def legacy_step_error_handler(step_name: str, error: Exception) -> None:
    """Standalone default: no mapping.

    Every mapping the previous version carried (``FeatureDeprecatedError`` to
    410, re-raising ``ModelManagerLockAcquisitionError`` /
    ``InferenceModelNotFound``) was for ``inference.core.exceptions`` classes
    and now lives in ``inference.core.interfaces.workflows_step_error_handlers``,
    which the server passes explicitly to ``ExecutionEngine.init``.
    """
    return None

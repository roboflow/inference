"""Step error handlers owned by the Workflows engine.

The engine ships only the mapping-free default below. A direct
``ExecutionEngine.init(...)`` caller that wants the Roboflow exception mappings
(``FeatureDeprecatedError`` to HTTP 410, model-access and remote-execution
status codes) must pass the server handler explicitly via ``step_error_handler``
-- it lives in ``inference.core.interfaces.workflows_step_error_handlers`` and
is selected by ``resolve_step_error_handler()``, which every server composition
root already does.
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

from __future__ import annotations

from typing import Optional

from inference_models import errors as models_errors
from inference_models.errors import (
    BaseInferenceModelsError,
    ModelPackageAlternativesExhaustedError,
    ModelPackageRestrictedError,
)
from inference_server.legacy.errors import MODEL_RESTRICTED_MESSAGE


class ModelLoadFailedError(Exception):
    """Model load failure of a kind the legacy server answers as an internal error."""


def load_failure_error(result: tuple) -> Optional[Exception]:
    """Rebuild the exception a gateway described in a load failure tuple.

    Args:
        result: Failure tuple of a gateway, ``("error", code)`` optionally
            followed by a dict describing the error that failed the load.

    Returns:
        The described error as an instance of its ``inference_models`` class,
        ``ModelLoadFailedError`` for any other class, or None when the tuple
        carries no description.
    """
    detail = result[2] if len(result) > 2 else None
    if not isinstance(detail, dict):
        return None

    message = str(detail.get("message") or "")
    help_url = detail.get("help_url")
    error_class = getattr(models_errors, str(detail.get("error_type")), None)
    if not (
        isinstance(error_class, type)
        and issubclass(error_class, BaseInferenceModelsError)
    ):
        return ModelLoadFailedError(message)

    if issubclass(error_class, ModelPackageAlternativesExhaustedError):
        alternatives = (
            [ModelPackageRestrictedError(MODEL_RESTRICTED_MESSAGE)]
            if detail.get("restricted")
            else []
        )
        error = error_class(
            message, help_url=help_url, alternatives_errors=alternatives
        )
    else:
        error = error_class(message, help_url=help_url)
    status_code = detail.get("status_code")
    if status_code is not None and getattr(error, "status_code", None) != status_code:
        error.status_code = status_code

    return error

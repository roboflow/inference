from __future__ import annotations

import keyword
from typing import Any, Optional

from inference_models import errors as models_errors
from inference_models.errors import (
    BaseInferenceModelsError,
    ModelPackageAlternativesExhaustedError,
    ModelPackageRestrictedError,
)
from inference_server.legacy.errors import MODEL_RESTRICTED_MESSAGE


class ModelLoadFailedError(Exception):
    """Model load failure of a kind the legacy server answers as an internal error."""


_MAX_LOAD_FAILURE_CLASS_NAME_LENGTH = 64
_MAX_LOAD_FAILURE_CLASSES = 128
_LOAD_FAILURE_CLASSES: dict[str, type] = {}


def _load_failure_class(error_type: Any) -> type:
    if (
        not isinstance(error_type, str)
        or len(error_type) > _MAX_LOAD_FAILURE_CLASS_NAME_LENGTH
        or not error_type.isidentifier()
        or keyword.iskeyword(error_type)
    ):
        return ModelLoadFailedError

    error_class = _LOAD_FAILURE_CLASSES.get(error_type)
    if error_class is not None:
        return error_class
    if len(_LOAD_FAILURE_CLASSES) >= _MAX_LOAD_FAILURE_CLASSES:
        return ModelLoadFailedError

    error_class = type(error_type, (ModelLoadFailedError,), {})
    _LOAD_FAILURE_CLASSES[error_type] = error_class

    return error_class


def _models_error_class(error_type: str) -> Any:
    error_class = getattr(models_errors, error_type, None)
    if error_class is not None:
        return error_class
    if error_type not in ("NotServableOnVLLMError", "AdapterNotServableError"):
        return None
    from inference_models.models.vllm_proxy import errors as vllm_proxy_errors

    return getattr(vllm_proxy_errors, error_type, None)


def load_failure_error(result: tuple) -> Optional[Exception]:
    """Rebuild the exception a gateway described in a load failure tuple.

    Args:
        result: Failure tuple of a gateway, ``("error", code)`` optionally
            followed by a dict describing the error that failed the load.

    Returns:
        The described error as an instance of its ``inference_models`` class,
        a ``ModelLoadFailedError`` named after any other class, or None when
        the tuple carries no description.
    """
    detail = result[2] if len(result) > 2 else None
    if not isinstance(detail, dict):
        return None

    message = str(detail.get("message") or "")
    help_url = detail.get("help_url")
    error_class = _models_error_class(str(detail.get("error_type")))
    if not (
        isinstance(error_class, type)
        and issubclass(error_class, BaseInferenceModelsError)
    ):
        return _load_failure_class(detail.get("error_type"))(message)

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
